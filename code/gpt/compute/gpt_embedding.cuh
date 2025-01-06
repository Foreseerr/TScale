#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
// embedding

struct TLabelInverseIndex
{
    struct TLabelPos
    {
        union {
            struct {
                int Pos;
                TLabelIndex Label;
            };
            ui64 SortValue;
        };

        TLabelPos() {}
        TLabelPos(TLabelIndex label, int pos) : Label(label), Pos(pos) {}
    };

    static const ui64 *GetSortValue(const TLabelPos &x)
    {
        return &x.SortValue;
    }

    TVector<TLabelPos> LabelPosArr;
    TVector<TLabelPos> LabelPosArrTempBuf;
    TVector<TLabelIndex> InvLabelArr;
    TVector<ui32> InvLabelPos;
    TVector<ui32> InvLabelPosPtr;

    // make list of references for each label
    // for labels with positions in previous iteration and no positions in current iteration create empty list
    void BuildInverseIndex(const TVector<TLabelIndex> &labelArr, const TVector<ui32> &labelPtr)
    {
        Y_ASSERT(sizeof(LABEL_NEGATIVE) == sizeof(InvLabelPos[0]));
        yint labelCount = YSize(labelArr);
        Y_ASSERT(labelCount == labelPtr.back());

        // add current iteration positions
        LabelPosArr.resize(labelCount);
        for (yint pos = 0; pos < YSize(labelPtr) - 1; ++pos) {
            for (yint k = labelPtr[pos]; k < labelPtr[pos + 1]; ++k) {
                TLabelIndex label = labelArr[k];
                TLabelIndex negMask = label & LABEL_NEGATIVE;
                LabelPosArr[k] = TLabelPos(label & LABEL_MASK, pos | negMask);
            }
        }

        // invert index
        LabelPosArrTempBuf.resize(labelCount);
        RadixUI64SortAscending(&LabelPosArr, &LabelPosArrTempBuf, GetSortValue);

        // make lists
        InvLabelArr.resize(0);
        InvLabelPos.resize(labelCount);
        InvLabelPosPtr.resize(0);
        TLabelIndex prevLabel = LABEL_INVALID_INDEX;
        for (yint k = 0; k < labelCount; ++k) {
            const TLabelPos &labelPos = LabelPosArr[k];
            TLabelIndex label = labelPos.Label;
            if (label != prevLabel) {
                InvLabelPosPtr.push_back(k);
                InvLabelArr.push_back(label);
                prevLabel = label;
            }
            InvLabelPos[k] = labelPos.Pos;
        }
        InvLabelPosPtr.push_back(labelCount);
    }
};


struct TLabelForwardIndex
{
    struct TLabelInfo
    {
        int Index = 0;
        int Timestamp = 0;
    };
    TVector<TLabelInfo> LabelIndex;
    TVector<TLabelIndex> RecodedLabelArr;
    TVector<TLabelIndex> UsedLabels;
    int Timestamp = 0;

public:
    void Init(int labelCount)
    {
        ClearPodArray(&LabelIndex, labelCount);
        Timestamp = 0;
    }
    void BuildUsedIndex(const TVector<TLabelIndex> &labelArr)
    {
        int curTimestamp = ++Timestamp;
        int sz = YSize(labelArr);
        RecodedLabelArr.yresize(sz);
        UsedLabels.yresize(sz);
        int dst = 0;
        for (int i = 0; i < sz; ++i) {
            TLabelIndex srcLabel = labelArr[i];
            TLabelIndex label = srcLabel & LABEL_MASK;
            TLabelInfo &info = LabelIndex[label];
            if (info.Timestamp != curTimestamp) {
                UsedLabels[dst] = label;
                info.Index = dst++;
                info.Timestamp = curTimestamp;
            }
            RecodedLabelArr[i] = info.Index | (srcLabel & LABEL_NEGATIVE);
        }
        UsedLabels.resize(dst);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// embed kernels
__global__ void CopyUsedEmbeddings(TCuda1DPtr<TLabelIndex> labelArr, TCuda2DPtr<TEmbedFloat> labelEmbedding, TCuda2DPtr<TEmbedFloat> dst)
{
    int tile = blockIdx.x;
    int k = blockIdx.y;
    int label = labelArr[k];
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;

    float vec[WSZ];
    LoadWarpVec<WSZ>(vec, labelEmbedding[label] + offset);
    StoreWarpVec<WSZ>(dst[k] + offset, vec);
}


template <int WSZ, class T>
__device__ void AddEmbedVec(float *dst, TCuda2DPtr<T> labelEmbedding, int offset, TLabelIndex idx)
{
    float vec[WSZ];
    LoadWarpVec<WSZ>(vec, labelEmbedding[idx & LABEL_MASK] + offset);
    if (idx & LABEL_NEGATIVE) {
        ScaleWarpVec<WSZ>(vec, -1);
    }
    AddWarpVec<WSZ>(dst, vec);
}

__global__ void AddEmbeddings(
    int len,
    TCuda1DPtr<TLabelIndex> labelArr, TCuda1DPtr<ui32> labelPtr, TCuda2DPtr<TEmbedFloat> labelEmbedding, float *labelEmbeddingScale, float labelEmbeddingStaticScale,
    TCuda2DPtr<TStateFloat> state
)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;

    float vec[WSZ];
    LoadZeroWarpVec<WSZ>(vec);
    if (t < len) {
        int start = labelPtr[t];
        int finish = labelPtr[t + 1];
        if (finish > start) {
            for (int z = start; z < finish; ++z) {
                AddEmbedVec<WSZ>(vec, labelEmbedding, offset, labelArr[z]);
            }
            ScaleWarpVec<WSZ>(vec, *labelEmbeddingScale * labelEmbeddingStaticScale);
        }
    }
    StoreWarpVec<WSZ>(state[t] + offset, vec);
}


template <class TGradFloat>
__global__ void BackpropEmbeddings(
    int dim,
    TCuda1DPtr<TLabelIndex> invLabelArr, TCuda1DPtr<ui32> invLabelPos, TCuda1DPtr<ui32> invLabelPosPtr,
    TCuda2DPtr<TGradFloat> stateGrad, float *stateGradScale,
    TCuda1DPtr<i8> deltaLabelEmbedding, TCuda1DPtr<float> deltaLabelTileScale
)
{
    CUDA_STATIC_ASSERT(MM_TILE == MODEL_INT8_DELTA_TILE);
    int tile = blockIdx.x;
    int labelId = blockIdx.y;
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;

    int start = invLabelPosPtr[labelId];
    int finish = invLabelPosPtr[labelId + 1];
    if (start == finish) {
        return;
    }

    // compute gradient
    float delta[WSZ];
    LoadZeroWarpVec<WSZ>(delta);
    for (int z = start; z < finish; ++z) {
        AddEmbedVec<WSZ>(delta, stateGrad, offset, invLabelPos[z]);
    }
    ScaleWarpVec<WSZ>(delta, *stateGradScale);

    // compute scale
    float maxVal = 0;
    for (int k = 0; k < WSZ; ++k) {
        maxVal = fmaxf(maxVal, fabs(delta[k]));
    }
    maxVal = WarpMax(maxVal);
    float scale = (maxVal > 0) ? maxVal / 127 : 0;
    float mult = (maxVal > 0) ? 1 / scale : 0;

    // pack delta
    __shared__ i8 packedDelta[MM_TILE];
    ScaleWarpVec<WSZ>(delta, mult);
    StoreWarpVec<WSZ>(packedDelta, delta);
    __syncwarp();

    // write
    int label = invLabelArr[labelId];
    int dstTileId = (label * dim + offset) / MM_TILE;
    int h = threadIdx.x;
    int thrOffset = h * 4;
    *(int *)(&deltaLabelEmbedding[dstTileId * MM_TILE + thrOffset]) = *(int *)(packedDelta + thrOffset);
    if (h == 0) {
        deltaLabelTileScale[dstTileId] = scale;
    }
    __threadfence_system(); // neccessary since we write to host memory
}
