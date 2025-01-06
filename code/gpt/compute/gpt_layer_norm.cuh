///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void ApplyDropoutSmem(int dim, int offset, TCuda1DPtr<ui32> dropTable, float *buf)
{
    int h = threadIdx.x;
    int blk = offset / 32;
    for (int base = 0; base < dim; base += WARP_SIZE, ++blk) {
        int d = base + h;
        if ((dropTable[blk] & (1 << h)) == 0) {
            buf[d] = 0;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TSrc, class TDst>
__global__ void LayerNormalizeStateVecs(int len, int dim, float vecScale,
    TCuda2DPtr<TSrc> state, TCuda1DPtr<ui32> dropTable,
    TCuda2DPtr<TDst> normState, TCuda2DPtr<float> normStateScale)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * dim;
    int h = threadIdx.x;
    extern __shared__ float buf[];

    if (t >= len) {
        StoreZeroSmemVec(dim, normState[t] + offset);
        if (h == 0) {
            normStateScale[tile][t] = 0;
        }
        return;
    }
    LoadSmemVec(dim, state[t] + offset, buf);
    ApplyDropoutSmem(dim, offset, dropTable, buf);
    float sum2 = CalcSum2Smem(dim, buf);
    float discrScale = 0;
    float mult = 0;
    if (sum2 > 0) {
        discrScale = sqrtf(sum2 / dim) * vecScale;
        mult = 1 / discrScale;
    }
    StoreScaledSmemVec(dim, buf, mult, normState[t] + offset);
    if (h == 0) {
        normStateScale[tile][t] = discrScale;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TSrc, class TGrad>
__global__ void BackpropLayerNormalize(int dim,
    TCuda1DPtr<ui32> dropTable,
    TCuda2DPtr<TSrc> normState, TCuda2DPtr<float> stateScale,
    TCuda2DPtr<float> dNormState,
    float *combinerScale,
    TCuda2DPtr<TGrad> pStateGrad, float *nextLayerGradMaxNorm
)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * dim;
    int h = threadIdx.x;
    extern __shared__ float buf[];
    float *srcBuf = buf;
    float *gradBuf = buf + dim;

    LoadSmemVec(dim, normState[t] + offset, srcBuf);
    ScaleSmemVec(dim, srcBuf, stateScale[tile][t]);
    ApplyDropoutSmem(dim, offset, dropTable, srcBuf);

    LoadSmemVec(dim, dNormState[t] + offset, gradBuf);
    ScaleSmemVec(dim, gradBuf, *combinerScale);

    BackpropNormalizeSmem(dim, srcBuf, gradBuf, gradBuf);
    ApplyDropoutSmem(dim, offset, dropTable, gradBuf);

    LoadAddSmemVec(dim, pStateGrad[t] + offset, gradBuf);
    float gradMax = CalcMaxAbsValueSmem(dim, gradBuf);
    StoreScaledSmemVec(dim, gradBuf, 1, pStateGrad[t] + offset);

    if (h == 0) {
        atomicMax((int *)nextLayerGradMaxNorm, __float_as_int(gradMax));
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TSrc, class TGrad>
__global__ void BackpropFinalNormalize(
    int len, int dim,
    TCuda1DPtr<ui32> dropTable,
    TCuda2DPtr<TSrc> state,
    TCuda2DPtr<TGrad> dNormState,
    TCuda2DPtr<TGrad> pStateGrad, float *nextLayerGradMaxNorm
)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * dim;
    int h = threadIdx.x;
    extern __shared__ float buf[];
    float *srcBuf = buf;
    float *gradBuf = buf + dim;

    if (t < len) {
        LoadSmemVec(dim, state[t] + offset, srcBuf);
        ApplyDropoutSmem(dim, offset, dropTable, srcBuf);

        LoadSmemVec(dim, dNormState[t] + offset, gradBuf);

        BackpropNormalizeSmem(dim, srcBuf, gradBuf, gradBuf);
        ApplyDropoutSmem(dim, offset, dropTable, gradBuf);

        float gradMax = CalcMaxAbsValueSmem(dim, gradBuf);
        StoreScaledSmemVec(dim, gradBuf, 1, pStateGrad[t] + offset);
        if (h == 0) {
            atomicMax((int *)nextLayerGradMaxNorm, __float_as_int(gradMax));
        }
    } else {
        StoreZeroSmemVec(dim, pStateGrad[t] + offset);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TNormFloat>
struct TStoreLayerAddDelta
{
    struct TParams
    {
        float Scale;
        float *ScalePtr;
        float VecScale; // for normalization
        TCuda2DPtr<TNormFloat> NormState;
        TCuda2DPtr<float> NormStateScale;
    };
    struct TShmem
    {
        float ScaledRes[2][TILE][MM_TILE];
    };

    template <class TRes, class T>
    __device__ static void Store(TParams &params, TShmem &shmem, const TTileCoord &tc, const TMatMulWarpResult<T> &mmRes, float resultScale, int offsetX, int offsetY, TCuda2DPtr<TRes> resBuf)
    {
        CUDA_STATIC_ASSERT(MM_TILE == 128);
        CUDA_STATIC_ASSERT(STATE_NORM_TILE == 128);
        int mmTileX = blockIdx.x * MM_TILE;
        int mmTileY = blockIdx.y * MM_TILE;
        int h = threadIdx.x;
        int tile = blockIdx.x;
        float sumScale = params.Scale * resultScale;
        if (params.ScalePtr) {
            sumScale *= *params.ScalePtr;
        }

        // process 2 rows at at time
        for (int ty = 0; ty < 4; ++ty) {
            __syncthreads();
            for (int tx = 0; tx < 4; ++tx) {
                mmRes.Sum[ty][tx].StoreScaled(tc, TCuda2DPtr<float>(&shmem.ScaledRes[offsetY / 64][0][offsetX + tx * TILE], MM_TILE * sizeof(float), 16, 16), sumScale);
            }
            __syncthreads();
            for (int base = 0; base < TILE; base += 2) {
                int rowId = base + (offsetX / 64);
                int dstY = mmTileY + offsetY + ty * TILE + rowId;
                constexpr int WSZ = MM_TILE / WARP_SIZE;
                float vec[WSZ];
                // add delta
                LoadWarpVec<WSZ>(vec, shmem.ScaledRes[offsetY / 64][rowId]);
                LoadAddWarpVec<WSZ>(vec, resBuf[dstY] + mmTileX);
                StoreWarpVec<WSZ>(resBuf[dstY] + mmTileX, vec);
                // normalize vec
                //ApplyDropoutSmem(dim, offset, dropTable, buf);
                float sum2 = CalcWarpVecSum2<WSZ>(vec);
                float discrScale = 0;
                if (sum2 > 0) {
                    discrScale = sqrt(sum2 / MM_TILE) * params.VecScale;
                    ScaleWarpVec<WSZ>(vec, 1 / discrScale);
                }
                StoreWarpVec<WSZ>(params.NormState[dstY] + mmTileX, vec);
                if (h == 0) {
                    params.NormStateScale[tile][dstY] = discrScale;
                }
            }
        }
    }
};
