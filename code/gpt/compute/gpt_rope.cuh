//typedef float TRopeFloat;
typedef half TRopeFloat;

struct TStoreRowTileNormalizeRope
{
    struct TParams
    {
        TCuda2DPtr<TRopeFloat> RopeBuf;
        float Scale;
        float VecScale; // for normalization
        TCuda2DPtr<float> ScaleBuf;
    };
    struct TShmem
    {
        float ScaledRes[2][TILE][MM_TILE];
    };

    template <class TRes, class T>
    __device__ static void Store(TParams &params, TShmem &shmem, const TTileCoord &tc, const TMatMulWarpResult<T> &mmRes, float resultScale, int offsetX, int offsetY, TCuda2DPtr<TRes> resBuf)
    {
        CUDA_STATIC_ASSERT(MM_TILE == 128);
        int mmTileX = blockIdx.x * MM_TILE;
        int mmTileY = blockIdx.y * MM_TILE;
        int h = threadIdx.x;
        int tile = blockIdx.x;
        float sumScale = params.Scale * resultScale;

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
                LoadWarpVec<WSZ>(vec, shmem.ScaledRes[offsetY / 64][rowId]);
                float sum2 = CalcWarpVecSum2<WSZ>(vec);
                float discrScale = 0;
                if (sum2 > 0) {
                    discrScale = sqrt(sum2 / MM_TILE) * params.VecScale;
                    ScaleWarpVec<WSZ>(vec, 1 / discrScale);
                }
                // apply rope
                float rope[WSZ];
                LoadWarpVec<WSZ>(rope, params.RopeBuf[dstY]);
                for (int k = 0; k < WSZ; k += 2) {
                    float cosValue = rope[k];
                    float sinValue = rope[k + 1];
                    float r0 = vec[k] * cosValue - vec[k + 1] * sinValue;
                    float r1 = vec[k] * sinValue + vec[k + 1] * cosValue;
                    vec[k] = r0;
                    vec[k + 1] = r1;
                }
                // save
                StoreWarpVec<WSZ>(resBuf[dstY] + mmTileX, vec);
                if (h == 0) {
                    params.ScaleBuf[tile][dstY] = discrScale;
                }
            }
        }
    }
};


template <int WSZ>
inline __device__ void ApplyWarpRopeImpl(const float *rope, float ropeRotateDir, float *vec)
{
    for (int k = 0; k < WSZ; k += 2) {
        float cosValue = rope[k];
        float sinValue = rope[k + 1] * ropeRotateDir;
        float r0 = vec[k] * cosValue - vec[k + 1] * sinValue;
        float r1 = vec[k] * sinValue + vec[k + 1] * cosValue;
        vec[k] = r0;
        vec[k + 1] = r1;
    }
}


// rope fwd/bwd (for bwd invert rotate dir)
template <class TGrad>
__global__ void ApplyRope(TCuda2DPtr<TRopeFloat> ropeBuf, float ropeRotateDir, TCuda2DPtr<TGrad> grad)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;

    float vec[WSZ];
    LoadWarpVec<WSZ>(vec, grad[t] + offset);
    float rope[WSZ];
    LoadWarpVec<WSZ>(rope, ropeBuf[t]);
    ApplyWarpRopeImpl<WSZ>(rope, ropeRotateDir, vec);
    StoreWarpVec<WSZ>(grad[t] + offset, vec);
}


inline void FillRopeBuf(TStream &stream, TCuda2DArray<TRopeFloat> *pRopeBuf, yint width, yint maxLen)
{
    TArray2D<float> rope;
    FillRopeBuf(&rope, width, maxLen);
    pRopeBuf->Allocate(width, maxLen);
    Put(stream, pRopeBuf, rope);
}
