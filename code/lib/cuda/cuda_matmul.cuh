#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"
#include "cuda_mma.cuh"
#include "vec_util.cuh"


namespace NCuda
{
constexpr int MM_TILE = 128;

template <class T>
struct TMatMulWarpResult
{
    TRegTile<T> Sum[4][4];

    __device__ void Clear()
    {
        for (int ty = 0; ty < 4; ++ty) {
            for (int tx = 0; tx < 4; ++tx) {
                Sum[ty][tx].Clear();
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// store result or add result
struct TStore
{
    struct TParams
    {
    };
    struct TShmem
    {
    };

    template <class TRes, class T>
    __device__ static void Store(TParams &params, TShmem &shmem, const TTileCoord &tc, const TMatMulWarpResult<T> &mmRes, float resultScale, int offsetX, int offsetY, TCuda2DPtr<TRes> resBuf)
    {
        int blkX = blockIdx.x * MM_TILE + offsetX;
        int blkY = blockIdx.y * MM_TILE + offsetY;
        if (resultScale == 1) {
            for (int ty = 0; ty < 4; ++ty) {
                for (int tx = 0; tx < 4; ++tx) {
                    mmRes.Sum[ty][tx].Store(tc, resBuf.Fragment(blkX + tx * TILE, blkY + ty * TILE));
                }
            }
        } else {
            for (int ty = 0; ty < 4; ++ty) {
                for (int tx = 0; tx < 4; ++tx) {
                    mmRes.Sum[ty][tx].StoreScaled(tc, resBuf.Fragment(blkX + tx * TILE, blkY + ty * TILE), resultScale);
                }
            }
        }
    }
};


struct TStoreScaled
{
    struct TParams
    {
        float *ScalePtr;
        float Scale;
    };
    struct TShmem
    {
    };

    template <class TRes, class T>
    __device__ static void Store(TParams &params, TShmem &shmem, const TTileCoord &tc, const TMatMulWarpResult<T> &mmRes, float resultScale, int offsetX, int offsetY, TCuda2DPtr<TRes> resBuf)
    {
        int blkX = blockIdx.x * MM_TILE + offsetX;
        int blkY = blockIdx.y * MM_TILE + offsetY;
        float tileScale = params.Scale * resultScale;
        if (params.ScalePtr) {
            tileScale *= *params.ScalePtr;
        }
        for (int ty = 0; ty < 4; ++ty) {
            for (int tx = 0; tx < 4; ++tx) {
                mmRes.Sum[ty][tx].StoreScaled(tc, resBuf.Fragment(blkX + tx * TILE, blkY + ty * TILE), tileScale);
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// add result
struct TStoreAdd
{
    struct TParams
    {
    };
    struct TShmem
    {
    };

    template <class TRes, class T>
    __device__ static void Store(TParams &params, TShmem &shmem, const TTileCoord &tc, const TMatMulWarpResult<T> &mmRes, float resultScale, int offsetX, int offsetY, TCuda2DPtr<TRes> resBuf)
    {
        int blkX = blockIdx.x * MM_TILE + offsetX;
        int blkY = blockIdx.y * MM_TILE + offsetY;
        for (int ty = 0; ty < 4; ++ty) {
            for (int tx = 0; tx < 4; ++tx) {
                mmRes.Sum[ty][tx].StoreAddScaled(tc, resBuf.Fragment(blkX + tx * TILE, blkY + ty * TILE), resultScale);
            }
        }
    }
};


struct TStoreAddScaled
{
    struct TParams
    {
        float *ScalePtr;
        float Scale;
    };
    struct TShmem
    {
    };

    template <class TRes, class T>
    __device__ static void Store(TParams &params, TShmem &shmem, const TTileCoord &tc, const TMatMulWarpResult<T> &mmRes, float resultScale, int offsetX, int offsetY, TCuda2DPtr<TRes> resBuf)
    {
        int blkX = blockIdx.x * MM_TILE + offsetX;
        int blkY = blockIdx.y * MM_TILE + offsetY;
        float tileScale = params.Scale * resultScale;
        if (params.ScalePtr) {
            tileScale *= *params.ScalePtr;
        }
        for (int ty = 0; ty < 4; ++ty) {
            for (int tx = 0; tx < 4; ++tx) {
                mmRes.Sum[ty][tx].StoreAddScaled(tc, resBuf.Fragment(blkX + tx * TILE, blkY + ty * TILE), tileScale);
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// store normalized result
template <int RELU>
struct TStoreRowTileNormalizeBase
{
    struct TParams
    {
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
                constexpr int WSZ = MM_TILE / WARP_SIZE;
                float vec[WSZ];
                LoadWarpVec<WSZ>(vec, shmem.ScaledRes[offsetY / 64][rowId]);
                float sum2 = CalcWarpVecSum2<WSZ>(vec);
                float discrScale = 0;
                if (sum2 > 0) {
                    discrScale = sqrt(sum2 / MM_TILE) * params.VecScale;
                    ScaleWarpVec<WSZ>(vec, 1 / discrScale);
                }
                if (RELU) {
                    for (int k = 0; k < WSZ; ++k) {
                        vec[k] = fmaxf(0, vec[k]);
                    }
                }
                int dstY = mmTileY + offsetY + ty * TILE + rowId;
                StoreWarpVec<WSZ>(resBuf[dstY] + mmTileX, vec);
                if (h == 0) {
                    params.ScaleBuf[tile][dstY] = discrScale;
                }
            }
        }
    }
};

typedef TStoreRowTileNormalizeBase<0> TStoreRowTileNormalize;
typedef TStoreRowTileNormalizeBase<1> TStoreRowTileNormalizeRelu;


///////////////////////////////////////////////////////////////////////////////////////////////////
// store max normalized result
struct TStoreRowTileMaxNormalize
{
    struct TParams
    {
        float *ScalePtr;
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
        float sumScale = resultScale;
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
                constexpr int WSZ = MM_TILE / WARP_SIZE;
                float vec[WSZ];
                LoadWarpVec<WSZ>(vec, shmem.ScaledRes[offsetY / 64][rowId]);
                float maxVal = CalcWarpVecMaxAbsValue<WSZ>(vec);
                float discrScale = 0;
                if (maxVal > 0) {
                    discrScale = GetMaxDiscrScale(maxVal, (TRes *)0);
                    ScaleWarpVec<WSZ>(vec, 1 / discrScale);
                }
                int dstY = mmTileY + offsetY + ty * TILE + rowId;
                StoreWarpVec<WSZ>(resBuf[dstY] + mmTileX, vec);
                if (h == 0) {
                    params.ScaleBuf[tile][dstY] = discrScale;
                }
            }
        }
    }
};

}
