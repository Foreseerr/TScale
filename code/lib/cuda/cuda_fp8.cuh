#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"
#include "cuda_mma.cuh"
#include "cuda_matmul.cuh"


namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline __device__ void MMAe4e4(TRegTile<float> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2, const TRegTile<float> &c)
{
#if (__CUDA_ARCH__ >= 890)
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        :
        "=f"(pD->x[0]), "=f"(pD->x[1]), "=f"(pD->x[2]), "=f"(pD->x[3]) // "=f" means overwrite, "+f" means read-modify-write
        :
        "r"(a1.nx[0]), "r"(a1.nx[1]), "r"(a2.nx[0]), "r"(a2.nx[1]),
        "r"(b1.nx[0]), "r"(b2.nx[0]),
        "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3])
        );
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        :
        "=f"(pD->x[4]), "=f"(pD->x[5]), "=f"(pD->x[6]), "=f"(pD->x[7])
        :
        "r"(a1.nx[0]), "r"(a1.nx[1]), "r"(a2.nx[0]), "r"(a2.nx[1]),
        "r"(b1.nx[1]), "r"(b2.nx[1]),
        "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7])
        );
#else
    printf("sm89 required\n");
#endif
}

__forceinline __device__ void MMAe4e4(TRegTile<float> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2)
{
    MMAe4e4(pD, a1, a2, b1, b2, *pD);
}


__forceinline __device__ void MMAe5e4(TRegTile<float> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2, const TRegTile<float> &c)
{
#if (__CUDA_ARCH__ >= 890)
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        :
        "=f"(pD->x[0]), "=f"(pD->x[1]), "=f"(pD->x[2]), "=f"(pD->x[3]) // "=f" means overwrite, "+f" means read-modify-write
        :
        "r"(a1.nx[0]), "r"(a1.nx[1]), "r"(a2.nx[0]), "r"(a2.nx[1]),
        "r"(b1.nx[0]), "r"(b2.nx[0]),
        "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3])
        );
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        :
        "=f"(pD->x[4]), "=f"(pD->x[5]), "=f"(pD->x[6]), "=f"(pD->x[7])
        :
        "r"(a1.nx[0]), "r"(a1.nx[1]), "r"(a2.nx[0]), "r"(a2.nx[1]),
        "r"(b1.nx[1]), "r"(b2.nx[1]),
        "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7])
        );
#else
    printf("sm89 required\n");
#endif
}

__forceinline __device__ void MMAe5e4(TRegTile<float> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2)
{
    MMAe5e4(pD, a1, a2, b1, b2, *pD);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TFloatA, class TFloatB>
inline __device__ void MMA(TRegTile<float> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2, TFloatA*, TFloatB*)
{
    CUDA_ASSERT(0 && "unsupported combination");
}

inline __device__ void MMA(TRegTile<float> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2, e4m3 *, e4m3 *)
{
    MMAe4e4(pD, a1, a2, b1, b2);
}

inline __device__ void MMA(TRegTile<float> *pD, const TRegTile<i8> &a1, const TRegTile<i8> &a2, const TRegTile<i8> &b1, const TRegTile<i8> &b2, e5m2 *, e4m3 *)
{
    MMAe5e4(pD, a1, a2, b1, b2);
}

template <class TStoreData>
struct TFp8MatMulData
{
    union {
        struct {
            T8SMemI8Tile aFrag[8];
            T8SMemI8Tile bFrag[8];
        };
        TStoreData StoreData;
    };
};

template <class TStoreFunc, class TRes, class TFloatA, class TFloatB>
__global__ void Fp8MatMulKernel(TCuda2DPtr<TFloatA> aMatr, TCuda2DPtr<TFloatB> bMatr, int yTiles, TCuda2DPtr<TRes> resBuf, typename TStoreFunc::TParams params)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);
    TTileCoord tc;

    // blockIdx.x - hidden dim
    // blockIdx.y - time

    int warpId = threadIdx.y;

    int aStride = aMatr.GetStrideInBytes();
    int bStride = bMatr.GetStrideInBytes();
    TFloatA *aPtr = aMatr[blockIdx.y * MM_TILE];
    TFloatB *bPtr = bMatr[blockIdx.x * MM_TILE];

    TMatMulWarpResult<float> res;
    res.Clear();
    int aWarpOffset = (warpId & 1) * 4;
    int bWarpOffset = (warpId >> 1) * 4;

    __shared__ TFp8MatMulData<typename TStoreFunc::TShmem> data;
    for (int yTile = 0; yTile < yTiles; ++yTile) {
        __syncthreads();
        Copy8TileArray(data.aFrag, warpId, TCuda2DPtr<TFloatA>(aPtr, aStride, I8_TILE_GROUP_SIZE, 8 * TILE), 8, TILE * aStride);
        Copy8TileArray(data.bFrag, warpId, TCuda2DPtr<TFloatB>(bPtr, bStride, I8_TILE_GROUP_SIZE, 8 * TILE), 8, TILE * bStride);
        __syncthreads();

        for (int k = 0; k < 8; k += 2) {
            // run out of registers if using single loop with preloading b1[4] & b2[4], so split into 2 loops
            TRegTile<i8> b1[2];
            TRegTile<i8> b2[2];
            b1[0] = TMmaColMajor::FragB(data.bFrag[bWarpOffset + 0], k);
            b1[1] = TMmaColMajor::FragB(data.bFrag[bWarpOffset + 1], k);
            b2[0] = TMmaColMajor::FragB(data.bFrag[bWarpOffset + 0], k + 1);
            b2[1] = TMmaColMajor::FragB(data.bFrag[bWarpOffset + 1], k + 1);
            for (int ty = 0; ty < 4; ++ty) {
                TRegTile<i8> a1 = TMmaRowMajor::FragA(data.aFrag[aWarpOffset + ty], k);
                TRegTile<i8> a2 = TMmaRowMajor::FragA(data.aFrag[aWarpOffset + ty], k + 1);
                for (int tx = 0; tx < 2; ++tx) {
                    MMA(&res.Sum[ty][tx], a1, a2, b1[tx], b2[tx], aPtr, bPtr);
                }
            }
            b1[0] = TMmaColMajor::FragB(data.bFrag[bWarpOffset + 2], k);
            b1[1] = TMmaColMajor::FragB(data.bFrag[bWarpOffset + 3], k);
            b2[0] = TMmaColMajor::FragB(data.bFrag[bWarpOffset + 2], k + 1);
            b2[1] = TMmaColMajor::FragB(data.bFrag[bWarpOffset + 3], k + 1);
            for (int ty = 0; ty < 4; ++ty) {
                TRegTile<i8> a1 = TMmaRowMajor::FragA(data.aFrag[aWarpOffset + ty], k);
                TRegTile<i8> a2 = TMmaRowMajor::FragA(data.aFrag[aWarpOffset + ty], k + 1);
                for (int tx = 0; tx < 2; ++tx) {
                    MMA(&res.Sum[ty][tx + 2], a1, a2, b1[tx], b2[tx], aPtr, bPtr);
                }
            }
        }
        aPtr += I8_TILE_GROUP_SIZE;
        bPtr += I8_TILE_GROUP_SIZE;
    }
    // save result
    TStoreFunc::Store(params, data.StoreData, tc, res, 1.0f, bWarpOffset * TILE, aWarpOffset * TILE, resBuf);
}
KERNEL_BLOCK_SIZE(Fp8MatMulKernel, WARP_SIZE, 4);


// XY,ZY->XZ
template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &Fp8MatMul(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles)
{
    return CudaCall(c, Fp8MatMulKernel<TStoreFunc, typename TRes::TElem, typename T1::TElem, typename T2::TElem>)
        .Grid(zTiles, xTiles)
        (aMatr, bMatr, yTiles)
        .Write(pResMatr);
}


}
