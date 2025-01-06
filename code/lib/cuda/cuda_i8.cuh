#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"
#include "cuda_mma.cuh"
#include "cuda_matmul.cuh"
#include "cuda_sort.cuh"


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr int I8_TRANSPOSE_WARPS = 8;

__global__ void TransposeI8Matrix(TCuda2DPtr<i8> src, TCuda2DPtr<i8> dst);
KERNEL_BLOCK_SIZE(TransposeI8Matrix, WARP_SIZE, I8_TRANSPOSE_WARPS);

// xTiles & yTiles are src dimensions
template <class TXSize, class TYSize>
void Transpose(TIntrusivePtr<TGraph> c, const TCuda2DArray<i8> &src, TXSize &&xTiles, TYSize &&yTiles, TCuda2DArray<i8> *pDst)
{
    CudaCall(c, TransposeI8Matrix).Grid(xTiles, yTiles)(src).Write(pDst);
}

template <class TXSize, class TYSize>
void Transpose(TIntrusivePtr<TGraph> c, const TCuda2DArray<e4m3> &src, TXSize &&xTiles, TYSize &&yTiles, TCuda2DArray<e4m3> *pDst)
{
    CudaCall(c, TransposeI8Matrix).Grid(xTiles, yTiles)(src).Write(pDst);
}

template <class TXSize, class TYSize>
void Transpose(TIntrusivePtr<TGraph> c, const TCuda2DArray<e5m2> &src, TXSize &&xTiles, TYSize &&yTiles, TCuda2DArray<e5m2> *pDst)
{
    CudaCall(c, TransposeI8Matrix).Grid(xTiles, yTiles)(src).Write(pDst);
}

template <class TXSize, class TYSize, class T>
void Transpose(TIntrusivePtr<TGraph> c, const TCuda2DArray<T> &src, TXSize &&xTiles, TYSize &&yTiles, TCuda2DArray<T> *pDst)
{
    Y_VERIFY(0 && "unsupported");
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// async data loader with prefetch, uses 4 warps
template <int LOOP_COUNT, int BUF_COUNT, int PREFETCH>
class TAsyncLoader
{
    T8SMemI8Tile *Buf;
    i8 *Data;
    int Idx;
    int StrideInBytes;

public:
    template <class T>
    __device__ TAsyncLoader(T8SMemI8Tile *buf, int warpId, T *data, int strideInBytes) : Buf(buf), Data((i8*)data), Idx(0), StrideInBytes(strideInBytes)
    {
        CUDA_STATIC_ASSERT(sizeof(T) == 1);
        CUDA_ASSERT(warpId >= 0 && warpId < 4);
        CUDA_ASSERT(PREFETCH < BUF_COUNT);
        for (int pr = 0; pr < PREFETCH; ++pr) {
            Copy8TileAsync(&Buf[pr], warpId, TCuda2DPtr<i8>(Data, StrideInBytes, 128, 16));
            Data += StrideInBytes * 16;
            AsyncCommitGroup();
        }
    }
    __device__ ~TAsyncLoader()
    {
        WaitAsyncCopy();
    }
    __device__ void Load(int warpId)
    {
        AsyncWaitGroup<PREFETCH - 1>();
        if (Idx < LOOP_COUNT - PREFETCH) {
            T8SMemI8Tile &dst = Buf[(Idx + PREFETCH) % BUF_COUNT];
            Copy8TileAsync(&dst, warpId, TCuda2DPtr<i8>(Data, StrideInBytes, 128, 16));
        }
        AsyncCommitGroup();
        __syncthreads();
    }
    __device__ int GetIndex() const
    {
        return Idx;
    }
    __device__ T8SMemI8Tile &GetBuf()
    {
        return Buf[Idx % BUF_COUNT];
    }
    __device__ bool IsValid() const
    {
        return Idx < LOOP_COUNT;
    }
    __device__ void Next()
    {
        ++Idx;
        Data += StrideInBytes * 16;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TStoreData>
struct TInt8MatMulData
{
    union {
        struct {
            T8SMemI8Tile aFrag[8];
            T8SMemI8Tile bFrag[8];
        };
        float RowScaleArr[128];
        TStoreData StoreData;
    };
};

struct TInt8MatMulCtx
{
    union {
        TMatMulWarpResult<int> Res;
        TMatMulWarpResult<float> ResScaled;
    };
    int aWarpOffset;
    int bWarpOffset;
    int aStride;
    int bStride;
    i8 *aPtr;
    i8 *bPtr;

    __device__ TInt8MatMulCtx(TCuda2DPtr<i8> aMatr, TCuda2DPtr<i8> bMatr, int warpId)
    {
        Res.Clear();
        aStride = aMatr.GetStrideInBytes();
        bStride = bMatr.GetStrideInBytes();
        aPtr = aMatr[blockIdx.y * MM_TILE];
        bPtr = bMatr[blockIdx.x * MM_TILE];

        aWarpOffset = (warpId & 1) * 4;
        bWarpOffset = (warpId >> 1) * 4;
    }

    template <class TStoreData>
    __device__ void LoadData(TInt8MatMulData<TStoreData> *pData, int warpId)
    {
        Copy8TileArray(pData->aFrag, warpId, TCuda2DPtr<i8>(aPtr, aStride, I8_TILE_GROUP_SIZE, 8 * TILE), 8, TILE * aStride);
        Copy8TileArray(pData->bFrag, warpId, TCuda2DPtr<i8>(bPtr, bStride, I8_TILE_GROUP_SIZE, 8 * TILE), 8, TILE * bStride);
    }

    template <class TStoreData>
    __device__ void Add(TInt8MatMulData<TStoreData> *pData, int warpId)
    {
        for (int k = 0; k < 8; ++k) {
            TRegTile<i8> b[4];
            b[0] = TMmaColMajor::FragB(pData->bFrag[bWarpOffset + 0], k);
            b[1] = TMmaColMajor::FragB(pData->bFrag[bWarpOffset + 1], k);
            b[2] = TMmaColMajor::FragB(pData->bFrag[bWarpOffset + 2], k);
            b[3] = TMmaColMajor::FragB(pData->bFrag[bWarpOffset + 3], k);
            for (int ty = 0; ty < 4; ++ty) {
                TRegTile<i8> a;
                a = TMmaRowMajor::FragA(pData->aFrag[aWarpOffset + ty], k);
                for (int tx = 0; tx < 4; ++tx) {
                    MMA(&Res.Sum[ty][tx], a, b[tx]);
                }
            }
        }
    }

    __device__ void NextTile()
    {
        aPtr += I8_TILE_GROUP_SIZE;
        bPtr += I8_TILE_GROUP_SIZE;
    }

    __device__ int GetResultOffsetX()
    {
        return bWarpOffset * TILE;
    }
    __device__ int GetResultOffsetY()
    {
        return aWarpOffset * TILE;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TStoreFunc, class TRes>
__global__ void Int8MatMulKernel(TCuda2DPtr<i8> aMatr, TCuda2DPtr<i8> bMatr, int yTiles, TCuda2DPtr<TRes> resBuf, typename TStoreFunc::TParams params)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);

    TTileCoord tc;

    // blockIdx.x - hidden dim
    // blockIdx.y - time

    int warpId = threadIdx.y;

    TInt8MatMulCtx ctx(aMatr, bMatr, warpId);
    __shared__ TInt8MatMulData<typename TStoreFunc::TShmem> data;

    for (int yTile = 0; yTile < yTiles; ++yTile) {
        __syncthreads();
        ctx.LoadData(&data, warpId);
        __syncthreads();
        ctx.Add(&data, warpId);
        ctx.NextTile();
    }
    // save result
    TStoreFunc::Store(params, data.StoreData, tc, ctx.Res, 1.0f, ctx.GetResultOffsetX(), ctx.GetResultOffsetY(), resBuf);
}
KERNEL_BLOCK_SIZE(Int8MatMulKernel, WARP_SIZE, 4);


// XY,ZY->XZ
template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &Int8MatMul(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles)
{
    return CudaCall(c, Int8MatMulKernel<TStoreFunc, typename TRes::TElem>)
        .Grid(zTiles, xTiles)
        (aMatr, bMatr, yTiles)
        .Write(pResMatr);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TStoreFunc, class TRes>
__global__ void Int8MatMulKernelRowScale(TCuda2DPtr<i8> aMatr, TCuda1DPtr<float> aRowScale, TCuda2DPtr<i8> bMatr, int yTiles, TCuda2DPtr<TRes> resBuf, typename TStoreFunc::TParams params)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);

    TTileCoord tc;

    // blockIdx.x - hidden dim
    // blockIdx.y - time

    int h = threadIdx.x;
    int warpId = threadIdx.y;

    TInt8MatMulCtx ctx(aMatr, bMatr, warpId);
    __shared__ TInt8MatMulData<typename TStoreFunc::TShmem> data;

    for (int yTile = 0; yTile < yTiles; ++yTile) {
        __syncthreads();
        ctx.LoadData(&data, warpId);
        __syncthreads();
        ctx.Add(&data, warpId);
        ctx.NextTile();
    }
    __syncthreads();
    {
        int y = warpId * WARP_SIZE + h;
        data.RowScaleArr[y] = aRowScale[blockIdx.y * MM_TILE + y];
    }
    __syncthreads();

    // inplace rescale result
    for (int ty = 0; ty < 4; ++ty) {
        TRegTileRow<float> scale;
        scale.Load(tc, data.RowScaleArr + (ctx.aWarpOffset + ty) * TILE);
        for (int tx = 0; tx < 4; ++tx) {
            TRegTile<float> &sumScaled = ctx.ResScaled.Sum[ty][tx];
            const TRegTile<int> &sum = ctx.Res.Sum[ty][tx];
            tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                sumScaled.x[elem] = sum.x[elem] * scale.x[rowIndex];
                });
        }
    }
    // save scaled result
    TStoreFunc::Store(params, data.StoreData, tc, ctx.ResScaled, 1.0f, ctx.GetResultOffsetX(), ctx.GetResultOffsetY(), resBuf);
}
KERNEL_BLOCK_SIZE(Int8MatMulKernelRowScale, WARP_SIZE, 4);


// XY,ZY->XZ
template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &Int8MatMulRowScale(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, TCudaVector<float> &aRowScale, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles)
{
    return CudaCall(c, Int8MatMulKernelRowScale<TStoreFunc, typename TRes::TElem>)
        .Grid(zTiles, xTiles)
        (aMatr, aRowScale, bMatr, yTiles)
        .Write(pResMatr);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TStoreFunc, class TRes>
__global__ void Int8MatMulYScaleKernel(TCuda2DPtr<i8> aMatr, TCuda2DPtr<i8> bMatr, TCuda1DPtr<float> yTileScale, int yTiles, TCuda2DPtr<TRes> resBuf, typename TStoreFunc::TParams params)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);

    TTileCoord tc;

    // blockIdx.x - hidden dim
    // blockIdx.y - time

    int warpId = threadIdx.y;

    TInt8MatMulCtx ctx(aMatr, bMatr, warpId);
    __shared__ TInt8MatMulData<typename TStoreFunc::TShmem> data;

    float prevTileScale = (yTiles > 0) ? yTileScale[0] : 0;
    for (int yTile = 0; yTile < yTiles; ++yTile) {
        __syncthreads();
        {
            float scale = yTileScale[yTile];
            if (scale == 0) {
                continue;
            }
            float mult = prevTileScale / scale;
            if (mult != 1) {
                for (int ty = 0; ty < 4; ++ty) {
                    for (int tx = 0; tx < 4; ++tx) {
                        ctx.Res.Sum[ty][tx].Scale(mult);
                    }
                }
            }
            prevTileScale = scale;
        }
        ctx.LoadData(&data, warpId);
        __syncthreads();
        ctx.Add(&data, warpId);
        ctx.NextTile();
    }
    // save result
    TStoreFunc::Store(params, data.StoreData, tc, ctx.Res, prevTileScale, ctx.GetResultOffsetX(), ctx.GetResultOffsetY(), resBuf);
}
KERNEL_BLOCK_SIZE(Int8MatMulYScaleKernel, WARP_SIZE, 4);


// XY,ZY->XZ
template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &Int8MatMulYScale(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TCudaVector<float> &yTileScale, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles)
{
    return CudaCall(c, Int8MatMulYScaleKernel<TStoreFunc, typename TRes::TElem>)
        .Grid(zTiles, xTiles)
        (aMatr, bMatr, yTileScale, yTiles)
        .Write(pResMatr);
}

}
