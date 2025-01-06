#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"
#include "cuda_mma.cuh"
#include "cuda_matmul.cuh"


namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
struct TMatMulDirect
{
    static __device__ int GetXStep(int stride) { (void)stride; return sizeof(T); }
    static __device__ int GetYStep(int stride) { return stride; }
    typedef TMmaRowMajor Rot;
};

template <class T>
struct TMatMulTranspose
{
    static __device__ int GetXStep(int stride) { return stride; }
    static __device__ int GetYStep(int stride) { (void)stride; return sizeof(T); }
    typedef TMmaColMajor Rot;
};


template <class TStoreData>
struct TFp16MatMulData
{
    union {
        struct {
            T4x4SMemHalfTile aFrag[2];
            T4x4SMemHalfTile bFrag[2];
        };
        TStoreData StoreData;
    };
};

template <class TStoreFunc, class TSumFloat, class ARotate, class BRotate, class TFloatA, class TFloatB>
struct TFp16MatMulCtx
{
    TMatMulWarpResult<TSumFloat> Res;
    int warpId;
    int aWarpBlk;
    int bWarpBlk;

    __device__ TFp16MatMulCtx()
    {
        Res.Clear();
        warpId = threadIdx.y;
        aWarpBlk = (warpId & 1);
        bWarpBlk = (warpId >> 1);
    }

    inline __device__ void ComputeFp16MatMul(const TTileCoord &tc, TFp16MatMulData<typename TStoreFunc::TShmem> &data, TCuda2DPtr<TFloatA> aMatr, TCuda2DPtr<TFloatB> bMatr, int yTiles)
    {
        CUDA_STATIC_ASSERT(MM_TILE == 128);

        // blockIdx.x - hidden dim
        // blockIdx.y - time

        int aStride = aMatr.GetStrideInBytes();
        int aXStep = ARotate::GetXStep(aStride);
        int aYStep = ARotate::GetYStep(aStride);

        int bStride = bMatr.GetStrideInBytes();
        int bXStep = BRotate::GetXStep(bStride);
        int bYStep = BRotate::GetYStep(bStride);

        ui8 *aPtr0 = aMatr.GetRawData() + (blockIdx.y * MM_TILE) * aYStep;
        ui8 *bPtr0 = bMatr.GetRawData() + (blockIdx.x * MM_TILE) * bXStep;
        ui8 *aPtr1 = aPtr0 + TILE_GROUP_SIZE * aYStep;
        ui8 *bPtr1 = bPtr0 + TILE_GROUP_SIZE * bXStep;

        for (int yTile = 0; yTile < yTiles * 2; ++yTile) {
            __syncthreads();
            Copy4x4Tile(&data.aFrag[0], warpId, TCuda2DPtr<TFloatA>(aPtr0, aStride, TILE_GROUP_SIZE, TILE_GROUP_SIZE));
            Copy4x4Tile(&data.aFrag[1], warpId, TCuda2DPtr<TFloatA>(aPtr1, aStride, TILE_GROUP_SIZE, TILE_GROUP_SIZE));
            Copy4x4Tile(&data.bFrag[0], warpId, TCuda2DPtr<TFloatB>(bPtr0, bStride, TILE_GROUP_SIZE, TILE_GROUP_SIZE));
            Copy4x4Tile(&data.bFrag[1], warpId, TCuda2DPtr<TFloatB>(bPtr1, bStride, TILE_GROUP_SIZE, TILE_GROUP_SIZE));
            __syncthreads();

            for (int k = 0; k < 4; ++k) {
                TRegTile<half> b[2];
                b[0] = BRotate::Rot::FragB(data.bFrag[bWarpBlk], 0, k);
                b[1] = BRotate::Rot::FragB(data.bFrag[bWarpBlk], 1, k);
                for (int ty = 0; ty < 4; ++ty) {
                    TRegTile<half> a;
                    a = ARotate::Rot::FragA(data.aFrag[aWarpBlk], k, ty);
                    for (int tx = 0; tx < 2; ++tx) {
                        MMA(&Res.Sum[ty][tx], a, b[tx]);
                    }
                }
                b[0] = BRotate::Rot::FragB(data.bFrag[bWarpBlk], 2, k);
                b[1] = BRotate::Rot::FragB(data.bFrag[bWarpBlk], 3, k);
                for (int ty = 0; ty < 4; ++ty) {
                    TRegTile<half> a;
                    a = ARotate::Rot::FragA(data.aFrag[aWarpBlk], k, ty);
                    for (int tx = 0; tx < 2; ++tx) {
                        MMA(&Res.Sum[ty][tx + 2], a, b[tx]);
                    }
                }
            }
            aPtr0 += aXStep * TILE_GROUP_SIZE;
            aPtr1 += aXStep * TILE_GROUP_SIZE;
            bPtr0 += bYStep * TILE_GROUP_SIZE;
            bPtr1 += bYStep * TILE_GROUP_SIZE;
        }
    }

    template <class TRes>
    inline __device__ void SaveResult(const TTileCoord &tc, TFp16MatMulData<typename TStoreFunc::TShmem> &data, TCuda2DPtr<TRes> resBuf, typename TStoreFunc::TParams params)
    {
        TStoreFunc::Store(params, data.StoreData, tc, Res, 1.0f, bWarpBlk * TILE_GROUP_SIZE, aWarpBlk * TILE_GROUP_SIZE, resBuf);
    }
};


template <class TStoreFunc, class TSumFloat, class ARotate, class BRotate, class TFloatA, class TFloatB, class TRes>
__global__ void Fp16MatMulKernel(TCuda2DPtr<TFloatA> aMatr, TCuda2DPtr<TFloatB> bMatr, int yTiles, TCuda2DPtr<TRes> resBuf, typename TStoreFunc::TParams params)
{
    TTileCoord tc;
    TFp16MatMulCtx<TStoreFunc, TSumFloat, ARotate, BRotate, TFloatA, TFloatB> ctx;
    __shared__ TFp16MatMulData<typename TStoreFunc::TShmem> data;
    ctx.ComputeFp16MatMul(tc, data, aMatr, bMatr, yTiles);
    ctx.SaveResult(tc, data, resBuf, params);
}
KERNEL_BLOCK_SIZE(Fp16MatMulKernel, WARP_SIZE, 4);


///////////////////////////////////////////////////////////////////////////////////////////////////
// XY,ZY->XZ
template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &Fp16MatMul(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles)
{
    typedef TMatMulDirect<typename T1::TElem> ARot;
    typedef TMatMulTranspose<typename T2::TElem> BRot;
    return CudaCall(c, Fp16MatMulKernel<TStoreFunc, float, ARot, BRot, typename T1::TElem, typename T2::TElem, typename TRes::TElem>)
        .Grid(zTiles, xTiles)
        (aMatr, bMatr, yTiles)
        .Write(pResMatr);
}


template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &MatMulXYoZYeXZ(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles)
{
    typedef TMatMulDirect<typename T1::TElem> ARot;
    typedef TMatMulTranspose<typename T2::TElem> BRot;
    return CudaCall(c, Fp16MatMulKernel<TStoreFunc, float, ARot, BRot, typename T1::TElem, typename T2::TElem, typename TRes::TElem>)
        .Grid(zTiles, xTiles)
        (aMatr, bMatr, yTiles)
        .Write(pResMatr);
}

template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &MatMulXYoZYeXZhalf(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles)
{
    typedef TMatMulDirect<typename T1::TElem> ARot;
    typedef TMatMulTranspose<typename T2::TElem> BRot;
    return CudaCall(c, Fp16MatMulKernel<TStoreFunc, half, ARot, BRot, typename T1::TElem, typename T2::TElem, typename TRes::TElem>)
        .Grid(zTiles, xTiles)
        (aMatr, bMatr, yTiles)
        .Write(pResMatr);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// XY,YZ->XZ
template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &MatMulXYoYZeXZ(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles)
{
    typedef TMatMulDirect<typename T1::TElem> ARot;
    typedef TMatMulDirect<typename T2::TElem> BRot;
    return CudaCall(c, Fp16MatMulKernel<TStoreFunc, float, ARot, BRot, typename T1::TElem, typename T2::TElem, typename TRes::TElem>)
        .Grid(zTiles, xTiles)
        (aMatr, bMatr, yTiles)
        .Write(pResMatr);
}

template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &MatMulXYoYZeXZhalf(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles)
{
    typedef TMatMulDirect<typename T1::TElem> ARot;
    typedef TMatMulDirect<typename T2::TElem> BRot;
    return CudaCall(c, Fp16MatMulKernel<TStoreFunc, half, ARot, BRot, typename T1::TElem, typename T2::TElem, typename TRes::TElem>)
        .Grid(zTiles, xTiles)
        (aMatr, bMatr, yTiles)
        .Write(pResMatr);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// XY,XZ->YZ
template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &MatMulXYoXZeYZ(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles)
{
    typedef TMatMulTranspose<typename T1::TElem> ARot;
    typedef TMatMulDirect<typename T2::TElem> BRot;
    return CudaCall(c, Fp16MatMulKernel<TStoreFunc, float, ARot, BRot, typename T1::TElem, typename T2::TElem, typename TRes::TElem>)
        .Grid(zTiles, yTiles)
        (aMatr, bMatr, xTiles)
        .Write(pResMatr);
}

template <class TStoreFunc, class T1, class T2, class TRes, class TXSize, class TYSize, class TZSize>
TKernelOp &MatMulXYoXZeYZhalf(TIntrusivePtr<TGraph> c,
    const T1 &aMatr, const T2 &bMatr, TRes *pResMatr,
    TXSize &&xTiles, TYSize &&yTiles, TZSize &&zTiles)
{
    typedef TMatMulTranspose<typename T1::TElem> ARot;
    typedef TMatMulDirect<typename T2::TElem> BRot;
    return CudaCall(c, Fp16MatMulKernel<TStoreFunc, half, ARot, BRot, typename T1::TElem, typename T2::TElem, typename TRes::TElem>)
        .Grid(zTiles, yTiles)
        (aMatr, bMatr, xTiles)
        .Write(pResMatr);
}

}
