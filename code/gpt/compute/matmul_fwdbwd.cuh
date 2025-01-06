#pragma once
#include "row_scale.cuh"
#include "row_tile_scale.cuh"
#include "par_matrix_cuda.cuh"
#include <gpt/model_params/model_dim.h>
#include <lib/cuda/cuda_sort.cuh>
#include <lib/cuda/cuda_i8.cuh>
#include <lib/cuda/cuda_fp8.cuh>
#include <lib/cuda/cuda_fp16.cuh>
#include <lib/cuda/vec_util.cuh>


namespace NCuda
{

constexpr bool DETERMINISTIC = true;
//constexpr bool DETERMINISTIC = false;


///////////////////////////////////////////////////////////////////////////////////////////////////
// forward matmul
template <class TStoreFunc, class TSrc, class TTransofrmFloat, class TDst, class TParamsType>
TKernelOp& MulForwardInt8(TIntrusivePtr<TGraph> c, TParamsType *pParams,
    int srcDim, int dstDim,
    TSrc &src, TIntrusivePtr<TCudaModelMatrix<TTransofrmFloat>> pTransform,
    TCuda2DArray<TDst> *pRes)
{
    int srcTiles = srcDim / MM_TILE;
    int dstTiles = dstDim / MM_TILE;
    return Int8MatMul<TStoreFunc>(c, src, pTransform->GetFast(), pRes, pParams->LenTiles, srcTiles, dstTiles).Struct();
}


template <class TStoreFunc, class TSrc, class TTransofrmFloat, class TDst, class TParamsType>
TKernelOp &MulForwardFp8(TIntrusivePtr<TGraph> c, TParamsType *pParams,
    int srcDim, int dstDim,
    TSrc &src, TIntrusivePtr<TCudaModelMatrix<TTransofrmFloat>> pTransform,
    TCuda2DArray<TDst> *pRes)
{
    int srcTiles = srcDim / MM_TILE;
    int dstTiles = dstDim / MM_TILE;
    return Fp8MatMul<TStoreFunc>(c, src, pTransform->GetFast(), pRes, pParams->LenTiles, srcTiles, dstTiles).Struct();
}


template <class TStoreFunc, class TSrc, class TTransofrmFloat, class TDst, class TParamsType>
TKernelOp &MulForwardFp16Half(TIntrusivePtr<TGraph> c, TParamsType *pParams,
    int srcDim, int dstDim,
    TSrc &src, TIntrusivePtr<TCudaModelMatrix<TTransofrmFloat>> pTransform,
    TCuda2DArray<TDst> *pRes)
{
    int srcTiles = srcDim / MM_TILE;
    int dstTiles = dstDim / MM_TILE;
    return MatMulXYoZYeXZhalf<TStoreFunc>(c, src, pTransform->GetFast(), pRes, pParams->LenTiles, srcTiles, dstTiles).Struct();
}


template <class TStoreFunc, class TSrc, class TTransofrmFloat, class TDst, class TParamsType>
TKernelOp &MulForwardFp16(TIntrusivePtr<TGraph> c, TParamsType *pParams,
    int srcDim, int dstDim,
    TSrc &src, TIntrusivePtr<TCudaModelMatrix<TTransofrmFloat>> pTransform,
    TCuda2DArray<TDst> *pRes)
{
    int srcTiles = srcDim / MM_TILE;
    int dstTiles = dstDim / MM_TILE;
    return MatMulXYoZYeXZ<TStoreFunc>(c, src, pTransform->GetFast(), pRes, pParams->LenTiles, srcTiles, dstTiles).Struct();
}


///////////////////////////////////////////////////////////////////////////////////////////////////
enum EResultStore {
    RESULT_GRAD_MOV,
    RESULT_GRAD_ADD,
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// fp16 backprop matmul
template <class TSrc, class TTransofrmFloat, class TSrcGradType, class TParamsType>
void BackpropMatMulFp16(TIntrusivePtr<TGraph> c, TParamsType *pParams,
    int srcDim, int gradDim,
    TCuda2DArray<TSrc> &src, TIntrusivePtr<TCudaModelMatrix<TTransofrmFloat>> pTransform, TCuda2DArray<half> &grad, TCudaPOD<float> gradScale,
    EResultStore rs,
    TSrcGradType *pSrcGrad, TPackedDeltaMatrix *pPackedDeltaMatrix)
{
    int srcTiles = srcDim / MM_TILE;
    int gradTiles = gradDim / MM_TILE;
    float normScale = CalcDotScale(srcDim) * MODEL_DISCR_SCALE;
    //TCudaPOD<float> trScale = pTransform->GetScale();
    if (rs == RESULT_GRAD_MOV) {
        MatMulXYoYZeXZ<TStoreScaled>(c, grad, pTransform->GetFast(), pSrcGrad, pParams->LenTiles, gradTiles, srcTiles).Struct()(nullptr, normScale);
    } else {
        MatMulXYoYZeXZ<TStoreAddScaled>(c, grad, pTransform->GetFast(), pSrcGrad, pParams->LenTiles, gradTiles, srcTiles).Struct()(nullptr, normScale);
    }

    // former sum rank one
    MatMulXYoXZeYZ<TStoreRowTileMaxNormalize>(c, grad, src, &pPackedDeltaMatrix->Data, pParams->LenTiles, gradTiles, srcTiles).Struct()(gradScale).Write(&pPackedDeltaMatrix->ScaleBuf);
}


template <class TSrc, class TTransofrmFloat, class TSrcGradType, class TParamsType>
void BackpropMatMulFp16Half(TIntrusivePtr<TGraph> c, TParamsType *pParams,
    int srcDim, int gradDim, float srcGradMult,
    TCuda2DArray<TSrc> &src, TIntrusivePtr<TCudaModelMatrix<TTransofrmFloat>> pTransform, TCuda2DArray<half> &grad, TCudaPOD<float> gradScale,
    EResultStore rs,
    TSrcGradType *pSrcGrad, TPackedDeltaMatrix *pPackedDeltaMatrix)
{
    int srcTiles = srcDim / MM_TILE;
    int gradTiles = gradDim / MM_TILE;
    float normScale = CalcDotScale(srcDim) * MODEL_DISCR_SCALE / srcGradMult;
    //TCudaPOD<float> trScale = pTransform->GetScale();
    if (rs == RESULT_GRAD_MOV) {
        MatMulXYoYZeXZ<TStoreScaled>(c, grad, pTransform->GetFast(), pSrcGrad, pParams->LenTiles, gradTiles, srcTiles).Struct()(nullptr, normScale);
    } else {
        MatMulXYoYZeXZ<TStoreAddScaled>(c, grad, pTransform->GetFast(), pSrcGrad, pParams->LenTiles, gradTiles, srcTiles).Struct()(nullptr, normScale);
    }

    // former sum rank one
    MatMulXYoXZeYZhalf<TStoreRowTileMaxNormalize>(c, grad, src, &pPackedDeltaMatrix->Data, pParams->LenTiles, gradTiles, srcTiles).Struct()(gradScale).Write(&pPackedDeltaMatrix->ScaleBuf);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TFp8MatMulBwdBuffers
{
    typedef e4m3 TFastGradFloat; // seems to be sufficient and is more precise
    //typedef e5m2 TFastGradFloat;

    int SrcDim = 0;
    int GradDim = 0;
    TCuda2DArray<TFastGradFloat> GradE5;
    TCuda2DArray<TFastGradFloat> GradE5T;
    TCuda2DArray<e4m3> SrcE4T;
    TCuda2DArray<e4m3> TransformT;

    void AllocateCuda(int srcDim, int gradDim, int maxLen, TIntrusivePtr<TCudaMemoryPool> pool)
    {
        SrcDim = srcDim;
        GradDim = gradDim;
        GradE5.AllocateCuda(gradDim, maxLen, pool);
        GradE5T.AllocateCuda(maxLen, gradDim, pool);
        SrcE4T.AllocateCuda(maxLen, srcDim, pool);
        TransformT.AllocateCuda(gradDim, srcDim, pool);
    }

    template <class TSrc, class TTransofrmFloat, class TGradFloat, class TSrcGradType, class TParamsType>
    void BackpropMatMul(TIntrusivePtr<TGraph> c, TParamsType *pParams,
        TCuda2DArray<TSrc> &src, TIntrusivePtr<TCudaModelMatrix<TTransofrmFloat>> pTransform, TCuda2DArray<TGradFloat> &grad, TCudaPOD<float> gradScale,
        EResultStore rs,
        TSrcGradType *pSrcGrad, TPackedDeltaMatrix *pPackedDeltaMatrix
    )
    {
        Y_VERIFY(0 && "unsupoorted combination of model and gradient/state float types");
    }

    template <class TGradFloat, class TSrcGradType, class TParamsType>
    void BackpropMatMul(TIntrusivePtr<TGraph> c, TParamsType *pParams,
        TCuda2DArray<e4m3> &src, TIntrusivePtr<TCudaModelMatrix<e4m3>> pTransform, TCuda2DArray<TGradFloat> &grad, TCudaPOD<float> gradScale,
        EResultStore rs,
        TSrcGradType *pSrcGrad, TPackedDeltaMatrix *pPackedDeltaMatrix
    )
    {
        int gradTiles = GradDim / MM_TILE;
        int srcTiles = SrcDim / MM_TILE;
        float mulForwardScale = CalcDotScale(SrcDim) * MODEL_DISCR_SCALE;

        CudaCall(c, ConvertMatrixScaled<TGradFloat, TFastGradFloat>).Grid(gradTiles, pParams->LenRound)(pParams->Len, grad, 1.0f).Write(&GradE5);

        // mul backward
        Transpose(c, pTransform->GetFast(), srcTiles, gradTiles, &TransformT);
        if (rs == RESULT_GRAD_MOV) {
            Fp8MatMul<TStoreScaled>(c, GradE5, TransformT, pSrcGrad, pParams->LenTiles, gradTiles, srcTiles).Struct()(nullptr, mulForwardScale);
        } else {
            Fp8MatMul<TStoreAddScaled>(c, GradE5, TransformT, pSrcGrad, pParams->LenTiles, gradTiles, srcTiles).Struct()(nullptr, mulForwardScale);
        }

        // former sum rank one
        Transpose(c, src, srcTiles, pParams->LenTiles, &SrcE4T);
        Transpose(c, GradE5, gradTiles, pParams->LenTiles, &GradE5T);
        Fp8MatMul<TStoreRowTileMaxNormalize>(c, GradE5T, SrcE4T, &pPackedDeltaMatrix->Data, gradTiles, pParams->LenTiles, srcTiles).Struct()(gradScale).Write(&pPackedDeltaMatrix->ScaleBuf);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// int8 backprop matmul

static __global__ void ShuffleTransposeKernel(TCuda2DPtr<i8> src, TCuda1DPtr<TSortNode> sortNode, int rowCount, TCuda2DPtr<i8> dst)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);
    int h = threadIdx.x;
    int xBlock = blockIdx.x * MM_TILE;
    int yBlock = blockIdx.y * MM_TILE;

    __shared__ i8 buf[128][128];
    constexpr int yStep = WARP_SIZE / 8;
    int yOffset = h / 8;
    int xOffset = 16 * (h & 7);
    for (int yBase = 0; yBase < 128; yBase += yStep) {
        int y = yBase + yOffset;
        int t = yBlock + y;
        int4 *pDst = (int4 *)&buf[y][xOffset];
        if (t < rowCount) {
            int nodeId = sortNode[t].NodeId;
            *pDst = *(int4 *)&src[nodeId][xBlock + xOffset];
        } else {
            *pDst = make_int4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    for (int yBase = 0; yBase < 128; yBase += yStep) {
        int y = yBase + yOffset;
        union {
            int4 column;
            i8 columnBytes[16];
        };
        for (int k = 0; k < 16; ++k) {
            int x = xOffset + k;
            columnBytes[k] = buf[x][y];
        }
        int4 *pDst = (int4 *)&dst[xBlock + y][yBlock + xOffset];
        *pDst = column;
    }
}


template <class TSrcFloat>
__global__ void ShuffleMaxScaleTransposeKernel(TCuda2DPtr<TSrcFloat> src, TCuda1DPtr<TSortNode> sortNode, int rowCount, TCuda2DPtr<i8> dst, TCuda1DPtr<float> tileScale)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);
    int h = threadIdx.x;
    int xBlock = blockIdx.x * MM_TILE;
    int yBlock = blockIdx.y * MM_TILE;

    float myMaxScale = 0;
    for (int k = h; k < 128; k += WARP_SIZE) {
        int t = yBlock + k;
        if (t < rowCount) {
            myMaxScale = fmaxf(myMaxScale, sortNode[t].Score);
        }
    }
    float maxScale = WarpMax(myMaxScale);
    if (maxScale == 0) {
        maxScale = 1;
    }
    maxScale = RoundFloatUp(maxScale);

    TCudaRngLCG rng(*(ui32 *)&myMaxScale, xBlock + *(ui32 *)&maxScale, yBlock + h);
    rng.Gen();

    __shared__ i8 buf[128][128];
    constexpr int yStep = WARP_SIZE / 8;
    int yOffset = h / 8;
    int xOffset = 16 * (h & 7);
    float mult = 1 / maxScale;
    for (int yBase = 0; yBase < 128; yBase += yStep) {
        int y = yBase + yOffset;
        int t = yBlock + y;
        int4 *pDst = (int4 *)&buf[y][xOffset];
        if (t < rowCount) {
            int nodeId = sortNode[t].NodeId;
            //float mult = sortNode[t].Score / maxScale;
            union {
                int4 row;
                i8 rowBytes[16];
            };
            //row = *(int4 *)&src[nodeId][xBlock + xOffset];
            //for (int k = 0; k < 16; ++k) {
            //    rowBytes[k] = CvtToI8(rowBytes[k] * mult);
            //}
            for (int k = 0; k < 16; ++k) {
                float rshift = rng.GenUniformFloat() - 0.5f;
                rowBytes[k] = CvtToI8(float(src[nodeId][xBlock + xOffset + k]) * mult + rshift); // slow, uncoalesced loads, use half instead of float source
            }
            *pDst = row;
        } else {
            *pDst = make_int4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    for (int yBase = 0; yBase < 128; yBase += yStep) {
        int y = yBase + yOffset;
        union {
            int4 column;
            i8 columnBytes[16];
        };
        for (int k = 0; k < 16; ++k) {
            int x = xOffset + k;
            columnBytes[k] = buf[x][y];
        }
        int4 *pDst = (int4 *)&dst[xBlock + y][yBlock + xOffset];
        *pDst = column;
    }
    if (h == 0 && xBlock == 0) {
        tileScale[blockIdx.y] = maxScale;
    }
}


struct TInt8MatMulBwdBuffers
{
    int SrcDim = 0;
    int GradDim = 0;
    TCuda2DArray<i8> Grad8;
    TCudaVector<float> Grad8RowScale;
    TCuda2DArray<i8> Grad8T;
    TCuda2DArray<i8> Src8T;
    TCudaVector<TSortNode> SortedSamples;
    TCudaVector<float> SortedTileScale;
    TCuda2DArray<i8> TransformT;

    void AllocateCuda(int srcDim, int gradDim, int maxLen, TIntrusivePtr<TCudaMemoryPool> pool)
    {
        SrcDim = srcDim;
        GradDim = gradDim;
        Grad8.AllocateCuda(gradDim, maxLen, pool);
        Grad8RowScale.AllocateCuda(maxLen, pool);
        Grad8T.AllocateCuda(maxLen, gradDim, pool);
        Src8T.AllocateCuda(maxLen, srcDim, pool);
        SortedSamples.AllocateCuda(maxLen, pool);
        SortedTileScale.AllocateCuda(maxLen / MM_TILE, pool);
        TransformT.AllocateCuda(gradDim, srcDim, pool);
    }

    template <class TSrc, class TTransofrmFloat, class TGradFloat, class TSrcGradType, class TParamsType>
    void BackpropMatMul(TIntrusivePtr<TGraph> c, TParamsType *pParams,
        TCuda2DArray<TSrc> &src, TIntrusivePtr<TCudaModelMatrix<TTransofrmFloat>> pTransform, TCuda2DArray<TGradFloat> &grad, TCudaPOD<float> gradScale,
        EResultStore rs,
        TSrcGradType *pSrcGrad, TPackedDeltaMatrix *pPackedDeltaMatrix
    )
    {
        Y_VERIFY(0 && "unsupoorted combination of model and gradient/state float types");
    }

    template <class TGradFloat, class TSrcGradType, class TParamsType>
    void BackpropMatMul(TIntrusivePtr<TGraph> c, TParamsType *pParams,
        TCuda2DArray<i8> &src, TIntrusivePtr<TCudaModelMatrix<i8>> pTransform, TCuda2DArray<TGradFloat> &grad, TCudaPOD<float> gradScale,
        EResultStore rs,
        TSrcGradType *pSrcGrad, TPackedDeltaMatrix *pPackedDeltaMatrix
    )
    {
        int gradTiles = GradDim / MM_TILE;
        int srcTiles = SrcDim / MM_TILE;
        //int tailTiles = MODEL_MATMUL_EXACT_BUF / MM_TILE;
        float mulForwardScale = CalcDotScale(SrcDim) * MODEL_DISCR_SCALE;

        NormalizeVecsRowMaxWithNoise(c, GradDim, pParams->Len, pParams->LenRound, grad, &Grad8, &Grad8RowScale);
        if (DETERMINISTIC) {
            //SortFloats(c, Grad8RowScale, pParams->Len, &SortedSamples);
            SortPositiveFloatsApproxStable(c, Grad8RowScale, pParams->Len, &SortedSamples);
        } else {
            SortPositiveFloatsApproxFast(c, Grad8RowScale, pParams->Len, &SortedSamples);
        }

        // mul backward
        Transpose(c, pTransform->GetFast(), srcTiles, gradTiles, &TransformT);
        if (rs == RESULT_GRAD_MOV) {
            Int8MatMulRowScale<TStoreScaled>(c, Grad8, Grad8RowScale, TransformT, pSrcGrad, pParams->LenTiles, gradTiles, srcTiles).Struct()(nullptr, mulForwardScale);
        } else {
            Int8MatMulRowScale<TStoreAddScaled>(c, Grad8, Grad8RowScale, TransformT, pSrcGrad, pParams->LenTiles, gradTiles, srcTiles).Struct()(nullptr, mulForwardScale);
        }

        // delta matrix
        // use high precision source to avoid precision loss in int8 reconversion to same scaling factor per sample
        // add noise to increase stability
        CudaCall(c, ShuffleMaxScaleTransposeKernel<TGradFloat>).Grid(gradTiles, pParams->LenTiles)(grad, SortedSamples, pParams->Len).Write(&Grad8T, &SortedTileScale);
        CudaCall(c, ShuffleTransposeKernel).Grid(srcTiles, pParams->LenTiles)(src, SortedSamples, pParams->Len).Write(&Src8T);
        Int8MatMulYScale<TStoreRowTileMaxNormalize>(c, Grad8T, Src8T, SortedTileScale, &pPackedDeltaMatrix->Data, gradTiles, pParams->LenTiles, srcTiles).Struct()(gradScale).Write(&pPackedDeltaMatrix->ScaleBuf);
    }
};

}
