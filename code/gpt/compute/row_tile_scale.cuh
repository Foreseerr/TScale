#pragma once
#include <lib/cuda/vec_util.cuh>
#include <lib/cuda/cuda_matmul.cuh>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// convert float vectors into i8 vectors + per MM_TILE scale
template <class TSrc, class TDst>
__global__ void NormalizeVecsRowTile(int len, float vecScale, TCuda2DPtr<TSrc> srcArr, TCuda2DPtr<TDst> resArr, TCuda2DPtr<float> resScale)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;
    float vec[WSZ];
    float discrScale = 0;
    if (t < len) {
        LoadWarpVec<WSZ>(vec, srcArr[t] + offset);
        float sum2 = CalcWarpVecSum2<WSZ>(vec);
        if (sum2 > 0) {
            discrScale = sqrt(sum2 / MM_TILE) * vecScale;
            ScaleWarpVec<WSZ>(vec, 1 / discrScale);
        }
    } else {
        LoadZeroWarpVec<WSZ>(vec);
    }
    StoreWarpVec<WSZ>(resArr[t] + offset, vec);
    if (threadIdx.x == 0) {
        resScale[tile][t] = discrScale;
    }
}


// backprop through per tile normalize
template <class TSrc, class TGrad>
__global__ void BackpropRowTileNormalize(TCuda2DPtr<TSrc> src, TCuda2DPtr<float> srcTileScale, TCuda2DPtr<TGrad> grad)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;

    // normalize
    float v[WSZ];
    LoadWarpVec<WSZ>(v, src[t] + offset);
    ScaleWarpVec<WSZ>(v, srcTileScale[tile][t]);
    float vGrad[WSZ];
    LoadWarpVec<WSZ>(vGrad, grad[t] + offset);
    float stateGrad[WSZ];
    StateNormalizeBackpropWarpVec<WSZ>(v, vGrad, stateGrad);
    StoreWarpVec<WSZ>(grad[t] + offset, stateGrad);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TSrc, class TDst>
__global__ void NormalizeVecsRowTileMax(int len, TCuda2DPtr<TSrc> srcArr, TCuda2DPtr<TDst> resArr, TCuda2DPtr<float> resScale)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;
    float vec[WSZ];
    float discrScale = 0;
    if (t < len) {
        LoadWarpVec<WSZ>(vec, srcArr[t] + offset);
        float maxVal = CalcWarpVecMaxAbsValue<WSZ>(vec);
        if (maxVal > 0) {
            discrScale = GetMaxDiscrScale(maxVal, (TDst *)0);
            ScaleWarpVec<WSZ>(vec, 1 / discrScale);
        }
    } else {
        LoadZeroWarpVec<WSZ>(vec);
    }
    StoreWarpVec<WSZ>(resArr[t] + offset, vec);
    if (threadIdx.x == 0) {
        resScale[tile][t] = discrScale;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TDst>
__global__ void FillArray(float val, TCuda2DPtr<TDst> resScale)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    resScale[tile][t] = val;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TSrc, class TDst>
__global__ void ConvertMatrixScaled(int len, TCuda2DPtr<TSrc> srcArr, float mult, TCuda2DPtr<TDst> resArr)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;
    float vec[WSZ];
    if (t < len) {
        LoadWarpVec<WSZ>(vec, srcArr[t] + offset);
    } else {
        LoadZeroWarpVec<WSZ>(vec);
    }
    ScaleWarpVec<WSZ>(vec, mult);
    StoreWarpVec<WSZ>(resArr[t] + offset, vec);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TDst>
__global__ void MatrixRelu(TCuda2DPtr<TDst> srcArr, TCuda2DPtr<TDst> resArr)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;
    float vec[WSZ];
    LoadWarpVec<WSZ>(vec, srcArr[t] + offset);
    for (int k = 0; k < WSZ; ++k) {
        if (vec[k] < 0) {
            vec[k] = 0;
        }
    }
    StoreWarpVec<WSZ>(resArr[t] + offset, vec);
}


template <class TSrc, class TGrad>
__global__ void BackpropRowTileNormalizeRelu(TCuda2DPtr<TSrc> src, TCuda2DPtr<float> srcTileScale, TCuda2DPtr<TGrad> grad)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;

    // load src with original scale
    float v[WSZ];
    LoadWarpVec<WSZ>(v, src[t] + offset);
    ScaleWarpVec<WSZ>(v, srcTileScale[tile][t]);
    float vGrad[WSZ];
    LoadWarpVec<WSZ>(vGrad, grad[t] + offset);

    // backprop relu
    for (int k = 0; k < WSZ; ++k) {
        if (v[k] < 0) {
            vGrad[k] = 0;
        }
    }

    // backprop row tile normalize
    float stateGrad[WSZ];
    StateNormalizeBackpropWarpVec<WSZ>(v, vGrad, stateGrad);
    StoreWarpVec<WSZ>(grad[t] + offset, stateGrad);
}
}
