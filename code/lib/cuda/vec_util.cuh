#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"

namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// vector utils
// one vector per warp (small dim vectors)
template <int WSZ, class TSrc>
__device__ void LoadWarpVec(float *res, const TSrc *src)
{
    int h = threadIdx.x;
    for (int k = 0; k < WSZ; ++k) {
        int d = k * WARP_SIZE + h;
        res[k] = float(src[d]);
    }
}

template <int WSZ, class TSrc>
__device__ void LoadAddWarpVec(float *res, const TSrc *src)
{
    int h = threadIdx.x;
    for (int k = 0; k < WSZ; ++k) {
        int d = k * WARP_SIZE + h;
        res[k] += float(src[d]);
    }
}

template <int WSZ, class TSrc>
__device__ void LoadMulWarpVec(float *res, const TSrc *src)
{
    int h = threadIdx.x;
    for (int k = 0; k < WSZ; ++k) {
        int d = k * WARP_SIZE + h;
        res[k] *= float(src[d]);
    }
}

template <int WSZ>
__device__ void LoadZeroWarpVec(float *res)
{
    for (int k = 0; k < WSZ; ++k) {
        res[k] = 0;
    }
}

// src & dst are local, can scale inplace
template <int WSZ>
inline __device__ void ScaleWarpVec(float *dst, float scale)
{
    for (int k = 0; k < WSZ; ++k) {
        dst[k] *= scale;
    }
}

// src & dst are local
template <int WSZ>
inline __device__ void AddWarpVec(float *dst, float *src)
{
    for (int k = 0; k < WSZ; ++k) {
        dst[k] += src[k];
    }
}

template <int WSZ>
__device__ float CalcWarpVecSum2(float *vec)
{
    float sum2 = 0;
    for (int k = 0; k < WSZ; ++k) {
        sum2 += vec[k] * vec[k];
    }
    return WarpSum(sum2);
}

template <int WSZ>
__device__ float CalcWarpVecMaxAbsValue(float *vec)
{
    float maxVal = 0;
    for (int k = 0; k < WSZ; ++k) {
        maxVal = max(maxVal, fabs(vec[k]));
    }
    return WarpMax(maxVal);
}

// src[WSZ] - not normalized source vector, grad[WSZ] - array gradient, returns gradient of pre normalization vector
template <int WSZ>
inline __device__ void StateNormalizeBackpropWarpVec(float *src, float *grad, float *dst)
{
    constexpr int STATE_DIM = WSZ * WARP_SIZE;
    float sum2 = 0;
    float dp = 0;
    for (int k = 0; k < WSZ; ++k) {
        float val = src[k];
        sum2 += val * val;
        dp += val * float(grad[k]);
    }
    sum2 = WarpSum(sum2);
    if (sum2 == 0) {
        for (int k = 0; k < WSZ; ++k) {
            dst[k] = 0;
        }
    } else {
        dp = WarpSum(dp);

        float sigma = dp / sum2;
        float scale = sqrtf(STATE_DIM / sum2);
        for (int k = 0; k < WSZ; ++k) {
            dst[k] = scale * (float(grad[k]) - float(src[k]) * sigma);
        }
    }
}


template <int WSZ, class TDst>
__device__ void StoreWarpVec(TDst *dst, float *src)
{
    int h = threadIdx.x;
    for (int k = 0; k < WSZ; ++k) {
        int d = k * WARP_SIZE + h;
        StoreConvertedFloat(src[k], &dst[d]);
    }
}


template <int WSZ, class TDst>
__device__ float NormalizeWarpVec(TDst *vec, float mult)
{
    float sum2 = CalcWarpVecSum2<WSZ>(vec);
    float discrScale = 0;
    if (sum2 > 0) {
        discrScale = sqrt(sum2 / (WSZ * WARP_SIZE));
        ScaleWarpVec<WSZ>(vec, mult / discrScale);
    }
    return discrScale;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// vector utils
// vectors reside in smem
template <class TSrc>
inline __device__ void LoadSmemVec(int dim, TSrc *src, float *buf)
{
    // could use async loads to fully utilize bandwidth with single warp?
    int h = threadIdx.x;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        int d = base + h;
        buf[d] = src[d];
    }
}


template <class TSrc>
inline __device__ void LoadAddSmemVec(int dim, TSrc *src, float *buf)
{
    // could use async loads to fully utilize bandwidth with single warp?
    int h = threadIdx.x;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        int d = base + h;
        buf[d] += float(src[d]);
    }
}


inline __device__ void ScaleSmemVec(int dim, float *buf, float mult)
{
    int h = threadIdx.x;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        buf[base + h] *= mult;
    }
}


inline __device__ float CalcSum2Smem(int dim, float *buf)
{
    int h = threadIdx.x;
    float sum2 = 0;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        int d = base + h;
        float val = buf[d];
        sum2 += val * val;
    }
    return WarpSum(sum2);
}


inline __device__ float CalcMaxAbsValueSmem(int dim, float *buf)
{
    int h = threadIdx.x;
    float maxVal = 0;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        int d = base + h;
        float val = buf[d];
        maxVal = max(maxVal, fabs(val));
    }
    return WarpMax(maxVal);
}


template <class TDst>
inline __device__ void StoreZeroSmemVec(int dim, TDst *dst)
{
    int h = threadIdx.x;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        int d = base + h;
        StoreConvertedFloat(0, &dst[d]);
    }
}


template <class TDst>
inline __device__ void StoreScaledSmemVec(int dim, float *buf, float mult, TDst *dst)
{
    int h = threadIdx.x;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        int d = base + h;
        StoreConvertedFloat(buf[d] * mult, &dst[d]);
    }
}


// res can point to src or grad
inline __device__ void BackpropNormalizeSmem(int dim, float *src, float *grad, float *res)
{
    int h = threadIdx.x;
    float sum2 = 0;
    float dp = 0;
    for (int base = 0; base < dim; base += WARP_SIZE) {
        int d = base + h;
        float val = src[d];
        float valGrad = grad[d];
        sum2 += val * val;
        dp += val * valGrad;
    }
    sum2 = WarpSum(sum2);
    if (sum2 == 0) {
        for (int base = 0; base < dim; base += WARP_SIZE) {
            int d = base + h;
            res[d] = 0;
        }
    } else {
        // add gradient and update gradMax
        dp = WarpSum(dp);

        float sigma = dp / sum2;
        float scale = sqrtf(dim / sum2);
        for (int base = 0; base < dim; base += WARP_SIZE) {
            int d = base + h;
            float resGrad = scale * (grad[d] - src[d] * sigma);
            res[d] = resGrad;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// debug kernels
template <int STATE_DIM>
__global__ void TestNan(int stepId, int id, TCuda2DPtr<float> vec)
{
    int h = threadIdx.x;
    int t = blockIdx.x;
    for (int k = 0; k < STATE_DIM / WARP_SIZE; ++k) {
        int d = k * WARP_SIZE + h;
        float val = vec[t][d];
        if (isnan(val) || !isfinite(val)) {
            printf("TestNan(%g / %g), t = %g, %g\n", stepId * 1., id * 1., t * 1., val);
            return;
        }
    }
}


template <int STATE_DIM>
__global__ void TestNanHalf(int stepId, int id, TCuda2DPtr<half> vec)
{
    int h = threadIdx.x;
    int t = blockIdx.x;
    for (int k = 0; k < STATE_DIM / WARP_SIZE; ++k) {
        int d = k * WARP_SIZE + h;
        float val = vec[t][d];
        if (isnan(val) || !isfinite(val)) {
            printf("TestNanHalf(%g / %g), t = %g, %g\n", stepId * 1., id * 1., t * 1., val);
            return;
        }
    }
}


template <int STATE_DIM, class T>
__global__ void VecsCheckSum(int len, TCuda2DPtr<T> vecs)
{
    int h = threadIdx.x;
    int chkSum = 0;
    for (int t = 0; t < len; ++t) {
        for (int k = 0; k < STATE_DIM / WARP_SIZE; ++k) {
            int d = k * WARP_SIZE + threadIdx.x;
            float val = vecs[t][d];
            chkSum += __float_as_int(val);
        }
    }
    chkSum = WarpIntSum(chkSum);
    if (h == 0) {
        printf("vecs %p, chksum %d\n", &vecs[0][0], chkSum);
    }
}


template <class T>
__global__ void PrintValue(T *p)
{
    if (threadIdx.x == 0) {
        printf("Value = %g\n", float(*p));
    }
}


template <int STATE_DIM, class T>
__global__ void PrintVec(int t, TCuda2DPtr<T> vecs)
{
    int h = threadIdx.x;
    for (int k = 0; k < STATE_DIM / WARP_SIZE; ++k) {
        int d = k * WARP_SIZE + h;
        float val = vecs[t][d];
        printf("gpu vec[%g] = %g\n", d * 1., val);
    }
}

}
