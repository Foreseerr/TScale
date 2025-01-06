#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <util/fp8.h>


namespace NCuda
{
const int WARP_SIZE = 32;


///////////////////////////////////////////////////////////////////////////////////////////////////
namespace staticassert
{
    template <bool x> struct CheckStruct;
    template <> struct CheckStruct<true> { int X; };// enum { value = 1 };
    template<int x> struct test {};
};

#define CUDA_STATIC_ASSERT( B )  typedef staticassert::test<sizeof(staticassert::CheckStruct< (bool)(B) >) > static_assert_chk_ ## __LINE__


///////////////////////////////////////////////////////////////////////////////////////////////////
// 
inline __device__ i32 CvtToI32(float x)
{
    int32_t res;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(res) : "f"(x));
    return res;
}

inline __device__ i8 CvtToI8(float x)
{
    int32_t res;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(res) : "f"(x));
    return res;
}


template <class TDst>
inline __device__ void StoreConvertedFloat(float x, TDst *p)
{
    *p = (TDst)x;
}

inline __device__ void StoreConvertedFloat(float x, half *p)
{
    asm volatile("cvt.rn.satfinite.f16.f32 %0, %1;" : "=h"(*(ui16 *)p) : "f"(x));
}

inline __device__ void StoreConvertedFloat(float x, i8 *p)
{
    *p = CvtToI8(x);
}

inline __device__ void StoreConvertedFloat(float x, e4m3 *p)
{
    p->Data = CvtToE4(x);
}

inline __device__ void StoreConvertedFloat(float x, e5m2 *p)
{
    p->Data = CvtToE5(x);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TDst>
inline __device__ void StoreZero(TDst *p)
{
    *p = 0;
}

inline __device__ void StoreZero(e4m3 *p)
{
    p->Data = 0;
}

inline __device__ void StoreZero(e5m2 *p)
{
    p->Data = 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void StoreConvertedFloat2(float2 val, float *p)
{
    *(float2 *)p = val;
}

inline __device__ void StoreConvertedFloat2(float2 val, half *p)
{
    asm volatile("cvt.rn.satfinite.f16x2.f32 %0, %2, %1;" : "=r"(*(ui32*)p) : "f"(val.x), "f"(val.y));
}

inline __device__ void StoreConvertedFloat2(float2 val, i8 *p)
{
    union {
        i8 val8[2];
        ui16 val16;
    };
    val8[0] = CvtToI8(val.x);
    val8[1] = CvtToI8(val.y);
    *(ui16 *)p = val16;
}

inline __device__ void StoreConvertedFloat2(float2 val, e4m3 *p)
{
#if (__CUDA_ARCH__ >= 890)
    float fHigh = val.y;
    float fLow = val.x;
    ui16 val16;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n" : "=h"(val16) : "f"(fLow), "f"(fHigh));
    *(ui16 *)p = val16;
#else
    printf("sm89 required\n");
#endif
}

inline __device__ void StoreConvertedFloat2(float2 val, e5m2 *p)
{
#if (__CUDA_ARCH__ >= 890)
    float fHigh = val.y;
    float fLow = val.x;
    ui16 val16;
    asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;\n" : "=h"(val16) : "f"(fLow), "f"(fHigh));
    *(ui16 *)p = val16;
#else
    printf("sm89 required\n");
#endif
}


inline __device__ void StoreScaledFloat2(float2 val, float *p, float mult)
{
    *(float2 *)p = make_float2(val.x * mult, val.y * mult);
}

inline __device__ void StoreScaledFloat2(float2 val, half *p, float mult)
{
    *(half2 *)p = make_half2(val.x * mult, val.y * mult);
}


inline __device__ void StoreAddScaledFloat2(float2 val, float *p, float mult)
{
    float2 old = *(float2 *)p;
    *(float2 *)p = make_float2(old.x + val.x * mult, old.y + val.y * mult);
}

inline __device__ void StoreAddScaledFloat2(float2 val, half *p, float mult)
{
    half2 old = *(half2 *)p;
    *(half2 *)p = old + make_half2(half(val.x * mult), half(val.y * mult));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void StoreConvertedFloat2(half2 val, float *p)
{
    *(float2 *)p = make_float2(val.x, val.y);
}

inline __device__ void StoreConvertedFloat2(half2 val, half *p)
{
    *(half2 *)p = val;
}

inline __device__ void StoreConvertedFloat2(half2 val, i8 *p)
{
    union {
        i8 val8[2];
        ui16 val16;
    };
    val8[0] = CvtToI8(val.x);
    val8[1] = CvtToI8(val.y);
    *(ui16 *)p = val16;
}

inline __device__ void StoreConvertedFloat2(half2 val, e4m3 *p)
{
#if (__CUDA_ARCH__ >= 890)
    int valSrc = *(int *)&val;
    ui16 val16;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(val16) : "r"(valSrc));
    *(ui16 *)p = val16;
#else
    printf("sm89 required\n");
#endif
}

inline __device__ void StoreConvertedFloat2(half2 val, e5m2 *p)
{
#if (__CUDA_ARCH__ >= 890)
    int valSrc = *(int *)&val;
    ui16 val16;
    asm volatile("cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;\n" : "=h"(val16) : "r"(valSrc));
    *(ui16 *)p = val16;
#else
    printf("sm89 required\n");
#endif
}


inline __device__ void StoreScaledFloat2(half2 val, float *p, float mult)
{
    *(float2 *)p = make_float2(float(val.x) * mult, float(val.y) * mult);
}

inline __device__ void StoreScaledFloat2(half2 val, half *p, float mult)
{
    *(half2 *)p = val * make_half2(mult, mult);
}


inline __device__ void StoreAddScaledFloat2(half2 val, float *p, float mult)
{
    float2 old = *(float2 *)p;
    *(float2 *)p = make_float2(old.x + float(val.x) * mult, old.y + float(val.y) * mult);
}

inline __device__ void StoreAddScaledFloat2(half2 val, half *p, float mult)
{
    half2 old = *(half2 *)p;
    *(half2 *)p = old + val * make_half2(mult, mult);
}



///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void StoreConvertedFloat2(int2 val, int *p)
{
    *(int2 *)p = val;
}

inline __device__ void StoreConvertedFloat2(int2 val, float *p)
{
    *(float2 *)p = make_float2(val.x, val.y);
}

inline __device__ void StoreConvertedFloat2(int2 val, half *p)
{
    *(half2 *)p = make_half2(val.x, val.y);
}


inline __device__ void StoreScaledFloat2(int2 val, int *p, float mult)
{
    *(int2 *)p = make_int2(CvtToI32(val.x * mult), CvtToI32(val.y * mult));
}

inline __device__ void StoreScaledFloat2(int2 val, float *p, float mult)
{
    *(float2 *)p = make_float2(val.x * mult, val.y * mult);
}

inline __device__ void StoreScaledFloat2(int2 val, half *p, float mult)
{
    *(half2 *)p = make_half2(val.x * mult, val.y * mult);
}


inline __device__ void StoreAddScaledFloat2(int2 val, int *p, float mult)
{
    int2 old = *(int2*)p;
    *(int2 *)p = make_int2(old.x + CvtToI32(val.x * mult), old.y + CvtToI32(val.y * mult));
}

inline __device__ void StoreAddScaledFloat2(int2 val, float *p, float mult)
{
    float2 old = *(float2 *)p;
    *(float2 *)p = make_float2(old.x + val.x * mult, old.y + val.y * mult);
}

inline __device__ void StoreAddScaledFloat2(int2 val, half *p, float mult)
{
    half2 old = *(half2 *)p;
    *(half2 *)p = old + make_half2(half(val.x * mult), half(val.y * mult));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// discr scale for normalizations to max value
template <class T>
static __device__ float GetMaxDiscrScale(float maxValue, T *)
{
    Y_VERIFY(0);
}
static __device__ float GetMaxDiscrScale(float maxValue, i8 *)
{
    return maxValue / 127;
}
static __device__ float GetMaxDiscrScale(float maxValue, e4m3 *)
{
    return maxValue / 256;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float RoundFloatUp(float x)
{
    // round scale to avoid precision loss in I8MatMulXYoZYeXZlarge() due to long chain of sum multiplication by close to 1 numbers
    float tail = __int_as_float(__float_as_int(x) | 0xfffff) - __int_as_float(__float_as_int(x) & 0xfff00000);
    float res = __int_as_float(__float_as_int(x + tail) & 0xfff00000); // round up
    return res;
}

inline __device__ float TruncateToPow2(float x)
{
    return __int_as_float(__float_as_int(x) & 0xff800000);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// warp sum
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
inline __device__ float WarpSum(float val)
{
    Y_ASSERT(WARP_SIZE == 32);
    float sum = val;
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    return sum;
}

inline __device__ int WarpIntSum(int val)
{
    Y_ASSERT(WARP_SIZE == 32);
    int sum = val;
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    return sum;
}

inline __device__ float HalfWarpSum(float val)
{
    Y_ASSERT(WARP_SIZE == 32);
    float sum = val;
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    return sum;
}

inline __device__ float WarpMax(float val)
{
    Y_ASSERT(WARP_SIZE == 32);
    float res = val;
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 16));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 8));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 4));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 2));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 1));
    return res;
}

inline __device__ float HalfWarpMax(float val)
{
    Y_ASSERT(WARP_SIZE == 32);
    float res = val;
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 8));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 4));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 2));
    res = fmaxf(res, __shfl_xor_sync(0xffffffff, res, 1));
    return res;
}

inline __device__ int WarpMinInt(int val)
{
    Y_ASSERT(WARP_SIZE == 32);
    int res = val;
    res = min(res, __shfl_xor_sync(0xffffffff, res, 16));
    res = min(res, __shfl_xor_sync(0xffffffff, res, 8));
    res = min(res, __shfl_xor_sync(0xffffffff, res, 4));
    res = min(res, __shfl_xor_sync(0xffffffff, res, 2));
    res = min(res, __shfl_xor_sync(0xffffffff, res, 1));
    return res;
}

inline __device__ int WarpMaxInt(int val)
{
    Y_ASSERT(WARP_SIZE == 32);
    int res = val;
    res = max(res, __shfl_xor_sync(0xffffffff, res, 16));
    res = max(res, __shfl_xor_sync(0xffffffff, res, 8));
    res = max(res, __shfl_xor_sync(0xffffffff, res, 4));
    res = max(res, __shfl_xor_sync(0xffffffff, res, 2));
    res = max(res, __shfl_xor_sync(0xffffffff, res, 1));
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <int BLOCK_SIZE>
inline __device__ float BlockSum(float x)
{
    __shared__ float val[BLOCK_SIZE];
    __syncthreads();
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    float sum = WarpSum(x);
    if (h == 0) {
        val[warpId] = sum;
    }
    __syncthreads();
    sum = val[0];
    for (int k = 1; k < BLOCK_SIZE; ++k) {
        sum += val[k];
    }
    return sum;
}


template <int BLOCK_SIZE>
inline __device__ float BlockMax(float x)
{
    __shared__ float val[BLOCK_SIZE];
    __syncthreads();
    int h = threadIdx.x;
    int warpId = threadIdx.y;
    float maxVal = WarpMax(x);
    if (h == 0) {
        val[warpId] = maxVal;
    }
    __syncthreads();
    maxVal = val[0];
    for (int k = 1; k < BLOCK_SIZE; ++k) {
        maxVal = fmaxf(maxVal, val[k]);
    }
    return maxVal;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ void atomicAddExact(float *pDst, float val)
{
    int *p = (int*)pDst;
    for (;;) {
        int assumed = *p;// assumed = old;
        if (atomicCAS(p, assumed, __float_as_int(val + __int_as_float(assumed))) == assumed) {
            return;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline __device__ T *AdvancePtr(T *p, int offset)
{
    return (T *)(((char *)p) + offset);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TCudaRngLCG
{
    ui32 State;

    __device__ TCudaRngLCG(ui32 a, ui32 b, ui32 c)
    {
        State = a * 0x398bf183 + b * 0x1affc3df + c * 0x9023a049;
    }
    __device__ ui32 Gen()
    {
        State = State * 0x448e3079 + 0xc484ef10;
        return State;
    }
    __device__ float GenUniformFloat()
    {
        return Gen() * (1.0f / float(1ll << 32));
    }
};
}
