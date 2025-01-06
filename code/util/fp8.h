#pragma once


inline float ConvertE4toFloat(i8 x)
{
    // slow implementation
    if (x == 0) {
        return 0;
    }
    int bias = 7;
    //int exponent_bits = 4;
    int mantissa_bits = 3;
    float sign = (x & 0x80) ? -1 : 1;
    int e = (x >> 3) & 15;
    int m = x & 7;
    if (e == 0 && m > 0) {
        return sign * exp2f(1 - bias) * (0 + exp2f(-mantissa_bits) * m);
    } else {
        return sign * exp2f(e - bias) * (1 + exp2f(-mantissa_bits) * m);
    }
}


inline float ConvertE5toFloat(i8 x)
{
    // slow implementation
    if (x == 0) {
        return 0;
    }
    int bias = 15;
    //int exponent_bits = 5;
    int mantissa_bits = 2;
    float sign = (x & 0x80) ? -1 : 1;
    int e = (x >> 2) & 31;
    int m = x & 3;
    if (e == 0 && m > 0) {
        return sign * exp2f(1 - bias) * (0 + exp2f(-mantissa_bits) * m);
    } else {
        return sign * exp2f(e - bias) * (1 + exp2f(-mantissa_bits) * m);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
#include <cuda_fp16.h>

inline __device__ i8 CvtToE4(float x)
{
#if (__CUDA_ARCH__ >= 890)
    float fHigh = 0;
    float fLow = x;
    ui16 p2;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n" : "=h"(p2) : "f"(fLow), "f"(fHigh));
    return p2 & 0xff;
#else
    printf("sm89 required\n");
    return 0;
#endif
}

inline __device__ half CvtE4ToHalf(i8 x)
{
#if (__CUDA_ARCH__ >= 890)
    ui16 p2 = x;
    union {
        half2 out2;
        int out2r;
    };
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out2r) : "h"(p2));
    return out2.x;
#else
    printf("sm89 required\n");
    return 0;
#endif
}

inline __device__ float CvtE4ToFloat(i8 x)
{
    return float(CvtE4ToHalf(x));
}

inline __device__ i8 CvtToE5(float x)
{
#if (__CUDA_ARCH__ >= 890)
    float fHigh = 0;
    float fLow = x;
    ui16 p2;
    asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;\n" : "=h"(p2) : "f"(fLow), "f"(fHigh));
    return p2 & 0xff;
#else
    printf("sm89 required\n");
    return 0;
#endif
}

inline __device__ half CvtE5ToHalf(i8 x)
{
#if (__CUDA_ARCH__ >= 890)
    ui16 p2 = x;
    union {
        half2 out2;
        int out2r;
    };
    asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(out2r) : "h"(p2));
    return out2.x;
#else
    printf("sm89 required\n");
    return 0;
#endif
}

inline __device__ float CvtE5ToFloat(i8 x)
{
    return float(CvtE5ToHalf(x));
}

#endif


///////////////////////////////////////////////////////////////////////////////////////////////////
struct e4m3
{
    i8 Data;

#ifdef __CUDACC__
    __host__ __device__ operator float() const
    {
#ifdef __CUDA_ARCH__
        return CvtE4ToFloat(Data);
#else
        return ConvertE4toFloat(Data);
#endif
    }
    __device__ operator half() const { return CvtE4ToHalf(Data); }
#else
    operator float() const { return ConvertE4toFloat(Data); }
#endif
};


struct e5m2
{
    i8 Data;

#ifdef __CUDACC__
    __host__ __device__ operator float() const
    {
#ifdef __CUDA_ARCH__
        return CvtE5ToFloat(Data);
#else
        return ConvertE5toFloat(Data);
#endif
    }
    __device__ operator half() const { return CvtE5ToHalf(Data); }
#else
    operator float() const { return ConvertE5toFloat(Data); }
#endif
};
