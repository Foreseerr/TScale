#pragma once
#include <immintrin.h>
#include <util/fp8.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
typedef i16 fp16;


///////////////////////////////////////////////////////////////////////////////////////////////////
inline void AddScaledArray(float *dst, const float *src, yint xSize, __m256 mult)
{
    Y_ASSERT((xSize & 7) == 0);
    __m256 *dstPtr = (__m256 *)dst;
    const __m256 *srcPtr = (const __m256 *)src;
    for (yint x8 = 0; x8 < xSize / 8; ++x8) {
        dstPtr[x8] = _mm256_add_ps(dstPtr[x8], _mm256_mul_ps(srcPtr[x8], mult));
    }
}

inline void ScaleArray(float *dst, yint xSize, __m256 mult)
{
    Y_ASSERT((xSize & 7) == 0);
    __m256 *dstPtr = (__m256 *)dst;
    for (yint x8 = 0; x8 < xSize / 8; ++x8) {
        dstPtr[x8] = _mm256_mul_ps(dstPtr[x8], mult);
    }
}

inline void AddScaledMatrixAligned(TArray2D<float> *dst, const TArray2D<float> &src, float scaleArg)
{
    yint xSize = src.GetXSize();
    yint ySize = src.GetYSize();
    Y_VERIFY(dst->GetXSize() == xSize);
    Y_VERIFY(dst->GetYSize() == ySize);
    __m256 scale = _mm256_set1_ps(scaleArg);
    for (yint y = 0; y < ySize; ++y) {
        AddScaledArray(dst->GetRow(y), src.GetRow(y), xSize, scale);
    }
}

inline void ScaleMatrixAligned(TArray2D<float> *dst, float scaleArg)
{
    yint xSize = dst->GetXSize();
    yint ySize = dst->GetYSize();
    __m256 scale = _mm256_set1_ps(scaleArg);
    for (yint y = 0; y < ySize; ++y) {
        ScaleArray(dst->GetRow(y), xSize, scale);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// https://stackoverflow.com/questions/51778721/how-to-convert-32-bit-float-to-8-bit-signed-char-41-packing-of-int32-to-int8
inline __m256i PackFloatToInt8(const __m256 *src, __m256 mult)
{
    // _mm256_loadu_ps() not needed, we expect aligned addresses
    __m256i a = _mm256_cvtps_epi32(_mm256_mul_ps(src[0], mult));
    __m256i b = _mm256_cvtps_epi32(_mm256_mul_ps(src[1], mult));
    __m256i c = _mm256_cvtps_epi32(_mm256_mul_ps(src[2], mult));
    __m256i d = _mm256_cvtps_epi32(_mm256_mul_ps(src[3], mult));
    __m256i ab = _mm256_packs_epi32(a, b);        // 16x int16_t
    __m256i cd = _mm256_packs_epi32(c, d);
    __m256i abcd = _mm256_packs_epi16(ab, cd);   // 32x int8_t
    // packed to one vector, but in [ a_lo, b_lo, c_lo, d_lo | a_hi, b_hi, c_hi, d_hi ] order
    // if you can deal with that in-memory format (e.g. for later in-lane unpack), great, you're done
    // but if you need sequential order, then vpermd:
    __m256i lanefix = _mm256_permutevar8x32_epi32(abcd, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    return lanefix;
}

inline void ConvertArray(i8 *dst, const float *src, yint xSize, __m256 mult)
{
    for (yint x = 0; x < xSize; x += 32) {
        *(__m256i *)(dst + x) = PackFloatToInt8((const __m256 *)(src + x), mult);
    }
}


inline void UnpackArray(float *dst, const i8 *src, yint xSize, __m256 mult)
{
    for (yint x = 0; x < xSize; x += 8) {
        __m128i src8 = _mm_cvtsi64_si128(*(const i64*)(src + x));
        __m256i src32 = _mm256_cvtepi8_epi32(src8);
        __m256 srcVal = _mm256_cvtepi32_ps(src32);
        *(__m256 *)(dst + x) = _mm256_mul_ps(srcVal, mult);
    }
}

inline void AddPackedArray(float *dst, const i8 *src, yint xSize, __m256 mult)
{
    for (yint x = 0; x < xSize; x += 8) {
        __m128i src8 = _mm_cvtsi64_si128(*(const i64 *)(src + x));
        __m256i src32 = _mm256_cvtepi8_epi32(src8);
        __m256 srcVal = _mm256_cvtepi32_ps(src32);
        __m256 *dstPtr = (__m256 *)(dst + x);
        *dstPtr = _mm256_add_ps(*dstPtr, _mm256_mul_ps(srcVal, mult));
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// fp16
static void ConvertToFp16(fp16 *dst, const float *src, yint xSize, __m256 mult)
{
    Y_ASSERT((xSize & 7) == 0);
    for (yint x = 0; x < xSize; x += 8) {
        // Load 8 floats from the input vector into a 256-bit register
        __m256 val = _mm256_mul_ps(_mm256_load_ps(src + x), mult);
        // Convert the 8 floats to 8 fp16 values and store them in a 128-bit register
        __m128i res = _mm256_cvtps_ph(val, 0);
        *(__m128i *)(dst + x) = res;
    }
}

inline void UnpackFp16Array(float *dst, const fp16 *src, yint xSize, __m256 mult)
{
    Y_ASSERT((xSize & 7) == 0);
    for (yint x = 0; x < xSize; x += 8) {
        __m256 srcVal = _mm256_cvtph_ps(*(__m128i *)(src + x));
        *(__m256 *)(dst + x) = _mm256_mul_ps(srcVal, mult);
    }
}

inline void AddScaledFp16Array(float *dst, const fp16 *src, yint xSize, __m256 mult)
{
    Y_ASSERT((xSize & 7) == 0);
    for (yint x = 0; x < xSize; x += 8) {
        __m256 srcVal = _mm256_cvtph_ps(*(__m128i *)(src + x));
        __m256 *dstPtr = (__m256 *)(dst + x);
        *dstPtr = _mm256_add_ps(*dstPtr, _mm256_mul_ps(srcVal, mult));
    }
}

inline void AddScaledFp16Array(fp16 *dst, const fp16 *src, yint xSize, __m256 mult)
{
    Y_ASSERT((xSize & 7) == 0);
    __m128i *dstPtr = (__m128i *)dst;
    const __m128i *srcPtr = (const __m128i *)src;
    for (yint x8 = 0; x8 < xSize / 8; ++x8) {
        __m256 srcVal = _mm256_cvtph_ps(srcPtr[x8]);
        __m256 dstVal = _mm256_cvtph_ps(dstPtr[x8]);
        dstPtr[x8] = _mm256_cvtps_ph(_mm256_add_ps(dstVal, _mm256_mul_ps(srcVal, mult)), 0);
    }
}

inline void ScaleFp16Array(fp16 *dst, yint xSize, __m256 mult)
{
    Y_ASSERT((xSize & 7) == 0);
    __m128i *dstPtr = (__m128i *)dst;
    for (yint x8 = 0; x8 < xSize / 8; ++x8) {
        __m256 dstVal = _mm256_cvtph_ps(dstPtr[x8]);
        dstPtr[x8] = _mm256_cvtps_ph(_mm256_mul_ps(dstVal, mult), 0);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// fp8
inline void ConvertToFp8e4m3(e4m3 *dst, const float *src, yint xSize, __m256 mult)
{
    Y_ASSERT((xSize & 7) == 0);
    for (yint x = 0; x < xSize; x += 8) {
        // Load 8 floats from the input vector into a 256-bit register
        __m256 vec = _mm256_mul_ps(_mm256_load_ps(src + x), mult);

        // round to nearest
        __m256i roundBits = _mm256_castps_si256(vec);
        __m256i round0 = _mm256_and_si256(roundBits, _mm256_set1_epi32(0xFF800000));
        __m256i round1 = _mm256_and_si256(roundBits, _mm256_set1_epi32(0xFF880000));
        vec = _mm256_add_ps(vec, _mm256_sub_ps(_mm256_castsi256_ps(round1), _mm256_castsi256_ps(round0)));

        // Extract sign, exponent, and mantissa
        __m256i bits = _mm256_castps_si256(vec);
        __m256i sign = _mm256_srli_epi32(bits, 31);
        __m256i exponent = _mm256_srli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7F800000)), 23);
        __m256i mantissa = _mm256_srli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x007FFFFF)), 20);

        // Adjust exponent bias from 127 to 7
        exponent = _mm256_sub_epi32(exponent, _mm256_set1_epi32(127 - 7));

        // Combine sign, exponent, and mantissa into FP8 format
        __m256i fp8dw = _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi32(sign, 7), _mm256_slli_epi32(exponent, 3)), mantissa);

        // cheap imprecise handling of overflow & underflow
        __m256i maxAbs = _mm256_or_si256(_mm256_slli_epi32(sign, 7), _mm256_set1_epi32(0x7f)); // 7f is NaN actually
        __m256i maskOverflow = _mm256_cmpgt_epi32(exponent, _mm256_set1_epi32(15));
        fp8dw = _mm256_blendv_epi8(fp8dw, maxAbs, maskOverflow);
        __m256i maskUnderflow = _mm256_cmpgt_epi32(_mm256_set1_epi32(0), exponent);
        fp8dw = _mm256_andnot_si256(maskUnderflow, fp8dw);

        // Pack dwords into bytes
        __m128i fp8w = _mm_packus_epi32(_mm256_castsi256_si128(fp8dw), _mm256_extracti128_si256(fp8dw, 1));
        __m128i fp8 = _mm_packus_epi16(fp8w, _mm_srli_si128(fp8w, 64));

        // Avoid NaNs
        fp8 = _mm_min_epu8(fp8, _mm_set1_epi8((char)0xfe));
        fp8 = _mm_min_epi8(fp8, _mm_set1_epi8(0x7e));

        // Store the result
        *(ui64 *)(dst + x) = _mm_extract_epi64(fp8, 0);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
inline float HorizontalSum(__m256 x)
{
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}


static __m256 CalcRowSum2(const float *src, yint xSize)
{
    Y_ASSERT((xSize & 7) == 0);
    __m256 rowSum2 = _mm256_setzero_ps();
    for (yint x = 0; x < xSize; x += 8) {
        __m256 val = _mm256_load_ps(src + x);
        rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
    }
    return rowSum2;
}

template <class TMatrix>
static float CalcMatrixSum2(const TMatrix &matr)
{
    yint xSize = matr.GetXSize();
    yint ySize = matr.GetYSize();
    __m256 sum2 = _mm256_setzero_ps();
    for (yint y = 0; y < ySize; ++y) {
        __m256 rowSum2 = CalcRowSum2(matr.GetRow(y), xSize);
        sum2 = _mm256_add_ps(sum2, rowSum2);
    }
    return HorizontalSum(sum2);
}
