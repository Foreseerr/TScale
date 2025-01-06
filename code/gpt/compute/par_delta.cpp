#include "stdafx.h"
#include "par_delta.h"
#include <immintrin.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
inline float HalfToFloat(fp16 x)
{
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(x)));
}

inline fp16 FloatToHalf(float x)
{
    return _mm_extract_epi16(_mm_cvtps_ph(_mm_set1_ps(x), 0), 0);
}

inline float RoundFloatPow2(float x)
{
    int val = *(int*) &x;
    val &= 0xff800000;
    return *(float *)&val;
}


void TModelMatrixHalfDelta::GetAllData(TArray2D<float> *p) const
{
    yint xSize = SizeX;
    yint ySize = SizeY;
    p->SetSizes(xSize, ySize);
    p->FillZero();
    for (yint y = 0; y < ySize; ++y) {
        const TRow &row = Rows[y];
        if (row.Scale == 0) {
            continue;
        }
        const fp16 *deltaRowPtr = GetRow(y);
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] = HalfToFloat(deltaRowPtr[x]) * row.Scale;
        }
    }
}


inline void CopyRow(TModelMatrixHalfDelta::TRow *pRow, yint xSize, fp16 *dstArg, const i8 *srcData, const float *srcDataScale)
{
    float sum2 = 0;
    float maxScale = 0;
    for (int tile = 0; tile < xSize / MODEL_INT8_DELTA_TILE; ++tile) {
        maxScale = fmaxf(maxScale, srcDataScale[tile]);
    }
    float newRowScale = RoundFloatPow2(maxScale / 64);
    if (maxScale > 0) {
        const ui64 *src = (const ui64 *)srcData;
        __m128i *dst = (__m128i *) dstArg;
        __m256 rowSum2 = _mm256_setzero_ps();
        for (int tile = 0; tile < xSize / MODEL_INT8_DELTA_TILE; ++tile) {
            __m256 srcMult = _mm256_set1_ps((1 / 64.0f / 64.0f) * srcDataScale[tile] / newRowScale);
            for (yint x8 = 0; x8 < MODEL_INT8_DELTA_TILE / 8; ++x8) {
                // Load the 64-bit integer into a 128-bit register
                __m128i src8 = _mm_cvtsi64_si128(src[x8]);
                // Unpack the 8 int8 integers into 32-bit integers
                __m256i src32 = _mm256_cvtepi8_epi32(src8);
                // convert to float and scale
                __m256 newDstVal = _mm256_mul_ps(_mm256_cvtepi32_ps(src32), srcMult);
                // convert to fp16 and write
                dst[x8] = _mm256_cvtps_ph(newDstVal, 0);
                // collect rowsum
                rowSum2 = _mm256_add_ps(_mm256_mul_ps(newDstVal, newDstVal), rowSum2);
            }
            src += MODEL_INT8_DELTA_TILE / 8;
            dst += MODEL_INT8_DELTA_TILE / 8;
        }
        sum2 = HorizontalSum(rowSum2);
    }
    Y_ASSERT(!isnan(sum2) && isfinite(sum2));
    pRow->Scale = newRowScale;
    pRow->Sum2 = sum2 * Sqr(pRow->Scale);
}


inline void AddRow(TModelMatrixHalfDelta::TRow *pRow, yint xSize, fp16 *dstArg, const i8 *srcData, const float *srcDataScale)
{
    float maxScale = 0;
    for (int tile = 0; tile < xSize / MODEL_INT8_DELTA_TILE; ++tile) {
        maxScale = fmaxf(maxScale, srcDataScale[tile]);
    }
    if (maxScale == 0) {
        return;
    }
    Y_ASSERT(pRow->Scale > 0 && maxScale > 0);
    float srcRowScale = RoundFloatPow2(maxScale / 64);
    float newRowScale = Max<float>(srcRowScale, pRow->Scale);
    __m256 dstMult = _mm256_set1_ps(1 * pRow->Scale / newRowScale);

    const ui64 *src = (const ui64 *)srcData;
    __m128i *dst = (__m128i *) dstArg;
    __m256 rowSum2 = _mm256_setzero_ps();
    for (int tile = 0; tile < xSize / MODEL_INT8_DELTA_TILE; ++tile) {
        __m256 srcMult = _mm256_set1_ps((1 / 64.0f / 64.0f) * srcDataScale[tile] / newRowScale);
        for (yint x8 = 0; x8 < MODEL_INT8_DELTA_TILE / 8; ++x8) {
            __m256 dstVal = _mm256_cvtph_ps(dst[x8]);
            // Load the 64-bit integer into a 128-bit register
            __m128i src8 = _mm_cvtsi64_si128(src[x8]);
            // Unpack the 8 int8 integers into 32-bit integers
            __m256i src32 = _mm256_cvtepi8_epi32(src8);
            // convert to float and scale
            __m256 srcVal = _mm256_cvtepi32_ps(src32);
            // new val
            __m256 newDstVal = _mm256_add_ps(_mm256_mul_ps(srcVal, srcMult), _mm256_mul_ps(dstVal, dstMult));
            // convert to fp16 and write
            dst[x8] = _mm256_cvtps_ph(newDstVal, 0);
            // collect rowsum
            rowSum2 = _mm256_add_ps(_mm256_mul_ps(newDstVal, newDstVal), rowSum2);
        }
        src += MODEL_INT8_DELTA_TILE / 8;
        dst += MODEL_INT8_DELTA_TILE / 8;
    }
    float sum2 = HorizontalSum(rowSum2);
    Y_ASSERT(!isnan(sum2) && isfinite(sum2));
    pRow->Scale = newRowScale;
    pRow->Sum2 = sum2 * Sqr(pRow->Scale);
}


void Copy(TModelMatrixHalfDelta *p, const TModelMatrixInt8Delta &delta)
{
    yint xSize = p->SizeX;
    yint ySize = p->SizeY;
    p->Delta.resize(xSize * ySize);
    p->Rows.resize(ySize);
    for (yint y = 0; y < ySize; ++y) {
        TModelMatrixHalfDelta::TRow &row = p->Rows[y];
        const i8 *src = delta.GetRow(y);
        const float *srcScale = delta.GetTileScaleRow(y);
        fp16 *dst = &p->Delta[y * xSize];
        CopyRow(&p->Rows[y], xSize, dst, src, srcScale);
    }
}


void Add(TModelMatrixHalfDelta *p, const TModelMatrixInt8Delta &delta)
{
    yint xSize = p->SizeX;
    yint ySize = p->SizeY;
    for (yint y = 0; y < ySize; ++y) {
        TModelMatrixHalfDelta::TRow &row = p->Rows[y];
        const i8 *src = delta.GetRow(y);
        const float *srcScale = delta.GetTileScaleRow(y);
        fp16 *dst = &p->Delta[y * xSize];
        if (row.Sum2 == 0) {
            CopyRow(&row, xSize, dst, src, srcScale);
        } else {
            AddRow(&row, xSize, dst, src, srcScale);
        }
    }
}


void Compress(TModelMatrixInt8Delta *p, const TArray2D<float> &data)
{
    Y_VERIFY((data.GetXSize() % MODEL_INT8_DELTA_TILE) == 0);
    yint xSize = data.GetXSize();
    yint ySize = data.GetYSize();
    yint tileId = 0;
    for (yint y = 0; y < ySize; ++y) {
        for (yint xOffset = 0; xOffset < xSize; xOffset += MODEL_INT8_DELTA_TILE) {
            float maxVal = 0;
            for (yint x = 0; x < MODEL_INT8_DELTA_TILE; ++x) {
                maxVal = Max<float>(maxVal, fabs(data[y][xOffset + x]));
            }
            if (maxVal == 0) {
                p->TileScale[tileId] = 0;
            } else {
                float scale = maxVal / 127;
                float mult = 1 / scale;
                p->TileScale[tileId] = scale;
                i8 *dstPtr = p->GetRow(y) + xOffset;
                ConvertArray(dstPtr, &data[y][xOffset], MODEL_INT8_DELTA_TILE, _mm256_set1_ps(mult));
            }
            ++tileId;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static void Add1(yint sz, const ui64 *a, ui64 *tail)
{
    for (yint k = 0; k < sz; ++k) {
        ui64 a1 = a[k];
        tail[k] = a1;
    }
}

static void Add2(yint sz, const ui64 *a, ui64 *tail, ui64 *res)
{
    for (yint k = 0; k < sz; ++k) {
        ui64 a1 = a[k];
        ui64 a2 = tail[k];
        res[k] = a1 & a2;
        tail[k] = a1 ^ a2;
    }
}

static void Add3(yint sz, const ui64 *a, const ui64 *b, ui64 *tail, ui64 *res)
{
    for (yint k = 0; k < sz; ++k) {
        ui64 a1 = a[k];
        ui64 a2 = b[k];
        ui64 a3 = tail[k];
        res[k] = (a1 & a2) | (a1 & a3) | (a2 & a3);
        tail[k] = a1 ^ a2 ^ a3;
    }
}


void SumBitDelta(const TModelMatrixBitDelta &a, const TModelMatrixBitDelta &b, TModelMatrixBitDeltaTail *pTail, TModelMatrixBitDelta *pRes)
{
    if (a.IsEmpty() && b.IsEmpty()) {
        // zero delta
        pRes->Clear();
        return;
    }
    // average row disp estimate
    Y_VERIFY(YSize(a.DeltaRowDisp) == YSize(b.DeltaRowDisp));
    yint rdCount = YSize(a.DeltaRowDisp);
    pRes->DeltaRowDisp.resize(rdCount);
    for (yint y = 0; y < rdCount; ++y) {
        pRes->DeltaRowDisp[y] = (a.DeltaRowDisp[y] + b.DeltaRowDisp[y]) * 0.5f;
    }
    // sum bit deltas
    yint sz = YSize(a.BitDelta);
    Y_VERIFY(YSize(b.BitDelta) == sz);
    Y_VERIFY(YSize(pTail->BitDelta) == sz);
    pRes->BitDelta.yresize(sz);
    Add3(sz, a.BitDelta.data(), b.BitDelta.data(), pTail->BitDelta.data(), pRes->BitDelta.data());
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static ui64 ByteMaskToInt[256];
static struct TInitByteMaskToInt
{
    TInitByteMaskToInt()
    {
        for (yint k = 0; k < 256; ++k) {
            ui64 res = 0;
            for (yint b = 0; b < 8; ++b) {
                if (k & (1ll << b)) {
                    res |= 0xffull << (b * 8);
                }
            }
            ByteMaskToInt[k] = res;
        }
    }
} initByteMaskToInt;


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TMatrixDeltaAddCtx
{
    TArray2D<float> &Matr;
    TArray2D<float> &AvrgDelta1;
    __m256 NewSum2;
    __m256 Beta1;
    __m256 BetaMult1;
    __m256 Weight0;
    __m256 Weight1;
    __m256 StepMult;
    __m256 ShrinkMult;
    __m256 AllSignBits;

    TMatrixDeltaAddCtx(TArray2D<float> &matr, TArray2D<float> &avrgDelta1, float step, float shrink, float beta1, float w0, float w1)
        : Matr(matr), AvrgDelta1(avrgDelta1)
    {
        NewSum2 = _mm256_setzero_ps();
        Beta1 = _mm256_set1_ps(beta1);
        BetaMult1 = _mm256_set1_ps(1 - beta1);
        Weight0 = _mm256_set1_ps(w0);
        Weight1 = _mm256_set1_ps(w1);
        StepMult = _mm256_set1_ps(step);
        ShrinkMult = _mm256_set1_ps(shrink);
        AllSignBits = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    }

    void AddRow(yint y, yint xSize)
    {
        __m256 *avrgDelta1Ptr = (__m256 *) & AvrgDelta1[y][0];
        __m256 *matrPtr = (__m256 *) & Matr[y][0];
        __m256 rowSum2 = _mm256_setzero_ps();
        for (yint x8 = 0; x8 < xSize / 8; ++x8) {
            __m256 oldAvrgDelta1 = avrgDelta1Ptr[x8];
            __m256 avrgDelta1 = _mm256_mul_ps(oldAvrgDelta1, Beta1);
            __m256 totalDelta = _mm256_mul_ps(avrgDelta1, Weight1);
            __m256 val = _mm256_add_ps(_mm256_mul_ps(matrPtr[x8], ShrinkMult), _mm256_mul_ps(totalDelta, StepMult));
            avrgDelta1Ptr[x8] = avrgDelta1;
            matrPtr[x8] = val;
            rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
        }
        NewSum2 = _mm256_add_ps(NewSum2, rowSum2);
    }

    void AddRow(yint y, yint xSize, const __m128i *deltaPtr, __m256 deltaScale)
    {
        __m256 *avrgDelta1Ptr = (__m256 *) & AvrgDelta1[y][0];
        __m256 *matrPtr = (__m256 *) & Matr[y][0];
        __m256 rowSum2 = _mm256_setzero_ps();
        for (yint x8 = 0; x8 < xSize / 8; ++x8) {
            __m256 deltaVal = _mm256_mul_ps(_mm256_cvtph_ps(deltaPtr[x8]), deltaScale);
            __m256 oldAvrgDelta1 = avrgDelta1Ptr[x8];
            __m256 avrgDelta1 = _mm256_add_ps(_mm256_mul_ps(oldAvrgDelta1, Beta1), _mm256_mul_ps(deltaVal, BetaMult1));
            __m256 totalDelta12 = _mm256_mul_ps(avrgDelta1, Weight1);
            __m256 totalDelta = _mm256_add_ps(totalDelta12, _mm256_mul_ps(deltaVal, Weight0));
            __m256 val = _mm256_add_ps(_mm256_mul_ps(matrPtr[x8], ShrinkMult), _mm256_mul_ps(totalDelta, StepMult));
            avrgDelta1Ptr[x8] = avrgDelta1;
            matrPtr[x8] = val;
            rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
        }
        NewSum2 = _mm256_add_ps(NewSum2, rowSum2);
    }

    void AddBitRow(yint y, yint xSize, ui8 *bitDeltaPtr)
    {
        __m256 *avrgDelta1Ptr = (__m256 *) & AvrgDelta1[y][0];
        __m256 *matrPtr = (__m256 *) & Matr[y][0];
        __m256 rowSum2 = _mm256_setzero_ps();
        for (int x8 = 0; x8 < xSize / 8; ++x8) {
            ui64 byteMask = ByteMaskToInt[bitDeltaPtr[x8]];
            __m256i mask = _mm256_cvtepi8_epi32(_mm_set_epi64x(0, byteMask));
            __m256 deltaSignBits = _mm256_and_ps(AllSignBits, _mm256_castsi256_ps(mask));
            __m256 oldAvrgDelta1 = avrgDelta1Ptr[x8];
            __m256 avrgDelta1 = _mm256_add_ps(_mm256_mul_ps(oldAvrgDelta1, Beta1), _mm256_xor_ps(deltaSignBits, BetaMult1));
            __m256 totalDelta12 = _mm256_mul_ps(avrgDelta1, Weight1);
            __m256 totalDelta = _mm256_add_ps(totalDelta12, _mm256_xor_ps(deltaSignBits, Weight0));
            __m256 val = _mm256_add_ps(_mm256_mul_ps(matrPtr[x8], ShrinkMult), _mm256_mul_ps(totalDelta, StepMult));
            avrgDelta1Ptr[x8] = avrgDelta1;
            matrPtr[x8] = val;
            rowSum2 = _mm256_add_ps(rowSum2, _mm256_mul_ps(val, val));
        }
        NewSum2 = _mm256_add_ps(NewSum2, rowSum2);
    }
};


void TModelMatrixData::AddDelta(const TModelMatrixHalfDelta &delta, const TTrainingStep &step)
{
    yint xSize = Matr.GetXSize();
    yint ySize = Matr.GetYSize();
    float dispDecay = step.DispDecay;

    TMatrixDeltaAddCtx ctx(Matr, AvrgDelta1, step.Rate, step.GetShrinkMult(), step.Beta1, step.Weight0, step.Weight1);
    SumWeight = SumWeight * dispDecay + 1;
    float rowDispNorm = 1 / SumWeight;
    if (YSize(RowDisp) > 1) {
        // separate row disp
        for (yint y = 0; y < ySize; ++y) {
            const TModelMatrixHalfDelta::TRow &row = delta.Rows[y];
            RowDisp[y] = (RowDisp[y] * dispDecay) + (row.Sum2 / xSize);

            if (row.Sum2 > 0) {
                const __m128i *deltaPtr = (const __m128i *)delta.GetRow(y);
                __m256 deltaScale = _mm256_set1_ps(row.Scale / sqrt(RowDisp[y] * rowDispNorm));
                ctx.AddRow(y, xSize, deltaPtr, deltaScale);
            } else {
                ctx.AddRow(y, xSize);
            }
        }

    } else {
        float sum2 = delta.CalcSum2() / xSize / ySize;
        RowDisp[0] = (RowDisp[0] * dispDecay) + sum2;
        float globalScale = (sum2 > 0) ? 1 / sqrtf(RowDisp[0] * rowDispNorm) : 0;
        for (int y = 0; y < ySize; ++y) {
            const TModelMatrixHalfDelta::TRow &row = delta.Rows[y];
            const __m128i *deltaPtr = (const __m128i *)delta.GetRow(y);
            __m256 deltaScale = _mm256_set1_ps(row.Scale * globalScale);
            ctx.AddRow(y, xSize, deltaPtr, deltaScale);
        }
    }
    Sum2 = HorizontalSum(ctx.NewSum2);
}


bool TModelMatrixData::AddBitDelta(const TModelMatrixBitDelta &bitDelta, const TTrainingStep &step)
{
    if (bitDelta.IsEmpty()) {
        return false;
    }
    yint xSize = GetXSize();
    yint ySize = GetYSize();
    float dispDecay = step.DispDecay;

    yint rdCount = YSize(RowDisp);
    Y_VERIFY(rdCount == YSize(bitDelta.DeltaRowDisp));
    SumWeight = SumWeight * dispDecay + 1;
    for (yint y = 0; y < rdCount; ++y) {
        RowDisp[y] = (RowDisp[y] * dispDecay) + bitDelta.DeltaRowDisp[y];
    }

    TMatrixDeltaAddCtx ctx(Matr, AvrgDelta1, step.Rate, step.GetShrinkMult(), step.Beta1, step.Weight0, step.Weight1);
    for (yint y = 0; y < ySize; ++y) {
        ui8 *bitDeltaPtr = (ui8 *)&bitDelta.BitDelta[y * xSize / 64];
        ctx.AddBitRow(y, xSize, bitDeltaPtr);
    }
    Sum2 = HorizontalSum(ctx.NewSum2);
    return true;
}


inline void CompressLine(ui8 *resPtr, const __m128i *deltaPtr, __m256 deltaScale, __m256 *deltaTailPtr, yint xSize, __m256 allSignBits, __m256 basicStep)
{
    for (yint x8 = 0; x8 < xSize / 8; ++x8) {
        __m256 deltaVal = _mm256_mul_ps(_mm256_cvtph_ps(deltaPtr[x8]), deltaScale);
        // val = tail + delta
        __m256 val = _mm256_add_ps(deltaTailPtr[x8], deltaVal);
        // signBit = val > 0
        __m256 signBit = _mm256_and_ps(allSignBits, val);
        // add = (val > 0) ? basicStep : -basicStep
        __m256 add = _mm256_or_ps(signBit, basicStep);
        // tail = val - add
        deltaTailPtr[x8] = _mm256_sub_ps(val, add);
        resPtr[x8] = _mm256_movemask_ps(signBit);
    }
}

void TModelMatrixData::CompressDelta(const TModelMatrixHalfDelta &delta, TModelMatrixBitDelta *pBitDelta, TArray2D<float> *pDeltaTail)
{
    TArray2D<float> &deltaTail = *pDeltaTail;
    yint xSize = Matr.GetXSize();
    yint ySize = Matr.GetYSize();
    Y_ASSERT((xSize % 64) == 0);
    __m256 allSignBits = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    bool perRowDisp = (YSize(RowDisp) > 1);

    __m256 basicStep = _mm256_setzero_ps();
    if (perRowDisp) {
        pBitDelta->DeltaRowDisp.yresize(ySize);
    } else {
        float disp = delta.CalcSum2() / xSize / ySize;
        float dispEstimate = (RowDisp[0] + disp) / (SumWeight + 1);
        basicStep = _mm256_set1_ps(sqrt(dispEstimate));
        // save disp estimate
        pBitDelta->DeltaRowDisp.yresize(1);
        pBitDelta->DeltaRowDisp[0] = disp;
    }

    pBitDelta->BitDelta.yresize(ySize * xSize / 64);
    for (yint y = 0; y < ySize; ++y) {
        const TModelMatrixHalfDelta::TRow &row = delta.Rows[y];
        if (perRowDisp) {
            // each row has separate scale
            // take into account current delta dispersion (somehow gives better results)
            float disp = row.Sum2 / xSize;
            float dispEstimate = (RowDisp[y] + disp) / (SumWeight + 1);
            basicStep = _mm256_set1_ps(sqrt(dispEstimate));
            // save disp estimate
            pBitDelta->DeltaRowDisp[y] = disp;
        }
        const __m128i *deltaPtr = (const __m128i *)delta.GetRow(y);
        __m256 deltaScale = _mm256_set1_ps(row.Scale);
        __m256 *deltaTailPtr = (__m256 *) deltaTail.GetRow(y);
        ui8 *resPtr = (ui8 *)&pBitDelta->BitDelta[y * xSize / 64];
        CompressLine(resPtr, deltaPtr, deltaScale, deltaTailPtr, xSize, allSignBits, basicStep);
    }
}
