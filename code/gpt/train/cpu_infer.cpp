#include "stdafx.h"
#include "cpu_infer.h"
#include <gpt/att/att.h>
#include <gpt/att/rope.h>
#include <gpt/data/data.h>
#include <gpt/model_params/model_dim.h>
#include <gpt/model_params/model_params.h>
#include <gpt/compute/model.h>
#include <gpt/compute/gpt_cuda.cuh>
#include <gpt/compute/gpt_cpu.h>
#include <gpt/att/sliding_window.h>
#include <lib/hp_timer/hp_timer.h>
#include <immintrin.h>


const float VEC_SCALE = 1.0f / 24;

///////////////////////////////////////////////////////////////////////////////////////////////////
static int Float2Int(float x)
{
    return _mm_cvtss_si32(_mm_set_ss(x));
}
float GetFp8value(float x)
{
    //// hypothetical e3m4
    //int bias = 3;
    //int exponent_bits = 3;
    //int mantissa_bits = 4;
    // e4m3
    int bias = 7;
    int exponent_bits = 4;
    int mantissa_bits = 3;
    //// e5m2
    //int bias = 15;
    //int exponent_bits = 5;
    //int mantissa_bits = 2;
    ////int exponent_bits = 5; // fp16
    ////int mantissa_bits = 10;

    int sign = x > 0 ? 1 : -1;
    float xAbs = fabs(x);
    int nExp = xAbs > 0 ? floor(log2f(xAbs)) : 0;
    float bit = exp2f(nExp);
    float man = xAbs > 0 ? xAbs / bit - 1 : 0;
    // e & m fields
    int e = nExp + bias;
    int m = Float2Int(man * (1 << mantissa_bits));
    float res = 0;
    if (e == 0 && m > 0) {
        res = sign * exp2f(1 - bias) * (0 + exp2f(-mantissa_bits) * m);
    } else {
        res = sign * exp2f(e - bias) * (1 + exp2f(-mantissa_bits) * m);
    }
    return res;
}

namespace NCPUInfer
{
struct TCPUModelParams
{
    struct TAttentionMatrices
    {
        TArray2D<i8> QK;
        TArray2D<i8> QV;
        TArray2D<i8> K;
        TArray2D<i8> V;
        TArray2D<i8> Combiner;
        TArray2D<i8> Expand;
        TArray2D<i8> Contract;
        float QVScale = 0;
        float VScale = 0;
        float CombinerScale = 0;
        float ContractScale = 0;
        int AttentionWidth = 0;
        float AlibiSlope = 0;
    };
    TModelDescr ModelDescr;
    TArray2D<i8> LabelEmbed;
    float LabelEmbedScale = 0;
    TVector<TAttentionMatrices> LayerArr;
    TArray2D<i8> FinalLayer;
    float FinalLayerScale = 0;
    TVector<float> Bias;
    yint BaseLabel = 0;
};


static i8 ConvertToInt8(float x)
{
    int res = _mm_cvtss_si32(_mm_set_ss(x));
    return ClampVal<int>(res, -127, 127); // -128 is incompatible with signed * unsigned ops
}

static float ConvertMatrix(const TArray2D<float> &data, TArray2D<i8> *p)
{
    yint xSize = data.GetXSize();
    yint ySize = data.GetYSize();
    float sum2 = 0;
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            sum2 += Sqr(data[y][x]);
        }
    }
    float sko = sqrt(sum2 / (xSize * ySize));
    float discrScale = sko * MODEL_DISCR_SCALE;
    float mult = (sko == 0) ? 0 : (1 / discrScale);
    p->SetSizes(xSize, ySize);
    double int8err = 0;
    double fp8err = 0;
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            float val = data[y][x] * mult;
            (*p)[y][x] = ConvertToInt8(val);
            int8err += Sqr(val - (*p)[y][x]);
            fp8err += Sqr(val * MODEL_DISCR_SCALE - GetFp8value(val * MODEL_DISCR_SCALE));
            //DebugPrintf("val %g, int8 %g, fp8 %g\n", val, val - (*p)[y][x], val - GetFp8value(val * MODEL_DISCR_SCALE) / MODEL_DISCR_SCALE);
        }
    }
    //DebugPrintf("int8 err %g, fp8 err %g\n", sqrt(int8err / xSize / ySize), sqrt(fp8err / xSize/ ySize) / MODEL_DISCR_SCALE);
    return discrScale;
}

static float ConvertMatrix(const TModelMatrix &data, TArray2D<i8> *p)
{
    return ConvertMatrix(data.GetMatrix(), p);
}

static void ConvertAtt(const TModelParams::TAttentionMatrices &att, TCPUModelParams::TAttentionMatrices *p)
{
    ConvertMatrix(att.MatrArr[MP_ATT_QK], &p->QK);
    p->QVScale = ConvertMatrix(att.MatrArr[MP_ATT_QV], &p->QV);
    ConvertMatrix(att.MatrArr[MP_ATT_K], &p->K);
    p->VScale = ConvertMatrix(att.MatrArr[MP_ATT_V], &p->V);
    p->CombinerScale = ConvertMatrix(att.MatrArr[MP_ATT_COMBINER], &p->Combiner);
    ConvertMatrix(att.MatrArr[MP_RELU_EXPAND], &p->Expand);
    p->ContractScale = ConvertMatrix(att.MatrArr[MP_RELU_CONTRACT], &p->Contract);
}

void ConvertModel(TModelParams &params, TCPUModelParams *p)
{
    p->ModelDescr = params.ModelDescr;
    p->LabelEmbedScale = ConvertMatrix(params.MatrArr[MP_MODEL_EMBED], &p->LabelEmbed);
    p->LayerArr.resize(YSize(params.LayerArr));
    for (yint layerId = 0; layerId < YSize(params.LayerArr); ++layerId) {
        TCPUModelParams::TAttentionMatrices &resAtt = p->LayerArr[layerId];
        ConvertAtt(params.LayerArr[layerId], &resAtt);
        // pick up width
        const TModelDescr::TAttentionPosParams &attPosParams = params.ModelDescr.Layers[layerId];
        resAtt.AttentionWidth = params.ModelDescr.AttentionWidthArr[attPosParams.AttentionWidthId];
        resAtt.AlibiSlope = attPosParams.AlibiSlope;
    }
    p->FinalLayerScale = ConvertMatrix(params.MatrArr[MP_MODEL_FINAL], &p->FinalLayer);
    p->Bias = params.Bias;
    p->BaseLabel = 1;
    if (!params.GetModelDescr().HasFlag(MPF_DISABLE_NOISE_LABELS)) {
        p->BaseLabel += NOISE_LABELS_COUNT;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// SSE utils
// 
inline int HorizontalSumInt(__m256i v)
{
    // Use SSE2 functions to extract the lower and higher 128 bits
    __m128i vlow = _mm256_castsi256_si128(v);
    __m128i vhigh = _mm256_extracti128_si256(v, 1);

    // Perform pairwise addition of 32-bit integers
    vlow = _mm_add_epi32(vlow, vhigh);

    // Shuffle and add until we get the sum across the vector
    __m128i shuf = _mm_shuffle_epi32(vlow, _MM_SHUFFLE(0, 3, 2, 1)); // Shuffle the elements
    vlow = _mm_add_epi32(vlow, shuf);
    shuf = _mm_shuffle_epi32(vlow, _MM_SHUFFLE(1, 0, 3, 2)); // Shuffle again
    vlow = _mm_add_epi32(vlow, shuf);

    // Extract the sum
    return _mm_extract_epi32(vlow, 0);
}


static inline __m256i dp64(const __m256i x1, const __m256i x2, const __m256i y1, const __m256i y2, const __m256i sum)
{
    // glorious Intel does not support VNNI in 12xxx - 14xxx cpus, use legacy instructions
    //sum = _mm256_dpbssd_epi32(aPtr[i], bPtr[i], sum);

    __m256i ax = _mm256_sign_epi8(x1, x1);
    __m256i sy = _mm256_sign_epi8(y1, x1);
    __m256i sum1 = _mm256_dpbusd_avx_epi32(sum, ax, sy);
    ax = _mm256_sign_epi8(x2, x2);
    sy = _mm256_sign_epi8(y2, x2);
    __m256i sum2 = _mm256_dpbusd_avx_epi32(sum1, ax, sy);
    return sum2;
}


static i32 DotInt8(const i8 *aData, const i8 *bData, yint sz)
{
    __m256i sum = _mm256_setzero_si256();
    const __m256i *aPtr = (const __m256i *)aData;
    const __m256i *bPtr = (const __m256i *)bData;
    for (yint i = 0; i < sz / 32; i += 2) {
        sum = dp64(aPtr[i], aPtr[i + 1], bPtr[i], bPtr[i + 1], sum);
        //_mm_prefetch((const char *)(aPtr + 4), _MM_HINT_NTA);
        //_mm_prefetch((const char *)(bPtr + 4), _MM_HINT_NTA);
    }
    return HorizontalSumInt(sum);
}

static i32 DotInt8(const TVector<i8> &a, const TVector<i8> &b)
{
    yint sz = YSize(a);
    Y_ASSERT(sz == YSize(b));
    return DotInt8(a.data(), b.data(), sz);
}


struct TSoftMaxBuf
{
    TVector<float> Buf;
    yint Ptr = 0;
    float MaxValue = 0;
    float Scale = 0;

    TSoftMaxBuf()
    {
        Buf.resize(8, -1e38f);
    }

    void Clear()
    {
        Ptr = 0;
        MaxValue = 0;
    }

    void Add(float x)
    {
        if (Ptr == YSize(Buf)) {
            Buf.resize(YSize(Buf) * 2, -1e38f);
        }
        Buf[Ptr++] = x;
        MaxValue = Max<float>(MaxValue, x);
    }

    void SoftMax()
    {
        yint sz = (Ptr + 7) / 8;
        float sumWeight = 0;
        __m256 *dataBuf = (__m256 *)Buf.data();
        __m256 sum = _mm256_setzero_ps();
        __m256 maxValue = _mm256_set1_ps(MaxValue);
        for (yint i = 0; i < sz; ++i) {
            // exp avx by Imperator@
            __m256 x = _mm256_sub_ps(dataBuf[i], maxValue);
            x = _mm256_max_ps(x, _mm256_set1_ps(-127));
            __m256 xf = _mm256_floor_ps(x);
            x = _mm256_sub_ps(x, xf);
            __m256 s = _mm256_sub_ps(x, xf);
            __m256i xfi = _mm256_cvtps_epi32(xf);

            __m256 x2 = _mm256_mul_ps(x, x);
            __m256 c0 = _mm256_set1_ps(-3.069678791803394491901405992213472390777e-1f);
            __m256 c1 = _mm256_set1_ps(-6.558811624324781017147952441210509604385e-2f);
            __m256 c2 = _mm256_set1_ps(-1.355574723481491770403079319055785445381e-2f);
            __m256 res = _mm256_fmadd_ps(_mm256_fmadd_ps(c2, x, c1), x, c0);

            __m256 one = _mm256_set1_ps(1);
            __m256 x_by_1_minus_x = _mm256_sub_ps(x, x2);
            res = _mm256_fmadd_ps(res, x_by_1_minus_x, x);
            res = _mm256_add_ps(res, one); //adding ymm_x and 1 separately in the end improves accuracy

            xfi = _mm256_slli_epi32(xfi, 23);
            res = _mm256_castsi256_ps(_mm256_add_epi32(xfi, _mm256_castps_si256(res)));
            dataBuf[i] = res;
            sum = _mm256_add_ps(sum, res);
        }
        Scale = 1 / HorizontalSum(sum);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// linear algebra
template <class T>
void PrintVec(const TVector<T> &vec)
{
    yint sz = YSize(vec); //Min<yint>(128, YSize(vec));
    for (yint i = 0; i < sz; ++i) {
        DebugPrintf("cpu vec[%g] = %g\n", i * 1., vec[i] * 1.);
    }
}

// resArr = kqv @ vecArr
template <class T>
static void MulForward(const TVector<i8> &vec, const TArray2D<i8> &kqv, TVector<T> *resArr)
{
    yint dim = YSize(vec);
    yint rDim = kqv.GetYSize();
    Y_ASSERT(dim == kqv.GetXSize());
    Y_ASSERT(rDim == kqv.GetYSize());
    resArr->resize(rDim);
    for (yint k = 0; k < rDim; ++k) {
        (*resArr)[k] = DotInt8(vec.data(), &kqv[k][0], dim);
    }
}


static void AddScaled(TVector<float> *pRes, const TVector<i32> &delta, float scale)
{
    yint sz = YSize(*pRes);
    Y_ASSERT(sz == YSize(delta));
    for (yint k = 0; k < sz; ++k) {
        (*pRes)[k] += delta[k] * scale;
    }
}


static void KVProduct(const TVector<i8> &kState, const TVector<float> &valLookup, yint headCount,
    TVector<i8> *pKVState)
{
    yint ttSum = YSize(kState);
    yint headDim = ttSum / headCount;
    pKVState->resize(ttSum * COMBINER_REP);
    for (yint head = 0; head < headCount; ++head) {
        yint headOffset = head * headDim;
        for (int blk = 0; blk < COMBINER_REP; ++blk) {
            yint base = head * headDim * COMBINER_REP + blk * headDim;
            for (yint z = 0; z < headDim; ++z) {
                i8 keyShfl = kState[headOffset + (z ^ blk)];
                float value = valLookup[headOffset + z];
                (*pKVState)[base + z] = ConvertToInt8(keyShfl * value);
            }
        }
    }
}


static void SoftMax(const TVector<float> &bias, const TVector<i32> &vec, float vecScale, TVector<float> *pPrediction)
{
    yint dim = YSize(bias);
    Y_ASSERT(YSize(vec) == dim);
    TSoftMaxBuf buf; // can be static
    for (yint k = 0; k < dim; ++k) {
        float w = vec[k] * vecScale + bias[k]; // can be vectorized
        buf.Add(w);
    }
    buf.SoftMax();
    pPrediction->resize(dim);
    for (yint k = 0; k < dim; ++k) {
        (*pPrediction)[k] = buf.Buf[k] * buf.Scale;
    }
}


template <class TSrc>
static void NormalizeState(TVector<i8> *pRes, const TVector<TSrc> &state, yint headCount, TVector<float> *pHeadScale)
{
    yint dim = YSize(state);
    yint headDim = dim / headCount;
    pRes->resize(dim);
    if (pHeadScale) {
        pHeadScale->resize(headCount);
    }
    for (yint h = 0; h < headCount; ++h) {
        yint offset = h * headDim;
        float sum2 = 0;
        for (yint x = 0; x < headDim; ++x) {
            sum2 += Sqr((float)state[offset + x]);
        }
        if (sum2 == 0) {
            for (yint x = 0; x < headDim; ++x) {
                (*pRes)[offset + x] = 0;
            }
            if (pHeadScale) {
                (*pHeadScale)[h] = 0;
            }
        } else {
            float sko = sqrt(sum2 / headDim);
            float discrScale = sko * VEC_SCALE;
            float mult = 1 / discrScale;
            for (yint x = 0; x < headDim; ++x) {
                (*pRes)[offset + x] = ConvertToInt8(state[offset + x] * mult);
            }
            if (pHeadScale) {
                (*pHeadScale)[h] = discrScale;
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Attention

struct TAttentionVecHistory
{
    TVector<TVector<i8>> QVState;
    TVector<TVector<float>> QVStateScaleArr;
    TVector<TVector<i8>> VState;

    void AddVectors(const TVector<i8> &qv, const TVector<float> &qvScaleArr, const TVector<i8> &v)
    {
        QVState.push_back(qv);
        QVStateScaleArr.push_back(qvScaleArr);
        VState.push_back(v);
    }
    yint GetLength() const { return YSize(VState); }
};


void ComputeValLookup(yint width, float alibiSlope, const TModelDims &dims,
    const TAttentionVecHistory &history,
    const TVector<i8> &qkState,
    TVector<float> *pValLookup)
{
    yint ttSum = dims.GetTTSum();
    yint headQDim = dims.QDim;
    yint headTTDim = dims.TTDim;

    TVector<int> toArr;
    yint cur = history.GetLength() - 1;
    if (width == 0) {
        toArr.push_back(cur);
    } else {
        yint ww = Min<yint>(cur, width);
        for (yint dt = 1; dt <= ww; ++dt) {
            toArr.push_back(cur - dt);
        }
    }

    yint toCount = YSize(toArr);
    TVector<float> weightArr;
    weightArr.resize(toCount);

    TVector<float> &valLookup = *pValLookup;
    ClearPodArray(&valLookup, ttSum);
    float attDotScale = CalcDotScale(headQDim) * CalcAttentionMult();
    for (yint head = 0; head < dims.HeadCount; ++head) {
        yint qOffset = head * headQDim;
        yint ttOffset = head * headTTDim;
        TSoftMaxBuf softMax;
        softMax.Add(0);
        for (yint z = 0; z < toCount; ++z) {
            yint to = toArr[z];
            i32 qProduct = DotInt8(qkState.data() + qOffset, history.QVState[to].data() + qOffset, headQDim);
            softMax.Add(qProduct * history.QVStateScaleArr[to][head] * attDotScale * VEC_SCALE - alibiSlope * (cur - to));
        }
        softMax.SoftMax();

        for (yint z = 0; z < toCount; ++z) {
            yint to = toArr[z];
            float w = softMax.Buf[z + 1] * softMax.Scale * VEC_SCALE;
            for (yint x = 0; x < headTTDim; ++x) {
                valLookup[ttOffset + x] += w * history.VState[to][ttOffset + x];
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// add product

static void AddLookupProduct(
    const TModelDescr &modelDescr,
    yint t,
    const TArray2D<float> &ropeBuf,
    const TCPUModelParams::TAttentionMatrices &att,
    TAttentionVecHistory *pKVCache,
    TVector<float> *pState)
{
    yint dim = modelDescr.Dims.Dim;
    yint headCount = modelDescr.Dims.HeadCount;
    Y_ASSERT(dim == YSize(*pState));
    TAttentionVecHistory &kvCache = *pKVCache;

    TVector<i8> normState;
    NormalizeState(&normState, *pState, dim / STATE_NORM_TILE, nullptr);

    TVector<float> qkSrc;
    MulForward(normState, att.QK, &qkSrc);
    TVector<float> qvSrc;
    MulForward(normState, att.QV, &qvSrc);
    TVector<i32> kSrc;
    MulForward(normState, att.K, &kSrc);
    TVector<i32> vSrc;
    MulForward(normState, att.V, &vSrc);
    ApplyRope(ropeBuf, 1, t, &qkSrc);
    ApplyRope(ropeBuf, 1, t, &qvSrc);
    TVector<i8> qk;
    NormalizeState(&qk, qkSrc, headCount, nullptr);
    TVector<i8> qv;
    TVector<float> qvHeadScale;
    NormalizeState(&qv, qvSrc, headCount, &qvHeadScale);
    TVector<i8> k;
    NormalizeState(&k, kSrc, headCount, nullptr);
    TVector<i8> v;
    NormalizeState(&v, vSrc, headCount, nullptr);
    //PrintVec(k);

    for (float &qvMult : qvHeadScale) {
        qvMult *= att.QVScale * VEC_SCALE * CalcDotScale(dim);
    }
    kvCache.AddVectors(qv, qvHeadScale, v);

    TVector<float> valLookup;
    ComputeValLookup(att.AttentionWidth, att.AlibiSlope, modelDescr.Dims, kvCache, qk, &valLookup);

    TVector<i8> kv;
    KVProduct(k, valLookup, headCount, &kv);
    TVector<i32> deltaState;
    MulForward(kv, att.Combiner, &deltaState);
    AddScaled(pState, deltaState, att.CombinerScale * VEC_SCALE * CalcDotScale(YSize(kv)));

    // relu
    NormalizeState(&normState, *pState, dim / STATE_NORM_TILE, nullptr);
    TVector<i32> reluSrc;
    MulForward(normState, att.Expand, &reluSrc);
    TVector<i8> relu;
    NormalizeState(&relu, reluSrc, headCount, nullptr);
    for (i8 &x : relu) {
        x = Max<i8>(0, x);
    }
    MulForward(relu, att.Contract, &deltaState);
    AddScaled(pState, deltaState, att.ContractScale * VEC_SCALE * CalcDotScale(YSize(relu)));
}



///////////////////////////////////////////////////////////////////////////////////////////////////
//

struct TCPUInferContext
{
    TVector<TAttentionVecHistory> KVcacheArr;
    TArray2D<float> RopeBuf;
    yint T = 0;

public:
    void Init(const TCPUModelParams &params)
    {
        yint depth = YSize(params.LayerArr);
        KVcacheArr.resize(depth);
        FillRopeBuf(&RopeBuf, params.ModelDescr.Dims.QDim, params.ModelDescr.GetWideLimitWindow());
        T = 0;
    }
};


void ComputePrediction(const TCPUModelParams &params, const TVector<TLabelIndex> &labels, TCPUInferContext *pCtx, TVector<float> *pResPrediction)
{
    TModelDescr modelDescr = params.ModelDescr;
    yint dim = modelDescr.Dims.Dim;

    // embedding
    TVector<float> state;
    ClearPodArray(&state, dim);
    for (TLabelIndex label : labels) {
        for (yint x = 0; x < dim; ++x) {
            state[x] += params.LabelEmbed[label][x] * params.LabelEmbedScale;
        }
    }

    // apply layers
    for (yint d = 0; d < YSize(params.LayerArr); ++d) {
        AddLookupProduct(modelDescr, pCtx->T, pCtx->RopeBuf, params.LayerArr[d], &pCtx->KVcacheArr[d], &state);
    }

    if (pResPrediction) {
        TVector<i8> finalState;
        NormalizeState(&finalState, state, dim / STATE_NORM_TILE, nullptr);

        TVector<i32> prediction;
        MulForward(finalState, params.FinalLayer, &prediction);

        float finalScale = CalcDotScale(dim) * CalcFinalLayerMult() * params.FinalLayerScale * VEC_SCALE;
        SoftMax(params.Bias, prediction, finalScale, pResPrediction);
    }
    pCtx->T += 1;
}


static int SampleFromDistr(TXRng &rng, const TVector<float> &distr, float temperature)
{
    // use gumbel max trick
    float best = -1e38f;
    yint res = 0;
    for (yint k = 0; k < YSize(distr); ++k) {
        //float score = distr[k] / -log(rng.GenRandReal3());
        float score = log(distr[k]) / temperature - log(-log(rng.GenRandReal3()));
        if (score > best) {
            best = score;
            res = k;
        }
    }
    return res;
}


void CpuInferenceProfile(const TCPUModelParams &cpuParams)
{
    TXRng rng(1313);
    DebugPrintf("start profiling\n");
    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        TLabelIndex prevLabel = 0;
        TCPUInferContext cpuCtx;
        cpuCtx.Init(cpuParams);
        for (yint t = 0; t < 100; ++t) {
            TVector<TLabelIndex> labelArr;
            labelArr.push_back(prevLabel);
            TVector<float> distr;
            ComputePrediction(cpuParams, labelArr, &cpuCtx, &distr);
            yint letter = rng.Uniform(cpuParams.ModelDescr.VocabSize);//SampleFromDistr(rng, distr, 1);
            prevLabel = letter + 1 + 1;
        }
        DebugPrintf("%g secs\n", NHPTimer::GetTimePassed(&tStart));
    }
}


void Check()
{
    TXRng rng(1313);

    TModelParams params;
    //Serialize(IO_READ, "D:/eden_gpt_1k.bin", params);
    //Serialize(IO_READ, "D:/eden_gpt_50k.bin", params);
    Serialize(IO_READ, "D:/models/owt_125m/eden_gpt_38k.bin", params);
    //params.LayerArr.resize(1);
    //params.ModelDescr.Layers.resize(1);

    TCPUModelParams cpuParams;
    ConvertModel(params, &cpuParams);

    //CpuInferenceProfile(cpuParams);

    const yint CHECK_BATCH_SIZE = 1;
    yint nodeCount = 100;

    TIntrusivePtr<IModel> gpuModel = CreateModel(1, params);
    //TIntrusivePtr<IComputeContext> gpuCtx = NCUDA_GPT::CreateContext(gpuModel, CHECK_BATCH_SIZE * nodeCount);
    TIntrusivePtr<IComputeContext> gpuCtx = NCPU_GPT::CreateContext(gpuModel, CHECK_BATCH_SIZE * nodeCount);

    yint baseLabel = params.GetModelDescr().HasFlag(MPF_DISABLE_NOISE_LABELS) ? 0 : NOISE_LABELS_COUNT;
    yint fragmentStartToken = 0;

    TFragment frag;
    frag.Text.push_back(fragmentStartToken); // start of fragment token
    TBPEToken prevToken = fragmentStartToken;
    TCPUInferContext cpuCtx;
    cpuCtx.Init(cpuParams);
    float gpuLoss = 0;
    float cpuLoss = 0;
    for (yint t = 0; t < 30; ++t) {
        TVector<TFragment> xxFrag;
        xxFrag.push_back(frag);
        MakeTest(xxFrag, gpuCtx.Get(), MAIN_DEVICE);
        TVector<TVector<float>> gpuPredArr;
        gpuCtx->ComputeFragmentPredictions(&gpuPredArr);

        TVector<TLabelIndex> labelArr;
        labelArr.push_back(cpuParams.BaseLabel + prevToken);
        TVector<float> cpuDistr;
        ComputePrediction(cpuParams, labelArr, &cpuCtx, &cpuDistr);

        for (yint k = 0; k < 5; ++k) {
            DebugPrintf("%g - %g\n", cpuDistr[k], gpuPredArr.back()[k]);
        }
        DebugPrintf("\n");

        //yint letter = rng.Uniform(params.GetModelDescr().VocabSize);
        //yint letter = SampleFromDistr(rng, cpuDistr, 1);
        yint letter = SampleFromDistr(rng, gpuPredArr.back(), 1);
        prevToken = letter;
        frag.Text.push_back(letter);

        cpuLoss -= log(cpuDistr[letter]);
        gpuLoss -= log(gpuPredArr.back()[letter]);
    }
    DebugPrintf("cpu loss %g\ngpu loss %g\n", cpuLoss, gpuLoss);
}
}
