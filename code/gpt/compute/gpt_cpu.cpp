#include "stdafx.h"
#include "gpt_cpu.h"
#include <gpt/att/rope.h>
#include <gpt/data/data.h>
#include <lib/random/rand_utils.h>
#include <lib/math/matrix_utils.h>
#include <xmmintrin.h> // for SSE intrinsics


namespace NCPU_GPT
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// Convert a float to an int
static int Float2I8(float arg)
{
    int x = _mm_cvt_ss2si(_mm_set_ss(arg));
    x = (x > 127) ? 127 : x;
    x = (x < -128) ? -128 : x;
    return x;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TMatrixFloat>
static TArray2D<float> GetData(TIntrusivePtr<TParModelMatrix<TMatrixFloat>> p)
{
    TArray2D<float> res;
    p->GetFastFloatData(&res);
    return res;
}


template <class T>
static void PrintVec(int t, const TArray2D<T> &arr)
{
    for (int k = 0; k < arr.GetXSize(); ++k) {
        printf("cpu vec[%g] = %g\n", k * 1., arr[t][k] * 1.);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// 
template <class T1, class T2>
static void CopyMatrix(TVector<TVector<T1>> *p, const TArray2D<T2> &src)
{
    p->resize(src.GetYSize());
    for (yint y = 0; y < src.GetYSize(); ++y) {
        (*p)[y].resize(src.GetXSize());
        for (yint x = 0; x < src.GetXSize(); ++x) {
            (*p)[y][x] = src[y][x];
        }
    }
}


template <class T1, class T2>
static void InitDeltaMatrix(TArray2D<T1> *p, const TArray2D<T2> &src)
{
    p->SetSizes(src.GetXSize(), src.GetYSize());
    p->FillZero();
}


template <class T1, class T2, class T3>
static void AddScaledMatrix(TArray2D<T1> *p, const TArray2D<T2> &src, T3 scale)
{
    yint xSize = src.GetXSize();
    yint ySize = src.GetYSize();
    Y_ASSERT(p->GetXSize() == xSize);
    Y_ASSERT(p->GetYSize() == ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] += src[y][x] * scale;
        }
    }
}


template <class T1, class T2>
static void ScaleMatrix(TArray2D<T1> *p, T2 scale)
{
    yint xSize = p->GetXSize();
    yint ySize = p->GetYSize();
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] *= scale;
        }
    }
}


template <class T1>
static void Relu(TArray2D<T1> *p)
{
    yint xSize = p->GetXSize();
    yint ySize = p->GetYSize();
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] = Max<T1>(0, (*p)[y][x]);
        }
    }
}


template <class T1, class T2>
static void BackpropRelu(const TArray2D<T1> &vals, TArray2D<T2> *grad)
{
    yint xSize = vals.GetXSize();
    yint ySize = vals.GetYSize();
    Y_ASSERT(xSize == grad->GetXSize());
    Y_ASSERT(ySize == grad->GetYSize());
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            if (vals[y][x] < 0) {
                (*grad)[y][x] = 0;
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// linear algebra

// resArr = kqv @ vecArr
static TArray2D<float> MulForward(const TArray2D<float> &vecArr, const TArray2D<float> &kqv)
{
    yint len = vecArr.GetYSize();
    yint dim = vecArr.GetXSize();
    yint rDim = kqv.GetYSize();
    Y_ASSERT(dim == kqv.GetXSize());
    Y_ASSERT(rDim == kqv.GetYSize());
    TArray2D<float> resArr;
    resArr.SetSizes(rDim, len);
    float normScale = CalcDotScale(dim);
    for (yint t = 0; t < len; ++t) {
        for (yint k = 0; k < rDim; ++k) {
            float res = 0;
            for (yint x = 0; x < dim; ++x) {
                res += vecArr[t][x] * kqv[k][x];
            }
            resArr[t][k] = res * normScale;
        }
    }
    return resArr;
}


static void MulBackwardWithAccum(TArray2D<float> *pVecArrGrad, const TArray2D<float> &kqv, const TArray2D<float> &resArrGrad)
{
    yint len = resArrGrad.GetYSize();
    yint dim = kqv.GetXSize();
    yint rDim = resArrGrad.GetXSize();
    Y_ASSERT(dim == kqv.GetXSize());
    Y_ASSERT(rDim == kqv.GetYSize());
    float normScale = CalcDotScale(dim);
    for (yint t = 0; t < len; ++t) {
        for (yint x = 0; x < dim; ++x) {
            float res = 0;
            for (yint k = 0; k < rDim; ++k) {
                res += resArrGrad[t][k] * kqv[k][x];
            }
            (*pVecArrGrad)[t][x] += res * normScale;
        }
    }
}


static void SumRankOne(const TArray2D<float> &vecArr, TArray2D<float> *pDelta, const TArray2D<float> &resArrGrad)
{
    yint len = vecArr.GetYSize();
    yint dim = vecArr.GetXSize();
    yint rDim = resArrGrad.GetXSize();
    Y_ASSERT(len == resArrGrad.GetYSize());
    pDelta->SetSizes(dim, rDim);
    pDelta->FillZero();
    for (yint k = 0; k < rDim; ++k) {
        for (yint x = 0; x < dim; ++x) {
            float res = 0;
            for (yint t = 0; t < len; ++t) {
                res += resArrGrad[t][k] * vecArr[t][x];
            }
            (*pDelta)[k][x] += res;
        }
    }
}


static void KVProduct(const TModelDims &dims, const TArray2D<float> &k, const TArray2D<float> &valLookup,
    TArray2D<float> *pKV)
{
    yint ttSum = dims.GetTTSum();
    yint headTTDim = dims.TTDim;
    yint headCount = dims.HeadCount;
    yint combinerDim = dims.GetCombinerDim();
    yint len = k.GetYSize();
    Y_ASSERT(k.GetXSize() == ttSum);
    Y_ASSERT(valLookup.GetXSize() == ttSum);
    Y_ASSERT(valLookup.GetYSize() == len);
    pKV->SetSizes(combinerDim, len);
    for (yint t = 0; t < len; ++t) {
        for (yint head = 0; head < headCount; ++head) {
            yint headOffset = head * headTTDim;
            for (int blk = 0; blk < COMBINER_REP; ++blk) {
                yint base = head * headTTDim * COMBINER_REP + blk * headTTDim;
                for (yint z = 0; z < headTTDim; ++z) {
                    float keyShfl = k[t][headOffset + (z ^ blk)];
                    float value = valLookup[t][headOffset + z];
                    (*pKV)[t][base + z] = keyShfl * value;
                }
            }
        }
    }
}


static void KVProductBackprop(const TModelDims &dims, const TArray2D<float> &k, const TArray2D<float> &valLookup, const TArray2D<float> &dkv,
    TArray2D<float> *pDK, TArray2D<float> *pDValLookup, TArray2D<float> *pDScale)
{
    yint ttSum = dims.GetTTSum();
    yint headTTDim = dims.TTDim;
    yint headCount = dims.HeadCount;
    yint combinerDim = dims.GetCombinerDim();
    yint len = k.GetYSize();
    Y_ASSERT(k.GetXSize() == ttSum);
    Y_ASSERT(valLookup.GetXSize() == ttSum);
    Y_ASSERT(valLookup.GetYSize() == len);
    Y_ASSERT(dkv.GetXSize() == combinerDim);
    Y_ASSERT(dkv.GetYSize() == len);

    TArray2D<float> &dKey = *pDK;
    dKey.SetSizes(ttSum, len);
    dKey.FillZero();
    TArray2D<float> &dValLookup = *pDValLookup;
    dValLookup.SetSizes(ttSum, len);
    dValLookup.FillZero();
    for (yint t = 0; t < len; ++t) {
        for (yint head = 0; head < headCount; ++head) {
            yint headOffset = head * headTTDim;
            for (int blk = 0; blk < COMBINER_REP; ++blk) {
                yint base = head * headTTDim * COMBINER_REP + blk * headTTDim;
                for (yint z = 0; z < headTTDim; ++z) {
                    float keyShfl = k[t][headOffset + (z ^ blk)];
                    float valueShfl = valLookup[t][headOffset + (z ^ blk)];
                    float grad = dkv[t][base + z];
                    float gradShfl = dkv[t][base + (z ^ blk)];
                    dKey[t][headOffset + z] += gradShfl * valueShfl;
                    dValLookup[t][headOffset + z] += grad * keyShfl;
                }
            }
        }
    }
    TArray2D<float> &dScale = *pDScale;
    dScale.SetSizes(len, dims.HeadCount);
    dScale.FillZero();
    for (yint t = 0; t < len; ++t) {
        for (yint k = 0; k < ttSum; ++k) {
            yint head = k / headTTDim;
            dScale[head][t] += dValLookup[t][k] * valLookup[t][k];
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// 
static void SoftMax(const TArray2D<float> &vecArr, TVector<TVector<float>> *pPrediction, const TVector<float> &bias)
{
    yint len = vecArr.GetYSize();
    yint dim = YSize(bias);
    Y_ASSERT(vecArr.GetXSize() == dim);
    pPrediction->resize(len);
    for (yint t = 0; t < len; ++t) {
        TVector<float> &dst = (*pPrediction)[t];
        dst.resize(dim);
        float maxVal = 0;
        for (yint k = 0; k < dim; ++k) {
            maxVal = fmaxf(maxVal, vecArr[t][k]);
        }
        double sumWeight = 0;
        for (yint k = 0; k < dim; ++k) {
            float val = vecArr[t][k];
            float w = exp2(val + bias[k] - maxVal);
            dst[k] = w;
            sumWeight += w;
        }
        float scale = 1 / sumWeight;
        for (yint k = 0; k < dim; ++k) {
            dst[k] *= scale;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
//
static TArray2D<float> NormalizeState(const TArray2D<float> &state, yint headCount)
{
    yint len = state.GetYSize();
    yint dim = state.GetXSize();
    yint headDim = dim / headCount;
    TArray2D<float> res;
    res.SetSizes(dim, len);
    for (yint t = 0; t < len; ++t) {
        for (yint h = 0; h < headCount; ++h) {
            yint offset = h * headDim;
            float sum2 = 0;
            for (yint x = 0; x < headDim; ++x) {
                sum2 += Sqr(state[t][offset + x]);
            }
            if (sum2 == 0) {
                for (yint x = 0; x < headDim; ++x) {
                    res[t][offset + x] = 0;
                }
            } else {
                float scale = sqrt(headDim / sum2);
                for (yint x = 0; x < headDim; ++x) {
                    res[t][offset + x] = state[t][offset + x] * scale;
                }
            }
        }
    }
    return res;
}


static void NormalizeStateBackward(const TArray2D<float> &state, yint headCount, const TArray2D<float> &dNormState, TArray2D<float> *pGrad)
{
    yint len = state.GetYSize();
    yint dim = state.GetXSize();
    yint headDim = dim / headCount;
    pGrad->SetSizes(dim, len);
    for (yint t = 0; t < len; ++t) {
        for (yint h = 0; h < headCount; ++h) {
            yint offset = h * headDim;
            float sum2 = 0;
            float dp = 0;
            for (yint x = 0; x < headDim; ++x) {
                float src = state[t][offset + x];
                float grad = dNormState[t][offset + x];
                sum2 += Sqr(src);
                dp += src * grad;
            }
            if (sum2 == 0) {
                for (yint x = 0; x < headDim; ++x) {
                    (*pGrad)[t][offset + x] = 0;
                }
            } else {
                float sigma = dp / sum2;
                float scale = sqrt(headDim / sum2);
                for (yint x = 0; x < headDim; ++x) {
                    float src = state[t][offset + x];
                    float grad = dNormState[t][offset + x];
                    (*pGrad)[t][offset + x] = scale * (grad - src * sigma);
                }
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Attention
// attention related compute
struct TAttentionComputer
{
    TArray2D<float> SumWeight;
    TArray2D<float> MaxDP;
    float AttDotScale = 0;
    float AlibiSlope = 0;

    TAttentionComputer(const TModelDims &dims, float alibiSlope) : AlibiSlope(alibiSlope)
    {
        AttDotScale = CalcDotScale(dims.QDim) * CalcAttentionMult();
    }

    float CalcDP(const TArray2D<float> &qk, const TArray2D<float> &qv,
        yint headQDim, yint qOffset, yint from, yint to)
    {
        float sum = 0;
        for (yint x = 0; x < headQDim; ++x) {
            sum += qk[from][qOffset + x] * qv[to][qOffset + x];
        }
        float dp = sum * AttDotScale;
        dp += GetAttentionDecay(from - to, AlibiSlope);
        return dp;
    }

    void ComputeValLookup(yint len, const TModelDims &dims,
        const TArray2D<float> &qk, const TArray2D<float> &qv, const TArray2D<float> &v,
        const TAttentionInfo &attInfo,
        TArray2D<float> *pValLookup)
    {
        yint ttSum = dims.GetTTSum();
        yint headQDim = dims.QDim;
        yint headTTDim = dims.TTDim;
        TVector<float> valLookup;
        pValLookup->SetSizes(ttSum, len);
        SumWeight.SetSizes(len, dims.HeadCount);
        MaxDP.SetSizes(len, dims.HeadCount);

        // compute weighted sum of val vectors
        for (yint head = 0; head < dims.HeadCount; ++head) {
            yint qOffset = head * headQDim;
            yint ttOffset = head * headTTDim;
            for (yint from = 0; from < len; ++from) {
                // find max dp
                float maxDP = 0;
                for (yint attIndex = attInfo.SpanPtr[from]; attIndex < attInfo.SpanPtr[from + 1]; ++attIndex) {
                    const TAttentionSpan &span = attInfo.Spans[attIndex];
                    for (yint to = span.Start; to <= span.Finish; ++to) {
                        float dp = CalcDP(qk, qv, headQDim, qOffset, from, to);
                        maxDP = fmaxf(maxDP, dp);
                    }
                }
                // sum
                float sumWeight = exp2(-maxDP); // initialize with zero vector of weight 1
                ClearPodArray(&valLookup, headTTDim);
                for (yint attIndex = attInfo.SpanPtr[from]; attIndex < attInfo.SpanPtr[from + 1]; ++attIndex) {
                    const TAttentionSpan &span = attInfo.Spans[attIndex];
                    for (yint to = span.Start; to <= span.Finish; ++to) {
                        float dp = CalcDP(qk, qv, headQDim, qOffset, from, to);
                        float w = exp2(dp - maxDP);
                        sumWeight += w;
                        for (yint x = 0; x < headTTDim; ++x) {
                            valLookup[x] += w * v[to][ttOffset + x];
                        }
                    }
                }
                Y_ASSERT(sumWeight > 0);
                SumWeight[head][from] = sumWeight;
                MaxDP[head][from] = maxDP;
                float sumWeight1 = 1 / sumWeight;
                for (yint x = 0; x < headTTDim; ++x) {
                    (*pValLookup)[from][ttOffset + x] = valLookup[x] * sumWeight1;
                }
            }
        }
    }

    struct TGradData
    {
        const TArray2D<float> &QK;
        const TArray2D<float> &QV;
        const TArray2D<float> &V;
        const TArray2D<float> &DValLookupArr;
        const TArray2D<float> &DScaleArr;
        yint HeadQDim = 0;
        yint HeadTTDim = 0;

        TGradData(const TModelDims &dims,
            const TArray2D<float> &qk, const TArray2D<float> &qv, const TArray2D<float> &v,
            const TArray2D<float> &dValLookupArr, const TArray2D<float> &dScaleArr)
            : QK(qk), QV(qv), V(v)
            , DValLookupArr(dValLookupArr)
            , DScaleArr(dScaleArr)
        {
            HeadQDim = dims.QDim;
            HeadTTDim = dims.TTDim;
        }
    };

    float CalcDDot(const TGradData &data,
        yint head, yint qOffset, yint ttOffset, yint from, yint to,
        float dp, float maxDP, float sumWeight)
    {
        float w = exp2(dp - maxDP) / sumWeight;
        Y_ASSERT(!isnan(w) && isfinite(w));

        float dW = 0;
        for (yint x = 0; x < data.HeadTTDim; ++x) {
            float dValLookup = data.DValLookupArr[from][ttOffset + x];
            float val = data.V[to][ttOffset + x]; // val2
            dW += dValLookup * val;
        }

        float dScale = data.DScaleArr[head][from];
        float dDot = w * (dW - dScale) * AttDotScale * LOG2;
        return dDot;
    }

    void GradQK(yint len, const TModelDims &dims,
        const TGradData &data,
        const TAttentionInfo &attInfo,
        TArray2D<float> *pDQK)
    {
        yint qSum = dims.GetQSum();
        yint ttSum = dims.GetTTSum();
        yint headQDim = dims.QDim;
        yint headTTDim = dims.TTDim;
        TArray2D<float> &dqk = *pDQK;
        dqk.SetSizes(qSum, len);
        dqk.FillZero();
        for (yint head = 0; head < dims.HeadCount; ++head) {
            yint qOffset = head * headQDim;
            yint ttOffset = head * headTTDim;
            for (yint from = 0; from < len; ++from) {
                for (yint attIndex = attInfo.SpanPtr[from]; attIndex < attInfo.SpanPtr[from + 1]; ++attIndex) {
                    const TAttentionSpan &span = attInfo.Spans[attIndex];
                    float sumWeight = SumWeight[head][from];
                    float maxDP = MaxDP[head][from];
                    for (yint to = span.Start; to <= span.Finish; ++to) {
                        float dp = CalcDP(data.QK, data.QV, data.HeadQDim, qOffset, from, to);
                        float dDot = CalcDDot(data, head, qOffset, ttOffset, from, to, dp, maxDP, sumWeight);
                        for (yint x = 0; x < headQDim; ++x) {
                            dqk[from][qOffset + x] += dDot * data.QV[to][qOffset + x];
                        }
                    }
                }
            }
        }
    }

    void GradQV(yint len, const TModelDims &dims,
        const TGradData &data,
        const TAttentionInfo &revAttInfo,
        TArray2D<float> *pDQV,
        TArray2D<float> *pDV)
    {
        yint qSum = dims.GetQSum();
        yint ttSum = dims.GetTTSum();
        TArray2D<float> &dqv = *pDQV;
        dqv.SetSizes(qSum, len);
        dqv.FillZero();
        TArray2D<float> &dv = *pDV;
        dv.SetSizes(ttSum, len);
        dv.FillZero();
        for (yint head = 0; head < dims.HeadCount; ++head) {
            yint qOffset = head * data.HeadQDim;
            yint ttOffset = head * data.HeadTTDim;
            for (yint to = 0; to < len; ++to) {
                for (yint attIndex = revAttInfo.SpanPtr[to]; attIndex < revAttInfo.SpanPtr[to + 1]; ++attIndex) {
                    const TAttentionSpan &span = revAttInfo.Spans[attIndex];
                    for (yint from = span.Start; from <= span.Finish; ++from) {
                        float sumWeight = SumWeight[head][from];
                        float maxDP = MaxDP[head][from];

                        float dp = CalcDP(data.QK, data.QV, data.HeadQDim, qOffset, from, to);
                        float dDot = CalcDDot(data, head, qOffset, ttOffset, from, to, dp, maxDP, sumWeight);
                        float w = exp2(dp - maxDP) / sumWeight;

                        for (yint x = 0; x < data.HeadQDim; ++x) {
                            dqv[to][qOffset + x] += dDot * data.QK[from][qOffset + x];
                        }

                        for (yint x = 0; x < data.HeadTTDim; ++x) {
                            dv[to][ttOffset + x] += w * data.DValLookupArr[from][ttOffset + x];
                        }
                    }
                }
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// add product

struct TFragmentStates
{
    TArray2D<float> State;

    void SetLength(yint len, yint dim)
    {
        State.SetSizes(dim, len);
        State.FillZero();
    }
    void Clear()
    {
        State.FillZero();
    }
};


struct TAttentionFB
{
    TAttentionInfo Att;
    TAttentionInfo RevAtt;

    void Assign(const TAttentionInfo &att)
    {
        Att = att;
        RevAtt = TransposeAttention(att);
    }
};


// compute product of two attention lookups
static void AddLookupProduct(
    const TModelDescr &modelDescr,
    const TArray2D<float> &ropeBuf,
    const TVector<const TAttentionParams *> &layerAtt,
    const TVector<TAttentionFB> &attFBArr,
    const TFragmentStates &prevState, TFragmentStates *pState, TFragmentStates *pStateRelu)
{
    const TModelDims &dims = modelDescr.Dims;
    yint dim = dims.Dim;
    yint len = prevState.State.GetYSize();
    yint headCount = modelDescr.Dims.HeadCount;
    yint reluDim = dims.GetReluDim();
    Y_ASSERT(dim == prevState.State.GetXSize());
    Y_ASSERT(YSize(layerAtt) == 1);

    TArray2D<float> normState = NormalizeState(prevState.State, dim / STATE_NORM_TILE);

    pState->State = prevState.State;
    for (const TAttentionParams *pAtt: layerAtt) {
        auto &attCombiner = pAtt->MatrArr[MP_ATT_COMBINER];
        auto &attQK = pAtt->MatrArr[MP_ATT_QK];
        auto &attQV = pAtt->MatrArr[MP_ATT_QV];
        auto &attK = pAtt->MatrArr[MP_ATT_K];
        auto &attV = pAtt->MatrArr[MP_ATT_V];

        TAttentionComputer attComp(modelDescr.Dims, pAtt->AlibiSlope);
        const TAttentionInfo &attInfo = attFBArr[pAtt->AttentionWidthId].Att;

        TArray2D<float> qkSrc = MulForward(normState, GetData(attQK));
        TArray2D<float> qvSrc = MulForward(normState, GetData(attQV));
        TArray2D<float> kSrc = MulForward(normState, GetData(attK));
        TArray2D<float> vSrc = MulForward(normState, GetData(attV));

        TArray2D<float> qk = NormalizeState(qkSrc, headCount);
        TArray2D<float> qv = qvSrc;
        TArray2D<float> k = NormalizeState(kSrc, headCount);
        TArray2D<float> v = NormalizeState(vSrc, headCount);
        ApplyRope(ropeBuf, 1, &qk);
        ApplyRope(ropeBuf, 1, &qv);

        TArray2D<float> valLookup;
        attComp.ComputeValLookup(len, modelDescr.Dims, qk, qv, v, attInfo, &valLookup);

        TArray2D<float> kv;
        KVProduct(dims, k, valLookup, &kv);
        TArray2D<float> deltaState = MulForward(kv, GetData(attCombiner));
        AddScaledMatrix(&pState->State, deltaState, 1);
    }

    pStateRelu->State = pState->State;
    normState = NormalizeState(pState->State, dim / STATE_NORM_TILE);
    for (const TAttentionParams *pAtt : layerAtt) {
        auto &attExpand = pAtt->MatrArr[MP_RELU_EXPAND];
        auto &attContract = pAtt->MatrArr[MP_RELU_CONTRACT];

        TArray2D<float> relu = MulForward(normState, GetData(attExpand));
        TArray2D<float> reluNorm = NormalizeState(relu, reluDim / STATE_NORM_TILE);
        Relu(&reluNorm);

        TArray2D<float> deltaState = MulForward(reluNorm, GetData(attContract));
        AddScaledMatrix(&pStateRelu->State, deltaState, 1);
    }
}


// add gradient of product of two attention lookups
static void AddLookupProductBackprop(
    const TModelDescr &modelDescr,
    const TArray2D<float> &ropeBuf,
    const TVector<const TAttentionParams *> &layerAtt,
    const TVector<TAttentionFB> &attFBArr,
    const TFragmentStates &prevState, const TFragmentStates &prevStateRelu,
    TFragmentStates *pGrad
    )
{
    const TModelDims &dims = modelDescr.Dims;
    yint dim = dims.Dim;
    yint len = prevState.State.GetYSize();
    yint headCount = modelDescr.Dims.HeadCount;
    yint reluDim = dims.GetReluDim();
    Y_ASSERT(YSize(layerAtt) == 1);

    // backprop relu
    TArray2D<float> normState = NormalizeState(prevStateRelu.State, dim / STATE_NORM_TILE);
    TArray2D<float> dNormState;
    InitDeltaMatrix(&dNormState, normState);
    for (const TAttentionParams *pAtt : layerAtt) {
        auto &attExpand = pAtt->MatrArr[MP_RELU_EXPAND];
        auto &attContract = pAtt->MatrArr[MP_RELU_CONTRACT];

        TArray2D<float> relu = MulForward(normState, GetData(attExpand));
        TArray2D<float> reluNorm = NormalizeState(relu, reluDim / STATE_NORM_TILE);
        Relu(&reluNorm);

        TArray2D<float> drelu;
        InitDeltaMatrix(&drelu, relu);
        MulBackwardWithAccum(&drelu, GetData(attContract), pGrad->State);
        TArray2D<float> deltaContract;
        SumRankOne(reluNorm, &deltaContract, pGrad->State);

        BackpropRelu(relu, &drelu);
        NormalizeStateBackward(relu, reluDim / STATE_NORM_TILE, drelu, &drelu);

        MulBackwardWithAccum(&dNormState, GetData(attExpand), drelu);
        TArray2D<float> deltaExpand;
        SumRankOne(normState, &deltaExpand, drelu);

        attExpand->ApplyDelta(deltaExpand);
        attContract->ApplyDelta(deltaContract);
    }
    TArray2D<float> deltaStateGrad;
    NormalizeStateBackward(prevStateRelu.State, dim / STATE_NORM_TILE, dNormState, &deltaStateGrad);
    AddScaledMatrix(&pGrad->State, deltaStateGrad, 1);


    // backprop att
    normState = NormalizeState(prevState.State, dim / STATE_NORM_TILE);
    InitDeltaMatrix(&dNormState, normState);
    for (const TAttentionParams *pAtt : layerAtt) {
        auto &attCombiner = pAtt->MatrArr[MP_ATT_COMBINER];
        auto &attQK = pAtt->MatrArr[MP_ATT_QK];
        auto &attQV = pAtt->MatrArr[MP_ATT_QV];
        auto &attK = pAtt->MatrArr[MP_ATT_K];
        auto &attV = pAtt->MatrArr[MP_ATT_V];

        TAttentionComputer attComp(dims, pAtt->AlibiSlope);
        const TAttentionInfo &attInfo = attFBArr[pAtt->AttentionWidthId].Att;
        const TAttentionInfo &revAttInfo = attFBArr[pAtt->AttentionWidthId].RevAtt;

        // recompute forward pass (could keep them)
        TArray2D<float> qkSrc = MulForward(normState, GetData(attQK));
        TArray2D<float> qvSrc = MulForward(normState, GetData(attQV));
        TArray2D<float> kSrc = MulForward(normState, GetData(attK));
        TArray2D<float> vSrc = MulForward(normState, GetData(attV));

        TArray2D<float> qk = NormalizeState(qkSrc, headCount);
        TArray2D<float> qv = qvSrc;
        TArray2D<float> k = NormalizeState(kSrc, headCount);
        TArray2D<float> v = NormalizeState(vSrc, headCount);
        ApplyRope(ropeBuf, 1, &qk);
        ApplyRope(ropeBuf, 1, &qv);

        TArray2D<float> valLookup;
        attComp.ComputeValLookup(len, modelDescr.Dims, qk, qv, v, attInfo, &valLookup);
        //PrintVec(10, valLookup);

        TArray2D<float> kv;
        KVProduct(dims, k, valLookup, &kv);
        TArray2D<float> dkv;
        InitDeltaMatrix(&dkv, kv);
        MulBackwardWithAccum(&dkv, GetData(attCombiner), pGrad->State);
        TArray2D<float> deltaCombiner;
        SumRankOne(kv, &deltaCombiner, pGrad->State);

        TArray2D<float> dK;
        TVector<float> dKscale;
        TArray2D<float> dValLookup;
        TArray2D<float> dScale;
        KVProductBackprop(dims, k, valLookup, dkv, &dK, &dValLookup, &dScale);

        TAttentionComputer::TGradData gradData(dims, qk, qv, v, dValLookup, dScale);

        TArray2D<float> dQK;
        attComp.GradQK(len, modelDescr.Dims, gradData, attInfo, &dQK);
        TArray2D<float> dQV;
        TArray2D<float> dV;
        attComp.GradQV(len, modelDescr.Dims, gradData, revAttInfo, &dQV, &dV);
        ApplyRope(ropeBuf, -1, &dQK);
        ApplyRope(ropeBuf, -1, &dQV);

        NormalizeStateBackward(qkSrc, headCount, dQK, &dQK);
        NormalizeStateBackward(kSrc, headCount, dK, &dK);
        NormalizeStateBackward(vSrc, headCount, dV, &dV);

        MulBackwardWithAccum(&dNormState, GetData(attQK), dQK);
        MulBackwardWithAccum(&dNormState, GetData(attQV), dQV);
        MulBackwardWithAccum(&dNormState, GetData(attK), dK);
        MulBackwardWithAccum(&dNormState, GetData(attV), dV);

        TArray2D<float> deltaQK;
        SumRankOne(normState, &deltaQK, dQK);
        TArray2D<float> deltaQV;
        SumRankOne(normState, &deltaQV, dQV);
        TArray2D<float> deltaK;
        SumRankOne(normState, &deltaK, dK);
        TArray2D<float> deltaV;
        SumRankOne(normState, &deltaV, dV);

        attCombiner->ApplyDelta(deltaCombiner);
        attQK->ApplyDelta(deltaQK);
        attQV->ApplyDelta(deltaQV);
        attK->ApplyDelta(deltaK);
        attV->ApplyDelta(deltaV);
    }
    NormalizeStateBackward(prevState.State, dim / STATE_NORM_TILE, dNormState, &deltaStateGrad);
    AddScaledMatrix(&pGrad->State, deltaStateGrad, 1);

    // can normalize pGrad, all deltas are normalized anyway
}


///////////////////////////////////////////////////////////////////////////////////////////////////
//
class TComputeContext : public IComputeContext
{
    TIntrusivePtr<IModel> Model;
    TVector<TVector<const TAttentionParams *>> LayerArr;
    TVector<TFragmentStates> AllStates;
    TVector<TLabelIndex> LabelArr;
    TVector<ui32> LabelPtr;
    TVector<TNodeTarget> KeepTarget;
    TVector<TAttentionFB> AttArr;
    TArray2D<float> RopeBuf;
    yint MaxNodeCount = 0;
    TArray2D<float> FinalNormState;
    TNodesBatch Nodes;
    TVector<ui32> DropTable;
public:
    TComputeContext(TIntrusivePtr<IModel> model, yint nodeCount) : Model(model), MaxNodeCount(nodeCount)
    {
        TModelDescr modelDescr = Model->GetModelDescr();
        LayerArr.resize(YSize(modelDescr.Layers));
        for (yint d = 0; d < YSize(modelDescr.Layers); ++d) {
            LayerArr[d].push_back(&Model->GetAttention(d));
        }
        FillRopeBuf(&RopeBuf, modelDescr.Dims.QDim, nodeCount);
    }

    IModel *GetModel() override
    {
        return Model.Get();
    }

    void OnParamsUpdate() override 
    {
    }

    yint GetDeviceCount() override
    {
        return 1;
    }

    TModelDescr GetModelDescr() override
    {
        return Model->GetModelDescr();
    }

    float GetAvrgTrainErr() override
    {
        Y_ASSERT(0 && "not implemented");
        return 0;
    }

    void WaitUpdates() override
    {
        Model->WaitDelayedCompute();
    }

    TNodesBatch &GetNodes(yint deviceId) override
    {
        Y_ASSERT(deviceId == 0);
        return Nodes;
    }

    TVector<ui32> &GetDropTable(yint deviceId) override
    {
        Y_ASSERT(deviceId == 0);
        return DropTable;
    }

    void Init(yint deviceId, EInitType initType) override
    {
        (void)initType;
        Y_ASSERT(deviceId == 0);
        Y_ASSERT(DropTable[0] == 0xffffffff); // TODO support dropout in cpu implementation
        TModelDescr modelDescr = Model->GetModelDescr();
        yint len = Nodes.GetNodeCount();
        Y_ASSERT(len <= MaxNodeCount);
        yint depth = YSize(modelDescr.Layers);
        AllStates.resize(depth * 2 + 1);
        for (yint k = 0; k < YSize(AllStates); ++k) {
            AllStates[k].SetLength(len, modelDescr.Dims.Dim);
        }
        LabelArr = Nodes.LabelArr;
        LabelPtr = Nodes.LabelPtr;
        KeepTarget = Nodes.Target;
        yint attentionWidthCount = modelDescr.GetAttentionWidthCount();
        AttArr.resize(attentionWidthCount);
        for (yint wa = 0; wa < attentionWidthCount; ++wa) {
            AttArr[wa].Assign(Nodes.AttArr[wa]);
        }
    }

    void ComputeEmbedding(const TArray2D<float> &labelEmbed)
    {
        TModelDescr modelDescr = Model->GetModelDescr();
        int dim = modelDescr.Dims.Dim;
        yint len = YSize(LabelPtr) - 1;

        AllStates[0].State.FillZero();
        for (yint t = 0; t < len; ++t) {
            for (yint k = LabelPtr[t], kFinish = LabelPtr[t + 1]; k < kFinish; ++k) {
                yint label = LabelArr[k];
                float w = (label & LABEL_NEGATIVE) ? -1 : 1;
                label &= LABEL_MASK;
                for (yint x = 0; x < dim; ++x) {
                    AllStates[0].State[t][x] += labelEmbed[label][x] * w;
                }
            }
        }
    }

    void ComputeForward(TVector<TVector<float>> *pPrediction, TVector<TVector<float>> *pStateVectors)
    {
        TModelDescr modelDescr = Model->GetModelDescr();
        int dim = modelDescr.Dims.Dim;

        auto labelEmbed = Model->GetEmbedding();
        auto finalLayer = Model->GetFinalLayer();

        // embedding
        ComputeEmbedding(GetData(labelEmbed));
        labelEmbed->AllowDelayedUpdates();

        // apply layers
        for (yint d = 0; d < YSize(LayerArr); ++d) {
            AddLookupProduct(modelDescr, RopeBuf, LayerArr[d], AttArr, AllStates[d * 2], &AllStates[d * 2 + 1], &AllStates[d * 2 + 2]);
        }

        FinalNormState = NormalizeState(AllStates.back().State, dim / STATE_NORM_TILE);

        if (pStateVectors) {
            CopyMatrix(pStateVectors, AllStates.back().State);
        }

        if (pPrediction) {
            TArray2D<float> predictionArr = MulForward(FinalNormState, GetData(finalLayer));
            ScaleMatrix(&predictionArr, CalcFinalLayerMult());
            SoftMax(predictionArr, pPrediction, Model->GetBias());
        }
    }

    void ComputeFinalStateVectors(TVector<TVector<float>> *pStateVectors) override
    {
        Model->WaitDelayedCompute();
        ComputeForward(0, pStateVectors);
    }

    void ComputeFragmentPredictions(TVector<TVector<float>> *pPrediction) override
    {
        Model->WaitDelayedCompute();
        ComputeForward(pPrediction, 0);
    }

    void ComputeFragmentPredictions(TVector<float> *pPrediction) override
    {
        TVector<TVector<float>> prediction;
        ComputeFragmentPredictions(&prediction);
        ClearPodArray(pPrediction, YSize(prediction));
        for (const TNodeTarget &nt : KeepTarget) {
            (*pPrediction)[nt.Node] = prediction[nt.Node][nt.TargetId];
        }
    }

    float ComputeScore() override
    {
        TVector<TVector<float>> prediction;
        ComputeFragmentPredictions(&prediction);
        float sum = 0;
        yint count = 0;
        for (const TNodeTarget &nt : KeepTarget) {
            sum += log(prediction[nt.Node][nt.TargetId]);
            count += 1;
        }
        return sum / count;
    }

    void Backprop(const TTrainingStep &step, EAddToModel addToModel) override
    {
        TModelDescr modelDescr = Model->GetModelDescr();
        yint len = YSize(LabelPtr) - 1;
        int dim = modelDescr.Dims.Dim;

        auto labelEmbed = Model->GetEmbedding();
        auto finalLayer = Model->GetFinalLayer();

        TVector<TVector<float>> predArr;
        ComputeForward(&predArr, 0);
        Y_ASSERT(YSize(predArr) == len);

        Model->WaitCompute();
        Model->StartIteration(step, addToModel);

        TFragmentStates grad;
        grad.SetLength(len, dim);
        {
            // final soft max gradient
            TArray2D<float> gradArr;
            gradArr.SetSizes(modelDescr.VocabSize, len);
            gradArr.FillZero();
            for (const TNodeTarget &nt : KeepTarget) {
                for (yint q = 0; q < modelDescr.VocabSize; ++q) {
                    gradArr[nt.Node][q] += -predArr[nt.Node][q];
                }
                gradArr[nt.Node][nt.TargetId] += 1;
            }

            // can be omitted, gradient scale does not change anything due to gradient normalization
            ScaleMatrix(&gradArr, CalcFinalLayerMult() * LOG2);

            TArray2D<float> normStateGrad;
            InitDeltaMatrix(&normStateGrad, FinalNormState);
            MulBackwardWithAccum(&normStateGrad, GetData(finalLayer), gradArr);

            // modify final layer
            TArray2D<float> deltaFinalLayer;
            SumRankOne(FinalNormState, &deltaFinalLayer, gradArr);
            finalLayer->ApplyDelta(deltaFinalLayer);

            NormalizeStateBackward(AllStates.back().State, dim / STATE_NORM_TILE, normStateGrad, &grad.State);
        }

        // modify layers
        for (yint d = YSize(LayerArr) - 1; d >= 0; --d) {
            AddLookupProductBackprop(modelDescr, RopeBuf, LayerArr[d], AttArr, AllStates[d * 2], AllStates[d * 2 + 1], &grad);
        }

        // modify embedding
        {
            TArray2D<float> deltaLabel;
            deltaLabel.SetSizes(dim, modelDescr.LabelCount);
            deltaLabel.FillZero();
            for (yint t = 0; t < len; ++t) {
                for (yint k = LabelPtr[t], kFinish = LabelPtr[t + 1]; k < kFinish; ++k) {
                    yint label = LabelArr[k];
                    float w = (label & LABEL_NEGATIVE) ? -1 : 1;
                    label &= LABEL_MASK;
                    for (yint x = 0; x < dim; ++x) {
                        deltaLabel[label][x] += grad.State[t][x] * w;
                    }
                }
            }
            labelEmbed->ApplyDelta(deltaLabel);
        }
    }
};

TIntrusivePtr<IComputeContext> CreateContext(TIntrusivePtr<IModel> pModel, yint nodeCount)
{
    return new TComputeContext(pModel, nodeCount);
}
}
