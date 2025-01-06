#include "stdafx.h"
#include "model_params.h"
#include "sse_utils.h"
#include <lib/random/rand_utils.h>


struct TFastNormalRNG : public TThrRefBase
{
    enum {
        BUF_SIZE = 1 << 16,
    };
    float Buf[BUF_SIZE];
    TXRng Rng;

    TFastNormalRNG(yint seed) : Rng(seed)
    {
        for (yint k = 0; k < BUF_SIZE; k += 2) {
            //(rng.GenRandReal3() * 2 - 1) * bound;
            float val = GenNormal(Rng);
            Buf[k] = val;
            Buf[k + 1] = -val;
        }
    }
    float Gen()
    {
        return Buf[((ui32)Rng.GenRand()) & (BUF_SIZE - 1)];
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
inline void InitMatrixNormal(TModelMatrix *pRes, TFastNormalRNG &rng, float sko)
{
    yint xSize = pRes->GetXSize();
    yint ySize = pRes->GetYSize();
    TArray2D<float> mm;
    mm.SetSizes(xSize, ySize);
    float bound = sko * 0.5f;
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            mm[y][x] = rng.Gen() * bound;
        }
    }
    pRes->SetMatrix(mm);
}

inline void InitEmbedMatrix(TModelMatrix *pRes, TFastNormalRNG &rng, bool noiseLabels)
{
    yint xSize = pRes->GetXSize();
    yint ySize = pRes->GetYSize();
    TArray2D<float> embed;
    embed.SetSizes(xSize, ySize);
    embed.FillZero();
    for (yint y = 0; y < ySize; ++y) {
        if (noiseLabels && y < NOISE_LABELS_COUNT) { // zero init noise tokens
            continue;
        }
        for (yint x = 0; x < xSize; ++x) {
            embed[y][x] = rng.Gen();
        }
    }
    pRes->SetMatrix(embed);
}

inline void InitIdentity(TModelMatrix *pRes)
{
    yint xSize = pRes->GetXSize();
    yint ySize = pRes->GetYSize();
    TArray2D<float> mm;
    mm.SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            mm[y][x] = (x == y) ? 1 : 0;
        }
    }
    pRes->SetMatrix(mm);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Model Params Initialize
static void AllocateAttention(TModelParams::TAttentionMatrices *p, const TModelDims &dims)
{
    yint dim = dims.Dim;
    yint qSum = dims.GetQSum();
    yint ttSum = dims.GetTTSum();
    yint reluDim = dims.GetReluDim();
    yint combinerDim = dims.GetCombinerDim();
    p->MatrArr.resize(MP_ATT_COUNT);
    p->MatrArr[MP_ATT_QK].SetSizes(dim, qSum, MM_MATRIX_DISP);
    p->MatrArr[MP_ATT_QV].SetSizes(dim, qSum, MM_MATRIX_DISP);
    p->MatrArr[MP_ATT_K].SetSizes(dim, ttSum, MM_MATRIX_DISP);
    p->MatrArr[MP_ATT_V].SetSizes(dim, ttSum, MM_MATRIX_DISP);
    p->MatrArr[MP_ATT_COMBINER].SetSizes(combinerDim, dim, MM_MATRIX_DISP);
    p->MatrArr[MP_RELU_EXPAND].SetSizes(dim, reluDim, MM_MATRIX_DISP);
    p->MatrArr[MP_RELU_CONTRACT].SetSizes(reluDim, dim, MM_MATRIX_DISP);
}

static void AllocateModel(TModelParams *p, const TModelDescr &descr)
{
    p->ModelDescr = descr;
    yint dim = descr.Dims.Dim;
    p->MatrArr.resize(MP_MODEL_COUNT);
    p->MatrArr[MP_MODEL_EMBED].SetSizes(dim, descr.LabelCount, MM_ROW_DISP);
    p->MatrArr[MP_MODEL_FINAL].SetSizes(dim, descr.VocabSize, MM_ROW_DISP);
    yint depth = YSize(descr.Layers);
    p->LayerArr.resize(depth);
    for (yint d = 0; d < depth; ++d) {
        AllocateAttention(&p->LayerArr[d], descr.Dims);
    }
    ClearPodArray(&p->Bias, descr.VocabSize);
}


static void InitAttention(TModelParams::TAttentionMatrices *p, TFastNormalRNG &rng, ECombinerInit combinerInit)
{
    InitMatrixNormal(&p->MatrArr[MP_ATT_QK], rng, 1);
    InitMatrixNormal(&p->MatrArr[MP_ATT_QV], rng, 1);
    //InitIdentity(&p->MatrArr[MP_ATT_QK]);
    //InitIdentity(&p->MatrArr[MP_ATT_QV]);
    InitMatrixNormal(&p->MatrArr[MP_ATT_K], rng, 1);
    InitMatrixNormal(&p->MatrArr[MP_ATT_V], rng, 1);
    if (combinerInit == COMBINER_INIT_RANDOM) {
        InitMatrixNormal(&p->MatrArr[MP_ATT_COMBINER], rng, 1); // required for binary classification to converge to interesting solutions?
        InitMatrixNormal(&p->MatrArr[MP_RELU_CONTRACT], rng, 1);
    } else if (combinerInit == COMBINER_INIT_ZERO) {
        InitMatrixNormal(&p->MatrArr[MP_ATT_COMBINER], rng, 0);
        InitMatrixNormal(&p->MatrArr[MP_RELU_CONTRACT], rng, 0);
    } else {
        Y_ASSERT(0 && "unsupported combiner init");
    }
    InitMatrixNormal(&p->MatrArr[MP_RELU_EXPAND], rng, 1);
}

void InitModel(TModelParams *pParams, TXRng &rngArg, const TModelDescr &modelDescr, ECombinerInit combinerInit, const TVector<float> &biasArr)
{
    TIntrusivePtr<TFastNormalRNG> rngHolder = new TFastNormalRNG(rngArg.GenRand());
    TFastNormalRNG &rng = *rngHolder.Get();
    bool noiseLabels = !modelDescr.HasFlag(MPF_DISABLE_NOISE_LABELS);
    AllocateModel(pParams, modelDescr);
    InitEmbedMatrix(&pParams->MatrArr[MP_MODEL_EMBED], rng, noiseLabels);
    InitMatrixNormal(&pParams->MatrArr[MP_MODEL_FINAL], rng, 1); // init for fixed final layer

    yint depth = YSize(pParams->LayerArr);
    for (yint d = 0; d < depth; ++d) {
        InitAttention(&pParams->LayerArr[d], rng, combinerInit);
    }
    Y_VERIFY(YSize(biasArr) == modelDescr.VocabSize);
    pParams->Bias = biasArr;
}


void InitModelZero(TModelParams *pParams, const TModelDescr &modelDescr, const TVector<float> &biasArr)
{
    AllocateModel(pParams, modelDescr);
    pParams->Bias = biasArr;
}


//////////////////////////////////////////////////////////////////////////////////////////////////
// perform op on each model matrix
template <class TMP, class T>
void ForEachModelMatrix(TMP &modelParams, T func)
{
    for (auto &mm : modelParams.MatrArr) {
        func(mm);
    }
    for (auto &att : modelParams.LayerArr) {
        for (auto &mm : att.MatrArr) {
            func(mm);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// model ops
//
yint CountModelSize(const TModelParams &params)
{
    yint res = 0;
    ForEachModelMatrix(params, [&](const TModelMatrix &mm) {
        res += mm.GetXSize() * mm.GetYSize();
        });
    res += YSize(params.Bias);
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void InitZero(TAllModelMatrices *p, const TModelDescr &descr)
{
    p->Clear();
    const TModelDims &dims = descr.Dims;
    int dim = dims.Dim;
    p->AddMatrix(dim, descr.LabelCount);
    p->AddMatrix(dim, descr.VocabSize);
    yint depth = YSize(descr.Layers);
    for (yint d = 0; d < depth; ++d) {
        yint qSum = dims.GetQSum();
        yint ttSum = dims.GetTTSum();
        yint reluDim = dims.GetReluDim();
        yint combinerDim = dims.GetCombinerDim();
        p->AddMatrix(dim, qSum);
        p->AddMatrix(dim, qSum);
        p->AddMatrix(dim, ttSum);
        p->AddMatrix(dim, ttSum);
        p->AddMatrix(combinerDim, dim);
        p->AddMatrix(dim, reluDim);
        p->AddMatrix(reluDim, dim);
    }
}

void GetMatrices(TAllModelMatrices *p, const TModelParams &params)
{
    p->Clear();
    ForEachModelMatrix(params, [&](const TModelMatrix &mm) {
        p->MatrArr.push_back(mm.GetMatrix());
        });
}

void SetMatrices(TModelParams *p, const TAllModelMatrices &params)
{
    yint k = 0;
    ForEachModelMatrix(*p, [&](TModelMatrix &mm) {
        mm.GetMatrix() = params.MatrArr[k++];
        });
}

void AddMatrices(TModelParams *p, const TAllModelMatrices &params, float scale)
{
    yint k = 0;
    ForEachModelMatrix(*p, [&](TModelMatrix &mm) {
        AddScaledMatrixAligned(&mm.GetMatrix(), params.MatrArr[k++], scale);
        });
}

void GetGradient(TAllModelMatrices *p, const TModelParams &params)
{
    p->Clear();
    ForEachModelMatrix(params, [&](const TModelMatrix &mm) {
        p->MatrArr.push_back(mm.GetGrad1());
        });
}

void SetGradient(TModelParams *p, const TAllModelMatrices &params)
{
    yint k = 0;
    ForEachModelMatrix(*p, [&](TModelMatrix &mm) {
        mm.GetGrad1() = params.MatrArr[k++];
        });
}




///////////////////////////////////////////////////////////////////////////////////////////////////
void PackMatrices(TBufferedStream &f, const TModelParams &params)
{
    ForEachModelMatrix(params, [&](const TModelMatrix &mm) {
        PackMatrix(f, mm.GetMatrix());
        });
}

void PackMatrices(TBufferedStream &f, const TAllModelMatrices &params)
{
    for (auto &mm : params.MatrArr) {
        PackMatrix(f, mm);
    }
}

void AddPackedMatrices(TAllModelMatrices *p, TBufferedStream &f, float scale)
{
    for (auto &mm : p->MatrArr) {
        AddPackedMatrix(&mm, f, scale);
    }
}

void AddPackedMatrices(TModelParams *p, TBufferedStream &f, float scale)
{
    ForEachModelMatrix(*p, [&](TModelMatrix &mm) {
        AddPackedMatrix(&mm.GetMatrix(), f, scale);
        });
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void GetRowDisp(TModelRowDisp *p, const TModelParams &params)
{
    p->Clear();
    ForEachModelMatrix(params, [&](const TModelMatrix &mm) {
        p->AddMatrixRowDisp(mm.GetRowDisp(), mm.GetSumWeight());
        });
}

void SetRowDisp(TModelParams *p, const TModelRowDisp &rd)
{
    yint ptr = 0;
    ForEachModelMatrix(*p, [&](TModelMatrix &mm) {
        yint sz = YSize(mm.GetRowDisp());
        TVector<float> rowDisp;
        for (yint k = 0; k < sz; ++k) {
            rowDisp.push_back(rd.RowDisp[ptr++]);
        }
        mm.SetRowDisp(rd.SumWeight, rowDisp);
        });
}
