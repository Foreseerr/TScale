#pragma once
#include "model_dim.h"
#include "model_matrix.h"
#include <gpt/rng/xrng.h>


enum {
    MP_ATT_QK = 0, // attention current token, Q in papers
    MP_ATT_QV = 1, // attention remote, K in papers
    MP_ATT_K = 2, // current token low rank, absent in papers
    MP_ATT_V = 3, // remote low rank, V in papers
    MP_ATT_COMBINER = 4, // // block diagonal combiner, (ttDim * COMBINER_REP) x (dim)
    MP_RELU_EXPAND = 5,
    MP_RELU_CONTRACT = 6,
    MP_ATT_COUNT = 7,
};

enum {
    MP_MODEL_EMBED = 0,
    MP_MODEL_FINAL = 1,
    MP_MODEL_COUNT = 2,
};

struct TModelParams
{
    struct TAttentionMatrices
    {
        TVector<TModelMatrix> MatrArr;
        SAVELOAD(MatrArr);
    };
    TModelDescr ModelDescr;
    TVector<TAttentionMatrices> LayerArr;
    TVector<TModelMatrix> MatrArr;
    TVector<float> Bias;
    SAVELOAD(ModelDescr, LayerArr, MatrArr, Bias);

    TModelDescr GetModelDescr() const
    {
        Y_ASSERT(ModelDescr.Dims.Dim == MatrArr[MP_MODEL_EMBED].GetXSize());
        Y_ASSERT(ModelDescr.LabelCount == MatrArr[MP_MODEL_EMBED].GetYSize());
        Y_ASSERT(ModelDescr.VocabSize == MatrArr[MP_MODEL_FINAL].GetYSize());
        Y_ASSERT(YSize(ModelDescr.Layers) == YSize(LayerArr));
        return ModelDescr;
    }
    template <class T>
    void ForEachModelMatrix(T func)
    {
        for (TModelMatrix &mm : MatrArr) {
            func(mm);
        }
        for (TAttentionMatrices &att : LayerArr) {
            for (TModelMatrix &mm : att.MatrArr) {
                func(mm);
            }
        }
    }
    void ResetGrad(EModelMatrixReset rr)
    {
        ForEachModelMatrix([&](TModelMatrix &mm) { mm.ResetGrad(rr); });
    }
    void ScaleGrad(float x)
    {
        ForEachModelMatrix([&](TModelMatrix &mm) { mm.ScaleGrad(x); });
    }
    bool IsEmpty() const { return Bias.empty(); }
};


struct TModelParamsHolder : public TThrRefBase
{
    TModelParams Params;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
void InitModel(TModelParams *pParams, TXRng &rng, const TModelDescr &modelDescr, ECombinerInit combinerInit, const TVector<float> &biasArr);
void InitModelZero(TModelParams *pParams, const TModelDescr &modelDescr, const TVector<float> &biasArr);

yint CountModelSize(const TModelParams &params);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TAllModelMatrices
{
    TVector<TArray2D<float>> MatrArr;
    SAVELOAD(MatrArr);

public:
    TAllModelMatrices() { MatrArr.reserve(10000); }
    void Clear()
    {
        MatrArr.resize(0);
    }
    bool IsEmpty() const
    {
        return MatrArr.empty();
    }
    void AddMatrix(yint xSize, yint ySize)
    {
        TArray2D<float> &mm = *MatrArr.insert(MatrArr.end());
        mm.SetSizes(xSize, ySize);
        mm.FillZero();
    }
    void FillZero()
    {
        for (auto &mm : MatrArr) {
            mm.FillZero();
        }
    }
    void AddScaled(const TAllModelMatrices &arg, float scale)
    {
        Y_VERIFY(YSize(MatrArr) == YSize(arg.MatrArr));
        for (yint k = 0; k < YSize(MatrArr); ++k) {
            AddScaledMatrixAligned(&MatrArr[k], arg.MatrArr[k], scale);
        }
    }
    void Scale(float scale)
    {
        for (yint k = 0; k < YSize(MatrArr); ++k) {
            ScaleMatrixAligned(&MatrArr[k], scale);
        }
    }
};


void InitZero(TAllModelMatrices *p, const TModelDescr &descr);
void GetMatrices(TAllModelMatrices *p, const TModelParams &params);
void SetMatrices(TModelParams *p, const TAllModelMatrices &params);
void AddMatrices(TModelParams *p, const TAllModelMatrices &params, float scale);
void GetGradient(TAllModelMatrices *p, const TModelParams &params);
void SetGradient(TModelParams *p, const TAllModelMatrices &params);


///////////////////////////////////////////////////////////////////////////////////////////////////
void PackMatrices(TBufferedStream &f, const TModelParams &params);
void PackMatrices(TBufferedStream &f, const TAllModelMatrices &params);
void AddPackedMatrices(TAllModelMatrices *p, TBufferedStream &f, float scale);
void AddPackedMatrices(TModelParams *p, TBufferedStream &f, float scale);


///////////////////////////////////////////////////////////////////////////////////////////////////
void GetRowDisp(TModelRowDisp *p, const TModelParams &params);
void SetRowDisp(TModelParams *p, const TModelRowDisp &rd);
