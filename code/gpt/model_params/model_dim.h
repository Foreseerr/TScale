#pragma once
#include <gpt/rng/xrng.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
// model matrices discretization step
//constexpr float MODEL_DISCR_SCALE = 1.f / 16;
//constexpr float MODEL_DISCR_SCALE = 1.f / 24;
constexpr float MODEL_DISCR_SCALE = 1.f / 32;

///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr yint COMBINER_REP = 4;

///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr yint NOISE_LABELS_COUNT = 32;

///////////////////////////////////////////////////////////////////////////////////////////////////
// what scaling is optimal and why we need it all is unclear
constexpr float LOG2 = 0.693147f;
constexpr float FINAL_LAYER_SOFTMAX_SCALE = 1;
constexpr float ATT_DOTPRODUCT_SCALE = 0.5f;
constexpr int STATE_NORM_TILE = 128;

constexpr float ATT_W_MULT = -128; // all weights are positive, -128 has higher dynamic range then 127
constexpr float ATT_DOT_MULT = 127;

#define CalcDotScale(dim) (sqrtf(1.0f / (dim)))

#define CalcFinalLayerMult() (FINAL_LAYER_SOFTMAX_SCALE / LOG2)

#define CalcAttentionMult() (ATT_DOTPRODUCT_SCALE / LOG2)

#define GetAttentionDecay(dist, alibiSlope) (-(alibiSlope) * (dist))


///////////////////////////////////////////////////////////////////////////////////////////////////
// model params flags
const ui64 MPF_NOFLAGS = 0;
const ui64 MPF_HASHED_EMBED = 0x1;
const ui64 MPF_PPM = 0x2;
const ui64 MPF_USE_DOC_START_TOKEN = 0x4;
const ui64 MPF_TAIL_LOSS = 0x20;
const ui64 MPF_SIM_QUANT_2BIT = 0x40;
const ui64 MPF_SIM_QUANT_4BIT = 0x80;
const ui64 MPF_GROK_BINARY_OP = 0x100;
const ui64 MPF_MLM_BERT = 0x200;
const ui64 MPF_ABS_POSITIONS = 0x400;
const ui64 MPF_DISABLE_NOISE_LABELS = 0x800;
const ui64 MPF_LMATCH = 0x1000;


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelDims
{
    yint Dim = 256;
    yint QDim = 128;
    yint TTDim = 128;
    yint HeadCount = 1;
    int ReluMult = 1;
    SAVELOAD(Dim, QDim, TTDim, HeadCount, ReluMult);

    yint GetQSum() const { return QDim * HeadCount; }
    yint GetTTSum() const { return TTDim * HeadCount; }
    yint GetCombinerDim() const { return TTDim * HeadCount * COMBINER_REP; }
    yint GetReluDim() const { return HeadCount * 128 * ReluMult; }
};

inline bool operator==(const TModelDims &a, const TModelDims &b)
{
    return a.Dim == b.Dim && a.QDim == b.QDim && a.TTDim == b.TTDim && a.HeadCount == b.HeadCount;
}

inline bool operator!=(const TModelDims &a, const TModelDims &b)
{
    return !(a == b);
}



///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelDescr
{
    struct TAttentionPosParams
    {
        float AlibiSlope = 0;
        yint AttentionWidthId = 0;

        TAttentionPosParams() {}
        TAttentionPosParams(float slope, yint attWidthId) : AlibiSlope(slope), AttentionWidthId(attWidthId) {}
    };


    TModelDims Dims;
    yint LabelCount = 0;
    yint VocabSize = 0;
    TVector<yint> AttentionWidthArr;
    TVector<TAttentionPosParams> Layers;
    ui64 Flags = 0;
    ui64 DocStartToken = 0;
    yint FragLen = 0;
    SAVELOAD(Dims, LabelCount, VocabSize, AttentionWidthArr, Layers, Flags, DocStartToken, FragLen);

    yint GetAttentionWidthCount() const
    {
        return YSize(AttentionWidthArr);
    }
    yint GetAttentionCount() const
    {
        return YSize(Layers);
    }
    yint GetWideLimitWindow() const
    {
        yint res = 1;
        for (yint x : AttentionWidthArr) {
            res = Max<yint>(x, res);
        }
        return res;
    }
    bool HasFlag(ui64 f) const
    {
        return (Flags & f) != 0;
    }
    void SetDocStartToken(ui64 token)
    {
        Flags |= MPF_USE_DOC_START_TOKEN;
        DocStartToken = token;
    }
    void AddLayer(yint attWidthId)
    {
        AddLayer(attWidthId, 0.f);
    }
    void AddLayer(yint attWidthId, float alibiSlope)
    {
        Layers.push_back(TAttentionPosParams(alibiSlope, attWidthId));
    }
};

inline bool operator==(const TModelDescr::TAttentionPosParams &a, const TModelDescr::TAttentionPosParams &b)
{
    return a.AlibiSlope == b.AlibiSlope && a.AttentionWidthId == b.AttentionWidthId;
}

inline bool operator==(const TModelDescr &a, const TModelDescr &b)
{
    return
        a.Dims == b.Dims &&
        a.LabelCount == b.LabelCount && a.VocabSize == b.VocabSize &&
        a.AttentionWidthArr == b.AttentionWidthArr && a.Layers == b.Layers && a.Flags == b.Flags &&
        a.DocStartToken == b.DocStartToken &&
        a.FragLen == b.FragLen;
}


enum EAlibi
{
    ALIBI_NONE,
    ALIBI_ABS_POS_ENCODING,
    ALIBI_V1,
    ALIBI_V2,
    ALIBI_V3,
};

enum ECombinerInit
{
    COMBINER_INIT_RANDOM,
    COMBINER_INIT_ZERO,
};


void InitModelDescr(TModelDescr *pRes, const TString &modelDescrStr, EAlibi alibi, yint vocabSize, yint labelCount, ui64 flags);
TString GetModelDescrString(const TModelDescr &modelDescr);


///////////////////////////////////////////////////////////////////////////////////////////////////
yint CalcDropTableSize(const TModelDescr &modelDescr);
void MakeDropTable(TXRng &rng, const TModelDescr &modelDescr, TVector<ui32> *pDropTable, float channelDrop);
