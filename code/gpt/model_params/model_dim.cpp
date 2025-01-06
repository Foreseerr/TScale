#include "stdafx.h"
#include "model_dim.h"
#include <lib/config/config.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelDescrString
{
    TModelDims Dims;
    yint Depth = 60;
    yint WideLimitWindow = 64;

    TModelDescrString() {}
    TModelDescrString(const TString &modelDescrStr);
};

TModelDescrString::TModelDescrString(const TString &modelDescrStr)
{
    TStringParams sp(modelDescrStr);
    for (TStringParams::TParam &param : sp.Params) {
        if (param.Name == "e") {
            Dims.Dim = param.Value;
        } else if (param.Name == "h") {
            Dims.HeadCount = param.Value;
        } else if (param.Name == "d") {
            Depth = param.Value;
        } else if (param.Name == "w") {
            WideLimitWindow = param.Value;
        } else if (param.Name == "relu") {
            Dims.ReluMult = param.Value;
        }
    }
}


TString GetModelDescrString(const TModelDescr &modelDescr)
{
    TModelDims defDims;
    TString res = Sprintf("e%g", modelDescr.Dims.Dim * 1.);
    if (modelDescr.Dims.HeadCount != defDims.HeadCount) {
        res += Sprintf("h%g", modelDescr.Dims.HeadCount * 1.);
    }
    yint depth = YSize(modelDescr.Layers);
    res += Sprintf("d%g", depth * 1.);
    if (modelDescr.Dims.ReluMult != defDims.ReluMult) {
        res += Sprintf("relu%g", modelDescr.Dims.ReluMult * 1.);
    }
    yint wideLimit = modelDescr.GetWideLimitWindow();
    res += Sprintf("w%g", wideLimit * 1.);
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static void InitAlibi(TModelDescr *p, EAlibi alibi, yint depth, yint wideLimitWindow)
{
    p->Layers.resize(0);
    p->Layers.resize(depth);
    // useful in attention profiling
    if (depth == 1) {
        p->AttentionWidthArr.push_back(wideLimitWindow);
        p->Layers[0] = TModelDescr::TAttentionPosParams(0.1f, 0);
        return;
    }
    // configure position encoding
    if (alibi == ALIBI_NONE) {
        ClearPodArray(&p->AttentionWidthArr, 1);

    } else if (alibi == ALIBI_ABS_POS_ENCODING) {
        p->AttentionWidthArr.resize(1);
        p->AttentionWidthArr[0] = wideLimitWindow;

    } else if (alibi == ALIBI_V3) {
        p->AttentionWidthArr.clear();
        // 
        p->AttentionWidthArr.push_back(0);
        p->AttentionWidthArr.push_back(1);
        p->AttentionWidthArr.push_back(64);
        p->AttentionWidthArr.push_back(wideLimitWindow);
        p->Layers.resize(0);
        if (depth > 0) {
            yint groupCount = DivCeil(depth, 2);
            yint w1count = Max<yint>(1, groupCount / 6);
            for (yint k = 0; k < w1count; ++k) {
                p->AddLayer(1);
                p->AddLayer(1);
            }
            for (yint k = w1count; k < groupCount; ++k) {
                p->AddLayer(2, 0.5);
                p->AddLayer(3);
            }
        }

    } else {
        Y_VERIFY("unknown alibi version");
    }
}


void InitModelDescr(TModelDescr *pRes, const TString &modelDescrStr, EAlibi alibi, yint vocabSize, yint labelCount, ui64 flags)
{
    TModelDescrString dims(modelDescrStr);

    TModelDescr &modelDescr = *pRes;
    modelDescr = TModelDescr();
    modelDescr.Dims = dims.Dims;
    modelDescr.LabelCount = labelCount;
    modelDescr.VocabSize = vocabSize;
    modelDescr.Flags = flags;
    if (modelDescr.HasFlag(MPF_ABS_POSITIONS)) {
        alibi = ALIBI_ABS_POS_ENCODING;
    }
    InitAlibi(&modelDescr, alibi, dims.Depth, dims.WideLimitWindow);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// drop table utils
yint CalcDropTableSize(const TModelDescr &modelDescr)
{
    return modelDescr.Dims.Dim / 32;
}

void MakeDropTable(TXRng &rng, const TModelDescr &modelDescr, TVector<ui32> *pDropTable, float channelDrop)
{
    yint sz = CalcDropTableSize(modelDescr);
    pDropTable->resize(sz);
    for (yint i = 0; i < sz; ++i) {
        ui32 mask = 0;
        for (int k = 0; k < 32; ++k) {
            if (rng.GenRandReal3() <= channelDrop) {
                mask |= 1 << k;
            }
        }
        (*pDropTable)[i] = mask;
    }
}
