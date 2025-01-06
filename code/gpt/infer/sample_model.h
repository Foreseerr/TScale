#pragma once
#include <gpt/data/bpe.h>
#include <gpt/data/ppm_window.h>
#include <gpt/data/ppm_lmatch.h>
#include <gpt/data/fragment_gen.h>
#include <gpt/data/data.h>
#include <gpt/model_params/model_params.h>
#include <gpt/compute/gpt_cuda.cuh>
#include <gpt/att/sliding_window.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TSamplingModel
{
    TIntrusivePtr<IModel> Model;
    TIntrusivePtr<IComputeContext> Ctx;
    TTokenizer Tokenizer;
    TIntrusivePtr<TLMatchSearch> LMatchSearch;
    yint DocStartToken = -1;
    yint MaxLen = 0;
    bool UsePPM = false;
    bool UseLMatch = false;

    TSamplingModel(TIntrusivePtr<TModelParamsHolder> mph, const TTokenizer &tokenizer, const TString &lmIndexDir)
        : DocStartToken(tokenizer.HasDocStartToken() ? tokenizer.GetDocStartToken() : -1)
    {
        if (!lmIndexDir.empty()) {
            LMatchSearch = new TLMatchSearch(lmIndexDir, DocStartToken);
        }
        const TModelParams &params = mph->Params;
        Tokenizer = tokenizer;
        Model = CreateModel(1, params);
        MaxLen = params.ModelDescr.FragLen;
        Ctx = NCUDA_GPT::CreateContext(Model, MaxLen);
        UsePPM = params.ModelDescr.HasFlag(MPF_PPM);
        UseLMatch = params.ModelDescr.HasFlag(MPF_LMATCH);
    }
};

TString SampleFromModel(TXRng &rng, TSamplingModel &model, const TString &prefix);
float ComputeLogLoss(TSamplingModel &model, const TVector<char> &text);

//TString GenerateFromModel(TXRng &rng, TTrainContext *pTrainCtx, const TModelParams &params, yint genLen, yint limitWindow, yint fragLen);
//TString BeamSampleFromModel(TXRng &rng, TTrainContext *pTrainCtx, const TModelParams &params, yint genLen, yint limitWindow, yint fragLen);
