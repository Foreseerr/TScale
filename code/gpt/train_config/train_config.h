#pragma once
#include "train_step.h"
#include <gpt/data/data.h>
#include <gpt/model_params/model_params.h>
#include <lib/config/cfg_file.h>
#include <math.h>


struct TDescentConfig
{
    yint TrainBatchSize = 64;
    yint TrainFragLen = 64;
    float TokenDrop = 0.9f;
    float ChannelDrop = 0.9f;
    TTrainingStep Step;
    float LRTail = 0;
    yint SlowStart = 1000;
    // SAVELOAD - serialized as POD

public:
    TDescentConfig() {}
    TDescentConfig(const TString &trainConfig, const TString &dropConfig);
    TString GetTrainConfig();
    TString GetDropConfig();

public:
    TTrainingStep GetStep(yint iter, yint maxIters) const
    {
        TTrainingStep res = Step;
        if (LRTail != 0) {
            float longFrac = (iter < maxIters) ? iter / (maxIters + 0.f) : 1;
            float scale = Min<float>(1, (1 - longFrac) * LRTail);
            res.ScaleRate(scale);
        }
        // w0 - gradient weight, w1 - gradient MA (with decay beta1) weight
        res.Weight0 = 0.2f;
        res.Weight1 = 0.8f;
        float decay = 1 - 0.5f / sqrt(1. + iter);
        //float decay = 1 - 0.1f / (iter + 1);
        res.Beta1 = decay;
        res.DispDecay = decay;
        // slow start
        if (SlowStart > 0 && iter < SlowStart) {
            res.ScaleRate((iter + 1.) / SlowStart); // linear
        }
        return res;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TTrainModelConfigParser
{
    TIntrusivePtr<TModelParamsHolder> StartParams;
    // params
    TString ModelDescrString = "e256d65";
    TString TrainConfig = "b64f64";
    TString DropConfig = "drop1ch1";

public:
    TDescentConfig MakeDescentConfig() const
    {
        return TDescentConfig(TrainConfig, DropConfig);
    }
    bool ParseScriptOp(const TConfigFile::TOp &op, TIntrusivePtr<IDataSource> data);
};
