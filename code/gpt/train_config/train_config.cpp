#include "stdafx.h"
#include "train_config.h"
#include <gpt/att/sliding_window.h>
#include <gpt/model_params/fed_model.h>
#include <lib/config/config.h>


TDescentConfig::TDescentConfig(const TString &trainConfig, const TString &dropConfig)
{
    double slide = 1;
    TStringParams tc(trainConfig);
    for (auto &param : tc.Params) {
        if (param.Name == "b") {
            TrainBatchSize = param.Value;
        } else if (param.Name == "f") {
            TrainFragLen = param.Value;
        } else {
            Y_VERIFY(0 && "unknown param");
        }
    }

    TStringParams dc(dropConfig);
    for (auto &param : dc.Params) {
        if (param.Name == "drop") {
            TokenDrop = param.Value;
        } else if (param.Name == "ch") {
            ChannelDrop = param.Value;
        } else if (param.Name == "lr") {
            Step.Rate = param.Value;
        } else if (param.Name == "slow") {
            SlowStart = param.Value;
        } else if (param.Name == "reg") {
            // ballpark estimate of optimal reg:
            //   reg = train_size / batch_size * 0.1
            // learning rate does not change optimal reg
            // dropout does not seem to change optimal reg
            // channel dropout does seem to change optimal reg, however ch1 wins
            if (param.Value > 0) {
                Step.L2Reg = 1 / param.Value;
            } else {
                Step.L2Reg = 0;
            }
        } else if (param.Name == "tail") {
            LRTail = param.Value;
        } else {
            Y_VERIFY(0 && "unknown param");
        }
    }
}

TString TDescentConfig::GetTrainConfig()
{
    TDescentConfig rr;
    TString res;
    res += Sprintf("b%gf%g", TrainBatchSize * 1., TrainFragLen * 1.);
    return res;
}

TString TDescentConfig::GetDropConfig()
{
    TDescentConfig rr;
    TString res = Sprintf("drop%gch%g", TokenDrop, ChannelDrop);
    if (Step.Rate != rr.Step.Rate) {
        res += Sprintf("lr%g", Step.Rate);
    }
    if (Step.L2Reg != rr.Step.L2Reg) {
        res += Sprintf("reg%g", 1 / Step.L2Reg);
    }
    if (SlowStart != rr.SlowStart) {
        res += Sprintf("slow%g", SlowStart * 1.);
    }
    if (LRTail != rr.LRTail) {
        res += Sprintf("tail%g", LRTail);
    }
    return res;
}


static TIntrusivePtr<TModelParamsHolder> CreateModel(const TConfigFile::TOp &op, const TString &modelDescrString, const IDataSource::TDataStats &stats)
{
    TXRng rng(1313);

    Y_VERIFY(stats.VocabSize > 0 && "unknown vocab size");
    ui64 modelFlags = 0;
    if (stats.UsePPM) {
        modelFlags |= MPF_PPM;
    }
    if (stats.UseLMatch) {
        modelFlags |= MPF_LMATCH;
    }
    for (const TString &flag : op.Args) {
        if (flag == "MPF_HASHED_EMBED") {
            modelFlags |= MPF_HASHED_EMBED;
        } else if (flag == "MPF_TAIL_LOSS") {
            modelFlags |= MPF_TAIL_LOSS;
        } else if (flag == "MPF_SIM_QUANT_2BIT") {
            modelFlags |= MPF_SIM_QUANT_2BIT;
        } else if (flag == "MPF_SIM_QUANT_4BIT") {
            modelFlags |= MPF_SIM_QUANT_4BIT;
        } else if (flag == "MPF_GROK_BINARY_OP") {
            modelFlags |= MPF_GROK_BINARY_OP;
        } else if (flag == "MPF_MLM_BERT") {
            modelFlags |= MPF_MLM_BERT;
        } else if (flag == "MPF_ABS_POSITIONS") {
            modelFlags |= MPF_ABS_POSITIONS;
        } else if (flag == "MPF_DISABLE_NOISE_LABELS") {
            modelFlags |= MPF_DISABLE_NOISE_LABELS;
        } else {
            DebugPrintf("unknown model flag %s\n", flag.c_str());
        }
    }
    // initialize model params
    TIntrusivePtr<TModelParamsHolder> res = new TModelParamsHolder();
    TModelDescr modelDescr;
    InitModelDescr(&modelDescr, modelDescrString, ALIBI_V3, stats.VocabSize, modelFlags);
    InitModel(&res->Params, rng, modelDescr, COMBINER_INIT_ZERO, stats.Bias);
    if (stats.DocStartToken>= 0) {
        res->Params.ModelDescr.SetDocStartToken(stats.DocStartToken);
    } else {
        Y_ASSERT(!res->Params.ModelDescr.HasFlag(MPF_USE_DOC_START_TOKEN));
    }
    return res;
}


bool TTrainModelConfigParser::ParseScriptOp(const TConfigFile::TOp &op, TIntrusivePtr<IDataSource> data)
{
    if (op.Op == CFG_OP_ASSIGNMENT) {
        if (op.Dst == "TRAIN_CONFIG") {
            TrainConfig = op.Args[0];
        } else if (op.Dst == "DROP_CONFIG") {
            DropConfig = op.Args[0];
        } else if (op.Dst == "MODEL_DIMS") {
            Y_VERIFY(StartParams == nullptr && "model dimenstion are useless, model already created");
            ModelDescrString = op.Args[0];
        } else {
            return false;
        }

    } else if (op.Op == CFG_OP_CALL) {
    // model ops
        if (op.Dst == "create_model") {
            StartParams = CreateModel(op, ModelDescrString, data->GetStats());

        } else if (op.Dst == "load_model") {
            Y_VERIFY(YSize(op.Args) == 1);
            DebugPrintf("Load model %s\n", op.Args[0].c_str());
            StartParams = new TModelParamsHolder();
            Serialize(IO_READ, op.Args[0], StartParams->Params);
            Y_VERIFY(!StartParams->Params.IsEmpty());

        } else if (op.Dst == "load_fed_model") {
            // iteration number from fed model is ignored, leads to divergence at start if training is continued from this model (due to rowdisp estimates most probably)
            Y_VERIFY(YSize(op.Args) == 1);
            DebugPrintf("Load fed center model %s\n", op.Args[0].c_str());
            StartParams = new TModelParamsHolder();
            TFedCenterModel fm;
            Serialize(IO_READ, op.Args[0], fm);
            ConvertToModelParams(&StartParams->Params, fm);
            Y_VERIFY(!StartParams->Params.IsEmpty());

        } else if (op.Dst == "reset_model_grad") {
            if (StartParams.Get()) {
                StartParams->Params.ResetGrad(MM_RESET_GRAD_AND_ROW_DISP);
            } else {
                DebugPrintf("can not reset model grad, no model\n");
            }

        } else {
            return false;
        }
    }
    return true;
}
