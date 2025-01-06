#include "stdafx.h"
#include "fed_sim.h"
#include <gpt/model_params/model_params.h>
#include <gpt/att/sliding_window.h>
#include <gpt/data_config/data_config.h>
#include <gpt/train_config/train_config.h>
#include <gpt/train_ctx/train_ctx.h>
#include <gpt/train_ctx/backprop.h>
#include <gpt/compute/gpt_cuda.cuh>


namespace NFedSim
{

#ifdef NDEBUG
const yint EVAL_INTERVAL = 100;
const yint EVAL_BATCH_COUNT = 2;
#else
const yint EVAL_INTERVAL = 10;
const yint EVAL_BATCH_COUNT = 2;
#endif


static TString DATA_SCRIPT =
    " set_vocab_size(50257, 1)"
    " set_doc_start_token(50256)" // eot == 50256
    " load_indexed_docset_folder('D:/text/fineweb')"
    " load_indexed_docset_folder('D:/text/fineweb_test')"
    ;

static TString TRAIN_SCRIPT =
    //" TRAIN_CONFIG = 'b64f128'"
    " TRAIN_CONFIG = 'b64f128'"
    " DROP_CONFIG = 'drop1ch1lr0.01'"
    ;


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
inline void PackMatrices(TMemStream *p, const T &pp)
{
    p->Seek(0);
    TBufferedStream bufIO(IO_WRITE, *p);
    PackMatrices(bufIO, pp);
}

template <class T>
inline void AddPackedMatrices(T *p, TMemStream &ms, float scale)
{
    ms.Seek(0);
    TBufferedStream bufIO(IO_READ, ms);
    AddPackedMatrices(p, bufIO, scale);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TCommonContext
{
    TAllModelMatrices Params;
};

struct TAgentContext : public TThrRefBase
{
    TMemStream Basepoint;
    TModelParams MyParams;
    TModelParams MyParamsTrained;
    TMemStream SentDelta;
    TMemStream SentGrad;
    float GlobalWeight = 0;

    TAgentContext(const TModelParams &pp)
    {
        PackMatrices(&Basepoint, pp);
        MyParams = pp;
        MyParamsTrained = pp;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TFedTrain
{
    TCommonContext Common;
    TVector<TIntrusivePtr<TAgentContext>> AgentArr;

    void AddAgent(const TModelParams &pp)
    {
        AgentArr.push_back(new TAgentContext(pp));
    }
};


void Train(TIntrusivePtr<IDataSource> data, const TTrainModelConfigParser &trainCfg, const TModelDescr &modelDescr, const TVector<float> &biasArr, yint startIter, TFedTrain *p)
{
    TFedTrain &fed = *p;
    yint agentCount = YSize(fed.AgentArr);

    yint deviceCount = 1;
    int limitNodeCount = 24 * 1024;
    const bool SAVE_MODEL = false;
    const yint MAX_ITERS = 10000000;

    TDescentConfig dc = trainCfg.MakeDescentConfig();
    TTrainContext trainCtx(data, dc, deviceCount, limitNodeCount, SAVE_MODEL, MAX_ITERS, EVAL_INTERVAL);
    const TDeviceBatchConfig &dbc = trainCtx.GetDeviceBatchConfig();

    // create compute ctx
    TModelParams centerParams;
    InitModelZero(&centerParams, modelDescr, biasArr);
    AddPackedMatrices(&centerParams, fed.AgentArr[0]->Basepoint, 1);
    TIntrusivePtr<IModel> pModel = CreateModel(deviceCount, centerParams);
    TIntrusivePtr<IComputeContext> pCtx = NCUDA_GPT::CreateContext(pModel, dbc.GetDeviceMaxNodeCount());

    // create batches for train & test score compute, can use different sizes
    trainCtx.MakeScoreBatches(EVAL_BATCH_COUNT, dbc.DeviceBatchSize, dbc.FragLen);

    constexpr bool UPDATE_LAG = true;
    //constexpr bool UPDATE_LAG = false;

    constexpr float FED_WEIGHT_SCALE = UPDATE_LAG ? 0.5f : 1;

    int globalIter = startIter;

    TAllModelMatrices sumDelta;
    InitZero(&sumDelta, modelDescr);

    for (yint metaIter = 0;; ++metaIter) {
        // compute train/test error
        {
            pCtx->SetParams(centerParams);
            float trainErr = CalcModelErr(trainCtx.GetScoreTrainBatches(), pCtx.Get()) * trainCtx.GetCompression();
            float testErr = CalcModelErr(trainCtx.GetScoreTestBatches(), pCtx.Get()) * trainCtx.GetCompression();
            DebugPrintf("train err %g, test err %g\n", trainErr, testErr); fflush(0);
            DebugPrintf("\n");
        }


        // each agent trains model on own data sample
        for (yint agentId = 0; agentId < agentCount; ++agentId) {
            TAgentContext &agent = *fed.AgentArr[agentId];

            pCtx->SetParams(agent.MyParams);
            for (yint iter = 0; iter < EVAL_INTERVAL; ++iter) {
                ui64 rngSeed = ((1313 + agentId) * 0xc949d7c7509e6557ULL + metaIter) * 0x9ae16a3b2f90404fULL + iter;
                TVector<TFragment> fragArr;
                trainCtx.SampleTrainBatches(rngSeed, &fragArr);

                TXRng iterRng(rngSeed);
                BackpropBatch(iterRng, dc, dbc, trainCtx.GetStep(globalIter + iter), fragArr, pCtx);
            }
            pCtx->GetParams(&agent.MyParamsTrained);
            DebugPrintf("meta iter %g, agent %g, err %g\n", metaIter * 1., agentId * 1., pCtx->GetAvrgTrainErr());
        }
        globalIter += EVAL_INTERVAL;


        // each agent compute delta (simulate lag if needed) and sum deltas
        sumDelta.FillZero();
        for (yint agentId = 0; agentId < agentCount; ++agentId) {
            TAgentContext &agent = *fed.AgentArr[agentId];
            if (!UPDATE_LAG) {
                agent.MyParams = agent.MyParamsTrained;
            }
            // compute delta = myParams - basepoint
            TAllModelMatrices delta;
            GetMatrices(&delta, agent.MyParams);
            AddPackedMatrices(&delta, agent.Basepoint, -1);

            // "send" delta and sum deltas from all agents
            PackMatrices(&agent.SentDelta, delta);
            float globalWeight = 1. / agentCount; // weight of this agent
            // 0.5 is required for convergence for 1 step sumDelta lag
            agent.GlobalWeight = globalWeight;
            AddPackedMatrices(&sumDelta, agent.SentDelta, agent.GlobalWeight * FED_WEIGHT_SCALE);

            // update params (simulates concurrent sending delta/receiving basepoint and training model)
            if (UPDATE_LAG) {
                agent.MyParams = agent.MyParamsTrained;
            }
        }


        // fed center add delta sum
        fed.Common.Params.AddScaled(sumDelta, 1);
        AddMatrices(&centerParams, sumDelta, 1);


        // agents update basepoint
        for (yint agentId = 0; agentId < agentCount; ++agentId) {
            TAgentContext &agent = *fed.AgentArr[agentId];
            // replace basepoint
            AddPackedMatrices(&agent.MyParams, agent.Basepoint, -1);
            PackMatrices(&agent.Basepoint, fed.Common.Params);
            AddPackedMatrices(&agent.MyParams, agent.Basepoint, 1);
            AddPackedMatrices(&agent.MyParams, agent.SentDelta, -agent.GlobalWeight * FED_WEIGHT_SCALE); // subtract our contribution
            // update gradient
            //agent.MyParams.ResetGrad(MM_RESET_GRAD);
            agent.MyParams.ScaleGrad(agent.GlobalWeight);
        }
    }
}



///////////////////////////////////////////////////////////////////////////////////////////////////

void Run()
{
#ifdef _MSC_VER
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    SetProcessAffinityMask(GetCurrentProcess(), 0xffff); // use P-cores, mask is cpu dependent
#endif

    //const yint AGENT_COUNT = 1;
    const yint AGENT_COUNT = 2;
    //const yint AGENT_COUNT = 5;
    //const yint AGENT_COUNT = 25;

    TString lmIndexDir;
    TIntrusivePtr<IDataSource> data = CreateDataSource(DATA_SCRIPT, &lmIndexDir);

    TModelParams startParams;
    //Serialize(IO_READ, "d:/fed/fed_start.bin", startParams);
    //startParams.ResetGrad();
    yint vocabSize = data->GetStats().VocabSize;
    yint modelFlags = 0;
    TModelDescr modelDescr;
    TXRng rng(1313);
    InitModelDescr(&modelDescr, "e256d30w128", ALIBI_V3, data->GetStats().VocabSize, MPF_TAIL_LOSS);
    InitModel(&startParams, rng, modelDescr, COMBINER_INIT_RANDOM, data->GetStats().Bias);

    TFedTrain fed;
    GetMatrices(&fed.Common.Params, startParams);
    yint startIteration = 20000;
    for (yint k = 0; k < AGENT_COUNT; ++k) {
        fed.AddAgent(startParams);
    }

    TTrainModelConfigParser trainCfg;
    {
        TConfigFile ts;
        ParseConfig(&ts, TRAIN_SCRIPT);
        for (auto &op : ts.OpArr) {
            trainCfg.ParseScriptOp(op, data);
        }
    }

    Train(data, trainCfg, startParams.ModelDescr, startParams.Bias, startIteration, &fed);
}
}
