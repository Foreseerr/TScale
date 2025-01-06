#pragma once
#include "batch_config.h"
#include <gpt/data/bpe.h>
#include <gpt/data/data.h>
#include <gpt/compute/model.h>
#include <gpt/train_config/train_config.h>
#include <lib/hp_timer/hp_timer.h>
#include <lib/random/mersenne.h>
#include <lib/random/rand_utils.h>


double CalcModelErr(const TVector<TFragment> &fragArr, IComputeContext *pCtx);
double CalcModelErr(const TVector<TVector<TFragment>> &batchArr, IComputeContext *pCtx);
double CalcTargetLoss(const TVector<TVector<float>> &predArr, const TVector<TNodeTarget> &target);


///////////////////////////////////////////////////////////////////////////////////////////////////
class TTrainContext
{
    TIntrusivePtr<IDataSource> Data;
    TDescentConfig DescentConfig;
    TDeviceBatchConfig DeviceBatchConfig;
    TVector<TVector<TFragment>> ScoreTrainBatches;
    TVector<TVector<TFragment>> ScoreTestBatches;
    bool SaveModel = false;
    yint MaxIters = 1000;
    yint EvalInterval = 100;

public:
    TTrainContext(TIntrusivePtr<IDataSource> data, const TDescentConfig &descent,
        yint deviceCount, yint limitNodeCount,
        bool saveModel, yint maxIters, yint evalInterval)
        : Data(data), DescentConfig(descent)
        , DeviceBatchConfig(deviceCount, limitNodeCount, descent.TrainBatchSize, descent.TrainFragLen)
        , SaveModel(saveModel), MaxIters(maxIters), EvalInterval(evalInterval)
    {
    }
    const TDescentConfig &GetDescentConfig() const { return DescentConfig; }
    const TDeviceBatchConfig &GetDeviceBatchConfig() const { return DeviceBatchConfig; }
    const TVector<TVector<TFragment>> &GetScoreTrainBatches() const { return ScoreTrainBatches; }
    const TVector<TVector<TFragment>> &GetScoreTestBatches() const { return ScoreTestBatches; }
    float GetCompression() const { return Data->GetStats().Compression; }
    bool IsSaveModel() const { return SaveModel; }
    yint GetMaxIters() const { return MaxIters; }
    yint GetEvalInterval() const { return EvalInterval; }
    TTrainingStep GetStep(yint iter) const
    {
        return DescentConfig.GetStep(iter, MaxIters);
    }

    void MakeScoreBatches(yint batchCount, yint batchSize, yint len)
    {
        if (Data->GetStats().HasTest) {
            yint chkRngSeed = 1313;
            for (yint k = 0; k < batchCount; ++k) {
                TVector<TFragment> &testBatch = *ScoreTestBatches.insert(ScoreTestBatches.end());
                Data->SampleFragments(IDataSource::TEST, chkRngSeed++, batchSize, len, &testBatch);
            }
        }
        yint chkRngSeed = 31313;
        for (yint k = 0; k < batchCount; ++k) {
            TVector<TFragment> &trainBatch = *ScoreTrainBatches.insert(ScoreTrainBatches.end());
            Data->SampleFragments(IDataSource::TRAIN, chkRngSeed++, batchSize, len, &trainBatch);
        }
    }

    void SampleTrainBatches(yint rngSeed, TVector<TFragment> *pRes) const
    {
        Data->SampleFragments(IDataSource::TRAIN, rngSeed, DescentConfig.TrainBatchSize, DescentConfig.TrainFragLen, pRes);
    }
};
