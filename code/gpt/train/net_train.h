#pragma once
#include <gpt/model_params/model_params.h>

class TTrainContext;

namespace NNetTrain
{
void RunWorker(yint port);
void RunMaster(yint startIteration, yint deviceCount, const TVector<TString> &workerAddrArr, const TTrainContext &trainCtx, TIntrusivePtr<TModelParamsHolder> pParams);
}
