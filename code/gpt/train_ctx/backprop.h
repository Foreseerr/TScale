#pragma once
#include "batch_config.h"
#include <gpt/rng/xrng.h>
#include <gpt/compute/model.h>
#include <gpt/train_config/train_config.h>


void BackpropBatch(TXRng &rng, const TDescentConfig &dc, const TDeviceBatchConfig &dbc,
    const TTrainingStep &step, const TVector<TFragment> &fragArr,
    TIntrusivePtr<IComputeContext> ctx);
