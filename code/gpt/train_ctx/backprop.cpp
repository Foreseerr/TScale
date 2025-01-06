#include "stdafx.h"
#include "backprop.h"
#include <gpt/att/sliding_window.h>


void BackpropBatch(TXRng &rng, const TDescentConfig &dc, const TDeviceBatchConfig &dbc,
    const TTrainingStep &step, const TVector<TFragment> &fragArr,
    TIntrusivePtr<IComputeContext> ctx)
{
    for (yint accStep = 0; accStep < dbc.AccumulateSteps; ++accStep) {
        EAddToModel addToModel = (accStep == dbc.AccumulateSteps - 1) ? GRADIENT_APPLY : GRADIENT_ACCUMULATE;
        yint base = accStep * dbc.DeviceCount * dbc.DeviceBatchSize;
        // provide train data to devices
        for (yint deviceId = 0; deviceId < dbc.DeviceCount; ++deviceId) {
            TVector<TFragment> devFrags;
            for (yint k = 0; k < dbc.DeviceBatchSize; ++k) {
                devFrags.push_back(fragArr[base + deviceId * dbc.DeviceBatchSize + k]);
            }
            MakeTrain(rng, devFrags, dc.TokenDrop, dc.ChannelDrop, ctx.Get(), deviceId);
        }
        // backprop
        ctx->Backprop(step, addToModel);
    }
}
