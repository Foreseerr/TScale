#include "stdafx.h"
#include "batch_config.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
TDeviceBatchConfig::TDeviceBatchConfig(yint deviceCount, yint limitNodeCount, yint batchSize, yint fragLen)
    : DeviceCount(deviceCount), FragLen(fragLen)
{
    if (fragLen > limitNodeCount) {
        DebugPrintf("train frag length %g does not fit into available %g nodes\n", fragLen * 1., limitNodeCount * 1.); fflush(0);
        abort();
    }
    yint trainFragPerDevice = batchSize / deviceCount;
    if (trainFragPerDevice == 0 || trainFragPerDevice * deviceCount != batchSize) {
        DebugPrintf("suboptimal configuration, %g fragments per device (%g fragments, %g devices)\n",
            trainFragPerDevice * 1., batchSize * 1., deviceCount * 1.); fflush(0);
        abort();
    }
    yint maxFragPerBatch = limitNodeCount / fragLen;
    Y_VERIFY(maxFragPerBatch > 0);
    for (yint fragPerBatch = Min<yint>(maxFragPerBatch, trainFragPerDevice); fragPerBatch >= 1; --fragPerBatch) {
        if ((trainFragPerDevice % fragPerBatch) == 0) {
            DeviceBatchSize = fragPerBatch;
            AccumulateSteps = trainFragPerDevice / fragPerBatch;
            TString strDeviceCount = (DeviceCount > 1) ? Sprintf("%g devices, ", DeviceCount * 1.).c_str() : "";
            TString strAccSteps = (AccumulateSteps > 1) ? Sprintf("%g accumulation steps, ", AccumulateSteps * 1.).c_str() : "";
            DebugPrintf("%s%s%g fragments per step, %g fragment legnth\n", strDeviceCount.c_str(), strAccSteps.c_str(), DeviceBatchSize * 1., FragLen * 1.); fflush(0);
            return;
        }
    }
}


yint TDeviceBatchConfig::GetDeviceMaxNodeCount() const
{
    return FragLen * DeviceBatchSize;
}
