#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
struct TDeviceBatchConfig
{
    yint DeviceCount = 0;
    yint AccumulateSteps = 0;
    yint DeviceBatchSize = 0;
    yint FragLen = 0;

public:
    TDeviceBatchConfig() {}
    TDeviceBatchConfig(yint deviceCount, yint limitNodeCount, yint batchSize, yint fragLen);
    yint GetDeviceMaxNodeCount() const;
};
