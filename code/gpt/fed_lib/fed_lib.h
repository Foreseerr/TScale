#pragma once
#include <lib/guid/guid.h>
#include <gpt/data/data.h>
#include <gpt/model_params/model_params.h>
#include <gpt/train_config/train_config.h>
#include <util/fast_io.h>
#include <util/mem_io.h>


extern TGuid FedToken;

const yint FED_HTTP_PORT = 18181;
const yint FED_GRAD_PORT = 18182;


struct TFedLogin
{
    TGuid UserId;
    TString UserName;
    SAVELOAD(UserId, UserName);
};

struct TFedParams
{
    TModelDescr ModelDescr;
    TModelRowDisp RowDisp;
    TVector<float> BiasArr;
    double TrainBatchCount = 0;
    TDescentConfig DescentConfig;
    float Compression = 0;
    TString DataServerAddr;
    float FedWeightScale = 0;
    SAVELOAD(ModelDescr, RowDisp, BiasArr, TrainBatchCount, DescentConfig, Compression, DataServerAddr, FedWeightScale);
};

bool IsValidUsername(const TString &x);
