#pragma once
#include "model_params.h"


struct TFedCenterModel
{
    TAllModelMatrices Params;
    TModelRowDisp RowDisp;
    TModelDescr ModelDescr;
    TVector<float> BiasArr;
    double TrainBatchCount = 0;
    SAVELOAD(Params, RowDisp, ModelDescr, BiasArr, TrainBatchCount);

    bool IsEmpty() const { return Params.IsEmpty(); }
};


void ConvertToModelParams(TModelParams *p, const TFedCenterModel &fm);
