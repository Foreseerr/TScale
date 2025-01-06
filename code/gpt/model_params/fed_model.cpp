#include "stdafx.h"
#include "fed_model.h"


void ConvertToModelParams(TModelParams *p, const TFedCenterModel &fm)
{
    InitModelZero(p, fm.ModelDescr, fm.BiasArr);
    AddMatrices(p, fm.Params, 1);
    SetRowDisp(p, fm.RowDisp);
}
