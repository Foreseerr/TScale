#include "stdafx.h"
#include "train_ctx.h"
#include <gpt/att/sliding_window.h>


const double LOSS_SCALE = 1;
//const double LOSS_SCALE = 1 / log(0.5); // bpc, bits per character


///////////////////////////////////////////////////////////////////////////////////////////////////
// 
double CalcModelErr(const TVector<TFragment> &fragArr, IComputeContext *pCtx)
{
    if (fragArr.empty()) {
        return 0;
    }
    MakeTest(fragArr, pCtx, MAIN_DEVICE);
    return pCtx->ComputeScore();
}


double CalcModelErr(const TVector<TVector<TFragment>> &batchArr, IComputeContext *pCtx)
{
    if (batchArr.empty()) {
        return 0;
    }
    double err = 0;
    for (const TVector<TFragment> &b : batchArr) {
        err += CalcModelErr(b, pCtx);
    }
    return err / YSize(batchArr);
}


double CalcTargetLoss(const TVector<TVector<float>> &predArr, const TVector<TNodeTarget> &target)
{
    double res = 0;
    for (const TNodeTarget &nt : target) {
        res += log(predArr[nt.Node][nt.TargetId]);
    }
    return res;
}
