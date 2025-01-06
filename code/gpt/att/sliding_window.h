#pragma once
#include <gpt/rng/xrng.h>
#include "nodes_batch.h"
#include <gpt/model_params/model_dim.h>


void InitModelDescr(TModelDescr *pRes, const TString &modelDescrStr, EAlibi alibi, yint vocabSize, ui64 flags);

struct TFragment;


enum {
    ATT_GRAPH_TRAIN_LOSS,
    ATT_GRAPH_TEST_LOSS,
};

void InitLabelData(const TModelDescr &modelDescr, TXRng &rng, float tokenDrop,
    const TVector<TFragment> &fragArr, yint lossType,
    TNodesBatch *pNodes);

// process set of fragments and init context
template <class TComputeContext>
inline void MakeTrain(TXRng &rng, const TVector<TFragment> &fragArr,
    float tokenDrop, float channelDrop,
    TComputeContext *pCtx, yint deviceId, TVector<TNodeTarget> *pTarget)
{
    TNodesBatch &nodes = pCtx->GetNodes(deviceId);
    TVector<ui32> &dropTable = pCtx->GetDropTable(deviceId);
    TModelDescr modelDescr = pCtx->GetModelDescr();
    InitLabelData(modelDescr, rng, tokenDrop, fragArr, ATT_GRAPH_TRAIN_LOSS, &nodes);
    MakeDropTable(rng, modelDescr, &dropTable, channelDrop);
    pCtx->Init(deviceId, TComputeContext::INIT_TRAIN);
    if (pTarget) {
        *pTarget = nodes.Target;
    }
}


extern TXRng NopRng;

template <class TComputeContext>
void MakeTest(const TVector<TFragment> &fragArr, TComputeContext *pCtx, yint deviceId)
{
    TNodesBatch &nodes = pCtx->GetNodes(deviceId);
    TVector<ui32> &dropTable = pCtx->GetDropTable(deviceId);
    TModelDescr modelDescr = pCtx->GetModelDescr();
    InitLabelData(modelDescr, NopRng, 1., fragArr, ATT_GRAPH_TEST_LOSS, &nodes);
    dropTable.resize(0);
    dropTable.resize(CalcDropTableSize(modelDescr), ~0);
    pCtx->Init(deviceId, TComputeContext::INIT_TEST);
}


// call when not interested in target loss computation
template <class TComputeContext>
inline void MakeTrain(TXRng &rng, const TVector<TFragment> &fragArr, float tokenDrop, float channelDrop, TComputeContext *pCtx, yint deviceId)
{
    MakeTrain(rng, fragArr, tokenDrop, channelDrop, pCtx, deviceId, nullptr);
}
