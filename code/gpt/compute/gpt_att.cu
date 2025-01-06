#include "stdafx.h"
#define KERNEL_UNIT "gpt_attention/"
#include <gpt/att/att.h>
#include <gpt/att/rope.h>
#include <gpt/model_params/model_dim.h>
#include <lib/cuda/cuda_i8.cuh>
#include <lib/cuda/cuda_fp8.cuh>
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/vec_util.cuh>
#include <lib/random/mersenne.h>
#include <lib/hp_timer/hp_timer.h>

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace NCuda;


namespace NTestKernelsNg8
{
constexpr float QK_VEC_SCALE = 1 / 32.0f;
constexpr float V_VEC_SCALE = 1 / 32.0f;
constexpr int Q_DIM = 128;
constexpr int ATT_GROUP = 64;
constexpr int ATT_ALIGN = 128;

#include "gpt_rope.cuh"
#include "gpt_att_fp8.cuh"
}
using namespace NTestKernelsNg8;

static void PutValue(TStream &stream, TCudaVector<float> *p, float val)
{
    TVector<float> vec;
    vec.push_back(val);
    Put(stream, p, vec);
}

void TestAttFp8()
{
    TMersenne<ui32> rng(1313);
    (void)ATT_ALIGN;

    TStream stream;
    TCuda2DArray<i8> qk;
    TCuda2DArray<float> qkScale;
    TCuda2DArray<i8> qv;
    TCuda2DArray<float> qvScale;
    TCuda2DArray<i8> vT;
    TCuda2DArray<half> valLookup;
    TCudaVector<TAttentionSpanGroup<ATT_GROUP>> attSpans;
    TCudaVector<int> attSpanPtr;
    TCuda2DArray<float> sumWeightLog;
    TCudaVector<float> qvGlobalScale;

    //const int ITER_COUNT = 1;
    const int ITER_COUNT = 10;

    yint len = 16 * 1024; // sample count
    yint headCount = 4;
    yint qSum = 128 * headCount;
    yint ttSum = 128 * headCount;
    int spanGroups = DivCeil(len, ATT_GROUP);
    qk.Allocate(qSum, len);
    qkScale.Allocate(headCount, len);
    qv.Allocate(qSum, len);
    qvScale.Allocate(headCount, len);
    vT.Allocate(len, ttSum); // transposed?!
    valLookup.Allocate(ttSum, len);
    attSpans.Allocate(spanGroups * 4);
    attSpanPtr.Allocate(spanGroups + 1);
    sumWeightLog.AllocateCuda(1, len);
    qvGlobalScale.Allocate(1);

    TIntrusivePtr<TGraph> computer = new TGraph;
    {
        TGraph *c = computer.Get();
        for (yint iter = 0; iter < ITER_COUNT; ++iter) {
            yint lenTiles = len / ATT_GROUP;
            float alibiSlope = 0.01f;
            TCudaPOD<float> qq = qvGlobalScale.GetElement(0);
            CudaCall(c, Fp8Att).Grid(headCount, lenTiles)
                (qq, qk, qv, qvScale, vT)
                (attSpans, attSpanPtr, alibiSlope)
                .Write(&valLookup, &sumWeightLog);
        }
    }

    // create all to all attention graph
    TAttentionInfoGrouped<ATT_GROUP> aig;
    aig.Init();
    for (yint t = 0; t < len / ATT_GROUP; ++t) {
        TAttentionSpanGroup<ATT_GROUP> ag;
        for (yint k = 0; k < ATT_GROUP; ++k) {
            ag.SpanStart[k] = 0;
            ag.SpanFinish[k] = len - 1;
        }
        ag.Start = 0;
        ag.Finish = len - MM_TILE;
        TVector<TAttentionSpanGroup<ATT_GROUP>> agArr;
        agArr.push_back(ag);
        aig.AddSpanGroups(agArr);
    }
    Put(stream, &attSpans, aig.SpanGroups);
    Put(stream, &attSpanPtr, aig.SpanGroupPtr);

    PutValue(stream, &qvGlobalScale, 1.0f);

    FillRandom(rng, stream, &qk);
    qkScale.ClearDeviceMem(stream);
    FillRandom(rng, stream, &qv);
    qvScale.ClearDeviceMem(stream);
    FillRandom(rng, stream, &vT);
    stream.Sync();
    double maxTFlops = 0;
    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        computer->Run(stream);
        stream.Sync();
        double tPassed = NHPTimer::GetTimePassed(&tStart);
        double tFlops = ITER_COUNT * (qSum * len * len + ttSum * len * len) * 2. / tPassed / 1e12;
        maxTFlops = Max(maxTFlops, tFlops);
        DebugPrintf("%g TFlops, %g\n", maxTFlops, tFlops);
    }
}


void TestAttGradQVfp8()
{
    TMersenne<ui32> rng(1313);

    TStream stream;
    TCuda2DArray<i8> qk;
    TCuda2DArray<i8> qkT;
    TCuda2DArray<float> qkScale;
    TCuda2DArray<i8> qv;
    TCuda2DArray<i8> qvT;
    TCuda2DArray<float> qvScale;
    TCuda2DArray<i8> v;
    TCuda2DArray<i8> dValLookupE4;
    TCuda2DArray<i8> dValLookupE4T;
    TCuda2DArray<float> vScale;
    TCuda2DArray<float> dV;
    TCuda2DArray<float> dQV;
    TCuda2DArray<float> dQK;
    TCudaVector<float> scaleVec;
    TCuda2DArray<float> dValLookupE4Scale;
    TCuda2DArray<float> dScaleArr;
    TCuda2DArray<float> sumWeightLog;
    TCudaVector<TAttentionSpanGroup<ATT_GROUP>> attSpans;
    TCudaVector<int> attSpanPtr;
    TCudaVector<float> qvGlobalScale;
    TCuda2DArray<TRopeFloat> ropeBuf;

    //const int ITER_COUNT = 1;
    const int ITER_COUNT = 10;

    yint len = 16 * 1024; // sample count
    yint headCount = 4;
    yint qSum = 128 * headCount;
    yint ttSum = 128 * headCount;
    int spanGroups = DivCeil(len, ATT_GROUP);
    qk.Allocate(qSum, len);
    qkT.Allocate(len, qSum);
    qkScale.Allocate(headCount, len);
    qv.Allocate(qSum, len);
    qvT.Allocate(len, qSum);
    qvScale.Allocate(headCount, len);
    v.Allocate(ttSum, len);
    vScale.Allocate(headCount, len);
    dValLookupE4.Allocate(ttSum, len);
    dValLookupE4T.Allocate(len, ttSum);
    dValLookupE4Scale.AllocateCuda(len, headCount);
    dV.Allocate(ttSum, len);
    dQV.Allocate(qSum, len);
    dQK.Allocate(qSum, len);
    dScaleArr.Allocate(len, headCount);
    dScaleArr.ClearDeviceMem(stream);
    sumWeightLog.Allocate(len, headCount);
    sumWeightLog.ClearDeviceMem(stream);
    attSpans.Allocate(spanGroups * 4);
    attSpanPtr.Allocate(spanGroups + 1);
    qvGlobalScale.Allocate(1);
    FillRopeBuf(stream, &ropeBuf, 128, len);

    float usefulOps = 0;
    usefulOps += (qSum * len * len * 2 + ttSum * len * len * 2) * 2.; // grad QV
    usefulOps += (qSum * len * len * 2 + ttSum * len * len) * 2.; // grad QK

    TIntrusivePtr<TGraph> computer = new TGraph;
    {
        TGraph *c = computer.Get();
        for (yint iter = 0; iter < ITER_COUNT; ++iter) {
            yint lenTiles = len / ATT_GROUP;
            float alibiSlope = 0.01f;
            float attGradMult = 1;
            TCudaPOD<float> qq = qvGlobalScale.GetElement(0);
            CudaCall(c, Fp8AttGradQV).Grid(headCount, lenTiles)
                (qq, qk, qv, qvScale)
                (v, qkT)
                (dValLookupE4, dValLookupE4T, dValLookupE4Scale)
                (dScaleArr, sumWeightLog)
                (attSpans, attSpanPtr, alibiSlope)
                (attGradMult)
                (ropeBuf)
                (vScale).Write(&dQV, &dQV);

            CudaCall(c, Fp8AttGradQK).Grid(headCount, lenTiles)
                (qq, qk, qv, qvScale)
                (v, qvT)
                (dValLookupE4, dValLookupE4Scale)
                (dScaleArr, sumWeightLog)
                (attSpans, attSpanPtr, alibiSlope)
                (attGradMult)
                (ropeBuf)
                (qkScale).Write(&dQK);
        }
    }

    // create all to all attention graph
    TAttentionInfoGrouped<ATT_GROUP> aig;
    aig.Init();
    for (yint t = 0; t < len / ATT_GROUP; ++t) {
        TAttentionSpanGroup<ATT_GROUP> ag;
        for (yint k = 0; k < ATT_GROUP; ++k) {
            ag.SpanStart[k] = 0;
            ag.SpanFinish[k] = len - 1;
        }
        ag.Start = 0;
        ag.Finish = len - MM_TILE;
        TVector<TAttentionSpanGroup<ATT_GROUP>> agArr;
        agArr.push_back(ag);
        aig.AddSpanGroups(agArr);
    }
    Put(stream, &attSpans, aig.SpanGroups);
    Put(stream, &attSpanPtr, aig.SpanGroupPtr);

    PutValue(stream, &qvGlobalScale, 1.0f);

    FillRandom(rng, stream, &qk);
    FillRandom(rng, stream, &qkT);
    FillRandom(rng, stream, &qv);
    FillRandom(rng, stream, &qvT);
    FillRandom(rng, stream, &v);
    FillRandom(rng, stream, &dValLookupE4);
    FillRandom(rng, stream, &dValLookupE4T);
    stream.Sync();
    double maxTFlops = 0;
    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        computer->Run(stream);
        stream.Sync();
        double tPassed = NHPTimer::GetTimePassed(&tStart);
        double tFlops = ITER_COUNT * usefulOps / tPassed / 1e12;
        maxTFlops = Max(maxTFlops, tFlops);
        DebugPrintf("%g TFlops, %g\n", maxTFlops, tFlops);
    }
}
