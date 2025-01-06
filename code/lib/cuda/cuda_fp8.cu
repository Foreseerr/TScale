#include "stdafx.h"
#define KERNEL_UNIT "cuda_fp8/"
#include "cuda_fp8.cuh"
#include "cuda_graph.cuh"
#include <lib/random/mersenne.h>
#include <lib/hp_timer/hp_timer.h>


using namespace NCuda;

void TestMatMulFp8()
{
    TMersenne<ui32> rng(1313);

    TStream stream;
    TCuda2DArray<e4m3> aMatr;
    TCuda2DArray<e4m3> bMatr;
    TCuda2DArray<float> resMatr;

    //const int ITER_COUNT = 1;
    const int ITER_COUNT = 100;

    int xSize = 16 * 1024; // sample count
    int ySize = 4096; // combiner width of tt=256
    int zSize = 1024; // state dim
    aMatr.Allocate(ySize, xSize);
    bMatr.Allocate(ySize, zSize);
    resMatr.Allocate(zSize, xSize);

    TIntrusivePtr<TGraph> computer = new TGraph;
    {
        TGraph *c = computer.Get();
        for (yint iter = 0; iter < ITER_COUNT; ++iter) {
            Fp8MatMul<TStoreScaled>(c, aMatr, bMatr, &resMatr, xSize / MM_TILE, ySize / MM_TILE, zSize / MM_TILE).Struct()(nullptr, 2.0f);
        }
    }

    FillRandom(rng, stream, &aMatr);
    FillRandom(rng, stream, &bMatr);
    stream.Sync();
    double maxTFlops = 0;
    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        computer->Run(stream);
        stream.Sync();
        double tPassed = NHPTimer::GetTimePassed(&tStart);
        double tFlops = 2. * ITER_COUNT * xSize * ySize * zSize / tPassed / 1e12;
        maxTFlops = Max(maxTFlops, tFlops);
        DebugPrintf("%g TFlops, %g\n", maxTFlops, tFlops);
    }
}
