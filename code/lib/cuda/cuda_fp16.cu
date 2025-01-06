#include "stdafx.h"
#define KERNEL_UNIT "cuda_fp16/"
#include "cuda_fp16.cuh"
//#include "cuda_graph.cuh"
#include <lib/random/mersenne.h>
#include <lib/hp_timer/hp_timer.h>


using namespace NCuda;

//__global__ void TestLoadTile(TCuda2DPtr<half> data, TCuda2DPtr<half> dst)
//{
//    TTileCoord tc;
//    __shared__ T4x4SMemHalfTile frag;
//    int warpId = threadIdx.y;
//
//    Copy4x4Tile(&frag, warpId, data[0], data.GetStride());
//    __syncthreads();
//
//    if (warpId == 0) {
//        for (int x = 0; x < 4; ++x) {
//            for (int y = 0; y < 4; ++y) {
//                TRegTile<half> tt;
//                //LoadTile(&tt, frag, x, y);
//                LoadTileTransposed(&tt, frag, x, y);
//                tt.Store(tc, dst[y * 16] + x * 16, dst.GetStride());
//            }
//        }
//    }
//}



void TestMatMulFp16(bool isHalfAccum)
{
    TMersenne<ui32> rng(1313);

    TStream stream;
    TCuda2DArray<half> aMatr;
    TCuda2DArray<half> bMatr;
    TCuda2DArray<float> resMatr;
    TCuda2DArray<float> resMatrRef;

#ifndef NDEBUG
    const int ITER_COUNT = 1;
#else
    const int ITER_COUNT = 100;
#endif

    // realistic setup
    int xSize = 16 * 1024; // sample count
    int ySize = 4096; // combiner width of tt=256
    int zSize = 1024; // state dim
    //// test transposed matmuls
    //int xSize = 4096;
    //int ySize = 4096;
    //int zSize = 4096;

    aMatr.Allocate(ySize, xSize);
    bMatr.Allocate(ySize, zSize);
    resMatr.Allocate(zSize, xSize);
    resMatrRef.Allocate(zSize, xSize);

    TIntrusivePtr<TGraph> computer = new TGraph;
    {
        TGraph *c = computer.Get();
        //CudaCall(c, TestLoadTile).Block(WARP_SIZE, MM_BATCH)(aMatr).Write(&aChk);
        for (yint iter = 0; iter < ITER_COUNT; ++iter) {
            if (isHalfAccum) {
                MatMulXYoZYeXZhalf<TStore>(c, aMatr, bMatr, &resMatr, xSize / MM_TILE, ySize / MM_TILE, zSize / MM_TILE).Struct();
            } else {
                MatMulXYoZYeXZ<TStore>(c, aMatr, bMatr, &resMatr, xSize / MM_TILE, ySize / MM_TILE, zSize / MM_TILE).Struct();
                //MatMulXYoYZeXZ<TStore>(c, aMatr, bMatr, &resMatr, xSize / MM_TILE, ySize / MM_TILE, zSize / MM_TILE).Struct();
                //MatMulXYoXZeYZ<TStore>(c, aMatr, bMatr, &resMatr, xSize / MM_TILE, ySize / MM_TILE, zSize / MM_TILE).Struct();
            }
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
        DebugPrintf("%g TFlops\n", maxTFlops);

#ifndef NDEBUG
        resMatr.CopyToHost(stream);
        resMatrRef.CopyToHost(stream);
        stream.Sync();
        TArray2D<float> ra, rb;
        GetAllData(resMatr, &ra);
        GetAllData(resMatrRef, &rb);
        for (yint y = 0; y < ra.GetYSize(); ++y) {
            for (yint x = 0; x < ra.GetXSize(); ++x) {
                Y_VERIFY(ra[y][x] == rb[y][x]);
            }
        }
#endif
    }
}
