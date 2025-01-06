#include "stdafx.h"
#define KERNEL_UNIT "cuda_i8/"
#include "cuda_i8.cuh"
#include "cuda_graph.cuh"
#include <lib/random/mersenne.h>
#include <lib/hp_timer/hp_timer.h>


namespace NCuda
{
__global__ void TransposeI8Matrix(TCuda2DPtr<i8> src, TCuda2DPtr<i8> dst)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);
    int xBlock = blockIdx.x * MM_TILE;
    int yBlock = blockIdx.y * MM_TILE;

    __shared__ i8 buf[128][128];

    int h = threadIdx.x;
    int warpId = threadIdx.y;
    for (int yBase = 0; yBase < 128; yBase += I8_TRANSPOSE_WARPS) {
        int y = yBase + warpId;
        int xOffset = h * 4;
        int xorAddr = y & ~3;
        int *pSrc = (int *)&src[yBlock + y][xBlock + xOffset];
        int *pDst = (int *)&buf[y][xOffset ^ xorAddr];
        *pDst = *pSrc;
    }
    __syncthreads();

    for (int yBase = 0; yBase < 128; yBase += I8_TRANSPOSE_WARPS) {
        int y = yBase + warpId;
        int xOffset = h * 4;
        union {
            int column;
            i8 columnBytes[4];
        };
        for (int k = 0; k < 4; ++k) {
            int readX = y;
            int readY = xOffset + k;
            int xorAddr = readY & ~3;
            columnBytes[k] = buf[readY][readX ^ xorAddr];
        }
        int *pDst = (int *)&dst[xBlock + y][yBlock + xOffset];
        *pDst = column;
    }
}
}


using namespace NCuda;

void TestMatMulInt8()
{
    TMersenne<ui32> rng(1313);

    TStream stream;
    TCuda2DArray<i8> aMatr;
    TCuda2DArray<i8> bMatr;
    TCuda2DArray<int> resMatr;
    TCuda2DArray<int> resMatrRef;
    TCuda2DArray<float> resMatrFloat;
    TCudaVector<float> aMatrRowScale;
    TCudaVector<float> yTileScale;

    //const int ITER_COUNT = 1;
    const int ITER_COUNT = 100;

    int xSize = 16 * 1024; // sample count
    int ySize = 4096; // combiner width of tt=256
    int zSize = 1024; // state dim
    aMatr.Allocate(ySize, xSize);
    bMatr.Allocate(ySize, zSize);
    resMatr.Allocate(zSize, xSize);
    resMatrRef.Allocate(zSize, xSize);
    resMatrFloat.Allocate(zSize, xSize);
    aMatrRowScale.AllocateCuda(xSize);
    aMatrRowScale.ClearDeviceMem(stream);
    yTileScale.Allocate(ySize / MM_TILE);

    TVector<float> yScale;
    ClearPodArray(&yScale, yTileScale.GetSize());
    for (yint i = 0; i < YSize(yScale); ++i) {
        yScale[i] = i + 1;
        //yScale[i] = 1;
    }
    Put(stream, &yTileScale, yScale);

    TIntrusivePtr<TGraph> computer = new TGraph;
    {
        TGraph *c = computer.Get();
        for (yint iter = 0; iter < ITER_COUNT; ++iter) {
            //Int8MatMulOld<TStore>(c, aMatr, bMatr, &resMatrRef, xSize / MM_TILE, ySize / MM_TILE, zSize / MM_TILE).Struct();
            Int8MatMul<TStore>(c, aMatr, bMatr, &resMatr, xSize / MM_TILE, ySize / MM_TILE, zSize / MM_TILE).Struct();
            //Int8MatMulRowScale<TStore>(c, aMatr, aMatrRowScale, bMatr, &resMatrFloat, xSize / MM_TILE, ySize / MM_TILE, zSize / MM_TILE).Struct();
            //Int8MatMulYScale<TStore>(c, aMatr, bMatr, yTileScale, &resMatrFloat, xSize / MM_TILE, ySize / MM_TILE, zSize / MM_TILE).Struct();
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

        resMatr.CopyToHost(stream);
        resMatrRef.CopyToHost(stream);
        stream.Sync();
        TArray2D<int> ra, rb;
        GetAllData(resMatr, &ra);
        GetAllData(resMatrRef, &rb);
        for (yint y = 0; y < ra.GetYSize(); ++y) {
            for (yint x = 0; x < ra.GetXSize(); ++x) {
                Y_ASSERT(ra[y][x] == rb[y][x]);
            }
        }
    }
}
