#include "stdafx.h"
#define KERNEL_UNIT "cuda_mma/"
#include "cuda_mma.cuh"
#include "cuda_graph.cuh"
#include <lib/random/rand_utils.h>
#include <lib/math/matrix_utils.h>


namespace NCuda
{
__global__ void TestIntMMA(TCuda2DPtr<i8> a, TCuda2DPtr<i8> b, TCuda2DPtr<int> ab)
{
    __shared__ T8SMemI8Tile shA;
    __shared__ T8SMemI8Tile shB;
    Copy8Tile(&shA, a);
    Copy8Tile(&shB, b);

    TTileCoord tc;
    TRegTile<int> res;
    res.Clear();
    TRegTile<i8> tileA;
    TRegTile<i8> tileB;
    for (int k = 0; k < 8; ++k) {
        LoadTile(&tileA, shA, k);
        LoadTile(&tileB, shB, k);
        MMA(&res, tileA, tileB);
    }
    res.Store(tc, ab);
}



template <class TRng, class T>
static void InitRandomMatrix(TRng &rng, TArray2D<T> *pRes, yint xSize, yint ySize)
{
    pRes->SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*pRes)[y][x] = rng.Uniform(100) - 50;
        }
    }
}


template <class T>
static TArray2D<double> Convert(const TArray2D<T> &matr)
{
    yint xSize = matr.GetXSize();
    yint ySize = matr.GetYSize();
    TArray2D<double> res;
    res.SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            res[y][x] = matr[y][x];
        }
    }
    return res;
}

}
using namespace NCuda;
void TestMMA()
{
    TStream stream;

    TMersenne<ui32> rng(1313);
    TIntrusivePtr<TGraph> c = new TGraph;
    TCuda2DArray<i8> a;
    TCuda2DArray<i8> b;
    TCuda2DArray<int> ab;
    a.Allocate(128, 16);
    b.Allocate(128, 16);
    ab.Allocate(16, 16);

    CudaCall(c, TestIntMMA)(a, b).Write(&ab);

    for (;;) {
        TArray2D<i8> refA;
        TArray2D<i8> refB;
        InitRandomMatrix(rng, &refA, a.GetXSize(), a.GetYSize());
        InitRandomMatrix(rng, &refB, b.GetXSize(), b.GetYSize());
        Put(stream, &a, refA);
        Put(stream, &b, refB);

        c->Run(stream);
        ab.CopyToHost(stream);
        stream.Sync();

        TArray2D<double> refAB;
        MatrixMult(Convert(refA), Transpose(Convert(refB)), &refAB);
        TArray2D<int> gpuAB;
        GetAllData(ab, &gpuAB);
        for (yint y = 0; y < gpuAB.GetYSize(); ++y) {
            for (yint x = 0; x < gpuAB.GetXSize(); ++x) {
                Y_VERIFY(refAB[y][x] == gpuAB[y][x]);
            }
        }
        printf(".");
    }
}
