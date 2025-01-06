#include "stdafx.h"
#define KERNEL_UNIT "cuda_sort/"
#include "cuda_sort.cuh"
#include <lib/random/mersenne.h>
#include <lib/hp_timer/hp_timer.h>


namespace NCuda
{

inline __device__ void Order(TSortNode *a, TSortNode *b)
{
    if (a->Score > b->Score || (a->Score == b->Score && a->NodeId > b->NodeId)) {
        TSortNode tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

template <int N_WARP_COUNT>
inline __device__ void SortPass(int thrIdx, TCuda1DPtr<TSortNode> nodes, int nodeCount, int sz, int bit, int bitValue)
{
    for (int i = thrIdx; i < nodeCount - bit; i += WARP_SIZE * N_WARP_COUNT) {
        int opp = i + bit;
        if ((i & bit) == bitValue && ((i ^ opp) & ~(sz - 1)) == 0) {
            Order(&nodes[i], &nodes[opp]);
        }
    }
    if (N_WARP_COUNT > 1) {
        __syncthreads();
    } else {
        __syncwarp();
    }
}

template <int N_WARP_COUNT>
inline __device__ void ButcherOddEvenMergeSort(int thrIdx, TCuda1DPtr<TSortNode> nodes, int nodeCount)
{
    // Batcher odd even merge sort (sorting network, to understand better search for picture)
    for (int bit = 1; bit < nodeCount; bit *= 2) {
        int sz = bit * 2; // size of lists to sort on this iteration
        SortPass<N_WARP_COUNT>(thrIdx, nodes, nodeCount, sz, bit, 0);
        for (int sub = bit / 2; sub > 0; sub /= 2) {
            SortPass<N_WARP_COUNT>(thrIdx, nodes, nodeCount, sz, sub, sub);
        }
    }
}


inline __device__ void PerformRegularSort(TCuda1DPtr<float> valArr, int nodeCount, TCuda1DPtr<TSortNode> nodes)
{
    int thrIdx = threadIdx.x + threadIdx.y * WARP_SIZE;
    for (int i = thrIdx; i < nodeCount; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
        TSortNode &dst = nodes[i];
        dst.NodeId = i;
        dst.Score = valArr[i];
    }
    __syncthreads();
    ButcherOddEvenMergeSort<FLOAT_SORT_WARP_COUNT>(thrIdx, nodes, nodeCount);
}


__global__ void SortFloatsKernel(TCuda1DPtr<float> valArr, int nodeCount, TCuda1DPtr<TSortNode> nodes)
{
    PerformRegularSort(valArr, nodeCount, nodes);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void HistSortFloatsKernel(TCuda1DPtr<float> valArr, int nodeCount, TCuda1DPtr<TSortNode> nodes)
{
    constexpr int N_BITS = 13;
    constexpr int SZ = 1 << N_BITS;
    __shared__ int hist[SZ];

    int thrIdx = threadIdx.x + threadIdx.y * WARP_SIZE;
    // clear histogram
    for (int i = thrIdx; i < SZ; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
        hist[i] = 0;
    }
    __syncthreads();

    // collect histogram
    for (int i = thrIdx; i < nodeCount; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
        CUDA_ASSERT(valArr[i] >= 0 && "implementation for positive floats");
        float val = valArr[i];
        ui32 bin = __float_as_uint(val) >> (31 - N_BITS);
        atomicAdd(&hist[bin], 1);
    }
    __syncthreads();

    // sum histogram
    for (int bit = 1; bit < SZ; bit *= 2) {
        for (int i = thrIdx; i < SZ; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
            if (i & bit) {
                hist[i] += hist[(i ^ bit) | (bit - 1)];
            }
        }
        __syncthreads();
    }

    // output result
    for (int i = thrIdx; i < nodeCount; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
        float val = valArr[i];
        ui32 bin = __float_as_uint(val) >> (31 - N_BITS);
        int ptr = atomicAdd(&hist[bin], -1) - 1;
        TSortNode &dst = nodes[ptr];
        dst.NodeId = i;
        dst.Score = val;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void HistSortFloatsStableKernel(TCuda1DPtr<float> valArr, int nodeCount, TCuda1DPtr<TSortNode> nodes)
{
    if (nodeCount < 500) {
        PerformRegularSort(valArr, nodeCount, nodes);
        return;
    }
    constexpr int N_BITS = 13;
    constexpr int SZ = 1 << N_BITS;
    int thrIdx = threadIdx.x + threadIdx.y * WARP_SIZE;

    __shared__ int hist[SZ + 1];

    // clear histogram
    for (int i = thrIdx; i < SZ; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
        hist[i] = 0;
    }
    __syncthreads();

    // collect histogram
    for (int i = thrIdx; i < nodeCount; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
        CUDA_ASSERT(valArr[i] >= 0 && "implementation for positive floats");
        float val = valArr[i];
        ui32 bin = __float_as_uint(val) >> (31 - N_BITS);
        atomicAdd(&hist[bin], 1);
    }
    __syncthreads();

    // sum histogram
    for (int bit = 1; bit < SZ; bit *= 2) {
        for (int i = thrIdx; i < SZ; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
            if (i & bit) {
                hist[i] += hist[(i ^ bit) | (bit - 1)];
            }
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        for (int i = threadIdx.x; i < nodeCount; i += WARP_SIZE) {
            float val = valArr[i];
            ui32 bin = __float_as_uint(val) >> (31 - N_BITS);
            int ptr = atomicAdd(&hist[bin], -1) - 1;
            TSortNode &dst = nodes[ptr];
            dst.NodeId = i;
            dst.Score = val;
        }
    }
    //// output result
    //for (int i = thrIdx; i < nodeCount; i += WARP_SIZE * FLOAT_SORT_WARP_COUNT) {
    //    float val = valArr[i];
    //    ui32 bin = __float_as_uint(val) >> (31 - N_BITS);
    //    int ptr = atomicAdd(&hist[bin], -1) - 1;
    //    TSortNode &dst = nodes[ptr];
    //    dst.NodeId = i;
    //    dst.Score = val;
    //}
    //if (thrIdx == 0) {
    //    hist[SZ] = nodeCount;
    //}
    //__syncthreads();

    //// sort within each bin
    //int h = threadIdx.x;
    //int warpId = threadIdx.y;
    //for (int largeBin = warpId; largeBin < SZ; largeBin += FLOAT_SORT_WARP_COUNT) {
    //    int beg = hist[largeBin];
    //    int fin = largeBin < SZ - 1 ? hist[largeBin + 1] : nodeCount;
    //    if (fin - beg > 1) {
    //        ButcherOddEvenMergeSort<1>(h, nodes + beg, fin - beg);
    //    }
    //}
}

}

using namespace NCuda;

static ui32 CalcResultHash(const TVector<TSortNode> &xx)
{
    ui64 res = 0xdeadf00d;
    for (const TSortNode &sn : xx) {
        res = res * 0x2349057 + sn.NodeId;
    }
    return res >> 32;
}

void TestCudaSort()
{
    TMersenne<ui32> rng(1313);

    TStream stream;
    TCudaVector<float> valArr;
    TCudaVector<TSortNode> sorted;

    //const int ITER_COUNT = 1;
    const int ITER_COUNT = 100;

    int len = 16 * 1024; // sample count
    valArr.Allocate(len);
    sorted.Allocate(len);

    TVector<float> xx;
    for (yint i = 0; i < len; ++i) {
        xx.push_back(rng.GenRandReal3());
    }
    Put(stream, &valArr, xx);

    TIntrusivePtr<TGraph> computer = new TGraph;
    {
        TGraph *c = computer.Get();
        for (yint iter = 0; iter < ITER_COUNT; ++iter) {
            //SortFloats(c, valArr, len, &sorted);
            //SortPositiveFloatsApproxFast(c, valArr, len, &sorted);
            SortPositiveFloatsApproxStable(c, valArr, len, &sorted);
        }
    }

    stream.Sync();
    double minTime = 1e10;
    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        computer->Run(stream);
        stream.Sync();

        sorted.CopyToHost(stream);
        stream.Sync();
        TVector<TSortNode> res;
        GetAllData(sorted, &res);
        //for (int k = 1; k < YSize(res); ++k) {
        //    Y_ASSERT(res[k - 1].Score <= res[k].Score);
        //}
        ui32 hh = CalcResultHash(res);
        if (hh != 1749808112) { // stability check
            __debugbreak();
        }

        double tPassed = NHPTimer::GetTimePassed(&tStart);
        minTime = Min(minTime, tPassed);
        DebugPrintf("%g ms, %g\n", minTime * 1000, tPassed * 1000);
    }
}
