#pragma once
#include <lib/cuda/cuda_util.cuh>
#include <lib/cuda/cuda_arrays.h>
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/vec_util.cuh>


namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TSrc, class TDst>
__global__ void NormalizeVecsRowMaxKernel(int dim, int len,
    TCuda2DPtr<TSrc> state,
    TCuda2DPtr<TDst> normState, TCuda1DPtr<float> normStateScale)
{
    int h = threadIdx.x;
    int t = blockIdx.x;
    extern __shared__ float buf[];

    float discrScale = 0;
    if (t >= len) {
        StoreZeroSmemVec(dim, normState[t]);
    } else {
        LoadSmemVec(dim, state[t], buf);
        float maxVal = CalcMaxAbsValueSmem(dim, buf);
        float mult = 0;
        if (maxVal > 0) {
            discrScale = GetMaxDiscrScale(maxVal, (TDst *)0);
            mult = 1 / discrScale;

            TCudaRngLCG rng(*(ui32 *)&buf[h], *(ui32 *)&buf[0], t);
            for (int base = 0; base < dim; base += WARP_SIZE) {
                int d = base + h;
                buf[d] = buf[d] * mult + rng.GenUniformFloat() - 0.5f;
            }
        }
        StoreScaledSmemVec(dim, buf, 1.f, normState[t]);
    }
    if (h == 0) {
        normStateScale[t] = discrScale;
    }
}


template <class TLen, class TSrc, class TDst>
void NormalizeVecsRowMaxWithNoise(TIntrusivePtr<TGraph> c,
    int stateDim, TLen &&len, TLen &&lenRoundSize,
    TSrc &src,
    TCuda2DArray<TDst> *pDst, TCudaVector<float> *pDstRowScale)
{
    CudaCall(c, NormalizeVecsRowMaxKernel<typename TSrc::TElem, TDst>).Shmem(stateDim * sizeof(float)).Grid(lenRoundSize)
        (stateDim, len, src).Write(pDst, pDstRowScale);
}



}
