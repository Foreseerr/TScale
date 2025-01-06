#include "stdafx.h"
#include "cuda_init.h"
#include <cuda_runtime.h>

yint GetCudaDeviceCount()
{
    int res = 0;
    Y_VERIFY(cudaSuccess == cudaGetDeviceCount(&res));
    return res;
}
