#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
// final layer kernels

// compute rowtile logusm
struct TStoreFinalLayerLogits
{
    struct TParams
    {
        float *ScalePtr;
        float NormScale;
        int VocabSize;
        TCuda1DPtr<float> BiasArr;
        TCuda2DPtr<float> RowTileLogSum;
    };
    struct TShmem
    {
        float ScaledRes[2][TILE][MM_TILE];
        float Bias[MM_TILE];
    };

    template <class TRes, class T>
    __device__ static void Store(TParams &params, TShmem &shmem, const TTileCoord &tc, TMatMulWarpResult<T> &mmRes, float resultScale, int offsetX, int offsetY, TCuda2DPtr<TRes> resBuf)
    {
        CUDA_STATIC_ASSERT(MM_TILE == 128);
        int mmTileX = blockIdx.x * MM_TILE;
        int mmTileY = blockIdx.y * MM_TILE;
        int h = threadIdx.x;
        int warpId = threadIdx.y;
        float sumScale = resultScale * params.NormScale;
        if (params.ScalePtr) {
            sumScale *= *params.ScalePtr;
        }
        int vocabSize = params.VocabSize;

        // load bias
        __syncthreads();
        {
            int cc = warpId * WARP_SIZE + h;
            shmem.Bias[cc] = (mmTileX + cc < vocabSize) ? params.BiasArr[mmTileX + cc] : 0;
        }

        // process 2 rows at at time
        for (int ty = 0; ty < 4; ++ty) {
            __syncthreads();
            for (int tx = 0; tx < 4; ++tx) {
                mmRes.Sum[ty][tx].StoreScaled(tc, TCuda2DPtr<float>(&shmem.ScaledRes[offsetY / 64][0][offsetX + tx * TILE], MM_TILE * sizeof(float), 16, 16), sumScale);
            }
            __syncthreads();
            for (int base = 0; base < TILE; base += 2) {
                int rowId = base + (offsetX / 64);
                constexpr int WSZ = MM_TILE / WARP_SIZE;
                float vec[WSZ];
                LoadWarpVec<WSZ>(vec, shmem.ScaledRes[offsetY / 64][rowId]);
                LoadAddWarpVec<WSZ>(vec, shmem.Bias);
                // find max
                float maxVal = vec[0];
                for (int k = 1; k < WSZ; ++k) {
                    maxVal = max(maxVal, vec[k]);
                }
                maxVal = WarpMax(maxVal);
                // sum
                float sum = 0;
                for (int k = 0; k < WSZ; ++k) {
                    sum += exp2f(vec[k] - maxVal);
                }
                sum = WarpSum(sum);
                float logSum = log2f(sum) + maxVal;
                if (h == 0) {
                    params.RowTileLogSum[blockIdx.x][mmTileY + offsetY + ty * TILE + rowId] = logSum;
                }
                // store relative to rowtile max to preserve precision
                for (int k = 0; k < WSZ; ++k) {
                    vec[k] -= logSum;
                }
                StoreWarpVec<WSZ>(resBuf[mmTileY + offsetY + ty * TILE + rowId] + mmTileX, vec);
            }
        }
    }
};


__global__ void SumLogWeight(int tileCount, TCuda2DPtr<float> rowTileLogSum)
{
    int tile = blockIdx.x;
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;

    // find max
    float maxVal[WSZ];
    LoadWarpVec<WSZ>(maxVal, rowTileLogSum[0] + offset);
    for (int tile = 1; tile < tileCount; ++tile) {
        float vec[WSZ];
        LoadWarpVec<WSZ>(vec, rowTileLogSum[tile] + offset);
        for (int k = 0; k < WSZ; ++k) {
            maxVal[k] = max(maxVal[k], vec[k]);
        }
    }
    // calc sum
    float sum[WSZ];
    LoadZeroWarpVec<WSZ>(sum);
    for (int tile = 0; tile < tileCount; ++tile) {
        float vec[WSZ];
        LoadWarpVec<WSZ>(vec, rowTileLogSum[tile] + offset);
        for (int k = 0; k < WSZ; ++k) {
            sum[k] += exp2f(vec[k] - maxVal[k]);
        }
    }
    float logSum[WSZ];
    for (int k = 0; k < WSZ; ++k) {
        logSum[k] = maxVal[k] + log2f(sum[k]);
    }
    // subtract sum
    for (int tile = 0; tile < tileCount; ++tile) {
        float vec[WSZ];
        LoadWarpVec<WSZ>(vec, rowTileLogSum[tile] + offset);
        for (int k = 0; k < WSZ; ++k) {
            vec[k] -= logSum[k];
        }
        StoreWarpVec<WSZ>(rowTileLogSum[tile] + offset, vec);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComputeFinalProbKernel(int targetOffset, int vocabSize, TCuda2DPtr<float> logitRowTileLogSum, TCuda1DPtr<int> targetArr, TCuda2DPtr<half> logitBuf, TCuda1DPtr<float> resTargetProb)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    int h = threadIdx.x;;
    int cc = targetArr[targetOffset + t];

    float rowTileLogSum = logitRowTileLogSum[tile][t];
    for (int base = 0; base < MM_TILE; base += WARP_SIZE) {
        int c = offset + base + h;
        if (c < vocabSize) {
            float pred = exp2f(float(logitBuf[t][c]) + rowTileLogSum);
            logitBuf[t][c] = pred;
            if (c == cc) {
                resTargetProb[targetOffset + t] = pred;
            }
        } else {
            logitBuf[t][c] = 0;
        }
    }
}


__global__ void ComputeGradient(
    int len,
    int targetOffset, int vocabSize, int vocabRoundSize,
    TCuda2DPtr<half> logitBuf, TCuda2DPtr<float> logitRowTileLogSum, TCuda1DPtr<int> targetArr,
    TCuda2DPtr<half> gradArr, TCuda1DPtr<float> sumTrainErr
)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    int h = threadIdx.x;
    int cc = -1;
    if (t < len) {
        cc = targetArr[targetOffset + t];
    }
    if (cc >= 0) {
        float rowTileLogSum = logitRowTileLogSum[tile][t];
        for (int base = 0; base < MM_TILE; base += WARP_SIZE) {
            int c = offset + base + h;
            if (c < vocabSize) {
                float pred = exp2f(float(logitBuf[t][c]) + rowTileLogSum);
                pred = fmaxf(pred, 1e-20f); // avoid nans
                // omit scale gradient by log2, constant scale does not change anything
                if (c == cc) {
                    gradArr[t][c] = (1 - pred);
                    atomicAdd(&sumTrainErr[0], 1);
                    atomicAdd(&sumTrainErr[1], log(pred));
                } else {
                    gradArr[t][c] = -pred;
                }
            } else {
                gradArr[t][c] = 0;
            }
        }
    } else {
        for (int base = 0; base < MM_TILE; base += WARP_SIZE) {
            int c = offset + base + h;
            gradArr[t][c] = 0;
        }
    }
}


__global__ void CollectSumTrainErr(TCuda1DPtr<float> sumTrainErr)
{
    if (threadIdx.x == 0) {
        sumTrainErr[2] += sumTrainErr[0];
        sumTrainErr[3] += sumTrainErr[1];
        sumTrainErr[0] = 0;
        sumTrainErr[1] = 0;
    }
}


__global__ void ComputeLossKernel(int offset, int len, TCuda2DPtr<half> logitBuf, TCuda2DPtr<float> logitRowTileLogSum, TCuda1DPtr<int> targetArr, TCuda1DPtr<float> resArr)
{
    int h = threadIdx.x;
    float sum = 0;
    for (int base = 0; base < len; base += WARP_SIZE) {
        int t = base + h;
        if (t < len) {
            int target = targetArr[offset + t];
            if (target >= 0) {
                sum += (float(logitBuf[t][target]) + logitRowTileLogSum[target / MM_TILE][t]) * LOG2;
            }
        }
    }
    sum = WarpSum(sum);
    if (h == 0) {
        resArr[0] += sum;
    }
}
