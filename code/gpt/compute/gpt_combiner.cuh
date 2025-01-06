#pragma once


///////////////////////////////////////////////////////////////////////////////////////////////////
template <int WSZ>
__device__ void KVProductImpl(int blk, const float *key, const float *value, i8 *dst)
{
    int h = threadIdx.x;
}

template <class TDst>
__global__ void KVProduct(
    TCuda2DPtr<half> kVecArr, TCuda2DPtr<half> vVecArr,
    float kvMult,
    TCuda2DPtr<TDst> kvVecArr
)
{
    int h = threadIdx.x;
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;

    float key[WSZ];
    LoadWarpVec<WSZ>(key, kVecArr[t] + offset);
    ScaleWarpVec<WSZ>(key, K_VEC_SCALE);
    float value[WSZ];
    LoadWarpVec<WSZ>(value, vVecArr[t] + offset);
    ScaleWarpVec<WSZ>(value, V_VEC_SCALE);
    for (int blk = 0; blk < COMBINER_REP; ++blk) {
        TDst *dst = kvVecArr[t] + tile * COMBINER_REP * MM_TILE + blk * MM_TILE;
        for (int k = 0; k < WSZ; ++k) {
            int d = k * WARP_SIZE + h;
            float keyShfl = __shfl_xor_sync(0xffffffff, key[k], blk);
            float val = keyShfl * value[k];
            StoreConvertedFloat(val * kvMult, &dst[d]);
        }
    }
}


//constexpr int KV_PRODUCT_TRANSP_BLOCK_SIZE = 8;
//template <int STATE_DIM>
//__global__ void KVProductShuffleTranspose(
//    int len, TSortNode *sortNode,
//    TCuda2DPtr<i8> kVecArr, TCuda2DPtr<i8> vVecArr,
//    TCuda2DPtr<i8> kvVecTArr
//)
//{
//    // .Grid(STATE_DIM / 32, len / 128)
//    int h = threadIdx.x;
//    int warpId = threadIdx.y;
//    constexpr int SAMPLE_PER_WARP = 128 / KV_PRODUCT_TRANSP_BLOCK_SIZE;
//    int blockDimBase = blockIdx.x * WARP_SIZE;
//    int blockTimeBase = blockIdx.y * 128;
//    int thrTimeBase = warpId * SAMPLE_PER_WARP;
//
//    float key[SAMPLE_PER_WARP];
//    float value[SAMPLE_PER_WARP];
//    for (int k = 0; k < SAMPLE_PER_WARP; ++k) {
//        int t = blockTimeBase + thrTimeBase + k;
//        if (t < len) {
//            int nodeId = sortNode[t].NodeId;
//            key[k] = kVecArr[nodeId][blockDimBase + h];
//            value[k] = vVecArr[nodeId][blockDimBase + h];
//        } else {
//            key[k] = 0;
//            value[k] = 0;
//        }
//    }
//
//    __shared__ i8 buf[32][128 + 4]; // avoid smem bank conflict
//    for (int blk = 0; blk < COMBINER_REP; ++blk) {
//        for (int k = 0; k < SAMPLE_PER_WARP; ++k) {
//            float keyShfl = __shfl_xor_sync(0xffffffff, key[k], blk);
//            buf[h][thrTimeBase + k] = CvtToI8(keyShfl * value[k] * (VEC_SCALE * VEC_SCALE / VEC_SCALE));
//        }
//        __syncthreads();
//        for (int y = 0; y < 32; y += KV_PRODUCT_TRANSP_BLOCK_SIZE) {
//            int *src = (int *)&buf[y + warpId][h * 4];
//            int *dst = (int *)&kvVecTArr[blockDimBase + blk * STATE_DIM + y + warpId][blockTimeBase + h * 4];
//            *dst = *src;
//        }
//        __syncthreads();
//    }
//}
//KERNEL_BLOCK_SIZE(KVProductShuffleTranspose, WARP_SIZE, KV_PRODUCT_TRANSP_BLOCK_SIZE);


template <int WSZ>
inline __device__ void KVProductBackpropImpl(float *key, float *value, half *dkvVec, float *dKey, float *dValue)
{
    LoadZeroWarpVec<WSZ>(dKey);
    LoadZeroWarpVec<WSZ>(dValue);
    for (int blk = 0; blk < COMBINER_REP; ++blk) {
        float dkv[WSZ];
        LoadWarpVec<WSZ>(dkv, dkvVec + blk * MM_TILE);
        // forward: kv = shfl(key) * value
        for (int k = 0; k < WSZ; ++k) {
            float keyShfl = __shfl_xor_sync(0xffffffff, key[k], blk);
            float valueShfl = __shfl_xor_sync(0xffffffff, value[k], blk);
            float grad = dkv[k];
            float gradShfl = __shfl_xor_sync(0xffffffff, dkv[k], blk);
            dKey[k] += gradShfl * valueShfl;
            dValue[k] += grad * keyShfl;
        }
    }

}

__global__ void KVProductBackprop(
    int len,
    TCuda2DPtr<half> kVecArr, TCuda2DPtr<half> vVecArr, TCuda2DPtr<half> dkvVecArr,
    float attGradMult,
    TCuda2DPtr<half> dkVecArr, TCuda2DPtr<half> dvVecArr
)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;

    if (t >= len) {
        float zero[WSZ];
        LoadZeroWarpVec<WSZ>(zero);
        StoreWarpVec<WSZ>(dkVecArr[t] + offset, zero);
        StoreWarpVec<WSZ>(dvVecArr[t] + offset, zero);
        return;
    }

    float key[WSZ];
    LoadWarpVec<WSZ>(key, kVecArr[t] + offset);
    ScaleWarpVec<WSZ>(key, K_VEC_SCALE);
    float value[WSZ];
    LoadWarpVec<WSZ>(value, vVecArr[t] + offset);
    ScaleWarpVec<WSZ>(value, V_VEC_SCALE);
    float dKey[WSZ];
    float dValue[WSZ];
    KVProductBackpropImpl<WSZ>(key, value, dkvVecArr[t] + tile * COMBINER_REP * MM_TILE, dKey, dValue);
    ScaleWarpVec<WSZ>(dKey, attGradMult);
    ScaleWarpVec<WSZ>(dValue, attGradMult);
    StoreWarpVec<WSZ>(dkVecArr[t] + offset, dKey);
    StoreWarpVec<WSZ>(dvVecArr[t] + offset, dValue);
}


// KVproduct backprop + normalize(K) backprop + dv convert rowtile max
template <class TVGradFloat>
__global__ void KVProductBackpropE4fused(
    int len,
    TCuda2DPtr<half> kVecArr, TCuda2DPtr<float> kTileScale, TCuda2DPtr<half> vVecArr, TCuda2DPtr<half> dkvVecArr,
    float attGradMult,
    TCuda2DPtr<half> dkVecArr, TCuda2DPtr<TVGradFloat> dvVecArr, TCuda2DPtr<float> dvScaleArr, TCuda2DPtr<float> dScaleArr
)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;

    if (t >= len) {
        float zero[WSZ];
        LoadZeroWarpVec<WSZ>(zero);
        StoreWarpVec<WSZ>(dkVecArr[t] + offset, zero);
        StoreWarpVec<WSZ>(dvVecArr[t] + offset, zero);
        if (threadIdx.x == 0) {
            dScaleArr[tile][t] = 0;
            dvScaleArr[tile][t] = 0;
        }
        return;
    }

    float key[WSZ];
    LoadWarpVec<WSZ>(key, kVecArr[t] + offset);
    ScaleWarpVec<WSZ>(key, K_VEC_SCALE);
    float value[WSZ];
    LoadWarpVec<WSZ>(value, vVecArr[t] + offset);
    ScaleWarpVec<WSZ>(value, V_VEC_SCALE);
    float dKey[WSZ];
    float dValue[WSZ];
    KVProductBackpropImpl<WSZ>(key, value, dkvVecArr[t] + tile * COMBINER_REP * MM_TILE, dKey, dValue);
    ScaleWarpVec<WSZ>(dKey, attGradMult);

    // compute DScale
    float dScale = 0;
    for (int k = 0; k < WSZ; ++k) {
        dScale += value[k] * dValue[k];
    }
    dScale = WarpSum(dScale);
    if (threadIdx.x == 0) {
        dScaleArr[tile][t] = dScale;
    }

    // backprop dKey
    LoadWarpVec<WSZ>(key, kVecArr[t] + offset);
    ScaleWarpVec<WSZ>(key, kTileScale[tile][t]);
    float dKeyResult[WSZ];
    StateNormalizeBackpropWarpVec<WSZ>(key, dKey, dKeyResult);
    StoreWarpVec<WSZ>(dkVecArr[t] + offset, dKeyResult);

    // dValue row tile max normalize
    float discrScale = 0;
    {
        float maxVal = 0;
        for (int k = 0; k < WSZ; ++k) {
            maxVal = max(maxVal, fabs(dValue[k]));
        }
        maxVal = WarpMax(maxVal);
        if (maxVal > 0) {
            discrScale = GetMaxDiscrScale(maxVal, (TVGradFloat *)0);
            ScaleWarpVec<WSZ>(dValue, 1 / discrScale);
        }
    }
    StoreWarpVec<WSZ>(dvVecArr[t] + offset, dValue);
    if (threadIdx.x == 0) {
        dvScaleArr[tile][t] = discrScale;
    }
}
