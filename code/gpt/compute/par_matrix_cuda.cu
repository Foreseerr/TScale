#include "stdafx.h"
#define KERNEL_UNIT "par_matrix_cuda/"
#include "par_matrix_cuda.cuh"
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/cuda_matmul.cuh>


namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////
TCudaModelMatrixScale::TCudaModelMatrixScale(TIntrusivePtr<TModelMatrixScale> pScale, TStream &stream) : MatrixScale(pScale)
{
    MatrixScaleDevice.AllocateCuda(pScale->GetSize());
}

void TCudaModelMatrixScale::CopyToDevice(TIntrusivePtr<TGraph> c)
{
    c->KernelCopy(&MatrixScaleDevice, MatrixScale->GetMatrixScaleHost());
}



///////////////////////////////////////////////////////////////////////////////////////////////////
//
__device__ void CopyTileScale(int xSize, int ySize, TCuda2DPtr<float> srcScaleArr, TCuda1DPtr<float> dstTileScale)
{
    int h = threadIdx.x;
    int warpId = threadIdx.y;

    int tilesPerRow = xSize / MODEL_INT8_DELTA_TILE;
    int totalTiles = tilesPerRow * ySize;
    int threadTile = h + warpId * WARP_SIZE;
    int curTile = threadTile % tilesPerRow;
    int curY = threadTile / tilesPerRow;
    int deltaTile = (COPY_DELTA_WARPS * WARP_SIZE) % tilesPerRow;
    int deltaY = (COPY_DELTA_WARPS * WARP_SIZE) / tilesPerRow;
    for (int baseTile = 0; baseTile < totalTiles; baseTile += COPY_DELTA_WARPS * WARP_SIZE) {
        if (curY < ySize) {
            dstTileScale[baseTile + threadTile] = srcScaleArr[curTile][curY];
        }
        curTile += deltaTile;
        curY += deltaY;
        if (curTile >= tilesPerRow) {
            curTile -= tilesPerRow;
            ++curY;
        }
    }
}


__global__ void CopyDelta(TCuda2DPtr<float> srcArr, int xSize, int ySize, TCuda1DPtr<int> iterCounter, TCuda2DPtr<float> tileScaleBuf, TCuda1DPtr<i8> dstArr, TCuda1DPtr<float> dstTileScale, int *launchFlag)
{
    CUDA_STATIC_ASSERT(MODEL_INT8_DELTA_TILE == 128);
    CUDA_ASSERT((xSize % 128) == 0);

    int h = threadIdx.x;
    int warpId = threadIdx.y;

    constexpr int LINE = 512;

    __shared__ i8 resPacked[COPY_DELTA_WARPS][LINE];
    __shared__ float resScale[COPY_DELTA_WARPS][LINE / 128];

    for (int offsetY = 0; offsetY < ySize; offsetY += COPY_DELTA_WARPS) {
        int y = offsetY + warpId;
        if (y >= ySize) {
            break;
        }
        int dstRowOffset = y * xSize;
        i8 *dstRowPtr = &dstArr[dstRowOffset];
        for (int offsetX = 0; offsetX < xSize; offsetX += LINE) {
            // each warp packs LINE elements in 128 width blocks
            // read and pack data
            for (int blk = 0; blk < LINE / 128; ++blk) {
                int x = offsetX + blk * 128 + h * 4;
                float maxVal = 0;
                float4 val4;
                if (x < xSize) {
                    val4 = *(float4 *)(srcArr[y] + x);
                    maxVal = max(maxVal, max(max(fabsf(val4.x), fabsf(val4.y)), max(fabsf(val4.z), fabsf(val4.w))));
                }
                maxVal = WarpMax(maxVal);

                float scale = (maxVal > 0) ? maxVal / 127 : 0;
                float mult = (maxVal > 0) ? 1 / scale : 0;
                resScale[warpId][blk] = scale;

                // convert
                if (x < xSize) {
                    union {
                        int res4;
                        i8 res[4];
                    };
                    res[0] = CvtToI8(val4.x * mult);
                    res[1] = CvtToI8(val4.y * mult);
                    res[2] = CvtToI8(val4.z * mult);
                    res[3] = CvtToI8(val4.w * mult);
                    *(int *)(&resPacked[warpId][blk * 128 + h * 4]) = res4;
                }
            }
            __syncwarp();
            // write packed data
            int thrOffset = h * 16;
            int4 packedData = *(int4 *)&resPacked[warpId][thrOffset];
            int writeX = offsetX + thrOffset;
            if (writeX < xSize) {
                *(int4 *)&dstRowPtr[writeX] = packedData;
            }

            // write scale
            int warpWidth = xSize - offsetX;
            if (h < warpWidth / 128) {
                tileScaleBuf[(offsetX / 128) + h][y] = resScale[warpId][h];
            }
            __syncwarp();
        }
    }
    __syncthreads();

    // copy scale
    CopyTileScale(xSize, ySize, tileScaleBuf, dstTileScale);
    // apply
    __threadfence_system(); // flush cache
    __syncthreads();
    if (h == 0 && warpId == 0) {
        *launchFlag = iterCounter[0];
    }
    __threadfence_system(); // neccessary, does not happen on kernel finish
}


__global__ void CopyPackedDelta(TCuda2DPtr<i8> srcArr, TCuda2DPtr<float> srcScaleArr, int xSize, int ySize, TCuda1DPtr<int> iterCounter, TCuda1DPtr<i8> dstArr, TCuda1DPtr<float> dstTileScale, int *launchFlag)
{
    CUDA_STATIC_ASSERT(MODEL_INT8_DELTA_TILE == 128);
    CUDA_STATIC_ASSERT(MODEL_INT8_DELTA_TILE == MM_TILE);
    CUDA_ASSERT((xSize % 128) == 0);

    int h = threadIdx.x;
    int warpId = threadIdx.y;

    // copy data
    for (int offsetY = 0; offsetY < ySize; offsetY += COPY_DELTA_WARPS) {
        int y = offsetY + warpId;
        if (y >= ySize) {
            break;
        }
        int dstRowOffset = y * xSize;
        i8 *dstRowPtr = &dstArr[dstRowOffset];
        for (int base = 0; base < xSize; base += 16 * WARP_SIZE) {
            int x = base + h * 16;
            int4 data;
            if (x < xSize) {
                data = *(int4 *)&srcArr[y][x];
            }
            __syncwarp();
            if (x < xSize) {
                *(int4 *)&dstRowPtr[x] = data;
            }
        }
    }
    // copy scale
    CopyTileScale(xSize, ySize, srcScaleArr, dstTileScale);
    // apply
    __threadfence_system(); // flush cache
    __syncthreads();
    if (h == 0 && warpId == 0) {
        *launchFlag = iterCounter[0];
    }
    __threadfence_system(); // neccessary, does not happen on kernel finish
}


__global__ void ClearHostMemKernel(int len, TCuda1DPtr<float> p)
{
    float4 zero = make_float4(0, 0, 0, 0);
    for (int base = 0; base < len; base += WARP_SIZE * 4) {
        int x = base + threadIdx.x * 4;
        if (x + 3 < len) {
            float4 *dst = (float4 *)&p[x];
            *dst = zero;
        } else if (x < len) {
            // clean tail
            for (int k = x; k < len; ++k) {
                p[k] = 0;
            }
        }
    }
    __threadfence_system(); // flush cache
}


__global__ void LaunchOpKernel(TCuda1DPtr<float> delta, TCuda1DPtr<float> rowScale, TCuda1DPtr<int> iterCounter, int *launchOpPtr)
{
    (void)delta; // needed for dependency
    (void)rowScale; // needed for dependency
    if (threadIdx.x == 0) {
        *launchOpPtr = iterCounter[0];
    }
    __threadfence_system(); // flush cache asap
}


__global__ void LaunchOpWaitDataKernel(TCuda2DPtr<i8> data, float *scale, TCuda1DPtr<int> iterCounter, int *launchOpPtr)
{
    (void)data; // needed for dependency
    (void)scale;
    if (threadIdx.x == 0) {
        *launchOpPtr = iterCounter[0];
    }
    __threadfence_system(); // flush cache asap
}


__global__ void AssignIterCounterKernel(int *hostIterCounter, TCuda1DPtr<int> iterCounter)
{
    if (threadIdx.x == 0) {
        iterCounter[0] = *hostIterCounter;
    }
}

}
