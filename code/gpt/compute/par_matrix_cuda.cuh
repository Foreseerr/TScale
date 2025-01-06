#pragma once
#include "par_matrix.h"
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/cuda_matmul.cuh>
#include <util/thread.h>

namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
class TGraph;

class TCudaModelMatrixScale : public TThrRefBase
{
    TIntrusivePtr<TModelMatrixScale> MatrixScale;
    TCudaVector<float> MatrixScaleDevice;
public:
    TCudaModelMatrixScale(TIntrusivePtr<TModelMatrixScale> pScale, TStream &stream);
    TIntrusivePtr<TModelMatrixScale> GetMatrixScale()
    {
        return MatrixScale;
    }
    void CopyToDevice(TIntrusivePtr<TGraph> c);
    TCudaPOD<float> GetElement(yint idx) const
    {
        return MatrixScaleDevice.GetElement(idx);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TPackedDeltaMatrix
{
    TCuda2DArray<i8> Data; // [y][x]
    TCuda2DArray<float> ScaleBuf; // [tile][y]

    void AllocateCuda(int xSize, int ySize, TIntrusivePtr<TCudaMemoryPool> pool)
    {
        Y_ASSERT((xSize % MODEL_INT8_DELTA_TILE) == 0);
        Data.AllocateCuda(xSize, ySize, pool);
        ScaleBuf.AllocateCuda(ySize, xSize / MODEL_INT8_DELTA_TILE, pool);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// kernels
const int COPY_DELTA_WARPS = 32;
__global__ void CopyDelta(TCuda2DPtr<float> srcArr, int xSize, int ySize, TCuda1DPtr<int> iterCounter, TCuda2DPtr<float> resTileScale, TCuda1DPtr<i8> dstArr, TCuda1DPtr<float> dstTileScale, int *launchFlag);
__global__ void CopyPackedDelta(TCuda2DPtr<i8> srcArr, TCuda2DPtr<float> srcScaleArr, int xSize, int ySize, TCuda1DPtr<int> iterCounter, TCuda1DPtr<i8> dstArr, TCuda1DPtr<float> dstTileScale, int *launchFlag);
__global__ void ClearHostMemKernel(int len, TCuda1DPtr<float> p);
__global__ void LaunchOpKernel(TCuda1DPtr<float> delta, TCuda1DPtr<float> rowScale, TCuda1DPtr<int> iterCounter, int *launchOpPtr);
__global__ void LaunchOpWaitDataKernel(TCuda2DPtr<i8> data, float *scale, TCuda1DPtr<int> iterCounter, int *launchOpPtr);
__global__ void AssignIterCounterKernel(int *hostIterCounter, TCuda1DPtr<int> iterCounter);


///////////////////////////////////////////////////////////////////////////////////////////////////
enum EModelMatrixMemory
{
    MM_MEM_HOST,
    MM_MEM_DEVICE,
};

template <class TMatrixFloat>
class TCudaModelMatrix : public TThrRefBase
{
    yint DeviceId = 0;
    EModelMatrixMemory Mem = MM_MEM_DEVICE;
    TIntrusivePtr<TParModelMatrix<TMatrixFloat>> Matrix;
    TIntrusivePtr<TCudaModelMatrixScale> CudaMatrixScale;
    TCuda2DArray<TMatrixFloat> FastDevice;
    TCuda2DArray<float> TileScaleBuf;

public:
    TCudaModelMatrix(yint deviceId, TIntrusivePtr<TCudaModelMatrixScale> pCudaMatrixScale, TIntrusivePtr<TParModelMatrix<TMatrixFloat>> pMatrix, EModelMatrixMemory mmm)
        : DeviceId(deviceId), Matrix(pMatrix), CudaMatrixScale(pCudaMatrixScale), Mem(mmm)
    {
        Y_ASSERT(Matrix->GetMatrixScale() == CudaMatrixScale->GetMatrixScale());
        if (Mem == MM_MEM_DEVICE) {
            yint xSize = pMatrix->GetXSize();
            yint ySize = pMatrix->GetYSize();
            Y_ASSERT((xSize % MM_TILE) == 0);
            yint roundYSize = DivCeil(ySize, MM_TILE) * MM_TILE;
            FastDevice.AllocateCuda(xSize, roundYSize);
            TileScaleBuf.AllocateCuda(ySize, xSize / MODEL_INT8_DELTA_TILE);
        }
    }

    void CopyToDevice(TIntrusivePtr<TGraph> c)
    {
        if (Mem == MM_MEM_HOST) {
            return;
        }
        TCuda2DArray<TMatrixFloat> &fastHost = Matrix->GetFastHost();
        // copy over PCIE bypassing CPU completely
        c->KernelCopy(&FastDevice, fastHost, fastHost.GetYSize());
        // on Windows under WDDM sometimes hangs due to some obscure buffering
        //c->CopyToDevice(&FastDevice, fastHost);
    }

    void ClearTileScale(TIntrusivePtr<TGraph> c)
    {
        TCudaVector<float> &rs = GetDeltaTileScale();
        int ySize = rs.GetSize();
        CudaCall(c, ClearHostMemKernel)(ySize).Write(&rs);
    }

    void CopyDeltaToHostAndApply(TIntrusivePtr<TGraph> c, TCuda2DArray<float> &delta, TCudaVector<int> &iterCounter)
    {
        // copy first rows, delta might have more rows due to size rounding
        TCudaVector<i8> &deltaHost = GetDelta();
        TCudaVector<float> &deltaTileScaleHost = GetDeltaTileScale();
        int xSize = Matrix->GetXSize();
        int ySize = Matrix->GetYSize();
        TCudaPOD<int> flag = Matrix->GetLaunchFlag(DeviceId);
        // add fake dependcy on iterCounter to make all copy deltas sequential
        CudaCall(c, CopyDelta).Block(WARP_SIZE, COPY_DELTA_WARPS)(delta, xSize, ySize).Write(&iterCounter, &TileScaleBuf, &deltaHost, &deltaTileScaleHost, &flag);
    }

    void CopyDeltaToHostAndApply(TIntrusivePtr<TGraph> c, TPackedDeltaMatrix &delta, TCudaVector<int> &iterCounter)
    {
        // copy first rows, delta might have more rows due to size rounding
        TCudaVector<i8> &deltaHost = GetDelta();
        TCudaVector<float> &deltaTileScaleHost = GetDeltaTileScale();
        int xSize = Matrix->GetXSize();
        int ySize = Matrix->GetYSize();
        TCudaPOD<int> flag = Matrix->GetLaunchFlag(DeviceId);
        // add fake dependcy on iterCounter to make all copy deltas sequential
        CudaCall(c, CopyPackedDelta).Block(WARP_SIZE, COPY_DELTA_WARPS)(delta.Data, delta.ScaleBuf, xSize, ySize).Write(&iterCounter, &deltaHost, &deltaTileScaleHost, &flag);
    }

    void ApplyHostDelta(TIntrusivePtr<TGraph> c, TCudaVector<int> &iterCounter)
    {
        TCudaVector<i8> &deltaHost = GetDelta(); // add dependency
        TCudaVector<float> &deltaTileScaleHost = GetDeltaTileScale();
        TCudaPOD<int> flag = Matrix->GetLaunchFlag(DeviceId);
        CudaCall(c, LaunchOpKernel)(deltaHost, deltaTileScaleHost, iterCounter).Write(&flag);
    }

    void AllowDelayedUpdates(TIntrusivePtr<TGraph> c, TCudaVector<int> &iterCounter)
    {
        TCuda2DArray<TMatrixFloat> &data = GetFast();
        TCudaPOD<float> scale = GetScale();
        TCudaPOD<int> flag = Matrix->GetAllowDelayedFlag(DeviceId);
        // add dependency, should wait all matrix reads, data type mismatch is fine, we do not use data[][] contents
        CudaCall(c, LaunchOpWaitDataKernel).Write(&data, &scale)(iterCounter).Write(&flag);
    }

    //
    TCuda2DArray<TMatrixFloat> &GetFast()
    {
        if (Mem == MM_MEM_HOST) {
            return Matrix->GetFastHost();
        } else {
            return FastDevice;
        }
    }
    TCudaPOD<float> GetScale() const { return CudaMatrixScale->GetElement(Matrix->GetMatrixScaleIndex()); }
    // direct delta manipulation
    TCudaVector<i8> &GetDelta() { return Matrix->GetDelta(DeviceId); }
    TCudaVector<float> &GetDeltaTileScale() { return Matrix->GetTileScale(DeviceId); }
};


inline void AssignIterCounter(TIntrusivePtr<TGraph> c, TCudaPOD<int> hostIterCounter, TCudaVector<int> *pIterCounter)
{
    CudaCall(c, AssignIterCounterKernel)(hostIterCounter).Write(pIterCounter);
}

}
