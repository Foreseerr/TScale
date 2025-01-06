#pragma once
#include <lib/cuda/cuda_arrays.h>
#include "cfg_precision.h"
#include "par_delta.h"
#include <gpt/train_config/train_step.h>
#include <util/thread.h>


namespace NCuda
{
    class TParModelMatrixBase;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
enum EAddToModel
{
    GRADIENT_ACCUMULATE,
    GRADIENT_APPLY,
};

struct IMMDeltaHook : public TThrRefBase
{
    virtual void OnDelta() = 0;
};

struct IMMDeltaHookGen : public TThrRefBase
{
    virtual IMMDeltaHook *CreateDeltaHook(yint idx, TIntrusivePtr<NCuda::TParModelMatrixBase> p) = 0;
    virtual void OnIterationStart() = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////
class TModelMatrixScale : public TThrRefBase
{
    TCudaVector<float> MatrixScaleHost;
    int IndexCount = 0;
public:
    TModelMatrixScale(yint sz);
    yint GetSize() { return MatrixScaleHost.GetSize(); }
    void SetScale(yint index, float val);
    float GetScale(yint index);
    int GetIndex()
    {
        Y_VERIFY(IndexCount < MatrixScaleHost.GetSize());
        return IndexCount++;
    }
    TCudaVector<float> &GetMatrixScaleHost()
    {
        return MatrixScaleHost;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// simulation model params quantization
enum EModelMatrixQuant
{
    MM_QUANT_NONE,
    MM_QUANT_158BIT, // -1 / 0 / 1
    MM_QUANT_2BIT,
    MM_QUANT_4BIT,
};

enum EModelMatrixDelayGradient
{
    MM_SYNC_GRADIENT,
    MM_DELAY_GRADIENT,
};


class TParModelMatrixBase : public TThrRefBase
{
    struct TDeviceData : public TThrRefBase
    {
        TCudaVector<i8> Delta;
        TCudaVector<float> TileScale;
        TCudaPOD<int> CudaLaunchFlag;
        TCudaPOD<int> CudaAllowDelayedFlag;
        yint XSize = 0;

        TDeviceData() : CudaLaunchFlag(0, 0, 0), CudaAllowDelayedFlag(0, 0, 0) {}
        TModelMatrixInt8Delta GetDelta()
        {
            TMemoryBlob blob = Delta.GetHostMem();
            return TModelMatrixInt8Delta(Delta.GetHostPtr(), TileScale.GetHostPtr(), XSize);
        }
    };

private:
    TVector<TIntrusivePtr<TDeviceData>> DeviceArr;
    TModelMatrixHalfDelta SumDelta;
    volatile bool HasDelta = false;
    TModelMatrixBitDelta BitDelta;
    TIntrusivePtr<TModelMatrixScale> MatrixScale;
    yint MatrixScaleIndex = 0;
    bool DelayGradient = false;
    volatile int *OpPointer = nullptr;

protected:
    TModelMatrixData Matr;
    float DiscrScale = 0;

protected:
    void Create(yint deviceCount, TIntrusivePtr<TModelMatrixScale> pScale,
        float discrScale, const TModelMatrix &data,
        EModelMatrixDelayGradient delayGrad);
    void SetDiscrScale(float discrScale)
    {
        MatrixScale->SetScale(MatrixScaleIndex, discrScale);
    }
    float GetMatrixScaleValue() const
    {
        return MatrixScale->GetScale(MatrixScaleIndex);
    }
    virtual void Convert() = 0;

public:
    enum {
        OP_NONE,
        OP_APPLY_DELAYED_DELTA_WAIT,
        OP_NEW_DELTA,
        OP_WAIT,
        OP_ADD_DELTA,
        OP_ADD_BIT_DELTA,
        OP_DELAY_DELTA,
        OP_CONVERT,
    };

    yint GetXSize() const { return Matr.GetXSize(); }
    yint GetYSize() const { return Matr.GetYSize(); }
    void AttachOp(int *opPointer, const TVector<TCudaPOD<int>> &cudaDeltaFlags, const TVector<TCudaPOD<int>> &cudaAllowDelayedFlags);
    int GetOp() const { return *OpPointer; }
    void SetOp(int op) { *OpPointer = op; }
    TIntrusivePtr<TModelMatrixScale> GetMatrixScale() { return MatrixScale; }
    yint GetMatrixScaleIndex() const { return MatrixScaleIndex; }
    bool IsDelayGradient() const { return DelayGradient; }
    //
    void AddToSumDelta(const TModelMatrixInt8Delta &delta);
    void AddDeviceToSumDelta(yint deviceId);
    //
    void AddDelta(const TTrainingStep &step);
    void AddBitDelta(const TTrainingStep &step);
    void ExtractDelta(TModelMatrixBitDelta *pBitDelta, TArray2D<float> *pDeltaTail);
    EModelMatrixDisp GetRowDispType() const { return Matr.GetRowDispType(); }
    //
    void GetData(TModelMatrix *p);
    void SetData(const TModelMatrix &data);
    void GetDeltaData(TModelMatrix *p) const;
    void PackMatrix(TBufferedStream &f) const;
    void AddPackedMatrixImpl(TBufferedStream &f, float scale);
    void PerformConvert();
    void GetRowDisp(TModelRowDisp *p);
    void SetRowDisp(const TModelRowDisp &rd, yint *pPtr);
    void ScaleGrad(float x);
    void ApplyDelta(const TArray2D<float> &data);
    void AllowDelayedUpdates();
    TModelMatrixBitDelta &GetBitDelta() { return BitDelta; }
    TCudaVector<i8> &GetDelta(yint deviceId) { return DeviceArr[deviceId]->Delta; }
    TCudaVector<float> &GetTileScale(yint deviceId) { return DeviceArr[deviceId]->TileScale; }
    TCudaPOD<int> GetLaunchFlag(yint deviceId) const { return DeviceArr[deviceId]->CudaLaunchFlag; }
    TCudaPOD<int> GetAllowDelayedFlag(yint deviceId) const { return DeviceArr[deviceId]->CudaAllowDelayedFlag; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
void ConvertToFastMatrixFloat(i8 *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant);
void ConvertToFastMatrixFloat(half *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant);
void ConvertToFastMatrixFloat(e4m3 *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant);

template <class TMatrixFloat>
class TParModelMatrix : public TParModelMatrixBase
{
    TCuda2DArray<TMatrixFloat> FastHost;
    EModelMatrixQuant Quantization;

    void Convert() override
    {
        yint xSize = Matr.GetXSize();
        yint ySize = Matr.GetYSize();

        float sko = sqrt(Matr.GetSum2() / (xSize * ySize));
        float discrScale = sko * DiscrScale;
        __m256 mult = _mm256_set1_ps((sko == 0) ? 0 : (1 / discrScale));

        SetDiscrScale(sko);

        TMemoryBlob fastMem = FastHost.GetHostMem();
        for (yint y = 0; y < ySize; ++y) {
            TMatrixFloat *dst = fastMem.GetElementAddress<TMatrixFloat>(0, y);
            const float *src = Matr.GetRow(y);
            ConvertToFastMatrixFloat(dst, src, mult, xSize, Quantization);
        }
    }

public:
    void Create(yint deviceCount, TIntrusivePtr<TModelMatrixScale> pScale,
        float discrScale, const TModelMatrix &data,
        EModelMatrixQuant quant, EModelMatrixDelayGradient delayGrad)
    {
        yint xSize = data.GetXSize();
        yint ySize = data.GetYSize();
        TParModelMatrixBase::Create(deviceCount, pScale, discrScale, data, delayGrad);
        FastHost.AllocateHost(xSize, ySize);
        Quantization = quant;
        Convert();
    }

    void GetFastFloatData(TArray2D<float> *p) const
    {
        yint xSize = Matr.GetXSize();
        yint ySize = Matr.GetYSize();
        THost2DPtr<TMatrixFloat> src = FastHost.GetHostPtr();
        float scale = GetMatrixScaleValue() * DiscrScale;
        p->SetSizes(xSize, ySize);
        for (yint y = 0; y < ySize; ++y) {
            for (yint x = 0; x < xSize; ++x) {
                (*p)[y][x] = float(src[y][x]) * scale;
            }
        }
    }

    TCuda2DArray<TMatrixFloat> &GetFastHost()
    {
        return FastHost;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCPUMatrixAdd : public TThrRefBase
{
    enum {
        BASE_WORKER_COUNT = 4,
    };

    enum EJob
    {
        MJ_SUM_DEVICE_DELTA,
        MJ_ON_DELTA_HOOK,
        MJ_ADD_DELTA,
        MJ_ADD_BIT_DELTA,
        MJ_CONVERT,
        MJ_NONE,
    };

    struct TJob
    {
        int MatrixId = -1;
        EJob Op = MJ_NONE;

        TJob() {}
        TJob(int id, EJob op) : MatrixId(id), Op(op) {}
    };

    struct TCudaLaunchFlags
    {
        TCudaVector<int> CudaFlag;
        TVector<int> PrevCudaFlag;

        void Init(yint sz)
        {
            CudaFlag.AllocateHost(sz);
            CudaFlag.ClearHostMem();
            ClearPodArray(&PrevCudaFlag, sz);
        }
        bool CheckFlag(yint k)
        {
            volatile int *cudaBuf = CudaFlag.GetHostPtr();
            int newCudaFlag = cudaBuf[k];
            if (newCudaFlag != PrevCudaFlag[k]) {
                // avoid modifying cudaAddDeltaFlag from cpu & gpu concurrently
                PrevCudaFlag[k] = newCudaFlag;
                return true;
            }
            return false;
        }
    };


    struct TDeviceData : public TThrRefBase
    {
        TCudaLaunchFlags DeltaFlag;
        TCudaLaunchFlags AllowDelayedFlag;
    };


    struct TWorkerData : public TThrRefBase
    {
        TThread Thr;
    };

private:
    yint MaxDeltaMatrices = 0;
    TVector<TIntrusivePtr<TDeviceData>> DeviceArr;
    TVector<int> MatrixOpArr;
    TVector<int> DeltaReadyDeviceCount;
    TVector<int> AllowDelayedReadyDeviceCount;
    TVector<TIntrusivePtr<TParModelMatrixBase>> MatrixArr;
    TIntrusivePtr<IMMDeltaHookGen> DeltaHookGen;
    TVector<TIntrusivePtr<IMMDeltaHook>> DeltaHookArr;
    TVector<TIntrusivePtr<TWorkerData>> WorkerArr;
    TSingleProducerJobCircleBuffer<TJob, 8192> JobQueue;
    std::atomic<yint> WorkerCount;
    std::atomic<yint> JobCount;
    std::atomic<yint> JobGeneratorFlag;
    std::atomic<bool> Exit;
    std::atomic<bool> IsIdle;
    std::atomic<bool> EnterIdle;
    yint MatrixCount = 0;
    TTrainingStep Step;
    EAddToModel AddToModel = GRADIENT_APPLY;
    TCudaVector<int> CudaIterCount;

    void NewJob(int k, EJob op);
    bool GenerateJobs();
    void Process(const TJob &job);
    bool StartDelayedOps(int newOp);
    void ResetAllowDelayedFlag();
    ~TCPUMatrixAdd();

public:
    TCPUMatrixAdd(yint deviceCount, yint maxDeltaMatrices, IMMDeltaHookGen *deltaHookGen);
    void AddMatrix(TParModelMatrixBase *p);
    void LaunchWorkers();
    void StartIteration(const TTrainingStep &step, EAddToModel addToModel); // assume no pending ops at this moment
    void Wait();
    void WaitDelayedCompute();
    void ConvertMatrices();
    NCuda::TCudaPOD<int> GetCurrentIteration() { return CudaIterCount.GetElement(0); }
    yint GetDeviceCount() const { return YSize(DeviceArr); }

public:
    void WorkerThread();
};


template <class TMatrixFloat>
TIntrusivePtr<TParModelMatrix<TMatrixFloat>> CreateModelMatrix(TIntrusivePtr<TCPUMatrixAdd> cpuAdd, TIntrusivePtr<TModelMatrixScale> pScale,
    float discrScale, const TModelMatrix &data,
    EModelMatrixQuant quant, EModelMatrixDelayGradient delayGrad)
{
    TIntrusivePtr<TParModelMatrix<TMatrixFloat>> res = new TParModelMatrix<TMatrixFloat>();
    res->Create(cpuAdd->GetDeviceCount(), pScale, discrScale, data, quant, delayGrad);
    cpuAdd->AddMatrix(res.Get());
    return res;

}

}
