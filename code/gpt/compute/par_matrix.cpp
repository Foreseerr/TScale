#include "stdafx.h"
#include "par_matrix.h"
#include <gpt/model_params/sse_utils.h>
#include <immintrin.h>


yint MatrixAddWorkerThreadCount = 0;

namespace NCuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////
TModelMatrixScale::TModelMatrixScale(yint sz)
{
    MatrixScaleHost.AllocateHost(sz);
}

void TModelMatrixScale::SetScale(yint index, float val)
{
    Y_ASSERT(index >= 0 && index < MatrixScaleHost.GetSize());
    MatrixScaleHost.GetHostPtr()[index] = val;
}

float TModelMatrixScale::GetScale(yint index)
{
    Y_ASSERT(index >= 0 && index < MatrixScaleHost.GetSize());
    return MatrixScaleHost.GetHostPtr()[index];
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static i8 QBitRecode2bit[256];
static i8 QBitRecode4bit[256];
static struct TInitBitTable
{
    TInitBitTable()
    {
        for (yint a = 0; a < 256; ++a) {
            i8 x = a;
            if (x < -24) {
                QBitRecode2bit[a] = -36;
            } else if (x < 0) {
                QBitRecode2bit[a] = -12;
            } else if (x <= 24) {
                QBitRecode2bit[a] = 12;
            } else {
                QBitRecode2bit[a] = 36;
            }
        }
        for (yint a = 0; a < 256; ++a) {
            i8 x = a;
            yint xx = x;
            yint bitVal = ClampVal<yint>((xx + 4 + 9 * 8) / 9, 0, 15); // 4.88512
            QBitRecode4bit[a] = bitVal * 9 + 4 - 9 * 8;
        }
    }
} initBitTable;

void ConvertToFastMatrixFloat(i8 *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant)
{
    ConvertArray(dst, src, xSize, mult);

    // simulate quantization
    if (quant == MM_QUANT_158BIT) {
        // 1.58 bit
        for (yint x = 0; x < xSize; ++x) {
            //dst[x] = (src[x] > 0) ? 32 : -32;
            if (dst[x] < -15) {
                dst[x] = -32;
            } else if (dst[x] > 15) {
                dst[x] = 32;
            } else {
                dst[x] = 0;
            }
        }
    } else if (quant == MM_QUANT_2BIT) {
        for (yint x = 0; x < xSize; ++x) {
            dst[x] = QBitRecode2bit[(ui8)dst[x]]; // can be speed up with SSE
        }
    } else if (quant == MM_QUANT_4BIT) {
        for (yint x = 0; x < xSize; ++x) {
            dst[x] = QBitRecode4bit[(ui8)dst[x]]; // can be speed up with SSE
        }
    }
}


void ConvertToFastMatrixFloat(half *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant)
{
    for (yint x = 0; x < xSize; x += 8) {
        // Load 8 floats from the input vector into a 256-bit register
        __m256 val = _mm256_mul_ps(_mm256_load_ps(src + x), mult);
        // Convert the 8 floats to 8 fp16 values and store them in a 128-bit register
        __m128i res = _mm256_cvtps_ph(val, 0);
        *(__m128i *)(dst + x) = res;
    }
    Y_VERIFY(quant == MM_QUANT_NONE);
}


void ConvertToFastMatrixFloat(e4m3 *dst, const float *src, __m256 mult, int xSize, EModelMatrixQuant quant)
{
    ConvertToFp8e4m3(dst, src, xSize, mult);
    Y_VERIFY(quant == MM_QUANT_NONE);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void TParModelMatrixBase::Create(yint deviceCount, TIntrusivePtr<TModelMatrixScale> pScale,
    float discrScale, const TModelMatrix &data,
    EModelMatrixDelayGradient delayGrad)
{
    yint xSize = data.GetXSize();
    yint ySize = data.GetYSize();
    DiscrScale = discrScale;
    Matr.SetData(data);
    SumDelta.Init(xSize, ySize);
    MatrixScale = pScale;
    MatrixScaleIndex = pScale->GetIndex();
    DelayGradient = (delayGrad == MM_DELAY_GRADIENT);
    DeviceArr.resize(deviceCount);
    for (yint deviceId = 0; deviceId < YSize(DeviceArr); ++deviceId) {
        DeviceArr[deviceId] = new TDeviceData;
        TDeviceData &dev = *DeviceArr[deviceId];
        dev.Delta.AllocateHost(xSize * ySize);
        dev.Delta.ClearHostMem();
        dev.TileScale.AllocateHost(xSize * ySize / MODEL_INT8_DELTA_TILE);
        dev.TileScale.ClearHostMem();
        dev.XSize = xSize;
    }
}


void TParModelMatrixBase::AttachOp(int *opPointer, const TVector<TCudaPOD<int>> &cudaDeltaFlags, const TVector<TCudaPOD<int>> &cudaAllowDelayedFlags)
{
    OpPointer = opPointer;
    Y_ASSERT(YSize(cudaDeltaFlags) == YSize(DeviceArr));
    for (yint deviceId = 0; deviceId < YSize(DeviceArr); ++deviceId) {
        TDeviceData &dev = *DeviceArr[deviceId];
        Y_ASSERT(dev.CudaLaunchFlag.GetOwner() == 0);
        Y_ASSERT(dev.CudaAllowDelayedFlag.GetOwner() == 0);
        dev.CudaLaunchFlag = cudaDeltaFlags[deviceId];
        dev.CudaAllowDelayedFlag = cudaAllowDelayedFlags[deviceId];
    }
}


void TParModelMatrixBase::AddToSumDelta(const TModelMatrixInt8Delta &delta)
{
    if (HasDelta) {
        Add(&SumDelta, delta);
    } else {
        Copy(&SumDelta, delta);
    }
    HasDelta = true;
}


void TParModelMatrixBase::AddDeviceToSumDelta(yint deviceId)
{
    AddToSumDelta(DeviceArr[deviceId]->GetDelta());
}


void TParModelMatrixBase::AddDelta(const TTrainingStep &step)
{
    Y_ASSERT(HasDelta);
    Y_ASSERT(*OpPointer == OP_WAIT);
    Matr.AddDelta(SumDelta, step);
    Convert();
    HasDelta = false;
    SetOp(OP_NONE);
}


void TParModelMatrixBase::AddBitDelta(const TTrainingStep &step)
{
    Y_ASSERT(*OpPointer == OP_WAIT);
    if (!Matr.AddBitDelta(BitDelta, step)) {
        SetOp(OP_NONE);
        return;
    }
    Convert();
    SetOp(OP_NONE);
}


void TParModelMatrixBase::ExtractDelta(TModelMatrixBitDelta *pBitDelta, TArray2D<float> *pDeltaTail)
{
    Matr.CompressDelta(SumDelta, pBitDelta, pDeltaTail);
    HasDelta = false;
}


void TParModelMatrixBase::GetData(TModelMatrix *p)
{
    Y_VERIFY(*OpPointer == OP_NONE);
    Matr.GetData(p);
}

void TParModelMatrixBase::SetData(const TModelMatrix &data)
{
    Y_VERIFY(*OpPointer == OP_NONE);
    Matr.SetData(data);
    Convert();
}

void TParModelMatrixBase::GetDeltaData(TModelMatrix *p) const
{
    Y_VERIFY(*OpPointer == OP_NONE);
    TArray2D<float> grad;
    SumDelta.GetAllData(&grad);
    p->SetSizes(GetXSize(), GetYSize(), GetRowDispType());
    p->SetMatrix(grad);
}

void TParModelMatrixBase::PackMatrix(TBufferedStream &f) const
{
    Y_VERIFY(*OpPointer == OP_NONE);
    Matr.PackMatrix(f);
}

void TParModelMatrixBase::AddPackedMatrixImpl(TBufferedStream &f, float scale)
{
    Y_VERIFY(*OpPointer == OP_NONE);
    Matr.AddPackedMatrix(f, scale);
    //Convert(); // should call Convert() separately to allow multiple add packed matrix with single convert
}

void TParModelMatrixBase::PerformConvert()
{
    Y_ASSERT(*OpPointer == OP_WAIT);
    Convert();
    SetOp(OP_NONE);
}

void TParModelMatrixBase::GetRowDisp(TModelRowDisp *p)
{
    Matr.GetRowDisp(p);
}

void TParModelMatrixBase::SetRowDisp(const TModelRowDisp &rd, yint *pPtr)
{
    Matr.SetRowDisp(rd, pPtr);
}

void TParModelMatrixBase::ScaleGrad(float x)
{
    Matr.ScaleGrad(x);
}


void TParModelMatrixBase::ApplyDelta(const TArray2D<float> &data)
{
    yint xSize = GetXSize();
    yint ySize = GetYSize();
    TVector<i8> arr;
    TVector<float> tileScale;
    ClearPodArray(&arr, xSize * ySize);
    ClearPodArray(&tileScale, xSize * ySize / MODEL_INT8_DELTA_TILE);
    TModelMatrixInt8Delta modelDelta(arr.data(), tileScale.data(), xSize);
    Compress(&modelDelta, data);
    AddToSumDelta(modelDelta);
    if (IsDelayGradient()) {
        SetOp(OP_DELAY_DELTA);
    } else {
        SetOp(OP_NEW_DELTA);
    }
}

void TParModelMatrixBase::AllowDelayedUpdates()
{
    // to get deterministic results we wait users to finish using current matrix contents before applying delayed delta
    if (*OpPointer == OP_APPLY_DELAYED_DELTA_WAIT) {
        *OpPointer = OP_NEW_DELTA;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void TCPUMatrixAdd::NewJob(int k, EJob op)
{
    JobCount.fetch_add(1);
    JobQueue.Add(TJob(k, op));
}


bool TCPUMatrixAdd::GenerateJobs()
{
    volatile int *matrixOpArr = MatrixOpArr.data();
    bool hasFinished = true;
    yint deviceCount = YSize(DeviceArr);
    yint matrixCount = MatrixCount;
    for (yint k = 0; k < matrixCount; ++k) {
        // perform ops until we run into waiting state
        for (bool keepProcess = true; keepProcess;) {
            int op = matrixOpArr[k];
            keepProcess = false;
            if (op == TParModelMatrixBase::OP_NONE) {
                // check if we got new delta
                for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                    TDeviceData &dev = *DeviceArr[deviceId];
                    if (dev.DeltaFlag.CheckFlag(k)) {
                        if (deviceCount > 1) {
                            if (++DeltaReadyDeviceCount[k] < deviceCount) {
                                continue;
                            }
                            DeltaReadyDeviceCount[k] = 0;
                        }
                        MatrixArr[k]->SetOp(TParModelMatrixBase::OP_WAIT);
                        NewJob(k, MJ_SUM_DEVICE_DELTA);
                        hasFinished = false;
                    }
                }
            } else if (op == TParModelMatrixBase::OP_APPLY_DELAYED_DELTA_WAIT) {
                // wait signal that matrix contents were used and delta can be applied, we are not finished
                hasFinished = false;
                for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
                    TDeviceData &dev = *DeviceArr[deviceId];
                    if (dev.AllowDelayedFlag.CheckFlag(k)) {
                        if (deviceCount > 1) {
                            if (++AllowDelayedReadyDeviceCount[k] < deviceCount) {
                                continue;
                            }
                            AllowDelayedReadyDeviceCount[k] = 0;
                        }
                        MatrixArr[k]->SetOp(TParModelMatrixBase::OP_NEW_DELTA);
                        keepProcess = true;
                    }
                }
            } else if (op == TParModelMatrixBase::OP_NEW_DELTA) {
                // start delta processing, continue processing
                if (DeltaHookArr[k].Get()) {
                    MatrixArr[k]->SetOp(TParModelMatrixBase::OP_WAIT);
                    NewJob(k, MJ_ON_DELTA_HOOK);
                    hasFinished = false;
                } else {
                    MatrixArr[k]->SetOp(TParModelMatrixBase::OP_ADD_DELTA);
                    keepProcess = true;
                }
            } else if (op == TParModelMatrixBase::OP_WAIT) {
                // we are waiting for network or something, iteration is incomplete
                hasFinished = false;
            } else if (op == TParModelMatrixBase::OP_ADD_DELTA) {
                MatrixArr[k]->SetOp(TParModelMatrixBase::OP_WAIT);
                NewJob(k, MJ_ADD_DELTA);
                hasFinished = false;
            } else if (op == TParModelMatrixBase::OP_ADD_BIT_DELTA) {
                MatrixArr[k]->SetOp(TParModelMatrixBase::OP_WAIT);
                NewJob(k, MJ_ADD_BIT_DELTA);
                hasFinished = false;
            } else if (op == TParModelMatrixBase::OP_DELAY_DELTA) {
                // do nothing, we have delta which will be applied on next iteration
            } else if (op == TParModelMatrixBase::OP_CONVERT) {
                MatrixArr[k]->SetOp(TParModelMatrixBase::OP_WAIT);
                NewJob(k, MJ_CONVERT);
                hasFinished = false;
            } else {
                Y_VERIFY(0 && "unknown add matrix state");
            }
        }
    }
    return hasFinished;
}


void TCPUMatrixAdd::Process(const TJob &job)
{
    yint deviceCount = YSize(DeviceArr);
    TIntrusivePtr<TParModelMatrixBase> matrix = MatrixArr[job.MatrixId];
    switch (job.Op) {

    case MJ_SUM_DEVICE_DELTA:
        for (yint srcDeviceId = 0; srcDeviceId < deviceCount; ++srcDeviceId) {
            matrix->AddDeviceToSumDelta(srcDeviceId);
        }
        if (AddToModel == GRADIENT_ACCUMULATE) {
            Y_ASSERT(matrix->GetOp() == TParModelMatrixBase::OP_WAIT);
            matrix->SetOp(TParModelMatrixBase::OP_NONE);
            break;
        }
        // got new delta
        if (matrix->IsDelayGradient()) {
            matrix->SetOp(TParModelMatrixBase::OP_DELAY_DELTA);
        } else {
            if (DeltaHookArr[job.MatrixId].Get()) {
                DeltaHookArr[job.MatrixId]->OnDelta();
            } else {
                matrix->AddDelta(Step);
            }
        }
        break;

    case MJ_ON_DELTA_HOOK:
        DeltaHookArr[job.MatrixId]->OnDelta();
        break;

    case MJ_ADD_DELTA:
        matrix->AddDelta(Step);
        break;

    case MJ_ADD_BIT_DELTA:
        matrix->AddBitDelta(Step);
        break;

    case MJ_CONVERT:
        matrix->PerformConvert();
        break;

    default:
        Y_VERIFY(0);
    }
    JobCount.fetch_add(-1);
}


void TCPUMatrixAdd::WorkerThread()
{
    yint workerId = WorkerCount.fetch_add(1);
    TWorkerData *data = WorkerArr[workerId].Get();
    while (!Exit) {
        TJob job;
        if (!IsIdle) {
            yint freeJobGenerator = 0;
            if (JobGeneratorFlag.compare_exchange_strong(freeJobGenerator, 1)) {
                // acquired job generator role
                for (;;) {
                    bool wasEnterIdle = EnterIdle;
                    int oldJobCount = JobCount.load();
                    bool hasFinished = GenerateJobs();
                    if (JobQueue.Get(&job)) {
                        // release job generator role, perform work
                        JobGeneratorFlag = 0;
                        Process(job);
                        break;
                    } else if (hasFinished && wasEnterIdle && oldJobCount == 0) {
                        EnterIdle = false;
                        IsIdle = true;
                        JobGeneratorFlag = 0;
                        break;
                    } else if (Exit || IsIdle) {
                        JobGeneratorFlag = 0;
                        break;
                    }
                }
            } else if (JobQueue.Get(&job)) {
                // has job to do
                Process(job);
            } else {
                _mm_pause();
            }
        } else {
            _mm_pause();
        }
    }
}


void TCPUMatrixAdd::ResetAllowDelayedFlag()
{
    for (auto &dev : DeviceArr) {
        for (yint k = 0; k < MatrixCount; ++k) {
            dev->AllowDelayedFlag.CheckFlag(k);
        }
    }
}


bool TCPUMatrixAdd::StartDelayedOps(int newOp)
{
    bool rv = false;
    for (yint k = 0; k < MatrixCount; ++k) {
        volatile int &op = MatrixOpArr[k];
        if (op == TParModelMatrixBase::OP_DELAY_DELTA) {
            op = newOp;
            rv = true;
        }
    }
    return rv;
}


TCPUMatrixAdd::~TCPUMatrixAdd()
{
    Exit = true;
}


TCPUMatrixAdd::TCPUMatrixAdd(yint deviceCount, yint maxDeltaMatrices, IMMDeltaHookGen *deltaHookGen)
    : DeltaHookGen(deltaHookGen), WorkerCount(0), JobCount(0), JobGeneratorFlag(0)
    , Exit(false), IsIdle(true), EnterIdle(false)
{
    MaxDeltaMatrices = maxDeltaMatrices;
    MatrixArr.resize(maxDeltaMatrices);
    DeltaHookArr.resize(maxDeltaMatrices);
    DeviceArr.resize(deviceCount);
    for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
        DeviceArr[deviceId] = new TDeviceData;
        TDeviceData &dev = *DeviceArr[deviceId];
        dev.DeltaFlag.Init(maxDeltaMatrices);
        dev.AllowDelayedFlag.Init(maxDeltaMatrices);
    }
    ClearPodArray(&MatrixOpArr, maxDeltaMatrices);
    ClearPodArray(&DeltaReadyDeviceCount, maxDeltaMatrices);
    ClearPodArray(&AllowDelayedReadyDeviceCount, maxDeltaMatrices);
    yint workerCount = BASE_WORKER_COUNT;
    if (MatrixAddWorkerThreadCount > 0) {
        workerCount = MatrixAddWorkerThreadCount;
    }
    WorkerArr.resize(workerCount);
    for (yint workerId = 0; workerId < workerCount; ++workerId) {
        WorkerArr[workerId] = new TWorkerData();
    }
    CudaIterCount.AllocateHost(1);
    CudaIterCount.ClearHostMem();
}


void TCPUMatrixAdd::AddMatrix(TParModelMatrixBase *p)
{
    yint idx = MatrixCount++;
    Y_VERIFY(idx < YSize(MatrixArr));
    MatrixArr[idx] = p;
    TVector<TCudaPOD<int>> cudaDeltaFlags;
    TVector<TCudaPOD<int>> cudaAllowDelayedFlags;
    for (yint deviceId = 0; deviceId < YSize(DeviceArr); ++deviceId) {
        cudaDeltaFlags.push_back(DeviceArr[deviceId]->DeltaFlag.CudaFlag.GetElement(idx));
        cudaAllowDelayedFlags.push_back(DeviceArr[deviceId]->AllowDelayedFlag.CudaFlag.GetElement(idx));
    }
    p->AttachOp(&MatrixOpArr[idx], cudaDeltaFlags, cudaAllowDelayedFlags);
    if (DeltaHookGen.Get()) {
        DeltaHookArr[idx] = DeltaHookGen->CreateDeltaHook(idx, p);
    }
}


void TCPUMatrixAdd::LaunchWorkers()
{
    // launch workers
    for (TIntrusivePtr<TWorkerData> &w : WorkerArr) {
        w->Thr.Create(this);
    }
}


void TCPUMatrixAdd::StartIteration(const TTrainingStep &step, EAddToModel addToModel)
{
    Y_VERIFY(IsIdle);
    Y_ASSERT(JobCount.load() == 0);
    CudaIterCount.GetHostPtr()[0] += 1;
    Step = step;
    AddToModel = addToModel;
    if (DeltaHookGen.Get()) {
        DeltaHookGen->OnIterationStart();
    }
    ResetAllowDelayedFlag();
    StartDelayedOps(TParModelMatrixBase::OP_APPLY_DELAYED_DELTA_WAIT);
    IsIdle = false;
}


void TCPUMatrixAdd::Wait()
{
    if (IsIdle) {
        return;
    }
    EnterIdle = true;
    // wait entering Idle state
    while (!IsIdle) {
        _mm_pause();
    }
    // wait pending jobs completion
    for (;;) {
        if (JobCount.load() == 0) {
            break;
        }
        _mm_pause();
    }
}


void TCPUMatrixAdd::WaitDelayedCompute()
{
    Wait();
    if (StartDelayedOps(TParModelMatrixBase::OP_NEW_DELTA)) {
        IsIdle = false;
        Wait();
    }
}


void TCPUMatrixAdd::ConvertMatrices()
{
    Y_VERIFY(IsIdle);
    for (yint k = 0; k < MatrixCount; ++k) {
        volatile int &op = MatrixOpArr[k];
        Y_VERIFY(op == TParModelMatrixBase::OP_NONE);
        op = TParModelMatrixBase::OP_CONVERT;
    }
    IsIdle = false;
    Wait();
}
}
