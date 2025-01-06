#include "stdafx.h"
#include "net_train.h"
#include "network.h"
#include <gpt/train_ctx/train_ctx.h>
#include <gpt/train_ctx/backprop.h>
#include <gpt/compute/gpt_cuda.cuh>
#include <gpt/compute/par_matrix.h>
#include <gpt/att/sliding_window.h>
#include <lib/hp_timer/hp_timer.h>
#include <lib/net/ip_address.h>
#include <lib/net/tcp_cmds.h>
#include <typeinfo>
#include <emmintrin.h>


using NCuda::TParModelMatrixBase;

using namespace NNet;

namespace NNetTrain
{

static TGuid NetTrainToken(0xbadf00d, 0x31337, 0xceedb0c0, 0x31415926);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TNetTrainContext;
struct TCommandPacket : public TCommandBase
{
    virtual void Exec(TNetTrainContext *pCtx) {}
    virtual yint GetP2PIteration() { return -1; }
};

typedef TMasterNetTempl<TCommandPacket> TMasterNet;

static TCommandFabric<TCommandPacket> cmdFabric;

enum ECommandResult
{
    CMD_OK,
};


///////////////////////////////////////////////////////////////////////////////////////////////////
//
class TMMNetDeltaReduceGen;
struct TNetTrainContext
{
    TIntrusivePtr<ITcpSendRecv> Net;
    TMasterConnection Master;
    TIntrusivePtr<TP2PNetwork> P2PNet;
    TIntrusivePtr<IModel> Model;
    TIntrusivePtr<IComputeContext> Ctx;
    TIntrusivePtr<TMMNetDeltaReduceGen> NetDeltaReduce;
    TVector<ui8> ModelSnapshot;
    TThread P2PThread;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCalcModelError : public TCommandPacket
{
    TVector<TFragment> FragArr;
    SAVELOAD_OVERRIDE(FragArr);
public:
    TCalcModelError() {}
    TCalcModelError(const TVector<TFragment> &fragArr) : FragArr(fragArr) {}
    void Exec(TNetTrainContext *p) override
    {
        double res = CalcModelErr(FragArr, p->Ctx.Get());
        p->Master.Send(res);
    }
};
REGISTER_PACKET(cmdFabric, TCalcModelError, 1);

static double DistributedCalcModelErr(TMasterNet &masterNet, const TVector<TVector<TFragment>> &batches)
{
    if (batches.empty()) {
        return 0;
    }
    yint batchCount = YSize(batches);
    auto it = masterNet.WorkerSet.begin();
    for (yint b = 0; b < batchCount; ++b) {
        masterNet.SendCommand(it->first, new TCalcModelError(batches[b]));
        if (++it == masterNet.WorkerSet.end()) {
            it = masterNet.WorkerSet.begin();
        }
    }
    // collect results
    double sum = 0;
    yint confirmCount = 0;
    while (confirmCount < batchCount) {
        TIntrusivePtr<TTcpPacketReceived> pkt;
        if (masterNet.Queue->Dequeue(&pkt)) {
            double score = 0;
            SerializeMem(IO_READ, &pkt->Data, score);
            sum += score;
            ++confirmCount;
        }
    }
    return sum / batchCount;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TDeltaMatrix : public TCommandPacket
{
    yint P2PIteration = 0;
    yint MatrixId = 0;
    yint SumLevel = 0;
    TModelMatrixBitDelta BitDelta;
    SAVELOAD_OVERRIDE(P2PIteration, MatrixId, SumLevel, BitDelta);
public:
    TDeltaMatrix() {}
    TDeltaMatrix(yint p2pIteration, yint matrixId, yint sumLevel, const TModelMatrixBitDelta &bitDelta)
        : P2PIteration(p2pIteration), MatrixId(matrixId), SumLevel(sumLevel), BitDelta(bitDelta)
    {
    }
    void Exec(TNetTrainContext *p) override;
    yint GetP2PIteration() override
    {
        return P2PIteration;
    }
};
REGISTER_PACKET(cmdFabric, TDeltaMatrix, 2);


// OnDelta() and AddRemoteDelta() are called from different threads
class TMMNetDeltaReduce : public IMMDeltaHook
{
    const static ui64 LOCAL_DATA = 0x8000; // debug is easier if we know which data is ready

    struct TReduceLevel : public TThrRefBase
    {
        std::atomic<yint> ReadyCount;
        TModelMatrixBitDelta RemoteSum;
        TModelMatrixBitDelta LocalSum;
        TModelMatrixBitDeltaTail Tail;

        TReduceLevel(yint xSize, yint ySize)
        {
            Tail.Init(xSize, ySize);
            ReadyCount = 0;
        }
    };

    yint P2PIteration = 0;
    yint MatrixId = 0;
    TIntrusivePtr<TP2PNetwork> P2PNet;
    TIntrusivePtr<TParModelMatrixBase> ModelMatrix;
    TVector<TIntrusivePtr<TReduceLevel>> ReduceArr;
    TArray2D<float> DeltaTail;

    static void SumDeltas(TReduceLevel *pLevel, TModelMatrixBitDelta *pRes)
    {
        SumBitDelta(pLevel->LocalSum, pLevel->RemoteSum, &pLevel->Tail, pRes);
        Y_VERIFY(pLevel->ReadyCount.load() == LOCAL_DATA + 1);
        pLevel->ReadyCount = 0;
    }

    void AddDeltaCount(yint level, ui64 c)
    {
        TReduceLevel &rl = *ReduceArr[level];
        if (rl.ReadyCount.fetch_add(c) + c == LOCAL_DATA + 1) {
            bool isFinalLevel = (level + 1 == YSize(ReduceArr));
            TModelMatrixBitDelta *resSum = isFinalLevel ? &ModelMatrix->GetBitDelta() : &ReduceArr[level + 1]->LocalSum;
            SumDeltas(&rl, resSum);
            if (isFinalLevel) {
                ModelMatrix->SetOp(TParModelMatrixBase::OP_ADD_BIT_DELTA);
            } else {
                TNetRank peerAddr = P2PNet->GetMyRank() ^ (1ull << (level + 1));
                P2PNet->Send(peerAddr, SerializeCommand(cmdFabric, new TDeltaMatrix(P2PIteration, MatrixId, level + 1, *resSum)));
                AddDeltaCount(level + 1, LOCAL_DATA);
            }
        }
    }

    void OnDelta() override
    {
        if (P2PNet->GetWorkerCount() == 1) {
            ModelMatrix->SetOp(TParModelMatrixBase::OP_ADD_DELTA);
            return;
        }
        //DebugPrintf("On delta, matrix %g\n", MatrixId * 1.);
        ModelMatrix->SetOp(TParModelMatrixBase::OP_WAIT);

        TModelMatrixBitDelta &localSum = ReduceArr[0]->LocalSum;
        ModelMatrix->ExtractDelta(&localSum, &DeltaTail);

        TNetRank peerAddr = P2PNet->GetMyRank() ^ 1;
        P2PNet->Send(peerAddr, SerializeCommand(cmdFabric, new TDeltaMatrix(P2PIteration, MatrixId, 0, localSum)));
        AddDeltaCount(0, LOCAL_DATA);
    }

public:
    TMMNetDeltaReduce(yint matrixId, TIntrusivePtr<TParModelMatrixBase> p, TIntrusivePtr<TP2PNetwork> p2pNet)
        : MatrixId(matrixId), P2PNet(p2pNet), ModelMatrix(p)
    {
        yint workerCount = P2PNet->GetWorkerCount();
        Y_VERIFY((workerCount & (workerCount - 1)) == 0);
        yint levelCount = 0;
        while ((1ll << levelCount) < workerCount) {
            ++levelCount;
        }
        yint xSize = p->GetXSize();
        yint ySize = p->GetYSize();
        ReduceArr.resize(levelCount);
        for (yint k = 0; k < levelCount; ++k) {
            ReduceArr[k] = new TReduceLevel(xSize, ySize);
        }
        DeltaTail.SetSizes(xSize, ySize);
        DeltaTail.FillZero();
    }

    void AddRemoteDelta(yint deltaP2PIteration, yint sumLevel, TModelMatrixBitDelta *pBitDelta)
    {
        if (deltaP2PIteration != P2PIteration) {
            DebugPrintf("delta iteration mismatch, remote %g, current %g\n", deltaP2PIteration * 1., P2PIteration * 1.);
        }
        pBitDelta->Swap(&ReduceArr[sumLevel]->RemoteSum);
        AddDeltaCount(sumLevel, 1); 
    }

    void SetP2PIteration(yint iter)
    {
        for (TIntrusivePtr<TReduceLevel> &lvl : ReduceArr) {
            Y_VERIFY(lvl->ReadyCount == 0);
        }
        P2PIteration = iter;
    }
};


class TMMNetDeltaReduceGen : public IMMDeltaHookGen
{
    TIntrusivePtr<TP2PNetwork> P2PNet;
    TVector<TIntrusivePtr<TMMNetDeltaReduce>> Arr;
    volatile yint CurrentP2PIteration = 0;

    IMMDeltaHook *CreateDeltaHook(yint idx, TIntrusivePtr<TParModelMatrixBase> p) override
    {
        TMMNetDeltaReduce *res = new TMMNetDeltaReduce(idx, p, P2PNet);
        if (YSize(Arr) <= idx) {
            Arr.resize(idx + 1);
        }
        Arr[idx] = res;
        return res;
    }

    void OnIterationStart() override
    {
        yint newIter = CurrentP2PIteration + 1;
        for (yint k = 0; k < YSize(Arr); ++k) {
            Arr[k]->SetP2PIteration(newIter);
        }
        CurrentP2PIteration = newIter; // after setting iteration for delta hooks
    }

public:
    TMMNetDeltaReduceGen(TIntrusivePtr<TP2PNetwork> p2pNet) : P2PNet(p2pNet)
    {
    }

    void AddRemoteDelta(yint deltaP2PIteration, yint matrixId, yint sumLevel, TModelMatrixBitDelta *pBitDelta)
    {
        Arr[matrixId]->AddRemoteDelta(deltaP2PIteration, sumLevel, pBitDelta);
    }

    yint GetCurrentP2PIteration() const
    {
        return CurrentP2PIteration;
    }
};

void TDeltaMatrix::Exec(TNetTrainContext *p)
{
    p->NetDeltaReduce->AddRemoteDelta(P2PIteration, MatrixId, SumLevel, &BitDelta);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TBackprop : public TCommandPacket
{
    yint Iter = 0;
    yint MaxIters = 0;
    TDescentConfig DescentConfig;
    TDeviceBatchConfig DeviceBatchConfig;
    TVector<TFragment> FragArr;
    SAVELOAD_OVERRIDE(Iter, MaxIters, DescentConfig, DeviceBatchConfig, FragArr);
public:
    TBackprop() {}
    TBackprop(yint iter, yint maxIters, const TDescentConfig &dc, const TDeviceBatchConfig &dbc, const TVector<TFragment> &fragArr)
        : Iter(iter), MaxIters(maxIters), DescentConfig(dc), DeviceBatchConfig(dbc), FragArr(fragArr)
    {
    }
    void Exec(TNetTrainContext *p) override
    {
        TXRng iterRng(Iter);
        TTrainingStep step = DescentConfig.GetStep(Iter, MaxIters);
        BackpropBatch(iterRng, DescentConfig, DeviceBatchConfig, step, FragArr, p->Ctx);
        p->Master.SendCopy(CMD_OK);
    }
};
REGISTER_PACKET(cmdFabric, TBackprop, 3);


///////////////////////////////////////////////////////////////////////////////////////////////////
// P2P network
class TGetP2PPort : public TCommandPacket
{
    void Exec(TNetTrainContext *p) override
    {
        yint port = p->P2PNet->GetPort();
        p->Master.Send(port);
    }
};
REGISTER_PACKET(cmdFabric, TGetP2PPort, 4);


static void P2PWorkerThread(void *p)
{
    TNetTrainContext &ctx = *(TNetTrainContext *)p;
    TVector<TIntrusivePtr<TCommandPacket>> delayedArr;
    yint p2pIteration = -1;
    for (;;) {
        yint newP2PIteration = ctx.NetDeltaReduce->GetCurrentP2PIteration();
        if (p2pIteration != newP2PIteration) {
            p2pIteration = newP2PIteration;
            yint dst = 0;
            for (yint k = 0; k < YSize(delayedArr); ++k) {
                TIntrusivePtr<TCommandPacket> cmd = delayedArr[k];
                if (cmd->GetP2PIteration() == p2pIteration) {
                    cmd->Exec(&ctx);
                } else {
                    Y_VERIFY(cmd->GetP2PIteration() > p2pIteration);
                    delayedArr[dst++] = cmd;
                }
            }
            delayedArr.resize(dst);
        }
        // recv command
        TIntrusivePtr<TCommandPacket> cmd = RecvCommand(cmdFabric, ctx.P2PNet->GetQueue());
        if (cmd.Get()) {
            //DebugPrintf("P2P got command %s\n", typeid(*cmd.Get()).name());
            if (cmd->GetP2PIteration() != p2pIteration) {
                // postpone commands from future iterations
                Y_VERIFY(cmd->GetP2PIteration() > p2pIteration);
                delayedArr.push_back(cmd);
            } else {
                cmd->Exec(&ctx);
            }
        }
    }
}


class TP2PConnect : public TCommandPacket
{
    TNetRank Rank = 0;
    TVector<TString> PeerList;
    SAVELOAD_OVERRIDE(Rank, PeerList);

    void Exec(TNetTrainContext *p) override
    {
        p->P2PNet->ConnectP2P(Rank, PeerList, NetTrainToken);
        DebugPrintf("p2p network complete\n");
        p->NetDeltaReduce = new TMMNetDeltaReduceGen(p->P2PNet);
        p->P2PThread.Create(P2PWorkerThread, p);
        p->Master.SendCopy(CMD_OK);
    }
public:
    TP2PConnect() {}
    TP2PConnect(TNetRank rank, const TVector<TString> &peerList) : Rank(rank), PeerList(peerList)
    {
    }
};
REGISTER_PACKET(cmdFabric, TP2PConnect, 5);


static void CreateP2PNetwork(TMasterNet &masterNet, const TVector<TString> &peerList)
{
    yint workerCount = YSize(peerList);

    TVector<yint> p2pPortArr;
    masterNet.BroadcastCommand(new TGetP2PPort(), &p2pPortArr);
    Y_ASSERT(YSize(p2pPortArr) == workerCount);

    TVector<TString> p2pPeers = peerList;
    for (yint k = 0; k < workerCount; ++k) {
        NNet::ReplacePort(&p2pPeers[k], p2pPortArr[k]);
    }

    DebugPrintf("p2p connect\n");
    for (auto it = masterNet.WorkerSet.begin(); it != masterNet.WorkerSet.end(); ++it) {
        masterNet.SendCommand(it->first, new TP2PConnect(it->second, p2pPeers));
    }
    TVector<ECommandResult> cmdResults;
    masterNet.CollectCommandResults(&cmdResults);
    DebugPrintf("p2p network complete\n");
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCreateModel : public TCommandPacket
{
    yint DeviceCount = 0;
    TModelParams Params;
    yint DeviceMaxNodeCount;
    SAVELOAD_OVERRIDE(DeviceCount, Params, DeviceMaxNodeCount);
public:
    TCreateModel() {}
    TCreateModel(yint deviceCount, const TModelParams &params, yint deviceMaxNodeCount)
        : DeviceCount(deviceCount), Params(params), DeviceMaxNodeCount(deviceMaxNodeCount)
    {
    }
    void Exec(TNetTrainContext *p) override
    {
        p->Model = CreateModel(DeviceCount, Params, p->NetDeltaReduce.Get());
        p->Ctx = NCUDA_GPT::CreateContext(p->Model, DeviceMaxNodeCount);
        p->Master.SendCopy(CMD_OK);
    }
};
REGISTER_PACKET(cmdFabric, TCreateModel, 7);


///////////////////////////////////////////////////////////////////////////////////////////////////
class TWaitDelayedUpdates : public TCommandPacket
{
public:
    void Exec(TNetTrainContext *p) override
    {
        p->Ctx->WaitUpdates();
        p->Master.SendCopy(CMD_OK);
    }
};
REGISTER_PACKET(cmdFabric, TWaitDelayedUpdates, 8);


class TMakeParamsSnapshot : public TCommandPacket
{
public:
    void Exec(TNetTrainContext *p) override
    {
        TModelParams params;
        p->Ctx->GetParams(&params);
        SerializeMem(IO_WRITE, &p->ModelSnapshot, params);
        yint sz = YSize(p->ModelSnapshot);
        p->Master.Send(sz);
    }
};
REGISTER_PACKET(cmdFabric, TMakeParamsSnapshot, 9);


class TGetParamsSnapshotFragment : public TCommandPacket
{
    yint Offset = 0;
    yint Size = 0;
    SAVELOAD_OVERRIDE(Offset, Size);
public:
    TGetParamsSnapshotFragment() {}
    TGetParamsSnapshotFragment(yint offset, yint size) : Offset(offset), Size(size) {}
    void Exec(TNetTrainContext *p) override
    {
        TVector<ui8> frag;
        frag.resize(Size);
        memcpy(frag.data(), p->ModelSnapshot.data() + Offset, Size);
        p->Master.Send(frag);
    }
};
REGISTER_PACKET(cmdFabric, TGetParamsSnapshotFragment, 10);


class TModelParamsFetcher
{
    enum {
        FRAG_COUNT = 1000
    };
    bool IsFetchingFlag = false;
    yint TotalSize = 0;
    yint Offset = 0;
    yint FragSize = 0;
    TVector<ui8> Buf;
    TString ResFilename;
public:
    bool IsFetching() const { return IsFetchingFlag; }
    void StartFetch(yint sz, const TString &resFilename)
    {
        Y_VERIFY(!IsFetchingFlag);
        IsFetchingFlag = true;
        TotalSize = sz;
        Offset = 0;
        FragSize = sz / FRAG_COUNT + 1;
        Buf.resize(sz);
        ResFilename = resFilename;
    }
    TGetParamsSnapshotFragment *MakeDownloadCommand()
    {
        Y_VERIFY(IsFetchingFlag);
        yint sz = Min<yint>(FragSize, TotalSize - Offset);
        return new TGetParamsSnapshotFragment(Offset, sz);
    }
    void GotDownloadCommandResult(const TVector<ui8> &result)
    {
        yint sz = YSize(result);
        memcpy(Buf.data() + Offset, result.data(), sz);
        Offset += sz;
        Y_VERIFY(Offset <= TotalSize);
        if (Offset == TotalSize) {
            TFileStream f(IO_WRITE, ResFilename.c_str());
            f.Write(Buf.data(), YSize(Buf));
            IsFetchingFlag = false;
        }
    }
};


static void FetchModelFragment(TMasterNet &masterNet, TModelParamsFetcher *p, TIntrusivePtr<ITcpConnection> modelFetchConn)
{
    TModelParamsFetcher &modelFetch = *p;
    masterNet.SendCommand(modelFetchConn, modelFetch.MakeDownloadCommand());
    TVector<ui8> result;
    WaitData(masterNet.Queue, modelFetchConn, &result);
    modelFetch.GotDownloadCommandResult(result);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void RunWorker(yint port)
{
    TNetTrainContext ctx;
    ctx.Net = CreateTcpSendRecv();
    ctx.Master.ConnectMaster(ctx.Net, port, NetTrainToken);
    ctx.P2PNet = new TP2PNetwork(ctx.Net, NetTrainToken);
    DebugPrintf("executing incoming commands\n");
    for (;;) {
        TIntrusivePtr<TCommandPacket> cmd = RecvCommand(cmdFabric, ctx.Master.GetQueue());
        if (cmd.Get()) {
            //DebugPrintf("Worker got command %s\n", typeid(*cmd.Get()).name());
            cmd->Exec(&ctx);
        }
    }
}


void RunMaster(yint startIteration, yint deviceCount, const TVector<TString> &workerAddrArr, const TTrainContext &trainCtx, TIntrusivePtr<TModelParamsHolder> pParams)
{
    const TDescentConfig &dc = trainCtx.GetDescentConfig();
    const TDeviceBatchConfig &dbc = trainCtx.GetDeviceBatchConfig();
    yint workerCount = YSize(workerAddrArr);
    Y_VERIFY(workerCount > 0 && (workerCount & (workerCount - 1)) == 0 && "pow2 worker count only is supported atm");

    TIntrusivePtr<ITcpSendRecv> net = CreateTcpSendRecv();
    TMasterNet masterNet(cmdFabric, net);
    masterNet.ConnectWorkers(workerAddrArr, NetTrainToken);

    DebugPrintf("create p2p network\n");
    CreateP2PNetwork(masterNet, workerAddrArr);

    DebugPrintf("create model\n");
    TVector<ECommandResult> cmdResults;
    masterNet.BroadcastCommand(new TCreateModel(deviceCount, pParams->Params, dbc.GetDeviceMaxNodeCount()), &cmdResults);
    pParams = 0;

    NHPTimer::STime tStart;
    NHPTimer::GetTime(&tStart);
    TModelParamsFetcher modelFetch;
    TIntrusivePtr<ITcpConnection> modelFetchConn = masterNet.WorkerSet.begin()->first;
    yint maxIters = trainCtx.GetMaxIters();
    for (yint iter = startIteration; iter <= maxIters; ++iter) {
        if ((iter % trainCtx.GetEvalInterval()) == 0) {
            // wait delayed updates on all hosts (otherwise no network exchange will happen)
            TVector<ECommandResult> cmdResults;
            masterNet.BroadcastCommand(new TWaitDelayedUpdates(), &cmdResults);
            if (trainCtx.IsSaveModel() && !modelFetch.IsFetching()) {
                // make model params snapshot on first host
                masterNet.SendCommand(modelFetchConn, new TMakeParamsSnapshot());
                yint sz;
                WaitData(masterNet.Queue, modelFetchConn, &sz);
                modelFetch.StartFetch(sz, Sprintf("d:/eden_gpt_%.8gk.bin", iter / 1000.));
            }
            float trainErr = DistributedCalcModelErr(masterNet, trainCtx.GetScoreTrainBatches()) * trainCtx.GetCompression();
            float testErr = DistributedCalcModelErr(masterNet, trainCtx.GetScoreTestBatches()) * trainCtx.GetCompression();
            if (testErr != 0) {
                DebugPrintf("iter %.8gk, %g sec, train err %g, test err %g\n", iter / 1000., NHPTimer::GetTimePassed(&tStart), trainErr, testErr); fflush(0);
            } else {
                DebugPrintf("iter %.8gk, %g sec, train err %g\n", iter / 1000., NHPTimer::GetTimePassed(&tStart), trainErr); fflush(0);
            }
        }

        // fetch model snapshot one fragment per iteration
        if (modelFetch.IsFetching()) {
            FetchModelFragment(masterNet, &modelFetch, modelFetchConn);
        }

        // backprop
        for (auto it = masterNet.WorkerSet.begin(); it != masterNet.WorkerSet.end(); ++it) {
            ui64 rank = it->second;
            ui64 rngSeed = (iter + 0xbadf00d) * 0x3148efull + rank * 0x0b08d424991cf81eull;
            TVector<TFragment> fragArr;
            trainCtx.SampleTrainBatches(rngSeed, &fragArr);

            masterNet.SendCommand(it->first, new TBackprop(iter, maxIters, dc, dbc, fragArr));
        }
        masterNet.CollectCommandResults(&cmdResults);
    }

    DebugPrintf("Fetch last iteration model\n");
    while (modelFetch.IsFetching()) {
        FetchModelFragment(masterNet, &modelFetch, modelFetchConn);
    }
}
}
