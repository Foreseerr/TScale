#include "stdafx.h"
#include <gpt/fed_lib/fed_lib.h>
#include <gpt/att/sliding_window.h>
#include <gpt/compute/gpt_cuda.cuh>
#include <gpt/data/net_data.h>
#include <gpt/train_config/train_config.h>
#include <gpt/train_ctx/batch_config.h>
#include <gpt/train_ctx/backprop.h>
#include <lib/cuda/cuda_init.h>
#include <lib/net/tcp_net.h>
#include <lib/file/dir.h>
#include <lib/config/config.h>
#include <lib/config/cfg_file.h>
#include <lib/hp_timer/hp_timer.h>


// applying deltas may take exoribitant amount of time like 60+ seconds. This seems to be caused by kcompactd
// To avoid this behavior try disabling "proactiveness":
// echo 0 >/proc/sys/vm/compaction_proactiveness

using namespace NNet;


///////////////////////////////////////////////////////////////////////////////////////////////////
static void PackModelParams(TMemStream *p, TIntrusivePtr<IComputeContext> pCtx, float weight)
{
    p->Seek(0);
    TBufferedStream bufIO(IO_WRITE, *p);
    IBinSaver bs(bufIO);
    bufIO.Write(&weight, sizeof(weight));
    pCtx->PackMatrices(bufIO);
    TModelRowDisp rd;
    pCtx->GetRowDisp(&rd);
    bs.Add(&rd);
}

static float AddPackedModelParamsScaled(TIntrusivePtr<IComputeContext> pCtx, TMemStream &pkt, float scale)
{
    float weight = 0;
    pkt.Seek(0);
    TBufferedStream bufIO(IO_READ, pkt);
    bufIO.Read(&weight, sizeof(weight));
    pCtx->AddPackedMatricesImpl(bufIO, scale);
    return weight;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static void ParseUserConfig(const TString &filename, TFedLogin *p)
{
    TVector<char> cfgText;
    Y_VERIFY(ReadWholeFile(filename, &cfgText));
    TConfigFile cfg;
    ParseConfig(&cfg, cfgText.data());
    for (TConfigFile::TOp &op : cfg.OpArr) {
        if (op.Op == CFG_OP_ASSIGNMENT) {
            if (op.Dst == "UserName") {
                p->UserName = op.Args[0];
            } else if (op.Dst == "UserId") {
                p->UserId = GetGuid(op.Args[0]);
            } else {
                DebugPrintf("Ignoring unknown variable %s\n", op.Dst.c_str());
            }
        } else {
            DebugPrintf("Ignoring unknown op %s\n", op.Dst.c_str());
        }
    }
}

static void PrintUserConfig(const TString &filename, const TFedLogin &login)
{
    TOFStream f(filename.c_str());
    f << "UserName = " << login.UserName.c_str() << "\n";
    f << "UserId = '" << GetGuidAsString(login.UserId).c_str() << "'\n";
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
static void SendData(TIntrusivePtr<ITcpSendRecv> net, TIntrusivePtr<ITcpConnection> conn, T &data)
{
    TIntrusivePtr<TTcpPacket> pkt = new TTcpPacket;
    SerializeMem(IO_WRITE, &pkt->Data, data);
    net->Send(conn, pkt);
}


static TIntrusivePtr<TTcpPacketReceived> RecvPacket(TIntrusivePtr<TTcpRecvQueue> q)
{
    TIntrusivePtr<TTcpPacketReceived> pkt;
    while (!q->Dequeue(&pkt)) {
        SchedYield();
    }
    return pkt;
}


template <class T>
void RecvData(TIntrusivePtr<TTcpRecvQueue> net, T *p)
{
    TIntrusivePtr<TTcpPacketReceived> pkt;
    while (!net->Dequeue(&pkt)) {
        SchedYield();
    }
    SerializeMem(IO_READ, &pkt->Data, *p);
}


void RecvWeightedModelParams(TIntrusivePtr<TTcpRecvQueue> net, TMemStream *p)
{
    TIntrusivePtr<TTcpPacketReceived> pkt;
    while (!net->Dequeue(&pkt)) {
        SchedYield();
    }
    p->Swap(&pkt->Data);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TDataQueriesCtx
{
    TIntrusivePtr<IDataSource> RemoteData;
    TIntrusivePtr<TSyncEvent> Ready;
    TIntrusivePtr<TSyncEvent> Query;
    TVector<TFragment> FragArr;
    yint FragCount = 0;
    yint FragLen = 0;
    TThread Thr;
    TXRng Rng;

    TDataQueriesCtx(TIntrusivePtr<ITcpSendRecv> net, const TString &addr, yint fragCount, yint fragLen)
        : FragCount(fragCount), FragLen(fragLen), Rng(GetCycleCount())
    {
        Ready = new TSyncEvent;
        Query = new TSyncEvent;
        RemoteData = ConnectDataServer(net, addr);
        Query->Set();
        Thr.Create(this);
    }
    void WorkerThread()
    {
        for (;;) {
            Query->Wait();
            RemoteData->SampleFragments(IDataSource::TRAIN, Rng.GenRand(), FragCount, FragLen, &FragArr);
            Ready->Set();
        }
    }
    void GetFragments(TVector<TFragment> *pRes)
    {
        Ready->Wait();
        pRes->swap(FragArr);
        Query->Set();
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
extern yint MatrixAddWorkerThreadCount;

int main(int argc, char **argv)
{
    TString centerAddr = "localhost";
    TString configFilename = "user.cfg";
    TFedLogin login;
    yint limitNodeCount = 24 * 1024;
    yint deviceCount = GetCudaDeviceCount();

    if (DoesFileExist(configFilename)) {
        ParseUserConfig(configFilename, &login);
        DebugPrintf("user %s key loaded from %s\n", login.UserName.c_str(), configFilename.c_str());
    }

    TOpt cmdline("c:n:d:u:t:", argc, argv);
    for (const TOpt::TParam &param : cmdline.Params) {
        if (param.Name == "c") {
            centerAddr = param.Args[0];
        } else if (param.Name == "n") {
            limitNodeCount = atoi(param.Args[0].c_str());
        } else if (param.Name == "d") {
            deviceCount = atoi(param.Args[0].c_str());
        } else if (param.Name == "u") {
            login.UserName = param.Args[0];
        } else if (param.Name == "t") {
            MatrixAddWorkerThreadCount = atoi(param.Args[0].c_str());
        }
    }

    if (!login.UserName.empty() && !IsValidUsername(login.UserName)) {
        DebugPrintf("illigal username %s\n", login.UserName.c_str());
        return 0;
    }

    if (!login.UserName.empty() && login.UserId.IsEmpty()) {
        CreateGuid(&login.UserId);
        PrintUserConfig(configFilename, login);
        DebugPrintf("user id saved to %s\n", configFilename.c_str());
    }

    TIntrusivePtr<ITcpSendRecv> net = CreateTcpSendRecv();
    TIntrusivePtr<TTcpRecvQueue> gradQueue = new TTcpRecvQueue();
    TIntrusivePtr<ITcpConnection> gradConn = Connect(centerAddr, FED_GRAD_PORT, FedToken);
    net->StartSendRecv(gradConn, gradQueue);

    // send login data
    SendData(net, gradConn, login);

    // get config
    TFedParams fedParams;
    RecvData(gradQueue, &fedParams);
    DebugPrintf("received fed params, current train batches %gk\n", fedParams.TrainBatchCount / 1000.); fflush(0);
    TDescentConfig dc = fedParams.DescentConfig;
    TDeviceBatchConfig dbc(deviceCount, limitNodeCount, dc.TrainBatchSize, dc.TrainFragLen);

    // run train fragment gen thread
    TDataQueriesCtx dataQuery(net, fedParams.DataServerAddr, dc.TrainBatchSize, dc.TrainFragLen);

    // recv basepoint
    TMemStream basepoint;
    RecvWeightedModelParams(gradQueue, &basepoint);
    DebugPrintf("received base point\n"); fflush(0);

    // create model
    TIntrusivePtr<IModel> pModel = CreateModel(deviceCount, fedParams.ModelDescr, fedParams.BiasArr, nullptr);
    TIntrusivePtr<IComputeContext> pCtx = NCUDA_GPT::CreateContext(pModel, dbc.GetDeviceMaxNodeCount());
    pCtx->SetRowDisp(fedParams.RowDisp);
    AddPackedModelParamsScaled(pCtx, basepoint, 1);
    pCtx->ApplyModifiedMatrices();
    double trainBatchCount = fedParams.TrainBatchCount;

    // our delta contribution
    TIntrusivePtr<TTcpPacket> myDeltaPkt = new TTcpPacket;
    // current epoch weight
    float currentWeight = 0;
    float prevWeight = 0;

    NHPTimer::STime tLastDelta;
    NHPTimer::GetTime(&tLastDelta);

    TXRng rng(GetCycleCount());
    for (yint iter = 0;; ++iter) {

        // receive batch
        TVector<TFragment> fragArr;
        dataQuery.GetFragments(&fragArr);
        Y_VERIFY(YSize(fragArr) == dc.TrainBatchSize);
        //DebugPrintf("got %g fragments\n", YSize(fragArr) * 1.); fflush(0);

        TTrainingStep step = dc.GetStep(trainBatchCount, trainBatchCount * 2);
        BackpropBatch(rng, dc, dbc, step, fragArr, pCtx);
        currentWeight += 1;
        trainBatchCount += 1;

        // process new basepoint if it has arrived
        TIntrusivePtr<TTcpPacketReceived> newBasepointPkt;
        if (gradQueue->Dequeue(&newBasepointPkt)) {
            pCtx->WaitUpdates();
            double iterTime = NHPTimer::GetTimePassed(&tLastDelta);
            float batchesPerMin = currentWeight / iterTime * 60;
            float avrgErr = pCtx->GetAvrgTrainErr() * fedParams.Compression;
            DebugPrintf("%g batches per minute, err %g, replacing basepoint, sz = %gmb\n", batchesPerMin, avrgErr, YSize(newBasepointPkt->Data) / 1000000.); fflush(0);

            // subtract old basepoint, assign new basepoint
            AddPackedModelParamsScaled(pCtx, basepoint, -1);
            basepoint.Swap(&newBasepointPkt->Data);

            // compute new delta
            TMemStream newMyDelta(&newBasepointPkt->Data); // reuse memory buffer
            newMyDelta.Seek(0);
            PackModelParams(&newMyDelta, pCtx, currentWeight);
            newMyDelta.Truncate();

            // add new basepoint
            float globalWeight = AddPackedModelParamsScaled(pCtx, basepoint, 1);
            Y_ASSERT(globalWeight >= prevWeight);
            float myFraction = prevWeight / globalWeight;
            trainBatchCount += globalWeight;

            // subtract our contribution (correctly weighted old delta)
            if (prevWeight > 0) {
                TMemStream myDelta(&myDeltaPkt->Data);
                AddPackedModelParamsScaled(pCtx, myDelta, -myFraction * fedParams.FedWeightScale);
                trainBatchCount -= prevWeight;
            }

            // reset grad
            //pCtx->ResetGrad();
            if (globalWeight > 0) {
                DebugPrintf("scaling gradient by %g\n", myFraction);
                pCtx->ScaleGrad(myFraction);
            }

            // keep weight
            prevWeight = currentWeight;
            currentWeight = 0;

            // assign new myDelta and send it
            newMyDelta.Swap(&myDeltaPkt->Data);
            net->Send(gradConn, myDeltaPkt);

            // convert new matrices
            pCtx->ApplyModifiedMatrices();

            // count time
            DebugPrintf("delta applied in %g secs, continue mining\n", NHPTimer::GetTimePassed(&tLastDelta)); fflush(0);
        }
    }

    return 0;
}
