#include "stdafx.h"
#include <lib/net/tcp_net.h>
#include <gpt/fed_lib/fed_lib.h>
#include <gpt/data/net_data.h>
#include <gpt/model_params/fed_model.h>
#include <lib/hp_timer/hp_timer.h>
#include <lib/net/http_server.h>
#include <lib/net/http_request.h>
#include <lib/net/html_compose.h>
#include <lib/net/poller.h>
#include <lib/config/config.h>
#include <lib/file/dir.h>
#include <lib/log/log.h>


TString FED_SCRIPT =
    " DELTA_COLLECT_TIMEOUT = 100"
    " DELTA_COLLECT_MIN_INTERVAL = 3"
    " SAVE_MODEL_INTERVAL = 5000"
    " TRAIN_CONFIG = 'b64f128'"
    " DROP_CONFIG = 'drop1ch1lr0.01'"
    " MODEL_DIMS = 'e256d30w128'"
    " FED_WEIGHT_SCALE = 0.5"
    // load data, create model, train
    " create_model(MPF_TAIL_LOSS)"
    //" load_model('fed/fed_start.bin')\n"
    //" START_TRAIN_BATCH_COUNT = 20000\n"
    " run_fed_center('fed/run_small')\n"
    ;

//TString FED_SCRIPT =
//    " KEEP_MODEL_COUNT = 100"
//    " DELTA_COLLECT_TIMEOUT = 300"
//    " DELTA_COLLECT_MIN_INTERVAL = 100"
//    " TRAIN_CONFIG = 'b96f1024'"
//    " DROP_CONFIG = 'drop1ch1'"
//    " MODEL_DIMS = 'e1024tt256d65w1024'" // 420M
//    // load data, create model, train
//    " create_model(MPF_TAIL_LOSS)"
//    " run_fed_center('fed/run_large')\n"
//    ;


const TString ModelFileExtension = ".fm";
const TString CenterStateFile = "users.bin";
const TString NewCenterStateFile = "users_new.bin";

const ui32 LOG_ID = 0xc280fe2c;
USE_LOG(LOG_ID);

using namespace NNet;


///////////////////////////////////////////////////////////////////////////////////////////////////
static void PackModelParams(TMemStream *p, const TAllModelMatrices &params, float weight)
{
    p->Seek(0);
    TBufferedStream bufIO(IO_WRITE, *p);
    bufIO.Write(&weight, sizeof(weight));
    PackMatrices(bufIO, params);
}

static float AddPackedModelParamsScaled(TAllModelMatrices *pParams, TModelRowDisp *pRowDisp, TMemStream &pkt)
{
    float weight = 0;
    pkt.Seek(0);
    TBufferedStream bufIO(IO_READ, pkt);
    IBinSaver bs(bufIO);
    bufIO.Read(&weight, sizeof(weight));
    AddPackedMatrices(pParams, bufIO, weight);
    TModelRowDisp rd;
    bs.Add(&rd);
    pRowDisp->AddScaled(rd, weight);
    return weight;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TFedUser
{
    TGuid UserId;
    TString UserName;
    double Batches = 0;
    SAVELOAD(UserId, UserName, Batches);
};

struct TFedCenterState
{
    THashMap<TGuid, TFedUser> Users;
    double TotalBatches = 0;
    SAVELOAD(Users, TotalBatches);
};

struct TFedWorkerStat
{
    TString Addr;
    float SumWeight = 0;
    float SumCount = 0;
    float CurrentWeight = 0;
};

struct TFedInfo
{
    struct TWorker
    {
        TFedWorkerStat Stats;
        TString UserName;
        TWorker() {}
        TWorker(const TFedWorkerStat &stats, const TString &userName) : Stats(stats), UserName(userName) {}
    };
    TVector<TWorker> WorkerArr;
    TVector<TFedUser> UserArr;
    float TimeSinceLastDelta = 0;
    double TotalBatches = 0;
    double ModelBatches = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
static void RenderRootPage(const TFedInfo &info, TString *pRes)
{
    TString css = NNet::DefaultTableCSS("center");
    NNet::THtmlPage page("Fed", css, "");

    // server state
    page += Sprintf("Total batches %gk, model batches %gk, time since last data: %g sec<br><br>", info.TotalBatches / 1000., info.ModelBatches / 1000., info.TimeSinceLastDelta);

    // workers
    page += "<table><tr><th>Username<th>ip addr<th>Avrg batches/iter<th>Last iter batches\n";
    for (const TFedInfo::TWorker &worker : info.WorkerArr) {
        page += Sprintf("<tr><td>%s<td>%s", worker.UserName.c_str(), worker.Stats.Addr.c_str());
        page += (worker.Stats.SumCount > 0) ? Sprintf("<td>%g\n", worker.Stats.SumWeight / worker.Stats.SumCount) : "<td>?";
        page += Sprintf("<td>%g\n", worker.Stats.CurrentWeight);
    }
    page += "</table><br>\n";

    // users
    page += "<table><tr><th>Username<th>Batches\n";
    for (const TFedUser &user : info.UserArr) {
        TString batches = (user.Batches > 2000) ? Sprintf("%gk", user.Batches / 1000.) : Sprintf("%g", user.Batches);
        page += Sprintf("<tr><td>%s<td>%s", user.UserName.c_str(), batches.c_str());
    }
    page += "</table><br>\n";

    // view log
    page += "<a href=\"log?max_lines=50\"><button style='font-size:large'>Logs</button></a><br><br>\n";

    // render result
    page.MakeHtml(pRes);
}


static void RenderLog(int maxLines, TString *pRes)
{
    TString css =
        "        table {border-collapse: collapse; font-family: monospace; font-size: smaller;}\n"
        "        tr,td{text-align:left; padding-right:1rem;}\n";

    THtmlPage page("Logs", css, "");
    page += "<table><tr style='background-color:lightgrey;'><th>Time<th>Msg";

    TVector<NLog::TLogEntry> logArr;
    NLog::GetLastMessages(NLog::NO_FILTER, maxLines, &logArr);
    for (yint i = YSize(logArr) - 1; i >= 0; --i) {
        const NLog::TLogEntry &le = logArr[i];
        std::tm *tm = localtime(&le.Time);
        page += Sprintf("<tr><td>%d.%d.%d %d:%02d:%02d<td>%s",
            tm->tm_mday, tm->tm_mon + 1, tm->tm_year + 1900, tm->tm_hour, tm->tm_min, tm->tm_sec,
            le.Msg.c_str());
    }
    page += "</table>\n";
    page += "<br>\n";
    page += "<a href=\"/\">Home</a>\n";
    page.MakeHtml(pRes);
}




///////////////////////////////////////////////////////////////////////////////////////////////////
class TFedCenterCtx
{
    struct TWorker
    {
        TFedWorkerStat Stat;
        TGuid UserId;
        TString UserName;
        bool GotDelta = true;
        bool FirstDelta = true;

        TWorker() {}
        TWorker(TIntrusivePtr<ITcpConnection> conn, const TGuid &userId, const TString &userName) : UserId(userId), UserName(userName)
        {
            Stat.Addr = conn->GetPeerAddress();
        }
    };

    TIntrusivePtr<ITcpSendRecv> Net;
    TIntrusivePtr<TTcpPacket> BasepointPkt;
    TFedCenterModel Model;
    TAllModelMatrices SumDelta;
    TModelRowDisp SumRowDisp;
    TFedCenterState CenterState;
    float SumDeltaWeight = 0;
    THashMap<TIntrusivePtr<ITcpConnection>, TWorker> WorkerSet;

public:
    TFedCenterCtx(TIntrusivePtr<ITcpSendRecv> net) : Net(net)
    {
    }

    TFedCenterModel &GetModel()
    {
        return Model;
    }

    TFedCenterState &GetCenterState()
    {
        return CenterState;
    }

    void InitModel(TIntrusivePtr<TModelParamsHolder> basepoint, float startTrainBatchCount, const TDescentConfig &descentConfig)
    {
        // copy model params
        Y_VERIFY(basepoint.Get() && !basepoint->Params.IsEmpty());
        GetMatrices(&Model.Params, basepoint->Params);
        Model.ModelDescr = basepoint->Params.ModelDescr;
        Model.BiasArr = basepoint->Params.Bias;
        GetRowDisp(&Model.RowDisp, basepoint->Params);
        Model.TrainBatchCount = startTrainBatchCount;
        // set train frag len
        Model.ModelDescr.FragLen = descentConfig.TrainFragLen;
    }

    void InitBasepoint(yint dataVocabSize)
    {
        Y_VERIFY(Model.ModelDescr.VocabSize == dataVocabSize);
        // serialize basepoint
        TMemStream wbp;
        PackModelParams(&wbp, Model.Params, 0);
        BasepointPkt = new TTcpPacket;
        wbp.Swap(&BasepointPkt->Data);
        // init sumdelta
        InitZero(&SumDelta, Model.ModelDescr);
        SumRowDisp.Clear();
        SumDeltaWeight = 0;
    }

    void CreateNewBasepoint(float fedWeightScale)
    {
        {
            // add sum delta to basepoint
            TMemStream wbp;
            wbp.Swap(&BasepointPkt->Data);
            if (SumDeltaWeight > 0) {
                CenterState.TotalBatches += SumDeltaWeight;
                Model.Params.AddScaled(SumDelta, fedWeightScale / SumDeltaWeight);
                Model.TrainBatchCount += SumDeltaWeight;
                Model.RowDisp = SumRowDisp;
                Model.RowDisp.Scale(1 / SumDeltaWeight);
            }
            PackModelParams(&wbp, Model.Params, SumDeltaWeight);
            // keep in basepoint packet
            wbp.Swap(&BasepointPkt->Data);
        }
        // clear sum delta
        if (SumDeltaWeight > 0) {
            SumDelta.FillZero();
            SumRowDisp.Clear();
            SumDeltaWeight = 0;
        }
        // update user total counters and reset worker state
        for (auto it = WorkerSet.begin(); it != WorkerSet.end(); ++it) {
            TWorker &worker = it->second;
            Y_VERIFY(worker.GotDelta);
            CenterState.Users[worker.UserId].Batches += worker.Stat.CurrentWeight;
            if (!worker.FirstDelta) {
                worker.Stat.SumWeight += worker.Stat.CurrentWeight;
                worker.Stat.SumCount += 1;
            }
            worker.Stat.CurrentWeight = 0;
            worker.GotDelta = false;
            worker.FirstDelta = false;
        }
    }

    void SendNewBasepoint()
    {
        // send new basepoint
        for (auto it = WorkerSet.begin(); it != WorkerSet.end(); ++it) {
            TWorker &worker = it->second;
            Net->Send(it->first, BasepointPkt);
            Log("send basepoint to %s", worker.Stat.Addr.c_str());
        }
    }

    void ReceiveDelta(TIntrusivePtr<TTcpPacketReceived> recvPkt)
    {
        if (WorkerSet.find(recvPkt->Conn) == WorkerSet.end()) {
            Log("ignore packet from closed connection %p", recvPkt->Conn.Get());
            return;
        }
        // process delta
        TMemStream delta;
        delta.Swap(&recvPkt->Data);
        // add delta
        float deltaWeight = AddPackedModelParamsScaled(&SumDelta, &SumRowDisp, delta);
        SumDeltaWeight += deltaWeight;
        // update worker
        TWorker &worker = WorkerSet[recvPkt->Conn];
        Log("got delta from %s", worker.Stat.Addr.c_str());
        Y_VERIFY(worker.GotDelta == false);
        worker.GotDelta = true;
        worker.Stat.CurrentWeight = deltaWeight;
    }

    void SendFedParams(TIntrusivePtr<ITcpConnection> conn, TFedParams *pFedParams)
    {
        TFedParams &fedParams = *pFedParams;
        fedParams.ModelDescr = Model.ModelDescr;
        fedParams.RowDisp = Model.RowDisp;
        fedParams.BiasArr = Model.BiasArr;
        fedParams.TrainBatchCount = Model.TrainBatchCount;
        Net->Send(conn, MakePacket(fedParams));
    }

    void NewLogin(TIntrusivePtr<ITcpConnection> conn, const TFedLogin &login, TFedParams *pFedParams)
    {
        bool ok = true;
        TString userName = login.UserName;
        if (login.UserId.IsEmpty()) {
            userName = "anon";
        } else {
            auto it = CenterState.Users.find(login.UserId);
            if (it == CenterState.Users.end()) {
                if (!IsValidUsername(userName)) {
                    ok = false;
                } else {
                    Log("add new user %s", userName.c_str());
                }
            }
        }
        if (ok) {
            CenterState.Users[login.UserId].UserName = userName;
            SendFedParams(conn, pFedParams);
            Net->Send(conn, BasepointPkt);
            WorkerSet[conn] = TWorker(conn, login.UserId, userName);
            Log("added worker %s from %s (connection %p), send basepoint", userName.c_str(), conn->GetPeerAddress().c_str(), conn.Get());
        } else {
            conn->Stop();
        }
    }

    bool CheckIfAllWorkersSentDelta(bool isTimeout)
    {
        bool deltaCollected = true;
        for (auto it = WorkerSet.begin(); it != WorkerSet.end();) {
            bool keep = false;
            if (it->first->IsValid()) {
                keep = true;
                TWorker &worker = it->second;
                if (!worker.GotDelta) {
                    if (isTimeout) {
                        Log("disconnect worker %s (connection %p) on delta collect timeout", worker.Stat.Addr.c_str(), it->first.Get());
                        it->first->Stop();
                        keep = false;
                    } else {
                        deltaCollected = false;
                    }
                }
            }
            if (keep) {
                ++it;
            } else {
                auto del = it++;
                WorkerSet.erase(del);
            }
        }
        return deltaCollected;
    }

    bool IsIdle() const
    {
        return WorkerSet.empty() && SumDeltaWeight == 0;
    }

    void CollectInfo(TFedInfo *p)
    {
        p->TotalBatches = CenterState.TotalBatches;
        p->ModelBatches = Model.TrainBatchCount;
        for (auto it = WorkerSet.begin(); it != WorkerSet.end(); ++it) {
            TWorker &worker = it->second;
            p->WorkerArr.push_back(TFedInfo::TWorker(worker.Stat, worker.UserName));
        }
        for (auto it = CenterState.Users.begin(); it != CenterState.Users.end(); ++it) {
            p->UserArr.push_back(it->second);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// save/load center state with processed batches
static void LoadCenterState(const TString &folder, TFedCenterState *p)
{
    TString fname = folder + "/" + CenterStateFile;
    TString fname2 = folder + "/" + NewCenterStateFile;
    if (DoesFileExist(fname2)) {
        Log("load center state from %s", fname2.c_str());
        Serialize(IO_READ, fname2, *p);
        if (DoesFileExist(fname)) {
            EraseFile(fname);
        }
        RenameFile(fname2, fname);
    } else if (DoesFileExist(fname)) {
        Log("load center state from %s", fname.c_str());
        Serialize(IO_READ, fname, *p);
    } else {
        Log("no center state found");
    }
}


static void SaveCenterState(const TString &folder, TFedCenterState &centerState)
{
    TString fname = folder + "/" + CenterStateFile;
    TString fname2 = folder + "/" + NewCenterStateFile;
    TString tmp = folder + "/center.tmp";
    Serialize(IO_WRITE, tmp, centerState);
    RenameFile(tmp, fname2);
    EraseFile(fname);
    RenameFile(fname2, fname);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TFedCenter
{
    TIntrusivePtr<ITcpSendRecv> Net;
    TTrainModelConfigParser ModelConfig;
    TString DataServerAddr;
    TIntrusivePtr<IDataSource> DataServer;
    float FedWeightScale = 0.5f;
    float DeltaCollectTimeout = 10 * 60; // in seconds
    float DeltaCollectMinInterval = 10; // in seconds
    float SaveModelInterval = 0;
    yint KeepModelCount = 50;
    double StartTrainBatchCount = 0;
    yint Version = 1; // checkpoint number

private:
    void EraseTmpFiles(const TString &folder)
    {
        TVector<TFindFileResult> dir;
        FindAllFiles(folder, &dir);
        for (const TFindFileResult &ff : dir) {
            if (ff.IsDir) {
                continue;
            }
            if (EndsWith(ff.Name, ".tmp")) {
                EraseFile(ff.Name);
            }
        }
    }

private:
    // save/load model checkpoints
    void LoadLastCheckpoint(const TString &folder, TFedCenterModel *pModel)
    {
        TVector<TFindFileResult> dir;
        FindAllFiles(folder, &dir);
        TString modelFile;
        THashMap<int, TString> allModels;
        for (const TFindFileResult &ff : dir) {
            if (ff.IsDir) {
                continue;
            }
            if (EndsWith(ff.Name, ModelFileExtension)) {
                TString sz = ff.Name.substr(0, YSize(ff.Name) - YSize(ModelFileExtension));
                yint numLen = 0;
                while (numLen < YSize(sz) && isdigit(sz[YSize(sz) - numLen - 1])) {
                    ++numLen;
                }
                yint version = atol(sz.substr(YSize(sz) - numLen).c_str());
                allModels[version] = folder + ff.Name;
                if (version >= Version) {
                    Version = version;
                    modelFile = folder + ff.Name;
                }
            }
        }
        if (!modelFile.empty()) {
            Log("load model version %d", (int)Version);
            Serialize(IO_READ, modelFile, *pModel);
            ++Version;
            if (KeepModelCount > 0) {
                for (auto it = allModels.begin(); it != allModels.end(); ++it) {
                    if (it->first < Version - KeepModelCount) {
                        Log("erase obsolete model %s", it->second.c_str());
                        EraseFile(it->second);
                    }
                }
            }
        }
    }

    void SaveCheckpoint(TFedCenterModel &model, const TString &folder)
    {
        TString tmpName = folder + "model.tmp";
        Serialize(IO_WRITE, tmpName, model);
        RenameFile(tmpName, Sprintf("%smodel_%d%s", folder.c_str(), (int)Version++, ModelFileExtension.c_str()));
        Log("model saved");
        if (KeepModelCount > 0) {
            TString oldFile = Sprintf("%smodel_%d%s", folder.c_str(), (int)(Version - KeepModelCount - 1), ModelFileExtension.c_str());
            if (DoesFileExist(oldFile)) {
                EraseFile(oldFile);
            }
        }
    }


private:
    struct TAvrgModelComputer
    {
        TAllModelMatrices AvrgModelParams;
        double Interval = 0;
        double SaveModelBatches = 0;
        double AvrgModelCount = 0;

        void Init(const TFedCenterModel &model, float interval)
        {
            AvrgModelParams = model.Params;
            Interval = interval;
            SaveModelBatches = (floor(model.TrainBatchCount / interval) + 2) * interval;
            AvrgModelCount = 1;
            Log("new save model batches border %g", SaveModelBatches);
        }
        void Add(const TString &folder, const TFedCenterModel &model)
        {
            AvrgModelParams.AddScaled(model.Params, 1);
            AvrgModelCount += 1;
            if (model.TrainBatchCount >= SaveModelBatches) {
                if (AvrgModelCount > 0) {
                    Log("save average over %g basepoints model at %gk batches, total %gk batches", AvrgModelCount, SaveModelBatches / 1000., model.TrainBatchCount / 1000.);
                    AvrgModelParams.Scale(1 / AvrgModelCount);
                    TFedCenterModel res = model;
                    res.Params = AvrgModelParams;
                    Serialize(IO_WRITE, Sprintf((folder + "fed_avrg_%.8gk.avg").c_str(), SaveModelBatches / 1000.), res);
                    AvrgModelParams.FillZero();
                    AvrgModelCount = 0;
                }
                SaveModelBatches = (floor(model.TrainBatchCount / Interval) + 1) * Interval;
                Log("new save model batches border %g", SaveModelBatches);
            }
        }
    };


private:
    void RunFedCenter(const TString &folder)
    {
        TDescentConfig descentConfig = ModelConfig.MakeDescentConfig();
        // accept connections
        TIntrusivePtr<TSocketEvent> ev = new TSocketEvent();
        TIntrusivePtr<ITcpAccept> gradAccept = Net->StartAccept(FED_GRAD_PORT, FedToken, ev);
        TIntrusivePtr<TTcpRecvQueue> gradQueue = new TTcpRecvQueue(ev);

        // http server
        TIntrusivePtr<THttpServer> srv(new THttpServer(FED_HTTP_PORT));

        // remove failed saves
        EraseTmpFiles(folder);

        // center state
        TFedCenterCtx ctx(Net);
        LoadCenterState(folder, &ctx.GetCenterState());

        // checkpoint
        LoadLastCheckpoint(folder, &ctx.GetModel());

        // init basepoint and sum delta
        if (ctx.GetModel().IsEmpty()) {
            Log("starting from config model");
            ctx.InitModel(ModelConfig.StartParams.Release(), StartTrainBatchCount, descentConfig);
        } else {
            ModelConfig.StartParams = 0;
        }
        ctx.InitBasepoint(DataServer->GetStats().VocabSize);

        // average model
        TAvrgModelComputer avrgModel;
        if (SaveModelInterval > 0) {
            avrgModel.Init(ctx.GetModel(), SaveModelInterval);
        }

        // information for workers
        TFedParams fedParams;
        fedParams.DescentConfig = descentConfig;
        fedParams.Compression = DataServer->GetStats().Compression;
        fedParams.DataServerAddr = DataServerAddr;
        fedParams.FedWeightScale = FedWeightScale;

        // timeout
        NHPTimer::STime tCurrent;
        NHPTimer::GetTime(&tCurrent);
        double collectTime = 0;

        THashMap<TIntrusivePtr<ITcpConnection>,bool> newConn;

        // collect updates and send basepoints
        Log("start serving queries");
        printf("start serving queries\n"); fflush(0);
        TTcpPoller poller;
        for (;;) {
            // do not consume 100% cpu when nothing is happening
            poller.Start();
            srv->Poll(&poller);
            poller.AddSocket(ev->GetSocket(), POLLRDNORM);

            poller.Poll(1);
            ev->Reset();

            TVector<THttpServer::TRequest> qArr;
            poller.Start();
            srv->OnPoll(&poller, &qArr);
            poller.CheckSocket(ev->GetSocket());

            double deltaT = NHPTimer::GetTimePassed(&tCurrent);

            // http commands
            for (THttpServer::TRequest &q : qArr) {
                Log("Http query %s", q.Req.Req.c_str());
                if (q.Req.Req == "") {
                    TFedInfo info;
                    info.TimeSinceLastDelta = collectTime;
                    ctx.CollectInfo(&info);
                    TString html;
                    RenderRootPage(info, &html);
                    q.ReplyHTML(html);
                } else if (q.Req.Req == "log") {
                    yint maxLines = q.Req.GetIntParam("max_lines");
                    maxLines = Max<yint>(20, maxLines);
                    TString html;
                    RenderLog(maxLines, &html);
                    q.ReplyHTML(html);
                } else {
                    q.ReplyNotFound();
                }
            }

            // erase failed connection attempts
            for (auto it = newConn.begin(); it != newConn.end();) {
                if (it->first->IsValid()) {
                    ++it;
                } else {
                    auto del = it++;
                    newConn.erase(del);
                }
            }

            // accept new gradient connections
            TIntrusivePtr<ITcpConnection> conn;
            while (gradAccept->GetNewConnection(&conn)) {
                Log("got connection from %s", conn->GetPeerAddress().c_str());
                conn->SetExitOnError(false);
                Net->StartSendRecv(conn, gradQueue);
                newConn[conn];
            }

            // process gradient requests
            TIntrusivePtr<TTcpPacketReceived> recvPkt;
            while (gradQueue->Dequeue(&recvPkt)) {
                TIntrusivePtr<ITcpConnection> conn = recvPkt->Conn;
                if (newConn.find(conn) != newConn.end()) {
                    newConn.erase(newConn.find(conn));
                    // process login
                    TFedLogin login;
                    SerializeMem(IO_READ, &recvPkt->Data, login);
                    ctx.NewLogin(conn, login, &fedParams);
                } else {
                    ctx.ReceiveDelta(recvPkt);
                }
            }

            // send new base point if time has come
            bool isTimeout = (collectTime > DeltaCollectTimeout);
            bool deltaCollected = ctx.CheckIfAllWorkersSentDelta(isTimeout);
            collectTime += deltaT;
            if (ctx.IsIdle()) {
                collectTime = 0;
            } else if (deltaCollected) {
                if (collectTime >= DeltaCollectMinInterval) {
                    collectTime = 0;
                    Log("delta collected, model version %d", (int)Version);
                    ctx.CreateNewBasepoint(FedWeightScale);
                    ctx.SendNewBasepoint();
                    SaveCheckpoint(ctx.GetModel(), folder);
                    SaveCenterState(folder, ctx.GetCenterState());
                    if (SaveModelInterval > 0) {
                        avrgModel.Add(folder, ctx.GetModel());
                    }
                }
            }
        }
    }

public:
    TFedCenter()
    {
        Net = CreateTcpSendRecv();
    }

    void ParseScript(const TString &scriptText)
    {
        TConfigFile cfg;
        ParseConfig(&cfg, scriptText);

        for (const TConfigFile::TOp &op : cfg.OpArr) {
            if (op.Op == CFG_OP_ASSIGNMENT) {
                if (op.Dst == "DELTA_COLLECT_TIMEOUT") {
                    DeltaCollectTimeout = atof(op.Args[0].c_str());
                } else if (op.Dst == "DELTA_COLLECT_MIN_INTERVAL") {
                    DeltaCollectMinInterval = atof(op.Args[0].c_str());
                } else if (op.Dst == "KEEP_MODEL_COUNT") {
                    KeepModelCount = atof(op.Args[0].c_str());
                } else if (op.Dst == "SAVE_MODEL_INTERVAL") {
                    SaveModelInterval = atof(op.Args[0].c_str());
                } else if (op.Dst == "START_TRAIN_BATCH_COUNT") {
                    StartTrainBatchCount = atof(op.Args[0].c_str());
                } else if (op.Dst == "FED_WEIGHT_SCALE") {
                    FedWeightScale = atof(op.Args[0].c_str());
                } else if (ModelConfig.ParseScriptOp(op, DataServer)) {
                    ;
                } else {
                    DebugPrintf("unknown config variable %s\n", op.Dst.c_str());
                    abort();
                }

            } else if (op.Op == CFG_OP_CALL) {
                if (op.Dst == "run_fed_center") {
                    Y_VERIFY(YSize(op.Args) == 1);
                    TString folder = op.Args[0];
                    if (!folder.empty() && !EndsWith(folder, "/")) {
                        folder += '/';
                    }
                    RunFedCenter(folder);

                } else if (ModelConfig.ParseScriptOp(op, DataServer)) {
                    ;

                } else {
                    DebugPrintf("unknown function %s\n", op.Dst.c_str());
                    abort();
                }
            }
        }
    }

    void SetDataSource(const TString &addr)
    {
        DataServerAddr = addr;
        DataServer = ConnectDataServer(Net, DataServerAddr);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    TString dataSourceAddr = "10.10.10.10";

    TOpt cmdline("d:s:", argc, argv);
    for (const TOpt::TParam &param : cmdline.Params) {
        if (param.Name == "s") {
            Log("Fed script %s", param.Args[0].c_str());
            TVector<char> cfg;
            Y_VERIFY(ReadWholeFile(param.Args[0], &cfg));
            Y_VERIFY(!cfg.empty() && "empty config");
            FED_SCRIPT = cfg.data();
        } else if (param.Name == "d") {
            dataSourceAddr = param.Args[0];
        }
    }

    // execute config script
    TFedCenter fed;
    fed.SetDataSource(dataSourceAddr);
    fed.ParseScript(FED_SCRIPT);

    return 0;
}
