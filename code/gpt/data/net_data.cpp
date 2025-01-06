#include "stdafx.h"
#include "net_data.h"
#include <lib/guid/guid.h>
#include <lib/net/tcp_cmds.h>
#include <lib/net/http_client.h>
#include <lib/net/http_server.h>
#include <lib/json/json.h>


using namespace NNet;
using namespace NJson;

const yint JSON_TRAIN_DATA_PORT = 19000;
const yint TRAIN_DATA_PORT = 18183;

static TGuid TrainDataToken(0x1e4da508, 0x5338525c, 0x40b87806, 0x4662301b);


///////////////////////////////////////////////////////////////////////////////////////////////////
namespace NNetData
{

struct IDataCmd : public TCommandBase
{
    virtual TIntrusivePtr<TTcpPacket> Exec(TIntrusivePtr<IDataSource> data) = 0;
};
static TCommandFabric<IDataCmd> cmdFabric;


class TGetStats : public IDataCmd
{
    TIntrusivePtr<TTcpPacket> Exec(TIntrusivePtr<IDataSource> data) override
    {
        IDataSource::TDataStats stats = data->GetStats();
        return MakePacket(stats);
    }
};
REGISTER_PACKET(cmdFabric, TGetStats, 1);


class TSampleFragments : public IDataCmd
{
    IDataSource::ETrainTest TRT = IDataSource::TRAIN;
    yint RngSeed = 1313;
    yint FragCount = 0;
    yint FragLen = 0;
    SAVELOAD_OVERRIDE(TRT, RngSeed, FragCount, FragLen);

    TIntrusivePtr<TTcpPacket> Exec(TIntrusivePtr<IDataSource> data) override
    {
        TVector<TFragment> fragArr;
        data->SampleFragments(TRT, RngSeed, FragCount, FragLen, &fragArr);
        return MakePacket(fragArr);

    }
public:
    TSampleFragments() {}
    TSampleFragments(IDataSource::ETrainTest trt, yint rngSeed, yint fragCount, yint fragLen)
        : TRT(trt), RngSeed(rngSeed), FragCount(fragCount), FragLen(fragLen)
    {
    }
};
REGISTER_PACKET(cmdFabric, TSampleFragments, 2);
}
using namespace NNetData;


///////////////////////////////////////////////////////////////////////////////////////////////////
static void AddArray(TJsonWriter &jw, const TString &name, const TVector<TBPEToken> &text)
{
    jw.AddArray(name);
    for (yint id : text) {
        jw.AddFloat("", id);
    }
    jw.Finish();
}

void RunDataServer(TIntrusivePtr<ITcpSendRecv> net, TIntrusivePtr<IDataSource> data)
{
    TTcpPoller poller;
    TIntrusivePtr<TSocketEvent> ev = new TSocketEvent();
    TIntrusivePtr<ITcpAccept> dataAccept = net->StartAccept(TRAIN_DATA_PORT, TrainDataToken, ev);
    TIntrusivePtr<TTcpRecvQueue> dataQueue = new TTcpRecvQueue(ev);
    TIntrusivePtr<THttpServer> jsonServer = new THttpServer(JSON_TRAIN_DATA_PORT);

    TXRng rng(GetCycleCount());
    for (;;) {
        poller.Start();
        jsonServer->Poll(&poller);
        poller.AddSocket(ev->GetSocket(), POLLRDNORM);

        poller.Poll(1);
        ev->Reset();

        TVector<THttpServer::TRequest> qArr;
        poller.Start();
        jsonServer->OnPoll(&poller, &qArr);
        poller.CheckSocket(ev->GetSocket());

        // process json queries
        for (THttpServer::TRequest &q : qArr) {
            const THttpRequest &req = q.Req;
            //DebugPrintf("http query %s\n", req.Req.c_str());
            TIntrusivePtr<TJson> response = new TJson();
            if (req.Req == "stats") {
                TJsonWriter jw(response);
                jw.AddObject("");
                const IDataSource::TDataStats &stats = data->GetStats();
                jw.AddBool("UsePPM", stats.UsePPM);
                jw.AddBool("UseLMatch", stats.UseLMatch);
                jw.AddFloat("Compression", stats.Compression);
                jw.AddFloat("VocabSize", stats.VocabSize);
                jw.AddFloat("DocStartToken", stats.DocStartToken);
                jw.AddFloat("FragmentStartToken", stats.FragmentStartToken);
                jw.AddArray("Bias");
                for (float w : stats.Bias) {
                    jw.AddFloat("", w);
                }
                jw.Finish();
                jw.AddBool("HasTest", stats.HasTest);
                jw.Finish();

            } else if (req.Req == "fragments") {
                IDataSource::ETrainTest trt = (IDataSource::ETrainTest)req.GetIntParam("trt");
                yint rngSeed = req.GetIntParam("seed");
                yint count = req.GetIntParam("count");
                yint len = req.GetIntParam("len");
                TVector<TFragment> fragArr;
                data->SampleFragments(trt, rngSeed, count, len, &fragArr);
                TJsonWriter jw(response);
                jw.AddArray("");
                for (const TFragment &frag : fragArr) {
                    jw.AddObject("");
                    AddArray(jw, "Text", frag.Text);
                    if (frag.PPM.empty()) {
                        AddArray(jw, "PPM", frag.PPM);
                    }
                    if (frag.LMatch.empty()) {
                        AddArray(jw, "LMatch", frag.LMatch);
                    }
                    AddArray(jw, "Target", frag.Target);
                    jw.Finish();
                }

            } else {
                q.ReplyNotFound();
                continue;
            }
            //DebugPrintf("response json:\n%s\n", RenderString(response).c_str());
            q.ReplyJson(Render(response));
        }

        // accept new data connections
        TIntrusivePtr<ITcpConnection> conn;
        while (dataAccept->GetNewConnection(&conn)) {
            conn->SetExitOnError(false);
            net->StartSendRecv(conn, dataQueue);
        }

        // process data requests
        TIntrusivePtr<TTcpPacketReceived> pkt;
        while (dataQueue->Dequeue(&pkt)) {
            TIntrusivePtr<IDataCmd> cmd = DeserializeCommand(cmdFabric, &pkt->Data);
            net->Send(pkt->Conn, cmd->Exec(data));
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
static void GetResponse(TIntrusivePtr<TSyncEvent> ev, TIntrusivePtr<TTcpRecvQueue> q, T *pRes)
{
    for (;;) {
        ev->Wait();
        TIntrusivePtr<TTcpPacketReceived> pkt;
        if (q->Dequeue(&pkt)) {
            SerializeMem(IO_READ, &pkt->Data, *pRes);
            return;
        }
    }
}

class TNetDataSource : public IDataSource
{
    TIntrusivePtr<ITcpSendRecv> Net;
    TIntrusivePtr<TSyncEvent> Event;
    TIntrusivePtr<TTcpRecvQueue> DataQueue;
    TIntrusivePtr<ITcpConnection> DataConn;
    TDataStats DataStats;

    const TDataStats &GetStats() const override
    {
        return DataStats;
    }
    void SampleFragments(ETrainTest trt, yint rngSeed, yint fragCount, yint len, TVector<TFragment> *pFragArr) override
    {
        SendCommand(cmdFabric, Net, DataConn, new TSampleFragments(trt, rngSeed, fragCount, len));
        GetResponse(Event, DataQueue, pFragArr);
    }
public:
    TNetDataSource(TIntrusivePtr<ITcpSendRecv> net, const TString &addr) : Net(net)
    {
        Event = new TSyncEvent();
        DataQueue = new TTcpRecvQueue(Event);
        DataConn = Connect(addr, TRAIN_DATA_PORT, TrainDataToken);
        Net->StartSendRecv(DataConn, DataQueue);

        SendCommand(cmdFabric, Net, DataConn, new TGetStats());
        GetResponse(Event, DataQueue, &DataStats);
    }
};


TIntrusivePtr<IDataSource> ConnectDataServer(TIntrusivePtr<ITcpSendRecv> net, const TString &addr)
{
    return new TNetDataSource(net, addr);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static void ParseArray(TJsonIterator &f, TVector<TBPEToken> *pRes)
{
    for (TJsonIterator w = f.Expand(); w.IsValid(); w.Next()) {
        pRes->push_back(w.GetFloatValue());
    }
}

class THttpDataSource : public IDataSource
{
    TIntrusivePtr<THttpConnection> Conn;
    TDataStats Stats;

    const TDataStats &GetStats() const override
    {
        return Stats;
    }
    void SampleFragments(ETrainTest trt, yint rngSeed, yint fragCount, yint len, TVector<TFragment> *pFragArr) override
    {
        TString q = Sprintf("/fragments?trt=%g&seed=%llu&count=%g&len=%g", trt * 1., rngSeed & 0x7fffffffffffffffull, fragCount * 1., len * 1.);
        TVector<char> frags;
        bool ok = Conn->FetchJson(q, TVector<char>(), &frags);
        Y_VERIFY(ok);
        TIntrusivePtr<TJson> json = ParseJson(frags);
        TJsonIterator root(json);
        Y_VERIFY(root.IsArray());
        for (TJsonIterator it = root.Expand(); it.IsValid(); it.Next()) {
            TFragment frag;
            for (TJsonIterator f = it.Expand(); f.IsValid(); f.Next()) {
                TString field = f.GetName();
                if (field == "Text") {
                    ParseArray(f, &frag.Text);
                } else if (field == "PPM") {
                    ParseArray(f, &frag.PPM);
                } else if (field == "LMatch") {
                    ParseArray(f, &frag.LMatch);
                } else if (field == "Target") {
                    ParseArray(f, &frag.Target);
                }
            }
            Y_VERIFY(!frag.Text.empty());
            if (frag.Target.empty()) {
                for (yint t = 1; t < YSize(frag.Text); ++t) {
                    frag.Target.push_back(frag.Text[t]);
                }
                frag.Text.pop_back();
            }
            Y_VERIFY(YSize(frag.Text) <= len);
            // replace first token with FragmentStartToken if needed
            if (Stats.FragmentStartToken >= 0 && YSize(frag.Text) > 0) {
                frag.Text[0] = Stats.FragmentStartToken;
            }
            pFragArr->push_back(frag);
        }
        Y_VERIFY(YSize(*pFragArr) == fragCount);
    }
public:
    THttpDataSource(const TString &addr) : Conn(new THttpConnection(addr, JSON_TRAIN_DATA_PORT))
    {
        TVector<char> stats;
        bool ok = Conn->FetchJson("/stats", TVector<char>(), &stats);
        Y_VERIFY(ok && "failed to query data source stats");
        TIntrusivePtr<TJson> json = ParseJson(stats);
        TJsonIterator root(json);
        Y_VERIFY(root.IsObject());
        for (TJsonIterator f = root.Expand(); f.IsValid(); f.Next()) {
            TString field = f.GetName();
            if (field == "UsePPM") {
                Stats.UsePPM = f.GetBoolValue();
            } else if (field == "UseLMatch") {
                Stats.UseLMatch = f.GetBoolValue();
            } else if (field == "Compression") {
                Stats.Compression = f.GetFloatValue();
            } else if (field == "VocabSize") {
                Stats.VocabSize = f.GetFloatValue();
            } else if (field == "DocStartToken") {
                Stats.DocStartToken = f.GetFloatValue();
            } else if (field == "FragmentStartToken") {
                Stats.FragmentStartToken = f.GetFloatValue();
            } else if (field == "HasTest") {
                Stats.HasTest = f.GetBoolValue();
            } else if (field == "Bias") {
                for (TJsonIterator b = f.Expand(); b.IsValid(); b.Next()) {
                    Stats.Bias.push_back(b.GetFloatValue());
                }
            } else {
                DebugPrintf("Unknown stats field %s\n", field.c_str());
            }
        }
        Y_VERIFY(YSize(Stats.Bias) == Stats.VocabSize);
    }
};


TIntrusivePtr<IDataSource> ConnectHttpDataServer(const TString &addr)
{
    return new THttpDataSource(addr);
}
