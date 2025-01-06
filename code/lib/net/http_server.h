#pragma once
#include "poller.h"
#include "http_request.h"
#include <lib/hp_timer/hp_timer.h>

namespace NNet
{
struct THttpPacket;

class THttpServer : public TThrRefBase
{
    SOCKET Listen = INVALID_SOCKET;
    int ListenPort = 0;

public:
    class TRequest;

private:
    struct TConnection : public TThrRefBase
    {
        SOCKET s = INVALID_SOCKET;
        bool Keepalive = false;
        TVector<char> RecvBuffer;
        yint RecvOffset = 0;
        yint RecvReqId = 0;
        TVector<char> SendBuffer;
        yint SendOffset = 0;
        yint SendReqId = 0;
        yint LastReqId = 1;
        THashMap<yint, TVector<char>> OutOfOrderReplies;

        TConnection(SOCKET ss) : s(ss) {}
        ~TConnection()
        {
            Close();
        }
        void RecvQueries(TVector<THttpServer::TRequest> *pReqArr);
        void SendBufferedData();
        void SendReply(yint reqId, TVector<char> *pData);
        void Close()
        {
            if (s != INVALID_SOCKET) {
                closesocket(s);
                s = INVALID_SOCKET;
            }
        }
        bool IsComplete() const
        {
            return !Keepalive && SendBuffer.empty() && SendReqId == LastReqId;
        }
        bool IsValid() { return s != INVALID_SOCKET; }
        bool IsKeepalive() const { return Keepalive; }
    };
    TVector<TIntrusivePtr<TConnection>> ConnArr;

public:
    class TRequest
    {
        TIntrusivePtr<TConnection> Conn;
        yint ReqId = 0;

        void Reply(const TString &reply, const TVector<char> &data, const char *content);
    public:
        THttpRequest Req;

        void ReplyNotFound();
        void ReplyXML(const TString &reply);
        void ReplyHTML(const TString &reply);
        void ReplyPlainText(const TString &reply);
        void ReplyJson(const TVector<char> &data);
        void ReplyBin(const TVector<char> &data);
        void ReplyBMP(const TVector<char> &data);

        friend class THttpServer;
    };

private:
    ~THttpServer();

public:
    THttpServer(int port);
    int GetPort() const { return ListenPort; }
    void Poll(TTcpPoller *pl);
    void OnPoll(TTcpPoller *pl, TVector<TRequest> *pReqArr);
};


inline void GetQueries(float timeout, TTcpPoller *poller, TIntrusivePtr<THttpServer> p, TVector<THttpServer::TRequest> *pQueries)
{
    poller->Start();
    p->Poll(poller);
    poller->Poll(timeout);
    poller->Start();
    p->OnPoll(poller, pQueries);
}
}
