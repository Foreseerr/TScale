#include "stdafx.h"
#include "tcp_net.h"
#include "net_util.h"
#include "ip_address.h"
#include "poller.h"
#include <lib/hp_timer/hp_timer.h>
#include <util/thread.h>

namespace NNet
{
const float CONNECT_TIMEOUT = 1;

static void MakeFastSocket(SOCKET s)
{
    // setting these buffers to any value somehow reduces window to 501 and destroys bandwidth
    //int bufSize = 1 << 20;
    //setsockopt(s, SOL_SOCKET, SO_SNDBUF, (char *)&bufSize, sizeof(int));
    //bufSize *= 2; // larger rcv buffer
    //setsockopt(s, SOL_SOCKET, SO_RCVBUF, (char *)&bufSize, sizeof(int));
    SetNoTcpDelay(s);
    MakeNonBlocking(s);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void TTcpRecvQueue::Enqueue(TIntrusivePtr<TTcpPacketReceived> pkt)
{
    RecvList.Enqueue(pkt);
    if (Event.Get()) {
        Event->Set();
    }
}


bool TTcpRecvQueue::Dequeue(TIntrusivePtr<TTcpPacketReceived> *p)
{
    return RecvList.DequeueFirst(p);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
class TTcpConnection : public ITcpConnection
{
    struct TTcpPacketHeader
    {
        yint Size;
    };

private:
    SOCKET Sock = INVALID_SOCKET;
    sockaddr_in PeerAddr;
    TIntrusivePtr<TTcpRecvQueue> RecvQueue;
    TSingleConsumerJobQueue<TIntrusivePtr<TTcpPacket>> SendQueue;
    volatile bool HasSendOps = false;
    volatile bool StopFlag = false;
    volatile bool ExitOnError = true;

    // recv data
    TTcpPacketHeader RecvHeader;
    yint RecvHeaderOffset = 0;
    TIntrusivePtr<TTcpPacketReceived> RecvPacket;
    yint RecvOffset = -1;

    // send data
    TTcpPacketHeader SendHeader;
    yint SendHeaderOffset = 0;
    TVector<TIntrusivePtr<TTcpPacket>> SendArr;
    yint SendOffset = -1;

private:
    ~TTcpConnection()
    {
        closesocket(Sock);
    }

    void OnFail(const TString &err)
    {
        if (Sock != INVALID_SOCKET) {
            closesocket(Sock);
            Sock = INVALID_SOCKET;
        }
        if (ExitOnError) {
            DebugPrintf("tcp connection failed\n");
            DebugPrintf("%s\n", err.c_str());
            fflush(0);
            exit(0);
        } else {
            StopFlag = true;
        }
    }

    int CheckRecvRetVal(int rv, const char *op)
    {
        if (rv == 0) {
            OnFail(Sprintf("%s, rv == 0, connection closed?\n", op));
            return 0;
        } else if (rv == SOCKET_ERROR) {
            yint err = errno;
            if (err != 0 && err != EWOULDBLOCK && err != EAGAIN) {
                OnFail(Sprintf("%s fail, rv %g, errno %g\n", op, rv * 1., err * 1.));
            }
            return 0;
        }
        return rv;
    }

    int CheckSendRetVal(int rv, const char *op)
    {
        if (rv == SOCKET_ERROR) {
            yint err = errno;
            if (err != 0 && err != EWOULDBLOCK && err != EAGAIN) {
                OnFail(Sprintf("%s fail, rv %g, errno %g\n", op, rv * 1., err * 1.));
            }
            return 0;
        }
        return rv;
    }

    void DoRecv()
    {
        if (RecvOffset == -1) {
            Y_ASSERT(RecvPacket == nullptr);
            char *data = (char *)&RecvHeader;
            int headerSize = sizeof(TTcpPacketHeader);
            int rv = recv(Sock, data + RecvHeaderOffset, headerSize - RecvHeaderOffset, 0);
            rv = CheckRecvRetVal(rv, "recv header");
            RecvHeaderOffset += rv;
            if (RecvHeaderOffset == headerSize) {
                RecvPacket = new TTcpPacketReceived();
                RecvPacket->Data.resize(RecvHeader.Size);
                RecvOffset = 0;
                RecvHeaderOffset = 0;
                if (RecvHeader.Size == 0) {
                    DoRecv();
                }
            }
        } else {
            yint sz = YSize(RecvPacket->Data) - RecvOffset;
            int rv = 0;
            if (sz > 0) {
                int szInt = Min<yint>(1 << 24, sz);
                rv = recv(Sock, (char *)RecvPacket->Data.data() + RecvOffset, szInt, 0);
                rv = CheckRecvRetVal(rv, "recv data");
            }
            if (rv == sz) {
                RecvOffset = -1;
                RecvPacket->Conn = this; // assign connection here to avoid self reference
                RecvQueue->Enqueue(RecvPacket.Release());
            } else {
                RecvOffset += rv;
            }
        }
    }

    void DoSend()
    {
        if (SendArr.empty()) {
            HasSendOps = false;
            SendQueue.DequeueAll(&SendArr);
            Reverse(SendArr.begin(), SendArr.end());
            if (!SendArr.empty()) {
                HasSendOps = true;
            }
        }
        if (!SendArr.empty()) {
            if (SendOffset < 0) {
                if (SendHeaderOffset == 0) {
                    SendHeader.Size = YSize(SendArr[0]->Data);
                }
                char *data = (char *)&SendHeader;
                int headerSize = sizeof(TTcpPacketHeader);
                int rv = send(Sock, data + SendHeaderOffset, headerSize - SendHeaderOffset, 0);
                rv = CheckSendRetVal(rv, "send header");
                SendHeaderOffset += rv;
                if (SendHeaderOffset == headerSize) {
                    SendHeaderOffset = 0;
                    SendOffset = 0;
                    DoSend();
                }
            } else {
                const TVector<ui8> &data = SendArr[0]->Data;
                yint sz = YSize(data) - SendOffset;
                int rv = 0;
                if (sz > 0) {
                    int szInt = Min<yint>(1 << 24, sz);
                    rv = send(Sock, (const char *)data.data() + SendOffset, szInt, 0);
                    rv = CheckSendRetVal(rv, "send data");
                }
                if (rv == sz) {
                    SendArr.erase(SendArr.begin());
                    SendOffset = -1;
                } else {
                    SendOffset += rv;
                }
            }
        }
    }


public:
    void Poll(TTcpPoller *pl)
    {
        yint events = HasSendOps ? (POLLRDNORM | POLLWRNORM) : POLLRDNORM;
        pl->AddSocket(Sock, events);
    }

    void OnPoll(TTcpPoller *pl)
    {
        yint events = pl->CheckSocket(Sock);
        if (events & ~(POLLRDNORM | POLLWRNORM)) {
            OnFail(Sprintf("Nontrivial events %x", events));
            return;
        }
        if (events & POLLRDNORM) {
            DoRecv();
        }
        if (events & POLLWRNORM) {
            DoSend();
        }
    }

    void Bind(TIntrusivePtr<TTcpRecvQueue> recvQueue)
    {
        RecvQueue = recvQueue;
    }

    void Send(TIntrusivePtr<TTcpPacket> pkt)
    {
        SendQueue.Enqueue(pkt);
        HasSendOps = true;
    }

public:
    TTcpConnection(SOCKET s, const sockaddr_in &peerAddr) : Sock(s), PeerAddr(peerAddr)
    {
    }

    // connect
    TTcpConnection(const TString &hostName, yint defaultPort, const TGuid &token)
    {
        TString pureHost;
        if (!ParseInetName(&PeerAddr, &pureHost, hostName.c_str(), defaultPort)) {
            DebugPrintf("Failed to parse server address %s\n", hostName.c_str());
            abort();
        }
        Sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (Sock == INVALID_SOCKET) {
            DebugPrintf("socket() failed\n");
            abort();
        }
        if (connect(Sock, (sockaddr *)&PeerAddr, sizeof(PeerAddr)) == SOCKET_ERROR) {
            DebugPrintf("Failed to connect to %s\n", hostName.c_str());
            abort();
        }
        MakeFastSocket(Sock);
        int rv = send(Sock, (const char *)&token, sizeof(token), 0);
        Y_VERIFY(rv == sizeof(token) && "should always be able to send few bytes after connect()");
    }

    virtual TString GetPeerAddress() override
    {
        return GetAddressString(PeerAddr);
    }

    void SetExitOnError(bool b) override
    {
        ExitOnError = b;
    }

    void Stop() override
    {
        StopFlag = true;
    }

    bool IsValid() override
    {
        return !StopFlag;
    }

    TTcpConnection *GetImpl() override
    {
        return this;
    }
};


TIntrusivePtr<ITcpConnection> Connect(const TString &hostName, yint defaultPort, const TGuid &token)
{
    return new TTcpConnection(hostName, defaultPort, token);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TTcpAccept : public ITcpAccept
{
    struct TConnectAttempt
    {
        SOCKET Sock = INVALID_SOCKET;
        sockaddr_in PeerAddr;
        float TimePassed = 0;

    public:
        TConnectAttempt() {}
        TConnectAttempt(SOCKET s, sockaddr_in peerAddr) : Sock(s), PeerAddr(peerAddr) {}
    };

    SOCKET Listen = INVALID_SOCKET;
    yint MyPort = 0;
    TGuid Token;
    TIntrusivePtr<ISyncEvent> Event;
    TSingleConsumerJobQueue<TIntrusivePtr<TTcpConnection>> NewConn;
    TVector<TConnectAttempt> AttemptArr;
    NHPTimer::STime TCurrent;
    volatile bool StopFlag = false;


private:
    ~TTcpAccept()
    {
        for (auto &x : AttemptArr) {
            closesocket(x.Sock);
        }
        closesocket(Listen);
    }

    void DoAccept()
    {
        sockaddr_in incomingAddr;
        socklen_t nIncomingAddrLen = sizeof(incomingAddr);
        SOCKET s = accept(Listen, (sockaddr*)&incomingAddr, &nIncomingAddrLen);
        if (s == INVALID_SOCKET) {
            int err = errno;
            // somehow errno 0 can happen on windows
            if (err != 0 && err != EWOULDBLOCK && err != EAGAIN) {
                DebugPrintf("accept() failed for signaled socket, errno %d\n", err);
                abort();
            }
        } else {
            MakeFastSocket(s);
            AttemptArr.push_back(TConnectAttempt(s, incomingAddr));
        }
    }

public:
    bool IsValid()
    {
        return !StopFlag;
    }

    void Poll(TTcpPoller *pl)
    {
        for (yint k = 0; k < YSize(AttemptArr); ++k) {
            TConnectAttempt &att = AttemptArr[k];
            pl->AddSocket(att.Sock, POLLRDNORM);
        }
        pl->AddSocket(Listen, POLLRDNORM);
    }

    void OnPoll(TTcpPoller *pl)
    {
        float deltaT = NHPTimer::GetTimePassed(&TCurrent);
        deltaT = ClampVal<float>(deltaT, 0, 0.5); // avoid spurious too large time steps

        yint dst = 0;
        for (yint k = 0; k < YSize(AttemptArr); ++k) {
            TConnectAttempt &att = AttemptArr[k];
            yint events = pl->CheckSocket(att.Sock);
            if (events & POLLRDNORM) {
                TGuid chk;
                int rv = recv(att.Sock, (char *)&chk, sizeof(TGuid), 0);
                if (rv == sizeof(TGuid) && chk == Token) {
                    NewConn.Enqueue(new TTcpConnection(att.Sock, att.PeerAddr));
                    if (Event.Get()) {
                        Event->Set();
                    }
                } else {
                    closesocket(att.Sock);
                }
            } else {
                att.TimePassed += deltaT;
                if (att.TimePassed <= CONNECT_TIMEOUT) {
                    AttemptArr[dst++] = att;
                } else {
                    closesocket(att.Sock);
                }
            }
        }
        AttemptArr.resize(dst);

        yint events = pl->CheckSocket(Listen);
        if ((events) & ~(POLLRDNORM | POLLWRNORM)) {
            DebugPrintf("Nontrivial accept events %x\n", events); fflush(0);
            Stop();
        } else if (events & POLLRDNORM) {
            DoAccept();
        }
    }

public:
    TTcpAccept(yint listenPort, const TGuid &token, TIntrusivePtr<ISyncEvent> ev) : Token(token), Event(ev)
    {
        Listen = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (Listen == INVALID_SOCKET) {
            DebugPrintf("Failed to create socket\n");
            abort();
        }
        sockaddr_in name;
        Zero(name);
        name.sin_family = AF_INET;
        //name.sin_addr = inaddr_any;
        name.sin_port = htons(listenPort);
        if (bind(Listen, (sockaddr *)&name, sizeof(name)) != 0) {
            DebugPrintf("Port %d already in use\n", ntohs(name.sin_port));
            abort();
        }
        if (listen(Listen, SOMAXCONN) != 0) {
            DebugPrintf("listen() failed\n");
            abort();
        }
        sockaddr_in localAddr;
        socklen_t len = sizeof(localAddr);
        if (getsockname(Listen, (sockaddr *)&localAddr, &len)) {
            Y_VERIFY(0 && "no self address");
        }
        MyPort = ntohs(localAddr.sin_port);
        MakeNonBlocking(Listen);
    }

    bool GetNewConnection(TIntrusivePtr<ITcpConnection> *p) override
    {
        TIntrusivePtr<TTcpConnection> conn;
        bool res = NewConn.DequeueFirst(&conn);
        *p = conn.Get();
        return res;
    }

    yint GetPort() override
    {
        return MyPort;
    }

    void Stop() override
    {
        StopFlag = true;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TTcpSendRecv : public ITcpSendRecv
{
    TThread Thr;
    TTcpPoller Poller;
    TIntrusivePtr<TSocketEvent> NewEvent;
    TSingleConsumerJobQueue<TIntrusivePtr<TTcpConnection>> NewConn;
    TSingleConsumerJobQueue<TIntrusivePtr<TTcpAccept>> NewListen;
    THashMap<TIntrusivePtr<TTcpConnection>, bool> ConnSet;
    THashMap<TIntrusivePtr<TTcpAccept>, bool> ListenSet;
    volatile bool Exit = false;

private:
    template <class T>
    void PollSet(T &coll)
    {
        for (auto it = coll.begin(); it != coll.end();) {
            if (it->first->IsValid()) {
                it->first->Poll(&Poller);
                ++it;
            } else {
                auto del = it++;
                coll.erase(del);
            }
        }
    }

    template <class T>
    void OnPollResults(T &coll)
    {
        for (auto it = coll.begin(); it != coll.end(); ++it) {
            it->first->OnPoll(&Poller);
        }
    }

    void ProcessQeueues()
    {
        TVector<TIntrusivePtr<TTcpConnection>> newConnArr;
        NewConn.DequeueAll(&newConnArr);
        for (auto x : newConnArr) {
            ConnSet[x];
        }

        TVector<TIntrusivePtr<TTcpAccept>> newListenArr;
        NewListen.DequeueAll(&newListenArr);
        for (auto x : newListenArr) {
            ListenSet[x];
        }
    }

    ~TTcpSendRecv()
    {
        Exit = true;
        NewEvent->Set();
        Thr.Join();
    }

public:
    TTcpSendRecv()
    {
        NewEvent = new TSocketEvent();
        Thr.Create(this);
    }

    void WorkerThread()
    {
        while (!Exit) {
            Poller.Start();
            PollSet(ConnSet);
            PollSet(ListenSet);
            Poller.AddSocket(NewEvent->GetSocket(), POLLRDNORM);

            float timeoutSec = 1;
            Poller.Poll(timeoutSec);

            Poller.Start();
            OnPollResults(ConnSet);
            OnPollResults(ListenSet);
            if (Poller.CheckSocket(NewEvent->GetSocket()) & POLLRDNORM) {
                NewEvent->Reset();
                ProcessQeueues();
            }
        }
    }

    void StartSendRecv(TIntrusivePtr<ITcpConnection> connArg, TIntrusivePtr<TTcpRecvQueue> q) override
    {
        TIntrusivePtr<TTcpConnection> conn = connArg->GetImpl();
        conn->Bind(q);
        NewConn.Enqueue(conn);
        NewEvent->Set();
    }

    TIntrusivePtr<ITcpAccept> StartAccept(yint port, const TGuid &token, TIntrusivePtr<ISyncEvent> ev) override
    {
        TIntrusivePtr<TTcpAccept> res = new TTcpAccept(port, token, ev);
        NewListen.Enqueue(res);
        NewEvent->Set();
        return res.Get();
    }

    void Send(TIntrusivePtr<ITcpConnection> connArg, TIntrusivePtr<TTcpPacket> pkt) override
    {
        connArg->GetImpl()->Send(pkt);
        NewEvent->Set();
    }
};


TIntrusivePtr<ITcpSendRecv> CreateTcpSendRecv()
{
    return new TTcpSendRecv();
}
}
