#pragma once
#include <lib/guid/guid.h>
#include <util/mem_io.h>
#include <util/thread.h>
#include <util/event.h>


namespace NNet
{
struct TTcpPacket : public TThrRefBase
{
    TVector<ui8> Data;
};

template <class T>
TIntrusivePtr<TTcpPacket> MakePacket(T &data)
{
    TIntrusivePtr<TTcpPacket> pkt = new TTcpPacket;
    SerializeMem(IO_WRITE, &pkt->Data, data);
    return pkt;
}


class TTcpConnection;
struct ITcpConnection : public TThrRefBase
{
    virtual TTcpConnection *GetImpl() = 0;
    virtual TString GetPeerAddress() = 0;
    virtual void SetExitOnError(bool b) = 0;
    virtual void Stop() = 0;
    virtual bool IsValid() = 0;
};
TIntrusivePtr<ITcpConnection> Connect(const TString &hostName, yint defaultPort, const TGuid &token);


struct TTcpPacketReceived : public TThrRefBase
{
    TIntrusivePtr<ITcpConnection> Conn;
    TVector<ui8> Data;
};


class TTcpRecvQueue : public TThrRefBase
{
    TSingleConsumerJobQueue<TIntrusivePtr<TTcpPacketReceived>> RecvList;
    TIntrusivePtr<ISyncEvent> Event;
public:
    TTcpRecvQueue() {}
    TTcpRecvQueue(TIntrusivePtr<ISyncEvent> ev) : Event(ev) {}
    void Enqueue(TIntrusivePtr<TTcpPacketReceived> pkt);
    bool Dequeue(TIntrusivePtr<TTcpPacketReceived> *p);
};


struct ITcpAccept : public TThrRefBase
{
    virtual bool GetNewConnection(TIntrusivePtr<ITcpConnection> *p) = 0;
    virtual yint GetPort() = 0;
    virtual void Stop() = 0;
};


struct ITcpSendRecv : public TThrRefBase
{
    virtual void StartSendRecv(TIntrusivePtr<ITcpConnection> connArg, TIntrusivePtr<TTcpRecvQueue> q) = 0;
    virtual TIntrusivePtr<ITcpAccept> StartAccept(yint port, const TGuid &token, TIntrusivePtr<ISyncEvent> ev) = 0;
    virtual void Send(TIntrusivePtr<ITcpConnection> connArg, TIntrusivePtr<TTcpPacket> pkt) = 0;
};


TIntrusivePtr<ITcpSendRecv> CreateTcpSendRecv();
}
