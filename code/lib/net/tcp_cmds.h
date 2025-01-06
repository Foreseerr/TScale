#pragma once
#include "tcp_net.h"

namespace NNet
{

typedef ui32 TTypeId;

// TCmd should inherit TThrRefBase and have virtual operator&(IBinSaver&)
struct TCommandBase : public TThrRefBase
{
    virtual int operator&(IBinSaver &f) { return 0; }
};

template <class TCmd>
struct TCommandFabric
{
    typedef TCommandBase *(*CreateObject)();
    THashMap<const std::type_info *, TTypeId, TPtrHash> Type2TypeId;
    THashMap<TTypeId, CreateObject> TypeId2Constructor;
};


#define REGISTER_PACKET(fabric, obj, id)\
    static TCommandBase* Create##id() { return new obj(); }\
    struct TRegisterPacket##id { TRegisterPacket##id() {\
        Y_ASSERT(fabric.TypeId2Constructor.find(id) == fabric.TypeId2Constructor.end());\
        fabric.Type2TypeId[&typeid(obj)] = id;\
        fabric.TypeId2Constructor[id] = Create##id;\
    } } registerPacket##id;


template <class TCmd, class TArg>
TIntrusivePtr<TTcpPacket> SerializeCommand(TCommandFabric<TCmd> &fabric, TArg *cmdArg)
{
    TIntrusivePtr<TCommandBase> cmd = cmdArg;
    TMemStream mem;
    {
        TBufferedStream bufIO(IO_WRITE, mem);
        TCommandBase *cmdPkt = cmd.Get();
        TTypeId objTypeId = fabric.Type2TypeId[&typeid(*cmdPkt)];
        bufIO.Write(&objTypeId, sizeof(objTypeId));
        IBinSaver bs(bufIO);
        bs.Add(cmdPkt);
    }
    TIntrusivePtr<TTcpPacket> res = new TTcpPacket;
    mem.Swap(&res->Data);
    return res;
}

template <class TCmd>
TIntrusivePtr<TCmd> DeserializeCommand(TCommandFabric<TCmd> &fabric, TVector<ui8> *p)
{
    TIntrusivePtr<TCommandBase> cmd;
    {
        TMemStream mem(p);
        TBufferedStream bufIO(IO_READ, mem);
        TTypeId objTypeId = 0;
        bufIO.Read(&objTypeId, sizeof(objTypeId));
        cmd = fabric.TypeId2Constructor[objTypeId]();
        IBinSaver bs(bufIO);
        bs.Add(cmd.Get());
    }
    return static_cast<TCmd*>(cmd.Release());
}


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TCmd>
TIntrusivePtr<TCmd> RecvCommand(TCommandFabric<TCmd> &fabric, TIntrusivePtr<TTcpRecvQueue> q)
{
    TIntrusivePtr<TTcpPacketReceived> pkt;
    if (q->Dequeue(&pkt)) {
        return DeserializeCommand(fabric, &pkt->Data);
    } else {
        return 0;
    }
}


template <class TCmd, class TArg>
void SendCommand(TCommandFabric<TCmd> &fabric, TIntrusivePtr<ITcpSendRecv> net, TIntrusivePtr<ITcpConnection> conn, TArg *cmdArg)
{
    net->Send(conn, SerializeCommand(fabric, cmdArg));
}

}
