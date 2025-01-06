#include "stdafx.h"
#include "network.h"
#include <lib/net/tcp_net.h>


namespace NNet
{

void TMasterNetBase::ConnectWorkers(const TVector<TString> &workerList, const TGuid &token)
{
    yint count = YSize(workerList);
    for (TNetRank rank = 0; rank < count; ++rank) {
        DebugPrintf("connect (%s)\n", workerList[rank].c_str());
        TIntrusivePtr<ITcpConnection> conn = Connect(workerList[rank], DEFAULT_WORKER_PORT, token);
        Net->StartSendRecv(conn, Queue);
        SendData(Net, conn, rank);
        WorkerSet[conn] = rank;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void TMasterConnection::ConnectMaster(TIntrusivePtr<ITcpSendRecv> net, yint port, const TGuid &token)
{
    Net = net;
    DebugPrintf("waiting master connect on port %g\n", port * 1.);
    TIntrusivePtr<TSyncEvent> ev = new TSyncEvent();
    TIntrusivePtr<ITcpAccept> acc = Net->StartAccept(port, token, ev);
    while (!acc->GetNewConnection(&Conn)) {
        ev->Wait();
    }
    acc->Stop();
    Queue = new TTcpRecvQueue;
    Net->StartSendRecv(Conn, Queue);
    WaitData(Queue, Conn, &MyRank);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void TP2PNetwork::Send(TNetRank rank, TIntrusivePtr<TTcpPacket> pkt)
{
    if (rank == MyRank) {
        DebugPrintf("send packet to self?\n");
        TIntrusivePtr<TTcpPacketReceived> recvPkt = new TTcpPacketReceived();
        recvPkt->Data.swap(pkt->Data);
        Queue->Enqueue(recvPkt);
    } else {
        Net->Send(Peers[rank], pkt);
    }
}


void TP2PNetwork::ConnectP2P(TNetRank myRank, const TVector<TString> &peerList, const TGuid &token)
{
    Y_ASSERT(Peers.empty());
    DebugPrintf("connect peers\n");
    MyRank = myRank;
    yint peerCount = YSize(peerList);
    Peers.resize(peerCount);
    for (TNetRank k = MyRank + 1; k < peerCount; ++k) {
        DebugPrintf("connect %g (%s)\n", k * 1., peerList[k].c_str());
        yint defaultPort = 0; // port should be explicitly specified in peerList[]
        Peers[k] = Connect(peerList[k], defaultPort, token);
        Net->StartSendRecv(Peers[k], Queue);
        SendData(Net, Peers[k], MyRank);
    }
    yint waitCount = MyRank;
    while (waitCount > 0) {
        TIntrusivePtr<ITcpConnection> conn;
        if (Accept->GetNewConnection(&conn)) {
            Net->StartSendRecv(conn, Queue);
            TNetRank rank = 0;
            WaitData(Queue, conn, &rank);
            DebugPrintf("add peer %g\n", rank * 1.);
            Peers[rank] = conn;
            --waitCount;
        }
    }
    Accept->Stop();
    Accept = 0;
}

}
