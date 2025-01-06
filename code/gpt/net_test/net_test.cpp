#include "stdafx.h"
#include <lib/net/tcp_net.h>
#include <lib/net/poller.h>
#include <lib/config/config.h>
#include <lib/hp_timer/hp_timer.h>


const int NET_TEST_PORT = 14897;
const yint REQUEST_SIZE = 100 * (1 << 20);
const yint RESPONSE_SIZE = 100 * (1 << 20);
static TGuid NetTestToken(0xbadf00d, 0x31337, 0x9ece30, 0x83291294);

using namespace NNet;

static void RunServer()
{
    TIntrusivePtr<ITcpSendRecv> net = CreateTcpSendRecv();
    TIntrusivePtr<TSocketEvent> ev = new TSocketEvent();
    TIntrusivePtr<ITcpAccept> connAccept = net->StartAccept(NET_TEST_PORT, NetTestToken, ev);
    TIntrusivePtr<TTcpRecvQueue> reqQueue = new TTcpRecvQueue(ev);

    // timeout
    NHPTimer::STime tCurrent;
    NHPTimer::GetTime(&tCurrent);
    double collectTime = 0;

    THashMap<TIntrusivePtr<ITcpConnection>, bool> newConn;

    // collect updates and send basepoints
    printf("run server\n"); fflush(0);
    TTcpPoller poller;
    for (;;) {
        // do not consume 100% cpu when nothing is happening
        poller.Start();
        poller.AddSocket(ev->GetSocket(), POLLRDNORM);
        poller.Poll(1);
        ev->Reset();

        TIntrusivePtr<ITcpConnection> conn;
        while (connAccept->GetNewConnection(&conn)) {
            printf("got connection from %s\n", conn->GetPeerAddress().c_str()); fflush(0);
            conn->SetExitOnError(false);
            net->StartSendRecv(conn, reqQueue);
        }

        TIntrusivePtr<TTcpPacketReceived> recvPkt;
        while (reqQueue->Dequeue(&recvPkt)) {
            conn = recvPkt->Conn;
            TIntrusivePtr<TTcpPacket> pkt = new TTcpPacket;
            pkt->Data.resize(RESPONSE_SIZE);
            net->Send(conn, pkt);
        }
    }
}


static void RunClient(const TString &serverAddr)
{
    TIntrusivePtr<ITcpSendRecv> net = CreateTcpSendRecv();
    TIntrusivePtr<TSyncEvent> ev = new TSyncEvent();
    TIntrusivePtr<TTcpRecvQueue> queue = new TTcpRecvQueue(ev);
    TIntrusivePtr<ITcpConnection> serverConn = Connect(serverAddr, NET_TEST_PORT, NetTestToken);
    net->StartSendRecv(serverConn, queue);

    TIntrusivePtr<TTcpPacket> query = new TTcpPacket;
    query->Data.resize(REQUEST_SIZE);

    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        net->Send(serverConn, query);
        for (;;) {
            ev->Wait();
            TIntrusivePtr<TTcpPacketReceived> response;
            if (queue->Dequeue(&response)) {
                double tPassed = NHPTimer::GetTimePassed(&tStart);
                printf("%g mb/sec\n", (REQUEST_SIZE + RESPONSE_SIZE) / 1000000. / tPassed);
                break;
            }
        }
    }
}

int main(int argc, char **argv)
{
    TOpt cmdline("c:", argc, argv);
    for (const TOpt::TParam &param : cmdline.Params) {
        if (param.Name == "c") {
            RunClient(param.Args[0]);
        }
    }
    RunServer();
    return 0;
}
