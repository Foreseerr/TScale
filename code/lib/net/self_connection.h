#pragma once
#include "ip_address.h"
#include "net_util.h"


class TSelfConnection : public TNonCopyable
{
    SOCKET LocalAccept = 0;
    SOCKET LocalServer = 0;
    SOCKET LocalClient = 0;
public:
    TSelfConnection()
    {
        LocalAccept = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        sockaddr_in localAddr;
        TString localHost;
        NNet::ParseInetName(&localAddr, &localHost, "127.0.0.1", 0);
        if (bind(LocalAccept, (sockaddr *)&localAddr, sizeof(localAddr)) != 0) {
            Y_VERIFY(0 && "bind fail");
        }
        if (listen(LocalAccept, 1) != 0) {
            Y_VERIFY(0 && "listen fail");
        }
        socklen_t len = sizeof(localAddr);
        if (getsockname(LocalAccept, (sockaddr *)&localAddr, &len)) {
            Y_VERIFY(0 && "no self address");
        }

        LocalClient = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (connect(LocalClient, (sockaddr *)&localAddr, len) == SOCKET_ERROR) {
            Y_VERIFY(0 && "can not connect to self");
        }
        NNet::SetNoTcpDelay(LocalClient);

        LocalServer = accept(LocalAccept, 0, 0);
    }
    ~TSelfConnection()
    {
        closesocket(LocalClient);
        closesocket(LocalServer);
        closesocket(LocalAccept);
    }

    void Wake()
    {
        // sends self
        char c = 0;
        send(LocalClient, &c, 1, 0);
    }
    void Recv()
    {
        char c;
        recv(LocalServer, &c, 1, 0);
    }
    SOCKET GetRecvSocket() const { return LocalServer; }
};


bool WaitForMessageOrWake(TSelfConnection *pSelfConnect, SOCKET dataSock, float timeoutSec);
