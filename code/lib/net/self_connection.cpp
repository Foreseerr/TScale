#include "stdafx.h"
#include "self_connection.h"


bool WaitForMessageOrWake(TSelfConnection *pSelfConnect, SOCKET dataSock, float timeoutSec)
{
    SOCKET self = pSelfConnect->GetRecvSocket();
    fd_set fs;
    FD_ZERO(&fs);
    FD_SET(dataSock, &fs);
    FD_SET(self, &fs);
    timeval tvTimeout = NNet::MakeTimeval(timeoutSec);
    if (select(Max<SOCKET>(dataSock, self) + 1, &fs, 0, 0, &tvTimeout) == 0) {
        return false;
    }
    if (FD_ISSET(self, &fs)) {
        pSelfConnect->Recv();
    }
    return FD_ISSET(dataSock, &fs);
}
