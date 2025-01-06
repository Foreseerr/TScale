#include "stdafx.h"
#include "net_util.h"

#ifndef _win_
#include <fcntl.h>
#endif


namespace NNet
{
void MakeNonBlocking(SOCKET s)
{
#if defined(_win_)
    unsigned long dummy = 1;
    ioctlsocket(s, FIONBIO, &dummy);
#else
    fcntl(s, F_SETFL, O_NONBLOCK);
    // added to prevent socket duplication into child processes
    fcntl(s, F_SETFD, FD_CLOEXEC);
#endif
}

void SetNoTcpDelay(SOCKET s)
{
    int flag = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
}
}
