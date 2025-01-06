#pragma once
#include "self_connection.h"
#include <util/event.h>

#ifdef _win_
#define pollfd WSAPOLLFD
#define poll WSAPoll
#else
#include <poll.h>
#endif


namespace NNet
{
struct TTcpPoller
{
    yint Ptr = 0;
    TVector<pollfd> FS;

    TTcpPoller()
    {
        ClearPodArray(&FS, 128);
    }

    void Start()
    {
        Ptr = 0;
    }

    void AddSocket(SOCKET s, yint events)
    {
        if (Ptr >= YSize(FS)) {
            pollfd zeroFD;
            Zero(zeroFD);
            FS.resize(Ptr * 2, zeroFD);
        }
        FS[Ptr].fd = s;
        FS[Ptr].events = events;
        ++Ptr;
    }

    void Poll(float waitSeconds)
    {
        int timeout = waitSeconds * 1000;
        poll(FS.data(), Ptr, timeout);
    }

    yint CheckSocket(SOCKET s)
    {
        Y_ASSERT(FS[Ptr].fd = s);
        return FS[Ptr++].revents;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TSocketEvent : public ISyncEvent
{
    TSelfConnection Self;
    std::atomic<ui32> Flag;
public:
    TSocketEvent();
    void Set() override;
    void Reset();
    SOCKET GetSocket() { return Self.GetRecvSocket(); }
};
}
