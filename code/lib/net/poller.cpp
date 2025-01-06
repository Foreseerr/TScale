#include "stdafx.h"
#include "poller.h"


namespace NNet
{
TSocketEvent::TSocketEvent() : Flag(0)
{
}

void TSocketEvent::Set()
{
    ui32 expected = 0;
    if (Flag.compare_exchange_strong(expected, 1)) {
        Self.Wake();
    }
}

void TSocketEvent::Reset()
{
    ui32 expected = 1;
    if (Flag.compare_exchange_strong(expected, 0)) {
        Self.Recv();
    }
}
}
