#include "stdafx.h"
#include "event.h"
#ifndef _win_
#include <unistd.h> // for syscall
#include <linux/futex.h> // for FUTEX_WAIT and FUTEX_WAKE
#include <sys/syscall.h> // for SYS_futex
#endif


#ifndef _win_
TSyncEvent::TSyncEvent() : Flag(false)
{
}

void TSyncEvent::Set()
{
    ui32 expected = 0;
    if (Flag.compare_exchange_strong(expected, 1)) {
        ui32 wakeThreadCount = 1; // INT_MAX;
        syscall(SYS_futex, &Flag, FUTEX_WAKE_PRIVATE, wakeThreadCount, nullptr, nullptr);
    }
}

void TSyncEvent::Wait()
{
    for (;;) {
        ui32 expected = 1;
        if (Flag.compare_exchange_strong(expected, 0)) {
            break;
        }
        syscall(SYS_futex, &Flag, FUTEX_WAIT_PRIVATE, 0, nullptr, nullptr);
    }
}

#else
TSyncEvent::TSyncEvent()
{
    Event = CreateEvent(0, FALSE, FALSE, 0);
}

void TSyncEvent::Set()
{
    SetEvent(Event);
}

void TSyncEvent::Wait()
{
    WaitForSingleObject(Event, INFINITE);
}
#endif

