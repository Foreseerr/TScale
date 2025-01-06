#pragma once


// single thread is expected to wait for this event
struct ISyncEvent : public TThrRefBase
{
    virtual void Set() = 0;
};


class TSyncEvent : public ISyncEvent
{
#ifndef _win_
    std::atomic<ui32> Flag;
#else 
    HANDLE Event;
#endif
public:
    TSyncEvent();
    void Set() override;
    void Wait();
};
