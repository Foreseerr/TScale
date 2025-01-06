#include "stdafx.h"
#include "thread.h"

#ifdef _MSC_VER
void SetIdlePriority()
{
    SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS);
}
#else
void SetIdlePriority()
{
}
#endif