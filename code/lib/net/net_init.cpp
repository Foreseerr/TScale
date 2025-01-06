#include "stdafx.h"

#ifdef _MSC_VER
#pragma comment(lib, "ws2_32.lib")

static struct TNetInit
{
    TNetInit()
    {
        WSADATA wsaData;
        Y_VERIFY(WSAStartup(MAKEWORD(2, 2), &wsaData) == 0);
    }
} NetInit;
#endif
