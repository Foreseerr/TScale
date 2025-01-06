#include "stdafx.h"
#include "guid.h"
#include "citymurmur.h"
#include <time.h>

#ifdef _MSC_VER
#define HOST_NAME_MAX 64
#define LOGIN_NAME_MAX 64
#else
#include <unistd.h>
#include <limits.h>
#endif

static TAtomic Counter;

struct TGuidSeed
{
    ui64 StartCycleCount = 0;
    char HostName[HOST_NAME_MAX] = "";
    char UserName[LOGIN_NAME_MAX] = "";
    ui64 StartTime = 0;
};

static TGuid CalcBaseGuid()
{
    TGuidSeed gs;
    gs.StartCycleCount = GetCycleCount();
    std::time_t t;
    std::time(&t);
    gs.StartTime = t;
#ifdef _MSC_VER
    DWORD len = HOST_NAME_MAX;
    GetComputerNameA(gs.HostName, &len);
    len = LOGIN_NAME_MAX;
    GetUserNameA(gs.UserName, &len);
#else
    gethostname(gs.HostName, HOST_NAME_MAX);
    getlogin_r(gs.UserName, LOGIN_NAME_MAX);
#endif
    TGuid res;
    CityMurmur(&gs, sizeof(gs), 0x31337, 0xbadf00d, &res.ll[0], &res.ll[1]);
    return res;
}


static TGuid BaseGuid;

void CreateGuid(TGuid *res)
{
    static bool guidSeedIsInit;
    if (!guidSeedIsInit) {
        BaseGuid = CalcBaseGuid();
        guidSeedIsInit = true;
    }
    *res = BaseGuid;
    res->ll[1] += Counter.fetch_add(1);
}


TString GetGuidAsString(const TGuid &g)
{
    char buf[1000];
    sprintf(buf, "%x-%x-%x-%x", g.dw[0], g.dw[1], g.dw[2], g.dw[3]);
    return buf;
}

TString CreateGuidAsString()
{
    TGuid guid;
    CreateGuid(&guid);
    return GetGuidAsString(guid);
}

TGuid GetGuid(const TString &s)
{
    TGuid g;
    sscanf(s.c_str(), "%x-%x-%x-%x", &g.dw[0], &g.dw[1], &g.dw[2], &g.dw[3]);
    return g;
}
