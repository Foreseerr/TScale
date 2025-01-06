#include "stdafx.h"
#include "log.h"

namespace NLog
{
const yint MAX_LOG_SIZE = 100000;

static TAtomic LogLock;
static TVector<TLogEntry> LogArr;

yint GetLogLineLen(const char *pszFormat, va_list va)
{
    return vsnprintf(0, 0, pszFormat, va);
}

void WriteLog(TLogId logId, yint lineLen, const char *pszFormat, va_list va)
{
    TLogEntry le;
    le.LogId = logId;
    std::time(&le.Time);

    TString &str = le.Msg;
    str.resize(lineLen + 1);
    yint resLen = vsnprintf(&str[0], YSize(str), pszFormat, va);
    str.resize(resLen);
#ifdef _MSC_VER
    if (IsDebuggerPresent()) {
        OutputDebugStringA(str.c_str());
        OutputDebugStringA("\n");
    }
#else
    printf("%s\n", str.c_str());
#endif
    //
    {
        TGuard<TAtomic> g(LogLock);
        LogArr.push_back(le);
        if (YSize(LogArr) > MAX_LOG_SIZE) {
            // truncate log
            yint delSize = Max<yint>(1, YSize(LogArr) * 0.1);
            LogArr.erase(LogArr.begin(), LogArr.begin() + delSize);
        }
    }
}   


void GetLastMessages(TLogId logId, yint maxCount, TVector<TLogEntry> *res)
{
    res->resize(0);
    TGuard<TAtomic> g(LogLock);
    for (yint i = YSize(LogArr) - 1; i >= 0; --i) {
        const TLogEntry &le = LogArr[i];
        if (logId == NO_FILTER || le.LogId == logId) {
            res->push_back(le);
            if (YSize(*res) == maxCount) {
                break;
            }
        }
    }
}
}
