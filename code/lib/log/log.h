#pragma once
#include <ctime>
//#include <stdarg.h>

namespace NLog
{
typedef ui32 TLogId;

const int NO_FILTER = 0;

struct TLogEntry
{
    TLogId LogId;
    std::time_t Time;
    TString Msg;
};

yint GetLogLineLen(const char *pszFormat, va_list va);
void WriteLog(TLogId logId, yint lineLen, const char *pszFormat, va_list va);
void GetLastMessages(TLogId logId, yint maxCount, TVector<TLogEntry> *res);

#define USE_CUSTOM_LOG(id, FuncName) static void FuncName(const char *p,...) {\
    va_list va;\
    va_start(va, p); yint lineLen = NLog::GetLogLineLen(p, va); va_end(va);\
    va_start(va, p); NLog::WriteLog(id, lineLen, p, va); va_end(va);\
}

#define USE_LOG(id) USE_CUSTOM_LOG(id, Log)

}
