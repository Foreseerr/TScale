#include "stdafx.h"
#include "tools.h"

TString Sprintf(const char *pszFormat, ...)
{
    TString res;

    va_list va;
    va_start(va, pszFormat);
    yint len = vsnprintf(0, 0, pszFormat, va);
    va_end(va);
    res.resize(len + 1);
    va_start(va, pszFormat);
    yint resLen = vsnprintf(&res[0], YSize(res), pszFormat, va);
    va_end(va);
    res.resize(resLen);
    //
    return res;
}


void DebugPrintf(const char *pszFormat, ...)
{
    TString res;
    va_list va;

    va_start(va, pszFormat);
    yint len = vsnprintf(0, 0, pszFormat, va);
    va_end(va);
    res.resize(len + 1);
    va_start(va, pszFormat);
    yint resLen = vsnprintf(&res[0], YSize(res), pszFormat, va);
    va_end(va);
    res.resize(resLen);

    printf("%s", res.c_str());
#ifdef _MSC_VER
    if (IsDebuggerPresent()) {
        OutputDebugStringA(res.c_str());
    }
#endif
}


void Split(char *pszBuf, TVector<const char*> *pRes, char cSplit)
{
    pRes->resize(0);
    pRes->push_back(pszBuf);
    for (char *pszData = pszBuf; *pszData; ++pszData) {
        if (*pszData == cSplit) {
            *pszData = 0;
            pRes->push_back(pszData + 1);
        }
    }
}


#ifdef _MSC_VER
#include <conio.h>
int GetKeyStroke()
{
    return _getch();
}

bool IsKeyPressed()
{
    return _kbhit();
}
#else
#include <termios.h>
#include <stdio.h>
#include <sys/ioctl.h>

int GetKeyStroke()
{
    static struct termios IOold, IOcurrent;
    tcgetattr(0, &IOold);
    IOcurrent = IOold;
    IOcurrent.c_lflag &= ~ICANON; // disable buffered IO
    //IOcurrent.c_lflag |= ECHO; // set echo mode
    IOcurrent.c_lflag &= ~ECHO; // set no echo mode
    tcsetattr(0, TCSANOW, &IOcurrent);

    char ch = getchar();

    tcsetattr(0, TCSANOW, &IOold);
    return (ui8)ch;
}

bool IsKeyPressed()
{
    static struct termios IOold, IOcurrent;
    tcgetattr(0, &IOold);
    IOcurrent = IOold;
    IOcurrent.c_lflag &= ~ICANON; // disable buffered IO
    //IOcurrent.c_lflag |= ECHO; // set echo mode
    IOcurrent.c_lflag &= ~ECHO; // set no echo mode
    tcsetattr(0, TCSANOW, &IOcurrent);

    int bytesWaiting;
    ioctl(0, FIONREAD, &bytesWaiting);

    tcsetattr(0, TCSANOW, &IOold);
    return bytesWaiting != 0;
}
#endif

bool StartsWith(const TString &str, const TString &prefix)
{
    yint sz = YSize(prefix);
    if (YSize(str) < sz) {
        return false;
    }
    for (yint k = 0; k < sz; ++k) {
        if (str[k] != prefix[k]) {
            return false;
        }
    }
    return true;
}


bool EndsWith(const TString &str, const TString &suffix)
{
    yint sz = YSize(suffix);
    yint pos = YSize(str) - sz;
    if (pos >= 0) {
        for (yint i = 0; i < sz; ++i) {
            if (str[pos + i] != suffix[i]) {
                return false;
            }
        }
        return true;
    }
    return false;
}


char *PrintInt(char *pDst, yint val)
{
    char buf[32];
    char *ptr = buf;
    if (val == 0) {
        *ptr++ = '0';
    } else {
        yint rem = val;
        while (rem > 0) {
            yint nextRem = rem / 10;
            *ptr++ = '0' + (rem - nextRem * 10);
            rem = nextRem;
        }
    }
    char *pRes = pDst;
    while (ptr > buf) {
        *pRes++ = *--ptr;
    }
    return pRes;
}

TString Itoa(yint val)
{
    TString buf;
    buf.resize(32);
    char *p = &buf[0];
    char *pFin = PrintInt(p, val);
    buf.resize(pFin - p);
    return buf;
}
