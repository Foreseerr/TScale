#pragma once

#define _CRT_SECURE_NO_WARNINGS
#define _WINSOCK_DEPRECATED_NO_WARNINGS

#if defined(__GNUC__) && !defined(__clang__)
#define __cdecl
#pragma GCC diagnostic ignored "-Wstringop-overflow="
#endif

#include <stdlib.h>
#include <memory.h>


// platform specific
#ifdef _MSC_VER
// win
#pragma warning ( disable : 4267 4018 4530)

// integer downcast
#pragma warning( disable : 4244 )

//#include <tchar.h>
#include <winsock2.h>
#include <windows.h>

#undef min
#undef max

#undef ASSERT
#ifdef _DEBUG
#  define ASSERT( a ) if ( !(a) ) __debugbreak();
#else
#  define ASSERT( a ) ((void)0)
#endif
#undef VERIFY
#define VERIFY( a ) if ( !(a) ) __debugbreak();
#define _win_

inline void SchedYield()
{
    Sleep(0);
}
inline void SleepSeconds(double x)
{
    Sleep(static_cast<DWORD>(x * 1000));
}

typedef int socklen_t;
#define NOINLINE __declspec(noinline)

inline unsigned long long GetCycleCount()
{
    return __rdtsc();
}


#else
// unix
#include <stddef.h>
#include <unistd.h>
#include <sched.h>
#include <assert.h>
#include <wchar.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <stdarg.h>
#include <ctime>

#define ASSERT assert

#undef VERIFY
#define VERIFY( a ) if ( !(a) ) abort();

inline void SchedYield()
{
    sched_yield();
}
inline void SleepSeconds(double x)
{
    usleep(x * 1000000);
}
typedef int SOCKET;
const int INVALID_SOCKET = -1;
const int SOCKET_ERROR = -1;
#define closesocket close

#define __debugbreak __builtin_trap
#define NOINLINE __attribute__((noinline))
#define __forceinline inline __attribute__((always_inline))

// A function that returns the value of the time stamp counter
inline unsigned long long GetCycleCount()
{
    unsigned int lo, hi;
    // Use the "=a" and "=d" constraints to store the low and high 32 bits of the result in the eax and edx registers
    // Use the "memory" clobber to prevent the compiler from reordering memory accesses around the instruction
    __asm__ __volatile__("rdtsc" : "=a" (lo), "=d" (hi) : : "memory");
    // Combine the low and high 32 bits into a 64-bit value
    return ((unsigned long long)hi << 32) | lo;
}
#endif


// common part
#define Y_ASSERT ASSERT
#define Y_VERIFY(a) if ( !(a) ) abort();

typedef unsigned short ui16;
typedef short i16;
typedef unsigned int ui32;
typedef int i32;
typedef long long yint;
typedef unsigned long long ui64;
typedef long long i64;
typedef char i8;
typedef unsigned char ui8;

#define LL(number)   ((i64)(number))
#define ULL(number)  ((ui64)(number))

// STL
#include "nvector.h"
#include "nlist.h"
#include "nstring.h"
#include "nhash_map.h"
#include <string.h>
using namespace nstl;

template<> struct nstl::hash<yint>
{
    size_t operator()(yint __s) const { return __s; }
};
template<> struct nstl::hash<ui64>
{
    size_t operator()(ui64 __s) const { return __s; }
};

struct TPtrHash
{
    yint operator()(const void *x) const
    {
        return (const char *)x - (const char *)0;
    }
};


#define TVector vector
#define THashMap hash_map
//typedef nstl::string Stroka;
typedef nstl::string TString;

#include <fstream>
#include <iostream>

typedef std::ofstream TOFStream;
#define Endl std::endl;

#include "tools.h"
//#include "ysafeptr.h"
#include "atomic.h"
#include "2Darray.h"
#include "bin_saver.h"

template<class T> struct nstl::hash<TIntrusivePtr<T>>
{
    size_t operator()(const TIntrusivePtr<T> & __s) const
    {
        return (const char *)__s.Get() - (const char *)0;
    }
};
