#include "stdafx.h"
#include "hp_timer.h"

using namespace NHPTimer;
static double fProcFreq1 = 1;

double NHPTimer::GetSeconds(const NHPTimer::STime& a)
{
	return (static_cast<double>(a)) * fProcFreq1;
}


double NHPTimer::GetClockRate()
{
	return 1 / fProcFreq1;
}


void NHPTimer::GetTime(STime* pTime)
{
	*pTime = GetCycleCount();
}


double NHPTimer::GetTimePassed(STime* pTime)
{
	STime old(*pTime);
	GetTime(pTime);
	return GetSeconds(*pTime - old);
}


#ifdef _MSC_VER
static void InitHPTimer()
{
	i64 pcFreq;
	QueryPerformanceFrequency((_LARGE_INTEGER *)&pcFreq);
	i64 pcStart, pcFin;
	STime tStart, tFin;
	QueryPerformanceCounter((_LARGE_INTEGER *)&pcFin);
	GetTime(&tFin);
	const yint N_ITER = 5;
	double freqArr[N_ITER];
	for (yint i = 0; i < N_ITER; ++i) {
		pcStart = pcFin;
		tStart = tFin;
		for (;;) {
			QueryPerformanceCounter((_LARGE_INTEGER *)&pcFin);
			GetTime(&tFin);
			if (pcFin - pcStart > 5000) { // defines frequency precision measurement, the longer the more precise
				break;
			}
			SchedYield();
		}
		double tscTicks = tFin - tStart;
		double pcTime = double(pcFin - pcStart) / pcFreq;
		freqArr[i] = tscTicks / pcTime;
	}
	Sort(freqArr, freqArr + N_ITER);
	fProcFreq1 = 1 / freqArr[N_ITER / 2];
	//printf("freq = %gMHz\n", 1e-6 / fProcFreq1);
}
#else
#include <time.h>
static double GetDelta(const timespec &a, const timespec &b)
{
	return (b.tv_sec - a.tv_sec) + (b.tv_nsec * 1e-9 - a.tv_nsec * 1e-9);
}

static void InitHPTimer()
{
	//CLOCK_REALTIME
	clockid_t clock = CLOCK_MONOTONIC_RAW;

	timespec interval;
	clock_getres(clock, &interval);
	double freq = 1e9 / interval.tv_nsec;

	timespec pcStart, pcFin;
	STime tStart, tFin;
	clock_gettime(clock, &pcFin);
	GetTime(&tFin);
	const yint N_ITER = 5;
	double freqArr[N_ITER];
	for (yint i = 0; i < N_ITER; ++i) {
		pcStart = pcFin;
		tStart = tFin;
		for (;;) {
			clock_gettime(clock, &pcFin);
			GetTime(&tFin);
			if (GetDelta(pcStart, pcFin) > 0.005) { // defines frequency precision measurement, the longer the more precise
				break;
			}
			SchedYield();
		}
		double tscTicks = tFin - tStart;
		double pcTime = GetDelta(pcStart, pcFin);
		freqArr[i] = tscTicks / pcTime;
	}
	Sort(freqArr, freqArr + N_ITER);
	fProcFreq1 = 1 / freqArr[N_ITER / 2];
	//printf("freq = %gMHz\n", 1e-6 / fProcFreq1);
}
#endif


////////////////////////////////////////////////////////////////////////////////////////////////////
// это вспомогательная структура для автоматической инициализации HP timer'а
struct SHPTimerInit
{
	SHPTimerInit() { InitHPTimer(); }
};
static SHPTimerInit hptInit;
