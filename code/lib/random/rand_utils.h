#pragma once
#include "mersenne.h"
#include "random250.h"

//inline float GetUniformRandom()
//{
//    return r250n(100000001) / 100000000.0f;
//}
//
//inline float GenNormDistr(float sko)
//{
//    for(;;) {
//        float x = GetUniformRandom() * 2 - 1;
//        float y = GetUniformRandom() * 2 - 1;
//        float r = x*x + y*y;
//        if (r > 1 || r == 0)
//            continue;
//        float fac = sqrt(-2 * log(r) / r);
//        return x * fac * sko;
//    }
//}

template <class TRng>
inline double GenNormal(TRng &rng)
{
    for(;;) {
        double x = rng.GenRandReal3() * 2 - 1;
        double y = rng.GenRandReal3() * 2 - 1;
        double r = x*x + y*y;
        if (r > 1 || r == 0)
            continue;
        double fac = sqrt(-2 * log(r) / r);
        return x * fac;
    }
}

template <class T>
void Shuffle(TVector<T> *res)
{
    yint sz = YSize(*res);
    for (yint i = 0; i < sz - 1; ++i) {
        DoSwap((*res)[i], (*res)[i + r250n(sz - i)]);
    }
}

template <class T, class TRnd>
void Shuffle(T beg, T fin, TRnd &&rnd)
{
    for (T ptr = beg; ptr != fin; ++ptr) {
        DoSwap(*ptr, ptr[rnd.Uniform(fin - ptr)]);
    }
}
