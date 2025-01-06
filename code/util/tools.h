#pragma once
#include <math.h>

template<class T>
void ClearPodArray(vector<T> *res, yint count)
{
    res->yresize(count);
    memset(res->data(), 0, sizeof(T) * count);
}

struct TVecHash
{
    int operator()(const vector<int> &a) const
    {
        int res = 1988712;
        for (int i = 0; i < a.size(); ++i)
            res = 984121 * res + a[i];
        return res;
    }
};


template <class T>
T Min(T a, T b) { return (a < b) ? a : b; }

template <class T>
T Max(T a, T b) { return (a > b) ? a : b; }

template<class T>
inline T ClampVal(T x, T minVal, T maxVal)
{
    if (x < minVal) {
        return minVal;
    } else if (x > maxVal) {
        return maxVal;
    } else {
        return x;
    }
}

inline yint DivCeil(yint k, yint block)
{
    return (k + block - 1) / block;
}


template <class T, class TElem>
inline bool IsInSet(const T &c, const TElem &e) { return find(c.begin(), c.end(), e) != c.end(); }

template <class T>
yint YSize(const T &x)
{
    return (yint)x.size();
}

#define ARRAY_SIZE( a ) ( sizeof( a ) / sizeof( (a)[0] ) )

template <class T>
void Sort(T a, T b)
{
    nstl::sort(a, b);
}

template <class T, class Cmp>
void Sort(T a, T b, Cmp cc)
{
    nstl::sort(a, b, cc);
}


template <class T>
inline T Sqr(T a)
{
    return a * a;
}

inline double LogAdd(double f1, double f2)
{
    //double f = Max( f1, f2 );
    //double fCheckRes = log( exp( f1 - f ) + exp( f2 - f ) ) + f;
    if (f1 < f2) {
        const double temp = f1;
        f1 = f2;
        f2 = temp;
    }
    double fDif = f2 - f1;
    if (fDif < -80)
        return f1;
    double fRes = log(exp(fDif) + 1) + f1;
    return fRes;
}

template <class T>
inline bool IsPow2(T x)
{
    return (x & (x - 1)) == 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// not used anymore
//inline constexpr float constexprSqrt(float x, float curr, float prev)
//{
//    // newton-raphson
//    return curr == prev ? curr : constexprSqrt(x, 0.5f * (curr + x / curr), curr);
//}
//
//inline constexpr float constexprSqrt(float x)
//{
//    return constexprSqrt(x, x, 0);
//}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TNonCopyable
{
private:  // emphasize the following members are private
    TNonCopyable(const TNonCopyable &);
    const TNonCopyable &operator=(const TNonCopyable &);
protected:
    TNonCopyable() {
    }

    ~TNonCopyable() {
    }
};

template<class T>
class THolder
{
    T *Ptr;

    THolder(const THolder &) {}
    void operator=(const THolder &) {}
public:
    THolder(T *p) : Ptr(p) {}
    ~THolder() { delete Ptr; }
    T *Get() { return Ptr; }
    T *operator->() { return Ptr; }
    T& operator*() { return *Ptr; }
};


template <class T>
void DoSwap(T &a, T &b)
{
    T x = a;
    a = b;
    b = x;
}


template <class TYPE>
inline void Zero(TYPE &val) { memset(&val, 0, sizeof(val)); }


template <class T>
void Reverse(T *beg, T *fin)
{
    while (beg < fin) {
        DoSwap(*beg++, *--fin);
    }
}


TString Sprintf(const char *pszFormat, ...);
void DebugPrintf(const char *pszFormat, ...);

void Split(char *pszBuf, TVector<const char *> *pRes, char cSplit);
inline void Split(char *pszBuf, TVector<const char *> *pRes) { Split(pszBuf, pRes, '\t'); }

int GetKeyStroke();
bool IsKeyPressed();

bool StartsWith(const TString &str, const TString &prefix);
bool EndsWith(const TString &str, const TString &suffix);

char *PrintInt(char *pDst, yint val);
TString Itoa(yint val);
