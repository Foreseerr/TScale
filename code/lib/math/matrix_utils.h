#pragma once
#include "matrix.h"

inline double Logit(double x)
{
    return log(x / (1 - x));
}

inline double Logistic(double x)
{
    return 1 / (1 + exp(-x));
}

// two independent evidences, equivalent to logistic add
inline double IndepMult(double a, double b)
{
    double p1 = a * b;
    double p0 = (1 - a) * (1 - b);
    return p1 / (p1 + p0);
}

inline double Shrink(double x, double a)
{
    if (x > 0) {
        return Max(0., x - a);
    } else {
        return Min(0., x + a);
    }
}


void Normalize(TVector<double> *res);

double Dot(const TVector<double> &a, const TVector<double> &b);
void AddScaled(TVector<double> *pRes, const TVector<double> &vec, double x);

void PowerMatrix(TArray2D<double> *res, double power);

inline void InvertMatrix(TArray2D<double> *res)
{
    PowerMatrix(res, -1);
}

inline void InvertSqrtMatrix(TArray2D<double> *res)
{
    PowerMatrix(res, -0.5);
}

void MatrixMult(const TArray2D<double> &a, const TArray2D<double> &b, TArray2D<double> *res);
void MatrixAdd(TArray2D<double> *res, const TArray2D<double> &b, double scale);
TArray2D<double> Transpose(const TArray2D<double> &a);
TArray2D<double> MakeE(yint sz);
double CalcNorm(const TArray2D<double> &a);
void MakeExactlySymmetric(TArray2D<double> *res);


template<class TRnd>
inline void MakeRandomPermutation(TArray2D<double> *res, TArray2D<double> *res1, yint sz, TRnd &&rnd)
{
    TVector<int> cc;
    for (yint k = 0; k < sz; ++k) {
        cc.push_back(k);
    }
    Shuffle(cc.begin(), cc.end(), rnd);
    res->SetSizes(sz, sz);
    res->FillZero();
    *res1 = *res;
    for (yint k = 0; k < sz; ++k) {
        (*res)[k][cc[k]] = 1;
        (*res1)[cc[k]][k] = 1;
    }
}


template<class TRng>
inline void BuildRandomOrthonormalMatrix(TArray2D<double> *res, yint featureCount, yint basisSize, TRng &rng)
{
    res->SetSizes(featureCount, basisSize);
    for (yint b = 0; b < basisSize; ++b) {
        TVector<double> dst;
        dst.resize(featureCount);
        for (yint f = 0; f < featureCount; ++f) {
            dst[f] = GenNormal(rng);
        }
        Normalize(&dst);
        for (yint z = 0; z < b; ++z) {
            double dot = 0;
            for (yint f = 0; f < featureCount; ++f) {
                dot += dst[f] * (*res)[z][f];
            }
            for (yint f = 0; f < featureCount; ++f) {
                dst[f] -= dot * (*res)[z][f];
            }
            Normalize(&dst);
        }
        for (yint f = 0; f < featureCount; ++f) {
            (*res)[b][f] = dst[f];
        }
    }
}


template<class TRng>
void MakeRandomReflection(TArray2D<double> *res, yint sz, TRng &rng)
{
    *res = MakeE(sz);
    TVector<double> dir;
    dir.resize(sz);
    for (yint f = 0; f < sz; ++f) {
        dir[f] = GenNormal(rng);
    }
    Normalize(&dir);
    for (yint y = 0; y < sz; ++y) {
        for (yint x = 0; x < sz; ++x) {
            (*res)[y][x] -= 2 * dir[y] * dir[x];
        }
    }
}


template<class TRng>
void MakeRandomRotation(TArray2D<double> *res, yint sz, TRng &rng)
{
    TArray2D<double> a, b;
    MakeRandomReflection(&a, sz, rng);
    MakeRandomReflection(&b, sz, rng);
    MatrixMult(a, b, res);
}


template<class T, class T1>
void MulLeft(TVector<T> *pRes, const TArray2D<T1> &m, const TVector<T> &s)
{
    TVector<T> &r = *pRes;
    yint xSize = m.GetXSize();
    yint ySize = m.GetYSize();
    ClearPodArray(&r, xSize);
    Y_ASSERT(ySize == s.size());
    for (int y = 0; y < ySize; ++y) {
        double mult = s[y];
        for (int x = 0; x < xSize; ++x) {
            r[x] += m[y][x] * mult;
        }
    }
}


inline double CalcQuadForm(const TArray2D<double> &m, const TVector<double> &vec)
{
    double res = 0;
    yint sz = YSize(vec);
    Y_ASSERT((yint)m.GetXSize() == sz);
    Y_ASSERT((yint)m.GetYSize() == sz);
    for (yint y = 0; y < sz; ++y) {
        for (yint x = 0; x < sz; ++x) {
            res += m[y][x] * vec[y] * vec[x];
        }
    }
    return res;
}

template <class T>
void Transpose(TArray2D<T> *p)
{
    yint xSize = p->GetXSize();
    yint ySize = p->GetYSize();
    Y_VERIFY(xSize == ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = y + 1; x < xSize; ++x) {
            DoSwap((*p)[y][x], (*p)[x][y]);
        }
    }
}
