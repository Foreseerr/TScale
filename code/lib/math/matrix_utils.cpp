#include "stdafx.h"
#include "matrix_utils.h"
#include "eigen.h"


void Normalize(TVector<double> *res)
{
    TVector<double> &v = *res;
    yint sz = YSize(v);
    double len2 = 0;
    for (yint x = 0; x < sz; ++x) {
        len2 += Sqr(v[x]);
    }
    Y_ASSERT(len2 > 0);
    double mult = 1 / sqrt(len2);
    for (yint x = 0; x < sz; ++x) {
        v[x] *= mult;
    }
}


double Dot(const TVector<double> &a, const TVector<double> &b)
{
    yint sz = YSize(a);
    Y_ASSERT(YSize(b) == sz);
    double res = 0;
    for (yint k = 0; k < sz; ++k) {
        res += a[k] * b[k];
    }
    return res;
}


void AddScaled(TVector<double> *pRes, const TVector<double> &vec, double x)
{
    yint sz = YSize(vec);
    Y_ASSERT(sz == YSize(*pRes));
    for (yint i = 0; i < sz; ++i) {
        (*pRes)[i] += vec[i] * x;
    }
}


// symmetrical matrices only
void PowerMatrix(TArray2D<double> *res, double power)
{
    TVector<double> eigenVals;
    TVector<TVector<double> > eigenVecs;
    NEigen::CalcEigenVectors(&eigenVals, &eigenVecs, *res);

    yint sz = res->GetXSize();
    TArray2D<double> m1;
    m1.SetSizes(sz, sz);
    m1.FillZero();
    for (yint f = 0; f < YSize(eigenVals); ++f) {
        if (eigenVals[f] <= 0) {
            continue; // avoid numerical errors
        }
        double e1 = exp(power * log(eigenVals[f]));
        const TVector<double> &vec = eigenVecs[f];
        for (yint y = 0; y < sz; ++y) {
            for (yint x = 0; x < sz; ++x) {
                m1[y][x] += e1 * (vec[y] * vec[x]);
            }
        }
    }
    res->Swap(m1);
}


void MatrixMult(const TArray2D<double> &a, const TArray2D<double> &b, TArray2D<double> *res)
{
    yint xSize = b.GetXSize();
    yint ySize = a.GetYSize();
    yint sz = a.GetXSize();
    Y_ASSERT(sz == (yint)b.GetYSize());
    TArray2D<double> rv;
    rv.SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            double val = 0;
            for (yint k = 0; k < sz; ++k) {
                val += a[y][k] * b[k][x];
            }
            rv[y][x] = val;
        }
    }
    res->Swap(rv);
}


void MatrixAdd(TArray2D<double> *res, const TArray2D<double> &b, double scale)
{
    yint xSize = b.GetXSize();
    yint ySize = b.GetYSize();
    Y_ASSERT((yint)res->GetXSize() == xSize);
    Y_ASSERT((yint)res->GetYSize() == ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*res)[y][x] += b[y][x] * scale;
        }
    }
}


TArray2D<double> Transpose(const TArray2D<double> &a)
{
    yint xSize = a.GetXSize();
    yint ySize = a.GetYSize();
    TArray2D<double> rv;
    rv.SetSizes(ySize, xSize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            rv[x][y] = a[y][x];
        }
    }
    return rv;
}

TArray2D<double> MakeE(yint sz)
{
    TArray2D<double> rv;
    rv.SetSizes(sz, sz);
    rv.FillZero();
    for (yint f = 0; f < sz; ++f) {
        rv[f][f] = 1;
    }
    return rv;
}


double CalcNorm(const TArray2D<double> &a)
{
    yint xSize = a.GetXSize();
    yint ySize = a.GetYSize();
    double rv = 0;
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            rv += Sqr(a[y][x]);
        }
    }
    return rv;
}


void MakeExactlySymmetric(TArray2D<double> *res)
{
    yint sz = res->GetXSize();
    for (yint y = 0; y < sz; ++y) {
        for (yint x = y + 1; x < sz; ++x) {
            double val = (*res)[y][x] + (*res)[x][y];
            (*res)[y][x] = val * 0.5;
            (*res)[x][y] = val * 0.5;
        }
    }
}
