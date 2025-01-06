#pragma once

namespace NEigen
{
void CalcEigenVectors(TVector<double> *pRes, TVector<TVector<double> > *pEigenVectors, const TArray2D<double> &m);
void CalcEigenVectors(TArray2D<double> *resEigen, TArray2D<double> *resBasis, const TArray2D<double> &m);
}
