#pragma once


template<class T>
void MakeE(TArray2D<T> *pMatrix)
{
	pMatrix->FillZero();
	for (int x = 0; x < Min(pMatrix->GetXSize(), pMatrix->GetYSize()); ++x)
		(*pMatrix)[x][x] = 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T, class T1>
void Mul(vector<T> *pRes, const TArray2D<T1> &m, const vector<T> &s)
{
	vector<T> &r = *pRes;
	r.resize(m.GetYSize());
	ASSERT(m.GetXSize() == s.size());
	for (int y = 0; y < r.size(); ++y)
	{
		T1 fRes = 0;
		for (int x = 0; x < m.GetXSize(); ++x)
			fRes += m[y][x] * s[x];
		r[y] = fRes;
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
bool InvertRobust(TArray2D<T> *pMatrix)
{
	const T about0((T)1.0e-20);
	const T zero(0);
	ASSERT(pMatrix->GetXSize() == pMatrix->GetYSize());
	if (pMatrix->GetXSize() != pMatrix->GetYSize())
		return false;
	int nSize = pMatrix->GetXSize();
	TArray2D<T> right, left(*pMatrix);
	right.SetSizes(nSize, nSize);
	MakeE(&right);
	for (int i=0; i<nSize; i++)
	{
		T diag=left[i][i], diag1=left[i][i];
		int maxi = i;
		for (int k=i+1; k<nSize; k++) 
		{
			if (fabs(left[k][i]) > diag1)
			{
				diag1 = left[k][i]; 
				maxi = k;
			}
		}
		if (maxi != i && fabs(diag)*((T)10) < fabs(diag1))
		{
			for (int u=0; u<nSize; u++)
			{
				left[i][u] += left[maxi][u];
				right[i][u] += right[maxi][u];
			}
			diag = left[i][i];
		}
		if (fabs(diag) < about0)
		{
			int h = i;
			while ((h < nSize-1) && (fabs(diag) < about0))
			{
				h++;
				if (fabs(left[h][i]) > about0)
				{
					for (int u=0; u<nSize; u++)
					{
						left[i][u] += left[h][u];
						right[i][u] += right[h][u];
					}
					diag = left[i][i];
				}
			}
			if (fabs(diag) < about0)
				return false;
		}
		T invdiag;
		invdiag=((T)1)/diag;
		for (int j=0; j<nSize; j++)
		{
			left[i][j] *= invdiag;
			right[i][j] *= invdiag;
		}
		for (int k=i+1; k<nSize; k++)
		{
			T  koef = left[k][i];
			T *le = &left[k][0], *lei = &left[i][0];
			T *ri = &right[k][0], *rii = &right[i][0];
			T *lefin = le + nSize;
			if (koef != zero)
			{
				//	for(s=0; s<size; s++){ left[k][s]-=koef*left[i][s]; right[k][s]-=koef*right[i][s]; }
				while (le < lefin)
				{
					le[0] -= koef*lei[0]; le++; lei++; 
					ri[0] -= koef*rii[0]; ri++; rii++;
				}
			}
		}
	}
	for (int i=nSize-1; i>=0; i--) 
	{
		for (int k=0; k<i; k++) 
		{
			T  koef = left[k][i];
			T *le = &left[k][0], *lei = &left[i][0];
			T *ri = &right[k][0], *rii = &right[i][0];
			T *lefin = le+nSize;
			if (koef != zero)
			{
				//	for(s=0; s<size; s++){ left[k][s]-=koef*left[i][s]; right[k][s]-=koef*right[i][s];}
				while (le < lefin)
				{
					le[0] -= koef*lei[0]; le++; lei++; 
					ri[0] -= koef*rii[0]; ri++; rii++;
				}
			}
		}
	}
	*pMatrix = right;
	return true;
}


template <class TTargefloat, class TSrcfloat>
static void CopyMatrix(TArray2D<TTargefloat> *pRes, const TArray2D<TSrcfloat> &src)
{
    yint xSize = src.GetXSize();
    yint ySize = src.GetYSize();
    pRes->SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*pRes)[y][x] = src[y][x];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void FindSomeLinearSolution(const TArray2D<double> &_m, vector<double> &proj, vector<double> *pRes);
