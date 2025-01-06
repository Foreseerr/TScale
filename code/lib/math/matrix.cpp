#include "stdafx.h"
#include "matrix.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
void FindSomeLinearSolution(const TArray2D<double> &_m, vector<double> &proj, vector<double> *pRes)
{
	int nSize = proj.size();
	ASSERT(_m.GetXSize() == nSize && _m.GetYSize() == nSize);
	pRes->resize(proj.size());
	TArray2D<double> left(_m);
	TArray2D<double> right;
	right.SetSizes(nSize, nSize);
	for (int y = 0; y < nSize; ++y)
	{
		for (int x = 0; x < nSize; ++x)
			right[y][x] = x == y;
	}

	const double about0 = 1e-5f;
	for (int y = 0; y < nSize; y++)
	{
		double fDiag = left[y][y], fMax = fabs(fDiag);
		int nBestRow = y;
		for (int k = y + 1; k < nSize; ++k)
		{
			double fTest = fabs(left[k][y]);
			if (fTest > 2 * fMax)
			{
				fMax = fTest;
				nBestRow = k;
			}
		}
		if (nBestRow != y)
		{
			double f = fMax * fDiag > 0 ? 1 : -1;
			for (int x = 0; x < nSize; ++x)
			{
				left[y][x] += f * left[ nBestRow ][x];
				right[y][x] += f * right[ nBestRow ][x];
			}
		}
		fDiag = left[y][y];
		if (fabs(fDiag) < about0)
		{
			int h = y;
			while((h < nSize - 1) && (fabs(fDiag) < about0))
			{
				h++;
				if(fabs(left[h][y]) > about0)
				{
					for(int u = 0; u < nSize; u++)
					{
						left[y][u] += left[h][u];
						right[y][u] += right[h][u];
					}
					fDiag = left[y][y];
				}
			}
			if (fabs(fDiag) < 1e-8)
			{
				// not needed contact
				/*for (int u = 0; u < nSize; ++u)
				{
				left[y][u] = 0; left[u][y] = 0;
				right[y][u] = 0; right[u][y] = 0;
				}*/
				continue;
			}
		}
		double fDiag1 = 1 / fDiag;
		for (int x = 0; x < nSize; ++x)
		{
			left[y][x] *= fDiag1;
			right[y][x] *= fDiag1;
		}
		for (int k = 0; k < nSize; ++k)
		{
			if (k == y)
				continue;
			double fK = left[k][y];
			for (int x = 0; x < nSize; ++x)
			{
				left[k][x] -= left[y][x] * fK;
				right[k][x] -= right[y][x] * fK;
			}
		}
	}
	for (int y = 0; y < nSize; ++y)
	{
		double fRes = 0;
		for (int x = 0; x < nSize; ++x)
			fRes += right[y][x] * proj[x];
		(*pRes)[y] = fRes;
	}
}
