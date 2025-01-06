#include "stdafx.h"
#include "eigen.h"

namespace NEigen
{
/**

Computes eigenvalues and eigenvectors of a double (non-complex)
matrix.
<P>
If A is symmetric, then A = V*D*V' where the eigenvalue matrix D is
diagonal and the eigenvector matrix V is orthogonal. That is,
the diagonal values of D are the eigenvalues, and
V*V' = I, where I is the identity matrix.  The columns of V
represent the eigenvectors in the sense that A*V = V*D.

<p>
(Adapted from JAMA, a Java Matrix Library, developed by jointly
by the Mathworks and NIST; see  http://math.nist.gov/javanumerics/jama).
**/

class Eigenvalue
{


    /** Row and column dimension (square matrix).  */
    int n;

    /** Arrays for internal storage of eigenvalues. */

    TVector<double> d;         /* double part */
    TVector<double> e;         /* img part */

    /** Array for internal storage of eigenvectors. */
    TArray2D<double> V;

    /** Array for internal storage of nonsymmetric Hessenberg form.
    @serial internal storage of nonsymmetric Hessenberg form.
    */
    TArray2D<double> H;



    // Symmetric Householder reduction to tridiagonal form.

    void tred2() {

        //  This is derived from the Algol procedures tred2 by
        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.

        for (int j = 0; j < n; j++) {
            d[j] = V[n-1][j];
        }

        // Householder reduction to tridiagonal form.

        for (int i = n-1; i > 0; i--) {

            // Scale to avoid under/overflow.

            double scale = 0.0;
            double h = 0.0;
            for (int k = 0; k < i; k++) {
                scale = scale + fabs(d[k]);
            }
            if (scale == 0.0) {
                e[i] = d[i-1];
                for (int j = 0; j < i; j++) {
                    d[j] = V[i-1][j];
                    V[i][j] = 0.0;
                    V[j][i] = 0.0;
                }
            } else {

                // Generate Householder vector.

                for (int k = 0; k < i; k++) {
                    d[k] /= scale;
                    h += d[k] * d[k];
                }
                double f = d[i-1];
                double g = sqrt(h);
                if (f > 0) {
                    g = -g;
                }
                e[i] = scale * g;
                h = h - f * g;
                d[i-1] = f - g;
                for (int j = 0; j < i; j++) {
                    e[j] = 0.0;
                }

                // Apply similarity transformation to remaining columns.

                for (int j = 0; j < i; j++) {
                    f = d[j];
                    V[j][i] = f;
                    g = e[j] + V[j][j] * f;
                    for (int k = j+1; k <= i-1; k++) {
                        g += V[k][j] * d[k];
                        e[k] += V[k][j] * f;
                    }
                    e[j] = g;
                }
                f = 0.0;
                for (int j = 0; j < i; j++) {
                    e[j] /= h;
                    f += e[j] * d[j];
                }
                double hh = f / (h + h);
                for (int j = 0; j < i; j++) {
                    e[j] -= hh * d[j];
                }
                for (int j = 0; j < i; j++) {
                    f = d[j];
                    g = e[j];
                    for (int k = j; k <= i-1; k++) {
                        V[k][j] -= (f * e[k] + g * d[k]);
                    }
                    d[j] = V[i-1][j];
                    V[i][j] = 0.0;
                }
            }
            d[i] = h;
        }

        // Accumulate transformations.

        for (int i = 0; i < n-1; i++) {
            V[n-1][i] = V[i][i];
            V[i][i] = 1.0;
            double h = d[i+1];
            if (h != 0.0) {
                for (int k = 0; k <= i; k++) {
                    d[k] = V[k][i+1] / h;
                }
                for (int j = 0; j <= i; j++) {
                    double g = 0.0;
                    for (int k = 0; k <= i; k++) {
                        g += V[k][i+1] * V[k][j];
                    }
                    for (int k = 0; k <= i; k++) {
                        V[k][j] -= g * d[k];
                    }
                }
            }
            for (int k = 0; k <= i; k++) {
                V[k][i+1] = 0.0;
            }
        }
        for (int j = 0; j < n; j++) {
            d[j] = V[n-1][j];
            V[n-1][j] = 0.0;
        }
        V[n-1][n-1] = 1.0;
        e[0] = 0.0;
    }

    // Symmetric tridiagonal QL algorithm.

    void tql2 () {

        //  This is derived from the Algol procedures tql2, by
        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.

        for (int i = 1; i < n; i++) {
            e[i-1] = e[i];
        }
        e[n-1] = 0.0;

        double f = 0.0;
        double tst1 = 0.0;
        double eps = exp(log(2.0) * -52);//pow(2.0,-52.0);
        for (int l = 0; l < n; l++) {

            // Find small subdiagonal element

            tst1 = Max(tst1, fabs(d[l]) + fabs(e[l]) );
            int m = l;

            // Original while-loop from Java code // modified by Andy
            while ( m < n - 1 )
            {
                if (fabs(e[m]) <= eps*tst1) {
                    break;
                }
                m++;
            }


            // If m == l, d[l] is an eigenvalue,
            // otherwise, iterate.

            if (m > l) {
                int iter = 0;
                do {
                    iter = iter + 1;  // (Could check iteration count here.)

                    // Compute implicit shift

                    double g = d[l];
                    double p = (d[l+1] - g) / (2.0 * e[l]);
                    double r = hypot(p,1.0);
                    if (p < 0) {
                        r = -r;
                    }
                    d[l] = e[l] / (p + r);
                    d[l+1] = e[l] * (p + r);
                    double dl1 = d[l+1];
                    double h = g - d[l];
                    for (int i = l+2; i < n; i++) {
                        d[i] -= h;
                    }
                    f = f + h;

                    // Implicit QL transformation.

                    p = d[m];
                    double c = 1.0;
                    double c2 = c;
                    double c3 = c;
                    double el1 = e[l+1];
                    double s = 0.0;
                    double s2 = 0.0;
                    for (int i = m-1; i >= l; i--) {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * e[i];
                        h = c * p;
                        r = hypot(p,e[i]);
                        //if ( i >= e.size() - 1 )
                        //    __debugbreak();
                        e[i+1] = s * r;
                        s = e[i] / r;
                        c = p / r;
                        p = c * d[i] - s * g;
                        d[i+1] = h + s * (c * g + s * d[i]);

                        // Accumulate transformation.

                        for (int k = 0; k < n; k++) {
                            h = V[k][i+1];
                            V[k][i+1] = s * V[k][i] + c * h;
                            V[k][i] = c * V[k][i] - s * h;
                        }
                    }
                    p = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p;
                    d[l] = c * p;

                    // Check for convergence.

                } while (fabs(e[l]) > eps*tst1);
            }
            d[l] = d[l] + f;
            e[l] = 0.0;
        }

        // Sort eigenvalues and corresponding vectors.

        for (int i = 0; i < n-1; i++) {
            int k = i;
            double p = d[i];
            for (int j = i+1; j < n; j++) {
                if (d[j] < p) {
                    k = j;
                    p = d[j];
                }
            }
            if (k != i) {
                d[k] = d[i];
                d[i] = p;
                for (int j = 0; j < n; j++) {
                    p = V[j][i];
                    V[j][i] = V[j][k];
                    V[j][k] = p;
                }
            }
        }
    }

public:


    /** Check for symmetry, then construct the eigenvalue decomposition
    @param A    Square double (non-complex) matrix
    */

    Eigenvalue(const TArray2D<double> &A) {
        n = A.GetXSize();
        V.SetSizes( n, n );
        V.FillZero();
        d.resize( n, 0 );
        e.resize( n, 0 );

        bool issymmetric = 1;
        for (int j = 0; (j < n) && issymmetric; j++) {
            for (int i = j+1; (i < n) && issymmetric; i++) {
                issymmetric = (A[i][j] == A[j][i]);
            }
        }

        if (!issymmetric)
            ASSERT(0);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                V[i][j] = A[i][j];
            }
        }

        // Tridiagonalize.
        tred2();

        // Diagonalize.
        tql2();

    }


    const TVector<double> &GetEigenvalues() const { return d; }
    void GetEigenVector(int nIdx, TVector<double> *pRes) const {
        pRes->resize(n);
        for (int i = 0; i < n; ++i)
            (*pRes)[i] = V[i][nIdx];
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
void CalcEigenVectors(TVector<double> *pRes, TVector<TVector<double> > *pEigenVectors, const TArray2D<double> &m)
{
    ASSERT(m.GetXSize() == m.GetYSize());
    Eigenvalue calc( m );
    *pRes = calc.GetEigenvalues();
    if (pEigenVectors) {
        pEigenVectors->resize(m.GetXSize());
        for (int i = 0; i < YSize(*pEigenVectors); ++i)
            calc.GetEigenVector(i, &(*pEigenVectors)[i]);
    }
}


// m = Transpose(basis) * eigen * basis
void CalcEigenVectors(TArray2D<double> *resEigen, TArray2D<double> *resBasis, const TArray2D<double> &m)
{
    TVector<double> eigenVals;
    TVector<TVector<double> > eigenVecs;
    CalcEigenVectors(&eigenVals, &eigenVecs, m);

    yint sz = YSize(eigenVals);
    TArray2D<double> &eigen = *resEigen;
    eigen.SetSizes(sz, sz);
    eigen.FillZero();
    TArray2D<double> &basis = *resBasis;
    basis.SetSizes(sz, sz);
    for (yint y = 0; y < sz; ++y) {
        eigen[y][y] = eigenVals[y];
        for (yint x = 0; x < sz; ++x) {
            basis[y][x] = eigenVecs[y][x];
        }
    }
    //{
    //    TArray2D<double> chk;
    //    MatrixMult(Transpose(basis), eigen, &chk);
    //    MatrixMult(chk, basis, &chk);
    //    chk = chk; // chk == m;
    //}
}


//////////////////////////////////////////////////////////////////////////
// Test
//static void Mult( TArray2D<double> *pRes, const TArray2D<double> &a )
//{
//    TArray2D<double> res;
//    res.SetSizes( pRes->GetXSize(), pRes->GetYSize() );
//    for ( int y = 0; y < res.GetYSize(); ++y )
//    {
//        for ( int x = 0; x < res.GetXSize(); ++x )
//        {
//            double f = 0;
//            for ( int i = 0; i < pRes->GetXSize(); ++i )
//                f += (*pRes)[y][i] * a[i][x];
//            res[y][x] = f;
//        }
//    }
//    *pRes = res;
//}
//static void Transpose( TArray2D<double> *pRes )
//{
//    TArray2D<double> &res = *pRes;
//    for ( int y = 0; y < res.GetYSize(); ++y )
//    {
//        for ( int x = y; x < res.GetXSize(); ++x )
//            swap( res[y][x], res[x][y] );
//    }
//}
//static TVector<double> Mult(const TArray2D<double> &m, const TVector<double> &v) {
//    ASSERT(m.GetXSize() == v.size());
//    TVector<double> res;
//    res.resize(m.GetYSize(), 0);
//    for (int y = 0; y < res.size(); ++y) {
//        double f = 0;
//        for (int x = 0; x < m.GetXSize(); ++x)
//            f += m[y][x] * v[x];
//        res[y] = f;
//    }
//    return res;
//}
//struct SEigTest
//{
//    SEigTest()
//    {
//        TArray2D<double> mtx, mRot;
//        mtx.SetSizes( 5, 5 );
//        mRot = mtx;
//        mtx.FillZero();
//        mtx[0][0] = 100;
//        mtx[1][1] = 34554;
//        mtx[2][2] = -1345;
//        mtx[3][3] = 0.01;
//        mtx[4][4] = -134;
//
//        for ( int k = 0; k < 1000; ++k )
//        {
//            mRot.FillZero();
//            for ( int i = 0; i < mRot.GetXSize(); ++i )
//                mRot[i][i] = 1;
//            int n1 = rand() % mtx.GetXSize();
//            int n2 = rand() % mtx.GetXSize();
//            if ( n1 == n2 )
//                continue;
//            float fAngle = ( rand() % 256 ) / 255.0f;
//            mRot[n2][n2] = mRot[n1][n1] = cos( fAngle );
//            mRot[n1][n2] = sin( fAngle );
//            mRot[n2][n1] = -sin( fAngle );
//            Mult( &mtx, mRot );
//            Transpose( &mRot );
//            Mult( &mRot, mtx );
//            mtx = mRot;
//        }
//        TVector<double> e;
//        TVector<TVector<double> > vecs;
//        CalcEigenTVectors(&e, &vecs, mtx);
//
//        TVector<double> vTest = vecs[0], vRes, vDel;
//        vRes = Mult(mtx, vTest);
//        for (int i = 0; i < vRes.size(); ++i)
//            vDel.push_back(vRes[i] / vTest[i]);
//        e[0] = 0;
//    }
//} tst;

}
