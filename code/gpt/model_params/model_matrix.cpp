#include "stdafx.h"
#include "model_matrix.h"

const bool FP16_PACK = true;
//const bool FP16_PACK = false;

void PackMatrix(TBufferedStream &f, const TArray2D<float> &matr)
{
    yint xSize = matr.GetXSize();
    yint ySize = matr.GetYSize();
    TVector<fp16> packedRow;
    packedRow.resize(xSize);
    Y_VERIFY((xSize % 16) == 0);
    for (yint y = 0; y < ySize; ++y) {
        if (FP16_PACK) {
            const float *matrRow = matr.GetRow(y);
            float sum2 = HorizontalSum(CalcRowSum2(matrRow, xSize));
            float sko = sqrt(sum2 / xSize);
            f.Write(&sko, sizeof(sko));
            if (sko != 0) {
                __m256 mult = _mm256_set1_ps(1 / sko);
                ConvertToFp16(packedRow.data(), matrRow, xSize, mult);
                f.Write(packedRow.data(), xSize * sizeof(packedRow[0]));
            }
        } else {
            const float *matrRow = matr.GetRow(y);
            f.Write(matrRow, xSize * sizeof(float));
        }
    }
}


void AddPackedMatrix(TArray2D<float> *p, TBufferedStream &f, float scale)
{
    TArray2D<float> &matr = *p;
    yint xSize = matr.GetXSize();
    yint ySize = matr.GetYSize();
    TVector<fp16> packedRow;
    packedRow.resize(xSize);
    TVector<float> vec;
    vec.resize(xSize);
    Y_VERIFY((xSize % 16) == 0);
    for (yint y = 0; y < ySize; ++y) {
        if (FP16_PACK) {
            float discrScale = 0;
            f.Read(&discrScale, sizeof(discrScale));
            if (discrScale != 0) {
                __m256 mult = _mm256_set1_ps(discrScale * scale);
                f.Read(packedRow.data(), xSize * sizeof(packedRow[0]));
                AddScaledFp16Array(matr.GetRow(y), packedRow.data(), xSize, mult);
            }
        } else {
            f.Read(vec.data(), xSize * sizeof(float));
            float *row = matr.GetRow(y);
            for (yint x = 0; x < xSize; ++x) {
                row[x] += vec[x] * scale;
            }
        }
    }
}

