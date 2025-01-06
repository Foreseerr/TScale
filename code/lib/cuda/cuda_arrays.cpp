#include "stdafx.h"
#include "cuda_arrays.h"


namespace NCuda
{
void GetAllData(const TCuda2DArray<half> &arr, TArray2D<float> *p)
{
    yint xSize = arr.GetXSize();
    yint ySize = arr.GetYSize();
    TArray2D<half> data;
    GetAllData(arr, &data);
    p->SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] = data[y][x];
        }
    }
}


void GetAllData(const TCuda2DArray<half> &arr, TVector<TVector<float>> *p)
{
    yint xSize = arr.GetXSize();
    yint ySize = arr.GetYSize();
    TArray2D<half> data;
    GetAllData(arr, &data);
    p->resize(ySize);
    for (yint y = 0; y < ySize; ++y) {
        (*p)[y].resize(xSize);
        for (yint x = 0; x < xSize; ++x) {
            (*p)[y][x] = data[y][x];
        }
    }
}


void Put(TStream &stream, TCuda2DArray<half> *arr, const TArray2D<float> &src)
{
    yint ySize = src.GetYSize();
    yint xSize = src.GetXSize();
    TArray2D<half> data;
    data.SetSizes(xSize, ySize);
    for (yint y = 0; y < ySize; ++y) {
        for (yint x = 0; x < xSize; ++x) {
            data[y][x] = src[y][x];
        }
    }
    Put(stream, arr, data);
}


void Put(TStream &stream, TCudaVector<half> *arr, const TVector<float> &src)
{
    yint sz = YSize(src);
    TVector<half> data;
    data.resize(sz);
    for (yint k = 0; k < sz; ++k) {
        data[k] = src[k];
    }
    Put(stream, arr, data);
}

}
