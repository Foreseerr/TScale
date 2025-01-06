#pragma once

template < class T >
struct CBoundCheck
{
    T *data;
    yint nSize;
    CBoundCheck(T *d, yint nS) { data = d; nSize = nS; }
    T &operator[](yint i) const { Y_ASSERT(i >= 0 && i < nSize); return data[i]; }
};


template < class T >
class TArray2D
{
    typedef T *PT;
    T *Data;
    T **pData;
    yint XSize, YSize;

    void Copy(const TArray2D &a)
    {
        if (this == &a) {
            return;
        }
        XSize = a.XSize;
        YSize = a.YSize;
        Create();
        if (std::is_trivially_copyable<T>::value) {
            memcpy(Data, a.Data, sizeof(T) * XSize * YSize);
        } else {
            for (yint i = 0; i < XSize * YSize; i++) {
                Data[i] = a.Data[i];
            }
        }
    }

    void Destroy()
    {
        delete[] Data;
        delete[] pData;
    }

    void Create()
    {
        Data = new T[XSize * YSize];
        pData = new PT[YSize];
        for (yint i = 0; i < YSize; i++) {
            pData[i] = Data + i * XSize;
        }
    }
public:
    TArray2D(yint xsize = 1, yint ysize = 1) { XSize = xsize; YSize = ysize; Create(); }
    TArray2D(const TArray2D &a) { Copy(a); }
    TArray2D &operator=(const TArray2D &a) { Destroy(); Copy(a); return *this; }
    ~TArray2D() { Destroy(); }
    void SetSizes(yint xsize, yint ysize) { if (XSize == xsize && YSize == ysize) return; Destroy(); XSize = xsize; YSize = ysize; Create(); }
    void Clear() { SetSizes(1, 1); }
#ifdef _DEBUG
    CBoundCheck<T> operator[](yint y) const { Y_ASSERT(y >= 0 && y < YSize); return CBoundCheck<T>(pData[y], XSize); }
#else
    T *operator[](yint y) const { ASSERT(y >= 0 && y < YSize); return pData[y]; }
#endif
    const T *GetRow(yint y) const { Y_ASSERT(y >= 0 && y < YSize); return pData[y]; }
    T *GetRow(yint y) { Y_ASSERT(y >= 0 && y < YSize); return pData[y]; }
    yint GetXSize() const { return XSize; }
    yint GetYSize() const { return YSize; }
    void FillZero() { memset(Data, 0, sizeof(T) * XSize * YSize); }
    void FillEvery(const T &a) { for (yint i = 0; i < XSize * YSize; i++) Data[i] = a; }
    void Swap(TArray2D &a) { swap(Data, a.Data); swap(pData, a.pData); swap(XSize, a.XSize); swap(YSize, a.YSize); }
};

template <class T>
yint GetXSize(const T &x) { return yint(x.GetXSize()); }
template <class T>
yint GetYSize(const T &x) { return yint(x.GetYSize()); }
