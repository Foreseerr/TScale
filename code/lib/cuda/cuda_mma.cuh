#pragma once
#include "cuda_util.cuh"
#include "cuda_graph.cuh"


//
// 16x16 half float tiles with hardware accelerated matrix multiplication (tensor core)


namespace NCuda
{
const int TILE = 16;
const int TILE_GROUP = 4;
const int TILE_GROUP_SIZE = TILE * TILE_GROUP;


struct T4x4SMemHalfTile { int4 Data[32 * 16]; };
struct T4SMemHalfTile { int4 Data[32 * 4]; };
struct TSwizzledSmemTile { int4 Data[64]; };
struct TSwizzledSmemHalfTile { int4 Data[32]; };
struct TSwizzledSmemI8Tile { int2 Data[32]; };


///////////////////////////////////////////////////////////////////////////////////////////////////
// async cp utils
__forceinline __device__ ui32 GetSharedAddress(const void *ptr)
{
    return static_cast<ui32>(__cvta_generic_to_shared(ptr));
}

__forceinline __device__ void AsyncCopy16(void *dst, const void *src)
{
    ui32 sharedDstAddr = GetSharedAddress(dst);
    // best type of caching is unclear
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sharedDstAddr), "l"(src));
}

__forceinline __device__ void WaitAsyncCopy()
{
    asm volatile("cp.async.wait_all;\n" ::);
}

__forceinline __device__ void AsyncCommitGroup()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__forceinline __device__ void AsyncWaitGroup()
{
    Y_VERIFY(0);
}
template <>
__forceinline __device__ void AsyncWaitGroup<0>()
{
    asm volatile("cp.async.wait_all;\n" ::);
}
template <>
__forceinline __device__ void AsyncWaitGroup<1>()
{
    asm volatile("cp.async.wait_group 1;\n" ::);
}
template <>
__forceinline __device__ void AsyncWaitGroup<2>()
{
    asm volatile("cp.async.wait_group 2;\n" ::);
}

// synchronize multiple warps, uses same bar ID, modify to use different IDs
__forceinline__ __device__ void BarSync(int count)
{
    asm volatile("bar.sync 1, %0;" : : "r"(count));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// reg tiles

struct TTileCoord
{
    enum {
        num_elements = 8
    };
    int TX, TY; // 8x8 tile layout over 32 threads
public:
    __device__ TTileCoord()
    {
        int h = threadIdx.x;
        TX = (h & 3) * 2;
        TY = h / 4;
    }
    __forceinline __device__ int GetX(int elemIdx) const
    {
        return TX + (elemIdx & 1) + (elemIdx & 4) * 2;
    }
    __forceinline __device__ int GetY(int elemIdx) const
    {
        return TY + (elemIdx & 2) * 4;
    }
    __forceinline __device__ int GetRegTileRowIndex(int elemIdx) const
    {
        return (elemIdx & 2) >> 1;
    }
    template <class T>
    __forceinline __device__ void EnumElements(T func) const
    {
        // func(elem, tx, ty, value)
        func(0, TX, TY, 0, 0);
        func(1, TX + 1, TY, 0, 1);
        func(4, TX + 8, TY, 0, 2);
        func(5, TX + 9, TY, 0, 3);
        func(2, TX, TY + 8, 1, 0);
        func(3, TX + 1, TY + 8, 1, 1);
        func(6, TX + 8, TY + 8, 1, 2);
        func(7, TX + 9, TY + 8, 1, 3);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
struct TRegTileRow
{
    enum {
        num_elements = 2,
    };
    T x[2];

    __device__ int GetNumElements() const
    {
        return num_elements;
    }
    __device__ void Clear()
    {
        x[0] = 0;
        x[1] = 0;
    }
    __device__ void FillEvery(float val)
    {
        x[0] = val;
        x[1] = val;
    }
    __device__ void Load(const TTileCoord &tc, const T *rmax)
    {
        x[0] = rmax[tc.TY];
        x[1] = rmax[tc.TY + 8];
    }
    __device__ void StoreOne(const TTileCoord &tc, T *rmax)
    {
        if (tc.TX == 0) {
            rmax[tc.TY] = x[0];
            rmax[tc.TY + 8] = x[1];
        }
    }
    __device__ void Add(T val)
    {
        x[0] += val;
        x[1] += val;
    }
    __device__ void Scale(T val)
    {
        x[0] *= val;
        x[1] *= val;
    }
    __device__ void Scale(TRegTileRow<T> val)
    {
        x[0] *= val.x[0];
        x[1] *= val.x[1];
    }
    // max functions
    __device__ void SetMax(float val)
    {
        x[0] = val;
        x[1] = val;
    }
    __device__ void LoadMax(const TTileCoord &tc, const T *rmax)
    {
        Load(tc, rmax);
    }
    __device__ void StoreMax(const TTileCoord &tc, T *rmax)
    {
        WarpMaxReduce();
        StoreOne(tc, rmax);
    }
    __device__ void WarpMaxReduce()
    {
        for (int k = 0; k < 2; ++k) {
            x[k] = max(x[k], __shfl_xor_sync(0xffffffff, x[k], 1));
            x[k] = max(x[k], __shfl_xor_sync(0xffffffff, x[k], 2));
        }
    }
    // sum functions
    __device__ void SetSum(const TTileCoord &tc, float val)
    {
        if (tc.TX == 0) {
            x[0] = val;
            x[1] = val;
        } else {
            x[0] = 0;
            x[1] = 0;
        }
    }
    __device__ void LoadSum(const TTileCoord &tc, const T *rsum)
    {
        if (tc.TX == 0) {
            x[0] = rsum[tc.TY];
            x[1] = rsum[tc.TY + 8];
        } else {
            x[0] = 0;
            x[1] = 0;
        }
    }
    __device__ void StoreSum(const TTileCoord &tc, T *rsum)
    {
        T sum0 = x[0] + __shfl_xor_sync(0xffffffff, x[0], 1);
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
        T sum1 = x[1] + __shfl_xor_sync(0xffffffff, x[1], 1);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
        // we do not modify x[], so can not use StoreOne()
        if (tc.TX == 0) {
            rsum[tc.TY] = sum0;
            rsum[tc.TY + 8] = sum1;
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
struct TRegTileColumn
{
    T x[4];

    __device__ void Clear()
    {
        x[0] = 0;
        x[1] = 0;
        x[2] = 0;
        x[3] = 0;
    }
    __device__ void Load(const TTileCoord &tc, const T *data)
    {
        x[0] = data[tc.TX];
        x[1] = data[tc.TX + 1];
        x[2] = data[tc.TX + 8];
        x[3] = data[tc.TX + 9];
    }
    __device__ void Add(T val)
    {
        x[0] += val;
        x[1] += val;
        x[2] += val;
        x[3] += val;
    }
    __device__ void Scale(T mult)
    {
        x[0] *= mult;
        x[1] *= mult;
        x[2] *= mult;
        x[3] *= mult;
    }
    // max functions
    __device__ void SetMax(float val)
    {
        x[0] = val;
        x[1] = val;
        x[2] = val;
        x[3] = val;
    }
    __device__ void StoreMax(const TTileCoord &tc, T *rmax)
    {
        WarpMaxReduce();
        // save result
        if (tc.TY == 0) {
            rmax[tc.TX] = x[0];
            rmax[tc.TX + 1] = x[1];
            rmax[tc.TX + 8] = x[2];
            rmax[tc.TX + 9] = x[3];
        }
    }
    __device__ void WarpMaxReduce()
    {
        for (int k = 0; k < 4; ++k) {
            x[k] = max(x[k], __shfl_xor_sync(0xffffffff, x[k], 4));
            x[k] = max(x[k], __shfl_xor_sync(0xffffffff, x[k], 8));
            x[k] = max(x[k], __shfl_xor_sync(0xffffffff, x[k], 16));
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
struct TRegTile
{};


///////////////////////////////////////////////////////////////////////////////////////////////////
template<>
struct TRegTile<half>
{
    enum {
        //num_elements = 8,
        num_packed_elements = 4
    };
    // 4 8x8 tiles
    union {
        int4 nnx; // all data
        ui32 nx[4]; // Val00, Val10, Val01, Val11;
        half2 xx[4]; // Val00, Val10, Val01, Val11;
        half x[8];
    };

    __device__ TRegTile() {}
    __device__ TRegTile(const TRegTile &rr)
    {
        for (int k = 0; k < num_packed_elements; ++k) {
            nx[k] = rr.nx[k];
        }
    }
    __device__ TRegTile& operator=(const TRegTile &rr)
    {
        for (int k = 0; k < num_packed_elements; ++k) {
            nx[k] = rr.nx[k];
        }
        return *this;
    }

    __device__ void Clear()
    {
        for (int i = 0; i < num_packed_elements; ++i) {
            nx[i] = 0;
        }
    }

    __device__ void FillEvery(float val)
    {
        for (int i = 0; i < num_packed_elements; ++i) {
            xx[i] = make_half2(val, val);
        }
    }

    __device__ void FillIdentity(const TTileCoord &tc)
    {
        x[0] = (tc.TX + 0) == (tc.TY);
        x[1] = (tc.TX + 1) == (tc.TY);
        x[2] = (tc.TX + 0) == (tc.TY + 8);
        x[3] = (tc.TX + 1) == (tc.TY + 8);
        x[4] = (tc.TX + 8) == (tc.TY);
        x[5] = (tc.TX + 9) == (tc.TY);
        x[6] = (tc.TX + 8) == (tc.TY + 8);
        x[7] = (tc.TX + 9) == (tc.TY + 8);
    }

    __device__ void Scale(float x)
    {
        half2 mult = half2(x, x);
        for (int i = 0; i < num_packed_elements; ++i) {
            xx[i] *= mult;
        }
    }

    __device__ void RowMax(TRegTileRow<float> *p)
    {
        p->x[0] = max(p->x[0], float(max(max(x[0], x[1]), max(x[4], x[5]))));
        p->x[1] = max(p->x[1], float(max(max(x[2], x[3]), max(x[6], x[7]))));
    }

    __device__ void RowSum(TRegTileRow<float> *p)
    {
        half2 hsum0 = xx[0] + xx[2];
        p->x[0] += float(hsum0.x) + float(hsum0.y);
        half2 hsum1 = xx[1] + xx[3];
        p->x[1] += float(hsum1.x) + float(hsum1.y);
    }

    __device__ void AddRowScaled(const TRegTileRow<float> &p, float scale)
    {
        float a0 = p.x[0] * scale;
        x[0] += a0;
        x[1] += a0;
        x[4] += a0;
        x[5] += a0;
        float a1 = p.x[1] * scale;
        x[2] += a1;
        x[3] += a1;
        x[6] += a1;
        x[7] += a1;
    }

    __device__ void Scale(const TRegTileRow<float> &p)
    {
        float a0 = p.x[0];
        x[0] *= a0;
        x[1] *= a0;
        x[4] *= a0;
        x[5] *= a0;
        float a1 = p.x[1];
        x[2] *= a1;
        x[3] *= a1;
        x[6] *= a1;
        x[7] *= a1;
    }

    __device__ TRegTile<half> Transpose() const
    {
        TRegTile<half> res;
        asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(res.nx[0]) : "r"(nx[0]));
        asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(res.nx[2]) : "r"(nx[1]));
        asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(res.nx[1]) : "r"(nx[2]));
        asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(res.nx[3]) : "r"(nx[3]));
        return res;
    }

    // swizzled load/store
    __device__ void Load(const TSwizzledSmemHalfTile &ht)
    {
        nnx = ht.Data[threadIdx.x];
    }
    __device__ void Store(TSwizzledSmemHalfTile *p)
    {
        p->Data[threadIdx.x] = nnx;
    }

    // store
    template <class T>
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<T> p) const
    {
        StoreConvertedFloat2(xx[0], &p[tc.TY][tc.TX]);
        StoreConvertedFloat2(xx[2], &p[tc.TY][tc.TX + 8]);
        StoreConvertedFloat2(xx[1], &p[tc.TY + 8][tc.TX]);
        StoreConvertedFloat2(xx[3], &p[tc.TY + 8][tc.TX + 8]);
    }

    template <class T>
    __device__ void StoreScaled(const TTileCoord &tc, TCuda2DPtr<T> p, float mult) const
    {
        StoreScaledFloat2(xx[0], &p[tc.TY][tc.TX], mult);
        StoreScaledFloat2(xx[2], &p[tc.TY][tc.TX + 8], mult);
        StoreScaledFloat2(xx[1], &p[tc.TY + 8][tc.TX], mult);
        StoreScaledFloat2(xx[3], &p[tc.TY + 8][tc.TX + 8], mult);
    }

    template <class T>
    __device__ void StoreAddScaled(const TTileCoord &tc, TCuda2DPtr<T> p, float mult) const
    {
        StoreAddScaledFloat2(xx[0], &p[tc.TY][tc.TX], mult);
        StoreAddScaledFloat2(xx[2], &p[tc.TY][tc.TX + 8], mult);
        StoreAddScaledFloat2(xx[1], &p[tc.TY + 8][tc.TX], mult);
        StoreAddScaledFloat2(xx[3], &p[tc.TY + 8][tc.TX + 8], mult);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template<>
struct TRegTile<float>
{
    enum {
        num_elements = 8
    };
    // 4 8x8 tiles
    union {
        float x[8];// Vall00a, Val00b, Val10a, Val10b, Val01a, Val01b, Val11a, Val11b;
        float2 xx[4];
        int4 nnx[2]; // all data
    };

    __device__ void Clear()
    {
        for (int i = 0; i < num_elements; ++i) {
            x[i] = 0;
        }
    }

    __device__ void FillEvery(float val)
    {
        for (int i = 0; i < num_elements; ++i) {
            x[i] = val;
        }
    }

    __device__ void Scale(float scale)
    {
        for (int i = 0; i < num_elements; ++i) {
            x[i] *= scale;
        }
    }

    __device__ void RowMax(TRegTileRow<float> *p)
    {
        p->x[0] = max(p->x[0], max(max(x[0], x[1]), max(x[4], x[5])));
        p->x[1] = max(p->x[1], max(max(x[2], x[3]), max(x[6], x[7])));
    }

    __device__ void RowSum(TRegTileRow<float> *p)
    {
        p->x[0] += x[0] + x[1] + x[4] + x[5];
        p->x[1] += x[2] + x[3] + x[6] + x[7];
    }

    __device__ void AddRowScaled(const TRegTileRow<float> &p, float scale)
    {
        float a0 = p.x[0] * scale;
        x[0] += a0;
        x[1] += a0;
        x[4] += a0;
        x[5] += a0;
        float a1 = p.x[1] * scale;
        x[2] += a1;
        x[3] += a1;
        x[6] += a1;
        x[7] += a1;
    }

    __device__ void Scale(const TRegTileRow<float> &p)
    {
        float a0 = p.x[0];
        x[0] *= a0;
        x[1] *= a0;
        x[4] *= a0;
        x[5] *= a0;
        float a1 = p.x[1];
        x[2] *= a1;
        x[3] *= a1;
        x[6] *= a1;
        x[7] *= a1;
    }

    // swizzled load/store
    __device__ void Load(const TSwizzledSmemTile &ht)
    {
        nnx[0] = ht.Data[threadIdx.x];
        nnx[1] = ht.Data[threadIdx.x + 32];
    }
    __device__ void Store(TSwizzledSmemTile *p) const
    {
        p->Data[threadIdx.x] = nnx[0];
        p->Data[threadIdx.x + 32] = nnx[1];
    }

    // store
    template <class T>
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<T> p) const
    {
        StoreConvertedFloat2(xx[0], &p[tc.TY][tc.TX]);
        StoreConvertedFloat2(xx[2], &p[tc.TY][tc.TX + 8]);
        StoreConvertedFloat2(xx[1], &p[tc.TY + 8][tc.TX]);
        StoreConvertedFloat2(xx[3], &p[tc.TY + 8][tc.TX + 8]);
    }

    template <class T>
    __device__ void StoreScaled(const TTileCoord &tc, TCuda2DPtr<T> p, float mult) const
    {
        StoreScaledFloat2(xx[0], &p[tc.TY][tc.TX], mult);
        StoreScaledFloat2(xx[2], &p[tc.TY][tc.TX + 8], mult);
        StoreScaledFloat2(xx[1], &p[tc.TY + 8][tc.TX], mult);
        StoreScaledFloat2(xx[3], &p[tc.TY + 8][tc.TX + 8], mult);
    }

    template <class T>
    __device__ void StoreAddScaled(const TTileCoord &tc, TCuda2DPtr<T> p, float mult) const
    {
        StoreAddScaledFloat2(xx[0], &p[tc.TY][tc.TX], mult);
        StoreAddScaledFloat2(xx[2], &p[tc.TY][tc.TX + 8], mult);
        StoreAddScaledFloat2(xx[1], &p[tc.TY + 8][tc.TX], mult);
        StoreAddScaledFloat2(xx[3], &p[tc.TY + 8][tc.TX + 8], mult);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template<>
struct TRegTile<int>
{
    enum {
        num_elements = 8
    };
    // 4 8x8 tiles
    union {
        int x[8];// Vall00a, Val00b, Val10a, Val10b, Val01a, Val01b, Val11a, Val11b;
        int2 xx[4];
    };

    __device__ void Clear()
    {
        for (int i = 0; i < num_elements; ++i) {
            x[i] = 0;
        }
    }

    __device__ void FillEvery(int val)
    {
        for (int i = 0; i < num_elements; ++i) {
            x[i] = val;
        }
    }

    __device__ void Scale(float val)
    {
        for (int i = 0; i < num_elements; ++i) {
            x[i] = CvtToI32(x[i] * val);
        }
    }

    __device__ void RowMax(TRegTileRow<int> *p)
    {
        p->x[0] = max(p->x[0], max(max(x[0], x[1]), max(x[4], x[5])));
        p->x[1] = max(p->x[1], max(max(x[2], x[3]), max(x[6], x[7])));
    }

    __device__ void RowSum(TRegTileRow<int> *p)
    {
        p->x[0] += x[0] + x[1] + x[4] + x[5];
        p->x[1] += x[2] + x[3] + x[6] + x[7];
    }

    __device__ void Scale(const TRegTileRow<float> &p)
    {
        float a0 = p.x[0];
        x[0] *= a0;
        x[1] *= a0;
        x[4] *= a0;
        x[5] *= a0;
        float a1 = p.x[1];
        x[2] *= a1;
        x[3] *= a1;
        x[6] *= a1;
        x[7] *= a1;
    }

    // store
    template <class T>
    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<T> p) const
    {
        StoreConvertedFloat2(xx[0], &p[tc.TY][tc.TX]);
        StoreConvertedFloat2(xx[2], &p[tc.TY][tc.TX + 8]);
        StoreConvertedFloat2(xx[1], &p[tc.TY + 8][tc.TX]);
        StoreConvertedFloat2(xx[3], &p[tc.TY + 8][tc.TX + 8]);
    }

    template <class T>
    __device__ void StoreScaled(const TTileCoord &tc, TCuda2DPtr<T> p, float mult) const
    {
        StoreScaledFloat2(xx[0], &p[tc.TY][tc.TX], mult);
        StoreScaledFloat2(xx[2], &p[tc.TY][tc.TX + 8], mult);
        StoreScaledFloat2(xx[1], &p[tc.TY + 8][tc.TX], mult);
        StoreScaledFloat2(xx[3], &p[tc.TY + 8][tc.TX + 8], mult);
    }

    template <class T>
    __device__ void StoreAddScaled(const TTileCoord &tc, TCuda2DPtr<T> p, float mult) const
    {
        StoreAddScaledFloat2(xx[0], &p[tc.TY][tc.TX], mult);
        StoreAddScaledFloat2(xx[2], &p[tc.TY][tc.TX + 8], mult);
        StoreAddScaledFloat2(xx[1], &p[tc.TY + 8][tc.TX], mult);
        StoreAddScaledFloat2(xx[3], &p[tc.TY + 8][tc.TX + 8], mult);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template<>
struct TRegTile<i8>
{
    enum {
        //num_elements = 8,
        num_packed_elements = 2
    };
    // 2 16x8 tiles
    union {
        int2 nnx;
        ui32 nx[2];// Val00, Val10
        i8 x[8];
    };

    __device__ void Clear()
    {
        for (int i = 0; i < num_packed_elements; ++i) {
            nx[i] = 0;
        }
    }

    // fast Transpose() seems to be impossible

    __device__ void Load(const TSwizzledSmemI8Tile &ht)
    {
        nnx = ht.Data[threadIdx.x];
    }

    __device__ void Store(TSwizzledSmemI8Tile *p)
    {
        p->Data[threadIdx.x] = nnx;
    }

    __device__ void Load(const TTileCoord &tc, TCuda2DPtr<i8> p)
    {
        nx[0] = *(ui32 *)&p[tc.TY][tc.TX * 2];
        nx[1] = *(ui32 *)&p[tc.TY + 8][tc.TX * 2];
    }

    __device__ void Store(const TTileCoord &tc, TCuda2DPtr<i8> p)
    {
        *(ui32 *)&p[tc.TY][tc.TX * 2] = nx[0];
        *(ui32 *)&p[tc.TY + 8][tc.TX * 2] = nx[1];
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// load from swizzled format
__forceinline __device__ TRegTile<half> LoadTile(const TSwizzledSmemHalfTile &ht)
{
    TRegTile<half> res;
    res.Load(ht);
    return res;
}

__forceinline __device__ TRegTile<i8> LoadTile(const TSwizzledSmemI8Tile &ht)
{
    TRegTile<i8> res;
    res.Load(ht);
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// load 16x16 fp16 tile from shared memory
// prone to bank conflicts
__forceinline __device__ void LoadFromSmem(TRegTile<half> *p, TCuda2DPtr<half> data)
{
    int h = threadIdx.x;
    int offsetX = (h & 16) >> 1;
    int offsetY = h & 15;
    ui32 sharedAddr = GetSharedAddress(&data[offsetY][offsetX]);
    asm volatile ("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(p->nx[0]), "=r"(p->nx[1]), "=r"(p->nx[2]), "=r"(p->nx[3])
        : "r"(sharedAddr));
}

__forceinline __device__ void LoadFromSmemTransposed(TRegTile<half> *p, TCuda2DPtr<half> data)
{
    int h = threadIdx.x;
    int offsetX = (h & 16) >> 1;
    int offsetY = h & 15;
    ui32 sharedAddr = GetSharedAddress(&data[offsetY][offsetX]);
    asm volatile ("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(p->nx[0]), "=r"(p->nx[2]), "=r"(p->nx[1]), "=r"(p->nx[3])
        : "r"(sharedAddr));
}

__forceinline __device__ void CopyTileToSmem(TCuda2DPtr<half> dst, TCuda2DPtr<half> src)
{
    int h = threadIdx.x;
    int sy = h / 2;
    int sx = h & 1;
    *(float4 *)&dst[sy][sx * 8] = *(float4 *)&src[sy][sx * 8];
}

__forceinline __device__ void CopyTileToSmemAsync(TCuda2DPtr<half> dst, TCuda2DPtr<half> src)
{
    int h = threadIdx.x;
    int sy = h / 2;
    int sx = h & 1;
    AsyncCopy16(&dst[sy][sx * 8], &src[sy][sx * 8]);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// shared memory 64x64 fp16 tile
// we operate 8x1 blocks (16 bytes), block address is computed as [y][x ^ (y&7)]
// using xor operation avoids bank conflicts in both situations
//   when storing blocks to smem horizontally (copying mem -> smem)
//   when load blocks from smem vertically (copying smem -> registers)

__forceinline __device__ void LoadTile(TRegTile<half> *p, const T4x4SMemHalfTile &ht, int x, int y)
{
    int h = threadIdx.x;
    int y7 = h & 7;
    int tx = h / 16;
    int ty = h & 15;
    int threadOffset = (ty * 8 + tx);
    int rowAddr = threadOffset + y * (16 * 8) + x * 2;
    ui32 sharedAddr = GetSharedAddress(&ht.Data[rowAddr ^ y7]);
    asm volatile ("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(p->nx[0]), "=r"(p->nx[1]), "=r"(p->nx[2]), "=r"(p->nx[3])
        : "r"(sharedAddr));
}

__forceinline __device__ void LoadTileTransposed(TRegTile<half> *p, const T4x4SMemHalfTile &ht, int x, int y)
{
    int h = threadIdx.x;
    int y7 = h & 7;
    int tx = h / 16;
    int ty = h & 15;
    int threadOffset = (ty * 8 + tx);
    int rowAddr = threadOffset + x * (16 * 8) + y * 2;
    ui32 sharedAddr = GetSharedAddress(&ht.Data[rowAddr ^ y7]);

    asm volatile ("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(p->nx[0]), "=r"(p->nx[2]), "=r"(p->nx[1]), "=r"(p->nx[3])
        : "r"(sharedAddr));
}

__forceinline __device__ TRegTile<half> LoadTile(const T4x4SMemHalfTile &ht, int x, int y)
{
    TRegTile<half> res;
    LoadTile(&res, ht, x, y);
    return res;
}

__forceinline __device__ TRegTile<half> LoadTileTransposed(const T4x4SMemHalfTile &ht, int x, int y)
{
    TRegTile<half> res;
    LoadTileTransposed(&res, ht, x, y);
    return res;
}

// load with 4 warps
__forceinline __device__ void Copy4x4Tile(T4x4SMemHalfTile *p, int warpId, TCuda2DPtr<half> data)
{
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    int4 *dst = p->Data + ((y * 8 + x) ^ y7);
    int4 *src = (int4 *)&data[y][x * 8];
    for (int k = 0; k < 4; ++k) {
        *dst = *src;
        dst = AdvancePtr(dst, TILE_GROUP_SIZE * TILE * sizeof(half));
        src = AdvancePtr(src, data.GetStrideInBytes() * TILE);
    }
}

// load with 4 warps
__forceinline __device__ void Copy4x4TileAsync(T4x4SMemHalfTile *p, int warpId, TCuda2DPtr<half> data)
{
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    int4 *dst = p->Data + ((y * 8 + x) ^ y7);
    int4 *src = (int4 *)&data[y][x * 8];
    for (int k = 0; k < 4; ++k) {
        AsyncCopy16(dst, src);
        dst = AdvancePtr(dst, TILE_GROUP_SIZE * TILE * sizeof(half));
        src = AdvancePtr(src, data.GetStrideInBytes() * TILE);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// shared memory 64x16 fp16 tile
__forceinline __device__ void LoadTile(TRegTile<half> *p, const T4SMemHalfTile &ht, int x)
{
    LoadTile(p, *(const T4x4SMemHalfTile *)&ht, x, 0);
}

__forceinline __device__ void LoadTileTransposed(TRegTile<half> *p, const T4SMemHalfTile &ht, int x)
{
    LoadTileTransposed(p, *(const T4x4SMemHalfTile *)&ht, 0, x);
}

__forceinline __device__ TRegTile<half> LoadTile(const T4SMemHalfTile &ht, int x)
{
    TRegTile<half> res;
    LoadTile(&res, ht, x);
    return res;
}

__forceinline __device__ TRegTile<half> LoadTileTransposed(const T4SMemHalfTile &ht, int x)
{
    TRegTile<half> res;
    LoadTileTransposed(&res, ht, x);
    return res;
}

// load with 4 warps
__forceinline __device__ void Copy4Tile(T4SMemHalfTile *p, int warpId, TCuda2DPtr<half> data)
{
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    p->Data[(y * 8 + x) ^ y7] = *(int4 *)&data[y][x * 8];
}

__forceinline __device__ void Copy4TileAsync(T4SMemHalfTile *p, int warpId, TCuda2DPtr<half> data)
{
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    AsyncCopy16(&p->Data[(y * 8 + x) ^ y7], &data[y][x * 8]);
}

// load with 1 warp
__forceinline __device__ void Copy4Tile(T4SMemHalfTile *p, TCuda2DPtr<half> data)
{
    for (int warpId = 0; warpId < 4; ++warpId) {
        Copy4Tile(p, warpId, data);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// shared memory 128x16 i8 tile
const int I8_TILE_GROUP = 8;
const int I8_TILE_GROUP_SIZE = TILE * I8_TILE_GROUP;

struct T8SMemI8Tile { int4 Data[16 * 8]; };

// load with 4 warps
template <class T>
__forceinline __device__ void Copy8Tile(T8SMemI8Tile *p, int warpId, TCuda2DPtr<T> data)
{
    CUDA_ASSERT(sizeof(T) == 1);
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    p->Data[(y * 8 + x) ^ y7] = *(int4 *)&data[y][x * 16];
}

// load with 4 warps
template <class T>
__forceinline __device__ void Copy8TileArray(T8SMemI8Tile *p, int warpId, TCuda2DPtr<T> data, int rowCount, int dataOffset)
{
    CUDA_ASSERT(sizeof(T) == 1);
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    int4 *dst = &p->Data[(y * 8 + x) ^ y7];
    int4 *src = (int4 *)&data[y][x * 16];
    for (int k = 0; k < rowCount; ++k) {
        *dst = *src;
        dst = AdvancePtr(dst, sizeof(T8SMemI8Tile));
        src = AdvancePtr(src, dataOffset);
    }
}

// load with 4 warps
template <class T>
__forceinline __device__ void Copy8TileAsync(T8SMemI8Tile *p, int warpId, TCuda2DPtr<T> data)
{
    CUDA_ASSERT(sizeof(T) == 1);
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    AsyncCopy16(&p->Data[(y * 8 + x) ^ y7], &data[y][x * 16]);
}

// load with 1 warp
template <class T>
__forceinline __device__ void Copy8Tile(T8SMemI8Tile *p, TCuda2DPtr<T> data)
{
    CUDA_ASSERT(sizeof(T) == 1);
    for (int warpId = 0; warpId < 4; ++warpId) {
        Copy8Tile(p, warpId, data);
    }
}

__forceinline __device__ void LoadTile(TRegTile<i8> *p, const T8SMemI8Tile &ht, int tileId)
{
    int h = threadIdx.x;
    int y7 = h & 7;
    int ty = h; // we use ldmatrix.x2, so only first 16 threads are utilized, first 8 threads load upper 16x8 tile, second 8 threads load bottom 16x8 tile
    int rowAddr = ty * 8 + tileId;
    ui32 sharedAddr = GetSharedAddress(&ht.Data[rowAddr ^ y7]);
    asm volatile ("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(p->nx[0]), "=r"(p->nx[1])
        : "r"(sharedAddr));
}

__forceinline __device__ TRegTile<i8> LoadTile(const T8SMemI8Tile &ht, int x)
{
    TRegTile<i8> res;
    LoadTile(&res, ht, x);
    return res;
}

template <class T>
__forceinline __device__ void LoadFromSmem(TRegTile<i8> *p, TCuda2DPtr<T> data)
{
    CUDA_ASSERT(sizeof(T) == 1);
    int h = threadIdx.x;
    ui32 sharedAddr = GetSharedAddress(&data[h & 15][0]);
    asm volatile ("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(p->nx[0]), "=r"(p->nx[1])
        : "r"(sharedAddr));
}

// instead of tileId pass tileOffset = tileId * 16, somehow generates faster code (skips x16?)
__forceinline __device__ void SetElement(T8SMemI8Tile *p, int tileOffset, int x, int y, i8 value)
{
    i8 *row = ((i8*)&p->Data[0]) + y * 128;
    int y7 = y & 7;
    row[x + (tileOffset ^ (y7 * 16))] = value;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// shared memory 64x16 and 64x64 fp16 tile, conversion from i8 source
union TConvertFloat4Half8
{
    int4 Int4;
    half Half8[8];
};

union TConvertFloat2Char8
{
    int2 Int2;
    i8 Char8[8];
};

__forceinline __device__ int4 Convert8xChartoHalf(int2 arg)
{
    TConvertFloat2Char8 src;
    src.Int2 = arg;
    TConvertFloat4Half8 dst;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        dst.Half8[k] = src.Char8[k];
    }
    return dst.Int4;
}

// load with 4 warps
__forceinline __device__ void Copy4Tile(T4SMemHalfTile *p, int warpId, TCuda2DPtr<i8> data)
{
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    int2 src = *(int2 *)&data[y][x * 8];
    p->Data[(y * 8 + x) ^ y7] = Convert8xChartoHalf(src);
}

// load with 1 warp
__forceinline __device__ void Copy4Tile(T4SMemHalfTile *p, TCuda2DPtr<i8> data)
{
    for (int warpId = 0; warpId < 4; ++warpId) {
        Copy4Tile(p, warpId, data);
    }
}

// load with 4 warps
__forceinline __device__ void Copy4x4Tile(T4x4SMemHalfTile *p, int warpId, TCuda2DPtr<i8> data)
{
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    int4 *dst = p->Data + ((y * 8 + x) ^ y7);
    int2 *src = (int2 *)&data[y][x * 8];
    for (int k = 0; k < 4; ++k) {
        *dst = Convert8xChartoHalf(*src);
        dst = AdvancePtr(dst, TILE_GROUP_SIZE * TILE * sizeof(half));
        src = AdvancePtr(src, data.GetStrideInBytes() * TILE);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// shared memory 64x16 and 64x64 fp16 tile, conversion from e4 source
__forceinline __device__ int4 Convert8xE4toHalf(int2 arg)
{
    TConvertFloat2Char8 src;
    src.Int2 = arg;
    TConvertFloat4Half8 dst;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        dst.Half8[k] = CvtE4ToHalf(src.Char8[k]);
    }
    return dst.Int4;
}

// load with 4 warps
__forceinline __device__ void Copy4Tile(T4SMemHalfTile *p, int warpId, TCuda2DPtr<e4m3> data)
{
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    int2 src = *(int2 *)&data[y][x * 8];
    p->Data[(y * 8 + x) ^ y7] = Convert8xE4toHalf(src);
}

// load with 1 warp
__forceinline __device__ void Copy4Tile(T4SMemHalfTile *p, TCuda2DPtr<e4m3> data)
{
    for (int warpId = 0; warpId < 4; ++warpId) {
        Copy4Tile(p, warpId, data);
    }
}

// load with 4 warps
__forceinline __device__ void Copy4x4Tile(T4x4SMemHalfTile *p, int warpId, TCuda2DPtr<e4m3> data)
{
    int h = threadIdx.x;
    int x = h & 7;
    int y = (h / 8) + warpId * 4;
    int y7 = y & 7;
    int4 *dst = p->Data + ((y * 8 + x) ^ y7);
    int2 *src = (int2 *)&data[y][x * 8];
    for (int k = 0; k < 4; ++k) {
        *dst = Convert8xE4toHalf(*src);
        dst = AdvancePtr(dst, TILE_GROUP_SIZE * TILE * sizeof(half));
        src = AdvancePtr(src, data.GetStrideInBytes() * TILE);
    }
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// tile mma, Matrix Multiply Add operations
// mul row col
__forceinline __device__ void MMA(TRegTile<float> *pD, const TRegTile<half> &a, const TRegTile<half> &b, const TRegTile<float> &c)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        :
    "=f"(pD->x[0]), "=f"(pD->x[1]), "=f"(pD->x[2]), "=f"(pD->x[3]) // "=f" means overwrite, "+f" means read-modify-write
        :
        "r"(a.nx[0]), "r"(a.nx[1]), "r"(a.nx[2]), "r"(a.nx[3]),
        "r"(b.nx[0]), "r"(b.nx[2]),
        "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3])
        );
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5, %6, %7 }," // A
        " { %8, %9 }," // B
        " { %10, %11, %12, %13 };" // C
        :
    "=f"(pD->x[4]), "=f"(pD->x[5]), "=f"(pD->x[6]), "=f"(pD->x[7])
        :
        "r"(a.nx[0]), "r"(a.nx[1]), "r"(a.nx[2]), "r"(a.nx[3]),
        "r"(b.nx[1]), "r"(b.nx[3]),
        "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7])
        );
}

__forceinline __device__ void MMA(TRegTile<float> *pD, const TRegTile<half> &a, const TRegTile<half> &b)
{
    MMA(pD, a, b, *pD);
}

__forceinline __device__ void MMA(TRegTile<half> *pD, const TRegTile<half> &a, const TRegTile<half> &b, const TRegTile<half> &c)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
        " { %0, %1 }," // D
        " { %2, %3, %4, %5 }," // A
        " { %6, %7 }," // B
        " { %8, %9 };" // C
        :
    "=r"(pD->nx[0]), "=r"(pD->nx[1])
        :
        "r"(a.nx[0]), "r"(a.nx[1]), "r"(a.nx[2]), "r"(a.nx[3]),
        "r"(b.nx[0]), "r"(b.nx[2]),
        "r"(c.nx[0]), "r"(c.nx[1])
        );
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
        " { %0, %1 }," // D
        " { %2, %3, %4, %5 }," // A
        " { %6, %7 }," // B
        " { %8, %9 };" // C
        :
    "=r"(pD->nx[2]), "=r"(pD->nx[3])
        :
        "r"(a.nx[0]), "r"(a.nx[1]), "r"(a.nx[2]), "r"(a.nx[3]),
        "r"(b.nx[1]), "r"(b.nx[3]),
        "r"(c.nx[2]), "r"(c.nx[3])
        );
}

__forceinline __device__ void MMA(TRegTile<half> *pD, const TRegTile<half> &a, const TRegTile<half> &b)
{
    MMA(pD, a, b, *pD);
}


__forceinline __device__ void MMA(TRegTile<int> *pD, const TRegTile<i8> &a, const TRegTile<i8> &b, const TRegTile<int> &c)
{
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5 }," // A
        " { %6 }," // B
        " { %7, %8, %9, %10 };" // C
        :
    "=r"(pD->x[0]), "=r"(pD->x[1]), "=r"(pD->x[2]), "=r"(pD->x[3]) // "=f" means overwrite, "+f" means read-modify-write
        :
        "r"(a.nx[0]), "r"(a.nx[1]),
        "r"(b.nx[0]),
        "r"(c.x[0]), "r"(c.x[1]), "r"(c.x[2]), "r"(c.x[3])
        );
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32"
        " { %0, %1, %2, %3 }," // D
        " { %4, %5 }," // A
        " { %6 }," // B
        " { %7, %8, %9, %10 };" // C
        :
    "=r"(pD->x[4]), "=r"(pD->x[5]), "=r"(pD->x[6]), "=r"(pD->x[7])
        :
        "r"(a.nx[0]), "r"(a.nx[1]),
        "r"(b.nx[1]),
        "r"(c.x[4]), "r"(c.x[5]), "r"(c.x[6]), "r"(c.x[7])
        );
}

__forceinline __device__ void MMA(TRegTile<int> *pD, const TRegTile<i8> &a, const TRegTile<i8> &b)
{
    MMA(pD, a, b, *pD);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
struct TMmaRowMajor
{
    static __device__ TRegTile<half> FragA(const TRegTile<half> &tt) { return tt; }
    static __device__ TRegTile<half> FragB(const TRegTile<half> &tt) { return tt.Transpose(); }
    static __device__ TRegTile<half> FragA(const T4x4SMemHalfTile &ht, int x, int y) { return LoadTile(ht, x, y); }
    static __device__ TRegTile<half> FragB(const T4x4SMemHalfTile &ht, int x, int y) { return LoadTileTransposed(ht, y, x); }
    static __device__ TRegTile<half> FragA(const T4SMemHalfTile &ht, int k) { return LoadTile(ht, k); }
    static __device__ TRegTile<half> FragB(const T4SMemHalfTile &ht, int k) { return LoadTileTransposed(ht, k); }
    static __device__ TRegTile<half> FragA(const TCuda2DPtr<half> data) { TRegTile<half> res; LoadFromSmem(&res, data); return res; }
    static __device__ TRegTile<half> FragB(const TCuda2DPtr<half> data) { TRegTile<half> res; LoadFromSmemTransposed(&res, data); return res; }
    static __device__ TRegTile<i8> FragA(const TRegTile<i8> &tt) { return tt; }
    static __device__ TRegTile<i8> FragA(const T8SMemI8Tile &ht, int k) { return LoadTile(ht, k); }
};

struct TMmaColMajor
{
    static __device__ TRegTile<half> FragA(const TRegTile<half> &tt) { return tt.Transpose(); }
    static __device__ TRegTile<half> FragB(const TRegTile<half> &tt) { return tt; }
    static __device__ TRegTile<half> FragA(const T4x4SMemHalfTile &ht, int x, int y) { return LoadTileTransposed(ht, x, y); }
    static __device__ TRegTile<half> FragB(const T4x4SMemHalfTile &ht, int x, int y) { return LoadTile(ht, y, x); }
    static __device__ TRegTile<half> FragA(const T4SMemHalfTile &ht, int k) { return LoadTileTransposed(ht, k); }
    static __device__ TRegTile<half> FragB(const T4SMemHalfTile &ht, int k) { return LoadTile(ht, k); }
    static __device__ TRegTile<half> FragA(const TCuda2DPtr<half> data) { TRegTile<half> res; LoadFromSmemTransposed(&res, data); return res; }
    static __device__ TRegTile<half> FragB(const TCuda2DPtr<half> data) { TRegTile<half> res; LoadFromSmem(&res, data); return res; }
    static __device__ TRegTile<i8> FragB(const TRegTile<i8> &tt) { return tt; }
    static __device__ TRegTile<i8> FragB(const T8SMemI8Tile &ht, int k) { return LoadTile(ht, k); }
};


void TestMMA();
}
