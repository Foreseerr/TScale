#pragma once
#include <cuda_runtime.h>

namespace NCuda
{
///////////////////////////////////////////////////////////////////////////////////////////////////
// cuda stream
class TStream : public TNonCopyable
{
    cudaStream_t Stream;
public:
    TStream() { cudaStreamCreate(&Stream); }
    ~TStream() { cudaStreamDestroy(Stream); }
    void Sync() { cudaStreamSynchronize(Stream); }
    operator cudaStream_t() const { return Stream; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// cuda memory pool
class TCudaMemoryPool : public TThrRefBase
{
    TIntrusivePtr<TCudaMemoryPool> Parent;
    yint TotalSize = 0;
    char *Ptr = 0;

    yint GetPoolOffset() const
    {
        yint res = 0;
        for (TCudaMemoryPool *p = Parent.Get(); p; p = p->Parent.Get()) {
            res += p->TotalSize;
        }
        return res;
    }
    void SetBaseDevicePtr(const void *base)
    {
        Ptr = ((char *)base) + GetPoolOffset();
    }
public:
    yint Allocate(yint sz)
    {
        Y_VERIFY(Ptr == 0 && "can not allocate after memory buffer is assigned");
        const yint ROUND_SIZE = 512;
        yint res = TotalSize;
        yint szRound = (sz + ROUND_SIZE - 1) & ~(ROUND_SIZE - 1);
        TotalSize += szRound;
        return res;
    }
    void *GetDevicePtr(yint offset) const
    {
        Y_VERIFY(Ptr != 0);
        return Ptr + offset;
    }
    yint GetMemSize() const
    {
        return GetPoolOffset() + TotalSize;
    }
    TCudaMemoryPool *GetParent() const
    {
        return Parent.Get();
    }

    friend class TCudaMemoryAllocator;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCudaMemoryAllocator : public TThrRefBase
{
    TVector<TIntrusivePtr<TCudaMemoryPool>> AllPools;
public:
    TIntrusivePtr<TCudaMemoryPool> CreatePool(TIntrusivePtr<TCudaMemoryPool> parent)
    {
        TIntrusivePtr<TCudaMemoryPool> p = new TCudaMemoryPool;
        p->Parent = parent;
        AllPools.push_back(p);
        return p;
    }
    TIntrusivePtr<TCudaMemoryPool> CreatePool()
    {
        return CreatePool(nullptr);
    }
    void AllocateMemory()
    {
        yint maxSize = 0;
        for (auto &p : AllPools) {
            maxSize = Max<yint>(maxSize, p->GetMemSize());
        }
        if (maxSize > 0) {
            void *ptr = 0;
            Y_VERIFY(cudaMalloc(&ptr, maxSize) == cudaSuccess);
            for (auto &p : AllPools) {
                p->SetBaseDevicePtr(ptr);
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TCudaMemory : public TNonCopyable
{
    yint SizeInBytes = 0;
    char *HostBuf = 0;
    void *DeviceBuf = 0;
    void *DeviceData = 0;
    TIntrusivePtr<TCudaMemoryPool> Pool;
    yint PoolOffset = 0;

public:
    enum {
        CUDA_ALLOC,
        CUDA_MAP,
    };

private:
    void Free()
    {
        if (HostBuf) {
            cudaFreeHost(HostBuf);
            HostBuf = 0;
        }
        if (DeviceBuf) {
            cudaFree(DeviceBuf);
            DeviceBuf = 0;
        }
        DeviceData = 0;
        Pool = 0;
    }
    void *GetDeviceBuf() const
    {
        if (DeviceBuf) {
            return DeviceBuf;
        }
        if (Pool.Get()) {
            return Pool->GetDevicePtr(PoolOffset);
        }
        return nullptr;
    }
    void *GetDeviceData() const
    {
        if (DeviceData) {
            return DeviceData;
        }
        if (Pool.Get()) {
            return Pool->GetDevicePtr(PoolOffset);
        }
        return nullptr;
    }

public:
    ~TCudaMemory()
    {
        Free();
    }
    yint GetSizeInBytes() const
    {
        return SizeInBytes;
    }

    // allocate
    void Allocate(yint sizeInBytes, int deviceAlloc, int hostAllocFlag)
    {
        Free();
        SizeInBytes = sizeInBytes;
        Y_VERIFY(sizeInBytes <= 0x20000000ll); // we are using int offsets for perf
        if (deviceAlloc == CUDA_ALLOC) {
            Y_VERIFY(cudaMalloc(&DeviceBuf, sizeInBytes) == cudaSuccess);
            DeviceData = DeviceBuf;
        }
        if (hostAllocFlag != -1) {
            Y_VERIFY(cudaHostAlloc(&HostBuf, sizeInBytes, hostAllocFlag) == cudaSuccess);
            if (DeviceData == 0) {
                Y_ASSERT(deviceAlloc == CUDA_MAP);
                Y_VERIFY(cudaHostGetDevicePointer(&DeviceData, HostBuf, 0) == cudaSuccess);
            }
        }
    }
    void AllocateFromCudaPool(yint sizeInBytes, TIntrusivePtr<TCudaMemoryPool> pool)
    {
        Free();
        SizeInBytes = sizeInBytes;
        Y_VERIFY(sizeInBytes <= 0x20000000ll); // we are using int offsets for perf
        Pool = pool;
        PoolOffset = pool->Allocate(sizeInBytes);
    }
    TIntrusivePtr<TCudaMemoryPool> GetMemPool() const
    {
        return Pool;
    }

    // mem ops
    void CopyToHost(const TStream &stream, yint sizeInBytes)
    {
        Y_ASSERT(sizeInBytes <= SizeInBytes);
        Y_ASSERT(HostBuf != 0);
        void *deviceBuf = GetDeviceBuf();
        if (deviceBuf) {
            cudaMemcpyAsync(HostBuf, deviceBuf, sizeInBytes, cudaMemcpyDeviceToHost, stream);
        } else {
            Y_VERIFY(0);
        }
    }
    void CopyToDevice(const TStream &stream, yint sizeInBytes)
    {
        Y_ASSERT(sizeInBytes <= SizeInBytes);
        Y_ASSERT(HostBuf != 0);
        void *deviceBuf = GetDeviceBuf();
        if (deviceBuf) {
            cudaMemcpyAsync(deviceBuf, HostBuf, sizeInBytes, cudaMemcpyHostToDevice, stream);
        } else {
            Y_VERIFY(0);
        }
    }
    void ClearHostMem()
    {
        memset(HostBuf, 0, SizeInBytes);
    }
    void ClearDeviceMem(const TStream &stream)
    {
        void *deviceBuf = GetDeviceBuf();
        if (deviceBuf) {
            cudaMemsetAsync(deviceBuf, 0, SizeInBytes, stream);
        } else {
            // no device memory in this case
            Y_VERIFY(0);
        }
    }

    // get mem address
    void *GetDevicePtr() const
    {
        return GetDeviceData();
    }
    void *GetHostPtr() const
    {
        return HostBuf;
    }
};

}
