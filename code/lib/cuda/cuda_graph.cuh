#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_arrays.h"

namespace NCuda
{

enum EOpDep
{
    DEP_NONE = 0,
    DEP_READ = 1, // op reads data
    DEP_ATOMICWRITE = 2, // op writes data atomically, can have several simlultaneious such ops
    DEP_READWRITE = 3, // op modifies or rewrites data, other writes should be complete before this op
    DEP_OVERWRITE = DEP_READWRITE, // no concurrent writes are allowed

    DEP_IS_READ = 1,
    DEP_IS_WRITE = 2,
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TGraphOp : public TThrRefBase
{
    bool NeedSetParams = false;
    TVector<const void *> ReadSet;
    TVector<const void *> WriteSet;
    TVector<TIntrusivePtr<TCudaMemoryPool>> MemPools;

protected:
    cudaGraph_t Graph = 0;
    cudaGraphNode_t Node = 0;

    virtual void SetParams(cudaGraphExec_t) {}
    virtual void CreateNode() = 0;
    virtual TIntrusivePtr<TCudaMemoryPool> GetNewMemPool()
    {
        return nullptr;
    }
    void ParamsUpdated()
    {
        NeedSetParams = false;
    }

public:
    TGraphOp(cudaGraph_t graph) : Graph(graph) {}
    void OnParamsChange()
    {
        NeedSetParams = true;
    }
    void UpdateParams(cudaGraphExec_t execGraph)
    {
        if (NeedSetParams) {
            SetParams(execGraph);
        }
    }

    // deps
    void AddDeps(EOpDep dep, const void *p)
    {
        if (dep & DEP_IS_READ) {
            ReadSet.push_back(p);
        }
        if (dep & DEP_IS_WRITE) {
            WriteSet.push_back(p);
        }
    }
    void AddDepOverwrite(const void *p)
    {
        // we want to wait other writes to complete, can do it by adding to ReadSet
        ReadSet.push_back(p);
        WriteSet.push_back(p);
    }
    void AddMemPool(TIntrusivePtr<TCudaMemoryPool> pool)
    {
        if (pool.Get()) {
            MemPools.push_back(pool);
        }
    }
    const void *MakeDepPtrFromDevicePtr(void *p)
    {
        // mark device pointer with lowest bit to avoid host&device pointers match
        char *ptr = (char *)p;
        return (ptr + 1);
    }

    friend class TGraph;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
class TOpParameter : public TNonCopyable
{
    struct TRef
    {
        TIntrusivePtr<TGraphOp> Op;
        T *Data;
    };
    T Val = T();
    TVector<TRef> RefArr;
public:
    void Set(const T &newValue)
    {
        if (Val != newValue) {
            Val = newValue;
            for (const TRef &x : RefArr) {
                *x.Data = newValue;
                x.Op->OnParamsChange();
            }
        }
    }
    void AddRef(T *p, TGraphOp *op)
    {
        TRef x;
        x.Data = p;
        x.Op = op;
        RefArr.push_back(x);
        *p = Val;
    }
    const T &Get() const
    {
        return Val;
    }
};



///////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA Kernel call op
class TKernelOp : public TGraphOp
{
    enum {
        PARAM_BUF_SIZE = 384
    };
    cudaKernelNodeParams CudaParams;
    char KParamBuf[PARAM_BUF_SIZE];
    TVector<void *> KParamList;
    yint KParamPtr = 0;
    bool IsStructParaam = false;
    yint StructOffset = 0;

    void CreateNode() override
    {
        CudaParams.kernelParams = KParamList.data();
        Y_ASSERT(Node == 0);
        cudaError_t err = cudaGraphAddKernelNode(&Node, Graph, 0, 0, &CudaParams);
        if (err != cudaSuccess) {
            abort();
        }
        ParamsUpdated();
    }

    void SetParams(cudaGraphExec_t execGraph)
    {
        cudaError_t err = cudaGraphExecKernelNodeSetParams(execGraph, Node, &CudaParams);
        if (err != cudaSuccess) {
            abort();
        }
        ParamsUpdated();
    }

    template <class T>
    struct TAlignmentComputer
    {
        char C;
        T Data;
    };

    template <class T>
    void AddParam(const T &val, EOpDep dep)
    {
        Y_ASSERT((dep & DEP_IS_WRITE) == 0); // allow write to arrays only atm
        if (IsStructParaam) {
            // alignment
            TAlignmentComputer<T> *pComputer = nullptr;
            yint alignSize = (char*) &pComputer->Data - (char*)nullptr;
            int alignedOffset = (StructOffset + (alignSize - 1)) & ~(alignSize - 1);
            KParamPtr += alignedOffset - StructOffset;
            StructOffset = alignedOffset + sizeof(T);
        } else {
            KParamList.push_back(KParamBuf + KParamPtr);
        }
        Y_VERIFY(KParamPtr + sizeof(T) <= PARAM_BUF_SIZE);
        T *pParamPlace = (T *)(KParamBuf + KParamPtr);
        *pParamPlace = val;
        KParamPtr += sizeof(T);
    }

    template <class T>
    void AddParam(const TCudaPOD<T> &param, EOpDep dep)
    {
        void *pDeviceData = param.GetDevicePtr();
        const void *owner = param.GetOwner();
        Y_ASSERT(owner);
        AddParam(pDeviceData, DEP_NONE);
        AddDeps(DEP_READ, owner); // should depend on writes to whole owner like clearmem
        AddDeps(dep, MakeDepPtrFromDevicePtr(pDeviceData));
        AddMemPool(param.GetMemPool());
    }

    template <class T>
    void AddParam(const TCuda2DArrayFragment<T> &param, EOpDep dep)
    {
        const void *owner = param.GetOwner();
        Y_ASSERT(owner);
        AddParam(param.GetDevicePtr(), DEP_NONE);
        AddDeps(dep, owner);
        AddMemPool(param.GetMemPool());
    }

    template <class T>
    void AddParam(const TCudaVector<T> &param, EOpDep dep)
    {
        AddParam(param.GetDevicePtr(), DEP_NONE);
        AddDeps(dep, &param);
        AddMemPool(param.GetMemPool());
    }

    template <class T>
    void AddParam(const TCuda2DArray<T> &param, EOpDep dep)
    {
        AddParam(param.GetDevicePtr(), DEP_NONE);
        AddDeps(dep, &param);
        AddMemPool(param.GetMemPool());
    }

    template <class T>
    void AddParam(TOpParameter<T> &param, EOpDep dep) // has to be non constant
    {
        T *pParamPlace = (T *)(KParamBuf + KParamPtr);
        AddParam(param.Get(), DEP_NONE);
        param.AddRef(pParamPlace, this);
        Y_ASSERT((dep & DEP_IS_WRITE) == 0); // writing to parameters is prohibited
    }

public:
    TKernelOp(cudaGraph_t graph, void *kernel) : TGraphOp(graph)
    {
        Zero(CudaParams);
        CudaParams.func = kernel;
        CudaParams.blockDim = dim3(WARP_SIZE);
        CudaParams.gridDim = dim3(1);
    }

    // Shmem
    TKernelOp &Shmem(int sz)
    {
        CudaParams.sharedMemBytes = sz;
        return *this;
    }

    // Grid
    TKernelOp &Grid(int x, int y = 1, int z = 1)
    {
        CudaParams.gridDim = dim3(x, y, z);
        return *this;
    }
    TKernelOp &Grid(TOpParameter<int> &x, int y = 1, int z = 1)
    {
        CudaParams.gridDim = dim3(0, y, z);
        Y_ASSERT(sizeof(CudaParams.gridDim.x) == sizeof(int));
        x.AddRef((int*)&CudaParams.gridDim.x, this);
        return *this;
    }
    TKernelOp &Grid(int x, TOpParameter<int> &y, int z = 1)
    {
        CudaParams.gridDim = dim3(x, 0, z);
        Y_ASSERT(sizeof(CudaParams.gridDim.y) == sizeof(int));
        y.AddRef((int *)&CudaParams.gridDim.y, this);
        return *this;
    }

    // Block
    TKernelOp &Block(int x, int y = 1, int z = 1)
    {
        CudaParams.blockDim = dim3(x, y, z);
        return *this;
    }

    // Struct
    TKernelOp &Struct()
    {
        IsStructParaam = true;
        char *pParamPlace = (KParamBuf + KParamPtr);
        KParamList.push_back(pParamPlace);
        StructOffset = 0;
        return *this;
    }
    TKernelOp &Params()
    {
        IsStructParaam = false;
        return *this;
    }

    // pass kernel parameters
    template <typename T>
    TKernelOp &operator()(const T &param)
    {
        AddParam(param, DEP_READ);
        return *this;
    }
    template <typename T>
    TKernelOp &operator()(T &param)
    {
        AddParam(param, DEP_READ);
        return *this;
    }
    template <typename T, typename... TRest>
    TKernelOp &operator()(const T &param, TRest&&... x)
    {
        AddParam(param, DEP_READ);
        return (*this)(x...);
    }
    template <typename T, typename... TRest>
    TKernelOp &operator()(T &param, TRest&&... x)
    {
        AddParam(param, DEP_READ);
        return (*this)(x...);
    }

    // kernel target params 
    template <typename T>
    TKernelOp &Write(T *param)
    {
        AddParam(*param, DEP_READWRITE);
        return *this;
    }
    template <typename T, typename... TRest>
    TKernelOp &Write(T *param, TRest... x)
    {
        AddParam(*param, DEP_READWRITE);
        return Write(x...);
    }

    // no write-write dependencies
    template <typename T>
    TKernelOp &AtomicWrite(T *param)
    {
        AddParam(*param, DEP_ATOMICWRITE);
        return *this;
    }
    template <typename T, typename... TRest>
    TKernelOp &AtomicWrite(T *param, TRest... x)
    {
        AddParam(*param, DEP_ATOMICWRITE);
        return AtomicWrite(x...);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TSetMemPoolOp : public TGraphOp
{
    TIntrusivePtr<TCudaMemoryPool> Pool;

    void CreateNode() override
    {
        if (cudaGraphAddEmptyNode(&Node, Graph, 0, 0) != cudaSuccess) {
            abort();
        }
    }
    TIntrusivePtr<TCudaMemoryPool> GetNewMemPool() override
    {
        return Pool;
    }
public:
    TSetMemPoolOp(cudaGraph_t graph, TIntrusivePtr<TCudaMemoryPool> pool) : TGraphOp(graph), Pool(pool)
    {
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TMemsetOp : public TGraphOp
{
    cudaMemsetParams Params;
    int RowSize = 0;
    int YSize = 0; // potential type mismatch with Params.height, so keep it separated
    int MaxYSize = 0;

    template <class T>
    void Init(T &arr)
    {
        TMemoryBlob blob = arr.GetDeviceMem();
        RowSize = blob.Stride;
        YSize = blob.YSize;
        MaxYSize = blob.YSize;
        Zero(Params);
        Params.dst = blob.Ptr;
        Params.elementSize = 1;
        Params.height = 1;
        Params.pitch = 0;
        Params.value = 0;
        Params.width = blob.Stride * YSize;
        // deps
        AddDepOverwrite(&arr);
    }

    void CreateNode() override
    {
        Y_ASSERT(YSize <= MaxYSize);
        Params.width = RowSize * YSize;
        Y_ASSERT(Node == 0);
        cudaError_t err = cudaGraphAddMemsetNode(&Node, Graph, 0, 0, &Params);
        Y_VERIFY(err == cudaSuccess);
        ParamsUpdated();
    }

    void SetParams(cudaGraphExec_t execGraph)
    {
        Y_ASSERT(YSize <= MaxYSize);
        Params.width = RowSize * YSize;
        cudaError_t err = cudaGraphExecMemsetNodeSetParams(execGraph, Node, &Params);
        Y_VERIFY(err == cudaSuccess);
        ParamsUpdated();
    }

public:
    template <class T>
    TMemsetOp(cudaGraph_t graph, T &arr) : TGraphOp(graph)
    {
        Init(arr);
    }
    template <class T>
    TMemsetOp(cudaGraph_t graph, T &arr, TOpParameter<int> &ySize) : TGraphOp(graph)
    {
        Init(arr);
        ySize.AddRef(&YSize, this);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TMemcpyOp : public TGraphOp
{
    void *Dst = 0;
    void *Src = 0;
    size_t Size = 0;
    cudaMemcpyKind OpType;

    void CreateNode() override
    {
        if (cudaGraphAddMemcpyNode1D(&Node, Graph, 0, 0, Dst, Src, Size, OpType) != cudaSuccess) {
            abort();
        }
    }
public:
    template <class TDst, class TSrc>
    TMemcpyOp(cudaGraph_t graph, TDst *dst, const TSrc &src, EMemType srcMemType, EMemType dstMemType, cudaMemcpyKind opType) : TGraphOp(graph), OpType(opType)
    {
        TMemoryBlob srcBlob = src.GetMem(srcMemType);
        TMemoryBlob dstBlob = dst->GetMem(dstMemType);
        Y_VERIFY(srcBlob.IsSameSize(dstBlob));
        Dst = dstBlob.Ptr;
        Src = srcBlob.Ptr;
        Size = dstBlob.GetSize();
        // deps
        AddDeps(DEP_READ, &src);
        AddDepOverwrite(dst);
    }
};


const int KERNEL_COPY_BLOCK = 32;
__global__ void KernelCopyImpl(int4 *dst, int4 *src, int len);


///////////////////////////////////////////////////////////////////////////////////////////////////
// default kernel block size
THashMap<TString, dim3> &GetKernelBlockSize();
#define KERNEL_BLOCK_SIZE(a, ...) namespace { struct TKernelBlock##a { TKernelBlock##a() {\
    Y_ASSERT(GetKernelBlockSize().find(KERNEL_UNIT #a) == GetKernelBlockSize().end());\
    GetKernelBlockSize()[KERNEL_UNIT #a] = dim3(__VA_ARGS__);\
} } setKernelBlockSize##a; }


///////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA Graph
struct TCudaOpDependencies;
class TGraph : public TThrRefBase
{
    cudaGraph_t Graph;
    cudaGraphExec_t ExecGraph = 0;
    TVector<TIntrusivePtr<TGraphOp>> OpArr;

    void AddMemPoolDeps(TCudaOpDependencies *pDep);
    void AddLinearDeps(TCudaOpDependencies *pDep);
    void AddDeps(TCudaOpDependencies *pDep);
    void CreateExecGraph();
public:
    TGraph()
    {
        cudaGraphCreate(&Graph, 0);
    }
    ~TGraph()
    {
        if (ExecGraph != 0) {
            cudaGraphExecDestroy(ExecGraph);
        }
        cudaGraphDestroy(Graph);
    }
    TKernelOp &CudaCallImplementation(const TString &kernelUnit, const TString &kernelName, void *kernel)
    {
        TKernelOp *p = new TKernelOp(Graph, kernel);
        TString kernelFuncName = kernelName.substr(0, kernelName.find('<'));
        const THashMap<TString, dim3> &kernelBlockSize = GetKernelBlockSize();
        auto it = kernelBlockSize.find(kernelUnit + kernelFuncName);
        if (it != kernelBlockSize.end()) {
            p->Block(it->second.x, it->second.y, it->second.z);
        }
        OpArr.push_back(p);
        return *p;
    }

    // memory ops
    void SetMemPool(TIntrusivePtr<TCudaMemoryPool> pool)
    {
        OpArr.push_back(new TSetMemPoolOp(Graph, pool));
    }
    template <class T>
    void ClearMem(T &arr)
    {
        OpArr.push_back(new TMemsetOp(Graph, arr));
    }
    template <class T>
    void ClearMem(T &arr, TOpParameter<int> &ySize)
    {
        OpArr.push_back(new TMemsetOp(Graph, arr, ySize));
    }
    template <class TDst, class TSrc>
    void CopyToHost(TDst *dst, const TSrc &src)
    {
        OpArr.push_back(new TMemcpyOp(Graph, dst, src, MT_DEVICE, MT_HOST, cudaMemcpyDeviceToHost));
    }
    template <class TDst, class TSrc>
    void CopyToDevice(TDst *dst, const TSrc &src)
    {
        OpArr.push_back(new TMemcpyOp(Graph, dst, src, MT_HOST, MT_DEVICE, cudaMemcpyHostToDevice));
    }

    // use kernel to copy arrays (avoid WDDM induced delays on Windows)
    template <class TDst, class TSrc>
    void KernelCopy(TDst *dst, const TSrc &src, yint ySize)
    {
        TMemoryBlob srcBlob = src.GetMem(MT_DEVICE);
        TMemoryBlob dstBlob = dst->GetMem(MT_DEVICE);
        Y_VERIFY(srcBlob.Stride == dstBlob.Stride && srcBlob.YSize >= ySize && dstBlob.YSize >= ySize);
        TIntrusivePtr<TKernelOp> p = new TKernelOp(Graph, (void*)KernelCopyImpl);
        (*p)(dstBlob.Ptr, srcBlob.Ptr, srcBlob.Stride * ySize);
        (*p).Block(WARP_SIZE, KERNEL_COPY_BLOCK);
        // deps
        p->AddDeps(DEP_READ, &src);
        p->AddDepOverwrite(dst);
        OpArr.push_back(p.Get());
    }
    template <class TDst, class TSrc>
    void KernelCopy(TDst *dst, const TSrc &src)
    {
        TMemoryBlob srcBlob = src.GetMem(MT_DEVICE);
        TMemoryBlob dstBlob = dst->GetMem(MT_DEVICE);
        Y_VERIFY(srcBlob.IsSameSize(dstBlob));
        KernelCopy(dst, src, srcBlob.YSize);
    }

    // run
    void Run(TStream &stream);
};

#define CudaCall(c, kernel, ...) c->CudaCallImplementation(KERNEL_UNIT, #kernel, (void*)kernel, ##__VA_ARGS__)

}
