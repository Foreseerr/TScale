#pragma once


void SetIdlePriority();

typedef void (*EdenThreadRoutine)(void *);

struct TThreadRunInfo
{
    EdenThreadRoutine Func;
    void *Arg;

    TThreadRunInfo(EdenThreadRoutine f, void *a) : Func(f), Arg(a) {}
};

#ifdef _MSC_VER
class TThread
{
    HANDLE H = 0;

    static DWORD WINAPI ThreadProc(void *param)
    {
        TThreadRunInfo *pInfo = (TThreadRunInfo *)param;
        pInfo->Func(pInfo->Arg);
        delete pInfo;
        return 0;
    }

    template <class T>
    static DWORD WINAPI RunMember(void *param)
    {
        T *p = (T *)param;
        p->WorkerThread();
        return 0;
    }
public:
    void Create(EdenThreadRoutine func, void *arg)
    {
        Y_ASSERT(H == 0);
        H = CreateThread(0, 0, ThreadProc, new TThreadRunInfo(func, arg), 0, 0);
    }
    template <class T>
    void Create(T *arg)
    {
        Y_ASSERT(H == 0);
        H = CreateThread(0, 0, RunMember<T>, arg, 0, 0);
    }
    void Join()
    {
        if (H != 0) {
            WaitForSingleObject(H, INFINITE);
            CloseHandle(H);
            H = 0;
        }
    }
    ~TThread() { Join(); }
};

#else
#include <pthread.h>

class TThread
{
    volatile bool HasCreated = false;
    pthread_t H;

    static void* ThreadProc(void *param)
    {
        TThreadRunInfo *pInfo = (TThreadRunInfo *)param;
        pInfo->Func(pInfo->Arg);
        delete pInfo;
        return 0;
    }

    template <class T>
    static void *RunMember(void *param)
    {
        T *p = (T*)param;
        p->WorkerThread();
        return 0;
    }
public:
    void Create(EdenThreadRoutine func, void *arg)
    {
        Y_VERIFY(!HasCreated);
        int rv = pthread_create(&H, 0, ThreadProc, new TThreadRunInfo(func, arg));
        Y_VERIFY(rv == 0);
        HasCreated = true;
    }
    template <class T>
    void Create(T *arg)
    {
        Y_VERIFY(!HasCreated);
        int rv = pthread_create(&H, 0, RunMember<T>, arg);
        Y_VERIFY(rv == 0);
        HasCreated = true;
    }
    void Join()
    {
        if (HasCreated) {
            int rv = pthread_join(H, 0);
            Y_VERIFY(rv == 0);
            HasCreated = false;
        }
    }
    ~TThread() { Join(); }
};

#endif


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
struct TSingleConsumerJobQueue : public TNonCopyable
{
    struct TNode
    {
        TNode *Next = 0;
        T Val;
        TNode(const T &val) : Val(val) {}
    };
    std::atomic<TNode *> Head;

    TSingleConsumerJobQueue()
    {
        Head = 0;
    }

    ~TSingleConsumerJobQueue()
    {
        while (Head.load()) {
            TNode *p = Head.load();
            Head = p->Next;
            delete p;
        }
    }

    bool IsEmpty() const
    {
        return Head.load() == nullptr;
    }

    void Enqueue(const T &val)
    {
        TNode *pNode = new TNode(val);
        for (;;) {
            TNode *p = Head.load();
            pNode->Next = p;
            if (Head.compare_exchange_strong(p, pNode)) {
                return;
            }
        }
    }

    // retrieves in reverse order
    bool DequeueAll(TVector<T> *resArr)
    {
        for (;;) {
            TNode *p = Head.load();
            if (p) {
                if (Head.compare_exchange_strong(p, 0)) {
                    // ABA is impossible with single consumer
                    while (p) {
                        resArr->push_back(p->Val);
                        TNode *next = p->Next;
                        delete p;
                        p = next;
                    }
                    return true;
                }
            } else {
                return false;
            }
        }
    }

    bool DequeueFirst(T *res)
    {
        // ABA is impossible with single consumer
        for (;;) {
            TNode *p = Head.load();
            if (p) {
                if (p->Next) {
                    TNode **pLast = &p->Next;
                    while ((*pLast)->Next) {
                        pLast = &(*pLast)->Next;
                    }
                    *res = (*pLast)->Val;
                    delete *pLast;
                    *pLast = nullptr;
                    return true;
                } else if (Head.compare_exchange_strong(p, 0)) {
                    Y_ASSERT(p->Next == nullptr);
                    *res = p->Val;
                    delete p;
                    return true;
                }
            } else {
                return false;
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TJob, int BUF_SIZE>
struct TSingleProducerJobCircleBuffer
{
    struct TRWPtr
    {
        union
        {
            ui64 Val;
            struct {
                ui32 Read;
                ui32 Write;
            };
        };
        TRWPtr() {}
        TRWPtr(ui64 v) : Val(v) {}
    };

    std::atomic<ui64> RWPtr;
    ui64 Padding[7];
    TJob JobArr[BUF_SIZE];

    TSingleProducerJobCircleBuffer()
    {
        Y_ASSERT(IsPow2(BUF_SIZE));
        RWPtr = 0;
    }
    void Add(const TJob &job)
    {
        for (;;) {
            TRWPtr ptr(RWPtr.load());
            if (ptr.Write - ptr.Read == BUF_SIZE) {
                SchedYield();
            } else {
                JobArr[ptr.Write & (BUF_SIZE - 1)] = job;
                break;
            }
        }
        for (;;) {
            TRWPtr old(RWPtr.load());
            TRWPtr ptr = old;
            ptr.Write += 1;
            if (RWPtr.compare_exchange_strong(old.Val, ptr.Val)) {
                break;
            }
        }
    }
    bool Get(TJob *p)
    {
        for (;;) {
            TRWPtr old(RWPtr.load());
            TRWPtr ptr = old;
            if (ptr.Read == ptr.Write) {
                return false;
            }
            *p = JobArr[ptr.Read & (BUF_SIZE - 1)];
            ptr.Read += 1;
            if (RWPtr.compare_exchange_strong(old.Val, ptr.Val)) {
                return true;
            }
        }
    }
};

