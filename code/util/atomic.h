#pragma once

#include <atomic>

typedef std::atomic<yint> TAtomic;
typedef yint TAtomicValue;

//////////////////////////////////////////////////////////////////////////
struct TThrRefBase
{
public:
    inline TThrRefBase() 
        : Counter(0)
    {
    }

    inline void Ref(yint d) {
        Counter.fetch_add(d);
    }

    inline void Ref() {
        Counter.fetch_add(1);
    }

    inline void UnRef(yint d) {
        yint resultCount = Counter.fetch_add(-d) - d;
        Y_ASSERT(resultCount >= 0);
        if (resultCount == 0) {
            delete this;
        }
    }

    inline void UnRef() {
        UnRef(1);
    }

    inline yint RefCount() const {
        return Counter;
    }

    inline void DecRef() {
        Counter.fetch_add(-1);
    }

    TThrRefBase(const TThrRefBase&)
        : Counter(0)
    {
    }

    void operator =(const TThrRefBase&) {
    }

    virtual ~TThrRefBase() {}

private:
    TAtomic Counter;
};

//////////////////////////////////////////////////////////////////////////
template <class T>
class TIntrusivePtr
{
public:
    inline TIntrusivePtr(T* t = 0)
        : Ptr(t)
    {
        Ref();
    }

    inline ~TIntrusivePtr() {
        UnRef();
    }

    inline TIntrusivePtr(const TIntrusivePtr &p)
        : Ptr(p.Ptr)
    {
        Ref();
    }

    template <class T1>
    inline TIntrusivePtr(const TIntrusivePtr<T1> &p)
        : Ptr(p.Get())
    {
        Ref();
    }

    inline TIntrusivePtr& operator= (TIntrusivePtr p) {
        p.Swap(*this);
        return *this;
    }

    inline T* Get() const {
        return Ptr;
    }

    inline void Swap(TIntrusivePtr& r) {
        DoSwap(Ptr, r.Ptr);
    }

    inline void Drop() {
        TIntrusivePtr(0).Swap(*this);
    }

    inline T* Release() {
        T* res = Ptr;
        if (Ptr) {
            Ptr->DecRef();
            Ptr = 0;
        }
        return res;
    }

    inline long RefCount() const {
        return Ptr ? Ptr->RefCount() : 0;
    }

    inline T& operator* () const {
        Y_ASSERT(this->AsT());

        return *(this->AsT());
    }

    inline T* operator-> () const {
        return AsT();
    }

    template <class C>
    inline bool operator== (const C& p) const {
        return (p == AsT());
    }

    template <class C>
    inline bool operator!= (const C& p) const {
        return (p != AsT());
    }

    inline bool operator! () const {
        return 0 == AsT();
    }

private:
    inline T* AsT() const {
        return Ptr;
    }

    inline void Ref() {
        if (Ptr) {
            Ptr->Ref();
        }
    }

    inline void UnRef() {
        if (Ptr) {
            Ptr->UnRef();
        }
    }

private:
    T* Ptr;
};

//////////////////////////////////////////////////////////////////////////
struct TAtomicOps
{
    static void Acquire(TAtomic *x)
    {
        for (;;) {
            TAtomicValue exp = 0;
            if (x->compare_exchange_strong(exp, 1)) {
                break;
            }
            SchedYield();
        }
    }
    static void Release(TAtomic *x)
    {
        Y_ASSERT(*x == 1);
        *x = 0;
    }
};

template <class T, class TOps = TAtomicOps>
class TGuard: public TNonCopyable {
    public:
        inline TGuard(const T& t) {
            Init(&t);
        }

        inline TGuard(const T* t) {
            Init(t);
        }

        inline ~TGuard() {
            Release();
        }

        inline void Release() {
            if (WasAcquired()) {
                TOps::Release(T_);
                T_ = 0;
            }
        }

        inline bool WasAcquired() const {
            return T_ != 0;
        }

    private:
        inline void Init(const T* t) {
            T_ = const_cast<T*>(t); TOps::Acquire(T_);
        }

    private:
        T* T_;
};
