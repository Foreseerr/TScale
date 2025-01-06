/*
 *
 * Copyright (c) 1994
 * Hewlett-Packard Company
 *
 * Copyright (c) 1996,1997
 * Silicon Graphics Computer Systems, Inc.
 *
 * Copyright (c) 1997
 * Moscow Center for SPARC Technology
 *
 * Copyright (c) 1999
 * Boris Fomitchev
 *
 * This material is provided "as is", with absolutely no warranty expressed
 * or implied. Any use is at your own risk.
 *
 * Permission to use or copy this software for any purpose is hereby granted
 * without fee, provided the above notices are retained on all copies.
 * Permission to modify the code and to distribute modified code is granted,
 * provided the above notices are retained, and a notice that the code was
 * modified is included with the above copyright notice.
 *
 */

 /* NOTE: This is an internal header file, included by other STL headers.
  *   You should not attempt to use it directly.
  */
#pragma once
#include "nuninitialized.h"
#include <type_traits>

namespace nstl
{

template <class _Tp >
class vector
{
private:
    _Tp *_M_start;
    _Tp *_M_finish;
    _Tp *_M_end_of_storage;
    _Tp *alloc(yint __n) { return (_Tp *) new char[__n * sizeof(_Tp)]; }
    void free(_Tp *p) { delete[]((char *)p); }
public:
    typedef _Tp value_type;
    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef value_type *iterator;
    typedef const value_type *const_iterator;

public:
    typedef value_type &reference;
    typedef const value_type &const_reference;

protected:
    // add volatile to avoid stupid memcpy warning
    template <class T>
    T* fast_uninitialized_copy(const T * volatile beg, const T *fin, T *dst)
    {
        if (std::is_trivially_copyable<T>::value) {
            yint len = ((const char*)fin) - ((const char*)beg);
            memcpy(dst, beg, len);
            return (T*)((char*)dst + len);
        } else {
            return uninitialized_copy(beg, fin, dst);
        }
    }

    // handles insertions on overflow
    void _M_insert_overflow(pointer __position, const _Tp &__x,
        yint __fill_len, bool __atend = false) {
        const yint __old_size = size();
        const yint __len = __old_size + (max)(__old_size, __fill_len);

        pointer __new_start = alloc(__len);
        pointer __new_finish = __new_start;
        __new_finish = fast_uninitialized_copy(this->_M_start, __position, __new_start);
        // handle insertion
        if (__fill_len == 1) {
            _Construct(__new_finish, __x);
            ++__new_finish;
        } else
            __new_finish = uninitialized_fill_n(__new_finish, __fill_len, __x);
        if (!__atend)
            // copy remainder
            __new_finish = fast_uninitialized_copy(__position, this->_M_finish, __new_finish);
        _M_clear();
        _M_set(__new_start, __new_finish, __new_start + __len);
    }

    // handles insertions on overflow
    void _M_insert_overflow1(pointer __position,
        yint __fill_len, bool __atend = false) {
        const yint __old_size = size();
        const yint __len = __old_size + (max)(__old_size, __fill_len);

        pointer __new_start = alloc(__len);//this->_M_end_of_storage.allocate(__len);
        pointer __new_finish = __new_start;
        __new_finish = fast_uninitialized_copy(this->_M_start, __position, __new_start);
        // handle insertion
        if (__fill_len == 1) {
            _Construct(__new_finish);
            ++__new_finish;
        } else
            __new_finish = uninitialized_fill_n(__new_finish, __fill_len);
        if (!__atend)
            // copy remainder
            __new_finish = fast_uninitialized_copy(__position, this->_M_finish, __new_finish);
        _M_clear();
        _M_set(__new_start, __new_finish, __new_start + __len);
    }


public:
    iterator begin() { return this->_M_start; }
    const_iterator begin() const { return this->_M_start; }
    iterator end() { return this->_M_finish; }
    const_iterator end() const { return this->_M_finish; }

    _Tp *data() { return this->_M_start; }
    const _Tp *data() const { return this->_M_start; }

    ptrdiff_t size() const { return (this->_M_finish - this->_M_start); }
    ptrdiff_t capacity() const { return (this->_M_end_of_storage - this->_M_start); }
    bool empty() const { return this->_M_start == this->_M_finish; }

    reference operator[](ptrdiff_t __n) { ASSERT(__n >= 0 && __n < size()); return *(begin() + __n); }
    const_reference operator[](ptrdiff_t __n) const { ASSERT(__n >= 0 && __n < size()); return *(begin() + __n); }

    reference front() { return *begin(); }
    const_reference front() const { return *begin(); }
    reference back() { return *(end() - 1); }
    const_reference back() const { return *(end() - 1); }

    explicit vector() : _M_start(0), _M_finish(0), _M_end_of_storage(0) {}

    vector(yint __n, const _Tp &__val)
    {
        _M_start = alloc(__n);//_M_end_of_storage.allocate(__n);
        _M_end_of_storage = _M_start + __n;
        this->_M_finish = uninitialized_fill_n(this->_M_start, __n, __val);
    }

    explicit vector(yint __n)
    {
        _M_start = alloc(__n);//_M_end_of_storage.allocate(__n);
        _M_end_of_storage = _M_start + __n;
        _M_finish = uninitialized_fill_n(this->_M_start, __n);
    }

    vector(const vector<_Tp> &__x)
    {
        if (__x._M_start == 0) {
            _M_start = 0;
            _M_finish = 0;
            _M_end_of_storage = 0;
        } else {
            _M_start = alloc(__x.size());//_M_end_of_storage.allocate(__n);
            _M_end_of_storage = _M_start + __x.size();
            this->_M_finish = fast_uninitialized_copy((const_pointer)__x._M_start,
                (const_pointer)__x._M_finish, this->_M_start);
        }
    }

    // Check whether it's an integral type.  If so, it's not an iterator.
    template <class _InputIterator>
    vector(_InputIterator __first, _InputIterator __last) : _M_start(0), _M_finish(0), _M_end_of_storage(0)
    {
        for (; __first != __last; ++__first)
            push_back(*__first);
    }


    ~vector()
    {
        _Destroy(this->_M_start, this->_M_finish);
        if (_M_start != 0)
            free(_M_start);//_M_end_of_storage.deallocate(_M_start, _M_end_of_storage._M_data - _M_start); 
    }

    vector<_Tp> &operator=(const vector<_Tp> &__x);

    void reserve(yint __n);

    // assign(), a generalized assignment member function.  Two
    // versions: one that takes a count, and one that takes a range.
    // The range version is a member template, so we dispatch on whether
    // or not the type is an integer.

    void assign(yint __n, const _Tp &__val) { _M_fill_assign(__n, __val); }
    void _M_fill_assign(yint __n, const _Tp &__val);


    void push_back(const _Tp &__x) {
        if (this->_M_finish != this->_M_end_of_storage) {
            _Construct(this->_M_finish, __x);
            ++this->_M_finish;
        } else
            _M_insert_overflow(this->_M_finish, __x, 1UL, true);
    }

    void swap(vector<_Tp> &__x) {
        nstl::swap(this->_M_start, __x._M_start);
        nstl::swap(this->_M_finish, __x._M_finish);
        nstl::swap(this->_M_end_of_storage, __x._M_end_of_storage);
    }

    iterator insert(iterator __position, const _Tp &__x) {
        yint __n = __position - begin();
        if (this->_M_finish != this->_M_end_of_storage) {
            if (__position == end()) {
                _Construct(this->_M_finish, __x);
                ++this->_M_finish;
            } else {
                _Construct(this->_M_finish, *(this->_M_finish - 1));
                ++this->_M_finish;
                _Tp __x_copy = __x;
                copy_backward(__position, this->_M_finish - 2, this->_M_finish - 1);
                *__position = __x_copy;
            }
        } else
            _M_insert_overflow(__position, __x, 1UL);
        return begin() + __n;
    }

    void push_back() { push_back(_Tp()); }
    iterator insert(iterator __position) { return insert(__position, _Tp()); }

    void _M_fill_insert(iterator __pos, yint __n, const _Tp &__x);
    void _M_fill_insert(iterator __pos, yint __n);

    template <class _InputIterator>
    void insert(iterator __pos, _InputIterator __first, _InputIterator __last) {
        for (; __first != __last; ++__first) {
            __pos = insert(__pos, *__first);
            ++__pos;
        }
    }

    void insert(iterator __pos, yint __n, const _Tp &__x) { _M_fill_insert(__pos, __n, __x); }

    void pop_back() {
        --this->_M_finish;
        _Destroy(this->_M_finish);
    }
    iterator erase(iterator __position) {
        if (__position + 1 != end())
            copy(__position + 1, this->_M_finish, __position);
        --this->_M_finish;
        _Destroy(this->_M_finish);
        return __position;
    }
    iterator erase(iterator __first, iterator __last) {
        pointer __i = copy(__last, this->_M_finish, __first);
        _Destroy(__i, this->_M_finish);
        this->_M_finish = __i;
        return __first;
    }

    void resize(yint __new_size, const _Tp &__x) {
        if (__new_size < size())
            erase(begin() + __new_size, end());
        else
            _M_fill_insert(end(), __new_size - size(), __x);
    }
    void resize(yint __new_size) {
        if (__new_size < size())
            erase(begin() + __new_size, end());
        else
            _M_fill_insert(end(), __new_size - size(), _Tp());
    }
    void yresize(yint __new_size)
    {
        if (__new_size < size())
            erase(begin() + __new_size, end());
        else
            _M_fill_insert(end(), __new_size - size());
    }
    void clear() {
        _Destroy(this->_M_start, this->_M_finish);
        if (_M_start != 0)
            free(_M_start);//_M_end_of_storage.deallocate(_M_start, _M_end_of_storage._M_data - _M_start); 
        _M_start = 0;
        _M_finish = 0;
        _M_end_of_storage = 0;
    }

protected:

    void _M_clear() {
        //    if (this->_M_start) {
        _Destroy(this->_M_start, this->_M_finish);
        free(_M_start);//this->_M_end_of_storage.deallocate(this->_M_start, this->_M_end_of_storage._M_data - this->_M_start);
        _M_start = 0;
        //    }
    }

    void _M_set(pointer __s, pointer __f, pointer __e) {
        this->_M_start = __s;
        this->_M_finish = __f;
        this->_M_end_of_storage = __e;
    }

    template <class _ForwardIterator>
    pointer _M_allocate_and_copy(yint __n, _ForwardIterator __first,
        _ForwardIterator __last)
    {
        pointer __result = alloc(__n);//this->_M_end_of_storage.allocate(__n);
        fast_uninitialized_copy(__first, __last, __result);
        return __result;
    }
};


template <class _Tp>
void inline vector<_Tp>::reserve(yint __n)
{
    if (capacity() < __n) {
        const yint __old_size = size();
        pointer __tmp;
        if (this->_M_start) {
            __tmp = _M_allocate_and_copy(__n, this->_M_start, this->_M_finish);
            _M_clear();
        } else {
            __tmp = alloc(__n);//this->_M_end_of_storage.allocate(__n);
        }
        _M_set(__tmp, __tmp + __old_size, __tmp + __n);
    }
}


template <class _Tp>
void inline vector<_Tp>::_M_fill_insert(iterator __position, yint __n, const _Tp &__x)
{
    if (__n != 0) {
        if (yint(this->_M_end_of_storage - this->_M_finish) >= __n) {
            _Tp __x_copy = __x;
            const yint __elems_after = this->_M_finish - __position;
            pointer __old_finish = this->_M_finish;
            if (__elems_after > __n) {
                fast_uninitialized_copy(this->_M_finish - __n, this->_M_finish, this->_M_finish);
                this->_M_finish += __n;
                copy_backward(__position, __old_finish - __n, __old_finish);
                nstl::fill(__position, __position + __n, __x_copy);
            } else {
                uninitialized_fill_n(this->_M_finish, __n - __elems_after, __x_copy);
                this->_M_finish += __n - __elems_after;
                fast_uninitialized_copy(__position, __old_finish, this->_M_finish);
                this->_M_finish += __elems_after;
                nstl::fill(__position, __old_finish, __x_copy);
            }
        } else
            _M_insert_overflow(__position, __x, __n);
    }
}


template <class _Tp>
void inline vector<_Tp>::_M_fill_insert(iterator __position, yint __n)
{
    if (__n != 0) {
        if (yint(this->_M_end_of_storage - this->_M_finish) >= __n) {
            const yint __elems_after = this->_M_finish - __position;
            pointer __old_finish = this->_M_finish;
            if (__elems_after > __n) {
                fast_uninitialized_copy(this->_M_finish - __n, this->_M_finish, this->_M_finish);
                this->_M_finish += __n;
                copy_backward(__position, __old_finish - __n, __old_finish);
                nstl::fill(__position, __position + __n, _Tp());
            } else {
                uninitialized_fill_n(this->_M_finish, __n - __elems_after);
                this->_M_finish += __n - __elems_after;
                fast_uninitialized_copy(__position, __old_finish, this->_M_finish);
                this->_M_finish += __elems_after;
                nstl::fill(__position, __old_finish, _Tp());
            }
        } else
            _M_insert_overflow1(__position, __n);
    }
}

template <class _Tp>
inline vector<_Tp> &vector<_Tp>::operator=(const vector<_Tp> &__x)
{
    if (&__x != this) {
        const yint __xlen = __x.size();
        if (__xlen > capacity()) {
            pointer __tmp = _M_allocate_and_copy(__xlen, (const_pointer)__x._M_start + 0, (const_pointer)__x._M_finish + 0);
            _M_clear();
            this->_M_start = __tmp;
            this->_M_end_of_storage = this->_M_start + __xlen;
        } else if (size() >= __xlen) {
            pointer __i = copy((const_pointer)__x._M_start + 0, (const_pointer)__x._M_finish + 0, (pointer)this->_M_start);
            _Destroy(__i, this->_M_finish);
        } else {
            copy((const_pointer)__x._M_start, (const_pointer)__x._M_start + size(), (pointer)this->_M_start);
            fast_uninitialized_copy((const_pointer)__x._M_start + size(), (const_pointer)__x._M_finish + 0, this->_M_finish);
        }
        this->_M_finish = this->_M_start + __xlen;
    }
    return *this;
}


template <class _Tp>
inline void vector<_Tp>::_M_fill_assign(yint __n, const _Tp &__val)
{
    if (__n > capacity()) {
        vector<_Tp> __tmp(__n, __val);
        __tmp.swap(*this);
    } else if (__n > size()) {
        fill(begin(), end(), __val);
        this->_M_finish = uninitialized_fill_n(this->_M_finish, __n - size(), __val);
    } else
        erase(nstl::fill_n(begin(), __n, __val), end());
}


template <class _Tp>
inline bool  operator==(const vector<_Tp> &__x, const vector<_Tp> &__y)
{
    return __x.size() == __y.size() &&
        equal(__x.begin(), __x.end(), __y.begin());
}

template <class T>
inline bool operator!=(const vector<T> &a, const vector<T> &b)
{
    return !(a == b);
}
}
