/*
 * Copyright (c) 1997-1999
 * Silicon Graphics Computer Systems, Inc.
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

#pragma once

#include "nalgobase.h"
#include "nhash_fun.h"

 // Standard C++ string class.  This class has performance
 // characteristics very much like vector<>, meaning, for example, that
 // it does not perform reference-count or copy-on-write, and that
 // concatenation of two strings is an O(N) operation. 

 // There are three reasons why basic_string is not identical to
 // vector.  First, basic_string always stores a null character at the
 // end; this makes it possible for c_str to be a fast operation.
 // Second, the C++ standard requires basic_string to copy elements
 // using char_traits<>::assign, char_traits<>::copy, and
 // char_traits<>::move.  This means that all of vector<>'s low-level
 // operations must be rewritten.  Third, basic_string<> has a lot of
 // extra functions in its interface that are convenient but, strictly
 // speaking, redundant.

 // Additionally, the C++ standard imposes a major restriction: according
 // to the standard, the character type T must be a POD type.  This
 // implementation weakens that restriction, and allows T to be a
 // a user-defined non-POD type.  However, T must still have a
 // default constructor.

namespace nstl
{
// ------------------------------------------------------------
// Class _String_base.  

// _String_base is a helper class that makes it it easier to write an
// exception-safe version of basic_string.  The constructor allocates,
// but does not initialize, a block of memory.  The destructor
// deallocates, but does not destroy elements within, a block of
// memory.  The destructor assumes that _M_start either is null, or else
// points to a block of memory that was allocated using _String_base's 
// allocator and whose size is _M_end_of_storage - _M_start.

template <class _Tp> class _String_base
{
public:
    _Tp *_M_start;
    _Tp *_M_finish;
    _Tp *_M_end_of_storage;

    // Precondition: 0 < __n <= max_size().
    void _M_allocate_block(size_t n)
    {
        _M_start = new _Tp[n];
        _M_finish = _M_start;
        _M_end_of_storage = _M_start + n;
    }

    _String_base()
        : _M_start(0), _M_finish(0), _M_end_of_storage(0) {}

    _String_base(size_t __n)
    {
        _M_allocate_block(__n);
    }

    ~_String_base() { delete[] _M_start; }
};


inline size_t __strlen(const char *p) { return strlen(p); }
inline size_t __strlen(const wchar_t *p) { return wcslen(p); }
inline yint __strncmp(const char *p1, const char *p2, size_t n) { return strncmp(p1, p2, n); }
inline yint __strncmp(const wchar_t *p1, const wchar_t *p2, size_t n) { return wcsncmp(p1, p2, n); }

// ------------------------------------------------------------
// Class basic_string.  

// Class invariants:
// (1) [start, finish) is a valid range.
// (2) Each iterator in [start, finish) points to a valid object
//     of type value_type.
// (3) *finish is a valid object of type value_type; in particular,
//     it is value_type().
// (4) [finish + 1, end_of_storage) is a valid range.
// (5) Each iterator in [finish + 1, end_of_storage) points to 
//     unininitialized memory.

// Note one important consequence: a string of length n must manage
// a block of memory whose size is at least n + 1.  

struct _Reserve_t {};

//template <class T> 
template <class T>
class basic_string : protected nstl::_String_base<T>
{
private:                        // Protected members inherited from base.
    typedef nstl::_String_base<T> _Base;
    typedef basic_string<T> _Self;

public:
    typedef T value_type;

    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef value_type &reference;
    typedef const value_type &const_reference;

    typedef const value_type *const_iterator;
    typedef value_type *iterator;

    static const size_t npos = ~(size_t)0;

public:                         // Constructor, destructor, assignment.
    basic_string() : _String_base<T>(8)
    {
        *this->_M_finish = 0; // _M_terminate_string();  
    }

    basic_string(_Reserve_t, size_t __n) : _String_base<T>(__n + 1)
    {
        *this->_M_finish = 0; // _M_terminate_string();  
    }

    basic_string(const basic_string<T> &__s)
    {
        _M_range_initialize(__s._M_start, __s._M_finish);
    }

    basic_string(const T *__s, size_t __n)
    {
        _M_range_initialize(__s, __s + __n);
    }

    basic_string(const T *__s)
    {
        _M_range_initialize(__s, __s + __strlen(__s));
    }

    basic_string(const T *__f, const T *__l)
    {
        _M_range_initialize(__f, __l);
    }


    _Self &operator=(const _Self &__s) {
        if (&__s != this)
            assign(__s._M_start, __s._M_finish);
        return *this;
    }

    _Self &operator=(const T *__s) {
        return assign(__s, __s + __strlen(__s));
    }

    _Self &operator=(T __c)
    {
        return assign(1, __c);
    }


private:
    // Helper functions used by constructors.  It is a severe error for
    // any of them to be called anywhere except from within constructors.



    template <class _InputIter> void _M_range_initialize(_InputIter __f, _InputIter __l) {
        size_t __n = size_t(__l - __f);
        this->_M_allocate_block(__n + 1);
        this->_M_finish = copy(__f, __l, this->_M_start);
        *this->_M_finish = 0; // _M_terminate_string();  
    }

public:                         // Iterators.
    iterator begin() { return this->_M_start; }
    iterator end() { return this->_M_finish; }
    const_iterator begin() const { return this->_M_start; }
    const_iterator end()   const { return this->_M_finish; }

public:                         // Size, capacity, etc.
    size_t size() const { return this->_M_finish - this->_M_start; }
    size_t length() const { return size(); }

    void resize(size_t __n, T __c) {
        if (__n <= size())
            erase(begin() + __n, end());
        else
            append(__n - size(), __c);
    }
    void resize(size_t __n) { resize(__n, 0); }

    void reserve(size_t __res_arg = 0)
    {
        size_t __n = (max)(__res_arg, size()) + 1;
        pointer __new_start = new T[__n];
        pointer __new_finish = __new_start;

        __new_finish = copy(this->_M_start, this->_M_finish, __new_start);
        *__new_finish = 0;// _M_construct_null(__new_finish);

        //_Destroy(this->_M_start, this->_M_finish + 1);
        delete[] this->_M_start;
        this->_M_start = __new_start;
        this->_M_finish = __new_finish;
        this->_M_end_of_storage = __new_start + __n;
    }

    size_t capacity() const { return (this->_M_end_of_storage - this->_M_start) - 1; }

    void clear() {
        if (!empty()) {
            *(this->_M_start) = 0;
            //_Destroy(this->_M_start+1, this->_M_finish+1);
            this->_M_finish = this->_M_start;
        }
    }

    bool empty() const { return this->_M_start == this->_M_finish; }

public:                         // Element access.

    const_reference operator[](size_t __n) const
    {
        ASSERT(__n < size());
        return *(this->_M_start + __n);
    }
    reference operator[](size_t __n)
    {
        ASSERT(__n < size());
        return *(this->_M_start + __n);
    }

public:                         // Append, operator+=, push_back.

    _Self &operator+=(const _Self &__s) { return append(__s); }
    _Self &operator+=(const T *__s) { return append(__s); }
    _Self &operator+=(T __c) { push_back(__c); return *this; }

    _Self &append(const _Self &__s)
    {
        return append(__s._M_start, __s._M_finish);
    }

    _Self &append(const _Self &__s,
        size_t __pos, size_t __n)
    {
        ASSERT(__pos <= __s.size());
        return append(__s._M_start + __pos,
            __s._M_start + __pos + (min)(__n, __s.size() - __pos));
    }

    _Self &append(const T *__s)
    {
        return append(__s, __s + __strlen(__s));
    }
    _Self &append(size_t __n, T __c)
    {
        if (size() + __n > capacity())
            reserve(size() + (max)(size(), __n));
        if (__n > 0) {
            fill_n(this->_M_finish + 1, __n - 1, __c);
            *(this->_M_finish + __n) = 0;//_M_construct_null(this->_M_finish + __n);
            *end() = __c;
            this->_M_finish += __n;
        }
        return *this;
    }

    // Check to see if _InputIterator is an integer type.  If so, then
    // it can't be an iterator.
    template <class _InputIter>
    _Self &append(_InputIter __first, _InputIter __last) {
        for (; __first != __last; ++__first)
            push_back(*__first);
        return *this;
    }

    void push_back(T __c) {
        if (this->_M_finish + 1 == this->_M_end_of_storage)
            reserve(size() + (max)(size(), size_t(1)));
        *(this->_M_finish + 1) = 0;
        *(this->_M_finish) = __c;
        ++this->_M_finish;
    }

    void pop_back() {
        *(this->_M_finish - 1) = 0;
        --this->_M_finish;
    }

public:                         // Assign

    _Self &assign(const _Self &__s) { return assign(__s._M_start, __s._M_finish); }

    _Self &assign(const T *__s, size_t __n) { return assign(__s, __s + __n); }

    _Self &assign(const T *__s) { return assign(__s, __s + __strlen(__s)); }



public:
    // Check to see if _InputIterator is an integer type.  If so, then
    // it can't be an iterator.
    template <class _InputIter> _Self &assign(_InputIter __f, _InputIter __l) {
        pointer __cur = this->_M_start;
        while (__f != __l && __cur != this->_M_finish) {
            *__cur = *__f;
            ++__f;
            ++__cur;
        }
        if (__f == __l)
            erase(__cur, end());
        else
            append(__f, __l);
        return *this;
    }

    // if member templates are on, this works as specialization 
    _Self &assign(const T *__f, const T *__l)
    {
        ptrdiff_t __n = __l - __f;
        if (size_t(__n) <= size()) {
            memcpy(this->_M_start, __f, sizeof(T) * __n);
            //_Traits::copy(this->_M_start, __f, __n);
            erase(begin() + __n, end());
        } else {
            memcpy(this->_M_start, __f, sizeof(T) * size());
            //_Traits::copy(this->_M_start, __f, size());
            append(__f + size(), __l);
        }
        return *this;
    }

public:                         // Insert

    _Self &insert(size_t __pos, const _Self &__s) {
        ASSERT(__pos <= size());
        insert(begin() + __pos, __s._M_start, __s._M_finish);
        return *this;
    }

    _Self &insert(size_t __pos, const _Self &__s,
        size_t __beg, size_t __n) {
        ASSERT(__pos <= size() && __beg <= __s.size());
        size_t __len = (min)(__n, __s.size() - __beg);
        insert(begin() + __pos,
            __s._M_start + __beg, __s._M_start + __beg + __len);
        return *this;
    }

    _Self &insert(size_t __pos, const T *__s) {
        ASSERT(__pos <= size());
        size_t __len = __strlen(__s);
        insert(this->_M_start + __pos, __s, __s + __len);
        return *this;
    }

    iterator insert(iterator __p, T __c) {
        if (__p == end()) {
            push_back(__c);
            return this->_M_finish - 1;
        } else
            return _M_insert_aux(__p, __c);
    }

    // Check to see if _InputIterator is an integer type.  If so, then
    // it can't be an iterator.
    template <class _InputIter> void insert(iterator __p, _InputIter __first, _InputIter __last) {
        for (; __first != __last; ++__first) {
            __p = insert(__p, *__first);
            ++__p;
        }
    }


private:                        // Helper functions for insert.
    pointer _M_insert_aux(pointer __p, T __c)
    {
        pointer __new_pos = __p;
        if (this->_M_finish + 1 < this->_M_end_of_storage)
        {
            this->_M_finish[1] = 0;//_M_construct_null(this->_M_finish + 1);
            memmove(__p + 1, __p, sizeof(T) * (this->_M_finish - __p));
            //_Traits::move(__p + 1, __p, this->_M_finish - __p);
            *__p = __c;//_Traits::assign(*__p, __c);
            ++this->_M_finish;
        } else
        {
            const size_t __old_len = size();
            const size_t __len = __old_len + (max)(__old_len, size_t(1)) + 1;
            pointer __new_start = new T[__len];
            pointer __new_finish = __new_start;
            __new_pos = copy(this->_M_start, __p, __new_start);
            *__new_pos = __c;
            __new_finish = __new_pos + 1;
            __new_finish = copy(__p, this->_M_finish, __new_finish);
            *__new_finish = 0;//_M_construct_null(__new_finish);
            //_Destroy(this->_M_start, this->_M_finish + 1);
            delete[] this->_M_start;
            this->_M_start = __new_start;
            this->_M_finish = __new_finish;
            this->_M_end_of_storage = __new_start + __len;
        }
        return __new_pos;
    }


public:                         // Erase.

    _Self &erase(size_t __pos = 0, size_t __n = npos) {
        ASSERT(__pos <= size());
        erase(begin() + __pos, begin() + __pos + (min)(__n, size() - __pos));
        return *this;
    }

    iterator erase(iterator __position) {
        // The move includes the terminating T().
        memmove(__position, __position + 1, sizeof(T) * (this->_M_finish - __position));
        --this->_M_finish;
        return __position;
    }

    iterator erase(iterator __first, iterator __last) {
        if (__first != __last) {
            // The move includes the terminating T().
            memmove(__first, __last, sizeof(T) * ((this->_M_finish - __last) + 1));
            pointer __new_finish = this->_M_finish - (__last - __first);
            this->_M_finish = __new_finish;
        }
        return __first;
    }

public:                         // Replace.  (Conceptually equivalent
    // to erase followed by insert.)
    _Self &replace(size_t __pos, size_t __n, const _Self &__s) {
        ASSERT(__pos <= size());
        const size_t __len = (min)(__n, size() - __pos);
        return replace(begin() + __pos, begin() + __pos + __len,
            __s._M_start, __s._M_finish);
    }

    _Self &replace(size_t __pos1, size_t __n1,
        const _Self &__s,
        size_t __pos2, size_t __n2) {
        ASSERT(__pos1 <= size() && __pos2 <= __s.size());
        const size_t __len1 = (min)(__n1, size() - __pos1);
        const size_t __len2 = (min)(__n2, __s.size() - __pos2);
        return replace(begin() + __pos1, begin() + __pos1 + __len1,
            __s._M_start + __pos2, __s._M_start + __pos2 + __len2);
    }

    _Self &replace(size_t __pos, size_t __n1,
        const T *__s, size_t __n2) {
        ASSERT(__pos <= size());
        const size_t __len = (min)(__n1, size() - __pos);
        return replace(begin() + __pos, begin() + __pos + __len,
            __s, __s + __n2);
    }

    _Self &replace(size_t __pos, size_t __n1,
        const T *__s) {
        ASSERT(__pos <= size());
        const size_t __len = (min)(__n1, size() - __pos);
        const size_t __n2 = __strlen(__s);
        return replace(begin() + __pos, begin() + __pos + __len,
            __s, __s + __strlen(__s));
    }

    _Self &replace(size_t __pos, size_t __n1,
        size_t __n2, T __c) {
        ASSERT(__pos <= size());
        const size_t __len = (min)(__n1, size() - __pos);
        return replace(begin() + __pos, begin() + __pos + __len, __n2, __c);
    }

    _Self &replace(iterator __first, iterator __last,
        const _Self &__s)
    {
        return replace(__first, __last, __s._M_start, __s._M_finish);
    }

    _Self &replace(iterator __first, iterator __last,
        const T *__s, size_t __n)
    {
        return replace(__first, __last, __s, __s + __n);
    }

    _Self &replace(iterator __first, iterator __last,
        const T *__s) {
        return replace(__first, __last, __s, __s + __strlen(__s));
    }

    _Self &replace(iterator __first, iterator __last,
        size_t __n, T __c);

    // Check to see if _InputIterator is an integer type.  If so, then
    // it can't be an iterator.
    template <class _InputIter> _Self &replace(iterator __first, iterator __last,
        _InputIter __f, _InputIter __l) {
        for (; __first != __last && __f != __l; ++__first, ++__f)
            *__first = *__f;

        if (__f == __l)
            erase(__first, __last);
        else
            insert(__last, __f, __l);
        return *this;
    }

private:                        // Helper functions for replace.


public:                         // Other modifier member functions.

    void swap(_Self &__s) {
        nstl::swap(this->_M_start, __s._M_start);
        nstl::swap(this->_M_finish, __s._M_finish);
        nstl::swap(this->_M_end_of_storage, __s._M_end_of_storage);
    }

public:                         // Conversion to C string.

    const T *c_str() const { return this->_M_start; }
    const T *data()  const { return this->_M_start; }

public:                         // find.

    size_t find(const _Self &__s, size_t __pos = 0) const
    {
        return find(__s._M_start, __pos, __s.size());
    }

    size_t find(const T *__s, size_t __pos = 0) const
    {
        return find(__s, __pos, __strlen(__s));
    }

    size_t find(const T *__s, size_t __pos, size_t __n) const;
    size_t find(T __c, size_t __pos = 0) const;

public:                         // rfind.
    size_t rfind(T __c, size_t __pos = npos) const;
public:                         // find_first_of

    size_t find_first_of(const _Self &__s, size_t __pos = 0) const
    {
        return find_first_of(__s._M_start, __pos, __s.size());
    }

    size_t find_first_of(const T *__s, size_t __pos = 0) const
    {
        return find_first_of(__s, __pos, __strlen(__s));
    }

    size_t find_first_of(const T *__s, size_t __pos,
        size_t __n) const;

    size_t find_first_of(T __c, size_t __pos = 0) const
    {
        return find(__c, __pos);
    }

public:                         // find_last_of

public:                         // find_first_not_of

    size_t find_first_not_of(const _Self &__s,
        size_t __pos = 0) const
    {
        return find_first_not_of(__s._M_start, __pos, __s.size());
    }

    size_t find_first_not_of(const T *__s, size_t __pos = 0) const
    {
        return find_first_not_of(__s, __pos, __strlen(__s));
    }

    size_t find_first_not_of(const T *__s, size_t __pos,
        size_t __n) const;

    size_t find_first_not_of(T __c, size_t __pos = 0) const;

public:                         // find_last_not_of

    size_t find_last_not_of(const _Self &__s,
        size_t __pos = npos) const
    {
        return find_last_not_of(__s._M_start, __pos, __s.size());
    }

    size_t find_last_not_of(const T *__s, size_t __pos = npos) const
    {
        return find_last_not_of(__s, __pos, __strlen(__s));
    }

    size_t find_last_not_of(const T *__s, size_t __pos,
        size_t __n) const;

    size_t find_last_not_of(T __c, size_t __pos = npos) const;

public:                         // Substring.

    _Self substr(size_t __pos = 0, size_t __n = npos) const {
        ASSERT(__pos <= size());
        return _Self(this->_M_start + __pos,
            this->_M_start + __pos + (min)(__n, size() - __pos));
    }

public:                         // Compare

    yint compare(const _Self &__s) const
    {
        return _M_compare(this->_M_start, this->_M_finish, __s._M_start, __s._M_finish);
    }

    yint compare(size_t __pos1, size_t __n1,
        const _Self &__s) const {
        ASSERT(__pos1 <= size());
        return _M_compare(this->_M_start + __pos1,
            this->_M_start + __pos1 + (min)(__n1, size() - __pos1),
            __s._M_start, __s._M_finish);
    }

    yint compare(size_t __pos1, size_t __n1,
        const _Self &__s,
        size_t __pos2, size_t __n2) const {
        ASSERT(__pos1 <= size() && __pos2 <= __s.size());
        return _M_compare(this->_M_start + __pos1,
            this->_M_start + __pos1 + (min)(__n1, size() - __pos1),
            __s._M_start + __pos2,
            __s._M_start + __pos2 + (min)(__n2, __s.size() - __pos2));
    }

    yint compare(const T *__s) const {
        return _M_compare(this->_M_start, this->_M_finish, __s, __s + __strlen(__s));
    }

    yint compare(size_t __pos1, size_t __n1, const T *__s) const {
        ASSERT(__pos1 <= size());
        return _M_compare(this->_M_start + __pos1,
            this->_M_start + __pos1 + (min)(__n1, size() - __pos1),
            __s, __s + __strlen(__s));
    }

    yint compare(size_t __pos1, size_t __n1, const T *__s,
        size_t __n2) const {
        ASSERT(__pos1 <= size());
        return _M_compare(this->_M_start + __pos1,
            this->_M_start + __pos1 + (min)(__n1, size() - __pos1),
            __s, __s + __n2);
    }

public:                        // Helper functions for compare.

    static yint  _M_compare(const T *__f1, const T *__l1,
        const T *__f2, const T *__l2) {
        const ptrdiff_t __n1 = __l1 - __f1;
        const ptrdiff_t __n2 = __l2 - __f2;
        const yint cmp = __strncmp(__f1, __f2, (min)(__n1, __n2));
        return cmp != 0 ? cmp : (__n1 < __n2 ? -1 : (__n1 > __n2 ? 1 : 0));
    }
};


// ------------------------------------------------------------
// Non-member functions.

template <class T> inline basic_string<T>
operator+(const basic_string<T> &__s,
    const basic_string<T> &__y)
{
    typedef basic_string<T> _Str;
    //typedef typename _Str::_Reserve_t _Reserve_t;
    _Str __result(_Reserve_t(), __s.size() + __y.size());
    __result.append(__s);
    __result.append(__y);
    return __result;
}

template <class T> inline basic_string<T>
operator+(const T *__s,
    const basic_string<T> &__y) {
    typedef basic_string<T> _Str;
    //typedef typename _Str::_Reserve_t _Reserve_t;
    const size_t __n = __strlen(__s);
    _Str __result(_Reserve_t(), __n + __y.size());
    __result.append(__s, __s + __n);
    __result.append(__y);
    return __result;
}

template <class T> inline basic_string<T>
operator+(T __c,
    const basic_string<T> &__y) {
    typedef basic_string<T> _Str;
    //typedef typename _Str::_Reserve_t _Reserve_t;
    _Str __result(_Reserve_t(), 1 + __y.size());
    __result.push_back(__c);
    __result.append(__y);
    return __result;
}

template <class T> inline basic_string<T>
operator+(const basic_string<T> &__x,
    const T *__s) {
    typedef basic_string<T> _Str;
    //typedef typename _Str::_Reserve_t _Reserve_t;
    const size_t __n = __strlen(__s);
    _Str __result(_Reserve_t(), __x.size() + __n);
    __result.append(__x);
    __result.append(__s, __s + __n);
    return __result;
}

template <class T> inline basic_string<T>
operator+(const basic_string<T> &__x,
    const T __c) {
    typedef basic_string<T> _Str;
    //typedef typename _Str::_Reserve_t _Reserve_t;
    _Str __result(_Reserve_t(), __x.size() + 1);
    __result.append(__x);
    __result.push_back(__c);
    return __result;
}


// Operator== and operator!=

template <class T> inline bool
    operator==(const basic_string<T> &__x,
        const basic_string<T> &__y) {
    return __x.size() == __y.size() && __strncmp(__x.data(), __y.data(), __x.size()) == 0;
}

template <class T> inline bool
    operator==(const T *__s,
        const basic_string<T> &__y) {
    size_t __n = __strlen(__s);
    return __n == __y.size() && __strncmp(__s, __y.data(), __n) == 0;
}

template <class T> inline bool
    operator==(const basic_string<T> &__x,
        const T *__s) {
    size_t __n = __strlen(__s);
    return __x.size() == __n && __strncmp(__x.data(), __s, __n) == 0;
}

// Operator< (and also >, <=, and >=).

template <class T> inline bool
    operator<(const basic_string<T> &__x,
        const basic_string<T> &__y) {
    return basic_string<T> ::_M_compare(__x.begin(), __x.end(),
        __y.begin(), __y.end()) < 0;
}

template <class T> inline bool
    operator<(const T *__s,
        const basic_string<T> &__y) {
    size_t __n = __strlen(__s);
    return basic_string<T> ::_M_compare(__s, __s + __n, __y.begin(), __y.end()) < 0;
}

template <class T> inline bool
    operator<(const basic_string<T> &__x,
        const T *__s) {
    size_t __n = __strlen(__s);
    return basic_string<T> ::_M_compare(__x.begin(), __x.end(), __s, __s + __n) < 0;
}

template <class T> inline bool
    operator!=(const basic_string<T> &__x,
        const basic_string<T> &__y) {
    return !(__x == __y);
}

template <class T> inline bool
    operator>(const basic_string<T> &__x,
        const basic_string<T> &__y) {
    return __y < __x;
}

template <class T> inline bool
    operator<=(const basic_string<T> &__x,
        const basic_string<T> &__y) {
    return !(__y < __x);
}

template <class T> inline bool
    operator>=(const basic_string<T> &__x,
        const basic_string<T> &__y) {
    return !(__x < __y);
}

template <class T> inline bool
    operator!=(const T *__s,
        const basic_string<T> &__y) {
    return !(__s == __y);
}

template <class T> inline bool
    operator!=(const basic_string<T> &__x,
        const T *__s) {
    return !(__x == __s);
}

template <class T> inline bool
    operator>(const T *__s,
        const basic_string<T> &__y) {
    return __y < __s;
}

template <class T> inline bool
    operator>(const basic_string<T> &__x,
        const T *__s) {
    return __s < __x;
}

template <class T> inline bool
    operator<=(const T *__s,
        const basic_string<T> &__y) {
    return !(__y < __s);
}

template <class T> inline bool
    operator<=(const basic_string<T> &__x,
        const T *__s) {
    return !(__s < __x);
}

template <class T> inline bool
    operator>=(const T *__s,
        const basic_string<T> &__y) {
    return !(__s < __y);
}

template <class T> inline bool
    operator>=(const basic_string<T> &__x,
        const T *__s) {
    return !(__x < __s);
}


template <class T> inline void
    swap(basic_string<T> &__x,
        basic_string<T> &__y) {
    __x.swap(__y);
}

///template <class T> void   _S_string_copy(const basic_string<T>& __s, T* __buf, size_t __n);

# undef basic_string

// ------------------------------------------------------------
// Non-inline declarations.


// Change the string's capacity so that it is large enough to hold
//  at least __res_arg elements, plus the terminating T().  Note that,
//  if __res_arg < capacity(), this member function may actually decrease
//  the string's capacity.

template <class T> basic_string<T> &basic_string<T> ::replace(iterator __first, iterator __last, size_t __n, T __c)
{
    size_t __len = (size_t)(__last - __first);

    if (__len >= __n) {
        fill_n(__first, __n, __c);//_Traits::assign(__first, __n, __c);
        erase(__first + __n, __last);
    } else {
        fill_n(__first, __len, __c);//_Traits::assign(__first, __len, __c);
        insert(__last, __n - __len, __c);
    }
    return *this;
}

struct Eq
{
    template<class T>
    bool operator()(const T &a, const T &b) const { return a == b; }
};

template <class T> size_t
    basic_string<T> ::find(const T *__s, size_t __pos, size_t __n) const
{
    if (__pos + __n > size())
        return npos;
    else {
        const const_pointer __result =
            nstl::search((const T *)this->_M_start + __pos, (const T *)this->_M_finish,
                __s, __s + __n, Eq());
        return __result != this->_M_finish ? __result - this->_M_start : npos;
    }
}

template <class T> size_t
    basic_string<T> ::find(T __c, size_t __pos) const
{
    if (__pos >= size())
        return npos;
    else {
        const const_pointer __result =
            nstl::find((const T *)this->_M_start + __pos, (const T *)this->_M_finish, __c);
        return __result != this->_M_finish ? __result - this->_M_start : npos;
    }
}

template <class T>
size_t basic_string<T> ::rfind(T __c, size_t __pos) const
{
    const size_t __len = size();

    if (__len < 1)
        return npos;
    else
    {
        const const_iterator __last = begin() + (min)(yint(__len) - 1, yint(__pos)) + 1;
        const_iterator __end = end() - 1;
        while (__end >= __last)
        {
            if (*__end == __c)
                return __end - begin();
            --__end;
        }
        return npos;
    }
}

template <class T> size_t
    basic_string<T> ::find_first_of(const T *__s, size_t __pos, size_t __n) const
{
    if (__pos >= size())
        return npos;
    else {
        const_iterator __result = __find_first_of(begin() + __pos, end(),
            __s, __s + __n);
        return __result != end() ? __result - begin() : npos;
    }
}


template<class T>
struct _Not_within_traits
{
    const T *f, *l;
    _Not_within_traits(const T *_f, const T *_l) : f(_f), l(_l) {}
    bool operator()(T c) { return find(f, l, c) == l; }
};

template <class T> size_t
    basic_string<T> ::find_first_not_of(const T *__s, size_t __pos, size_t __n) const
{
    if (__pos > size())
        return size_t(npos);
    else {
        const_pointer __result = nstl::find_if((const T *)this->_M_start + __pos,
            (const T *)this->_M_finish,
            _Not_within_traits<T>((const T *)__s, (const T *)__s + __n));
        return __result != this->_M_finish ? __result - this->_M_start : npos;
    }
}

template<class T>
struct _Neq_char_bound
{
    T c;
    _Neq_char_bound(T _c) : c(_c) {}
    bool operator()(const T a) { return a != c; }
};

template <class T> size_t
    basic_string<T> ::find_first_not_of(T __c, size_t __pos) const
{
    if (__pos > size())
        return size_t(npos);
    else {
        const_pointer __result = nstl::find_if((const T *)this->_M_start + __pos, (const T *)this->_M_finish,
            _Neq_char_bound<T>(__c));
        return __result != this->_M_finish ? __result - this->_M_start : npos;
    }
}




template <class T> size_t
    basic_string<T> ::find_last_not_of(const T *__s, size_t __pos, size_t __n) const
{
    const size_t __len = size();

    if (__len < 1)
        return npos;
    else {
        const_iterator __last = begin() + (min)(__len - 1, __pos) + 1;
        _Not_within_traits<T> test((const T *)__s, (const T *)__s + __n);
        while (__last != begin())
        {
            --__last;
            if (test(*__last))
                return __last - begin();
        }
        return npos;
    }
}

template <class T> size_t
    basic_string<T> ::find_last_not_of(T __c, size_t __pos) const
{
    const size_t __len = size();

    if (__len < 1)
        return npos;
    else {
        const_iterator __last = begin() + (min)(__len - 1, __pos) + 1;
        while (__last != begin())
        {
            --__last;
            if (*__last != __c)
                return __last - begin();
        }
        return npos;
    }
}



typedef basic_string<char> string;

typedef basic_string<wchar_t> wstring;



template<> struct hash<nstl::string>
{
    size_t operator()(const nstl::string &s) const { return nstl::__stl_hash_string(s.c_str()); }
};

template<> struct hash<nstl::wstring>
{
    size_t operator()(const nstl::wstring &s) const
    {
        size_t r = 0;
        for (size_t k = 0; k < s.length(); ++k)
            r = 0xbadf00d * r + ((size_t)s[k]);
        return r;
    }
};
}
