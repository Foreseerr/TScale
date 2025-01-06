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

#include "nalgobase.h"
#include <new>

namespace nstl
{
template <class _Tp>
inline void _Destroy(_Tp *__pointer) {
    //  __pointer;
    __pointer->~_Tp();
}


template <class _T1, class _T2>
inline void _Construct(_T1 *__p, const _T2 &__val) {
    new(__p) _T1(__val);
}


template <class _T1>
inline void _Construct(_T1 *__p) {
    new(__p) _T1;
}


template <class _ForwardIterator>
inline void _Destroy(_ForwardIterator __first, _ForwardIterator __last) {
    for (; __first != __last; ++__first)
        _Destroy(&*__first);
}


template <class _T1, class _T2>
inline void construct(_T1 *__p, const _T2 &__val) { _Construct(__p, __val); }
template <class _T1>
inline void construct(_T1 *__p) { _Construct(__p); }


// uninitialized_copy

// Valid if copy construction is equivalent to assignment, and if the
//  destructor is trivial.

template <class _InputIter, class _ForwardIter>
inline _ForwardIter uninitialized_copy(_InputIter __first, _InputIter __last, _ForwardIter __result)
{
    _ForwardIter __cur = __result;
    for (; __first != __last; ++__first, ++__cur)
        _Construct(&*__cur, *__first);
    return __cur;
}

// Valid if copy construction is equivalent to assignment, and if the
// destructor is trivial.
template <class _ForwardIter, class _Tp>
inline void uninitialized_fill(_ForwardIter __first, _ForwardIter __last, const _Tp &__x)
{
    _ForwardIter __cur = __first;
    for (; __cur != __last; ++__cur)
        _Construct(&*__cur, __x);
}

// Valid if copy construction is equivalent to assignment, and if the
//  destructor is trivial.
template <class _ForwardIter, class _Size, class _Tp>
inline _ForwardIter
    uninitialized_fill_n(_ForwardIter __first, _Size __n, const _Tp &__x) {
    _ForwardIter __cur = __first;
    for (; __n > 0; --__n, ++__cur)
        _Construct(&*__cur, __x);
    return __cur;
}

template <class _ForwardIter, class _Size>
inline _ForwardIter
    uninitialized_fill_n(_ForwardIter __first, _Size __n) {
    _ForwardIter __cur = __first;
    for (; __n > 0; --__n, ++__cur)
        _Construct(&*__cur);
    return __cur;
}
}
