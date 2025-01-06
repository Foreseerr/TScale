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

#include "npair.h"


namespace nstl
{
// swap and iter_swap
template <class _Tp>
inline void swap(_Tp &__a, _Tp &__b) {
    _Tp __tmp = __a;
    __a = __b;
    __b = __tmp;
}


struct less {
    template<class T>
    inline bool operator()(const T a, const T b) { return a < b; }
};
struct equal_to {
    template<class T>
    inline bool operator()(const T a, const T b) { return a == b; }
};


template<class T>
yint inline distance(T a, T b)
{
    yint nRes = 0;
    for (; a != b; ++a)
        ++nRes;
    return nRes;
}
//--------------------------------------------------
// min and max

template <class _Tp>
inline const _Tp &(min)(const _Tp &__a, const _Tp &__b) { return __b < __a ? __b : __a; }
template <class _Tp>
inline const _Tp &(max)(const _Tp &__a, const _Tp &__b) { return  __a < __b ? __b : __a; }

template <class _Tp, class _Compare>
inline const _Tp &(min)(const _Tp &__a, const _Tp &__b, _Compare __comp) {
    return __comp(__b, __a) ? __b : __a;
}

template <class _Tp, class _Compare>
inline const _Tp &(max)(const _Tp &__a, const _Tp &__b, _Compare __comp) {
    return __comp(__a, __b) ? __b : __a;
}

//--------------------------------------------------
// copy

// All of these auxiliary functions serve two purposes.  (1) Replace
// calls to copy with memmove whenever possible.  (Memmove, not memcpy,
// because the input and output ranges are permitted to overlap.)
// (2) If we're using random access iterators, then write the loop as
// a for loop with an explicit count.

template <class _InputIter, class _OutputIter>
inline _OutputIter copy(_InputIter __first, _InputIter __last,
    _OutputIter __result) {
    for (; __first != __last; ++__result, ++__first)
        *__result = *__first;
    return __result;
}

//--------------------------------------------------
// copy_backward auxiliary functions

template <class _BidirectionalIter1, class _BidirectionalIter2>
inline _BidirectionalIter2 copy_backward(_BidirectionalIter1 __first,
    _BidirectionalIter1 __last,
    _BidirectionalIter2 __result)
{
    while (__first != __last)
        *--__result = *--__last;
    return __result;
}

//--------------------------------------------------
// fill and fill_n


template <class _ForwardIter, class _Tp>
inline
    void fill(_ForwardIter __first, _ForwardIter __last, const _Tp &__val) {
    for (; __first != __last; ++__first)
        *__first = __val;
}

template <class _OutputIter, class _Size, class _Tp>
inline
    _OutputIter fill_n(_OutputIter __first, _Size __n, const _Tp &__val) {
    for (; __n > 0; --__n, ++__first)
        *__first = __val;
    return __first;
}


// Specialization: for one-byte types we can use memset.

inline void fill(unsigned char *__first, unsigned char *__last,
    const unsigned char &__val) {
    unsigned char __tmp = __val;
    memset(__first, __tmp, __last - __first);
}

inline void fill(char *__first, char *__last, const char &__val) {
    char __tmp = __val;
    memset(__first, static_cast<unsigned char>(__tmp), __last - __first);
}

#ifdef _NSTL_FUNCTION_TMPL_PARTIAL_ORDER

template <class _Size>
inline unsigned char *fill_n(unsigned char *__first, _Size __n,
    const unsigned char &__val) {
    fill(__first, __first + __n, __val);
    return __first + __n;
}

template <class _Size>
inline signed char *fill_n(char *__first, _Size __n,
    const signed char &__val) {
    fill(__first, __first + __n, __val);
    return __first + __n;
}

template <class _Size>
inline char *fill_n(char *__first, _Size __n, const char &__val) {
    fill(__first, __first + __n, __val);
    return __first + __n;
}

#endif /* _NSTL_FUNCTION_TMPL_PARTIAL_ORDER */


//--------------------------------------------------
// equal and mismatch
template <class _InputIter1, class _InputIter2>
inline
    bool equal(_InputIter1 __first1, _InputIter1 __last1,
        _InputIter2 __first2) {
    for (; __first1 != __last1; ++__first1, ++__first2)
        if (!(*__first1 == *__first2))
            return false;
    return true;
}

template <class _InputIter1, class _InputIter2, class _BinaryPredicate>
inline
    bool equal(_InputIter1 __first1, _InputIter1 __last1,
        _InputIter2 __first2, _BinaryPredicate __binary_pred) {
    for (; __first1 != __last1; ++__first1, ++__first2)
        if (!__binary_pred(*__first1, *__first2))
            return false;
    return true;
}

//--------------------------------------------------
// lexicographical_compare and lexicographical_compare_3way.
// (the latter is not part of the C++ standard.)

template <class _InputIter1, class _InputIter2>
inline
    bool lexicographical_compare(_InputIter1 __first1, _InputIter1 __last1,
        _InputIter2 __first2, _InputIter2 __last2)
{
    for (; __first1 != __last1 && __first2 != __last2; ++__first1, ++__first2)
    {
        if (*__first1 < *__first2)
            return true;
        if (*__first2 < *__first1)
            return false;
    }
    return __first1 == __last1 && __first2 != __last2;
}

template <class _InputIter1, class _InputIter2, class _Compare>
inline
    bool lexicographical_compare(_InputIter1 __first1, _InputIter1 __last1,
        _InputIter2 __first2, _InputIter2 __last2,
        _Compare __comp)
{
    for (; __first1 != __last1 && __first2 != __last2; ++__first1, ++__first2)
    {
        if (__comp(*__first1, *__first2))
            return true;
        if (__comp(*__first2, *__first1))
            return false;
    }
    return __first1 == __last1 && __first2 != __last2;
}

inline bool
    lexicographical_compare(const unsigned char *__first1,
        const unsigned char *__last1,
        const unsigned char *__first2,
        const unsigned char *__last2)
{
    const size_t __len1 = __last1 - __first1;
    const size_t __len2 = __last2 - __first2;

    const yint __result = memcmp(__first1, __first2, (min)(__len1, __len2));
    return __result != 0 ? (__result < 0) : (__len1 < __len2);
}

// count
template <class _InputIter, class _Tp>
inline yint count(_InputIter __first, _InputIter __last, const _Tp &__val)
{
    yint __n = 0;
    for (; __first != __last; ++__first)
        if (*__first == __val)
            ++__n;
    return __n;
}

// find and find_if. Note find may be expressed in terms of find_if if appropriate binder was available.
template <class _InputIter, class _Tp>
inline
    _InputIter find(_InputIter __first, _InputIter __last, const _Tp &__val)
{
    while (__first != __last && !(*__first == __val))
        ++__first;
    return __first;
}
template <class _InputIter, class _Predicate>
inline
    _InputIter find_if(_InputIter __first, _InputIter __last, _Predicate __pred)
{
    while (__first != __last && !__pred(*__first))
        ++__first;
    return __first;
}

// search.
template <class _ForwardIter1, class _ForwardIter2, class _BinaryPred>
inline
    _ForwardIter1 search(_ForwardIter1 __first1, _ForwardIter1 __last1,
        _ForwardIter2 __first2, _ForwardIter2 __last2, _BinaryPred  __predicate)
{
    // Test for empty ranges
    if (__first1 == __last1 || __first2 == __last2)
        return __first1;

    // Test for a pattern of length 1.
    _ForwardIter2 __tmp(__first2);
    ++__tmp;
    if (__tmp == __last2) {
        while (__first1 != __last1 && !__predicate(*__first1, *__first2))
            ++__first1;
        return __first1;
    }

    // General case.

    _ForwardIter2 __p1, __p;

    __p1 = __first2; ++__p1;

    //  _ForwardIter1 __current = __first1;

    while (__first1 != __last1) {
        while (__first1 != __last1) {
            if (__predicate(*__first1, *__first2))
                break;
            ++__first1;
        }
        while (__first1 != __last1 && !__predicate(*__first1, *__first2))
            ++__first1;
        if (__first1 == __last1)
            return __last1;

        __p = __p1;
        _ForwardIter1 __current = __first1;
        if (++__current == __last1) return __last1;

        while (__predicate(*__current, *__p)) {
            if (++__p == __last2)
                return __first1;
            if (++__current == __last1)
                return __last1;
        }

        ++__first1;
    }
    return __first1;
}

// find_first_of
template <class _InputIter, class _ForwardIter>
inline
    _InputIter __find_first_of(_InputIter __first1, _InputIter __last1,
        _ForwardIter __first2, _ForwardIter __last2)
{
    for (; __first1 != __last1; ++__first1)
        for (_ForwardIter __iter = __first2; __iter != __last2; ++__iter)
            if (*__first1 == *__iter)
                return __first1;
    return __last1;
}

// replace
template <class _ForwardIter, class _Tp>
inline void
    replace(_ForwardIter __first, _ForwardIter __last,
        const _Tp &__old_value, const _Tp &__new_value) {
    for (; __first != __last; ++__first)
        if (*__first == __old_value)
            *__first = __new_value;
}


template <class _ForwardIter, class _Tp, class _Compare, class _Distance>
_ForwardIter __lower_bound(_ForwardIter __first, _ForwardIter __last,
    const _Tp &__val, _Compare __comp, _Distance *)
{
    _Distance __len = __last - __first;//distance(__first, __last);
    _Distance __half;
    _ForwardIter __middle;

    while (__len > 0) {
        __half = __len >> 1;
        __middle = __first;
        __middle += __half; // advance()
        if (__comp(*__middle, __val)) {
            __first = __middle;
            ++__first;
            __len = __len - __half - 1;
        } else
            __len = __half;
    }
    return __first;
}





// transform
template <class _InputIter, class _OutputIter, class _UnaryOperation>
_OutputIter
    transform(_InputIter __first, _InputIter __last, _OutputIter __result, _UnaryOperation __opr) {
    for (; __first != __last; ++__first, ++__result)
        *__result = __opr(*__first);
    return __result;
}
template <class _InputIter1, class _InputIter2, class _OutputIter, class _BinaryOperation>
_OutputIter
    transform(_InputIter1 __first1, _InputIter1 __last1,
        _InputIter2 __first2, _OutputIter __result, _BinaryOperation __binary_op) {
    for (; __first1 != __last1; ++__first1, ++__first2, ++__result)
        *__result = __binary_op(*__first1, *__first2);
    return __result;
}
// remove, remove_if, remove_copy, remove_copy_if

template <class _InputIter, class _OutputIter, class _Tp>
_OutputIter
    remove_copy(_InputIter __first, _InputIter __last, _OutputIter __result, const _Tp &__val) {
    for (; __first != __last; ++__first)
        if (!(*__first == __val)) {
            *__result = *__first;
            ++__result;
        }
    return __result;
}

template <class _InputIter, class _OutputIter, class _Predicate>
_OutputIter
    remove_copy_if(_InputIter __first, _InputIter __last, _OutputIter __result, _Predicate __pred) {
    for (; __first != __last; ++__first)
        if (!__pred(*__first)) {
            *__result = *__first;
            ++__result;
        }
    return __result;
}

template <class _ForwardIter, class _Tp>
inline _ForwardIter
    remove(_ForwardIter __first, _ForwardIter __last, const _Tp &__val) {
    __first = find(__first, __last, __val);
    if (__first == __last)
        return __first;
    else {
        _ForwardIter __next = __first;
        return remove_copy(++__next, __last, __first, __val);
    }
}

template <class _ForwardIter, class _Predicate>
_ForwardIter
    remove_if(_ForwardIter __first, _ForwardIter __last, _Predicate __pred) {
    __first = find_if(__first, __last, __pred);
    if (__first == __last)
        return __first;
    else {
        _ForwardIter __next = __first;
        return remove_copy_if(++__next, __last, __first, __pred);
    }
}
template <class _BidirectionalIter>
inline void
    reverse(_BidirectionalIter __first, _BidirectionalIter __last) {
    while (true)
        if (__first == __last || __first == --__last)
            return;
        else
            swap(*__first++, *__last);
}

# define  __stl_threshold  16

template<class T, class TComp>
inline void JoinSortStage(yint step, const T *src, T *pDst, yint nCount, TComp compare)
{
    yint i, n1, n2, step2, dest, numSteps;
    step2 = step * 2;
    if (step * 2 > nCount)
    {
        numSteps = nCount - step;
        for (yint k = 0; k < step; ++k)
            pDst[nCount - step + k] = src[nCount - step + k];
        //		memcpy( pDst + nCount - step, pSrc + nCount - step, step * sizeof(pSrc[0]) );
    } else {
        numSteps = step;
    }
    numSteps = min(step, nCount - step);
    for (i = 0; i < numSteps; i++)
    {
        n1 = i;
        n2 = i + step;
        dest = i;
        while (1)
        {
            if (!compare(*src[n1], *src[n2]))
            {
                pDst[dest] = src[n2];
                n2 += step2;
                dest += step;
                if (n2 >= nCount) {
                    while (n1 < nCount)
                    {
                        pDst[dest] = src[n1];
                        n1 += step2;
                        dest += step;
                    }
                    break;
                }
            } else {
                pDst[dest] = src[n1];
                n1 += step2;
                dest += step;
                if (n1 >= nCount) {
                    while (n2 < nCount)
                    {
                        pDst[dest] = src[n2];
                        n2 += step2;
                        dest += step;
                    }
                    break;
                }
            }
        }
    }
}

template <class _ForwardIterator>
void _Destroy(_ForwardIterator __first, _ForwardIterator __last);

template<class T, class T1>
inline void SortReorder(T pRes, T *sorted, yint nSize, T1 a)
{
    T1 *pData = (T1 *)new char[sizeof(T1) * nSize];
    for (yint k = 0; k < nSize; ++k)
        new(&pData[k]) T1(*sorted[k]);
    for (yint k = 0; k < nSize; ++k)
        *pRes++ = pData[k];
    _Destroy(pData, pData + nSize);
    delete[]((char *)pData);
}

template <class T, class _Compare>
void sort(T __first, T __last, _Compare __comp)
{
    if (__first == __last)
        return;
    yint nElems = distance(__first, __last);
    if (nElems <= 1)
        return;
    T *data = new T[nElems], *tmpData = new T[nElems], **pSort, **pSortDst;
    {
        T *pDest = data;
        for (T p = __first; p != __last; ++p)
            *pDest++ = p;
    }

    pSort = &data;
    pSortDst = &tmpData;
    yint nStep = 0x40000000;
    while (nStep >= nElems) nStep >>= 1; // determine initial step for join sort

    while (nStep > 0)
    {
        JoinSortStage(nStep, *pSort, *pSortDst, nElems, __comp);
        nStep >>= 1;
        swap(pSort, pSortDst);
    }
    SortReorder(__first, *pSort, nElems, *__first);
    delete[] data;
    delete[] tmpData;
}

template <class _RandomAccessIter>
void sort(_RandomAccessIter __first, _RandomAccessIter __last)
{
    sort(__first, __last, less());
}

// Binary search (lower_bound, upper_bound, equal_range, binary_search).

template <class _ForwardIter, class _Tp>
inline _ForwardIter lower_bound(_ForwardIter __first, _ForwardIter __last,
    const _Tp &__val) {
    return __lower_bound(__first, __last, __val, less(), (yint *)0);
}

template <class _ForwardIter, class _Tp, class _Compare>
inline _ForwardIter lower_bound(_ForwardIter __first, _ForwardIter __last,
    const _Tp &__val, _Compare __comp) {
    return __lower_bound(__first, __last, __val, __comp, (yint *)0);
}
}
