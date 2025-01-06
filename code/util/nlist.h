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
  //#include <functional>

namespace nstl
{

struct _List_node_base {
    _List_node_base *_M_next;
    _List_node_base *_M_prev;
};

template <class _Dummy>
class _List_global {
public:
    typedef _List_node_base _Node;
    static void  _Transfer(_List_node_base *__position,
        _List_node_base *__first, _List_node_base *__last);
};

typedef _List_global<bool> _List_global_inst;

template <class _Tp>
struct _List_node : public _List_node_base {
    _Tp _M_data;

    _List_node() {}
    _List_node(const _Tp &a) : _M_data(a) {}
    //  __TRIVIAL_STUFF(_List_node)
};



template<class _Tp, class TRet>//, class _Traits>
struct _List_it
{
    _Tp *_M_node;

    _List_it(_Tp *__x) : _M_node(__x) {}
    template<class T1, class TR1>
    _List_it(const _List_it<T1, TR1> &a) : _M_node(a._M_node) {}
    _List_it() {}

    TRet &operator*() const { return _M_node->_M_data; }

    TRet *operator->() const { return &_M_node->_M_data; }

    _List_it &operator++() {
        _M_node = (_Tp *)_M_node->_M_next;
        return *this;
    }
    _List_it operator++(int) {
        _List_it __tmp = *this;
        _M_node = (_Tp *)_M_node->_M_next;
        return __tmp;
    }
    _List_it &operator--() {
        _M_node = (_Tp *)_M_node->_M_prev;
        return *this;
    }
    _List_it operator--(int) {
        _List_it __tmp = *this;
        _M_node = (_Tp *)_M_node->_M_prev;
        return __tmp;
    }
    bool operator==(const _List_it &__y) const { return _M_node == __y._M_node; }
    bool operator!=(const _List_it &__y) const { return _M_node != __y._M_node; }
};


// Base class that encapsulates details of allocators and helps 
// to simplify EH

template <class _Tp>
class _List_base
{
protected:
    typedef _List_node<_Tp> _Node;
public:

    _List_base()
    {
        _Node *__n = (_Node *)new char[sizeof(_Node)];
        __n->_M_next = __n;
        __n->_M_prev = __n;
        _M_node = __n;
    }
    ~_List_base() {
        clear();
        delete[]((char *)_M_node);
    }

    void clear()
    {
        _Node *__cur = (_Node *)this->_M_node->_M_next;
        while (__cur != this->_M_node) {
            _Node *__tmp = __cur;
            __cur = (_Node *)__cur->_M_next;
            delete __tmp;
        }
        this->_M_node->_M_next = this->_M_node;
        this->_M_node->_M_prev = this->_M_node;
    }

public:
    _Node *_M_node;
};

template <class _Tp>
class list;

// helper functions to reduce code duplication
template <class _Tp, class _Predicate>
void _S_remove_if(list<_Tp> &__that, _Predicate __pred);

template <class _Tp, class _BinaryPredicate>
void _S_unique(list<_Tp> &__that, _BinaryPredicate __binary_pred);

template <class _Tp, class _StrictWeakOrdering>
void _S_merge(list<_Tp> &__that, list<_Tp> &__x,
    _StrictWeakOrdering __comp);

template <class _Tp, class _StrictWeakOrdering>
void _S_sort(list<_Tp> &__that, _StrictWeakOrdering __comp);

template <class _Tp>
class list : public _List_base<_Tp> {
    typedef _List_base<_Tp> _Base;
    typedef list<_Tp> _Self;
public:
    typedef _Tp value_type;
    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef value_type &reference;
    typedef const value_type &const_reference;
    typedef _List_node<_Tp> _Node;

public:
    typedef _List_it<_Node, _Tp > iterator;
    typedef _List_it<const _Node, const _Tp > const_iterator;

protected:
    _Node *_M_create_node(const _Tp &__x)
    {
        _Node *__p = new _Node(__x);
        return __p;
    }

    _Node *_M_create_node()
    {
        _Node *__p = new _Node();
        return __p;
    }

public:

    explicit list() { LOOK_INIT(*this); }

    iterator begin() { return iterator((_Node *)(this->_M_node->_M_next)); }
    const_iterator begin() const { return const_iterator((_Node *)(this->_M_node->_M_next)); }

    iterator end() { return this->_M_node; }
    const_iterator end() const { return this->_M_node; }

    bool empty() const { return this->_M_node->_M_next == this->_M_node; }
    yint size() const {
        yint __result = distance(begin(), end());
        return __result;
    }
    yint max_size() const { return yint(-1); }

    reference front() { return *begin(); }
    const_reference front() const { return *begin(); }
    reference back() { return *(--end()); }
    const_reference back() const { return *(--end()); }

    void swap(list<_Tp> &__x) {
        nstl::swap(this->_M_node, __x._M_node);
    }

    iterator insert(iterator __position, const _Tp &__x) {

        _Node *__tmp = _M_create_node(__x);
        _List_node_base *__n = __position._M_node;
        _List_node_base *__p = __n->_M_prev;
        __tmp->_M_next = __n;
        __tmp->_M_prev = __p;
        __p->_M_next = __tmp;
        __n->_M_prev = __tmp;
        return __tmp;
    }

    iterator insert(iterator __position) {

        _Node *__tmp = _M_create_node();
        _List_node_base *__n = __position._M_node;
        _List_node_base *__p = __n->_M_prev;
        __tmp->_M_next = __n;
        __tmp->_M_prev = __p;
        __p->_M_next = __tmp;
        __n->_M_prev = __tmp;
        return __tmp;
    }

    template <class _InputIterator>
    void insert(iterator __position, _InputIterator __first, _InputIterator __last) {
        for (; __first != __last; ++__first)
            insert(__position, *__first);
    }
    void insert(iterator __pos, yint __n, const _Tp &__x)
    {
        for (; __n > 0; --__n)
            insert(__pos, __x);
    }

    void push_front(const _Tp &__x) { insert(begin(), __x); }
    void push_back(const _Tp &__x) { insert(end(), __x); }

    void push_front() { insert(begin()); }
    void push_back() { insert(end()); }

    iterator erase(iterator __position) {
        _List_node_base *__next_node = __position._M_node->_M_next;
        _List_node_base *__prev_node = __position._M_node->_M_prev;
        _Node *__n = (_Node *)__position._M_node;
        __prev_node->_M_next = __next_node;
        __next_node->_M_prev = __prev_node;
        delete __n;
        return iterator((_Node *)__next_node);
    }

    iterator erase(iterator __first, iterator __last) {
        while (__first != __last)
            erase(__first++);
        return __last;
    }

    void pop_front() { erase(begin()); }
    void pop_back() {
        iterator __tmp = end();
        erase(--__tmp);
    }
    list(yint __n, const _Tp &__val)
    {
        this->insert(begin(), __n, __val); LOOK_INIT(*this);
    }
    explicit list(yint __n)
    {
        this->insert(begin(), __n, _Tp()); LOOK_INIT(*this);
    }

    // We don't need any dispatching tricks here, because insert does all of
    // that anyway.
    template <class _InputIterator>
    list(_InputIterator __first, _InputIterator __last)
    {
        insert(begin(), __first, __last); LOOK_INIT(*this);
    }

    list(const list<_Tp> &__x)
    {
        insert(begin(), __x.begin(), __x.end()); LOOK_INIT(*this);
    }

    ~list() { }

    list<_Tp> &operator=(const list<_Tp> &__x);

public:
    // assign(), a generalized assignment member function.  Two
    // versions: one that takes a count, and one that takes a range.
    // The range version is a member template, so we dispatch on whether
    // or not the type is an integer.

    void assign(yint __n, const _Tp &__val) { _M_fill_assign(__n, __val); }

    void _M_fill_assign(yint __n, const _Tp &__val);

public:
    void splice(iterator __position, _Self &__x) {
        if (!__x.empty())
            _List_global_inst::_Transfer(__position._M_node, __x.begin()._M_node, __x.end()._M_node);
    }
    void splice(iterator __position, _Self &, iterator __i) {
        iterator __j = __i;
        ++__j;
        if (__position == __i || __position == __j) return;
        _List_global_inst::_Transfer(__position._M_node, __i._M_node, __j._M_node);
    }
    void splice(iterator __position, _Self &, iterator __first, iterator __last) {
        if (__first != __last)
            _List_global_inst::_Transfer(__position._M_node, __first._M_node, __last._M_node);
    }

    void remove(const _Tp &__val) {
        iterator __first = begin();
        iterator __last = end();
        while (__first != __last) {
            iterator __next = __first;
            ++__next;
            if (__val == *__first) erase(__first);
            __first = __next;
        }
    }

    void unique() {
        _S_unique(*this, equal_to());//<_Tp>());
    }

    void merge(_Self &__x) {
        _S_merge(*this, __x, less());//<_Tp>());
    }

    void reverse() {
        _List_node_base *__p = this->_M_node;
        _List_node_base *__tmp = __p;
        do {
            nstl::swap(__tmp->_M_next, __tmp->_M_prev);
            __tmp = __tmp->_M_prev;     // Old next node is now prev.
        } while (__tmp != __p);
    }

    void sort() {
        _S_sort(*this, less());//<_Tp>());
    }

    template <class _Predicate> void remove_if(_Predicate __pred) {
        _S_remove_if(*this, __pred);
    }
    template <class _BinaryPredicate>
    void unique(_BinaryPredicate __binary_pred) {
        _S_unique(*this, __binary_pred);
    }

    template <class _StrictWeakOrdering>
    void merge(list<_Tp> &__x,
        _StrictWeakOrdering __comp) {
        _S_merge(*this, __x, __comp);
    }

    template <class _StrictWeakOrdering>
    void sort(_StrictWeakOrdering __comp) {
        _S_sort(*this, __comp);
    }
};

template <class _Tp>
inline bool
    operator==(const list<_Tp> &__x, const list<_Tp> &__y)
{
    typedef typename list<_Tp>::const_iterator const_iterator;
    const_iterator __end1 = __x.end();
    const_iterator __end2 = __y.end();

    const_iterator __i1 = __x.begin();
    const_iterator __i2 = __y.begin();
    while (__i1 != __end1 && __i2 != __end2 && *__i1 == *__i2) {
        ++__i1;
        ++__i2;
    }
    return __i1 == __end1 && __i2 == __end2;
}

template <class T>
inline bool operator!=(const list<T> &a, const list<T> &b) {
    return !(a == b);
}

template <class _Tp>
inline bool   operator<(const list<_Tp> &__x,
    const list<_Tp> &__y)
{
    return lexicographical_compare(__x.begin(), __x.end(),
        __y.begin(), __y.end());
}



template <class _Dummy>
void
    _List_global<_Dummy>::_Transfer(_List_node_base *__position,
        _List_node_base *__first, _List_node_base *__last) {
    if (__position != __last) {
        // Remove [first, last) from its old position.
        ((_Node *)(__last->_M_prev))->_M_next = __position;
        ((_Node *)(__first->_M_prev))->_M_next = __last;
        ((_Node *)(__position->_M_prev))->_M_next = __first;

        // Splice [first, last) into its new position.
        _Node *__tmp = (_Node *)(__position->_M_prev);
        __position->_M_prev = __last->_M_prev;
        __last->_M_prev = __first->_M_prev;
        __first->_M_prev = __tmp;
    }
}


template <class _Tp>
list<_Tp> &list<_Tp>::operator=(const list<_Tp> &__x)
{
    if (this != &__x) {
        iterator __first1 = begin();
        iterator __last1 = end();
        const_iterator __first2 = __x.begin();
        const_iterator __last2 = __x.end();
        while (__first1 != __last1 && __first2 != __last2)
            *__first1++ = *__first2++;
        if (__first2 == __last2)
            erase(__first1, __last1);
        else
            insert(__last1, __first2, __last2);
    }
    return *this;
}

template <class _Tp>
void list<_Tp>::_M_fill_assign(yint __n, const _Tp &__val) {
    iterator __i = begin();
    for (; __i != end() && __n > 0; ++__i, --__n)
        *__i = __val;
    if (__n > 0)
        insert(end(), __n, __val);
    else
        erase(__i, end());
}

template <class _Tp, class _Predicate>
void _S_remove_if(list<_Tp> &__that, _Predicate __pred) {
    typename list<_Tp>::iterator __first = __that.begin();
    typename list<_Tp>::iterator __last = __that.end();
    while (__first != __last) {
        typename list<_Tp>::iterator __next = __first;
        ++__next;
        if (__pred(*__first)) __that.erase(__first);
        __first = __next;
    }
}

template <class _Tp, class _BinaryPredicate>
void _S_unique(list<_Tp> &__that, _BinaryPredicate __binary_pred) {
    typename list<_Tp>::iterator __first = __that.begin();
    typename list<_Tp>::iterator __last = __that.end();
    if (__first == __last) return;
    typename list<_Tp>::iterator __next = __first;
    while (++__next != __last) {
        if (__binary_pred(*__first, *__next))
            __that.erase(__next);
        else
            __first = __next;
        __next = __first;
    }
}

template <class _Tp, class _StrictWeakOrdering>
void _S_merge(list<_Tp> &__that, list<_Tp> &__x,
    _StrictWeakOrdering __comp) {
    typedef typename list<_Tp>::iterator _Literator;
    _Literator __first1 = __that.begin();
    _Literator __last1 = __that.end();
    _Literator __first2 = __x.begin();
    _Literator __last2 = __x.end();
    while (__first1 != __last1 && __first2 != __last2)
        if (__comp(*__first2, *__first1)) {
            _Literator __next = __first2;
            _List_global_inst::_Transfer(__first1._M_node, __first2._M_node, (++__next)._M_node);
            __first2 = __next;
        } else
            ++__first1;
        if (__first2 != __last2) _List_global_inst::_Transfer(__last1._M_node, __first2._M_node, __last2._M_node);
}

template <class _Tp, class _StrictWeakOrdering>
void _S_sort(list<_Tp> &__that, _StrictWeakOrdering __comp) {
    // Do nothing if the list has length 0 or 1.
    if (__that._M_node->_M_next != __that._M_node &&
        (__that._M_node->_M_next)->_M_next != __that._M_node) {
        list<_Tp> __carry;
        list<_Tp> __counter[64];
        yint __fill = 0;
        while (!__that.empty()) {
            __carry.splice(__carry.begin(), __that, __that.begin());
            yint __i = 0;
            while (__i < __fill && !__counter[__i].empty()) {
                _S_merge(__counter[__i], __carry, __comp);
                __carry.swap(__counter[__i++]);
            }
            __carry.swap(__counter[__i]);
            if (__i == __fill) ++__fill;
        }

        for (yint __i = 1; __i < __fill; ++__i)
            _S_merge(__counter[__i], __counter[__i - 1], __comp);
        __that.swap(__counter[__fill - 1]);
    }
}



}
