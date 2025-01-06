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

#include "nhash_table.h"
#include "nhash_fun.h"

namespace nstl
{

template <class _Key, class _Tp, class _HashFcn = hash<_Key> >
class hash_map
{
private:
    typedef hashtable<pair < const _Key, _Tp >, _Key, _HashFcn> _Ht;
    typedef hash_map<_Key, _Tp, _HashFcn> _Self;

public:
    typedef typename _Ht::key_type key_type;
    typedef typename _Ht::value_type value_type;
    typedef typename _Ht::hasher hasher;

    typedef typename _Ht::iterator iterator;
    typedef typename _Ht::const_iterator const_iterator;

private:
    _Ht _M_ht;

public:
    hash_map() : _M_ht(10, hasher()) {}
    explicit hash_map(yint __n)
        : _M_ht(__n, hasher()) {}
    hash_map(yint __n, const hasher &__hf)
        : _M_ht(__n, __hf) {}

    template <class _InputIterator>
    hash_map(_InputIterator __f, _InputIterator __l)
        : _M_ht(10, hasher())
    {
        _M_ht.insert_unique(__f, __l);
    }
    template <class _InputIterator>
    hash_map(_InputIterator __f, _InputIterator __l, yint __n)
        : _M_ht(__n, hasher())
    {
        _M_ht.insert_unique(__f, __l);
    }
    template <class _InputIterator>
    hash_map(_InputIterator __f, _InputIterator __l, yint __n,
        const hasher &__hf)
        : _M_ht(__n, __hf)
    {
        _M_ht.insert_unique(__f, __l);
    }

public:
    yint size() const { return _M_ht.size(); }
    bool empty() const { return _M_ht.empty(); }
    void swap(_Self &__hs) { _M_ht.swap(__hs._M_ht); }
    iterator begin() { return _M_ht.begin(); }
    iterator end() { return _M_ht.end(); }
    const_iterator begin() const { return _M_ht.begin(); }
    const_iterator end() const { return _M_ht.end(); }

public:
    pair<iterator, bool> insert(const value_type &__obj)
    {
        return _M_ht.insert_unique(__obj);
    }
    template <class _InputIterator>
    void insert(_InputIterator __f, _InputIterator __l)
    {
        _M_ht.insert_unique(__f, __l);
    }
    pair<iterator, bool> insert_noresize(const value_type &__obj)
    {
        return _M_ht.insert_unique_noresize(__obj);
    }

    iterator find(const key_type &__key) { return _M_ht.find(__key); }
    const_iterator find(const key_type &__key) const { return _M_ht.find(__key); }

    _Tp &operator[](const key_type &__key) {
        iterator __it = _M_ht.find(__key);
        return (__it == _M_ht.end() ?
            _M_ht._M_insert(value_type(__key, _Tp())).second :
            (*__it).second);
    }

    yint count(const key_type &__key) const { return _M_ht.count(__key); }

    pair<iterator, iterator> equal_range(const key_type &__key)
    {
        return _M_ht.equal_range(__key);
    }
    pair<const_iterator, const_iterator>
        equal_range(const key_type &__key) const
    {
        return _M_ht.equal_range(__key);
    }

    yint erase(const key_type &__key) { return _M_ht.erase(__key); }
    void erase(iterator __it) { _M_ht.erase(__it); }
    void erase(iterator __f, iterator __l) { _M_ht.erase(__f, __l); }
    void clear() { _M_ht.clear(); }

    void resize(yint __hint) { _M_ht.resize(__hint); }
    static bool _M_equal(const _Self &__x, const _Self &__y) {
        return _Ht::_M_equal(__x._M_ht, __y._M_ht);
    }
};


template <class _Key, class _Tp, class _HashFcn>
inline bool
    operator==(const hash_map<_Key, _Tp, _HashFcn> &__hm1,
        const hash_map<_Key, _Tp, _HashFcn> &__hm2)
{
    return hash_map<_Key, _Tp, _HashFcn>::_M_equal(__hm1, __hm2);
}

template <class _Key, class _Tp, class _HashFcn>
inline bool
    operator!=(const hash_map<_Key, _Tp, _HashFcn> &__hm1,
        const hash_map<_Key, _Tp, _HashFcn> &__hm2)
{
    return !(__hm1 == __hm2);
}


template <class _Key, class _Tp, class _HashFcn>
inline void
    swap(hash_map<_Key, _Tp, _HashFcn> &__hm1,
        hash_map<_Key, _Tp, _HashFcn> &__hm2)
{
    __hm1.swap(__hm2);
}

}
