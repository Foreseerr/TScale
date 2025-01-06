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

#include "nvector.h"
#include "nalgobase.h"
#include "npair.h"

namespace nstl
{
// Hashtable class, used to implement the hashed associative containers
// hash_set, hash_map, hash_multiset, and hash_multimap.

template <class _Val>
struct _Hashtable_node
{
    typedef _Hashtable_node<_Val> _Self;
    _Self *_M_next;
    _Val _M_val;

    _Hashtable_node() {}
    _Hashtable_node(const _Val &a) : _M_next(0), _M_val(a) {}
};

// some compilers require the names of template parameters to be the same
template <class _Val, class _Key, class _HF>
class hashtable;


template <class _Val, class _Key, class _HF, class _Node, class TRes>
struct _Ht_it
{
    typedef hashtable<_Val, _Key, _HF> _Hashtable;

    _Node *_M_cur;
    const _Hashtable *_M_ht;

    _Node *_M_skip_to_next()
    {
        size_t __bucket = _M_ht->_M_bkt_num(_M_cur->_M_val);
        size_t __h_sz;
        __h_sz = this->_M_ht->bucket_count();

        _Node *__i = 0;
        while (__i == 0 && ++__bucket < __h_sz)
            __i = (_Node *)_M_ht->_M_buckets[__bucket];
        return __i;
    }

    typedef _Ht_it<_Val, _Key, _HF, _Node, TRes> _Self;

    _Ht_it(_Node *_p, const _Hashtable *_t) : _M_cur(_p), _M_ht(_t) {}
    _Ht_it() {}
    template<class TN1, class TR1>
    _Ht_it(const _Ht_it<_Val, _Key, _HF, TN1, TR1> &a) : _M_cur(a._M_cur), _M_ht(a._M_ht) { }

    TRes &operator*() const {
        return this->_M_cur->_M_val;
    }
    TRes *operator->() const { return &_M_cur->_M_val; }

    _Self &operator++() {
        _Node *__n = this->_M_cur->_M_next;
        this->_M_cur = (__n != 0 ? __n : this->_M_skip_to_next());
        return *this;
    }
    inline  _Self operator++(int) {
        _Self __tmp = *this;
        ++*this;
        return __tmp;
    }
};

template <class _Val, class _Key, class _HF, class TN1, class TR1, class TN2, class TR2>
inline bool
    operator==(const _Ht_it<_Val, _Key, _HF, TN1, TR1> &__x,
        const _Ht_it<_Val, _Key, _HF, TN2, TR2> &__y) {
    return __x._M_cur == __y._M_cur;
}

template <class _Val, class _Key, class _HF, class TN1, class TR1, class TN2, class TR2>
inline bool
    operator!=(const _Ht_it<_Val, _Key, _HF, TN1, TR1> &__x,
        const _Ht_it<_Val, _Key, _HF, TN2, TR2> &__y) {
    return __x._M_cur != __y._M_cur;
}

#define __stl_num_primes  30
template <class _Tp>
class _Stl_prime {
public:
    static const size_t _M_list[__stl_num_primes];
};
template <class _Tp>
const size_t _Stl_prime<_Tp>::_M_list[__stl_num_primes] =
{
13ul, 29ul,
53ul,         97ul,         193ul,       389ul,       769ul,
1543ul,       3079ul,       6151ul,      12289ul,     24593ul,
49157ul,      98317ul,      196613ul,    393241ul,    786433ul,
1572869ul,    3145739ul,    6291469ul,   12582917ul,  25165843ul,
50331653ul,   100663319ul,  201326611ul, 402653189ul, 805306457ul,
1610612741ul, 3221225473ul, 4294967291ul
};


typedef _Stl_prime<bool> _Stl_prime_type;


template <class _Val, class _Key, class _HF>
class hashtable
{
    typedef hashtable<_Val, _Key, _HF> _Self;
    static inline const _Key &_M_get_key(const _Val &a) { return a.first; }
public:
    typedef _Key key_type;
    typedef _Val value_type;
    typedef _HF hasher;

    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef value_type &reference;
    typedef const value_type &const_reference;

    hasher hash_funct() const { return _M_hash; }

public:
    typedef _Hashtable_node<_Val> _Node;

private:
    typedef vector<void *> _BucketVector;
private:
    hasher                _M_hash;
    _BucketVector         _M_buckets;
    size_t _M_num_elements;
    const _Node *_M_get_bucket(size_t __n) const { return (_Node *)_M_buckets[__n]; }

public:
    typedef _Ht_it<_Val, _Key, _HF, _Node, _Val> iterator;
    typedef _Ht_it<_Val, _Key, _HF, const _Node, const _Val> const_iterator;
    friend iterator;
    friend const_iterator;

public:
    hashtable(size_t __n, const _HF &__hf)
        :
        _M_hash(__hf),
        _M_num_elements(0)
    {
        _M_initialize_buckets(__n);
    }

    hashtable(const _Self &__ht)
        :
        _M_hash(__ht._M_hash),
        _M_num_elements(0)
    {
        _M_copy_from(__ht);
    }

    _Self &operator= (const _Self &__ht)
    {
        if (&__ht != this) {
            clear();
            _M_hash = __ht._M_hash;
            _M_copy_from(__ht);
        }
        return *this;
    }

    ~hashtable() { clear(); }

    size_t size() const { return _M_num_elements; }
    bool empty() const { return size() == 0; }

    void swap(_Self &__ht)
    {
        nstl::swap(_M_hash, __ht._M_hash);
        _M_buckets.swap(__ht._M_buckets);
        nstl::swap(_M_num_elements, __ht._M_num_elements);
    }

    iterator begin()
    {
        for (size_t __n = 0; __n < _M_buckets.size(); ++__n)
            if (_M_buckets[__n])
                return iterator((_Node *)_M_buckets[__n], this);
        return end();
    }

    iterator end() { return iterator((_Node *)0, this); }

    const_iterator begin() const
    {
        for (size_t __n = 0; __n < _M_buckets.size(); ++__n)
            if (_M_buckets[__n])
                return const_iterator((_Node *)_M_buckets[__n], this);
        return end();
    }

    const_iterator end() const { return const_iterator((_Node *)0, this); }

    static bool inline _M_equal(const hashtable<_Val, _Key, _HF> &,
        const hashtable<_Val, _Key, _HF> &);

public:

    size_t bucket_count() const { return _M_buckets.size(); }

    size_t elems_in_bucket(size_t __bucket) const
    {
        size_t __result = 0;
        for (_Node *__cur = (_Node *)_M_buckets[__bucket]; __cur; __cur = __cur->_M_next)
            __result += 1;
        return __result;
    }

    pair<iterator, bool> insert_unique(const value_type &__obj)
    {
        resize(_M_num_elements + 1);
        return insert_unique_noresize(__obj);
    }

    iterator insert_equal(const value_type &__obj)
    {
        resize(_M_num_elements + 1);
        return insert_equal_noresize(__obj);
    }

    pair<iterator, bool> insert_unique_noresize(const value_type &__obj)
    {
        const size_t __n = _M_bkt_num(__obj);
        _Node *__first = (_Node *)_M_buckets[__n];

        for (_Node *__cur = __first; __cur; __cur = __cur->_M_next)
            if ((_M_get_key(__cur->_M_val) == _M_get_key(__obj)))
                return pair<iterator, bool>(iterator(__cur, this), false);

        _Node *__tmp = _M_new_node(__obj);
        __tmp->_M_next = __first;
        _M_buckets[__n] = __tmp;
        ++_M_num_elements;
        return pair<iterator, bool>(iterator(__tmp, this), true);
    }
    iterator insert_equal_noresize(const value_type &__obj)
    {
        const size_t __n = _M_bkt_num(__obj);
        _Node *__first = (_Node *)_M_buckets[__n];

        for (_Node *__cur = __first; __cur; __cur = __cur->_M_next)
            if ((_M_get_key(__cur->_M_val) == _M_get_key(__obj))) {
                _Node *__tmp = _M_new_node(__obj);
                __tmp->_M_next = __cur->_M_next;
                __cur->_M_next = __tmp;
                ++_M_num_elements;
                return iterator(__tmp, this);
            }

        _Node *__tmp = _M_new_node(__obj);
        __tmp->_M_next = __first;
        _M_buckets[__n] = __tmp;
        ++_M_num_elements;
        return iterator(__tmp, this);
    }

    template <class _InputIterator>
    void insert_unique(_InputIterator __f, _InputIterator __l)
    {
        for (; __f != __l; ++__f)
            insert_unique(*__f);
    }

    template <class _InputIterator>
    void insert_equal(_InputIterator __f, _InputIterator __l)
    {
        for (; __f != __l; ++__f)
            insert_equal(*__f);
    }


    reference find_or_insert(const value_type &__obj)
    {
        _Node *__first = _M_find(_M_get_key(__obj));
        if (__first)
            return __first->_M_val;
        else
            return _M_insert(__obj);
    }

private:
    template <class _KT>
    _Node *_M_find(const _KT &__key) const
    {
        size_t __n = (size_t)(((size_t)_M_hash(__key)) % _M_buckets.size());
        _Node *__first;
        for (__first = (_Node *)_M_buckets[__n];
            __first && !(_M_get_key(__first->_M_val) == __key);
            __first = __first->_M_next)
        {
        }
        return __first;
    }

public:
    template <class _KT>
    iterator find(const _KT &__key)
    {
        return iterator(_M_find(__key), this);
    }

    template <class _KT>
    const_iterator find(const _KT &__key) const
    {
        return const_iterator(_M_find(__key), this);
    }

    size_t count(const key_type &__key) const
    {
        const size_t __n = _M_bkt_num_key(__key);
        size_t __result = 0;

        for (const _Node *__cur = (_Node *)_M_buckets[__n]; __cur; __cur = __cur->_M_next)
            if ((_M_get_key(__cur->_M_val) == __key))
                ++__result;
        return __result;
    }

    pair<iterator, iterator>
        equal_range(const key_type &__key)
    {
        typedef pair<iterator, iterator> _Pii;
        const size_t __n = _M_bkt_num_key(__key);

        for (_Node *__first = (_Node *)_M_buckets[__n]; __first; __first = __first->_M_next)
            if ((_M_get_key(__first->_M_val) == __key)) {
                for (_Node *__cur = __first->_M_next; __cur; __cur = __cur->_M_next)
                    if (!(_M_get_key(__cur->_M_val) == __key))
                        return _Pii(iterator(__first, this), iterator(__cur, this));
                for (size_t __m = __n + 1; __m < _M_buckets.size(); ++__m)
                    if (_M_buckets[__m])
                        return _Pii(iterator(__first, this),
                            iterator((_Node *)_M_buckets[__m], this));
                return _Pii(iterator(__first, this), end());
            }
        return _Pii(end(), end());
    }

    pair<const_iterator, const_iterator>
        equal_range(const key_type &__key) const
    {
        typedef pair<const_iterator, const_iterator> _Pii;
        const size_t __n = _M_bkt_num_key(__key);

        for (const _Node *__first = (_Node *)_M_buckets[__n];
            __first;
            __first = __first->_M_next) {
            if ((_M_get_key(__first->_M_val) == __key)) {
                for (const _Node *__cur = __first->_M_next;
                    __cur;
                    __cur = __cur->_M_next)
                    if (!(_M_get_key(__cur->_M_val) == __key))
                        return _Pii(const_iterator(__first, this),
                            const_iterator(__cur, this));
                for (size_t __m = __n + 1; __m < _M_buckets.size(); ++__m)
                    if (_M_buckets[__m])
                        return _Pii(const_iterator(__first, this),
                            const_iterator((_Node *)_M_buckets[__m], this));
                return _Pii(const_iterator(__first, this), end());
            }
        }
        return _Pii(end(), end());
    }

    size_t erase(const key_type &__key)
    {
        const size_t __n = _M_bkt_num_key(__key);
        _Node *__first = (_Node *)_M_buckets[__n];
        size_t __erased = 0;

        if (__first) {
            _Node *__cur = __first;
            _Node *__next = __cur->_M_next;
            while (__next) {
                if ((_M_get_key(__next->_M_val) == __key)) {
                    __cur->_M_next = __next->_M_next;
                    _M_delete_node(__next);
                    __next = __cur->_M_next;
                    ++__erased;
                    --_M_num_elements;
                } else {
                    __cur = __next;
                    __next = __cur->_M_next;
                }
            }
            if ((_M_get_key(__first->_M_val) == __key)) {
                _M_buckets[__n] = __first->_M_next;
                _M_delete_node(__first);
                ++__erased;
                --_M_num_elements;
            }
        }
        return __erased;
    }
    void erase(const const_iterator &__it)
    {
        const _Node *__p = __it._M_cur;
        if (__p) {
            const size_t __n = _M_bkt_num(__p->_M_val);
            _Node *__cur = (_Node *)_M_buckets[__n];

            if (__cur == __p) {
                _M_buckets[__n] = __cur->_M_next;
                _M_delete_node(__cur);
                --_M_num_elements;
            } else {
                _Node *__next = __cur->_M_next;
                while (__next) {
                    if (__next == __p) {
                        __cur->_M_next = __next->_M_next;
                        _M_delete_node(__next);
                        --_M_num_elements;
                        break;
                    } else {
                        __cur = __next;
                        __next = __cur->_M_next;
                    }
                }
            }
        }
    }

    void erase(const_iterator __first, const_iterator __last);
    void resize(size_t __num_elements_hint);
    void clear();

public:
    // this is for hash_map::operator[]
    reference _M_insert(const value_type &__obj)
    {
        resize(_M_num_elements + 1);

        size_t __n = _M_bkt_num(__obj);
        _Node *__first = (_Node *)_M_buckets[__n];

        _Node *__tmp = _M_new_node(__obj);
        __tmp->_M_next = __first;
        _M_buckets[__n] = __tmp;
        ++_M_num_elements;
        return __tmp->_M_val;
    }

private:

    size_t _M_next_size(size_t __n) const;

    void _M_initialize_buckets(size_t __n)
    {
        const size_t __n_buckets = _M_next_size(__n);
        _M_buckets.reserve(__n_buckets);
        _M_buckets.insert(_M_buckets.end(), __n_buckets, (void *)0);
        _M_num_elements = 0;
    }

    size_t _M_bkt_num_key(const key_type &__key) const
    {
        return _M_bkt_num_key(__key, _M_buckets.size());
    }

    size_t _M_bkt_num(const value_type &__obj) const
    {
        return _M_bkt_num_key(_M_get_key(__obj));
    }

    size_t _M_bkt_num_key(const key_type &__key, size_t __n) const
    {
        return (size_t)(((size_t)_M_hash(__key)) % __n);
    }

    size_t _M_bkt_num(const value_type &__obj, size_t __n) const
    {
        return _M_bkt_num_key(_M_get_key(__obj), __n);
    }

    _Node *_M_new_node(const value_type &_obj)
    {
        _Node *__n = new _Node(_obj);//_M_num_elements.allocate(1);
        __n->_M_next = 0;
        //_Construct(&__n->_M_val, __obj);
        return __n;
    }

    void _M_delete_node(_Node *__n)
    {
        delete __n;
    }

    void _M_erase_bucket(const size_t __n, _Node *__first, _Node *__last);
    void _M_erase_bucket(const size_t __n, _Node *__last);

    void _M_copy_from(const _Self &__ht);
};

template <class _Val, class _Key, class _HF>
inline bool operator==(const hashtable<_Val, _Key, _HF> &__ht1,
    const hashtable<_Val, _Key, _HF> &__ht2)
{
    return hashtable<_Val, _Key, _HF>::_M_equal(__ht1, __ht2);
}

template <class _Val, class _Key, class _HF>
inline void swap(hashtable<_Val, _Key, _HF> &__ht1,
    hashtable<_Val, _Key, _HF> &__ht2) {
    __ht1.swap(__ht2);
}



template <class _Val, class _Key, class _HF>
size_t
    hashtable<_Val, _Key, _HF>::_M_next_size(size_t __n) const {
    const size_t *__first = (const size_t *)_Stl_prime_type::_M_list;
    const size_t *__last = (const size_t *)_Stl_prime_type::_M_list + (size_t)__stl_num_primes;
    const size_t *pos = __lower_bound(__first, __last, __n, less(), (size_t *)0);
    return (pos == __last ? *(__last - 1) : *pos);
}

template <class _Val, class _Key, class _HF>
bool
    hashtable<_Val, _Key, _HF>::_M_equal(
        const hashtable<_Val, _Key, _HF> &__ht1,
        const hashtable<_Val, _Key, _HF> &__ht2)
{
    //  typedef _Hashtable_node<_Val> _Node;
    if (__ht1.bucket_count() != __ht2.bucket_count())
        return false;
    for (size_t __n = 0; __n < __ht1.bucket_count(); ++__n) {
        const _Node *__cur1 = __ht1._M_get_bucket(__n);
        const _Node *__cur2 = __ht2._M_get_bucket(__n);
        for (; __cur1 && __cur2 && __cur1->_M_val == __cur2->_M_val;
            __cur1 = __cur1->_M_next, __cur2 = __cur2->_M_next)
        {
        }
        if (__cur1 || __cur2)
            return false;
    }
    return true;
}


template <class _Val, class _Key, class _HF>
void hashtable<_Val, _Key, _HF>
    ::erase(const_iterator _c_first, const_iterator _c_last)
{
    iterator &__first = (iterator &)_c_first;
    iterator &__last = (iterator &)_c_last;
    size_t __f_bucket = __first._M_cur ?
        _M_bkt_num(__first._M_cur->_M_val) : _M_buckets.size();
    size_t __l_bucket = __last._M_cur ?
        _M_bkt_num(__last._M_cur->_M_val) : _M_buckets.size();
    if (__first._M_cur == __last._M_cur)
        return;
    else if (__f_bucket == __l_bucket)
        _M_erase_bucket(__f_bucket, __first._M_cur, __last._M_cur);
    else {
        _M_erase_bucket(__f_bucket, __first._M_cur, 0);
        for (size_t __n = __f_bucket + 1; __n < __l_bucket; ++__n)
            _M_erase_bucket(__n, 0);
        if (__l_bucket != _M_buckets.size())
            _M_erase_bucket(__l_bucket, __last._M_cur);
    }
}

template <class _Val, class _Key, class _HF>
void hashtable<_Val, _Key, _HF>
    ::resize(size_t __num_elements_hint)
{
    const size_t __old_n = _M_buckets.size();
    if (__num_elements_hint > __old_n) {
        const size_t __n = _M_next_size(__num_elements_hint);
        if (__n > __old_n) {
            _BucketVector __tmp(__n, (void *)(0));
            for (size_t __bucket = 0; __bucket < __old_n; ++__bucket) {
                _Node *__first = (_Node *)_M_buckets[__bucket];
                while (__first) {
                    size_t __new_bucket = _M_bkt_num(__first->_M_val, __n);
                    _M_buckets[__bucket] = __first->_M_next;
                    __first->_M_next = (_Node *)__tmp[__new_bucket];
                    __tmp[__new_bucket] = __first;
                    __first = (_Node *)_M_buckets[__bucket];
                }
            }
            _M_buckets.swap(__tmp);
        }
    }
}

template <class _Val, class _Key, class _HF>
void hashtable<_Val, _Key, _HF>
    ::_M_erase_bucket(const size_t __n, _Node *__first, _Node *__last)
{
    _Node *__cur = (_Node *)_M_buckets[__n];
    if (__cur == __first)
        _M_erase_bucket(__n, __last);
    else {
        _Node *__next;
        for (__next = __cur->_M_next;
            __next != __first;
            __cur = __next, __next = __cur->_M_next)
            ;
        while (__next != __last) {
            __cur->_M_next = __next->_M_next;
            _M_delete_node(__next);
            __next = __cur->_M_next;
            --_M_num_elements;
        }
    }
}

template <class _Val, class _Key, class _HF>
void hashtable<_Val, _Key, _HF>
    ::_M_erase_bucket(const size_t __n, _Node *__last)
{
    _Node *__cur = (_Node *)_M_buckets[__n];
    while (__cur && __cur != __last) {
        _Node *__next = __cur->_M_next;
        _M_delete_node(__cur);
        __cur = __next;
        _M_buckets[__n] = __cur;
        --_M_num_elements;
    }
}

template <class _Val, class _Key, class _HF>
void hashtable<_Val, _Key, _HF>::clear()
{
    for (size_t __i = 0; __i < _M_buckets.size(); ++__i) {
        _Node *__cur = (_Node *)_M_buckets[__i];
        while (__cur != 0) {
            _Node *__next = __cur->_M_next;
            _M_delete_node(__cur);
            __cur = __next;
        }
        _M_buckets[__i] = 0;
    }
    _M_num_elements = 0;
}


template <class _Val, class _Key, class _HF>
void hashtable<_Val, _Key, _HF>
    ::_M_copy_from(const hashtable<_Val, _Key, _HF> &__ht)
{
    _M_buckets.clear();
    _M_buckets.reserve(__ht._M_buckets.size());
    _M_buckets.insert(_M_buckets.end(), __ht._M_buckets.size(), (void *)0);
    for (size_t __i = 0; __i < __ht._M_buckets.size(); ++__i) {
        const _Node *__cur = (_Node *)__ht._M_buckets[__i];
        if (__cur) {
            _Node *__xcopy = _M_new_node(__cur->_M_val);
            _M_buckets[__i] = __xcopy;

            for (_Node *__next = __cur->_M_next;
                __next;
                __cur = __next, __next = __cur->_M_next) {
                __xcopy->_M_next = _M_new_node(__next->_M_val);
                __xcopy = __xcopy->_M_next;
            }
        }
    }
    _M_num_elements = __ht._M_num_elements;
}

# undef __stl_num_primes
}
