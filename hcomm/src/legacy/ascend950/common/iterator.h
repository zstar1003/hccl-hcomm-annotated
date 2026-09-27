/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ITERATOR_H
#define HCCLV2_ITERATOR_H

#include <iterator>
#include <memory>

namespace Hccl {

template <typename T, typename Enable = void>
struct IsSharedPtr final {
    static const bool value = false;
};

template <typename T>
struct IsSharedPtr<
    T, typename std::enable_if<std::is_same<T, std::shared_ptr<typename T::element_type>>::value>::type> {
    static const bool value = true;
};

template <typename T, typename Enable = void>
struct IsUniquePtr final {
    static const bool value = false;
};

template <typename T>
struct IsUniquePtr<
    T, typename std::enable_if<std::is_same<T, std::unique_ptr<typename T::element_type>>::value>::type> {
    static const bool value = true;
};

template <typename T, typename Enable = void>
struct IsSmartPtr final {
    static const bool value = false;
};

template <typename T>
struct IsSmartPtr<
    T, typename std::enable_if<
           IsSharedPtr<T>::value || IsUniquePtr<T>::value
           || std::is_same<T, std::weak_ptr<typename T::element_type>>::value>::type> {
    static const bool value = true;
};

template <template <class U, typename _Alloc = std::allocator<U>> class Sequence, typename T, typename Enable = void>
class BaseConstIterator {};

template <template <class U, typename _Alloc = std::allocator<U>> class Sequence, typename T>
class BaseConstIterator<Sequence, T, typename std::enable_if<IsSharedPtr<T>::value || IsUniquePtr<T>::value>::type> {
public:
    using V = typename T::element_type;

    BaseConstIterator() : iter(nullptr), end(nullptr) {}

    explicit BaseConstIterator(const Sequence<T>& seq) : iter(seq.cbegin()), end(seq.cend()) {}

    virtual const V& operator*() { return *(*iter); }

    virtual const V* operator->() { return (*iter).get(); }

    virtual BaseConstIterator& Next()
    {
        iter++;
        return *this;
    }

    virtual BaseConstIterator& operator++()
    {
        iter++;
        return *this;
    }

    virtual bool HasNext() { return iter != end; }

    virtual ~BaseConstIterator() {};

protected:
    typename Sequence<T>::const_iterator iter;
    typename Sequence<T>::const_iterator end;
};

template <template <class U, typename _Alloc = std::allocator<U>> class Sequence, typename T>
class BaseConstIterator<Sequence, T, typename std::enable_if<!IsSmartPtr<T>::value>::type> {
public:
    BaseConstIterator() : iter(nullptr), end(nullptr) {}

    explicit BaseConstIterator(const Sequence<T>& seq) : iter(seq.cbegin()), end(seq.cend()) {}

    virtual const T& operator*() { return *iter; }

    virtual const T* operator->() { return &*iter; }

    virtual BaseConstIterator& Next()
    {
        iter++;
        return *this;
    }

    virtual BaseConstIterator& operator++()
    {
        iter++;
        return *this;
    }

    virtual bool HasNext() { return iter != end; }

protected:
    typename Sequence<T>::const_iterator iter;
    typename Sequence<T>::const_iterator end;
};

} // namespace Hccl

#endif // HCCLV2_ITERATOR_H
