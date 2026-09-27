/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ITERATOR_UNCONST_H
#define HCCLV2_ITERATOR_UNCONST_H

namespace Hccl {
using namespace std;

template <template <class U, typename _Alloc = std::allocator<U>> class Sequence, typename T, typename Enable = void>
class BaseIterator {};

template <template <class U, typename _Alloc = std::allocator<U>> class Sequence, typename T>
class BaseIterator<Sequence, T, typename enable_if<IsSharedPtr<T>::value || IsUniquePtr<T>::value>::type> {
public:
    using V = typename T::element_type;

    BaseIterator() : iter(nullptr), end(nullptr) {};

    explicit BaseIterator(Sequence<T>& seq) : iter(seq.begin()), end(seq.end()) {};
    virtual ~BaseIterator() {};

    virtual V& operator*() { return *(*iter); };

    virtual V* operator->() { return (*iter).get(); };

    virtual BaseIterator& Next()
    {
        iter++;
        return *this;
    };

    virtual BaseIterator& operator++()
    {
        iter++;
        return *this;
    }

    virtual bool HasNext() { return iter != end; };

protected:
    typename Sequence<T>::iterator iter;
    typename Sequence<T>::iterator end;
};

} // namespace Hccl

#endif // HCCLV2_ITERATOR_H
