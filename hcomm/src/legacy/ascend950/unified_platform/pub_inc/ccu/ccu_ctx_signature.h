/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_CTX_SIGNATURE_H
#define HCCL_CCU_CTX_SIGNATURE_H

#include <sstream>
#include <iostream>
#include <string>
#include <ios>
#include "hash_utils.h"
#include "string_util.h"

namespace Hccl {

class CcuCtxSignature {
public:
    CcuCtxSignature() = default;
    ~CcuCtxSignature() = default;
    CcuCtxSignature(const CcuCtxSignature& other)
    {
        // 实现复制构造函数
        data << other.data.str();
    }

    void operator=(const CcuCtxSignature& other)
    {
        // 实现赋值操作
        data << other.data.str();
    }

    bool operator==(const CcuCtxSignature& rhs) const { return this == &rhs || data.str() == rhs.data.str(); }

    // 用法Append<T>(t)
    template <typename T>
    void Append(T t)
    {
        data << t;
    }

    void Append(const CcuCtxSignature& other) { data << other.data.str(); }

    std::string Describe() const { return StringFormat("CcuCtxSignature[data=%s]", data.str().c_str()); }

    // 下掉CcuContext GetSignatrue
    std::string GetData() const { return data.str(); }

    friend class std::hash<Hccl::CcuCtxSignature>;

private:
    std::ostringstream data;
};

} // namespace Hccl

namespace std {

template <>
class hash<Hccl::CcuCtxSignature> {
public:
    size_t operator()(const Hccl::CcuCtxSignature& signature) const
    {
        auto dataHash = hash<string>{}(signature.GetData());
        return Hccl::HashCombine({dataHash});
    }
};

} // namespace std

#endif // HCCL_CCU_CTX_SIGNATURE_H
