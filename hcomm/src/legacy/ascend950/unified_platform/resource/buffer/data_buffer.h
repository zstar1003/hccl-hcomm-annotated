/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_Data_DataBuffer_H
#define HCCL_Data_DataBuffer_H

#include <cstdint>
#include <cstddef>
#include <string>
#include "string_util.h"
#include "hash_utils.h"
namespace Hccl {

class DataBuffer {
public:
    DataBuffer(uintptr_t addr, std::size_t size) : addr(addr), size(size) {};

    explicit DataBuffer(std::size_t size) : size(size) {};

    virtual ~DataBuffer() = default;

    uintptr_t GetAddr() const { return addr; }

    size_t GetSize() const { return size; }

    virtual std::string Describe() const { return StringFormat("Buffer[addr=0x%llx, size=0x%llx]", addr, size); }

    // "=="运算符
    bool operator==(const DataBuffer& that) const { return (addr == that.GetAddr()) && (size == that.GetSize()); }

    // "!="运算符
    bool operator!=(const DataBuffer& that) const { return (addr != that.GetAddr()) || (size != that.GetSize()); }

protected:
    uintptr_t addr{0};
    std::size_t size{0};
};
} // namespace Hccl

namespace std {

template <>
class hash<Hccl::DataBuffer> {
public:
    size_t operator()(const Hccl::DataBuffer& dataBuffer) const
    {
        auto addrHash = hash<uint8_t>{}(dataBuffer.GetAddr());
        auto sizeHash = hash<uint8_t>{}(dataBuffer.GetSize());

        return Hccl::HashCombine({addrHash, sizeHash});
    }
};
} // namespace std
#endif // HCCL_DataBuffer_H
