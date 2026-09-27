/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BUFFER_KEY_H
#define BUFFER_KEY_H
#include <string>
#include <stdint.h>

namespace hcomm {
template <typename A, typename S>
class BufferKey {
public:
    using AddrType = A;
    using SizeType = S;

private:
    AddrType addr_;
    SizeType size_;

public:
    BufferKey(AddrType addr, SizeType size) : addr_(addr), size_(size) {}

    AddrType Addr() const { return addr_; }

    SizeType Size() const { return size_; }

    bool operator==(const BufferKey& other) const { return addr_ == other.addr_ && size_ == other.size_; }

    bool operator!=(const BufferKey& other) const { return addr_ != other.addr_ || size_ != other.size_; }

    bool operator<(const BufferKey& other) const
    {
        return addr_ < other.addr_ || (addr_ == other.addr_ && size_ < other.size_);
    }

    // 不含等于情况
    inline bool IsSubset(const BufferKey& other) const
    {
        // 子集判断：当前 key 的起始地址 >= 其他 key 的起始地址，且结束地址 <= 其他 key 的结束地址
        return *this != other && addr_ >= other.addr_ && (addr_ + size_) <= (other.addr_ + other.size_);
    }

    // 不含等于情况
    inline bool IsSuperset(const BufferKey& other) const
    {
        // 超集判断：当前 key 的起始地址 < 其他 key 的起始地址，且结束地址 > 其他 key 的结束地址
        return *this != other && addr_ <= other.addr_ && (addr_ + size_) >= (other.addr_ + other.size_);
    }

    inline bool IsIntersect(const BufferKey& other) const
    {
        // 检查是否有交集：两个区域重叠。
        return (addr_ < other.addr_ + other.size_ && addr_ + size_ > other.addr_)
               || (other.addr_ < addr_ + size_ && other.addr_ + other.size_ > addr_);
    }

    inline bool IsDisjoint(const BufferKey& other) const
    {
        // 检查是否没有交集：一方完全在另一方之前或之后。
        return (addr_ + size_ <= other.addr_) || (addr_ >= other.addr_ + other.size_);
    }

    inline std::string ToString() const
    {
        return std::string("addr:") + std::to_string(addr_) + std::string(", size:") + std::to_string(size_);
    }
};
} // namespace hcomm

// 兼容旧 Hccl namespace 引用
namespace Hccl {
template <typename A, typename S>
using BufferKey = hcomm::BufferKey<A, S>;
}

// 兼容旧 hccl namespace 引用
namespace hccl {
template <typename A, typename S>
using BufferKey = hcomm::BufferKey<A, S>;
}
#endif
