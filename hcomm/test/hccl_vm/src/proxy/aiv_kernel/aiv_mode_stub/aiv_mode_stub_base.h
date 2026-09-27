/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_AIV_MODE_STUB_BASE_H
#define AIV_AIV_MODE_STUB_BASE_H

#include <cstdint>
#include <sstream>
#include <string>

namespace AivSim {
using RankId = uint32_t;
using BlockId = uint32_t;
using TaskId = uint32_t;
constexpr uint32_t AIV_OP_KERNEL_NAME_MAX_LEN = 256;

struct AivOpParam {
    uint32_t dataType;
    uint64_t len;
    uint32_t reduceOp;
    uint32_t root;
    uint32_t sliceId;
    uint64_t inputStride;
    uint64_t outputStride;
    char kernelName[AIV_OP_KERNEL_NAME_MAX_LEN];
};

enum class AivBufferType : uint32_t {
    INPUT = 0,
    OUTPUT,
    CCL,
    UB,
    AIV_COMM,
};

std::string inline GetAivBufferTypeName(AivBufferType bufferType)
{
    switch (bufferType) {
        case AivBufferType::INPUT:
            return "INPUT";
        case AivBufferType::OUTPUT:
            return "OUTPUT";
        case AivBufferType::CCL:
            return "CCL";
        case AivBufferType::UB:
            return "UB";
        case AivBufferType::AIV_COMM:
            return "AIV_COMM";
        default:
            return "UNKNOWN";
    }
}

class AivDataSlice {
public:
    AivDataSlice() : type_(AivBufferType::INPUT), offset_(0), size_(0) {}
    AivDataSlice(AivBufferType type, uint64_t offset, uint64_t size) : type_(type), offset_(offset), size_(size) {}
    inline AivBufferType GetType() const { return type_; }
    inline uint64_t GetOffset() const { return offset_; }
    inline uint64_t GetSize() const { return size_; }
    void SetBufferType(const AivBufferType type) { type_ = type; }
    void SetOffset(uint64_t offset) { offset_ = offset; }
    void SetSize(uint64_t size) { size_ = size; }

    std::string Describe() const
    {
        std::stringstream ss;
        ss << "DataSlice[type=" << GetAivBufferTypeName(type_) << ", offset=0x" << std::uppercase << std::hex << offset_
           << std::dec << ", size=" << size_ << "]";
        return ss.str();
    }

private:
    AivBufferType type_;
    uint64_t offset_;
    uint64_t size_;
};

struct Mem {
    uint64_t addr{0};
    uint64_t size{0};

    std::string Describe() const
    {
        std::stringstream ss;
        ss << "{Addr=0x" << std::hex << addr << std::dec << ", Size=" << size << "}";
        return ss.str();
    }
};

enum class ReduceOp : uint32_t {
    REDUCE_SUM = 0,
    REDUCE_PROD,
    REDUCE_MAX,
    REDUCE_MIN,
    REDUCE_RESERVED,
};

std::string inline GetReduceOpName(ReduceOp reduceOp)
{
    switch (reduceOp) {
        case ReduceOp::REDUCE_SUM:
            return "SUM";
        case ReduceOp::REDUCE_PROD:
            return "PROD";
        case ReduceOp::REDUCE_MAX:
            return "MAX";
        case ReduceOp::REDUCE_MIN:
            return "MIN";
        default:
            return "UNKNOWN";
    }
}
} // namespace AivSim

#endif // AIV_AIV_MODE_STUB_BASE_H
