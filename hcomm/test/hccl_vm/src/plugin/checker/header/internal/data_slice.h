/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DATA_SLICE_H
#define DATA_SLICE_H
#include "enum_factory.h"
#include "sim_common.h"

class DataSlice {
public:
    DataSlice() : type_(BufferType::INPUT), offset_(0), size_(0), rawAddr_(0) {}
    DataSlice(BufferType type, uint64_t offset, uint64_t size) : type_(type), offset_(offset), size_(size), rawAddr_(0)
    {}
    std::string Describe() const
    {
        return StringFormat("DataSlice[type=%d, offset=0x%llX, size=0x%llX]", type_, offset_, size_);
    }
    inline BufferType GetType() const { return type_; }
    inline uint64_t GetOffset() const { return offset_; }
    inline uint64_t GetSize() const { return size_; }
    inline uint64_t GetRawAddr() const { return rawAddr_; }
    void SetBufferType(const BufferType type) { type_ = type; }
    void SetOffset(uint64_t offset) { offset_ = offset; }
    void SetSize(uint64_t size) { size_ = size; }
    void SetRawAddr(uint64_t rawAddr) { rawAddr_ = rawAddr; }

private:
    BufferType type_;
    uint64_t offset_;
    uint64_t size_;
    uint64_t rawAddr_;
};

#endif
