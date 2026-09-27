/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LLT_DATA_SLICE_H
#define LLT_DATA_SLICE_H

#include "checker_buffer_type.h"

#include "llt_common.h"

#include <string>
#include <iostream>
#include <sstream>
#include <memory>

#include "checker_string_util.h"

namespace checker {

class DataSlice {
public:
    explicit DataSlice() : type(BufferType::INPUT), offset(0), size(0) {}

    explicit DataSlice(u64 size) : type(BufferType::INPUT), offset(0), size(size) {}

    DataSlice(BufferType bufferType, u64 offset, u64 size) : type(bufferType), offset(offset), size(size) {}

    bool operator==(const DataSlice& other) const
    {
        return (type == other.type && offset == other.offset && size == other.size);
    }

    bool operator!=(const DataSlice& other) const
    {
        return (type != other.type || offset != other.offset || size != other.size);
    }

    std::string Describe() const
    {
        return StringFormat("DataSlice[%s, offset=%llu, size=%llu]", type.Describe().c_str(), offset, size);
    }

    inline BufferType GetType() const { return type; }

    inline u64 GetOffset() const { return offset; }

    inline u64 GetSize() const { return size; }

    void SetBufferType(const BufferType bufferType) { type = bufferType; }

    void SetOffset(u64 off) { offset = off; }

    void SetSize(u64 sizeOfSlice) { size = sizeOfSlice; }

private:
    BufferType type;
    u64 offset;
    u64 size;
};

bool DataSliceSizeIsEqual(std::unique_ptr<DataSlice>& a, std::unique_ptr<DataSlice>& b);
bool DataSliceSizeIsEqual(std::unique_ptr<DataSlice>& a, std::unique_ptr<DataSlice>& b, std::unique_ptr<DataSlice>& c);

} // namespace checker
#endif
