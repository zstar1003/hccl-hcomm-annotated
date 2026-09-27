/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "buffer.h"
#include "range_utils.h"
#include "log.h"
#include "string_util.h"
#include "exception_util.h"
#include "internal_exception.h"
namespace Hccl {

Buffer::Buffer(uintptr_t addr, std::size_t size) : addr_(addr), size_(size) {}

Buffer::Buffer(std::size_t size) : size_(size) {}

Buffer::Buffer(uintptr_t addr, std::size_t size, HcclMemType memType) : addr_(addr), size_(size), memType_(memType) {}

Buffer::Buffer(uintptr_t addr, std::size_t size, HcclMemType memType, const char* memInfo)
    : addr_(addr),
      size_(size),
      memType_(memType)
{
    SetMemInfo(memInfo);
}

Buffer::Buffer(uintptr_t addr, std::size_t size, const char* memInfo) : addr_(addr), size_(size)
{
    SetMemInfo(memInfo);
}

void Buffer::SetMemInfo(const char* memInfo)
{
    if (memInfo == nullptr || memInfo[0] == '\0') {
        memInfo_[0] = '\0'; // 初始化为空字符串
        return;
    }
    size_t memLen = strlen(memInfo);
    if (memLen >= sizeof(memInfo_)) {
        THROW<InternalException>("[Buffer] memInfo too long, len[%zu], max[%zu].", memLen, sizeof(memInfo_) - 1);
    }
    int sret = snprintf_s(memInfo_, sizeof(memInfo_), memLen, "%s", memInfo);
    if (sret <= 0) {
        THROW<InternalException>(
            "[Buffer] snprintf_s failed, dest[%p], destMax[%zu], count[%zu], ret[%d].", static_cast<void*>(memInfo_),
            sizeof(memInfo_), memLen, sret);
    }
}
uintptr_t Buffer::GetAddr() const { return addr_; }

size_t Buffer::GetSize() const { return size_; }

HcclMemType Buffer::GetMemType() const { return memType_; }

const std::string Buffer::GetMemInfo() const { return memInfo_; }

std::string Buffer::Describe() const
{
    return StringFormat("Buffer[addr=0x%llx, size=0x%llx, memInfo=%s]", addr_, size_, memInfo_);
}

bool Buffer::Contains(Buffer* buf) const { return IsRangeInclude(addr_, size_, buf->addr_, buf->size_); }

bool Buffer::Contains(uintptr_t bufAddr, size_t bufSize) const
{
    return IsRangeInclude(addr_, size_, bufAddr, bufSize);
}

Buffer Buffer::Range(std::size_t offset, std::size_t givenSize) const
{
    HCCL_INFO("[Buffer::Range] offset[%zu] givenSize[%zu] size[%zu]", offset, givenSize, size_);
    if (addr_ != 0 && (offset + givenSize) <= size_) {
        return Buffer(addr_ + offset, givenSize);
    } else {
        HCCL_WARNING("Buffer range[%zu] size[%zu Byte] error or addr[0x%llx] null", offset + givenSize, size_, addr_);
        return Buffer(0, 0);
    }
}

} // namespace Hccl
