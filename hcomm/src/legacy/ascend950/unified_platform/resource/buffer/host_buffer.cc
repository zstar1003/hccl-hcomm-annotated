/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_buffer.h"
#include "string_util.h"
#include "internal_exception.h"
#include "exception_util.h"
#include "orion_adapter_rts.h"
namespace Hccl {

HostBuffer::HostBuffer(uintptr_t devAddr, std::size_t devSize) : Buffer(devSize), selfOwned(false)
{
    addr_ = devAddr;
    size_ = devSize;
}

HostBuffer::HostBuffer(std::size_t allocSize) : Buffer(allocSize), selfOwned(true)
{
    if (allocSize == 0) {
        std::string msg = "allocaSize should not be 0!";
        THROW<InternalException>(msg);
    }
    addr_ = reinterpret_cast<uintptr_t>(HrtMallocHost(allocSize));
}

HostBuffer::~HostBuffer()
{
    if (selfOwned) {
        DECTOR_TRY_CATCH("Buffer", HrtFreeHost(reinterpret_cast<void*>(addr_)))
    }
}

std::string HostBuffer::Describe() const
{
    return StringFormat("HostBuffer[addr=0x%llx, size=0x%llx, selfOwned=%d]", addr_, size_, selfOwned);
}

bool HostBuffer::GetSelfOwned() const { return selfOwned; }

} // namespace Hccl
