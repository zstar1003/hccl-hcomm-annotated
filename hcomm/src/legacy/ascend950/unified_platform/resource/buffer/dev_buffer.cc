/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "dev_buffer.h"
#include "string_util.h"
#include "internal_exception.h"
#include "exception_util.h"
#include "orion_adapter_rts.h"
namespace Hccl {

DevBuffer::DevBuffer(uintptr_t devAddr, std::size_t devSize) : Buffer(devSize), selfOwned(false)
{
    addr_ = devAddr;
    size_ = devSize;
}

DevBuffer::DevBuffer(std::size_t allocSize) : Buffer(allocSize), selfOwned(true)
{
    if (allocSize == 0) {
        std::string msg = "allocaSize should not be 0!";
        THROW<InternalException>(msg);
    }
    addr_ = ReinterpretAs<uintptr_t>(HrtMalloc(allocSize, static_cast<u32>(ACL_MEM_TYPE_HIGH_BAND_WIDTH)));
}

std::shared_ptr<DevBuffer> DevBuffer::Create(uintptr_t devAddr, std::size_t devSize)
{
    if (devAddr == 0 || devSize == 0) {
        return nullptr;
    }
    return std::shared_ptr<DevBuffer>(new DevBuffer(devAddr, devSize));
}
DevBuffer::DevBuffer(std::size_t allocSize, std::uint32_t policy, PolicyTag /*tag*/)
    : Buffer(allocSize),
      selfOwned(true)
{
    if (allocSize == 0) {
        std::string msg = "allocaSize should not be 0!";
        THROW<InternalException>(msg);
    }
    addr_ = ReinterpretAs<uintptr_t>(
        HrtMalloc(allocSize, static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH) | static_cast<int>(policy)));
}

std::shared_ptr<DevBuffer> DevBuffer::CreateHugePageBuf(std::size_t size)
{
    return std::make_shared<DevBuffer>(size, static_cast<int>(ACL_MEM_MALLOC_HUGE_ONLY), PolicyTag{});
}

DevBuffer::~DevBuffer()
{
    if (selfOwned) {
        DECTOR_TRY_CATCH("Buffer", HrtFree(ReinterpretAs<void*>(addr_)))
    }
}

std::string DevBuffer::Describe() const
{
    return StringFormat("DevBuffer[addr=0x%llx, size=0x%llx, selfOwned=%d]", addr_, size_, selfOwned);
}

bool DevBuffer::GetSelfOwned() const { return selfOwned; }

} // namespace Hccl
