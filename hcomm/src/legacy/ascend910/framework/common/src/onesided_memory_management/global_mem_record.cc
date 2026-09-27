/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "global_mem_record.h"
#include <sstream>

namespace hccl {
GlobalMemRecord::GlobalMemRecord(const HcclMem* mem)
    : type_(mem->type),
      addr_(mem->addr),
      size_(mem->size),
      pLock_(std::make_unique<std::mutex>())
{}

GlobalMemRecord::GlobalMemRecord(const HcclMem& mem)
    : type_(mem.type),
      addr_(mem.addr),
      size_(mem.size),
      pLock_(std::make_unique<std::mutex>())
{}

GlobalMemRecord::GlobalMemRecord(GlobalMemRecord&& other) noexcept
    : type_(other.type_),
      addr_(other.addr_),
      size_(other.size_),
      pLock_(std::move(other.pLock_)),
      boundComm_(std::move(other.boundComm_))
{}

bool GlobalMemRecord::HasOverlap(const GlobalMemRecord& other) const
{
    if (type_ != other.GetMemType()) {
        // 不同类型不判断
        return false;
    }

    const auto thisBegin = reinterpret_cast<uintptr_t>(addr_);
    const auto thisEnd = thisBegin + size_;
    const auto otherBegin = reinterpret_cast<uintptr_t>(other.GetAddr());
    const auto otherEnd = otherBegin + other.GetSize();

    return (thisBegin < otherEnd) && (otherBegin < thisEnd);
}

HcclResult GlobalMemRecord::BindToComm(const std::string& commIdentifier)
{
    std::unique_lock<std::mutex> lock(*pLock_);
    const auto insertRet = boundComm_.insert(commIdentifier);

    CHK_PRT_RET(
        insertRet.second == false,
        HCCL_ERROR(
            "[GlobalMemRecord][BindToComm] The mem[%s] has been bound to the comm[%s] already.", PrintInfo().c_str(),
            commIdentifier.c_str()),
        HCCL_E_PARA);

    HCCL_INFO(
        "[GlobalMemRecord][BindToComm] The mem[%s] is bound to the comm[%s].", PrintInfo().c_str(),
        commIdentifier.c_str());

    return HCCL_SUCCESS;
}

HcclResult GlobalMemRecord::UnbindFromComm(const std::string& commIdentifier)
{
    std::unique_lock<std::mutex> lock(*pLock_);
    const auto eraseCount = boundComm_.erase(commIdentifier);

    CHK_PRT_RET(
        eraseCount == 0,
        HCCL_ERROR(
            "[GlobalMemRecord][UnbindFromComm] The mem[%s] is not bound to the comm[%s].", PrintInfo().c_str(),
            commIdentifier.c_str()),
        HCCL_E_PARA);

    HCCL_INFO(
        "[GlobalMemRecord][UnbindFromComm] The mem[%s] has been unbound from the comm[%s].", PrintInfo().c_str(),
        commIdentifier.c_str());

    return HCCL_SUCCESS;
}

std::string GlobalMemRecord::PrintInfo() const
{
    std::stringstream ss;
    if (type_ == HCCL_MEM_TYPE_DEVICE) {
        ss << "type:DEVICE, ";
    } else if (type_ == HCCL_MEM_TYPE_HOST) {
        ss << "type:HOST, ";
    }

    ss << "addr:" << addr_ << ", ";
    ss << "size:" << size_;
    return ss.str();
}

} // namespace hccl
