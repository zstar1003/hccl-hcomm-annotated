/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "local_ipc_rma_buffer.h"
#include "local_ipc_rma_buffer_impl.h"

namespace hccl {
LocalIpcRmaBuffer::LocalIpcRmaBuffer(const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType)
    : RmaBuffer(netDevCtx, addr, size, memType, RmaType::IPC_RMA)
{
    pimpl_ = std::make_unique<LocalIpcRmaBufferImpl>(netDevCtx, addr, size, memType);
}

LocalIpcRmaBuffer::LocalIpcRmaBuffer(
    const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType, const LocalIpcRmaBuffer& parent)
    : RmaBuffer(netDevCtx, addr, size, memType, RmaType::IPC_RMA, true)
{
    pimpl_ = std::make_unique<LocalIpcRmaBufferImpl>(netDevCtx, addr, size, memType, *parent.pimpl_);
    this->devAddr = pimpl_->GetDevAddr();
    HCCL_INFO("[LocalIpcRmaBuffer] alias constructor");
}

LocalIpcRmaBuffer::~LocalIpcRmaBuffer()
{
    if (!isAlias_) {
        HcclResult res = Destroy();
        if (res != HCCL_SUCCESS) {
            HCCL_ERROR("[LocalIpcRmaBuffer][~LocalIpcRmaBuffer]failed, ret[%d]", res);
        }
    }
}

HcclResult LocalIpcRmaBuffer::Init()
{
    CHK_PTR_NULL(addr);
    CHK_PRT_RET(
        (memType >= RmaMemType::TYPE_NUM),
        HCCL_ERROR("[LocalIpcRmaBuffer][Init]RmaMemType[%d] is invalid.", static_cast<int>(memType)), HCCL_E_PARA);
    CHK_PRT_RET(
        (size == 0 || (memType == RmaMemType::HOST && size >= HOST_MEM_MAX_COUNT)
         || (memType == RmaMemType::DEVICE && size >= DEVICE_MEM_MAX_COUNT)),
        HCCL_ERROR(
            "[LocalIpcRmaBuffer][Init]memory size[%llu] should be greater than 0 and less than [%llu].", size,
            (memType == RmaMemType::DEVICE ? DEVICE_MEM_MAX_COUNT : HOST_MEM_MAX_COUNT)),
        HCCL_E_PARA);

    CHK_SMART_PTR_NULL(pimpl_);
    HcclResult ret = pimpl_->Init();
    if (ret != HCCL_SUCCESS) {
        pimpl_ = nullptr;
        HCCL_ERROR("[LocalIpcRmaBuffer][Init]Init failed, ret[%d]", ret);
        return ret;
    }

    this->devAddr = pimpl_->GetDevAddr();

    return HCCL_SUCCESS;
}

HcclResult LocalIpcRmaBuffer::Destroy()
{
    if (pimpl_ != nullptr) {
        HcclResult ret = pimpl_->Destroy();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[LocalIpcRmaBuffer][Destroy]Destroy failed, ret[%d]", ret);
        }
        pimpl_ = nullptr;
        addr = nullptr;
        size = 0;
        devAddr = nullptr;
        return ret;
    }
    return HCCL_SUCCESS;
}

std::string& LocalIpcRmaBuffer::Serialize() { return pimpl_->Serialize(); }

HcclResult LocalIpcRmaBuffer::Grant(u32 remotePid, u32 remoteSdid) { return pimpl_->Grant(remotePid, remoteSdid); }
} // namespace hccl
