/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "remote_rdma_rma_buffer.h"
#include "remote_rdma_rma_buffer_impl.h"

namespace hccl {
RemoteRdmaRmaBuffer::RemoteRdmaRmaBuffer() : RmaBuffer(nullptr, nullptr, 0, RmaMemType::TYPE_NUM, RmaType::RDMA_RMA)
{
    pimpl_ = std::make_unique<RemoteRdmaRmaBufferImpl>();
}

RemoteRdmaRmaBuffer::~RemoteRdmaRmaBuffer()
{
    if (pimpl_ != nullptr) {
        pimpl_ = nullptr;
        addr = nullptr;
        size = 0;
        devAddr = nullptr;
    }
}

HcclResult RemoteRdmaRmaBuffer::Deserialize(const std::string& msg)
{
    std::istringstream iss(msg);
    u8 type{static_cast<u8>(RmaType::RMA_TYPE_RESERVED)};
    iss.read(reinterpret_cast<char_t*>(&type), sizeof(type));
    iss.read(reinterpret_cast<char_t*>(&addr), sizeof(addr));
    iss.read(reinterpret_cast<char_t*>(&size), sizeof(size));
    iss.read(reinterpret_cast<char_t*>(&devAddr), sizeof(devAddr));
    iss.read(reinterpret_cast<char_t*>(&memType), sizeof(memType));
    CHK_PTR_NULL(addr);
    CHK_PTR_NULL(devAddr);
    CHK_PRT_RET(
        (memType >= RmaMemType::TYPE_NUM),
        HCCL_ERROR("[RemoteRdmaRmaBuffer][Deserialize]RmaMemType[%d] is invalid.", static_cast<int>(memType)),
        HCCL_E_PARA);
    CHK_PRT_RET(
        type >= static_cast<u8>(RmaType::RMA_TYPE_RESERVED),
        HCCL_ERROR("[RemoteRdmaRmaBuffer][Deserialize]rmaType[%d] is invalid.", type), HCCL_E_PARA);
    CHK_PRT_RET(
        (size == 0 || (memType == RmaMemType::HOST && size >= HOST_MEM_MAX_COUNT)
         || (memType == RmaMemType::DEVICE && size >= DEVICE_MEM_MAX_COUNT)),
        HCCL_ERROR(
            "[RemoteRdmaRmaBuffer][Deserialize]memory size[%llu] should be greater than 0 and less than [%llu].", size,
            (memType == RmaMemType::DEVICE ? DEVICE_MEM_MAX_COUNT : HOST_MEM_MAX_COUNT)),
        HCCL_E_PARA);

    HCCL_DEBUG(
        "[RemoteRdmaRmaBuffer][Deserialize]addr[%p], size[%llu], devAddr[%p], memType[%d]", addr, size, devAddr,
        memType);

    CHK_SMART_PTR_NULL(pimpl_);
    if (rmaType != static_cast<RmaType>(type)) {
        HCCL_ERROR("[RemoteRdmaRmaBuffer][Deserialize]rmaType[%d] is not match to [%d].", type, rmaType);
        return HCCL_E_INTERNAL;
    }

    std::string remainingMsg = std::string((std::istreambuf_iterator<char>(iss)), std::istreambuf_iterator<char>());
    HcclResult ret = pimpl_->Deserialize(remainingMsg);
    if (ret != HCCL_SUCCESS) {
        pimpl_ = nullptr;
        HCCL_ERROR("[RemoteRdmaRmaBuffer]Deserialize failed, ret[%d]", ret);
        return ret;
    }
    return HCCL_SUCCESS;
}

u32 RemoteRdmaRmaBuffer::GetKey() const { return pimpl_->GetKey(); }
} // namespace hccl
