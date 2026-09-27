/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "local_rdma_rma_buffer_v2.h"
#include "hccp.h"
#include "exchange_rdma_buffer_dto.h"

namespace Hccl {

LocalRdmaRmaBuffer::LocalRdmaRmaBuffer(std::shared_ptr<Buffer> buf, RdmaHandle rdmaHandle)
    : LocalRmaBuffer(buf, RmaType::RDMA),
      rdmaHandle(rdmaHandle)
{
    if (rdmaHandle == nullptr || buf == nullptr) {
        string nullParam = rdmaHandle == nullptr ? "rdmaHandle" : "buf";
        THROW<NullPtrException>("LocalRdmaRmaBuffer's %s is nullptr", nullParam.c_str());
    }
    const uintptr_t bufAddr = buf->GetAddr();
    size_t bufSize = buf->GetSize();
    if (bufAddr == 0 || bufSize <= 0) {
        HCCL_ERROR(
            "[LocalRdmaRmaBuffer]buffer size[%zu Byte] and addr[%zu] should be greater than 0.", bufAddr, bufSize);
        THROW<InvalidParamsException>("[%s] failed, param error.", __func__);
    }
    // 注册内存
    struct MrInfoT mrInfo;
    mrInfo.addr = reinterpret_cast<void*>(bufAddr);
    mrInfo.size = bufSize;
    mrInfo.access = RA_ACCESS_REMOTE_WRITE | RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_READ | RA_ACCESS_REMOTE_ATOMIC;

    s32 ret = RaRegisterMr(rdmaHandle, &mrInfo, &mrHandle);
    if (ret != 0 || mrHandle == nullptr) {
        HCCL_ERROR("[HrtRaRegisterMr] RaRegisterMr failed, call interface error[%d]", ret);
        THROW<InternalException>("[%s] failed, call interface error[%d].", __func__, ret);
    }
    lkey = mrInfo.lkey;
    rkey = mrInfo.rkey;
    HCCL_INFO(
        "LocalRdmaRmaBuffer[rdmaHandle=%p, mrHandle = %p, buf=%s]", rdmaHandle, mrHandle, buf->Describe().c_str());
}

LocalRdmaRmaBuffer::LocalRdmaRmaBuffer(
    std::shared_ptr<Buffer> buf, RdmaHandle rdmaHandle, u32 lkey, u32 rkey, MrHandle mrHandle)
    : LocalRmaBuffer(buf, RmaType::RDMA, true),
      rdmaHandle(rdmaHandle),
      lkey(lkey),
      rkey(rkey),
      mrHandle(mrHandle)
{
    if (rdmaHandle == nullptr || buf == nullptr) {
        string nullParam = rdmaHandle == nullptr ? "rdmaHandle" : "buf";
        THROW<NullPtrException>("LocalRdmaRmaBuffer alias: %s is nullptr", nullParam.c_str());
    }
    const uintptr_t bufAddr = buf->GetAddr();
    size_t bufSize = buf->GetSize();
    if (bufAddr == 0 || bufSize <= 0) {
        HCCL_ERROR(
            "[LocalRdmaRmaBuffer] alias buffer addr[%p] and size[%llu Byte] should be greater than 0.",
            reinterpret_cast<void*>(bufAddr), static_cast<unsigned long long>(bufSize));
        THROW<InvalidParamsException>("[%s] alias failed, param error.", __func__);
    }
    if (mrHandle == nullptr) {
        THROW<NullPtrException>("LocalRdmaRmaBuffer alias: mrHandle is nullptr");
    }
    HCCL_INFO(
        "LocalRdmaRmaBuffer alias[rdmaHandle=%p, mrHandle=%p, lkey=%u, buf=%s]", rdmaHandle, mrHandle, lkey,
        buf->Describe().c_str());
}

LocalRdmaRmaBuffer::~LocalRdmaRmaBuffer()
{
    if (mrHandle && !isAlias_) {
        s32 ret = RaDeregisterMr(rdmaHandle, mrHandle);
        if (ret != 0) {
            HCCL_ERROR(
                "[HrtRaDeRegisterMr]errNo[0x%016llx] RaDeregisterMr failed, return[%d]",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), ret);
            // THROW<InternalException>("[%s] failed, call interface error[%d].", __func__, ret);
        }
        mrHandle = nullptr;
    }
}

string LocalRdmaRmaBuffer::Describe() const
{
    return StringFormat(
        "LocalRdmaRmaBuffer[rdmaHandle=%p, mrHandle = %p, buf=%s]", rdmaHandle, mrHandle, buf->Describe().c_str());
}

std::unique_ptr<Serializable> LocalRdmaRmaBuffer::GetExchangeDto()
{
    std::unique_ptr<ExchangeRdmaBufferDto> dto
        = make_unique<ExchangeRdmaBufferDto>(buf->GetAddr(), buf->GetSize(), this->rkey, buf->GetMemInfo().c_str());
    return std::unique_ptr<Serializable>(dto.release());
}

} // namespace Hccl
