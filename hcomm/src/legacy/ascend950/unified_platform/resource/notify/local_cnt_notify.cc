/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "local_cnt_notify.h"

#include "exchange_ub_buffer_dto.h"
#include "rdma_handle_manager.h"
#include "local_ub_rma_buffer.h"

namespace Hccl {

LocalCntNotify::LocalCntNotify(RdmaHandle rdmaHandle, RtsCntNotify* notify)
    : rdmaHandle(rdmaHandle),
      notify(notify),
      tokenValue(GetUbToken()),
      addr(notify->GetAddr()),
      size(notify->GetSize())
{
    std::pair<u64, u64> alignBuf = BufAlign(addr, size);
    bufKey_ = BufferKey<uintptr_t, u64>{alignBuf.first, alignBuf.second};
    auto tokenIdInfoPair = RdmaHandleManager::GetInstance().GetTokenIdInfo(rdmaHandle, bufKey_);
    tokenIdHandle_ = tokenIdInfoPair.first;
    tokenId = tokenIdInfoPair.second;
    HCCL_INFO("[LocalCntNotify] tokenIdHandle=0x[%llx].", tokenIdHandle_);
    HrtRaUbLocMemRegParam memRegInput(alignBuf.first, alignBuf.second, tokenValue, tokenIdHandle_);
    reqReg = HrtRaUbLocalMemReg(rdmaHandle, memRegInput);
    keySize = reqReg.keySize;
    memHandle = reqReg.handle;
    (void)memcpy_s(key, HRT_UB_MEM_KEY_MAX_LEN, reqReg.key, HRT_UB_MEM_KEY_MAX_LEN);
}

std::unique_ptr<Serializable> LocalCntNotify::GetExchangeDto()
{
    std::unique_ptr<ExchangeUbBufferDto> dto
        = make_unique<ExchangeUbBufferDto>(addr, size, tokenValue, tokenId, keySize, notify->GetId());
    (void)memcpy_s(dto->key, HRT_UB_MEM_KEY_MAX_LEN, key, HRT_UB_MEM_KEY_MAX_LEN);
    return std::unique_ptr<Serializable>(dto.release());
}

std::string LocalCntNotify::Describe() const
{
    return StringFormat(
        "UbLocalNotify:notify=%s, addr=0x%llx, keySize=%u, memHandle=0x%llx", notify->Describe().c_str(), addr, keySize,
        memHandle);
}

LocalCntNotify::~LocalCntNotify()
{
    const bool ctxValid = rdmaHandle != nullptr && RdmaHandleManager::GetInstance().IsHandleValid(rdmaHandle);

    if (rdmaHandle && memHandle != 0) {
        if (!ctxValid) {
            HCCL_WARNING(
                "[LocalCntNotify][%s] skip HrtRaUbLocalMemUnreg, "
                "rdmaHandle=%p invalid (DeInit/DestroyAll done), memHandle=0x%llx",
                __func__, rdmaHandle, static_cast<unsigned long long>(memHandle));
        } else {
            DECTOR_TRY_CATCH("LocalCntNotify", HrtRaUbLocalMemUnreg(rdmaHandle, memHandle));
        }
        memHandle = 0;
    }

    if (ctxValid) {
        RdmaHandleManager::GetInstance().PutTokenIdInfo(rdmaHandle, bufKey_, tokenIdHandle_);
    } else if (rdmaHandle != nullptr) {
        HCCL_WARNING("[LocalCntNotify][%s] skip PutTokenIdInfo, rdmaHandle=%p invalid", __func__, rdmaHandle);
    }
    tokenIdHandle_ = 0; // 防御性置零
}
} // namespace Hccl
