/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_local_notify.h"

#include "dev_capability.h"
#include "not_support_exception.h"
#include "exchange_ub_buffer_dto.h"
#include "rdma_handle_manager.h"
#include "local_ub_rma_buffer.h"

namespace Hccl {

UbLocalNotify::UbLocalNotify(RdmaHandle rdmaHandle, bool devUsed)
    : BaseLocalNotify(RmaType::UB, devUsed),
      rdmaHandle(rdmaHandle)
{
    auto devType = HrtGetDeviceType(); // 先查询，避免后续失败资源泄露
    HrtDevResInfo devResInfo;
    devResInfo.dieId = 0;
    devResInfo.procType = HrtDevResProcType::PROCESS_HCCP;
    devResInfo.resType = HrtDevResType::RES_TYPE_STARS_NOTIFY_RECORD;
    devResInfo.resId = GetNotify()->GetId();
    devResInfo.flag = 0;
    auto resAddrInfo = HrtGetDevResAddress(devResInfo);
    addr = resAddrInfo.address;
    DevCapability::GetInstance().Init(devType); // 单例初始化
    size = DevCapability::GetInstance().GetNotifySize();
    std::pair<u64, u64> alignBuf = BufAlign(addr, size);
    bufKey_ = BufferKey<uintptr_t, u64>{alignBuf.first, alignBuf.second};
    auto tokenIdInfoPair = RdmaHandleManager::GetInstance().GetTokenIdInfo(rdmaHandle, bufKey_);
    tokenIdHandle_ = tokenIdInfoPair.first;
    tokenId = tokenIdInfoPair.second;
    HCCL_INFO("[UbLocalNotify] tokenIdHandle=0x[%llx]", tokenIdHandle_);
    HCCL_INFO("mapped addr=[%llx]", addr);
    HCCL_INFO("UB notify size=[%u]", size);

    // halNotifyMap 返回的地址不保证4K对齐，
    // notify的地址还是使用hal接口返回的addr，但是注册mem的时候我们需要自己做向下对齐
    tokenValue = GetUbToken();
    HrtRaUbLocMemRegParam lmemReg{alignBuf.first, alignBuf.second, tokenValue, tokenIdHandle_, 1};
    reqReg = HrtRaUbLocalMemReg(rdmaHandle, lmemReg);
    keySize = reqReg.keySize;
    memHandle = reqReg.handle;
    (void)memcpy_s(key, HRT_UB_MEM_KEY_MAX_LEN, reqReg.key, HRT_UB_MEM_KEY_MAX_LEN);
}

string UbLocalNotify::Describe() const
{
    return StringFormat(
        "UbLocalNotify:notify=%s, addr=0x%llx, keySize=%u, memHandle=0x%llx", GetNotify()->Describe().c_str(), addr,
        keySize, memHandle);
}

void UbLocalNotify::Wait(const Stream& stream, u32 timeout) const { GetNotify()->Wait(stream, timeout); }

void UbLocalNotify::Post([[maybe_unused]] const Stream& stream) const
{
    std::string msg = "UbLocalNotify does not support submitting record task";
    MACRO_THROW(NotSupportException, msg);
}

std::unique_ptr<Serializable> UbLocalNotify::GetExchangeDto()
{
    std::unique_ptr<ExchangeUbBufferDto> dto
        = make_unique<ExchangeUbBufferDto>(addr, size, tokenValue, tokenId, keySize, GetNotify()->GetId());
    (void)memcpy_s(dto->key, HRT_UB_MEM_KEY_MAX_LEN, key, HRT_UB_MEM_KEY_MAX_LEN);
    return std::unique_ptr<Serializable>(dto.release());
}

void UbLocalNotify::ReleaseResource()
{
    const bool ctxValid = rdmaHandle != nullptr && RdmaHandleManager::GetInstance().IsHandleValid(rdmaHandle);

    if (rdmaHandle && memHandle != 0) {
        if (!ctxValid) {
            HCCL_WARNING(
                "[UbLocalNotify][%s] skip HrtRaUbLocalMemUnreg, "
                "rdmaHandle=%p invalid (DeInit/DestroyAll done), memHandle=0x%llx",
                __func__, rdmaHandle, static_cast<unsigned long long>(memHandle));
            memHandle = 0;
        } else {
            HrtRaUbLocalMemUnreg(rdmaHandle, memHandle);
            memHandle = 0;
        }
    }

    if (ctxValid) {
        RdmaHandleManager::GetInstance().PutTokenIdInfo(rdmaHandle, bufKey_, tokenIdHandle_);
    } else if (rdmaHandle != nullptr) {
        HCCL_WARNING("[UbLocalNotify][%s] skip PutTokenIdInfo, rdmaHandle=%p invalid", __func__, rdmaHandle);
    }

    HrtDevResInfo devResInfo;
    devResInfo.dieId = 0;
    devResInfo.procType = HrtDevResProcType::PROCESS_HCCP;
    devResInfo.resType = HrtDevResType::RES_TYPE_STARS_NOTIFY_RECORD;
    devResInfo.resId = GetNotify()->GetId();
    devResInfo.flag = 0;
    HrtReleaseDevResAddress(devResInfo);
}

UbLocalNotify::~UbLocalNotify() { DECTOR_TRY_CATCH("UbLocalNotify", ReleaseResource()); }
} // namespace Hccl
