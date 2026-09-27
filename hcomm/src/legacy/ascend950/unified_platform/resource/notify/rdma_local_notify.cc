/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rdma_local_notify.h"
#include "not_support_exception.h"
#include "dev_capability.h"
#include "hccp.h"
#include "exchange_rdma_buffer_dto.h"

namespace Hccl {

RdmaLocalNotify::RdmaLocalNotify(RdmaHandle rdmaHandle, bool devUsed)
    : BaseLocalNotify(RmaType::RDMA, devUsed),
      rdmaHandle(rdmaHandle)
{
    auto devType = HrtGetDeviceType(); // 先查询，避免后续失败资源泄露
    HrtDevResInfo devResInfo;
    devResInfo.dieId = 0;
    devResInfo.procType = HrtDevResProcType::PROCESS_CP1;
    devResInfo.resType = HrtDevResType::RES_TYPE_STARS_NOTIFY_RECORD;
    devResInfo.resId = GetNotify()->GetId();
    devResInfo.flag = HRT_DEV_RES_FLAG_USE_UNIQUE_VA;
    auto resAddrInfo = HrtGetDevResAddress(devResInfo);
    addr = resAddrInfo.address;
    DevCapability::GetInstance().Init(devType); // 单例初始化
    size = DevCapability::GetInstance().GetNotifySize();
    HCCL_DEBUG("[RdmaLocalNotify] addr=0x%llx, size=%u", addr, size);
    // 注册内存
    struct MrInfoT mrInfo;
    mrInfo.addr = reinterpret_cast<void*>(addr);
    mrInfo.size = size;
    mrInfo.access = RA_ACCESS_REMOTE_WRITE | RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_READ;
    s32 ret = RaRegisterMr(rdmaHandle, &mrInfo, &mrHandle);
    if (ret != 0 || mrHandle == nullptr) {
        HCCL_ERROR("[RdmaLocalNotify] RaRegisterMr failed, call interface error[%d] mrHandle[%p]", ret, mrHandle);
        THROW<InternalException>("[%s] failed, call interface error[%d].", __func__, ret);
    }
    lkey = mrInfo.lkey;
    rkey = mrInfo.rkey;
}

RdmaLocalNotify::~RdmaLocalNotify()
{
    if (mrHandle) {
        s32 ret = RaDeregisterMr(rdmaHandle, mrHandle);
        if (ret != 0) {
            HCCL_ERROR(
                "[~RdmaLocalNotify]errNo[0x%016llx] RaDeregisterMr failed, return[%d]", HCCL_ERROR_CODE(HCCL_E_NETWORK),
                ret);
        }
        mrHandle = nullptr;
    }

    HrtDevResInfo devResInfo;
    devResInfo.dieId = 0;
    devResInfo.procType = HrtDevResProcType::PROCESS_CP1;
    devResInfo.resType = HrtDevResType::RES_TYPE_STARS_NOTIFY_RECORD;
    devResInfo.resId = GetNotify()->GetId();
    devResInfo.flag = HRT_DEV_RES_FLAG_USE_UNIQUE_VA;
    HrtReleaseDevResAddress(devResInfo);
}

void RdmaLocalNotify::Wait(const Stream& stream, u32 timeout) const { GetNotify()->Wait(stream, timeout); }

void RdmaLocalNotify::Post([[maybe_unused]] const Stream& stream) const
{
    HCCL_ERROR("RdmaLocalNotify does not support submit record task");
    throw NotSupportException("RdmaLocalNotify does not support submit record task");
}

string RdmaLocalNotify::Describe() const
{
    return StringFormat(
        "RdmaLocalNotify[notify=%s, addr=0x%llx, size=%u]", GetNotify()->Describe().c_str(), addr, size);
}

std::unique_ptr<Serializable> RdmaLocalNotify::GetExchangeDto()
{
    std::unique_ptr<ExchangeRdmaBufferDto> dto = make_unique<ExchangeRdmaBufferDto>(addr, size, rkey, "RdmaNotify");
    return std::unique_ptr<Serializable>(dto.release());
}

} // namespace Hccl
