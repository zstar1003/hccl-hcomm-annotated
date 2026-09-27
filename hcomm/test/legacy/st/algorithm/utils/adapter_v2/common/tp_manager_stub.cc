/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tp_manager.h"

#include "hccp_ctx.h"
#include "hccl_common_v2.h"
#include "orion_adapter_rts.h"
#include "rdma_handle_manager.h"

namespace Hccl {

void TpManager::Init() {}

TpManager& TpManager::GetInstance(const int32_t deviceLogicId)
{
    static TpManager tpManager[MAX_MODULE_DEVICE_NUM];

    if (deviceLogicId < 0 || static_cast<uint32_t>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) {
        THROW<InvalidParamsException>(
            "[TpManager][%s] Failed to get instance. "
            "devLogicId[%d] should be less than %u.",
            __func__, deviceLogicId, MAX_MODULE_DEVICE_NUM);
    }

    return tpManager[deviceLogicId];
}

HcclResult TpManager::GetTpInfo(const RaUbGetTpInfoParam& param, TpInfo& tpInfo, bool isSync)
{
    TpInfo info;
    info.tpHandle = 1;
    info.mappedJettyPriority = static_cast<uint32_t>(param.qos & 0xFU);
    info.hasMappedJettyPriority = true;
    tpInfo = info;
    return HCCL_SUCCESS;
}

HcclResult TpManager::ReleaseTpInfo(const RaUbGetTpInfoParam& param, const TpInfo& tpInfo)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::GetTpAttr(const GetTpAttrParam& param, TpAttrInfo& tpAttrInfo, RdmaHandle rdmaHandle)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::ReleaseTpAttr(const TpHandle tpHandle, const TpAttrInfo& tpAttrInfo)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::GetTpTotalTimeout(const TpAttrInfo& tpAttrInfo, uint32_t& tpTimeOutMs)
{
    (void)tpAttrInfo;
    tpTimeOutMs = 0;
    return HcclResult::HCCL_SUCCESS;
}

uint8_t TpManager::CalcTaTimeout(TpProtocol tpProtocol, uint8_t taTimeOut, uint32_t tpTimeOutMs)
{
    (void)tpTimeOutMs;
    if (taTimeOut != TA_TIMEOUT_NOT_SET) {
        return taTimeOut;
    }
    return (tpProtocol == TpProtocol::CTP) ? 8U : 16U;
}

} // namespace Hccl
