/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_dfx.h"

#include "hccl_common_v2.h"
#include "exception_util.h"
#include "ccu_error_handler.h"

namespace Hccl {
using namespace std;

HcclResult GetCcuErrorMsg(
    s32 deviceId, uint16_t status, const ParaCcu& ccuTaskParam, const std::string& groupRankContent,
    std::vector<CcuErrorInfo>& errorInfo)
{
    TRY_CATCH_RETURN(
        HCCL_RUN_INFO(
            "[CcuDfx]GetCcuErrorMsg: deviceId[%d], dieId[%u], missionId[%u], execMissionId[%u], executeId[%llu].",
            deviceId, static_cast<u32>(ccuTaskParam.dieId), static_cast<u32>(ccuTaskParam.missionId),
            static_cast<u32>(ccuTaskParam.execMissionId), ccuTaskParam.executeId);

        // 入参校验
        CHK_PRT_RET(
            (deviceId < 0 || static_cast<u32>(deviceId) >= MAX_MODULE_DEVICE_NUM),
            HCCL_ERROR(
                "[CcuDfx][GetCcuErrorMsg]deviceId[%d] error, MAX_MODULE_DEVICE_NUM[%u].", deviceId,
                MAX_MODULE_DEVICE_NUM),
            HcclResult::HCCL_E_PARA);

        CcuErrorHandler::GetCcuErrorMsg(deviceId, status, ccuTaskParam, groupRankContent, errorInfo););
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GetCcuJettys(s32 deviceLogicId, const ParaCcu& ccuTaskParam, std::vector<CcuJetty*>& ccuJettys)
{
    TRY_CATCH_RETURN(CcuErrorHandler::GetCcuJettys(deviceLogicId, ccuTaskParam, ccuJettys););
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
