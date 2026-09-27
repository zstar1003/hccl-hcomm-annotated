/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "orion_adapter_tsd.h"
#include "log.h"
#include "rt_external_device.h"

namespace Hccl {

const std::string HDC_TYPE_EXT_PAM = "--hdcType=18";

HcclResult HrtOpenTsdProcess(u32 deviceLogicId)
{
    rtNetServiceOpenArgs openArgs;
    rtProcExtParam extParam{};
    extParam.paramInfo = HDC_TYPE_EXT_PAM.c_str();
    extParam.paramLen = HDC_TYPE_EXT_PAM.size();
    openArgs.extParamCnt = 1UL;
    openArgs.extParamList = &extParam;
    rtError_t aclret = rtOpenNetService(&openArgs);
    if (aclret != 0) {
        HCCL_INFO("deviceLogicId = %u TsdProcessOpen fail aclret:%d", deviceLogicId, aclret);
        return HcclResult::HCCL_E_UNAVAIL;
    } else {
        HCCL_INFO("deviceLogicId = %u TsdProcessOpen success", deviceLogicId);
        return HcclResult::HCCL_SUCCESS;
    }
}
} // namespace Hccl
