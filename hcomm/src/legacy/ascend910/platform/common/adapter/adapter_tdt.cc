/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <adapter_tdt.h>
#include "log.h"
#include "workflow_pub.h"
#include "driver/ascend_hal.h"
#include "sal_pub.h"
using namespace hccl;

#ifdef __cplusplus
extern "C" {
#endif
HcclResult hrtOpenTsd()
{
#ifndef HCCD
    std::string extPam("--hdcType=" + std::to_string(HDC_SERVICE_TYPE_RDMA));
    rtNetServiceOpenArgs openArgs = {};
    rtProcExtParam extParam{};
    extParam.paramInfo = extPam.c_str();
    extParam.paramLen = extPam.size();
    openArgs.extParamCnt = 1UL;
    openArgs.extParamList = &extParam;
    CHK_RET(hrtOpenNetService(&openArgs));
    HCCL_INFO("hrtOpenTsd success");
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtOpenTsd]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult __hrtOpenNetService([[maybe_unused]] rtNetServiceOpenArgs* openArgs)
{
#if !defined(CCL_KERNEL) && !defined(HCCD)
    aclError aclret = rtOpenNetService(openArgs);
    if (aclret != ACL_RT_SUCCESS) {
        HCCL_ERROR("[hrtOpenNetService]Open NetService failed, err code : %d", aclret);
        return HcclResult::HCCL_E_UNAVAIL;
    }

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtOpenNetService]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
weak_alias(__hrtOpenNetService, hrtOpenNetService);

HcclResult __hrtCloseNetService()
{
#if !defined(CCL_KERNEL) && !defined(HCCD)
    aclError aclret = rtCloseNetService();
    if (aclret != ACL_RT_SUCCESS) {
        HCCL_ERROR("[hrtCloseNetService]Open NetService failed, err code : %d", aclret);
        return HcclResult::HCCL_E_UNAVAIL;
    }

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtCloseNetService]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
weak_alias(__hrtCloseNetService, hrtCloseNetService);

HcclResult hrtTsdCapabilityGet(
    [[maybe_unused]] uint32_t deviceLogicId, [[maybe_unused]] int32_t type, [[maybe_unused]] uint64_t ptr)
{
#ifndef HCCD
    uint32_t tdtStatus = DlTdtFunction::GetInstance().dlTsdCapabilityGet(deviceLogicId, type, ptr);
    if (tdtStatus != 0) {
        HCCL_ERROR(
            "[Get][TsdCapability]Get TsdCapability failed, tdt error code: %u, error deviceLogicId[%u], ", tdtStatus,
            deviceLogicId);
        return HCCL_E_UNAVAIL;
    }
    HCCL_INFO("Get TsdCapability success. deviceLogicId[%u]", deviceLogicId);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtTsdCapabilityGet]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
#ifdef __cplusplus
} // extern "C"
#endif
