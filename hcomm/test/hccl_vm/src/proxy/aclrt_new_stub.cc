/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "NEW_STUB"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "ascend_hal.h"
#include "atrace_types.h"
#include "dtype_common.h"
#include "sim_log.h"
#include "hccp_common.h"
#include "rts_device.h"
#include "sim_models.h"
#include "db_sim_runner_common.h"
#include "db_sim_runner_ops.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

HcclResult hrtGetDeviceType(DevType& devType)
{
    auto devKey = sim::GetCurrDeviceKey();
    auto device = RunnerDB::GetById<sim::Device>(devKey);
    if (!device.has_value()) {
        // not find
        HCCL_VM_ERROR("current device type not found: {:d}", devKey);
        return HCCL_E_NOT_FOUND;
    }
    if (strcmp(device->soc_version, "Ascend950") == 0) {
        devType = DevType::DEV_TYPE_950;
#ifdef BUILD_A6_CCU_INSTR
    } else if (strcmp(device->soc_version, "Ascend960") == 0) {
        devType = DevType::DEV_TYPE_960;
#endif
    } else {
        HCCL_VM_ERROR("not support device soc version: {:s}", device->soc_version);
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_VM_TRACE("Get current device type: {}", static_cast<int>(devType));
    return HCCL_SUCCESS;
}

rtError_t rtOpenNetService(const rtNetServiceOpenArgs* args)
{
    (void)args;
    // hccpThreadStatus = 1;
    return ACL_SUCCESS;
}

rtError_t rtGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    (void)deviceId;
    (void)moduleType;
    (void)infoType;
    (void)value;
    return RT_ERROR_NONE;
}

rtError_t rtUbDevQueryInfo(rtUbDevQueryCmd cmd, void* devInfo)
{
    (void)cmd;
    (void)devInfo;
    return RT_ERROR_NONE;
}

rtError_t rtCntNotifyCreateServer(rtCntNotify_t* const cntNotify, uint64_t flags)
{
    (void)cntNotify;
    (void)flags;
    return 0;
}

rtError_t rtsCntNotifyGetId(rtCntNotify_t cntNotify, uint32_t* notifyId)
{
    (void)cntNotify;
    (void)notifyId;
    return ACL_SUCCESS;
}

rtError_t rtGetDevResAddress(rtDevResInfo* const resInfo, rtDevResAddrInfo* const addrInfo)
{
    if (resInfo == nullptr
        || (resInfo->resType != rtDevResType_t::RT_RES_TYPE_STARS_NOTIFY_RECORD
            && resInfo->resType != rtDevResType_t::RT_RES_TYPE_STARS_CNT_NOTIFY_BIT_WR)) {
        // 非NotifyRecord场景暂不处理
        return RT_ERROR_NONE;
    }

    uint32_t notifyId = resInfo->resId;
    uint32_t len = 8;
    *(addrInfo->resAddress) = static_cast<uint64_t>(notifyId);
    *(addrInfo->len) = len;
    return RT_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif // __cplusplus
