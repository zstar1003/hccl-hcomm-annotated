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
#define HCCL_VM_MODULE "ACLRT_STUB"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "sim_log.h"
#include "runtime/base.h"
#include "db_sim_runner_ops.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

aclError aclsysGetVersionStr(char* pkgName, char* versionStr)
{
    (void)pkgName;
    memcpy(versionStr, "9.0.0", sizeof("9.0.0"));
    HCCL_VM_DEBUG("get version:{}", versionStr);
    return ACL_SUCCESS;
}

rtError_t rtEnableP2P(uint32_t devIdDes, uint32_t phyIdSrc, uint32_t flag)
{
    (void)devIdDes;
    (void)phyIdSrc;
    (void)flag;
    return ACL_SUCCESS;
}

rtError_t rtDisableP2P(uint32_t devIdDes, uint32_t phyIdSrc)
{
    (void)devIdDes;
    (void)phyIdSrc;
    return ACL_SUCCESS;
}
#ifdef __cplusplus
}
#endif // __cplusplus
