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
#define HCCL_VM_MODULE "EXEC_STUB"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "sim_log.h"
#include "db_sim_runner_ops.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

aclError aclrtLaunchCallback(aclrtCallback fn, void* userData, aclrtCallbackBlockType blockType, aclrtStream stream)
{
    (void)fn;
    (void)userData;
    (void)blockType;
    (void)stream;
    HCCL_VM_TRACE("not supported");
    return ACL_SUCCESS;
}

aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    (void)threadId;
    (void)stream;
    HCCL_VM_TRACE("not supported");
    return ACL_SUCCESS;
}

aclError aclrtProcessReport(int32_t timeout)
{
    (void)timeout;
    HCCL_VM_TRACE("not supported");
    return ACL_SUCCESS;
}

aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    (void)threadId;
    (void)stream;
    HCCL_VM_TRACE("not supported");
    return ACL_SUCCESS;
}

aclError aclrtSubscribeHostFunc(uint64_t hostFuncThreadId, aclrtStream exeStream)
{
    (void)hostFuncThreadId;
    (void)exeStream;
    HCCL_VM_TRACE("not supported");
    return ACL_SUCCESS;
}

aclError aclrtProcessHostFunc(int32_t timeout)
{
    (void)timeout;
    HCCL_VM_TRACE("not supported");
    return ACL_SUCCESS;
}

aclError aclrtUnSubscribeHostFunc(uint64_t hostFuncThreadId, aclrtStream exeStream)
{
    (void)hostFuncThreadId;
    (void)exeStream;
    HCCL_VM_TRACE("not supported");
    return ACL_SUCCESS;
}

aclError aclrtGetOpTimeoutInterval(uint64_t* interval)
{
    (void)interval;
    HCCL_VM_TRACE("not supported");
    return ACL_SUCCESS;
}

aclError aclrtSetOpExecuteTimeOut(uint32_t timeout)
{
    (void)timeout;
    HCCL_VM_TRACE("not supported");
    return ACL_SUCCESS;
}

aclError aclrtSetOpExecuteTimeOutV2(uint64_t timeout, uint64_t* actualTimeout)
{
    (void)timeout;
    (void)actualTimeout;
    HCCL_VM_TRACE("not supported");
    return ACL_SUCCESS;
}

aclError aclrtSetOpExecuteTimeOutWithMs(uint32_t timeout)
{
    (void)timeout;
    HCCL_VM_TRACE("not supported");
    return ACL_SUCCESS;
}

aclError aclrtGetThreadLastTaskId(uint32_t* taskId)
{
    (void)taskId;
    HCCL_VM_TRACE("not supported");
    return ACL_SUCCESS;
}

aclError aclrtLaunchHostFunc(aclrtStream stream, aclrtHostFunc fn, void* args)
{
    (void)stream;
    (void)fn;
    (void)args;
    HCCL_VM_TRACE("not supported");
    return ACL_SUCCESS;
}
#ifdef __cplusplus
}
#endif // __cplusplus
