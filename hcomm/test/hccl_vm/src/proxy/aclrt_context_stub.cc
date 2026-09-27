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
#define HCCL_VM_MODULE "CTX_STUB"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "hccl/base.h"
#include "sim_common_macro.h"
#include "sim_log.h"
#include "db_sim_runner_ops.h"
#include "db_sim_runner_common.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
aclError aclrtCreateContext(aclrtContext* context, int32_t deviceId)
{
    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device& d) {
        return d.logic_id == (uint32_t)deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("device not found:{:d}", deviceId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    // 找不到context，则默认新增一个context
    sim::Context ctx{};
    ctx.device_id = ret.first.id;
    ctx.run_id = runner.id;
    auto currCtxId = RunnerDB::Add<sim::Context>(ctx);

    // 新建context，默认新建一个stream
    sim::Stream stream{};
    stream.ctx_id = currCtxId;
    stream.is_primary_default = 1;
    stream.activated = 1;
    RunnerDB::Add<sim::Stream>(stream);
    *context = (aclrtContext)currCtxId;
    return ACL_SUCCESS;
}

aclError aclrtDestroyContext(aclrtContext context)
{
    uint64_t ctxId = (uint64_t)(uintptr_t)context;

    auto ret = RunnerDB::GetOneByPred<sim::Stream>([ctxId](const sim::Stream& stm) {
        return stm.ctx_id == ctxId && stm.is_primary_default == 1;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("stream not found ctxId:{:d}", ctxId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    RunnerDB::Delete<sim::Stream>(ret.first.id);
    RunnerDB::Delete<sim::Context>(ctxId);
    return ACL_SUCCESS;
}

aclError aclrtSetCurrentContext(aclrtContext context)
{
    uint64_t ctxId = (uint64_t)(uintptr_t)context;
    sim::SetCurrCtxTls(ctxId);
    return ACL_SUCCESS;
}

aclError aclrtGetCurrentContext(aclrtContext* context)
{
    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *context = (aclrtContext)runner.current_ctx_id;
    return ACL_SUCCESS;
}

aclError aclrtCtxGetCurrentDefaultStream(aclrtStream* stream)
{
    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto curCtxId = runner.current_ctx_id;
    auto stm = RunnerDB::GetOneByPred<sim::Stream>([curCtxId](const sim::Stream& stm) {
        return stm.ctx_id == curCtxId && stm.is_primary_default == 1;
    });
    if (!stm.second) {
        HCCL_VM_ERROR("stream not found ctxId:{:d}", curCtxId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    *stream = (aclrtStream)stm.first.id;
    return ACL_SUCCESS;
}

aclError aclrtGetPrimaryCtxState(int32_t deviceId, uint32_t* flags, int32_t* active)
{
    (void)deviceId;
    (void)flags;
    auto serverId = sim::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::Runner runner;
    if (!sim::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("ctx not found:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    *active = (currCtx->is_default == 1) ? 1 : 0;
    return ACL_SUCCESS;
}

aclError aclrtCtxSetSysParamOpt(aclSysParamOpt opt, int64_t value)
{
    (void)opt;
    (void)value;
    return ACL_SUCCESS;
}

aclError aclrtCtxGetSysParamOpt(aclSysParamOpt opt, int64_t* value)
{
    (void)opt;
    (void)value;
    return ACL_SUCCESS;
}

aclError aclrtCtxGetFloatOverflowAddr(void** overflowAddr)
{
    (void)overflowAddr;
    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
