/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KFC_DEPRECATED_PROCESS_H
#define AICPU_KFC_DEPRECATED_PROCESS_H

#include "common/aicpu_hccl_def.h"
#include "aicpu_kfc_rpc_server.h"

class AicpuKfcDeprecatedProcess {
public:
    ~AicpuKfcDeprecatedProcess() = default;
    static HcclResult AICPU_RpcServerUnfoldStageWait(AicpuComContext* ctx, AicpuKfcRpcServer& rpc);
    static HcclResult RunRpcServerTwoStageWait(AicpuComContext* ctx, AicpuKfcRpcServer& rpc);
    static HcclResult TryRunRpcServerOneStageWait(AicpuComContext* ctx, AicpuKfcRpcServer& rpc);

private:
    static HcclResult
    LaunchHcclOp(AicpuComContext* ctx, AivAicpuOpParam* commParam, uint32_t& beginSqePos, uint32_t& endSqePos);
    static HcclResult RetryLaunchHcclOp(AicpuComContext* ctx, AivAicpuOpParam* commParam, uint32_t& endSqePos);
    static HcclResult RunRpcServerOneStageWait(AicpuComContext* ctx, AicpuKfcRpcServer& rpc);
    static HcclResult
    HcclOpExecFsmWaitEndProcess(AicpuComContext* ctx, HcclOpExecFSM& state, KfcError& errorCode, u32 retryCnt);
    static HcclResult HcclOpExecFsmWaitRetryProcess(AicpuComContext* ctx, HcclOpExecFSM& state, KfcError& errorCode);
    static HcclResult HcclOpExecFsmLaunchProcess(
        AicpuComContext* ctx, HcclOpExecFSM& state, KfcError& errorCode, AivAicpuOpParam& opParams,
        uint32_t& beginSqePos, uint32_t& endSqePos);
    static HcclResult HcclOpExecFsmRetryProcess(
        AicpuComContext* ctx, HcclOpExecFSM& state, KfcError& errorCode, uint32_t& retryCnt, AivAicpuOpParam& opParams,
        uint32_t& endSqePos);
};

#endif
