/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KFC_PROCESS_H
#define AICPU_KFC_PROCESS_H

#include "common/aicpu_hccl_def.h"
#include "hccl_tiling_msg.h"
#include "aicpu_kfc_rpc_server.h"

class AicpuKfcProcess {
public:
    ~AicpuKfcProcess() = default;
    static u32 AicpuRpcResInit(HccCommResParamTask* commParam);
    static u32 GetStreamRankIdx(s32 actualStreamId);
    static HcclResult DealReturnValue(const AicpuComContext* ctx, const HcclResult ret);
    static HcclResult AddTaskForHcclMsg(
        AicpuComContext* ctx, AicpuKfcRpcServer& rpc, CommonHcclMsg* hcclMsg, AivAicpuOpParam* msg, u64 tilingBase);
    static HcclResult RunRpcServerApi(AicpuComContext* ctx, AicpuKfcRpcServer& rpc, u64 tilingBase = 0UL);
    static HcclResult AicpuRunRpcServerForApi(AicpuComContext* ctx, u64 tilingBase);
    static u32 AicpuRunRpcServerForMC2V2(KFCTaskV2* task, const HcclApi::Mc2InitTilingInner* tilingData);
    static u32 AicpuRunRpcServerForMC2(KFCTaskV2* task);

private:
    friend class AicpuKfcDeprecatedProcess;
    static HcclResult AicpuCcOpExe(AivAicpuOpParam* commParam, AivAicpuOpParam* commParamNext, AicpuComContext* ctx);
    static HcclResult WaitTaskFinish(AicpuComContext* ctx, bool isWaitTask = true);
    static HcclResult ResetSqBuff(AicpuComContext* ctx);
    static u32 GetActiveSqId(AicpuComContext* ctx);
    static HcclResult InitStreamInfo(HccCommResParamTask* commParam, AicpuComContext* ctx);
};

#endif
