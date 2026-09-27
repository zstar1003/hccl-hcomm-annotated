/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_kfc_interface.h"

#include "common/aicpu_hccl_common.h"
#include "common/aicpu_hccl_def.h"
#include "common/aicpu_sqe_context.h"
#include "dfx/mc2_trace_utils.h"
#include "dfx/aicpu_profiling_manager.h"
#include "framework/aicpu_hccl_process.h"
#include "aicpu_kfc/framework/aicpu_kfc_process.h"
#include "aicpu_kfc/decoupler/comm_kfc_dispatcher.h"
#include "aicpu_kfc/framework/aicpu_kfc_deprecated_process.h"
#include "aicpu_kfc/framework/aicpu_kfc_batchwrite_process.h"
#include "utils/hccl_aicpu_utils.h"
#include "common/aicpu_kfc_utils.h"
#include "aicpu_kfc/common/aicpu_kfc_tiling_utils.h"
#include "aicpu_kfc/framework/aicpu_kfc_prof.h"
#include "utils/aicpu_hdc_utils.h"

using namespace HcclApi;
namespace {
u64 GetTensorAddr(uint16_t index, uint8_t* tensorPtr)
{
    uint64_t* dataAddr = reinterpret_cast<uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    uint64_t* resPtr = dataAddr + (tensorPtrOffset >> 3);
    return u64(*(resPtr + index));
}

u64 GetUpdatedOpIdx()
{
    static uint64_t aicpuOpIdx[MAX_AICPU_NUM_BLOCKS] = {0UL};
    const u32 blockNum = HcclAicpuUtils::GetBlockNum();
    const u32 blockIdx = HcclAicpuUtils::GetBlockIdx();
    ++aicpuOpIdx[blockIdx];
    if (blockIdx == blockNum - 1U) {
        for (u32 i = blockIdx + 1U; i < MAX_AICPU_NUM_BLOCKS; ++i) {
            ++aicpuOpIdx[i];
        }
    }
    return aicpuOpIdx[blockIdx];
}

HcclResult AicpuRunRpcServer(AicpuComContext* ctx, KFCTask* taskInfo)
{
    // 启动RPC服务
    static AicpuKfcRpcServer rpc;
    rpc.Init(ctx->workSpaceAddr, ctx->notifyOff, ctx->notifyBeginCnt, taskInfo);
    AicpuKfcProf::GetProInst(*ctx).commInitEndTime = GetCurCpuTimestamp(true);
    if (rpc.GetPreparePosition() == TASK_PREPARE_KERNEL) {
        auto ret = AicpuKfcProcess::RunRpcServerApi(ctx, rpc);
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), false);
        if (ret != HCCL_SUCCESS) {
            return AicpuKfcProcess::DealReturnValue(ctx, ret);
        } else {
            CHK_RET(AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kEnd, KfcError::kNone, 0));
            return ret;
        }
    }

    if (rpc.GetTaskType() == HCCL_KFC_TASK_HCCL_ONLY_EXE) {
        auto ret = AicpuKfcDeprecatedProcess::AICPU_RpcServerUnfoldStageWait(ctx, rpc);
        if ((ret != HCCL_SUCCESS) && (ret != HCCL_E_SUSPENDING)) {
            HCCL_ERROR(
                "AicpuRpcStageWait failed, commType:%d, reducekind:%d, totalCnt:%lu, totalTurnCnt:%u", ctx->commType,
                ctx->reducekind, ctx->totalCnt, ctx->totalTurnCnt);
            return ret;
        }
        if (ret == HCCL_E_SUSPENDING) {
            HCCL_RUN_INFO("[NsRecovery][AICPU] Suspending");
            return ret;
        }
    } else if (ctx->devType == DevType::DEV_TYPE_310P1 || ctx->devType == DevType::DEV_TYPE_310P3) {
        auto ret = AicpuKfcDeprecatedProcess::RunRpcServerTwoStageWait(ctx, rpc);
        if ((ret != HCCL_SUCCESS) && (ret != HCCL_E_SUSPENDING)) {
            return ret;
        }
        if (ret == HCCL_E_SUSPENDING) {
            HCCL_RUN_INFO("[NsRecovery][MC2] Suspending");
            return ret;
        }
    } else {
        auto ret = AicpuKfcDeprecatedProcess::TryRunRpcServerOneStageWait(ctx, rpc);
        if ((ret != HCCL_SUCCESS) && (ret != HCCL_E_SUSPENDING)) {
            return ret;
        }
        if (ret == HCCL_E_SUSPENDING) {
            HCCL_RUN_INFO("[NsRecovery][MC2] Suspending");
            return ret;
        }
    }

    return HCCL_SUCCESS;
}

u32 RunAicpuInnerRpcSrvGroupLaunch(void* args[], KFCGroupTilingDataAuto* tilingData, CommKfcParamDesc* desc)
{
    constexpr int DESC_POS = 0;

    if (tilingData == nullptr || tilingData->groupNum == 0) {
        HCCL_ERROR("tilingData is nullptr or groupNum is 0.");
        return HCCL_E_PARA;
    }
    for (uint32_t i = 0; i < tilingData->groupNum; ++i) {
        KFCTask singleTask;
        singleTask.inputA = u64(args[tilingData->msg[i].sendArgIndex + desc->hasFfts + desc->itemNum + 1]);
        singleTask.outputC = GetTensorAddr(
            i, reinterpret_cast<uint8_t*>(args[tilingData->msg[i].recvArgIndex + desc->hasFfts + desc->itemNum + 1]));
        singleTask.commOut = 0;
        singleTask.context = u64(args[DESC_POS + desc->hasFfts + desc->itemNum]);
        singleTask.workSpace = u64(args[desc->tilingOff - 1]);
        singleTask.tilingData = u64(&tilingData->msg[i]);
        uint32_t ret = RunAicpuRpcSrvLaunch(&singleTask);
        if (ret != 0) {
            HCCL_ERROR("RunAicpuRpcSrvGroupLaunch runs failed.");
            return HCCL_E_PARA;
        }
    }
    return 0;
}

u32 RunKernelAicpuServerV1(void* args[], CommKfcParamDesc* desc)
{
    HcclKFCTilingData* tilingData = static_cast<HcclKFCTilingData*>(args[desc->tilingOff]);
    HCCL_INFO("RunAicpuKfcSrvLaunch, tiling.sendArgIndex %lu", tilingData->sendArgIndex);
    HCCL_INFO("RunAicpuKfcSrvLaunch, tiling.recvArgIndex %lu", tilingData->recvArgIndex);
    HCCL_INFO("RunAicpuKfcSrvLaunch, tiling.commOutArgIndex %lu", tilingData->commOutArgIndex);
    HCCL_INFO("RunAicpuKfcSrvLaunch, tiling.hasCommOut %lu", tilingData->hasCommOut);
    KFCTask task;
    task.tilingData = u64(tilingData);
    task.inputA = u64(args[tilingData->sendArgIndex + desc->hasFfts + desc->itemNum + 1]);
    task.outputC = u64(args[tilingData->recvArgIndex + desc->hasFfts + desc->itemNum + 1]);
    task.context = u64(args[desc->hasFfts + desc->itemNum]);
    task.workSpace = u64(args[desc->tilingOff - 1]);
    if (tilingData->commOutArgIndex != u64(0xff)) {
        task.commOut = u64(args[tilingData->commOutArgIndex + desc->hasFfts + desc->itemNum + 1]);
    } else {
        task.commOut = 0;
    }
    AicpuKfcUtils::PrintKFCTask(task);
    HCCL_INFO("Task Assembled. Start to launch RunAicpuRpcSrvLaunch");
    const uint32_t ret = RunAicpuRpcSrvLaunch(&task);
    HCCL_INFO("RunAicpuKfcSrvLaunch ends with result %lu.", ret);
    return ret;
}

HcclResult KfcProf(u64 launchEntryTime, KFCTaskV2& task, u32 turnOffset = 0U)
{
    if (AicpuKfcProf::NeedRecordTimeTaken()) {
        AicpuKfcProf::SetCurrentProf(launchEntryTime);
        HcclOpResParam* commParam = reinterpret_cast<HcclOpResParam*>(task.context[0]);
        AicpuKfcProf::GetCurrentAicpuProf()->rankId = commParam->topoInfo.userRank;
        AicpuKfcProf::GetCurrentAicpuProf()->endTime = GetCurCpuTimestamp(true);
    }

    CHK_RET(dfx::AicpuProfilingManager::ReportTaskExecTimeLine(AicpuKfcProf::GetCurrentAicpuProf(), turnOffset));
    AicpuKfcProf::OutputProfLog(
        AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_TIME_TAKEN), AicpuKfcProf::GetaicpuProfInst());
    AicpuKfcProf::AddProfLoopCnt();
    return HCCL_SUCCESS;
}

u32 RunKernelAicpuServerV2(void* args[], CommKfcParamDesc* desc, void* tilingData)
{
    u64 launchEntryTime = GetCurCpuTimestamp(true);
    // MC2目前只支持OP_BASE
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    static uint64_t aicpuOpIdx = 0;

    Mc2ServerCfg* cfg = MC2TilingGetServerCfg(tilingData);
    AicpuKfcProf::SetDebugMode(cfg->debugMode);
    if (AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_ONLY_CUBE)) {
        HCCL_INFO("[%s]DebugMode is set to be 1 (i.e. computation only).", __func__);
        return HCCL_SUCCESS;
    }

    KFCTaskV2 task;
    task.tilingData = reinterpret_cast<u64>(tilingData);
    task.ctxNum = desc->itemNum;
    if (task.ctxNum > MAX_COMM_CTX_NUM) {
        HCCL_ERROR("group num must be smaller than %u.", MAX_COMM_CTX_NUM);
        return HCCL_E_PARA;
    }
    for (int i = 0; i < desc->itemNum; i++) {
        task.context[i] = reinterpret_cast<u64>(args[desc->hasFfts + i + 1]);
        if (task.context[i] == 0) {
            HCCL_ERROR("idx %d ctx is null, please check the input ctx.", i);
            return HCCL_E_PARA;
        }
        HcclAicpuUtils::PrintHcclOpResParam(reinterpret_cast<HcclOpResParam*>(task.context[i]));
    }
    task.workSpace = reinterpret_cast<u64>(args[desc->tilingOff - 1]);
    aicpuOpIdx++;
    if (AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_PRINT_MSG) || AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_PRINT_BUFF)) {
        HCCL_RUN_INFO("Server start, MC2 opIdx:%lu", aicpuOpIdx);
    }
    HCCL_INFO("Start launch RunAicpuInnerKfcSrvLaunch");
    AicpuKfcProf::GetCurrentAicpuProf()->workCnt = 0;
    const u32 ret = AicpuKfcProcess::AicpuRunRpcServerForMC2(&task);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("Server runs failed, error code %u.", ret);
        return ret;
    }
    CHK_RET(KfcProf(launchEntryTime, task));
    HCCL_INFO("end kfc server.");
    return 0;
}

u32 RunAicpuApiRpcSrvLaunchV1(void* args[], CommKfcParamDesc* desc)
{
    static u32 aicpuOpIdx = 0;
    u64 launchEntryTime = GetCurCpuTimestamp(true);

    HccCommResParamTask* contextParam = nullptr;
    for (uint64_t i = 1; i <= desc->itemNum; ++i) {
        contextParam = reinterpret_cast<HccCommResParamTask*>(args[desc->hasFfts + i]);
        if (contextParam != nullptr) {
            HCCL_INFO("Idx %llu ctx addr %p.", i, contextParam);
            break;
        }
    }
    if (contextParam == nullptr) {
        HCCL_ERROR("Context args is null.");
        return HCCL_E_PARA;
    }

    AicpuComContext* ctx = AicpuGetComContext();
    if (ctx == nullptr || !ctx->alreadyInit || strcmp(ctx->hcomId, contextParam->hcomId) != 0) {
        HCCL_ERROR("The comm domain %s does not exist.", contextParam->hcomId);
        return HCCL_E_PARA;
    }
    AicpuServerRole role = AicpuKfcBatchwriteProcess::GetVerifiedServerRole(*ctx);
    if (role == AicpuServerRole::INVALID) {
        HCCL_INFO("aicpu server role is invalid, return");
        return HCCL_SUCCESS;
    } else if (role == AicpuServerRole::SLAVE) {
        return AicpuKfcBatchwriteProcess::RunSlaveRpcServerForApi(ctx);
    }
    if (ctx->dfxExtendInfo.cqeStatus != dfx::CqeStatus::kDefault
        || ctx->dfxExtendInfo.pollStatus == PollStatus::kStopAsException) {
        HCCL_ERROR(
            "Exist errors before, cqeStatus:%d, pollStatus:%d, group[%s]", ctx->dfxExtendInfo.cqeStatus,
            ctx->dfxExtendInfo.pollStatus, contextParam->hcomId);
        return HCCL_E_INTERNAL;
    }

    Mc2InitTilingInner* tilingData = reinterpret_cast<Mc2InitTilingInner*>(args[desc->tilingOff]);
    ctx->debugMode = tilingData->debugMode;
    if (ctx->debugMode == MC2_DEBUG_ONLY_CUBE) {
        HCCL_INFO("[%s]DebugMode is set to be 1 (i.e. computation only).", __func__);
        return HCCL_SUCCESS;
    }

    HcclComSuspendingFlag kfcFlag;
    CHK_RET(AicpuHdcUtils::GetSuspendingStatus(ctx->kfcControlTransferH2D, kfcFlag));
    if (kfcFlag == HcclComSuspendingFlag::isSuspending) {
        HCCL_WARNING("[NsRecovery] the op should not be launched in the suspending status");
        return HCCL_SUCCESS;
    }
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, endStopLaunch), false);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isStopLaunch), false);
    AicpuSqeContext::SyncVariable();
    auto profInst = AicpuKfcProf::GetProInst(*ctx);
    if (AicpuKfcUtils::NeedRecordTimeTaken(*ctx)) {
        profInst.tid = syscall(__NR_gettid);
        profInst.clusterId = ctx->clusterId;
        profInst.rankId = ctx->rankId;
        profInst.launchEntryTime = launchEntryTime;
    }

    ctx->preparePosition = TASK_PREPARE_KERNEL;
    ctx->notifyOff = 0;
    ctx->notifyBeginCnt = 0;
    ctx->notifyEndCnt = 0;
    ctx->totalCnt = 0;
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
        HCCL_INFO("Align hcclmsgarea from %p to %p", ctx->workSpaceAddr, newAddr);
    }
    ctx->workSpaceAddr = newAddr;
    ctx->commAlg = COMM_ALG_FULL_MESH;
    ctx->curTurnCnt = 0;
    profInst.workCnt = 0;
    ctx->msgPosForKernel = 0;
    ctx->curTurnCntForKernel = 0;
    ctx->totalTurnCntForKernel = 0;
    ctx->gatherOut = 0UL;
    AicpuKfcUtils::PrintTilingData(*tilingData);
    AicpuKfcUtils::PrintMC2AicpuContext(*ctx);

    CHK_RET(MC2TraceUtils::Submit(ctx)); // 上报ctx消息

    aicpuOpIdx++;
    if (ctx->debugMode == MC2_DEBUG_PRINT_MSG || ctx->debugMode == MC2_DEBUG_PRINT_BUFF) {
        HCCL_RUN_INFO("Server start, MC2 opIdx:%u", aicpuOpIdx);
    }
    HcclResult ret = AicpuKfcProcess::AicpuRunRpcServerForApi(ctx, reinterpret_cast<u64>(tilingData));
    if (ret == HCCL_E_SUSPENDING) {
        HCCL_INFO("mc2 opp is suspended");
        return AICPUSUSPENDING_ERROR;
    } else if (ret != HCCL_SUCCESS) {
        AicpuKfcUtils::PrintTilingData(*tilingData, true);
        AicpuKfcUtils::PrintMC2AicpuContext(*ctx, true);
        HCCL_ERROR("Server failed, MC2 opIdx:%u", aicpuOpIdx);
        return ret;
    }
    if (AicpuKfcUtils::NeedRecordTimeTaken(*ctx)) {
        profInst.endTime = GetCurCpuTimestamp(true);
    }
    CHK_RET(dfx::AicpuProfilingManager::ReportTaskExecTimeLine(&profInst));
    AicpuComContext* contextBase = nullptr;
    u32 contextNum = 0;
    AicpuGetAllComContext(contextBase, contextNum);
    AicpuKfcProf::OutputProfLog(
        AicpuKfcUtils::IsDebugModeEquals(*ctx, MC2_DEBUG_TIME_TAKEN), contextBase[0].acprof, contextBase[1].acprof);
    AicpuKfcProf::AddProfLoopCnt();
    AicpuSqeContext::SaveVariable();
    CHK_RET(AicpuSqeContext::ClearLocalBuff());
    HCCL_INFO("Kfc server ends successfully.");
    return HCCL_SUCCESS;
}

u32 RunAicpuApiRpcSrvLaunchV2(void* args[], CommKfcParamDesc* desc)
{
    // MC2目前只支持OP_BASE
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    u64 launchEntryTime = GetCurCpuTimestamp(true);

    const Mc2InitTilingInner* tilingData = static_cast<const Mc2InitTilingInner*>(args[desc->tilingOff]);
    AicpuKfcProf::SetDebugMode(tilingData->debugMode);
    if (AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_ONLY_CUBE)) {
        HCCL_INFO("[%s]DebugMode is set to be 1 (i.e. computation only).", __func__);
        return HCCL_SUCCESS;
    }

    KFCTaskV2 task{};
    for (uint64_t i = 0; i < desc->itemNum; i++) {
        u64 arg = reinterpret_cast<u64>(args[desc->hasFfts + i + 1]);
        if (arg != 0UL) {
            HCCL_INFO("Ctx idx %u, addr %#llx.", task.ctxNum, arg);
            task.context[task.ctxNum++] = arg;
        }
    }
    CHK_PRT_RET(
        task.ctxNum == 0 || task.ctxNum > MAX_COMM_CTX_NUM, HCCL_ERROR("Invalid ctx number %u.", task.ctxNum),
        HCCL_E_PARA);
    task.workSpace = 0;
    AicpuKfcProf::GetCurrentAicpuProf()->workCnt = 0;
    const u64 aicpuOpIdx = GetUpdatedOpIdx();
    if (AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_PRINT_MSG) || AicpuKfcProf::IsDebugModeEquals(MC2_DEBUG_PRINT_BUFF)) {
        HCCL_RUN_INFO("Server start, MC2 opIdx:%lu", aicpuOpIdx);
    }
    HCCL_INFO("Start %s, aicpuOpIdx %lu", __func__, aicpuOpIdx);
    const u32 ret = AicpuKfcProcess::AicpuRunRpcServerForMC2V2(&task, tilingData);
    if (ret != HCCL_SUCCESS) {
        AicpuKfcUtils::PrintTilingData(*tilingData, true);
        HCCL_ERROR("[%s] aicpuOpIdx %lu", __func__, aicpuOpIdx);
        return ret;
    }

    const u32 blockNum = HcclAicpuUtils::GetBlockNum();
    const u32 blockIdx = HcclAicpuUtils::GetBlockIdx();
    const u32 totalQueueNum = tilingData->commBlockNum * tilingData->queueNum;
    // blockNum 为 0 时，后续 totalQueueNum/blockNum 与 %blockNum 会除零，作为非法入参直接返回
    if (blockNum == 0) {
        HCCL_ERROR("Invalid blockNum 0, division by zero.");
        return HCCL_E_PARA;
    }
    const u32 turnOffset = blockIdx * (totalQueueNum / blockNum) + std::min(blockIdx, totalQueueNum % blockNum);
    CHK_RET(KfcProf(launchEntryTime, task, turnOffset));
    HCCL_INFO("End %s, aicpuOpIdx %lu", __func__, aicpuOpIdx);
    return 0;
}

u32 RunKernelAicpuServerForTilingApi(void* args[], CommKfcParamDesc* desc)
{
    if (AicpuHcclProcess::AicpuGetInnerDevType() == DevType::DEV_TYPE_910_93) {
        return RunAicpuApiRpcSrvLaunchV2(args, desc);
    } else {
        return RunAicpuApiRpcSrvLaunchV1(args, desc);
    }
}
} // namespace

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuKfcResInit(void* args)
{
    if (args == nullptr) {
        HCCL_ERROR("args is null.");
        return HCCL_E_PARA;
    }

    KFCResInitTask* ctxArgs = static_cast<KFCResInitTask*>(args);
    return AicpuKfcProcess::AicpuRpcResInit(reinterpret_cast<HccCommResParamTask*>(ctxArgs->context));
}

// aclgraph 销毁触发的 tag 清理入口。host 端走 KFCResInitTask{context, isCustom} 模式投递，这里从 context 反解 payload
// 交给 AicpuHcclProcess
__attribute__((visibility("default"))) uint32_t RunAicpuKfcClearOpRes(void* args)
{
    if (args == nullptr) {
        HCCL_ERROR("[RunAicpuKfcClearOpRes] args is null.");
        return HCCL_E_PARA;
    }
    KFCResInitTask* ctxArgs = static_cast<KFCResInitTask*>(args);
    if (ctxArgs->context == 0) {
        HCCL_ERROR("[RunAicpuKfcClearOpRes] ctxArgs->context is null.");
        return HCCL_E_PARA;
    }
    const HcclKfcClearOpResTilingData* tilingData
        = reinterpret_cast<const HcclKfcClearOpResTilingData*>(ctxArgs->context);
    return AicpuHcclProcess::AicpuRpcClearOpRes(tilingData);
}

__attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvLaunch(void* args)
{
    KfcState state;
    static uint32_t aicpuOpIdx = 0;
    u64 launchEntryTime = GetCurCpuTimestamp(true);

    if (args == nullptr) {
        HCCL_ERROR("args is null.");
        return HCCL_E_PARA;
    }

    KFCTask* task = reinterpret_cast<KFCTask*>(args);
    HCCL_INFO(
        "KFCTask inputA %p, outputC %p, commOut %p, context %p, workSpace %p, tilingData %p", task->inputA,
        task->outputC, task->commOut, task->context, task->workSpace, task->tilingData);
    HcclKFCTilingData* tilingData = reinterpret_cast<HcclKFCTilingData*>(task->tilingData);
    HccCommResParamTask* contextParam = reinterpret_cast<HccCommResParamTask*>(task->context);
    if (tilingData == nullptr || contextParam == nullptr) {
        HCCL_ERROR("tilingData or context args is null.");
        return HCCL_E_PARA;
    }

    AicpuComContext* ctx = AicpuGetComContext();
    if (ctx == nullptr || !ctx->alreadyInit || strcmp(ctx->hcomId, contextParam->hcomId) != 0) {
        HCCL_ERROR("The comm domain %s does not exist.", contextParam->hcomId);
        return HCCL_E_PARA;
    }
    if ((ctx->dfxExtendInfo.cqeStatus != dfx::CqeStatus::kDefault)
        || (ctx->dfxExtendInfo.pollStatus == PollStatus::kStopAsException)) {
        HCCL_ERROR(
            "Exist errors before, cqeStatus:%d, pollStatus:%d, group[%s]", ctx->dfxExtendInfo.cqeStatus,
            ctx->dfxExtendInfo.pollStatus, contextParam->hcomId);
        return HCCL_E_INTERNAL;
    }
    ctx->debugMode = tilingData->debugMode;
    if (ctx->debugMode == MC2_DEBUG_ONLY_CUBE) {
        HCCL_INFO("[%s]DebugMode is set to be 1 (i.e. computation only).", __func__);
        return HCCL_SUCCESS;
    }
    if (!tilingData->useBufferType) {
        ctx->gatherOut = task->commOut;
    } else {
        ctx->gatherOut = task->outputC;
    }
    // 这里就是判断Suspending通道的内容
    HCCL_DEBUG("[NsRecovery]check the suspending status");
    HcclComSuspendingFlag kfcFlag = HcclComSuspendingFlag ::isNull;
    CHK_RET(AicpuHdcUtils::GetSuspendingStatus(ctx->kfcControlTransferH2D, kfcFlag));
    if (kfcFlag == HcclComSuspendingFlag::isSuspending) {
        HCCL_WARNING("[NsRecovery] the op should not be launched in the suspending status");
        return 0;
    }
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, endStopLaunch), false);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isStopLaunch), false);
    AicpuSqeContext::SyncVariable();
    auto profInst = AicpuKfcProf::GetProInst(*ctx);
    if (AicpuKfcUtils::NeedRecordTimeTaken(*ctx)) {
        profInst.tid = syscall(__NR_gettid);
        profInst.clusterId = ctx->clusterId;
        profInst.rankId = ctx->rankId;
        profInst.launchEntryTime = launchEntryTime;
    }
    HCCL_INFO("RunAicpuRpcSrvLaunch, preparePosition %u", tilingData->preparePosition);
    if (tilingData->preparePosition > 1) {
        HCCL_ERROR("invalid preparePosition %u", tilingData->preparePosition);
        return HCCL_E_PARA;
    }
    ctx->preparePosition = static_cast<TASK_PREPARE_POSITION>(tilingData->preparePosition);
    if (ctx->preparePosition == TASK_PREPARE_HOST) {
        ctx->notifyOff = tilingData->notifyOff;
        ctx->notifyBeginCnt = tilingData->notifyBeginCnt;
        ctx->notifyEndCnt = tilingData->notifyEndCnt;
        ctx->totalCnt = tilingData->totalCnt;
    } else {
        ctx->notifyOff = 0;
        ctx->notifyBeginCnt = 0;
        ctx->notifyEndCnt = 0;
        ctx->totalCnt = 0;
        u64 newAddr = ctx->workSpaceAddr;
        if (newAddr & 0x1ff) {
            newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
            HCCL_INFO("Align hcclmsgarea from %p to %p", ctx->workSpaceAddr, newAddr);
        }
        ctx->workSpaceAddr = newAddr;
    }
    tilingData->commAlg
        = (ctx->devType == DevType::DEV_TYPE_910B) ? static_cast<u8>(COMM_ALG_FULL_MESH) : tilingData->commAlg;
    ctx->commAlg = tilingData->commAlg;
    ctx->skipLocalDataCopy = tilingData->hasCommOut ? false : true;
    ctx->curTurnCnt = 0;
    profInst.workCnt = 0;
    ctx->msgPosForKernel = 0;
    ctx->curTurnCntForKernel = 0;
    ctx->sendCntRecord[0] = AicpuKfcUtils::GetSendCnt(ctx);
    ctx->recvCntRecord[0] = AicpuKfcUtils::GetRecvCnt(ctx);
    ctx->totalTurnCntForKernel = 0;
    AicpuKfcUtils::PrintTilingData(*tilingData);
    AicpuKfcUtils::PrintMC2AicpuContext(*ctx);

    CHK_RET(MC2TraceUtils::Submit(task, tilingData));
    CHK_RET(MC2TraceUtils::Submit(ctx)); // 上报ctx消息

    aicpuOpIdx++;
    if (ctx->debugMode == MC2_DEBUG_PRINT_MSG || ctx->debugMode == MC2_DEBUG_PRINT_BUFF) {
        HCCL_RUN_INFO("Server start, MC2 opIdx:%u", aicpuOpIdx);
    }
    auto ret = AicpuRunRpcServer(ctx, task);
    ctx->sendCntRecord[3] = AicpuKfcUtils::GetSendCnt(ctx); // 3 记录执行结束时的sendCnt
    ctx->recvCntRecord[3] = AicpuKfcUtils::GetRecvCnt(ctx); // 3 记录执行结束时的recvCnt
    if ((ret != HCCL_SUCCESS) && (ret != HCCL_E_SUSPENDING)) {
        AicpuKfcUtils::PrintTilingData(*tilingData, true);
        AicpuKfcUtils::PrintMC2AicpuContext(*ctx, true);
        if (ctx->preparePosition == TASK_PREPARE_HOST) { // host展开时aicore会通过workspace回传维测信息, 解析后打印
            HCCL_ERROR(
                "Run rpc error, opIdx:%u, sndCnt:%d %d %d %d, rcvCnt:%d %d %d %d", aicpuOpIdx, ctx->sendCntRecord[0],
                ctx->sendCntRecord[1], ctx->sendCntRecord[2], ctx->sendCntRecord[3], ctx->recvCntRecord[0],
                ctx->recvCntRecord[1], ctx->recvCntRecord[2], ctx->recvCntRecord[3]);
        }
        HCCL_ERROR("Failed to run aicpu server, MC2 opIdx:%u", aicpuOpIdx);
        return ret;
    } else if (ret == HCCL_E_SUSPENDING) {
        HCCL_INFO("mc2 opp is suspended");
        return AICPUSUSPENDING_ERROR;
    }
    if (AicpuKfcUtils::NeedRecordTimeTaken(*ctx)) {
        profInst.endTime = GetCurCpuTimestamp(true);
    }
    CHK_RET(dfx::AicpuProfilingManager::ReportTaskExecTimeLine(&profInst));
    AicpuComContext* contextBase = nullptr;
    u32 contextNum = 0;
    AicpuGetAllComContext(contextBase, contextNum);
    AicpuKfcProf::OutputProfLog(
        AicpuKfcUtils::IsDebugModeEquals(*ctx, MC2_DEBUG_TIME_TAKEN), contextBase[0].acprof, contextBase[1].acprof);
    AicpuKfcProf::AddProfLoopCnt();
    AicpuSqeContext::SaveVariable();
    CHK_RET(AicpuSqeContext::ClearLocalBuff());
    HCCL_INFO("end RunAicpuRpcSrvLaunch");
    return 0;
}

__attribute__((visibility("default"))) uint32_t RunAicpuRpcSrvGroupLaunch(void* args)
{
    KfcState state;
    if (args == nullptr) {
        HCCL_ERROR("args is null.");
        return HCCL_E_PARA;
    }

    KFCTask* task = reinterpret_cast<KFCTask*>(args);
    HCCL_INFO(
        "KFCTask inputA %p, outputC %p, commOut %p, context %p, workSpace %p, tilingData %p", task->inputA,
        task->outputC, task->commOut, task->context, task->workSpace, task->tilingData);
    KFCGroupTilingData* tilingData = reinterpret_cast<KFCGroupTilingData*>(task->tilingData);

    if (tilingData == nullptr || tilingData->groupNum == 0) {
        HCCL_ERROR("tilingData is nullptr or groupNum is 0.");
        return HCCL_E_PARA;
    }

    for (uint32_t i = 0; i < tilingData->groupNum; ++i) {
        KFCTask singleTask;
        singleTask.inputA = task->inputA;
        singleTask.outputC = *reinterpret_cast<u64*>(task->outputC + sizeof(void*) * i);
        singleTask.commOut = task->commOut;
        singleTask.context = task->context;
        singleTask.workSpace = task->workSpace;
        singleTask.tilingData = reinterpret_cast<u64>(&tilingData->msg[i]);
        uint32_t ret = RunAicpuRpcSrvLaunch(&singleTask);
        if (ret != 0) {
            HCCL_ERROR("RunAicpuRpcSrvGroupLaunch runs failed.");
            return HCCL_E_PARA;
        }
    }

    return 0;
}

constexpr u32 GROUP_DYN_FLAG = 23U;
constexpr u32 GROUP_TILING_MAGIC_NUM = 99U;
__attribute__((visibility("default"))) uint32_t RunAicpuKfcSrvLaunch(void* args[])
{
    if (args == nullptr) {
        HCCL_ERROR("args is null.");
        return HCCL_E_PARA;
    }
    constexpr int DESC_POS = 0;
    uint64_t desc_value = u64(args[DESC_POS]);
    uint64_t* desc_addr = &desc_value;
    CommKfcParamDesc* desc = reinterpret_cast<CommKfcParamDesc*>(desc_addr);
    AicpuKfcUtils::PrintHcclCommParamDesc(*desc);
    if (desc->version == DECOUPLED_CTX_VER) {
        return CommKfcDispatcher::Run(&(args[1]), desc->itemNum);
    }
    void* tiling = reinterpret_cast<void*>(args[desc->tilingOff]);
    if (tiling == nullptr) {
        HCCL_ERROR("tiling is null.");
        return HCCL_E_PARA;
    }
    KfcState state;
    bool profL1Open = dfx::ProfilingManager::IsProfL1On();
    bool profL0Open = dfx::ProfilingManager::IsProfL0On();
    HCCL_INFO("profL1Open:%d, profL0Open:%d", profL1Open, profL0Open);
    const uint32_t ver = MC2TilingGetVer(tiling);
    HCCL_INFO("Start RunAicpuKfcSrvLaunch with tiling version %u.", ver);
    uint32_t ret;
    switch (ver) {
        case TILING_DATA_VER_OLD_FOR_HOST: {
            KFCGroupTilingDataAuto* tilingData = static_cast<KFCGroupTilingDataAuto*>(tiling);
            if (desc->isDyn == GROUP_DYN_FLAG && tilingData->groupTilingMagicNum == GROUP_TILING_MAGIC_NUM) {
                ret = RunAicpuInnerRpcSrvGroupLaunch(args, tilingData, desc);
            } else {
                ret = RunKernelAicpuServerV1(args, desc);
            }
            break;
        }
        case TILING_DATA_VER_OLD_FOR_KERNEL:
            ret = RunKernelAicpuServerV1(args, desc);
            break;
        case TILING_DATA_VER_OLD_FOR_KERNEL_V2:
            ret = RunKernelAicpuServerV2(args, desc, tiling);
            break;
        case TILING_DATA_VER_FOR_TILING_API:
            ret = RunKernelAicpuServerForTilingApi(args, desc);
            break;
        default:
            HCCL_ERROR("Invalid tiling version %u.", ver);
            ret = HCCL_E_PARA;
    }
    return ret;
}
}
