/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_launch.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "adapter_rts_common.h"
#include "ccu_res.h"

#include "ccu_log.h"

#include "hcom_common.h"

#include "ccu_kernel_mgr.h"
#include "ccu_instance_mgr.h"
#include "ccu_var_event_res_mgr.h"

#include "log.h"
#include "thread.h"
#include "dfx_dlprof_function.h"

#include "env_config/env_config_v2.h" // 暂时引用orion的环境变量处理模块

#include "hcomm_adapter_rts.h"

#include "task_param.h"

#include "ccu_assist_v1.h"

#include "config_plf_log_v2.h"
using Hccl::PLF_TASK;

CcuResult HcommCcuKernelRegisterStart(CcuInsHandle insHandle)
{
    const uint32_t userDevId = HcclGetThreadDeviceId();
    auto* ccuIns = hcomm::CcuInstanceMgr::GetInstance(userDevId).Get(insHandle);
    CCU_CHK_PTR_NULL(ccuIns);

    CCU_CHK_RET(ccuIns->BeginRegister());

    CcuResult ret = ccuIns->Reset();
    if (ret != CcuResult::CCU_SUCCESS) {
        (void)ccuIns->EndRegister();
        HCCL_ERROR("[%s] failed, Reset failed[%d], rollback register state.", __func__, ret);
        return ret;
    }
    return CcuResult::CCU_SUCCESS;
}

static CcuResult CcuKernelTryRegister(
    hcomm::CcuInstance* ccuIns, hcomm::CcuResPack* resPack, uint32_t userDevId, uint32_t dieId,
    const char* kernelFuncName, const void* kernelFunc, const void** kernelArgs, uint32_t argNum,
    CcuKernelHandle& newHandle)
{
    CCU_EXCEPTION_HANDLE_BEGIN
    auto& kernelMgr = hcomm::CcuKernelMgr::GetInstance(userDevId);
    CCU_CHK_RET(kernelMgr.Register(*resPack, dieId, kernelFuncName, kernelFunc, kernelArgs, argNum, ccuIns, newHandle));
    CCU_CHK_RET(ccuIns->SaveKernel(newHandle));
    CCU_EXCEPTION_HANDLE_END
    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuKernelRegister(
    CcuInsHandle insHandle, uint32_t dieId, const char* kernelFuncName, const void* kernelFunc, const void** kernelArgs,
    uint32_t argNum, CcuKernelHandle* kernelHandle)
{
    HCCL_RUN_INFO("Entry-%s", __func__);
    HcclUs startut = TIME_NOW();

    CCU_CHK_PTR_NULL(kernelFunc);
    CCU_CHK_PTR_NULL(kernelHandle);

    if (argNum != 0) {
        CCU_CHK_PTR_NULL(kernelArgs);
    }

    const uint32_t userDevId = HcclGetThreadDeviceId();
    auto* ccuIns = hcomm::CcuInstanceMgr::GetInstance(userDevId).Get(insHandle);
    CCU_CHK_PTR_NULL(ccuIns);

    CCU_CHK_RET(ccuIns->CheckRegistering());

    auto* resPack = ccuIns->GetResPack();
    CCU_CHK_PTR_NULL(resPack);

    CcuKernelHandle newHandle{0};
    CcuResult ret = CcuKernelTryRegister(
        ccuIns, resPack, userDevId, dieId, kernelFuncName, kernelFunc, kernelArgs, argNum, newHandle);
    if (ret != CcuResult::CCU_SUCCESS) {
        ccuIns->AbortRegister();
        if (CCU_CHK_RES_UNAVAIL(ret)) {
            HCCL_WARNING(
                "[%s] register kernel resource unavailable[%d], current register round aborted.", __func__, ret);
            return CcuResult::CCU_E_UNAVAIL;
        } else {
            HCCL_ERROR("[%s] failed, register kernel failed[%d], current register round aborted.", __func__, ret);
            return ret;
        }
    }

    *kernelHandle = newHandle;
    HCCL_INFO("[%s] success, take time [%lld]us.", __func__, DURATION_US(TIME_NOW() - startut).count());
    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle insHandle)
{
    const uint32_t userDevId = HcclGetThreadDeviceId();
    auto* ccuIns = hcomm::CcuInstanceMgr::GetInstance(userDevId).Get(insHandle);
    CCU_CHK_PTR_NULL(ccuIns);

    CCU_CHK_RET(ccuIns->EndRegister());
    const auto& newKernels = ccuIns->GetUntranslatedKernels();

    auto& kernelMgr = hcomm::CcuKernelMgr::GetInstance(userDevId);
    // 当前翻译内部流程可能抛异常
    CCU_EXCEPTION_HANDLE_BEGIN
    CCU_CHK_RET(kernelMgr.Translate(newKernels));
    CCU_EXCEPTION_HANDLE_END

    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuGetTaskArgsNum(CcuKernelHandle kernelHandle, uint32_t* taskArgsNum)
{
    HCCL_INFO("Entry-%s", __func__);
    CCU_CHK_PTR_NULL(taskArgsNum);

    const uint32_t userDevId = HcclGetThreadDeviceId();
    auto& kernelMgr = hcomm::CcuKernelMgr::GetInstance(userDevId);

    hcomm::CcuKernelInfo info{};
    CCU_CHK_RET(kernelMgr.GetCcuKernelInfo(kernelHandle, info));

    *taskArgsNum = info.maxTaskArgsNum;
    HCCL_INFO("[%s] success, kernelHandle[%llx], taskArgsNum[%u].", __func__, kernelHandle, *taskArgsNum);
    return CcuResult::CCU_SUCCESS;
}

static std::shared_ptr<std::vector<Hccl::CcuProfilingInfo>>
ConstructCcuDetailInfo(const std::vector<hcomm::CcuProfilingInfo>& allCcuProfilingInfo, bool isSaveProfilingData)
{
    if (allCcuProfilingInfo.empty() || !isSaveProfilingData) {
        return nullptr;
    }

    std::vector<Hccl::CcuProfilingInfo> converted(allCcuProfilingInfo.size());
    for (u32 idx = 0; idx < allCcuProfilingInfo.size(); ++idx) {
        auto& src = allCcuProfilingInfo[idx];
        auto& dst = converted[idx];
        dst.name = src.name;
        dst.type = src.type;
        dst.dieId = src.dieId;
        dst.missionId = src.missionId;
        dst.instrId = src.instrId;
        dst.reduceOpType = src.reduceOpType;
        dst.inputDataType = src.inputDataType;
        dst.outputDataType = src.outputDataType;
        dst.dataSize = src.dataSize;
        dst.ckeId = src.ckeId;
        dst.mask = src.mask;
        (void)memcpy_s(dst.channelId, sizeof(dst.channelId), src.channelId, sizeof(src.channelId));
        (void)memcpy_s(dst.channelHandle, sizeof(dst.channelHandle), src.channelHandle, sizeof(src.channelHandle));
    }
    return std::make_shared<std::vector<Hccl::CcuProfilingInfo>>(std::move(converted));
}

static Hccl::TaskParam ConstructCcuTaskParam(
    const hcomm::CcuTaskParam& ccuParam, const CcuKernelHandle kernelHandle,
    const std::shared_ptr<std::vector<Hccl::CcuProfilingInfo>>& ccuDetailInfo, u64 beginTime, u64 endTime,
    bool isMaster)
{
    Hccl::TaskParam taskParam{};
    taskParam.beginTime = beginTime;
    taskParam.endTime = endTime;
    taskParam.taskType = Hccl::TaskParamType::TASK_CCU;
    taskParam.taskPara.Ccu.dieId = ccuParam.dieId;
    taskParam.taskPara.Ccu.missionId = ccuParam.missionId;
    taskParam.taskPara.Ccu.execMissionId = ccuParam.missionId;
    taskParam.taskPara.Ccu.instrId = ccuParam.instStartId;
    taskParam.taskPara.Ccu.executeId = kernelHandle;
    taskParam.taskPara.Ccu.ccuKernelHandle = kernelHandle;
    taskParam.isMaster = isMaster;
    taskParam.ccuDetailInfo = ccuDetailInfo;
    return taskParam;
}

static void LogCcuTaskInfo(const std::vector<hcomm::CcuTaskParam>& ccuParams, const CcuKernelHandle kernelHandle)
{
    if (!HcclCheckLogLevel(HCCL_LOG_INFO)) {
        return;
    }
    const uint32_t execTimeOutSec = Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
    for (u32 idx = 0; idx < ccuParams.size(); idx++) {
        const auto& param = ccuParams[idx];
        PLF_CONFIG_INFO(
            PLF_TASK,
            "[%s] start ccu task, dieId[%u], missionId[%u], execMissionId[%u], instStartId[%u], instCnt[%u], "
            "argSize[%u], timeout[%u]s, executeId[0x%llx], ccuKernelHandle[0x%llx]",
            __func__, param.dieId, param.missionId, param.missionId, param.instStartId, param.instCnt, param.argSize,
            execTimeOutSec, kernelHandle, kernelHandle);
    }
}

static void ConstructProfilingInfoLog(const std::vector<hcomm::CcuProfilingInfo>& allCcuProfilingInfo)
{
    if (!HcclCheckLogLevel(HCCL_LOG_INFO)) {
        return;
    }
    for (const hcomm::CcuProfilingInfo& profInfo : allCcuProfilingInfo) {
        for (int idx = 0; idx < hcomm::CCU_MAX_CHANNEL_NUM; idx++) {
            if (profInfo.channelId[idx] == hcomm::INVALID_VALUE_CHANNELID) {
                break;
            }
            HCCL_INFO(
                "[%s]idx[%d]: channelId[%u], channelHandle[0x%llx]", __func__, idx, profInfo.channelId[idx],
                profInfo.channelHandle[idx]);
        }
    }
}

static CcuResult ConstructProfilingInfo(
    hcomm::CcuKernel* kernel, const uint64_t* taskArgs, uint32_t argNum,
    std::vector<hcomm::CcuProfilingInfo>& allCcuProfilingInfo, bool isSaveProfilingData)
{
    if (!isSaveProfilingData) {
        return CcuResult::CCU_SUCCESS;
    }

    CCU_CHK_RET(kernel->GetCcuProfilingInfo(taskArgs, argNum, allCcuProfilingInfo));
    if (allCcuProfilingInfo.empty()) {
        return CcuResult::CCU_SUCCESS;
    }
    ConstructProfilingInfoLog(allCcuProfilingInfo);
    return CcuResult::CCU_SUCCESS;
}

static CcuResult ReportCcuTaskDfx(const ThreadHandle threadHandle, const Hccl::TaskParam& taskParam)
{
    auto* rtsThread = reinterpret_cast<hccl::Thread*>(threadHandle);
    CCU_CHK_PTR_NULL(rtsThread);

    auto callback = rtsThread->GetCallback();
    if (!callback) {
        HCCL_WARNING("[%s] task info callback is not registered on thread, skip ccu profiling report.", __func__);
        return CcuResult::CCU_SUCCESS;
    }
    u32 streamId = INVALID_UINT;
    u32 taskId = INVALID_UINT;
    CCU_CHK_RET(hrtGetTaskIdAndStreamID(taskId, streamId));
    CCU_CHK_RET(callback(streamId, taskId, taskParam, DFX_INVALID_U64));
    return CcuResult::CCU_SUCCESS;
}

static HcclResult LaunchCcuTasks(const hcomm::CcuTaskParam& param, const aclrtStream stream)
{
    const uint32_t execTimeOutSec = Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
    rtCcuTaskInfo_t taskInfo{};
    taskInfo.dieId = param.dieId;
    taskInfo.missionId = param.missionId;
    taskInfo.instStartId = param.instStartId;
    taskInfo.instCnt = param.instCnt;
    taskInfo.key = param.key;
    taskInfo.argSize = param.argSize;
    taskInfo.timeout = execTimeOutSec >= std::numeric_limits<u16>::max() ? std::numeric_limits<u16>::max() :
                                                                           static_cast<u16>(execTimeOutSec);
    std::copy(std::begin(param.args), std::end(param.args), std::begin(taskInfo.args));

    auto ret = rtCCULaunch(&taskInfo, stream);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[%s] failed to launch ccu, ret[%d]", __func__, ret);
        return HcclResult::HCCL_E_RUNTIME;
    }

    return HcclResult::HCCL_SUCCESS;
}

CcuResult
HcommCcuKernelLaunch(ThreadHandle threadHandle, CcuKernelHandle kernelHandle, const void* taskArgs, uint32_t argNum)
{
    const auto& startus = TIME_NOW();

    CHK_PRT_RET(threadHandle == 0, HCCL_ERROR("[%s] failed, thread handle is empty.", __func__), CcuResult::CCU_E_PARA);
    CHK_PRT_RET(kernelHandle == 0, HCCL_ERROR("[%s] failed, kernel handle is empty.", __func__), CcuResult::CCU_E_PARA);
    CHK_PRT_RET(
        argNum > 0 && taskArgs == nullptr,
        HCCL_ERROR("[%s] failed, taskArgs is nullptr while argNum[%u] > 0.", __func__, argNum), CcuResult::CCU_E_PTR);

    PLF_CONFIG_INFO(
        PLF_TASK, "[HcommCcuKernelLaunch] threadHandle[0x%llx] kernelHandle[0x%llx].", threadHandle, kernelHandle);

    const auto* rtsThread = reinterpret_cast<hccl::Thread*>(threadHandle);
    const auto* threadStream = rtsThread->GetStream();
    CCU_CHK_PTR_NULL(threadStream);
    auto* streamPtr = threadStream->ptr();
    CCU_CHK_PTR_NULL(streamPtr);

    const uint32_t userDevId = HcclGetThreadDeviceId();
    auto& kernelMgr = hcomm::CcuKernelMgr::GetInstance(userDevId);
    auto* kernel = kernelMgr.GetKernel(kernelHandle);
    CCU_CHK_PTR_NULL(kernel);

    CCU_EXCEPTION_HANDLE_BEGIN
    std::vector<hcomm::CcuTaskParam> taskParams{};
    auto ret = kernel->GeneTaskParams(static_cast<const uint64_t*>(taskArgs), argNum, taskParams);
    CHK_PRT_RET(
        ret != CcuResult::CCU_SUCCESS,
        HCCL_ERROR("[%s] failed, threadHandle[0x%llx] kernelHandle[0x%llx].", __func__, threadHandle, kernelHandle),
        ret);

    if (taskParams.empty()) {
        HCCL_INFO("[%s] passed, ccu params are empty.", __func__);
        return CcuResult::CCU_SUCCESS;
    }
    bool isProfilingEnabledL1 = Hccl::DfxProfilingHandler::GetInstance().GetHcclL1State();
    bool isProfilingEnabledL0 = Hccl::DfxProfilingHandler::GetInstance().GetHcclL0State();
    bool isCached = Hccl::DfxProfilingHandler::GetInstance().GetCachedFlag();
    bool isSaveProfilingData = isProfilingEnabledL1 || isProfilingEnabledL0 || isCached;

    std::vector<hcomm::CcuProfilingInfo> allCcuProfilingInfo;
    CCU_CHK_RET(ConstructProfilingInfo(
        kernel, static_cast<const uint64_t*>(taskArgs), argNum, allCcuProfilingInfo, isSaveProfilingData));
    LogCcuTaskInfo(taskParams, kernelHandle);
    auto ccuDetailInfo = ConstructCcuDetailInfo(allCcuProfilingInfo, isSaveProfilingData);
    for (u32 idx = 0; idx < taskParams.size(); idx++) {
        u64 beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
        CCU_CHK_RET(LaunchCcuTasks(taskParams[idx], streamPtr));
        u64 endTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
        Hccl::TaskParam taskParam = ConstructCcuTaskParam(
            taskParams[idx], kernelHandle, ccuDetailInfo, beginTime, endTime, rtsThread->GetMaster());
        CCU_CHK_RET(ReportCcuTaskDfx(threadHandle, taskParam));
    }
    CCU_EXCEPTION_HANDLE_END
    HCCL_INFO("[%s] success, take time [%lld]us.", __func__, DURATION_US(TIME_NOW() - startus).count());
    return CcuResult::CCU_SUCCESS;
}

// 定位实例资源仓后转发给 CcuVarEventResMgr；预约、地址映射与失败回滚均由该类内部完成
static CcuResult CcuAllocVarEventRes(
    CcuInsHandle insHandle, hcomm::CcuVarEventType type, uint8_t dieId, uint32_t num, uint64_t& outHandle)
{
    const uint32_t userDevId = HcclGetThreadDeviceId();
    auto* ccuIns = hcomm::CcuInstanceMgr::GetInstance(userDevId).Get(insHandle);
    CCU_CHK_PTR_NULL(ccuIns);

    auto* resPack = ccuIns->GetResPack();
    CCU_CHK_PTR_NULL(resPack);

    CCU_CHK_RET(hcomm::CcuVarEventResMgr::GetInstance(userDevId).Acquire(
        insHandle, resPack->GetCcuResRepo(), type, dieId, num, outHandle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuVariableAlloc(CcuInsHandle insHandle, uint8_t dieId, uint32_t num, CcuVariableHandle* varHandle)
{
    CCU_CHK_PTR_NULL(varHandle);
    *varHandle = 0;
    CCU_CHK_RET(CcuAllocVarEventRes(insHandle, hcomm::CcuVarEventType::VARIABLE, dieId, num, *varHandle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuEventAlloc(CcuInsHandle insHandle, uint8_t dieId, uint32_t num, CcuEventHandle* eventHandle)
{
    CCU_CHK_PTR_NULL(eventHandle);
    *eventHandle = 0;
    CCU_CHK_RET(CcuAllocVarEventRes(insHandle, hcomm::CcuVarEventType::EVENT, dieId, num, *eventHandle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuVariableGetAddr(CcuVariableHandle varHandle, uint32_t index, uint64_t* va)
{
    CCU_CHK_PTR_NULL(va);

    const uint32_t userDevId = HcclGetThreadDeviceId();
    CCU_CHK_RET(hcomm::CcuVarEventResMgr::GetInstance(userDevId).GetSavedAddr(
        hcomm::CcuVarEventType::VARIABLE, varHandle, index, *va));

    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuEventGetAddr(CcuEventHandle eventHandle, uint32_t index, uint64_t* va)
{
    CCU_CHK_PTR_NULL(va);

    const uint32_t userDevId = HcclGetThreadDeviceId();
    CCU_CHK_RET(hcomm::CcuVarEventResMgr::GetInstance(userDevId).GetSavedAddr(
        hcomm::CcuVarEventType::EVENT, eventHandle, index, *va));

    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuGetMemToken(uint64_t srcVa, uint64_t size, uint64_t* tokenInfo)
{
    CCU_CHK_PTR_NULL(tokenInfo);

    if (srcVa == 0 || size == 0) {
        HCCL_ERROR(
            "[%s] failed, srcVa[0x%llx] size[%llu] should not be 0.", __func__, static_cast<unsigned long long>(srcVa),
            static_cast<unsigned long long>(size));
        return CcuResult::CCU_E_PARA;
    }
    // 注意token信息属于安全信息，均不允许打印
    hcomm::rtMemUbTokenInfo info{};
    info.va = srcVa;
    info.size = size;

    const uint32_t userDevId = HcclGetThreadDeviceId();
    uint64_t tokenId = 0;
    uint64_t tokenValue = 0;
    CcuResult ret = hcomm::CcuQueryTokenInfo(userDevId, info.va, info.size, tokenId, tokenValue);
    if (ret == CcuResult::CCU_E_NOT_FOUND || ret == CcuResult::CCU_E_NOT_SUPPORT) {
        // 非CCU资源空间或非960设备：回退既有RTS查询路径
        CCU_CHK_RET(hcomm::RtsUbDevQueryInfo(QUERY_PROCESS_TOKEN, info));
    } else if (ret != CcuResult::CCU_SUCCESS) {
        HCCL_ERROR("[%s] failed, ret[%d].", __func__, ret);
        return ret;
    } else {
        info.tokenId = static_cast<uint32_t>(tokenId);
        info.tokenValue = static_cast<uint32_t>(tokenValue);
    }
    *tokenInfo = hcomm::CcuRep::CcuCombineTokenInfo(info.tokenId, info.tokenValue, 1);

    return CcuResult::CCU_SUCCESS;
}
