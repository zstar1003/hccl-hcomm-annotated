/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcclCommDfx.h"
#include "dfx_dlprof_function.h"
#include "ccu_rep_context_v1.h"
#include "task_info.h"

namespace hccl {

std::shared_mutex HcclCommDfx::baseLock_;
std::mutex HcclCommDfx::taskIdMutex_;
std::unordered_map<std::string, std::unordered_map<u64, u32>> HcclCommDfx::channelRemoteRankId_;
std::unordered_map<u32, u32> HcclCommDfx::streamIdToTaskId_;
HcclCommDfx::HcclCommDfx() {}

HcclCommDfx::~HcclCommDfx()
{
    setAddTaskCallback_ = nullptr;
    setAddDpuTaskCallback_ = nullptr;
}

HcclResult HcclCommDfx::Init(u32 deviceId, const std::string& comTag, u32 myRankId)
{
    if (initializedFlag_) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[%s]deviceId[%u], comTag[%s], myRankId[%u]", __func__, deviceId, comTag.c_str(), myRankId);
    deviceId_ = deviceId;
    commTag_ = comTag;
    myRankId_ = myRankId;
    // 1. 如果mirrorTaskManager_为空，则创建新的MirrorTaskManager
    if (!mirrorTaskManager_) {
        mirrorTaskManager_
            = std::make_unique<Hccl::MirrorTaskManager>(deviceId_, &Hccl::GlobalMirrorTasks::Instance(), false);
    }

    // 2. 创建Profiling管理类
    EXCEPTION_CATCH(
        profiling_ = std::make_unique<HcclCommProfiling>(deviceId_, mirrorTaskManager_.get()), return HCCL_E_PTR);
    CHK_RET(profiling_->Init());
    // profiling_->Init() 内部触发 DfxDlProfFunctionInit(dlopen libprofapi.so)，必须在 dlsym 完成后计算
    // groupNameHash_，否则首 comm 走 stub 返回 0
    groupNameHash_ = Hccl::DfxDlProfFunction::GetInstance().dlMsprofStr2Id(commTag_.c_str(), commTag_.length());

    // 3. 注册回调
    setAddTaskCallback_ = [this](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle) {
        return this->AddTaskInfoCallback(streamId, taskId, taskParam, handle);
    };
    setAddDpuTaskCallback_ = [this](const Hccl::TaskParam& taskParam, u64 handle) {
        return this->AddDpuTaskInfoCallback(taskParam, handle);
    };
    initializedFlag_ = true;
    return HCCL_SUCCESS; // 初始化成功返回成功码
}

HcclResult HcclCommDfx::GetOpModeFlags(bool& isOpBase, bool& isCached)
{
    auto opMode = mirrorTaskManager_->GetOpMode();
    isOpBase = opMode == Hccl::OpMode::OPBASE || opMode == Hccl::OpMode::ACLGRAPH;
    isCached = opMode == Hccl::OpMode::OFFLOAD || opMode == Hccl::OpMode::ACLGRAPH;
    return HCCL_SUCCESS;
}

// 回调注册实现
void HcclCommDfx::AddTaskInfoCallbackLog(
    const Hccl::TaskParam& taskParam, const std::unordered_map<u64, u32>& handleMap) const
{
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO) == 0)) {
        return;
    }
    for (size_t i = 0; i < taskParam.ccuDetailInfo->size(); ++i) {
        const Hccl::CcuProfilingInfo& profInfo = (*taskParam.ccuDetailInfo)[i];
        for (int idx = 0; idx < hcomm::CCU_MAX_CHANNEL_NUM; idx++) {
            if (profInfo.channelId[idx] == hcomm::INVALID_VALUE_CHANNELID) {
                break;
            }
            auto handleIt = handleMap.find(profInfo.channelHandle[idx]);
            if (handleIt == handleMap.end()) {
                continue;
            }
            HCCL_INFO(
                "[%s]idx[%d]: channelId[%u], remoteRankId[%u], channelHandle[0x%llx]", __func__, idx,
                profInfo.channelId[idx], handleIt->second, profInfo.channelHandle[idx]);
        }
    }
}

HcclResult HcclCommDfx::AddTaskInfoCallback(u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle)
{
    u32 remoteRankId = INVALID_UINT;
    if (handle != DFX_INVALID_U64) {
        CHK_RET(GetChannelRemoteRankId(commTag_, handle, remoteRankId));
    }

    Hccl::PrintTaskLog(streamId, taskId, taskParam, remoteRankId);

    if (taskParam.taskType == Hccl::TaskParamType::TASK_CCU && taskParam.ccuDetailInfo != nullptr) {
        std::shared_lock<std::shared_mutex> rwLock(baseLock_);
        auto commIt = channelRemoteRankId_.find(commTag_);
        if (commIt == channelRemoteRankId_.end()) {
            HCCL_ERROR("[%s] commTag:[%s] not found in CCU batch lookup", __func__, commTag_.c_str());
            return HCCL_E_PARA;
        }
        const auto& handleMap = commIt->second;
        AddTaskInfoCallbackLog(taskParam, handleMap);
        for (size_t i = 0; i < taskParam.ccuDetailInfo->size(); ++i) {
            Hccl::CcuProfilingInfo& profInfo = (*taskParam.ccuDetailInfo)[i];
            for (int idx = 0; idx < hcomm::CCU_MAX_CHANNEL_NUM; idx++) {
                if (profInfo.channelId[idx] == hcomm::INVALID_VALUE_CHANNELID) {
                    break;
                }
                auto handleIt = handleMap.find(profInfo.channelHandle[idx]);
                if (handleIt == handleMap.end()) {
                    HCCL_ERROR(
                        "[%s] Failed to get remote rank for channelHandle[0x%llx]", __func__,
                        profInfo.channelHandle[idx]);
                    return HCCL_E_PARA;
                }
                profInfo.remoteRankId[idx] = handleIt->second;
            }
        }
    }
    HcclResult ret = mirrorTaskManager_->AddTaskInfo(
        streamId, taskId, remoteRankId, taskParam, mirrorTaskManager_->GetCurrDfxOpInfo(), taskParam.isMaster,
        Hccl::DfxProfilingHandler::GetCachedTid());
    CHK_RET(ret);
    return HCCL_SUCCESS;
}

HcclResult HcclCommDfx::AddDpuTaskInfoCallback(const Hccl::TaskParam& taskParam, u64 handle)
{
    u32 streamId = dpuStreamId_;
    u32 taskId = GetTaskId(streamId);
    Hccl::TaskParam localTaskParam = taskParam;
    localTaskParam.aicpuTaskId = aicpuTaskId_;
    localTaskParam.npuDevId = deviceId_;
    HCCL_INFO(
        "[%s] streamId[%u], taskId[%u], aicpuTaskId[%llu], npuDevId[%u].", __func__, streamId, taskId,
        localTaskParam.aicpuTaskId, static_cast<u32>(localTaskParam.npuDevId));
    return AddTaskInfoCallback(streamId, taskId, localTaskParam, handle);
}

HcclResult HcclCommDfx::SetCurrDfxOpInfo(std::shared_ptr<Hccl::DfxOpInfo> dfxOpInfo)
{
    if (dfxOpInfo != nullptr) {
        // commTag_ 与 op_.opTag 恒等（单算子入口 HcclDfxRegOpInfoByCommId 设 op_.opTag=collComm->GetCommId()，
        // HcclCommDfx::Init 的 comTag 入参来源同为 collComm->GetCommId()），故 groupNameHash_ 可代表 op_.opTag 哈希
        dfxOpInfo->groupNameHash_ = groupNameHash_;
    }
    profiling_->SetCurrDfxOpInfo(dfxOpInfo);
    return HCCL_SUCCESS;
}

// HcclCommDfx接口实现 - 修改为返回HcclResult类型
HcclResult HcclCommDfx::ReportAllTasks(bool cachedReq)
{
    EXCEPTION_CATCH(profiling_->ReportAllTasks(cachedReq), return HCCL_E_PTR);
    return HCCL_SUCCESS;
}

HcclResult HcclCommDfx::ReportOp(uint64_t beginTime, bool cachedReq, bool isOpBase)
{
    EXCEPTION_CATCH(profiling_->ReportOp(beginTime, cachedReq, isOpBase), return HCCL_E_PTR);
    return HCCL_SUCCESS;
}

// 返回值Mc2要改
void HcclCommDfx::ReportMc2CommInfo(const Mc2CommInfo& mc2CommInfo) { profiling_->ReportMc2CommInfo(mc2CommInfo); }

HcclResult HcclCommDfx::UpdateProfStat()
{
    profiling_->UpdateProfStat();
    return HCCL_SUCCESS;
}

Hccl::MirrorTaskManager* HcclCommDfx::GetMirrorTaskManager() const { return mirrorTaskManager_.get(); }

// 将remoteRankId添加到channelRemoteRankId_表中
void HcclCommDfx::AddChannelRemoteRankId(const std::string& commTag, u64 handle, u32 remoteRankId)
{
    std::unique_lock<std::shared_mutex> rwLock(baseLock_);
    HCCL_INFO(
        "[HcclCommDfx][AddChannelRemoteRankId] commTag:[%s], handle:[%llu], remoteRankId:[%u]", commTag.c_str(), handle,
        remoteRankId);
    channelRemoteRankId_[commTag][handle] = remoteRankId;
}

// 在channelRemoteRankId_表中对remoteRankId进行查找（原有逻辑补充返回值）
HcclResult HcclCommDfx::GetChannelRemoteRankId(const std::string& commTag, u64 handle, u32& remoteRankId)
{
    std::shared_lock<std::shared_mutex> rwLock(baseLock_);
    auto commIt = channelRemoteRankId_.find(commTag);
    if (commIt == channelRemoteRankId_.end()) {
        HCCL_ERROR("[HcclCommDfx]commTag:[%s] not found", commTag.c_str());
        return HCCL_E_PARA;
    }
    auto handleIt = commIt->second.find(handle);
    if (handleIt == commIt->second.end()) {
        HCCL_ERROR("[HcclCommDfx]handle not found,commTag:[%s],handle:[%llu]", commTag.c_str(), handle);
        return HCCL_E_PARA;
    }
    remoteRankId = handleIt->second;
    return HCCL_SUCCESS; // 查找成功补充返回成功码
}

HcclResult HcclCommDfx::ReportKernel(
    uint64_t beginTime, const std::string& commTag, const std::string& kernelName, uint32_t threadId, bool cachedReq)
{
    CHK_RET(profiling_->ReportKernel(beginTime, commTag, kernelName, threadId, cachedReq));
    return HCCL_SUCCESS;
}

u32 HcclCommDfx::GetTaskId(u32 streamId)
{
    std::lock_guard<std::mutex> lock(taskIdMutex_);
    auto& taskIdRef = streamIdToTaskId_[streamId];
    constexpr u32 TASK_ID_MODULO = 65536;
    taskIdRef = (taskIdRef + 1) % TASK_ID_MODULO;
    u32 retTaskId = taskIdRef;
    return retTaskId;
}

void HcclCommDfx::SetDpuStreamId(u32 dpuStreamId) { dpuStreamId_ = dpuStreamId; }

} // namespace hccl
