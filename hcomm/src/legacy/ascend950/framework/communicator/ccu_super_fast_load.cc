/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_super_fast_load.h"

#include "exception_util.h"
#include "internal_exception.h"
#include "communicator_impl.h"
#include "ccu_jetty_mgr.h"
#include "coll_service_device_mode.h"

namespace Hccl {

static void SFLReportCcuProfilingInfoInitPart(
    uint64_t execId, std::vector<CcuProfilingInfo>&& streamProfilingInfo, const CommunicatorImpl& comm,
    TaskParam& taskParam)
{
    if (streamProfilingInfo.empty()) {
        HCCL_INFO("There is no ccu profiling info.");
        return;
    }
    taskParam.taskPara.Ccu.dieId = streamProfilingInfo[0].dieId;
    taskParam.taskPara.Ccu.missionId = streamProfilingInfo[0].missionId;
    taskParam.taskPara.Ccu.execMissionId = streamProfilingInfo[0].missionId;
    taskParam.taskPara.Ccu.instrId = streamProfilingInfo[0].instrId;
    taskParam.taskPara.Ccu.executeId = execId;

    CcuJettyMgr* ccuJettyMgr = dynamic_cast<CollServiceDeviceMode*>(comm.GetCollService())
                                   ->GetCcuInsPreprocessor()
                                   ->GetCcuComm()
                                   ->GetCcuJettyMgr();
    for (auto& profInfo : streamProfilingInfo) {
        for (int idx = 0; idx < CCU_MAX_CHANNEL_NUM; idx++) {
            if (profInfo.channelId[idx] == INVALID_VALUE_CHANNELID) {
                break;
            }
            profInfo.remoteRankId[idx]
                = ccuJettyMgr->GetRemoteRankIdByChannelId(profInfo.dieId, profInfo.channelId[idx]);
        }
    }
    taskParam.ccuDetailInfo = std::make_shared<std::vector<CcuProfilingInfo>>(std::move(streamProfilingInfo));
}

CachedCCUParams::CachedCCUParams(
    std::vector<std::vector<Hccl::CcuTaskParam>>&& ccuInstruction,
    std::vector<std::vector<CcuProfilingInfo>>&& profilingInfo, std::size_t execId, CcuInstType insType, bool isSlave,
    void* comm)
    : execId(execId),
      insType(insType),
      isSlave(isSlave)
{
    std::vector<std::vector<rtCcuTaskInfo_t>> ccuTaskInstruction{};
    auto& commImpl = *(static_cast<CommunicatorImpl*>(comm));
    for (auto& vec : ccuInstruction) {
        std::vector<rtCcuTaskInfo_t> ccuTaskVec{};
        for (auto& ccuTask : vec) {
            rtCcuTaskInfo_t taskInfo{};
            taskInfo.dieId = ccuTask.dieId;
            taskInfo.missionId = ccuTask.missionId;
            taskInfo.instStartId = ccuTask.instStartId;
            taskInfo.instCnt = ccuTask.instCnt;
            taskInfo.key = ccuTask.key;
            taskInfo.argSize = ccuTask.argSize;
            taskInfo.timeout = commImpl.GetNotifyTimeoutCfg().GetNotifyTimeout();
            std::copy(std::begin(ccuTask.args), std::end(ccuTask.args), std::begin(taskInfo.args));
            ccuTaskVec.push_back(taskInfo);
        }
        ccuTaskInstruction.push_back(ccuTaskVec);
    }
    constexpr std::size_t alignment = alignof(std::max_align_t);
    ccuParams = alloc_and_memcpy_aligned(
        std::forward<std::vector<std::vector<rtCcuTaskInfo_t>>>(ccuTaskInstruction), alignment);
    taskParams.reserve(ccuInstruction.size());
    for (std::size_t i = 0; i < ccuInstruction.size(); i++) {
        TaskParam taskParam = {};
        taskParam.taskType = TaskParamType::TASK_CCU;
        SFLReportCcuProfilingInfoInitPart(
            execId, std::forward<std::vector<CcuProfilingInfo>>(profilingInfo[i]), commImpl, taskParam);
        taskParams.emplace_back(std::move(taskParam));
    }
    HCCL_RUN_INFO("Save CcuInstType: %d", insType);
}

CachedCCUParams::CachedCCUParams(CachedCCUParams&& other) noexcept
    : ccuParams(std::exchange(other.ccuParams, nullptr)),
      count(std::move(other.count)),
      taskParams(std::move(other.taskParams)),
      execId(other.execId),
      totalCounts(other.totalCounts),
      insType(other.insType),
      isSlave(other.isSlave)
{}

CachedCCUParams& CachedCCUParams::operator=(CachedCCUParams&& other) noexcept
{
    if (this != &other) {
        aligned_free(ccuParams);
        ccuParams = std::exchange(other.ccuParams, nullptr);
        execId = other.execId;
        count = std::move(other.count);
        totalCounts = other.totalCounts;
        isSlave = other.isSlave;
        taskParams = std::move(other.taskParams);
        insType = other.insType;
    }
    return *this;
}

CachedCCUParams::~CachedCCUParams() { aligned_free(ccuParams); }

rtCcuTaskInfo_t*
CachedCCUParams::alloc_and_memcpy_aligned(const std::vector<std::vector<rtCcuTaskInfo_t>>& vecs, std::size_t alignment)
{
    if (alignment < alignof(CcuTaskParam)) {
        THROW<InternalException>(StringFormat("[CachedCCUParams] alignment must be larger than type alignment."));
    }
    count.resize(vecs.size());
    std::size_t countIndex = 0;
    count[countIndex++] = vecs[0].size();
    for (const auto& vec : vecs) {
        totalCounts += vec.size();
        HCCL_INFO("CachedCCUParams: vec.size[%llu]", static_cast<std::uint64_t>(vec.size()));
    }
    if (totalCounts == 0) {
        THROW<InternalException>(StringFormat("[CachedCCUParams] total count is zero."));
    }
    HCCL_INFO("CachedCCUParams: totalCounts[%llu]", static_cast<std::uint64_t>(totalCounts));
    std::size_t bytes = totalCounts * sizeof(rtCcuTaskInfo_t);
    void* raw = alloc_aligned_raw(alignment, bytes);
    if (!raw) {
        THROW<InternalException>(StringFormat("[CachedCCUParams] failed to allocate memory, size %zu.", bytes));
    }
    rtCcuTaskInfo_t* dst = static_cast<rtCcuTaskInfo_t*>(raw);
    rtCcuTaskInfo_t* cur = dst;
    if (!vecs[0].empty()) {
        auto ret
            = memcpy_s(cur, count[0] * sizeof(rtCcuTaskInfo_t), vecs[0].data(), count[0] * sizeof(rtCcuTaskInfo_t));
        if (ret != EOK) {
            aligned_free(dst);
            THROW<InternalException>(StringFormat("[CachedCCUParams] failed to memcpy, ret %d.", ret));
        }
        cur += count[0];
    }
    u32 reqStreamNum = vecs.size() - 1;
    for (std::size_t i = 1; i <= reqStreamNum; i++) {
        auto& vec = vecs[i];
        if (!vec.empty()) {
            auto ret
                = memcpy_s(cur, vec.size() * sizeof(rtCcuTaskInfo_t), vec.data(), vec.size() * sizeof(rtCcuTaskInfo_t));
            if (ret != EOK) {
                aligned_free(dst);
                THROW<InternalException>(StringFormat("[CachedCCUParams] failed to memcpy, ret %d.", ret));
            }
            count[countIndex++] = vec.size();
            cur += vec.size();
        }
    }
    for (std::size_t i = 0; i < count.size(); ++i) {
        HCCL_INFO("CachedCCUParams: count value[%llu]", static_cast<std::uint64_t>(count[i]));
    }
    return dst;
}
} // namespace Hccl
