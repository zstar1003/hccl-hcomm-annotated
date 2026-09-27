/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dfx_profiling_handler_lite.h"
#include "res_pub.h"
#include "log.h"
#include "sqe_a5.h"
#include "prof_sal_lite.h"
#include <limits>
#include <map>

namespace aicpu {
enum DfxStatusT : uint8_t { AICPU_ERROR_NONE = 0, AICPU_ERROR_FAILED = 1 };
DfxStatusT __attribute__((weak)) GetTaskAndStreamId(uint64_t& taskId, uint32_t& streamId);
} // namespace aicpu

namespace Hccl {

static constexpr u32 aging = 1;
constexpr std::uint32_t HCCLINFO_REPORT_BATCH_NUM = 2;
constexpr u32 TASK_ID_HIGH_WORD_SHIFT_BITS = 16;
constexpr u32 ADDR_WORD_SHIFT_BITS = 32;
DfxProfilingHandlerLite DfxProfilingHandlerLite::instance_;

static const std::map<OpTypeVal, std::string> OP_TYPE_NAME_MAP = {
    {OpTypeVal::OP_TYPE_ALLREDUCE, "OpType::ALLREDUCE"},
    {OpTypeVal::OP_TYPE_BROADCAST, "OpType::BROADCAST"},
    {OpTypeVal::OP_TYPE_ALLGATHER, "OpType::ALLGATHER"},
    {OpTypeVal::OP_TYPE_REDUCESCATTER, "OpType::REDUCESCATTER"},
    {OpTypeVal::OP_TYPE_SEND, "OpType::SEND"},
    {OpTypeVal::OP_TYPE_RECV, "OpType::RECV"},
    {OpTypeVal::OP_TYPE_BARRIER, "OpType::BARRIER"},
    {OpTypeVal::OP_TYPE_ALLTOALL, "OpType::ALLTOALL"},
    {OpTypeVal::OP_TYPE_REDUCE, "OpType::REDUCE"},
    {OpTypeVal::OP_TYPE_GATHER, "OpType::GATHER"},
    {OpTypeVal::OP_TYPE_SCATTER, "OpType::SCATTER"},
    {OpTypeVal::OP_TYPE_ALLTOALLV, "OpType::ALLTOALLV"},
    {OpTypeVal::OP_TYPE_ALLTOALLVC, "OpType::ALLTOALLVC"},
    {OpTypeVal::OP_TYPE_HALFALLTOALLV, "OpType::HALFALLTOALLV"},
    {OpTypeVal::OP_TYPE_BATCHSENDRECV, "OpType::BATCHSENDRECV"},
    {OpTypeVal::OP_TYPE_BATCHGET, "OpType::BATCHGET"},
    {OpTypeVal::OP_TYPE_BATCHPUT, "OpType::BATCHPUT"},
    {OpTypeVal::OP_TYPE_ALLGATHERV, "OpType::ALLGATHERV"},
    {OpTypeVal::OP_TYPE_REDUCESCATTERV, "OpType::REDUCESCATTERV"},
};

static const std::map<TaskParamTypeVal, std::string> TASK_PARAM_TYPE_NAME_MAP = {
    {TaskParamTypeVal::TASK_SDMA, "Memcpy"},
    {TaskParamTypeVal::TASK_RDMA, "RDMASend"},
    {TaskParamTypeVal::TASK_REDUCE_INLINE, "Reduce_Inline"},
    {TaskParamTypeVal::TASK_REDUCE_TBE, "Reduce_TBE"},
    {TaskParamTypeVal::TASK_NOTIFY_RECORD, "Notify_Record"},
    {TaskParamTypeVal::TASK_NOTIFY_WAIT, "Notify_Wait"},
    {TaskParamTypeVal::TASK_SEND_NOTIFY, "Send_Notify"},
    {TaskParamTypeVal::TASK_SEND_PAYLOAD, "Send_Payload"},
    {TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY, "Write_With_Notify"},
    {TaskParamTypeVal::TASK_UB_INLINE_WRITE, "Ub_Inline_Write"},
    {TaskParamTypeVal::TASK_UB_REDUCE_INLINE, "Ub_Reduce_Inline"},
    {TaskParamTypeVal::TASK_UB, "Ub_Write_Or_Read"},
    {TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY, "Reuce_With_Notify"},
    {TaskParamTypeVal::TASK_CCU, "Ccu"},
    {TaskParamTypeVal::TASK_AICPU_KERNEL, "AicpuKernel"},
    {TaskParamTypeVal::TASK_AICPU_REDUCE, "Aicpu_Reduce"},
    {TaskParamTypeVal::TASK_AIV, "AivKernel"},
    {TaskParamTypeVal::TASK_DPU_KERNEL, "DpuKernel"},
    {TaskParamTypeVal::TASK_DPU_THREAD_FENCE, "Dpu_ThreadFence"},
    {TaskParamTypeVal::TASK_DPU_CHANNEL_FENCE, "Dpu_ChannelFence"},
    {TaskParamTypeVal::TASK_DPU_INLINE_WRITE, "Dpu_Notify_Record"},
    {TaskParamTypeVal::TASK_DPU_NOTIFY_WAIT, "Dpu_Notify_Wait"},
    {TaskParamTypeVal::TASK_DPU_WRITE_WITH_NOTIFY, "Dpu_Write_With_Notify"},
};

static const std::map<AlgTypeVal, std::string> ALG_TYPE_NAME_MAP = {
    {AlgTypeVal::ALG_TYPE_NOT_SPECIFIED, "AlgType::NOT_SPECIFIED"},
    {AlgTypeVal::ALG_TYPE_RING, "AlgType::RING"},
    {AlgTypeVal::ALG_TYPE_MULTI_RING, "AlgType::MULTI_RING"},
    {AlgTypeVal::ALG_TYPE_MESH, "AlgType::MESH"},
    {AlgTypeVal::ALG_TYPE_RECURSIVE_HD, "AlgType::RECURSIVE_HD"},
    {AlgTypeVal::ALG_TYPE_BINARY_HD, "AlgType::BINARY_HD"},
    {AlgTypeVal::ALG_TYPE_PAIR_WISE, "AlgType::PAIR_WISE"},
};

DfxProfilingHandlerLite::DfxProfilingHandlerLite() {}

DfxProfilingHandlerLite::~DfxProfilingHandlerLite() {}

DfxProfilingHandlerLite& DfxProfilingHandlerLite::GetInstance() { return instance_; }

void DfxProfilingHandlerLite::BindProfilingHandles()
{
    if (MsprofReportBatchAdditionalInfo == nullptr) {
        if (AdprofReportAdditionalInfo != nullptr) {
            reportAdditionalInfo_ = AdprofReportAdditionalInfo;
        }
        if (AdprofGetHashId != nullptr) {
            getProfHashId_ = AdprofGetHashId;
        }
        if (AdprofReportBatchAdditionalInfo != nullptr) {
            reportBatchAdditionalInfo_ = AdprofReportBatchAdditionalInfo;
        }
    } else {
        if (MsprofReportAdditionalInfo != nullptr) {
            reportAdditionalInfo_ = [](uint32_t flag, const void* data, uint32_t len) -> int32_t {
                return MsprofReportAdditionalInfo(flag, const_cast<void*>(data), len);
            };
        }
        if (MsprofStr2Id != nullptr) {
            getProfHashId_ = MsprofStr2Id;
        }
        reportBatchAdditionalInfo_ = [](uint32_t flag, const void* data, uint32_t len) -> int32_t {
            return MsprofReportBatchAdditionalInfo(flag, const_cast<void*>(data), len);
        };
    }
}

void DfxProfilingHandlerLite::InitHashCaches()
{
    for (const auto& item : TASK_PARAM_TYPE_NAME_MAP) {
        u8 taskTypeValue = static_cast<u8>(item.first);
        taskTypeHashCache_[static_cast<uint32_t>(taskTypeValue)]
            = GetProfHashId(item.second.c_str(), item.second.length());
    }
    cachedAlgTypeHashId_ = GetProfHashId("AlgType::NHR", strlen("AlgType::NHR"));
    for (const auto& item : OP_TYPE_NAME_MAP) {
        u8 opTypeValue = static_cast<u8>(item.first);
        if (opTypeHashCache_.find(opTypeValue) == opTypeHashCache_.end()) {
            opTypeHashCache_[opTypeValue] = GetProfHashId(item.second.c_str(), item.second.length());
        }
    }
    for (const auto& item : ALG_TYPE_NAME_MAP) {
        u8 algTypeValue = static_cast<u8>(item.first);
        if (algTypeHashCache_.find(algTypeValue) == algTypeHashCache_.end()) {
            algTypeHashCache_[algTypeValue] = GetProfHashId(item.second.c_str(), item.second.length());
        }
    }
}

HcclResult DfxProfilingHandlerLite::Init()
{
    if (initializedFlag_) {
        return HCCL_SUCCESS;
    }
    BindProfilingHandles();
    if (reportAdditionalInfo_ == nullptr) {
        HCCL_ERROR(
            "[DfxProfilingHandlerLite][Init] reportAdditionalInfo_ is nullptr, profiling report will be skipped");
        return HCCL_E_PROFILING;
    }
    if (getProfHashId_ == nullptr) {
        HCCL_ERROR("[DfxProfilingHandlerLite][Init] getProfHashId_ is nullptr, profiling hash will be invalid");
        return HCCL_E_PROFILING;
    }
    InitHashCaches();
    initializedFlag_ = true;
    return HCCL_SUCCESS;
}

void DfxProfilingHandlerLite::ReportHcclOpInfo(const DfxDfxOpInfo& opInfo, const DfxCommContext& ctx) const
{
    if (!GetProfL0State()) {
        HCCL_INFO("[DfxProfilingHandlerLite][ReportHcclOpInfo] l0 is false.");
        return;
    }
    if (aicpu::GetTaskAndStreamId == nullptr) {
        HCCL_WARNING("[DfxProfilingHandlerLite][ReportHcclOpInfo] GetTaskAndStreamId is nullptr.");
        return;
    }
    uint64_t taskId = 0U;
    uint32_t streamId = 0;
    if (aicpu::GetTaskAndStreamId(taskId, streamId) != aicpu::DfxStatusT::AICPU_ERROR_NONE) {
        HCCL_ERROR("[DfxProfilingHandlerLite][ReportHcclOpInfo] Failed to get task id and stream id.");
        return;
    }
    if (taskId > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        HCCL_ERROR("[DfxProfilingHandlerLite][ReportHcclOpInfo] taskId is larger than u32.");
        return;
    }
    MsprofAdditionalInfo reporterData{};
    reporterData.level = MSPROF_REPORT_AICPU_LEVEL;
    reporterData.type = MSPROF_REPORT_AICPU_HCCL_OP_INFO;
    reporterData.threadId = static_cast<u32>(SalGetTidLite());
    reporterData.dataLen = sizeof(MsprofAicpuHCCLOPInfo);
    reporterData.timeStamp = ProfGetCurCpuTimestampLite();
    auto* hcclOpInfo = reinterpret_cast<MsprofAicpuHCCLOPInfo*>(reporterData.data);

    hcclOpInfo->algType = cachedAlgTypeHashId_;
    hcclOpInfo->taskId = static_cast<uint32_t>(taskId);
    hcclOpInfo->streamId = streamId;
    hcclOpInfo->count = opInfo.count;
    hcclOpInfo->dataType = opInfo.dataType;
    hcclOpInfo->groupName = ctx.groupName;
    hcclOpInfo->ranksize = ctx.rankSize;
    HCCL_INFO(
        "[DfxProfilingHandlerLite][ReportHcclOpInfo] relay:%u, retry:%u, dataType:%u, algType:%u, count:%llu, "
        "groupName:%lu, ranksize:%u, taskId:%u, streamId:%u",
        hcclOpInfo->relay, hcclOpInfo->retry, hcclOpInfo->dataType, hcclOpInfo->algType, hcclOpInfo->count,
        hcclOpInfo->groupName, hcclOpInfo->ranksize, hcclOpInfo->taskId, hcclOpInfo->streamId);
    ReportAdditionInfo(reporterData);
}

void DfxProfilingHandlerLite::ReportMainStreamTask(const DfxFlagTaskInfo& flagTaskInfo) const
{
    if (!GetProfL0State()) {
        HCCL_INFO("[DfxProfilingHandlerLite][ReportMainStreamTask] l0 is false.");
        return;
    }
    if (aicpu::GetTaskAndStreamId == nullptr) {
        HCCL_WARNING("[DfxProfilingHandlerLite][ReportMainStreamTask] aicpu::GetTaskAndStreamId is nullptr.");
        return;
    }
    uint64_t aicpuKernelTaskId = 0U;
    uint32_t aicpuKernelStreamId = 0;
    if (aicpu::GetTaskAndStreamId(aicpuKernelTaskId, aicpuKernelStreamId) != aicpu::DfxStatusT::AICPU_ERROR_NONE) {
        HCCL_ERROR("[DfxProfilingHandlerLite][ReportMainStreamTask] Failed to get task id and stream id.");
        return;
    }
    if (aicpuKernelTaskId > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        HCCL_ERROR("[DfxProfilingHandlerLite][ReportMainStreamTask] aicpuKernelTaskId is larger than u32.");
        return;
    }
    MsprofAdditionalInfo reporterData{};
    reporterData.level = MSPROF_REPORT_AICPU_LEVEL;
    reporterData.type = MSPROF_REPORT_AICPU_HCCL_FLAG_TASK;
    reporterData.threadId = SalGetTidLite();
    reporterData.dataLen = sizeof(MsprofAicpuHcclMainStreamTask);
    reporterData.timeStamp = ProfGetCurCpuTimestampLite();
    auto* flagtask = reinterpret_cast<MsprofAicpuHcclMainStreamTask*>(reporterData.data);
    flagtask->taskId = static_cast<uint16_t>(flagTaskInfo.taskId >> TASK_ID_HIGH_WORD_SHIFT_BITS);
    flagtask->streamId = static_cast<uint16_t>(flagTaskInfo.taskId);
    flagtask->type = flagTaskInfo.type;
    uint32_t aicpuKernelTaskIdLow32 = static_cast<uint32_t>(aicpuKernelTaskId);
    flagtask->aicpuTaskId = static_cast<uint16_t>(aicpuKernelTaskIdLow32 >> TASK_ID_HIGH_WORD_SHIFT_BITS);
    flagtask->aicpuStreamId = static_cast<uint16_t>(aicpuKernelTaskIdLow32);
    HCCL_INFO(
        "[DfxProfilingHandlerLite][ReportMainStreamTask] streamId:%u, taskId:%u, type:%u,"
        "aicpuStreamId:%u, aicpuTaskId:%u",
        flagtask->streamId, flagtask->taskId, flagtask->type, flagtask->aicpuStreamId, flagtask->aicpuTaskId);
    ReportAdditionInfo(reporterData);
}

void DfxProfilingHandlerLite::ReportAdditionInfo(const MsprofAdditionalInfo& reporterData) const
{
    if (reportAdditionalInfo_(aging, &reporterData, sizeof(MsprofAdditionalInfo)) != 0) {
        HCCL_ERROR("[DfxProfilingHandlerLite] ReportAdditionalInfo failed.");
    }
}

bool DfxProfilingHandlerLite::FillBatchReporterData(
    uint32_t batchId, const MsprofAicpuHcclTaskInfo* taskInfos, MsprofAdditionalInfo& addInfo) const
{
    addInfo.level = MSPROF_REPORT_AICPU_LEVEL;
    addInfo.type = MSPROF_REPORT_AICPU_MC2_BATCH_HCCL_INFO;
    addInfo.threadId = SalGetTidLite();
    addInfo.timeStamp = 0;
    addInfo.dataLen = sizeof(MsprofAicpuHcclTaskInfo) * batchId;
    s32 sret = memcpy_s(addInfo.data, sizeof(addInfo.data), taskInfos, addInfo.dataLen);
    if (sret != 0) {
        HCCL_WARNING("[DfxProfilingHandlerLite][FillBatchReporterData] memcpy failed, sret[%d]", sret);
        return false;
    }
    return true;
}

bool DfxProfilingHandlerLite::ReportBatchAddInfo(
    uint32_t batchId, const MsprofAicpuHcclTaskInfo* taskInfos, MsprofAdditionalInfo* addInfoVec, uint32_t& addInfoIndx,
    uint32_t maxBatchNum, bool isLastBatch) const
{
    if (!FillBatchReporterData(batchId, taskInfos, addInfoVec[addInfoIndx])) {
        return false;
    }
    addInfoIndx++;
    if (addInfoIndx == maxBatchNum || isLastBatch) {
        if (reportBatchAdditionalInfo_(aging, addInfoVec, addInfoIndx * sizeof(MsprofAdditionalInfo)) != 0) {
            HCCL_WARNING("[DfxProfilingHandlerLite][ReportStreamTaskDetails] reportBatchAdditionalInfo failed");
            return false;
        }
        addInfoIndx = 0;
    }
    return true;
}

void DfxProfilingHandlerLite::UpdateProfSwitch()
{
    IsProfSwitchOn(DfxProfilingLevel::L0);
    IsProfSwitchOn(DfxProfilingLevel::L1);
}

bool DfxProfilingHandlerLite::IsProfOn(uint64_t feature) const
{
    if (MsprofReportBatchAdditionalInfo == nullptr) {
        if (AdprofCheckFeatureIsOn == nullptr) {
            return false;
        }
        return AdprofCheckFeatureIsOn(feature) > 0;
    } else {
        if (feature == ADPROF_TASK_TIME_L1) {
            return enableHcclL1_;
        } else if (feature == ADPROF_TASK_TIME_L0) {
            return enableHcclL0_;
        }
    }
    return false;
}

bool DfxProfilingHandlerLite::IsProfSwitchOn(DfxProfilingLevel level)
{
    bool res = false;
    if (level == DfxProfilingLevel::L0) {
        res = IsProfOn(ADPROF_TASK_TIME_L0);
        enableHcclL0_ = res;
    } else if (level == DfxProfilingLevel::L1) {
        res = IsProfOn(ADPROF_TASK_TIME_L1);
        enableHcclL1_ = res;
    }
    return res;
}

void DfxProfilingHandlerLite::SetProL1On(bool val)
{
    HCCL_INFO("[%s] val = [%d]", __func__, val);
    enableHcclL1_ = val;
}

void DfxProfilingHandlerLite::SetProL0On(bool val)
{
    HCCL_INFO("[%s] val = [%d]", __func__, val);
    enableHcclL0_ = val;
}

uint64_t DfxProfilingHandlerLite::GetProfHashId(const char* name, uint32_t len) const
{
    if (name == nullptr || len == 0) {
        HCCL_WARNING("HashData is empty.  name:%s, len:%u", name, len);
        return DFX_INVALID_U64;
    }
    if (getProfHashId_ != nullptr) {
        return getProfHashId_(name, len);
    }
    return DFX_INVALID_U64;
}

uint64_t DfxProfilingHandlerLite::GetCachedAlgTypeHashId() const { return cachedAlgTypeHashId_; }

void DfxProfilingHandlerLite::GetTaskDetailInfosFromDfxTaskInfo(
    const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos, const DfxCommContext& ctx) const
{
    if (!it->IsTaskTypeValid()) {
        HCCL_WARNING("[DfxProfilingHandlerLite] invalid taskType[%u], skip task detail", it->taskType);
        return;
    }

    auto cacheIt = taskTypeHashCache_.find(static_cast<uint32_t>(it->taskType));
    taskDetailsInfos.itemId = (cacheIt != taskTypeHashCache_.end()) ? cacheIt->second : DFX_INVALID_U64;

    FillCclTagAndRemoteRank(it, taskDetailsInfos, ctx);

    taskDetailsInfos.groupName = ctx.groupName;
    taskDetailsInfos.localRank = ctx.localRank;
    taskDetailsInfos.rankSize = ctx.rankSize;
    taskDetailsInfos.stage = 0;
    taskDetailsInfos.timeStamp = ProfGetCurCpuTimestampLite();
    taskDetailsInfos.durationEstimated = 0;

    switch (static_cast<TaskParamTypeVal>(it->taskType)) {
        case TaskParamTypeVal::TASK_SDMA:
        case TaskParamTypeVal::TASK_RDMA:
            FillSdmaRdmaDetail(it, taskDetailsInfos);
            break;
        case TaskParamTypeVal::TASK_REDUCE_INLINE:
            FillReduceInlineDetail(it, taskDetailsInfos);
            break;
        case TaskParamTypeVal::TASK_UB:
        case TaskParamTypeVal::TASK_UB_INLINE_WRITE:
        case TaskParamTypeVal::TASK_UB_REDUCE_INLINE:
        case TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY:
        case TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY:
            FillUbDmaDetail(it, taskDetailsInfos);
            break;
        case TaskParamTypeVal::TASK_NOTIFY_RECORD:
        case TaskParamTypeVal::TASK_NOTIFY_WAIT:
            FillNotifyDetail(it, taskDetailsInfos);
            break;
        default:
            FillDefaultDetail(it, taskDetailsInfos);
            break;
    }

    FillCommonTailFields(it, taskDetailsInfos);
}

void DfxProfilingHandlerLite::ReportStreamTaskDetailsLog(TaskInfoCircularQueue& taskQueue) const
{
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO) == 0)) {
        return;
    }
    u16 begin = taskQueue.GetBegin();
    u16 count = taskQueue.GetCount();
    for (u16 idx = begin, i = 0; i < count; idx = (idx + 1) % taskQueue.GetCapacity(), i++) {
        DfxTaskInfo* ptr = taskQueue.GetSlot(idx);
        if (ptr == nullptr) {
            continue;
        }
        HCCL_INFO(
            "[DfxProfilingHandlerLite] DumpTaskDetails sqId:%u, taskId:%u, taskType:%u", ptr->sqId, ptr->taskId,
            ptr->taskType);
    }
}

void DfxProfilingHandlerLite::ReportStreamTaskDetails(TaskInfoCircularQueue& taskQueue, const DfxCommContext& ctx) const
{
    u16 begin = taskQueue.GetBegin();
    u16 count = taskQueue.GetCount();
    if (taskQueue.IsEmpty()) {
        return;
    }

    ReportStreamTaskDetailsLog(taskQueue);

    MsprofAicpuHcclTaskInfo taskInfos[HCCLINFO_REPORT_BATCH_NUM] = {};
    bool isSupportBatchReport = (reportBatchAdditionalInfo_ != nullptr);
    constexpr int32_t MAX_BATCH_REPORT_NUM = 512;
    MsprofAdditionalInfo addInfoVec[MAX_BATCH_REPORT_NUM] = {};
    uint32_t addInfoIndx = 0;
    uint32_t batchId = 0;
    u32 totalTasks = count;
    u32 reportedTasks = 0;

    for (u16 idx = begin, i = 0; i < count; idx = (idx + 1) % taskQueue.GetCapacity(), i++) {
        DfxTaskInfo* ptr = taskQueue.GetSlot(idx);
        if (ptr == nullptr) {
            continue;
        }
        GetTaskDetailInfosFromDfxTaskInfo(ptr, taskInfos[batchId++], ctx);
        reportedTasks++;
        if (batchId == HCCLINFO_REPORT_BATCH_NUM || reportedTasks == totalTasks) {
            if (!isSupportBatchReport) {
                MsprofAdditionalInfo reporterData{};
                if (!FillBatchReporterData(batchId, taskInfos, reporterData)) {
                    return;
                }
                ReportAdditionInfo(reporterData);
            } else {
                if (!ReportBatchAddInfo(
                        batchId, taskInfos, addInfoVec, addInfoIndx, MAX_BATCH_REPORT_NUM,
                        reportedTasks == totalTasks)) {
                    return;
                }
            }
            batchId = 0;
            memset_s(taskInfos, sizeof(taskInfos), 0, sizeof(taskInfos));
        }
    }
}

void DfxProfilingHandlerLite::FillReduceInlineDetail(
    const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos) const
{
    taskDetailsInfos.notifyID = it->taskPara.Reduce.notifyId;
    taskDetailsInfos.srcAddr = it->taskPara.Reduce.srcAddr;
    taskDetailsInfos.dstAddr = it->taskPara.Reduce.dstAddr;
    taskDetailsInfos.dataSize = static_cast<uint64_t>(it->taskPara.Reduce.size);
    taskDetailsInfos.linkType = it->linkType;
    taskDetailsInfos.opType = it->taskPara.Reduce.reduceOp;
}

void DfxProfilingHandlerLite::FillSdmaRdmaDetail(const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos) const
{
    taskDetailsInfos.linkType = it->linkType;
    taskDetailsInfos.opType = 0;
    taskDetailsInfos.notifyID = DFX_INVALID_U64;
    void* sqePtr = reinterpret_cast<void*>(it->taskPara.Dma.sqeAddr);
    if (sqePtr != nullptr) {
        auto* header = reinterpret_cast<Hccl::Rt91095StarsSqeHeader*>(sqePtr);
        if (static_cast<Hccl::Rt91095StarsSqeType>(header->type) == Hccl::Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA) {
            auto* dmaSqe = reinterpret_cast<Hccl::Rt91095StarsMemcpySqe*>(sqePtr);
            taskDetailsInfos.srcAddr = (static_cast<u64>(dmaSqe->u.strideMode0.srcAddrHigh) << ADDR_WORD_SHIFT_BITS)
                                       | dmaSqe->u.strideMode0.srcAddrLow;
            taskDetailsInfos.dstAddr = (static_cast<u64>(dmaSqe->u.strideMode0.dstAddrHigh) << ADDR_WORD_SHIFT_BITS)
                                       | dmaSqe->u.strideMode0.dstAddrLow;
            taskDetailsInfos.dataSize = dmaSqe->u.strideMode0.lengthMove;
        }
    }
}

void DfxProfilingHandlerLite::FillUbDmaDetail(const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos) const
{
    taskDetailsInfos.notifyID = it->taskPara.ubDma.notifyId;
    taskDetailsInfos.srcAddr = it->taskPara.ubDma.srcAddr;
    taskDetailsInfos.dstAddr = it->taskPara.ubDma.dstAddr;
    taskDetailsInfos.dataSize = static_cast<uint64_t>(it->taskPara.ubDma.size);
    taskDetailsInfos.linkType = it->linkType;
    if (it->taskType == static_cast<u8>(TaskParamTypeVal::TASK_UB_REDUCE_INLINE)
        || it->taskType == static_cast<u8>(TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY)) {
        taskDetailsInfos.opType = it->taskPara.Reduce.reduceOp;
    } else {
        taskDetailsInfos.opType = 0;
    }
}

void DfxProfilingHandlerLite::FillNotifyDetail(const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos) const
{
    taskDetailsInfos.linkType = it->linkType;
    taskDetailsInfos.notifyID = DFX_INVALID_U64;
    void* sqePtr = reinterpret_cast<void*>(it->taskPara.Notify.sqeAddr);
    if (sqePtr != nullptr) {
        auto* header = reinterpret_cast<Hccl::Rt91095StarsSqeHeader*>(sqePtr);
        if (static_cast<Hccl::Rt91095StarsSqeType>(header->type)
                == Hccl::Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD
            || static_cast<Hccl::Rt91095StarsSqeType>(header->type)
                   == Hccl::Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT) {
            auto* notifySqe = reinterpret_cast<Hccl::Rt91095StarsNotifySqe*>(sqePtr);
            taskDetailsInfos.notifyID = notifySqe->notifyId;
        }
    }
}

void DfxProfilingHandlerLite::FillDefaultDetail(const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos) const
{
    taskDetailsInfos.linkType = it->linkType;
    taskDetailsInfos.notifyID = DFX_INVALID_U64;
}

void DfxProfilingHandlerLite::FillCclTagAndRemoteRank(
    const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos, const DfxCommContext& ctx) const
{
    if (it->dfxOpInfo != DFX_INVALID_U64) {
        auto* opInfo = reinterpret_cast<const DfxDfxOpInfo*>(it->dfxOpInfo);
        auto opTypeCclTagIt = opTypeHashCache_.find(opInfo->opType);
        taskDetailsInfos.cclTag = (opTypeCclTagIt != opTypeHashCache_.end()) ? opTypeCclTagIt->second : DFX_INVALID_U64;
        taskDetailsInfos.remoteRank = INVALID_U32;
        if (ctx.channelRemoteRankIdMap != nullptr && it->channelHandle != DFX_INVALID_U64) {
            auto rankIt = ctx.channelRemoteRankIdMap->find(it->channelHandle);
            if (rankIt != ctx.channelRemoteRankIdMap->end()) {
                taskDetailsInfos.remoteRank = rankIt->second;
            }
        }
    } else {
        taskDetailsInfos.cclTag = DFX_INVALID_U64;
        taskDetailsInfos.remoteRank = INVALID_U32;
    }
}

void DfxProfilingHandlerLite::FillCommonTailFields(
    const DfxTaskInfo* it, MsprofAicpuHcclTaskInfo& taskDetailsInfos) const
{
    taskDetailsInfos.taskId = it->taskId;
    taskDetailsInfos.streamId = it->sqId;
    taskDetailsInfos.planeID = 0;
    if (it->dfxOpInfo != DFX_INVALID_U64) {
        taskDetailsInfos.dataType = reinterpret_cast<const DfxDfxOpInfo*>(it->dfxOpInfo)->dataType;
    }
    taskDetailsInfos.transportType = static_cast<int32_t>(it->transportType);
    if (taskDetailsInfos.remoteRank == Hccl::DFX_INVALID_RANKID) {
        taskDetailsInfos.remoteRank = taskDetailsInfos.localRank;
    }
    taskDetailsInfos.rdmaType = 0;
    taskDetailsInfos.role = static_cast<uint32_t>(DfxTaskRole::NEW_TASK_ROLE_DST);
    taskDetailsInfos.workFlowMode = static_cast<uint32_t>(DfxWorkflowMode::NEW_WORKFLOW_MODE_OP_BASE);
}

} // namespace Hccl
