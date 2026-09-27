/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_PROFILING_HANDLER_H
#define HCCL_PROFILING_HANDLER_H
#include <unordered_map>
#include <queue>
#include <mutex>
#include <atomic>
#include "hccl/hccl_types.h"
#include "task_info.h"
#include "rt_external.h"
#include "profiling_common.h"
#include "stream_manager.h"
#include "task_param.h"

namespace Hccl {
MAKE_ENUM(kernelType, AICPU_KERNEL = 0, CCU_KERNEL);

// ccu 上报数据结构
constexpr unsigned int MSPROF_REPORT_CCU_TASK_INFO = 14U;
constexpr unsigned int MSPROF_REPORT_CCU_WAIT_SIGNAL_INFO = 15U;
constexpr unsigned int MSPROF_REPORT_CCU_GROUP_INFO = 16U;
constexpr uint8_t INVALID_TYPE_VALUE = 0xFF; // reduceOpType、inputDataType、outputDataType非法值

MAKE_ENUM(ProfTaskType, TASK_HCCL_INFO, TASK_DPU_HCCL_INFO);

struct MsprofCcuTaskInfo {
    uint8_t version;
    uint8_t workFlowMode;
    uint64_t itemId;    // CCU任务名 hash id
    uint64_t groupName; // 通信域 hash id
    uint32_t rankId;
    uint32_t ranksize; // CCU任务设计的Chip数目

    uint16_t streamId;
    uint32_t taskId;
    uint8_t dieId;     // CCU任务执行的DieId
    uint8_t missionId; // CCU任务执行的MissionId
    uint16_t instrId;
};

struct MsprofCcuGroupInfo {
    uint8_t version;
    uint64_t itemId;    // CCU任务名 hash id
    uint64_t groupName; // 通信域 hash id
    uint32_t rankId;
    uint32_t ranksize; // CCU任务设计的Chip数目
    uint8_t workFlowMode;

    uint16_t streamId;
    uint32_t taskId;
    uint8_t dieId; // CCU任务执行的DieId
    uint16_t instrId;
    uint8_t missionId; // CCU任务执行的MissionId

    uint8_t reduceOpType;   // 与HcclReduceOp类型保持一致
    uint8_t inputDataType;  // 与HcclDataType类型保持一致
    uint8_t outputDataType; // 与HcclDataType类型保持一致
    uint64_t dataSize;      // 输入数据大小

    uint16_t channelId[CCU_MAX_CHANNEL_NUM];    // LoopGroup所包含的搬运指令使用的ChannelId
    uint32_t remoteRankId[CCU_MAX_CHANNEL_NUM]; // LoopGroup所包含的搬运指令的对端
};

struct MsprofCcuWaitSignalInfo {
    uint8_t version;
    uint64_t itemId;    // CCU任务名 hash id
    uint64_t groupName; // 通信域 hash id
    uint32_t rankId;
    uint32_t ranksize; // CCU任务设计的Chip数目
    uint8_t workFlowMode;

    uint16_t streamId;
    uint32_t taskId;
    uint8_t dieId; // CCU任务执行的DieId
    uint16_t instrId;
    uint8_t missionId; // CCU任务执行的MissionId

    uint32_t ckeId;
    uint32_t mask;
    uint16_t channelId[CCU_MAX_CHANNEL_NUM];    // LoopGroup所包含的搬运指令使用的ChannelId
    uint32_t remoteRankId[CCU_MAX_CHANNEL_NUM]; // LoopGroup所包含的搬运指令的对端
};

struct HCCLReportData {
    std::string fileTag;
    uint64_t ts;
    uint32_t type;
    MsprofHcclInfo profInfo;
    MsprofDpuHcclTrack dpuProfInfo;
    std::string tag;
    std::string groupName;
};

const std::map<OpType, std::string> PROF_OP_NAME_V2
    = {{OpType::INVALID, "hcom_invalid_"},
       {OpType::ALLREDUCE, "hcom_allReduce_"},
       {OpType::BROADCAST, "hcom_broadcast_"},
       {OpType::REDUCE, "hcom_reduce_"},
       {OpType::SEND, "hcom_send_"},
       {OpType::RECV, "hcom_receive_"},
       {OpType::ALLGATHER, "hcom_allGather_"},
       {OpType::REDUCESCATTER, "hcom_reduceScatter_"},
       {OpType::SCATTER, "hcom_scatter_"},
       {OpType::ALLTOALL, "hcom_alltoall_"},
       {OpType::ALLTOALLV, "hcom_alltoallv_"},
       {OpType::ALLGATHERV, "hcom_allGatherv_"},
       {OpType::REDUCESCATTERV, "hcom_reduceScatterv_"},
       {OpType::ALLTOALLVC, "hcom_alltoallvc_"},
       {OpType::BATCHSENDRECV, "hcom_batchSendRecv_"},
       {OpType::BATCHPUT, "hccl_batchPut_"},
       {OpType::BATCHGET, "hccl_batchGet_"},
       {OpType::DEBUGCASE, "hccl_debugCase_"},
       {OpType::BARRIER, "hccl_barrier_"},
       {OpType::HALFALLTOALLV, "hccl_halfAlltoallv_"},
       {OpType::HCCLGROUPOP, "hccl_groupOp_"}};

inline std::string GetProfOpName(OpType opType)
{
    CHK_PRT_RET(PROF_OP_NAME_V2.empty(), HCCL_ERROR("PROF_OP_NAME_V2 has not inited."), "hcom_invalid_");
    auto it = PROF_OP_NAME_V2.find(opType);
    if (it != PROF_OP_NAME_V2.end()) {
        return it->second;
    }
    return PROF_OP_NAME_V2.begin()->second;
}

class ProfilingHandler {
public:
    ~ProfilingHandler();

    ProfilingHandler(const ProfilingHandler& that) = delete;

    ProfilingHandler& operator=(const ProfilingHandler& that) = delete;

    static ProfilingHandler& GetInstance();

    static int32_t CommandHandleWrapper(uint32_t rtType, void* data, uint32_t len);

    void ReportKernel() const;

    void ReportHostApi(OpType opType, uint64_t beginTime, uint64_t endTime, bool cachedReq, bool isAiCpu);

    void ReportHcclOp(const DfxOpInfo& opInfo, bool cachedReq);

    void ReportHcclTaskApi(
        TaskParamType taskType, uint64_t beginTime, uint64_t endTime, bool isMasterStream, bool cachedReq,
        bool ignoreLevel = false);

    void ReportHcclTaskDetails(const TaskInfo& taskInfo, bool cachedReq);
    void ReportHcclTaskDetailsBatch(const std::vector<TaskInfo*>& taskInfos, bool cachedReq);

    bool GetHostApiState() const;
    bool GetHcclNodeState() const;
    bool GetHcclL0State() const;
    bool GetHcclL1State() const;
    inline void SetOpModeFlags(bool isOpBase, bool isCached)
    {
        isOpbase_ = isOpBase;
        isCached_ = isCached;
    }
    inline bool GetOpBaseFlag() const { return isOpbase_; }
    inline bool GetCachedFlag() const { return isCached_; }
    int32_t CommandHandle(uint32_t rtType, void* data, uint32_t len) const;
    HcclResult Init();
    void ReportHcclMC2CommInfo(
        const Stream& kfcStream, const Stream& stream, const std::vector<Stream*>& aicpuStreams, const std::string& id,
        RankId myRank, u32 rankSize, RankId rankInParentComm);
    void ReportHcclMC2CommInfo(
        const u32 kfcStreamId, const std::vector<u32>& aicpuStreamsId, const std::string& id, RankId myRank,
        u32 rankSize, RankId rankInParentComm);
    void ReportNodeApi(uint64_t beginTime, uint64_t endTime, uint64_t cmdItemId, uint32_t threadId, bool cachedReq);
    void ReportNodeBasicInfo(uint64_t timeStamp, uint64_t cmdItemId, uint32_t threadId, bool cachedReq);
    uint64_t GetProfHashId(const char* name, uint32_t len) const;
    uint64_t GetCachedAlgTypeHashId() const { return cachedAlgTypeHashId_.load(); }

private:
    explicit ProfilingHandler();

    void ReportAclApi(
        uint32_t cmdType, uint64_t beginTime, uint64_t endTime, uint64_t cmdItemId, uint32_t threadId, bool cachedReq);

    void ReportHcclOpInfo(uint64_t timeStamp, const DfxOpInfo& opInfo, uint32_t threadId, bool cachedReq);
    void ReportAdditionInfo(MsprofAdditionalInfo& reporterData) const;

    void StartSubscribe(uint64_t profconfig);
    void StartTaskApiSubscribe();
    void StartHostApiSubscribe();
    void StartAdditionInfoSubscribe();
    void StartHostHcclOpSubscribe();
    void StartCcuSubscribe();
    void StopSubscribe();

    void CallProfRegHostApi() const;
    void ReportStoragedCompactInfo();
    void ReportStoragedNodeBasicInfo();
    void ReportMc2AdditionInfo();

    void CallProfRegTaskTypeApi() const;
    void ReportStoragedTaskApi();
    void ReportStoragedAclApi();

    void CallProfRegHcclOpApi() const;

    void ReportStoragedAdditionInfo();

    void GetHCCLReportData(const TaskInfo& taskInfo, HCCLReportData& hcclReportData) const;
    void FillProfCommonInfo(const TaskInfo& taskInfo, MsprofAdditionalInfo& reporterData) const;
    void FillProfTaskSpecificInfo(const TaskInfo& taskInfo, MsprofHcclInfo* profInfo) const;
    void FillDpuProfInfo(const TaskInfo& taskInfo, MsprofAdditionalInfo& reporterData) const;
    void FillDpuTaskParaDetails(const TaskInfo& taskInfo, MsprofDpuHcclTrack* dpuProfInfo) const;
    void ConvertHcclInfoToDpuTrack(MsprofAdditionalInfo& reporterData) const;
    void FillTaskAdditionInfo(const TaskInfo& taskInfo, MsprofAdditionalInfo& reporterData) const;
    uint32_t GetTaskTypeValue(TaskParamType taskType) const;
    void CallAdditionInfo(MsprofAdditionalInfo& reporterData) const;

    void ReportCcuInfo(const TaskInfo& taskInfo) const;
    void GetCcuTaskInfo(const TaskInfo& taskInfo, const CcuProfilingInfo& info) const;
    void GetCcuWaitSignalInfo(const TaskInfo& taskInfo, const CcuProfilingInfo& info) const;
    void GetCcuGroupInfo(const TaskInfo& taskInfo, const CcuProfilingInfo& info) const;

    void DumpHCCLReportData(const TaskInfo& taskInfo, const MsprofAdditionalInfo& reporterData) const;
    void DumpCcuGroupInfo(const MsprofCcuGroupInfo& ccuGroupInfo) const;
    void ReportMc2AdditionInfo(uint64_t timeStamp, const void* data, int len);
    void SetCachedCclTag();
    void InitLog() const;
    void ReportHcclMC2CommInfoLog(
        const Stream& kfcStream, const Stream& stream, const std::vector<Stream*>& aicpuStreams, const std::string& id,
        RankId myRank, u32 rankSize, RankId rankInParentComm) const;
    void ReportHcclMC2CommInfoLog(
        const u32 kfcStreamId, const std::vector<u32>& aicpuStreamsId, const std::string& id, RankId myRank,
        u32 rankSize, RankId rankInParentComm) const;
    void ReportCcuInfoLog(const TaskInfo& taskInfo) const;
    void LogCcuTaskInfo(
        const CcuProfilingInfo& info, const TaskInfo& taskInfo, uint64_t itemId, uint64_t groupName, u32 rankId,
        u32 ranksize) const;
    void LogCcuWaitSignalInfo(
        const CcuProfilingInfo& info, const TaskInfo& taskInfo, uint64_t itemId, uint64_t groupName, u32 rankId,
        u32 ranksize) const;
    void LogCcuGroupInfo(
        const CcuProfilingInfo& info, const TaskInfo& taskInfo, uint64_t itemId, uint64_t groupName, u32 rankId,
        u32 ranksize) const;
    void ReportHcclTaskDetailsBatchLog(const std::vector<TaskInfo*>& taskInfos) const;
    void ReportStoragedAdditionInfoLog() const;

private:
    static ProfilingHandler instance_;
    bool initializedFlag_{false};
    bool enableHostApi_{false};
    bool enableHcclNode_{false};
    bool enableHcclL0_{false};
    bool enableHcclL1_{false};
    bool isOpbase_{false};
    bool isCached_{false};

    std::vector<TaskInfo> cacheTaskInfos_{};
    std::queue<MsprofApi> cachedTaskApiInfo_{};
    std::queue<MsprofApi> cachedAclApiInfo_{};
    std::queue<MsprofCompactInfo> cacheHcclOpInfo_{};
    std::queue<MsprofCompactInfo> cacheNodeBasicInfo_{};
    std::queue<MsprofAdditionalInfo> cacheHcclAdditionInfo_{};
    std::atomic<uint64_t> cachedAlgTypeHashId_{0};

    std::map<uint32_t, uint64_t> cachedNewCclTag_{};
    mutable std::mutex cacheTaskInfosMutex_;
    std::mutex cachedTaskApiInfoMutex_;
    std::mutex cachedAclApiInfoMutex_;
    std::mutex cacheHcclOpInfoMutex_;
    std::mutex cacheNodeBasicInfoMutex_;
    std::mutex cacheHcclAdditionInfoMutex_;
};
} // namespace Hccl

#endif // HCCL_PROFILING_HANDLER_H
