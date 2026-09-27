/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_TASKEXCEPTION_LIFT_H
#define HCCL_COMM_TASKEXCEPTION_LIFT_H

#include "daemon_func.h"
#include "res_pub.h"
#include "coll_comm_aicpu.h"
#include "aicpu_hccl_sqcq.h"
#include "error_message_v2.h"
#include "coll_comm_aicpu_mgr.h"

namespace Hccl {
template <typename T, uint32_t CAPACITY>
class DfxCircularQueue;
}

namespace hcomm {

class HcclCommTaskExceptionLite : public Hccl::DaemonFunc {
public:
    static HcclCommTaskExceptionLite& GetInstance();
    void Init(u32 devId);
    void Call() override;
    HcclResult PrintAllCommTaskException(); // 打印所有通信域所有流的task信息
    HcclResult PrintCommTaskException(CollCommAicpu* aicpuComm);

private:
    HcclCommTaskExceptionLite() = default;
    ~HcclCommTaskExceptionLite() override = default;

    // 检测流上的异常cqe并进行打印和上报
    HcclResult HandleExceptionCqe();
    HcclResult GetThreadCqe(hccl::Thread* thread, rtLogicCqReport_t& cqeException, CqeStatus& cqeStatus);
    HcclResult ProcessCqe(
        CollCommAicpu* aicpuComm, const rtLogicCqReport_t& exceptionInfo, const CqeStatus& cqeStatus,
        const std::vector<std::pair<std::string, CollCommAicpu*>>& aicpuCommInfo);

    // errMsg上报到host
    HcclResult ReportErrMsg(CollCommAicpu* aicpuComm, const rtLogicCqReport_t& exceptionInfo);
    HcclResult GenerateErrorMessageReport(
        CollCommAicpu* aicpuComm, const Hccl::DfxTaskInfo& taskInfo, const rtLogicCqReport_t& exceptionInfo,
        Hccl::ErrorMessageReport& errMsgInfo);
    void GenerateTaskErrMsg(
        const Hccl::DfxTaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo,
        const rtLogicCqReport_t& exceptionInfo);
    void FillNotifyErrMsg(const Hccl::DfxTaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo);
    void FillReduceErrMsg(
        const Hccl::DfxTaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo,
        const rtLogicCqReport_t& exceptionInfo);
    void FillDmaErrMsg(
        const Hccl::DfxTaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo,
        const rtLogicCqReport_t& exceptionInfo);
    void FillSdmaErrMsg(const Hccl::DfxTaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo);
    void FillUbErrMsg(
        const Hccl::DfxTaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo,
        const rtLogicCqReport_t& exceptionInfo);
    void FillReduceInlineErrMsg(const Hccl::DfxTaskInfo& taskInfo, Hccl::ErrorMessageReport& errMsgInfo);
    HcclResult SendTaskExceptionByMBox(const u32 notifyId, const u32 tsId, const rtLogicCqReport_t& exceptionInfo);
    uint16_t SwitchUBCqeErrCodeToTsErrCode(u32 cqeErrCode);
    uint16_t SwitchSdmaCqeErrCodeToTsErrCode(u32 cqeErrCode);

    // 打印流上的task信息的方法函数
    HcclResult PrintTaskExceptionBySqeId(CollCommAicpu* aicpuComm, u32 sqId, u32 sqeId);
    HcclResult PrintTaskContextInfo(CollCommAicpu* aicpuComm, u32 sqId, u32 taskId);
    HcclResult
    CollectTaskContext(CollCommAicpu* aicpuComm, u32 sqId, u32 taskId, std::vector<Hccl::DfxTaskInfo*>& taskContext);
    std::string GetParaInfo(const Hccl::DfxTaskInfo& taskInfo);
    std::string GetGroupInfo(CollCommAicpu* aicpuComm);
    u32 GetSqeId(uint16_t taskId, uint16_t streamId);

    Hccl::DfxTaskInfo* FindDfxTaskInfo(CollCommAicpu* aicpuComm, u32 sqId, u32 sqeId);
    Hccl::DfxCircularQueue<Hccl::DfxTaskInfo, Hccl::DFX_TASK_INFO_QUEUE_CAPACITY>*
    GetTaskQueueBySqId(CollCommAicpu* aicpuComm, u32 sqId);
    void GetEidFromChannelHandle(const Hccl::DfxTaskInfo& taskInfo, Hccl::Eid& locEid, Hccl::Eid& rmtEid);
    u32 GetRemoteRankId(const Hccl::DfxTaskInfo& taskInfo);
    std::string GetConciseTaskName(const Hccl::DfxTaskInfo& taskInfo);
    std::string GetNotifyInfo(const Hccl::DfxTaskInfo& taskInfo);
    u32 GetNotifyIdFromSqe(u64 sqeAddr);
    void GetSdmaSqeFields(u64 sqeAddr, u64& src, u64& dst, u64& size);
    u32 GetOpIndex(const Hccl::DfxTaskInfo* taskInfo);
    void PrintOpDataInfo(const Hccl::DfxTaskInfo* taskInfo);
    // dpu相关
    HcclResult HandleDpuTaskexception(CollCommAicpu* aicpuComm);
    HcclResult IsHandleDpuStop(uint8_t* taskexceptionVa, bool& isStop);

private:
    bool stopCall_{false};
    u32 devId_{INVALID_UINT};
    std::unordered_map<u32, u32> threadsPrinted_;
};

} // namespace hcomm

#endif
