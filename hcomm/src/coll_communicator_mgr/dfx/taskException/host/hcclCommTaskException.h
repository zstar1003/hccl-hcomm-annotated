/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_TASKEXCEPTION_H
#define HCCL_COMM_TASKEXCEPTION_H

#include <array>
#include <map>
#include "types.h"
#include "hccl_types.h"
#include "orion_adapter_rts.h"
#include "global_mirror_tasks.h"
#include "error_message_v2.h"
#include "orion_adapter_hccp.h"
#include "rdma_handle_manager.h"
#include "coll_comm.h"

namespace hcomm {
using RdmaHandle = void*;

using GetAicpuCqeErrInfoCallBackHcomm = void (*)(
    u32 RemoteLocalId, u32 LocDeviceId, uint16_t status, std::string LocalEid, std::string RemoteEid,
    std::string RemoteInsId); // 获取远端rankId的回调函数类型
void RegisterGetAicpuCqeErrInfoCallBackHcomm(GetAicpuCqeErrInfoCallBackHcomm p1); // 注册获取远端rankId的回调函数

using AicpuGetErrStatusVecCallBack = std::vector<std::string> (*)(s32 deviceLogicID);
void RegisterAicpuGetErrStatusVecCallBack(AicpuGetErrStatusVecCallBack p1);

class TaskExceptionHost {
public:
    static TaskExceptionHost* GetInstance(s32 deviceLogicID);
    TaskExceptionHost() = default;
    ~TaskExceptionHost();

    HcclResult Register(u64 commHandle);
    HcclResult UnRegister(u64 commHandle);
    static void ProcessCallback(rtExceptionInfo_t* exceptionInfo);

private:
    void Process(rtExceptionInfo_t* exceptionInfo);
    void ProcessHcclTaskException(rtExceptionInfo_t* exceptionInfo);
    void HandleAicpuErrorReport(
        rtExceptionInfo_t* exceptionInfo, const Hccl::ErrorMessageReport& errorMessage,
        const Hccl::TaskInfo& taskInfo) const;
    void HandleHostErrorReport(rtExceptionInfo_t* exceptionInfo, const Hccl::TaskInfo& taskInfo) const;
    void ReportErrorMsg(
        const Hccl::TaskInfo& exceptionTaskInfo, const std::string& groupRankContent,
        const Hccl::ErrorMessageReport& errorMessage, rtExceptionInfo_t* exceptionInfo,
        const std::string& stageErrInfo) const;
    void ReportEI0018Error(
        const Hccl::TaskInfo& exceptionTaskInfo, const Hccl::ErrorMessageReport& errorMessage,
        const std::string& stageErrInfo) const;
    bool ShouldReportError() const;

    std::string GetGroupRankInfo(const Hccl::TaskInfo& taskInfo) const;
    void ProcessException(rtExceptionInfo_t* exceptionInfo, const Hccl::TaskInfo& taskInfo);
    void
    PrintTaskContextInfo(uint32_t deviceId, uint32_t streamId, uint32_t taskId, const std::string& stageErrInfo) const;
    void PrintUbDfxInfo(rtExceptionInfo_t* exceptionInfo, const Hccl::ErrorMessageReport& errorMessage) const;
    void PrintGroupErrorMessage(
        const Hccl::ErrorMessageReport& errorMessage, const Hccl::TaskInfo& exceptionTaskInfo,
        std::string& groupRankContent, std::string& stageErrInfo) const;
    void PrintOpDataErrorMessage(
        u32 deviceId, const Hccl::ErrorMessageReport& errorMessage, std::string& stageErrInfo) const;
    HcclResult PrintUbRegisters(s32 devLogicId, const RdmaHandle rdmaHandle) const;
    void ClusterMoniterGetAicpuCqeErrInfo(
        u32 remoteLocalId, u32 locDeviceId, uint16_t status, std::string localEid, std::string remoteEid,
        std::string remoteInsId) const;
    void GetAicpuCqeErrInfo(
        rtExceptionInfo_t* exceptionInfo, const Hccl::ErrorMessageReport& errorMessage,
        const Hccl::TaskInfo& taskInfo) const;
    void GetAicpuCqeErrRemoteLocalIdByRankId(hccl::CollComm* collComm, uint32_t rankid, u32& remoteLocalId) const;
    void GetAicpuCqeErrNetInstanceByRankId(hccl::CollComm* collComm, uint32_t rankid, std::string& netInstanceId) const;
    void NotifyControlPlaneOnUbError(
        u32 devPhyId, RdmaHandle rdmaHandle, const Hccl::ErrorMessageReport& errorMessage) const;
    bool ProcessDpuException(const rtExceptionInfo_t* exceptionInfo) const;

private:
    std::mutex taskExceptionMutex_;
    std::unordered_set<u64> CommRegisterMap_;
    bool hasAicpuReport_{false};
    mutable std::mutex ei0002ReportedMutex_;
    mutable bool ei0002Reported_{false};
};
} // namespace hcomm

#endif
