/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TASK_EXCEPTION_HANDLER_H
#define HCCL_TASK_EXCEPTION_HANDLER_H

#include <array>
#include "types.h"
#include "hccl_types.h"
#include "orion_adapter_rts.h"
#include "ccu_device_manager.h"
#include "global_mirror_tasks.h"
#include "ccu_dfx.h"
#include "ccu_task_param.h"
#include "error_message_v2.h"
#include "orion_adapter_hccp.h"
#include "rdma_handle_manager.h"

namespace Hccl {
using GetAicpuTaskExceptionCallBack = std::function<ErrorMessageReport()>;
class TaskExceptionHandler {
public:
    // 构造函数使用初始化列表初始化devId_
    explicit TaskExceptionHandler(int deviceId);
    ~TaskExceptionHandler();

    // 获取设备ID
    int GetDeviceId() const { return devId_; }
    void Register() const;                                 // 向rts注册异常处理方法
    void UnRegister() const;                               // 向rts注销异常处理方法
    static void Process(rtExceptionInfo_t* exceptionInfo); // 处理异常信息
    static void PrintAicpuErrorMessage(rtExceptionInfo_t* exceptionInfo, bool& isExistAicpuError);

private:
    static std::string GetGroupRankInfo(const TaskInfo& taskInfo);
    static void ProcessException(rtExceptionInfo_t* exceptionInfo, const TaskInfo& taskInfo);
    static void PrintTaskContextInfo(uint32_t deviceId, uint32_t streamId, uint32_t taskId);
    static void ProcessCcuMC2Exception(rtExceptionInfo_t* exceptionInfo);
    static std::vector<CcuTaskParam> GetMC2AlgTaskParam(const TaskInfo& taskInfo);
    static void ProcessCcuException(const rtExceptionInfo_t* exceptionInfo, const TaskInfo& taskInfo);
    static void PrintCcuErrorInfo(uint32_t deviceId, uint16_t status, const TaskInfo& taskInfo);
    static void PrintCcuErrorLog(const std::vector<CcuErrorInfo>& errorInfos, const TaskInfo& taskInfo);
    static void ProcessAivException(rtExceptionInfo_t* exceptionInfo, const TaskInfo& taskInfo);
    static void PrintAivPreviousTaskException(rtExceptionInfo_t* exceptionInfo);

    static std::string GetCcuErrorMsgByType(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgLoop(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgMission(const CcuErrorInfo& ccuErrorInfo);
    static std::string GetCcuErrorMsgDefault(const CcuErrorInfo& ccuErrorInfo);
    static std::string GetCcuErrorMsgLoopGroup(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgLocPostSem(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgLocWaitSem(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgRemPostSem(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgRemWaitSem(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgRemPostVar(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgRemWaitGroup(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgPostSharedVar(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgPostSharedSem(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgRead(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgWrite(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgLocalCpy(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgLocalReduce(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgBufRead(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgBufWrite(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgBufLocRead(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgBufLocWrite(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static std::string GetCcuErrorMsgBufReduce(const CcuErrorInfo& ccuErrorInfo, const TaskInfo& taskInfo);
    static RankId GetRankIdByChannelId(uint16_t channelId, const TaskInfo& taskInfo);
    static void PrintGroupErrorMessage(
        ErrorMessageReport& errorMessage, const TaskInfo& exceptionTaskInfo, string& groupRankContent,
        string& stageErrInfo);
    static void PrintOpDataErrorMessage(u32 deviceId, ErrorMessageReport& errorMessage, string& stageErrInfo);
    static std::pair<IpAddress, IpAddress> GetAddrPairByChannelId(uint16_t channelId, const TaskInfo& taskInfo);
    static std::string GetCcuLenErrorMsg(const uint64_t len);

private:
    uint32_t devId_; // 当前设备id
};

const std::string LOG_KEYWORDS_TIMEOUT = "Timeout";         // 算子执行阶段超时
const std::string LOG_KEYWORDS_RUN_FAILED = "RunFailed";    // 算子执行阶段失败，如SDMA ERROR
const std::string LOG_KEYWORDS_TASK_EXEC = "TaskExecStage"; // 算子执行阶段异常
const std::string LOG_KEYWORDS_AICPU = "AICPU";
} // namespace Hccl

#endif // HCCL_TASK_EXCEPTION_HANDLER_H
