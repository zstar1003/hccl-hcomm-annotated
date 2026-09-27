/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_EXCEPTION_TEST_COMMON_H
#define TASK_EXCEPTION_TEST_COMMON_H

#include <mockcpp/mockcpp.hpp>
#include "task_exception_handler.h"

using namespace Hccl;

HcclResult MockGetCcuErrorMsg(
    s32 deviceId, uint16_t missionStatus, uint16_t currIns, const ParaCcu& ccuTaskParam,
    const std::string& groupRankContent, std::vector<CcuErrorInfo>& errorInfo);

inline rtExceptionInfo_t PrepareNullDfxCcuException(MirrorTaskManager& mgr)
{
    auto taskInfo = std::make_unique<TaskInfo>(0, 0, 0, TaskParam{}, nullptr);
    taskInfo->taskParam_.taskType = TaskParamType::TASK_CCU;
    mgr.AddTaskInfo(std::move(taskInfo));

    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.expandInfo.type = RT_EXCEPTION_FUSION;
    exceptionInfo.expandInfo.u.fusionInfo.type = RT_FUSION_AICORE_CCU;
    exceptionInfo.deviceid = 0;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.ccuMissionNum = 1;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.missionInfo[0].dieId = 0;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.missionInfo[0].missionId = 1;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.missionInfo[0].instrId = 2;
    return exceptionInfo;
}

inline void SetupAndProcessCcuException()
{
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.expandInfo.type = RT_EXCEPTION_FUSION;
    exceptionInfo.expandInfo.u.fusionInfo.type = RT_FUSION_AICORE_CCU;
    exceptionInfo.deviceid = 10;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.ccuMissionNum = 1;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.missionInfo[0].dieId = 0;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.missionInfo[0].missionId = 1;
    exceptionInfo.expandInfo.u.fusionInfo.u.aicoreCcuInfo.ccuDetailMsg.missionInfo[0].instrId = 2;

    MOCKER(GetCcuErrorMsg).stubs().will(returnValue(HcclResult::HCCL_SUCCESS)).then(invoke(MockGetCcuErrorMsg));
    CcuTaskParam ccuTaskParam{};
    ccuTaskParam.dieId = 0;
    ccuTaskParam.missionId = 2;
    ccuTaskParam.instStartId = 5;
    vector<CcuTaskParam> mockAlgTaskParams{ccuTaskParam};
    MOCKER(TaskExceptionHandler::GetMC2AlgTaskParam).stubs().will(returnValue(mockAlgTaskParams));

    MOCKER(CcuCleanDieCkes).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuComponent::Init).stubs();
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(0));
    MOCKER(HrtRaTlvRequestForCustomChannel).stubs();

    TaskExceptionHandler::Process(&exceptionInfo);
}

#endif // TASK_EXCEPTION_TEST_COMMON_H
