/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiling_manager_pub.h"
#include "hccl_comm_pub.h"
#include "hccl/hcom.h"
#include "hcom_host_profiling.h"
#include "adapter_prof.h"
#include "hccl/hccl_types.h"
using namespace hccl;
extern HcclResult HcommProfilingReportKernel(uint64_t beginTime, const char* profName)
{
    std::string profNames(profName);
    uint64_t endTime = hrtMsprofSysCycleTime();
    uint32_t threadId = SalGetTid();
    return ProfilingManagerPub::CallMsprofReportNodeInfo(beginTime, endTime, profName, threadId);
}

// 上报acl host
extern HcclResult HcommProfilingReportOp(HcomProInfo profInfo)
{
    // 数据恢复
    HcclCMDType cmdTypeTemp = static_cast<HcclCMDType>(profInfo.cmdType);
    HcclDataType dataTypeTemp = static_cast<HcclDataType>(profInfo.dataType);
    uint64_t groupName = hrtMsprofGetHashId(profInfo.commName, strlen(profInfo.commName));
    std::string algTypeStr(profInfo.algType);

    AlgType algType;
    CHK_PRT_RET(
        TransferStrToAlgType(algTypeStr, algType) == false,
        HCCL_ERROR("[%s] Fail to transfer [%s] to AlgType", __func__, algTypeStr.c_str()), HCCL_E_PARA);

    HCCL_INFO(
        "[%s] cmdType[%u], dataType[%u], groupName[%llu], groupNameStr[%s], algTypeStr[%s], blockDim[%u]", __func__,
        cmdTypeTemp, dataTypeTemp, groupName, profInfo.commName, profInfo.algType, profInfo.blockDim);

    CHK_RET_AND_PRINT_IDE(
        ProfilingManagerPub::CallMsprofReportHostApi(
            cmdTypeTemp, profInfo.beginTime, profInfo.dataCount, dataTypeTemp, algType, groupName, profInfo.blockDim),
        profInfo.commName);
    return HCCL_SUCCESS;
}

extern HcclResult HcommProfilingRegThread(HcomProInfo profInfo, ThreadHandle* threads)
{
    CHK_PTR_NULL(threads);
    std::string identifier(profInfo.commName);
    std::string tag(profInfo.tag);
    HCCL_PROFILER_ADD_GROUPRANK(identifier, profInfo.rankSize, profInfo.userRank);
    if (profInfo.isAiv) {
        HCCL_PROFILER_ADD_TAG_AIV(tag, identifier, GetWorkflowMode());
    } else {
        HCCL_PROFILER_ADD_TAG(tag, identifier, GetWorkflowMode());
    }

    uint32_t mainStreamId = reinterpret_cast<Thread*>(threads[0])->GetStream()->id();
    HCCL_INFO(
        "[%s] mainStreamId[%u], identifier[%s], tag[%s]", __func__, mainStreamId, profInfo.commName, profInfo.tag);
    HCCL_PROFILER_ADD_OPDATA_OP(
        profInfo.tag, profInfo.dataCount, nullptr, nullptr, static_cast<HcclDataType>(profInfo.dataType), profInfo.root,
        profInfo.commName, HcclReduceOp::HCCL_REDUCE_RESERVED);

    std::string algTypeStr(profInfo.algType);
    AlgType algType;
    CHK_PRT_RET(
        TransferStrToAlgType(algTypeStr, algType) == false,
        HCCL_ERROR("[%s] Fail to transfer [%s] to AlgType", __func__, algTypeStr.c_str()), HCCL_E_PARA);
    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(mainStreamId, tag, 0, algType);

    if (((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
         && !hccl::ProfilingManagerPub::GetAdditionInfoState() && !hccl::ProfilingManagerPub::GetTaskApiState())
        && !profInfo.isCapture) {
        return HCCL_SUCCESS;
    }
    // 从流信息profiling开关打开的话再注册
    for (u32 streamIndex = 1; streamIndex <= profInfo.slaveThreadNum; streamIndex++) {
        uint32_t slaveStreamId = reinterpret_cast<Thread*>(threads[streamIndex])->GetStream()->id();
        HCCL_INFO("[%s] slaveStreamId[%u]", __func__, slaveStreamId);
        HCCL_PROFILER_ADD_STREAM_BY_STREAMID(slaveStreamId, profInfo.tag, streamIndex, algType);
    }
    return HCCL_SUCCESS;
}

extern HcclResult HcommProfilingUnRegThread(HcomProInfo profInfo, ThreadHandle* threads)
{
    CHK_PTR_NULL(threads);
    std::string tag(profInfo.tag);
    std::string identifier(profInfo.commName);

    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_GROUPRANK(identifier);

    uint32_t mainStreamId = reinterpret_cast<Thread*>(threads[0])->GetStream()->id();
    HCCL_PROFILER_DEL_STREAM_BY_STREAMID(mainStreamId);
    HCCL_PROFILER_DEL_OPDATA(tag);
    if (((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
         && !hccl::ProfilingManagerPub::GetAdditionInfoState() && !hccl::ProfilingManagerPub::GetTaskApiState())
        && !profInfo.isCapture) {
        return HCCL_SUCCESS;
    }

    for (u32 streamIndex = 1; streamIndex <= profInfo.slaveThreadNum; streamIndex++) {
        uint32_t slaveStreamId = reinterpret_cast<Thread*>(threads[streamIndex])->GetStream()->id();
        HCCL_INFO("[%s] slaveStreamId[%u]", __func__, slaveStreamId);
        HCCL_PROFILER_DEL_STREAM_BY_STREAMID(slaveStreamId);
    }
    return HCCL_SUCCESS;
}
