/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include <unordered_map>
#include "common/aicpu_sqe_context.h"
#include "task_profiling_pub.h"
#include "common/aicpu_hccl_common.h"
#include "profiling_manager_device.h"
#include "profiling_extend_info.h"

namespace dfx {
uint64_t g_taskHashIds[static_cast<uint64_t>(hccl::ProfTaskType::TASK_INVALID)];
const std::vector<hccl::ProfTaskType> kfcTaskTypes
    = {hccl::ProfTaskType::TASK_HCCL_INFO, // 当前未支持的用这个来暂替
       hccl::ProfTaskType::TASK_NOTIFY_RECORD,
       hccl::ProfTaskType::TASK_NOTIFY_WAIT,
       hccl::ProfTaskType::TASK_SDMA,
       hccl::ProfTaskType::TASK_INTER_RANK_RECORD,
       hccl::ProfTaskType::TASK_INTER_PROCESSOR_SYNC,
       hccl::ProfTaskType::TASK_REDUCE_INLINE,
       hccl::ProfTaskType::TASK_RDMA};

namespace {
    void ParseNotifySqeInfo(const SqeInfo& sqeInfo, MsprofAicpuHcclTaskInfo& msprofAicpuMC2HcclInfo)
    {
        msprofAicpuMC2HcclInfo.notifyID = sqeInfo.notifyId;
        msprofAicpuMC2HcclInfo.remoteRank = sqeInfo.remoteRank;
        msprofAicpuMC2HcclInfo.itemId = g_taskHashIds[static_cast<uint64_t>(hccl::ProfTaskType::TASK_NOTIFY_RECORD)];
    };

    void ParseWaitqeInfo(const SqeInfo& sqeInfo, MsprofAicpuHcclTaskInfo& msprofAicpuMC2HcclInfo)
    {
        msprofAicpuMC2HcclInfo.notifyID = sqeInfo.notifyId;
        msprofAicpuMC2HcclInfo.remoteRank = sqeInfo.remoteRank;
        msprofAicpuMC2HcclInfo.role = static_cast<uint32_t>(hccl::TaskRole::DST);
        msprofAicpuMC2HcclInfo.itemId = g_taskHashIds[static_cast<uint64_t>(hccl::ProfTaskType::TASK_NOTIFY_WAIT)];
    };

    void ParseSdmaSqeInfo(const SqeInfo& sqeInfo, MsprofAicpuHcclTaskInfo& msprofAicpuMC2HcclInfo)
    {
        // addrXHigh存的是地址的高32位
        msprofAicpuMC2HcclInfo.srcAddr = (static_cast<uint64_t>(sqeInfo.addr1High) << 32) | sqeInfo.addr1Low;
        // addrXLow存的是地址的低32位
        msprofAicpuMC2HcclInfo.dstAddr = (static_cast<uint64_t>(sqeInfo.addr2High) << 32) | sqeInfo.addr2Low;
        msprofAicpuMC2HcclInfo.remoteRank = sqeInfo.remoteRank;
        msprofAicpuMC2HcclInfo.role = static_cast<uint32_t>(hccl::TaskRole::DST);
        if (msprofAicpuMC2HcclInfo.localRank == msprofAicpuMC2HcclInfo.remoteRank) {
            msprofAicpuMC2HcclInfo.transportType = static_cast<uint32_t>(hccl::SimpleTaskType::LOCAL);
        } else {
            msprofAicpuMC2HcclInfo.transportType = static_cast<uint32_t>(hccl::SimpleTaskType::SDMA);
        }
        msprofAicpuMC2HcclInfo.linkType = sqeInfo.taskRelated.linkType;
        msprofAicpuMC2HcclInfo.dataSize = sqeInfo.length;
        if (sqeInfo.opCode == 0) { // 0表示不做随路规约
            // SDMA不展示数据类型
            msprofAicpuMC2HcclInfo.itemId = g_taskHashIds[static_cast<uint64_t>(hccl::ProfTaskType::TASK_SDMA)];
        } else {
            TranslateOpcode(sqeInfo.opCode, msprofAicpuMC2HcclInfo.opType);
            msprofAicpuMC2HcclInfo.itemId
                = g_taskHashIds[static_cast<uint64_t>(hccl::ProfTaskType::TASK_REDUCE_INLINE)];
        }
    }

    void ParseCommonSqeInfo([[maybe_unused]] const SqeInfo& sqeInfo, MsprofAicpuHcclTaskInfo& msprofAicpuMC2HcclInfo)
    {
        HCCL_WARNING("Unsupported SQE type");
        msprofAicpuMC2HcclInfo.itemId = g_taskHashIds[static_cast<uint64_t>(hccl::ProfTaskType::TASK_HCCL_INFO)];
    };

    // write value用于卡间的record
    void ParseWriteValueSqeInfo(const SqeInfo& sqeInfo, MsprofAicpuHcclTaskInfo& msprofAicpuMC2HcclInfo)
    {
        msprofAicpuMC2HcclInfo.remoteRank = sqeInfo.remoteRank;
        if (sqeInfo.subType == RT_STARS_WRITE_VALUE_SUB_TYPE_RDMA_DB_SEND) {
            msprofAicpuMC2HcclInfo.itemId = g_taskHashIds[static_cast<uint64_t>(hccl::ProfTaskType::TASK_RDMA)];
            msprofAicpuMC2HcclInfo.linkType = static_cast<uint32_t>(hccl::LinkType::LINK_ROCE);       // reserved value
            msprofAicpuMC2HcclInfo.transportType = static_cast<uint32_t>(hccl::SimpleTaskType::RDMA); // reserved value
            msprofAicpuMC2HcclInfo.dataSize = sqeInfo.length;                                         // wr len
            msprofAicpuMC2HcclInfo.rdmaType = sqeInfo.taskRelated.rdmaType;
        } else {
            msprofAicpuMC2HcclInfo.itemId
                = g_taskHashIds[static_cast<uint64_t>(hccl::ProfTaskType::TASK_INTER_RANK_RECORD)];
        }
    };

    void ParseCondSqeInfo([[maybe_unused]] const SqeInfo& sqeInfo, MsprofAicpuHcclTaskInfo& msprofAicpuMC2HcclInfo)
    {
        msprofAicpuMC2HcclInfo.itemId
            = g_taskHashIds[static_cast<uint64_t>(hccl::ProfTaskType::TASK_INTER_PROCESSOR_SYNC)];
    };
} // namespace

void ProfilingExtendInfoHelper::SqeInfo2MsprofAicpuMC2HcclInfo(
    const SqeInfo& sqeInfo, MsprofAicpuHcclTaskInfo& msprofAicpuMC2HcclInfo)
{
    msprofAicpuMC2HcclInfo.taskId = sqeInfo.taskId;
    msprofAicpuMC2HcclInfo.streamId = sqeInfo.streamId;
    return ProfilingExtendInfoHelper::AssembleProfInfoByType(sqeInfo, msprofAicpuMC2HcclInfo);
}

void ProfilingExtendInfoHelper::AssembleProfInfoByType(
    const SqeInfo& sqeInfo, MsprofAicpuHcclTaskInfo& msprofAicpuMC2HcclInfo)
{
    static const std::unordered_map<uint8_t, Handle> funcMap
        = {{RT_STARS_SQE_TYPE_WRITE_VALUE, ParseWriteValueSqeInfo},
           {RT_STARS_SQE_TYPE_NOTIFY_RECORD, ParseNotifySqeInfo},
           {RT_STARS_SQE_TYPE_NOTIFY_WAIT, ParseWaitqeInfo},
           {RT_STARS_SQE_TYPE_SDMA, ParseSdmaSqeInfo},
           {RT_STARS_SQE_TYPE_COND, ParseCondSqeInfo}};
    auto it = funcMap.find(sqeInfo.type);
    if (it == funcMap.cend()) {
        return ParseCommonSqeInfo(sqeInfo, msprofAicpuMC2HcclInfo);
    }
    (it->second)(sqeInfo, msprofAicpuMC2HcclInfo);
}

void ProfilingExtendInfoHelper::InitHcclInfo(MsprofAicpuHcclTaskInfo& msprofAicpuMC2HcclInfo)
{
    msprofAicpuMC2HcclInfo.linkType = static_cast<uint8_t>(hccl::LinkType::LINK_RESERVED);
    msprofAicpuMC2HcclInfo.rdmaType = static_cast<uint8_t>(hccl::RdmaType::RDMA_TYPE_RESERVED);
    msprofAicpuMC2HcclInfo.dataType = static_cast<uint8_t>(HcclDataType::HCCL_DATA_TYPE_RESERVED);
    msprofAicpuMC2HcclInfo.opType = static_cast<uint8_t>(HcclReduceOp::HCCL_REDUCE_RESERVED);
    msprofAicpuMC2HcclInfo.workFlowMode = static_cast<uint8_t>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED);
    msprofAicpuMC2HcclInfo.stage = 0;
    msprofAicpuMC2HcclInfo.role = static_cast<uint8_t>(hccl::TaskRole::SRC);
    msprofAicpuMC2HcclInfo.transportType = static_cast<uint8_t>(hccl::SimpleTaskType::LOCAL);
}

void ProfilingExtendInfoHelper::InitProfItemId()
{
    if (MsprofReportBatchAdditionalInfo == nullptr) {
        if (AdprofGetHashId == nullptr) {
            HCCL_INFO("AdprofGetHashId is null, InitProfItemId just return");
            return;
        }
        for (const auto taskType : kfcTaskTypes) {
            // index保证是有效的
            g_taskHashIds[static_cast<uint64_t>(taskType)] = AdprofGetHashId(
                hccl::GetProfTaskOpName(taskType).c_str(), hccl::GetProfTaskOpName(taskType).length());
        }
    } else {
        if (MsprofStr2Id == nullptr) {
            HCCL_INFO("MsprofStr2Id is null, InitProfItemId just return");
            return;
        }
        for (const auto taskType : kfcTaskTypes) {
            // index保证是有效的
            g_taskHashIds[static_cast<uint64_t>(taskType)]
                = MsprofStr2Id(hccl::GetProfTaskOpName(taskType).c_str(), hccl::GetProfTaskOpName(taskType).length());
        }
    }
    return;
}
} // namespace dfx
