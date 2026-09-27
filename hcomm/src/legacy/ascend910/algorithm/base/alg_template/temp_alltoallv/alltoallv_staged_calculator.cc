/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_staged_calculator.h"
#include "log.h"

namespace hccl {
using namespace std;

AlltoAllVStagedCalculator::AlltoAllVStagedCalculator() {}

AlltoAllVStagedCalculator::~AlltoAllVStagedCalculator() {}

// / STATIC MEMBER FUNCTIONS BEGINS
void AlltoAllVStagedCalculator::CalcWorkSpaceMemSize(
    const AlltoAllUserRankInfo& userRankInfo, const std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo,
    u64& workspaceMemSize, u32 meshAggregationRankSize)
{
    for (const auto& oneMeshAggregationSendRecvInfo : allMeshAggregationSendRecvInfo) {
        for (const auto& sendLength : oneMeshAggregationSendRecvInfo.sendLength) {
            HCCL_DEBUG("[CalcWorkSpaceMemSize] sendLength[%llu]", sendLength);
        }
        for (const auto& sendOffset : oneMeshAggregationSendRecvInfo.sendOffset) {
            HCCL_DEBUG("[CalcWorkSpaceMemSize] sendOffset[%llu]", sendOffset);
        }
        for (const auto& recvLength : oneMeshAggregationSendRecvInfo.recvLength) {
            HCCL_DEBUG("[CalcWorkSpaceMemSize] recvLength[%llu]", recvLength);
        }
        for (const auto& recvOffset : oneMeshAggregationSendRecvInfo.recvOffset) {
            HCCL_DEBUG("[CalcWorkSpaceMemSize] recvOffset[%llu]", recvOffset);
        }
    }
    if (allMeshAggregationSendRecvInfo.size() % meshAggregationRankSize != 0
        || allMeshAggregationSendRecvInfo.size() == 0) {
        workspaceMemSize = 0;
        HCCL_ERROR("Invalid Send Recv Info Size[%u]", allMeshAggregationSendRecvInfo.size());
        return;
    }
    workspaceMemSize = 0;
    u32 meshAggregationIndex = userRankInfo.userRank / meshAggregationRankSize;
    u32 meshAggregationRankBegin = meshAggregationIndex * meshAggregationRankSize;
    HCCL_DEBUG(
        "[AlltoAllVStagedCalculator][CalcWorkSpaceMemSize]meshAggregationRankBegin is %u", meshAggregationRankBegin);
    for (u32 infoIndex = userRankInfo.userRank % meshAggregationRankSize; infoIndex < userRankInfo.userRankSize;
         infoIndex += meshAggregationRankSize) {
        for (u32 k = meshAggregationRankBegin; k < meshAggregationRankBegin + meshAggregationRankSize; k++) {
            workspaceMemSize += allMeshAggregationSendRecvInfo[k].sendLength[infoIndex];
        }
    }

    if (workspaceMemSize == 0) {
        HCCL_INFO("[AlltoAllVStagedCalculator][CalcWorkSpaceMemSize] workspaceMemSize is 0, use tiny mem size");
        workspaceMemSize = TINY_MEM_SIZE;
    }
    HCCL_INFO("[AlltoAllVStagedCalculator][CalcWorkSpaceMemSize]: workspaceMemSize[%llu]", workspaceMemSize);
}
// / STATIC MEMBER FUNCTIONS ENDS
} // namespace hccl
