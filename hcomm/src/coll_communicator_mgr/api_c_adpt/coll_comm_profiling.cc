/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "prof_common.h"
#include "adapter_prof.h"
#include "coll_comm_profiling.h"
#include "profiling_manager_pub.h"
#include "hccl_comm_pub.h"
// 通信域内首次才能上报
HcclResult HcclStreamProfilingReport(HcclComm comm, u32 threadNum, const u32* threadId)
{
    HCCL_INFO("[%s] threadNum = [%u]", __func__, threadNum);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(threadId);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    ProfilingDeviceCommResInfo hcclMc2Info;
    std::string GroupName(hcclComm->GetIdentifier());
    hcclMc2Info.groupName = hrtMsprofGetHashId(GroupName.c_str(), GroupName.length());
    hcclComm->GetRankSize(hcclMc2Info.rankSize);
    hcclComm->GetGroupRank(hcclMc2Info.rankId);
    hcclMc2Info.usrRankId = hcclComm->GetRealUserRank();
    hcclMc2Info.aicpuKfcStreamId = 0; // 暂无 先报stream 0
    hcclMc2Info.reserve = 0;
    uint32_t reportId = 0;
    // 上报所有得threadid
    const uint32_t ONCE_REPORT_STREAM_NUM_MAX = 8;
    for (uint32_t streamIndex = 0; streamIndex < threadNum; streamIndex++) {
        u32 id = threadId[streamIndex];
        HCCL_DEBUG("[%s] streamIndex:[%u], reportId:[%u], streamId[%u]", __func__, streamIndex, reportId, id);
        hcclMc2Info.commStreamIds[reportId++] = id;
        if (reportId == ONCE_REPORT_STREAM_NUM_MAX) {
            hcclMc2Info.commStreamSize = reportId;
            CHK_RET(hccl::ProfilingManagerPub::CallMsprofReportMc2CommInfo(
                hrtMsprofSysCycleTime(), &hcclMc2Info, sizeof(hcclMc2Info)));
            reportId = 0;
        }
    }
    // 上报剩余的stream
    if (reportId > 0) {
        HCCL_DEBUG("[%s] last reportId[%u]", __func__, reportId);
        hcclMc2Info.commStreamSize = reportId;
        CHK_RET(hccl::ProfilingManagerPub::CallMsprofReportMc2CommInfo(
            hrtMsprofSysCycleTime(), &hcclMc2Info, sizeof(hcclMc2Info)));
        reportId = 0;
    }
    HCCL_INFO("[%s] success", __func__);
    return HCCL_SUCCESS;
}
