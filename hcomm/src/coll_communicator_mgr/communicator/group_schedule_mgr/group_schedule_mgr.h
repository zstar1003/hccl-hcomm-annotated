/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUP_SCHEDULE_MGR_H
#define GROUP_SCHEDULE_MGR_H

#include <deque>
#include <vector>
#include <string>
#include <map>
#include <hccl/hccl_types.h>
#include <hccl/hccl_launch.h>
#include "acl/acl_base_rt.h"
#include "hcomm_res_defs.h"

constexpr int32_t MAX_P2P_TASK_NUM = 2048;

namespace hccl {

struct HcclP2pPair {
    uint32_t sendRank;
    uint32_t recvRank;
};

struct HcclP2pTask {
    HcclOpP2pDesc desc;
    aclrtStream stream;
    aclrtStream usrStream;
    HcclKernelFuncInfo funcInfo;
    uint8_t args[P2P_MAX_ARG_SIZE];
    uint32_t argSize;
};

struct HcclP2pSendRecvQueue {
    std::deque<HcclP2pTask> sendQue;
    std::deque<HcclP2pTask> recvQue;
};

void ClearHcclGroupCommList();
std::vector<HcclComm>& GetHcclGroupCommList();
int32_t GetHcclP2pTaskNums();
void SetHcclP2pTaskNums(int32_t targetP2pTaskNums);

class GroupScheduleMgr {
public:
    GroupScheduleMgr() : userRank_(0), serverNum_(0), nTasksP2p_(-1), usrStream_(nullptr) {};
    ~GroupScheduleMgr();

    HcclResult GetUsrStream(aclrtStream& usrStream);
    HcclResult SetUsrStream(const aclrtStream& usrStream);

    HcclResult AppendGroupP2pTask(HcclComm comm, const HcclP2pTask& task, const HcclOpP2pDesc& p2pDesc);
    HcclResult GetP2pTaskSchedule(std::vector<HcclP2pTask>& sortedSendQue, std::vector<HcclP2pTask>& sortedRecvQue);

private:
    HcclResult GetCurLocalRank(uint32_t& localRank);
    HcclResult CalculateGroupSize();
    uint32_t GenerateP2pSchedule(
        const std::vector<uint32_t>& groupToServer, const std::vector<uint32_t>& groupToLocalRankBase,
        uint32_t curGroupIdx, uint32_t curGroupLocalRankIdx);
    HcclResult InitGroupPlanner(HcclComm comm);
    HcclResult HcclP2pSchedulerGenerate();

private:
    uint32_t userRank_;
    uint32_t serverNum_;
    std::map<uint32_t, uint32_t> serverToRankSize_;
    std::map<uint32_t, std::vector<uint32_t>> serverToRankList_;
    uint32_t rankSize_;
    uint32_t groupSize_;
    uint32_t nGroups_;

    int32_t nTasksP2p_;
    aclrtStream usrStream_;
    std::vector<HcclP2pPair> p2pSchedule_;
    std::vector<HcclP2pSendRecvQueue> peers_;
};

} // namespace hccl
#endif // GroupScheduleMgr
