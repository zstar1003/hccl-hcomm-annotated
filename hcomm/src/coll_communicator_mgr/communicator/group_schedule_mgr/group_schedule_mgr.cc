/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "group_schedule_mgr.h"
#include "log.h"
#include "coll_alg_utils.h"
#include "hccl_comm_pub.h"

thread_local int32_t hcclP2pTaskNums;
thread_local std::vector<HcclComm> hcclGroupCommListV2;

constexpr uint32_t NUM = 1U;
constexpr uint32_t BIT_MAX = 31U;

namespace {
uint32_t pow2Up(uint32_t n)
{
    if (n > (NUM << BIT_MAX)) {
        return 0;
    }

    uint32_t power = 1;
    while (power < n) {
        power = power << 1U;
    }
    return power;
}
} // namespace

namespace hccl {

void ClearHcclGroupCommList() { hcclGroupCommListV2.clear(); }

std::vector<HcclComm>& GetHcclGroupCommList() { return hcclGroupCommListV2; }

int32_t GetHcclP2pTaskNums() { return hcclP2pTaskNums; }

void SetHcclP2pTaskNums(int32_t targetP2pTaskNums) { hcclP2pTaskNums = targetP2pTaskNums; }

GroupScheduleMgr::~GroupScheduleMgr() {}

HcclResult GroupScheduleMgr::GetUsrStream(aclrtStream& usrStream)
{
    CHK_PTR_NULL(this->usrStream_);
    usrStream = this->usrStream_;
    return HCCL_SUCCESS;
}

HcclResult GroupScheduleMgr::SetUsrStream(const aclrtStream& usrStream)
{
    CHK_PTR_NULL(usrStream);
    this->usrStream_ = usrStream;
    return HCCL_SUCCESS;
}

HcclResult GroupScheduleMgr::InitGroupPlanner(HcclComm comm)
{
    CHK_PTR_NULL(comm);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_PTR_NULL(hcclComm);
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    constexpr uint32_t netLayerServer = 0;
    uint32_t* serverSizeList = nullptr;
    uint32_t serverNum = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, netLayerServer, &serverSizeList, &serverNum));
    CHK_PTR_NULL(serverSizeList);
    this->userRank_ = collComm->GetMyRankId();
    this->rankSize_ = collComm->GetRankSize();
    this->nTasksP2p_ = 0;
    this->serverNum_ = serverNum;
    for (uint32_t serverIdx = 0, rankIdx = 0; serverIdx < serverNum; serverIdx++) {
        this->serverToRankSize_[serverIdx] = serverSizeList[serverIdx];
        for (uint32_t localIdx = 0; localIdx < serverSizeList[serverIdx]; localIdx++) {
            this->serverToRankList_[serverIdx].emplace_back(rankIdx++);
        }
    }
    HCCL_INFO(
        "[InitGroupPlanner] ranksize:%u, serverNum:%u nTaskP2p:%d", this->rankSize_, this->serverNum_,
        this->nTasksP2p_);

    return HCCL_SUCCESS;
}

HcclResult GroupScheduleMgr::GetCurLocalRank(uint32_t& localRank)
{
    uint32_t curServerIdx = 0;
    for (uint32_t cumulativeRank = 0, serverIdx = 0; serverIdx < this->serverNum_; serverIdx++) {
        cumulativeRank += this->serverToRankSize_.at(serverIdx);
        if (this->userRank_ < cumulativeRank) {
            curServerIdx = serverIdx;
            break;
        }
    }

    u32 curLocalRank = 0;
    for (u32 rankIdx : this->serverToRankList_.at(curServerIdx)) {
        if (this->userRank_ == rankIdx) {
            break;
        }
        curLocalRank++;
    }

    if (curLocalRank >= this->serverToRankSize_.at(curServerIdx)) {
        HCCL_ERROR(
            "[getCurLocalRank] curLocalRank[%u] is invalid, valid range [0, %u)", curLocalRank,
            this->serverToRankSize_.at(curServerIdx));
        return HCCL_E_INTERNAL;
    }

    localRank = curLocalRank;
    return HCCL_SUCCESS;
}

HcclResult GroupScheduleMgr::CalculateGroupSize()
{
    if (this->serverNum_ == 0) {
        return HCCL_E_INTERNAL;
    }

    if (this->serverNum_ == 1) {
        this->groupSize_ = this->rankSize_;
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> serverRankSizeList;
    for (const auto& pair : this->serverToRankSize_) {
        serverRankSizeList.emplace_back(pair.second);
    }
    this->groupSize_ = hccl::CalGCD(serverRankSizeList);
    return HCCL_SUCCESS;
}

uint32_t GroupScheduleMgr::GenerateP2pSchedule(
    const std::vector<uint32_t>& groupToServer, const std::vector<uint32_t>& groupToLocalRankBase, uint32_t curGroupIdx,
    uint32_t curGroupLocalRankIdx)
{
    this->p2pSchedule_.resize(this->rankSize_);
    uint32_t round = 0;
    uint32_t groupRound = 0;
    uint32_t groupDelta = 0;
    uint32_t nGroupsPow2 = pow2Up(this->nGroups_);
    if (nGroupsPow2 == 0) {
        return 0;
    }

    do {
        if (groupDelta < this->nGroups_) {
            uint32_t sendGroupIdx = (curGroupIdx + groupDelta) % this->nGroups_;
            uint32_t recvGroupIdx = (curGroupIdx - groupDelta + this->nGroups_) % this->nGroups_;
            uint32_t sendServerIdx = groupToServer[sendGroupIdx];
            uint32_t recvServerIdx = groupToServer[recvGroupIdx];

            for (uint32_t delta = 0; delta < this->groupSize_; delta++) {
                uint32_t sendLocalIdx
                    = groupToLocalRankBase[sendGroupIdx] + (curGroupLocalRankIdx + delta) % this->groupSize_;
                uint32_t recvLocalIdx = groupToLocalRankBase[recvGroupIdx]
                                        + (curGroupLocalRankIdx - delta + this->groupSize_) % this->groupSize_;

                this->p2pSchedule_[round].sendRank = this->serverToRankList_.at(sendServerIdx)[sendLocalIdx];
                this->p2pSchedule_[round].recvRank = this->serverToRankList_.at(recvServerIdx)[recvLocalIdx];
                round++;
            }
        }
        groupRound++;
        groupDelta = (groupDelta + groupRound) & (nGroupsPow2 - 1);
    } while (groupRound != nGroupsPow2);

    return round;
}

HcclResult GroupScheduleMgr::HcclP2pSchedulerGenerate()
{
    uint32_t curLocalRank = 0; // 当前rank所在server的局部排序号
    CHK_RET(GetCurLocalRank(curLocalRank));
    CHK_RET(CalculateGroupSize());
    if (this->groupSize_ == 0) {
        HCCL_ERROR("[HcclP2pSchedulerGenerate] groupSize is zero");
        return HCCL_E_INTERNAL;
    }

    uint32_t curGroupLocalRankIdx = curLocalRank % this->groupSize_;
    uint32_t curGroupIdx = this->userRank_ / this->groupSize_;
    this->nGroups_ = this->rankSize_ / this->groupSize_;
    HCCL_INFO(
        "[HcclP2pSchedulerGenerate] userRank:%u, localRank:%u, groupSize:%u, nGroups:%u", this->userRank_, curLocalRank,
        this->groupSize_, this->nGroups_);

    std::vector<uint32_t> groupToServer(this->nGroups_);
    std::vector<uint32_t> groupToLocalRankBase(this->nGroups_);
    uint32_t groupIdx = 0;
    for (const auto& pair : this->serverToRankList_) {
        uint32_t serverIdx = pair.first;
        uint32_t localRankSize = this->serverToRankSize_.at(serverIdx);
        uint32_t localGroupSize = localRankSize / this->groupSize_;
        for (uint32_t localGroupIdx = 0; localGroupIdx < localGroupSize; localGroupIdx++) {
            groupToServer[groupIdx] = serverIdx;
            groupToLocalRankBase[groupIdx] = localGroupIdx * this->groupSize_;
            groupIdx++;
        }
    }

    uint32_t round = GenerateP2pSchedule(groupToServer, groupToLocalRankBase, curGroupIdx, curGroupLocalRankIdx);
    if (this->rankSize_ != round) {
        HCCL_ERROR("[HcclP2pSchedulerGenerate] round:%u is not equal to rankSize:%u", round, this->rankSize_);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[HcclP2pSchedulerGenerate] schedule generated, round:%u", round);
    return HCCL_SUCCESS;
}

HcclResult GroupScheduleMgr::AppendGroupP2pTask(HcclComm comm, const HcclP2pTask& task, const HcclOpP2pDesc& p2pDesc)
{
    CHK_PTR_NULL(comm);
    if (hcclP2pTaskNums == MAX_P2P_TASK_NUM) {
        HCCL_ERROR("[hcclGroupAddP2pTask] P2pTaskNums is out of %d", MAX_P2P_TASK_NUM);
        return HCCL_E_INTERNAL;
    }
    if (this->nTasksP2p_ == -1) {
        CHK_RET(InitGroupPlanner(comm));
        CHK_RET(HcclP2pSchedulerGenerate());
        this->peers_.resize(this->rankSize_);
    }

    if (p2pDesc.cmdType == HcclCMDType::HCCL_CMD_SEND) {
        this->peers_[p2pDesc.remoteRank].sendQue.emplace_back(task);
    } else {
        this->peers_[p2pDesc.remoteRank].recvQue.emplace_back(task);
    }
    this->nTasksP2p_ += 1;
    hcclP2pTaskNums++;
    auto itComm = std::find(hcclGroupCommListV2.begin(), hcclGroupCommListV2.end(), comm);
    if (itComm == hcclGroupCommListV2.end()) {
        hcclGroupCommListV2.emplace_back(comm);
    }

    return HCCL_SUCCESS;
}

HcclResult
GroupScheduleMgr::GetP2pTaskSchedule(std::vector<HcclP2pTask>& sortedSendQue, std::vector<HcclP2pTask>& sortedRecvQue)
{
    HCCL_INFO("[HcclP2pTaskSchedule] nTaskP2p:%d", this->nTasksP2p_);

    uint32_t epoch = 0;
    uint32_t maxEpochNum = this->nTasksP2p_;
    while (this->nTasksP2p_ > 0) {
        for (uint32_t round = 0; round < this->rankSize_; round++) {
            uint32_t sendRank = this->p2pSchedule_[round].sendRank;
            uint32_t recvRank = this->p2pSchedule_[round].recvRank;

            if (!this->peers_[sendRank].sendQue.empty()) {
                auto& sendTask = this->peers_[sendRank].sendQue.front();
                sortedSendQue.emplace_back(sendTask);
                this->peers_[sendRank].sendQue.pop_front();
                this->nTasksP2p_--;
            }

            if (!this->peers_[recvRank].recvQue.empty()) {
                auto& recvTask = this->peers_[recvRank].recvQue.front();
                sortedRecvQue.emplace_back(recvTask);
                this->peers_[recvRank].recvQue.pop_front();
                this->nTasksP2p_--;
            }
        }

        epoch++;
        if (epoch > maxEpochNum) {
            HCCL_ERROR("[GetP2pTaskSchedule] epoch:%u is more than max epoch:%u", epoch, maxEpochNum);
            return HCCL_E_INTERNAL;
        }
    }
    HCCL_INFO(
        "[HcclP2pTaskSchedule] done, use epochs:%u, sendQueSize:%u, recvQueSize:%u ", epoch,
        static_cast<uint32_t>(sortedSendQue.size()), static_cast<uint32_t>(sortedRecvQue.size()));

    return HCCL_SUCCESS;
}
} // namespace hccl
