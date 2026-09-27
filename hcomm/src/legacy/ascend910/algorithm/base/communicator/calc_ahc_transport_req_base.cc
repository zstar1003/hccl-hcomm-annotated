/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "calc_ahc_transport_req_base.h"

namespace hccl {
CalcAHCTransportReqBase::CalcAHCTransportReqBase(
    std::vector<std::vector<u32>>& subCommPlaneVector, std::vector<bool>& isBridgeVector, u32 userRank,
    std::vector<std::vector<std::vector<u32>>>& globalSubGroups, std::map<AHCConcOpType, TemplateType>& ahcAlgOption,
    std::unordered_map<u32, bool>& isUsedRdmaMap)
    : CalcTransportReqBase(subCommPlaneVector, isBridgeVector, userRank),
      globalSubGroups_(globalSubGroups),
      ahcAlgOption_(ahcAlgOption),
      isUsedRdmaMap_(isUsedRdmaMap)
{}

CalcAHCTransportReqBase::~CalcAHCTransportReqBase() {}

HcclResult CalcAHCTransportReqBase::DisposeSubGroups(u32 rank)
{
    (void)rank;
    return HCCL_SUCCESS;
}

HcclResult CalcAHCTransportReqBase::CalcDstRanks(u32 rank, std::set<u32>& dstRanks, u32 ringIndex)
{
    (void)rank;
    (void)dstRanks;
    (void)ringIndex;
    return HCCL_SUCCESS;
}

HcclResult CalcAHCTransportReqBase::CommAHCInfoInit(std::vector<std::vector<u32>>& subGroups)
{
    (void)subGroups;
    return HCCL_SUCCESS;
}

HcclResult CalcAHCTransportReqBase::CalcTransportRequest(
    const std::string& tag, TransportMemType inputMemType, TransportMemType outputMemType,
    const CommParaInfo& commParaInfo, std::vector<SingleSubCommTransport>& commTransport, u32 subUserRankRoot)
{
    (void)subUserRankRoot;
    u32 ringSize = subCommPlaneVector_.size();
    commTransport.resize(ringSize);
    if (tag.find("AllReduce", 0) != std::string::npos) {
        opType_ = AHCOpType::AHC_OP_TYPE_ALLREDUCE;
    } else if (tag.find("ReduceScatter", 0) != std::string::npos) {
        opType_ = AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER;
    } else if (tag.find("AllGather", 0) != std::string::npos) {
        opType_ = AHCOpType::AHC_OP_TYPE_ALLGATHER;
    }

    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        if (commParaInfo.commPlane == COMM_LEVEL1_AHC && !isBridgeVector_[ringIndex]) {
            continue; // 跳出本次循环
        }

        u32 rank = GetSubCollectiveRank(subCommPlaneVector_[ringIndex]);
        if (rank == INVALID_VALUE_RANKID) {
            continue;
        }

        u32 rankSize = subCommPlaneVector_[ringIndex].size();
        SingleSubCommTransport& subCommTransport = commTransport[ringIndex];
        subCommTransport.transportRequests.resize(rankSize);
        // 只有一张卡时不需要建链
        if (rankSize == HCCL_RANK_SIZE_EQ_ONE) {
            HCCL_INFO("[CalcAHCTransportReqBase] comm base needn't to create links, rankSize_[%u].", rankSize);
            return HCCL_SUCCESS;
        }

        std::set<u32> dstRanks;
        CHK_RET(CalcDstRanks(rank, dstRanks, ringIndex));

        // 建链
        for (u32 dstRank : dstRanks) {
            CHK_PRT_RET(
                dstRank >= rankSize,
                HCCL_ERROR(
                    "[CalcAHCTransportReqBase][CalcTransportRequest] dstRank [%u] exceed rankSize [%u] error", dstRank,
                    rankSize),
                HCCL_E_INTERNAL);

            if (dstRank != rank) {
                TransportRequest& tmpTransport = subCommTransport.transportRequests[dstRank];
                tmpTransport.isValid = true;
                tmpTransport.localUserRank = userRank_;
                tmpTransport.remoteUserRank = subCommPlaneVector_[ringIndex][dstRank];
                tmpTransport.inputMemType = inputMemType;
                tmpTransport.outputMemType = outputMemType;
                HCCL_INFO(
                    "[CalcAHCTransportReqBase] param_.tag[%s] ringIndex[%u], localRank[%u], "
                    "remoteRank[%u], inputMemType[%d], outputMemType[%d]",
                    tag.c_str(), ringIndex, userRank_, tmpTransport.remoteUserRank, inputMemType, outputMemType);
            }
        }

        // 刷新RDMA建链标记
        RefreshTransportIsUsedRdma(rank, ringIndex, commTransport);
    }
    return HCCL_SUCCESS;
}

void CalcAHCTransportReqBase::RefreshTransportIsUsedRdma(
    u32 rank, u32 ringIndex, std::vector<SingleSubCommTransport>& commTransport)
{
    // 组内和组间通信域计算
    std::vector<u32> intraCommGroup;
    std::vector<std::vector<u32>> interCommGroupList;

    commAHCBaseInfo_->GetIntraCommGroup(rank, intraCommGroup);
    commAHCBaseInfo_->GetInterCommGroupList(rank, interCommGroupList);

    SingleSubCommTransport& subCommTransport = commTransport[ringIndex];

    // 组内子通信域粒度刷新
    bool isUsedRdma = false;
    for (u32 i = 0; i < intraCommGroup.size(); i++) {
        u32 dstRank = intraCommGroup[i];
        HCCL_DEBUG(
            "[CalcAHCTransportReqBase][RefreshTransportIsUsedRdma] intraCommGroup localRank[%u], dstRank [%u] ", rank,
            dstRank);
        if (isUsedRdmaMap_[subCommPlaneVector_[ringIndex][dstRank]]) {
            isUsedRdma = true;
            HCCL_DEBUG(
                "[CalcAHCTransportReqBase][RefreshTransportIsUsedRdma] intraCommGroup userrank[%u] rdma map is true",
                subCommPlaneVector_[ringIndex][dstRank]);
            break;
        }
    }
    for (u32 i = 0; i < intraCommGroup.size(); i++) {
        u32 dstRank = intraCommGroup[i];
        TransportRequest& tmpTransport = subCommTransport.transportRequests[dstRank];
        tmpTransport.isUsedRdma = isUsedRdma;
    }

    // 组间子通信域粒度刷新
    for (u32 i = 0; i < interCommGroupList.size(); i++) {
        isUsedRdma = false;
        for (u32 j = 0; j < interCommGroupList[i].size(); j++) {
            u32 dstRank = interCommGroupList[i][j];
            HCCL_DEBUG(
                "[CalcAHCTransportReqBase][RefreshTransportIsUsedRdma] interCommGroupList index[%u] localRank[%u], "
                "dstRank [%u] ",
                i, rank, dstRank);
            if (isUsedRdmaMap_[subCommPlaneVector_[ringIndex][dstRank]]) {
                isUsedRdma = true;
                HCCL_DEBUG(
                    "[CalcAHCTransportReqBase][RefreshTransportIsUsedRdma] interCommGroupList userrank[%u] rdma map is "
                    "true",
                    subCommPlaneVector_[ringIndex][dstRank]);
                break;
            }
        }
        for (u32 j = 0; j < interCommGroupList[i].size(); j++) {
            u32 dstRank = interCommGroupList[i][j];
            TransportRequest& tmpTransport = subCommTransport.transportRequests[dstRank];
            tmpTransport.isUsedRdma = isUsedRdma;
        }
    }
}
} // namespace hccl
