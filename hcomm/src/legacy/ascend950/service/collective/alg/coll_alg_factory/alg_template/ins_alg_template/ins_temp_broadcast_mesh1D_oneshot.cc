/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"

#include "alg_data_trans_wrapper.h"
#include "ins_alg_template/ins_temp_broadcast_mesh1D_oneshot.h"

namespace Hccl {
InsTempBroadcastMesh1DOneShot::InsTempBroadcastMesh1DOneShot(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempBroadcastMesh1DOneShot::~InsTempBroadcastMesh1DOneShot() {}

HcclResult InsTempBroadcastMesh1DOneShot::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = tempVTopo_[0].size() - 1 > 0 ? tempVTopo_[0].size() - 1 : 1;
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    QId centerQ = 0;
    tempResReq.localWaitGroupCntNotify.emplace_back(centerQ, 0);
    tempResReq.localBcastPostCntNotify.emplace_back(centerQ, 0);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    HCCL_DEBUG(
        "[InsTempBroadcastMesh1DOneShot]CalcRes: queNum[%u], myRank[%d], tempRankSize[%u]", tempResReq.queNum, myRank_,
        tempRankSize_);
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempBroadcastMesh1DOneShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    if (op_.opMode == OpMode::OPBASE) {
        return 1;
    } else {
        return 0;
    }
}

HcclResult InsTempBroadcastMesh1DOneShot::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    if (tempVTopo_[0].size() == 1) {
        return HcclResult::HCCL_SUCCESS;
    }

    HCCL_INFO("[InsTempBroadcastMesh1DOneShot][Run] Broadcast1DMesh start: rank[%d] end", myRank_);
    opMode_ = tempFuncs.opMode;
    queNum_ = tempVTopo_[0].size() - 1;
    CHK_PRT_RET(
        queNum_ != tempInsQues.size(),
        HCCL_ERROR("[CollAlgFactory] [InsTempBroadcastMesh1DOneShot] Rank [%d], requiredQue Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    UsrData usrData;
    usrData.usrInSlices.emplace_back(
        BufferType::INPUT, templateDataParams.buffInfo.inBuffBaseOff, templateDataParams.sliceSize);
    usrData.scratchInSlices.emplace_back(BufferType::SCRATCH, 0, templateDataParams.sliceSize);
    usrData.scratchOutSlices.emplace_back(BufferType::SCRATCH, 0, templateDataParams.sliceSize);
    usrData.usrOutSlices.emplace_back(
        BufferType::INPUT, templateDataParams.buffInfo.inBuffBaseOff, templateDataParams.sliceSize);
    if (root_ == u32(myRank_)) {
        CHK_RET(PreSyncInterQueues(tempInsQues));
        for (u32 i = 0; i < tempVTopo_[0].size() - 1; i++) {
            u32 neighborRank = (myRank_ + 1 + i) % tempVTopo_[0].size();
            const LinkData& linkSend = tempLinks.at(neighborRank)[0];
            std::vector<DataSlice> txSlices;
            if (opMode_ == OpMode::OPBASE) {
                txSlices = usrData.scratchOutSlices;
            } else {
                txSlices = usrData.usrInSlices;
            }
            SlicesList txSlicesList(usrData.usrInSlices, txSlices);
            DataInfo sendData(linkSend, txSlicesList);
            CHK_PRT_RET(
                Send(sendData, tempInsQues[i], 0, true, DmaMode::PUT),
                HCCL_ERROR("[InsTempBroadcastMesh1DOneShot] BatchSend failed"), HcclResult::HCCL_E_INTERNAL);
        }
        CHK_RET(PostSyncInterQueues(tempInsQues));
    } else {
        const LinkData& linkRecv = tempLinks.at(root_)[0];
        std::vector<DataSlice> rxSlices;
        if (opMode_ == OpMode::OPBASE) {
            rxSlices = usrData.scratchOutSlices;
        } else {
            rxSlices = usrData.usrInSlices;
        }
        SlicesList rxSlicesList(usrData.usrInSlices, rxSlices);
        DataInfo recvData(linkRecv, rxSlicesList);
        CHK_PRT_RET(
            Recv(recvData, tempInsQues[0], 0, true, DmaMode::PUT),
            HCCL_ERROR("[InsTempBroadcastMesh1DOneShot] BatchRecv failed"), HcclResult::HCCL_E_INTERNAL);
        if (opMode_ == OpMode::OPBASE) {
            LocalCopySlices(tempInsQues[0], usrData.scratchOutSlices, usrData.usrInSlices);
        }
    }
    HCCL_INFO("[InsTempBroadcastMesh1DOneShot][Run] Broadcast1DMesh finished: rank[%d] end", myRank_);

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
