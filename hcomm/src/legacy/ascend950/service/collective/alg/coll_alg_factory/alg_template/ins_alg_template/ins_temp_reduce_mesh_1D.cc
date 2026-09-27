/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_data_trans_wrapper.h"
#include "ins_temp_reduce_mesh_1D.h"

namespace Hccl {

InsTempReduceMesh1D::InsTempReduceMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempReduceMesh1D::~InsTempReduceMesh1D() {}

HcclResult InsTempReduceMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    HCCL_INFO("[InsTempReduceMesh1D] rank[%d] CalcRes start", myRank_);

    CHK_PRT_RET(tempRankSize_ == 0, HCCL_ERROR("[InsTempReduceMesh1D] rankSize is 0"), HcclResult::HCCL_E_INTERNAL);

    tempResReq.queNum = tempRankSize_;
    tempResReq.streamNum = tempResReq.queNum;
    tempResReq.queNotifys = CreateMasterSlaveQueNotifiesRequest(tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));

    HCCL_INFO(
        "[InsTempReduceMesh1D] rank[%d] CalcRes finished, need queNum[%u], queNotifyNum[%u], linkNum[%u]", myRank_,
        tempResReq.queNum, tempResReq.queNotifys.size(), tempResReq.links.size());
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempReduceMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) const
{
    (void)inBuffType;
    (void)outBuffType;

    CHK_PRT_RET(tempRankSize_ == 0, HCCL_ERROR("[InsTempReduceMesh1D] rankSize is 0"), HcclResult::HCCL_E_INTERNAL);

    HCCL_INFO("[InsTempReduceMesh1D] rank[%d] scratch multiple is [%u]", myRank_, tempRankSize_);
    return tempRankSize_;
}

HcclResult InsTempReduceMesh1D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& dataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    HCCL_INFO("[InsTempReduceMesh1D] rank[%d] GenExtIns start", myRank_);

    // 处理数据量为0场景
    if (dataParams.sliceSize == 0) {
        HCCL_INFO("[InsTempReduceMesh1D] sliceSize is 0, no need to process");
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_PRT_RET(tempRankSize_ == 0, HCCL_ERROR("[InsTempReduceMesh1D] rankSize is 0"), HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        tempVTopo_.size() != 1,
        HCCL_ERROR("[InsTempReduceMesh1D] level num of vtopo need to be 1, current is [%zu]", tempVTopo_.size()),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        tempVTopo_.at(0).size() != tempRankSize_,
        HCCL_ERROR(
            "[InsTempReduceMesh1D] rank num of level 0 in vtopo should be equal to rankSize[%u], current is [%zu]",
            tempRankSize_, tempVTopo_.at(0).size()),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(root_ == INVALID_U32, HCCL_ERROR("[InsTempReduceMesh1D] root is invalid"), HcclResult::HCCL_E_INTERNAL);

    opMode_ = tempFuncs.opMode;
    buffInfo_ = dataParams.buffInfo;

    queNum_ = tempRankSize_;
    CHK_PRT_RET(
        tempInsQues.size() != queNum_,
        HCCL_ERROR("[InsTempReduceMesh1D] resource queNum[%zu] is invalid, need[%u]", tempInsQues.size(), queNum_),
        HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(
        tempVirtRankMap_.count(myRank_) == 0,
        HCCL_ERROR("[InsTempReduceMesh1D] rank[%d] is not in virtRankMap", myRank_), HcclResult::HCCL_E_INTERNAL);
    myIdx_ = tempVirtRankMap_.at(myRank_);
    CHK_PRT_RET(
        myIdx_ >= tempRankSize_,
        HCCL_ERROR(
            "[InsTempReduceMesh1D] rank idx[%u] in virtRankMap is invalid, it should be less than rankSize[%u]", myIdx_,
            tempRankSize_),
        HcclResult::HCCL_E_INTERNAL);

    CHK_RET(RunReduce(dataParams, tempLinks, tempInsQues));

    HCCL_INFO("[InsTempReduceMesh1D] rank[%d] GenExtIns finished", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh1D::RunReduce(
    const TemplateDataParams& dataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues)
{
    if (u32(myRank_) == root_) {
        // 主从队列同步
        if (tempInsQues.size() > 1) {
            CHK_RET(PreSyncInterQueues(tempInsQues));
        }
        // Gather数据
        CHK_RET(GatherData(dataParams, tempLinks, tempInsQues));
        // 主从队列同步
        if (tempInsQues.size() > 1) {
            CHK_RET(PostSyncInterQueues(tempInsQues));
        }
        // 规约数据
        CHK_RET(ReduceData(dataParams, tempInsQues));
    } else {
        // Gather数据
        CHK_RET(SendData(dataParams, tempLinks, tempInsQues));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh1D::SendData(
    const TemplateDataParams& dataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues)
{
    DataSlice srcDataSlice(buffInfo_.inBuffType, buffInfo_.inBuffBaseOff, dataParams.sliceSize);

    const LinkData& SendLink = tempLinks.at(root_).at(0);

    DataSlice dstDataSlice(buffInfo_.scratBuffType, dataParams.sliceSize * myIdx_, dataParams.sliceSize);
    SlicesList sendSlicesList({srcDataSlice}, {dstDataSlice});
    DataInfo sendInfo(SendLink, sendSlicesList);

    CHK_PRT_RET(
        Send(sendInfo, tempInsQues.at(0), 0, true, DmaMode::PUT), HCCL_ERROR("[InsTempReduceMesh1D] Send data failed"),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh1D::GatherData(
    const TemplateDataParams& dataParams, const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues)
{
    DataSlice srcDataSlice(buffInfo_.inBuffType, buffInfo_.inBuffBaseOff, dataParams.sliceSize);

    // 主流将数据从inBuff拷贝到outBuff
    if (buffInfo_.inBuffType != buffInfo_.outBuffType) {
        DataSlice dstCopySlice(buffInfo_.outBuffType, buffInfo_.inBuffBaseOff, dataParams.sliceSize);
        CHK_PRT_RET(
            LocalCopy(tempInsQues.at(0), srcDataSlice, dstCopySlice),
            HCCL_ERROR("[InsTempReduceMesh1D] LocalCopy failed"), HcclResult::HCCL_E_INTERNAL);
    }

    // 单卡场景做完LocalCopy就直接返回
    if (tempRankSize_ == 1) {
        HCCL_INFO("[InsTempReduceMesh1D] rankSize is 1, copy data from inBuff to outBuff and return");
        return HcclResult::HCCL_SUCCESS;
    }

    // 从流接收来自其它rank的数据
    u32 queId = 1;
    for (u32 idx = 0; idx < tempVTopo_.at(0).size(); ++idx) {
        if (idx == myIdx_) {
            continue;
        }

        RankId rmtRank = tempVTopo_.at(0).at(idx);
        const LinkData& recvLink = tempLinks.at(rmtRank).at(0);

        DataSlice dstDataSlice(buffInfo_.scratBuffType, dataParams.sliceSize * idx, dataParams.sliceSize);
        SlicesList recvSlicesList({srcDataSlice}, {dstDataSlice});
        DataInfo recvInfo(recvLink, recvSlicesList);

        CHK_PRT_RET(
            Recv(recvInfo, tempInsQues.at(queId), 0, true, DmaMode::PUT),
            HCCL_ERROR("[InsTempReduceMesh1D] Recv data failed"), HcclResult::HCCL_E_INTERNAL);

        queId++;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh1D::ReduceData(const TemplateDataParams& dataParams, std::vector<InsQuePtr>& tempInsQues)
{
    if (tempRankSize_ == 1) {
        // 当rankSize为1时，数据已经拷贝至output，无需规约，直接返回
        return HcclResult::HCCL_SUCCESS;
    }

    DataSlice dstDataSlice(buffInfo_.outBuffType, buffInfo_.outBuffBaseOff, dataParams.sliceSize);

    for (u32 idx = 0; idx < tempVTopo_.at(0).size(); ++idx) {
        if (idx == myIdx_) {
            continue;
        }

        DataSlice srcDataSlice(buffInfo_.scratBuffType, dataParams.sliceSize * idx, dataParams.sliceSize);
        CHK_PRT_RET(
            LocalReduce(tempInsQues.at(0), srcDataSlice, dstDataSlice, dataType_, redOp_),
            HCCL_ERROR("[InsTempReduceMesh1D] Local reduce data failed"), HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
