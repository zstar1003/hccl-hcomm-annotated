/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "template_utils.h"
#include "log.h"
#include "buffer.h"

namespace Hccl {
HcclResult GetUnitAllignSize(const AllignInfo& allignInfo, u64& unitAlignSize)
{
    u32 dataSizePerVolume = DataTypeSizeGet(allignInfo.dataType);

    if (allignInfo.enableAllign) {
        CHK_PRT_RET(
            allignInfo.allignSize < dataSizePerVolume,
            HCCL_ERROR("[CollAlgFactory] Invalid input alignSize [%u].", allignInfo.allignSize),
            HcclResult::HCCL_E_PARA);
        unitAlignSize = (allignInfo.allignSize % dataSizePerVolume == 0) ? allignInfo.allignSize :
                                                                           allignInfo.allignSize * dataSizePerVolume;
    } else {
        unitAlignSize = dataSizePerVolume;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GetAlgRank(const RankId virtRank, const std::vector<RankId>& tempVTopo, u32& algRank)
{
    std::vector<RankId>::const_iterator topoVecIter = std::find(tempVTopo.begin(), tempVTopo.end(), virtRank);
    CHK_PRT_RET(
        topoVecIter == tempVTopo.end(), HCCL_ERROR("[CollAlgFactory] Invalid virtual Rank!"), HcclResult::HCCL_E_PARA);
    algRank = distance(tempVTopo.begin(), topoVecIter);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CalcRsAgSliceInfoConcurrMesh(
    const RankId myRank, const std::vector<std::vector<RankId>>& tempVTopo, const AllignInfo& allignInfo,
    const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    // multi-dimensional mesh
    u64 unitAlignSize;
    CHK_RET(GetUnitAllignSize(allignInfo, unitAlignSize));

    u32 dimSize0 = tempVTopo[0].size();
    u32 dimSize1 = tempVTopo[1].size();
    u64 sliceSize0
        = min(dataSize, RoundUp(dataSize, ((dimSize0 + dimSize1) * unitAlignSize)) * dimSize0 * unitAlignSize);
    u64 sliceSize1 = dataSize - sliceSize0;
    u64 accumOff = 0;
    u32 tempRankSize = dimSize0 * dimSize1;
    for (u32 rankIdx = 0; rankIdx < tempRankSize; rankIdx++) {
        SliceInfo slice0 = {accumOff, sliceSize0};
        sliceInfoVec[rankIdx][0] = slice0;
        accumOff += sliceSize0;

        SliceInfo slice1 = {accumOff, sliceSize1};
        sliceInfoVec[rankIdx][1] = slice1;
        accumOff += sliceSize1;
    }

    CHK_PRT_RET(
        (sliceInfoVec[tempRankSize - 1][1].offset + sliceInfoVec[tempRankSize - 1][1].size != dataSize * tempRankSize),
        HCCL_ERROR("[CollAlgFactory] Rank [%d], SliceInfo calculation error!", myRank), HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CalcRsAgSliceInfoMesh(
    const RankId myRank, const u32 tempRankSize, const AllignInfo& allignInfo, const u64 dataSize,
    RankSliceInfo& sliceInfoVec)
{
    (void)allignInfo;
    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < sliceInfoVec.size(); rankIdx++) {
        SliceInfo slice = {accumOff, dataSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += dataSize;
    }
    CHK_PRT_RET(
        (sliceInfoVec[tempRankSize - 1][0].offset + sliceInfoVec[tempRankSize - 1][0].size != dataSize * tempRankSize),
        HCCL_ERROR("[CollAlgFactory] Rank [%d], SliceInfo calculation error!", myRank), HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CalcRsAgSliceInfoRing(
    const RankId myRank, const std::vector<std::vector<RankId>>& tempVTopo, const AllignInfo& allignInfo,
    const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    u32 queNum = tempVTopo.size();
    u32 tempRankSize = tempVTopo[0].size();
    u64 unitAlignSize;
    CHK_RET(GetUnitAllignSize(allignInfo, unitAlignSize));

    u64 queSliceSize = RoundUp(dataSize, (queNum * unitAlignSize)) * unitAlignSize;

    u64 resChunkSize = dataSize;
    std::vector<u64> queSlice;
    for (u32 queIdx = 0; queIdx < queNum; queIdx++) {
        // split data on queues
        u64 currQueSliceSize = (resChunkSize > queSliceSize) ? queSliceSize : resChunkSize;
        queSlice.push_back(currQueSliceSize);
        resChunkSize -= currQueSliceSize;
    }
    CHK_PRT_RET(
        resChunkSize != 0, HCCL_ERROR("[CollAlgFactory] Rank [%d], SliceInfo calculation error!", myRank),
        HcclResult::HCCL_E_INTERNAL);

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < tempRankSize; rankIdx++) {
        for (u32 queIdx = 0; queIdx < queNum; queIdx++) {
            u64 currSliceSize = queSlice[queIdx];
            SliceInfo currSlice = {accumOff, currSliceSize};
            accumOff += currSliceSize;
            sliceInfoVec[rankIdx][queIdx] = currSlice;
        }

        CHK_PRT_RET(
            (accumOff != dataSize * (rankIdx + 1)),
            HCCL_ERROR("[CollAlgFactory] Rank [%d], SliceInfo calculation error!", myRank),
            HcclResult::HCCL_E_INTERNAL);
    }

    CHK_PRT_RET(
        (sliceInfoVec[tempRankSize - 1][queNum - 1].offset + sliceInfoVec[tempRankSize - 1][queNum - 1].size
         != dataSize * tempRankSize),
        HCCL_ERROR("[CollAlgFactory] Rank [%d], SliceInfo calculation error!", myRank), HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CalcRsAgSliceInfoNHR(
    const RankId myRank, const u32 tempRankSize, const AllignInfo& allignInfo, const u64 dataSize,
    RankSliceInfo& sliceInfoVec)
{
    (void)allignInfo;
    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < sliceInfoVec.size(); rankIdx++) {
        SliceInfo slice = {accumOff, dataSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += dataSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec[tempRankSize - 1][0].offset + sliceInfoVec[tempRankSize - 1][0].size != dataSize * tempRankSize),
        HCCL_ERROR("[CollAlgFactory] Rank [%d], SliceInfo calculation error!", myRank), HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CalcResLinksMesh(
    const RankId myRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const u32 linkNumBtwPeers, AlgTempResReq& tempResReq)
{
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank, tempVTopo[0], myAlgRank));

    for (u32 queIdx = 0; queIdx < tempVTopo[0].size() - 1; queIdx++) {
        // find neighbors : virtualRank
        RankId neighborRank = tempVTopo[0][(myAlgRank + 1 + queIdx) % tempRankSize];

        // LinkNum
        tempResReq.links[neighborRank] = linkNumBtwPeers;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CalcResLinksMesh2D(
    const RankId myRank, const std::vector<std::vector<RankId>>& tempVTopo, const u32 linkNumBtwPeers,
    AlgTempResReq& tempResReq)
{
    u32 myAlgRank;
    for (u32 dim = 0; dim < tempVTopo.size(); dim++) {
        CHK_RET(GetAlgRank(myRank, tempVTopo[dim], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempVTopo[dim].size() - 1; queIdx++) {
            u32 neighborAlgRank = (myAlgRank + 1 + queIdx) % (tempVTopo[dim].size());
            CHK_PRT_RET(neighborAlgRank > (tempVTopo[dim].size() - 1),
                        HCCL_ERROR(
                            "[CalcResLinksMesh2D] neighborAlgRank[%u] is invalid, "
                            "the Max rank[%zu].",
                            neighborAlgRank, tempVTopo[dim].size() - 1);
                        , HcclResult::HCCL_E_INTERNAL);
            RankId neighborRank = tempVTopo[dim][neighborAlgRank];
            tempResReq.links[neighborRank] = linkNumBtwPeers;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GetDetourSendRecvLinksIn4P(
    const RankId myRank, const RankId neighborRank, const ResLinks& tempLinks,
    std::vector<std::vector<LinkDataIterator>>& sendRecvLinks)
{
    HCCL_DEBUG("[CollAlgFactory] [GetDetourSendRecvLinksIn4P] Rank [%d], NeighborRank [%d].", myRank, neighborRank);
    LinkDataIterator neighborLinkDataIter = tempLinks.at(neighborRank).begin();
    while (neighborLinkDataIter != tempLinks.at(neighborRank).end()) {
        if ((*neighborLinkDataIter).GetDirection() == LinkDirection::BOTH) {
            sendRecvLinks[0][0] = (neighborLinkDataIter);
            sendRecvLinks[0][1] = (neighborLinkDataIter);
        } else if ((*neighborLinkDataIter).GetDirection() == LinkDirection::RECV_ONLY) {
            // 当前算法是根据Linkdata属性来判断哪条绕路链路负责收数据，哪条负责发数据，后续方案会改进
            sendRecvLinks[1][1] = (neighborLinkDataIter);
        } else {
            sendRecvLinks[1][0] = (neighborLinkDataIter);
        }
        neighborLinkDataIter++;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CalcResLinksRing(
    const RankId myRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    AlgTempResReq& tempResReq)
{
    std::vector<std::vector<RankId>>::const_iterator tempVTopoIter;
    for (tempVTopoIter = tempVTopo.begin(); tempVTopoIter != tempVTopo.end(); tempVTopoIter++) {
        // locate myRank in tempVTopo -> algRank
        u32 myAlgRank;
        CHK_RET(GetAlgRank(myRank, (*tempVTopoIter), myAlgRank));

        // find neighbors -> virtualRank
        RankId sendToRank = tempVTopoIter->at((myAlgRank + 1) % tempRankSize);
        RankId recvFromRank = tempVTopoIter->at((myAlgRank - 1 + tempRankSize) % tempRankSize); // virtualRank

        // LinkNum
        tempResReq.links[sendToRank] = 1;
        tempResReq.links[recvFromRank] = 1;
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 GetLinkNum(const RankGraph* rankGraph, RankId srcRank, RankId dstRank)
{
    std::set<u32> levelSet = rankGraph->GetLevels(srcRank);
    u32 linkNum = 0;
    for (u32 levelIdx : levelSet) {
        std::vector<NetInstance::Path> paths = rankGraph->GetPaths(levelIdx, srcRank, dstRank);
        linkNum += paths.size();
    }
    return linkNum;
}

// NHR的算法步数 = Ceil(log2(N))
u32 GetNHRStepNum(u32 rankSize)
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[NHRBase][GetStepNumInterServer] rankSize[%u] nSteps[%u]", rankSize, nSteps);

    return nSteps;
}

HcclResult CalcResLinksNHR(
    const RankId myRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    AlgTempResReq& tempResReq)
{
    CHK_PRT_RET(
        tempVTopo.size() != 1,
        HCCL_ERROR("[CollAlgFactory][CalcResLinksNHR] invalid tempVTopo size[%zu]", tempVTopo.size()),
        HcclResult::HCCL_E_PARA);
    const std::vector<RankId>& tree = tempVTopo[0];
    CHK_PRT_RET(
        tree.size() != tempRankSize,
        HCCL_ERROR("[CollAlgFactory][CalcResLinksNHR] tempRankSize[%u] != tree.size[%zu]", tempRankSize, tree.size()),
        HcclResult::HCCL_E_PARA);
    u32 nSteps = GetNHRStepNum(tempRankSize);

    RankId sendToRank;
    RankId recvFromRank;
    // locate myRank in tempVTopo -> algRank
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank, tree, myAlgRank));

    for (u32 currentStep = 0; currentStep < nSteps; currentStep++) {
        u32 deltaRank = nSteps - 1 - currentStep;
        // send info
        sendToRank = tree[(myAlgRank + (1 << deltaRank)) % tempRankSize];
        // receive Info
        recvFromRank = tree[(myAlgRank + tempRankSize - (1 << deltaRank)) % tempRankSize];
        tempResReq.links[sendToRank] = 1;
        tempResReq.links[recvFromRank] = 1;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GetLocalSendRecvInfoforAlltoall(
    const CollAlgOperator& opParam, const u32 userRank, const u32 userRankSize, A2ASendRecvInfo& localSendRecvInfo)
{
    u64 curSendDispls = 0;
    u64 curSendOffset = 0;
    u64 curRecvDispls = 0;
    u64 curRecvOffset = 0;
    for (u32 j = 0; j < userRankSize; j++) {
        u64 curSendCounts = opParam.all2AllDataDes.sendCount;
        u64 curSendLength = curSendCounts * DataTypeSizeGet(opParam.all2AllDataDes.sendType);
        localSendRecvInfo.sendCounts[j] = curSendCounts;
        localSendRecvInfo.sendDispls[j] = curSendDispls;
        localSendRecvInfo.sendLength[j] = curSendLength;
        localSendRecvInfo.sendOffset[j] = curSendOffset;
        curSendDispls += curSendCounts;
        curSendOffset += curSendLength;

        u64 curRecvCounts = opParam.all2AllDataDes.sendCount;
        u64 curRecvLength = curRecvCounts * DataTypeSizeGet(opParam.all2AllDataDes.recvType);
        localSendRecvInfo.recvCounts[j] = curRecvCounts;
        localSendRecvInfo.recvDispls[j] = curRecvDispls;
        localSendRecvInfo.recvLength[j] = curRecvLength;
        localSendRecvInfo.recvOffset[j] = curRecvOffset;
        curRecvDispls += curRecvCounts;
        curRecvOffset += curRecvLength;
        HCCL_DEBUG(
            "[GetLocalSendRecvInfoforAlltoall] rank[%u], sendCounts[%llu], sendDispls[%llu] "
            "recvCounts[%llu], recvDispls[%llu], sendLength[%llu], recvLength[%llu]",
            userRank, localSendRecvInfo.sendCounts[j], localSendRecvInfo.sendDispls[j], localSendRecvInfo.recvCounts[j],
            localSendRecvInfo.recvDispls[j], localSendRecvInfo.sendLength[j], localSendRecvInfo.recvLength[j]);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GetLocalSendRecvInfoforAlltoallV(
    const CollAlgOperator& opParam, const u32 userRank, const u32 userRankSize, A2ASendRecvInfo& localSendRecvInfo)
{
    CHK_PTR_NULL(opParam.all2AllVDataDes.sendCounts);
    CHK_PTR_NULL(opParam.all2AllVDataDes.sdispls);
    CHK_PTR_NULL(opParam.all2AllVDataDes.recvCounts);
    CHK_PTR_NULL(opParam.all2AllVDataDes.rdispls);
    for (u32 j = 0; j < userRankSize; j++) {
        u64 curSendCounts = *(static_cast<const u64*>(opParam.all2AllVDataDes.sendCounts) + j);
        u64 curSendDispls = *(static_cast<const u64*>(opParam.all2AllVDataDes.sdispls) + j);
        localSendRecvInfo.sendCounts[j] = curSendCounts;
        localSendRecvInfo.sendDispls[j] = curSendDispls;
        localSendRecvInfo.sendLength[j] = curSendCounts * DataTypeSizeGet(opParam.all2AllVDataDes.sendType);
        localSendRecvInfo.sendOffset[j] = curSendDispls * DataTypeSizeGet(opParam.all2AllVDataDes.sendType);

        u64 curRecvCounts = *(static_cast<const u64*>(opParam.all2AllVDataDes.recvCounts) + j);
        u64 curRecvDispls = *(static_cast<const u64*>(opParam.all2AllVDataDes.rdispls) + j);
        localSendRecvInfo.recvCounts[j] = curRecvCounts;
        localSendRecvInfo.recvDispls[j] = curRecvDispls;
        localSendRecvInfo.recvLength[j] = curRecvCounts * DataTypeSizeGet(opParam.all2AllVDataDes.recvType);
        localSendRecvInfo.recvOffset[j] = curRecvDispls * DataTypeSizeGet(opParam.all2AllVDataDes.recvType);

        HCCL_DEBUG(
            "[GetLocalSendRecvInfoforAlltoallV] rank[%u], sendCounts[%llu], sendDispls[%llu] "
            "recvCounts[%llu], recvDispls[%llu], sendLength[%llu], recvLength[%llu]",
            userRank, localSendRecvInfo.sendCounts[j], localSendRecvInfo.sendDispls[j], localSendRecvInfo.recvCounts[j],
            localSendRecvInfo.recvDispls[j], localSendRecvInfo.sendLength[j], localSendRecvInfo.recvLength[j]);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GetLocalSendRecvInfoforAlltoallVC(
    const CollAlgOperator& opParam, const u32 userRank, const u32 userRankSize, A2ASendRecvInfo& localSendRecvInfo)
{
    u64 curSendDispls = 0;
    u64 curSendOffset = 0;
    u64 curRecvDispls = 0;
    u64 curRecvOffset = 0;
    for (u32 j = 0; j < userRankSize; j++) {
        u64 curSendCounts
            = *(static_cast<const u64*>(opParam.all2AllVCDataDes.sendCountMatrix) + userRank * userRankSize + j);
        u64 curSendLength = curSendCounts * DataTypeSizeGet(opParam.all2AllVCDataDes.sendType);
        localSendRecvInfo.sendCounts[j] = curSendCounts;
        localSendRecvInfo.sendDispls[j] = curSendDispls;
        localSendRecvInfo.sendLength[j] = curSendLength;
        localSendRecvInfo.sendOffset[j] = curSendOffset;
        curSendDispls += curSendCounts;
        curSendOffset += curSendLength;

        u64 curRecvCounts
            = *(static_cast<const u64*>(opParam.all2AllVCDataDes.sendCountMatrix) + userRank + userRankSize * j);
        u64 curRecvLength = curRecvCounts * DataTypeSizeGet(opParam.all2AllVCDataDes.recvType);
        localSendRecvInfo.recvCounts[j] = curRecvCounts;
        localSendRecvInfo.recvDispls[j] = curRecvDispls;
        localSendRecvInfo.recvLength[j] = curRecvLength;
        localSendRecvInfo.recvOffset[j] = curRecvOffset;
        curRecvDispls += curRecvCounts;
        curRecvOffset += curRecvLength;
        HCCL_DEBUG(
            "[GetLocalSendRecvInfoforAlltoallVC] rank[%u], sendCounts[%llu], sendDispls[%llu] "
            "recvCounts[%llu], recvDispls[%llu]",
            userRank, localSendRecvInfo.sendCounts[j], localSendRecvInfo.sendDispls[j], localSendRecvInfo.recvCounts[j],
            localSendRecvInfo.recvDispls[j]);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GetAlltoAllLocalSendRecvInfo(
    const CollAlgOperator& opParam, const u32 userRank, const u32 userRankSize, A2ASendRecvInfo& localSendRecvInfo)
{
    HCCL_DEBUG("[GetAlltoAllLocalSendRecvInfo] rank[%u], userRankSize[%u]", userRank, userRankSize);
    localSendRecvInfo.sendCounts.resize(userRankSize, 0);
    localSendRecvInfo.sendDispls.resize(userRankSize, 0);
    localSendRecvInfo.sendLength.resize(userRankSize, 0);
    localSendRecvInfo.sendOffset.resize(userRankSize, 0);

    localSendRecvInfo.recvCounts.resize(userRankSize, 0);
    localSendRecvInfo.recvDispls.resize(userRankSize, 0);
    localSendRecvInfo.recvLength.resize(userRankSize, 0);
    localSendRecvInfo.recvOffset.resize(userRankSize, 0);
    if (opParam.opType == OpType::ALLTOALLV) {
        CHK_RET(GetLocalSendRecvInfoforAlltoallV(opParam, userRank, userRankSize, localSendRecvInfo));
    } else if (opParam.opType == OpType::ALLTOALL) {
        CHK_RET(GetLocalSendRecvInfoforAlltoall(opParam, userRank, userRankSize, localSendRecvInfo));
    } else if (opParam.opType == OpType::ALLTOALLVC) {
        CHK_RET(GetLocalSendRecvInfoforAlltoallVC(opParam, userRank, userRankSize, localSendRecvInfo));
    } else if (opParam.opType != OpType::HALFALLTOALLV) {
        HCCL_ERROR("Only support optype alltoall , alltoallv, halfalltoallv and alltoallvc !");
    }
    HCCL_DEBUG("[GetAlltoAllLocalSendRecvInfo] GetAlltoAllLocalSendRecvInfo success");
    return HcclResult::HCCL_SUCCESS;
}

/*
 * 一个基本的 Allreduce 数据切分函数，用于ReduceScatter + Allgather组合成的 Allreduce 算。
 * 输入的 dataSize 是一张卡上完整的数据量
 * 函数会将 dataSize 切分成 rankSize 份，最后一份尾块可能会比其他的切分出来的子块大。
 */
HcclResult CalcSliceInfoAllReduce(
    const AllignInfo& allignInfo, const u32 rankSize, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    sliceInfoVec.clear();
    sliceInfoVec.resize(rankSize);

    u32 dataSizePerVolume = DataTypeSizeGet(allignInfo.dataType);
    u64 unitAlignSize;
    CHK_RET(GetUnitAllignSize(allignInfo, unitAlignSize));
    u64 unitPerSlice = dataSize / unitAlignSize / rankSize;
    HCCL_DEBUG("unitAlignSize[%llu] unitPerSlice[%llu]", unitAlignSize, unitPerSlice);

    u64 accumOff = 0;
    SliceInfo currSlice;
    for (u32 rankIdx = 0; rankIdx < rankSize; rankIdx++) {
        if (rankIdx == rankSize - 1) {
            currSlice.offset = accumOff;
            currSlice.size = dataSize - accumOff;
        } else {
            currSlice.offset = accumOff;
            currSlice.size = unitPerSlice * unitAlignSize;
        }
        CHK_PRT_RET(
            currSlice.size % dataSizePerVolume != 0,
            HCCL_ERROR(
                "[Calc][SliceInfo]rank[%u] slice size[%llu] is invalid, dataSizePerVolume[%llu]", rankIdx,
                currSlice.size, dataSizePerVolume),
            HcclResult::HCCL_E_INTERNAL);
        sliceInfoVec[rankIdx].push_back(currSlice);
        accumOff += currSlice.size;
    }

    CHK_PRT_RET(
        (sliceInfoVec[rankSize - 1][0].offset + sliceInfoVec[rankSize - 1][0].size != dataSize),
        HCCL_ERROR(
            "[CalcSliceInfoAllReduce] SliceInfo calculation error! DataSize[%llu], "
            "lastoffset[%llu], lastsize[%llu]",
            dataSize, sliceInfoVec[rankSize - 1][0].offset, sliceInfoVec[rankSize - 1][0].size),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult BufferTypeToAddr(const BufferType& bufferType, CollAlgOperator& op, uint64_t& addr)
{
    Buffer* buffer = op.GetBuffer(bufferType);
    CHK_PTR_NULL(buffer);
    addr = buffer->GetAddr();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CalcDataSplitRateForLinks(const std::vector<LinkData>& links, std::vector<float>& dataSplitRate)
{
    // 取到第一个对端的link数量来作为数据切分的依据
    std::vector<u8> linkPortGroupSizes;
    linkPortGroupSizes.resize(links.size());
    for (u32 linkIdx = 0; linkIdx < links.size(); linkIdx++) {
        const LinkData& linkData = links[linkIdx];
        linkPortGroupSizes[linkIdx] = linkData.GetPortGroupSize();
    }
    u32 totalPortNum = accumulate(linkPortGroupSizes.begin(), linkPortGroupSizes.end(), 0);
    if (totalPortNum == 0) {
        HCCL_ERROR("totalPortNum is zero");
        return HcclResult::HCCL_E_INTERNAL;
    }
    for (u32 linkIdx = 0; linkIdx < linkPortGroupSizes.size(); linkIdx++) {
        dataSplitRate[linkIdx] = static_cast<float>(linkPortGroupSizes[linkIdx]) / totalPortNum;
    }
    return HcclResult::HCCL_SUCCESS;
}

DataSlice CalcDataSliceForLinks(
    const DataSlice& recvSrcSliceAllLinks, std::vector<float> dataSplitRate, u32 j, DataType dataType_)
{
    BufferType type = recvSrcSliceAllLinks.GetType();
    u64 offset = recvSrcSliceAllLinks.GetOffset();
    u64 size = recvSrcSliceAllLinks.GetSize();
    u64 AccSize = 0;
    u64 typeSize = DataTypeSizeGet(dataType_);
    u64 dataCnt = size / typeSize;
    u64 linkNum = dataSplitRate.size();
    std::vector<DataSlice> dataSliceForLinks(linkNum);
    HCCL_INFO("[InsTempAllGatherNHR] Slice data for links");
    for (u32 linkIdx = 0; linkIdx < linkNum; linkIdx++) {
        if (linkIdx != linkNum - 1) {
            dataSliceForLinks[linkIdx].SetSize(
                static_cast<u64>(static_cast<float>(dataCnt) * dataSplitRate[linkIdx]) * typeSize);
        } else {
            dataSliceForLinks[linkIdx].SetSize(size - AccSize);
        }
        dataSliceForLinks[linkIdx].SetOffset(offset + AccSize);
        AccSize += dataSliceForLinks[linkIdx].GetSize();
        dataSliceForLinks[linkIdx].SetBufferType(type);
    }
    return dataSliceForLinks[j];
}

} // namespace Hccl
