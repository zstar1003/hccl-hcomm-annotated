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
#include "ins_temp_reduce_mesh_2D.h"

namespace Hccl {

InsTempReduceMesh2D::InsTempReduceMesh2D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : InsAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

InsTempReduceMesh2D::~InsTempReduceMesh2D() {}

HcclResult InsTempReduceMesh2D::CalcRes(AlgTempResReq& tempResReq)
{
    HCCL_INFO("[InsTempReduceMesh2D] Calculate communication resources start");

    CHK_PRT_RET(
        tempVTopo_.size() != AXIS_NUM,
        HCCL_ERROR(
            "[InsTempReduceMesh2D] The dimension of topo is invalid, expect [%u], now is [%u]", AXIS_NUM,
            tempVTopo_.size()),
        HcclResult::HCCL_E_INTERNAL);

    axisRankSize_[AXIS_X] = tempVTopo_.at(AXIS_X).size();
    axisRankSize_[AXIS_Y] = tempVTopo_.at(AXIS_Y).size();

    CHK_PRT_RET(
        axisRankSize_[AXIS_X] == 0 || axisRankSize_[AXIS_Y] == 0,
        HCCL_ERROR(
            "[InsTempReduceMesh2D] The rankSize of dimension is invalid, xRankSize is [%u], yRankSize is [%u]",
            axisRankSize_[AXIS_X], axisRankSize_[AXIS_Y]),
        HcclResult::HCCL_E_INTERNAL);

    tempResReq.queNum = axisRankSize_[AXIS_X] + axisRankSize_[AXIS_Y];
    tempResReq.streamNum = tempResReq.queNum;

    tempResReq.queNotifys = CreateNotifiesRequest(axisRankSize_[AXIS_X], axisRankSize_[AXIS_Y]);

    CHK_RET(CalcResLinksConcurrMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));

    HCCL_INFO(
        "[InsTempReduceMesh2D] Calculate communication resources finished, queNum[%u], streamNum[%u], "
        "queNotifyNum[%zu] linkNum[%zu]",
        tempResReq.queNum, tempResReq.streamNum, tempResReq.queNotifys.size(), tempResReq.links.size());

    return HcclResult::HCCL_SUCCESS;
}

std::vector<std::tuple<QId, QId, u32>> InsTempReduceMesh2D::CreateNotifiesRequest(u32 xQueueNum, u32 yQueueNum) const
{
    std::vector<std::tuple<QId, QId, u32>> notifyRequests;
    if (xQueueNum == 0) {
        HCCL_INFO("[InsTempReduceMesh2D] xQueueNum is zero, return empty notifyRequests");
        return notifyRequests;
    }
    if (yQueueNum == 0) {
        HCCL_INFO("[InsTempReduceMesh2D] yQueueNum is zero, return empty notifyRequests");
        return notifyRequests;
    };
    u32 queueNum = xQueueNum + yQueueNum;

    u32 slaveNum = queueNum - 1;
    if (slaveNum < 1) {
        HCCL_INFO("[InsTempReduceMesh2D] slaveNum is zero, return empty notifyRequests");
        return notifyRequests;
    }

    u32 ctrlNotfiyReqNum = 2;                // X轴向的控制流（主流）和Y轴向的控制流之间的Notify
    u32 xNotifyReqNum = (xQueueNum - 1) * 2; // X轴向的控制流和业务流之间的Notify
    u32 yNotifyReqNum = (yQueueNum - 1) * 2; // Y轴向的控制流和业务流之间的Notify
    u32 totalNotifyReqNum = ctrlNotfiyReqNum + xNotifyReqNum + yNotifyReqNum;
    notifyRequests.reserve(totalNotifyReqNum);

    QId xCtrlId = 0;
    QId yCtrlId = xCtrlId + xNotifyReqNum;

    notifyRequests.emplace_back(std::make_tuple(xCtrlId, yCtrlId, 0));
    notifyRequests.emplace_back(std::make_tuple(yCtrlId, xCtrlId, 0));

    for (QId xId = xCtrlId + 1; xId < xNotifyReqNum; ++xId) {
        notifyRequests.emplace_back(std::make_tuple(xCtrlId, xId, 0));
        notifyRequests.emplace_back(std::make_tuple(xId, xCtrlId, 0));
    }

    for (QId yId = yCtrlId + 1; yId < xNotifyReqNum + yNotifyReqNum; ++yId) {
        notifyRequests.emplace_back(std::make_tuple(yCtrlId, yId, 0));
        notifyRequests.emplace_back(std::make_tuple(yId, yCtrlId, 0));
    }

    HCCL_DEBUG(
        "[InsTempReduceMesh2D] Create notifies request: "
        "totalNotifyReqNum[%u], ctrlNotifyReqNum[%u], xNotifyReqNum[%u], yNotifyReqNum[%u]",
        totalNotifyReqNum, ctrlNotfiyReqNum, xNotifyReqNum, yNotifyReqNum);

    return notifyRequests;
}

HcclResult InsTempReduceMesh2D::CalcResLinksConcurrMesh(
    const RankId myRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const u32 linkNumBtwPeers, AlgTempResReq& tempResReq) const
{
    (void)tempRankSize;
    u32 myAlgRank;
    for (u32 dim = 0; dim < tempVTopo.size(); dim++) {
        CHK_RET(GetAlgRank(myRank, tempVTopo[dim], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempVTopo[dim].size() - 1; queIdx++) {
            RankId neighborRank = tempVTopo[dim][(myAlgRank + 1 + queIdx) % (tempVTopo[dim].size())];
            tempResReq.links[neighborRank] = linkNumBtwPeers;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempReduceMesh2D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;

    // 数据会在2个维度间切换通信，选择最大的维度切分scratch方便数据处理
    u32 scratchMultiple = max(tempVTopo_.at(AXIS_X).size(), tempVTopo_.at(AXIS_Y).size());
    HCCL_INFO("[InsTempReduceMesh2D] Scratch multiple is [%u]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempReduceMesh2D::GenExtIns(
    const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    (void)tempFuncs;
    HCCL_INFO("[InsTempReduceMesh2D] GenExtIns start rank[%d]", myRank_);

    CHK_RET(CalcParams(templateDataParams));

    // 单卡场景可以直接Input拷贝到Output，单独判断
    if (tempRankSize_ == 1) {
        CHK_RET(LocalCopyFromInputToOutput(templateDataParams, tempInsQues));
        return HcclResult::HCCL_SUCCESS;
    }

    // 将队列分为2组，一组负责X轴向通信，一组负责Y轴向通信，两组中的第一条流兼任控制流
    std::vector<InsQuePtr> ctrlTempInsQues;
    std::vector<InsQuePtr> xTempInsQues;
    std::vector<InsQuePtr> yTempInsQues;
    CHK_RET(SplitInsQues(tempInsQues, ctrlTempInsQues, xTempInsQues, yTempInsQues));

    CHK_RET(PreSyncInterQueues(ctrlTempInsQues)); // XY轴并行启动

    if (u32(myRank_) == root_) {
        // 数据片A第一步通信
        CHK_RET(GatherFromInput(SLICE_A, AXIS_X, tempLinks, xTempInsQues));
        CHK_RET(ReduceToScratch(SLICE_A, AXIS_X, xTempInsQues));
        // 数据片B第一步通信
        CHK_RET(GatherFromInput(SLICE_B, AXIS_Y, tempLinks, yTempInsQues));
        CHK_RET(ReduceToScratch(SLICE_B, AXIS_Y, yTempInsQues));
        // X轴和Y轴控制流同步，然后交换处理数据
        CHK_RET(PreSyncInterQueues(ctrlTempInsQues));
        CHK_RET(PostSyncInterQueues(ctrlTempInsQues));
        // 数据片A第二步通信
        CHK_RET(GatherFromScratch(SLICE_A, AXIS_Y, tempLinks, yTempInsQues));
        CHK_RET(ReduceToOutput(SLICE_A, AXIS_Y, yTempInsQues));
        // 数据片B第二步通信
        CHK_RET(GatherFromScratch(SLICE_B, AXIS_X, tempLinks, xTempInsQues));
        CHK_RET(ReduceToOutput(SLICE_B, AXIS_X, xTempInsQues));
    } else if (axisRank_[AXIS_X] == axisRoot_[AXIS_X]) {
        // 数据片A通信
        CHK_RET(GatherFromInput(SLICE_A, AXIS_X, tempLinks, xTempInsQues));
        CHK_RET(ReduceToScratch(SLICE_A, AXIS_X, xTempInsQues));
        CHK_RET(SendFromScratch(SLICE_A, AXIS_Y, tempLinks, xTempInsQues));
        // 数据片B通信
        CHK_RET(SendFromInput(SLICE_B, AXIS_Y, tempLinks, yTempInsQues));
    } else if (axisRank_[AXIS_Y] == axisRoot_[AXIS_Y]) {
        // 数据片A通信
        CHK_RET(SendFromInput(SLICE_A, AXIS_X, tempLinks, xTempInsQues));
        // 数据片B通信
        CHK_RET(GatherFromInput(SLICE_B, AXIS_Y, tempLinks, yTempInsQues));
        CHK_RET(ReduceToScratch(SLICE_B, AXIS_Y, yTempInsQues));
        CHK_RET(SendFromScratch(SLICE_B, AXIS_X, tempLinks, yTempInsQues));
    } else {
        // 数据片A通信
        CHK_RET(SendFromInput(SLICE_A, AXIS_X, tempLinks, xTempInsQues));
        // 数据片B通信
        CHK_RET(SendFromInput(SLICE_B, AXIS_Y, tempLinks, yTempInsQues));
    }

    CHK_RET(PostSyncInterQueues(ctrlTempInsQues)); // 返回主流

    HCCL_INFO("[InsTempReduceMesh2D] GenExtIns finished rank[%d]", myRank_);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh2D::CalcParams(const TemplateDataParams& templateDataParams)
{
    axisRankSize_[AXIS_X] = tempVTopo_.at(AXIS_X).size();
    axisRankSize_[AXIS_Y] = tempVTopo_.at(AXIS_Y).size();
    axisRank_[AXIS_X] = u32(myRank_) % axisRankSize_[AXIS_X];
    axisRank_[AXIS_Y] = u32(myRank_) / axisRankSize_[AXIS_X];
    axisRoot_[AXIS_X] = root_ % axisRankSize_[AXIS_X];
    axisRoot_[AXIS_Y] = root_ / axisRankSize_[AXIS_X];

    u32 dataTypeSize = DataTypeSizeGet(dataType_);
    // 用count均分，防止数据截断；并且保证在奇数情况下SLICE_A的切分比SLICE_B大
    sliceSize_[SLICE_A] = (templateDataParams.sliceSize / dataTypeSize + 1) / SLICE_NUM * dataTypeSize;
    sliceSize_[SLICE_B] = templateDataParams.sliceSize - sliceSize_[SLICE_A];

    sliceInputBaseOffset_[SLICE_A] = templateDataParams.buffInfo.inBuffBaseOff;
    sliceInputBaseOffset_[SLICE_B] = sliceInputBaseOffset_[SLICE_A] + sliceSize_[SLICE_A];

    sliceOutputBaseOffset_[SLICE_A] = templateDataParams.buffInfo.outBuffBaseOff;
    sliceOutputBaseOffset_[SLICE_B] = sliceOutputBaseOffset_[SLICE_A] + sliceSize_[SLICE_A];

    // Scratch切分时，上下两部分都按照最大的轴向RankSize来切分，从而保证数据换轴通信时有足够的暂存Buffer来做确定性计算
    u32 maxAxisRankSize = max(axisRankSize_[AXIS_X], axisRankSize_[AXIS_Y]);
    sliceScratchBaseOffset_[SLICE_A] = templateDataParams.buffInfo.scratchBuffBaseOff;
    sliceScratchBaseOffset_[SLICE_B] = sliceScratchBaseOffset_[SLICE_A] + sliceSize_[SLICE_A] * maxAxisRankSize;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh2D::SplitInsQues(
    std::vector<InsQuePtr>& tempInsQues, std::vector<InsQuePtr>& ctrlTempInsQues, std::vector<InsQuePtr>& xTempInsQues,
    std::vector<InsQuePtr>& yTempInsQues)
{
    u32 expectQueNum = axisRankSize_[AXIS_X] + axisRankSize_[AXIS_Y];
    CHK_PRT_RET(
        tempInsQues.size() != expectQueNum,
        HCCL_ERROR(
            "[InsTempReduceMesh2D] The count of queues is invalid, expect [%u], now is [%u]", expectQueNum,
            tempInsQues.size()),
        HcclResult::HCCL_E_INTERNAL);

    ctrlTempInsQues.emplace_back(tempInsQues.at(0));
    ctrlTempInsQues.emplace_back(tempInsQues.at(axisRankSize_[AXIS_X]));
    xTempInsQues = std::vector<InsQuePtr>(tempInsQues.begin(), tempInsQues.begin() + axisRankSize_[AXIS_X]);
    yTempInsQues = std::vector<InsQuePtr>(tempInsQues.begin() + axisRankSize_[AXIS_X], tempInsQues.end());

    HCCL_INFO(
        "[InsTempReduceMesh2D] splitInsQues success, ctrlTempInsQuesNum[%zu], xTempInsQuesNum[%zu], "
        "yTempInsQuesNum[%zu]",
        ctrlTempInsQues.size(), xTempInsQues.size(), yTempInsQues.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh2D::LocalCopyFromInputToOutput(
    const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues) const
{
    DataSlice srcLocalSlice(BufferType::INPUT, 0, templateDataParams.sliceSize);
    DataSlice dstLocalSlice(BufferType::OUTPUT, 0, templateDataParams.sliceSize);
    CHK_PRT_RET(
        LocalCopy(tempInsQues[0], srcLocalSlice, dstLocalSlice),
        HCCL_ERROR("[InsTempReduceMesh2D] LocalCopy data failed"), HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh2D::GatherFromInput(
    const u32 slice, const u32 axis, const ResLinks& tempLinks, std::vector<InsQuePtr>& axisTempInsQues)
{
    HCCL_DEBUG("[InsTempReduceMesh2D] Gather from input start.");

    CHK_PRT_RET(
        axisTempInsQues.empty(), HCCL_ERROR("[InsTempReduceMesh2D][GatherFromInput] axisTempInsQues is empty."),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(axisTempInsQues[0]);
    u64 sliceSize = sliceSize_[slice];
    u64 sliceScratchBaseOffset = sliceScratchBaseOffset_[slice];

    DataSlice srcDataSlice(BufferType::INPUT, sliceInputBaseOffset_[slice], sliceSize);

    if (axisTempInsQues.size() > 1) {
        CHK_RET(PreSyncInterQueues(axisTempInsQues));
    }

    // 主队列本地拷贝，从Input拷贝到Scratch
    DataSlice dstLocalSlice(BufferType::SCRATCH, sliceScratchBaseOffset + axisRoot_[axis] * sliceSize, sliceSize);
    CHK_PRT_RET(
        LocalCopy(axisTempInsQues[0], srcDataSlice, dstLocalSlice),
        HCCL_ERROR("[InsTempReduceMesh2D] LocalCopy data failed"), HcclResult::HCCL_E_INTERNAL);

    // 从队列负责接收来自其它rank的数据
    u32 queIdx = 1;
    for (u32 axisRank = 0; axisRank < tempVTopo_.at(axis).size(); ++axisRank) {
        RankId rmtRank = tempVTopo_.at(axis).at(axisRank);
        if (rmtRank == myRank_) {
            continue;
        }

        const LinkData& recvLink = tempLinks.at(rmtRank).at(0);
        // 按照发送rank的序号来计算接收数据存放的偏移
        DataSlice dstDataSlice(BufferType::SCRATCH, sliceScratchBaseOffset + axisRank * sliceSize, sliceSize);
        SlicesList recvSlicesList({srcDataSlice}, {dstDataSlice});
        DataInfo recvInfo(recvLink, recvSlicesList);
        CHK_PRT_THROW(
            queIdx >= axisTempInsQues.size(),
            HCCL_ERROR(
                "[InsTempReduceMesh2D] queIdx[%u] is bigger than axisTempInsQues size[%zu].", queIdx,
                axisTempInsQues.size()),
            InvalidParamsException, "queIdx is invalid");
        CHK_PRT_RET(
            Recv(recvInfo, axisTempInsQues[queIdx], 0, true, DmaMode::PUT),
            HCCL_ERROR("[InsTempReduceMesh2D] Recv data failed"), HcclResult::HCCL_E_INTERNAL);

        queIdx++;
    }

    if (axisTempInsQues.size() > 1) {
        CHK_RET(PostSyncInterQueues(axisTempInsQues));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh2D::GatherFromScratch(
    const u32 slice, const u32 axis, const ResLinks& tempLinks, std::vector<InsQuePtr>& axisTempInsQues)
{
    HCCL_DEBUG("[InsTempReduceMesh2D] Gather from scratch start");

    u64 sliceSize = sliceSize_[slice];
    u64 sliceScratchBaseOffset = sliceScratchBaseOffset_[slice];

    DataSlice srcDataSlice(BufferType::SCRATCH, sliceScratchBaseOffset + axisRoot_[axis] * sliceSize, sliceSize);

    if (axisTempInsQues.size() > 1) {
        CHK_RET(PreSyncInterQueues(axisTempInsQues));
    }

    // 主队列本地拷贝，从Scratch拷贝到Output
    DataSlice dstLocalSlice(BufferType::OUTPUT, sliceOutputBaseOffset_[slice], sliceSize);
    CHK_PRT_RET(
        LocalCopy(axisTempInsQues[0], srcDataSlice, dstLocalSlice),
        HCCL_ERROR("[InsTempReduceMesh2D] LocalCopy data failed"), HcclResult::HCCL_E_INTERNAL);

    // 从队列负责接收来自其它rank的数据
    u32 queIdx = 1;
    for (u32 axisRank = 0; axisRank < tempVTopo_.at(axis).size(); ++axisRank) {
        RankId rmtRank = tempVTopo_.at(axis).at(axisRank);
        if (rmtRank == myRank_) {
            continue;
        }

        const LinkData& recvLink = tempLinks.at(rmtRank).at(0);
        // 按照发送rank的序号来计算接收数据存放的偏移
        DataSlice dstDataSlice(BufferType::SCRATCH, sliceScratchBaseOffset + axisRank * sliceSize, sliceSize);
        SlicesList recvSlicesList({srcDataSlice}, {dstDataSlice});
        DataInfo recvInfo(recvLink, recvSlicesList);

        CHK_PRT_RET(
            Recv(recvInfo, axisTempInsQues[queIdx], 0, true, DmaMode::PUT),
            HCCL_ERROR("[InsTempReduceMesh2D] Recv data failed"), HcclResult::HCCL_E_INTERNAL);

        queIdx++;
    }

    if (axisTempInsQues.size() > 1) {
        CHK_RET(PostSyncInterQueues(axisTempInsQues));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh2D::SendFromInput(
    const u32 slice, const u32 axis, const ResLinks& tempLinks, std::vector<InsQuePtr>& axisTempInsQues)
{
    HCCL_DEBUG("[InsTempReduceMesh2D] Send from input start");

    u64 sliceSize = sliceSize_[slice];

    RankId rmtRank = tempVTopo_.at(axis).at(axisRoot_[axis]);
    const LinkData& sendLink = tempLinks.at(rmtRank).at(0);

    DataSlice srcDataSlice(BufferType::INPUT, sliceInputBaseOffset_[slice], sliceSize);
    DataSlice dstDataSlice(
        BufferType::SCRATCH, sliceScratchBaseOffset_[slice] + axisRank_[axis] * sliceSize, sliceSize);
    SlicesList sendSlicesList({srcDataSlice}, {dstDataSlice});
    DataInfo sendInfo(sendLink, sendSlicesList);

    CHK_PRT_RET(
        Send(sendInfo, axisTempInsQues[0], 0, true, DmaMode::PUT), HCCL_ERROR("[InsTempReduceMesh2D] Send data failed"),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh2D::SendFromScratch(
    const u32 slice, const u32 axis, const ResLinks& tempLinks, std::vector<InsQuePtr>& axisTempInsQues)
{
    HCCL_DEBUG("[InsTempReduceMesh2D] Send from scratch start");

    u64 sliceSize = sliceSize_[slice];
    u64 sliceScratchBaseOffset = sliceScratchBaseOffset_[slice];

    RankId rmtRank = tempVTopo_.at(axis).at(axisRoot_[axis]);
    const LinkData& sendLink = tempLinks.at(rmtRank).at(0);

    DataSlice srcDataSlice(BufferType::SCRATCH, sliceScratchBaseOffset + axisRoot_[axis] * sliceSize, sliceSize);
    DataSlice dstDataSlice(BufferType::SCRATCH, sliceScratchBaseOffset + axisRank_[axis] * sliceSize, sliceSize);
    SlicesList sendSlicesList({srcDataSlice}, {dstDataSlice});
    DataInfo sendInfo(sendLink, sendSlicesList);

    CHK_PRT_RET(
        Send(sendInfo, axisTempInsQues[0], 0, true, DmaMode::PUT), HCCL_ERROR("[InsTempReduceMesh2D] Send data failed"),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempReduceMesh2D::ReduceToScratch(const u32 slice, const u32 axis, std::vector<InsQuePtr>& axisTempInsQues)
{
    HCCL_DEBUG("[InsTempReduceMesh2D] Reduce to scratch start");

    u64 sliceSize = sliceSize_[slice];
    u64 sliceScratchBaseOffset = sliceScratchBaseOffset_[slice];

    // 数据规约到scratch时，下一步会交换处理另一片数据，因此数据规约至axisRoot_[1-axis]的偏移位置，便于后续数据搬运
    DataSlice dstDataSlice(BufferType::SCRATCH, sliceScratchBaseOffset + axisRoot_[1 - axis] * sliceSize, sliceSize);

    // 另一轴Root值大于等于当前轴的RankSize时，需要将数据规约至原本无数据的区域，需要先拷贝第一片数据
    bool needLocalCopy = axisRoot_[1 - axis] >= axisRankSize_[axis];
    if (needLocalCopy) {
        DataSlice srcLocalSlice(BufferType::SCRATCH, sliceScratchBaseOffset, sliceSize);
        CHK_PRT_RET(
            LocalCopy(axisTempInsQues[0], srcLocalSlice, dstDataSlice),
            HCCL_ERROR("[InsTempReduceMesh2D] LocalCopy data failed"), HcclResult::HCCL_E_INTERNAL);

        for (u32 sliceId = 1; sliceId < axisRankSize_[axis]; ++sliceId) {
            DataSlice srcDataSlice(BufferType::SCRATCH, sliceScratchBaseOffset + sliceId * sliceSize, sliceSize);
            CHK_PRT_RET(
                LocalReduce(axisTempInsQues[0], srcDataSlice, dstDataSlice, dataType_, redOp_),
                HCCL_ERROR("[InsTempReduceMesh2D] Local reduce data failed"), HcclResult::HCCL_E_INTERNAL);
        }

        return HcclResult::HCCL_SUCCESS;
    }

    // 另一轴Root值小于当前轴RankSize时，按照数据片顺序逐个Reduce
    for (u32 sliceId = 0; sliceId < axisRankSize_[axis]; ++sliceId) {
        if (sliceId == axisRoot_[1 - axis]) {
            continue;
        }
        DataSlice srcDataSlice(BufferType::SCRATCH, sliceScratchBaseOffset + sliceId * sliceSize, sliceSize);
        CHK_PRT_RET(
            LocalReduce(axisTempInsQues[0], srcDataSlice, dstDataSlice, dataType_, redOp_),
            HCCL_ERROR("[InsTempReduceMesh2D] Local reduce data failed"), HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceMesh2D::ReduceToOutput(const u32 slice, const u32 axis, std::vector<InsQuePtr>& axisTempInsQues)
{
    HCCL_DEBUG("[InsTempReduceMesh2D] Reduce to output start");

    u64 sliceSize = sliceSize_[slice];
    u64 sliceScratchBaseOffset = sliceScratchBaseOffset_[slice];

    DataSlice dstDataSlice(BufferType::OUTPUT, sliceOutputBaseOffset_[slice], sliceSize);

    for (u32 sliceId = 0; sliceId < axisRankSize_[axis]; ++sliceId) {
        if (sliceId == axisRoot_[axis]) { // 跳过轴向root的数据片，这一片已经提前拷贝至Output
            continue;
        }
        DataSlice srcDataSlice(BufferType::SCRATCH, sliceScratchBaseOffset + sliceId * sliceSize, sliceSize);
        CHK_PRT_RET(
            LocalReduce(axisTempInsQues[0], srcDataSlice, dstDataSlice, dataType_, redOp_),
            HCCL_ERROR("[InsTempReduceMesh2D] Local reduce data failed"), HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
