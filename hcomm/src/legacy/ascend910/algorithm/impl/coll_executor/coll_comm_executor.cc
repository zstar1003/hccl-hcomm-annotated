/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_executor.h"
#include "stream_active_manager.h"
#include "device_capacity.h"

namespace hccl {
CollCommExecutor::CollCommExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollNativeExecutorBase(dispatcher, topoMatcher)
{}

HcclResult CollCommExecutor::GetSubStreamInfoOnOneRing(
    const u32 ringIndex, std::vector<Stream>& subStreamsInOneRing,
    std::vector<std::shared_ptr<LocalNotify>>& mainSignalsInOneRing,
    std::vector<std::shared_ptr<LocalNotify>>& subSignalsInOneRing)
{
    u32 ringNum = GetLevel0RingNum();
    if (ringNum == LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING) {
        // double ring
        subStreamsInOneRing.push_back(algResResp_->slaveStreams[ringIndex + 1]);
        mainSignalsInOneRing.push_back(algResResp_->notifiesMain[ringIndex + 1]);
        subSignalsInOneRing.push_back(algResResp_->notifiesAux[ringIndex + 1]);
    } else if (ringNum == LEVEL0_PLANE_NUM_IN_NPRING_SINGLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING) {
        // single ring
        subStreamsInOneRing.push_back(algResResp_->slaveStreams[ringIndex]);
        mainSignalsInOneRing.push_back(algResResp_->notifiesMain[ringIndex]);
        subSignalsInOneRing.push_back(algResResp_->notifiesAux[ringIndex]);
    }
    return HCCL_SUCCESS;
}

u32 CollCommExecutor::GetLevel0RingNum() const { return algResResp_->slaveStreams.size() + 1; }

HcclResult CollCommExecutor::MultiRingAllReduce(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count, const HcclDataType dataType,
    const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>>& multRingsSliceZero, Stream stream,
    s32 profStage, const u64 baseOffset)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(
            singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        SubCommInfo level0RingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);

        u32 rankSize = level0RingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;
        std::unique_ptr<AlgTemplateBase> tempAlg;
        tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_RING in COMM_LEVEL0", __func__);
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(reduceAttr));

        if (ringIndex != (ringNum - 1)) {                                                    // 0~ringNum-2的环
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                CHK_RET(StreamActiveManager::GetInstance(topoAttr_.deviceLogicId)
                            .StreamActive(algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr()));
            }

            ret = LocalNotify::Wait(
                algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u] wait failed", ringIndex), ret);
            ret = tempAlg->Prepare(
                inputMem, outputMem, outputMem, count, dataType, algResResp_->slaveStreams[ringIndex], reductionOp,
                LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingAllReduce]stream[%u], AllReduce(ring) prepare failed,"
                    "return[%d]",
                    ringIndex, ret),
                ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                    + level0RingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingAllReduce]stream[%u], AllReduce(ring) register Profiler "
                    "failed,return[%d]",
                    ringIndex, ret),
                ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingAllReduce]stream[%u], AllReduce(ring) run failed,"
                    "return[%d]",
                    ringIndex, ret),
                ret);

            ret = LocalNotify::Post(
                algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesMain[ringIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u] record failed", ringIndex), ret);

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u] record failed", ringIndex), ret);
        } else { // 主环
            tempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_RING in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(tempAlg);
            CHK_RET(tempAlg->Prepare(reduceAttr));

            ret = tempAlg->Prepare(
                inputMem, outputMem, outputMem, count, dataType, stream, reductionOp, LEVEL0_BRIDGE_RANK_ID,
                singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingAllReduce]stream[%u], AllReduce(ring) prepare failed, "
                    "return[%d]",
                    ringIndex, ret),
                ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                    + level0RingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingAllReduce]stream[%u], AllReduce(ring) register Profiler "
                    "failed,return[%d]",
                    ringIndex, ret),
                ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingAllReduce]stream[%u], AllReduce(ring) run failed, "
                    "return[%d]",
                    ringIndex, ret),
                ret);

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                /* 等待executor执行完毕 */
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u] wait failed", ring), ret);
            }
        }
    }
    // 添加空task,保证执行时不乱序
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::UpdateOffsetBasedOnStrideCount(
    const OpParam& param, std::vector<std::vector<Slice>>& multRingsUserMemSlice) const
{
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    for (u32 ringIndex = 0; ringIndex < multRingsUserMemSlice.size(); ringIndex++) {
        for (u32 sliceIndex = 0; sliceIndex < multRingsUserMemSlice[ringIndex].size(); sliceIndex++) {
            u64 selfRank = multRingsUserMemSlice[ringIndex][sliceIndex].offset / (param.DataDes.count * perDataSize);
            HCCL_DEBUG(
                "rank[%u], ringIndex[%u], sliceIndex[%u], slice.offset=[%llu], slice.size=[%llu], selfRank[%llu].",
                topoAttr_.userRank, ringIndex, sliceIndex, multRingsUserMemSlice[ringIndex][sliceIndex].offset,
                multRingsUserMemSlice[ringIndex][sliceIndex].size, selfRank);
            multRingsUserMemSlice[ringIndex][sliceIndex].offset
                = multRingsUserMemSlice[ringIndex][sliceIndex].offset
                  + selfRank * ((param.DataDes.strideCount - param.DataDes.count) * perDataSize);
            HCCL_DEBUG(
                "rank[%u], ringIndex[%u], sliceIndex[%u], slice.offset=[%llu], slice.size=[%llu], selfRank[%llu] "
                "updated.",
                topoAttr_.userRank, ringIndex, sliceIndex, multRingsUserMemSlice[ringIndex][sliceIndex].offset,
                multRingsUserMemSlice[ringIndex][sliceIndex].size, selfRank);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingAllGather(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream, s32 profStage, const u64 baseOffset,
    const HcomCollOpInfo* opInfo, const std::vector<std::vector<Slice>> multRingsUserMemSlice,
    const CommPlane leveIndex)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(leveIndex, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));
    // 拿到ring环映射关系
    SubCommInfo level0ZeroCommInfo = GetSubCommInfo(leveIndex, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    TopoType topoType = topoType_;

    if (leveIndex == COMM_LEVEL0_LOGICAL) {
        std::vector<u32> mockNicList;
        mockNicList.reserve(level0ZeroCommInfo.localRankSize);
        for (u32 rankIndex = 0; rankIndex < level0ZeroCommInfo.localRankSize; rankIndex++) {
            mockNicList.push_back(rankIndex);
        }
        nicList = mockNicList;
        u32 ARSRankSize = topoMatcher_->GetCommPlaneRanks(COMM_LEVEL0_LOGICAL)[0].size();
        bool ARSDoubleRing
            = ((ARSRankSize > FACTOR_TWO) && (ARSRankSize % FACTOR_TWO == 0) && topoAttr_.isARSDoubleRing);

        if (ARSDoubleRing) {
            topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
        } else {
            topoType = TopoType::TOPO_TYPE_NP_SINGLE_RING;
        }
    }
    std::vector<std::vector<u32>> multiRingsOrder
        = GetRingsOrderByTopoType(level0ZeroCommInfo.localRankSize, topoType, nicList);

    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(
            singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]"
                       "singleRingSliceZero is empty"),
            HCCL_E_INTERNAL);

        // 910_93场景 生成userMemOut_上对应的slices
        std::vector<Slice> userMemOutputSlices;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(
                dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemOutputSlices));
        } else {
            userMemOutputSlices = multRingsUserMemSlice[ringIndex];
        }
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));

        SubCommInfo level0RingCommInfo = GetSubCommInfo(leveIndex, ringIndex);

        u32 rankSize = level0RingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        // 910_93场景 准备环中的从流
        std::vector<Stream> subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(
                GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing));
        }
        if (ringIndex != (ringNum - 1)) { // 最后一个环是主stream，所以这里减1，符合条件的走从stream
            if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())
                && workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                if (opInfo != nullptr) {
                    algResResp_->threadManage[ringIndex]->Prepare(
                        outputMem, outputMem, inputMem, count, dataType, algResResp_->slaveStreams[ringIndex],
                        HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset,
                        ringNics[ringIndex], tag, profStage, level0RingCommInfo, algResResp_->notifiesAux[ringIndex],
                        algResResp_->notifiesMain[ringIndex], ringIndex, ExecutorType::ALLGATHER_RING_DIRECT, 0, opInfo,
                        subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices);
                } else {
                    algResResp_->threadManage[ringIndex]->Prepare(
                        outputMem, outputMem, inputMem, count, dataType, algResResp_->slaveStreams[ringIndex],
                        HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset,
                        ringNics[ringIndex], tag, profStage, level0RingCommInfo, algResResp_->notifiesAux[ringIndex],
                        algResResp_->notifiesMain[ringIndex], ringIndex, ExecutorType::ALLGATHER_RING);
                }
                algResResp_->threadManage[ringIndex]->NotifyStart(); // 给线程发信号启动处理
            } else {
                ret = LocalNotify::Wait(
                    algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] wait failed", ringIndex), ret);
                // 如何判断是否环内是否有数据, 以ring的第一个rank的 size为判断依据
                std::unique_ptr<AlgTemplateBase> tempAlg;
                if (opInfo != nullptr) {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT, dispatcher_);
                    HCCL_CONFIG_INFO(
                        HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(
                        const_cast<HcomCollOpInfo*>(opInfo), topoAttr_.userRank, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices));
                } else {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                }

                ret = tempAlg->Prepare(
                    outputMem, outputMem, inputMem, count, dataType, algResResp_->slaveStreams[ringIndex],
                    HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset,
                    ringNics[ringIndex]);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingAllGather]stream[%u],AllGather(ring) prepare "
                        "failed,return[%d]",
                        ringIndex, ret),
                    ret);
                ret = tempAlg->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID)
                        + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingAllGather]stream[%u],AllGather(ring) register "
                        "Profiler failed,return[%d]",
                        ringIndex, ret),
                    ret);

                ret = RunTemplate(tempAlg, level0RingCommInfo);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingAllGather]stream[%u],AllGather(ring) run failed, "
                        "return[%d]",
                        ringIndex, ret),
                    ret);

                ret = LocalNotify::Post(
                    algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesMain[ringIndex], profStage);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] record failed", ringIndex), ret);
            }

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] record failed", ringIndex), ret);
        } else { // 主环
            std::unique_ptr<AlgTemplateBase> tempAlg;
            if (opInfo != nullptr) {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT, dispatcher_);
                HCCL_CONFIG_INFO(
                    HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
                CHK_RET(tempAlg->Prepare(
                    const_cast<HcomCollOpInfo*>(opInfo), topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemOutputSlices));
            } else {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
            }

            ret = tempAlg->Prepare(
                outputMem, outputMem, inputMem, count, dataType, stream, HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID,
                singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingAllGather]stream[%u],AllGather(ring) prepare failed,"
                    "return[%d]",
                    ringIndex, ret),
                ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                    + level0RingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingAllGather]stream[%u], AllGather(ring) register Profiler "
                    "failed,return[%d]",
                    ringIndex, ret),
                ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingAllGather]stream[%u], AllGather(ring) run failed,"
                    "return[%d]",
                    ringIndex, ret),
                ret);

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())
                    && workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    algResResp_->threadManage[ring]->WaitDone(); // 单算子模式，等待线程处理完成信号
                }
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] wait failed", ring), ret);
            }
        }
    }
    // 添加空task,保证执行时不乱序
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingAllGatherConcurrent(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsSliceZero, Stream stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo* opInfo,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsUserMemSlice)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size(); // 环数, 当前为4环

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));
    auto halfRingSize = ringNum;
    if (ringNum > RDMA_PLANE_NUM_IN_NPRING_DOUBLE) {
        halfRingSize = ringNum / 2; // 2环
    }
    // 拿到ring环映射关系
    CHK_RET(CheckCommSize(COMM_LEVEL0_ANYPATH_SDMA, COMM_INDEX_1));
    SubCommInfo level0ZeroCommInfo = GetSubCommInfo(COMM_LEVEL0_ANYPATH_SDMA, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder
        = GetRingsOrderForAnyPath(level0ZeroCommInfo.localRankSize, topoType_, nicList);

    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex].second; // 取出sdma/rdma的数据块
        CHK_PRT_RET(
            singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]"
                       "singleRingSliceZero is empty"),
            HCCL_E_INTERNAL);

        // 910_93场景 生成userMemOut_上对应的slices
        std::vector<Slice> userMemOutputSlices;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(
                dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemOutputSlices));
        } else {
            userMemOutputSlices = multRingsUserMemSlice[ringIndex].second;
        }
        std::vector<u32> rankOrder;
        u32 commIndex = ringIndex % halfRingSize;
        CHK_RET(GetRankOrder(multiRingsOrder, commIndex, rankOrder));

        SubCommInfo level0RingCommInfo = multRingsSliceZero[ringIndex].first ?
                                             GetSubCommInfo(COMM_LEVEL0_ANYPATH_SDMA, commIndex) :
                                             GetSubCommInfo(COMM_LEVEL0_ANYPATH_RDMA, commIndex);

        u32 rankSize = level0RingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        // 910_93场景 准备环中的从流
        std::vector<Stream> subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(
                GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing));
        }
        bool isSdma = multRingsSliceZero[ringIndex].first;
        if (ringIndex != (ringNum - 1)) { // 最后一个环是主stream，所以这里减1，符合条件的走从stream
            if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())
                && workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                if (opInfo != nullptr) {
                    ExecutorType type
                        = isSdma ? ExecutorType::ALLGATHER_RING_DIRECT : ExecutorType::ALLGATHER_RING_DIRECT_RDMA;
                    algResResp_->threadManage[ringIndex]->Prepare(
                        outputMem, outputMem, inputMem, count, dataType, algResResp_->slaveStreams[ringIndex],
                        HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset,
                        ringNics[ringIndex % halfRingSize], tag, profStage, level0RingCommInfo,
                        algResResp_->notifiesAux[ringIndex], algResResp_->notifiesMain[ringIndex], ringIndex, type, 0,
                        opInfo, subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder,
                        userMemOutputSlices);
                } else {
                    algResResp_->threadManage[ringIndex]->Prepare(
                        outputMem, outputMem, inputMem, count, dataType, algResResp_->slaveStreams[ringIndex],
                        HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset,
                        ringNics[ringIndex % halfRingSize], tag, profStage, level0RingCommInfo,
                        algResResp_->notifiesAux[ringIndex], algResResp_->notifiesMain[ringIndex], ringIndex,
                        ExecutorType::ALLGATHER_RING);
                }
                algResResp_->threadManage[ringIndex]->NotifyStart(); // 给线程发信号启动处理
            } else {
                ret = LocalNotify::Wait(
                    algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] wait failed", ringIndex),
                    ret);
                // 如何判断是否环内是否有数据, 以ring的第一个rank的 size为判断依据
                std::unique_ptr<AlgTemplateBase> tempAlg;
                if (opInfo != nullptr) {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT, dispatcher_);
                    HCCL_CONFIG_INFO(
                        HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(
                        const_cast<HcomCollOpInfo*>(opInfo), topoAttr_.userRank, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices, isSdma));
                } else {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                }
                ret = tempAlg->Prepare(
                    outputMem, outputMem, inputMem, count, dataType, algResResp_->slaveStreams[ringIndex],
                    HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset,
                    ringNics[ringIndex % halfRingSize]);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],AllGather(ring) prepare "
                        "failed,return[%d]",
                        ringIndex, ret),
                    ret);
                ret = tempAlg->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID)
                        + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],AllGather(ring) register "
                        "Profiler failed,return[%d]",
                        ringIndex, ret),
                    ret);

                ret = RunTemplate(tempAlg, level0RingCommInfo);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],AllGather(ring)"
                        " run failed,return[%d]",
                        ringIndex, ret),
                    ret);

                ret = LocalNotify::Post(
                    algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesMain[ringIndex], profStage);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] record failed", ringIndex),
                    ret);
            }

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] record failed", ringIndex), ret);
        } else { // 主环
            std::unique_ptr<AlgTemplateBase> tempAlg;
            if (opInfo != nullptr) {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT, dispatcher_);
                HCCL_CONFIG_INFO(
                    HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
                CHK_RET(tempAlg->Prepare(
                    const_cast<HcomCollOpInfo*>(opInfo), topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemOutputSlices, isSdma));
            } else {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
            }
            ret = tempAlg->Prepare(
                outputMem, outputMem, inputMem, count, dataType, stream, HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID,
                singleRingSliceZero, baseOffset, ringNics[ringIndex % halfRingSize]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],AllGather(ring) prepare"
                    " failed,return[%d]",
                    ringIndex, ret),
                ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                    + level0RingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],AllGather(ring) register "
                    "Profiler failed, return[%d]",
                    ringIndex, ret),
                ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],AllGather(ring) run failed,"
                    "return[%d]",
                    ringIndex, ret),
                ret);

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())
                    && workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    algResResp_->threadManage[ring]->WaitDone(); // 单算子模式，等待线程处理完成信号
                }
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] wait failed", ring), ret);
            }
        }
    }
    // 添加空task,保证执行时不乱序
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::Level1AllGatherConcurrent(
    DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType, Stream stream, s32 profStage,
    std::vector<Slice>& level1DataSegsSlice, u32 syncTrans)
{
    std::vector<std::pair<bool, std::vector<Slice>>> level1MultSlice;
    std::vector<Slice> level1DataSegsSliceSdma;
    std::vector<Slice> level1DataSegsSliceRdma;
    bool isAnyPathCommLevel0 = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING
                                && workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) ?
                                   true :
                                   false;
    CommPlane commplane = (isAnyPathCommLevel0) ? COMM_LEVEL0_ANYPATH_SDMA : COMM_LEVEL0;
    CHK_RET(CheckCommSize(commplane, COMM_INDEX_1));
    SubCommInfo level0CommInfo = GetSubCommInfo(commplane, COMM_INDEX_0);
    u32 level0ServerIndex = level0CommInfo.localRank;
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1_ANYPATH_SDMA, level0ServerIndex);
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    HcclResult ret = HCCL_SUCCESS;
    level1MultSlice.resize(RDMA_PLANE_NUM_IN_NPRING_DOUBLE);

    for (u32 i = 0; i < level1CommInfo.localRankSize; i++) {
        Slice sdmaSlice;
        Slice rdmaSlice;
        u64 sdmaSliceSize
            = ((level1DataSegsSlice[i].size <= HCCL_MIN_SLICE_ALIGN_910_93) || (syncTrans == MAX_SPLIT_VALUE)) ?
                  level1DataSegsSlice[i].size :
                  ((syncTrans * level1DataSegsSlice[i].size / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_93)
                      * HCCL_MIN_SLICE_ALIGN_910_93;
        sdmaSlice.size = sdmaSliceSize;
        sdmaSlice.offset = level1DataSegsSlice[i].offset;
        rdmaSlice.size = level1DataSegsSlice[i].size - sdmaSliceSize;
        rdmaSlice.offset = level1DataSegsSlice[i].offset + sdmaSliceSize;
        level1DataSegsSliceSdma.push_back(sdmaSlice);
        level1DataSegsSliceRdma.push_back(rdmaSlice);
        HCCL_DEBUG(
            "Level1 index:[%u], Original [offset %llu, size %llu], sdma [offset %llu, size %llu], "
            "rdma [offset %llu, size %llu]",
            i, level1DataSegsSlice[i].offset, level1DataSegsSlice[i].size, sdmaSlice.offset, sdmaSlice.size,
            rdmaSlice.offset, rdmaSlice.size);
    }
    level1MultSlice[0] = std::make_pair(true, level1DataSegsSliceSdma);
    level1MultSlice[1] = std::make_pair(false, level1DataSegsSliceRdma);

    u32 commPlaneNum = level1MultSlice.size();
    for (u32 planeIndex = 0; planeIndex < commPlaneNum; planeIndex++) {
        std::vector<Slice>& singleSlice = level1MultSlice[planeIndex].second;
        SubCommInfo level1RdmaCommInfo = GetSubCommInfo(COMM_LEVEL1_ANYPATH_RDMA, level0ServerIndex);
        SubCommInfo level1TempCommInfo = level1MultSlice[planeIndex].first ? level1CommInfo : level1RdmaCommInfo;
        std::unique_ptr<AlgTemplateBase> level1TempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL1", __func__);
        } else {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL1", __func__);
        }
        CHK_SMART_PTR_NULL(level1TempAlg);

        if (planeIndex != (commPlaneNum - 1)) {
            ret = LocalNotify::Wait(
                algResResp_->slaveStreams[planeIndex], dispatcher_, algResResp_->notifiesAux[planeIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("stream[%u] wait failed", planeIndex), ret);

            CHK_RET(level1TempAlg->Prepare(
                outputMem, outputMem, inputMem, count, dataType, algResResp_->slaveStreams[planeIndex],
                HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, singleSlice, 0));

            CHK_RET(level1TempAlg->RegisterProfiler(
                (level1TempCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[planeIndex]));

            CHK_RET(RunTemplate(level1TempAlg, level1TempCommInfo));
            ret = LocalNotify::Post(
                algResResp_->slaveStreams[planeIndex], dispatcher_, algResResp_->notifiesMain[planeIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
            // 主环record启动从环
            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[planeIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
        } else {
            CHK_RET(level1TempAlg->Prepare(
                outputMem, outputMem, inputMem, count, dataType, stream, HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID,
                singleSlice, 0));
            CHK_RET(level1TempAlg->RegisterProfiler(
                (level1TempCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream));

            CHK_RET(RunTemplate(level1TempAlg, level1TempCommInfo));
            for (u32 ring = 0; ring < (commPlaneNum - 1); ring++) {
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("stream[%u] wait failed", ring), ret);
            }
        }
    }
    HCCL_INFO("Level1AllGatherConcurrent run success");
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

u32 CollCommExecutor::CalcOptimalIntraRingsize(u64 count, HcclDataType dataType, HcclCMDType opType)
{
    if (!topoMatcher_->GetARSFlag())
        return 0;

    u32 level0RankSize = topoMatcher_->GetCommPlaneRanks(COMM_LEVEL0)[0].size();
    u32 rankSizeInSuperPod = topoMatcher_->GetCommPlaneRanks(COMM_ARS)[0].size();
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
    // 不支持 ARS 或环内卡数不是 2 的倍数
    u32 level0RingSize = 1;
    if (!topoAttr_.isARSDoubleRing || (level0RankSize % FACTOR_TWO != 0)) {
        HCCL_INFO(
            "not Support ARS doubleRing, level0RingSize:[%u], level0RankSize[%u].", level0RingSize, level0RankSize);
        return level0RingSize;
    }
    // --- 1. 带宽 & 基本参数 ---
    float bwHCCS, bwHBM, bwSIO;
    constexpr u32 level0 = 0;
    constexpr u32 level2 = 2;
    constexpr u32 level3 = 3;
    CHK_RET(GetBandWidthPerNPU(level0, topoAttr_.userRankSize, topoAttr_.deviceNumPerAggregation, bwHCCS));
    CHK_RET(GetBandWidthPerNPU(level2, topoAttr_.userRankSize, topoAttr_.deviceNumPerAggregation, bwHBM));
    CHK_RET(GetBandWidthPerNPU(level3, topoAttr_.userRankSize, topoAttr_.deviceNumPerAggregation, bwSIO));
    float latency = BASE_COMM_LATENCY / MULTIPLIER_MS2US; // ms
    // --- 2. 数据总量 (GB) ---
    float baseSizeGB = static_cast<double>(count) * perDataSize / (1024 * 1024 * 1024);
    float totalSize = baseSizeGB;
    HCCL_INFO(
        "CalcOptimalIntraRingsize: count[%u], totalSize:[%lf]GB, perDataSize[%u].", count, totalSize, perDataSize);
    if (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        totalSize *= rankSizeInSuperPod;
    }
    // --- 3. 枚举可能的环大小 ---
    std::vector<u32> factors;
    for (u32 i = 1; i <= rankSizeInSuperPod / i; ++i) {
        if (rankSizeInSuperPod % i == 0) {
            factors.push_back(i);
            if (i != rankSizeInSuperPod / i) {
                factors.push_back(rankSizeInSuperPod / i);
            }
        }
    }
    std::sort(factors.begin(), factors.end());
    // --- 4. 计算最优带宽 ---
    double maxBwARS = 0.0;
    for (u32 N1 : factors) {
        u32 N2 = rankSizeInSuperPod / N1;
        // 静态时延 (ms)
        double interStep = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) ? (N2 - 1) : log2(N2);
        double latencyStep = (interStep + (N1 - 1)) * latency;
        // 传输时延 (ms)
        double latencyIntra;
        if ((N1 % FACTOR_TWO == 0) && (N1 > FACTOR_TWO)) {
            latencyIntra = (N1 - 1) * totalSize * MULTIPLIER_S2MS / N1 / bwHCCS / FACTOR_TWO;
        } else if (N1 == FACTOR_TWO) {
            latencyIntra = totalSize * MULTIPLIER_S2MS / FACTOR_TWO / bwSIO;
        } else {
            latencyIntra = (N1 - 1) * totalSize * MULTIPLIER_S2MS / N1 / bwHCCS;
        }
        double latencyInter = (N2 - 1) * totalSize * MULTIPLIER_S2MS / N1 / N2 / bwHCCS;
        // HBM 拷贝时延 (ms)
        double latencyCopy = totalSize * MULTIPLIER_S2MS / bwHBM;
        u8 mul = (opType == HcclCMDType::HCCL_CMD_ALLREDUCE) ? FACTOR_TWO : 1;
        double timeCost = mul * (latencyStep + latencyIntra + latencyInter) + latencyCopy;
        double bwARS = totalSize / timeCost; // GB/ms
        if (bwARS > maxBwARS) {
            maxBwARS = bwARS;
            level0RingSize = N1;
        }
    }
    HCCL_INFO("level0RingSize:[%u], totalSize:[%lf]GB, level0RankSize[%u].", level0RingSize, totalSize, level0RankSize);
    return level0RingSize;
}

HcclResult CollCommExecutor::CollectMultiRingsUserMemSlices(
    u32 ringNum, const HcclDataType dataType, const HcomCollOpInfo* opInfo,
    const std::vector<std::vector<Slice>>& multRingsSliceZero, const std::vector<std::vector<u32>>& multiRingsOrder,
    const std::vector<std::vector<Slice>>& multRingsUserMemSlice,
    std::vector<std::vector<Slice>>& userMemSlicesOfMultiRings)
{
    CHK_PTR_NULL(opInfo);
    CHK_PRT_RET(
        0 < opInfo->strideCount && opInfo->strideCount < opInfo->count,
        HCCL_ERROR(
            "[CollCommExecutor][CollectMultiRingsUserMemSlices]strideCount[%llu] is smaller than opCount[%llu]",
            opInfo->strideCount, opInfo->count),
        HCCL_E_PARA);
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(
            singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);
        std::vector<Slice> userMemSlices;
        HCCL_DEBUG(
            "[CollCommExecutor][CollectMultiRingsUserMemSlices]multRingsUserMemSlice.size()[%zu], strideCount[%llu], "
            "opCount[%llu]",
            multRingsUserMemSlice.size(), opInfo->strideCount, opInfo->count);
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemSlices));
        } else {
            userMemSlices = multRingsUserMemSlice[ringIndex];
        }
        userMemSlicesOfMultiRings.push_back(userMemSlices);
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::CollectMultiRingsRankOrder(
    u32 ringNum, const std::vector<std::vector<u32>>& multiRingsOrder, std::vector<std::vector<u32>>& rankOrders)
{
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));
        rankOrders.push_back(rankOrder);
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingReduceScatter(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream,
    s32 profStage, const u64 baseOffset, const HcomCollOpInfo* opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice, const CommPlane levelIndex)
{
    HCCL_INFO("[MultiRingReduceScatter] MultiRingReduceScatter starts");
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(levelIndex, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));
    // 拿到ring环映射关系
    SubCommInfo level0ZeroCommInfo = GetSubCommInfo(levelIndex, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;

    TopoType topoType = topoType_;

    if (levelIndex == COMM_LEVEL0_LOGICAL) {
        std::vector<u32> mockNicList;
        mockNicList.reserve(level0ZeroCommInfo.localRankSize);
        for (u32 rankIndex = 0; rankIndex < level0ZeroCommInfo.localRankSize; rankIndex++) {
            mockNicList.push_back(rankIndex);
        }
        nicList = mockNicList;
        u32 ARSRankSize = topoMatcher_->GetCommPlaneRanks(COMM_LEVEL0_LOGICAL)[0].size();
        bool ARSDoubleRing
            = ((ARSRankSize > FACTOR_TWO) && (ARSRankSize % FACTOR_TWO == 0) && topoAttr_.isARSDoubleRing);
        if (ARSDoubleRing) {
            topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
        } else {
            topoType = TopoType::TOPO_TYPE_NP_SINGLE_RING;
        }
    }
    std::vector<std::vector<u32>> multiRingsOrder
        = GetRingsOrderByTopoType(level0ZeroCommInfo.localRankSize, topoType, nicList);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(
            singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(
                dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemInputSlices));
        } else {
            userMemInputSlices = multRingsUserMemSlice[ringIndex];
        }

        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));

        SubCommInfo level0RingCommInfo = GetSubCommInfo(levelIndex, ringIndex);
        u32 rankSize = level0RingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        std::vector<Stream> subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(
                GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing));
        }
        if (ringIndex != (ringNum - 1)) {                                                    // 0~ringNum-2的环
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                ret = StreamActiveManager::GetInstance(topoAttr_.deviceLogicId)
                          .StreamActive(algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr());
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]active stream[%u], failed", ringIndex), ret);
            }
            if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())
                && workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                /* 更新线程参数 */
                if (opInfo != nullptr) {
                    algResResp_->threadManage[ringIndex]->Prepare(
                        inputMem, inputMem, outputMem, count, dataType, algResResp_->slaveStreams[ringIndex],
                        reductionOp, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex], tag,
                        profStage, level0RingCommInfo, algResResp_->notifiesAux[ringIndex],
                        algResResp_->notifiesMain[ringIndex], ringIndex, ExecutorType::REDUCE_SCATTER_RING_DIRECT,
                        reduceAttr, opInfo, subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder,
                        userMemInputSlices);
                } else {
                    algResResp_->threadManage[ringIndex]->Prepare(
                        inputMem, inputMem, outputMem, count, dataType, algResResp_->slaveStreams[ringIndex],
                        reductionOp, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex], tag,
                        profStage, level0RingCommInfo, algResResp_->notifiesAux[ringIndex],
                        algResResp_->notifiesMain[ringIndex], ringIndex, ExecutorType::REDUCE_SCATTER_RING, reduceAttr);
                }

                algResResp_->threadManage[ringIndex]->NotifyStart(); // 给线程发通知启动线程执行
            } else {
                std::unique_ptr<AlgTemplateBase> tempAlg;
                if (opInfo != nullptr) {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_REDUCESCATTER_RING_DIRECT, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING_DIRECT in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(
                        reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                        subSignalsInOneRing, rankOrder, userMemInputSlices));
                } else {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(reduceAttr));
                }
                CHK_SMART_PTR_NULL(tempAlg);

                ret = LocalNotify::Wait(
                    algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] wait failed", ringIndex), ret);
                ret = tempAlg->Prepare(
                    inputMem, inputMem, outputMem, count, dataType, algResResp_->slaveStreams[ringIndex], reductionOp,
                    LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingReduceScatter]stream[%u],ReduceScatter(ring) "
                        "prepare failed,return[%d]",
                        ringIndex, ret),
                    ret);
                ret = tempAlg->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID)
                        + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingReduceScatter]stream[%u],ReduceScatter(ring) "
                        "register Profiler failed,return[%d]",
                        ringIndex, ret),
                    ret);

                ret = RunTemplate(tempAlg, level0RingCommInfo);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingReduceScatter]stream[%u],ReduceScatter(ring) run "
                        "failed,return[%d]",
                        ringIndex, ret),
                    ret);

                ret = LocalNotify::Post(
                    algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesMain[ringIndex], profStage);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] record failed", ringIndex), ret);
            }
            /* 主环record启动从环 */
            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] record failed", ringIndex), ret);
        } else { // 主环 最后一个环
            std::unique_ptr<AlgTemplateBase> tempAlg;
            if (opInfo != nullptr) {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_RING_DIRECT, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING_DIRECT in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
                CHK_RET(tempAlg->Prepare(
                    reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemInputSlices));
            } else {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
                CHK_RET(tempAlg->Prepare(reduceAttr));
            }
            CHK_SMART_PTR_NULL(tempAlg);
            ret = tempAlg->Prepare(
                inputMem, inputMem, outputMem, count, dataType, stream, reductionOp, LEVEL0_BRIDGE_RANK_ID,
                singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingReduceScatter]stream[%u],ReduceScatter(ring) prepare "
                    "failed,return[%d]",
                    ringIndex, ret),
                ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                    + level0RingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingReduceScatter]stream[%u],ReduceScatter(ring) register "
                    "Profiler failed,return[%d]",
                    ringIndex, ret),
                ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingReduceScatter]stream[%u],ReduceScatter(ring) run "
                    "failed,return[%d]",
                    ringIndex, ret),
                ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())
                    && workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    algResResp_->threadManage[ring]->WaitDone();
                }
                /* 等待executor执行完毕 */
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);

                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] wait failed", ring), ret);
            }
        }
    }

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingGather(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::vector<Slice>> multRingsSliceZero, HcclReduceOp op, u32 root, Stream stream, s32 profStage)
{
    u32 ringNum = multRingsSliceZero.size();
    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    HcclResult ret;

    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(
            singleRingSliceZero.empty(), HCCL_ERROR("[CommonOperator][MultiRingGather]singleRingSliceZero is empty"),
            HCCL_E_INTERNAL);

        SubCommInfo level0RingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);
        u32 rankSize = level0RingCommInfo.localRankSize;
        u32 rootRank = 0;
        ret = GetRankByUserRank(COMM_LEVEL0, ringIndex, root, rootRank);
        CHK_PRT_RET(
            ret == HCCL_E_PARA,
            HCCL_ERROR("[CommonOperator][MultiRingGather]invalid root rank[%u] to get user rank", root), ret);

        std::unique_ptr<AlgTemplateBase> tempAlg = nullptr;
        EXCEPTION_CATCH(
            (tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_GATHER_RING, dispatcher_)),
            return HCCL_E_PTR);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_GATHER_RING in COMM_LEVEL0", __func__);
        CHK_SMART_PTR_NULL(tempAlg);

        if (ringIndex != (ringNum - 1)) {                                                    // 0~ringNum-2的环
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                CHK_RET(StreamActiveManager::GetInstance(topoAttr_.deviceLogicId)
                            .StreamActive(algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr()));
            }
            ret = LocalNotify::Wait(
                algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]in stream[%u] wait failed", ringIndex), ret);
            if (singleRingSliceZero[0].size != 0) {
                ret = tempAlg->Prepare(
                    inputMem, outputMem, outputMem, count, dataType, algResResp_->slaveStreams[ringIndex], op, rootRank,
                    singleRingSliceZero, 0, ringNics[ringIndex]);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CommonOperator][MultiRingGather]stream[%u],gather(ring) prepare failed, "
                        "return[%d]",
                        ringIndex, ret),
                    ret);

                ret = tempAlg->RegisterProfiler(
                    level0RingCommInfo.localRank, profStage, HCCL_EXEC_STEP_NOT_SET,
                    algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CommonOperator][MultiRingGather]stream[%u], gather(ring) register profiler "
                        "failed,return[%d]",
                        ringIndex, ret),
                    ret);

                ret = RunTemplate(tempAlg, level0RingCommInfo);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CommonOperator][MultiRingGather]stream[%u],gather(ring) run failed,return[%d]", ringIndex,
                        ret),
                    ret);
            }
            ret = LocalNotify::Post(
                algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesMain[ringIndex], profStage);

            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u] record failed", ringIndex),
                ret);

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u] record failed", ringIndex),
                ret);
        } else { // 主环
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_GATHER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_GATHER_RING in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(tempAlg);

            ret = tempAlg->Prepare(
                inputMem, outputMem, outputMem, count, dataType, stream, op, rootRank, singleRingSliceZero, 0,
                ringNics[ringIndex]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CommonOperator][MultiRingGather]stream[%u],gather(ring) prepare failed, "
                    "return[%d]",
                    ringIndex, ret),
                ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                    + level0RingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CommonOperator][MultiRingGather]stream[%u], gather(ring) register "
                    "profiler failed,return[%d]",
                    ringIndex, ret),
                ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CommonOperator][MultiRingGather]stream[%u],gather(ring) run failed, "
                    "return[%d]",
                    ringIndex, ret),
                ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                /* 等待executor执行完毕 , 当前环没有分配数据，跳过此环处理，继续下一个环 */
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS, HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u] wait failed", ring),
                    ret);
            }
        }
    }

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingReduceScatterConcurrent(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const HcclReduceOp reductionOp, const std::vector<std::pair<bool, std::vector<Slice>>> multRingsSliceZero,
    Stream stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo* opInfo,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsUserMemSlice)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));
    u32 halfRingSize = ringNum;
    u32 DoubleRing = 2;
    if (ringNum > RDMA_PLANE_NUM_IN_NPRING_DOUBLE) {
        halfRingSize = ringNum / DoubleRing;
    }

    // 拿到ring环映射关系
    SubCommInfo level0ZeroCommInfo = GetSubCommInfo(COMM_LEVEL0_ANYPATH_SDMA, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder
        = GetRingsOrderForAnyPath(level0ZeroCommInfo.localRankSize, topoType_, nicList);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex].second;
        CHK_PRT_RET(
            singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]singleRingSliceZero is empty"),
            HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        u32 commIndex = ringIndex % halfRingSize;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(
                dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemInputSlices));
        } else {
            userMemInputSlices = multRingsUserMemSlice[ringIndex].second;
        }
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, commIndex, rankOrder));

        SubCommInfo level0RingCommInfo = multRingsSliceZero[ringIndex].first ?
                                             GetSubCommInfo(COMM_LEVEL0_ANYPATH_SDMA, commIndex) :
                                             GetSubCommInfo(COMM_LEVEL0_ANYPATH_RDMA, commIndex);
        u32 rankSize = level0RingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        std::vector<Stream> subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(
                GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing));
        }
        bool isSdma = multRingsSliceZero[ringIndex].first;
        if (ringIndex != (ringNum - 1)) {                                                    // 0~ringNum-2的环
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                ret = StreamActiveManager::GetInstance(topoAttr_.deviceLogicId)
                          .StreamActive(algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr());
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingReduceScatterConcurrent]active stream[%u], failed", ringIndex),
                    ret);
            }

            if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())
                && workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                /* 更新线程参数 */
                if (opInfo != nullptr) {
                    ExecutorType type = isSdma ? ExecutorType::REDUCE_SCATTER_RING_DIRECT :
                                                 ExecutorType::REDUCE_SCATTER_RING_DIRECT_RDMA;
                    algResResp_->threadManage[ringIndex]->Prepare(
                        inputMem, inputMem, outputMem, count, dataType, algResResp_->slaveStreams[ringIndex],
                        reductionOp, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset,
                        ringNics[ringIndex % halfRingSize], tag, profStage, level0RingCommInfo,
                        algResResp_->notifiesAux[ringIndex], algResResp_->notifiesMain[ringIndex], ringIndex, type,
                        reduceAttr, opInfo, subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder,
                        userMemInputSlices);
                } else {
                    algResResp_->threadManage[ringIndex]->Prepare(
                        inputMem, inputMem, outputMem, count, dataType, algResResp_->slaveStreams[ringIndex],
                        reductionOp, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset,
                        ringNics[ringIndex % halfRingSize], tag, profStage, level0RingCommInfo,
                        algResResp_->notifiesAux[ringIndex], algResResp_->notifiesMain[ringIndex], ringIndex,
                        ExecutorType::REDUCE_SCATTER_RING, reduceAttr);
                }

                algResResp_->threadManage[ringIndex]->NotifyStart(); // 给线程发通知启动线程执行
            } else {
                std::unique_ptr<AlgTemplateBase> tempAlg;
                if (opInfo != nullptr) {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_REDUCESCATTER_RING_DIRECT, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING_DIRECT in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(
                        reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                        subSignalsInOneRing, rankOrder, userMemInputSlices, isSdma));
                    HCCL_DEBUG("[MultiRingReduceScatterConcurrent]run in COMM_LEVEL0 ends");
                } else {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(reduceAttr));
                }
                HCCL_DEBUG("[MultiRingReduceScatterConcurrent]run in COMM_LEVEL0 ends");
                CHK_SMART_PTR_NULL(tempAlg);

                ret = LocalNotify::Wait(
                    algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] wait failed", ringIndex),
                    ret);
                ret = tempAlg->Prepare(
                    inputMem, inputMem, outputMem, count, dataType, algResResp_->slaveStreams[ringIndex], reductionOp,
                    LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex % halfRingSize]);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],ReduceScatter(ring) "
                        "prepare failed,return[%d]",
                        ringIndex, ret),
                    ret);
                ret = tempAlg->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID)
                        + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],ReduceScatter(ring) "
                        "register Profiler failed,return[%d]",
                        ringIndex, ret),
                    ret);

                ret = RunTemplate(tempAlg, level0RingCommInfo);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],ReduceScatter(ring)"
                        " run failed,return[%d]",
                        ringIndex, ret),
                    ret);

                ret = LocalNotify::Post(
                    algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesMain[ringIndex], profStage);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] record failed", ringIndex),
                    ret);
            }
            /* 主环record启动从环 */
            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] record failed", ringIndex),
                ret);
        } else { // 主环 最后一个环
            std::unique_ptr<AlgTemplateBase> tempAlg;
            if (opInfo != nullptr) {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_RING_DIRECT, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING_DIRECT in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
                CHK_RET(tempAlg->Prepare(
                    reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemInputSlices, isSdma));
            } else {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
                CHK_RET(tempAlg->Prepare(reduceAttr));
            }
            CHK_SMART_PTR_NULL(tempAlg);
            ret = tempAlg->Prepare(
                inputMem, inputMem, outputMem, count, dataType, stream, reductionOp, LEVEL0_BRIDGE_RANK_ID,
                singleRingSliceZero, baseOffset, ringNics[ringIndex % halfRingSize]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],ReduceScatter(ring) "
                    "prepare failed,return[%d]",
                    ringIndex, ret),
                ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                    + level0RingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],ReduceScatter(ring) "
                    "register Profiler failed,return[%d]",
                    ringIndex, ret),
                ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],ReduceScatter(ring) run "
                    "failed,return[%d]",
                    ringIndex, ret),
                ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())
                    && workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    algResResp_->threadManage[ring]->WaitDone();
                }
                /* 等待executor执行完毕 */
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] wait failed", ring),
                    ret);
            }
        }
    }

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::Level1ReduceScatterConcurrent(
    DeviceMem inputMem, DeviceMem scratchMem, const u64 count, const HcclDataType dataType,
    const HcclReduceOp reductionOp, Stream stream, s32 profStage, std::vector<Slice>& level1DataSegsSlice,
    u32 syncTrans, u64 reduceAttr)
{
    (void)profStage;
    std::vector<std::pair<bool, std::vector<Slice>>> level1MultSlice;
    level1MultSlice.resize(RDMA_PLANE_NUM_IN_NPRING_DOUBLE);
    std::vector<Slice> sdmaSlice;
    std::vector<Slice> rdmaSlice;
    for (u32 segsIndex = 0; segsIndex < level1DataSegsSlice.size(); segsIndex++) {
        u64 totalSize = level1DataSegsSlice[segsIndex].size;
        u64 sdmaSliceOffset = level1DataSegsSlice[segsIndex].offset;
        u64 sdmaSliceSize = ((totalSize <= HCCL_MIN_SLICE_ALIGN_910_93) || (syncTrans == MAX_SPLIT_VALUE)) ?
                                totalSize :
                                ((syncTrans * totalSize / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_93)
                                    * HCCL_MIN_SLICE_ALIGN_910_93;
        Slice sdmaSliceTmp;
        sdmaSliceTmp.offset = sdmaSliceOffset;
        sdmaSliceTmp.size = sdmaSliceSize;
        Slice rdmaSliceTmp;
        rdmaSliceTmp.offset = sdmaSliceOffset + sdmaSliceSize;
        rdmaSliceTmp.size = totalSize - sdmaSliceSize;
        sdmaSlice.push_back(sdmaSliceTmp);
        rdmaSlice.push_back(rdmaSliceTmp);
        HCCL_DEBUG(
            "Level1 data segId:%u, Original [offset %llu, size %llu], sdma [offset %llu, size %llu], "
            "rdma [offset %llu, size %llu]",
            segsIndex, sdmaSliceOffset, totalSize, sdmaSliceTmp.offset, sdmaSliceTmp.size, rdmaSliceTmp.offset,
            rdmaSliceTmp.size);
    }
    level1MultSlice[0] = std::make_pair(true, sdmaSlice);  // true表示使用sdma
    level1MultSlice[1] = std::make_pair(false, rdmaSlice); // false表示rdma

    u32 commPlaneNum = level1MultSlice.size();
    bool isAnyPathCommLevel0 = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING
                                && workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) ?
                                   true :
                                   false;
    CommPlane commplane = (isAnyPathCommLevel0) ? COMM_LEVEL0_ANYPATH_SDMA : COMM_LEVEL0;
    u32 commIndex = GetSubCommInfo(commplane, COMM_INDEX_0).localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1_ANYPATH_SDMA, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1_ANYPATH_SDMA, commIndex);
    CHK_RET(CheckCommSize(COMM_LEVEL1_ANYPATH_RDMA, commIndex + 1));
    SubCommInfo level1RdmaCommInfo = GetSubCommInfo(COMM_LEVEL1_ANYPATH_RDMA, commIndex);
    for (u32 planeIndex = 0; planeIndex < commPlaneNum; planeIndex++) {
        std::vector<Slice>& singleSlice = level1MultSlice[planeIndex].second;
        SubCommInfo level1TempCommInfo = level1MultSlice[planeIndex].first ? level1CommInfo : level1RdmaCommInfo;
        std::unique_ptr<AlgTemplateBase> level1TempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NB in COMM_LEVEL1", __func__);
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL1", __func__);
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
        HcclResult ret = HCCL_SUCCESS;

        if (planeIndex != (commPlaneNum - 1)) {
            ret = LocalNotify::Wait(
                algResResp_->slaveStreams[planeIndex], dispatcher_, algResResp_->notifiesAux[planeIndex], reductionOp);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("stream[%u] wait failed", planeIndex), ret);

            CHK_RET(level1TempAlg->Prepare(
                inputMem, inputMem, scratchMem, count, dataType, algResResp_->slaveStreams[planeIndex], reductionOp,
                LEVEL0_BRIDGE_RANK_ID, singleSlice));

            CHK_RET(level1TempAlg->RegisterProfiler(
                (level1TempCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1TempCommInfo.localRank,
                reductionOp, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[planeIndex]));

            CHK_RET(RunTemplate(level1TempAlg, level1TempCommInfo));
            ret = LocalNotify::Post(
                algResResp_->slaveStreams[planeIndex], dispatcher_, algResResp_->notifiesMain[planeIndex], reductionOp);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
            // 主环record启动从环
            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[planeIndex], reductionOp);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
        } else {
            CHK_RET(level1TempAlg->Prepare(
                inputMem, inputMem, scratchMem, count, dataType, stream, reductionOp, LEVEL0_BRIDGE_RANK_ID,
                singleSlice));
            CHK_RET(level1TempAlg->RegisterProfiler(
                (level1TempCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1TempCommInfo.localRank,
                reductionOp, HCCL_EXEC_STEP_NOT_SET, stream));

            CHK_RET(RunTemplate(level1TempAlg, level1TempCommInfo));
            for (u32 ring = 0; ring < (commPlaneNum - 1); ring++) {
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], reductionOp);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("param.stream[%u] wait failed", ring), ret);
            }
        }
    }
    HCCL_INFO("Level1ReduceScatterConcurrent run success");
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, scratchMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingMultiRootScatter(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::vector<Slice>>& multRingsSliceZero, u32 root, Stream stream, const u64 baseOffset)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(
            singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        SubCommInfo level0RingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);

        u32 rankSize = level0RingCommInfo.localRankSize;
        std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_MULTI_ROOT_SCATTER_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_MULTI_ROOT_SCATTER_RING in COMM_LEVEL0", __func__);
        CHK_SMART_PTR_NULL(tempAlg);

        if (ringIndex != (ringNum - 1)) {
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                CHK_RET(StreamActiveManager::GetInstance(topoAttr_.deviceLogicId)
                            .StreamActive(algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr()));
            }
        }

        u32 rootRank = 0;
        ret = GetRankByUserRank(COMM_LEVEL0, ringIndex, root, rootRank);
        CHK_PRT_RET(
            ret == HCCL_E_PARA,
            HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]invalid root [%u] to get userrank", root), ret);

        if (ringIndex != (ringNum - 1)) { // 0~ringNum-2的环
            ret = LocalNotify::Wait(
                algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesAux[ringIndex], PROF_STAGE_0);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]in stream[%u] wait failed", ringIndex), ret);

            ret = tempAlg->Prepare(
                inputMem, outputMem, outputMem, count, dataType, algResResp_->slaveStreams[ringIndex],
                HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset,
                ringNics[ringIndex]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingMultiRootScatter]stream[%u],multirootscatter(ring) "
                    "prepare failed,return[%d]",
                    ringIndex, ret),
                ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                    + level0RingCommInfo.localRank,
                PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingMultiRootScatter]stream[%u], multirootscatter(ring) "
                    "register profiler failed,return[%d]",
                    ringIndex, ret),
                ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingMultiRootScatter]stream[%u],multirootscatter(ring) "
                    "failed,return[%d]",
                    ringIndex, ret),
                ret);

            ret = LocalNotify::Post(
                algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesMain[ringIndex], PROF_STAGE_0);

            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u] record failed", ringIndex), ret);

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], PROF_STAGE_0);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u] record failed", ringIndex), ret);
        } else { // 主环
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_MULTI_ROOT_SCATTER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_MULTI_ROOT_SCATTER_RING in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(tempAlg);
            ret = tempAlg->Prepare(
                inputMem, outputMem, outputMem, count, dataType, stream, HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID,
                singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingMultiRootScatter]stream[%u],multirootscatter(ring) "
                    "prepare failed,return[%d]",
                    ringIndex, ret),
                ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                    + level0RingCommInfo.localRank,
                PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingMultiRootScatter]stream[%u], multirootscatter(ring) "
                    "register profiler failed,return[%d]",
                    ringIndex, ret),
                ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiRingMultiRootScatter]stream[%u],multirootscatter(ring) run "
                    "failed,return[%d]",
                    ringIndex, ret),
                ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                /* 等待executor执行完毕 , 当前环没有分配数据，跳过此环处理，继续下一个环 */
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], PROF_STAGE_0);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u] wait failed", ring), ret);
            }
        }
    }

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiStreamReduceScatterMeshAtomic(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count, const HcclDataType dataType,
    const HcclReduceOp reductionOp, const std::vector<Slice>& dataSliceVct, Stream& stream,
    const CommPlane commLevelIndex, const u64 baseOffset, HcomCollOpInfo* opInfo)
{
    (void)tag;
    u32 unitSize = SIZE_TABLE[dataType];

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);
    std::unique_ptr<AlgTemplateBase> tempAlg;
    DeviceMem deviceOutputMem = inputMem;
    if (topoAttr_.isSingleMeshAggregation && (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
        && static_cast<bool>((reduceAttr & INLINE_REDUCE_BITMASK)) && (opInfo != nullptr)) {
        if (((opInfo->count) * unitSize <= HCCL_SMALL_COUNT_32_KB)
            && (topoAttr_.deviceNumPerAggregation == DEVICE_EIGHT)) {
            deviceOutputMem = outputMem;
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_HDSTAGE, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_HDSTAGE in COMM_LEVEL0", __func__);
        } else {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_MESH_DIRECT, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_MESH_DIRECT in COMM_LEVEL0", __func__);
        }
    } else {
        tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_MESH_ATOMIC, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_MESH_ATOMIC in COMM_LEVEL0", __func__);
    }
    CHK_SMART_PTR_NULL(tempAlg);

    CHK_RET(CheckCommSize(commLevelIndex, COMM_INDEX_0 + 1));
    const SubCommInfo subCommInfo = GetSubCommInfo(commLevelIndex, COMM_INDEX_0);
    CHK_RET(tempAlg->Prepare(
        inputMem, deviceOutputMem, outputMem, count, dataType, stream, reductionOp, LEVEL0_BRIDGE_RANK_ID, dataSliceVct,
        baseOffset, reduceAttr, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank, opInfo));

    CHK_RET(tempAlg->RegisterProfiler(
        (subCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + subCommInfo.localRank, PROF_STAGE_0,
        HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(RunTemplate(tempAlg, subCommInfo));

    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiStreamReduceScatterMesh(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>>& multStreamsSlice, Stream stream,
    const CommPlane commLevelIndex, const u64 baseOffset)
{
    (void)tag;
    HcclResult ret = HCCL_SUCCESS;
    u64 streamNum = multStreamsSlice.size();
    HCCL_INFO("MultiStreamReduceScatterMesh streamNum[%llu]", streamNum);
    CHK_RET(CheckCommSize(commLevelIndex, streamNum));
    const SubCommInfo zeroCommInfo = GetSubCommInfo(commLevelIndex, COMM_INDEX_0);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    for (u32 streamIndex = 0; streamIndex < streamNum; streamIndex++) {
        std::vector<Slice> singleStreamSlice = multStreamsSlice[streamIndex];
        CHK_PRT_RET(
            singleStreamSlice.size() <= 0,
            HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]singleStreamSlice is empty"), HCCL_E_INTERNAL);

        const SubCommInfo subCommInfo = GetSubCommInfo(commLevelIndex, streamIndex);
        u32 commIndex = subCommInfo.localRank;
        CHK_PRT_RET(
            commIndex >= singleStreamSlice.size(),
            HCCL_ERROR(
                "[CollCommExecutor][MultiStreamReduceScatterMesh]commIndex[%u] => "
                "singleStreamSlice size[%zu]",
                commIndex, singleStreamSlice.size()),
            HCCL_E_INTERNAL);

        u32 rankSize = subCommInfo.localRankSize;
        u32 ringIndexOp = streamIndex;
        std::unique_ptr<AlgTemplateBase> tempAlg;

        if (topoAttr_.isDiffDeviceType) {
            HCCL_DEBUG("[CollCommExecutor][MultiStreamReduceScatterMesh]isDiffDeviceType");
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_MESH_MIX_SS, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_MESH_MIX_SS in COMM_LEVEL0", __func__);
        } else {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_MESH, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_MESH in COMM_LEVEL0", __func__);
        }
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(reduceAttr, streamIndex));

        if (streamIndex != (streamNum - 1)) { // 0~ringNum-2的环
            HCCL_INFO("MultiStreamReduceScatterMesh step into subStream");
            ret = LocalNotify::Wait(
                algResResp_->slaveStreams[streamIndex], dispatcher_, algResResp_->notifiesAux[streamIndex],
                PROF_STAGE_0);
            // 等待executor执行完毕
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] wait failed", streamIndex), ret);

            ret = tempAlg->Prepare(
                inputMem, inputMem, outputMem, count, dataType, algResResp_->slaveStreams[streamIndex], reductionOp,
                LEVEL0_BRIDGE_RANK_ID, singleStreamSlice, baseOffset);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u],ReduceScatter(mesh) "
                    "prepare failed,return[%d]",
                    streamIndex, ret),
                ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                    + zeroCommInfo.localRank,
                PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[streamIndex]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u],ReduceScatter(mesh) "
                    "register Profiler failed,return[%d]",
                    streamIndex, ret),
                ret);

            ret = RunTemplate(tempAlg, subCommInfo);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u],ReduceScatter(mesh) run "
                    "failed,return[%d]",
                    streamIndex, ret),
                ret);

            ret = LocalNotify::Post(
                algResResp_->slaveStreams[streamIndex], dispatcher_, algResResp_->notifiesMain[streamIndex],
                PROF_STAGE_0);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] record failed", streamIndex),
                ret);

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[streamIndex], PROF_STAGE_0);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] record failed", streamIndex),
                ret);
        } else { // 主环
            HCCL_INFO("MultiStreamReduceScatterMesh step into mainStream");

            ret = tempAlg->Prepare(
                inputMem, inputMem, outputMem, count, dataType, stream, reductionOp, LEVEL0_BRIDGE_RANK_ID,
                singleStreamSlice, baseOffset);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u], "
                    "ReduceScatter(mesh) prepare failed, return[%d]",
                    streamIndex, ret),
                ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                    + zeroCommInfo.localRank,
                PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u], ReduceScatter(mesh) "
                    "register Profiler failed, return[%d]",
                    streamIndex, ret),
                ret);

            ret = RunTemplate(tempAlg, subCommInfo);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u], "
                    "ReduceScatter(mesh) run failed, return[%d]",
                    streamIndex, ret),
                ret);

            for (u32 streamIndex = 0; streamIndex < (streamNum - 1); streamIndex++) {
                //  等待executor执行完毕
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[streamIndex], PROF_STAGE_0);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] wait failed", streamIndex),
                    ret);
            }
        }
    }

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return ret;
}

HcclResult CollCommExecutor::PrepareReduceScatterSliceData(
    u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice>& dataSlice) const
{
    CHK_PRT_RET(
        (sliceNum == 0), HCCL_ERROR("[CollCommExecutor][PrepareReduceScatterSliceData]sliceNum is zero."), HCCL_E_PARA);

    dataSlice.resize(sliceNum);
    u64 sliceSize = dataCount * unitSize;
    for (u32 i = 0; i < sliceNum; i++) {
        dataSlice[i].size = sliceSize;
        dataSlice[i].offset = (i * sliceSize);
    }
    return HCCL_SUCCESS;
}

std::vector<std::vector<u32>>
CollCommExecutor::GetRingsOrderByTopoType(u32 ranksSize, TopoType topoType, std::vector<u32>& nicList) const
{
    std::vector<std::vector<u32>> multiRingOrder;
    if (topoType == TopoType::TOPO_TYPE_8P_RING) { // 4 ring 场景
        // 每个环的排序是按照设备物理ID进行的
        std::vector<u32> tmpLevel00 = {0, 1, 2, 6, 5, 4, 7, 3}; // 环0
        std::vector<u32> tmpLevel01 = {0, 3, 7, 4, 5, 6, 2, 1}; // 环1
        std::vector<u32> tmpLevel02 = {0, 2, 3, 1, 5, 7, 6, 4}; // 环2
        std::vector<u32> tmpLevel03 = {0, 4, 6, 7, 5, 1, 3, 2}; // 环3

        // 填充8pring 多环的comm level0 四个环的顺序
        multiRingOrder.push_back(tmpLevel00);
        multiRingOrder.push_back(tmpLevel01);
        multiRingOrder.push_back(tmpLevel02);
        multiRingOrder.push_back(tmpLevel03);
    } else if (topoType == TopoType::TOPO_TYPE_NP_DOUBLE_RING) { // 2 ring 场景
        std::vector<u32> tmpLevel00;                             // 环0
        std::vector<u32> tmpLevel01;                             // 环1
        tmpLevel00 = nicList;                                    // { 0, 1, 2, 3, 4, 5, 6, 7 };
        tmpLevel01.reserve(ranksSize);
        tmpLevel01.push_back(nicList[0]);
        tmpLevel01.insert(tmpLevel01.end(), tmpLevel00.rbegin(), tmpLevel00.rend() - 1);
        // 填充 double ring 两环的comm level0的顺序
        multiRingOrder.push_back(tmpLevel00);
        multiRingOrder.push_back(tmpLevel01);
    } else {                                   // 1 ring 场景
        std::vector<u32> tmpLevel00 = nicList; // 环0

        // 填充 single ring 单环的comm level0的顺序
        multiRingOrder.push_back(tmpLevel00);
    }
    // 打印多个环
    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        for (size_t i = 0; i < multiRingOrder.size(); i++) {
            auto ring = multiRingOrder[i];
            std::ostringstream stringRepresentation;
            for (std::vector<uint32_t>::iterator it = ring.begin(); it != ring.end(); it++) {
                stringRepresentation << *it << " ";
            }
            std::string ringString = stringRepresentation.str();
            const char* charRing = ringString.c_str();
            HCCL_DEBUG("[GetRingsOrderByTopoType] The No.%zu ring: %s", i, charRing);
        }
    }
    return multiRingOrder;
}

std::vector<std::vector<u32>>
CollCommExecutor::GetRingsOrderForAnyPath(u32 ranksSize, TopoType topoType, std::vector<u32>& nicList) const
{
    std::vector<std::vector<u32>> multiRingOrder;
    if (topoType == TopoType::TOPO_TYPE_NP_DOUBLE_RING) { // 2 ring 场景
        std::vector<u32> tmpLevel00;                      // 环0
        std::vector<u32> tmpLevel01;                      // 环1
        std::vector<u32> rohLevel0;
        if (topoMatcher_->CheckSdmaWithRohTopo(nicList, rohLevel0)) {
            tmpLevel00 = rohLevel0;        // 环0, 8卡 { 0, 1, 3, 2, 4, 5, 7, 6 };
            tmpLevel01.reserve(ranksSize); // 环1, 8卡 { 0, 6, 7, 5, 4, 2, 3, 1 };
            tmpLevel01.push_back(rohLevel0[0]);
            tmpLevel01.insert(tmpLevel01.end(), rohLevel0.rbegin(), rohLevel0.rend() - 1);
        } else {
            tmpLevel00 = nicList; // { 0, 1, 2, 3, 4, 5, 6, 7 };
            tmpLevel01.reserve(ranksSize);
            tmpLevel01.push_back(nicList[0]);
            tmpLevel01.insert(tmpLevel01.end(), tmpLevel00.rbegin(), tmpLevel00.rend() - 1);
        }
        // 填充 double ring 两环的comm level0的顺序
        multiRingOrder.push_back(tmpLevel00);
        multiRingOrder.push_back(tmpLevel01);
    } else {                                   // 1 ring 场景
        std::vector<u32> tmpLevel00 = nicList; // 环0

        // 填充 single ring 单环的comm level0的顺序
        multiRingOrder.push_back(tmpLevel00);
    }
    // 打印多个环
    for (size_t i = 0; i < multiRingOrder.size(); i++) {
        auto ring = multiRingOrder[i];
        std::ostringstream stringRepresentation;
        for (std::vector<uint32_t>::iterator it = ring.begin(); it != ring.end(); it++) {
            stringRepresentation << *it << " ";
        }
        std::string ringString = stringRepresentation.str();
        const char* charRing = ringString.c_str();
        HCCL_INFO("[GetRingsOrderByRdmaSdmaConcurrent] The No.%zu ring: %s", i, charRing);
    }
    return multiRingOrder;
}

HcclResult CollCommExecutor::MutliSegSlicePrepare(
    const std::vector<Slice>& dataSegsSlice, std::vector<std::vector<Slice>>& mutliSegsSlices, u32 ringCount) const
{
    std::vector<Slice> singleSegSlices;
    singleSegSlices.reserve(ringCount);
    for (u32 rankId = 0; rankId < dataSegsSlice.size(); rankId++) {
        Slice rankSliceTemp;
        u64 rankDataSize = dataSegsSlice[rankId].size;
        u32 ringIndex = 0;
        u64 offsetStart = dataSegsSlice[rankId].offset;
        if (rankDataSize > 0 && ringCount != 0) {
            u64 sizeTemp = (rankDataSize + ringCount - 1) / ringCount; /* 1是为了向上取整 */
            u64 sizePerRing = AlgTemplateBase::RoundUpWithDivisor(sizeTemp, HCCL_MIN_SLICE_ALIGN);
            u64 residueSize = rankDataSize;

            while (residueSize > 0) {
                u64 singleRingSize = sizePerRing < residueSize ? sizePerRing : residueSize;
                rankSliceTemp.size = singleRingSize;
                rankSliceTemp.offset = offsetStart + rankDataSize - residueSize;
                ringIndex++;
                if (singleRingSize == 0) {
                    HCCL_ERROR(
                        "[CollCommExecutor][MutliSegSlicePrepare]"
                        "Multrings slices prepare: singleRingSize[%llu]",
                        singleRingSize);
                    return HCCL_E_INTERNAL;
                }
                residueSize -= singleRingSize;
                singleSegSlices.push_back(rankSliceTemp);
            }
        }
        while (ringIndex < ringCount) {
            rankSliceTemp.size = 0;
            rankSliceTemp.offset = offsetStart;
            ringIndex++;
            singleSegSlices.push_back(rankSliceTemp);
        }
        mutliSegsSlices.push_back(singleSegSlices); // rings_slice 判断大小不为 8 则异常
        singleSegSlices.clear();
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MutliSegSlicePrepareAvoidCceRewrite(
    const std::vector<Slice>& dataSegsSlice, std::vector<std::vector<Slice>>& mutliSegsSlices, u32 ringCount) const
{
    for (u32 rankId = 0; rankId < dataSegsSlice.size(); rankId++) {
        Slice rankSliceTemp;
        std::vector<Slice> singleSegSlices;
        for (u32 ringIndex = 0; ringIndex < ringCount; ringIndex++) {
            if (ringIndex < ringCount - 1) {
                rankSliceTemp.size = 0;
                rankSliceTemp.offset = dataSegsSlice[rankId].offset;
            } else {
                rankSliceTemp.size = dataSegsSlice[rankId].size;
                rankSliceTemp.offset = dataSegsSlice[rankId].offset;
            }
            singleSegSlices.push_back(rankSliceTemp);
        }
        mutliSegsSlices.push_back(singleSegSlices); // rings_slice 判断大小不为 8 则异常
    }
    return HCCL_SUCCESS;
}

void CollCommExecutor::NicSendSizeCal(
    const std::vector<std::vector<Slice>>& mutliSegsSlices, u32 ringCount, u32 chunkSize,
    const std::vector<u32>& nicList, const std::string& tag)
{
    // 计算每个网口最终会发送的数据量大小
    std::vector<u64> sizeList;
    sizeList.reserve(nicList.size());
    for (u32 nicIdx = 0; nicIdx < nicList.size(); nicIdx++) {
        u64 tempSize = 0;
        for (u32 chunkIdx = 0; chunkIdx < chunkSize; chunkIdx++) {
            for (u32 ringIdx = 0; ringIdx < ringCount; ringIdx++) {
                tempSize += mutliSegsSlices[nicIdx * chunkSize + chunkIdx][ringIdx].size;
            }
        }
        sizeList.push_back(tempSize);
    }
    SetNicSendSize(tag, sizeList);
}

std::vector<std::vector<Slice>> CollCommExecutor::PrepareMultiRingSlice(
    const std::vector<Slice>& dataSegsSlice, const std::string& tag, bool avoidCceRewrite, std::vector<u32> nicList,
    CommPlane commLevelIndex)
{
    // get ranksSize
    u32 ranksSize = GetSubCommInfo(commLevelIndex, COMM_INDEX_0).localRankSize;
    // 获取每个ring上设备的排布顺序，顺序均为deviceID
    sort(nicList.begin(), nicList.end());
    std::vector<std::vector<u32>> multiRingsOrder;
    if (topoMatcher_->GetARSFlag()) {
        multiRingsOrder = GetRingsOrderByTopoType(nicList.size(), TopoType::TOPO_TYPE_NP_DOUBLE_RING, nicList);
    } else {
        multiRingsOrder = GetRingsOrderByTopoType(ranksSize, topoType_, nicList);
    }
    HCCL_INFO("[%s], multiRingsOrder.size() = %u", __func__, multiRingsOrder.size());
    std::vector<std::vector<Slice>> mutliRingsSlices;
    std::vector<std::vector<Slice>> mutliSegsSlices;
    u32 ringCount = multiRingsOrder.size();
    // 单环场景不应该走入此流程，需要在函数外校验
    CHK_PRT_RET(
        ringCount <= 1, HCCL_ERROR("[CollCommExecutor][PrepareMultiRingSlice] ringCount[%u] <= 1", ringCount),
        mutliRingsSlices);

    u32 ringRanks = multiRingsOrder[0].size(); // 获取单个 ring 上设备的数量

    // 将数每块据切分为 ringCount 份
    mutliSegsSlices.reserve(dataSegsSlice.size());
    HcclResult ret;
    if (avoidCceRewrite) {
        ret = MutliSegSlicePrepareAvoidCceRewrite(dataSegsSlice, mutliSegsSlices, ringCount);
    } else {
        ret = MutliSegSlicePrepare(dataSegsSlice, mutliSegsSlices, ringCount);
    }
    if (ret != HCCL_SUCCESS) {
        return mutliRingsSlices;
    }
    u32 chunkSize = ringRanks / nicList.size();
    HCCL_DEBUG("[CollCommExecutor][PrepareMultiRingSlice]chunkSize is %u", chunkSize);
    (void)NicSendSizeCal(mutliSegsSlices, ringCount, chunkSize, nicList, tag);
    std::vector<u32> rankList;
    std::vector<Slice> singleRingSlices;
    std::vector<std::vector<u32>> ringRankList;

    ringRankList.reserve(ringCount);
    singleRingSlices.reserve(ringRanks);
    rankList.reserve(ringRanks);

    for (u32 ringIndex = 0; ringIndex < ringCount; ringIndex++) {
        for (u32 segsIndex = 0; segsIndex < ringRanks; segsIndex++) {
            u32 deviceIdx = multiRingsOrder[ringIndex][segsIndex];
            std::vector<u32>::iterator iterRank = std::find(nicList.begin(), nicList.end(), deviceIdx);
            if (iterRank != nicList.end()) {
                rankList.push_back(segsIndex);
                u32 nicPosition = distance(nicList.begin(), iterRank);
                for (u32 chunkIdx = 0; chunkIdx < chunkSize; chunkIdx++) {
                    Slice tempSlice = mutliSegsSlices[nicPosition * chunkSize + chunkIdx][ringIndex];
                    singleRingSlices.push_back(tempSlice);
                }
            }
        }
        mutliRingsSlices.push_back(singleRingSlices);
        ringRankList.push_back(rankList);
        singleRingSlices.clear();
        rankList.clear();
    }

    ret = SetRingNics(tag, ringRankList);
    if (ret != HCCL_SUCCESS) {
        std::vector<std::vector<Slice>> emptySlice;
        HCCL_ERROR("[Prepare][MultiRingSlice]set nics in ring failed, ret[%u]", ret);
        return emptySlice;
    }
    return mutliRingsSlices;
}

std::vector<std::vector<Slice>> CollCommExecutor::AnyPathPrepareMultiRingSlice(
    const std::vector<Slice>& dataSegsSlice, const std::string& tag, bool avoidCceRewrite, std::vector<u32> nicList)
{
    CheckCommSize(COMM_LEVEL0_ANYPATH_SDMA, COMM_INDEX_1);
    u32 ranksSize = GetSubCommInfo(COMM_LEVEL0_ANYPATH_SDMA, COMM_INDEX_0).localRankSize;
    // 获取每个ring上设备的排布顺序，顺序均为deviceID
    sort(nicList.begin(), nicList.end());
    std::vector<std::vector<u32>> multiRingsOrder = GetRingsOrderForAnyPath(ranksSize, topoType_, nicList);
    std::vector<std::vector<Slice>> mutliRingsSlices;
    std::vector<std::vector<Slice>> mutliSegsSlices;
    u32 ringCount = multiRingsOrder.size();
    // 单环场景不应该走入此流程，需要在函数外校验
    CHK_PRT_RET(
        ringCount <= 1, HCCL_ERROR("[CollCommExecutor][PrepareMultiRingSlice] ringCount[%u] <= 1", ringCount),
        mutliRingsSlices);

    u32 ringRanks = multiRingsOrder[0].size(); // 获取单个 ring 上设备的数量

    // 将数每块据切分为 ringCount 份
    HcclResult ret;
    mutliSegsSlices.reserve(dataSegsSlice.size());
    if (avoidCceRewrite) {
        ret = MutliSegSlicePrepareAvoidCceRewrite(dataSegsSlice, mutliSegsSlices, ringCount);
    } else {
        ret = MutliSegSlicePrepare(dataSegsSlice, mutliSegsSlices, ringCount);
    }
    if (ret != HCCL_SUCCESS) {
        return mutliRingsSlices;
    }
    u32 chunkSize = ringRanks / nicList.size();
    (void)NicSendSizeCal(mutliSegsSlices, ringCount, chunkSize, nicList, tag);
    std::vector<std::vector<u32>> ringRankList;
    std::vector<Slice> singleRingSlices;
    std::vector<u32> rankList;

    ringRankList.reserve(ringCount);
    rankList.reserve(ringRanks);
    singleRingSlices.reserve(ringRanks);

    for (u32 ringIndex = 0; ringIndex < ringCount; ringIndex++) {
        for (u32 segsIndex = 0; segsIndex < ringRanks; ++segsIndex) {
            u32 deviceIdx = multiRingsOrder[ringIndex][segsIndex];
            std::vector<u32>::iterator iterRank = std::find(nicList.begin(), nicList.end(), deviceIdx);
            if (iterRank != nicList.end()) {
                u32 nicPosition = distance(nicList.begin(), iterRank);
                for (u32 chunkIdx = 0; chunkIdx < chunkSize; chunkIdx++) {
                    Slice tempSlice = mutliSegsSlices[nicPosition * chunkSize + chunkIdx][ringIndex];
                    singleRingSlices.push_back(tempSlice);
                }
                rankList.push_back(segsIndex);
            }
        }
        mutliRingsSlices.push_back(singleRingSlices);
        singleRingSlices.clear();
        ringRankList.push_back(rankList);
        rankList.clear();
    }

    ret = SetRingNics(tag, ringRankList);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Prepare][MultiRingSlice]set nics in ring failed, ret[%u]", ret);
        std::vector<std::vector<Slice>> emptySlice;
        return emptySlice;
    }
    return mutliRingsSlices;
}

u64 CollCommExecutor::GetReduceAttr(DeviceMem& inputMem, DeviceMem& outputMem, HcclDataType dataType, HcclReduceOp op)
{
    u64 reduceAttr = 0;
    bool isInlineReduce = IsSupportSDMAReduce(inputMem.ptr(), outputMem.ptr(), dataType, op);
    if (isInlineReduce && algoAttr_.inlineReduceSwitchOn) {
        SalSetBitOne(reduceAttr, ATTR_POS_INLINE_REDUCE);
    }

    bool isRdmaReduce = IsSupportRDMAReduce(dataType, op);
    if (isRdmaReduce) {
        SalSetBitOne(reduceAttr, ATTR_POS_SUPPORT_RDMA_REDUCE);
    }

    return reduceAttr;
}

HcclResult CollCommExecutor::CalUserMemSlices(
    const HcclDataType dataType, const HcomCollOpInfo* opInfo, const std::vector<Slice>& singleRingSliceZero,
    u32 ringIndex, const std::vector<std::vector<u32>>& multiRingsOrder, std::vector<Slice>& userMemSlices)
{
    if (opInfo == nullptr || opInfo->inputAddr == nullptr || opInfo->outputAddr == nullptr) {
        // 910_93场景下，allreduce算子的userMem上的slice信息
        userMemSlices = singleRingSliceZero;
        return HCCL_SUCCESS;
    }
    // 910_93场景下，reduce scatter和AllGather算子的userMem上的slice信息
    std::vector<u32> ring0 = multiRingsOrder[0];
    for (u32 sliceIdx = 0; sliceIdx < singleRingSliceZero.size(); sliceIdx++) {
        Slice userMemSlice;
        u32 deviceId;
        if (ringIndex >= SLICES_FACTOR) {
            deviceId = multiRingsOrder[ringIndex % SLICES_FACTOR][sliceIdx];
        } else {
            deviceId = multiRingsOrder[ringIndex][sliceIdx];
        }

        u32 pos = distance(ring0.begin(), find(ring0.begin(), ring0.end(), deviceId));
        // 专用于MC2调用的 strideCount 特性
        u64 count = (opInfo->strideCount == 0) ? opInfo->count : opInfo->strideCount;
        userMemSlice.offset = pos * count * SIZE_TABLE[dataType] + singleRingSliceZero[0].offset;
        userMemSlice.size = singleRingSliceZero[sliceIdx].size;
        userMemSlices.push_back(userMemSlice);
        HCCL_DEBUG(
            "[CollCommExecutor][CalUserMemSlices] Push back userMemSlice offset[%llu], size[%llu] at rank[%u]",
            userMemSlice.offset, userMemSlice.size, topoAttr_.userRank);
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::GetRankOrder(
    const std::vector<std::vector<u32>>& multiRingsOrder, u32 ringIndex, std::vector<u32>& rankOrder) const
{
    std::vector<u32> ring0 = multiRingsOrder[0];
    std::vector<u32> ringOrder = multiRingsOrder[ringIndex];
    for (u32 i = 0; i < ringOrder.size(); i++) {
        u32 deviceId = ringOrder[i];
        u32 pos = distance(ring0.begin(), find(ring0.begin(), ring0.end(), deviceId));
        rankOrder.push_back(pos);
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::MultiRingScatter(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::vector<Slice>> multRingsSliceZero, u32 root, Stream stream, const HcomCollOpInfo* opInfo,
    const u64 baseOffset)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();

    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    // 拿到ring环映射关系
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder
        = GetRingsOrderByTopoType(level0CommInfo.localRankSize, topoType_, nicList);

    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(
            singleRingSliceZero.empty(), HCCL_ERROR("[CollCommExecutor][MultiRingScatter]singleRingSliceZero is empty"),
            HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        CHK_RET(
            CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemInputSlices));
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));
        SubCommInfo level0RingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);
        u32 rankSize = level0RingCommInfo.localRankSize;

        std::vector<Stream> subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        std::unique_ptr<AlgTemplateBase> tempAlg;
        if (opInfo == nullptr) {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s][KernelRun] Run TEMPLATE_SCATTER_RING in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(tempAlg);
        } else if (opInfo->inputAddr != nullptr) {
            CHK_RET(
                GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing));
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT, dispatcher_);
            HCCL_CONFIG_INFO(
                HCCL_ALG, "[%s][KernelRun] Run TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(tempAlg);
            CHK_RET(tempAlg->Prepare(
                const_cast<HcomCollOpInfo*>(opInfo), topoAttr_.userRank, subStreamsInOneRing, mainSignalsInOneRing,
                subSignalsInOneRing, rankOrder, userMemInputSlices));
        } else {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_SCATTER_RING_DIRECT, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s][KernelRun] Run TEMPLATE_SCATTER_RING_DIRECT in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(tempAlg);
            CHK_RET(tempAlg->Prepare(
                const_cast<HcomCollOpInfo*>(opInfo), topoAttr_.userRank, rankOrder, userMemInputSlices));
        }

        if (ringIndex != (ringNum - 1)) {
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                ret = StreamActiveManager::GetInstance(topoAttr_.deviceLogicId)
                          .StreamActive(algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr());
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u],active stream failed", ringIndex), ret);
            }
        }

        u32 rootRank = 0;
        ret = GetRankByUserRank(COMM_LEVEL0, ringIndex, root, rootRank);
        CHK_PRT_RET(
            ret == HCCL_E_PARA,
            HCCL_ERROR("[CollCommExecutor][MultiRingScatter]invalid root [%u] to get userrank", root), ret);

        if (ret == HCCL_SUCCESS) {
            if (ringIndex != (ringNum - 1)) { // 0~ringNum-2的环
                ret = LocalNotify::Wait(
                    algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesAux[ringIndex],
                    PROF_STAGE_0);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]in stream[%u] wait failed", ringIndex), ret);

                ret = tempAlg->Prepare(
                    inputMem, inputMem, outputMem, count, dataType, algResResp_->slaveStreams[ringIndex],
                    HCCL_REDUCE_RESERVED, rootRank, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingScatter]stream[%u],scatter(ring) prepare failed, "
                        "return[%d]",
                        ringIndex, ret),
                    ret);

                ret = tempAlg->RegisterProfiler(
                    ((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID)
                        + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                    PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingScatter]stream[%u], scatter(ring) register profiler "
                        "failed,return[%d]",
                        ringIndex, ret),
                    ret);

                ret = RunTemplate(tempAlg, level0RingCommInfo);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingScatter]stream[%u],scatter(ring) run failed, "
                        "return[%d]",
                        ringIndex, ret),
                    ret);

                ret = LocalNotify::Post(
                    algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesMain[ringIndex],
                    PROF_STAGE_0);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u] record failed", ringIndex), ret);
                /* 主环record启动从环 */
                ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], PROF_STAGE_0);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u] record failed", ringIndex), ret);
            } else { // 主环
                std::unique_ptr<AlgTemplateBase> tempAlg;
                if (opInfo == nullptr) {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_SCATTER_RING, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s][KernelRun] Run TEMPLATE_SCATTER_RING in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                } else if (opInfo->inputAddr != nullptr) {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT, dispatcher_);
                    HCCL_CONFIG_INFO(
                        HCCL_ALG, "[%s][KernelRun] Run TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT in COMM_LEVEL0",
                        __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(
                        const_cast<HcomCollOpInfo*>(opInfo), topoAttr_.userRank, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices));
                } else {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_SCATTER_RING_DIRECT, dispatcher_);
                    HCCL_CONFIG_INFO(
                        HCCL_ALG, "[%s][KernelRun] Run TEMPLATE_SCATTER_RING_DIRECT in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(
                        const_cast<HcomCollOpInfo*>(opInfo), topoAttr_.userRank, rankOrder, userMemInputSlices));
                }

                ret = tempAlg->Prepare(
                    inputMem, inputMem, outputMem, count, dataType, stream, HCCL_REDUCE_RESERVED, rootRank,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingScatter]stream[%u],scatter(ring) prepare failed, "
                        "return[%d]",
                        ringIndex, ret),
                    ret);
                ret = tempAlg->RegisterProfiler(
                    ((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID)
                        + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                    PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingScatter]stream[%u], scatter(ring) register profiler "
                        "failed,return[%d]",
                        ringIndex, ret),
                    ret);

                ret = RunTemplate(tempAlg, level0RingCommInfo);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[CollCommExecutor][MultiRingScatter]stream[%u],scatter(ring) run failed, "
                        "return[%d]",
                        ringIndex, ret),
                    ret);

                for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                    /* 等待executor执行完毕 , 当前环没有分配数据，跳过此环处理，继续下一个环 */
                    ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], PROF_STAGE_0);
                    CHK_PRT_RET(
                        ret != HCCL_SUCCESS,
                        HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u] wait failed", ring), ret);
                }
            }
        }
    }

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::SetRingNics(const std::string& tag, const std::vector<std::vector<u32>>& ringNics)
{
    std::unique_lock<std::mutex> lock(ringNicListLock_);
    ringNicList_[tag] = ringNics;
    return HCCL_SUCCESS;
}
HcclResult CollCommExecutor::GetRingNics(const std::string& tag, std::vector<std::vector<u32>>& ringNics)
{
    std::unique_lock<std::mutex> lock(ringNicListLock_);
    auto iterRingNic = ringNicList_.find(tag);
    if (iterRingNic == ringNicList_.end()) {
        ringNics = {{0, 1, 2, 3, 4, 5, 6, 7}};
    } else {
        ringNics = iterRingNic->second;
    }
    return HCCL_SUCCESS;
}
HcclResult CollCommExecutor::SetNicSendSize(const std::string& tag, std::vector<u64>& sizeList)
{
    std::unique_lock<std::mutex> lock(nicSendSizeListLock_);
    nicSendSizeList_[tag] = sizeList;
    return HCCL_SUCCESS;
}
HcclResult CollCommExecutor::PrepareLevel1CommInfo(
    u32& segmentIdx, u32& commIndex, u64& hdSize, const SubCommInfo& commInfo,
    const std::vector<std::vector<Slice>>& multRingsSliceZero, const std::string& tag)
{
    segmentIdx = topoAttr_.devicePhyId;
    commIndex = topoAttr_.devicePhyId;
    CHK_PRT_RET(multRingsSliceZero.empty(), HCCL_ERROR("[Prepare][Level1CommInfo]slice map is empty"), HCCL_E_PARA);
    if (multRingsSliceZero.size() > 1) {
        std::vector<u32>::const_iterator iterNic
            = std::find(topoAttr_.nicList.begin(), topoAttr_.nicList.end(), topoAttr_.devicePhyId);
        if (iterNic != topoAttr_.nicList.end()) { // 如果当前rank为通信网口
            u32 nicIdx = std::distance(topoAttr_.nicList.begin(), iterNic);
            std::unique_lock<std::mutex> lock(nicSendSizeListLock_);
            auto iter = nicSendSizeList_.find(tag);
            CHK_PRT_RET(
                iter == nicSendSizeList_.end(),
                HCCL_ERROR(
                    "[Prepare][Level1CommInfo]find tag[%s] in "
                    "nicSendSizeList_ failed",
                    tag.c_str()),
                HCCL_E_INTERNAL);
            CHK_PRT_RET(
                nicIdx >= iter->second.size(),
                HCCL_ERROR(
                    "[Prepare][Level1CommInfo]tag[%s] nicIdx[%u] "
                    "invalid, expect less than %zu",
                    tag.c_str(), nicIdx, iter->second.size()),
                HCCL_E_INTERNAL);
            hdSize = iter->second[nicIdx];                // 通过nicSendSizeList_得到该网口传输数据量
            u32 ringRanks = multRingsSliceZero[0].size(); // 获取单个 ring 上设备的数量
            segmentIdx = ringRanks / topoAttr_.nicList.size() * nicIdx; // 通过网口位置得到该网口传输数据的起始位置
            if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
                segmentIdx = commInfo.localRank;
                hdSize = iter->second[segmentIdx];
                commIndex = segmentIdx;
            }
        } else { // 如果当前rank不是通信网口，则不发送数据
            hdSize = 0;
        }
    } else if (multRingsSliceZero.size() == 1) {
        segmentIdx = commInfo.localRank; // 针对0、4device下
        CHK_PRT_RET(
            segmentIdx >= multRingsSliceZero[0].size(),
            HCCL_ERROR(
                "[Prepare][Level1CommInfo]index is out of "
                "range. Idx[%u] Slice size[%zu]",
                segmentIdx, multRingsSliceZero[0].size()),
            HCCL_E_PARA);
        hdSize = multRingsSliceZero[0][segmentIdx].size;
        commIndex = segmentIdx;
    } else {
        return HCCL_E_PARA;
    }
    HCCL_INFO(
        "[CollCommExecutor][PrepareLevel1CommInfo]userRank[%u] segmentIdx[%u] commIndex[%u] hdSize[%llu]",
        topoAttr_.userRank, segmentIdx, commIndex, hdSize);
    return HCCL_SUCCESS;
}

/* ↓↓ ====================== 用于ZerocopyExecutor ====================== ↓↓ */
HcclResult CollCommExecutor::CalcIntraServerDataSlicesDiscontinuous(
    const OpParam& param, const ExecMem& execMem, u32 level0RankSize, u32 level1RankSize, u32 level2RankSize,
    std::vector<Slice>& dataSegsSlice)
{
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    u64 level0Count = execMem.count * level0RankSize;
    u64 level0StrideCount = param.DataDes.strideCount * level0RankSize;
    u64 sliceSize = perDataSize * execMem.count;
    u64 strideSize = perDataSize * ((level0StrideCount != 0) ? level0StrideCount : level0Count);
    dataSegsSlice.resize(topoAttr_.userRankSize);
    for (u32 i = 0; i < level0RankSize; i++) {
        for (u32 j = 0; j < level1RankSize * level2RankSize; j++) {
            u32 index = i * level1RankSize * level2RankSize + j;
            dataSegsSlice[index].size = sliceSize;
            dataSegsSlice[index].offset = j * strideSize + i * sliceSize;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::CalcIntraServerDataSlicesContinuous(
    const OpParam& param, const ExecMem& execMem, u32 level0RankSize, u32 level1RankSize, u32 level2RankSize,
    std::vector<Slice>& dataSegsSlice) const
{
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    u64 level0Count = execMem.count * level1RankSize * level2RankSize;
    u64 level0StrideCount = param.DataDes.strideCount * level1RankSize * level1RankSize;
    u64 sliceSize = perDataSize * level0Count;
    u64 strideSize = perDataSize * ((level0StrideCount != 0) ? level0StrideCount : level0Count);
    dataSegsSlice.resize(level0RankSize);
    for (u32 i = 0; i < level0RankSize; i++) {
        dataSegsSlice[i].size = sliceSize;
        dataSegsSlice[i].offset = (i * strideSize);
    }
    return HCCL_SUCCESS;
}

void CollCommExecutor::CalcLevel1DataSlices(
    u64 sliceSize, u32 level1RankSize, u32 level2RankSize, std::vector<Slice>& level1DataSegsSlice) const
{
    level1DataSegsSlice.resize(level1RankSize);
    u64 level1SliceSize = sliceSize * level2RankSize;
    for (u32 i = 0; i < level1RankSize; i++) {
        level1DataSegsSlice[i].size = level1SliceSize;
        level1DataSegsSlice[i].offset = i * level1SliceSize;
    }
}

HcclResult CollCommExecutor::GetCommRankInfoNormal(
    u32& level0Rank, u32& level0RankSize, u32& level1Rank, u32& level1RankSize, u32& level2Rank, u32& level2RankSize,
    bool isAHCAlgo)
{
    // 获取通信域信息
    // ==> Level0
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    level0Rank = level0CommInfo.localRank;
    level0RankSize = level0CommInfo.localRankSize;
    // ==> Level1
    CommPlane commPlaneLevel1 = isAHCAlgo ? COMM_LEVEL1_AHC : COMM_LEVEL1;
    CHK_RET(CheckCommSize(commPlaneLevel1, level0Rank + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(commPlaneLevel1, level0Rank);
    level1Rank = level1CommInfo.localRank;
    level1RankSize = level1CommInfo.localRankSize;
    // ==> Level2
    if (isAHCAlgo) { // bypass level2
        level2Rank = 0;
        level2RankSize = 1;
    } else {
        CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
        level2Rank = level2CommInfo.localRank;
        level2RankSize = level2CommInfo.localRankSize;
    }
    return HCCL_SUCCESS;
}
/* ↑↑ ====================== 用于ZerocopyExecutor ======================= ↑↑ */

/* ↓↓ ====================== 用于ExchangeExecutor ====================== ↓↓ */
HcclResult CollCommExecutor::CalExchangeRemoteRankForReduceScatter(u32& remoteRankSend, u32& remoteRankRecv)
{
    u32 userRank = topoAttr_.userRank;
    u32 userRankSize = topoAttr_.userRankSize;
    u32 l2Size = topoAttr_.superPodNum;
    CHK_PRT_RET(
        l2Size == 0, HCCL_ERROR("[CollCommExecutor][CalExchangeRemoteRank] invalid rank size, level2RankSize is 0"),
        HCCL_E_PARA);
    u32 l1Size = topoAttr_.serverNum / l2Size;
    CHK_PRT_RET(
        l1Size == 0, HCCL_ERROR("[CollCommExecutor][CalExchangeRemoteRank] invalid rank size, level1RankSize is 0"),
        HCCL_E_PARA);
    u32 l0Size = userRankSize / l1Size / l2Size;
    u32 l0Index = userRank % l0Size;
    u32 l1ServerIndex = userRank % (l0Size * l1Size) / l0Size;
    u32 l2ServerIndex = userRank / l0Size / l1Size;

    // 计算本端将要发送数据的目标rank
    remoteRankSend = l0Index * l2Size * l1Size + l1ServerIndex * l2Size + l2ServerIndex;

    // 计算本端将要接收数据的目标rank
    u32 r0 = userRank / (l1Size * l2Size);
    u32 r1 = userRank % (l1Size * l2Size) / l2Size;
    u32 r2 = userRank % (l1Size * l2Size) % l2Size;
    remoteRankRecv = r2 * l1Size * l0Size + r1 * l0Size + r0;
    return HCCL_SUCCESS;
}

HcclResult CollCommExecutor::GetTransportForExchange(u32 remoteUserRank, LINK& targetLink)
{
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0RankSize = level0CommInfo.localRankSize;
    CommPlane commPlane = IsLevel0Neighbor(remoteUserRank, level0RankSize) ? COMM_LEVEL0 : COMM_COMBINE_ORDER;

    CHK_PRT_RET(
        COMM_INDEX_0 >= algResResp_->opTransportResponse[commPlane].size(),
        HCCL_ERROR(
            "[%s] commIndex[%u] is larger than opTransportResponse size[%zu]", __func__, COMM_INDEX_0,
            algResResp_->opTransportResponse[commPlane].size()),
        HCCL_E_PARA);
    SingleSubCommTransport& commCombined = algResResp_->opTransportResponse[commPlane][COMM_INDEX_0];

    CHK_PRT_RET(
        commCombined.userRank2subCommRank.count(remoteUserRank) == 0,
        HCCL_ERROR("[%s] remoteUserRank[%u] not found in userRank2subCommRank map.", __func__, remoteUserRank),
        HCCL_E_PARA);

    u32 remoteRank = commCombined.userRank2subCommRank[remoteUserRank];
    CHK_PRT_RET(
        remoteRank >= commCombined.links.size(),
        HCCL_ERROR(
            "[%s] remoteUserRank[%u], get remoteRank[%u], the size of combinedComm links is [%zu]", __func__,
            remoteUserRank, remoteRank, commCombined.links.size()),
        HCCL_E_PARA);
    targetLink = commCombined.links[remoteRank];
    CHK_PTR_NULL(targetLink);

    return HCCL_SUCCESS;
}

bool CollCommExecutor::IsLevel0Neighbor(u32 remoteRank, u32 level0RankSize)
{
    CHK_PRT_RET(level0RankSize == 0, HCCL_ERROR("[%s] invalid rank size, Level0RankSize is 0", __func__), HCCL_E_PARA);
    bool isSameServer = remoteRank / level0RankSize == topoAttr_.userRank / level0RankSize;
    bool isLeftNeighbor = (topoAttr_.userRank + 1) % level0RankSize == remoteRank % level0RankSize;
    bool isRightNeighbor = (topoAttr_.userRank + level0RankSize - 1) % level0RankSize == remoteRank % level0RankSize;
    return isSameServer && (isLeftNeighbor || isRightNeighbor);
}
/* ↑↑ ====================== 用于ExchangeExecutor ======================= ↑↑ */

HcclResult CollCommExecutor::GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo)
{
    HCCL_INFO("[GetAdjInfo] Entry GetAdjInfo.");
    algResResp_ = &algRes;
    SubCommInfo level1CommInfo = {};
    AdjInfo nslbAdjInfo = {};
    if (Getlevel1CommRank(level1CommInfo) != HCCL_SUCCESS) {
        HCCL_WARNING("[GetAdjInfo] Getlevel1CommRank is NULL.");
        return HCCL_SUCCESS;
    }
    u32 localRank = level1CommInfo.localRank;
    u32 localRankSize = level1CommInfo.localRankSize;
    HCCL_INFO("[GetAdjInfo] level1CommInfo.localRank = [%u] localRankSize = [%u].", localRank, localRankSize);

    if (localRankSize == 1) {
        return HCCL_SUCCESS;
    }

    if (level1CommInfo.links.size() < localRankSize) {
        return HCCL_SUCCESS;
    }

    std::unique_ptr<AlgTemplateBase> nslbdp_levelTempAlg;
    if (SelectTempAlg(nslbdp_levelTempAlg, localRankSize) != HCCL_SUCCESS) {
        HCCL_WARNING("[GetAdjInfo] SelectTempAlg is unsuccessful.");
        return HCCL_SUCCESS;
    }
    if (nslbdp_levelTempAlg == nullptr) {
        return HCCL_SUCCESS;
    }
    CHK_RET(nslbdp_levelTempAlg->GetNslbAdjInfo(localRank, localRankSize, level1CommInfo.links, nslbAdjInfo));

    adjInfo.dstRankNum = nslbAdjInfo.dstRankNum;
    HCCL_INFO("[GetAdjInfo] adjInfo.dstRankNum[%u].", adjInfo.dstRankNum);

    for (size_t i = 0; i < nslbAdjInfo.nsAdjInfo.size(); i++) {
        NslbDpAdjInfo dpAdjInfo = {};
        dpAdjInfo.dstLocalRankId = nslbAdjInfo.nsAdjInfo[i].dstLocalRankId;
        dpAdjInfo.phaseId = nslbAdjInfo.nsAdjInfo[i].phaseId;
        dpAdjInfo.rev = 0;
        adjInfo.nsAdjInfo.push_back(dpAdjInfo);
        HCCL_INFO(
            "[GetAdjInfo] dstLocalRankId[%u], phaseId[%u].", nslbAdjInfo.nsAdjInfo[i].dstLocalRankId,
            nslbAdjInfo.nsAdjInfo[i].phaseId);
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
