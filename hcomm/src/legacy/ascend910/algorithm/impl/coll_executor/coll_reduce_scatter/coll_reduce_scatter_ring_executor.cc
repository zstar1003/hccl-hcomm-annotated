/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_ring_executor.h"
#include "alg_template_register.h"

namespace hccl {

CollReduceScatterRingExecutor::CollReduceScatterRingExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_
        = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
           && topoAttr_.deviceType == DevType::DEV_TYPE_910_93);
}

void CollReduceScatterRingExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    // 是否需要scratch memory
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && (topoAttr_.deviceType == DevType::DEV_TYPE_910B || topoAttr_.deviceType == DevType::DEV_TYPE_910_93)
        && isSupportSDMAReduce_ && IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType)) {
        scratchMemFlag_ = false;
    } else {
        scratchMemFlag_ = true;
    }

    // 记录图模式总数据量
    HCCL_DEBUG("[CollReduceScatterRingExecutor][ParseParam]scratchMemFlag is %d", scratchMemFlag_);
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
}

HcclResult CollReduceScatterRingExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    if (scratchMemFlag_) {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            scratchMemSize = inCCLbufferSize_;
        } else {
            scratchMemSize = totalSize_;
        }
    } else {
        scratchMemSize = 0U;
    }
    HCCL_INFO(
        "[CollReduceScatterRingExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%llu]", tag_.c_str(),
        scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 1U;
    if (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_8P_RING) {
        totalStreamNum = LEVEL0_PLANE_NUM_IN_8PRING;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollReduceScatterRingExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult
CollReduceScatterRingExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        if (scratchMemFlag_) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::CCL_OUTPUT;
        }
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        if (scratchMemFlag_) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::PARAM_OUTPUT;
        }
    }
    HCCL_INFO(
        "[CollReduceScatterRingExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollReduceScatterRingExecutor][CalcLevel0CommInfo]tag[%s] start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollReduceScatterRingExecutor][CalcLevel0CommInfo]tag[%s] Calc RingComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

u64 CollReduceScatterRingExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop = inCCLbufferSize_ / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

bool CollReduceScatterRingExecutor::IsHugeData(const u64 curSize, [[maybe_unused]] OpParam* param)
{
    bool hugeData;
    if (DMAReduceFlag_) {
        hugeData = curSize > SDMA_SEND_MAX_SIZE;
    } else {
        hugeData = (curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE)
                   || (curSize > SDMA_SEND_MAX_SIZE);
    }

    return hugeData;
}

HcclResult CollReduceScatterRingExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterRingExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 ringNum
        = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? LEVEL0_PLANE_NUM_IN_8PRING : LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;

    u32 commIndex = (ringNum == LEVEL0_PLANE_NUM_IN_8PRING) ? topoAttr_.devicePhyId : level0CommInfo.localRank;

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    /* ******************网口裁剪步骤: 节点内allreduce *******************************/
    std::vector<Slice> dataSegsSlice;                 // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    u32 sliceNum = level0CommInfo.localRankSize;
    // Slice sliceTemp;
    bool isMultiNic = topoType_ == TopoType::TOPO_TYPE_8P_RING && topoAttr_.nicList.size() != DEVICE_EIGHT;
    if (isMultiNic) {
        u64 inputDataCount = execMem.inputMem.size() / perDataSize;
        CHK_RET(AlgTemplateBase::PrepareSliceData(inputDataCount, perDataSize, sliceNum, 0, dataSegsSlice));
        multiStreamSlice = PrepareMultiRingSlice(dataSegsSlice, param.tag);
        CHK_PRT_RET(
            multiStreamSlice.size() != ringNum,
            HCCL_ERROR(
                "[CollReduceScatterRingExecutor][KernelRun]ringNum[%u] != multiStreamSlice size[%zu]", ringNum,
                multiStreamSlice.size()),
            HCCL_E_INTERNAL);

        CHK_RET(MultiRingAllReduce(
            param.tag, execMem.inputMem, execMem.scratchMem, inputDataCount, param.DataDes.dataType, param.reduceType,
            multiStreamSlice, param.stream, PROF_STAGE_0));

        CHK_RET(
            HcclD2DMemcpyAsync(dispatcher_, execMem.inputMem, execMem.scratchMem, const_cast<Stream&>(param.stream)));
    }

    std::vector<u32>& nicList = const_cast<std::vector<u32>&>(topoAttr_.nicList);
    std::vector<u32>::iterator iterNic = std::find(nicList.begin(), nicList.end(), topoAttr_.devicePhyId);
    bool innRunRet = isMultiNic && (iterNic == nicList.end());
    if (!innRunRet) { // 1. 8P ring的拓扑。2. 网口不满配。3. 当前device不出网口。 的情况下不进行节点间的reduce scatter
        /* ******************第一步: 节点间reducescatter *******************************/
        u32 level1RankSize = level1CommInfo.localRankSize;
        if (level1RankSize > 1) {
            u64 reduceAttr
                = GetReduceAttr(execMem.inputMem, execMem.scratchMem, param.DataDes.dataType, param.reduceType);
            std::unique_ptr<AlgTemplateBase> level1TempAlg;

            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
                HCCL_INFO("ReduceScatter ring: using ring algo inter-server.");
                CHK_SMART_PTR_NULL(level1TempAlg);
                CHK_RET(level1TempAlg->Prepare(reduceAttr));

                u64 ringSize = execMem.inputMem.size() / level1RankSize;
                u64 ringCount = ringSize / perDataSize;

                CHK_RET(level1TempAlg->Prepare(
                    execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount, param.DataDes.dataType,
                    param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0)));
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
                HCCL_INFO("ReduceScatter ring: using nhr algo inter-server.");
                CHK_SMART_PTR_NULL(level1TempAlg);
                CHK_RET(level1TempAlg->Prepare(reduceAttr, false));

                u64 ringSize = execMem.inputMem.size() / level1RankSize;
                u64 ringCount = ringSize / perDataSize;
                CHK_RET(level1TempAlg->Prepare(
                    execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount, param.DataDes.dataType,
                    param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0)));
                level1TempAlg->CloseBarrier();
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_NHR_V1, dispatcher_);
                HCCL_INFO("ReduceScatter ring: using nhr_v1 algo inter-server.");
                CHK_SMART_PTR_NULL(level1TempAlg);
                CHK_RET(level1TempAlg->Prepare(reduceAttr));

                u64 ringSize = execMem.inputMem.size() / level1RankSize;
                u64 ringCount = ringSize / perDataSize;
                CHK_RET(level1TempAlg->Prepare(
                    execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount, param.DataDes.dataType,
                    param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0)));
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
                HCCL_INFO("ReduceScatter ring: using nonuniform-bruck algo inter-server.");
                CHK_SMART_PTR_NULL(level1TempAlg);
                CHK_RET(level1TempAlg->Prepare(reduceAttr));

                u64 ringSize = execMem.inputMem.size() / level1RankSize;
                u64 ringCount = ringSize / perDataSize;
                CHK_RET(level1TempAlg->Prepare(
                    execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount, param.DataDes.dataType,
                    param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0)));
            } else {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_RECURSIVE_HD, dispatcher_);
                HCCL_INFO("ReduceScatter ring: using halving-doubling algo inter-server.");

                CHK_SMART_PTR_NULL(level1TempAlg);
                CHK_RET(level1TempAlg->Prepare(reduceAttr));
                u64 inputDataCount = execMem.inputMem.size() / perDataSize; // count是output的数据个数
                CHK_RET(level1TempAlg->Prepare(
                    execMem.inputMem, execMem.inputMem, execMem.scratchMem, inputDataCount, param.DataDes.dataType,
                    param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0)));
            }
            CHK_RET(level1TempAlg->RegisterProfiler(
                (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_0,
                HCCL_EXEC_STEP_NOT_SET, param.stream));
            CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
        }
    }

    /* ***********第二步: 节点内reducescatter(正常场景), 节点内多根结点scatter(网口裁剪)*****************************/
    CHK_RET(ActiveSlaveStreams(param.stream));

    bool useInlineRduce = false;
    bool isInlineReduce = IsSupportSDMAReduce(
        execMem.inputMem.ptr(), execMem.scratchMem.ptr(), param.DataDes.dataType, param.reduceType);
    useInlineRduce = isInlineReduce && algoAttr_.inlineReduceSwitchOn;
    multiStreamSlice
        = ReduceScatterRingSlicePrepare(ringNum, sliceNum, useInlineRduce, execMem.outputMem, dataSegsSlice, param.tag);
    bool bRet = (multiStreamSlice.size() != ringNum);
    CHK_PRT_RET(
        bRet,
        HCCL_ERROR(
            "[CollReduceScatterRingExecutor][KernelRun]sliceNum-1[%u] != multiStreamSlice size[%zu]", sliceNum - 1,
            multiStreamSlice.size()),
        HCCL_E_INTERNAL);

    if (isMultiNic) { // 网口裁剪情况下需要改变slice最终在rank上位置
        PrepareMultiRingSlice(dataSegsSlice, param.tag, false, nicList); // 刷新多环ringRankList信息
        std::vector<std::vector<u32>> ringNics;
        CHK_RET(GetRingNics(param.tag, ringNics));

        for (u32 ringIdx = 0; ringIdx < ringNum; ringIdx++) { // 按第一个网口位置改变slice最终在rank上的位置
            u32 firstNicIdx = ringNics[ringIdx][0];
            std::rotate(
                multiStreamSlice[ringIdx].begin(), multiStreamSlice[ringIdx].begin() + firstNicIdx,
                multiStreamSlice[ringIdx].end());
        }
    }

    DeviceMem srcMem;
    if (isMultiNic) {
        u32 level1RankSize = topoAttr_.userRankSize / DEVICE_EIGHT; // currComm->commLevel0[0]->UserRankSize();
        // 每个server分配的slice大小
        CHK_PRT_RET(
            level1RankSize == 0, HCCL_ERROR("[CollReduceScatterRingExecutor][KernelRun]level1RankSize is illegal"),
            HCCL_E_PARA);
        u64 serverSliceSize = execMem.inputMem.size() / level1RankSize;
        // 每个服务器对应的偏移
        u32 serverIndex = level1CommInfo.localRank;
        CHK_PRT_RET(
            serverIndex == INVALID_VALUE_RANKID,
            HCCL_ERROR(
                "[CollReduceScatterRingExecutor][KernelRun]get rank of "
                "bridgeRank failed, commIdx[%u]",
                commIndex),
            HCCL_E_PARA);
        u64 serverSliceOffset = serverSliceSize * serverIndex;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            CHK_RET(HcclD2DMemcpyAsync(
                dispatcher_, execMem.scratchMem, execMem.inputMem, const_cast<Stream&>(param.stream)));
        }
        DeviceMem reduceScatterRingOutput = execMem.scratchMem.range(serverSliceOffset, serverSliceSize);
        CHK_SMART_PTR_NULL(reduceScatterRingOutput.ptr());
        u64 countLocal = serverSliceSize / perDataSize;
        CHK_RET(MultiRingMultiRootScatter(
            param.tag, reduceScatterRingOutput, reduceScatterRingOutput, countLocal, param.DataDes.dataType,
            multiStreamSlice, serverIndex * DEVICE_EIGHT, param.stream, serverSliceOffset));

        srcMem
            = reduceScatterRingOutput.range(dataSegsSlice[topoAttr_.devicePhyId].offset, execMem.count * perDataSize);
        CHK_SMART_PTR_NULL(srcMem.ptr());
    } else {
        u32 level1RankSize = level1CommInfo.localRankSize;
        // 每个server分配的slice大小
        u64 serverSliceSize = execMem.inputMem.size() / level1RankSize;
        // 每个服务器对应的偏移
        u32 serverIndex = level1CommInfo.localRank;
        u64 serverSliceOffset = serverSliceSize * serverIndex;
        HCCL_DEBUG(
            "inputMem.size=%llu, level0CommInfo.localRankSize=%u, serverSliceSize=%llu, serverSliceOffset=%llu "
            "commIndex=%u commLevel1[commIndex]->rank=%u",
            execMem.inputMem.size(), level0CommInfo.localRankSize, serverSliceSize, serverSliceOffset, commIndex,
            level1CommInfo.localRank);
        DeviceMem reduceScatterRingInput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
        CHK_SMART_PTR_NULL(reduceScatterRingInput.ptr());
        DeviceMem reduceScatterRingOutput = execMem.scratchMem.range(serverSliceOffset, serverSliceSize);
        CHK_SMART_PTR_NULL(reduceScatterRingOutput.ptr());
        u64 countLocal = serverSliceSize / perDataSize;

        HcomCollOpInfo opInfo = {"",
                                 execMem.inputPtr,
                                 execMem.outputPtr,
                                 param.DataDes.count,
                                 param.DataDes.dataType,
                                 param.root,
                                 param.reduceType,
                                 0};
        HcomCollOpInfo* opInfoPtr = nullptr;
        if (DMAReduceFlag_) {
            opInfoPtr = &opInfo;
        }

        CHK_RET(MultiRingReduceScatter(
            param.tag, reduceScatterRingInput, reduceScatterRingOutput, countLocal, param.DataDes.dataType,
            param.reduceType, multiStreamSlice, param.stream, PROF_STAGE_1, serverSliceOffset, opInfoPtr));

        srcMem
            = execMem.inputMem.range(serverSliceOffset + dataSegsSlice[commIndex].offset, execMem.count * perDataSize);
        CHK_SMART_PTR_NULL(srcMem.ptr());
    }

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));

    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingExecutor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
{
    if (CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1) != HCCL_SUCCESS) {
        return HCCL_E_UNAVAIL;
    }
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 ringNum
        = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? LEVEL0_PLANE_NUM_IN_8PRING : LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
    u32 commIndex = (ringNum == LEVEL0_PLANE_NUM_IN_8PRING) ? topoAttr_.devicePhyId : level0CommInfo.localRank;

    if (CheckCommSize(COMM_LEVEL1, commIndex + 1) != HCCL_SUCCESS) {
        return HCCL_E_UNAVAIL;
    }
    level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    return HCCL_SUCCESS;
}

HcclResult
CollReduceScatterRingExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NHR_V1, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RECURSIVE_HD, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
        }
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}

REGISTER_EXEC("ReduceScatterRingExecutor", ReduceScatterRing, CollReduceScatterRingExecutor);

} // namespace hccl
