/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_mesh_executor.h"
#include "alg_template_register.h"

namespace hccl {

CollReduceScatterMeshExecutor::CollReduceScatterMeshExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

void CollReduceScatterMeshExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    // 910B 图模式非确定计算，inlineReduce使能，MESH拓扑场景下，创建一个mesh平面
    bool isInlineReduce
        = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    meshSinglePlane_ = (topoAttr_.deviceType == DevType::DEV_TYPE_910B)
                       && topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE && isInlineReduce
                       && (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_DEBUG("[CollReduceScatterMeshExecutor][ParseParam]meshSinglePlane is %d", meshSinglePlane_);

    // 是否需要scratch memory
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && (topoAttr_.deviceType == DevType::DEV_TYPE_910B || topoAttr_.deviceType == DevType::DEV_TYPE_910_93)
        && isSupportSDMAReduce_ && IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType)) {
        scratchMemFlag_ = false;
    } else {
        scratchMemFlag_ = true;
    }

    // 记录图模式总数据量
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
}

HcclResult CollReduceScatterMeshExecutor::CalcScratchMemSize(u64& scratchMemSize)
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
        "[CollReduceScatterMeshExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%llu]", tag_.c_str(),
        scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollReduceScatterMeshExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult
CollReduceScatterMeshExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
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
        "[CollReduceScatterMeshExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = meshSinglePlane_;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollReduceScatterMeshExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = inCCLbufferSize_ / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

bool CollReduceScatterMeshExecutor::IsHugeData(const u64 curSize, [[maybe_unused]] OpParam* param)
{
    bool hugeData = (curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE)
                    || (curSize > SDMA_SEND_MAX_SIZE);
    return hugeData;
}

HcclResult CollReduceScatterMeshExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterMeshExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    /* ******************第一步: 节点间reducescatter *******************************/
    u32 commIndex = level0CommInfo.localRank; // 找到rank所在的节点间平面

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    u32 level1RankSize = level1CommInfo.localRankSize;
    if (level1RankSize > 1) {
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<AlgTemplateBase> level1TempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
            HCCL_INFO("ReduceScatter mesh: using ring algo inter-server.");
            u64 ringSize = execMem.inputMem.size() / level1RankSize;
            u64 ringCount = ringSize / perDataSize;
            CHK_RET(level1TempAlg->Prepare(
                execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount, param.DataDes.dataType, param.stream,
                param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
            HCCL_INFO("ReduceScatter mesh: using nhr algo inter-server.");
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr, false));
            u64 ringSize = execMem.inputMem.size() / level1RankSize;
            u64 ringCount = ringSize / perDataSize;
            CHK_RET(level1TempAlg->Prepare(
                execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount, param.DataDes.dataType, param.stream,
                param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0)));
            level1TempAlg->CloseBarrier();
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NHR_V1, dispatcher_);
            HCCL_INFO("ReduceScatter mesh: using nhr_v1 algo inter-server.");
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
            u64 ringSize = execMem.inputMem.size() / level1RankSize;
            u64 ringCount = ringSize / perDataSize;
            CHK_RET(level1TempAlg->Prepare(
                execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount, param.DataDes.dataType, param.stream,
                param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            HCCL_INFO("ReduceScatter mesh: using nonuniform-bruck algo inter-server.");
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
            u64 ringSize = execMem.inputMem.size() / level1RankSize;
            u64 ringCount = ringSize / perDataSize;
            CHK_RET(level1TempAlg->Prepare(
                execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount, param.DataDes.dataType, param.stream,
                param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RECURSIVE_HD, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
            HCCL_INFO("ReduceScatter mesh: using halving-doubling algo inter-server.");
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

    /* *******************第二步: 节点内reducescatter ******************************************/
    CHK_RET(ActiveSlaveStreams(param.stream));

    u32 sliceNum = level0CommInfo.localRankSize;
    // 根据数据量算每个环上数据的偏移和大小，把做完hd的slice均分成RankSize份
    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareReduceScatterSliceData(execMem.count, perDataSize, sliceNum, dataSegsSlice));

    // 每个server分配的slice大小
    u64 serverSliceSize = execMem.inputMem.size() / level1RankSize;
    // 每个服务器对应的偏移
    u64 serverSliceOffset = serverSliceSize * level1CommInfo.localRank;

    HCCL_DEBUG(
        "inputMem.size=%llu, level0CommInfo.localRankSize=%u, serverSliceSize=%llu, serverSliceOffset=%llu "
        "commIndex=%u level1CommInfo.localRank=%u",
        execMem.inputMem.size(), level0CommInfo.localRankSize, serverSliceSize, serverSliceOffset, commIndex,
        level1CommInfo.localRank);

    DeviceMem reduceScatterMeshInput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(reduceScatterMeshInput);
    DeviceMem reduceScatterMeshOutput = execMem.scratchMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(reduceScatterMeshOutput);

    HcomCollOpInfo* opInfoPtr = nullptr;

    if (topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE
        && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)
        && (topoAttr_.deviceType == DevType::DEV_TYPE_910B && param.reduceType != HCCL_REDUCE_PROD)) {
        CHK_RET(MultiStreamReduceScatterMeshAtomic(
            param.tag, reduceScatterMeshInput, reduceScatterMeshOutput, // 非确定性
            execMem.count, param.DataDes.dataType, param.reduceType, dataSegsSlice, const_cast<Stream&>(param.stream),
            COMM_LEVEL0, serverSliceOffset, opInfoPtr));
    } else {
        std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
        // mesh算法stream数量为rank数减1
        CHK_RET(AlgTemplateBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));
        CHK_RET(MultiStreamReduceScatterMesh(
            param.tag, reduceScatterMeshInput, reduceScatterMeshOutput, // 确定性
            execMem.count, param.DataDes.dataType, param.reduceType, multiStreamSlice,
            const_cast<Stream&>(param.stream), COMM_LEVEL0, serverSliceOffset));
    }

    DeviceMem srcMem
        = execMem.inputMem.range(serverSliceOffset + dataSegsSlice[commIndex].offset, execMem.count * perDataSize);
    CHK_SMART_PTR_NULL(srcMem);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));

    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshExecutor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
{
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 commIndex = level0CommInfo.localRank; // 找到rank所在的节点间平面
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));

    level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    return HCCL_SUCCESS;
}

HcclResult
CollReduceScatterMeshExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
            HCCL_INFO("ReduceScatter mesh: using nhr algo inter-server.");
            CHK_SMART_PTR_NULL(level1TempAlg);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NHR_V1, dispatcher_);
            HCCL_INFO("ReduceScatter mesh: using nhr_v1 algo inter-server.");
            CHK_SMART_PTR_NULL(level1TempAlg);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            HCCL_INFO("ReduceScatter mesh: using nonuniform-bruck algo inter-server.");
            CHK_SMART_PTR_NULL(level1TempAlg);
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RECURSIVE_HD, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
        }
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterMeshExecutor", ReduceScatterMesh, CollReduceScatterMeshExecutor);
} // namespace hccl
