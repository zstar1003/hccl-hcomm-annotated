/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_mesh_executor.h"

namespace hccl {

CollReduceMeshExecutor::CollReduceMeshExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceExecutor(dispatcher, topoMatcher)
{}

void CollReduceMeshExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    root_ = param.root;
    bool isInlineReduce
        = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    meshSinglePlane_ = (topoAttr_.deviceType == DevType::DEV_TYPE_910B)
                       && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE && isInlineReduce
                       && (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
}

HcclResult CollReduceMeshExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollReduceMeshExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceMeshExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceMeshExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollReduceMeshExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(), inputType,
        outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceMeshExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = meshSinglePlane_;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceMeshExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceMeshExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    std::vector<Slice> dataSegsSlice;                 // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    // std::unique_ptr<AlgTemplateBase> level1TempAlg;
    // std::unique_ptr<AlgTemplateBase> level0TempAlg;

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 sliceNum = level0CommInfo.localRankSize;
    // 根据数据量算每个环上数据的偏移和大小
    CHK_RET(AlgTemplateBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));
    // mesh算法stream数量为server内rank数减1

    CHK_RET(ActiveSlaveStreams(param.stream));

    if (topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE
        && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)
        && (topoAttr_.deviceType == DevType::DEV_TYPE_910B && param.reduceType != HCCL_REDUCE_PROD)) {
        CHK_RET(MultiStreamReduceScatterMeshAtomic(
            tag_, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.reduceType,
            dataSegsSlice, const_cast<Stream&>(param.stream), COMM_LEVEL0));
    } else {
        std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
        // mesh算法stream数量为rank数减1
        CHK_RET(AlgTemplateBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));
        CHK_RET(MultiStreamReduceScatterMesh(
            tag_, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.reduceType,
            multiStreamSlice, const_cast<Stream&>(param.stream), COMM_LEVEL0));
    }
    HCCL_INFO("[CollReduceMeshExecutor]reduce mesh stage0 run success");

    // step2: 节点间的reduce
    u32 commIndex = level0CommInfo.localRank;
    CHK_PRT_RET(
        commIndex >= dataSegsSlice.size(),
        HCCL_ERROR(
            "[CollReduceMeshExecutor][Run]commIndex[%u] >= "
            "dataSegsSlice size[%zu]",
            commIndex, dataSegsSlice.size()),
        HCCL_E_INTERNAL);

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    HCCL_DEBUG(
        "commIdx:%u TagCommInfo[%s].commLevel1.size():%zu", commIndex, tag_.c_str(), level1CommInfo.links.size());

    DeviceMem reduceInput = execMem.inputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(reduceInput);
    DeviceMem reduceOutput = execMem.outputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(reduceOutput);

    u32 rankSize = level1CommInfo.localRankSize;
    if (rankSize > 1) {
        u64 reduceAttr = GetReduceAttr(reduceInput, reduceOutput, param.DataDes.dataType, param.reduceType);

        u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);

        CHK_PRT_RET(
            subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR(
                "[ReduceOperator][ReduceMeshExecutor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
                subUserrankRoot, topoAttr_.userRank, param.root),
            HCCL_E_INTERNAL);

        u32 planeRoot = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, planeRoot));

        std::unique_ptr<AlgTemplateBase> level1TempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCE_RING, dispatcher_);
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCE_RECURSIVE_HALVING_DOUBLING, dispatcher_);
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
        // 节点间的hd 使用环0来记录
        u64 hdCount = dataSegsSlice[commIndex].size / perDataSize;

        CHK_RET(level1TempAlg->Prepare(
            reduceInput, reduceOutput, reduceOutput, hdCount, param.DataDes.dataType, param.stream, param.reduceType,
            planeRoot, std::vector<Slice>(0), dataSegsSlice[commIndex].offset));

        CHK_RET(level1TempAlg->RegisterProfiler(
            (level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_1,
            HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    } else {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, reduceOutput, reduceInput, const_cast<Stream&>(param.stream)));
    }

    HCCL_INFO("[CollReduceMeshExecutor]reduce mesh stage1 run success");

    SingleSubCommTransport& level0TransportInfo
        = const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[COMM_LEVEL0][COMM_INDEX_0]);

    if (level0TransportInfo.userRank2subCommRank.find(param.root) != level0TransportInfo.userRank2subCommRank.end()) {
        const u32 rootRank = level0TransportInfo.userRank2subCommRank[param.root];

        std::unique_ptr<AlgTemplateBase> level0TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_GATHER_MESH, dispatcher_);
        CHK_SMART_PTR_NULL(level0TempAlg);
        CHK_RET(level0TempAlg->Prepare(
            algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank));
        CHK_RET(level0TempAlg->Prepare(
            execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count, param.DataDes.dataType,
            const_cast<Stream&>(param.stream), param.reduceType, rootRank, dataSegsSlice));

        u32 rankSize = level0CommInfo.localRankSize;
        CHK_RET(level0TempAlg->RegisterProfiler(
            (0 << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                + level0CommInfo.localRank,
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    }
    HCCL_INFO("[CollReduceMeshExecutor]reduce mesh stage2 run success");

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceMeshExecutor", ReduceMesh, CollReduceMeshExecutor);

} // namespace hccl
