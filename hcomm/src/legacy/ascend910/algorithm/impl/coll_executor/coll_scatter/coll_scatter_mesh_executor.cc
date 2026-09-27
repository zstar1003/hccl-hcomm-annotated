/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_scatter_mesh_executor.h"

namespace hccl {
CollScatterMeshExecutor::CollScatterMeshExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollScatterExecutor(dispatcher, topoMatcher)
{}

HcclResult CollScatterMeshExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollScatterMeshExecutor][CalcLevel0CommInfo]tag[%s] start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);

    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollScatterMeshExecutor][CalcLevel0CommInfo]tag[%s] Calc meshComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollScatterMeshExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    return HCCL_SUCCESS;
}

HcclResult CollScatterMeshExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollScatterMeshExecutor] scatter starts.");
    Stream& stream = const_cast<Stream&>(param.stream);

    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0LocalRank = level0CommInfo.localRank;
    u32 level0LocalRankSize = level0CommInfo.localRankSize;

    u32 commIndex = level0LocalRank;

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    u32 level1LocalRank = level1CommInfo.localRank;
    u32 level1LocalRankSize = level1CommInfo.localRankSize;

    bool bRet = level0LocalRankSize == 0;
    CHK_PRT_RET(
        bRet, HCCL_ERROR("[CollScatterMeshExecutor][KernelRun]tag[%s],comm level0 is empty", tag_.c_str()),
        HCCL_E_INTERNAL);

    /* ***********第一步: 节点间scatter ****************************/
    u32 subRoot = INVALID_VALUE_RANKID;
    CHK_RET(topoMatcher_->GetSubRootForScatter(param.root, subRoot));
    CHK_PRT_RET(
        subRoot == INVALID_VALUE_RANKID,
        HCCL_ERROR(
            "[CollScatterMeshExecutor][KernelRun]GetSubRootForScatter failed, "
            "userRank[%u], root[%u], subRoot[%u]",
            topoAttr_.userRank, param.root, subRoot),
        HCCL_E_INTERNAL);
    HCCL_DEBUG(
        "[CollScatterMeshExecutor][KernelRun]GetSubRootForScatter, userRank[%u], root[%u], subRoot[%u]",
        topoAttr_.userRank, param.root, subRoot);
    CHK_RET(KernelRunLevel1(
        execMem.inputMem, execMem.count, param.DataDes.dataType, commIndex, param.root, subRoot, COMM_LEVEL1, stream));

    /* ***********第二步: 节点内scatter*****************************/
    // 根据数据量算每个环上数据的偏移和大小
    u32 sliceNum = level0LocalRankSize;
    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareDataSlice(execMem.count, perDataSize, sliceNum, dataSegsSlice));

    // 每个server分配的slice大小
    u64 serverSliceSize = execMem.inputMem.size() / level1LocalRankSize;
    // 每个服务器对应的偏移
    u64 serverSliceOffset = serverSliceSize * level1LocalRank;
    DeviceMem scatterMeshInput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(scatterMeshInput);
    DeviceMem scatterMeshOutput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(scatterMeshOutput);

    std::unique_ptr<AlgTemplateBase> level0TempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_MESH, dispatcher_);
    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(level0LocalRank, level0LocalRankSize));
    // 偏移需要带入prepare
    u32 rootRankLevel0 = 0;
    CHK_RET(GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, subRoot, rootRankLevel0));
    CHK_PRT_RET(
        rootRankLevel0 == INVALID_VALUE_RANKID,
        HCCL_ERROR(
            "[CollScatterMeshExecutor][KernelRun]rootRankLevel0[%u] is invalid, userRank[%u], subRoot[%u]",
            rootRankLevel0, topoAttr_.userRank, subRoot),
        HCCL_E_INTERNAL);

    CHK_RET(level0TempAlg->Prepare(
        scatterMeshInput, scatterMeshOutput, execMem.inputMem, execMem.count, param.DataDes.dataType, stream,
        HCCL_REDUCE_RESERVED, rootRankLevel0, dataSegsSlice, serverSliceOffset));

    HcclResult ret = RunTemplate(level0TempAlg, level0CommInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollScatterMeshExecutor][KernelRun]scatter(mesh) RunTemplate failed,return[%d]", ret), ret);

    // 将scratchMem赋值给outputMem
    u8* scatterMeshOutputPtr = static_cast<u8*>(scatterMeshOutput.ptr());
    DeviceMem resultMem(scatterMeshOutputPtr + execMem.outputMem.size() * level0LocalRank, execMem.outputMem.size());
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, resultMem, stream));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ScatterMeshExecutor", ScatterMesh, CollScatterMeshExecutor);

} // namespace hccl
