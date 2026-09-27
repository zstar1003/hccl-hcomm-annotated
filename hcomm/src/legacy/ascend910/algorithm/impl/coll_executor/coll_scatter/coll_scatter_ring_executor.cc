/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_scatter_ring_executor.h"

namespace hccl {
CollScatterRingExecutor::CollScatterRingExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollScatterExecutor(dispatcher, topoMatcher)
{}

HcclResult CollScatterRingExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollScatterRingExecutor][CalcLevel0CommInfo]tag[%s] start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);

    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollScatterRingExecutor][CalcLevel0CommInfo]tag[%s] Calc RingComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollScatterRingExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
    if (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_8P_RING) {
        totalStreamNum = LEVEL0_PLANE_NUM_IN_8PRING;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollScatterRingExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollScatterRingExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollScatterRingExecutor] scatter starts.");
    Stream& stream = const_cast<Stream&>(param.stream);

    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0LocalRank = level0CommInfo.localRank;
    u32 level0LocalRankSize = level0CommInfo.localRankSize;

    u32 commIndex = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? topoAttr_.devicePhyId : level0LocalRank;

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    u32 level1LocalRank = level1CommInfo.localRank;
    u32 level1LocalRankSize = level1CommInfo.localRankSize;

    bool bRet = level0LocalRankSize == 0;
    CHK_PRT_RET(
        bRet, HCCL_ERROR("[CollScatterRingExecutor][KernelRun]tag[%s],comm level0 is empty", tag_.c_str()),
        HCCL_E_INTERNAL);

    /* ***********第一步: 节点间scatter ****************************/
    u32 subRoot = INVALID_VALUE_RANKID;
    CHK_RET(topoMatcher_->GetSubRootForScatter(param.root, subRoot));
    CHK_PRT_RET(
        subRoot == INVALID_VALUE_RANKID,
        HCCL_ERROR(
            "[CollScatterRingExecutor][KernelRun]GetSubRootForScatter failed, "
            "userRank[%u], root[%u], subRoot[%u]",
            topoAttr_.userRank, param.root, subRoot),
        HCCL_E_INTERNAL);
    HCCL_DEBUG(
        "[CollScatterRingExecutor][KernelRun]GetSubRootForScatter, userRank[%u], root[%u], subRoot[%u]",
        topoAttr_.userRank, param.root, subRoot);
    CHK_RET(KernelRunLevel1(
        execMem.inputMem, execMem.count, param.DataDes.dataType, commIndex, param.root, subRoot, COMM_LEVEL1, stream));

    /* ***********第二步: 节点内scatter*****************************/
    u32 sliceNum = level0LocalRankSize;
    std::vector<Slice> dataSegsSlice;
    u32 outputOffset = level0LocalRank;
    CHK_RET(PrepareScatterRingSliceData(execMem.count, perDataSize, sliceNum, dataSegsSlice, outputOffset));

    // 每个server分配的slice大小
    u64 serverSliceSize = execMem.inputMem.size() / level1LocalRankSize;
    // 每个服务器对应的偏移
    u64 serverSliceOffset = serverSliceSize * level1LocalRank;
    DeviceMem scatterRingInput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(scatterRingInput);
    DeviceMem scatterRingOutput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(scatterRingOutput);

    std::unique_ptr<AlgTemplateBase> level0TempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_RING, dispatcher_);
    CHK_SMART_PTR_NULL(level0TempAlg);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_SCATTER_RING in COMM_LEVEL0", __func__);

    // 偏移需要带入prepare
    u32 rootRankLevel0 = 0;
    CHK_RET(GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, subRoot, rootRankLevel0));
    CHK_PRT_RET(
        rootRankLevel0 == INVALID_VALUE_RANKID,
        HCCL_ERROR(
            "[CollScatterRingExecutor][KernelRun]rootRankLevel0[%u] is invalid, userRank[%u], subRoot[%u]",
            rootRankLevel0, topoAttr_.userRank, subRoot),
        HCCL_E_INTERNAL);

    CHK_RET(level0TempAlg->Prepare(
        scatterRingInput, scatterRingOutput, execMem.inputMem, execMem.count, param.DataDes.dataType, stream,
        HCCL_REDUCE_RESERVED, rootRankLevel0, dataSegsSlice, serverSliceOffset));

    HcclResult ret = RunTemplate(level0TempAlg, level0CommInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollScatterRingExecutor][KernelRun]scatter(ring) RunTemplate failed,return[%d]", ret), ret);

    // 将scratchMem赋值给outputMem
    u8* scatterRingOutputPtr = static_cast<u8*>(scatterRingOutput.ptr());
    DeviceMem resultMem(scatterRingOutputPtr + execMem.outputMem.size() * outputOffset, execMem.outputMem.size());
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, resultMem, stream));
    return HCCL_SUCCESS;
}

HcclResult CollScatterRingExecutor::PrepareScatterRingSliceData(
    u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice>& dataSlice, u32& outputOffset)
{
    CHK_PRT_RET(
        (sliceNum == 0), HCCL_ERROR("[CollScatterRingExecutor][PrepareScatterRingSliceData]sliceNum is zero."),
        HCCL_E_PARA);

    // 根据数据量算每个环上数据的偏移和大小
    CHK_RET(PrepareDataSlice(dataCount, unitSize, sliceNum, dataSlice));

    if (topoType_ == TopoType::TOPO_TYPE_8P_RING) {
        std::vector<u32> tmpRing0 = {0, 1, 2, 6, 5, 4, 7, 3};
        outputOffset = tmpRing0[outputOffset];
        CHK_RET(ReorderSlice(dataSlice, tmpRing0));
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ScatterRingExecutor", ScatterRing, CollScatterRingExecutor);

} // namespace hccl
