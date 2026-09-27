/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_ring_plus_hd_executor.h"

namespace hccl {

CollReduceRingPlusHdExecutor::CollReduceRingPlusHdExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceExecutor(dispatcher, topoMatcher)
{}

HcclResult CollReduceRingPlusHdExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 1U;

    if (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_8P_RING) {
        totalStreamNum = LEVEL0_PLANE_NUM_IN_8PRING;
    } else if (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING) {
        totalStreamNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollReduceRingPlusHdExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceRingPlusHdExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceRingPlusHdExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollReduceRingPlusHdExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceRingPlusHdExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollReduceRingPlusHdExecutor][CalcLevel0CommInfo]tag[%s] start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollReduceRingPlusHdExecutor][CalcLevel0CommInfo]tag[%s] Calc RingComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollReduceRingPlusHdExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceRingPlusHdExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    std::vector<Slice> dataSegsSlice;             // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> mulRingSlice; // 数据基于该rank上环0的偏移

    // step1: 节点内的reducescatter
    u32 ringNum
        = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? LEVEL0_PLANE_NUM_IN_8PRING : LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;

    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 按ranksize得到内存切分slice数为8
    u32 sliceNum = level0CommInfo.localRankSize;
    CHK_RET(AlgTemplateBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));

    /* 外层:reducescatter */
    // 将每slice再切分成4份，按各ring的dev顺序排列
    if (ringNum == LEVEL0_PLANE_NUM_IN_8PRING) {
        // 构造ring algorithm对应的reduce-scatter实例
        mulRingSlice = PrepareMultiRingSlice(dataSegsSlice, tag_, false, topoAttr_.nicList);
        CHK_PRT_RET(
            mulRingSlice.size() != ringNum,
            HCCL_ERROR(
                "[CollReduceRingPlusHdExecutor]ringNum[%u] "
                "!=mulRingSlice size[%zu]",
                ringNum, mulRingSlice.size()),
            HCCL_E_INTERNAL);
    } else {
        mulRingSlice.push_back(dataSegsSlice); // 应该offset全为0，而大小和dataSegsSlice中一样,里面的offset不使用
    }
    CHK_RET(MultiRingReduceScatter(
        tag_, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.reduceType,
        mulRingSlice, param.stream, PROF_STAGE_0, 0, nullptr));

    HCCL_INFO("reduce 8PringHD stage0 run success");

    // step2: 节点间的reduce
    u64 hdSize;
    u32 segmentIdx;
    u32 commIndex;
    CHK_RET(PrepareLevel1CommInfo(segmentIdx, commIndex, hdSize, level0CommInfo, mulRingSlice, tag_));

    u64 hdCount = hdSize / perDataSize;

    HCCL_DEBUG(
        "commIdx:%u TagCommInfo[%s].commLevel1.size():%u", commIndex, tag_.c_str(), level0CommInfo.localRankSize);

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    DeviceMem reduceInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
    CHK_SMART_PTR_NULL(reduceInput);
    DeviceMem reduceOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
    CHK_SMART_PTR_NULL(reduceOutput);

    u64 reduceAttr = GetReduceAttr(reduceInput, reduceOutput, param.DataDes.dataType, param.reduceType);

    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCE_RING, dispatcher_);
    } else {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCE_RECURSIVE_HALVING_DOUBLING, dispatcher_);
    }
    CHK_SMART_PTR_NULL(level1TempAlg);
    CHK_RET(level1TempAlg->Prepare(reduceAttr));

    u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
    CHK_PRT_RET(
        subUserrankRoot == INVALID_VALUE_RANKID,
        HCCL_ERROR(
            "[ReduceOperator][ReduceRingPlusHd]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]", subUserrankRoot,
            topoAttr_.userRank, param.root),
        HCCL_E_INTERNAL);

    u32 planeRoot = 0;
    CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, planeRoot));

    u32 ranksize = level1CommInfo.localRankSize;
    // 节点间的hd 使用环0来记录
    CHK_RET(level1TempAlg->Prepare(
        reduceInput, reduceOutput, reduceOutput, hdCount, param.DataDes.dataType, param.stream, param.reduceType,
        planeRoot, std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));

    CHK_RET(level1TempAlg->RegisterProfiler(
        (ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
        param.stream));

    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));

    HCCL_INFO("reduce 8PringHD stage1 run success");

    // step3: 节点内的gathering，只有在root所在server内进行gather操作
    SingleSubCommTransport& level0TransportInfo
        = const_cast<SingleSubCommTransport&>(algResResp_->opTransportResponse[COMM_LEVEL0][COMM_INDEX_0]);

    if (level0TransportInfo.userRank2subCommRank.find(param.root) != level0TransportInfo.userRank2subCommRank.end()) {
        CHK_RET(MultiRingGather(
            tag_, execMem.outputMem, execMem.outputMem, hdCount, param.DataDes.dataType, mulRingSlice, param.reduceType,
            param.root, const_cast<Stream&>(param.stream), PROF_STAGE_2));
    }
    HCCL_INFO("reduce 8PringHD stage2 run success");

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceRingPlusHd", ReduceRingPlusHd, CollReduceRingPlusHdExecutor);

} // namespace hccl
