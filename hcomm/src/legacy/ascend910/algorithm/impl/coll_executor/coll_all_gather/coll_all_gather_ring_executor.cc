/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_ring_executor.h"

namespace hccl {
CollAllGatherRingExecutor::CollAllGatherRingExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollAllGatherRingExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 1U;
    if (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_8P_RING) {
        totalStreamNum = LEVEL0_PLANE_NUM_IN_8PRING;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollAllGatherRingExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollAllGatherRingExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollAllGatherRingExecutor][CalcLevel0CommInfo]tag[%s] start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollAllGatherRingExecutor][CalcLevel0CommInfo]tag[%s] Calc RingComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

u64 CollAllGatherRingExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

HcclResult CollAllGatherRingExecutor::SelectAlgorithmTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg)
{
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
        HCCL_INFO("AllGather ring: using ring algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        HCCL_INFO("AllGather ring: using nhr algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHRV1, dispatcher_);
        HCCL_INFO("AllGather ring: using nhr_v1 algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
        HCCL_INFO("AllGather ring: using nonuniform-bruck algo inter-server.");
    } else {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING, dispatcher_);
        HCCL_INFO("AllGather ring: using halving-doubling algo inter-server.");
    }
    CHK_SMART_PTR_NULL(level1TempAlg);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherRingExecutor][KernelRun]The AllGatherRingExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    CHK_PRT_RET(
        perDataSize == 0,
        HCCL_ERROR(
            "[CollAllGatherRingExecutor][KernelRun]errNo[0x%016llx] datatype[%d] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), param.DataDes.dataType),
        HCCL_E_PARA);

    // 获取子通信域的信息
    u32 ringNum
        = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? LEVEL0_PLANE_NUM_IN_8PRING : LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;

    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = (ringNum == LEVEL0_PLANE_NUM_IN_8PRING) ? topoAttr_.devicePhyId : level0CommInfo.localRank;
    u32 level0RankSize = level0CommInfo.localRankSize;

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    u32 serverIndex = level1CommInfo.localRank;

    //  第一步，如果非DMA消减，将数据从input内存拷贝到output内存的对应位置
    u64 inputMemSize = execMem.inputMem.size();
    u64 baseOffset = serverIndex * inputMemSize * level0RankSize;
    u64 level0Offset = commIndex * inputMemSize;
    DeviceMem dstMem = execMem.outputMem.range(baseOffset + level0Offset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);

    HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAllGatherRingExecutor][KernelRun]AllGather 8PringHD memcpy Failed, "
            "Offset[%llu], Size[%llu]",
            baseOffset + level0Offset, inputMemSize),
        ret);

    // 第二步，各个AI Server 内 multi ring AllGather
    std::vector<Slice> dataSegsSlice;                   // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    u32 sliceNum = level0RankSize;
    CHK_RET(PrepareAllgatherSlice(sliceNum, inputMemSize, dataSegsSlice));

    //  多环数据切分
    if (ringNum == LEVEL0_PLANE_NUM_IN_8PRING) {
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice);
    }
    CHK_PRT_RET(
        multRingsSliceZero.size() != ringNum,
        HCCL_ERROR(
            "[CollAllGatherRingExecutor][KernelRun]ringNum[%u] != multRingsSliceZero size[%zu]", ringNum,
            multRingsSliceZero.size()),
        HCCL_E_INTERNAL);

    //  抽取当前用于多环AllGather 的output内存数据
    DeviceMem currentOutputMem = execMem.outputMem.range(baseOffset, inputMemSize * level0RankSize);
    CHK_SMART_PTR_NULL(currentOutputMem);

    CHK_RET(ActiveSlaveStreams(param.stream));

    CHK_RET(MultiRingAllGather(
        param.tag, execMem.inputMem, currentOutputMem, execMem.count, param.DataDes.dataType, multRingsSliceZero,
        param.stream, PROF_STAGE_1, baseOffset, nullptr));

    HCCL_INFO("AllGather 8PringHD level0 run success");

    //  第三步， AI server 间 recursive halving doubling AllGather
    u64 hdSize = 0;
    std::vector<u32> nicList = const_cast<std::vector<u32>&>(topoAttr_.nicList);
    std::vector<u32>::iterator iterNic = std::find(nicList.begin(), nicList.end(), topoAttr_.devicePhyId);
    if (iterNic != nicList.end()) {
        hdSize = inputMemSize * level0RankSize;
    }
    u64 hdCount = hdSize / perDataSize;

    bool isMultiNic = topoType_ == TopoType::TOPO_TYPE_8P_RING && nicList.size() != DEVICE_EIGHT;
    bool innRunRet = isMultiNic && (iterNic == nicList.end());
    if (!innRunRet) { // 满足以下条件, 不做server间通信: 1. 8P ring的拓扑 2. 网口不满配 3. 当前device不出网口
        std::unique_ptr<AlgTemplateBase> level1TempAlg;
        CHK_RET(SelectAlgorithmTempAlg(level1TempAlg));

        //  此处虽然带入inputMem作为scratch mem, 但inputMem 不能被使用
        CHK_RET(level1TempAlg->Prepare(
            execMem.outputMem, execMem.outputMem, execMem.inputMem, hdCount, param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, std::vector<Slice>(COMM_INDEX_0), 0));

        u32 rankSize = level1CommInfo.localRankSize;
        CHK_RET(level1TempAlg->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + serverIndex, PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET,
            param.stream));

        CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    }
    HCCL_INFO("AllGather 8PringHD level1 run success");

    //  网口裁剪：AI server 内多网口的allgather
    if (topoType_ == TopoType::TOPO_TYPE_8P_RING && nicList.size() != DEVICE_EIGHT) {
        CHK_RET(ActiveSlaveStreams(param.stream)); // 为什么要active两遍

        u32 perDataSize = 0;
        CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
        u64 tempCount = execMem.outputMem.size() / perDataSize;
        CHK_RET(AlgTemplateBase::PrepareSliceData(tempCount, perDataSize, sliceNum, 0, dataSegsSlice));
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, nicList);
        CHK_PRT_RET(
            multRingsSliceZero.size() != ringNum,
            HCCL_ERROR(
                "[CollAllGatherRingExecutor][KernelRun]"
                "ringNum[%u] != multRingsSliceZero size[%zu]",
                ringNum, multRingsSliceZero.size()),
            HCCL_E_INTERNAL);

        CHK_RET(MultiRingAllGather(
            param.tag, execMem.outputMem, execMem.outputMem, tempCount / DEVICE_EIGHT, param.DataDes.dataType,
            multRingsSliceZero, param.stream, PROF_STAGE_1));

        HCCL_INFO("AllGather 8PringHD level1 chunk run success");
    }
    return HCCL_SUCCESS;
}
HcclResult CollAllGatherRingExecutor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
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

HcclResult CollAllGatherRingExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        CHK_RET(SelectAlgorithmTempAlg(level1TempAlg));
    }
    return HCCL_E_UNAVAIL;
}

REGISTER_EXEC("AllGatherRingExecutor", AllGatherRing, CollAllGatherRingExecutor);

} // namespace hccl
