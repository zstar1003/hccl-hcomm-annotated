/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_ring_executor.h"

namespace hccl {

CollBroadcastRingExecutor::CollBroadcastRingExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        DMAReduceFlag_ = true;
    } else {
        DMAReduceFlag_ = false;
    }
}

HcclResult CollBroadcastRingExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum
        = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? LEVEL0_PLANE_NUM_IN_8PRING : LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;

    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            totalStreamNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
        } else {
            totalStreamNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
        }
    }

    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollBroadcastRingExecutor][CalcStreamNum] tag[%s] streamNum_[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastRingExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastRingExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollBroadcastRingExecutor][CalcLevel0CommInfo]tag[%s] start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollBroadcastRingExecutor][CalcLevel0CommInfo]tag[%s] Calc RingComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastRingExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollBroadcastRingExecutor][KernelRun]The CollBroadcastRingExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    std::vector<Slice> dataSegsSlice;                   // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    // step1: 节点内的scatter
    u32 ringNum
        = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? LEVEL0_PLANE_NUM_IN_8PRING : LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;

    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    // 按ranksize得到内存切分slice数
    u32 sliceNum = level0CommInfo.localRankSize;
    // 将根节点数据切分成sliceNum份
    CHK_RET(AlgTemplateBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));
    HCCL_DEBUG(
        "[CollBroadcastRingExecutor][KernelRun] execMem.count[%llu], perDataSize[%u], sliceNum[%u], ringNum[%u] ",
        execMem.count, perDataSize, sliceNum, ringNum);

    /* 外层:scatter */
    // 将每slice再切分成4份，按各ring的dev顺序排列
    if (ringNum == LEVEL0_PLANE_NUM_IN_8PRING) {
        // 构造ring algorithm对应的scatter实例
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
        CHK_PRT_RET(
            multRingsSliceZero.size() != ringNum,
            HCCL_ERROR(
                "[CollBroadcastRingExecutor]"
                "ringNum[%u] !=multRingsSliceZero size[%zu]",
                ringNum, multRingsSliceZero.size()),
            HCCL_E_INTERNAL);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice); // 应该offset全为0，而大小和dataSegsSlice中一样,里面的offset不使用
    }

    HcomCollOpInfo* scatterOpInfoPtr = nullptr;
    HcomCollOpInfo scatterOpInfo
        = {"", execMem.inputPtr, nullptr, execMem.count, param.DataDes.dataType, param.root, param.reduceType, 0};
    if (DMAReduceFlag_) {
        scatterOpInfoPtr = &scatterOpInfo;
    }
    CHK_RET(MultiRingScatter(
        param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, multRingsSliceZero,
        param.root, param.stream, scatterOpInfoPtr));

    HCCL_INFO("broadcast 8PringHD stage0 run success");

    // step2: 节点间的broadcast
    u64 hdSize;
    u32 segmentIdx;
    u32 commIndex;
    CHK_RET(PrepareLevel1CommInfo(segmentIdx, commIndex, hdSize, level0CommInfo, multRingsSliceZero, param.tag));

    u64 hdCount = hdSize / perDataSize;
    auto nicList = topoAttr_.nicList;
    bool isMultiNic = topoType_ == TopoType::TOPO_TYPE_8P_RING && nicList.size() != DEVICE_EIGHT;
    std::vector<u32>::iterator iterNic = std::find(nicList.begin(), nicList.end(), topoAttr_.devicePhyId);
    bool innRunRet = isMultiNic && (iterNic == nicList.end());
    if (!innRunRet) { // 满足以下条件, 不做server间通信: 1. 8P ring的拓扑 2. 网口不满配 3. 当前device不出网口
        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
        u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        bool isUsedRegister = false;
        std::unique_ptr<AlgTemplateBase> level1TempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            HCCL_DEBUG(
                "broadcast ring: curSize[%llu] deviceNumPerAggregation[%u] commLevel0Size[%u].", curSize,
                topoAttr_.deviceNumPerAggregation, level0CommInfo.localRankSize);
            if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_BCAST_SMALL_SIZE) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_BROADCAST_NHR_ONESHOT, dispatcher_);
            } else {
                level1TempAlg
                    = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR, dispatcher_);
            }
            HCCL_INFO("broadcast ring: using nhr algo inter-server");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            isUsedRegister = true;
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR_V1, dispatcher_);
            HCCL_INFO("broadcast ring: using nhr_v1 algo inter-server");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            const u32 level1RankSize = level1CommInfo.localRankSize;
            if (ShouldUseBinaryBroadcastOfNB(
                    curSize / topoAttr_.deviceNumPerAggregation, level1RankSize, topoAttr_.userRankSize,
                    topoAttr_.deviceNumPerAggregation)) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_BROADCAST_NB_BINARY, dispatcher_);
            } else {
                level1TempAlg
                    = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NB, dispatcher_);
            }
            HCCL_INFO("broadcast ring: using nonuniform-bruck algo inter-server");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_BROADCAST_RECURSIVE_HD, dispatcher_);
            HCCL_INFO("broadcast ring: using Recursive halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);

        u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
        CHK_PRT_RET(
            subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR(
                "[BroadCastOperator][BroadCastRingExecutor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
                subUserrankRoot, topoAttr_.userRank, param.root),
            HCCL_E_INTERNAL);
        u32 planeRoot = 0;
        u32 level1RankSize = level1CommInfo.localRankSize;
        u32 level1LocalRank = level1CommInfo.localRank;
        CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, planeRoot));

        // 节点间的hd 使用环0来记录
        if (isUsedRegister) {
            PrepareData prepareData;
            prepareData.inputMem = execMem.inputMem;
            prepareData.outputMem = execMem.inputMem;
            prepareData.scratchMem = execMem.outputMem;
            prepareData.count = hdCount;
            prepareData.dataType = param.DataDes.dataType;
            prepareData.stream = param.stream;
            prepareData.reductionOp = HCCL_REDUCE_RESERVED;
            prepareData.root = planeRoot;
            prepareData.baseOffset = dataSegsSlice[segmentIdx].offset;
            CHK_RET(level1TempAlg->Prepare(prepareData));
        } else {
            CHK_RET(level1TempAlg->Prepare(
                execMem.inputMem, execMem.inputMem, execMem.outputMem, hdCount, param.DataDes.dataType, param.stream,
                HCCL_REDUCE_RESERVED, planeRoot, std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));
        }

        CHK_RET(level1TempAlg->RegisterProfiler(
            (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1LocalRank, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
            param.stream));

        CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    }
    HCCL_INFO("broadcast 8PringHD stage1 run success");

    // step3: 节点内的allgatherring
    HcomCollOpInfo* allgatherOpInfoPtr = nullptr;
    HcomCollOpInfo allgatherOpInfo
        = {"", nullptr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, HCCL_REDUCE_RESERVED, 0};
    if (DMAReduceFlag_) {
        allgatherOpInfoPtr = &allgatherOpInfo;
    }
    CHK_RET(MultiRingAllGather(
        param.tag, execMem.inputMem, execMem.outputMem, hdCount, param.DataDes.dataType, multRingsSliceZero,
        param.stream, PROF_STAGE_2, 0, allgatherOpInfoPtr));

    HCCL_INFO("broadcast 8PringHD stage2 run success");
    return HCCL_SUCCESS;
}
HcclResult CollBroadcastRingExecutor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
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

HcclResult CollBroadcastRingExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR, dispatcher_);
            HCCL_INFO("broadcast ring: using nhr algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR_V1, dispatcher_);
            HCCL_INFO("broadcast ring: using nhr_v1 algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NB, dispatcher_);
            HCCL_INFO("broadcast ring: using nonuniform-bruck algo inter-server.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_BROADCAST_RECURSIVE_HD, dispatcher_);
            HCCL_INFO("broadcast ring: using Recursive halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}

REGISTER_EXEC("BroadCastRingExecutor", BroadcastRing, CollBroadcastRingExecutor);

} // namespace hccl
