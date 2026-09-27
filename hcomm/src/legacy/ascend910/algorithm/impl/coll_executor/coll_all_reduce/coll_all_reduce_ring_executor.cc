/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_ring_executor.h"

namespace hccl {

CollAllReduceRingExecutor::CollAllReduceRingExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        DMAReduceFlag_ = true;
    } else {
        DMAReduceFlag_ = false;
    }
}

HcclResult CollAllReduceRingExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 1U;
    if (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_8P_RING) {
        totalStreamNum = LEVEL0_PLANE_NUM_IN_8PRING;
    } else if (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING) {
        if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                totalStreamNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
            } else {
                totalStreamNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
            }
        }
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollAllReduceRingExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceRingExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceRingExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollAllReduceRingExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d].", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceRingExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollAllReduceRingExecutor][CalcLevel0CommInfo]tag[%s] start.", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollAllReduceRingExecutor][CalcLevel0CommInfo]tag[%s] Calc RingComm finish.", tag_.c_str());
    return HCCL_SUCCESS;
}

bool CollAllReduceRingExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize / topoAttr_.deviceNumPerAggregation / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE
                    || curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

bool CollAllReduceRingExecutor::IsSmallData([[maybe_unused]] const u64 totalSize, const u64 curSize)
{
    bool smallData = DMAReduceFlag_ ? false : IsAllReduceSmallData(curSize);
    return smallData;
}

HcclResult CollAllReduceRingExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllReduceRingExecutor][Run]The CollAllReduceRingExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    std::vector<Slice> dataSegsSlice;                   // 数据分成ranksize份，每份的起始偏移和大小
    u32 ringNum
        = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? LEVEL0_PLANE_NUM_IN_8PRING : LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;

    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 sliceNum = level0CommInfo.localRankSize;
    // 根据数据量计算每个环上数据的偏移和大小
    CHK_RET(AlgTemplateBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));

    /* 三步算法step1：外层 - 节点内 reduce-scatter */
    /* 外层:reduce_scatter */
    if (ringNum == LEVEL0_PLANE_NUM_IN_8PRING) {
        // 构造ring algorithm对应的reduce-scatter实例
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
        CHK_PRT_RET(
            multRingsSliceZero.size() != ringNum,
            HCCL_ERROR(
                "[CollAllReduceRingExecutor]"
                "ringNum[%u] != multRingsSliceZero size[%zu]",
                ringNum, multRingsSliceZero.size()),
            HCCL_E_INTERNAL);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice); // 应该offset全为0，而大小和dataSegsSlice中一样,里面的offset不使用
    }

    HcomCollOpInfo* reduceScatterOpInfoPtr = nullptr;
    // 第一步的reducescatter输出放在CCL buffer上，通过设置nullptr指示不做最后一步的DMA削减动作
    HcomCollOpInfo reduceScatterOpInfo
        = {"", execMem.inputPtr, nullptr, execMem.count, param.DataDes.dataType, param.root, param.reduceType, 0};
    if (DMAReduceFlag_) {
        reduceScatterOpInfoPtr = &reduceScatterOpInfo;
    }
    CHK_RET(MultiRingReduceScatter(
        param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.reduceType,
        multRingsSliceZero, param.stream, PROF_STAGE_0, 0, reduceScatterOpInfoPtr));

    HCCL_INFO("AllReduce ringhd stage0 run success");

    /* 三步算法step2: 内层 - 节点间 allreduce */
    u64 hdSize;
    u32 segmentIdx;
    u32 commIndex;
    CHK_RET(PrepareLevel1CommInfo(segmentIdx, commIndex, hdSize, level0CommInfo, multRingsSliceZero, param.tag));

    u64 hdCount = hdSize / perDataSize;
    auto nicList = topoAttr_.nicList;
    bool isMultiNic = topoType_ == TopoType::TOPO_TYPE_8P_RING && nicList.size() != DEVICE_EIGHT;
    std::vector<u32>::iterator iterNic = std::find(nicList.begin(), nicList.end(), topoAttr_.devicePhyId);
    bool innRunRet = isMultiNic && (iterNic == nicList.end());
    HCCL_DEBUG("[CollAllReduceRingExecutor][KernelRun]isMultiNic is %d, innRunRet is %d", isMultiNic, innRunRet);
    if (!innRunRet) { // 满足以下条件, 不做server间通信: 1. 8P ring的拓扑 2. 网口不满配 3. 当前device不出网口
        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

        DeviceMem allreduceInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
        CHK_SMART_PTR_NULL(allreduceInput);
        DeviceMem allreduceOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
        CHK_SMART_PTR_NULL(allreduceOutput);

        u64 reduceAttr = GetReduceAttr(allreduceInput, allreduceOutput, param.DataDes.dataType, param.reduceType);

        std::unique_ptr<AlgTemplateBase> level1TempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
            HCCL_INFO("AllReduce ring: using ring algo inter-server.");
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType]; // 单位 byte
            HCCL_DEBUG(
                "AllReduce ring: curSize[%llu] deviceNumPerAggregation[%u] commLevel0Size[%u]", curSize,
                topoAttr_.deviceNumPerAggregation, level0CommInfo.localRankSize);
            if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_ALLREDUCE_SMALL_SIZE) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_REDUCE_NHR_ONESHOT, dispatcher_);
            } else {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
            }
            HCCL_INFO("AllReduce ring: using nhr algo inter-server.");
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
            level1TempAlg->CloseBarrier();
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR_V1, dispatcher_);
            HCCL_INFO("AllReduce ring: using nhr_v1 algo inter-server.");
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NB, dispatcher_);
            HCCL_INFO("AllReduce ring: using nonuniform-bruck algo inter-server.");
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING, dispatcher_);
            HCCL_INFO("AllReduce ring: using Recursive halving-doubling algo inter-server.");
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
        }

        CHK_SMART_PTR_NULL(level1TempAlg);

        // 节点间的hd 使用环0来记录
        CHK_RET(level1TempAlg->Prepare(
            allreduceInput, allreduceOutput, allreduceOutput, hdCount, param.DataDes.dataType, param.stream,
            param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));

        CHK_RET(level1TempAlg->RegisterProfiler(
            (level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_1,
            HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    }
    HCCL_INFO("AllReduce ringhd stage1 run success");

    /* 三步算法step3：外层 - 节点内 allgather */
    HcomCollOpInfo* allgatherOpInfoPtr = nullptr;
    // 第三步的allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
    HcomCollOpInfo allgatherOpInfo
        = {"", nullptr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType, 0};
    if (DMAReduceFlag_) {
        allgatherOpInfoPtr = &allgatherOpInfo;
    }
    CHK_RET(MultiRingAllGather(
        param.tag, execMem.inputMem, execMem.outputMem, hdCount, param.DataDes.dataType, multRingsSliceZero,
        param.stream, PROF_STAGE_2, 0, allgatherOpInfoPtr));
    HCCL_INFO("AllReduce ringhd stage2 run success");
    return HCCL_SUCCESS;
}
HcclResult CollAllReduceRingExecutor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
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

HcclResult CollAllReduceRingExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
            HCCL_INFO("AllReduce ring: using ring algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
            HCCL_INFO("AllReduce ring: using nhr algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR_V1, dispatcher_);
            HCCL_INFO("AllReduce ring: using nhr_v1 algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
            // 获取通信域分组信息
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_AHC, dispatcher_);
            HCCL_INFO("AllReduce ring: using ahc algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_REDUCE_AHC_BROKE, dispatcher_);
            HCCL_INFO("AllReduce ring: using ahc-broke algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NB, dispatcher_);
            HCCL_INFO("AllReduce ring: using nonuniform-bruck algo inter-server.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING, dispatcher_);
            HCCL_INFO("AllReduce ring: using Recursive halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}
REGISTER_EXEC("AllReduceRingExecutor", AllReduceRing, CollAllReduceRingExecutor);

} // namespace hccl
