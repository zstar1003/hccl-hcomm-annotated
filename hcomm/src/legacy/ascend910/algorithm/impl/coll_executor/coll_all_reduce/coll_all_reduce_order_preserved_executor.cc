/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_order_preserved_executor.h"

namespace hccl {

CollAllReduceOrderPreservedExecutor::CollAllReduceOrderPreservedExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
    CCLMemSlice_ = false;
}

void CollAllReduceOrderPreservedExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    u64 sizePerBlock = (param.DataDes.count + topoAttr_.userRankSize - 1) / topoAttr_.userRankSize
                       * SIZE_TABLE[param.DataDes.dataType];
    sizePerBlock = AlgTemplateBase::RoundUpWithDivisor(sizePerBlock, HCCL_MIN_SLICE_ALIGN);

    // 是否需要scratch memory（图模式没有cclbuffer，需要额外申请scratchMem）
    u64 inputSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    scratchMemFlag_ = (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
                      && (inputSize < (topoAttr_.userRankSize - 1) * sizePerBlock);

    totalSize_ = std::max(sizePerBlock * topoAttr_.userRankSize, inputSize);
}

HcclResult CollAllReduceOrderPreservedExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    scratchMemSize = scratchMemFlag_ ? totalSize_ : 0U;
    HCCL_INFO("[%s]tag[%s] scratchMemSize[%llu]", __func__, tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

u32 CollAllReduceOrderPreservedExecutor::CalReduceStreamNum(const u32& localRankSize)
{
    return (1 << static_cast<int>(std::floor(log2(localRankSize))));
}

HcclResult CollAllReduceOrderPreservedExecutor::CalcStreamNum(u32& streamNum)
{
    if (topoAttr_.deviceNumPerAggregation == 1) {
        u32 level1StreamNum = CalReduceStreamNum(topoAttr_.moduleNum);
        streamNum = std::min(level1StreamNum, DEVICE_EIGHT + DEVICE_EIGHT / FACTOR_NUM_TWO - 1);
        HCCL_INFO(
            "[%s]tag[%s] single rank per module, level1StreamNum[%u], streamNum[%u]", __func__, tag_.c_str(),
            level1StreamNum, streamNum);
        return HCCL_SUCCESS;
    }

    // Level0RankSize条流给alltoall，剩下的流给LocalReduce使用
    u32 level0StreamNum = topoAttr_.deviceNumPerAggregation - 1 + CalReduceStreamNum(topoAttr_.deviceNumPerAggregation);
    // level1主流分给alltoall，从流给LocalReduce使用
    u32 level1StreamNum = CalReduceStreamNum(topoAttr_.moduleNum);
    // 总流数上限：7（alltoall使用，提前的本地拷贝任务不需要并行）+ 4（LocalReduce使用）
    streamNum
        = std::min(std::max(level0StreamNum - 1, level1StreamNum), DEVICE_EIGHT + DEVICE_EIGHT / FACTOR_NUM_TWO - 1);

    HCCL_INFO(
        "[%s]tag[%s] level0StreamNum[%u], level1StreamNum[%u], streamNum[%u]", __func__, tag_.c_str(), level0StreamNum,
        level1StreamNum, streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedExecutor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    // 图模式场景使用PARAM_INPUT/OUTPUT -> userInput/userOutPut，不需要scratchMem
    inputType = workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ? TransportMemType::PARAM_INPUT :
                                                                                TransportMemType::CCL_INPUT;
    outputType = workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ? TransportMemType::PARAM_OUTPUT :
                                                                                 TransportMemType::CCL_OUTPUT;

    if (scratchMemFlag_) {
        outputType = TransportMemType::SCRATCH;
    }

    HCCL_INFO("[%s]tag[%s] inputType[%d], outputType[%d]", __func__, tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedExecutor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    if (topoAttr_.moduleNum > 1) {
        CommParaInfo commParaLevel1(COMM_LEVEL1, CommType::COMM_TAG_MESH);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[COMM_LEVEL1], inputType, outputType));
    }
    return HCCL_SUCCESS;
}

bool CollAllReduceOrderPreservedExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize / topoAttr_.deviceNumPerAggregation / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE
                    || curSize > SDMA_SEND_MAX_SIZE;
    HCCL_DEBUG(
        "[%s]isHugeData[%d], curSize[%llu], topoAttr_.deviceNumPerAggregation[%u]", __func__, hugeData, curSize,
        topoAttr_.deviceNumPerAggregation);
    return hugeData;
}

void CollAllReduceOrderPreservedExecutor::CalcSizePerBlock(const OpParam& param, ExecMem& execMem)
{
    sizePerBlock_
        = (execMem.count + topoAttr_.userRankSize - 1) / topoAttr_.userRankSize * SIZE_TABLE[param.DataDes.dataType];
    sizePerBlock_ = AlgTemplateBase::RoundUpWithDivisor(sizePerBlock_, HCCL_MIN_SLICE_ALIGN);
}

void CollAllReduceOrderPreservedExecutor::CalGroupSlices(const OpParam& param, const ExecMem& execMem)
{
    groupSize_.clear();
    u64 sizeRemain = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    for (u32 rankId = 0; rankId < topoAttr_.userRankSize; rankId++) {
        u64 size = (sizeRemain > sizePerBlock_) ? sizePerBlock_ : sizeRemain;
        groupSize_.push_back(size);
        sizeRemain -= size;
    }
}

HcclResult CollAllReduceOrderPreservedExecutor::RunReduceScatterLevel0(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo)
{
    if (level0CommInfo.localRankSize == 1) {
        all2allOffset_ = topoAttr_.moduleNum > 1 ? 1 : 0;
        HCCL_INFO("[%s] single rank per module, skip L0 AllToAll and LocalReduce, tag[%s]", __func__, tag_.c_str());
        CHK_RET(RunReduceScatterLevel0SingleRank(param, execMem, level0CommInfo));
        return HCCL_SUCCESS;
    }

    // 切分数据(ReduceScatter分组，记录每组的起始偏移和大小)
    GroupSlicesInfo groupSlicesInfoLevel0;
    for (u32 groupId = 0; groupId < topoAttr_.moduleNum; groupId++) {
        MemBlockInfo memInfo;
        for (u32 dataId = 0; dataId < level0CommInfo.localRankSize; dataId++) {
            u64 globalDataId = groupId * level0CommInfo.localRankSize + dataId;
            u64 size = groupSize_[globalDataId];
            u64 offset = globalDataId * sizePerBlock_;
            memInfo.size.push_back(size);
            memInfo.userInputOffsets.push_back(offset);
            memInfo.inputOffsets.push_back(offset);
            memInfo.outputOffsets.push_back(offset);
        }
        groupSlicesInfoLevel0.push_back(memInfo);
    }

    CHK_RET(ActiveSlaveStreams(param.stream));
    all2allOffset_ = topoAttr_.moduleNum > 1 ? 1 : 0; // 多机场景需要偏移1（给L1预留计算位，减少拷贝次数）

    std::unique_ptr<AlgTemplateBase> level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE, dispatcher_);
    CHK_SMART_PTR_NULL(level0TempAlg);

    DeviceMem outputMem = scratchMemFlag_ ? execMem.scratchMem : execMem.outputMem;
    CHK_RET(level0TempAlg->Prepare(
        execMem.inputPtr, execMem.inputMem, outputMem, param.stream, algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, groupSlicesInfoLevel0, param.reduceType, all2allOffset_,
        param.DataDes.dataType, true));

    CHK_RET(level0TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedExecutor::RunReduceScatterLevel0SingleRank(
    const OpParam& param, ExecMem& execMem, const SubCommInfo& level0CommInfo) const
{
    (void)level0CommInfo;
    u64 size = execMem.count * SIZE_TABLE[param.DataDes.dataType];

    DeviceMem srcMem = DeviceMem::create(execMem.inputPtr, size);
    DeviceMem dstMem = scratchMemFlag_ ? execMem.scratchMem.range(0, size) : execMem.inputMem.range(0, size);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));

    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedExecutor::RunReduceScatterLevel1(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo)
{
    u32 commIndex = level0CommInfo.localRank;
    u32 level0Ranksize = level0CommInfo.localRankSize;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    // 切分数据，记录每组的起始偏移和大小（仅1组）
    MemBlockInfo memInfo;
    u32 inputBaseIndex
        = (all2allOffset_ + commIndex) % level0Ranksize; // 多机场景需要偏移1（给L1预留计算位，减少拷贝次数）
    for (u32 dataId = 0; dataId < level1CommInfo.localRankSize; dataId++) {
        u64 inputIndex = inputBaseIndex + dataId * level0Ranksize;
        memInfo.inputOffsets.push_back(inputIndex * sizePerBlock_);
        memInfo.size.push_back(groupSize_[commIndex + dataId * level0Ranksize]);
        u64 outputIndex = commIndex + dataId * level0Ranksize;
        memInfo.outputOffsets.push_back(outputIndex * sizePerBlock_);
        memInfo.userInputOffsets.push_back(outputIndex * sizePerBlock_);
    }

    DeviceMem outputMem = scratchMemFlag_ ? execMem.scratchMem : execMem.outputMem;
    std::unique_ptr<AlgTemplateBase> level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE_COMBINE, dispatcher_);
    CHK_SMART_PTR_NULL(level1TempAlg);

    u32 level0LastRank = level0Ranksize - 1;
    bool isUseCclIn = (level0Ranksize == 1) || (commIndex == level0LastRank - 1);
    CHK_RET(level1TempAlg->Prepare(
        execMem.inputMem, outputMem, param.stream, algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux, memInfo, param.reduceType, param.DataDes.dataType, isUseCclIn,
        commIndex == level0LastRank, true));

    CHK_RET(level1TempAlg->RegisterProfiler(
        (level0Ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedExecutor::RunAllGatherLevel0(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo)
{
    u32 level0RankSize = level0CommInfo.localRankSize;
    u32 commIndex = level0CommInfo.localRank;
    u64 count = execMem.count / topoAttr_.userRankSize;
    u64 serverOffsetConut = topoAttr_.userRank / level0RankSize * level0RankSize;

    // allgather 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> dataSegsSlice;
    for (u32 rank = 0; rank < level0RankSize; rank++) {
        Slice userslice;
        userslice.size = groupSize_[rank + serverOffsetConut];
        userslice.offset = userslice.size == 0 ? 0 : (rank + serverOffsetConut) * sizePerBlock_;
        dataSegsSlice.emplace_back(std::move(userslice));
    }

    DeviceMem outputMem = scratchMemFlag_ ? execMem.scratchMem : execMem.outputMem;

    std::unique_ptr<AlgTemplateBase> level0TempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_MESH_ATOMIC, dispatcher_);

    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank, nullptr,
        commIndex, level0RankSize));
    CHK_RET(level0TempAlg->Prepare(
        outputMem, outputMem, outputMem, count, param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED,
        LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0));

    CHK_RET(level0TempAlg->RegisterProfiler(
        (level0RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commIndex, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
        param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedExecutor::RunAllGatherLevel1(
    const OpParam& param, ExecMem& execMem, const SubCommInfo& level0CommInfo)
{
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    std::unique_ptr<AlgTemplateBase> level1TempAlg; // Level1Allgather(根据算法选择)
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
        HCCL_INFO("AllGather mesh: using ring algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
        HCCL_INFO("AllGather mesh: using nonuniform-bruck algo inter-server.");
    } else {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        HCCL_INFO("AllGather mesh: using nhr algo inter-server.");
    }

    DeviceMem outputMem = scratchMemFlag_ ? execMem.scratchMem : execMem.outputMem;

    u32 level0RankSize = level0CommInfo.localRankSize;
    u64 count = execMem.count / level1CommInfo.localRankSize;

    std::vector<u64> level1GroupSize;
    for (u32 rank = 0; rank < level1CommInfo.localRankSize; rank++) {
        u64 size = 0;
        for (u32 level0RankId = 0; level0RankId < level0RankSize; level0RankId++) {
            size += groupSize_[rank * level0RankSize + level0RankId];
        }
        level1GroupSize.push_back(size);
    }

    // allgather 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> dataSegsSlice;
    for (u32 rank = 0; rank < level1CommInfo.localRankSize; rank++) {
        Slice userslice;
        userslice.size = level1GroupSize[rank];
        userslice.offset = rank * level0RankSize * sizePerBlock_;
        dataSegsSlice.emplace_back(std::move(userslice));
    }

    CHK_SMART_PTR_NULL(level1TempAlg);
    CHK_RET(level1TempAlg->Prepare(
        outputMem, outputMem, outputMem, count, param.DataDes.dataType, param.stream,
        HcclReduceOp::HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, dataSegsSlice));

    CHK_RET(level1TempAlg->RegisterProfiler(
        (level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s]The CollAllReduceOrderPreservedExecutor starts, tag[%s]", __func__, tag_.c_str());
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    CalcSizePerBlock(param, execMem);
    CalGroupSlices(param, execMem);

    u64 inputSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    scratchMemFlag_ = (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
                      && (inputSize < (topoAttr_.userRankSize - 1) * sizePerBlock_);

    // L0 节点内 reduce scatter
    CHK_RET(RunReduceScatterLevel0(param, execMem, level0CommInfo));
    // L1 节点间 reduce scatter
    if (topoAttr_.moduleNum > 1) {
        CHK_RET(RunReduceScatterLevel1(param, execMem, level0CommInfo));
    }

    // Level0 节点内 AllGatherMeshAtomic
    CHK_RET(RunAllGatherLevel0(param, execMem, level0CommInfo));
    if (topoAttr_.moduleNum > 1) {
        // L1 节点间 allgather
        CHK_RET(RunAllGatherLevel1(param, execMem, level0CommInfo));
    }

    // 单算子需要 execMem.outputMem最后拷贝至UserOut
    if (scratchMemFlag_ || workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u64 dataSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        void* srcPtr = scratchMemFlag_ ? execMem.scratchMem.ptr() : execMem.outputMem.ptr();
        DeviceMem srcMem = DeviceMem::create(srcPtr, dataSize);
        DeviceMem dstMem = DeviceMem::create(execMem.outputPtr, dataSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    HCCL_INFO("[%s]order preserved AllReduce run success, tag[%s]", __func__, tag_.c_str());
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceOrderPreservedExecutor", AllReduceOrderPreserved, CollAllReduceOrderPreservedExecutor);
} // namespace hccl
