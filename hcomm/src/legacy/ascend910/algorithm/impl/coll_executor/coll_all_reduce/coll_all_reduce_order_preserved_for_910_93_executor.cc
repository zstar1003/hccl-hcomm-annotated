/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_order_preserved_for_910_93_executor.h"

namespace hccl {

CollAllReduceOrderPreservedFor91093Executor::CollAllReduceOrderPreservedFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
    CCLMemSlice_ = false;
    desc_.deterministic = DETERMINISTIC_STRICT;
}

void CollAllReduceOrderPreservedFor91093Executor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;

    u64 sizePerBlock = (param.DataDes.count + topoAttr_.userRankSize - 1) / topoAttr_.userRankSize
                       * SIZE_TABLE[param.DataDes.dataType];
    sizePerBlock = AlgTemplateBase::RoundUpWithDivisor(sizePerBlock, HCCL_MIN_SLICE_ALIGN);

    // 是否需要scratch memory（图模式没有cclbuffer，需要额外申请scratchMem）
    u64 inputSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    scratchMemFlag_ = (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
                      && (inputSize < (topoAttr_.userRankSize - 1) * sizePerBlock);

    totalSize_ = std::max(sizePerBlock * topoAttr_.userRankSize, inputSize);
}

HcclResult CollAllReduceOrderPreservedFor91093Executor::CalcScratchMemSize(u64& scratchMemSize)
{
    scratchMemSize = scratchMemFlag_ ? totalSize_ : 0U;
    HCCL_INFO("[%s]tag[%s] scratchMemSize[%llu]", __func__, tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

u32 CollAllReduceOrderPreservedFor91093Executor::CalReduceStreamNum(const u32& localRankSize) const
{
    return (1 << static_cast<int>(std::floor(log2(localRankSize))));
}

HcclResult CollAllReduceOrderPreservedFor91093Executor::CalcStreamNum(u32& streamNum)
{
    // 获取超节点内rank数
    u32 devNumInlocalPod = 0;
    u32 rankIdxInPod = 0;
    CHK_RET(topoMatcher_->GetLocalSuperPodRankSize(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));

    if (devNumInlocalPod == 1) {
        u32 level2StreamNum = std::min(CalReduceStreamNum(topoAttr_.superPodNum) - 1, DEVICE_FOUR);
        streamNum = level2StreamNum;
        HCCL_INFO(
            "[%s]tag[%s] single rank per module, level2StreamNum[%u], streamNum[%u]", __func__, tag_.c_str(),
            level2StreamNum, streamNum);
        return HCCL_SUCCESS;
    }

    // all2allStreamNum条流给alltoall
    u32 all2allStreamNum = std::min(devNumInlocalPod, DEVICE_EIGHT);
    // reduceStreamNum主流分给alltoall，从流给LocalReduce使用
    u32 reduceStreamNum = std::min(CalReduceStreamNum(devNumInlocalPod) - 1, DEVICE_FOUR);
    // level2StreamNum超节点间reducescatter
    u32 level2StreamNum = std::min(CalReduceStreamNum(topoAttr_.superPodNum) - 1, DEVICE_FOUR);
    // 总流数上限：7（alltoall使用，提前的本地拷贝任务不需要并行）+ 4（LocalReduce使用）
    streamNum = std::max(all2allStreamNum + reduceStreamNum - 1, level2StreamNum);

    HCCL_INFO(
        "[%s]tag[%s] all2allStreamNum[%u], reduceStreamNum[%u], level2StreamNum[%u], streamNum[%u]", __func__,
        tag_.c_str(), all2allStreamNum, reduceStreamNum, level2StreamNum, streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedFor91093Executor::CalcTransportMemType(
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

HcclResult CollAllReduceOrderPreservedFor91093Executor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel1(COMM_COMBINE_L1, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[COMM_COMBINE_L1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedFor91093Executor::CalcLevel2CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    if (topoAttr_.superPodNum > 1) {
        CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MESH);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    }
    return HCCL_SUCCESS;
}

bool CollAllReduceOrderPreservedFor91093Executor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize / topoAttr_.deviceNumPerAggregation / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE
                    || curSize > SDMA_SEND_MAX_SIZE;
    HCCL_DEBUG(
        "[%s]isHugeData[%d], curSize[%llu], topoAttr_.deviceNumPerAggregation[%u]", __func__, hugeData, curSize,
        topoAttr_.deviceNumPerAggregation);
    return hugeData;
}

void CollAllReduceOrderPreservedFor91093Executor::CalcSizePerBlock(const OpParam& param, ExecMem& execMem)
{
    sizePerBlock_
        = (execMem.count + topoAttr_.userRankSize - 1) / topoAttr_.userRankSize * SIZE_TABLE[param.DataDes.dataType];
    sizePerBlock_ = AlgTemplateBase::RoundUpWithDivisor(sizePerBlock_, HCCL_MIN_SLICE_ALIGN);
}

void CollAllReduceOrderPreservedFor91093Executor::CalGroupSlices(const OpParam& param, const ExecMem& execMem)
{
    groupSize_.clear();
    u64 sizeRemain = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    for (u32 rankId = 0; rankId < topoAttr_.userRankSize; rankId++) {
        u64 size = (sizeRemain > sizePerBlock_) ? sizePerBlock_ : sizeRemain;
        groupSize_.push_back(size);
        sizeRemain -= size;
    }
}

HcclResult CollAllReduceOrderPreservedFor91093Executor::RunReduceScatterLevel1(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level1CommInfo)
{
    if (level1CommInfo.localRankSize == 1) {
        all2allOffset_ = topoAttr_.superPodNum > 1 ? 1 : 0;
        HCCL_INFO("[%s] single rank per module, skip L1 AllToAll and LocalReduce, tag[%s]", __func__, tag_.c_str());
        CHK_RET(RunReduceScatterLevel1SingleRank(param, execMem, level1CommInfo));
        return HCCL_SUCCESS;
    }

    // 切分数据(ReduceScatter分组，记录每组的起始偏移和大小)
    GroupSlicesInfo groupSlicesInfoLevel1;
    for (u32 groupId = 0; groupId < topoAttr_.superPodNum; groupId++) {
        MemBlockInfo memInfo;
        for (u32 dataId = 0; dataId < level1CommInfo.localRankSize; dataId++) {
            u64 globalDataId = groupId * level1CommInfo.localRankSize + dataId;
            u64 size = groupSize_[globalDataId];
            u64 offset = globalDataId * sizePerBlock_;
            memInfo.size.push_back(size);
            memInfo.userInputOffsets.push_back(offset);
            memInfo.inputOffsets.push_back(offset);
            memInfo.outputOffsets.push_back(offset);
        }
        groupSlicesInfoLevel1.push_back(memInfo);
    }

    CHK_RET(ActiveSlaveStreams(param.stream));
    all2allOffset_ = topoAttr_.superPodNum > 1 ? 1 : 0; // 多机场景需要偏移1（给L1预留计算位，减少拷贝次数）

    std::unique_ptr<AlgTemplateBase> level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE in COMM_COMBINE_L1", __func__);
    CHK_SMART_PTR_NULL(level1TempAlg);

    DeviceMem outputMem = scratchMemFlag_ ? execMem.scratchMem : execMem.outputMem;
    CHK_RET(level1TempAlg->Prepare(
        execMem.inputPtr, execMem.inputMem, outputMem, param.stream, algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, groupSlicesInfoLevel1, param.reduceType, all2allOffset_,
        param.DataDes.dataType, true, false, true));

    CHK_RET(level1TempAlg->RegisterProfiler(
        (level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedFor91093Executor::RunReduceScatterLevel1SingleRank(
    const OpParam& param, ExecMem& execMem, const SubCommInfo& level1CommInfo) const
{
    (void)level1CommInfo;
    u64 size = execMem.count * SIZE_TABLE[param.DataDes.dataType];

    DeviceMem srcMem = DeviceMem::create(execMem.inputPtr, size);
    DeviceMem dstMem = scratchMemFlag_ ? execMem.scratchMem.range(0, size) : execMem.inputMem.range(0, size);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));

    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedFor91093Executor::RunReduceScatterLevel2(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level1CommInfo)
{
    u32 commIndex = level1CommInfo.localRank;
    u32 level1Ranksize = level1CommInfo.localRankSize;
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    // 切分数据，记录每组的起始偏移和大小（仅1组）
    MemBlockInfo memInfo;
    u32 inputBaseIndex
        = (all2allOffset_ + commIndex) % level1Ranksize; // 多机场景需要偏移1（给L1预留计算位，减少拷贝次数）
    for (u32 dataId = 0; dataId < level2CommInfo.localRankSize; dataId++) {
        u64 inputIndex = inputBaseIndex + dataId * level1Ranksize;
        memInfo.inputOffsets.push_back(inputIndex * sizePerBlock_);
        memInfo.size.push_back(groupSize_[commIndex + dataId * level1Ranksize]);
        u64 outputIndex = commIndex + dataId * level1Ranksize;
        memInfo.outputOffsets.push_back(outputIndex * sizePerBlock_);
        memInfo.userInputOffsets.push_back(outputIndex * sizePerBlock_);
    }

    DeviceMem outputMem = scratchMemFlag_ ? execMem.scratchMem : execMem.outputMem;
    std::unique_ptr<AlgTemplateBase> level2TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE_COMBINE, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE_COMBINE in COMM_LEVEL2", __func__);
    CHK_SMART_PTR_NULL(level2TempAlg);

    u32 level1LastRank = level1Ranksize - 1;
    bool isUseCclIn = level1Ranksize == 1 || commIndex == level1LastRank - 1;
    CHK_RET(level2TempAlg->Prepare(
        execMem.inputMem, outputMem, param.stream, algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux, memInfo, param.reduceType, param.DataDes.dataType, isUseCclIn,
        commIndex == level1LastRank, true));

    CHK_RET(level2TempAlg->RegisterProfiler(
        (level1Ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level2TempAlg, level2CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedFor91093Executor::RunAllGatherLevel1(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level1CommInfo)
{
    u32 level1RankSize = level1CommInfo.localRankSize;
    u32 commIndex = level1CommInfo.localRank;
    u64 count = execMem.count / topoAttr_.userRankSize;
    u64 serverOffsetConut = topoAttr_.userRank / level1RankSize * level1RankSize;

    // allgather 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> dataSegsSlice;
    for (u32 rank = 0; rank < level1RankSize; rank++) {
        Slice userslice;
        userslice.size = groupSize_[rank + serverOffsetConut];
        userslice.offset = userslice.size == 0 ? 0 : (rank + serverOffsetConut) * sizePerBlock_;
        dataSegsSlice.emplace_back(std::move(userslice));
    }

    DeviceMem outputMem = scratchMemFlag_ ? execMem.scratchMem : execMem.outputMem;

    std::unique_ptr<AlgTemplateBase> level1TempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_COMBINE_L1", __func__);

    CHK_SMART_PTR_NULL(level1TempAlg);
    CHK_RET(level1TempAlg->Prepare(
        outputMem, outputMem, outputMem, count, param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED,
        LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0));

    CHK_RET(level1TempAlg->RegisterProfiler(
        (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commIndex, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
        param.stream));

    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedFor91093Executor::RunAllGatherLevel2(
    const OpParam& param, const ExecMem& execMem, const SubCommInfo& level1CommInfo)
{
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    std::unique_ptr<AlgTemplateBase> level2TempAlg; // Level1Allgather(根据算法选择)
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_RING) {
        level2TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL2", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        level2TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL2", __func__);
    } else {
        level2TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL2", __func__);
    }

    DeviceMem outputMem = scratchMemFlag_ ? execMem.scratchMem : execMem.outputMem;

    u32 level1RankSize = level1CommInfo.localRankSize;
    u64 count = execMem.count / level2CommInfo.localRankSize;

    std::vector<u64> level2GroupSize;
    for (u32 rank = 0; rank < level2CommInfo.localRankSize; rank++) {
        u64 size = 0;
        for (u32 level1RankId = 0; level1RankId < level1RankSize; level1RankId++) {
            size += groupSize_[rank * level1RankSize + level1RankId];
        }
        level2GroupSize.push_back(size);
    }

    // allgather 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> dataSegsSlice;
    for (u32 rank = 0; rank < level2CommInfo.localRankSize; rank++) {
        Slice userslice;
        userslice.size = level2GroupSize[rank];
        userslice.offset = rank * level1RankSize * sizePerBlock_;
        dataSegsSlice.emplace_back(std::move(userslice));
    }

    CHK_SMART_PTR_NULL(level2TempAlg);
    CHK_RET(level2TempAlg->Prepare(
        outputMem, outputMem, outputMem, count, param.DataDes.dataType, param.stream,
        HcclReduceOp::HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, dataSegsSlice));

    CHK_RET(level2TempAlg->RegisterProfiler(
        (level2CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level2TempAlg, level2CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceOrderPreservedFor91093Executor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[%s]The CollAllReduceOrderPreservedFor91093Executor starts, tag[%s]", __func__, tag_.c_str());
    CHK_RET(CheckCommSize(COMM_COMBINE_L1, COMM_INDEX_0 + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_COMBINE_L1, COMM_INDEX_0);

    CalcSizePerBlock(param, execMem);
    CalGroupSlices(param, execMem);

    u64 inputSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    scratchMemFlag_ = (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
                      && (inputSize < (topoAttr_.userRankSize - 1) * sizePerBlock_);

    // L1 节点内 reduce scatter
    CHK_RET(RunReduceScatterLevel1(param, execMem, level1CommInfo));
    // L2 节点间 reduce scatter
    if (topoAttr_.superPodNum > 1) {
        CHK_RET(RunReduceScatterLevel2(param, execMem, level1CommInfo));
    }

    // Level1 节点内 AllGatherMeshAtomic
    CHK_RET(RunAllGatherLevel1(param, execMem, level1CommInfo));
    if (topoAttr_.superPodNum > 1) {
        // L2 节点间 allgather
        CHK_RET(RunAllGatherLevel2(param, execMem, level1CommInfo));
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

REGISTER_EXEC(
    "AllReduceOrderPreservedFor91093Executor", AllReduceOrderPreservedFor91093,
    CollAllReduceOrderPreservedFor91093Executor);
} // namespace hccl
