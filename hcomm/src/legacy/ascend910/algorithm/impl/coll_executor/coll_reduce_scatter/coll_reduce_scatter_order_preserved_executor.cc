/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_order_preserved_executor.h"

namespace hccl {

CollReduceScatterOrderPreservedExecutor::CollReduceScatterOrderPreservedExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
}

void CollReduceScatterOrderPreservedExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    // 是否需要scratch memory（图模式没有cclbuffer，需要额外申请scratchMem）
    scratchMemFlag_ = (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    u64 sizePerRank = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    totalSize_ = topoAttr_.userRankSize * sizePerRank;

    // 单算子场景 单机2次幂场景小数据量使用HD性能更优
    const bool isSmallData = sizePerRank <= HCCL_SMALL_COUNT_32_KB;
    const bool isSingleModule = topoAttr_.moduleNum == 1;
    const bool isOpBase = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    const bool isPowerOfTwoDevices
        = (topoAttr_.deviceNumPerAggregation == DEVICE_EIGHT) || (topoAttr_.deviceNumPerAggregation == DEVICE_FOUR);
    isUseHDAlg_ = isSmallData && isSingleModule && isOpBase && isPowerOfTwoDevices;
}

HcclResult CollReduceScatterOrderPreservedExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    scratchMemSize = scratchMemFlag_ ? totalSize_ : 0U;
    HCCL_INFO("[%s]tag[%s] scratchMemSize[%llu]", __func__, tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

u32 CollReduceScatterOrderPreservedExecutor::CalReduceStreamNum(const u32& localRankSize)
{
    return (1 << static_cast<int>(std::floor(log2(localRankSize))));
}

HcclResult CollReduceScatterOrderPreservedExecutor::CalcStreamNum(u32& streamNum)
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

HcclResult CollReduceScatterOrderPreservedExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterOrderPreservedExecutor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    // scratchMemFlag_ 对应图模式场景（图模式没有cclbuffer）, PARAM_INPUT -> userInput
    inputType = scratchMemFlag_ ? TransportMemType::PARAM_INPUT : TransportMemType::CCL_INPUT;
    outputType = scratchMemFlag_ ? TransportMemType::SCRATCH : TransportMemType::CCL_OUTPUT;
    HCCL_INFO("[%s]tag[%s] inputType[%d], outputType[%d]", __func__, tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterOrderPreservedExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterOrderPreservedExecutor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    if (topoAttr_.moduleNum > 1) {
        CommParaInfo commParaLevel1(COMM_LEVEL1, CommType::COMM_TAG_MESH);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[COMM_LEVEL1], inputType, outputType));
    }
    return HCCL_SUCCESS;
}

bool CollReduceScatterOrderPreservedExecutor::IsSmallData(const u64 totalSize, const u64 curSize)
{
    (void)curSize;
    // 子图复用的阈值（opmeta全一致时，ffts子图复用）
    return totalSize <= HCCL_SMALL_COUNT_32_KB;
}

HcclResult CollReduceScatterOrderPreservedExecutor::RunReduceScatterLevel0SingleRank(
    const OpParam& param, ExecMem& execMem, const SubCommInfo& level0CommInfo) const
{
    (void)level0CommInfo;
    u64 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 curSize = execMem.count * unitSize;
    DeviceMem bufferMem = scratchMemFlag_ ? execMem.scratchMem : execMem.inputMem;
    DeviceMem dstMem;
    DeviceMem srcMem;
    for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
        // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为output的size
        dstMem = bufferMem.range(curSize * i, curSize);
        srcMem = DeviceMem::create(static_cast<u8*>(execMem.inputPtr) + param.DataDes.count * unitSize * i, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterOrderPreservedExecutor::RunReduceScatterLevel0HD(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo)
{
    std::unique_ptr<AlgTemplateBase> level0TempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_HDSTAGE, dispatcher_);

    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
    HcomCollOpInfo opInfo = {"",
                             execMem.inputPtr,
                             execMem.outputPtr,
                             param.DataDes.count,
                             param.DataDes.dataType,
                             param.root,
                             param.reduceType,
                             0};

    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(
        execMem.inputMem, execMem.scratchMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.stream,
        param.reduceType, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0, reduceAttr, algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank, &opInfo));

    CHK_RET(level0TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterOrderPreservedExecutor::RunReduceScatterLevel0(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo)
{
    if (level0CommInfo.localRankSize == 1) {
        all2allOffset_ = topoAttr_.moduleNum > 1 ? 1 : 0;
        HCCL_INFO("[%s] single rank per module, skip L0 AllToAll and LocalReduce, tag[%s]", __func__, tag_.c_str());
        CHK_RET(RunReduceScatterLevel0SingleRank(param, execMem, level0CommInfo));
        return HCCL_SUCCESS;
    }

    CHK_RET(ActiveSlaveStreams(param.stream));
    if (isUseHDAlg_) {
        CHK_RET(RunReduceScatterLevel0HD(param, execMem, level0CommInfo));
    } else {
        // 切分数据(ReduceScatter分组，记录每组的起始偏移和大小)
        GroupSlicesInfo groupSlicesInfoLevel0;
        u64 size = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        for (u32 groupId = 0; groupId < topoAttr_.moduleNum; groupId++) {
            MemBlockInfo memInfo;
            for (u32 dataId = 0; dataId < level0CommInfo.localRankSize; dataId++) {
                u64 offset = (dataId + groupId * level0CommInfo.localRankSize) * size;
                u64 userMemInOffset = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType]
                                      * (dataId + groupId * level0CommInfo.localRankSize);
                memInfo.size.push_back(size);
                memInfo.userInputOffsets.push_back(userMemInOffset);
                memInfo.inputOffsets.push_back(offset);
                memInfo.outputOffsets.push_back(offset);
            }
            groupSlicesInfoLevel0.push_back(memInfo);
        }

        all2allOffset_ = topoAttr_.moduleNum > 1 ? 1 : 0; // 多机场景需要偏移1（给L1预留计算位，减少拷贝次数）
        std::unique_ptr<AlgTemplateBase> level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE, dispatcher_);
        CHK_SMART_PTR_NULL(level0TempAlg);

        // execMem.scratchMem在单算子模式下为cclout，图模式为scratch，因此output传入scratch即可
        CHK_RET(level0TempAlg->Prepare(
            execMem.inputPtr, execMem.inputMem, execMem.scratchMem, param.stream, algResResp_->slaveStreams,
            algResResp_->notifiesMain, algResResp_->notifiesAux, groupSlicesInfoLevel0, param.reduceType,
            all2allOffset_, param.DataDes.dataType, false));
        CHK_RET(level0TempAlg->RegisterProfiler(
            (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_2,
            HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterOrderPreservedExecutor::RunReduceScatterLevel1(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level0CommInfo)
{
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    // 切分数据，记录每组的起始偏移和大小（仅1组）
    u64 size = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    MemBlockInfo memInfo;
    u32 level0Ranksize = level0CommInfo.localRankSize;
    u32 inputBaseIndex
        = (all2allOffset_ + commIndex) % level0Ranksize; // 多机场景需要偏移1（给L1预留计算位，减少拷贝次数）
    for (u32 dataId = 0; dataId < level1CommInfo.localRankSize; dataId++) {
        u64 inputIndex = inputBaseIndex + dataId * level0Ranksize;
        memInfo.inputOffsets.push_back(inputIndex * size);
        u64 outputIndex = commIndex + dataId * level0Ranksize;
        memInfo.outputOffsets.push_back(outputIndex * size);
        memInfo.userInputOffsets.push_back(outputIndex * size);
        memInfo.size.push_back(size);
    }

    std::unique_ptr<AlgTemplateBase> level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE_COMBINE, dispatcher_);
    CHK_SMART_PTR_NULL(level1TempAlg);

    u32 level0LastRank = level0Ranksize - 1;
    bool isUseCclIn = (level0Ranksize == 1) || (commIndex == level0LastRank - 1);
    bool borrowSpace = level0Ranksize == 1;
    CHK_RET(level1TempAlg->Prepare(
        execMem.inputMem, execMem.scratchMem, param.stream, algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux, memInfo, param.reduceType, param.DataDes.dataType, isUseCclIn,
        commIndex == level0LastRank, borrowSpace));
    CHK_RET(level1TempAlg->RegisterProfiler(
        (level0Ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterOrderPreservedExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s]CollReduceScatterOrderPreservedExecutor starts, tag[%s]", __func__, tag_.c_str());
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // L0 节点内 reduce scatter
    CHK_RET(RunReduceScatterLevel0(param, execMem, level0CommInfo));
    // L1 节点间 reduce scatter
    if (topoAttr_.moduleNum > 1) {
        CHK_RET(RunReduceScatterLevel1(param, execMem, level0CommInfo));
    }

    if (!isUseHDAlg_) {
        // 非HD算法 execMem.scratchMem最后拷贝至UserOut
        u64 dataSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        DeviceMem srcMem = execMem.scratchMem.range(dataSize * topoAttr_.userRank, dataSize);
        DeviceMem dstMem = DeviceMem::create(execMem.outputPtr, dataSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    HCCL_INFO("[%s]order preserved ReduceScatter run success, tag[%s]", __func__, tag_.c_str());
    return HCCL_SUCCESS;
}

REGISTER_EXEC(
    "ReduceScatterOrderPreservedExecutor", ReduceScatterOrderPreserved, CollReduceScatterOrderPreservedExecutor);
} // namespace hccl
