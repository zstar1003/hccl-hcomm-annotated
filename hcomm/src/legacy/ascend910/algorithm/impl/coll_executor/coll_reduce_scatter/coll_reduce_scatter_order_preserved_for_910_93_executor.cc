/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_order_preserved_for_910_93_executor.h"

namespace hccl {

CollReduceScatterOrderPreservedFor91093Executor::CollReduceScatterOrderPreservedFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
    desc_.deterministic = DETERMINISTIC_STRICT;
}

void CollReduceScatterOrderPreservedFor91093Executor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;

    // 是否需要scratch memory（图模式没有cclbuffer，需要额外申请scratchMem）
    scratchMemFlag_ = (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    u64 sizePerRank = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    totalSize_ = topoAttr_.userRankSize * sizePerRank;
}

HcclResult CollReduceScatterOrderPreservedFor91093Executor::CalcScratchMemSize(u64& scratchMemSize)
{
    scratchMemSize = scratchMemFlag_ ? totalSize_ : 0U;
    HCCL_INFO("[%s]tag[%s] scratchMemSize[%llu]", __func__, tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

u32 CollReduceScatterOrderPreservedFor91093Executor::CalReduceStreamNum(const u32& localRankSize) const
{
    return (1 << static_cast<int>(std::floor(log2(localRankSize))));
}

HcclResult CollReduceScatterOrderPreservedFor91093Executor::CalcStreamNum(u32& streamNum)
{
    // 获取超节点内rank数
    u32 devNumInlocalPod = 0;
    u32 rankIdxInPod = 0;
    CHK_RET(topoMatcher_->GetLocalSuperPodRankSize(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));

    // 单卡节点场景，L1(超节点内)不需要流，仅计算L2流数
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

HcclResult
CollReduceScatterOrderPreservedFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterOrderPreservedFor91093Executor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    // scratchMemFlag_ 对应图模式场景（图模式没有cclbuffer）, PARAM_INPUT -> userInput
    inputType = scratchMemFlag_ ? TransportMemType::PARAM_INPUT : TransportMemType::CCL_INPUT;
    outputType = scratchMemFlag_ ? TransportMemType::SCRATCH : TransportMemType::CCL_OUTPUT;
    HCCL_INFO("[%s]tag[%s] inputType[%d], outputType[%d]", __func__, tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterOrderPreservedFor91093Executor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel1(COMM_COMBINE_L1, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[COMM_COMBINE_L1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterOrderPreservedFor91093Executor::CalcLevel2CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    if (topoAttr_.superPodNum > 1) {
        CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MESH);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    }
    return HCCL_SUCCESS;
}

bool CollReduceScatterOrderPreservedFor91093Executor::IsSmallData(const u64 totalSize, const u64 curSize)
{
    (void)curSize;
    // 子图复用的阈值（opmeta全一致时，ffts子图复用）
    return totalSize <= HCCL_SMALL_COUNT_32_KB;
}

HcclResult CollReduceScatterOrderPreservedFor91093Executor::RunReduceScatterLevel1SingleRank(
    const OpParam& param, ExecMem& execMem, const SubCommInfo& level1CommInfo) const
{
    (void)level1CommInfo;
    HCCL_INFO("[%s] single rank per module, skip L1 AllToAll and LocalReduce, tag[%s]", __func__, tag_.c_str());

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

HcclResult CollReduceScatterOrderPreservedFor91093Executor::RunReduceScatterLevel1(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level1CommInfo)
{
    if (level1CommInfo.localRankSize == 1) {
        all2allOffset_ = topoAttr_.superPodNum > 1 ? 1 : 0;
        CHK_RET(RunReduceScatterLevel1SingleRank(param, execMem, level1CommInfo));
        return HCCL_SUCCESS;
    }

    CHK_RET(ActiveSlaveStreams(param.stream));

    // 切分数据(ReduceScatter分组，记录每组的起始偏移和大小)
    GroupSlicesInfo groupSlicesInfoLevel0;
    u64 size = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    for (u32 groupId = 0; groupId < topoAttr_.superPodNum; groupId++) {
        MemBlockInfo memInfo;
        for (u32 dataId = 0; dataId < level1CommInfo.localRankSize; dataId++) {
            u64 offset = (dataId + groupId * level1CommInfo.localRankSize) * size;
            u64 userMemInOffset = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType]
                                  * (dataId + groupId * level1CommInfo.localRankSize);
            memInfo.size.push_back(size);
            memInfo.userInputOffsets.push_back(userMemInOffset);
            memInfo.inputOffsets.push_back(offset);
            memInfo.outputOffsets.push_back(offset);
        }
        groupSlicesInfoLevel0.push_back(memInfo);
    }

    all2allOffset_ = topoAttr_.superPodNum > 1 ? 1 : 0; // 多机场景需要偏移1（给L1预留计算位，减少拷贝次数）
    std::unique_ptr<AlgTemplateBase> level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE in COMM_COMBINE_L1", __func__);
    CHK_SMART_PTR_NULL(level1TempAlg);

    // execMem.scratchMem在单算子模式下为cclout，图模式为scratch，因此output传入scratch即可
    CHK_RET(level1TempAlg->Prepare(
        execMem.inputPtr, execMem.inputMem, execMem.scratchMem, param.stream, algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, groupSlicesInfoLevel0, param.reduceType, all2allOffset_,
        param.DataDes.dataType, false, false, true));
    CHK_RET(level1TempAlg->RegisterProfiler(
        (level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));

    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterOrderPreservedFor91093Executor::RunReduceScatterLevel2(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level1CommInfo)
{
    u32 commIndex = level1CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    // 切分数据，记录每组的起始偏移和大小（仅1组）
    u64 size = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    MemBlockInfo memInfo;
    u32 level0Ranksize = level1CommInfo.localRankSize;
    u32 inputBaseIndex
        = (all2allOffset_ + commIndex) % level0Ranksize; // 多机场景需要偏移1（给L1预留计算位，减少拷贝次数）
    for (u32 dataId = 0; dataId < level2CommInfo.localRankSize; dataId++) {
        u64 inputIndex = inputBaseIndex + dataId * level0Ranksize;
        memInfo.inputOffsets.push_back(inputIndex * size);
        u64 outputIndex = commIndex + dataId * level0Ranksize;
        memInfo.outputOffsets.push_back(outputIndex * size);
        memInfo.userInputOffsets.push_back(outputIndex * size);
        memInfo.size.push_back(size);
    }

    std::unique_ptr<AlgTemplateBase> level2TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE_COMBINE, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE_COMBINE in COMM_LEVEL2", __func__);
    CHK_SMART_PTR_NULL(level2TempAlg);

    u32 level0LastRank = level0Ranksize - 1;
    bool isUseCclIn = (level0Ranksize == 1) || (commIndex == level0LastRank - 1);
    bool borrowSpace = level0Ranksize == 1;
    CHK_RET(level2TempAlg->Prepare(
        execMem.inputMem, execMem.scratchMem, param.stream, algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux, memInfo, param.reduceType, param.DataDes.dataType, isUseCclIn,
        commIndex == level0LastRank, borrowSpace));
    CHK_RET(level2TempAlg->RegisterProfiler(
        (level0Ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level2TempAlg, level2CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterOrderPreservedFor91093Executor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[%s]CollReduceScatterOrderPreservedFor91093Executor starts, tag[%s]", __func__, tag_.c_str());
    CHK_RET(CheckCommSize(COMM_COMBINE_L1, COMM_INDEX_0 + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_COMBINE_L1, COMM_INDEX_0);

    // L1 节点内 reduce scatter
    CHK_RET(RunReduceScatterLevel1(param, execMem, level1CommInfo));
    // L2 节点间 reduce scatter
    if (topoAttr_.superPodNum > 1) {
        CHK_RET(RunReduceScatterLevel2(param, execMem, level1CommInfo));
    }

    // 非HD算法 execMem.scratchMem最后拷贝至UserOut
    u64 dataSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    DeviceMem srcMem = execMem.scratchMem.range(dataSize * topoAttr_.userRank, dataSize);
    DeviceMem dstMem = DeviceMem::create(execMem.outputPtr, dataSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));

    HCCL_INFO("[%s]order preserved ReduceScatter run success, tag[%s]", __func__, tag_.c_str());
    return HCCL_SUCCESS;
}

REGISTER_EXEC(
    "ReduceScatterOrderPreservedFor91093Executor", ReduceScatterOrderPreservedFor91093,
    CollReduceScatterOrderPreservedFor91093Executor);
} // namespace hccl
