/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_pipeline_for_910_93_executor.h"
#include "hccl_types.h"
#include "alg_template_register.h"
#include "alg_template_base_pub.h"

namespace hccl {
constexpr u32 PIPELINE_NUM = 2;
constexpr u32 PIPELINE_EXTRA_STREAM_NUM = PIPELINE_NUM;

CollAllGatherPipelineFor91093Executor::CollAllGatherPipelineFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    desc_.level1SupportedAlgos
        = {AlgTypeLevel1::ALG_LEVEL1_NHR, AlgTypeLevel1::ALG_LEVEL1_NB, AlgTypeLevel1::ALG_LEVEL1_RING};
    desc_.level2SupportedAlgos
        = {AlgTypeLevel2::ALG_LEVEL2_NHR, AlgTypeLevel2::ALG_LEVEL2_NB, AlgTypeLevel2::ALG_LEVEL2_RING};
}

HcclResult CollAllGatherPipelineFor91093Executor::CalcStreamNum(u32& streamNum)
{
    // 计算三级流水线所需的流数量
    HCCL_INFO(
        "[CollAllGatherPipelineFor91093Executor][CalcStreamNum] topoType_[%u], workflowMode_[%u]", topoType_,
        workflowMode_);
    // 基本流数量计算
    u32 totalStreamNum
        = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
                                                             LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) { // 工作流模式，双倍的流，用于并行操作
        totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;             // *2
    }

    // 为三级流水线增加额外的流
    // 从流用于L2，主流用于L1 + L0
    // 新增从流中，一条用于L2流水线，一条用于多申请2个notify，共新增4个notify用于两块内存的主从流之间的同步
    totalStreamNum += PIPELINE_EXTRA_STREAM_NUM;

    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollAllGatherPipelineFor91093Executor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherPipelineFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

// level0 ring
HcclResult CollAllGatherPipelineFor91093Executor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

// level2 NHR
HcclResult CollAllGatherPipelineFor91093Executor::CalcLevel2CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
    } else {
        commParaInfo.commType = CommType::COMM_TAG_RING_INNER;
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherPipelineFor91093Executor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;
    HCCL_INFO(
        "[CollAllGatherPipelineFor91093Executor][CalcTransportMemType]"
        "tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

// 每次循环处理的数据量，双流水的情况下需要满足每个流水线都能满载
u64 CollAllGatherPipelineFor91093Executor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    // 分成两片，做流水ping-pong
    u64 maxCountPerLoop
        = cclBuffSize / PIPELINE_NUM / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;
    HCCL_INFO("[%s] tag[%s] maxCountPerLoop[%llu]", __func__, tag_.c_str(), maxCountPerLoop);

    return maxCountPerLoop;
}

// 编排
HcclResult CollAllGatherPipelineFor91093Executor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherPipelineFor91093Executor][Orchestrate] begins.");

    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    // 设置L0和L1通信域信息
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    level0CommInfo_ = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo_.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    level1CommInfo_ = GetSubCommInfo(COMM_LEVEL1, commIndex);
    // 获取L2通信域信息
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    level2CommInfo_ = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    // L1/L0 runs on the main stream; L2 runs on the reserved slave stream.
    mainStreamL1L0_ = param.stream;
    subStreams_ = algResResp_->slaveStreams;
    mainStreamL2_ = subStreams_.back();
    const u32 baseStreamNum = algResResp_->slaveStreams.size() - PIPELINE_EXTRA_STREAM_NUM;
    notifyL1L0ToL2A_ = algResResp_->notifiesAux[baseStreamNum];
    notifyL1L0ToL2B_ = algResResp_->notifiesAux[baseStreamNum + 1];
    notifyL2ToL1L0A_ = algResResp_->notifiesMain[baseStreamNum];
    notifyL2ToL1L0B_ = algResResp_->notifiesMain[baseStreamNum + 1];
    HCCL_INFO(
        "[CollAllGatherPipelineFor91093Executor][RunLoop] NotifyIds: "
        "L1L0ToL2A: Aux[%u], L1L0ToL2B: Aux[%u], L2ToL1L0A: Main[%u], L2ToL1L0B: Main[%u]",
        baseStreamNum, baseStreamNum + 1, baseStreamNum, baseStreamNum + 1);
    notifyRingMain_.assign(
        algResResp_->notifiesMain.begin(), algResResp_->notifiesMain.end() - PIPELINE_EXTRA_STREAM_NUM);
    notifyRingSub_.assign(algResResp_->notifiesAux.begin(), algResResp_->notifiesAux.end() - PIPELINE_EXTRA_STREAM_NUM);
    ringSubStreams_.assign(subStreams_.begin(), subStreams_.end() - PIPELINE_EXTRA_STREAM_NUM);

    // 计算通信域信息和内存类型
    unitSize_ = SIZE_TABLE[param.DataDes.dataType];
    cclInputSizeHalved_ = algResResp_->cclInputMem.size() / PIPELINE_NUM;
    cclInputAMem_ = algResResp_->cclInputMem.range(0, cclInputSizeHalved_);
    cclInputBMem_ = algResResp_->cclInputMem.range(cclInputSizeHalved_, cclInputSizeHalved_);
    cclOutputSizeHalved_ = algResResp_->cclOutputMem.size() / PIPELINE_NUM;
    cclOutputAMem_ = algResResp_->cclOutputMem.range(0, cclOutputSizeHalved_);
    cclOutputBMem_ = algResResp_->cclOutputMem.range(cclOutputSizeHalved_, cclOutputSizeHalved_);

    CHK_RET(RunLoop(param)); // 运行循环，循环内执行三级流水线

    HCCL_INFO(
        "tag[%s], Allgather executor orchestrate success, take time [%lld]us.", tag_.c_str(),
        DURATION_US(TIME_NOW() - startut));

    return HCCL_SUCCESS;
}

HcclResult CollAllGatherPipelineFor91093Executor::RunL2Stage(
    const OpParam& param, ExecMem& execMem, u64 loopIdx, u64 memIdx, u64 bufferSliceNum)
{
    // superpod数量不超过1不需要跨超节点；最后一轮循环处理L1L0的最后一片数据，L2不需要参与通信，跳过L2阶段
    if (loopIdx >= bufferSliceNum) {
        return HCCL_SUCCESS;
    }
    // Loop 0 waits for the main stream start signal. Later ping-pong buffer reuse waits for L1/L0.
    if (loopIdx == 0 || loopIdx >= PIPELINE_NUM) {
        auto notifyL1L0ToL2 = (memIdx == 0) ? notifyL1L0ToL2A_ : notifyL1L0ToL2B_;
        CHK_RET(LocalNotify::Wait(mainStreamL2_, dispatcher_, notifyL1L0ToL2, INVALID_VALUE_STAGE));
    }

    // Local Copy: UserIn -> Ccl
    u64 curSize = execMem.count * unitSize_;
    DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(execMem.inputPtr), curSize);
    DeviceMem dstMem = execMem.inputMem.range(0, curSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, mainStreamL2_));

    // OPBASE模式
    u64 dstMemOffset = topoAttr_.userRank * curSize;
    DeviceMem dmaDst = execMem.outputMem.range(dstMemOffset, curSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dmaDst, srcMem, mainStreamL2_));

    // L2 Rx -> DMA[0/1]
    u64 baseOffset = memIdx == 0 ? 0 : cclInputSizeHalved_;
    CHK_RET(KernelRunInterSuperPod(param, execMem, baseOffset));
    auto notifyL2ToL1L0 = (memIdx == 0) ? notifyL2ToL1L0A_ : notifyL2ToL1L0B_;
    CHK_RET(LocalNotify::Post(mainStreamL2_, dispatcher_, notifyL2ToL1L0, INVALID_VALUE_STAGE));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherPipelineFor91093Executor::RunL1L0Stage(
    const OpParam& param, ExecMem& lastExecMem, u64 loopIdx, u64 memIdx, u64 bufferSliceNum)
{
    // 第一轮等待L2处理完
    if (loopIdx < 1) {
        return HCCL_SUCCESS;
    }
    // 同步：等待上一轮的 L2 通信完成
    auto notifyL2ToL1L0 = (memIdx == 0) ? notifyL2ToL1L0A_ : notifyL2ToL1L0B_;
    CHK_RET(LocalNotify::Wait(mainStreamL1L0_, dispatcher_, notifyL2ToL1L0, INVALID_VALUE_STAGE));

    u64 baseOffset = memIdx == 0 ? 0 : cclInputSizeHalved_;
    if (level1CommInfo_.localRankSize > 1) {
        CHK_RET(KernelRunInterServer(param, lastExecMem, baseOffset));
    }
    CHK_RET(KernelRunIntraServer(param, lastExecMem, baseOffset));

    if (loopIdx + 1 < bufferSliceNum) {
        auto notifyL1L0ToL2 = (memIdx == 0) ? notifyL1L0ToL2A_ : notifyL1L0ToL2B_;
        CHK_RET(LocalNotify::Post(mainStreamL1L0_, dispatcher_, notifyL1L0ToL2, INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherPipelineFor91093Executor::RunLoop(OpParam& param)
{
    u8* userInputPtr = static_cast<u8*>(param.inputPtr);
    u8* userOutputPtr = static_cast<u8*>(param.outputPtr);
    CHK_PTR_NULL(userInputPtr);
    CHK_PTR_NULL(userOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(algResResp_->cclInputMem.size(), unitSize_);
    CHK_PRT_RET(
        maxCountPerLoop == 0,
        HCCL_ERROR(
            "[CollAllGatherPipelineFor91093Executor][RunLoop]tag[%s] userRankSize[%u] maxCountPerLoop[%llu]",
            tag_.c_str(), topoAttr_.userRankSize, maxCountPerLoop),
        HCCL_E_PARA);
    u64 bufferSliceNum = (param.DataDes.count + maxCountPerLoop - 1) / maxCountPerLoop;
    if (bufferSliceNum == 0) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "[CollAllGatherPipelineFor91093Executor][%s] maxCountPerLoop[%llu] bufferSliceNum[%llu]", __func__,
        maxCountPerLoop, bufferSliceNum);
    u64 loopNum = bufferSliceNum + 1;
    u64 countLeft = param.DataDes.count; // 剩余的数据量

    u32 memIdx = 0;
    ExecMem lastExecMem;
    // AllGather starts with L2, so the main stream releases the first L2 stage before the loop.
    CHK_RET(LocalNotify::Post(mainStreamL1L0_, dispatcher_, notifyL1L0ToL2A_, INVALID_VALUE_STAGE));
    for (u64 loopIdx = 0; loopIdx < loopNum; loopIdx++) {
        u64 curCount = countLeft > maxCountPerLoop ? maxCountPerLoop : countLeft; // 当前循环处理的数据量
        countLeft -= curCount;

        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = memIdx == 0 ? cclInputAMem_ : cclInputBMem_;
        execMem.outputMem = memIdx == 0 ? cclOutputAMem_ : cclOutputBMem_;
        execMem.inputPtr = userInputPtr;
        execMem.outputPtr = userOutputPtr;

        CHK_RET(RunL2Stage(param, execMem, loopIdx, memIdx, bufferSliceNum));
        CHK_RET(RunL1L0Stage(param, lastExecMem, loopIdx, 1 - memIdx, bufferSliceNum));

        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));

        u64 curSize = curCount * unitSize_;
        userInputPtr += curSize;
        userOutputPtr += curSize;
        memIdx = 1 - memIdx; // 双缓冲交替使用
        lastExecMem = execMem;
    }

    return HCCL_SUCCESS;
}

// 跨超节点
HcclResult
CollAllGatherPipelineFor91093Executor::KernelRunInterSuperPod(const OpParam& param, ExecMem& execMem, u64 baseOffset)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] begins, topoType_[%u], DMAReduceFlag_[%u]", __func__, topoType_, DMAReduceFlag_);
    std::unique_ptr<AlgTemplateBase> level2AGExecutor;
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        level2AGExecutor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL2", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        level2AGExecutor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL2", __func__);
    } else {
        level2AGExecutor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL2", __func__);
    }
    CHK_SMART_PTR_NULL(level2AGExecutor);

    u64 curDataSegsSliceSize = execMem.count * unitSize_;
    std::vector<Slice> level2DataSegsSlice
        = PrepareSlicesL2(param, level2CommInfo_, level1CommInfo_, level0CommInfo_, unitSize_, curDataSegsSliceSize);
    CHK_RET(level2AGExecutor->Prepare(
        execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count, param.DataDes.dataType, mainStreamL2_,
        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level2DataSegsSlice, baseOffset));

    CHK_RET(level2AGExecutor->RegisterProfiler(
        (level2CommInfo_.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo_.localRank, PROF_STAGE_0,
        HCCL_EXEC_STEP_NOT_SET, mainStreamL2_));

    CHK_RET(RunTemplate(level2AGExecutor, level2CommInfo_));
    HCCL_INFO("[%s] AllGather level2 AllGather run success, topoType_[%u]", __func__, topoType_);
    return HCCL_SUCCESS;
}

// 超节点内的节点间通信 L1nhr
HcclResult
CollAllGatherPipelineFor91093Executor::KernelRunInterServer(const OpParam& param, ExecMem& execMem, u64 baseOffset)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] begins, topoType_[%u], DMAReduceFlag_[%u]", __func__, topoType_, DMAReduceFlag_);
    u64 curDataSegsSliceSize = execMem.count * unitSize_;
    std::vector<Slice> level1DataSegsSlice
        = PrepareSlicesL1(param, level2CommInfo_, level1CommInfo_, level0CommInfo_, unitSize_, curDataSegsSliceSize);

    std::unique_ptr<AlgTemplateBase> level1AGExecutor;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1AGExecutor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL1", __func__);
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        level1AGExecutor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL1", __func__);
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        level1AGExecutor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL1", __func__);
    } else {
        HCCL_ERROR("AllGather ring: unsupported algtype [%s].", AlgTypeToStr(algType_).c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_SMART_PTR_NULL(level1AGExecutor);
    CHK_RET(level1AGExecutor->Prepare(
        execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count, param.DataDes.dataType, mainStreamL1L0_,
        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice, baseOffset));

    CHK_RET(level1AGExecutor->RegisterProfiler(
        (level1CommInfo_.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo_.localRank, PROF_STAGE_1,
        HCCL_EXEC_STEP_NOT_SET, mainStreamL1L0_));

    CHK_RET(RunTemplate(level1AGExecutor, level1CommInfo_));
    HCCL_INFO("[%s] AllGather level1 AllGather run success, topoType_[%u]", __func__, topoType_);
    return HCCL_SUCCESS;
}

// Server内的通信
HcclResult
CollAllGatherPipelineFor91093Executor::KernelRunIntraServer(const OpParam& param, ExecMem& execMem, u64 baseOffset)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] begins, topoType_[%u], DMAReduceFlag_[%u]", __func__, topoType_, DMAReduceFlag_);
    // 节点内做AllGather ring
    u64 curDataSegsSliceSize = execMem.count * unitSize_;
    std::vector<std::vector<Slice>> multRingsSlice;
    CHK_RET(PrepareSlicesL0(
        multRingsSlice, param, level2CommInfo_, level1CommInfo_, level0CommInfo_, unitSize_, curDataSegsSliceSize));

    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    CHK_RET(PrepareUserMemSlices(
        multRingsUserMemSlice, multRingsSlice, param, level2CommInfo_, level1CommInfo_, level0CommInfo_, unitSize_,
        curDataSegsSliceSize));

    // allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
    l0OpInfo_.inputAddr = nullptr;
    l0OpInfo_.outputAddr = execMem.outputPtr;
    l0OpInfo_.dataType = param.GetDataType();
    l0OpInfo_.count = execMem.count;
    l0OpInfo_.root = 0;
    l0OpInfo_.reduceOp = HCCL_REDUCE_RESERVED;
    l0OpInfo_.strideCount = param.DataDes.strideCount;

    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        CHK_RET(DoubleRingAllGather(
            param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, multRingsSlice,
            mainStreamL1L0_, PROF_STAGE_2, baseOffset, &l0OpInfo_, multRingsUserMemSlice));
    } else if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
        CHK_RET(MultiRingAllGather(
            param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, multRingsSlice,
            mainStreamL1L0_, PROF_STAGE_2, baseOffset, &l0OpInfo_, multRingsUserMemSlice, COMM_LEVEL0));
    } else {
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("[%s] AllGather level0 Ring run success, topoType_[%u]", __func__, topoType_);
    return HCCL_SUCCESS;
}

std::vector<Slice> CollAllGatherPipelineFor91093Executor::PrepareSlicesL1(
    [[maybe_unused]] const OpParam& param, const SubCommInfo& level2CommInfo, const SubCommInfo& level1CommInfo,
    const SubCommInfo& level0CommInfo, [[maybe_unused]] u32 perDataSize, u64 inputMemSize) const
{
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level0ServerIndex = level0CommInfo.localRank;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    std::vector<Slice> level1DataSegsSlice;
    for (u32 j = 0; j < level1RankSize; j++) {
        for (u32 i = 0; i < level2RankSize; i++) {
            Slice level1Slice;
            level1Slice.size = inputMemSize;
            level1Slice.offset
                = inputMemSize * (i * level1RankSize * level0RankSize + j * level0RankSize + level0ServerIndex);

            HCCL_DEBUG(
                "[CollAllGatherPipelineFor91093Executor][PrepareSlicesL1] rank[%u], level1index[%u], level2index[%u], "
                "slices.offset=%llu, slices.size=%llu",
                level0CommInfo.localRank, j, i, level1Slice.offset, level1Slice.size);

            level1DataSegsSlice.push_back(level1Slice);
        }
    }
    return level1DataSegsSlice;
}

std::vector<Slice> CollAllGatherPipelineFor91093Executor::PrepareSlicesL2(
    [[maybe_unused]] const OpParam& param, const SubCommInfo& level2CommInfo, const SubCommInfo& level1CommInfo,
    const SubCommInfo& level0CommInfo, [[maybe_unused]] u32 perDataSize, u64 inputMemSize) const
{
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level0ServerIndex = level0CommInfo.localRank;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level1ServerIndex = level1CommInfo.localRank;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    std::vector<Slice> level2DataSegsSlice;
    for (u32 i = 0; i < level2RankSize; i++) {
        Slice sliceTemp;
        sliceTemp.size = inputMemSize;
        sliceTemp.offset
            = inputMemSize
              * (i * level1RankSize * level0RankSize + level1ServerIndex * level0RankSize + level0ServerIndex);
        level2DataSegsSlice.push_back(sliceTemp);
    }
    return level2DataSegsSlice;
}

HcclResult CollAllGatherPipelineFor91093Executor::PrepareSlicesL0(
    std::vector<std::vector<Slice>>& multRingsSlice, const OpParam& param, const SubCommInfo& level2CommInfo,
    const SubCommInfo& level1CommInfo, const SubCommInfo& level0CommInfo, [[maybe_unused]] u32 perDataSize,
    u64 inputMemSize)
{
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;

    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareAllgatherSlice(level0RankSize, inputMemSize, dataSegsSlice));

    // 多环数据切分
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移

    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING
        && !IsSupportUnifiedMarch(param, topoType_, topoAttr_.serverNum, topoAttr_.superPodNum)) {
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice);
    }
    for (u32 ringIndex = 0; ringIndex < multRingsSliceZero.size(); ringIndex++) {
        std::vector<Slice> level2DataSlice;
        CHK_RET(CalculateLevel2AllgatherSlice(
            inputMemSize, level0RankSize, level1RankSize, level2RankSize, multRingsSliceZero, level2DataSlice,
            ringIndex));
        multRingsSlice.push_back(level2DataSlice);
    }

    return HCCL_SUCCESS;
}

HcclResult CollAllGatherPipelineFor91093Executor::PrepareUserMemSlices(
    std::vector<std::vector<Slice>>& userMemSlices, const std::vector<std::vector<Slice>>& multRingsSlice,
    const OpParam& param, [[maybe_unused]] const SubCommInfo& level2CommInfo,
    [[maybe_unused]] const SubCommInfo& level1CommInfo, [[maybe_unused]] const SubCommInfo& level0CommInfo,
    u32 perDataSize, u64 inputMemSize)
{
    CHK_PRT_RET(
        0 < param.DataDes.strideCount && param.DataDes.strideCount < param.DataDes.count,
        HCCL_ERROR(
            "[CollAllGatherPipelineFor91093Executor][KernelRun]strideCount[%llu] is smaller than opCount[%llu]",
            param.DataDes.strideCount, param.DataDes.count),
        HCCL_E_PARA);
    HCCL_DEBUG(
        "[CollAllGatherPipelineFor91093Executor][KernelRun]strideCount[%llu], opCount[%llu]", param.DataDes.strideCount,
        param.DataDes.count);

    for (u32 ringIndex = 0; ringIndex < multRingsSlice.size(); ringIndex++) {
        std::vector<Slice> userMemSlice;
        for (const auto& cclSlice : multRingsSlice[ringIndex]) {
            Slice tmpSlice;
            u64 count = (param.DataDes.strideCount == 0) ? param.DataDes.count : param.DataDes.strideCount;
            tmpSlice.size = cclSlice.size;
            tmpSlice.offset
                = (cclSlice.offset / inputMemSize) * count * perDataSize + multRingsSlice[ringIndex][0].offset;
            userMemSlice.push_back(tmpSlice);
            HCCL_DEBUG(
                "rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]", topoAttr_.userRank, ringIndex,
                tmpSlice.offset, tmpSlice.size);
        }
        userMemSlices.push_back(userMemSlice);
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherPipelineFor91093Executor::DoubleRingAllGather(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream, s32 profStage, const u64 baseOffset,
    const HcomCollOpInfo* opInfo, const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollAllGatherPipelineFor91093Executor]userRank[%u], count[%llu]", topoAttr_.userRank, count);

    (void)tag;
    HCCL_INFO("[CollAllGatherPipelineFor91093Executor][DoubleRingAllGather] DoubleRingAllGather starts");
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));
    // 拿到ring环映射关系
    SubCommInfo level0ZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder
        = GetRingsOrderByTopoType(level0ZeroCommInfo.localRankSize, topoType_, nicList);
    // 生成两个ring上的userMemOut_上对应的slices
    std::vector<std::vector<Slice>> userMemOutputSlicesOfDoubleRing;
    CHK_RET(CollectMultiRingsUserMemSlices(
        ringNum, dataType, opInfo, multRingsSliceZero, multiRingsOrder, multRingsUserMemSlice,
        userMemOutputSlicesOfDoubleRing));
    // 生成两个ring上的rankOrder
    std::vector<std::vector<u32>> rankOrders;
    CHK_RET(CollectMultiRingsRankOrder(ringNum, multiRingsOrder, rankOrders));
    // 初始化executor
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALIGNED_ALL_GATHER_DOUBLE_RING, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALIGNED_ALL_GATHER_DOUBLE_RING in COMM_LEVEL0", __func__);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(
        const_cast<HcomCollOpInfo*>(opInfo), topoAttr_.userRank, ringSubStreams_, notifyRingMain_, notifyRingSub_,
        rankOrders, userMemOutputSlicesOfDoubleRing));

    ret = tempAlg->Prepare(
        outputMem, outputMem, inputMem, count, dataType, stream, multRingsSliceZero, HCCL_REDUCE_RESERVED,
        LEVEL0_BRIDGE_RANK_ID, baseOffset);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAllGatherPipelineFor91093Executor][DoubleRingAllGather]Double ring "
            "AllGather failed, return[%d]",
            ret),
        ret);
    u32 ringIndexOp = COMM_INDEX_0;
    u32 rankSize = level0ZeroCommInfo.localRankSize;
    ret = tempAlg->RegisterProfiler(
        ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
            + level0ZeroCommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAllGatherPipelineFor91093Executor][DoubleRingAllGather]Double ring "
            "AllGather failed, return[%d]",
            ret),
        ret);

    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    ret = RunTemplate(tempAlg, level0ZeroCommInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAllGatherPipelineFor91093Executor][DoubleRingAllGather] Double ring "
            "AllGather failed, return[%d]",
            ret),
        ret);
    // 添加空task,保证执行时不乱序
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherPipelineFor91093Executor::GetSubStreamInfoOnOneRing(
    const u32 ringIndex, std::vector<Stream>& subStreamsInOneRing,
    std::vector<std::shared_ptr<LocalNotify>>& mainSignalsInOneRing,
    std::vector<std::shared_ptr<LocalNotify>>& subSignalsInOneRing)
{
    // slaveStreams 不包含主流，先 +1 补回主流，再减去 2 条 Pipeline 专用流，得到基类 ring 数。
    const u32 ringNum = algResResp_->slaveStreams.size() + 1 - PIPELINE_EXTRA_STREAM_NUM;
    if (ringNum == LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING) {
        subStreamsInOneRing.push_back(algResResp_->slaveStreams[ringIndex + 1]);
        mainSignalsInOneRing.push_back(algResResp_->notifiesMain[ringIndex + 1]);
        subSignalsInOneRing.push_back(algResResp_->notifiesAux[ringIndex + 1]);
    } else if (ringNum == LEVEL0_PLANE_NUM_IN_NPRING_SINGLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING) {
        subStreamsInOneRing.push_back(algResResp_->slaveStreams[ringIndex]);
        mainSignalsInOneRing.push_back(algResResp_->notifiesMain[ringIndex]);
        subSignalsInOneRing.push_back(algResResp_->notifiesAux[ringIndex]);
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherPipelineFor91093Executor", AllGatherPipelineFor91093, CollAllGatherPipelineFor91093Executor);

} // namespace hccl
