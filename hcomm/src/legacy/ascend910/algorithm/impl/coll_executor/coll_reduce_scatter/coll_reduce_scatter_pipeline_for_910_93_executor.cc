/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_pipeline_for_910_93_executor.h"
#include "alg_template_register.h"

namespace hccl {
namespace {
    constexpr u32 PIPELINE_BUFFER_NUM = 2;
    constexpr u32 PIPELINE_PINGPONG_NOTIFY_PAIRS = PIPELINE_BUFFER_NUM;
    // HCCL 按 stream:notify = 1:1 分配；pipeline 仅使用最后一条额外 stream 承载 L2，
    // 其余额外 stream 仅用于获取 ping-pong notify 对。
    constexpr u32 PIPELINE_EXTRA_STREAM_NUM = PIPELINE_PINGPONG_NOTIFY_PAIRS;
} // namespace

CollReduceScatterPipelineFor91093Executor::CollReduceScatterPipelineFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterRingFor91093Executor(dispatcher, topoMatcher)
{}

HcclResult CollReduceScatterPipelineFor91093Executor::CalcStreamNum(u32& streamNum)
{
    CHK_RET(CollReduceScatterRingFor91093Executor::CalcStreamNum(streamNum));
    streamNum += PIPELINE_EXTRA_STREAM_NUM;
    HCCL_INFO(
        "[CollReduceScatterPipelineFor91093Executor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

u64 CollReduceScatterPipelineFor91093Executor::CalcLoopMaxCount(const u32 unitSize)
{
    const u64 maxSizePerLoopUnaligned = inCCLbufferSize_ / topoAttr_.userRankSize / PIPELINE_BUFFER_NUM;
    const u64 maxSizePerLoop = maxSizePerLoopUnaligned / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    const u64 maxCountPerLoop = maxSizePerLoop / unitSize;
    HCCL_INFO(
        "[CollReduceScatterPipelineFor91093Executor][CalcLoopMaxCount] "
        "maxCountPerLoop[%llu], maxSizePerLoop[%llu]",
        maxCountPerLoop, maxSizePerLoop);
    return maxCountPerLoop;
}

HcclResult CollReduceScatterPipelineFor91093Executor::RunLoop(OpParam& param, AlgResourceResponse& algRes)
{
    if (param.DataDes.count == 0) {
        return HCCL_SUCCESS;
    }

    const u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    Stream streamL0L1 = param.stream;
    Stream streamL2 = algResResp_->slaveStreams.back();
    const u32 baseNotifyIndex = algResResp_->notifiesMain.size() - PIPELINE_PINGPONG_NOTIFY_PAIRS;
    auto notifyL0L1toL2A = algResResp_->notifiesAux[baseNotifyIndex];
    auto notifyL0L1toL2B = algResResp_->notifiesAux[baseNotifyIndex + 1];
    auto notifyL2toL0L1A = algResResp_->notifiesMain[baseNotifyIndex];
    auto notifyL2toL0L1B = algResResp_->notifiesMain[baseNotifyIndex + 1];
    HCCL_INFO(
        "[CollReduceScatterPipelineFor91093Executor][RunLoop] NotifyIds: "
        "L0L1toL2A: Aux[%u], L0L1toL2B: Aux[%u], L2toL0L1A: Main[%u], L2toL0L1B: Main[%u]",
        baseNotifyIndex, baseNotifyIndex + 1, baseNotifyIndex, baseNotifyIndex + 1);
    PipelineLoopContext ctx;
    CHK_RET(BuildPipelineLoopContext(param, algRes, unitSize, ctx));
    CHK_RET(GetLevelCommInfo());

    auto getForwardNotify = [&](u64 blockIdx) -> std::shared_ptr<LocalNotify> {
        return (blockIdx % PIPELINE_PINGPONG_NOTIFY_PAIRS == 0) ? notifyL0L1toL2A : notifyL0L1toL2B;
    };
    auto getBackwardNotify = [&](u64 blockIdx) -> std::shared_ptr<LocalNotify> {
        return (blockIdx % PIPELINE_PINGPONG_NOTIFY_PAIRS == 0) ? notifyL2toL0L1A : notifyL2toL0L1B;
    };

    const u64 numLoopTotal = ctx.numBlockTotal + 1;
    for (u64 i = 0; i < numLoopTotal; ++i) {
        if (i < ctx.numBlockTotal) {
            if (i >= PIPELINE_BUFFER_NUM) {
                CHK_RET(LocalNotify::Wait(streamL0L1, dispatcher_, getBackwardNotify(i)));
            }
            CHK_RET(RunL0L1Phase(param, ctx, i, streamL0L1));
            CHK_RET(LocalNotify::Post(streamL0L1, dispatcher_, getForwardNotify(i)));
        }
        if (i >= 1 && i <= ctx.numBlockTotal) {
            const u64 blockIdx = i - 1;
            CHK_RET(LocalNotify::Wait(streamL2, dispatcher_, getForwardNotify(blockIdx)));
            CHK_RET(RunL2Phase(param, ctx, blockIdx, streamL2));
            CHK_RET(LocalNotify::Post(streamL2, dispatcher_, getBackwardNotify(blockIdx)));
        }
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }

    CHK_RET(WaitForRemainingL2Signals(param, ctx.numBlockTotal, streamL0L1, notifyL2toL0L1A, notifyL2toL0L1B));
    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    HCCL_INFO("[CollReduceScatterPipelineFor91093Executor][RunLoop] Pipeline run success");
    return HCCL_SUCCESS;
}

// 由 RunLoop 调用
HcclResult CollReduceScatterPipelineFor91093Executor::BuildPipelineLoopContext(
    OpParam& param, AlgResourceResponse& algRes, const u32 unitSize, PipelineLoopContext& ctx)
{
    u8* curInputPtr = static_cast<u8*>(param.inputPtr);
    u8* curOutputPtr = static_cast<u8*>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    const u64 maxCountDataPerLoop = CalcLoopMaxCount(unitSize);
    const u64 targetCountDataPerLoop = HCCL_SMALL_COUNT_8_MB / unitSize;
    const u64 countDataPerLoop
        = maxCountDataPerLoop < targetCountDataPerLoop ? maxCountDataPerLoop : targetCountDataPerLoop;
    CHK_PRT_RET(
        countDataPerLoop == 0,
        HCCL_ERROR("[CollReduceScatterPipelineFor91093Executor][BuildPipelineLoopContext]"
                   " countDataPerLoop is zero."),
        HCCL_E_INTERNAL);

    const u64 countDataLastLoopTemp = param.DataDes.count % countDataPerLoop;
    const u64 countDataLastLoop = countDataLastLoopTemp > 0 ? countDataLastLoopTemp : countDataPerLoop;
    const u64 cclInputBufferSize = algRes.cclInputMem.size() / PIPELINE_BUFFER_NUM;
    const u64 cclOutputBufferSize = algRes.cclOutputMem.size() / PIPELINE_BUFFER_NUM;
    ctx.countDataPerLoop = countDataPerLoop;
    ctx.countDataLastLoop = countDataLastLoop;
    ctx.sizeDataPerLoop = countDataPerLoop * unitSize;
    ctx.numBlockTotal = (param.DataDes.count - countDataLastLoop) / countDataPerLoop + 1;
    ctx.cclInputBufferSize = cclInputBufferSize;
    ctx.cclInputAMem = algRes.cclInputMem.range(0, cclInputBufferSize);
    ctx.cclInputBMem = algRes.cclInputMem.range(cclInputBufferSize, cclInputBufferSize);
    ctx.cclOutputAMem = algRes.cclOutputMem.range(0, cclOutputBufferSize);
    ctx.cclOutputBMem = algRes.cclOutputMem.range(cclOutputBufferSize, cclOutputBufferSize);
    ctx.curInputPtr = curInputPtr;
    ctx.curOutputPtr = curOutputPtr;

    HCCL_INFO(
        "[CollReduceScatterPipelineFor91093Executor][BuildPipelineLoopContext] "
        "tag[%s] numBlockTotal[%llu] numLoopTotal[%llu] maxCountDataPerLoop[%llu] "
        "targetCountDataPerLoop[%llu] countDataPerLoop[%llu] countDataLastLoop[%llu]",
        param.tag.c_str(), ctx.numBlockTotal, ctx.numBlockTotal + 1, maxCountDataPerLoop, targetCountDataPerLoop,
        ctx.countDataPerLoop, ctx.countDataLastLoop);
    return HCCL_SUCCESS;
}

// 由 RunLoop 调用
HcclResult CollReduceScatterPipelineFor91093Executor::WaitForRemainingL2Signals(
    const OpParam& param, u64 numBlockTotal, Stream& streamL0L1, const std::shared_ptr<LocalNotify>& notifyL2toL0L1A,
    const std::shared_ptr<LocalNotify>& notifyL2toL0L1B)
{
    const u64 remainingSignals
        = (numBlockTotal >= PIPELINE_PINGPONG_NOTIFY_PAIRS) ? PIPELINE_PINGPONG_NOTIFY_PAIRS : numBlockTotal;
    const u64 firstBlockIdx = numBlockTotal - remainingSignals;
    for (u64 blockIdx = firstBlockIdx; blockIdx < numBlockTotal; ++blockIdx) {
        auto notify = (blockIdx % PIPELINE_PINGPONG_NOTIFY_PAIRS == 0) ? notifyL2toL0L1A : notifyL2toL0L1B;
        HcclResult ret = LocalNotify::Wait(streamL0L1, dispatcher_, notify);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollReduceScatterPipelineFor91093Executor][WaitForRemainingL2Signals] "
                "PostSync wait error, tag[%s] blockIdx[%llu]",
                param.tag.c_str(), blockIdx),
            ret);
    }
    return HCCL_SUCCESS;
}

// 由 RunLoop 循环体调用
HcclResult CollReduceScatterPipelineFor91093Executor::RunL0L1Phase(
    OpParam& param, const PipelineLoopContext& ctx, u64 blockIdx, Stream& streamL0L1)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollReduceScatterPipelineFor91093Executor][RunL0L1Phase] blockIdx[%llu] useBufferA[%d]", blockIdx,
        (blockIdx % PIPELINE_BUFFER_NUM == 0));

    const bool useBufferA = (blockIdx % PIPELINE_BUFFER_NUM == 0);
    const bool isLastBlock = (blockIdx == ctx.numBlockTotal - 1);
    ExecMem execMem;
    execMem.count = isLastBlock ? ctx.countDataLastLoop : ctx.countDataPerLoop;
    execMem.inputMem = useBufferA ? ctx.cclInputAMem : ctx.cclInputBMem;
    execMem.outputMem = useBufferA ? ctx.cclOutputAMem : ctx.cclOutputBMem;
    execMem.scratchMem = execMem.outputMem;
    execMem.inputPtr = ctx.curInputPtr + blockIdx * ctx.sizeDataPerLoop;
    execMem.outputPtr = ctx.curOutputPtr + blockIdx * ctx.sizeDataPerLoop;

    const u64 bufferBaseOffset = useBufferA ? 0 : ctx.cclInputBufferSize;
    SliceExecMem(param, execMem);

    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterPipelineFor91093Executor][RunL0L1Phase] chunk starts");

    HcclResult ret = KernelRunLevel0To1(param, execMem, streamL0L1, bufferBaseOffset);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterPipelineFor91093Executor][RunL0L1Phase] kernel run error, tag[%s]", param.tag.c_str()),
        ret);
    return HCCL_SUCCESS;
}

// 由 RunLoop 循环体调用
HcclResult CollReduceScatterPipelineFor91093Executor::RunL2Phase(
    OpParam& param, const PipelineLoopContext& ctx, u64 blockIdx, Stream& streamL2)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollReduceScatterPipelineFor91093Executor][RunL2Phase] blockIdx[%llu] L2 phase", blockIdx);

    const bool useBufferA = (blockIdx % PIPELINE_BUFFER_NUM == 0);
    const bool isLastBlock = (blockIdx == ctx.numBlockTotal - 1);
    ExecMem execMem;
    execMem.count = isLastBlock ? ctx.countDataLastLoop : ctx.countDataPerLoop;
    execMem.inputMem = useBufferA ? ctx.cclInputAMem : ctx.cclInputBMem;
    execMem.outputMem = useBufferA ? ctx.cclOutputAMem : ctx.cclOutputBMem;
    execMem.scratchMem = execMem.outputMem;
    execMem.inputPtr = ctx.curInputPtr + blockIdx * ctx.sizeDataPerLoop;
    execMem.outputPtr = ctx.curOutputPtr + blockIdx * ctx.sizeDataPerLoop;

    const u64 l2BaseOffset = useBufferA ? 0 : ctx.cclInputBufferSize;
    SliceExecMem(param, execMem);

    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterPipelineFor91093Executor][RunL2Phase] chunk starts");

    HcclResult ret = KernelRunLevel2(param, execMem, streamL2, l2BaseOffset);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterPipelineFor91093Executor][RunL2Phase] kernel run error, tag[%s]", param.tag.c_str()),
        ret);
    return HCCL_SUCCESS;
}

// 由 RunL0L1Phase、RunL2Phase 调用
void CollReduceScatterPipelineFor91093Executor::SliceExecMem(const OpParam& param, ExecMem& execMem) const
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 curSize = execMem.count * unitSize;
    u32 sliceNum = topoAttr_.userRankSize;
    execMem.inputMem = execMem.inputMem.range(0, curSize * sliceNum);
    execMem.outputMem = execMem.outputMem.range(0, curSize);
}

// 由 KernelRunLevel0To1、KernelRunLevel2 调用
// Pipeline 约束 !isAHCAlgo，AHC 分支不可达，直接走 COMM_LEVEL2。
HcclResult CollReduceScatterPipelineFor91093Executor::GetLevel2CommInfo(SubCommInfo& level2CommInfo)
{
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    return HCCL_SUCCESS;
}

u32 CollReduceScatterPipelineFor91093Executor::GetLevel0RingNum() const
{
    // slaveStreams 不包含主流，先 +1 补回主流，再减去 2 条 Pipeline 专用流，得到基类 ring 数。
    return algResResp_->slaveStreams.size() + 1 - PIPELINE_EXTRA_STREAM_NUM;
}

HcclResult CollReduceScatterPipelineFor91093Executor::RunIntraSeverReduceScatter(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count, const HcclDataType& dataType,
    const HcclReduceOp& reductionOp, const std::vector<std::vector<Slice>>& multRingsSliceZero, const Stream& stream,
    s32 profStage, const u64 baseOffset, const HcomCollOpInfo* opInfo,
    const std::vector<std::vector<Slice>>& multRingsUserMemSlice, const bool disableDMAReduce)
{
    // SemiRing（IsUnifiedMarch）分支不可达：Pipeline 约束 workflowMode_==OP_BASE 排除图模式，
    // superPodNum>1 排除单 server，IsUnifiedMarch 恒为 false。
    HcclResult ret = HCCL_SUCCESS;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        ret = DoubleRingReduceScatter(
            tag, inputMem, outputMem, count, dataType, reductionOp, multRingsSliceZero, stream, profStage, baseOffset,
            opInfo, multRingsUserMemSlice, disableDMAReduce);
    } else {
        ret = CollReduceScatterRingFor91093Executor::RunIntraSeverReduceScatter(
            tag, inputMem, outputMem, count, dataType, reductionOp, multRingsSliceZero, stream, profStage, baseOffset,
            opInfo, multRingsUserMemSlice, disableDMAReduce);
    }

    CHK_RET(ret);
    return HCCL_SUCCESS;
}

// 逻辑与 CollAlignedReduceScatterDoubleRingFor91093Executor::DoubleRingReduceScatter 一致
HcclResult CollReduceScatterPipelineFor91093Executor::DoubleRingReduceScatter(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream,
    s32 profStage, const u64 baseOffset, const HcomCollOpInfo* opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice, const bool disableDMAReduce)
{
    (void)tag;
    HCCL_CONFIG_INFO(
        HCCL_ALG,
        "[CollReduceScatterPipelineFor91093Executor][DoubleRingReduceScatter] DoubleRingReduceScatter starts");
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);
    SubCommInfo level0RingCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    std::vector<std::vector<Slice>> userMemInputSlicesOfDoubleRing;
    std::vector<std::vector<u32>> rankOrders;
    CHK_RET(PrepareDoubleRingSlices(
        ringNum, dataType, opInfo, multRingsSliceZero, multRingsUserMemSlice, userMemInputSlicesOfDoubleRing,
        rankOrders));

    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_DB_RING, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_DB_RING in COMM_LEVEL0", __func__);
    CHK_SMART_PTR_NULL(tempAlg);
    // 排除尾部 Pipeline 专用资源流（NotifyReserve + StreamL2），使模板看到与基类一致的流数
    std::vector<Stream> baseSlaveStreams(
        algResResp_->slaveStreams.begin(), algResResp_->slaveStreams.end() - PIPELINE_EXTRA_STREAM_NUM);
    std::vector<std::shared_ptr<LocalNotify>> baseNotifiesMain(
        algResResp_->notifiesMain.begin(), algResResp_->notifiesMain.end() - PIPELINE_PINGPONG_NOTIFY_PAIRS);
    std::vector<std::shared_ptr<LocalNotify>> baseNotifiesAux(
        algResResp_->notifiesAux.begin(), algResResp_->notifiesAux.end() - PIPELINE_PINGPONG_NOTIFY_PAIRS);
    HcclResult ret = tempAlg->Prepare(
        inputMem, inputMem, outputMem, count, dataType, stream, multRingsSliceZero, reductionOp, LEVEL0_BRIDGE_RANK_ID,
        baseOffset, disableDMAReduce, reduceAttr, opInfo, topoAttr_.userRank, baseSlaveStreams, baseNotifiesMain,
        baseNotifiesAux, rankOrders, userMemInputSlicesOfDoubleRing);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterPipelineFor91093Executor][DoubleRingReduceScatter] "
            "Double ring ReduceScatter failed,return[%d]",
            ret),
        ret);

    u32 ringIndexOp = COMM_INDEX_0;
    u32 rankSize = level0RingCommInfo.localRankSize;
    ret = tempAlg->RegisterProfiler(
        ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
            + level0RingCommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterPipelineFor91093Executor][DoubleRingReduceScatter] "
            "Double ring ReduceScatter RegisterProfiler failed,return[%d]",
            ret),
        ret);

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    ret = RunTemplate(tempAlg, level0RingCommInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceScatterPipelineFor91093Executor][DoubleRingReduceScatter] "
            "Double ring ReduceScatter RunTemplate failed,return[%d]",
            ret),
        ret);

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

// 由 DoubleRingReduceScatter 调用
HcclResult CollReduceScatterPipelineFor91093Executor::PrepareDoubleRingSlices(
    u32 ringNum, const HcclDataType dataType, const HcomCollOpInfo* opInfo,
    const std::vector<std::vector<Slice>>& multRingsSliceZero,
    const std::vector<std::vector<Slice>>& multRingsUserMemSlice,
    std::vector<std::vector<Slice>>& userMemInputSlicesOfDoubleRing, std::vector<std::vector<u32>>& rankOrders)
{
    SubCommInfo level0ZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder
        = GetRingsOrderByTopoType(level0ZeroCommInfo.localRankSize, topoType_, nicList);
    CHK_RET(CollectMultiRingsUserMemSlices(
        ringNum, dataType, opInfo, multRingsSliceZero, multiRingsOrder, multRingsUserMemSlice,
        userMemInputSlicesOfDoubleRing));
    CHK_RET(CollectMultiRingsRankOrder(ringNum, multiRingsOrder, rankOrders));
    return HCCL_SUCCESS;
}

// 拆分自 CollReduceScatterRingFor91093Executor::KernelRun 的 L0+L1 部分
HcclResult CollReduceScatterPipelineFor91093Executor::KernelRunLevel0To1(
    const OpParam& param, ExecMem& execMem, Stream& streamL0L1, const u64 baseOffset)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] executor starts", __func__);
    u32 perDataSize = 0;
    const HcclDataType dataType = param.GetDataType();
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    u32 ringNum;
    u32 sliceNum = logicalLevel0CommInfo_.localRankSize;
    u32 commIndex = logicalLevel0CommInfo_.localRank;

    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        ringNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE;
    } else {
        ringNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
    }

    SubCommInfo level2CommInfo;
    CHK_RET(GetLevel2CommInfo(level2CommInfo));
    const u32 level2RankSize = level2CommInfo.localRankSize;
    const u32 level1RankSize = logicalLevel1CommInfo_.localRankSize;

    std::vector<std::vector<Slice>> multiStreamSlice;
    std::vector<std::vector<Slice>> level0DataSegsSlice;
    CHK_RET(CalLevel0DataSegsSlice(
        execMem, multiStreamSlice, param, ringNum, sliceNum, level1RankSize, level2RankSize, dataType,
        level0DataSegsSlice));

    HcomCollOpInfo opInfo = GetHcomCollOpInfo(param, execMem);
    HcomCollOpInfo* opInfoPtr = &opInfo;

    bool disableDMAReduce = algOpContext_.opRetryHandler.retryEnable
                            && (algOpContext_.opRetryHandler.inPlaceSupportRetryStatus
                                    == InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE1
                                || algOpContext_.opRetryHandler.inPlaceSupportRetryStatus
                                       == InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE2);
    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    CHK_RET(CalUserMemDataSegsSlice(
        execMem, level0DataSegsSlice, multiStreamSlice, param, ringNum, sliceNum, level1RankSize, level2RankSize,
        dataType, perDataSize, opInfoPtr, disableDMAReduce, multRingsUserMemSlice));

    HcomCollOpInfo opInfoByReduceScatterDMAreduce = *opInfoPtr;
    opInfoByReduceScatterDMAreduce.outputAddr = nullptr;
    CHK_RET(RunIntraSeverReduceScatter(
        param.tag, execMem.inputMem, execMem.scratchMem, execMem.count, dataType, param.reduceType, level0DataSegsSlice,
        streamL0L1, PROF_STAGE_1, baseOffset, &opInfoByReduceScatterDMAreduce, multRingsUserMemSlice,
        disableDMAReduce));

    if (level1RankSize > 1) {
        CHK_RET(RunLevel1Template(
            param, execMem, streamL0L1, baseOffset, commIndex, sliceNum, level1RankSize, level2RankSize, perDataSize));
    }

    return HCCL_SUCCESS;
}

// 由 KernelRunLevel0To1 调用
HcclResult CollReduceScatterPipelineFor91093Executor::RunLevel1Template(
    const OpParam& param, ExecMem& execMem, Stream& streamL0L1, u64 baseOffset, u32 commIndex, u32 sliceNum,
    u32 level1RankSize, u32 level2RankSize, u32 perDataSize)
{
    const HcclDataType dataType = param.GetDataType();
    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, dataType, param.reduceType);
    std::unique_ptr<AlgTemplateBase> level1TempAlg;

    std::vector<Slice> level1DataSegsSlice;
    CHK_RET(CalLevel1DataSegsSlice(
        execMem, param, logicalLevel1plane_, commIndex, sliceNum, level1RankSize, level2RankSize, perDataSize,
        level1DataSegsSlice));

    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL1", __func__);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NB in COMM_LEVEL1", __func__);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    } else {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NHR in COMM_LEVEL1", __func__);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr, false));
    }

    CHK_RET(level1TempAlg->Prepare(
        execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count, dataType, streamL0L1, param.reduceType,
        LEVEL0_BRIDGE_RANK_ID, level1DataSegsSlice, baseOffset));
    CHK_RET(level1TempAlg->RegisterProfiler(
        (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + logicalLevel1CommInfo_.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, streamL0L1));
    CHK_RET(RunTemplate(level1TempAlg, logicalLevel1CommInfo_));
    return HCCL_SUCCESS;
}

// 拆分自 CollReduceScatterRingFor91093Executor::KernelRun 的 L2+copyOut 部分，stream 替换为 streamL2
HcclResult CollReduceScatterPipelineFor91093Executor::KernelRunLevel2(
    const OpParam& param, ExecMem& execMem, Stream& streamL2, const u64 baseOffset)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] executor starts", __func__);
    u32 perDataSize = 0;
    const HcclDataType dataType = param.GetDataType();
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    SubCommInfo level2CommInfo;
    CHK_RET(GetLevel2CommInfo(level2CommInfo));
    const u32 level2RankSize = level2CommInfo.localRankSize;

    CHK_RET(RunLevel2Template(param, execMem, streamL2, baseOffset, level2CommInfo, level2RankSize, perDataSize));

    HcomCollOpInfo opInfo = GetHcomCollOpInfo(param, execMem);
    HcomCollOpInfo* opInfoPtr = &opInfo;

    const u64 offset = CalcSrcMemOffset(execMem, param, perDataSize);
    DeviceMem srcMem = execMem.inputMem.range(offset, execMem.outputMem.size());
    if (opInfoPtr != nullptr) {
        DeviceMem dstMem = DeviceMem::create(static_cast<u8*>(opInfoPtr->outputAddr), execMem.outputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, streamL2));
    } else {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, streamL2));
    }

    return HCCL_SUCCESS;
}

// 由 KernelRunLevel2 调用
HcclResult CollReduceScatterPipelineFor91093Executor::RunLevel2Template(
    const OpParam& param, ExecMem& execMem, Stream& streamL2, u64 baseOffset, const SubCommInfo& level2CommInfo,
    u32 level2RankSize, u32 perDataSize)
{
    const HcclDataType dataType = param.GetDataType();
    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, dataType, param.reduceType);

    std::vector<Slice> level2DataSegsSlice;
    CHK_RET(CalLevel2DataSegsSlice(execMem, param, level2RankSize, perDataSize, level2DataSegsSlice));

    std::unique_ptr<AlgTemplateBase> level2TempAlg;
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        level2TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NB in COMM_LEVEL2", __func__);
        CHK_SMART_PTR_NULL(level2TempAlg);
        CHK_RET(level2TempAlg->Prepare(reduceAttr));
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        level2TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NHR in COMM_LEVEL2", __func__);
        CHK_SMART_PTR_NULL(level2TempAlg);
        CHK_RET(level2TempAlg->Prepare(reduceAttr, false));
        if (algoAttr_.isSupportAtomicWrite) {
            level2TempAlg->CloseBarrier();
        }
    } else {
        level2TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL2", __func__);
        CHK_SMART_PTR_NULL(level2TempAlg);
        CHK_RET(level2TempAlg->Prepare(reduceAttr));
    }

    CHK_RET(level2TempAlg->Prepare(
        execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count, dataType, streamL2, param.reduceType,
        LEVEL0_BRIDGE_RANK_ID, level2DataSegsSlice, baseOffset));
    CHK_RET(level2TempAlg->RegisterProfiler(
        (level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, streamL2));
    CHK_RET(RunTemplate(level2TempAlg, level2CommInfo));
    return HCCL_SUCCESS;
}

REGISTER_EXEC(
    "ReduceScatterPipelineFor91093Executor", ReduceScatterPipelineFor91093, CollReduceScatterPipelineFor91093Executor);
} // namespace hccl
