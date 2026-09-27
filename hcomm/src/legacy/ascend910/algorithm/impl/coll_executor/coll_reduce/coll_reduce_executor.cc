/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_executor.h"

namespace hccl {

CollReduceExecutor::CollReduceExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollCommExecutor(dispatcher, topoMatcher)
{}

HcclResult CollReduceExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();

    tag_ = param.tag;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD || algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD) {
        std::string appendTag = "";
        u32 serverNumPerSuperPod
            = topoAttr_.superPodNum == 0 ? topoAttr_.moduleNum : topoAttr_.moduleNum / topoAttr_.superPodNum;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
            u32 part1Size = FACTOR_TWO * (serverNumPerSuperPod - (1 << static_cast<u32>(log2(serverNumPerSuperPod))));
            u32 rootId = param.root / topoAttr_.deviceNumPerAggregation % serverNumPerSuperPod;
            appendTag += "L1_" + std::to_string((rootId >= part1Size) || ((rootId % FACTOR_TWO) == 0));
        }
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD) {
            u32 part1Size = FACTOR_TWO * (topoAttr_.superPodNum - (1 << static_cast<u32>(log2(topoAttr_.superPodNum))));
            u32 rootId = param.root / topoAttr_.deviceNumPerAggregation / serverNumPerSuperPod;
            appendTag += (appendTag.empty() ? "L2_" : "_L2_")
                         + std::to_string((rootId >= part1Size) || ((rootId % FACTOR_TWO) == 0));
        }
        tag_ = param.tag + '_' + appendTag;
        if (param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, param.tag));
        }
    }

    algResResp_ = &algRes;
    HcclResult ret = HCCL_SUCCESS;
    bool needLaunchAtTheEnd = true; // 是否需要在Orchestrate()结束时launch任务
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    // 图模式和单卡场景下不需要Loop
    HCCL_DEBUG("[CollReduceExecutor][Orchestrate]workflowMode is %d", workflowMode_);
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        ret = KernelRun(param, execMem);
        if (algOpContext_.opRetryHandler.isPostSync == true) {
            // post Sync
            CHK_RET(RetryPostSync(param, execMem));
        }
    } else if (topoAttr_.userRankSize == 1) {
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        ret = KernelRun(param, execMem);
        needLaunchAtTheEnd = false;
    } else {
        ret = RunLoop(param, algRes);
        needLaunchAtTheEnd = false;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceExecutor][Orchestrate]errNo[0x%016llx]reduce executor kernel run failed", HCCL_ERROR_CODE(ret)),
        ret);

    // Enforce task launch at the end of Orchestrate
    // 注意: 不要删除这里的强制launch, 否则会导致aicpu cache功能问题
    if (needLaunchAtTheEnd) {
        HCCL_INFO("%s: enforce task launch at the end of Orchestrate", __func__);
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }

    HCCL_INFO(
        "tag[%s], Reduce executor orchestrate success, take time [%lld]us.", tag_.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollReduceExecutor::RunLoop(OpParam& param, AlgResourceResponse& algRes)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    ReduceType reduceType
        = ((param.reduceType != HCCL_REDUCE_PROD) && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
              ReduceType::INLINE_REDUCE :
              ReduceType::TBE_REDUCE;

    u8* curInputPtr = static_cast<u8*>(param.inputPtr);
    u8* curOutputPtr = static_cast<u8*>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(unitSize, algRes); // override

    HCCL_DEBUG(
        "[CollReduceExecutor][RunLoop]tag[%s], userRankSize is [%u], maxCountPerLoop is [%llu].", tag_.c_str(),
        topoAttr_.userRankSize, maxCountPerLoop);

    u64 inputOffset = 0;
    u64 outputOffset = 0;
    u64 countLeft = param.DataDes.count;
    while (countLeft > 0) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        // 判断剩余数据量对应的output size是否大于中转output size
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        HCCL_DEBUG(
            "[CollReduceExecutor][RunLoop]tag[%s], inputOffset[%llu], outputOffset[%llu], "
            "sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%d].",
            tag_.c_str(), inputOffset, outputOffset, curInputPtr, curOutputPtr, curCount, param.DataDes.dataType);

        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;

        CHK_RET(RunLoopInner(param, reduceType, execMem));

        countLeft -= curCount;
        inputOffset = curSize;
        outputOffset = curSize;
    }
    if (algOpContext_.opRetryHandler.isPostSync == true) {
        ExecMem execMem;
        execMem.count = param.DataDes.count;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        // post Sync
        CHK_RET(RetryPostSync(param, execMem));
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceExecutor::RunLoopInner(OpParam& param, const ReduceType& reduceType, ExecMem& execMem)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 curSize = execMem.count * unitSize; // 单位：字节
    HCCL_DEBUG(
        "[CollReduceExecutor][RunLoopInner]inputMem[%p][%llu], outputMem[%p][%llu], "
        "inputPtr[%p], outputPtr[%p], curCount[%llu], curSize[%llu]",
        execMem.inputMem.ptr(), execMem.inputMem.size(), execMem.outputMem.ptr(), execMem.outputMem.size(),
        execMem.inputPtr, execMem.outputPtr, execMem.count, curSize);
    CHK_PRT_RET(
        (execMem.count == 0), HCCL_ERROR("[CollReduceExecutor][RunLoopInner]In OP_BASE curCount is zero."),
        HCCL_E_PARA);

    /* 设置子图复用标志 */
    bool isRootRank = param.root == topoAttr_.realUserRank ? true : false;
    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    bool hugeData = IsHugeData(curSize); // override
    /* TBE reduce 当总count数超过INT32_MAX时，不使能子图复用 */
    if (reduceType == ReduceType::TBE_REDUCE) {
        hugeData = hugeData || param.DataDes.count > INT32_MAX;
    }
    HCCL_DEBUG("[CollReduceExecutor][RunLoopInner]IsHugeData:[%u]", hugeData);
    u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();
    auto opMeta = HcclOpMetaInfo::GetOneForReduce(
        isRootRank, param.root, autoSelectedAlgTypeLevel1, param.DataDes.dataType, reduceType, hugeData, deterministic);
    CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));

    execMem.inputMem = DeviceMem::create(execMem.inputMem.ptr(), curSize);
    execMem.outputMem = DeviceMem::create(execMem.outputMem.ptr(), curSize);

    // 执行
    // 如果使用in CCL buffer，需要将user buffer in中的结果拷贝到CCL buffer in
    DeviceMem inMem(execMem.inputPtr, curSize);
    DeviceMem inCommMem = execMem.inputMem.range(0, curSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, param.stream));
    HCCL_DEBUG("[CollReduceExecutor][RunLoopInner]copy from user in to ccl in.");

    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollReduceExecutor][RunLoopInner]errNo[0x%016llx]kernel run error, tag[%s], "
            "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d], reduce op type[%d]",
            HCCL_ERROR_CODE(ret), tag_.c_str(), execMem.inputMem.ptr(), execMem.outputMem.ptr(), execMem.count,
            param.DataDes.dataType, param.reduceType),
        ret);

    if (topoAttr_.realUserRank == param.root) { // 只root rank需要把数据从中转内存拷贝出去
        DeviceMem outMem(execMem.outputPtr, curSize);
        DeviceMem outCommMem = execMem.outputMem.range(0, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outMem, outCommMem, param.stream));
    }

    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    return ret;
}

u64 CollReduceExecutor::CalcLoopMaxCount(const u32 unitSize, const AlgResourceResponse& algRes)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = algRes.cclInputMem.size() / unitSize;
    HCCL_WARNING(
        "[CollReduceExecutor][CalcLoopMaxCount]"
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.",
        maxCountPerLoop);
    return maxCountPerLoop;
}

bool CollReduceExecutor::IsHugeData(const u64 curSize)
{
    HCCL_WARNING("[CollReduceExecutor][IsHugeData]opMeta is using the default option.");
    bool hugeData = (curSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE) || (curSize > SDMA_SEND_MAX_SIZE);
    return hugeData;
}

HcclResult CollReduceExecutor::RetryPostSync(OpParam& param, ExecMem& execMem)
{
    if ((algResResp_->slaveStreams).size() == 0) {
        CHK_RET(PostSyncWithoutSubstream(param, execMem));
    } else {
        PrepareData postSyncPrepareData;
        postSyncPrepareData.subStreamsPtr = &algResResp_->slaveStreams;
        postSyncPrepareData.signalPtr = &algResResp_->notifiesMain;
        postSyncPrepareData.signalAuxPtr = &algResResp_->notifiesAux;
        postSyncPrepareData.stream = param.stream;
        CHK_RET(PostSyncWithSubstream(param, execMem, postSyncPrepareData));
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
