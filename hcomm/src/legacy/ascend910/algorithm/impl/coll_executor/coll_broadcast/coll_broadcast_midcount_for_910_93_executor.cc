/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_midcount_for_910_93_executor.h"

namespace hccl {
CollBroadcastMidCountFor91093Executor::CollBroadcastMidCountFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
    desc_.level1SupportedAlgos = {
        AlgTypeLevel1::ALG_LEVEL1_NHR,
    };
    desc_.level2SupportedAlgos = {
        AlgTypeLevel2::ALG_LEVEL2_NHR,
    };
}
HcclResult CollBroadcastMidCountFor91093Executor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0;
    HCCL_INFO("[CollBroadcastCommExecutor][CalcStreamNum]tag[%s] streamNum_ is [%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMidCountFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMidCountFor91093Executor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        HCCL_ERROR("BroadcastMidCountFor91093Executor do not support offload mode");
        return HCCL_E_UNAVAIL;
    }
    HCCL_INFO(
        "[CollBroadcastMidCountFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMidCountFor91093Executor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaCombineL1(COMM_COMBINE_L1, CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaCombineL1, opTransport[COMM_COMBINE_L1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMidCountFor91093Executor::CalcLevel2CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollBroadcastMidCountFor91093Executor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / HCCL_MIN_SLICE_ALIGN_910_93 * HCCL_MIN_SLICE_ALIGN_910_93 / unitSize;
    if (maxCountPerLoop == 0) {
        HCCL_ERROR("[CollBroadcastMidCountFor91093Executor][CalcLoopMaxCount] cclbuffer size is too small");
    }
    return maxCountPerLoop;
}

HcclResult CollBroadcastMidCountFor91093Executor::RunLevel2ByNHR(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level1CommInfo, SubCommInfo& level2CommInfo) const
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[MidCountBroadcast][RunLevel2ByNHR] userRank[%u] starts.", topoAttr_.userRank);

    u32 unitSize = 0;
    const HcclDataType dataType = param.GetDataType();
    CHK_RET(SalGetDataTypeSize(dataType, unitSize));

    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR_ONESHOT, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NHR_ONESHOT in COMM_LEVEL2", __func__);
    CHK_SMART_PTR_NULL(tempAlg);

    // 获取root
    u32 rootRank = param.root / level1CommInfo.localRankSize;
    CHK_RET(tempAlg->Prepare(
        execMem.inputMem, execMem.inputMem, execMem.inputMem, execMem.count, param.DataDes.dataType, param.stream,
        HCCL_REDUCE_RESERVED, rootRank));

    CHK_RET(RunTemplate(tempAlg, level2CommInfo));

    HCCL_INFO("MidCountBroadcast run success in level2");
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMidCountFor91093Executor::RunLevel1ByNHR(
    const OpParam& param, ExecMem& execMem, SubCommInfo& level1CommInfo, [[maybe_unused]] SubCommInfo& level2CommInfo)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[MidCountBroadcast][RunLevel1ByNHR] userRank[%u] starts.", topoAttr_.userRank);
    u32 unitSize = 0;
    const HcclDataType dataType = param.GetDataType();
    CHK_RET(SalGetDataTypeSize(dataType, unitSize));

    std::unique_ptr<AlgTemplateBase> tempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR_ONESHOT, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_BROADCAST_NHR_ONESHOT in COMM_COMBINE_L1", __func__);
    CHK_SMART_PTR_NULL(tempAlg);

    // 获取root
    u32 rootRank = 0;
    CHK_RET(GetRankByUserRank(COMM_COMBINE_L1, COMM_INDEX_0, param.root, rootRank));
    CHK_RET(tempAlg->Prepare(
        execMem.inputMem, execMem.inputMem, execMem.inputMem, execMem.count, param.DataDes.dataType, param.stream,
        HCCL_REDUCE_RESERVED, rootRank));

    CHK_RET(RunTemplate(tempAlg, level1CommInfo));

    HCCL_INFO("MidCountBroadcast run success in level1");
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMidCountFor91093Executor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] The MidCountFor91093Executor starts", __func__);

    SubCommInfo level1CommInfo;
    SubCommInfo level2CommInfo;
    CHK_RET(CheckCommSize(COMM_COMBINE_L1, COMM_INDEX_0 + 1));
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    level1CommInfo = GetSubCommInfo(COMM_COMBINE_L1, COMM_INDEX_0);
    level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    u32 unitSize = 0;
    const HcclDataType dataType = param.GetDataType();
    CHK_RET(SalGetDataTypeSize(dataType, unitSize));

    u32 rootSideIndex = param.root / level1CommInfo.localRankSize;
    u32 rankSideIndex = topoAttr_.userRank / level1CommInfo.localRankSize;
    if (rootSideIndex == rankSideIndex) {
        CHK_RET(RunLevel1ByNHR(param, execMem, level1CommInfo, level2CommInfo));
    }
    CHK_RET(RunLevel2ByNHR(param, execMem, level1CommInfo, level2CommInfo));

    HCCL_INFO("MidCountBroadcast run success.");
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMidCountFor91093Executor::RunLoopInner(OpParam& param, ExecMem& execMem)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 totalSize = unitSize * param.DataDes.count;
    bool isRootRank = param.root == topoAttr_.realUserRank ? true : false;
    u64 curSize = execMem.count * unitSize; // 单位：字节
    auto inCCLbufferSize = execMem.inputMem.size();

    HCCL_DEBUG(
        "[CollBroadcastMidCountFor91093Executor][RunLoopInner]inputMem[%p], outputMem[%p]"
        "inputPtr[%p], curCount[%llu], curSize[%llu]",
        execMem.inputMem.ptr(), execMem.outputMem.ptr(), execMem.inputPtr, execMem.count, curSize);
    CHK_PRT_RET(
        (execMem.count == 0),
        HCCL_ERROR("[CollBroadcastMidCountFor91093Executor][RunLoop]In OP_BASE curCount is zero."), HCCL_E_PARA);

    bool hugeData
        = (inCCLbufferSize / topoAttr_.deviceNumPerAggregation > RDMA_SEND_MAX_SIZE) || (curSize > SDMA_SEND_MAX_SIZE);
    bool isSmallData = IsBroadcastSmallData(curSize, totalSize);
    u64 sliceNum = 0;
    CHK_RET(GetSliceNum(curSize, isSmallData, sliceNum));
    CopyPattern copy = DMAReduceFlag_ ? CopyPattern::ZCOPY : CopyPattern::BCOPY;
    auto meta = HcclOpMetaInfo::GetOneForBroadcast(isRootRank, param.root, hugeData, isSmallData, sliceNum, copy);
    CHK_RET(InitTask(dispatcher_, param.stream, meta.isEnableCache, meta.GetCacheKey()));

    // 执行
    HcclResult ret;
    // 如果使用in CCL buffer，需要将user buffer in中的结果拷贝到CCL buffer in
    DeviceMem inCommMem = execMem.inputMem.range(0, curSize);
    DeviceMem inMem(execMem.inputPtr, curSize);
    if (topoAttr_.userRank == param.root) {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, param.stream));
    }

    ret = KernelRun(param, execMem);
    if (topoAttr_.realUserRank != param.root) {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inMem, inCommMem, param.stream));
    }

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollBroadcastMidCountFor91093Executor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], "
            "inputMem ptr[%p], count[%llu], dataType[%d]",
            HCCL_ERROR_CODE(ret), param.tag.c_str(), execMem.inputMem.ptr(), execMem.count, param.DataDes.dataType),
        ret);

    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    return ret;
}

HcclResult CollBroadcastMidCountFor91093Executor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    tag_ = param.tag;
    algResResp_ = &algRes;
    /* ------------执行算法-------------- */
    HcclUs startut = TIME_NOW();

    // 图模式和单卡场景下不需要Loop
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.inputPtr;
    HCCL_INFO(
        "Orchestrate UserRank[%u], devicePhyId[%u], inputPtr[%p], outputPtr[%p], root[%u]", topoAttr_.userRank,
        topoAttr_.devicePhyId, param.inputPtr, param.outputPtr, param.root);

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u8* curInputPtr = static_cast<u8*>(param.inputPtr);
    CHK_PTR_NULL(curInputPtr);
    u64 maxCountPerLoop = CalcLoopMaxCount(algRes.cclInputMem.size(), unitSize);
    HCCL_DEBUG(
        "[CollBroadcastMidCountFor91093Executor][RunLoop]tag[%s], userRankSize is [%u], maxCountPerLoop is [%llu].",
        param.tag.c_str(), topoAttr_.userRankSize, maxCountPerLoop);

    u64 totalCount = param.DataDes.count;
    for (u64 countLeft = totalCount, curCount = 0, inputOffset = 0; countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        // 判断剩余数据量对应的output size是否大于中转output size
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = algRes.cclOutputMem;
        execMem.outputMem = algRes.cclOutputMem; // ccl buffer 均只使用out buffer
        execMem.inputPtr = curInputPtr;
        HCCL_DEBUG(
            "[CollBroadcastMidCountFor91093Executor] RunLoop tag[%s], inputOffset[%llu], "
            "curInputPtr[%p], sendCount[%llu], sendSize[%llu], dataType[%s], realUserRank[%u]",
            param.tag.c_str(), inputOffset, curInputPtr, curCount, curSize,
            GetDataTypeEnumStr(param.DataDes.dataType).c_str(), topoAttr_.realUserRank);

        CHK_RET(RunLoopInner(param, execMem));

        inputOffset = curSize;
    }

    HCCL_INFO(
        "tag[%s], Broadcast executor orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("BroadcastMidCountFor91093Executor", BroadcastMidCountFor91093, CollBroadcastMidCountFor91093Executor);
} // namespace hccl
