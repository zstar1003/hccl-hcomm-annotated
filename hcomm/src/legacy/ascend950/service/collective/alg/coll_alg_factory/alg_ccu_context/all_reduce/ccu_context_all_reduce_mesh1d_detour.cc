/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_reduce_mesh1d_detour.h"
#include "ccu_instruction_all_reduce_mesh1d_detour.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;

CcuContextAllReduceMeshDetour1D::CcuContextAllReduceMeshDetour1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgAllReduceMeshDetour1D* ctxArg = dynamic_cast<const CcuCtxArgAllReduceMeshDetour1D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMeshDetour1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    rankSize_ = ctxArg->dimSize_[0];
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextAllReduceMeshDetour1D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }
    reduceOp_ = ctxArg->op_.reduceOp;
    singleTransportSize = ctxArg->singleTransportSize_;
    detourPathNum = ctxArg->detourPathNum_;
    pathNumPerPeer = ctxArg->pathNumPerPeer_;
    HCCL_INFO(
        "[CcuContextAllReduceMeshDetour1D] Init, CtxArgs are rankId_[%u], rankSize_[%u], dataType[%s], "
        "outputDataType[%s], reduceOp[%s]",
        rankId_, rankSize_, dataType_.Describe().c_str(), outputDataType_.Describe().c_str(),
        reduceOp_.Describe().c_str());

    HCCL_INFO("[CcuContextAllReduceMeshDetour1D] transport.size[%zu]", transports.size());
    if (transports.size() < rankSize_ - 1) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMeshDetour1D transports size is less"));
    }
    for (uint64_t i = 0; i < pathNumPerPeer; i++) {
        // 到每个对端有pathNum个transport，故detourTransport中共有pathNum组
        detourTransports_.emplace_back(std::vector<CcuTransport*>());
    }
    uint64_t directPathNum = pathNumPerPeer - detourPathNum;
    for (uint64_t i = 0; i < directPathNum; i++) {
        // 有pathNum-detourPathNum组的直连链路，每组重复
        for (uint64_t j = 0; j < rankSize_ - 1; j++) {
            detourTransports_[i].emplace_back(transports[j]);
        }
        HCCL_INFO(
            "[CcuContextAllReduceMeshDetour1D] Add directTransports[%llu], size[%zu]", i, detourTransports_[i].size());
    }
    for (uint64_t i = 0; i < detourPathNum; i++) {
        for (uint64_t j = 0; j < rankSize_ - 1; j++) {
            detourTransports_[i + directPathNum].emplace_back(transports[(i + 1) * (rankSize_ - 1) + j]);
            detourTransports_[i + directPathNum].emplace_back(
                transports[(i + 1) * (rankSize_ - 1) + j + detourPathNum * (rankSize_ - 1)]);
            HCCL_INFO(
                "detourTransports_ emplace_back sendLink[%u], recvLink[%u]", (i + 1) * (rankSize_ - 1) + j,
                (i + 1) * (rankSize_ - 1) + j + detourPathNum * (rankSize_ - 1));
        }
    }
}

void CcuContextAllReduceMeshDetour1D::CreateMultiOpReduceDetour(
    DataType& dataType, DataType& outputDataType, ReduceOp& opType)
{
    moConfig.loopCount = CcuRep::CCU_MS_DEFAULT_LOOP_COUNT;
    moConfig.msInterleave = pathNumPerPeer * rankSize_;
    if (moRes.executor.size() == 0) {
        moRes.maskSignal = CreateBlockMaskSignal(moConfig.loopCount);
        moRes.executor = CreateBlockExecutor(moConfig.loopCount);
        moRes.ccuBuffer = CreateBlockCcuBuffer(moConfig.loopCount * moConfig.msInterleave);
    }
    std::string loopType = "reduceDetour";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }
    CcuRep::LoopBlock lb(this, loopType + "_loop");
    {
        // loopblock的形参
        std::vector<CcuRep::Memory> dst;
        std::vector<CcuRep::Memory> src;
        std::vector<CcuRep::Variable> lengths;
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            lengths.emplace_back(CreateVariable());
            dst.emplace_back(CreateMemory());
            for (uint64_t j = 0; j < rankSize_; j++) {
                src.emplace_back(CreateMemory());
            }
        }

        lb(src, dst, lengths);
        std::vector<std::vector<CcuRep::CcuBuffer>> bufs;
        bufs.resize(pathNumPerPeer);
        std::vector<CcuRep::MaskSignal> sems;

        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            for (uint64_t j = 0; j < rankSize_; j++) {
                bufs[i].emplace_back(moRes.ccuBuffer[i * rankSize_ + j]);
            }
            sems.emplace_back(moRes.maskSignal[i]);
        }

        // 先读远端直连的到本地MS
        uint64_t directPathNum = pathNumPerPeer - detourPathNum;
        for (uint64_t i = 0; i < directPathNum; i++) {
            for (uint32_t j = 0; j < detourTransports_[i].size(); j++) {
                if (detourTransports_[i][j] == nullptr) {
                    THROW<CcuApiException>("transport is nullptr");
                }
                Read(*detourTransports_[i][j], bufs[i][j], src[i * rankSize_ + j], lengths[i], sems[i], 1 << j);
            }
        }
        // 再读远端绕路的到本地MS
        for (uint64_t i = directPathNum; i < pathNumPerPeer; i++) {
            for (uint64_t j = 0; j < rankSize_ - 1; j++) {
                if (detourTransports_[i][j * 2 + 1] == nullptr) { // j * 2 + 1是recvOnly Link
                    THROW<CcuApiException>("transport is nullptr");
                }
                Read(*detourTransports_[i][j * 2 + 1], bufs[i][j], src[i * rankSize_ + j], lengths[i], sems[i], 1 << j);
            }
        }

        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            LocalCopy(
                bufs[i][rankSize_ - 1], src[i * rankSize_ + rankSize_ - 1], lengths[i], sems[i], 1 << (rankSize_ - 1));
        }
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            LocalWait(sems[i], (1 << rankSize_) - 1);
        }
        if (rankSize_ > 1) {
            for (uint64_t i = 0; i < pathNumPerPeer; i++) {
                LocalReduce(bufs[i], rankSize_, dataType, outputDataType, opType, sems[i], lengths[i]);
                LocalWait(sems[i]);
            }
        }
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            LocalCopy(dst[i], bufs[i][0], lengths[i], sems[i]);
            LocalWait(sems[i]);
        }
    }
    registeredLoop.insert(loopType);
    return;
}

void CcuContextAllReduceMeshDetour1D::GroupReduceDetour(
    std::vector<CcuRep::Memory>& src, std::vector<CcuRep::Memory>& dst, DataType& dataType, DataType& outputDataType,
    ReduceOp& opType)
{
    CreateMultiOpReduceDetour(dataType, outputDataType, opType);
    uint32_t interLeave = 8;

    CCU_IF(iterNum_ != 0)
    {
        CcuRep::Variable loopParam = CreateVariable();
        CcuRep::Variable paraCfg = CreateVariable();
        CcuRep::Variable offsetCfg = CreateVariable();

        loopParam = CcuRep::GetLoopParam(
            0, singleTransportSize * moConfig.loopCount, 0); // 下次迭代的偏移是单次总搬运量*loopNum
        loopParam += iterNum_;                               // 加上loop的迭代次数构成完整loop参数
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1); // loop固定展开到128个
        offsetCfg = CcuRep::GetOffsetParam(singleTransportSize, interLeave, pathNumPerPeer); // 下一个loop偏移量
        auto lc = Loop("reduceDetour_loop")(src, dst, lengths_);
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }
    return;
}

void CcuContextAllReduceMeshDetour1D::CreateMultiOpBroadcastDetour()
{
    moConfig.loopCount = CcuRep::CCU_MS_DEFAULT_LOOP_COUNT;
    moConfig.msInterleave = pathNumPerPeer * 1; // Bcast为msNum*1，Reduce为msNum*rankSize_
    if (moRes.executor.size() == 0) {
        moRes.executor = CreateBlockExecutor(moConfig.loopCount);
        moRes.maskSignal = CreateBlockMaskSignal(moConfig.loopCount);
        moRes.ccuBuffer = CreateBlockCcuBuffer(moConfig.loopCount * moConfig.msInterleave);
    }

    std::string loopType = "broadcastDetour";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }

    CcuRep::LoopBlock lb(this, loopType + "_loop");
    {
        // loopblock的形参
        std::vector<CcuRep::Memory> src;
        std::vector<CcuRep::Memory> dst;
        std::vector<CcuRep::Variable> lengths;
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            lengths.emplace_back(CreateVariable());
            src.emplace_back(CreateMemory());
            for (uint64_t j = 0; j < rankSize_; j++) {
                dst.emplace_back(CreateMemory());
            }
        }

        lb(src, dst, lengths);
        std::vector<CcuRep::CcuBuffer> bufs;
        std::vector<CcuRep::MaskSignal> sems;
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            bufs.emplace_back(moRes.ccuBuffer[i]);
            sems.emplace_back(moRes.maskSignal[i]);
        }

        // 从本地搬运多片数据到多个MS
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            LocalCopy(bufs[i], src[i], lengths[i], sems[i]);
        }
        // 等待数据搬到MS
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            LocalWait(sems[i]);
        }
        // 给每个peer搬运多个MS上的数据
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            for (uint64_t j = 0; j < rankSize_ - 1; j++) {
                if (detourTransports_[i][j * 2] == nullptr) { // j * 2是sendOnly Link
                    THROW<CcuApiException>("transport is nullptr");
                }
                Write(*detourTransports_[i][j * 2], dst[i * rankSize_ + j], bufs[i], lengths[i], sems[i], 1 << j);
            }
            LocalCopy(dst[i * rankSize_ + rankSize_ - 1], bufs[i], lengths[i], sems[i], 1 << (rankSize_ - 1));
        }
        // 等待给所有远端写完数据
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            LocalWait(sems[i], (1 << rankSize_) - 1);
        }
    }

    registeredLoop.insert(loopType);
    return;
}

void CcuContextAllReduceMeshDetour1D::GroupBroadcastDetour(
    std::vector<CcuRep::Variable>& lengths, std::vector<CcuRep::Memory>& src, std::vector<CcuRep::Memory>& dst)
{
    CreateMultiOpBroadcastDetour();
    uint32_t interLeave = 8;

    CCU_IF(iterNum_ != 0)
    {
        CcuRep::Variable loopParam = CreateVariable();
        CcuRep::Variable paraCfg = CreateVariable();
        CcuRep::Variable offsetCfg = CreateVariable();

        loopParam = CcuRep::GetLoopParam(0, singleTransportSize * moConfig.loopCount, 0); // 偏移是单次总搬运量*loopNum
        loopParam += iterNum_; // 加上loop的迭代次数构成完整loop参数
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1); // loop固定展开到128个
        offsetCfg = CcuRep::GetOffsetParam(singleTransportSize, interLeave, pathNumPerPeer); // 下一个loop偏移量
        auto lc = Loop("broadcastDetour_loop")(src, dst, lengths);
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }
    return;
}

void CcuContextAllReduceMeshDetour1D::ReduceScatterFirstStep()
{
    std::vector<CcuRep::Memory> reduceSrc;
    std::vector<CcuRep::Memory> reduceDst;

    // 为每个直连或绕路transport分别准备reduceSrc与reduceDst
    for (uint64_t i = 0; i < pathNumPerPeer; i++) {
        reduceDst.emplace_back(CreateMemory());
        for (uint64_t j = 0; j < rankSize_; j++) {
            reduceSrc.emplace_back(CreateMemory());
        }
    }

    // reduceDst填充
    reduceDst[0].addr = output_[rankId_];
    reduceDst[0].addr += offset_;
    reduceDst[0].token = token_[rankId_];
    for (uint64_t i = 1; i < pathNumPerPeer; i++) {
        reduceDst[i].addr = reduceDst[i - 1].addr + lengths_[i - 1];
        reduceDst[i].token = token_[rankId_];
    }
    // 直连transport的reduceSrc填充
    uint32_t srcId = 0;
    uint32_t curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = srcId;
            srcId++;
        } else {
            curId = rankSize_ - 1;
        }
        reduceSrc[curId].addr = input_[rankIdx];
        reduceSrc[curId].addr += offset_;
        reduceSrc[curId].token = token_[rankIdx];
    }
    // 绕路transport的reduceSrc相比直连src再做偏移
    for (uint64_t i = 1; i < pathNumPerPeer; i++) {
        for (uint64_t j = 0; j < rankSize_; j++) {
            reduceSrc[i * rankSize_ + j].addr = reduceSrc[(i - 1) * rankSize_ + j].addr + lengths_[i - 1];
            reduceSrc[i * rankSize_ + j].token = reduceSrc[(i - 1) * rankSize_ + j].token;
        }
    }

    // 整块数据用绕路Reduce
    GroupReduceDetour(reduceSrc, reduceDst, dataType_, outputDataType_, reduceOp_);
    return;
}

void CcuContextAllReduceMeshDetour1D::ReduceScatterSecondStep()
{
    // 余下的尾块用直连Reduce
    std::vector<CcuRep::Memory> tailSrc;
    CcuRep::Memory tailDst = CreateMemory();
    for (uint64_t i = 0; i < rankSize_; i++) {
        tailSrc.emplace_back(CreateMemory());
    }
    tailDst.addr = output_[rankId_];
    tailDst.addr += offset_;
    tailDst.addr += tailOffset_;
    tailDst.token = token_[rankId_];
    uint32_t srcId = 0;
    uint32_t curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = srcId;
            srcId++;
        } else {
            curId = rankSize_ - 1;
        }
        tailSrc[curId].addr = input_[rankIdx];
        tailSrc[curId].addr += tailOffset_;
        tailSrc[curId].addr += offset_;
        tailSrc[curId].token = token_[rankIdx];
    }

    GroupReduce(detourTransports_[0], tailDst, tailSrc, groupOpSize_, dataType_, outputDataType_, reduceOp_);
    return;
}

void CcuContextAllReduceMeshDetour1D::AllGatherFirstStep()
{
    // 开始AllGather
    std::vector<CcuRep::Memory> allGatherSrc;
    std::vector<CcuRep::Memory> allGatherDst;

    // 为每个直连或绕路transport分别准备src与dst
    for (uint64_t i = 0; i < pathNumPerPeer; i++) {
        allGatherSrc.emplace_back(CreateMemory());
        for (uint64_t j = 0; j < rankSize_; j++) {
            allGatherDst.emplace_back(CreateMemory());
        }
    }
    // allGather 的输入就是 reduceScatter 的输出
    allGatherSrc[0].addr = output_[rankId_]; // 直连源地址
    allGatherSrc[0].addr += offset_;
    allGatherSrc[0].token = token_[rankId_];
    for (uint64_t i = 1; i < pathNumPerPeer; i++) {
        allGatherSrc[i].addr = allGatherSrc[i - 1].addr + lengths_[i - 1];
        allGatherSrc[i].token = token_[rankId_];
    }

    // 直连的allGatherDst填充
    uint32_t curId = 0;
    uint32_t dstId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = rankSize_ - 1;
        }
        allGatherDst[curId].addr = output_[rankIdx];
        allGatherDst[curId].addr += offset_;
        allGatherDst[curId].token = token_[rankIdx];
    }

    // 绕路的allGatherDst填充，相比直连做偏移
    for (uint64_t i = 1; i < pathNumPerPeer; i++) {
        for (uint64_t j = 0; j < rankSize_; j++) {
            allGatherDst[i * rankSize_ + j].addr = allGatherDst[(i - 1) * rankSize_ + j].addr + lengths_[i - 1];
            allGatherDst[i * rankSize_ + j].token = allGatherDst[(i - 1) * rankSize_ + j].token;
        }
    }
    GroupBroadcastDetour(lengths_, allGatherSrc, allGatherDst);
    return;
}

void CcuContextAllReduceMeshDetour1D::AllGatherSecondStep()
{
    // 余下的尾块用直连transport发送
    CcuRep::Memory bcastTailSrc = CreateMemory();
    std::vector<CcuRep::Memory> bcastTailDst;
    for (uint64_t i = 0; i < rankSize_; i++) {
        bcastTailDst.emplace_back(CreateMemory());
    }
    bcastTailSrc.addr = output_[rankId_];
    bcastTailSrc.addr += offset_;
    bcastTailSrc.addr += tailOffset_;
    bcastTailSrc.token = token_[rankId_];
    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = rankSize_ - 1;
        }
        bcastTailDst[curId].addr = output_[rankIdx];
        bcastTailDst[curId].addr += offset_;
        bcastTailDst[curId].addr += tailOffset_;
        bcastTailDst[curId].token = token_[rankIdx];
    }
    GroupBroadcast(detourTransports_[0], bcastTailDst, bcastTailSrc, groupOpSize_);
    return;
}

void CcuContextAllReduceMeshDetour1D::Algorithm()
{
    HCCL_INFO("[CcuContextAllReduceMeshDetour1D] AllReduceMeshDetour1D run.");
    uint16_t selfBit = 1 << rankId_;
    uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_));

    // 初始化资源
    uint16_t transportIdx = 0;
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextAllReduceMeshDetour1D] MyRank[%u], PeerId[%llu], TransportId[%u]", rankId_, peerId,
                transportIdx);
            CHK_PRT_RET(
                detourTransports_[0][transportIdx] == nullptr,
                HCCL_ERROR("[CcuContextAllReduceMeshDetour1D] Algorithm transport ptr is null"), );
            input_.push_back(CreateVariable((*detourTransports_[0][transportIdx]), INPUT_XN_ID));
            output_.push_back(CreateVariable((*detourTransports_[0][transportIdx]), OUTPUT_XN_ID));
            token_.push_back(CreateVariable((*detourTransports_[0][transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    offset_ = CreateVariable();
    iterNum_ = CreateVariable();
    tailOffset_ = CreateVariable();
    tailSize_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();
    for (uint64_t i = 0; i < pathNumPerPeer; i++) {
        lengths_.emplace_back(CreateVariable());
    }

    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(offset_);
    Load(iterNum_);
    Load(tailOffset_);
    Load(tailSize_);
    Load(groupOpSize_);
    for (uint64_t i = 0; i < pathNumPerPeer; i++) {
        Load(lengths_[i]);
    }

    for (auto& t : detourTransports_[0]) {
        WriteVariableWithSignal(*t, input_[rankId_], INPUT_XN_ID, CKE_IDX_1, selfBit);
        WriteVariableWithSignal(*t, output_[rankId_], OUTPUT_XN_ID, CKE_IDX_2, selfBit);
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_3, selfBit);
    }

    GroupWait(*transportGroup, CKE_IDX_1, allBit);
    GroupWait(*transportGroup, CKE_IDX_2, allBit);
    GroupWait(*transportGroup, CKE_IDX_3, allBit);

    ReduceScatterFirstStep();
    ReduceScatterSecondStep();

    AllGatherFirstStep();
    AllGatherSecondStep();

    for (auto t : detourTransports_[0]) {
        RemotePost(*t, CKE_IDX_0, selfBit);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit);

    HCCL_INFO("[CcuContextAllReduceMeshDetour1D] AllReduceMeshDetour1D end.");
    return;
}

std::vector<uint64_t> CcuContextAllReduceMeshDetour1D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgAllReduceMeshDetour1D* taskArg = dynamic_cast<const CcuTaskArgAllReduceMeshDetour1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMeshDetour1D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t offset = taskArg->offset_;
    uint64_t iterNum = taskArg->iterNum_;
    uint64_t tailOffset = taskArg->tailOffset_;
    uint64_t tailSize = taskArg->tailSize_;
    auto goSize = CalGoSize(tailSize);

    HCCL_INFO(
        "[CcuContextAllReduceMeshDetour1D] GeneArgs, taskArg are inputAddr[%llu], outputAddr[%llu], "
        "offset[%llu], iterNum[%llu], tailOffset[%llu], tailSize[%llu]",
        inputAddr, outputAddr, offset, iterNum, tailOffset, tailSize);
    std::vector<uint64_t> sqeArgs = {inputAddr, outputAddr, tokenInfo, offset,    iterNum,  tailOffset,
                                     tailSize,  goSize[0],  goSize[1], goSize[2], goSize[3]};
    for (auto len : taskArg->lengths_) {
        sqeArgs.emplace_back(len);
    }
    return sqeArgs;
}

} // namespace Hccl
