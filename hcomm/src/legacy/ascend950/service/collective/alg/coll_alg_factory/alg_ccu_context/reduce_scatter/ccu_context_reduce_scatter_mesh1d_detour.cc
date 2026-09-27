/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_reduce_scatter_mesh1d_detour.h"
#include "ccu_instruction_reduce_scatter_mesh1d_detour.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;

CcuContextReduceScatterMeshDetour1D::CcuContextReduceScatterMeshDetour1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgReduceScatterMeshDetour1D* ctxArg = dynamic_cast<const CcuCtxArgReduceScatterMeshDetour1D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMeshDetour1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    rankSize_ = ctxArg->dimSize_[0];
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextReduceScatterMeshDetour1D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }
    reduceOp_ = ctxArg->op_.reduceOp;
    singleTransportSize_ = ctxArg->singleTransportSize_;
    detourPathNum_ = ctxArg->detourPathNum_;
    pathNumPerPeer_ = ctxArg->pathNumPerPeer_;
    HCCL_INFO(
        "[CcuContextReduceScatterMeshDetour1D] Init, CtxArgs are rankId[%u], rankSize[%u], dataType[%s], "
        "outputDataType[%s], reduceOp[%s]",
        rankId_, rankSize_, dataType_.Describe().c_str(), outputDataType_.Describe().c_str(),
        reduceOp_.Describe().c_str());
    if (transports.size() == 0 || transports.size() < rankSize_ - 1) {
        THROW<NullPtrException>(
            StringFormat("CcuContextReduceScatterMeshDetour1D transports is empty or size is less"));
    }
    HCCL_INFO("[CcuContextReduceScatterMeshDetour1D] transport.size[%zu]", transports.size());
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        // 到每个对端有pathNum个transport，故detourTransport中共有pathNum组
        detourTransports_.emplace_back(std::vector<CcuTransport*>());
    }
    uint64_t directPathNum = pathNumPerPeer_ - detourPathNum_;
    for (uint64_t i = 0; i < directPathNum; i++) {
        // 有pathNum-detourPathNum组的直连链路，每组重复
        for (uint32_t j = 0; j < rankSize_ - 1; j++) {
            detourTransports_[i].emplace_back(transports[j]);
        }
        HCCL_INFO(
            "[CcuContextReduceScatterMeshDetour1D] Add directTransports[%llu], size[%zu]", i,
            detourTransports_[i].size());
    }
    for (uint32_t i = 0; i < detourPathNum_; i++) {
        for (uint32_t j = 0; j < rankSize_ - 1; j++) {
            detourTransports_[i + directPathNum].emplace_back(transports[(i + 1) * (rankSize_ - 1) + j]);
            detourTransports_[i + directPathNum].emplace_back(
                transports[(i + 1) * (rankSize_ - 1) + j + detourPathNum_ * (rankSize_ - 1)]);
            HCCL_INFO(
                "detourTransports_ emplace_back sendLink[%u], recvLink[%u]", (i + 1) * (rankSize_ - 1) + j,
                (i + 1) * (rankSize_ - 1) + j + detourPathNum_ * (rankSize_ - 1));
        }
    }
}

void CcuContextReduceScatterMeshDetour1D::CreateMultiOpReduceDetour(
    DataType& dataType, DataType& outputDataType, ReduceOp& opType)
{
    moConfig.loopCount = CcuRep::CCU_MS_DEFAULT_LOOP_COUNT;
    moConfig.msInterleave = pathNumPerPeer_ * rankSize_;
    if (moRes.executor.size() == 0) {
        moRes.executor = CreateBlockExecutor(moConfig.loopCount);
        moRes.maskSignal = CreateBlockMaskSignal(moConfig.loopCount);
        moRes.ccuBuffer = CreateBlockCcuBuffer(moConfig.loopCount * moConfig.msInterleave);
    }
    std::string loopType = "reduceDetour";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }
    CcuRep::LoopBlock lb(this, loopType + "_loop");
    {
        // loopblock的形参
        std::vector<CcuRep::Memory> src;
        std::vector<CcuRep::Memory> dst;
        std::vector<CcuRep::Variable> lengths;
        for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
            lengths.emplace_back(CreateVariable());
            dst.emplace_back(CreateMemory());
            for (uint32_t j = 0; j < rankSize_; j++) {
                src.emplace_back(CreateMemory());
            }
        }

        lb(src, dst, lengths);
        std::vector<std::vector<CcuRep::CcuBuffer>> bufs;
        bufs.resize(pathNumPerPeer_);
        std::vector<CcuRep::MaskSignal> sems;

        for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
            for (uint32_t j = 0; j < rankSize_; j++) {
                bufs[i].emplace_back(moRes.ccuBuffer[i * rankSize_ + j]);
            }
            sems.emplace_back(moRes.maskSignal[i]);
        }

        // 先读远端直连的到本地MS
        uint64_t directPathNum = pathNumPerPeer_ - detourPathNum_;
        for (uint32_t i = 0; i < directPathNum; i++) {
            for (uint32_t j = 0; j < detourTransports_[i].size(); j++) {
                if (detourTransports_[i][j] == nullptr) {
                    THROW<CcuApiException>("transport is nullptr");
                }
                Read(*detourTransports_[i][j], bufs[i][j], src[i * rankSize_ + j], lengths[i], sems[i], 1 << j);
            }
        }
        // 再读远端绕路的到本地MS
        for (uint32_t i = directPathNum; i < pathNumPerPeer_; i++) {
            for (uint32_t j = 0; j < rankSize_ - 1; j++) {
                if (detourTransports_[i][j * 2 + 1] == nullptr) { // j * 2 + 1是recvOnly Link
                    THROW<CcuApiException>("transport is nullptr");
                }
                Read(*detourTransports_[i][j * 2 + 1], bufs[i][j], src[i * rankSize_ + j], lengths[i], sems[i], 1 << j);
            }
        }

        for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
            LocalCopy(
                bufs[i][rankSize_ - 1], src[i * rankSize_ + rankSize_ - 1], lengths[i], sems[i], 1 << (rankSize_ - 1));
        }
        for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
            LocalWait(sems[i], (1 << rankSize_) - 1);
        }
        if (rankSize_ > 1) {
            for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
                LocalReduce(bufs[i], rankSize_, dataType, outputDataType, opType, sems[i], lengths[i]);
                LocalWait(sems[i]);
            }
        }
        for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
            LocalCopy(dst[i], bufs[i][0], lengths[i], sems[i]);
            LocalWait(sems[i]);
        }
    }
    registeredLoop.insert(loopType);
    return;
}

void CcuContextReduceScatterMeshDetour1D::GroupReduceDetour(
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
            0, singleTransportSize_ * moConfig.loopCount, 0); // 下次迭代的偏移是单次总搬运量*loopNum
        loopParam += iterNum_;                                // 加上loop的迭代次数构成完整loop参数
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1); // loop固定展开到128个
        offsetCfg = CcuRep::GetOffsetParam(singleTransportSize_, interLeave, pathNumPerPeer_); // 下一个loop偏移量
        auto lc = Loop("reduceDetour_loop")(src, dst, lengths_);
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }
    return;
}

void CcuContextReduceScatterMeshDetour1D::Algorithm()
{
    HCCL_INFO("[CcuContextReduceScatterMeshDetour1D] ReduceScatterMeshDetour1D run");
    uint16_t selfBit = 1 << rankId_;
    uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_));
    output_.push_back(CreateVariable());
    // 初始化资源
    uint16_t transportIdx = 0;
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextReduceScatterMeshDetour1D] MyRank[%u], PeerId[%llu], TransportId[%u]", rankId_, peerId,
                transportIdx);
            CHK_PRT_RET(
                detourTransports_[0][transportIdx] == nullptr,
                HCCL_ERROR("[CcuContextReduceScatterMeshDetour1D] Algorithm transport ptr is null"), );
            input_.push_back(CreateVariable((*detourTransports_[0][transportIdx]), INPUT_XN_ID));
            token_.push_back(CreateVariable((*detourTransports_[0][transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    offset_ = CreateVariable();
    iterNum_ = CreateVariable();
    tailOffset_ = CreateVariable();
    tailSize_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        lengths_.emplace_back(CreateVariable());
    }

    Load(input_[rankId_]);
    Load(output_[0]);
    Load(token_[rankId_]);
    Load(offset_);
    Load(iterNum_);
    Load(tailOffset_);
    Load(tailSize_);
    Load(groupOpSize_);
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        Load(lengths_[i]);
    }

    for (auto& t : detourTransports_[0]) {
        WriteVariableWithSignal(*t, input_[rankId_], INPUT_XN_ID, CKE_IDX_1, selfBit);
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_3, selfBit);
    }

    GroupWait(*transportGroup, CKE_IDX_1, allBit);
    GroupWait(*transportGroup, CKE_IDX_3, allBit);
    // 如果是4p*2场景，template里可以都传4k进来，transport和length通过<直连4k>, <直连4k>,
    // <绕路4k>这样构造达成数据量2:1的效果

    std::vector<CcuRep::Memory> reduceSrc;
    std::vector<CcuRep::Memory> reduceDst;

    // 为每个直连或绕路transport分别准备reduceSrc与reduceDst
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        reduceDst.emplace_back(CreateMemory());
        for (uint32_t j = 0; j < rankSize_; j++) {
            reduceSrc.emplace_back(CreateMemory());
        }
    }

    // reduceDst填充
    reduceDst[0].addr = output_[0];
    // reduceDst[0].addr += offset_;
    reduceDst[0].token = token_[rankId_];
    for (uint32_t i = 1; i < pathNumPerPeer_; i++) {
        reduceDst[i].addr = reduceDst[i - 1].addr + lengths_[i - 1];
        reduceDst[i].token = token_[rankId_];
    }
    // 直连transport的reduceSrc填充
    uint32_t srcId = 0;
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = srcId;
            srcId++;
        } else {
            curId = rankSize_ - 1;
        }
        reduceSrc[curId].addr = input_[rankIdx];
        reduceSrc[curId].token = token_[rankIdx];
        reduceSrc[curId].addr += offset_;
    }
    // 绕路transport的reduceSrc相比直连src再做偏移
    for (uint32_t i = 1; i < pathNumPerPeer_; i++) {
        for (uint32_t j = 0; j < rankSize_; j++) {
            reduceSrc[i * rankSize_ + j].addr = reduceSrc[(i - 1) * rankSize_ + j].addr + lengths_[i - 1];
            reduceSrc[i * rankSize_ + j].token = reduceSrc[(i - 1) * rankSize_ + j].token;
        }
    }

    GroupReduceDetour(reduceSrc, reduceDst, dataType_, outputDataType_, reduceOp_);

    // 余下的尾块用直连Reduce
    std::vector<CcuRep::Memory> tailSrc;
    CcuRep::Memory tailDst = CreateMemory();
    for (uint32_t i = 0; i < rankSize_; i++) {
        tailSrc.emplace_back(CreateMemory());
    }
    tailDst.addr = output_[0];
    // tailDst.addr += offset_;
    tailDst.addr += tailOffset_;
    tailDst.token = token_[rankId_];
    srcId = 0;
    curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = srcId;
            srcId++;
        } else {
            curId = rankSize_ - 1;
        }
        tailSrc[curId].addr = input_[rankIdx];
        tailSrc[curId].addr += offset_;
        tailSrc[curId].addr += tailOffset_;
        tailSrc[curId].token = token_[rankIdx];
    }

    GroupReduce(detourTransports_[0], tailDst, tailSrc, groupOpSize_, dataType_, outputDataType_, reduceOp_);

    for (auto t : detourTransports_[0]) {
        RemotePost(*t, CKE_IDX_0, selfBit);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit);

    HCCL_INFO("[CcuContextReduceScatterMeshDetour1D] ReduceScatterMeshDetour1D end");
    return;
}

std::vector<uint64_t> CcuContextReduceScatterMeshDetour1D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgReduceScatterMeshDetour1D* taskArg = dynamic_cast<const CcuTaskArgReduceScatterMeshDetour1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMeshDetour1D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t offset = taskArg->offset_;
    uint64_t iterNum = taskArg->iterNum_;
    uint64_t tailOffset = taskArg->tailOffset_;
    uint64_t tailSize = taskArg->tailSize_;
    auto goSize = CalGoSize(tailSize); // ***

    HCCL_INFO(
        "[CcuContextReduceScatterMeshDetour1D] GeneArgs, taskArg are inputAddr[%llu], outputAddr[%llu], "
        "offset[%llu], iterNum[%llu], tailOffset[%llu], tailSize[%llu]",
        inputAddr, outputAddr, offset, iterNum, tailOffset, tailSize);
    std::vector<uint64_t> sqeArgs = {inputAddr, outputAddr, tokenInfo, offset,    iterNum,  tailOffset,
                                     tailSize,  goSize[0],  goSize[1], goSize[2], goSize[3]};
    for (auto len : taskArg->lengths_) {
        HCCL_INFO("get lengths");
        sqeArgs.emplace_back(len);
    }
    return sqeArgs;
}

} // namespace Hccl
