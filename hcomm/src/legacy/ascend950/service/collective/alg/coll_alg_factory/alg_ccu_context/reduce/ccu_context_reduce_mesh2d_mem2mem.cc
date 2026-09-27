/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_reduce_mesh2d_mem2mem.h"
#include "ccu_instruction_reduce_mesh2d_mem2mem.h"

namespace Hccl {

CcuContextReduceMeshMem2Mem2D::CcuContextReduceMeshMem2Mem2D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgReduceMeshMem2Mem2D* ctxArg = dynamic_cast<const CcuCtxArgReduceMeshMem2Mem2D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceMeshMem2Mem2D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    dimSize_ = ctxArg->dimSize_;
    axisId_ = ctxArg->axisId_; // 要进行操作的是 行或列

    if (dimSize_.size() != 2 || axisId_ > 1 || dimSize_[0] == 0) { // 2D 拓扑校验
        THROW<NullPtrException>(StringFormat(
            "[CcuContextReduceMeshMem2Mem2D] dimSize[%u] or axisId[%u] or dimSize[0] [%u] is invalid", dimSize_.size(),
            axisId_, dimSize_[0]));
    }
    dimId_.emplace_back(rankId_ % dimSize_[0]);
    dimId_.emplace_back(rankId_ / dimSize_[0]);
    localId_ = dimId_[axisId_];     // 本rank所在的行/列
    localSize_ = dimSize_[axisId_]; // 本rank所在的行/列的总数

    HCCL_INFO(
        "[CcuContextReduceMeshMem2Mem2D] RankId[%u], DimSize0[%llu], DimSize1[%llu], localId[%llu], localSize[%llu]",
        rankId_, dimSize_[0], dimSize_[1], localId_, localSize_);

    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextReduceMeshMem2Mem2D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }
    reduceOp_ = ctxArg->op_.reduceOp;
    rootId_ = ctxArg->rootId_;
    rootDimId_.emplace_back(rootId_ % dimSize_[0]); // root的x
    rootDimId_.emplace_back(rootId_ / dimSize_[0]); // root的y
    HCCL_INFO(
        "[CcuContextReduceMeshMem2Mem2D] init end, ctxArg->dimSize size[%zu] localSize_[%u]", ctxArg->dimSize_.size(),
        localSize_);

    localAxisSignalName_ = "CcuContextReduceMeshMem2Mem2DAxisSync_" + std::to_string(axisId_);
    anotherAxisSignalName_ = "CcuContextReduceMeshMem2Mem2DAxisSync_" + std::to_string(1 - axisId_);
}

void CcuContextReduceMeshMem2Mem2D::InitResources()
{
    localAxisSignal_ = CreateMaskSignal();
    anotherAxisSignal_ = CreateMaskSignal();
    ExportMaskSignal(localAxisSignal_, localAxisSignalName_);
    anotherAxisSignal_ = ImportMaskSignal(anotherAxisSignalName_);

    output_ = CreateVariable();
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceMeshMem2Mem2D transports is empty"));
    }
    uint32_t transportIdx = 0;
    for (uint32_t peerId = 0; peerId < localSize_; peerId++) {
        if (peerId == localId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextReduceMeshMem2Mem2D] MyRank[%u], PeerId[%u], TransportId[%u]", localId_, peerId,
                transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr,
                HCCL_ERROR("[CcuContextReduceMeshMem2Mem2D] Algorithm transport ptr is null"), );
            input_.push_back(
                CreateVariable((*transports[transportIdx]), INPUT_XN_ID)); // 获取transport中id=1的Var来传递output
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    locMask_ = CreateMaskSignal();
    xAxisGroupOpSize_ = CreateGroupOpSize();
    yAxisGroupOpSize_ = CreateGroupOpSize();
    xAxisSize_ = CreateVariable();
    yAxisSize_ = CreateVariable();
    yAxisOffset_ = CreateVariable();
    curGoSize_ = CreateGroupOpSize();
    for (uint16_t roundId = 0; roundId < (localSize_ - 1); roundId++) {
        xChunkSize_.push_back(CreateVariable());
        yChunkSize_.push_back(CreateVariable());
        chunkSize_.push_back(CreateVariable());
    }
    chunkOffset_ = CreateVariable();
    HCCL_INFO("[CcuContextReduceMeshMem2Mem2D] InitResources finished");
}

void CcuContextReduceMeshMem2Mem2D::LoadArgs()
{
    Load(input_[localId_]);
    Load(output_);
    Load(token_[localId_]);
    Load(xAxisSize_);
    Load(yAxisSize_);
    Load(yAxisOffset_);
    for (uint16_t i = 0; i < (localSize_ - 1); i++) {
        Load(xChunkSize_[i]);
    }
    for (uint16_t i = 0; i < (localSize_ - 1); i++) {
        Load(yChunkSize_[i]);
    }
    Load(xAxisGroupOpSize_);
    Load(yAxisGroupOpSize_);
    // 只有step2会用到localcopy
    curGoSize_ = (axisId_ == X_AXIS_ID) ? yAxisGroupOpSize_ : xAxisGroupOpSize_;
    HCCL_INFO("[CcuContextReduceMeshMem2Mem2D] LoadArgs run finished");
}

void CcuContextReduceMeshMem2Mem2D::PreSync() // 前同步
{
    uint16_t selfBit = 1 << localId_;
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        WriteVariableWithSignal(*t, input_[localId_], INPUT_XN_ID, CKE_IDX_1, selfBit); // index = 1，传递output信息
        WriteVariableWithSignal(*t, token_[localId_], TOKEN_XN_ID, CKE_IDX_2, selfBit); // index = 2，传递token信息
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit); // index = 1，传递output信息
    GroupWait(*transportGroup, CKE_IDX_2, allBit); // index = 2，传递token信息
    HCCL_INFO("[CcuContextReduceMeshMem2Mem2D] PreSync run finished");
}

void CcuContextReduceMeshMem2Mem2D::PostSync(uint32_t signalIndex)
{
    uint16_t selfBit = 1 << localId_;
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        RemotePost(*t, signalIndex, selfBit);
    }
    GroupWait(*transportGroup, signalIndex, allBit);
    HCCL_INFO("[CcuContextReduceMeshMem2Mem2D] PostSync run finished");
}

void CcuContextReduceMeshMem2Mem2D::AxisSync(uint32_t signalIndex) // 轴间同步
{
    const uint32_t DIE_NUM = 2;
    LocalCtxPost(anotherAxisSignal_, 1 << (axisId_ + signalIndex * DIE_NUM));
    LocalWait(localAxisSignal_, 1 << (1 - axisId_ + signalIndex * DIE_NUM));
    HCCL_INFO("[CcuContextReduceMeshMem2Mem2D] AxisSync run finished");
    return;
}

void CcuContextReduceMeshMem2Mem2D::ReduceStep1()
{
    HCCL_INFO("[CcuContextReduceMeshMem2Mem2D] RankId [%u], axisId [%u],Reduce Step1 starts", rankId_, axisId_);
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));
    CcuRep::Memory dst = CreateMemory();
    CcuRep::Memory src = CreateMemory();
    dst.addr = input_[localId_]; // step1 reduce到input
    dst.token = token_[localId_];
    bool isYAxis = (axisId_ == Y_AXIS_ID);
    CcuRep::Memory tmpDst = CreateMemory();
    chunkSize_ = isYAxis ? yChunkSize_ : xChunkSize_;
    for (uint16_t i = 0; i < (localSize_ - 1); i++) { // 外层循环控制步数=chunk数量
        // 读不同rank的不同chunk
        for (uint16_t rmtId = 0; rmtId < localSize_; ++rmtId) {
            if (rmtId == localId_) {
                continue;
            }
            src.addr = input_[rmtId];
            src.token = token_[rmtId];
            tmpDst.addr = dst.addr;
            tmpDst.token = dst.token;
            if (isYAxis) { // 第一步y slice size要在y轴方向reduce
                src.addr += yAxisOffset_;
                tmpDst.addr += yAxisOffset_;
            }
            chunkOffset_ = 0;
            uint16_t chkId = 0;
            if (rmtId < localId_) {
                chkId = (i + rmtId) % (localSize_ - 1);
            } else {
                chkId = (i + rmtId - 1) % (localSize_ - 1);
            }
            // 计算一下offset 0~(child-1)
            for (uint16_t j = 0; j < chkId; ++j) {
                chunkOffset_ += chunkSize_[j];
            }
            // 更新对应的addr
            src.addr += chunkOffset_;
            tmpDst.addr += chunkOffset_;
            CCU_IF(chunkSize_[chkId] == 0) { LocalPost(locMask_, 1 << rmtId); }

            CCU_IF(chunkSize_[chkId] != 0)
            {
                uint16_t transId = rmtId < localId_ ? rmtId : rmtId - 1;
                ReadReduce(
                    *transports[transId], tmpDst, src, chunkSize_[chkId], dataType_, reduceOp_, locMask_, 1 << rmtId);
            }
        }
        LocalWait(locMask_, allBit);
    }
    HCCL_INFO("[CcuContextReduceMeshMem2Mem2D] Reduce Step1 ends");
}

void CcuContextReduceMeshMem2Mem2D::ReduceStep2()
{
    HCCL_INFO("[CcuContextReduceMeshMem2Mem2D] RankId [%u] Reduce Step2 starts", rankId_);
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));
    CcuRep::Memory dst = CreateMemory();
    CcuRep::Memory src = CreateMemory();
    dst.addr = output_; // 第二步reduce是从input reduce到root的output
    dst.token = token_[localId_];

    src.addr = input_[localId_];
    src.token = token_[localId_];
    bool isXAxis = (axisId_ == X_AXIS_ID);
    chunkSize_ = isXAxis ? yChunkSize_ : xChunkSize_;
    if (isXAxis) // 第二步y slice size要在x轴方向reduce
    {
        src.addr += yAxisOffset_;
        dst.addr += yAxisOffset_;
    }
    GroupCopy(dst, src, curGoSize_);
    for (uint16_t i = 0; i < (localSize_ - 1); i++) {
        for (uint16_t rmtId = 0; rmtId < localSize_; ++rmtId) {
            if (rmtId == localId_) {
                continue;
            }
            dst.addr = output_;
            src.addr = input_[rmtId];
            src.token = token_[rmtId];
            if (isXAxis) {
                src.addr += yAxisOffset_;
                dst.addr += yAxisOffset_;
            }
            uint16_t chkId = 0;
            chunkOffset_ = 0;
            if (rmtId < localId_) {
                chkId = (i + rmtId) % (localSize_ - 1);
            } else {
                chkId = (i + rmtId - 1) % (localSize_ - 1);
            }
            // 计算一下offset 0~(child-1)
            for (uint16_t j = 0; j < chkId; ++j) {
                chunkOffset_ += chunkSize_[j];
            }
            // 更新对应的addr
            src.addr += chunkOffset_;
            dst.addr += chunkOffset_;
            CCU_IF(chunkSize_[chkId] == 0) { LocalPost(locMask_, 1 << rmtId); }
            CCU_IF(chunkSize_[chkId] != 0)
            {
                uint16_t transId = rmtId < localId_ ? rmtId : rmtId - 1;
                ReadReduce(
                    *transports[transId], dst, src, chunkSize_[chkId], dataType_, reduceOp_, locMask_, 1 << rmtId);
            }
        }
        LocalWait(locMask_, allBit);
    }
    HCCL_INFO("[CcuContextReduceMeshMem2Mem2D] Reduce Step2 ends");
}

void CcuContextReduceMeshMem2Mem2D::Algorithm()
{
    HCCL_INFO("[CcuContextReduceMeshMem2Mem2D] ReduceMeshMem2Mem2D run");
    InitResources();
    LoadArgs();
    PreSync(); // 前同步
    if (rankId_ == rootId_ || (dimId_[1] == rootDimId_[1] && axisId_ == Y_AXIS_ID)
        || (dimId_[0] == rootDimId_[0] && axisId_ == X_AXIS_ID)) {
        // 与root同行的元素要在Y方向规约 同列元素要在X方向规约
        ReduceStep1();
    }
    AxisSync(0);
    PostSync(CKE_IDX_3);
    AxisSync(1);
    if (rankId_ == rootId_) { // 第二步只有root进行readreduce
        ReduceStep2();
    }
    AxisSync(0);
    PostSync(CKE_IDX_0);
    AxisSync(1);
}

std::vector<uint64_t> CcuContextReduceMeshMem2Mem2D::CalMeshChunkSlice(uint64_t dataSize, uint64_t sliceNum)
{
    uint64_t dataCount = dataSize / DataTypeSizeGet(dataType_);
    uint64_t bigDataSliceNum = dataCount % sliceNum;
    uint64_t bigDataSliceSize = (dataCount / sliceNum + 1) * DataTypeSizeGet(dataType_);
    uint64_t smallDataSliceNum = sliceNum - dataCount % sliceNum;
    uint64_t smallDataSliceSize = dataCount / sliceNum * DataTypeSizeGet(dataType_);
    return {bigDataSliceNum, bigDataSliceSize, smallDataSliceNum, smallDataSliceSize};
}

std::vector<uint64_t> CcuContextReduceMeshMem2Mem2D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgReduceMeshMem2Mem2D* taskArg = dynamic_cast<const CcuTaskArgReduceMeshMem2Mem2D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuTaskArgReduceMeshMem2Mem2D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t xAxisSize = taskArg->xAxisSize_;
    uint64_t yAxisSize = taskArg->yAxisSize_;
    uint64_t yAxisOffset = xAxisSize;
    auto xAxisGoSize = CalGoSize(xAxisSize);
    auto yAxisGoSize = CalGoSize(yAxisSize);
    std::vector<uint64_t> processReturn = {inputAddr, outputAddr, tokenInfo, xAxisSize, yAxisSize, yAxisOffset};
    HCCL_INFO(
        "[CcuContextReduceMeshMem2Mem2D] ReduceMeshMem2Mem2D inputAddr [%llu] outputAddr [%llu] "
        "xAxisSize [%llu] yAxisSize [%llu],yAxisOffset[%llu],",
        inputAddr, outputAddr, xAxisSize, yAxisSize, yAxisOffset);
    // mesh chunk for x slice size
    std::vector<uint64_t> xChunkVec = CalMeshChunkSlice(xAxisSize, localSize_ - 1);
    for (uint64_t i = 0; i < xChunkVec[0]; i++) {
        processReturn.push_back(xChunkVec[1]);
    }
    for (uint64_t i = 0; i < xChunkVec[2]; i++) {
        processReturn.push_back(xChunkVec[3]);
    }
    // mesh chunk for y slice size
    std::vector<uint64_t> yChunkVec = CalMeshChunkSlice(yAxisSize, localSize_ - 1);
    for (uint64_t i = 0; i < yChunkVec[0]; i++) {
        processReturn.push_back(yChunkVec[1]);
    }
    for (uint64_t i = 0; i < yChunkVec[2]; i++) {
        processReturn.push_back(yChunkVec[3]);
    }
    // for gosize
    processReturn.insert(processReturn.end(), xAxisGoSize.begin(), xAxisGoSize.end());
    processReturn.insert(processReturn.end(), yAxisGoSize.begin(), yAxisGoSize.end());
    return processReturn;
}
} // namespace Hccl
