/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_reduce_scatter_mesh2d_mem2mem.h"
#include "ccu_instruction_reduce_scatter_mesh2d_mem2mem.h"

namespace Hccl {

constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;
constexpr int CKE_IDX_4 = 4;
constexpr int INPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int FST_AXIS_ID = 0;
constexpr int SEC_AXIS_ID = 1;
constexpr int X_AXIS_ID = 0;
constexpr int Y_AXIS_ID = 1;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint64_t LOOP_COUNT = 8;
constexpr uint64_t LOCAL_COPY_MS = 8;
constexpr uint32_t max_dimSize = 2;

CcuContextReduceScatterMeshMem2Mem2D::CcuContextReduceScatterMeshMem2Mem2D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    HCCL_INFO("[CcuContextReduceScatterMeshMem2Mem2D] Enter Constructor");

    const CcuCtxArgReduceScatterMeshMem2Mem2D* ctxArg = dynamic_cast<const CcuCtxArgReduceScatterMeshMem2Mem2D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMeshMem2Mem2D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    dimSize_ = ctxArg->dimSize_;
    axisId_ = ctxArg->axisId_;
    if (dimSize_.size() != max_dimSize || axisId_ > 1 || dimSize_[0] == 0) { // 2D 拓扑校验
        THROW<NullPtrException>(StringFormat(
            "[CcuContextReduceScatterMeshMem2Mem2D] dimSize[%zu] or axisId[%u] or dimSize[0] [%u] is invalid",
            dimSize_.size(), axisId_, dimSize_[0]));
    }
    dimId_.emplace_back(rankId_ % dimSize_[0]); // 当前 rank 所在列编号, 亦即 x 方向的 localId_
    dimId_.emplace_back(rankId_ / dimSize_[0]); // 当前 rank 所在行编号, 亦即 y 方向的 localId_
    localId_ = dimId_[axisId_];                 // 当前 rank 在 axisId_ 轴上的编号
    localSize_ = dimSize_[axisId_];             // mesh2d 拓扑在 axisId_ 轴上的维度
    oppsiteSize_ = dimSize_[1 - axisId_];

    HCCL_INFO(
        "[CcuContextReduceScatterMeshMem2Mem2D] RankId[%u], DimSize0[%u], "
        "DimSize1[%u], localId[%u], localSize[%u], oppositeSize[%u]",
        rankId_, dimSize_[0], dimSize_[1], localId_, localSize_, oppsiteSize_);

    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;

    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextReduceScatterMeshMem2Mem2D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }

    reduceOp_ = ctxArg->op_.reduceOp;

    localAxisSignalName_ = "CcuContextReduceScatterMeshMem2Mem2DAxisSync_" + std::to_string(axisId_);
    anotherAxisSignalName_ = "CcuContextReduceScatterMeshMem2Mem2DAxisSync_" + std::to_string(1 - axisId_);

    moConfig.loopCount = LOOP_COUNT;                 // loop 展开 8 次 or 16 次
    moConfig.msInterleave = LOCAL_COPY_MS;           // 一个 loop 8 个 MS
    moConfig.memSlice = LOCAL_COPY_MS * CCU_MS_SIZE; // 32k
    if (moRes.executor.size() == 0) {
        moRes.executor = CreateBlockExecutor(moConfig.loopCount);
        moRes.maskSignal = CreateBlockMaskSignal(moConfig.loopCount);
        moRes.ccuBuffer = CreateBlockCcuBuffer(moConfig.loopCount * moConfig.msInterleave);
    }
}

void CcuContextReduceScatterMeshMem2Mem2D::InitResources()
{
    step0BaseOffset_ = CreateVariable();
    step0AddOffset_ = CreateVariable();
    step1AddOffset_ = CreateVariable();
    yAxisOffset_ = CreateVariable();
    xAxisSize_ = CreateVariable();
    yAxisSize_ = CreateVariable();
    localAxisSignal_ = CreateMaskSignal();
    anotherAxisSignal_ = CreateMaskSignal();
    xAxisGroupOpSize_ = CreateGroupOpSize();
    yAxisGroupOpSize_ = CreateGroupOpSize();
    curGoSize_ = CreateGroupOpSize();

    ExportMaskSignal(localAxisSignal_, localAxisSignalName_);
    anotherAxisSignal_ = ImportMaskSignal(anotherAxisSignalName_);

    output_.push_back(CreateVariable());

    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMeshMem2Mem2D transports is empty"));
    }

    // 从小到大遍历 transports, 遇到本 rank 就填充本地资源
    // 否则依次取远端资源, 要求给框架返回的 Link 同样是按顺序排列的
    uint32_t transportIdx = 0;
    for (uint64_t peerId = 0; peerId < localSize_; peerId++) {
        if (peerId == localId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextReduceScatterMeshMem2Mem2D] MyRankLocalId[%u], PeerId[%llu], TransportId[%u]", localId_,
                peerId, transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr,
                HCCL_ERROR("[CcuContextReduceScatterMeshMem2Mem2D] Algorithm transport ptr is null"), );
            input_.push_back(CreateVariable((*transports[transportIdx]), INPUT_XN_ID));
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }

    strideSize_ = CreateVariable();
    for (uint64_t i = 0; i < localSize_ - 1; i++) {
        xlocalSlice_.push_back(CreateVariable());
        ylocalSlice_.push_back(CreateVariable());
    }

    HCCL_INFO("[CcuContextReduceScatterMeshMem2Mem2D] InitResources finished");
}

void CcuContextReduceScatterMeshMem2Mem2D::AxisSync(uint32_t signalIndex)
{
    const uint32_t DIE_NUM = 2;
    if (signalIndex > 1) {
        THROW<InvalidParamsException>(
            StringFormat("[CcuContextReduceScatterMeshMem2Mem2D] Unexpected SignalIndex[%u]", signalIndex));
    }
    LocalCtxPost(anotherAxisSignal_, 1 << (axisId_ + signalIndex * DIE_NUM));
    LocalWait(localAxisSignal_, 1 << (1 - axisId_ + signalIndex * DIE_NUM));
    HCCL_INFO("[CcuContextReduceScatterMeshMem2Mem2D] AxisSync run finished");
}

void CcuContextReduceScatterMeshMem2Mem2D::PreSync()
{
    uint16_t selfBit = 1 << localId_; // selfBit = 1*2^{localId_}
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        WriteVariableWithSignal(*t, input_[localId_], INPUT_XN_ID, CKE_IDX_1, selfBit); // index = 1，传递output信息
        WriteVariableWithSignal(*t, token_[localId_], TOKEN_XN_ID, CKE_IDX_2, selfBit); // index = 2，传递token信息
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit); // index = 1，传递output信息
    GroupWait(*transportGroup, CKE_IDX_2, allBit); // index = 2，传递token信息
    HCCL_INFO("[CcuContextReduceScatterMeshMem2Mem2D] PreSync run finished");
}

void CcuContextReduceScatterMeshMem2Mem2D::PostSync(uint32_t signalIndex)
{
    uint16_t selfBit = 1 << localId_;
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        RemotePost(*t, signalIndex, selfBit);
    }
    GroupWait(*transportGroup, signalIndex, allBit);
    HCCL_INFO("[CcuContextReduceScatterMeshMem2Mem2D] PostSync run finished");
}

void CcuContextReduceScatterMeshMem2Mem2D::LoadArgs()
{
    Load(input_[localId_]);
    Load(output_[0]);
    Load(token_[localId_]);

    Load(step0BaseOffset_);
    Load(step0AddOffset_);
    Load(step1AddOffset_);

    Load(yAxisOffset_);
    Load(xAxisSize_);
    Load(yAxisSize_);

    for (uint64_t i = 0; i < localSize_ - 1; i++) {
        Load(xlocalSlice_[i]);
    }
    for (uint64_t i = 0; i < localSize_ - 1; i++) {
        Load(ylocalSlice_[i]);
    }

    Load(xAxisGroupOpSize_);
    Load(yAxisGroupOpSize_);

    curGoSize_ = (axisId_ == 0) ? yAxisGroupOpSize_ : xAxisGroupOpSize_;

    HCCL_INFO("[CcuContextReduceScatterMeshMem2Mem2D] LoadArgs run finished");
}

void CcuContextReduceScatterMeshMem2Mem2D::CreateLocalCopyLoop()
{
    std::string loopType = "reducescatter";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }

    for (uint32_t index = 0; index < 2; index++) { // 需要 2 个 Loop
        CcuRep::Memory src = CreateMemory();
        CcuRep::Memory dst = CreateMemory();
        CcuRep::Variable len = CreateVariable();
        CcuRep::LoopBlock lb(this, loopType + "_localcopy_loop_" + std::to_string(index));
        lb(src, dst, len);

        CcuRep::MaskSignal sem = moRes.maskSignal[index];
        std::vector<CcuRep::CcuBuffer> bufs;
        for (uint32_t i = 0; i < LOCAL_COPY_MS; i++) {
            bufs.push_back(moRes.ccuBuffer[i]);
        }

        LocalCopy(bufs[0], src, len, sem);
        LocalWait(sem);
        LocalCopy(dst, bufs[0], len, sem);
        LocalWait(sem);
    }
    registeredLoop.insert(loopType);
}

void CcuContextReduceScatterMeshMem2Mem2D::LocalCopyByLoopGroup(CcuRep::Memory dst, CcuRep::Memory src)
{
    CreateLocalCopyLoop();

    CCU_IF(curGoSize_.addrOffset != 0)
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam = CcuRep::GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += curGoSize_.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        auto lc = Loop("reducescatter_localcopy_loop_0")(src, dst, sliceSize);

        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }

    CCU_IF(curGoSize_.parallelParam != 0)
    {
        CcuRep::Condition cond(this, curGoSize_.parallelParam != 0);

        src.addr += curGoSize_.addrOffset;
        dst.addr += curGoSize_.addrOffset;
        auto lc0 = Loop("reducescatter_localcopy_loop_0")(src, dst, curGoSize_.residual);

        src.addr += curGoSize_.residual;
        dst.addr += curGoSize_.residual;
        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        auto lc1 = Loop("reducescatter_localcopy_loop_1")(src, dst, sliceSize);

        CcuRep::Variable loopCfg0 = CreateVariable();
        loopCfg0 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, curGoSize_.parallelParam, offsetCfg);
    }
}

std::vector<uint64_t> CcuContextReduceScatterMeshMem2Mem2D::CalMeshChunkSlice(uint64_t dataSize, uint64_t sliceNum)
{
    uint64_t dataCount = dataSize / DataTypeSizeGet(dataType_);

    uint64_t bigDataSliceNum = dataCount % sliceNum;
    uint64_t bigDataSliceSize = (dataCount / sliceNum + 1) * DataTypeSizeGet(dataType_);
    uint64_t smallDataSliceNum = sliceNum - dataCount % sliceNum;
    uint64_t smallDataSliceSize = dataCount / sliceNum * DataTypeSizeGet(dataType_);

    return {bigDataSliceNum, bigDataSliceSize, smallDataSliceNum, smallDataSliceSize};
}

void CcuContextReduceScatterMeshMem2Mem2D::Step1Reduce()
{
    std::vector<CcuRep::Memory> src;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        src.push_back(CreateMemory());
    }
    for (uint32_t localIdx = 0; localIdx < localSize_; localIdx++) {
        src[localIdx].addr = input_[localIdx];
        src[localIdx].token = token_[localIdx];
    }

    CcuRep::Memory dst = CreateMemory();
    dst.addr = input_[localId_];
    dst.token = token_[localId_];

    CcuRep::Memory tempDst = CreateMemory();
    CcuRep::Memory tempSrc = CreateMemory();

    for (uint32_t oppositeIdx = 0; oppositeIdx < oppsiteSize_; oppositeIdx++) {
        // 由 oppositeIdx 造成的地址偏移
        for (uint32_t localIdx = 0; localIdx < localSize_; localIdx++) {
            src[localIdx].addr += (oppositeIdx == 0) ? step0BaseOffset_ : step0AddOffset_;
        }
        dst.addr += (oppositeIdx == 0) ? step0BaseOffset_ : step0AddOffset_;

        bool isXAxis = (axisId_ == X_AXIS_ID);
        CcuRep::Variable len = isXAxis ? xAxisSize_ : yAxisSize_;
        std::vector<CcuRep::Variable> sliceSize = isXAxis ? xlocalSlice_ : ylocalSlice_;
        uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

        CcuRep::MaskSignal localMask = CreateMaskSignal();
        for (uint32_t i = 0; i < localSize_ - 1; i++) {
            CcuRep::Variable sliceOffset = CreateVariable();
            CcuRep::Variable strideOffset = CreateVariable();
            sliceOffset = 0;
            strideOffset = 0;
            for (uint64_t j = 0; j < localSize_ - 1; j++) { // 遍历 rmt rank
                tempDst.addr = dst.addr;                    // local tempDst 初始化
                tempDst.token = dst.token;
                tempDst.addr += sliceOffset; // local tempDst 地址按照 chunk 大小偏移
                uint16_t nextNum = i + j + 1;
                if (nextNum >= localSize_) {
                    nextNum += 1;
                }
                uint16_t rmtRank = (localId_ + nextNum) % localSize_;
                tempSrc.addr = src[rmtRank].addr;
                tempSrc.token = src[rmtRank].token;
                tempSrc.addr += sliceOffset; // rmt tempSrc 地址按照 chunk 大小偏移

                CCU_IF(sliceSize[j] == 0)
                {
                    LocalPost(localMask, 1 << rmtRank);
                    continue;
                }
                uint16_t rmtTransport;
                if (rmtRank < localId_) {
                    rmtTransport = rmtRank;
                } else {
                    rmtTransport = rmtRank - 1;
                }
                ReadReduce(
                    *transports[rmtTransport], tempDst, tempSrc, sliceSize[j], dataType_, reduceOp_, localMask,
                    1 << rmtRank);
                sliceOffset += sliceSize[j];
            }
            LocalWait(localMask, allBit);
        }
    }

    HCCL_INFO("[CcuContextReduceScatterMeshMem2Mem2D] Step1Reduce run finished");
}

void CcuContextReduceScatterMeshMem2Mem2D::Step2Reduce()
{
    std::vector<CcuRep::Memory> src;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        src.push_back(CreateMemory());
    }
    for (uint32_t localIdx = 0; localIdx < localSize_; localIdx++) {
        src[localIdx].addr = input_[localIdx];
        src[localIdx].addr += step1AddOffset_;
        src[localIdx].token = token_[localIdx];
    }

    CcuRep::Memory dst = CreateMemory();
    dst.addr = output_[0];
    dst.token = token_[localId_];
    if (axisId_ == X_AXIS_ID) {
        dst.addr += yAxisOffset_;
    }

    CcuRep::Memory tempDst = CreateMemory();
    CcuRep::Memory tempSrc = CreateMemory();

    bool isXAxis = (axisId_ == X_AXIS_ID);
    CcuRep::Variable len = isXAxis ? yAxisSize_ : xAxisSize_;
    std::vector<CcuRep::Variable> sliceSize = isXAxis ? ylocalSlice_ : xlocalSlice_;
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    CcuRep::MaskSignal localMask = CreateMaskSignal();
    for (uint32_t i = 0; i < localSize_ - 1; i++) {
        CcuRep::Variable sliceOffset = CreateVariable();
        sliceOffset = 0;
        for (uint64_t j = 0; j < localSize_ - 1; j++) { // 遍历 rmt rank
            tempDst.addr = src[localId_].addr;
            tempDst.token = src[localId_].token;
            tempDst.addr += sliceOffset; // dst 地址偏移
            uint16_t nextNum = i + j + 1;
            if (nextNum >= localSize_) {
                nextNum += 1;
            }
            uint16_t rmtRank = (localId_ + nextNum) % localSize_;
            tempSrc.addr = src[rmtRank].addr;
            tempSrc.token = src[rmtRank].token;
            tempSrc.addr += sliceOffset; // src 地址偏移

            CCU_IF(sliceSize[j] == 0)
            {
                LocalPost(localMask, 1 << rmtRank);
                continue;
            }
            uint16_t rmtTransport;
            if (rmtRank < localId_) {
                rmtTransport = rmtRank;
            } else {
                rmtTransport = rmtRank - 1;
            }
            ReadReduce(
                *transports[rmtTransport], tempDst, tempSrc, sliceSize[j], dataType_, reduceOp_, localMask,
                1 << rmtRank);
            sliceOffset += sliceSize[j];
        }
        LocalWait(localMask, allBit);
    }

    LocalCopyByLoopGroup(dst, src[localId_]); // 将计算结果 copy 到 output_

    HCCL_INFO("[CcuContextReduceScatterMeshMem2Mem2D] Step2Reduce run finished");
}

void CcuContextReduceScatterMeshMem2Mem2D::Algorithm()
{
    HCCL_INFO("[CcuContextReduceScatterMeshMem2Mem2D] ReduceScatterMeshMem2Mem2D run");

    InitResources(); // 读取数据
    LoadArgs();
    HCCL_INFO("[CcuContextReduceScatterMeshMem2Mem2D] Algorithm first step begins");
    PreSync();

    Step1Reduce();
    PostSync(CKE_IDX_3);
    AxisSync(FST_AXIS_ID);

    HCCL_INFO("[CcuContextReduceScatterMeshMem2Mem2D] Algorithm second step begins");
    PostSync(CKE_IDX_4);

    Step2Reduce();
    PostSync(CKE_IDX_0);
    AxisSync(SEC_AXIS_ID);

    HCCL_INFO("[CcuContextReduceScatterMeshMem2Mem2D] ReduceScatterMeshMem2Mem2D end");
    return;
}

std::vector<uint64_t> CcuContextReduceScatterMeshMem2Mem2D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgReduceScatterMeshMem2Mem2D* taskArg
        = dynamic_cast<const CcuTaskArgReduceScatterMeshMem2Mem2D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceScatterMeshMem2Mem2D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t outputSize = taskArg->outputSize_;
    uint64_t offset = taskArg->offset_;
    uint64_t yAxisOffset = taskArg->xAxisSize_;
    uint64_t xAxisSize = taskArg->xAxisSize_;
    uint64_t yAxisSize = taskArg->yAxisSize_;

    /* @brief 计算 mesh-chunk 分块数量和数据大小
     * @param 总是分成 localSize_ - 1 块,
     * @param xlocalSlice: 把 xAxisSize 切分成 localSize_ - 1 份
     * @param ylocalSlice: 把 yAxisSize 切分成 localSize_ - 1 份
     * @vector bigDataSliceNum, bigDataSliceSize, smallDataSliceNum, smallDataSliceSize
     */
    std::vector<uint64_t> xlocalSlice = CalMeshChunkSlice(xAxisSize, localSize_ - 1);
    std::vector<uint64_t> ylocalSlice = CalMeshChunkSlice(yAxisSize, localSize_ - 1);

    // 计算不同die的数据
    uint64_t step0BaseOffset
        = (axisId_ == 0) ? dimId_[0] * outputSize + offset : dimId_[1] * dimSize_[0] * outputSize + offset + xAxisSize;
    uint64_t step0AddOffset = (axisId_ == 0) ? dimSize_[0] * outputSize : outputSize;
    uint64_t step1AddOffset = rankId_ * outputSize + offset + (axisId_ == 0 ? xAxisSize : 0);
    auto xAxisGoSize = CalGoSize(xAxisSize);
    auto yAxisGoSize = CalGoSize(yAxisSize);

    std::vector<uint64_t> processReturn = {inputAddr,      outputAddr,  tokenInfo, step0BaseOffset, step0AddOffset,
                                           step1AddOffset, yAxisOffset, xAxisSize, yAxisSize};

    for (uint64_t i = 0; i < xlocalSlice[0]; i++) { // bigData 块
        processReturn.push_back(xlocalSlice[1]);
    }
    for (uint64_t i = 0; i < xlocalSlice[2]; i++) { // smallData 块
        processReturn.push_back(xlocalSlice[3]);
    }

    for (uint64_t i = 0; i < ylocalSlice[0]; i++) { // bigData 块
        processReturn.push_back(ylocalSlice[1]);
    }
    for (uint64_t i = 0; i < ylocalSlice[2]; i++) { // smallData 块
        processReturn.push_back(ylocalSlice[3]);
    }

    for (auto goSize : {xAxisGoSize, yAxisGoSize}) {
        for (auto val : goSize) {
            processReturn.push_back(val);
        }
    }
    HCCL_INFO(
        "[CcuContextReduceScatterMeshMem2Mem2D] GeneArgs: inputAddr[%llu], outputAddr[%llu],"
        "step0BaseOffset[%llu], step0AddOffset[%llu], step1AddOffset[%llu]",
        inputAddr, outputAddr, step0BaseOffset, step0AddOffset, step1AddOffset);

    return processReturn;
}
} // namespace Hccl
