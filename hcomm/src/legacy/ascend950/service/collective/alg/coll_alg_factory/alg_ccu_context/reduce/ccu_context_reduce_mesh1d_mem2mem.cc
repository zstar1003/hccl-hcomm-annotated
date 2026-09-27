/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_reduce_mesh1d_mem2mem.h"
#include "ccu_instruction_reduce_mesh1d_mem2mem.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;

using CurrentCtxArg = CcuCtxArgReduceMeshMem2Mem1D;
using CurrentTaskArg = CcuTaskArgReduceMeshMem2Mem1D;

CcuContextReduceMeshMem2Mem1D::CcuContextReduceMeshMem2Mem1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CurrentCtxArg* ctxArg = dynamic_cast<const CurrentCtxArg*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceMeshMem2Mem1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    rankSize_ = ctxArg->dimSize_[0];
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextReduceMeshMem2Mem1D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }
    if (ctxArg->dimSize_.size() > 0) {
        rankSize_ = ctxArg->dimSize_[0];
    }
    HCCL_INFO("[CcuContextReduceMeshMem2Mem1D] CtxArg: rankId[%u] rankSize[%llu]", rankId_, rankSize_);
    reduceOp_ = ctxArg->op_.reduceOp;
    rootId_ = ctxArg->rootId_;
    HCCL_INFO(
        "[CcuContextReduceMeshMem2Mem1D] init end, ctxArg->dimSize size[%zu] rankSize[%llu]", ctxArg->dimSize_.size(),
        rankSize_);
}

void CcuContextReduceMeshMem2Mem1D::InitResource()
{
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceMeshMem2Mem1D transports is empty"));
    }
    HCCL_INFO("[CcuContextReduceMeshMem2Mem1D]transports.size: [%zu]", transports.size());
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
                "[CcuContextReduceMeshMem2Mem1D] MyRank[%u], PeerId[%llu], TransportId[%hu]", rankId_, peerId,
                transportIdx);
            // 判断transport是否为空，为空直接报错
            CHK_PRT_RET(
                transports[transportIdx] == nullptr || transportIdx >= transports.size(),
                HCCL_ERROR("[CcuContextReduceMeshMem2Mem1D] Algorithm transport ptr is null or transportIdx is out of "
                           "bounds"), );
            input_.push_back(
                CreateVariable((*transports[transportIdx]), CKE_IDX_0)); // 获取transport中id=1的Var来传递input
            output_.push_back(
                CreateVariable((*transports[transportIdx]), CKE_IDX_1)); // 获取transport中id=2的Var来传递output
            token_.push_back(CreateVariable((*transports[transportIdx]), CKE_IDX_2));
            transportIdx++;
        }
    }
    for (uint16_t roundId = 0; roundId < (rankSize_ - 1); roundId++) {
        chunkSize_.push_back(CreateVariable());
    }
    inputRepeatStride_ = CreateVariable();
    outputRepeatStride_ = CreateVariable();
    normalSliceSize_ = CreateVariable();
    lastSliceSize_ = CreateVariable();
    repeatNumVar_ = CreateVariable();
    flag_ = CreateVariable();
    isInputOutputEqual_ = CreateVariable();
    selfBit_ = 1 << rankId_;
    allBit_ = ((1 << rankSize_) - 1) & (~(1 << rankId_));

    srcMem_ = CreateMemory();
    dstMem_ = CreateMemory();
    locMask_ = CreateMaskSignal();
    localGoSize_ = CreateGroupOpSize();
    chunkOffset_ = CreateVariable();
    AllocGoResource(CCU_MS_LOCAL_COPY_LOOP_COUNT, LOCAL_COPY_MS_PER_LOOP);
}

void CcuContextReduceMeshMem2Mem1D::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(isInputOutputEqual_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(repeatNumVar_);
    for (uint16_t i = 0; i < (rankSize_ - 1); i++) {
        Load(chunkSize_[i]);
    }
    Load(localGoSize_);
}

void CcuContextReduceMeshMem2Mem1D::PreSync()
{
    // 互换内存信息
    HCCL_INFO("[CcuContextReduceMeshMem2Mem1D] ReduceMeshMem2Mem1D LocalPost begin");
    for (auto t : transports) {
        WriteVariableWithSignal(*t, input_[rankId_], INPUT_XN_ID, CKE_IDX_1, selfBit_); // index = 1，传递input信息
        WriteVariableWithSignal(*t, output_[rankId_], OUTPUT_XN_ID, CKE_IDX_2, selfBit_); // index = 0，传递output信息
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_3, selfBit_); // index = 2，传递token信息
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit_);
    GroupWait(*transportGroup, CKE_IDX_2, allBit_);
    GroupWait(*transportGroup, CKE_IDX_3, allBit_);
    HCCL_INFO("[CcuContextReduceMeshMem2Mem1D] ReduceMeshMem2Mem1D wait all end");
}

void CcuContextReduceMeshMem2Mem1D::PostSync()
{
    for (auto t : transports) {
        RemotePost(*t, CKE_IDX_0, selfBit_);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit_);
    HCCL_INFO("[CcuContextReduceMeshMem2Mem1D] ReduceMesh1D Reduce groupwait end");
}

void CcuContextReduceMeshMem2Mem1D::DoRepeatReduce(
    const std::vector<CcuRep::Variable>& srcAddr, const CcuRep::Variable& dstAddr)
{
    // 从远程设备读取数据并逐步归约到本地设备
    CHK_PRT_THROW(
        srcAddr.size() != transports.size() + 1,
        HCCL_ERROR(
            "[ReadReduceRmtToLoc] srcAddr.size[%zu] != transports size[%zu] +1", srcAddr.size(), transports.size()),
        InvalidParamsException, "Invalid srcAddr size");

    dstMem_.addr = dstAddr;
    dstMem_.token = token_[rankId_];

    srcMem_.addr = srcAddr[rankId_];
    srcMem_.token = token_[rankId_];
    CCU_IF(flag_ != 0)
    {
        // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
        dstMem_.addr += outputRepeatStride_;
        srcMem_.addr += inputRepeatStride_;
    }
    CCU_IF(isInputOutputEqual_ == 0) { GroupCopy(dstMem_, srcMem_, localGoSize_); }
    for (uint16_t i = 0; i < (rankSize_ - 1); i++) { // 外层循环控制step
        // 读不同rank的不同chunk
        for (uint16_t rmtId = 0; rmtId < rankSize_; ++rmtId) {
            if (rmtId == rootId_) {
                continue;
            }
            chunkOffset_ = 0;
            dstMem_.addr = dstAddr;
            srcMem_.addr = srcAddr[rmtId];
            srcMem_.token = token_[rmtId];

            CCU_IF(flag_ != 0)
            {
                // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
                dstMem_.addr += outputRepeatStride_;
                srcMem_.addr += inputRepeatStride_;
            }
            uint16_t chkId = 0;
            if (rmtId < rankId_) {
                chkId = (i + rmtId) % (rankSize_ - 1);
            } else {
                chkId = (i + rmtId - 1) % (rankSize_ - 1);
            }
            uint16_t transId = rmtId < rootId_ ? rmtId : rmtId - 1;
            HCCL_DEBUG(
                "[ReadReduceRmtToLoc] debug rankId[%llu], root[%llu] chkId[%llu], rmtId[%llu] transId[%llu]", rankId_,
                rootId_, chkId, rmtId, transId);

            // 计算一下offset 0~(child-1)
            for (uint16_t j = 0; j < chkId; ++j) {
                chunkOffset_ += chunkSize_[j];
            }
            // 更新对应的addr
            srcMem_.addr += chunkOffset_;
            dstMem_.addr += chunkOffset_;

            CCU_IF(chunkSize_[chkId] == 0) { LocalPost(locMask_, 1 << rmtId); }

            CCU_IF(chunkSize_[chkId] != 0)
            {
                ReadReduce(
                    *transports[transId], dstMem_, srcMem_, chunkSize_[chkId], dataType_, reduceOp_, locMask_,
                    1 << rmtId);
            }
        }
        LocalWait(locMask_, allBit_);
    }

    HCCL_INFO("[CcuContextReduceMeshMem2Mem1D] ReduceMeshMem2Mem1D ReadReduce end");
}

void CcuContextReduceMeshMem2Mem1D::Algorithm()
{
    HCCL_INFO("[CcuContextReduceMeshMem2Mem1D] ReduceMeshMem2Mem1D run");
    InitResource();
    HCCL_INFO("[CcuContextReduceMeshMem2Mem1D] ReduceMeshMem2Mem1D load input variables, id: [%u]", rankId_);
    LoadArgs();
    PreSync();
    CCU_IF(normalSliceSize_ != 0) // 所有rank
    {
        if (rankId_ == rootId_) {
            CcuRep::Variable repeatNumAdd = CreateVariable();
            repeatNumAdd = 1;
            flag_ = 0;
            CCU_WHILE(repeatNumVar_ != UINT64_MAX)
            { // 循环repeatNum_次
                // root要去读每个rank每个chunk的数据
                DoRepeatReduce(input_, output_[rankId_]);
                repeatNumVar_ += repeatNumAdd;
                flag_ = 1;
            }
        }
    }
    PostSync();
    HCCL_INFO("[CcuContextReduceMeshMem2Mem1D] ReduceMeshMem2Mem1D end");
    return;
}

std::vector<uint64_t> CcuContextReduceMeshMem2Mem1D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgReduceMeshMem2Mem1D* taskArg = dynamic_cast<const CcuTaskArgReduceMeshMem2Mem1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextReduceMeshMem2Mem1D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;

    uint64_t bigDataSliceNum = taskArg->bigDataSliceNum_;
    uint64_t bigDataSliceSize = taskArg->bigDataSliceSize_;
    uint64_t smallDataSliceNum = taskArg->smallDataSliceNum_;
    uint64_t smallDataSliceSize = taskArg->smallDataSliceSize_;
    uint64_t inputRepeatStride = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize = taskArg->normalSliceSize_;
    uint64_t lastSliceSize = taskArg->lastSliceSize_;
    uint64_t repeatNumVar = taskArg->repeatNumVar_;
    uint64_t isInputOutputEqual = (inputAddr == outputAddr) ? 1 : 0;
    std::vector<uint64_t> taskArgs = {
        inputAddr,          outputAddr,      tokenInfo,     isInputOutputEqual, inputRepeatStride,
        outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar,
    };
    for (uint64_t i = 0; i < bigDataSliceNum; i++) {
        taskArgs.push_back(bigDataSliceSize);
    }
    for (uint64_t i = 0; i < smallDataSliceNum; i++) {
        taskArgs.push_back(smallDataSliceSize);
    }

    auto localGoSize = CalGoSize(normalSliceSize);
    taskArgs.push_back(localGoSize[0]);
    taskArgs.push_back(localGoSize[1]);
    taskArgs.push_back(localGoSize[2]);
    taskArgs.push_back(localGoSize[3]);
    HCCL_INFO(
        "[CcuContextReduceMeshMem2Mem1D] TaskArgs: inputAddr[%llu], outputAddr[%llu], inputRepeatStride[%llu], "
        "outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu], repeatNumVar[%llu], "
        "bigDataSliceNum[%llu], bigDataSliceSize[%llu], smallDataSliceNum[%llu], smallDataSliceSize[%llu], ",
        inputAddr, outputAddr, inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar,
        bigDataSliceNum, bigDataSliceSize, smallDataSliceNum, smallDataSliceSize);
    return taskArgs;
}

} // namespace Hccl
