/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_reduce_mesh1d_one_shot.h"
#include "ccu_instruction_all_reduce_mesh1d_one_shot.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;

CcuContextAllReduceMesh1DOneShot::CcuContextAllReduceMesh1DOneShot(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgAllReduceMesh1DOneShot* ctxArg = dynamic_cast<const CcuCtxArgAllReduceMesh1DOneShot*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuCtxArgAllReduceMesh1DOneShot::ctxArg ptr is null"));
    }
    notifySignal_ = ctxArg->notifySignal_;
    rankId_ = ctxArg->rankId_;
    rankSize_ = ctxArg->dimSize_[0];
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    reduceOp_ = ctxArg->op_.reduceOp;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextAllReduceMesh1DOneShot] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }
    HCCL_INFO(
        "[CcuContextAllReduceMesh1DOneShot] Init, CtxArgs are notifySignal_[%s], rankId[%u], rankSize[%llu], "
        "dataType[%s], "
        "outputDataType[%s], reduceOp[%s]",
        notifySignal_.c_str(), rankId_, rankSize_, dataType_.Describe().c_str(), outputDataType_.Describe().c_str(),
        reduceOp_.Describe().c_str());
}

void CcuContextAllReduceMesh1DOneShot::Algorithm()
{
    HCCL_INFO("[CcuContextAllReduceMesh1DOneShot] AllReduceMesh1DOneShot start");
    InitResource();
    LoadArgs(); // 加载 taskArg 参数
    Presync();  // 跨卡前同步，交换参数信息

    DoGroupReduce();

    Postsync(); // 所有搬运任务结束后，跨卡后同步

    HCCL_INFO("[CcuContextAllReduceMesh1DOneShot] AllReduceMesh1DOneShot end");
    return;
}

void CcuContextAllReduceMesh1DOneShot::InitResource()
{
    HCCL_INFO("[CcuContextAllReduceMesh1DOneShot] InitResource start");
    // 初始化资源
    output_ = CreateVariable();
    uint16_t transportIdx = 0;
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMesh1DOneShot transports is empty"));
    }
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextAllReduceMesh1DOneShot] MyRank[%u], PeerId[%llu], TransportId[%u]", rankId_, peerId,
                transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr,
                HCCL_ERROR("[CcuContextAllReduceMesh1DOneShot] Algorithm transport ptr is null"), );
            input_.push_back(CreateVariable((*transports[transportIdx]), INPUT_XN_ID));
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    groupOpSize_ = CreateGroupOpSize();

    HCCL_INFO("[CcuContextAllReduceMesh1DOneShot] InitResource end");
}

void CcuContextAllReduceMesh1DOneShot::LoadArgs()
{
    HCCL_INFO("[CcuContextAllReduceMesh1DOneShot] LoadArgs start");
    Load(input_[rankId_]);
    Load(output_);
    Load(token_[rankId_]);
    Load(groupOpSize_);
    HCCL_INFO("[CcuContextAllReduceMesh1DOneShot] LoadArgs end");
}

void CcuContextAllReduceMesh1DOneShot::Presync()
{
    HCCL_INFO("[CcuContextAllReduceMesh1DOneShot] Presync start");
    uint16_t selfBit = 1 << rankId_;
    uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_));
    for (auto t : transports) {
        WriteVariableWithSignal(*t, input_[rankId_], INPUT_XN_ID, CKE_IDX_1, selfBit);
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_2, selfBit);
    }

    GroupWait(*transportGroup, CKE_IDX_1, allBit);
    GroupWait(*transportGroup, CKE_IDX_2, allBit);
    HCCL_INFO("[CcuContextAllReduceMesh1DOneShot] Presync end");
}

void CcuContextAllReduceMesh1DOneShot::Postsync()
{
    HCCL_INFO("[CcuContextAllReduceMesh1DOneShot] Postsync start");
    uint16_t selfBit = 1 << rankId_;
    uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_));
    for (auto t : transports) {
        RemotePost(*t, CKE_IDX_0, selfBit);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit);
    HCCL_INFO("[CcuContextAllReduceMesh1DOneShot] Postsync end");
}

void CcuContextAllReduceMesh1DOneShot::DoGroupReduce()
{
    HCCL_INFO("[CcuContextAllReduceMesh1DOneShot] DoGroupReduce start");
    // 初始化地址寄存器
    std::vector<CcuRep::Memory> reduceSrc;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceSrc.push_back(CreateMemory());
    }
    CcuRep::Memory reduceDst = CreateMemory();

    // 填充地址
    uint32_t dstId = 0;
    uint32_t curId = 0;
    // SRC
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = rankSize_ - 1;
        }
        reduceSrc[curId].addr = input_[rankIdx];
        reduceSrc[curId].token = token_[rankIdx];
    }

    // DST
    reduceDst.addr = output_;
    reduceDst.token = token_[rankId_];

    // 执行 reduce 操作
    GroupReduce(transports, reduceDst, reduceSrc, groupOpSize_, dataType_, outputDataType_, reduceOp_);
    HCCL_INFO("[CcuContextAllReduceMesh1DOneShot] DoGroupReduce end");
    return;
}

std::vector<uint64_t> CcuContextAllReduceMesh1DOneShot::GeneArgs(const CcuTaskArg& arg)
{
    HCCL_INFO("[CcuContextAllReduceMesh1DOneShot] GeneArgs start");
    const CcuTaskArgAllReduceMesh1DOneShot* taskArg = dynamic_cast<const CcuTaskArgAllReduceMesh1DOneShot*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMesh1DOneShot::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t sliceSize = taskArg->sliceSize_;

    auto mainBlockGoSize = CalGoSize(sliceSize);

    HCCL_INFO(
        "[CcuContextAllReduceMesh1DOneShot] GeneArgs, taskArg are inputAddr[%llu], outputAddr[%llu], "
        "sliceSize[%llu]",
        inputAddr, outputAddr, sliceSize);

    std::vector<uint64_t> taskArgList{inputAddr, outputAddr, tokenInfo};
    for (auto val : mainBlockGoSize) {
        taskArgList.push_back(val);
    }

    HCCL_INFO("[CcuContextAllReduceMesh1DOneShot] GeneArgs end");
    return taskArgList;
}

} // namespace Hccl
