/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "orion_adapter_rts.h"
#include "ccu_context_all_reduce_mesh1d.h"
#include "ccu_instruction_all_reduce_mesh1d.h"

namespace Hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;

CcuContextAllReduceMesh1D::CcuContextAllReduceMesh1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgAllReduceMesh1D* ctxArg = dynamic_cast<const CcuCtxArgAllReduceMesh1D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMesh1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    rankSize_ = ctxArg->dimSize_[0];
    dataType_ = ctxArg->op_.dataType;
    outputDataType_ = ctxArg->op_.outputDataType;
    if (outputDataType_ == DataType::INVALID) {
        outputDataType_ = dataType_;
        HCCL_INFO(
            "[CcuContextAllReduceMesh1D] outputDataType is [INVALID], set outputDataType to[%s]",
            outputDataType_.Describe().c_str());
    }

    reduceOp_ = ctxArg->op_.reduceOp;
    HCCL_DEBUG(
        "[CcuContextAllReduceMesh1D] Init, CtxArgs are rankId[%u], rankSize[%u], dataType[%s], "
        "outputDataType[%s], reduceOp[%s]",
        rankId_, rankSize_, dataType_.Describe().c_str(), outputDataType_.Describe().c_str(),
        reduceOp_.Describe().c_str());

    // 判断device类型
    int32_t devLogicId = HrtGetDevice();
    if (CcuDeviceManager::GetCcuVersion(devLogicId, ccuVersion_) != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("Cannot get ccu version: %s", __func__);
    }
}

void CcuContextAllReduceMesh1D::RunBroadcast(std::vector<CcuRep::Memory>& dst, CcuRep::Memory& src)
{
    if (ccuVersion_ == CcuVersion::CCU_V1) {
        GroupBroadcast(transports, dst, src, groupOpSize_);
    } else {
        THROW<NotSupportException>(StringFormat("CCU version not support, version[%u]", ccuVersion_));
    }
}

void CcuContextAllReduceMesh1D::RunReduce(CcuRep::Memory& dst, std::vector<CcuRep::Memory>& src)
{
    if (ccuVersion_ == CcuVersion::CCU_V1) {
        GroupReduce(transports, dst, src, groupOpSize_, dataType_, outputDataType_, reduceOp_);
    } else {
        THROW<NotSupportException>(StringFormat("CCU version not support, version[%u]", ccuVersion_));
    }
}

void CcuContextAllReduceMesh1D::Algorithm()
{
    HCCL_INFO("[CcuContextAllReduceMesh1D] AllReduceMesh1D run");
    uint16_t selfBit = 1 << rankId_;
    uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_));

    // 初始化资源
    uint16_t transportIdx = 0;
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMesh1D transports is empty"));
    }
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextAllReduceMesh1D] MyRank[%u], PeerId[%llu], TransportId[%u]", rankId_, peerId, transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr,
                HCCL_ERROR("[CcuContextAllReduceMesh1D] Algorithm transport ptr is null"), );
            input_.push_back(CreateVariable((*transports[transportIdx]), INPUT_XN_ID));
            output_.push_back(CreateVariable((*transports[transportIdx]), OUTPUT_XN_ID));
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    offset_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();

    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(offset_);

    if (ccuVersion_ == CcuVersion::CCU_V1) {
        Load(groupOpSize_);
    } else {
        THROW<NotSupportException>(StringFormat("CCU version not support, version[%u]", ccuVersion_));
    }

    for (auto t : transports) {
        WriteVariableWithSignal(*t, input_[rankId_], INPUT_XN_ID, CKE_IDX_1, selfBit);
        WriteVariableWithSignal(*t, output_[rankId_], OUTPUT_XN_ID, CKE_IDX_2, selfBit);
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_3, selfBit);
    }

    GroupWait(*transportGroup, CKE_IDX_1, allBit);
    GroupWait(*transportGroup, CKE_IDX_2, allBit);
    GroupWait(*transportGroup, CKE_IDX_3, allBit);

    std::vector<CcuRep::Memory> reduceScatterSrc;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceScatterSrc.push_back(CreateMemory());
    }
    CcuRep::Memory reduceScatterDst = CreateMemory();
    // DST
    reduceScatterDst.addr = output_[rankId_];
    reduceScatterDst.addr += offset_;
    reduceScatterDst.token = token_[rankId_];

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
        reduceScatterSrc[curId].addr = input_[rankIdx];
        reduceScatterSrc[curId].addr += offset_;
        reduceScatterSrc[curId].token = token_[rankIdx];
    }
    RunReduce(reduceScatterDst, reduceScatterSrc);

    CcuRep::Memory allGatherSrc = CreateMemory();
    std::vector<CcuRep::Memory> allGatherDst;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        allGatherDst.push_back(CreateMemory());
    }
    // allGather 的输入就是 reduceScatter 的输出
    allGatherSrc.addr = output_[rankId_];
    allGatherSrc.addr += offset_;
    allGatherSrc.token = token_[rankId_];

    dstId = 0;
    curId = 0;
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
    RunBroadcast(allGatherDst, allGatherSrc);

    for (auto t : transports) {
        RemotePost(*t, CKE_IDX_0, selfBit);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit);
    HCCL_INFO("[CcuContextAllReduceMesh1D] AllReduceMesh1D end");
    return;
}

std::vector<uint64_t> CcuContextAllReduceMesh1D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgAllReduceMesh1D* taskArg = dynamic_cast<const CcuTaskArgAllReduceMesh1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllReduceMesh1D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t sliceSize = taskArg->sliceSize_;
    uint64_t offset = taskArg->offset_;

    if (ccuVersion_ == CcuVersion::CCU_V1) {
        auto goSize = CalGoSize(sliceSize);

        HCCL_INFO(
            "[CcuContextAllReduceMesh1D] GeneArgs, taskArg are inputAddr[%llu], outputAddr[%llu], "
            "offset[%llu], sliceSize[%llu]",
            inputAddr, outputAddr, offset, sliceSize);
        return {inputAddr, outputAddr, tokenInfo, offset, goSize[0], goSize[1], goSize[2], goSize[3]};
    } else {
        THROW<NotSupportException>(StringFormat("CCU version not support, version[%u]", ccuVersion_));
    }

    return {};
}
} // namespace Hccl
