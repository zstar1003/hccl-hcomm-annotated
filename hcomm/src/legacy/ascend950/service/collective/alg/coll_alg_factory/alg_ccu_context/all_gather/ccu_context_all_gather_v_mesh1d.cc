/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_gather_v_mesh1d.h"
#include "ccu_instruction_all_gather_v_mesh1d.h"

namespace Hccl {

constexpr int OUTPUT_XN_ID = 0;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr uint32_t LOOP_NUMS_PCIE_STD = 16;
constexpr uint32_t MS_NUMS_PER_LOOP_PCIE_STD = 8;

CcuContextAllGatherVMesh1D::CcuContextAllGatherVMesh1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgAllGatherVMesh1D* ctxArg = dynamic_cast<const CcuCtxArgAllGatherVMesh1D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllGatherVMesh1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    rankSize_ = ctxArg->dimSize_[0];
}

void CcuContextAllGatherVMesh1D::LoadArgs()
{
    Load(input_);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(mySliceOffset_);
    Load(groupOpSize_);
    return;
}

void CcuContextAllGatherVMesh1D::InitResources()
{
    uint16_t transportIdx = 0;
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextAllGatherVMesh1D transports is empty"));
    }
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextAllGatherVMesh1D] MyRank[%u], PeerId[%llu], TransportId[%u]", rankId_, peerId,
                transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr,
                HCCL_ERROR("[CcuContextAllGatherVMesh1D] Algorithm transport ptr is null"), );
            output_.push_back(
                CreateVariable((*transports[transportIdx]), OUTPUT_XN_ID)); // 获取transport中id=1的Var来传递output
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    mySliceOffset_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();
    input_ = CreateVariable();
    AllocGoResource(LOOP_NUMS_PCIE_STD, MS_NUMS_PER_LOOP_PCIE_STD);
    return;
}

void CcuContextAllGatherVMesh1D::PreSync()
{
    uint16_t selfBit = 1 << rankId_;
    uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_));
    for (auto t : transports) {
        WriteVariableWithSignal(*t, output_[rankId_], OUTPUT_XN_ID, CKE_IDX_1, selfBit); // index = 1，传递output信息
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_2, selfBit); // index = 2，传递token信息
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit); // index = 1，传递output信息
    GroupWait(*transportGroup, CKE_IDX_2, allBit); // index = 2，传递token信息
    return;
}

void CcuContextAllGatherVMesh1D::PostSync()
{
    uint16_t selfBit = 1 << rankId_;
    uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_));
    for (auto t : transports) {
        RemotePost(*t, CKE_IDX_0, selfBit);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit);
}

void CcuContextAllGatherVMesh1D::DoGroupBroadcast()
{
    CcuRep::Memory src = CreateMemory();
    std::vector<CcuRep::Memory> dst;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        dst.push_back(CreateMemory());
    }
    src.addr = input_;
    src.token = token_[rankId_];

    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = rankSize_ - 1;
        }
        dst[curId].addr = output_[rankIdx];
        dst[curId].addr += mySliceOffset_;
        dst[curId].token = token_[rankIdx];
    }
    GroupBroadcast(transports, dst, src, groupOpSize_);
}

void CcuContextAllGatherVMesh1D::Algorithm()
{
    HCCL_INFO("[CcuContextAllGatherVMesh1D] Algorithm run");
    InitResources();
    LoadArgs();
    PreSync();
    DoGroupBroadcast();
    PostSync();
    HCCL_INFO("[CcuContextAllGatherVMesh1D] Algorithm end");
    return;
}

std::vector<uint64_t> CcuContextAllGatherVMesh1D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgAllGatherVMesh1D* taskArg = dynamic_cast<const CcuTaskArgAllGatherVMesh1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllGatherVMesh1D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t mySliceOutputOffset = taskArg->offset_;
    uint64_t mySliceSize = taskArg->sliceSize_;
    auto goSize = CalGoSize(mySliceSize);
    std::vector<uint64_t> args = {inputAddr, outputAddr, tokenInfo, mySliceOutputOffset};
    for (auto& item : goSize) {
        args.push_back(item);
    }
    return args;
}
} // namespace Hccl
