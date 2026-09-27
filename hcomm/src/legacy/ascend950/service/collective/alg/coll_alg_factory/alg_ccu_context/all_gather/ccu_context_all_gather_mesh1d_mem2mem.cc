/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_gather_mesh1d_mem2mem.h"
#include "ccu_instruction_all_gather_mesh1d_mem2mem.h"

namespace Hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;

CcuContextAllGatherMeshMem2Mem1D::CcuContextAllGatherMeshMem2Mem1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem1D] Enter Constructor.");
    const CcuCtxArgAllGatherMeshMem2Mem1D* ctxArg = dynamic_cast<const CcuCtxArgAllGatherMeshMem2Mem1D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllGatherMeshMem2Mem1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    if (ctxArg->dimSize_.size() > 0) {
        rankSize_ = ctxArg->dimSize_[0];
    }

    input_.push_back(CreateVariable());
    uint16_t transportIdx = 0;
    // 按照rank号从小到大遍历transports，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextAllGatherMeshMem2Mem1D] MyRank[%u], PeerId[%llu], TransportId[%u]", rankId_, peerId,
                transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr || transportIdx >= transports.size(),
                HCCL_ERROR("[CcuContextAllGatherMeshMem2Mem1D] Algorithm transport ptr is null or transportIdx is out "
                           "of bounds"), );
            output_.push_back(
                CreateVariable((*transports[transportIdx]), OUTPUT_XN_ID)); // 获取transport中id=1的Var来传递output
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }
    offset_ = CreateVariable();
    sliceSize_ = CreateVariable();
    localGoSize_ = CreateGroupOpSize();
}

void CcuContextAllGatherMeshMem2Mem1D::Algorithm()
{
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem1D] AllgatherMesh1D run.");
    uint16_t selfBit = 1 << rankId_;
    uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_));

    Load(input_[0]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(offset_);
    Load(sliceSize_);
    Load(localGoSize_);

    for (auto t : transports) {
        WriteVariableWithSignal(*t, output_[rankId_], OUTPUT_XN_ID, CKE_IDX_1, selfBit); // index = 1，传递output信息
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_2, selfBit); // index = 2，传递token信息
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit); // index = 1，传递output信息
    GroupWait(*transportGroup, CKE_IDX_2, allBit); // index = 2，传递token信息

    CcuRep::Memory src = CreateMemory();
    std::vector<CcuRep::Memory> dst;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        dst.push_back(CreateMemory());
    }

    u32 transportId = 0;
    CcuRep::MaskSignal locMask = CreateMaskSignal();
    src.addr = input_[0];
    src.token = token_[rankId_];
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        dst[rankIdx].addr = output_[rankIdx];
        dst[rankIdx].addr += offset_;
        dst[rankIdx].token = token_[rankIdx];
        if (rankIdx == rankId_) {
            LocalPost(locMask, 1 << rankIdx);
        } else {
            Write(*transports[transportId], dst[rankIdx], src, sliceSize_, locMask, 1 << rankIdx);
            transportId++;
        }
    }
    GroupCopy(dst[rankId_], src, localGoSize_);
    LocalWait(locMask, (1 << rankSize_) - 1);

    for (auto t : transports) {
        RemotePost(*t, CKE_IDX_0, selfBit);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit);
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem1D] AllgatherMesh1D end.");
    return;
}

std::vector<uint64_t> CcuContextAllGatherMeshMem2Mem1D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgAllGatherMeshMem2Mem1D* taskArg = dynamic_cast<const CcuTaskArgAllGatherMeshMem2Mem1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllGatherMeshMem2Mem1D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t offset = taskArg->offset_;
    uint64_t sliceSize = taskArg->sliceSize_;
    auto localGoSize = CalGoSize(sliceSize);

    return {inputAddr,      outputAddr,     tokenInfo,      offset,        sliceSize,
            localGoSize[0], localGoSize[1], localGoSize[2], localGoSize[3]};
}
} // namespace Hccl
