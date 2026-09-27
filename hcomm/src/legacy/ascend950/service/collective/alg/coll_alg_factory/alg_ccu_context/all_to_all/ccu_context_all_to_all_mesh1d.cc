/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_to_all_mesh1d.h"
#include "ccu_instruction_all_to_all_mesh1d.h"

namespace Hccl {

constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;

CcuContextAllToAllMesh1D::CcuContextAllToAllMesh1D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    const CcuCtxArgAllToAllMesh1D* ctxArg = dynamic_cast<const CcuCtxArgAllToAllMesh1D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllToAllMesh1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId;
    if (ctxArg->dimSize.size() > 0) {
        rankSize_ = ctxArg->dimSize[0];
    }
    if (transports.size() == 0) {
        THROW<NullPtrException>(StringFormat("CcuContextAllToAllMesh1D transports is empty"));
    }
    loadFromMem_ = ctxArg->loadFromMem;
}

void CcuContextAllToAllMesh1D::Algorithm()
{
    HCCL_INFO("[ccuAllToAllMesh1D_context] AllToAllMesh1D run.");
    // 创建Variable，用于交换地址及token
    u32 transportId = 0;
    for (u64 id = 0; id < rankSize_; id++) {
        if (id == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else { // 非本地，使用远端Variable
            CHK_PRT_RET(
                transports[transportId] == nullptr,
                HCCL_ERROR("[CcuContextAllToAllMesh1D] Algorithm transport ptr is null"), );
            input_.push_back(CreateVariable((*transports[transportId]), CKE_IDX_0));
            output_.push_back(CreateVariable((*transports[transportId]), CKE_IDX_1));
            token_.push_back(CreateVariable((*transports[transportId]), CKE_IDX_2));
            transportId++;
        }
    }
    sliceSize_ = CreateVariable();
    srcStride_ = CreateVariable();
    srcOffset_ = CreateVariable();
    dstOffset_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();

    // 从SQE load args，本rank需要的input、output地址等信息
    // inputAddr, outputAddr, tokenInfo, srcStride, srcOffset, dstOffset, groupOpSize
    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(sliceSize_); // 本轮传输的分片大小
    Load(srcStride_); // 单片数据大小
    Load(srcOffset_);
    Load(dstOffset_);
    Load(groupOpSize_);

    // 前同步。交换信息，将本Rank load的in\out等地址信息写到所有对端的对应Variable中，并同步
    uint16_t selfBit = 1 << rankId_; // 本rank的mask
    uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_));

    srcOffset_ += input_[rankId_];

    for (auto t : transports) {
        // （transport, param, paramID, SemID, mask）
        WriteVariableWithSignal(*t, output_[rankId_], CKE_IDX_1, CKE_IDX_1, selfBit); // index = 1，传递output信息
        WriteVariableWithSignal(*t, token_[rankId_], CKE_IDX_2, CKE_IDX_2, selfBit);  // index = 2，传递token信息
    }

    GroupWait(*transportGroup, CKE_IDX_1, allBit); // index = 1，传递output信息
    GroupWait(*transportGroup, CKE_IDX_2, allBit); // index = 2，传递token信息

    // 创建GSA， src为本地的各片HBM地址GSA列表，dst为所有对端的HBM地址GSA列表
    std::vector<CcuRep::Memory> src;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        src.push_back(CreateMemory());
    }
    std::vector<CcuRep::Memory> dst;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        dst.push_back(CreateMemory());
    }

    // 考虑stride信息
    for (uint64_t r = 0; r < rankSize_; r++) {
        src[r].token = token_[r];
        dst[r].token = token_[r];

        // src[r] = srcOffset + r*srcStride
        src[r].addr = srcOffset_;
        for (uint64_t i = 0; i < r; i++) {
            src[r].addr += srcStride_;
        }
        // dst[r] = recvBuf[r] + dstOffset
        dst[r].addr = output_[r];
        dst[r].addr += dstOffset_;
    }

    // 创建CKE，源端保序
    CcuRep::MaskSignal locMask = CreateMaskSignal();
    //  all2all 数据搬运
    transportId = 0;
    for (uint64_t r = 0; r < rankSize_; r++) {
        if (r != rankId_) {
            Write(*transports[transportId], dst[r], src[r], sliceSize_, locMask, 1 << r);
            transportId++;
        }
    }
    GroupCopy(dst[rankId_], src[rankId_], groupOpSize_);
    LocalWait(locMask, allBit);

    //  后同步
    for (auto t : transports) {
        if (t == nullptr) {
            THROW<NullPtrException>(StringFormat("CcuContextAllToAllMesh1D::Algorithm transport ptr is null"));
        }
        RemotePost(*t, CKE_IDX_0, selfBit);
    }
    GroupWait(*transportGroup, CKE_IDX_0, allBit);
    HCCL_INFO("[AllToAllAlgo] AllToAllMesh1D end");

    return;
}

std::vector<uint64_t> CcuContextAllToAllMesh1D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgAllToAllMesh1D* taskArg = dynamic_cast<const CcuTaskArgAllToAllMesh1D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllToAllMesh1D::taskArg ptr is null"));
    }
    uint64_t inputAddr = taskArg->inputAddr;
    uint64_t outputAddr = taskArg->outputAddr;
    uint64_t tokenInfo = taskArg->token;

    uint64_t srcStride = taskArg->srcStride;
    uint64_t srcOffset = taskArg->srcOffset;
    uint64_t dstOffset = taskArg->dstOffset;

    uint64_t sliceSize = taskArg->sliceSize;
    auto goSize = CalGoSize(sliceSize);
    HCCL_INFO(
        "[AllToAllAlgo] inputAddr[%llu], outputAddr[%llu], sliceSize[%llu], srcStride[%llu], srcOffset[%llu], "
        "dstOffset[%llu].",
        inputAddr, outputAddr, sliceSize, srcStride, srcOffset, dstOffset);

    return {inputAddr, outputAddr, tokenInfo, sliceSize, srcStride, srcOffset,
            dstOffset, goSize[0],  goSize[1], goSize[2], goSize[3]};
}

} // namespace Hccl
