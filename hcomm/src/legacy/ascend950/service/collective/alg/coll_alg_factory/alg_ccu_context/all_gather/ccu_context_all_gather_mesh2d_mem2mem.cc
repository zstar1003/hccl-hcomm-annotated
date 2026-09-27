/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_all_gather_mesh2d_mem2mem.h"
#include "ccu_instruction_all_gather_mesh2d_mem2mem.h"

namespace Hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0; // 约定：尾同步均使用第0个CKE
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;
constexpr int CKE_IDX_4 = 4;
constexpr int FST_AXIS_ID = 0;
constexpr int SEC_AXIS_ID = 1;

CcuContextAllGatherMeshMem2Mem2D::CcuContextAllGatherMeshMem2Mem2D(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
    : CcuContextAlgBase(arg, transports, group)
{
    xAxisSize_ = CreateVariable();
    yAxisSize_ = CreateVariable();
    offset_ = CreateVariable();
    sliceSize_ = CreateVariable();
    firstOutOffset_ = CreateVariable();
    firstInOffset_ = CreateVariable();
    secondInOutBaseOffset_ = CreateVariable();
    secondInOutStepOffset_ = CreateVariable();
    goASize_ = CreateGroupOpSize();
    goBSize_ = CreateGroupOpSize();
    localAxisSignal_ = CreateMaskSignal();

    const CcuCtxArgAllGatherMeshMem2Mem2D* ctxArg = dynamic_cast<const CcuCtxArgAllGatherMeshMem2Mem2D*>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllGatherMeshMem2Mem2D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    dimSize_ = ctxArg->dimSize_;
    axisId_ = ctxArg->axisId_;
    // DIM_SIZE由头文件传入，=2
    if (dimSize_.size() != DIM_SIZE or axisId_ > 1) {
        THROW<NullPtrException>(StringFormat(
            "CcuContextAllGatherMeshMem2Mem2D::dimSize[%u] or axisId[%u] is invalid", dimSize_.size(), axisId_));
    }
    CHK_PRT_THROW(
        dimSize_[0] == 0 || dimSize_[1] == 0,
        HCCL_ERROR(
            "[CcuContextAllGatherMeshMem2Mem2D] dimSize0[%llu] or dimSize1[%llu] is zero", dimSize_[0], dimSize_[1]),
        InvalidParamsException, "dimSize[0] or dimSize[1] is invalid");
    dimId_.emplace_back(rankId_ % dimSize_[0]);
    dimId_.emplace_back(rankId_ / dimSize_[0]);
    localId_ = dimId_[axisId_];
    localSize_ = dimSize_[axisId_];
    localAxisSignalName_ = "CcuContextAllGatherMeshMem2Mem2DAxisSync_" + std::to_string(axisId_);
    anotherAxisSignalName_ = "CcuContextAllGatherMeshMem2Mem2DAxisSync_" + std::to_string(1 - axisId_);
    HCCL_INFO(
        "[CcuContextAllGatherMeshMem2Mem2D] RankId[%u], DimSize: D0[%u]--D1[%u], localId[%u], localSize[%u]", rankId_,
        dimSize_[0], dimSize_[1], localId_, localSize_);
}

void CcuContextAllGatherMeshMem2Mem2D::InitResources()
{
    input_.push_back(CreateVariable());

    uint32_t transportIdx = 0;
    for (uint32_t peerId = 0; peerId < localSize_; peerId++) {
        if (peerId == localId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO(
                "[CcuContextAllGatherMeshMem2Mem2D] MyRank[%u], peerId[%u], transportIdx[%u]", rankId_, peerId,
                transportIdx);
            CHK_PRT_RET(
                transports[transportIdx] == nullptr,
                HCCL_ERROR("[CcuContextAllGatherMeshMem2Mem2D] Algorithm transport ptr is null"), );
            output_.push_back(CreateVariable((*transports[transportIdx]), OUTPUT_XN_ID));
            token_.push_back(CreateVariable((*transports[transportIdx]), TOKEN_XN_ID));
            transportIdx++;
        }
    }

    ExportMaskSignal(localAxisSignal_, localAxisSignalName_);
    anotherAxisSignal_ = ImportMaskSignal(anotherAxisSignalName_);
    return;
}

// 需要与GeneArgs返回时的顺序相同
void CcuContextAllGatherMeshMem2Mem2D::LoadArgs()
{
    Load(input_[0]);
    Load(output_[localId_]);
    Load(token_[localId_]);
    Load(xAxisSize_);
    Load(yAxisSize_);
    Load(offset_);
    Load(sliceSize_);
    Load(firstInOffset_);
    Load(firstOutOffset_);
    Load(secondInOutBaseOffset_);
    Load(secondInOutStepOffset_);
    Load(goASize_);
    Load(goBSize_);

    return;
}

void CcuContextAllGatherMeshMem2Mem2D::ExchangeInfoAndSync()
{
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] ExchangeInfoAndSync run begins.");
    uint16_t selfBit = 1 << localId_;
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        if (t == nullptr) {
            THROW<NullPtrException>(StringFormat("CcuContextAllGatherMeshMem2Mem2D::Algorithm transport ptr is null"));
        }
        WriteVariableWithSignal(*t, output_[localId_], OUTPUT_XN_ID, CKE_IDX_1, selfBit); // index = 1，传递output信息
        WriteVariableWithSignal(*t, token_[localId_], TOKEN_XN_ID, CKE_IDX_2, selfBit); // index = 2，传递token信息
        HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] change addr success");
    }
    GroupWait(*transportGroup, CKE_IDX_1, allBit);
    GroupWait(*transportGroup, CKE_IDX_2, allBit);
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] ExchangeInfoAndSync run finished.");
    return;
}

void CcuContextAllGatherMeshMem2Mem2D::RankSync(uint32_t signalIndex)
{
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] RankSync run begins.");
    uint16_t selfBit = 1 << localId_;
    uint16_t allBit = ((1 << localSize_) - 1) & (~(1 << localId_));

    for (auto t : transports) {
        if (t == nullptr) {
            THROW<NullPtrException>(StringFormat("CcuContextAllGatherMeshMem2Mem2D::Algorithm transport ptr is null"));
        }
        RemotePost(*t, signalIndex, selfBit);
    }
    GroupWait(*transportGroup, signalIndex, allBit);
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] RankSync run ends.");
    return;
}

void CcuContextAllGatherMeshMem2Mem2D::AxisSync(uint32_t signalIndex)
{
    constexpr uint32_t DIE_NUM = 2;
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] AxisSync run begins.");
    LocalCtxPost(anotherAxisSignal_, 1 << (axisId_ + (signalIndex * DIE_NUM)));
    LocalWait(localAxisSignal_, 1 << ((1 - axisId_) + (signalIndex * DIE_NUM)));
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] AxisSync run ends.");
    return;
}

void CcuContextAllGatherMeshMem2Mem2D::FirstStep()
{
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] firstStep run begins.");
    CcuRep::Memory src = CreateMemory();
    src.addr = input_[0];
    src.addr += firstInOffset_;
    src.token = token_[localId_];

    std::vector<CcuRep::Memory> dst;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        dst.push_back(CreateMemory());
    }
    CcuRep::Variable tempSize = CreateVariable();
    tempSize = (axisId_ == 0) ? xAxisSize_ : yAxisSize_;
    GroupOpSize tempGoSize = CreateGroupOpSize();
    tempGoSize = (axisId_ == 0) ? goASize_ : goBSize_;
    CCU_IF(tempSize != 0)
    {
        u32 transportId = 0;
        CcuRep::MaskSignal locMask = CreateMaskSignal();
        for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
            dst[rankIdx].addr = output_[rankIdx];
            dst[rankIdx].addr += firstOutOffset_;
            dst[rankIdx].token = token_[rankIdx];
            if (rankIdx == localId_) {
                LocalPost(locMask, (1 << rankIdx));
            } else {
                Write(*transports[transportId], dst[rankIdx], src, tempSize, locMask, 1 << rankIdx);
                transportId++;
            }
        }
        GroupCopy(dst[localId_], src, tempGoSize);
        LocalWait(locMask, (1 << localSize_) - 1);
    }

    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] firstStep run ends.");
}

void CcuContextAllGatherMeshMem2Mem2D::SecondStep()
{
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] secondStep run begins.");
    uint64_t anotherSize = dimSize_[1 - axisId_];

    CcuRep::Memory src = CreateMemory();
    src.addr = output_[localId_];
    src.token = token_[localId_];

    std::vector<CcuRep::Memory> dst;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        dst.push_back(CreateMemory());
    }

    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        dst[rankIdx].addr = output_[rankIdx];
        dst[rankIdx].token = token_[rankIdx];
    }

    for (uint64_t localIdx = 0; localIdx < anotherSize; localIdx++) { // 分片数等于另外一个轴的大小
        if (localIdx == 0) {
            src.addr += secondInOutBaseOffset_;
        } else {
            src.addr += secondInOutStepOffset_;
        }
        for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
            if (localIdx == 0) {
                dst[rankIdx].addr += secondInOutBaseOffset_;
            } else {
                dst[rankIdx].addr += secondInOutStepOffset_;
            }
        }
        auto tempSize = (axisId_ == 0) ? yAxisSize_ : xAxisSize_;
        CCU_IF(tempSize != 0)
        {
            u32 transportId = 0;
            CcuRep::MaskSignal locMask = CreateMaskSignal();
            for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
                if (rankIdx == localId_) {
                    LocalPost(locMask, (1 << rankIdx));
                } else {
                    Write(*transports[transportId], dst[rankIdx], src, tempSize, locMask, 1 << rankIdx);
                    transportId++;
                }
            }
            LocalWait(locMask, (1 << localSize_) - 1);
        }
    }
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] secondStep run ends.");
}

void CcuContextAllGatherMeshMem2Mem2D::Algorithm()
{
    // 初始化寄存器资源 & 加载外部输入参数
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] AllgatherMesh2D Algorithm Init Begins.");
    InitResources();
    LoadArgs();

    // 第一轮
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] Algorithm first step begins.");
    ExchangeInfoAndSync();
    FirstStep();
    RankSync(CKE_IDX_3);
    AxisSync(FST_AXIS_ID);

    // 第二轮
    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] Algorithm second step begins.");
    RankSync(CKE_IDX_4);
    SecondStep();
    RankSync(CKE_IDX_0);
    AxisSync(SEC_AXIS_ID);

    HCCL_INFO("[CcuContextAllGatherMeshMem2Mem2D] Algorithm Ends.");
    return;
}

std::vector<uint64_t> CcuContextAllGatherMeshMem2Mem2D::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgAllGatherMeshMem2Mem2D* taskArg = dynamic_cast<const CcuTaskArgAllGatherMeshMem2Mem2D*>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuContextAllGatherMeshMem2Mem2D::taskArg ptr is null"));
    }

    // input&output&buffer地址
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t xAxisSize = taskArg->xAxisSize_;
    uint64_t yAxisSize = taskArg->yAxisSize_;
    uint64_t offset = taskArg->offset_;
    uint64_t sliceSize = xAxisSize + yAxisSize;
    uint64_t tokenValue = taskArg->token_;

    auto goSizeAxis = CalGoSize(xAxisSize);
    auto goSizeBxis = CalGoSize(yAxisSize);

    uint64_t firstInOffset;
    uint64_t firstOutOffset = 0;
    uint64_t secondInOutBaseOffset = 0;
    uint64_t secondInOutStepOffset = 0;

    for (uint32_t i = 0; i < rankId_; i++) {
        firstOutOffset += offset;
    }

    if (axisId_ == 0) {
        firstInOffset = 0;
        for (uint32_t i = 0; i < dimId_[0]; i++) {
            secondInOutBaseOffset += offset;
        }
        secondInOutBaseOffset += xAxisSize;
        for (uint64_t i = 0; i < dimSize_[0]; i++) {
            secondInOutStepOffset += offset;
        }
    } else {
        firstInOffset = xAxisSize;
        firstOutOffset += xAxisSize;
        for (uint32_t i = 0; i < dimId_[1]; i++) {
            for (uint64_t j = 0; j < dimSize_[0]; j++) {
                secondInOutBaseOffset += offset;
            }
        }
        secondInOutStepOffset = offset;
    }

    HCCL_INFO(
        "[CcuContextAllGatherMeshMem2Mem2D][GeneArgs] RankId[%u]--AxisId[%u], inputAddr[%llu], outputAddr[%llu], "
        "aSize[%llu], bSize[%llu], offset[%llu], sliceSize[%llu], firstInOffset[%llu], firstOutOffset[%llu], "
        "secondInOutBaseOffset[%llu], secondInOutStepOffset[%llu]",
        rankId_, axisId_, inputAddr, outputAddr, xAxisSize, yAxisSize, offset, sliceSize, firstInOffset, firstOutOffset,
        secondInOutBaseOffset, secondInOutStepOffset);

    return {
        inputAddr,
        outputAddr,
        tokenValue,
        xAxisSize,
        yAxisSize,
        offset,
        sliceSize,
        firstInOffset,
        firstOutOffset,
        secondInOutBaseOffset,
        secondInOutStepOffset,
        goSizeAxis[0],
        goSizeAxis[1],
        goSizeAxis[2],
        goSizeAxis[3],
        goSizeBxis[0],
        goSizeBxis[1],
        goSizeBxis[2],
        goSizeBxis[3]};
}

} // namespace Hccl
