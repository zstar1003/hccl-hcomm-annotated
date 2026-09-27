/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_resource_manager.h"

#include <cstdint>
#include <cstdio>
#include <new>

#include "sim_log.h"
#include "sim_op_db_types.h"
#include "sim_common_defs.h"

using HcclSim::HcclVmResult;

namespace {
constexpr uint8_t INPUT_BUFFER_TYPE = 0;
constexpr uint8_t OUTPUT_BUFFER_TYPE = 1;
constexpr uint8_t CCL_BUFFER_TYPE = 2;
const char* GetBufferTypeName(uint8_t bufferType)
{
    switch (bufferType) {
        case INPUT_BUFFER_TYPE:
            return "input";
        case OUTPUT_BUFFER_TYPE:
            return "output";
        case CCL_BUFFER_TYPE:
            return "ccl";
        default:
            return "unknown";
    }
}

AivBufferResource* GetBufferSlot(AivRankResource& rankResource, uint8_t bufferType)
{
    switch (bufferType) {
        case INPUT_BUFFER_TYPE:
            return &rankResource.inputBuffer;
        case OUTPUT_BUFFER_TYPE:
            return &rankResource.outputBuffer;
        case CCL_BUFFER_TYPE:
            return &rankResource.cclBuffer;
        default:
            return nullptr;
    }
}
} // namespace

AivResourceManager& AivResourceManager::GetInstance()
{
    static AivResourceManager instance;
    return instance;
}

AivResourceManager::~AivResourceManager() { Reset(); }

HcclVmResult AivResourceManager::Init(uint32_t rankId, const sim::OpMemInfoTab& opMemInfo, uint32_t rankSize)
{
    auto ret = EnsureInitialized(rankSize);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        return ret;
    }
    if (rankId >= rankSize) {
        HCCL_VM_ERROR("rankId={} out of range, rankSize={}", rankId, rankSize);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    ret = MapOpMemBuffer(rankSize, rankId, INPUT_BUFFER_TYPE, opMemInfo.inputAddr, opMemInfo.inputSize);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        return ret;
    }
    ret = MapOpMemBuffer(rankSize, rankId, OUTPUT_BUFFER_TYPE, opMemInfo.outputAddr, opMemInfo.outputSize);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        return ret;
    }
    ret = MapOpMemBuffer(rankSize, rankId, CCL_BUFFER_TYPE, opMemInfo.cclAddr, opMemInfo.cclSize);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        return ret;
    }

    HCCL_VM_DEBUG(
        "init rank op mem success, rankId={}, rankSize={}, "
        "input={:x}/{} output={:x}/{} ccl={:x}/{}",
        rankId, rankSize, opMemInfo.inputAddr, opMemInfo.inputSize, opMemInfo.outputAddr, opMemInfo.outputSize,
        opMemInfo.cclAddr, opMemInfo.cclSize);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

void AivResourceManager::Reset()
{
    for (const auto& phyMem : acquiredPhyMemBlocks_) {
        sim::ReleaseInNoHostProcess(phyMem);
    }
    acquiredPhyMemBlocks_.clear();
    aivCommInfoBufferOwners_.clear();
    rankResources_.clear();
}

const std::vector<AivRankResource>& AivResourceManager::GetAllRankResources() const { return rankResources_; }

const AivRankResource* AivResourceManager::GetRankResource(uint32_t rankId) const
{
    if (rankId >= rankResources_.size()) {
        return nullptr;
    }
    return &rankResources_[rankId];
}

HcclVmResult AivResourceManager::EnsureInitialized(uint32_t rankSize)
{
    if (rankSize == 0) {
        HCCL_VM_ERROR("rankSize is 0");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    if (rankResources_.empty()) {
        rankResources_.resize(rankSize);
        auto ret = InitAivCommInfoBuffers(rankSize);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            Reset();
            return ret;
        }
        HCCL_VM_INFO("init base resource success, rankSize={}", rankSize);
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    if (rankResources_.size() != rankSize) {
        HCCL_VM_ERROR("rankSize mismatch, current={}, incoming={}", rankResources_.size(), rankSize);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AivResourceManager::MapBuffer(
    uint32_t rankSize, uint32_t rankId, uint8_t bufferType, uint64_t startAddr, uint64_t size, bool allowDuplicateSame)
{
    if (rankId >= rankSize) {
        HCCL_VM_ERROR("rankId={} out of range, rankSize={}", rankId, rankSize);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    auto& rankResource = rankResources_[rankId];
    AivBufferResource* bufferSlot = GetBufferSlot(rankResource, bufferType);
    if (bufferSlot == nullptr) {
        HCCL_VM_ERROR("unsupported buffer type={}, rankId={}", bufferType, rankId);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    if (bufferSlot->realAddr != nullptr) {
        if (allowDuplicateSame && bufferSlot->virtualAddr == startAddr && bufferSlot->size == size) {
            HCCL_VM_DEBUG(
                "duplicate same {} buffer skipped, "
                "rankId={}, virtualAddr={:x}, size={}",
                GetBufferTypeName(bufferType), rankId, startAddr, size);
            return HcclVmResult::HCCL_SIM_SUCCESS;
        }
        HCCL_VM_ERROR(
            "duplicate {} buffer found, rankId={}, virtualAddr={:x}, size={}", GetBufferTypeName(bufferType), rankId,
            startAddr, size);
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    sim::PhyMemBlock phyMem{};
    void* realAddr = sim::AcquireDevPtrInNoHostProcess(reinterpret_cast<void*>(startAddr), phyMem);
    if (realAddr == nullptr) {
        HCCL_VM_ERROR("translate addr failed, rankId={}, type={}, addr={:x}", rankId, bufferType, startAddr);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    AivBufferResource bufferResource;
    bufferResource.realAddr = realAddr;
    bufferResource.virtualAddr = startAddr;
    bufferResource.size = size;

    *bufferSlot = bufferResource;

    acquiredPhyMemBlocks_.push_back(phyMem);
    HCCL_VM_DEBUG(
        "rankId={}, type={}, virtualAddr={:x}, realAddr={}, size={}", rankId, bufferType, startAddr, realAddr, size);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AivResourceManager::MapOpMemBuffer(
    uint32_t rankSize, uint32_t rankId, uint8_t bufferType, uint64_t startAddr, uint64_t size)
{
    if (startAddr == 0 || size == 0) {
        if (startAddr != 0 || size != 0) {
            HCCL_VM_WARN(
                "incomplete {} buffer skipped, "
                "rankId={}, virtualAddr={:x}, size={}",
                GetBufferTypeName(bufferType), rankId, startAddr, size);
        }
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    return MapBuffer(rankSize, rankId, bufferType, startAddr, size, true);
}

HcclVmResult AivResourceManager::InitAivCommInfoBuffers(uint32_t rankSize)
{
    aivCommInfoBufferOwners_.resize(rankSize);
    for (uint32_t rankId = 0; rankId < rankSize; ++rankId) {
        auto aivCommInfoBuffer = std::make_unique<uint8_t[]>(AivCommInfoLayout::SIZE_BYTES);
        if (aivCommInfoBuffer == nullptr) {
            HCCL_VM_ERROR(
                "alloc AIV commInfo buffer failed, rankId={}, size={}", rankId, AivCommInfoLayout::SIZE_BYTES);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }

        rankResources_[rankId].aivCommInfoBuffer.realAddr = aivCommInfoBuffer.get();
        rankResources_[rankId].aivCommInfoBuffer.size = AivCommInfoLayout::SIZE_BYTES;
        aivCommInfoBufferOwners_[rankId] = std::move(aivCommInfoBuffer);

        HCCL_VM_DEBUG(
            "AIV commInfo rankId={}, realAddr={}, size={}", rankId, rankResources_[rankId].aivCommInfoBuffer.realAddr,
            rankResources_[rankId].aivCommInfoBuffer.size);
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}
