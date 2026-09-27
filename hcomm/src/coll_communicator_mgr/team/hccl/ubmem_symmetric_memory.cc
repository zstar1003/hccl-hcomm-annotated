/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ubmem_symmetric_memory.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <list>
#include <limits>
#include <new>
#include <utility>

#include "adapter_rts_common.h"
#include "coll_comm.h"
#include "hccl_common.h"
#include "hccl_team_mgr.h"
#include "hcomm_team.h"
#include "log.h"

namespace hccl {

// 与A3 SimpleVaAllocator保持一致，管理每个LSA成员stride内的Window相对偏移。
class UbMemSymmetricMemory::SimpleVaAllocator {
public:
    HcclResult Init(size_t totalSize)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        CHK_PRT_RET(totalSize == 0, HCCL_ERROR("[%s] total size is zero", __func__), HCCL_E_PARA);
        freeList_.clear();
        HcclResult ret = HCCL_SUCCESS;
        EXCEPTION_CATCH(freeList_.push_back({0, totalSize}), ret = HCCL_E_MEMORY);
        if (ret == HCCL_SUCCESS) {
            totalSize_ = totalSize;
        }
        return ret;
    }

    void Destroy()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        freeList_.clear();
        totalSize_ = 0;
    }

    HcclResult Reserve(size_t size, size_t align, size_t& offset)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        CHK_PRT_RET(
            size == 0 || align == 0 || (align & (align - 1U)) != 0,
            HCCL_ERROR("[%s] invalid size[%zu] or alignment[%zu]", __func__, size, align), HCCL_E_PARA);
        for (auto iter = freeList_.begin(); iter != freeList_.end(); ++iter) {
            size_t alignedOffset = 0;
            if (!AlignOffset(iter->offset, align, alignedOffset)) {
                continue;
            }
            size_t blockEnd = iter->offset + iter->size;
            if (alignedOffset > blockEnd || size > blockEnd - alignedOffset) {
                continue;
            }
            CHK_RET(ReserveFromBlock(iter, alignedOffset, size));
            offset = alignedOffset;
            return HCCL_SUCCESS;
        }
        HCCL_ERROR("[%s] no free VA block for size[%zu], alignment[%zu]", __func__, size, align);
        return HCCL_E_MEMORY;
    }

    HcclResult Release(size_t offset, size_t size)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        CHK_PRT_RET(
            size == 0 || offset > totalSize_ || size > totalSize_ - offset,
            HCCL_ERROR(
                "[%s] release range[offset:%zu,size:%zu] exceeds total[%zu]", __func__, offset, size, totalSize_),
            HCCL_E_PARA);
        auto next = freeList_.begin();
        while (next != freeList_.end() && next->offset < offset) {
            ++next;
        }
        CHK_RET(CheckReleaseRange(next, offset, size));
        std::list<FreeBlock>::iterator released;
        HcclResult ret = HCCL_SUCCESS;
        EXCEPTION_CATCH(released = freeList_.insert(next, {offset, size}), ret = HCCL_E_MEMORY);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        MergeAdjacentBlocks(released);
        return HCCL_SUCCESS;
    }

private:
    struct FreeBlock {
        size_t offset;
        size_t size;
    };

    using FreeBlockIterator = std::list<FreeBlock>::iterator;

    static bool AlignOffset(size_t offset, size_t align, size_t& alignedOffset)
    {
        if (offset > std::numeric_limits<size_t>::max() - (align - 1U)) {
            return false;
        }
        alignedOffset = (offset + align - 1U) & ~(align - 1U);
        return true;
    }

    HcclResult ReserveFromBlock(FreeBlockIterator iter, size_t offset, size_t size)
    {
        size_t blockEnd = iter->offset + iter->size;
        size_t frontSize = offset - iter->offset;
        size_t backSize = blockEnd - offset - size;
        if (frontSize > 0 && backSize > 0) {
            HcclResult ret = HCCL_SUCCESS;
            EXCEPTION_CATCH(freeList_.insert(std::next(iter), {offset + size, backSize}), ret = HCCL_E_MEMORY);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            iter->size = frontSize;
        } else if (frontSize > 0) {
            iter->size = frontSize;
        } else if (backSize > 0) {
            iter->offset = offset + size;
            iter->size = backSize;
        } else {
            freeList_.erase(iter);
        }
        return HCCL_SUCCESS;
    }

    HcclResult CheckReleaseRange(FreeBlockIterator next, size_t offset, size_t size) const
    {
        if (next != freeList_.begin()) {
            auto previous = std::prev(next);
            CHK_PRT_RET(
                previous->offset + previous->size > offset,
                HCCL_ERROR("[%s] release range overlaps previous free block", __func__), HCCL_E_PARA);
        }
        CHK_PRT_RET(
            next != freeList_.end() && offset + size > next->offset,
            HCCL_ERROR("[%s] release range overlaps next free block", __func__), HCCL_E_PARA);
        return HCCL_SUCCESS;
    }

    void MergeAdjacentBlocks(FreeBlockIterator released)
    {
        auto next = std::next(released);
        if (next != freeList_.end() && released->offset + released->size == next->offset) {
            released->size += next->size;
            freeList_.erase(next);
        }
        if (released != freeList_.begin()) {
            auto previous = std::prev(released);
            if (previous->offset + previous->size == released->offset) {
                previous->size += released->size;
                freeList_.erase(released);
            }
        }
    }

    std::list<FreeBlock> freeList_;
    std::mutex mutex_;
    size_t totalSize_{0};
};

namespace {
    constexpr uintptr_t UB_SYMMETRIC_VA_HINT = 40ULL * 1024ULL * 1024ULL * 1024ULL * 1024ULL;
} // namespace

static_assert(
    sizeof(UbmemShareableInfo) <= UBMEM_PACKET_DATA_MAX_LEN,
    "UB Memory shareable information exceeds the Agent packet payload");

UbMemSymmetricMemory::UbMemSymmetricMemory(
    CollComm* collComm, HcommTeamHandle lsaTeam, uint32_t netLayer, const std::vector<uint32_t>& worldRankIds)
    : collComm_(collComm),
      lsaTeamSize_(static_cast<uint32_t>(worldRankIds.size())),
      netLayer_(netLayer),
      worldRankIds_(worldRankIds),
      lsaTeam_(lsaTeam)
{
    vaAllocator_.reset(new (std::nothrow) SimpleVaAllocator());
}

UbMemSymmetricMemory::~UbMemSymmetricMemory()
{
    Finalize();
    // Finalize清理失败时记录会保留；对象销毁前仍需移除Owner索引，避免留下悬空通信域指针。
    for (const auto& entry : windowsByHandle_) {
        EraseHcommWindowOwner(entry.first);
    }
}

HcclResult UbMemSymmetricMemory::Init()
{
    CHK_PTR_NULL(collComm_);
    rankGraph_ = collComm_->GetRankGraph();
    selfRank_ = collComm_->GetMyRankId();
    commId_ = collComm_->GetCommId();
    CHK_PTR_NULL(rankGraph_);
    auto selfIter = std::find(worldRankIds_.begin(), worldRankIds_.end(), selfRank_);
    CHK_PRT_RET(
        selfIter == worldRankIds_.end(),
        HCCL_ERROR("[%s] self rank[%u] is not in UB Memory LSA team", __func__, selfRank_), HCCL_E_PARA);
    selfMember_ = static_cast<uint32_t>(selfIter - worldRankIds_.begin());

    EXCEPTION_CATCH(
        agent_ = std::make_unique<UbMemSymmetricMemoryAgent>(rankGraph_, selfRank_, worldRankIds_, netLayer_, commId_),
        return HCCL_E_MEMORY);
    CHK_SMART_PTR_NULL(agent_);
    return CheckPrebuiltLsaTeam();
}

HcclResult UbMemSymmetricMemory::CheckPrebuiltLsaTeam()
{
    CHK_PRT_RET(
        lsaTeam_ == nullptr, HCCL_ERROR("[%s] prebuilt UB Memory LSA team not found, layer[%u]", __func__, netLayer_),
        HCCL_E_NOT_FOUND);
    std::vector<uint32_t> registeredRanks;
    EXCEPTION_CATCH(
        registeredRanks
        = HcclTeamMgr::GetInstance().GetPrebuiltWorldTeamRanks(collComm_, COMM_PROTOCOL_UB_MEM, netLayer_),
        return HCCL_E_MEMORY);
    CHK_PRT_RET(
        registeredRanks != worldRankIds_, HCCL_ERROR("[%s] UB Memory LSA team membership changed", __func__),
        HCCL_E_INTERNAL);
    HCCL_RUN_INFO(
        "[%s] UB Memory LSA team ready, comm[%s], team[%p], selfRank[%u], lsaTeamSize[%u], layer[%u]", __func__,
        commId_.c_str(), lsaTeam_, selfRank_, lsaTeamSize_, netLayer_);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::EnsureRuntime()
{
    CHK_PTR_NULL(collComm_);
    CHK_SMART_PTR_NULL(agent_);
    std::call_once(runtimeInitFlag_, [this]() {
        // 与A3一致：先准备对称VA，再建立用于成员信息交换的LSA Ring。
        runtimeInitResult_ = InitSymmetricVa();
        if (runtimeInitResult_ != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] initialize UB Memory symmetric VA failed, ret[%d]", __func__, runtimeInitResult_);
            FinalizeSymmetricVa();
            return;
        }
        runtimeInitResult_ = agent_->Init();
        if (runtimeInitResult_ != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] initialize UB Memory LSA ring failed, ret[%d]", __func__, runtimeInitResult_);
            FinalizeSymmetricVa();
            return;
        }
        runtimeInitResult_ = GetAllMemberPids();
        if (runtimeInitResult_ != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] exchange UB Memory LSA member pids failed, ret[%d]", __func__, runtimeInitResult_);
            broken_ = true; // pid 交换已进入集合阶段，失败后重试会导致成员间失步。
            agent_->Finalize();
            FinalizeSymmetricVa();
        }
    });
    return runtimeInitResult_;
}

HcclResult UbMemSymmetricMemory::GetAllMemberPids()
{
    int32_t localPid = 0;
    aclError ret = aclrtDeviceGetBareTgid(&localPid);
    CHK_PRT_RET(
        ret != ACL_SUCCESS || localPid <= 0,
        HCCL_ERROR("[%s] get local bare tgid failed, ret[%d], pid[%d]", __func__, ret, localPid), HCCL_E_RUNTIME);
    EXCEPTION_CATCH(memberPids_.resize(lsaTeamSize_), return HCCL_E_MEMORY);
    CHK_RET(agent_->ExchangeInfo(&localPid, memberPids_.data(), sizeof(localPid)));
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::InitSymmetricVa()
{
    if (arenaBase_ != nullptr) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(collComm_);
    uint64_t strideGb = collComm_->GetCommConfig().GetConfigSymmetricMemoryStride();
    CHK_PRT_RET(
        lsaTeamSize_ == 0 || selfMember_ >= lsaTeamSize_ || strideGb == 0,
        HCCL_ERROR("[%s] invalid symmetric VA configuration", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        strideGb > std::numeric_limits<uint64_t>::max() / BYTES_PER_GB,
        HCCL_ERROR("[%s] stride[%llu]GB overflows", __func__, static_cast<unsigned long long>(strideGb)), HCCL_E_PARA);
    stride_ = strideGb * BYTES_PER_GB;

    size_t freeHbmSize = 0;
    size_t totalHbmSize = 0;
    aclError ret = aclrtGetMemInfo(ACL_HBM_MEM_HUGE, &freeHbmSize, &totalHbmSize);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] get HBM memory information failed, ret[%d]", __func__, ret),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        stride_ > totalHbmSize,
        HCCL_ERROR(
            "[%s] stride[%llu] exceeds total HBM size[%zu]", __func__, static_cast<unsigned long long>(stride_),
            totalHbmSize),
        HCCL_E_PARA);
    CHK_RET(InitGranularity());
    CHK_RET(ReserveArena());
    HcclResult allocatorRet = InitWindowOffsetAllocator();
    if (allocatorRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] initialize Window offset allocator failed, ret[%d]", __func__, allocatorRet);
        FinalizeSymmetricVa();
        return allocatorRet;
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::InitGranularity()
{
    int32_t deviceId = 0;
    CHK_PRT_RET(
        aclrtGetDevice(&deviceId) != ACL_SUCCESS, HCCL_ERROR("[%s] get device failed", __func__), HCCL_E_RUNTIME);
    aclrtPhysicalMemProp property{};
    property.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    property.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    property.memAttr = ACL_HBM_MEM_HUGE;
    property.location.id = deviceId;
    property.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    aclError ret = aclrtMemGetAllocationGranularity(&property, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity_);
    CHK_PRT_RET(
        ret != ACL_SUCCESS || granularity_ == 0,
        HCCL_ERROR("[%s] get allocation granularity failed, ret[%d], granularity[%zu]", __func__, ret, granularity_),
        HCCL_E_RUNTIME);
    CHK_PRT_RET(
        stride_ % granularity_ != 0,
        HCCL_ERROR(
            "[%s] stride[%llu] is not aligned to[%zu]", __func__, static_cast<unsigned long long>(stride_),
            granularity_),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::ReserveArena()
{
    CHK_PRT_RET(lsaTeamSize_ == 0, HCCL_ERROR("[%s] LSA team size is zero", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        stride_ > std::numeric_limits<size_t>::max() / lsaTeamSize_,
        HCCL_ERROR("[%s] total arena size overflows", __func__), HCCL_E_PARA);
    size_t arenaSize = static_cast<size_t>(stride_ * lsaTeamSize_);
    void* hint = reinterpret_cast<void*>(UB_SYMMETRIC_VA_HINT);
    aclError ret = aclrtReserveMemAddressNoUCMemory(&arenaBase_, arenaSize, 0, hint, 0);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] reserve UB Memory symmetric VA failed, size[%zu], ret[%d]", __func__, arenaSize, ret);
        arenaBase_ = nullptr;
        return HCCL_E_RUNTIME;
    }
    uintptr_t base = reinterpret_cast<uintptr_t>(arenaBase_);
    if (arenaSize > std::numeric_limits<uintptr_t>::max() - base) {
        HCCL_ERROR("[%s] reserved UB Memory symmetric VA range overflows", __func__);
        (void)aclrtReleaseMemAddress(arenaBase_);
        arenaBase_ = nullptr;
        return HCCL_E_RUNTIME;
    }
    HCCL_RUN_INFO(
        "[%s] UB Memory symmetric VA ready, base[%p], stride[%llu], lsaTeamSize[%u]", __func__, arenaBase_,
        static_cast<unsigned long long>(stride_), lsaTeamSize_);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::InitWindowOffsetAllocator()
{
    CHK_SMART_PTR_NULL(vaAllocator_);
    size_t allocatorSize = static_cast<size_t>(stride_);
    CHK_PRT_RET(
        static_cast<uint64_t>(allocatorSize) != stride_,
        HCCL_ERROR("[%s] stride[%llu] exceeds size_t range", __func__, static_cast<unsigned long long>(stride_)),
        HCCL_E_PARA);
    return vaAllocator_->Init(allocatorSize);
}

void UbMemSymmetricMemory::FinalizeSymmetricVa()
{
    if (activeMappingCount_ != 0) {
        HCCL_ERROR("[%s] refuse to release symmetric VA with[%zu] active mappings", __func__, activeMappingCount_);
        return;
    }
    if (arenaBase_ != nullptr) {
        aclError ret = aclrtReleaseMemAddress(arenaBase_);
        if (ret != ACL_SUCCESS) {
            HCCL_ERROR("[%s] release symmetric VA[%p] failed, ret[%d]", __func__, arenaBase_, ret);
            return;
        }
    }
    if (vaAllocator_ != nullptr) {
        vaAllocator_->Destroy();
    }
    arenaBase_ = nullptr;
    stride_ = 0;
    granularity_ = 0;
}

HcclResult UbMemSymmetricMemory::ValidateRegisterRange(void* ptr, size_t size) const
{
    CHK_PTR_NULL(ptr);
    CHK_PRT_RET(size == 0, HCCL_ERROR("[%s] window size is zero", __func__), HCCL_E_PARA);
    uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    CHK_PRT_RET(begin + size < begin, HCCL_ERROR("[%s] address overflow", __func__), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::GetMemoryInfo(void* ptr, size_t size, VaMappingInfo& mapping) const
{
    CHK_RET(ValidateRegisterRange(ptr, size));
    CHK_PRT_RET(arenaBase_ == nullptr, HCCL_ERROR("[%s] symmetric VA is not initialized", __func__), HCCL_E_UNAVAIL);
    void* allocationBase = nullptr;
    size_t allocationSize = 0;
    aclError ret = aclrtMemGetAddressRange(ptr, &allocationBase, &allocationSize);
    CHK_PRT_RET(
        ret != ACL_SUCCESS || allocationBase == nullptr || allocationSize == 0,
        HCCL_ERROR("[%s] get allocation range failed, ptr[%p], ret[%d]", __func__, ptr, ret), HCCL_E_PARA);
    uintptr_t request = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t base = reinterpret_cast<uintptr_t>(allocationBase);
    CHK_PRT_RET(
        request < base || request - base > allocationSize || size > allocationSize - (request - base),
        HCCL_ERROR("[%s] requested range is outside allocation", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        granularity_ == 0 || allocationSize % granularity_ != 0,
        HCCL_ERROR("[%s] allocation size[%zu] is not aligned to[%zu]", __func__, allocationSize, granularity_),
        HCCL_E_PARA);

    mapping.allocationBase = allocationBase;
    mapping.allocationSize = allocationSize;
    mapping.userOffset = static_cast<uint64_t>(request - base);
    ret = aclrtMemRetainAllocationHandle(allocationBase, &mapping.localHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS || mapping.localHandle == nullptr,
        HCCL_ERROR("[%s] retain local physical handle failed, ret[%d]", __func__, ret), HCCL_E_RUNTIME);
    HCCL_INFO(
        "[%s] retained PA handle[%p], allocationBase[%p], allocationSize[%zu]", __func__, mapping.localHandle,
        mapping.allocationBase, mapping.allocationSize);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::ExportLocalMapping(VaMappingInfo& mapping, std::vector<uint8_t>& shareableDesc) const
{
    CHK_PTR_NULL(mapping.localHandle);
    aclError ret = aclrtMemExportToShareableHandleV2(
        mapping.localHandle, 0, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, static_cast<void*>(&mapping.shareableHandle));
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] export local physical handle failed, ret[%d]", __func__, ret),
        HCCL_E_RUNTIME);
    const auto* begin = reinterpret_cast<const uint8_t*>(&mapping.shareableHandle);
    HcclResult descResult = HCCL_SUCCESS;
    EXCEPTION_CATCH(shareableDesc.assign(begin, begin + sizeof(mapping.shareableHandle)), descResult = HCCL_E_MEMORY);
    return descResult;
}

HcclResult UbMemSymmetricMemory::GrantLocalMemory(const VaMappingInfo& mapping)
{
    CHK_PRT_RET(
        memberPids_.size() != lsaTeamSize_ || mapping.localHandle == nullptr,
        HCCL_ERROR("[%s] invalid local grant parameters", __func__), HCCL_E_PARA);
    aclrtMemFabricHandle shareableHandle = mapping.shareableHandle;
    aclError ret = aclrtMemSetPidToShareableHandleV2(
        static_cast<void*>(&shareableHandle), ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, memberPids_.data(), memberPids_.size());
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] grant local memory failed, ret[%d]", __func__, ret), HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::SynchronizeMemberResult(HcclResult localResult, HcclResult& globalResult)
{
    std::vector<int32_t> memberResults;
    EXCEPTION_CATCH(memberResults.resize(lsaTeamSize_), return HCCL_E_MEMORY);
    int32_t result = static_cast<int32_t>(localResult);
    CHK_RET(agent_->ExchangeInfo(&result, memberResults.data(), sizeof(result)));
    globalResult = HCCL_SUCCESS;
    for (uint32_t member = 0; member < lsaTeamSize_; ++member) {
        if (memberResults[member] != static_cast<int32_t>(HCCL_SUCCESS)) {
            globalResult = static_cast<HcclResult>(memberResults[member]);
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::ValidateMappingRange(uint64_t windowOffset, size_t mapSize) const
{
    CHK_PRT_RET(
        arenaBase_ == nullptr || windowOffset > stride_ || mapSize > stride_ - windowOffset,
        HCCL_ERROR(
            "[%s] mapping exceeds stride, offset[%llu], size[%zu], stride[%llu]", __func__,
            static_cast<unsigned long long>(windowOffset), mapSize, static_cast<unsigned long long>(stride_)),
        HCCL_E_PARA);
    CHK_PRT_RET(
        granularity_ == 0 || windowOffset % granularity_ != 0 || mapSize % granularity_ != 0,
        HCCL_ERROR("[%s] mapping is not aligned to granularity[%zu]", __func__, granularity_), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::ImportMemberHandle(
    uint32_t member, const std::vector<uint8_t>& shareableDesc, const VaMappingInfo& mapping, aclrtDrvMemHandle& handle,
    bool& ownsHandle) const
{
    if (member == selfMember_) {
        handle = mapping.localHandle;
        ownsHandle = false;
        CHK_PRT_RET(handle == nullptr, HCCL_ERROR("[%s] local physical handle is null", __func__), HCCL_E_PTR);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        shareableDesc.size() != sizeof(aclrtMemFabricHandle),
        HCCL_ERROR("[%s] member[%u] invalid shareable descriptor size[%zu]", __func__, member, shareableDesc.size()),
        HCCL_E_PARA);
    aclrtMemFabricHandle remoteHandle{};
    CHK_PRT_RET(
        memcpy_s(&remoteHandle, sizeof(remoteHandle), shareableDesc.data(), shareableDesc.size()) != EOK,
        HCCL_ERROR("[%s] copy member[%u] shareable handle failed", __func__, member), HCCL_E_MEMORY);
    aclError ret = aclrtMemImportFromShareableHandleV2(
        static_cast<void*>(&remoteHandle), ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0, &handle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS || handle == nullptr,
        HCCL_ERROR("[%s] import member[%u] physical handle failed, ret[%d]", __func__, member, ret), HCCL_E_RUNTIME);
    ownsHandle = true;
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::MapAllMembers(
    uint64_t windowOffset, const std::vector<std::vector<uint8_t>>& shareableDescs, VaMappingInfo& mapping,
    size_t windowSize, std::vector<CommMem>& memberMems)
{
    CHK_RET(ValidateMappingRange(windowOffset, mapping.allocationSize));
    CHK_PRT_RET(
        mapping.userOffset > mapping.allocationSize || windowSize > mapping.allocationSize - mapping.userOffset,
        HCCL_ERROR("[%s] logical window exceeds mapped allocation", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        shareableDescs.size() != lsaTeamSize_, HCCL_ERROR("[%s] member descriptor count mismatch", __func__),
        HCCL_E_PARA);
    EXCEPTION_CATCH(mapping.peers.assign(lsaTeamSize_, PeerMappingInfo{}), return HCCL_E_MEMORY);
    EXCEPTION_CATCH(memberMems.assign(lsaTeamSize_, CommMem{}), return HCCL_E_MEMORY);
    for (uint32_t member = 0; member < lsaTeamSize_; ++member) {
        PeerMappingInfo& peer = mapping.peers[member];
        HcclResult ret = ImportMemberHandle(member, shareableDescs[member], mapping, peer.handle, peer.ownsHandle);
        if (ret != HCCL_SUCCESS) {
            return CleanupPartialMapping(mapping, ret);
        }
        uintptr_t base = reinterpret_cast<uintptr_t>(arenaBase_);
        uint64_t memberOffset = static_cast<uint64_t>(member) * stride_ + windowOffset;
        peer.address = reinterpret_cast<void*>(base + memberOffset);
        aclError aclRet = aclrtMapMem(peer.address, mapping.allocationSize, 0, peer.handle, 0);
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR("[%s] map member[%u] at[%p] failed, ret[%d]", __func__, member, peer.address, aclRet);
            return CleanupPartialMapping(mapping, HCCL_E_RUNTIME);
        }
        peer.mapped = true;
        ++activeMappingCount_;
        memberMems[member].addr = static_cast<uint8_t*>(peer.address) + mapping.userOffset;
        memberMems[member].size = windowSize;
        memberMems[member].type = COMM_MEM_TYPE_DEVICE;
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::CleanupPartialMapping(VaMappingInfo& mapping, HcclResult originalResult)
{
    HcclResult cleanupResult = UnmapAllMembers(mapping);
    if (cleanupResult != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] cleanup partial mapping failed, original[%d], cleanup[%d]", __func__, originalResult, cleanupResult);
    }
    return originalResult;
}

HcclResult UbMemSymmetricMemory::ReleasePeerMapping(PeerMappingInfo& peer)
{
    if (peer.mapped) {
        aclError ret = aclrtUnmapMem(peer.address);
        if (ret != ACL_SUCCESS) {
            HCCL_ERROR("[%s] unmap address[%p] failed, ret[%d]", __func__, peer.address, ret);
            return HCCL_E_RUNTIME;
        }
        peer.mapped = false;
        --activeMappingCount_;
    }
    if (peer.handle != nullptr && peer.ownsHandle) {
        aclError ret = aclrtFreePhysical(peer.handle);
        if (ret != ACL_SUCCESS) {
            HCCL_ERROR("[%s] free imported physical handle[%p] failed, ret[%d]", __func__, peer.handle, ret);
            return HCCL_E_RUNTIME;
        }
    }
    peer = PeerMappingInfo{};
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::UnmapAllMembers(VaMappingInfo& mapping)
{
    HcclResult firstError = HCCL_SUCCESS;
    for (PeerMappingInfo& peer : mapping.peers) {
        HcclResult ret = ReleasePeerMapping(peer);
        if (ret != HCCL_SUCCESS && firstError == HCCL_SUCCESS) {
            firstError = ret;
        }
    }
    // localHandle标识用户持有的本地物理allocation，本模块仅释放Import得到的远端Handle。
    mapping.localHandle = nullptr;
    const bool allReleased = std::all_of(mapping.peers.begin(), mapping.peers.end(), [](const PeerMappingInfo& peer) {
        return !peer.mapped && peer.handle == nullptr;
    });
    if (allReleased) {
        mapping.peers.clear();
    }
    return firstError;
}

HcclResult UbMemSymmetricMemory::AllocateWindowOffset(size_t size, uint64_t& offset)
{
    CHK_PRT_RET(
        granularity_ == 0 || size == 0 || size % granularity_ != 0,
        HCCL_ERROR("[%s] invalid mapping size[%zu], granularity[%zu]", __func__, size, granularity_), HCCL_E_PARA);
    CHK_PRT_RET(
        size > stride_,
        HCCL_ERROR(
            "[%s] window size[%zu] exceeds stride[%llu]", __func__, size, static_cast<unsigned long long>(stride_)),
        HCCL_E_MEMORY);
    CHK_SMART_PTR_NULL(vaAllocator_);
    size_t allocatedOffset = 0;
    CHK_RET(vaAllocator_->Reserve(size, granularity_, allocatedOffset));
    offset = static_cast<uint64_t>(allocatedOffset);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::ReleaseWindowOffset(uint64_t offset, size_t size)
{
    CHK_SMART_PTR_NULL(vaAllocator_);
    size_t releasedOffset = static_cast<size_t>(offset);
    CHK_PRT_RET(
        static_cast<uint64_t>(releasedOffset) != offset,
        HCCL_ERROR("[%s] offset[%llu] exceeds size_t range", __func__, static_cast<unsigned long long>(offset)),
        HCCL_E_PARA);
    return vaAllocator_->Release(releasedOffset, size);
}

HcclResult UbMemSymmetricMemory::BreakFailed(HcclResult result)
{
    broken_ = true;
    HCCL_ERROR(
        "[%s] UB Memory collective registration is broken, ret[%d]; deregister remaining windows and destroy the "
        "communicator",
        __func__, result);
    return result;
}

HcclResult UbMemSymmetricMemory::RegisterInternal(PaMappingInfo& paMapping)
{
    // 同一PA Handle仅在首次注册时执行Export、Grant和Map，重叠注册复用已有映射。
    bool firstRegistration = paMapping.refCount == 1U;
    HcclResult localExportResult = HCCL_SUCCESS;
    if (firstRegistration) {
        localExportResult = ExportLocalMapping(paMapping.vaMapping, paMapping.shareableDesc);
    }
    HcclResult globalExportResult = HCCL_SUCCESS;
    HcclResult exportSyncResult = SynchronizeMemberResult(localExportResult, globalExportResult);
    CHK_PRT_RET(
        exportSyncResult != HCCL_SUCCESS,
        HCCL_ERROR("[%s] synchronize export result failed, ret[%d]", __func__, exportSyncResult),
        BreakFailed(exportSyncResult));
    CHK_PRT_RET(
        globalExportResult != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] export LSA member memory failed, localResult[%d], globalResult[%d]", __func__, localExportResult,
            globalExportResult),
        globalExportResult);

    UbmemShareableInfo localInfo{
        paMapping.heapBaseOffset, paMapping.vaMapping.allocationSize, paMapping.vaMapping.shareableHandle};
    std::vector<UbmemShareableInfo> memberInfos;
    EXCEPTION_CATCH(memberInfos.resize(lsaTeamSize_), return BreakFailed(HCCL_E_MEMORY));
    HcclResult exchangeResult = agent_->ExchangeInfo(&localInfo, memberInfos.data(), sizeof(localInfo));
    CHK_PRT_RET(
        exchangeResult != HCCL_SUCCESS,
        HCCL_ERROR("[%s] exchange UB Memory shareable info failed, ret[%d]", __func__, exchangeResult),
        BreakFailed(exchangeResult));
    for (uint32_t member = 0; member < lsaTeamSize_; ++member) {
        CHK_PRT_RET(
            memberInfos[member].offset != paMapping.heapBaseOffset
                || memberInfos[member].size != paMapping.vaMapping.allocationSize,
            HCCL_ERROR(
                "[%s] member[%u] layout[offset:%llu,size:%llu] differs from local[offset:%llu,size:%zu]; "
                "ensure every LSA member invokes registration in the same order",
                __func__, member, static_cast<unsigned long long>(memberInfos[member].offset),
                static_cast<unsigned long long>(memberInfos[member].size),
                static_cast<unsigned long long>(paMapping.heapBaseOffset), paMapping.vaMapping.allocationSize),
            BreakFailed(HCCL_E_PARA));
    }

    // A5 UB Memory在全部成员完成Handle导出与交换后配置decoder，再同步授权结果后执行远端映射。
    HcclResult localGrantResult = firstRegistration ? GrantLocalMemory(paMapping.vaMapping) : HCCL_SUCCESS;
    HcclResult globalGrantResult = HCCL_SUCCESS;
    HcclResult grantSyncResult = SynchronizeMemberResult(localGrantResult, globalGrantResult);
    CHK_PRT_RET(
        grantSyncResult != HCCL_SUCCESS,
        HCCL_ERROR("[%s] synchronize grant result failed, ret[%d]", __func__, grantSyncResult),
        BreakFailed(grantSyncResult));
    CHK_PRT_RET(
        globalGrantResult != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] grant LSA member memory failed, localResult[%d], globalResult[%d]", __func__, localGrantResult,
            globalGrantResult),
        globalGrantResult);
    if (!firstRegistration) {
        return HCCL_SUCCESS;
    }

    std::vector<std::vector<uint8_t>> shareableDescs;
    EXCEPTION_CATCH(shareableDescs.resize(lsaTeamSize_), return BreakFailed(HCCL_E_MEMORY));
    for (uint32_t member = 0; member < lsaTeamSize_; ++member) {
        const auto* begin = reinterpret_cast<const uint8_t*>(&memberInfos[member].handle);
        EXCEPTION_CATCH(
            shareableDescs[member].assign(begin, begin + sizeof(memberInfos[member].handle)),
            return BreakFailed(HCCL_E_MEMORY));
    }
    HcclResult mapResult = MapAllMembers(
        paMapping.heapBaseOffset, shareableDescs, paMapping.vaMapping, paMapping.vaMapping.allocationSize,
        paMapping.memberMems);
    CHK_PRT_RET(
        mapResult != HCCL_SUCCESS, HCCL_ERROR("[%s] map LSA member memory failed, ret[%d]", __func__, mapResult),
        BreakFailed(mapResult));
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::PublishWindow(WindowRecord& record)
{
    CHK_SMART_PTR_NULL(record.paMapping);
    CHK_PTR_NULL(record.deviceWindow);
    std::vector<CommMem> userMemberMems;
    EXCEPTION_CATCH(userMemberMems = record.paMapping->memberMems, return HCCL_E_MEMORY);
    for (CommMem& memberMem : userMemberMems) {
        memberMem.addr = static_cast<uint8_t*>(memberMem.addr) + record.userOffset;
        memberMem.size = record.userSize;
    }
    void* baseVa = static_cast<uint8_t*>(arenaBase_) + record.paMapping->heapBaseOffset + record.userOffset;
    HcommResult ret = HcommTeamBindUbSymmetricWindow(
        record.deviceWindow, lsaTeam_, agent_->GetNetLayer(), userMemberMems.data(), lsaTeamSize_, baseVa, stride_,
        record.userSize);
    CHK_PRT_RET(
        ret != HCOMM_SUCCESS,
        HCCL_ERROR(
            "[%s] fill HcommWindow LSA information failed, window[%p], ret[%d]", __func__, record.deviceWindow, ret),
        static_cast<HcclResult>(ret));
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::AddWindowRecord(std::unique_ptr<WindowRecord>& record, HcclComm comm)
{
    uintptr_t userAddress = reinterpret_cast<uintptr_t>(record->userBase);
    HcclCommSymWindow handle = record->deviceWindow;
    WindowRecord* recordPtr = record.get();
    decltype(windowsByAddress_)::iterator addressIter;
    HcclResult ret = HCCL_SUCCESS;
    EXCEPTION_CATCH(addressIter = windowsByAddress_.emplace(userAddress, nullptr), ret = HCCL_E_MEMORY);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    addressIter->second = std::move(record);

    bool inserted = false;
    EXCEPTION_CATCH(inserted = windowsByHandle_.emplace(handle, recordPtr).second, ret = HCCL_E_MEMORY);
    if (ret == HCCL_SUCCESS && !inserted) {
        HCCL_ERROR("[%s] window handle[%p] already exists", __func__, handle);
        ret = HCCL_E_INTERNAL;
    }
    if (ret == HCCL_SUCCESS) {
        ret = RecordHcommWindowOwner(handle, comm);
    }
    if (ret != HCCL_SUCCESS) {
        if (inserted) {
            windowsByHandle_.erase(handle);
        }
        record = std::move(addressIter->second);
        windowsByAddress_.erase(addressIter);
        return ret;
    }
    recordPtr->state = WindowState::ACTIVE;
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::ReleasePaMapping(WindowRecord& record)
{
    if (!record.mappingRefHeld || record.paMapping == nullptr) {
        return HCCL_SUCCESS;
    }
    std::shared_ptr<PaMappingInfo> paMapping = record.paMapping;
    CHK_PRT_RET(paMapping->refCount == 0, HCCL_ERROR("[%s] invalid PA mapping reference", __func__), HCCL_E_INTERNAL);
    if (paMapping->refCount > 1U) {
        --paMapping->refCount;
    } else {
        HcclResult ret = UnmapAllMembers(paMapping->vaMapping);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] unmap member memory failed, ret[%d]", __func__, ret), ret);
        CHK_RET(ReleaseWindowOffset(paMapping->heapBaseOffset, paMapping->vaMapping.allocationSize));
        auto iter = paMappingMap_.find(paMapping->paHandle);
        if (iter != paMappingMap_.end() && iter->second.get() == paMapping.get()) {
            paMappingMap_.erase(iter);
        }
        paMapping->refCount = 0;
    }
    record.mappingRefHeld = false;
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemory::CleanupWindow(WindowRecord& record)
{
    // HcommWindow由CollComm统一创建和销毁，此处只释放UB Memory的LSA对称VA映射。
    return ReleasePaMapping(record);
}

HcclResult UbMemSymmetricMemory::RegisterWindow(void* ptr, size_t size, HcclCommSymWindow winHandle, HcclComm comm)
{
    CHK_PTR_NULL(winHandle);
    CHK_PTR_NULL(comm);
    std::lock_guard<std::mutex> lock(mutex_);
    CHK_PRT_RET(finalized_, HCCL_ERROR("[%s] symmetric memory manager is finalized", __func__), HCCL_E_UNAVAIL);
    CHK_PRT_RET(
        broken_,
        HCCL_ERROR(
            "[%s] UB Memory symmetric memory is broken after a failed collective registration; deregister "
            "remaining windows and destroy the communicator",
            __func__),
        HCCL_E_UNAVAIL);
    CHK_RET(EnsureRuntime());

    VaMappingInfo localMapping;
    CHK_RET(GetMemoryInfo(ptr, size, localMapping));
    const uint64_t userOffset = localMapping.userOffset;
    // 底层仍按完整allocation建立共享映射；各外层Window发布时再叠加用户注册地址偏移。
    localMapping.userOffset = 0;
    std::unique_ptr<WindowRecord> window;
    HcclResult ret = HCCL_SUCCESS;
    EXCEPTION_CATCH(window = std::make_unique<WindowRecord>(), ret = HCCL_E_MEMORY);
    if (ret != HCCL_SUCCESS) {
        (void)UnmapAllMembers(localMapping);
        return ret;
    }

    std::shared_ptr<PaMappingInfo> paMapping;
    auto mappingIter = paMappingMap_.find(localMapping.localHandle);
    if (mappingIter != paMappingMap_.end()) {
        paMapping = mappingIter->second;
        if (paMapping == nullptr || paMapping->vaMapping.allocationBase != localMapping.allocationBase
            || paMapping->vaMapping.allocationSize != localMapping.allocationSize) {
            HCCL_ERROR("[%s] PA handle is associated with a different allocation", __func__);
            (void)UnmapAllMembers(localMapping);
            return HCCL_E_INTERNAL;
        }
        CHK_RET(UnmapAllMembers(localMapping));
        CHK_PRT_RET(
            paMapping->refCount == std::numeric_limits<uint32_t>::max(),
            HCCL_ERROR("[%s] PA mapping reference is exhausted", __func__), HCCL_E_UNAVAIL);
        ++paMapping->refCount;
        HCCL_INFO("[%s] reuse PA handle[%p], refCount[%u]", __func__, paMapping->paHandle, paMapping->refCount);
    } else {
        uint64_t heapBaseOffset = 0;
        ret = AllocateWindowOffset(localMapping.allocationSize, heapBaseOffset);
        if (ret != HCCL_SUCCESS) {
            (void)UnmapAllMembers(localMapping);
            return ret;
        }
        EXCEPTION_CATCH(paMapping = std::make_shared<PaMappingInfo>(), ret = HCCL_E_MEMORY);
        if (ret != HCCL_SUCCESS) {
            (void)ReleaseWindowOffset(heapBaseOffset, localMapping.allocationSize);
            (void)UnmapAllMembers(localMapping);
            return ret;
        }
        paMapping->paHandle = localMapping.localHandle;
        paMapping->vaMapping = std::move(localMapping);
        paMapping->heapBaseOffset = heapBaseOffset;
        paMapping->refCount = 1U;
        bool inserted = false;
        EXCEPTION_CATCH(inserted = paMappingMap_.emplace(paMapping->paHandle, paMapping).second, ret = HCCL_E_MEMORY);
        if (ret != HCCL_SUCCESS || !inserted) {
            (void)ReleaseWindowOffset(heapBaseOffset, paMapping->vaMapping.allocationSize);
            (void)UnmapAllMembers(paMapping->vaMapping);
            return ret != HCCL_SUCCESS ? ret : HCCL_E_INTERNAL;
        }
    }

    window->userBase = ptr;
    window->userSize = size;
    window->userOffset = userOffset;
    window->deviceWindow = winHandle;
    window->paMapping = paMapping;
    window->mappingRefHeld = true;

    ret = RegisterInternal(*paMapping);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] register UB Memory mapping failed, ret[%d]", __func__, ret);
        (void)CleanupWindow(*window);
        return ret;
    }
    ret = PublishWindow(*window);
    if (ret != HCCL_SUCCESS) {
        (void)CleanupWindow(*window);
        return BreakFailed(ret);
    }
    ret = AddWindowRecord(window, comm);
    if (ret != HCCL_SUCCESS) {
        if (window != nullptr) {
            (void)CleanupWindow(*window);
        }
        return BreakFailed(ret);
    }
    HCCL_RUN_INFO(
        "[%s] A5 UB Memory symmetric window registered, comm[%s], userBase[%p], userSize[%zu], "
        "allocationBase[%p], allocationSize[%zu], userOffset[%llu], window[%p]",
        __func__, commId_.c_str(), ptr, size, paMapping->vaMapping.allocationBase, paMapping->vaMapping.allocationSize,
        static_cast<unsigned long long>(userOffset), winHandle);
    return HCCL_SUCCESS;
}

void UbMemSymmetricMemory::EraseWindowAddressRecord(uintptr_t userAddress, const WindowRecord* record)
{
    auto range = windowsByAddress_.equal_range(userAddress);
    for (auto iter = range.first; iter != range.second; ++iter) {
        if (iter->second.get() == record) {
            windowsByAddress_.erase(iter);
            return;
        }
    }
}

HcclResult UbMemSymmetricMemory::DeregisterWindow(HcclCommSymWindow winHandle)
{
    CHK_PTR_NULL(winHandle);
    std::lock_guard<std::mutex> lock(mutex_);
    CHK_PRT_RET(finalized_, HCCL_ERROR("[%s] symmetric memory manager is finalized", __func__), HCCL_E_UNAVAIL);
    auto handleIter = windowsByHandle_.find(winHandle);
    CHK_PRT_RET(
        handleIter == windowsByHandle_.end(), HCCL_ERROR("[%s] window[%p] not found", __func__, winHandle),
        HCCL_E_NOT_FOUND);
    WindowRecord* record = handleIter->second;
    CHK_PTR_NULL(record);
    uintptr_t userAddress = reinterpret_cast<uintptr_t>(record->userBase);
    record->state = WindowState::RETIRING;
    HcclResult ret = CleanupWindow(*record);
    if (ret != HCCL_SUCCESS) {
        record->state = WindowState::ACTIVE;
        return ret;
    }
    windowsByHandle_.erase(handleIter);
    EraseWindowAddressRecord(userAddress, record);
    return HCCL_SUCCESS;
}

void UbMemSymmetricMemory::Finalize()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (finalized_) {
        return;
    }
    for (auto iter = windowsByAddress_.begin(); iter != windowsByAddress_.end();) {
        WindowRecord& record = *iter->second;
        HcclResult ret = CleanupWindow(record);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] cleanup window[%p] failed, ret[%d]", __func__, record.deviceWindow, ret);
            ++iter;
            continue;
        }
        EraseHcommWindowOwner(record.deviceWindow);
        windowsByHandle_.erase(record.deviceWindow);
        iter = windowsByAddress_.erase(iter);
    }
    if (!windowsByAddress_.empty() || !paMappingMap_.empty()) {
        HCCL_ERROR(
            "[%s] UB Memory still has windows[%zu] or PA mappings[%zu]", __func__, windowsByAddress_.size(),
            paMappingMap_.size());
        return;
    }
    windowsByHandle_.clear();
    if (agent_ != nullptr) {
        agent_->Finalize();
    }
    FinalizeSymmetricVa();
    finalized_ = true;
}

} // namespace hccl
