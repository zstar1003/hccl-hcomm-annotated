/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UBMEM_SYMMETRIC_MEMORY_H
#define UBMEM_SYMMETRIC_MEMORY_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "acl/acl_rt.h"
#include "hcomm_c_adpt.h"
#include "hcomm_res_defs.h"
#include "hcomm_team_defs.h"
#include "hccl/hccl_types.h"
#include "ubmem_symmetric_memory_agent.h"

namespace hccl {

class CollComm;

struct UbmemShareableInfo {
    uint64_t offset;
    uint64_t size;
    aclrtMemFabricHandle handle;
};

class UbMemSymmetricMemory {
public:
    UbMemSymmetricMemory(
        CollComm* collComm, HcommTeamHandle lsaTeam, uint32_t netLayer, const std::vector<uint32_t>& worldRankIds);
    ~UbMemSymmetricMemory();

    HcclResult Init();
    void Finalize();
    HcclResult RegisterWindow(void* ptr, size_t size, HcclCommSymWindow winHandle, HcclComm comm);
    HcclResult DeregisterWindow(HcclCommSymWindow winHandle);
    HcommTeamHandle GetLsaTeam() const { return lsaTeam_; }

private:
    class SimpleVaAllocator;

    static constexpr uint64_t BYTES_PER_GB = 1024ULL * 1024ULL * 1024ULL;

    enum class WindowState : uint8_t { REGISTERING, ACTIVE, RETIRING };

    struct PeerMappingInfo {
        void* address{nullptr};
        aclrtDrvMemHandle handle{nullptr};
        bool mapped{false};
        bool ownsHandle{false};
    };

    struct VaMappingInfo {
        void* allocationBase{nullptr};
        size_t allocationSize{0};
        uint64_t userOffset{0};
        aclrtDrvMemHandle localHandle{nullptr};
        aclrtMemFabricHandle shareableHandle{};
        std::vector<PeerMappingInfo> peers;
    };

    // 一个物理allocation只建立一套LSA对称VA映射，多个外层Window通过refCount共享。
    struct PaMappingInfo {
        aclrtDrvMemHandle paHandle{nullptr};
        VaMappingInfo vaMapping;
        std::vector<uint8_t> shareableDesc;
        std::vector<CommMem> memberMems;
        uint64_t heapBaseOffset{0};
        uint32_t refCount{0};
    };

    struct WindowRecord {
        void* userBase{nullptr};
        size_t userSize{0};
        uint64_t userOffset{0}; // 用户注册地址相对完整allocation基址的偏移
        HcclCommSymWindow deviceWindow{nullptr};
        std::shared_ptr<PaMappingInfo> paMapping;
        bool mappingRefHeld{false};
        WindowState state{WindowState::REGISTERING};
    };

    HcclResult CheckPrebuiltLsaTeam();
    HcclResult EnsureRuntime();
    HcclResult GetAllMemberPids();
    HcclResult ValidateRegisterRange(void* ptr, size_t size) const;
    HcclResult InitSymmetricVa();
    HcclResult InitGranularity();
    HcclResult ReserveArena();
    HcclResult InitWindowOffsetAllocator();
    void FinalizeSymmetricVa();
    HcclResult GetMemoryInfo(void* ptr, size_t size, VaMappingInfo& mapping) const;
    HcclResult ExportLocalMapping(VaMappingInfo& mapping, std::vector<uint8_t>& shareableDesc) const;
    HcclResult GrantLocalMemory(const VaMappingInfo& mapping);
    HcclResult SynchronizeMemberResult(HcclResult localResult, HcclResult& globalResult);
    HcclResult ValidateMappingRange(uint64_t windowOffset, size_t mapSize) const;
    HcclResult ImportMemberHandle(
        uint32_t member, const std::vector<uint8_t>& shareableDesc, const VaMappingInfo& mapping,
        aclrtDrvMemHandle& handle, bool& ownsHandle) const;
    HcclResult MapAllMembers(
        uint64_t windowOffset, const std::vector<std::vector<uint8_t>>& shareableDescs, VaMappingInfo& mapping,
        size_t windowSize, std::vector<CommMem>& memberMems);
    HcclResult CleanupPartialMapping(VaMappingInfo& mapping, HcclResult originalResult);
    HcclResult ReleasePeerMapping(PeerMappingInfo& peer);
    HcclResult UnmapAllMembers(VaMappingInfo& mapping);
    HcclResult AllocateWindowOffset(size_t size, uint64_t& offset);
    HcclResult ReleaseWindowOffset(uint64_t offset, size_t size);
    HcclResult RegisterInternal(PaMappingInfo& paMapping);
    // 集合注册中 ring 交换完成后任一本成员局部失败都会使组内成员失步，不可安全重试，置永久失败。
    HcclResult BreakFailed(HcclResult result);
    HcclResult PublishWindow(WindowRecord& record);
    HcclResult AddWindowRecord(std::unique_ptr<WindowRecord>& record, HcclComm comm);
    HcclResult ReleasePaMapping(WindowRecord& record);
    HcclResult CleanupWindow(WindowRecord& record);
    void EraseWindowAddressRecord(uintptr_t userAddress, const WindowRecord* record);

    CollComm* collComm_{nullptr};
    RankGraph* rankGraph_{nullptr};
    uint32_t selfRank_{0};
    uint32_t selfMember_{0};
    uint32_t lsaTeamSize_{0};
    uint32_t netLayer_{0};
    std::string commId_;
    std::vector<uint32_t> worldRankIds_;
    HcommTeamHandle lsaTeam_{nullptr};
    std::unique_ptr<UbMemSymmetricMemoryAgent> agent_;
    void* arenaBase_{nullptr};
    uint64_t stride_{0};
    size_t granularity_{0};
    size_t activeMappingCount_{0};
    std::once_flag runtimeInitFlag_;
    HcclResult runtimeInitResult_{HCCL_E_INTERNAL};
    std::vector<int32_t> memberPids_;
    std::unique_ptr<SimpleVaAllocator> vaAllocator_;
    std::multimap<uintptr_t, std::unique_ptr<WindowRecord>> windowsByAddress_;
    std::unordered_map<HcclCommSymWindow, WindowRecord*> windowsByHandle_;
    std::unordered_map<aclrtDrvMemHandle, std::shared_ptr<PaMappingInfo>> paMappingMap_;
    mutable std::mutex mutex_;
    bool finalized_{false};
    bool broken_{false}; // 集合注册失步后置true，拒绝后续注册，需销毁重建通信域
};

} // namespace hccl

#endif // UBMEM_SYMMETRIC_MEMORY_H
