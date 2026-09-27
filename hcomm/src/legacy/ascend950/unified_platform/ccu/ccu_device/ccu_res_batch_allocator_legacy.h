/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_RES_BATCH_ALLOCATOR_H
#define HCCL_CCU_RES_BATCH_ALLOCATOR_H

#include <mutex>
#include <vector>
#include <memory>
#include <unordered_map>
#include "ccu_res_specs_legacy.h"
#include "ccu_device_manager.h"

namespace Hccl {

struct BlockInfo {
    uint32_t id{0};
    uint32_t startId{0};
    uint32_t num{0};
    uintptr_t handle{0};
    bool allocated{false};
};

struct CcuResBlockNums {
    uint32_t loopNum{0};
    uint32_t msNum{0};
    uint32_t ckeNum{0};
    uint32_t xnNum{0};
    uint32_t gsaNum{0};
};

class CcuResBatchAllocator {
public:
    CcuResBatchAllocator(const CcuResBatchAllocator& that) = delete;
    CcuResBatchAllocator& operator=(const CcuResBatchAllocator& that) = delete;
    ~CcuResBatchAllocator() = default; // 不允许在析构中调用CcuComponent，会引起未定义行为

    static CcuResBatchAllocator& GetInstance(const int32_t deviceLogicId);
    void Init();
    void Deinit();

    HcclResult AllocResHandle(const CcuResReq& resReq, CcuResHandle& resHandle);
    HcclResult ReleaseResHandle(const CcuResHandle& handle);
    HcclResult GetResource(const CcuResHandle& handle, CcuResRepository& ccuResRepo);
    HcclResult QueryRemainRes(uint8_t dieId, ResType resType, uint32_t& remainNum) const;
    // 获取指定 die 上可分配的块类型资源数量 = 预分配块数量 × CcuBlockResStrategy 对应块大小
    // 仅支持 ResType::LOOP / MS / CKE / XN / GSA
    HcclResult GetAllocatableMaxBlockResNum(ResType resType, uint8_t dieId, uint32_t& num) const;

private:
    class CcuMissionMgr {
    public:
        CcuMissionMgr() = default;
        ~CcuMissionMgr() = default;

        void Reset();
        HcclResult PreAlloc(
            const int32_t devLogicId, const uint32_t blockSize, const std::array<bool, MAX_CCU_IODIE_NUM> dieFlags);
        HcclResult Alloc(const uintptr_t handleKey, const MissionReq& missionReq, MissionResInfo& missionInfos);
        void Release(MissionResInfo& missionInfos);
        const std::vector<BlockInfo>& GetBlocks() const { return blocks; }

    private:
        uint32_t stragtegy{0};
        std::array<bool, MAX_CCU_IODIE_NUM> dieEnableFlags;
        std::vector<BlockInfo> blocks;
    };

    // 键值为CcuResRepository裸指针转换的uintptr_t
    std::unordered_map<uintptr_t, std::unique_ptr<CcuResRepository>> handleMap;
    CcuMissionMgr missionMgr;
    CcuResBlockNums maxResBlockNums{};

    int32_t devLogicId{0};
    std::array<bool, MAX_CCU_IODIE_NUM> dieEnableFlags;
    std::array<CcuBlockResStrategy, MAX_CCU_IODIE_NUM> resStrategies;
    bool preAllocated{false};
    std::array<std::unordered_map<ResType, std::vector<BlockInfo>, std::EnumClassHash>, MAX_CCU_IODIE_NUM> resBlocks{};
    mutable std::mutex innerMutex;

    explicit CcuResBatchAllocator() = default;

    CcuResBlockNums GetPreAllocatedMaxBlockNums(const uint8_t dieId) const;

    HcclResult PreAllocBlockRes();
    HcclResult TryAllocResHandle(
        const uintptr_t handleKey, const CcuResReq& resReq, std::unique_ptr<CcuResRepository>& resRepoPtr);
    HcclResult
    AllocBlockRes(const uintptr_t handleKey, const CcuResReq& resReq, std::unique_ptr<CcuResRepository>& resRepoPtr);
    HcclResult AllocConsecutiveRes(const CcuResReq& resReq, std::unique_ptr<CcuResRepository>& resRepoPtr) const;
    HcclResult AllocDiscreteRes(const CcuResReq& resReq, std::unique_ptr<CcuResRepository>& resRepoPtr) const;
    HcclResult ReleaseResource(std::unique_ptr<CcuResRepository>& resRepoPtr);
    void ReleaseBlockResource(std::unique_ptr<CcuResRepository>& resRepoPtr);
    HcclResult ReleaseNonBlockTypeRes(std::unique_ptr<CcuResRepository>& resRepoPtr) const;
    // 根据 resType 解析对应的 blocks 指针，用于 QueryRemainRes 降低圈复杂度
    HcclResult ResolveBlocksPtr(uint8_t dieId, ResType resType, const std::vector<BlockInfo>*& blocksPtr) const;
};

}; // namespace Hccl

#endif
