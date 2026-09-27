/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_TEAM_MGR_H
#define HCOMM_TEAM_MGR_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "hcomm_res.h"
#include "hcomm_res_defs.h"
#include "hcomm_team.h"
#include "hcomm_team_defs.h"
#include "hcomm_team_entity_defs.h"

namespace hcomm {

struct WindowEntry {
    HcommWindow hostWindow{};
    void* devWindow{nullptr};
    /* netWin.baseRemoteMemAddr 指向的扁平 addr 数组的 host 副本
     * （按 netLayer 分段，槽位=worldTeamAccumulateId[L]+worldTeamId），sync 到 devRemoteAddrs 后
     * 由 baseRemoteMemAddr 记录 device 基地址；windowSize 为标量（所有 remoteMem size 相同） */
    uint64_t* hostRemoteAddrs{nullptr};
    void* devRemoteAddrs{nullptr};
    /* netWin.worldTeamAccumulateId 的 device 副本（SyncWindowToDevice 前替换 hostWindow 指针为 device 地址） */
    uint32_t* devWorldTeamAccumulateId{nullptr};
    uint32_t remoteMemsTotal{0}; /* 表总长度 = worldTeamAccumulateId[netLayerNum-1] + 最后一层 size */
    HcommTeamHandle teamHandle{nullptr};
    bool ownsDeviceWindow{false};
    /* 本端窗口注册信息（RegisterWindow 时由 L2 经 HcommTeamWindowSetSelfInfo 登记用户 VA/size。
     * 回填本端槽位时 addr/size 取自此处；未登记（selfVa 为空）则跳过本端槽位回填 */
    void* selfVa{nullptr};
    uint64_t selfSize{0};
    uint32_t selfRankId{0}; // 本端 rankId（仅日志/排障用）
    /* 本端槽位（L2 用 GetRankLayerSlots(selfRankId) + 前缀和算好传入，
     * 算法与 remoteRank 槽位一致：sum(sizes[0..L-1]) + 该层 worldTeamId） */
    std::vector<uint32_t> selfSlots;
};

struct TeamEntry {
    HcommTeam hostTeam{};
    HcommTeamHandle devTeam{nullptr};
    void* devWorldTeamIds{nullptr};
    void* devChannelNums{nullptr};
    void* devChannels{nullptr};
    void* devRemoteMems{nullptr};

    uint32_t* hostWorldTeamIds{nullptr};
    std::vector<uint32_t> hostChannelNums;
    std::vector<std::vector<uint64_t>> channelsList;

    CommMem* hostRemoteMems{nullptr};

    HcommTeamHandle worldTeamHandle{nullptr};
    bool isSubTeam{false};

    HcommTeamSyncMemRequirement syncMemReq{};
    uint64_t syncMemSize{0};
};

class HcommTeamMgr {
public:
    static HcommTeamMgr& GetInstance();

    HcommResult TeamCreate(
        HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* outSyncMemSize);
    HcommResult TeamDestroy(HcommTeamHandle team);
    HcommResult WindowRegister(void* devLegacySymWin, HcclCommSymWindow* handle);
    HcommResult WindowDeregister(HcclCommSymWindow handle);
    HcommResult BindUbSymmetricWindow(
        HcclCommSymWindow handle, HcommTeamHandle lsaTeam, uint32_t netLayer, const CommMem* memberMems,
        uint32_t memberNum, void* baseVa, size_t stride, size_t userSize);
    // 对各 netLayer 段的槽位更新远端内存并同步到 device（供 CollComm 统一回填 HcommWindow）
    // sizes/sizeNum：各层 worldTeam 大小（首次调用时按 sum(sizes[0..sizeNum-1]) 分配层分段表并填
    // netLayerNum） slots/slotNum：最终槽位数组（值 = worldTeamAccumulateId[L] +
    // worldTeamId），由调用方（L2）计算
    HcommResult UpdateWindowRemoteMemByRank(
        HcclCommSymWindow handle, const uint32_t* sizes, uint32_t sizeNum, const uint32_t* slots, uint32_t slotNum,
        const CommMem& remoteMem);
    // 登记本端窗口注册信息（用户 VA/size/rankId + 本端层槽位），供回填本端槽位用
    HcommResult SetWindowSelfInfo(
        HcclCommSymWindow handle, void* selfVa, uint64_t selfSize, const uint32_t* selfSlots, uint32_t selfSlotNum);
    // 用窗口自身注册信息回填本端（self rank）层槽位并 H2D 同步（幂等：重复写相同值无害）
    HcommResult UpdateWindowSelfSlots(WindowEntry* winEntry);
    HcommResult BindChannels(HcommTeamHandle team, const HcommTeamBindChannelsDesc* desc);
    HcommResult BindSyncMem(HcommTeamHandle team, const HcommTeamBindSyncMemDesc* desc);
    HcommResult GetNetLayer(HcommTeamHandle team, uint32_t* netLayer);
    HcommResult GetEngine(HcommTeamHandle team, CommEngine* engine);

private:
    HcommTeamMgr() = default;
    ~HcommTeamMgr();
    HcommTeamMgr(const HcommTeamMgr&) = delete;
    HcommTeamMgr& operator=(const HcommTeamMgr&) = delete;

    TeamEntry* FindTeamByHandleLocked(HcommTeamHandle handle);
    WindowEntry* FindWindowByHandleLocked(HcclCommSymWindow handle);
    HcommResult SyncTeamToDevice(TeamEntry* entry);
    HcommResult SyncWindowToDevice(WindowEntry* entry);
    void ClearWindowNetWin(WindowEntry* winEntry);
    HcommResult AllocWindowAddrTable(WindowEntry* winEntry, uint32_t total);
    HcommResult AllocWindowAccumulateIdArray(WindowEntry* winEntry, const uint32_t* sizes, uint32_t sizeNum);
    HcommResult AllocWindowNetWin(WindowEntry* winEntry, const uint32_t* sizes, uint32_t sizeNum);
    HcommResult ValidateUbSymmetricWindowLayout(
        const CommMem* memberMems, uint32_t memberNum, const void* baseVa, size_t stride, size_t userSize) const;
    HcommResult AllocAndCopyWorldTeamIds(TeamEntry* entry, const uint32_t* src, uint32_t memberNum);
    HcommResult AllocAndCopyChannels(TeamEntry* entry);
    HcommResult AllocChannelEntities(TeamEntry* entry);
    HcommResult AllocChannelNumsArray(TeamEntry* entry);
    HcommResult AllocAndCopyRemoteMems(TeamEntry* entry, const CommMem* src, uint32_t memberNum);
    // 首次分配 hostRemoteMems（calloc+memcpy）与 devRemoteMems（hrtMalloc+hrtMemSyncCopy），并写入 syncMem 字段
    HcommResult AllocRemoteMems(TeamEntry* entry, const CommMem* src, uint32_t memberNum);
    // 重复 bind：校验维度一致后更新 hostRemoteMems（memcpy）并重新 sync 到 devRemoteMems
    HcommResult UpdateRemoteMems(TeamEntry* entry, const CommMem* src, uint32_t memberNum);
    HcommResult ValidateSubTeam(TeamEntry* worldEntry, const HcommTeamCreateDesc* desc);
    void InitTeamEntry(TeamEntry* entry, const HcommTeamCreateDesc* desc, HcommTeamHandle worldTeam);
    HcommResult AllocAndSyncTeam(TeamEntry* entry, const HcommTeamCreateDesc* desc);
    void MergeChannelLists(
        TeamEntry* entry, const HcommTeamBindChannelsDesc* desc, std::vector<std::vector<uint64_t>>& newChannels);
    void FreeTeamResources(TeamEntry* entry);
    void FreeWindowResources(WindowEntry* entry);

    std::shared_mutex teamsRwMutex_; // 保护 teams_
    std::unordered_map<HcommTeamHandle, std::unique_ptr<TeamEntry>> teams_;
    std::shared_mutex windowsRwMutex_; // 保护 windows_
    std::unordered_map<HcclCommSymWindow, std::unique_ptr<WindowEntry>> windows_;
};

} // namespace hcomm

#endif // HCOMM_TEAM_MGR_H
