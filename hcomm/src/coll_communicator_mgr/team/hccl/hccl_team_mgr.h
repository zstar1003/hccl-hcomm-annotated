/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TEAM_MGR_H
#define HCCL_TEAM_MGR_H

#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "coll_comm.h"
#include "hccl/hccl_res.h"
#include "hcomm_team.h"
#include "hcomm_team_defs.h"

namespace hccl {
/**
 * @note 职责：进程级 team 管理器，统一维护：
 *       1) WorldTeam 与 SubTeam 的父子关系（world→sub 列表 / sub→world）；
 *       2) Team 粒度的 syncMem 本地内存（HcommTeamCreate 返回大小后 hrtMalloc，待 channel 交换）；
 *       3) 预制 worldTeam 的 (protocol,netLayer) 索引。
 *       window 不再由 HcclTeamMgr 管理，统一归属通信域（CollComm）。
 */
struct TeamEntry {
    CollComm* collComm{nullptr};                   // 反查通信域
    HcommTeamHandle worldTeam{nullptr};            // 父 world team；world team 自身为 nullptr
    CommProtocol protocol{COMM_PROTOCOL_RESERVED}; // 该 team 使用的通信协议（worldTeam 预制时持久化）
    void* syncMemPtr{nullptr};                     // 本地 syncMem 内存指针（hrtMalloc 申请）
    uint64_t syncMemSize{0};                       // syncMem 内存大小
    // team 粒度 syncMem 内存句柄（首次 WindowRegister 时注册一次）
    HcclMemHandle syncMemHandle{nullptr};
    std::string syncMemTag;
    bool syncMemExchanged{false}; // syncMemHandle 是否已参与建链交换，避免重复交换
    // memberId→rankId 映射，下标=memberId，值=rankId。L2 维护，rankId 不下沉 L3。
    std::vector<uint32_t> rankIds;
};

class HcclTeamMgr {
public:
    static HcclTeamMgr& GetInstance();

    // —— 预制 worldTeam 索引（通信域内 protocol+netLayer 唯一）——
    // 注册预制 worldTeam：建立 (collComm, protocol,netLayer)→worldTeam 索引。worldTeam 不通信，不创建 syncMem。
    HcclResult RegisterPrebuiltWorldTeam(
        HcommTeamHandle worldTeam, CollComm* collComm, CommProtocol protocol, uint32_t netLayer,
        const uint32_t* rankIds, uint32_t rankNum);
    // 在指定通信域内按 (protocol, netLayer) 查找预制的 worldTeam，未找到返回 nullptr。
    HcommTeamHandle FindWorldTeamByProtoLayer(CollComm* collComm, CommProtocol protocol, uint32_t netLayer);
    // 取预制 worldTeam 的 rankIds 拷贝，未找到返回空。
    std::vector<uint32_t> GetPrebuiltWorldTeamRanks(CollComm* collComm, CommProtocol protocol, uint32_t netLayer);

    // —— 创建/销毁 team 时调用 ——
    // 注册 sub team：校验 worldTeam 存在，建父子关系，存 syncMem + rankIds（collComm 取自 worldTeam 条目）。
    HcclResult RegisterSubTeam(
        HcommTeamHandle worldTeam, HcommTeamHandle subTeam, void* syncMemPtr, uint64_t syncMemSize,
        const uint32_t* rankIds, uint32_t rankNum);
    // 销毁 team：hrtFree syncMem + erase 自身。memHandles 不注销。
    void UnregisterTeam(HcommTeamHandle team);

    // —— 查询 ——
    CollComm* FindCollComm(HcommTeamHandle team);
    // sub→world；world team 自身返回自身，未找到返回 nullptr。
    HcommTeamHandle FindWorldTeam(HcommTeamHandle team);
    // 取 team 的 rankIds（memberId→rankId 映射）拷贝，未找到返回空。
    std::vector<uint32_t> GetRankIds(HcommTeamHandle team);
    HcclResult GetLsaTeam(CollComm* collComm, HcommTeamHandle& lsaTeam);
    HcclResult MemberToRank(CollComm* collComm, HcommTeamHandle team, uint32_t memberId, uint32_t& rankId);
    HcclResult RankToMember(CollComm* collComm, HcommTeamHandle team, uint32_t rankId, uint32_t& memberId);

    // —— syncMem ——
    void* GetSyncMemPtr(HcommTeamHandle team);
    uint64_t GetSyncMemSize(HcommTeamHandle team);

    // —— team 粒度 syncMem 内存句柄（ChannelsCreate 首次注册并取用）——
    void SetTeamSyncMemHandle(HcommTeamHandle team, HcclMemHandle handle, const std::string& tag);
    HcclMemHandle GetTeamSyncMemHandle(HcommTeamHandle team);
    std::string GetTeamSyncMemTag(HcommTeamHandle team);

    // 收集未交换的 syncMemHandle，标记已交换，避免重复建链交换。
    std::vector<HcclMemHandle> CollectPendingMemHandles(HcommTeamHandle worldTeam, HcommTeamHandle team);

    // —— 预制 worldTeam 各层信息（供 HcommWindow.netWin.baseRemoteMemAddr 层分段表计算，按通信域隔离）——
    // 取指定通信域各 netLayer 的 worldTeam 大小（变长出参，下标=netLayer，无该层填 0）。
    void GetWorldTeamSizesPerNetLayer(CollComm* collComm, std::vector<uint32_t>& sizes);
    // 取 rankId 在指定通信域各层出现的槽位：(netLayer, 该层 worldTeam 中的 worldTeamId) 列表。
    void GetRankLayerSlots(CollComm* collComm, uint32_t rankId, std::vector<std::pair<uint32_t, uint32_t>>& slots);

    // 取 worldTeam 下所有 subTeam 的 handle（worldTeam 销毁时连带销毁用）。锁内收集，调用方锁外销毁。
    std::vector<HcommTeamHandle> GetSubTeams(HcommTeamHandle worldTeam);

    // 取该通信域下所有 subTeam（subTeam 建链即成功、存活即已建链；worldTeam 预制不通信，排除）。
    // window 后注册时对其补交换（建链参数 engine 由 HcommTeamGetEngine 运行时查询）。
    std::vector<HcommTeamHandle> GetLinkedSubTeams(CollComm* collComm);

    // —— CollComm 析构兜底：清理属于该 comm 的所有 team 条目（hrtFree syncMem + erase）——
    void ClearByCollComm(CollComm* collComm);

private:
    HcclTeamMgr() = default;
    ~HcclTeamMgr() = default;
    HcclTeamMgr(const HcclTeamMgr&) = delete;
    HcclTeamMgr& operator=(const HcclTeamMgr&) = delete;

    // ClearByCollComm 的单 team 销毁信息（锁内收集，锁外销毁）
    struct TeamCleanupInfo {
        HcommTeamHandle handle{nullptr};
        void* syncMemPtr{nullptr};
    };
    // 锁内收集该通信域下所有 team 的销毁信息并 erase teamMap 条目
    std::vector<TeamCleanupInfo> CollectTeamCleanupInfo(CollComm* collComm);
    // 锁外依次销毁：window（L3）→ team（L3）→ syncMem（L2 本地内存）
    void ExecuteTeamCleanup(const std::vector<TeamCleanupInfo>& cleanupInfos);

    std::unordered_map<HcommTeamHandle, TeamEntry> teamMap_;
    std::shared_mutex mutex_; // 读写锁：查询类接口多读，注册/销毁类接口独占写

    // 预制 worldTeam 的 (protocol,netLayer)→worldTeam 索引，保证组合唯一性
    // key = protocol << NET_LAYER_BITS | netLayer：protocol 枚举（int32 量级）占高位，netLayer 占低 32 位
    static constexpr uint32_t NET_LAYER_BITS = 32;
    static uint64_t MakeProtoLayerKey(CommProtocol protocol, uint32_t netLayer)
    {
        return (static_cast<uint64_t>(protocol) << NET_LAYER_BITS) | static_cast<uint64_t>(netLayer);
    }
    // 通信域→(protocol,netLayer→worldTeam)
    std::unordered_map<CollComm*, std::unordered_map<uint64_t, HcommTeamHandle>> worldTeamIndex_;
    // 通信域→(netLayer→该层 rankIds)。同通信域内同层各协议 rankIds 相同只记一份，供层槽位计算。
    std::unordered_map<CollComm*, std::unordered_map<uint32_t, std::vector<uint32_t>>> layerRanksMap_;
};
} // namespace hccl

#endif // HCCL_TEAM_MGR_H
