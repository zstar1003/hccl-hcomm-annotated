/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_team_mgr.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

#include "adapter_rts_common.h"
#include "log.h"

namespace hccl {

HcclTeamMgr& HcclTeamMgr::GetInstance()
{
    static std::once_flag instanceFlag;
    static HcclTeamMgr* instance = nullptr;
    std::call_once(instanceFlag, [&] {
        instance = new HcclTeamMgr();
    });
    return *instance;
}

HcclResult HcclTeamMgr::RegisterPrebuiltWorldTeam(
    HcommTeamHandle worldTeam, CollComm* collComm, CommProtocol protocol, uint32_t netLayer, const uint32_t* rankIds,
    uint32_t rankNum)
{
    if (worldTeam == nullptr || collComm == nullptr || rankIds == nullptr) {
        HCCL_ERROR(
            "[HcclTeamMgr][%s] invalid param, worldTeam[%p] collComm[%p] rankIds[%p]", __func__, worldTeam, collComm,
            rankIds);
        return HCCL_E_PTR;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto worldIt = teamMap_.find(worldTeam);
    if (worldIt != teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] worldTeam[%p] already registered", __func__, worldTeam);
        return HCCL_E_PARA;
    }
    uint64_t key = MakeProtoLayerKey(protocol, netLayer);
    auto& commIndex = worldTeamIndex_[collComm];
    auto idxIt = commIndex.find(key);
    if (idxIt != commIndex.end()) {
        HCCL_ERROR(
            "[HcclTeamMgr][%s] prebuilt worldTeam for protocol[%d] netLayer[%u] already exists", __func__,
            static_cast<int32_t>(protocol), netLayer);
        return HCCL_E_PARA;
    }
    TeamEntry& entry = teamMap_[worldTeam];
    entry.collComm = collComm;
    entry.worldTeam = nullptr;
    entry.protocol = protocol;
    entry.syncMemPtr = nullptr; // worldTeam 不通信，不创建 syncMem
    entry.syncMemSize = 0;
    entry.syncMemHandle = nullptr;
    entry.syncMemTag.clear();
    entry.rankIds.assign(rankIds, rankIds + rankNum);
    commIndex[key] = worldTeam;
    // 同通信域内同层各协议 worldTeam 的 rankIds 相同，只记一份（供 UpdateWindowRemoteMemByRank 计算层槽位）
    auto& commLayers = layerRanksMap_[collComm];
    if (commLayers.find(netLayer) == commLayers.end()) {
        commLayers[netLayer] = entry.rankIds;
    }
    HCCL_INFO(
        "[HcclTeamMgr][%s] prebuilt worldTeam registered, handle[%p], protocol[%d], netLayer[%u], rankNum[%u]",
        __func__, worldTeam, static_cast<int32_t>(protocol), netLayer, rankNum);
    return HCCL_SUCCESS;
}

HcommTeamHandle HcclTeamMgr::FindWorldTeamByProtoLayer(CollComm* collComm, CommProtocol protocol, uint32_t netLayer)
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (collComm == nullptr) {
        return nullptr;
    }
    auto commIt = worldTeamIndex_.find(collComm);
    if (commIt == worldTeamIndex_.end()) {
        return nullptr;
    }
    auto it = commIt->second.find(MakeProtoLayerKey(protocol, netLayer));
    if (it == commIt->second.end()) {
        return nullptr;
    }
    return it->second;
}

std::vector<uint32_t>
HcclTeamMgr::GetPrebuiltWorldTeamRanks(CollComm* collComm, CommProtocol protocol, uint32_t netLayer)
{
    if (collComm == nullptr) {
        HCCL_ERROR("[HcclTeamMgr][%s] collComm is nullptr", __func__);
        return {};
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto commIt = worldTeamIndex_.find(collComm);
    if (commIt == worldTeamIndex_.end()) {
        return {};
    }
    auto indexIt = commIt->second.find(MakeProtoLayerKey(protocol, netLayer));
    if (indexIt == commIt->second.end()) {
        return {};
    }
    auto teamIt = teamMap_.find(indexIt->second);
    return teamIt == teamMap_.end() ? std::vector<uint32_t>() : teamIt->second.rankIds;
}

void HcclTeamMgr::GetWorldTeamSizesPerNetLayer(CollComm* collComm, std::vector<uint32_t>& sizes)
{
    sizes.clear();
    if (collComm == nullptr) {
        return;
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto commIt = layerRanksMap_.find(collComm);
    if (commIt == layerRanksMap_.end()) {
        return;
    }
    // 先确定最大 netLayer，按层填 0 再覆盖实际大小（无该层填 0，保证下标=netLayer）
    uint32_t maxLayer = 0;
    for (const auto& pair : commIt->second) {
        maxLayer = std::max(maxLayer, pair.first);
    }
    sizes.assign(static_cast<size_t>(maxLayer) + 1, 0);
    for (const auto& pair : commIt->second) {
        sizes[pair.first] = static_cast<uint32_t>(pair.second.size());
    }
}

void HcclTeamMgr::GetRankLayerSlots(
    CollComm* collComm, uint32_t rankId, std::vector<std::pair<uint32_t, uint32_t>>& slots)
{
    slots.clear();
    if (collComm == nullptr) {
        return;
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto commIt = layerRanksMap_.find(collComm);
    if (commIt == layerRanksMap_.end()) {
        return;
    }
    for (const auto& pair : commIt->second) {
        for (uint32_t memberId = 0; memberId < pair.second.size(); memberId++) {
            if (pair.second[memberId] == rankId) {
                slots.emplace_back(pair.first, memberId);
                break;
            }
        }
    }
}

HcclResult HcclTeamMgr::RegisterSubTeam(
    HcommTeamHandle worldTeam, HcommTeamHandle subTeam, void* syncMemPtr, uint64_t syncMemSize, const uint32_t* rankIds,
    uint32_t rankNum)
{
    if (worldTeam == nullptr || subTeam == nullptr || rankIds == nullptr) {
        HCCL_ERROR(
            "[HcclTeamMgr][%s] invalid param, worldTeam[%p] subTeam[%p] rankIds[%p]", __func__, worldTeam, subTeam,
            rankIds);
        return HCCL_E_PTR;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto worldIt = teamMap_.find(worldTeam);
    if (worldIt == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] worldTeam[%p] not registered", __func__, worldTeam);
        return HCCL_E_PARA;
    }
    if (teamMap_.find(subTeam) != teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] subTeam[%p] already registered", __func__, subTeam);
        return HCCL_E_PARA;
    }
    CollComm* collComm = worldIt->second.collComm;
    TeamEntry& entry = teamMap_[subTeam];
    entry.collComm = collComm;
    entry.worldTeam = worldTeam;
    entry.protocol = worldIt->second.protocol; // subTeam 继承 worldTeam 的 protocol
    entry.syncMemPtr = syncMemPtr;
    entry.syncMemSize = syncMemSize;
    entry.syncMemHandle = nullptr;
    entry.syncMemTag.clear();
    entry.rankIds.assign(rankIds, rankIds + rankNum);
    return HCCL_SUCCESS;
}

void HcclTeamMgr::UnregisterTeam(HcommTeamHandle team)
{
    if (team == nullptr) {
        return;
    }
    void* syncMemPtr = nullptr;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto it = teamMap_.find(team);
        if (it == teamMap_.end()) {
            return;
        }
        syncMemPtr = it->second.syncMemPtr;
        // 若为 worldTeam 且带 protocol，清理预制索引（限定其所属通信域的内层 map）
        if (it->second.worldTeam == nullptr && it->second.protocol != COMM_PROTOCOL_RESERVED) {
            CollComm* collComm = it->second.collComm;
            auto commIt = worldTeamIndex_.find(collComm);
            if (commIt != worldTeamIndex_.end()) {
                for (auto idxIt = commIt->second.begin(); idxIt != commIt->second.end(); ++idxIt) {
                    if (idxIt->second == team) {
                        commIt->second.erase(idxIt);
                        break;
                    }
                }
                if (commIt->second.empty()) {
                    worldTeamIndex_.erase(commIt);
                }
            }
        }
        teamMap_.erase(it);
    }
    if (syncMemPtr != nullptr) {
        (void)hrtFree(syncMemPtr);
    }
}

CollComm* HcclTeamMgr::FindCollComm(HcommTeamHandle team)
{
    CHK_PRT_RET(team == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] team[%p] is invalid", __func__, team), nullptr);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team);
        return nullptr;
    }
    return it->second.collComm;
}

HcommTeamHandle HcclTeamMgr::FindWorldTeam(HcommTeamHandle team)
{
    CHK_PRT_RET(team == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] team[%p] is invalid", __func__, team), nullptr);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team);
        return nullptr;
    }
    // world team 自身 worldTeam 字段为 nullptr，返回自身
    return (it->second.worldTeam != nullptr) ? it->second.worldTeam : team;
}

std::vector<uint32_t> HcclTeamMgr::GetRankIds(HcommTeamHandle team)
{
    std::vector<uint32_t> result;
    if (team == nullptr) {
        return result;
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        return result;
    }
    result = it->second.rankIds;
    return result;
}

HcclResult HcclTeamMgr::GetLsaTeam(CollComm* collComm, HcommTeamHandle& lsaTeam)
{
    CHK_PRT_RET(collComm == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] collComm is nullptr", __func__), HCCL_E_PTR);
    lsaTeam = nullptr;
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto commIt = worldTeamIndex_.find(collComm);
    if (commIt == worldTeamIndex_.end()) {
        HCCL_DEBUG("[HcclTeamMgr][%s] communicator has no prebuilt world team", __func__);
        return HCCL_E_NOT_FOUND;
    }
    for (const auto& indexEntry : commIt->second) {
        auto teamIt = teamMap_.find(indexEntry.second);
        CHK_PRT_RET(
            teamIt == teamMap_.end(), HCCL_ERROR("[HcclTeamMgr][%s] invalid world team index", __func__),
            HCCL_E_INTERNAL);
        if (teamIt->second.protocol != COMM_PROTOCOL_UB_MEM) {
            continue;
        }
        CHK_PRT_RET(
            lsaTeam != nullptr,
            HCCL_ERROR("[HcclTeamMgr][%s] communicator has multiple UB Memory world teams", __func__), HCCL_E_INTERNAL);
        CHK_PRT_RET(
            teamIt->second.collComm != collComm || teamIt->second.worldTeam != nullptr,
            HCCL_ERROR("[HcclTeamMgr][%s] invalid UB Memory world team metadata", __func__), HCCL_E_INTERNAL);
        lsaTeam = indexEntry.second;
    }
    if (lsaTeam == nullptr) {
        HCCL_DEBUG("[HcclTeamMgr][%s] communicator has no LSA team", __func__);
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclTeamMgr::MemberToRank(CollComm* collComm, HcommTeamHandle team, uint32_t memberId, uint32_t& rankId)
{
    CHK_PRT_RET(
        collComm == nullptr || team == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] invalid parameter", __func__),
        HCCL_E_PTR);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto teamIt = teamMap_.find(team);
    CHK_PRT_RET(
        teamIt == teamMap_.end(), HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team),
        HCCL_E_NOT_FOUND);
    CHK_PRT_RET(
        teamIt->second.collComm != collComm,
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] does not belong to communicator", __func__, team), HCCL_E_PARA);
    CHK_PRT_RET(
        memberId >= teamIt->second.rankIds.size(),
        HCCL_ERROR(
            "[HcclTeamMgr][%s] memberId[%u] exceeds team size[%zu]", __func__, memberId, teamIt->second.rankIds.size()),
        HCCL_E_PARA);
    rankId = teamIt->second.rankIds[memberId];
    return HCCL_SUCCESS;
}

HcclResult HcclTeamMgr::RankToMember(CollComm* collComm, HcommTeamHandle team, uint32_t rankId, uint32_t& memberId)
{
    CHK_PRT_RET(
        collComm == nullptr || team == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] invalid parameter", __func__),
        HCCL_E_PTR);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto teamIt = teamMap_.find(team);
    CHK_PRT_RET(
        teamIt == teamMap_.end(), HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team),
        HCCL_E_NOT_FOUND);
    CHK_PRT_RET(
        teamIt->second.collComm != collComm,
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] does not belong to communicator", __func__, team), HCCL_E_PARA);
    const auto& rankIds = teamIt->second.rankIds;
    auto rankIt = std::find(rankIds.begin(), rankIds.end(), rankId);
    CHK_PRT_RET(
        rankIt == rankIds.end(),
        HCCL_ERROR("[HcclTeamMgr][%s] rankId[%u] is not a member of team[%p]", __func__, rankId, team),
        HCCL_E_NOT_FOUND);
    memberId = static_cast<uint32_t>(rankIt - rankIds.begin());
    return HCCL_SUCCESS;
}

void* HcclTeamMgr::GetSyncMemPtr(HcommTeamHandle team)
{
    CHK_PRT_RET(team == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] team[%p] is invalid", __func__, team), nullptr);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team);
        return nullptr;
    }
    return it->second.syncMemPtr;
}

uint64_t HcclTeamMgr::GetSyncMemSize(HcommTeamHandle team)
{
    CHK_PRT_RET(team == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] team[%p] is invalid", __func__, team), 0);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team);
        return 0;
    }
    return it->second.syncMemSize;
}

void HcclTeamMgr::SetTeamSyncMemHandle(HcommTeamHandle team, HcclMemHandle handle, const std::string& tag)
{
    if (team == nullptr) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] is invalid", __func__, team);
        return;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team);
        return;
    }
    it->second.syncMemHandle = handle;
    it->second.syncMemTag = tag;
}

HcclMemHandle HcclTeamMgr::GetTeamSyncMemHandle(HcommTeamHandle team)
{
    CHK_PRT_RET(team == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] team[%p] is invalid", __func__, team), nullptr);
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team);
        return nullptr;
    }
    return it->second.syncMemHandle;
}

std::string HcclTeamMgr::GetTeamSyncMemTag(HcommTeamHandle team)
{
    CHK_PRT_RET(team == nullptr, HCCL_ERROR("[HcclTeamMgr][%s] team[%p] is invalid", __func__, team), std::string());
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = teamMap_.find(team);
    if (it == teamMap_.end()) {
        HCCL_ERROR("[HcclTeamMgr][%s] team[%p] not registered", __func__, team);
        return std::string();
    }
    return it->second.syncMemTag;
}

std::vector<HcclMemHandle> HcclTeamMgr::CollectPendingMemHandles(HcommTeamHandle worldTeam, HcommTeamHandle team)
{
    std::vector<HcclMemHandle> result;
    if (worldTeam == nullptr || team == nullptr) {
        return result;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto itTeam = teamMap_.find(team);
    if (itTeam == teamMap_.end()) {
        return result;
    }
    // syncMemHandle：首次注册且未交换时收集
    if (!itTeam->second.syncMemExchanged && itTeam->second.syncMemHandle != nullptr) {
        result.push_back(itTeam->second.syncMemHandle);
        itTeam->second.syncMemExchanged = true;
    }
    // window localMemHandle 不再收集：window 远端内存交换统一由 HcclChannelAcquire 内部
    // AppendSymmetricMemHandles 用 __hccl_sym_win__ tag 处理，AIN 侧无需重复注册。
    (void)worldTeam; // worldTeam 保留用于未来扩展，当前不再收集 window memHandle
    return result;
}

std::vector<HcommTeamHandle> HcclTeamMgr::GetSubTeams(HcommTeamHandle worldTeam)
{
    std::vector<HcommTeamHandle> result;
    if (worldTeam == nullptr) {
        return result;
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    // subTeam 的 worldTeam 字段指向其父 worldTeam；worldTeam 自身该字段为 nullptr。
    for (const auto& pair : teamMap_) {
        if (pair.second.worldTeam == worldTeam) {
            result.push_back(pair.first);
        }
    }
    return result;
}

std::vector<HcommTeamHandle> HcclTeamMgr::GetLinkedSubTeams(CollComm* collComm)
{
    std::vector<HcommTeamHandle> result;
    if (collComm == nullptr) {
        return result;
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    // subTeam（worldTeam != nullptr）建链即成功、存活即已建链；预制 worldTeam 不通信，排除
    for (const auto& pair : teamMap_) {
        if (pair.second.collComm == collComm && pair.second.worldTeam != nullptr) {
            result.push_back(pair.first);
        }
    }
    return result;
}

void HcclTeamMgr::ClearByCollComm(CollComm* collComm)
{
    if (collComm == nullptr) {
        return;
    }
    // 锁内收集并 erase，锁外销毁，避免持 L2 锁调 L3 接口（锁嵌套/死锁风险）
    std::vector<TeamCleanupInfo> cleanupInfos = CollectTeamCleanupInfo(collComm);
    ExecuteTeamCleanup(cleanupInfos);
}

std::vector<HcclTeamMgr::TeamCleanupInfo> HcclTeamMgr::CollectTeamCleanupInfo(CollComm* collComm)
{
    std::vector<TeamCleanupInfo> cleanupInfos;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (auto it = teamMap_.begin(); it != teamMap_.end();) {
        if (it->second.collComm != collComm) {
            ++it;
            continue;
        }
        TeamCleanupInfo info;
        info.handle = it->first;
        info.syncMemPtr = it->second.syncMemPtr;
        cleanupInfos.push_back(std::move(info));
        it = teamMap_.erase(it);
    }
    // 该通信域登记的两个索引一并按外层 key 清理（worldTeamIndex_ 内层条目随通信域整体失效）
    worldTeamIndex_.erase(collComm);
    layerRanksMap_.erase(collComm);
    return cleanupInfos;
}

void HcclTeamMgr::ExecuteTeamCleanup(const std::vector<TeamCleanupInfo>& cleanupInfos)
{
    // 锁外依次销毁：team（L3 device 资源）→ syncMem（L2 本地内存）
    for (const auto& info : cleanupInfos) {
        (void)HcommTeamDestroy(info.handle);
        if (info.syncMemPtr != nullptr) {
            (void)hrtFree(info.syncMemPtr);
        }
    }
}

} // namespace hccl
