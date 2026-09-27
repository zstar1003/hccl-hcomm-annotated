/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_team_mgr.h"

#include <cstdlib>

#include "adapter_rts_common.h"
#include "hcomm_res_entity_defs.h"
#include "hcomm_result_defs.h"
#include "hcomm_team_entity_defs.h"
#include "log.h"
#include "securec.h"

namespace hcomm {

HcommTeamMgr& HcommTeamMgr::GetInstance()
{
    static std::once_flag instanceFlag;
    static HcommTeamMgr* instance = nullptr;
    std::call_once(instanceFlag, [&] {
        instance = new HcommTeamMgr();
    });
    return *instance;
}

HcommTeamMgr::~HcommTeamMgr()
{
    {
        std::unique_lock<std::shared_mutex> lock(teamsRwMutex_);
        for (auto& pair : teams_) {
            if (pair.second != nullptr) {
                FreeTeamResources(pair.second.get());
            }
        }
        teams_.clear();
    }
    {
        std::unique_lock<std::shared_mutex> lock(windowsRwMutex_);
        for (auto& pair : windows_) {
            if (pair.second != nullptr) {
                FreeWindowResources(pair.second.get());
            }
        }
        windows_.clear();
    }
}

TeamEntry* HcommTeamMgr::FindTeamByHandleLocked(HcommTeamHandle handle)
{
    auto it = teams_.find(handle);
    if (it == teams_.end()) {
        HCCL_ERROR("[FindTeamByHandleLocked] team handle[%p] not found", handle);
        return nullptr;
    }
    return it->second.get();
}

WindowEntry* HcommTeamMgr::FindWindowByHandleLocked(HcclCommSymWindow handle)
{
    auto it = windows_.find(handle);
    if (it == windows_.end()) {
        HCCL_ERROR("[FindWindowByHandleLocked] window handle[%p] not found", handle);
        return nullptr;
    }
    return it->second.get();
}

HcommResult HcommTeamMgr::SyncTeamToDevice(TeamEntry* entry)
{
    if (entry->devTeam == nullptr) {
        HCCL_ERROR("[SyncTeamToDevice] devTeam is null");
        return HCOMM_E_PTR;
    }
    return static_cast<HcommResult>(hrtMemSyncCopy(
        entry->devTeam, sizeof(HcommTeam), &entry->hostTeam, sizeof(HcommTeam),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
}

HcommResult HcommTeamMgr::SyncWindowToDevice(WindowEntry* entry)
{
    if (entry->devWindow == nullptr) {
        HCCL_ERROR("[SyncWindowToDevice] devWindow is null");
        return HCOMM_E_PTR;
    }
    // SyncWindowToDevice 前，把 host 指针替换为 device 副本地址，同步后还原，避免 device 侧拿到无效 host 指针
    uint32_t* hostAccumIdPtr = entry->hostWindow.netWin.worldTeamAccumulateId;
    if (entry->devWorldTeamAccumulateId != nullptr) {
        entry->hostWindow.netWin.worldTeamAccumulateId = entry->devWorldTeamAccumulateId;
    }
    HcommResult ret = static_cast<HcommResult>(hrtMemSyncCopy(
        entry->devWindow, sizeof(HcommWindow), &entry->hostWindow, sizeof(HcommWindow),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    entry->hostWindow.netWin.worldTeamAccumulateId = hostAccumIdPtr; // 还原 host 指针，供后续更新
    return ret;
}

HcommResult HcommTeamMgr::AllocAndCopyWorldTeamIds(TeamEntry* entry, const uint32_t* src, uint32_t memberNum)
{
    // worldTeam（src==nullptr）：worldTeamIds 置为 [0,memberNum) 连续序列（memberId 自身）
    std::vector<uint32_t> sequentialIds;
    const uint32_t* srcPtr = src;
    if (src == nullptr) {
        sequentialIds.resize(memberNum);
        for (uint32_t i = 0; i < memberNum; i++) {
            sequentialIds[i] = i;
        }
        srcPtr = sequentialIds.data();
    }

    void* devPtr = nullptr;
    HcommResult ret = static_cast<HcommResult>(hrtMalloc(&devPtr, static_cast<uint64_t>(memberNum * sizeof(uint32_t))));
    CHK_PRT_RET(ret != HCOMM_SUCCESS, HCCL_ERROR("[AllocAndCopyWorldTeamIds] hrtMalloc failed, ret[%d]", ret), ret);

    ret = static_cast<HcommResult>(hrtMemSyncCopy(
        devPtr, static_cast<uint64_t>(memberNum * sizeof(uint32_t)), srcPtr,
        static_cast<uint64_t>(memberNum * sizeof(uint32_t)), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocAndCopyWorldTeamIds] hrtMemSyncCopy failed, ret[%d]", ret);
        (void)hrtFree(devPtr);
        return ret;
    }

    entry->devWorldTeamIds = devPtr;
    entry->hostTeam.worldTeamIds = static_cast<uint32_t*>(devPtr);

    entry->hostWorldTeamIds = static_cast<uint32_t*>(malloc(memberNum * sizeof(uint32_t)));
    if (entry->hostWorldTeamIds == nullptr) {
        HCCL_ERROR("[AllocAndCopyWorldTeamIds] malloc hostWorldTeamIds failed");
        (void)hrtFree(devPtr);
        entry->devWorldTeamIds = nullptr;
        entry->hostTeam.worldTeamIds = nullptr;
        return HCOMM_E_MEMORY;
    }
    errno_t memRet
        = memcpy_s(entry->hostWorldTeamIds, memberNum * sizeof(uint32_t), srcPtr, memberNum * sizeof(uint32_t));
    if (memRet != EOK) {
        HCCL_ERROR("[AllocAndCopyWorldTeamIds] memcpy_s hostWorldTeamIds failed, ret[%d]", memRet);
        free(entry->hostWorldTeamIds);
        entry->hostWorldTeamIds = nullptr;
        (void)hrtFree(devPtr);
        entry->devWorldTeamIds = nullptr;
        entry->hostTeam.worldTeamIds = nullptr;
        return HCOMM_E_MEMORY;
    }

    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::AllocChannelEntities(TeamEntry* entry)
{
    uint32_t memberNum = entry->hostTeam.memberNum;
    entry->hostChannelNums.resize(memberNum, 0);
    uint32_t totalChannels = 0;
    for (uint32_t i = 0; i < memberNum; i++) {
        uint32_t chNum = static_cast<uint32_t>(entry->channelsList[i].size());
        entry->hostChannelNums[i] = chNum;
        totalChannels += chNum;
    }

    if (totalChannels == 0) {
        entry->devChannels = nullptr;
        entry->hostTeam.channelsBaseAddr = 0;
        return HCOMM_SUCCESS;
    }

    void* devPtr = nullptr;
    HcommResult ret
        = static_cast<HcommResult>(hrtMalloc(&devPtr, static_cast<uint64_t>(totalChannels * sizeof(ChannelEntity))));
    CHK_PRT_RET(
        ret != HCOMM_SUCCESS,
        HCCL_ERROR("[AllocChannelEntities] hrtMalloc failed, totalChannels[%u] ret[%d]", totalChannels, ret), ret);

    // 逐个 channel 做 D2D 拷贝：把 ChannelHandle 指向的 ChannelEntity 本体拷贝到连续内存对应偏移
    // 偏移按前缀和 channelCntAccumulatePerMember[peer-1] + channelIdx 计算（此处用线性 offset 累加实现）
    uint32_t offset = 0;
    for (uint32_t peer = 0; peer < memberNum; peer++) {
        for (uint32_t idx = 0; idx < entry->channelsList[peer].size(); idx++) {
            void* dst = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(devPtr) + offset * sizeof(ChannelEntity));
            void* src = reinterpret_cast<void*>(static_cast<uintptr_t>(entry->channelsList[peer][idx]));
            ret = static_cast<HcommResult>(hrtMemSyncCopy(
                dst, sizeof(ChannelEntity), src, sizeof(ChannelEntity),
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE));
            if (ret != HCOMM_SUCCESS) {
                HCCL_ERROR(
                    "[AllocChannelEntities] hrtMemSyncCopy D2D failed, peer[%u] idx[%u] ret[%d]", peer, idx, ret);
                (void)hrtFree(devPtr);
                return ret;
            }
            offset++;
        }
    }

    entry->devChannels = devPtr;
    entry->hostTeam.channelsBaseAddr = reinterpret_cast<uint64_t>(devPtr);
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::AllocChannelNumsArray(TeamEntry* entry)
{
    uint32_t memberNum = entry->hostTeam.memberNum;
    // 计算前缀和（起始下标）：channelCntAccumulatePerMember[i] = sum(channelNumPerMember[0..i-1])
    std::vector<uint32_t> accumulateNums(memberNum, 0);
    uint32_t acc = 0;
    for (uint32_t i = 0; i < memberNum; i++) {
        accumulateNums[i] = acc;
        acc += entry->hostChannelNums[i];
    }
    void* devPtr = nullptr;
    HcommResult ret = static_cast<HcommResult>(hrtMalloc(&devPtr, static_cast<uint64_t>(memberNum * sizeof(uint32_t))));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocChannelNumsArray] hrtMalloc channelCntAccumulatePerMember failed, ret[%d]", ret);
        return ret;
    }

    ret = static_cast<HcommResult>(hrtMemSyncCopy(
        devPtr, static_cast<uint64_t>(memberNum * sizeof(uint32_t)), accumulateNums.data(),
        static_cast<uint64_t>(memberNum * sizeof(uint32_t)), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocChannelNumsArray] hrtMemSyncCopy channelCntAccumulatePerMember failed, ret[%d]", ret);
        (void)hrtFree(devPtr);
        return ret;
    }
    entry->devChannelNums = devPtr;
    entry->hostTeam.channelCntAccumulatePerMember = static_cast<uint32_t*>(devPtr);
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::AllocAndCopyChannels(TeamEntry* entry)
{
    uint32_t memberNum = entry->hostTeam.memberNum;
    if (memberNum == 0 || entry->channelsList.empty()) {
        HCCL_ERROR("[AllocAndCopyChannels] memberNum[%u] or channelsList is empty", memberNum);
        return HCOMM_E_PARA;
    }

    /* 两阶段提交：先摘下旧 device 资源，新的分配+拷贝全部成功后统一替换并释放旧资源；
     * 任一步失败只回收本次新分配的部分，entry 的 device 字段保持旧值，避免半途失败
     * 释放旧数组导致 channelsBaseAddr 悬空（UAF）。 */
    void* oldDevChannels = entry->devChannels;
    void* oldDevChannelNums = entry->devChannelNums;
    entry->devChannels = nullptr;
    entry->devChannelNums = nullptr;

    HcommResult ret = AllocChannelEntities(entry);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocAndCopyChannels] AllocChannelEntities failed, ret[%d]", ret);
        entry->devChannels = oldDevChannels;
        entry->devChannelNums = oldDevChannelNums;
        return ret;
    }

    ret = AllocChannelNumsArray(entry);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocAndCopyChannels] AllocChannelNumsArray failed, ret[%d]", ret);
        /* 只回收本次新分配的 channels 数组；nums 数组失败时内部已自清 */
        if (entry->devChannels != nullptr) {
            (void)hrtFree(entry->devChannels);
            entry->devChannels = nullptr;
            entry->hostTeam.channelsBaseAddr = reinterpret_cast<uint64_t>(oldDevChannels);
        }
        entry->devChannelNums = oldDevChannelNums;
        return ret;
    }

    /* 提交：释放旧 device 资源（新资源已写入 entry） */
    if (oldDevChannels != nullptr) {
        (void)hrtFree(oldDevChannels);
    }
    if (oldDevChannelNums != nullptr) {
        (void)hrtFree(oldDevChannelNums);
    }
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::AllocAndCopyRemoteMems(TeamEntry* entry, const CommMem* src, uint32_t memberNum)
{
    if (entry->hostRemoteMems == nullptr) {
        return AllocRemoteMems(entry, src, memberNum);
    }
    return UpdateRemoteMems(entry, src, memberNum);
}

HcommResult HcommTeamMgr::AllocRemoteMems(TeamEntry* entry, const CommMem* src, uint32_t memberNum)
{
    // 首次分配：calloc hostRemoteMems + memcpy 拷入；hrtMalloc devRemoteMems + hrtMemSyncCopy 到 device
    entry->hostRemoteMems = static_cast<CommMem*>(calloc(memberNum, sizeof(CommMem)));
    if (entry->hostRemoteMems == nullptr) {
        HCCL_ERROR("[AllocRemoteMems] calloc hostRemoteMems failed");
        return HCOMM_E_MEMORY;
    }
    errno_t memRet = memcpy_s(entry->hostRemoteMems, memberNum * sizeof(CommMem), src, memberNum * sizeof(CommMem));
    if (memRet != EOK) {
        HCCL_ERROR("[AllocRemoteMems] memcpy_s failed, ret[%d]", memRet);
        free(entry->hostRemoteMems);
        entry->hostRemoteMems = nullptr;
        return HCOMM_E_MEMORY;
    }

    void* devPtr = nullptr;
    HcommResult ret = static_cast<HcommResult>(hrtMalloc(&devPtr, static_cast<uint64_t>(memberNum * sizeof(CommMem))));
    CHK_PRT_RET(ret != HCOMM_SUCCESS, HCCL_ERROR("[AllocRemoteMems] hrtMalloc failed, ret[%d]", ret), ret);

    ret = static_cast<HcommResult>(hrtMemSyncCopy(
        devPtr, static_cast<uint64_t>(memberNum * sizeof(CommMem)), src,
        static_cast<uint64_t>(memberNum * sizeof(CommMem)), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocRemoteMems] hrtMemSyncCopy failed, ret[%d]", ret);
        (void)hrtFree(devPtr);
        return ret;
    }

    entry->devRemoteMems = devPtr;
    entry->hostTeam.syncMem.remoteMems = static_cast<CommMem*>(devPtr);
    entry->hostTeam.syncMem.remoteMemsNum = memberNum;
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::UpdateRemoteMems(TeamEntry* entry, const CommMem* src, uint32_t memberNum)
{
    // 重复 bind：校验维度一致后更新 hostRemoteMems 并重新 sync 到 devRemoteMems
    CHK_PRT_RET(
        memberNum != entry->hostTeam.syncMem.remoteMemsNum,
        HCCL_ERROR(
            "[UpdateRemoteMems] memberNum[%u] != remoteMemNum[%u], dimension mismatch", memberNum,
            entry->hostTeam.syncMem.remoteMemsNum),
        HCOMM_E_PARA);

    errno_t memRet = memcpy_s(entry->hostRemoteMems, memberNum * sizeof(CommMem), src, memberNum * sizeof(CommMem));
    CHK_PRT_RET(memRet != EOK, HCCL_ERROR("[UpdateRemoteMems] memcpy_s failed, ret[%d]", memRet), HCOMM_E_MEMORY);

    HcclResult hrtRet = hrtMemSyncCopy(
        entry->devRemoteMems, static_cast<uint64_t>(memberNum * sizeof(CommMem)), entry->hostRemoteMems,
        static_cast<uint64_t>(memberNum * sizeof(CommMem)), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    HcommResult ret = static_cast<HcommResult>(hrtRet);
    CHK_PRT_RET(ret != HCOMM_SUCCESS, HCCL_ERROR("[UpdateRemoteMems] hrtMemSyncCopy failed, ret[%d]", ret), ret);
    return HCOMM_SUCCESS;
}

void HcommTeamMgr::FreeTeamResources(TeamEntry* entry)
{
    if (entry->devRemoteMems != nullptr) {
        (void)hrtFree(entry->devRemoteMems);
        entry->devRemoteMems = nullptr;
    }
    if (entry->hostRemoteMems != nullptr) {
        free(entry->hostRemoteMems);
        entry->hostRemoteMems = nullptr;
    }

    if (entry->devChannels != nullptr) {
        (void)hrtFree(entry->devChannels);
        entry->devChannels = nullptr;
    }
    if (entry->devChannelNums != nullptr) {
        (void)hrtFree(entry->devChannelNums);
        entry->devChannelNums = nullptr;
    }
    entry->hostTeam.channelsBaseAddr = 0;
    entry->hostTeam.channelCntAccumulatePerMember = nullptr;

    if (entry->devWorldTeamIds != nullptr) {
        (void)hrtFree(entry->devWorldTeamIds);
        entry->devWorldTeamIds = nullptr;
    }
    if (entry->hostWorldTeamIds != nullptr) {
        free(entry->hostWorldTeamIds);
        entry->hostWorldTeamIds = nullptr;
    }

    if (entry->hostTeam.syncMem.shadowMem.addr != nullptr) {
        (void)hrtFree(entry->hostTeam.syncMem.shadowMem.addr);
        entry->hostTeam.syncMem.shadowMem.addr = nullptr;
        entry->hostTeam.syncMem.shadowMem.size = 0;
    }

    if (entry->devTeam != nullptr) {
        (void)hrtFree(entry->devTeam);
        entry->devTeam = nullptr;
    }

    entry->channelsList.clear();
    entry->hostChannelNums.clear();
}

void HcommTeamMgr::FreeWindowResources(WindowEntry* entry)
{
    if (entry->devRemoteAddrs != nullptr) {
        (void)hrtFree(entry->devRemoteAddrs);
        entry->devRemoteAddrs = nullptr;
    }
    if (entry->hostRemoteAddrs != nullptr) {
        free(entry->hostRemoteAddrs);
        entry->hostRemoteAddrs = nullptr;
    }
    if (entry->devWorldTeamAccumulateId != nullptr) {
        (void)hrtFree(entry->devWorldTeamAccumulateId);
        entry->devWorldTeamAccumulateId = nullptr;
    }
    if (entry->hostWindow.netWin.worldTeamAccumulateId != nullptr) {
        free(entry->hostWindow.netWin.worldTeamAccumulateId);
        entry->hostWindow.netWin.worldTeamAccumulateId = nullptr;
    }
    entry->hostWindow.netWin.netLayerNum = 0;
    entry->hostWindow.netWin.baseRemoteMemAddr = 0;
    entry->hostWindow.netWin.windowSize = 0;

    if (entry->ownsDeviceWindow && entry->devWindow != nullptr) {
        (void)hrtFree(entry->devWindow);
        entry->devWindow = nullptr;
    }
}

HcommResult HcommTeamMgr::ValidateSubTeam(TeamEntry* worldEntry, const HcommTeamCreateDesc* desc)
{
    /* worldEntry 为 nullptr 表示创建 world team（无父 team），无需 sub team 成员校验。 */
    CHK_PRT_RET(
        worldEntry == nullptr, HCCL_INFO("[TeamCreate] worldEntry is null, skip sub team validation"), HCOMM_SUCCESS);
    CHK_PRT_RET(
        desc->memberNum > worldEntry->hostTeam.memberNum,
        HCCL_ERROR(
            "[TeamCreate] sub team memberNum[%u] > world team memberNum[%u]", desc->memberNum,
            worldEntry->hostTeam.memberNum),
        HCOMM_E_PARA);

    uint32_t worldMemberNum = worldEntry->hostTeam.memberNum;
    for (uint32_t i = 0; i < desc->memberNum; i++) {
        CHK_PRT_RET(
            desc->worldMemberIds[i] >= worldMemberNum,
            HCCL_ERROR(
                "[TeamCreate] worldMemberIds[%u]=%u >= world team memberNum[%u]", i, desc->worldMemberIds[i],
                worldMemberNum),
            HCOMM_E_PARA);
    }
    return HCOMM_SUCCESS;
}

void HcommTeamMgr::InitTeamEntry(TeamEntry* entry, const HcommTeamCreateDesc* desc, HcommTeamHandle worldTeam)
{
    entry->hostTeam.header.version = HCOMM_TEAM_VERSION;
    entry->hostTeam.header.magicWord = HCOMM_TEAM_MAGIC_WORD;
    entry->hostTeam.header.size = sizeof(HcommTeam);
    entry->hostTeam.header.reserved = 0;
    entry->hostTeam.memberNum = desc->memberNum;
    entry->hostTeam.selfMemberId = desc->selfMemberId;
    entry->hostTeam.netLayer = desc->netLayer;
    entry->hostTeam.engine = desc->engine;
    entry->syncMemReq = desc->requirement;
    entry->syncMemSize
        = static_cast<uint64_t>(
              desc->requirement.signalCount + desc->requirement.counterCount + desc->requirement.barrierCount)
          * sizeof(uint64_t) * desc->memberNum;
    entry->hostTeam.syncMem.syncMemReq = desc->requirement;
    entry->hostTeam.syncMem.syncMemSize = entry->syncMemSize;

    if (worldTeam != nullptr) {
        entry->worldTeamHandle = worldTeam;
        entry->isSubTeam = true;
    }
}

HcommResult HcommTeamMgr::AllocAndSyncTeam(TeamEntry* entry, const HcommTeamCreateDesc* desc)
{
    HcommResult ret = AllocAndCopyWorldTeamIds(entry, desc->worldMemberIds, desc->memberNum);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocAndSyncTeam] AllocAndCopyWorldTeamIds failed");
        FreeTeamResources(entry);
        return ret;
    }

    void* devTeamPtr = nullptr;
    ret = static_cast<HcommResult>(hrtMalloc(&devTeamPtr, static_cast<uint64_t>(sizeof(HcommTeam))));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocAndSyncTeam] hrtMalloc devTeam failed, ret[%d]", ret);
        FreeTeamResources(entry);
        return ret;
    }
    entry->devTeam = static_cast<HcommTeamHandle>(devTeamPtr);

    ret = SyncTeamToDevice(entry);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocAndSyncTeam] SyncTeamToDevice failed, ret[%d]", ret);
        FreeTeamResources(entry);
        return ret;
    }
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::TeamCreate(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* outSyncMemSize)
{
    TeamEntry* worldEntry = nullptr;
    if (worldTeam != nullptr) {
        std::shared_lock<std::shared_mutex> lock(teamsRwMutex_);
        worldEntry = FindTeamByHandleLocked(worldTeam);
        CHK_PRT_RET(
            worldEntry == nullptr, HCCL_ERROR("[TeamCreate] worldTeam handle[%p] not found", worldTeam),
            HCOMM_E_NOT_FOUND);
    }

    HcommResult ret = ValidateSubTeam(worldEntry, desc);
    CHK_PRT_RET(ret != HCOMM_SUCCESS, HCCL_ERROR("[TeamCreate] ValidateSubTeam failed"), ret);

    auto entry = std::make_unique<TeamEntry>();
    InitTeamEntry(entry.get(), desc, worldTeam);

    ret = AllocAndSyncTeam(entry.get(), desc);
    CHK_PRT_RET(ret != HCOMM_SUCCESS, HCCL_ERROR("[TeamCreate] AllocAndSyncTeam failed"), ret);

    *outSyncMemSize = entry->syncMemSize;
    *team = entry->devTeam;

    {
        std::unique_lock<std::shared_mutex> lock(teamsRwMutex_);
        teams_[*team] = std::move(entry);
    }

    HCCL_INFO(
        "[TeamCreate] team created, memberNum[%u], selfMemberId[%u], "
        "netLayer[%u], protocol[%d], "
        "signalCount[%u], counterCount[%u], barrierCount[%u], "
        "syncMemSize[%llu], isSubTeam[%d], devTeam[%p]",
        desc->memberNum, desc->selfMemberId, desc->netLayer, static_cast<int32_t>(desc->protocol),
        desc->requirement.signalCount, desc->requirement.counterCount, desc->requirement.barrierCount, *outSyncMemSize,
        static_cast<int32_t>(worldTeam != nullptr), *team);

    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::TeamDestroy(HcommTeamHandle team)
{
    std::unique_lock<std::shared_mutex> lock(teamsRwMutex_);
    auto it = teams_.find(team);
    CHK_PRT_RET(
        it == teams_.end(), HCCL_WARNING("[TeamDestroy] team handle[%p] not found, maybe already destroyed", team),
        HCOMM_SUCCESS);

    bool isSubTeam = it->second->isSubTeam;
    FreeTeamResources(it->second.get());
    teams_.erase(it);

    HCCL_INFO("[TeamDestroy] team[%p] destroyed, isSubTeam[%d]", team, static_cast<int32_t>(isSubTeam));
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::WindowRegister(void* devLegacySymWin, HcclCommSymWindow* handle)
{
    auto winEntry = std::make_unique<WindowEntry>();
    winEntry->hostWindow.header.version = HCOMM_WINDOW_VERSION;
    winEntry->hostWindow.header.magicWord = HCOMM_WINDOW_MAGIC_WORD;
    winEntry->hostWindow.header.size = sizeof(HcommWindow);
    winEntry->hostWindow.header.reserved = 0;
    winEntry->hostWindow.netWin.baseRemoteMemAddr = 0;
    winEntry->hostWindow.netWin.windowSize = 0;
    winEntry->hostWindow.netWin.worldTeamAccumulateId = nullptr;
    winEntry->hostWindow.netWin.netLayerNum = 0;
    winEntry->hostWindow.lsaWin.baseVa = 0;
    winEntry->hostWindow.lsaWin.stride = 0;
    winEntry->hostWindow.lsaWin.userSize = 0;
    winEntry->hostWindow.legacySymWindow = reinterpret_cast<uint64_t>(devLegacySymWin);
    winEntry->ownsDeviceWindow = true;

    void* devWin = nullptr;
    HcclResult mallocRet = hrtMalloc(&devWin, sizeof(HcommWindow));
    CHK_PRT_RET(
        mallocRet != HCCL_SUCCESS, HCCL_ERROR("[%s] hrtMalloc failed, ret[%d]", __func__, mallocRet), HCOMM_E_MEMORY);
    winEntry->devWindow = devWin;
    HcommResult ret = SyncWindowToDevice(winEntry.get());
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[%s] SyncWindowToDevice failed, ret[%d]", __func__, ret);
        FreeWindowResources(winEntry.get());
        return ret;
    }
    *handle = devWin;
    // hostWindow.netWin 字段首次 UpdateWindowRemoteMemByRank 回填时填充
    // （baseRemoteMemAddr/windowSize/worldTeamAccumulateId/netLayerNum）
    {
        std::unique_lock<std::shared_mutex> lock(windowsRwMutex_);
        CHK_PRT_RET(
            windows_.find(*handle) != windows_.end(),
            HCCL_ERROR("[%s] window[%p] already registered", __func__, *handle), HCOMM_E_PARA);
        windows_[*handle] = std::move(winEntry);
    }
    HCCL_INFO("[%s] window registered, devWindow[%p]", __func__, *handle);
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::WindowDeregister(HcclCommSymWindow handle)
{
    std::unique_lock<std::shared_mutex> lock(windowsRwMutex_);
    auto it = windows_.find(handle);
    CHK_PRT_RET(
        it == windows_.end(), HCCL_WARNING("[%s] window[%p] not found, maybe already unregistered", __func__, handle),
        HCOMM_SUCCESS);
    FreeWindowResources(it->second.get());
    windows_.erase(it);
    HCCL_INFO("[%s] window unregistered, devWindow[%p]", __func__, handle);
    return HCOMM_SUCCESS;
}

/* 清空 window 的 netWin 层分段表资源（失败回滚/重置共用）：释放 host/dev 内存并清零相关字段。 */
void HcommTeamMgr::ClearWindowNetWin(WindowEntry* winEntry)
{
    if (winEntry->devRemoteAddrs != nullptr) {
        (void)hrtFree(winEntry->devRemoteAddrs);
        winEntry->devRemoteAddrs = nullptr;
    }
    if (winEntry->hostRemoteAddrs != nullptr) {
        free(winEntry->hostRemoteAddrs);
        winEntry->hostRemoteAddrs = nullptr;
    }
    if (winEntry->devWorldTeamAccumulateId != nullptr) {
        (void)hrtFree(winEntry->devWorldTeamAccumulateId);
        winEntry->devWorldTeamAccumulateId = nullptr;
    }
    if (winEntry->hostWindow.netWin.worldTeamAccumulateId != nullptr) {
        free(winEntry->hostWindow.netWin.worldTeamAccumulateId);
        winEntry->hostWindow.netWin.worldTeamAccumulateId = nullptr;
    }
    winEntry->hostWindow.netWin.netLayerNum = 0;
    winEntry->hostWindow.netWin.baseRemoteMemAddr = 0;
    winEntry->hostWindow.netWin.windowSize = 0;
    winEntry->remoteMemsTotal = 0;
}

/* 分配 addr 扁平数组（total 个 uint64_t）：calloc host + hrtMalloc dev + 填 baseRemoteMemAddr。
 * 失败时自清。 */
HcommResult HcommTeamMgr::AllocWindowAddrTable(WindowEntry* winEntry, uint32_t total)
{
    winEntry->remoteMemsTotal = total;
    winEntry->hostRemoteAddrs = static_cast<uint64_t*>(calloc(total, sizeof(uint64_t)));
    CHK_PRT_RET(
        winEntry->hostRemoteAddrs == nullptr, HCCL_ERROR("[AllocWindowAddrTable] calloc hostRemoteAddrs failed"),
        HCOMM_E_MEMORY);
    void* devAddrsPtr = nullptr;
    HcommResult ret
        = static_cast<HcommResult>(hrtMalloc(&devAddrsPtr, static_cast<uint64_t>(total * sizeof(uint64_t))));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocWindowAddrTable] hrtMalloc devRemoteAddrs failed, ret[%d]", ret);
        ClearWindowNetWin(winEntry);
        return ret;
    }
    winEntry->devRemoteAddrs = devAddrsPtr;
    winEntry->hostWindow.netWin.baseRemoteMemAddr = reinterpret_cast<uint64_t>(winEntry->devRemoteAddrs);
    return HCOMM_SUCCESS;
}

/* 分配 worldTeamAccumulateId（长度 sizeNum）：calloc host + 填前缀和 + hrtMalloc dev 副本 + H2D 同步。
 * 失败时自清（连同已分配的 addr/size 表一并释放，由 ClearWindowNetWin 统一处理）。 */
HcommResult HcommTeamMgr::AllocWindowAccumulateIdArray(WindowEntry* winEntry, const uint32_t* sizes, uint32_t sizeNum)
{
    winEntry->hostWindow.netWin.worldTeamAccumulateId = static_cast<uint32_t*>(calloc(sizeNum, sizeof(uint32_t)));
    if (winEntry->hostWindow.netWin.worldTeamAccumulateId == nullptr) {
        HCCL_ERROR("[AllocWindowAccumulateIdArray] calloc worldTeamAccumulateId failed");
        ClearWindowNetWin(winEntry);
        return HCOMM_E_PTR;
    }
    // 填前缀和（起始下标）：accumulateId[i] = sum(sizes[0..i-1])
    uint32_t acc = 0;
    for (uint32_t i = 0; i < sizeNum; i++) {
        winEntry->hostWindow.netWin.worldTeamAccumulateId[i] = acc;
        acc += sizes[i];
    }
    winEntry->hostWindow.netWin.netLayerNum = sizeNum;
    void* devAccumIdPtr = nullptr;
    HcommResult ret
        = static_cast<HcommResult>(hrtMalloc(&devAccumIdPtr, static_cast<uint64_t>(sizeNum * sizeof(uint32_t))));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocWindowAccumulateIdArray] hrtMalloc devWorldTeamAccumulateId failed, ret[%d]", ret);
        ClearWindowNetWin(winEntry);
        return ret;
    }
    winEntry->devWorldTeamAccumulateId = static_cast<uint32_t*>(devAccumIdPtr);
    ret = static_cast<HcommResult>(hrtMemSyncCopy(
        winEntry->devWorldTeamAccumulateId, static_cast<uint64_t>(sizeNum * sizeof(uint32_t)),
        winEntry->hostWindow.netWin.worldTeamAccumulateId, static_cast<uint64_t>(sizeNum * sizeof(uint32_t)),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    CHK_PRT_RET(
        ret != HCOMM_SUCCESS,
        HCCL_ERROR("[AllocWindowAccumulateIdArray] hrtMemSyncCopy devWorldTeamAccumulateId failed, ret[%d]", ret), ret);
    return HCOMM_SUCCESS;
}

/* 首次回填时分配 netWin 层分段表：addr 扁平数组 + worldTeamAccumulateId 副本 + 整体 SyncWindowToDevice。
 * 任一步失败时全部已分配资源清零。 */
HcommResult HcommTeamMgr::AllocWindowNetWin(WindowEntry* winEntry, const uint32_t* sizes, uint32_t sizeNum)
{
    uint32_t total = 0;
    for (uint32_t i = 0; i < sizeNum; i++) {
        total += sizes[i];
    }
    CHK_PRT_RET(total == 0, HCCL_ERROR("[AllocWindowNetWin] no prebuilt worldTeam sizes available"), HCOMM_E_PARA);

    HcommResult ret = AllocWindowAddrTable(winEntry, total);
    CHK_PRT_RET(ret != HCOMM_SUCCESS, HCCL_ERROR("[AllocWindowNetWin] AllocWindowAddrTable failed"), ret);
    ret = AllocWindowAccumulateIdArray(winEntry, sizes, sizeNum);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[AllocWindowNetWin] AllocWindowAccumulateIdArray failed, ret[%d]", ret);
        ClearWindowNetWin(winEntry);
        return ret;
    }

    // 首次分配后整体同步，device 侧才能看到 netWin.baseRemoteMemAddr/windowSize/worldTeamAccumulateId
    ret = SyncWindowToDevice(winEntry);
    CHK_PRT_RET(ret != HCOMM_SUCCESS, HCCL_ERROR("[AllocWindowNetWin] SyncWindowToDevice failed, ret[%d]", ret), ret);
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::ValidateUbSymmetricWindowLayout(
    const CommMem* memberMems, uint32_t memberNum, const void* baseVa, size_t stride, size_t userSize) const
{
    uintptr_t base = reinterpret_cast<uintptr_t>(baseVa);
    uint64_t lastMember = static_cast<uint64_t>(memberNum - 1);
    CHK_PRT_RET(
        userSize > stride || lastMember > (UINTPTR_MAX - base) / stride,
        HCCL_ERROR("[%s] symmetric VA layout overflows", __func__), HCOMM_E_PARA);
    for (uint32_t member = 0; member < memberNum; ++member) {
        void* expected = reinterpret_cast<void*>(base + static_cast<uint64_t>(member) * stride);
        CHK_PRT_RET(
            memberMems[member].addr != expected || memberMems[member].size != userSize
                || memberMems[member].type != COMM_MEM_TYPE_DEVICE,
            HCCL_ERROR(
                "[%s] member[%u] memory does not match fixed VA slot, addr[%p], expected[%p], size[%llu]", __func__,
                member, memberMems[member].addr, expected, memberMems[member].size),
            HCOMM_E_PARA);
    }
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::BindUbSymmetricWindow(
    HcclCommSymWindow handle, HcommTeamHandle lsaTeam, uint32_t netLayer, const CommMem* memberMems, uint32_t memberNum,
    void* baseVa, size_t stride, size_t userSize)
{
    CHK_PRT_RET(
        handle == nullptr || lsaTeam == nullptr || memberMems == nullptr || baseVa == nullptr,
        HCCL_ERROR("[%s] null parameter", __func__), HCOMM_E_PTR);
    CHK_PRT_RET(
        memberNum == 0 || stride == 0 || userSize == 0,
        HCCL_ERROR(
            "[%s] invalid parameter, layer[%u], memberNum[%u], stride[%zu], userSize[%zu]", __func__, netLayer,
            memberNum, stride, userSize),
        HCOMM_E_PARA);
    HcommResult validateRet = ValidateUbSymmetricWindowLayout(memberMems, memberNum, baseVa, stride, userSize);
    CHK_PRT_RET(
        validateRet != HCOMM_SUCCESS, HCCL_ERROR("[%s] invalid UB symmetric window layout", __func__), validateRet);

    {
        std::shared_lock<std::shared_mutex> lock(teamsRwMutex_);
        TeamEntry* teamEntry = FindTeamByHandleLocked(lsaTeam);
        CHK_PRT_RET(
            teamEntry == nullptr, HCCL_ERROR("[%s] LSA team[%p] not found", __func__, lsaTeam), HCOMM_E_NOT_FOUND);
        CHK_PRT_RET(
            teamEntry->isSubTeam, HCCL_ERROR("[%s] LSA team[%p] is not a world team", __func__, lsaTeam), HCOMM_E_PARA);
        CHK_PRT_RET(
            teamEntry->hostTeam.netLayer != netLayer,
            HCCL_ERROR(
                "[%s] LSA team[%p] netLayer[%u] does not match window netLayer[%u]", __func__, lsaTeam,
                teamEntry->hostTeam.netLayer, netLayer),
            HCOMM_E_PARA);
        CHK_PRT_RET(
            teamEntry->hostTeam.memberNum != memberNum,
            HCCL_ERROR(
                "[%s] LSA team[%p] memberNum[%u] does not match window memberNum[%u]", __func__, lsaTeam,
                teamEntry->hostTeam.memberNum, memberNum),
            HCOMM_E_PARA);
    }

    std::unique_lock<std::shared_mutex> lock(windowsRwMutex_);
    WindowEntry* winEntry = FindWindowByHandleLocked(handle);
    CHK_PRT_RET(winEntry == nullptr, HCCL_ERROR("[%s] window[%p] not found", __func__, handle), HCOMM_E_NOT_FOUND);
    CHK_PRT_RET(
        winEntry->hostWindow.lsaWin.baseVa != 0 || winEntry->hostWindow.lsaWin.stride != 0
            || winEntry->hostWindow.lsaWin.userSize != 0,
        HCCL_ERROR("[%s] window[%p] has already bound UB Memory", __func__, handle), HCOMM_E_PARA);

    // netWin和legacySymWindow由URMA路径维护；UB Memory仅补充同一个HcommWindow中的lsaWin。
    winEntry->hostWindow.lsaWin.baseVa = reinterpret_cast<uint64_t>(baseVa);
    winEntry->hostWindow.lsaWin.stride = static_cast<uint64_t>(stride);
    winEntry->hostWindow.lsaWin.userSize = static_cast<uint64_t>(userSize);
    HcommResult ret = SyncWindowToDevice(winEntry);
    if (ret != HCOMM_SUCCESS) {
        winEntry->hostWindow.lsaWin.baseVa = 0;
        winEntry->hostWindow.lsaWin.stride = 0;
        winEntry->hostWindow.lsaWin.userSize = 0;
    }
    return ret;
}

/* 合并语义：selfVa/selfSize 与 selfSlots 各自可选（传空表示保持原值不变）——
 * RegisterWindow 先登记 VA/size（槽位未知），UpdateHcommWindowRemoteMem 首次调用时补齐槽位。
 * 幂等：重复登记相同值无害。 */
HcommResult HcommTeamMgr::SetWindowSelfInfo(
    HcclCommSymWindow handle, void* selfVa, uint64_t selfSize, const uint32_t* selfSlots, uint32_t selfSlotNum)
{
    CHK_PRT_RET(handle == nullptr, HCCL_ERROR("[SetWindowSelfInfo] window handle is nullptr"), HCOMM_E_PTR);
    CHK_PRT_RET(
        (selfVa == nullptr || selfSize == 0) && (selfSlots == nullptr || selfSlotNum == 0),
        HCCL_ERROR(
            "[SetWindowSelfInfo] nothing to set, handle[%p] selfVa[%p] selfSize[%llu] selfSlotNum[%u]", handle, selfVa,
            selfSize, selfSlotNum),
        HCOMM_E_PARA);
    std::unique_lock<std::shared_mutex> lock(windowsRwMutex_);
    WindowEntry* winEntry = FindWindowByHandleLocked(handle);
    CHK_PRT_RET(
        winEntry == nullptr, HCCL_ERROR("[SetWindowSelfInfo] window handle[%p] not found", handle), HCOMM_E_NOT_FOUND);
    if (selfVa != nullptr && selfSize != 0) {
        winEntry->selfVa = selfVa;
        winEntry->selfSize = selfSize;
    }
    if (selfSlots != nullptr && selfSlotNum != 0) {
        winEntry->selfSlots.assign(selfSlots, selfSlots + selfSlotNum);
    }
    HCCL_INFO(
        "[SetWindowSelfInfo] handle[%p] selfVa[%p] selfSize[%llu] selfSlotNum[%u]", handle, selfVa, selfSize,
        selfSlotNum);
    return HCOMM_SUCCESS;
}

/* 用窗口自身注册信息回填本端（self rank）层槽位（幂等：重复写相同值无害，无分配）。
 * 未登记本端注册信息（selfVa 为空）时跳过并告警，不影响远端槽位回填。 */
HcommResult HcommTeamMgr::UpdateWindowSelfSlots(WindowEntry* winEntry)
{
    if (winEntry->selfVa == nullptr || winEntry->selfSize == 0) {
        HCCL_WARNING("[UpdateWindowSelfSlots] handle[%p] no self register info, skip self slots", winEntry->devWindow);
        return HCOMM_SUCCESS;
    }
    const std::vector<uint32_t>& selfSlots = winEntry->selfSlots;
    if (selfSlots.empty()) {
        HCCL_WARNING("[UpdateWindowSelfSlots] handle[%p] has no layer slots, skip self slots", winEntry->devWindow);
        return HCOMM_SUCCESS;
    }
    uint64_t selfAddr = reinterpret_cast<uint64_t>(winEntry->selfVa);
    uint64_t selfSize = winEntry->selfSize;
    // windowSize 为标量（所有 remoteMem size 相同），首次回填时写入
    winEntry->hostWindow.netWin.windowSize = selfSize;
    for (uint32_t slot : selfSlots) {
        CHK_PRT_RET(
            slot >= winEntry->remoteMemsTotal,
            HCCL_ERROR("[UpdateWindowSelfSlots] slot[%u] >= total[%u]", slot, winEntry->remoteMemsTotal), HCOMM_E_PARA);
        winEntry->hostRemoteAddrs[slot] = selfAddr;
        HcommResult ret = static_cast<HcommResult>(hrtMemSyncCopy(
            static_cast<void*>(reinterpret_cast<uint8_t*>(winEntry->devRemoteAddrs) + slot * sizeof(uint64_t)),
            sizeof(uint64_t), &winEntry->hostRemoteAddrs[slot], sizeof(uint64_t),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        CHK_PRT_RET(
            ret != HCOMM_SUCCESS,
            HCCL_ERROR("[UpdateWindowSelfSlots] hrtMemSyncCopy addr failed, slot[%u] ret[%d]", slot, ret), ret);
        HCCL_INFO(
            "[UpdateWindowSelfSlots] slot[%u] filled with self mem, addr[%llu] size[%llu]", slot, selfAddr, selfSize);
    }
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::UpdateWindowRemoteMemByRank(
    HcclCommSymWindow handle, const uint32_t* sizes, uint32_t sizeNum, const uint32_t* slots, uint32_t slotNum,
    const CommMem& remoteMem)
{
    std::unique_lock<std::shared_mutex> lock(windowsRwMutex_);
    WindowEntry* winEntry = FindWindowByHandleLocked(handle);
    CHK_PRT_RET(
        winEntry == nullptr, HCCL_ERROR("[UpdateWindowRemoteMemByRank] window handle[%p] not found", handle),
        HCOMM_E_NOT_FOUND);

    // 首次调用：分配层分段表（不含 UB_MEM）
    if (winEntry->hostRemoteAddrs == nullptr || winEntry->devRemoteAddrs == nullptr) {
        HcommResult allocRet = AllocWindowNetWin(winEntry, sizes, sizeNum);
        CHK_PRT_RET(
            allocRet != HCOMM_SUCCESS,
            HCCL_ERROR("[UpdateWindowRemoteMemByRank] AllocWindowNetWin failed, ret[%d]", allocRet),
            static_cast<HcclResult>(allocRet));
    }

    // 回填本端（self rank）槽位（幂等：重复调用重复写相同值无害，无额外分配）
    HcommResult selfRet = UpdateWindowSelfSlots(winEntry);
    CHK_PRT_RET(
        selfRet != HCOMM_SUCCESS,
        HCCL_ERROR("[UpdateWindowRemoteMemByRank] UpdateWindowSelfSlots failed, ret[%d]", selfRet), selfRet);

    // 对每个层槽位回填（slots 已由调用方算好：sum(sizes[0..L-1]) + worldTeamId）
    uint64_t remoteAddr = reinterpret_cast<uint64_t>(remoteMem.addr);
    // windowSize 为标量（所有 remoteMem size 相同），首次回填时写入
    if (winEntry->hostWindow.netWin.windowSize == 0) {
        winEntry->hostWindow.netWin.windowSize = remoteMem.size;
        HcommResult syncRet = SyncWindowToDevice(winEntry);
        CHK_PRT_RET(
            syncRet != HCOMM_SUCCESS,
            HCCL_ERROR("[UpdateWindowRemoteMemByRank] SyncWindowToDevice for windowSize failed, ret[%d]", syncRet),
            syncRet);
    }
    for (uint32_t s = 0; s < slotNum; s++) {
        uint32_t slot = slots[s];
        CHK_PRT_RET(
            slot >= winEntry->remoteMemsTotal,
            HCCL_ERROR("[UpdateWindowRemoteMemByRank] slot[%u] >= total[%u]", slot, winEntry->remoteMemsTotal),
            HCOMM_E_PARA);
        winEntry->hostRemoteAddrs[slot] = remoteAddr;
        HcommResult ret = static_cast<HcommResult>(hrtMemSyncCopy(
            static_cast<void*>(reinterpret_cast<uint8_t*>(winEntry->devRemoteAddrs) + slot * sizeof(uint64_t)),
            sizeof(uint64_t), &winEntry->hostRemoteAddrs[slot], sizeof(uint64_t),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        CHK_PRT_RET(
            ret != HCOMM_SUCCESS,
            HCCL_ERROR("[UpdateWindowRemoteMemByRank] hrtMemSyncCopy addr failed, slot[%u] ret[%d]", slot, ret), ret);
    }
    return HCOMM_SUCCESS;
}

void HcommTeamMgr::MergeChannelLists(
    TeamEntry* entry, const HcommTeamBindChannelsDesc* desc, std::vector<std::vector<uint64_t>>& newChannels)
{
    if (entry->channelsList.empty()) {
        newChannels.resize(entry->hostTeam.memberNum);
    } else {
        newChannels = entry->channelsList;
    }

    for (uint32_t i = 0; i < entry->hostTeam.memberNum; i++) {
        uint64_t chNum = desc->channelNumPerMember[i];
        uint64_t* chSrc = desc->channelsByMemberId[i];
        for (uint64_t j = 0; j < chNum; j++) {
            newChannels[i].push_back(chSrc[j]);
        }
    }
}

HcommResult HcommTeamMgr::BindChannels(HcommTeamHandle team, const HcommTeamBindChannelsDesc* desc)
{
    std::shared_lock<std::shared_mutex> lock(teamsRwMutex_);
    TeamEntry* entry = FindTeamByHandleLocked(team);
    CHK_PRT_RET(entry == nullptr, HCCL_ERROR("[BindChannels] team handle[%p] not found", team), HCOMM_E_NOT_FOUND);

    CHK_PRT_RET(
        desc->memberNum != entry->hostTeam.memberNum,
        HCCL_ERROR("[BindChannels] memberNum[%u] != team memberNum[%u]", desc->memberNum, entry->hostTeam.memberNum),
        HCOMM_E_PARA);

    for (uint32_t i = 0; i < entry->hostTeam.memberNum; i++) {
        CHK_PRT_RET(
            desc->channelNumPerMember[i] != 0 && desc->channelsByMemberId[i] == nullptr,
            HCCL_ERROR(
                "[BindChannels] member[%u] channelNum[%u] > 0 but channels is null", i, desc->channelNumPerMember[i]),
            HCOMM_E_PARA);
    }

    // 二维数组，需要填入HcommTeam中的channels字段
    std::vector<std::vector<uint64_t>> newChannels;
    MergeChannelLists(entry, desc, newChannels);

    std::vector<std::vector<uint64_t>> oldChannels = std::move(entry->channelsList);
    entry->channelsList = std::move(newChannels);

    HcommResult ret = AllocAndCopyChannels(entry);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[BindChannels] AllocAndCopyChannels failed, this bind does not take effect, "
                   "previously bound channels are not affected");
        entry->channelsList = std::move(oldChannels);
        return ret;
    }

    ret = SyncTeamToDevice(entry);
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[BindChannels] SyncTeamToDevice failed");
        return ret;
    }

    HCCL_INFO(
        "[BindChannels] channels appended and synced, team[%p], channelNumPerMember[%p], "
        "channelsByMemberId[%p], memberNum[%u], isSubTeam[%d]",
        team, desc->channelNumPerMember, desc->channelsByMemberId, entry->hostTeam.memberNum, entry->isSubTeam);
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::BindSyncMem(HcommTeamHandle team, const HcommTeamBindSyncMemDesc* desc)
{
    std::shared_lock<std::shared_mutex> lock(teamsRwMutex_);
    TeamEntry* entry = FindTeamByHandleLocked(team);
    CHK_PRT_RET(entry == nullptr, HCCL_ERROR("[BindSyncMem] team handle[%p] not found", team), HCOMM_E_NOT_FOUND);

    CHK_PRT_RET(
        desc->remoteMemNum != entry->hostTeam.memberNum,
        HCCL_ERROR("[BindSyncMem] remoteMemNum[%u] != memberNum[%u]", desc->remoteMemNum, entry->hostTeam.memberNum),
        HCOMM_E_PARA);

    HcommResult ret = AllocAndCopyRemoteMems(entry, desc->remoteMems, desc->remoteMemNum);
    CHK_PRT_RET(ret != HCOMM_SUCCESS, HCCL_ERROR("[BindSyncMem] AllocAndCopyRemoteMems failed"), ret);

    if (entry->hostTeam.syncMem.shadowMem.addr != nullptr) {
        (void)hrtFree(entry->hostTeam.syncMem.shadowMem.addr);
        entry->hostTeam.syncMem.shadowMem.addr = nullptr;
        entry->hostTeam.syncMem.shadowMem.size = 0;
    }
    void* devShadowMemPtr = nullptr;
    uint64_t shadowMemSize = entry->hostTeam.syncMem.syncMemSize;
    ret = static_cast<HcommResult>(hrtMalloc(&devShadowMemPtr, shadowMemSize));
    if (ret != HCOMM_SUCCESS) {
        HCCL_ERROR("[BindSyncMem] hrtMalloc shadowMem failed, size[%llu] ret[%d]", shadowMemSize, ret);
        return ret;
    }
    entry->hostTeam.syncMem.shadowMem.type = COMM_MEM_TYPE_DEVICE;
    entry->hostTeam.syncMem.shadowMem.addr = devShadowMemPtr;
    entry->hostTeam.syncMem.shadowMem.size = shadowMemSize;

    ret = SyncTeamToDevice(entry);
    CHK_PRT_RET(ret != HCOMM_SUCCESS, HCCL_ERROR("[BindSyncMem] SyncTeamToDevice failed"), ret);

    HCCL_INFO("[BindSyncMem] syncMem bound, remoteMems[%p], remoteMemNum[%u]", desc->remoteMems, desc->remoteMemNum);
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::GetNetLayer(HcommTeamHandle team, uint32_t* netLayer)
{
    std::shared_lock<std::shared_mutex> lock(teamsRwMutex_);
    TeamEntry* entry = FindTeamByHandleLocked(team);
    CHK_PRT_RET(entry == nullptr, HCCL_ERROR("[GetNetLayer] team handle[%p] not found", team), HCOMM_E_NOT_FOUND);

    *netLayer = entry->hostTeam.netLayer;
    return HCOMM_SUCCESS;
}

HcommResult HcommTeamMgr::GetEngine(HcommTeamHandle team, CommEngine* engine)
{
    std::shared_lock<std::shared_mutex> lock(teamsRwMutex_);
    TeamEntry* entry = FindTeamByHandleLocked(team);
    CHK_PRT_RET(entry == nullptr, HCCL_ERROR("[GetEngine] team handle[%p] not found", team), HCOMM_E_NOT_FOUND);

    *engine = entry->hostTeam.engine;
    return HCOMM_SUCCESS;
}

} // namespace hcomm
