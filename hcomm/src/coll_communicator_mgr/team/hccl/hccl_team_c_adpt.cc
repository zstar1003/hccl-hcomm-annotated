/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_team_c_adpt.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "adapter_rts_common.h"
#include "hccl_comm_pub.h"
#include "hccl_team.h"
#include "hccl_team_mgr.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "hcomm_team.h"
#include "hcomm_team_c_adpt.h"
#include "hcomm_team_mgr.h"
#include "symmetric_memory.h"

using namespace hccl;

/**
 * @note 职责：集合通信的通信域HcclTeam管理的C接口的C到C++适配
 */

/* HcclTeamCreate入参校验 */
static HcclResult CheckTeamCreateParam(HcclComm comm, const HcclTeamCreateDesc* desc, HcommTeamHandle* team)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(desc);
    CHK_PTR_NULL(team);
    CHK_PRT_RET(
        (desc->rankNum == 0 || desc->rankNum == 1), HCCL_ERROR("[%s] team can not be empty or single rank", __func__),
        HCCL_E_PARA);
    CHK_PRT_RET(desc->rankIds == nullptr, HCCL_ERROR("[%s] rankIds is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(
        (desc->requirement.signalCount != 0 || desc->requirement.counterCount != 0),
        HCCL_ERROR(
            "[%s] signalCount[%u] and counterCount[%u] must be 0", __func__, desc->requirement.signalCount,
            desc->requirement.counterCount),
        HCCL_E_PARA);
    CHK_PRT_RET(
        desc->requirement.barrierCount == 0, HCCL_ERROR("[%s] barrierCount must be >= 1", __func__), HCCL_E_PARA);
    CHK_PRT_RET(desc->channelCnt == 0, HCCL_ERROR("[%s] channelCnt is 0", __func__), HCCL_E_PARA);
    CHK_PRT_RET(desc->engine != COMM_ENGINE_AIV, HCCL_ERROR("[%s] only support AIV engine", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        desc->protocol != COMM_PROTOCOL_UB_CTP && desc->protocol != COMM_PROTOCOL_UBC_TP
            && desc->protocol != COMM_PROTOCOL_UBOE && desc->protocol != COMM_PROTOCOL_UB_RTP,
        HCCL_ERROR("[%s] only support URMA protocol", __func__), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

/* 在 rankIds 中查找 selfRankId 的下标作为 memberId。
 * 例如 rankIds=[1,3,5,7]，selfRankId=3，则 selfMemberId=1。未找到返回 HCCL_E_NOT_FOUND。 */
static HcclResult FindSelfMemberId(const HcclTeamCreateDesc* desc, uint32_t& selfMemberId)
{
    for (uint32_t i = 0; i < desc->rankNum; i++) {
        if (desc->rankIds[i] == desc->selfRankId) {
            selfMemberId = i;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("[FindSelfMemberId] selfRankId[%u] not found in rankIds", desc->selfRankId);
    return HCCL_E_NOT_FOUND;
}

static HcclResult GetTeamQueryCollComm(HcclComm comm, CollComm*& collComm)
{
    CHK_PTR_NULL(comm);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_PRT_RET(
        !hcclComm->IsCommunicatorV2(), HCCL_ERROR("[%s] only supports communicator V2", __func__), HCCL_E_NOT_SUPPORT);
    collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    return HCCL_SUCCESS;
}

/* 从 HcclTeamCreateDesc 填充 HcommTeamCreateDesc。
 * worldMemberIds：worldTeam 传 nullptr（L3 生成 [0,memberNum)）；subTeam 传 worldTeam 的 memberId 列表。 */
static void FillHcommTeamCreateDesc(
    const HcclTeamCreateDesc* desc, uint32_t selfMemberId, const uint32_t* worldMemberIds,
    HcommTeamCreateDesc& hcommDesc)
{
    /* 用头文件 Init 初始化 ABI 头部（magicWord/version/size）与默认字段，再覆盖业务字段。 */
    (void)HcommTeamCreateDescInit(&hcommDesc);
    hcommDesc.memberNum = desc->rankNum;
    hcommDesc.selfMemberId = selfMemberId;
    hcommDesc.worldMemberIds = worldMemberIds;
    hcommDesc.requirement.signalCount = 0;  // 不支持配置，强制为0
    hcommDesc.requirement.counterCount = 0; // 不支持配置，强制为0
    hcommDesc.requirement.barrierCount = desc->requirement.barrierCount;
    hcommDesc.netLayer = desc->netLayer;
    hcommDesc.protocol = desc->protocol;
    hcommDesc.engine = desc->engine;
}

/* 反查 subTeam.rankIds 在 worldTeam.rankIds 中的下标作为 worldMemberIds（worldTeam 的 memberId）。
 * 用 unordered_map 预建 rankId→worldMemberId 反查表，O(N+M) 避免双重循环。 */
static HcclResult BuildSubTeamWorldMemberIds(
    HcommTeamHandle worldTeam, const uint32_t* subRankIds, uint32_t subRankNum, std::vector<uint32_t>& worldMemberIds)
{
    std::vector<uint32_t> worldRankIds = HcclTeamMgr::GetInstance().GetRankIds(worldTeam);
    CHK_PRT_RET(
        worldRankIds.empty(), HCCL_ERROR("[%s] GetRankIds failed, worldTeam[%p] not registered", __func__, worldTeam),
        HCCL_E_PARA);
    std::unordered_map<uint32_t, uint32_t> rankToMember;
    rankToMember.reserve(worldRankIds.size());
    for (uint32_t m = 0; m < worldRankIds.size(); m++) {
        rankToMember[worldRankIds[m]] = m;
    }
    worldMemberIds.resize(subRankNum);
    for (uint32_t i = 0; i < subRankNum; i++) {
        auto it = rankToMember.find(subRankIds[i]);
        CHK_PRT_RET(
            it == rankToMember.end(),
            HCCL_ERROR("[%s] subTeam rankId[%u] not in worldTeam rankIds", __func__, subRankIds[i]), HCCL_E_PARA);
        worldMemberIds[i] = it->second;
    }
    return HCCL_SUCCESS;
}

/* 创建 subTeam 并申请 syncMem 内存。 */
static HcclResult CreateSubTeamWithSyncMem(
    HcommTeamHandle worldTeam, const HcclTeamCreateDesc* desc, const HcommTeamCreateDesc& hcommDesc,
    HcommTeamHandle* team)
{
    uint64_t syncMemSize = 0;
    HcommResult ret = HcommTeamCreate(worldTeam, &hcommDesc, team, &syncMemSize);
    CHK_PRT_RET(
        ret != 0 || *team == nullptr,
        HCCL_ERROR(
            "[%s] HcommTeamCreate failed, ret[%d] rankNum[%u] selfRankId[%u]", __func__, ret, desc->rankNum,
            desc->selfRankId),
        HCCL_E_INTERNAL);
    if (syncMemSize == 0) {
        HCCL_ERROR("[%s] syncMemSize is 0", __func__);
        (void)HcommTeamDestroy(*team);
        *team = nullptr;
        return HCCL_E_PARA;
    }

    /* 申请 syncMem 本地内存。失败时回滚已创建的 sub team。 */
    void* syncMemPtr = nullptr;
    HcclResult mallocRet = hrtMalloc(&syncMemPtr, syncMemSize);
    if (mallocRet != HCCL_SUCCESS || syncMemPtr == nullptr) {
        HCCL_ERROR("[%s] hrtMalloc failed, ret[%d] size[%llu]", __func__, mallocRet, syncMemSize);
        (void)HcommTeamDestroy(*team);
        *team = nullptr;
        return HCCL_E_MEMORY;
    }

    /* 注册 sub team 到 HcclTeamMgr。 */
    HcclResult regRet = HcclTeamMgr::GetInstance().RegisterSubTeam(
        worldTeam, *team, syncMemPtr, syncMemSize, desc->rankIds, desc->rankNum);
    if (regRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] RegisterSubTeam failed, ret[%d]", __func__, regRet);
        (void)hrtFree(syncMemPtr);
        (void)HcommTeamDestroy(*team);
        *team = nullptr;
        return regRet;
    }
    return HCCL_SUCCESS;
}

/* 注册 team 的 syncMem 内存（team 粒度，仅首次注册一次）。
 * syncMemHandle 已存在则跳过注册；syncMemHandle 出参返回当前句柄（首次或已存在），供调用方日志使用。 */
static HcclResult RegisterTeamSyncMem(HcommTeamHandle team, CollComm* collComm)
{
    HcclMemHandle syncMemHandle = HcclTeamMgr::GetInstance().GetTeamSyncMemHandle(team);
    const std::string commId = collComm->GetCommId();
    if (syncMemHandle != nullptr) {
        HCCL_INFO(
            "[RegisterTeamSyncMem] syncMemHandle[%p] already registered, comm[%s] team[%p]", syncMemHandle,
            commId.c_str(), team);
        return HCCL_SUCCESS;
    }
    void* syncMemPtr = HcclTeamMgr::GetInstance().GetSyncMemPtr(team);
    uint64_t syncMemSize = HcclTeamMgr::GetInstance().GetSyncMemSize(team);
    if (syncMemPtr == nullptr || syncMemSize == 0) {
        HCCL_ERROR(
            "[RegisterTeamSyncMem] syncMemPtr[%p] or syncMemSize[%llu] is invalid, comm[%s] team[%p]", syncMemPtr,
            syncMemSize, commId.c_str(), team);
        return HCCL_E_INTERNAL;
    }
    CommMem syncMemVar{};
    syncMemVar.type = COMM_MEM_TYPE_DEVICE;
    syncMemVar.addr = syncMemPtr;
    syncMemVar.size = syncMemSize;
    std::ostringstream syncMemTagStream;
    syncMemTagStream << HCCL_TEAM_SYNCMEM_TAG_PREFIX << commId << "_team_" << team << "_addr_" << syncMemPtr << "_size_"
                     << syncMemSize;
    const std::string syncMemTagStr = syncMemTagStream.str();
    auto myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);
    CommMems* commMem = myRank->GetCommMems();
    CHK_PTR_NULL(commMem);
    HcclResult ret = commMem->CommRegMem(syncMemTagStr, syncMemVar, &syncMemHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[RegisterTeamSyncMem] register syncMem failed, comm[%s] ret[%d] size[%llu]", commId.c_str(), ret,
            syncMemSize),
        ret);
    HcclTeamMgr::GetInstance().SetTeamSyncMemHandle(team, syncMemHandle, syncMemTagStr);
    return HCCL_SUCCESS;
}

/* 查询本rank到peerRank的link，填充 HcclChannelDesc 的 endpoint/protocol 字段。*/
static HcclResult
FillChannelDescForPeer(HcclComm comm, HcommTeamHandle team, uint32_t selfRank, uint32_t peerRank, HcclChannelDesc& desc)
{
    uint32_t netLayer = 0;
    HcommResult hRet = HcommTeamGetNetLayer(team, &netLayer);
    CHK_PRT_RET(hRet != 0, HCCL_ERROR("[%s] HcommTeamGetNetLayer failed, ret[%d]", __func__, hRet), HCCL_E_INTERNAL);

    CommLink* links = nullptr;
    uint32_t linkNum = 0;
    HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, selfRank, peerRank, &links, &linkNum);
    if (ret != HCCL_SUCCESS || links == nullptr || linkNum == 0) {
        HCCL_ERROR("[%s] no link found from rank[%u] to rank[%u]", __func__, selfRank, peerRank);
        return HCCL_E_NOT_FOUND;
    }
    const CommLink& link = links[0];
    desc.remoteRank = peerRank;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint = link.srcEndpointDesc;
    desc.remoteEndpoint = link.dstEndpointDesc;
    return HCCL_SUCCESS;
}

namespace hccl {
/* 从 HcclTeamMgr 取 team 的 rankIds（memberId→rankId），推算 memberNum 与 selfMemberId。 */
HcclResult GetTeamMemberInfo(HcommTeamHandle team, uint32_t selfRank, ChannelsCreateCtx& ctx)
{
    ctx.rankIds = HcclTeamMgr::GetInstance().GetRankIds(team);
    CHK_PRT_RET(
        ctx.rankIds.empty(), HCCL_ERROR("[%s] GetRankIds failed, team[%p] not registered", __func__, team),
        HCCL_E_PARA);
    ctx.memberNum = static_cast<uint32_t>(ctx.rankIds.size());
    bool selfFound = false;
    for (uint32_t m = 0; m < ctx.memberNum; m++) {
        if (ctx.rankIds[m] == selfRank) {
            ctx.selfMemberId = m;
            selfFound = true;
            break;
        }
    }
    CHK_PRT_RET(!selfFound, HCCL_ERROR("[%s] selfRank[%u] not in team rankIds", __func__, selfRank), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

/* 取 worldTeam 的 syncMem/memHandles、worldTeam 维度信息并计算 curToWorld 映射。 */
HcclResult GetWorldTeamContext(HcommTeamHandle team, uint32_t selfRank, ChannelsCreateCtx& ctx)
{
    ctx.worldTeam = HcclTeamMgr::GetInstance().FindWorldTeam(team);
    CHK_PTR_NULL(ctx.worldTeam);
    ctx.syncMemTag = HcclTeamMgr::GetInstance().GetTeamSyncMemTag(team);
    // 只收集 syncMemHandle（window memHandle 由 AppendSymmetricMemHandles 统一处理）
    ctx.memHandles = HcclTeamMgr::GetInstance().CollectPendingMemHandles(ctx.worldTeam, team);

    ctx.worldRankIds = HcclTeamMgr::GetInstance().GetRankIds(ctx.worldTeam);
    CHK_PRT_RET(
        ctx.worldRankIds.empty(),
        HCCL_ERROR("[%s] GetRankIds failed, worldTeam[%p] not registered", __func__, ctx.worldTeam), HCCL_E_PARA);
    ctx.worldMemberNum = static_cast<uint32_t>(ctx.worldRankIds.size());
    bool worldSelfFound = false;
    for (uint32_t w = 0; w < ctx.worldMemberNum; w++) {
        if (ctx.worldRankIds[w] == selfRank) {
            ctx.worldSelfMemberId = w;
            worldSelfFound = true;
            break;
        }
    }
    CHK_PRT_RET(
        !worldSelfFound, HCCL_ERROR("[%s] selfRank[%u] not in worldTeam rankIds", __func__, selfRank), HCCL_E_PARA);
    // curToWorld[m] = 当前 team memberId m 的 rankId 在 worldRankIds 中的下标（worldMemberId）
    ctx.curToWorld.assign(ctx.memberNum, 0);
    for (uint32_t m = 0; m < ctx.memberNum; m++) {
        uint32_t rankId = ctx.rankIds[m];
        for (uint32_t w = 0; w < ctx.worldMemberNum; w++) {
            if (ctx.worldRankIds[w] == rankId) {
                ctx.curToWorld[m] = w;
                break;
            }
        }
    }
    return HCCL_SUCCESS;
}
} // namespace hccl

/* 对每个 peer member 创建 channelCnt 个 channel，结果存 ctx.channelsByMember（self 为空）。 */
static HcclResult AcquireChannels(
    HcclComm comm, HcommTeamHandle team, const HcclTeamCreateDesc* desc, uint32_t selfRank, ChannelsCreateCtx& ctx)
{
    uint32_t channelCnt = desc->channelCnt;
    ctx.channelsByMember.assign(ctx.memberNum, {});
    for (uint32_t m = 0; m < ctx.memberNum; m++) {
        if (m == ctx.selfMemberId) {
            continue; // 本 member 不建到自己的 channel
        }
        uint32_t peerRank = ctx.rankIds[m];
        std::vector<HcclChannelDesc> channelDescs(channelCnt);
        HcclResult ret = HcclChannelDescInit(channelDescs.data(), channelCnt);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] HcclChannelDescInit failed, ret[%d]", __func__, ret), ret);
        for (uint32_t c = 0; c < channelCnt; c++) {
            ret = FillChannelDescForPeer(comm, team, selfRank, peerRank, channelDescs[c]);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[%s] FillChannelDescForPeer failed, peerRank[%u] ret[%d]", __func__, peerRank, ret), ret);
            channelDescs[c].notifyNum = desc->notifyNum;
            if (!ctx.memHandles.empty()) {
                channelDescs[c].memHandles = ctx.memHandles.data();
                channelDescs[c].memHandleNum = static_cast<uint32_t>(ctx.memHandles.size());
            }
        }
        ctx.channelsByMember[m].assign(channelCnt, 0);

        // 创建并配置ChannelConfig
        HcclChannelConfig config = nullptr;
        CHK_RET(HcclChannelConfigCreate(&config));
        CHK_RET(HcclChannelConfigSetInt(config, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, desc->isSharedQueue));
        if (desc->isSharedQueue != 0 && desc->sharedQueueTag[0] != '\0') {
            CHK_RET(HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, desc->sharedQueueTag));
        }

        ret = HcclChannelAcquireWithConfig(
            comm, desc->engine, channelDescs.data(), channelCnt, config, ctx.channelsByMember[m].data());
        (void)HcclChannelConfigDestroy(config);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] HcclChannelAcquire failed, peerMember[%u] peerRank[%u] ret[%d]", __func__, m, peerRank, ret),
            ret);
    }
    return HCCL_SUCCESS;
}

/* 构造 HcommTeamBindChannelsDesc，绑定 team 与 channel。 */
static HcclResult BindTeamChannels(HcommTeamHandle team, const std::string& commId, ChannelsCreateCtx& ctx)
{
    std::vector<uint32_t> channelNumPerMember(ctx.memberNum, 0);
    std::vector<uint64_t*> channelsByMemberIdPtrs(ctx.memberNum, nullptr);
    // 保持各 member 的 channel 数组生命周期覆盖 BindChannels 调用
    for (uint32_t m = 0; m < ctx.memberNum; m++) {
        channelNumPerMember[m] = static_cast<uint32_t>(ctx.channelsByMember[m].size());
        channelsByMemberIdPtrs[m] = ctx.channelsByMember[m].data();
    }
    HcommTeamBindChannelsDesc bindDesc{};
    (void)HcommTeamBindChannelsDescInit(&bindDesc);
    bindDesc.memberNum = ctx.memberNum;
    bindDesc.channelNumPerMember = channelNumPerMember.data();
    bindDesc.channelsByMemberId = channelsByMemberIdPtrs.data();
    HcommResult hRet = HcommTeamBindChannels(team, &bindDesc);
    CHK_PRT_RET(
        hRet != 0,
        HCCL_ERROR(
            "[%s] HcommTeamBindChannels failed, comm[%s] ret[%d] memberNum[%u]", __func__, commId.c_str(), hRet,
            ctx.memberNum),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

/* 收集所有 peer channel 的远端 syncMem。selfMemberId 槽填本地 syncMem，peer 槽按 memberId 填远端 syncMem。
 * window 远端内存已由 CollComm::UpdateHcommWindowRemoteMem 统一回填，此处不再处理。 */
static HcclResult CollectSyncMem(HcclComm comm, HcommTeamHandle team, ChannelsCreateCtx& ctx)
{
    ctx.syncMemRemoteMems.assign(ctx.memberNum, CommMem{});
    // selfMemberId 槽填本地 syncMem 内存
    void* localSyncMemPtr = HcclTeamMgr::GetInstance().GetSyncMemPtr(team);
    uint64_t localSyncMemSize = HcclTeamMgr::GetInstance().GetSyncMemSize(team);
    if (localSyncMemPtr != nullptr && localSyncMemSize > 0) {
        ctx.syncMemRemoteMems[ctx.selfMemberId].type = COMM_MEM_TYPE_DEVICE;
        ctx.syncMemRemoteMems[ctx.selfMemberId].addr = localSyncMemPtr;
        ctx.syncMemRemoteMems[ctx.selfMemberId].size = localSyncMemSize;
    }
    for (uint32_t m = 0; m < ctx.memberNum; m++) {
        if (m == ctx.selfMemberId || ctx.channelsByMember[m].empty()) {
            continue;
        }
        for (ChannelHandle channel : ctx.channelsByMember[m]) {
            if (channel == 0) {
                continue;
            }
            uint32_t memNum = 0;
            CommMem* remoteMems = nullptr;
            char** memTags = nullptr;
            HcclResult gRet = HcclChannelGetRemoteMems(comm, channel, &memNum, &remoteMems, &memTags);
            CHK_PRT_RET(
                gRet != HCCL_SUCCESS,
                HCCL_ERROR("[%s] HcclChannelGetRemoteMems failed, member[%u] ret[%d]", __func__, m, gRet), gRet);
            for (uint32_t r = 0; r < memNum; r++) {
                std::string tag
                    = (memTags != nullptr && memTags[r] != nullptr) ? std::string(memTags[r]) : std::string();
                if (!ctx.syncMemTag.empty()
                    && tag.compare(0, strlen(HCCL_TEAM_SYNCMEM_TAG_PREFIX), HCCL_TEAM_SYNCMEM_TAG_PREFIX) == 0) {
                    ctx.syncMemRemoteMems[m] = remoteMems[r];
                    break; // 每个 channel 只有一个 syncMem
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

/* 调 HcommTeamBindRemoteSyncMem 绑定 syncMem。window 绑定已由 CollComm::UpdateHcommWindowRemoteMem 统一完成。 */
static HcclResult BindSyncMem(HcommTeamHandle team, const std::string& commId, ChannelsCreateCtx& ctx)
{
    HcommTeamBindSyncMemDesc syncDesc{};
    (void)HcommTeamBindSyncMemDescInit(&syncDesc);
    syncDesc.remoteMems = ctx.syncMemRemoteMems.data();
    syncDesc.remoteMemNum = static_cast<uint32_t>(ctx.syncMemRemoteMems.size());
    HcommResult bindSyncRet = HcommTeamBindRemoteSyncMem(team, &syncDesc);
    CHK_PRT_RET(
        bindSyncRet != 0,
        HCCL_ERROR(
            "[%s] HcommTeamBindRemoteSyncMem failed, comm[%s] team[%p] ret[%d] remoteMemNum[%zu]", __func__,
            commId.c_str(), team, bindSyncRet, ctx.syncMemRemoteMems.size()),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

/* 建链管道：注册 syncMem → 取成员信息 → 建链 → 绑定 channel → 取远端 syncMem → 绑定 syncMem。
 * 失败由调用方统一调 HcclTeamDestroy 回滚（team 已注册到 HcclTeamMgr，Destroy 即完整逆操作）。 */
static HcclResult
SetupTeamChannels(HcclComm comm, HcommTeamHandle team, const HcclTeamCreateDesc* desc, CollComm* collComm)
{
    const std::string commId = collComm->GetCommId();
    uint32_t selfRank = collComm->GetMyRankId();
    /* window 此时已由 HcclCommSymWinRegister 注册到 HcclTeamMgr（属于通信域），GetWorldTeamContext 取之。 */
    CHK_RET(RegisterTeamSyncMem(team, collComm));
    ChannelsCreateCtx ctx{};
    CHK_RET(GetTeamMemberInfo(team, selfRank, ctx));
    CHK_RET(GetWorldTeamContext(team, selfRank, ctx));
    CHK_RET(AcquireChannels(comm, team, desc, selfRank, ctx));
    CHK_RET(BindTeamChannels(team, commId, ctx));
    CHK_RET(CollectSyncMem(comm, team, ctx));
    CHK_RET(BindSyncMem(team, commId, ctx));
    return HCCL_SUCCESS;
}

HcclResult HcclTeamCreate(HcclComm comm, const HcclTeamCreateDesc* desc, HcommTeamHandle* team)
{
    CHK_RET(CheckTeamCreateParam(comm, desc, team));

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    const std::string commId = collComm->GetCommId();

    /* 1. 在本通信域内按 protocol+netLayer 查找初始化时预制的 worldTeam（不存在则报错，不再内部创建）。 */
    HcommTeamHandle worldTeam
        = HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(collComm, desc->protocol, desc->netLayer);
    CHK_PRT_RET(
        worldTeam == nullptr,
        HCCL_ERROR(
            "[%s] no worldTeam for protocol[%d] netLayer[%u], comm[%s]", __func__, desc->protocol, desc->netLayer,
            commId.c_str()),
        HCCL_E_NOT_FOUND);

    /* 2. 反查 subTeam.rankIds 在 worldTeam.rankIds 中的下标作为 worldMemberIds。 */
    uint32_t selfMemberId = 0;
    CHK_RET(FindSelfMemberId(desc, selfMemberId));
    std::vector<uint32_t> worldMemberIds;
    CHK_RET(BuildSubTeamWorldMemberIds(worldTeam, desc->rankIds, desc->rankNum, worldMemberIds));

    /* 3. 填充 HcommTeamCreateDesc 并创建 subTeam。 */
    HcommTeamCreateDesc hcommDesc{};
    FillHcommTeamCreateDesc(desc, selfMemberId, worldMemberIds.data(), hcommDesc);

    /* 4. 创建 subTeam 并申请 syncMem 内存，失败时内部清理 */
    CHK_RET(CreateSubTeamWithSyncMem(worldTeam, desc, hcommDesc, team));

    /* 5. 建链管道。失败统一回滚：HcclTeamDestroy = CommUnregMem(syncMem) + hrtFree(syncMem) + L2 条目 erase
     *    + L3 team 释放，即成功创建的完整逆操作。 */
    HcclResult ret = SetupTeamChannels(comm, *team, desc, collComm);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] SetupTeamChannels failed, team[%p] ret[%d], rollback", __func__, *team, ret);
        (void)HcclTeamDestroy(*team);
        *team = nullptr;
        return ret;
    }

    HCCL_INFO(
        "[%s] success, comm[%s] team[%p] worldTeam[%p] protocol[%d] netLayer[%u] channelCnt[%u] memberNum[%u]",
        __func__, commId.c_str(), *team, worldTeam, desc->protocol, desc->netLayer, desc->channelCnt, desc->rankNum);
    return HCCL_SUCCESS;
}

HcclResult HcclTeamDestroy(HcommTeamHandle team)
{
    CHK_PTR_NULL(team);
    if (HcclTeamMgr::GetInstance().FindWorldTeam(team) == team) {
        HCCL_ERROR("[%s] prebuilt worldTeam can not be destroyed", __func__);
        return HCCL_E_PARA;
    }

    /* 先查出 syncMem 解注册信息（UnregisterTeam 后 TeamEntry 已 erase，无法再查）。 */
    HcclMemHandle syncMemHandle = HcclTeamMgr::GetInstance().GetTeamSyncMemHandle(team);
    std::string syncMemTag = HcclTeamMgr::GetInstance().GetTeamSyncMemTag(team);
    CollComm* collComm = HcclTeamMgr::GetInstance().FindCollComm(team);
    /* 解注册 syncMem 的 CommRegMem（CommUnregMem），释放 memHandle 资源。 */
    if (syncMemHandle != nullptr && !syncMemTag.empty() && collComm != nullptr) {
        auto myRank = collComm->GetMyRank();
        CHK_PTR_NULL(myRank);
        CommMems* commMem = myRank->GetCommMems();
        CHK_PTR_NULL(commMem);
        HcclResult unregRet = commMem->CommUnregMem(syncMemTag, syncMemHandle);
        if (unregRet != HCCL_SUCCESS && unregRet != HCCL_E_NOT_FOUND) {
            HCCL_WARNING("[%s] CommUnregMem syncMem failed, tag[%s] ret[%d]", __func__, syncMemTag.c_str(), unregRet);
        }
    }

    /* 释放该 team 的 syncMem（hrtFree）+ erase 条目。 */
    HcclTeamMgr::GetInstance().UnregisterTeam(team);

    HcommResult ret = HcommTeamDestroy(team);
    CHK_PRT_RET(ret != 0, HCCL_ERROR("[%s] HcommTeamDestroy failed, ret[%d]", __func__, ret), HCCL_E_INTERNAL);

    HCCL_INFO("[%s] success", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcclTeamGetLsaTeam(HcclComm comm, HcommTeamHandle* lsaTeam)
{
    CHK_PTR_NULL(lsaTeam);
    *lsaTeam = nullptr;
    CollComm* collComm = nullptr;
    CHK_RET(GetTeamQueryCollComm(comm, collComm));
    return HcclTeamMgr::GetInstance().GetLsaTeam(collComm, *lsaTeam);
}

HcclResult HcclTeamMemberToRank(HcclComm comm, HcommTeamHandle team, uint32_t memberId, uint32_t* rankId)
{
    CHK_PTR_NULL(team);
    CHK_PTR_NULL(rankId);
    CollComm* collComm = nullptr;
    CHK_RET(GetTeamQueryCollComm(comm, collComm));
    return HcclTeamMgr::GetInstance().MemberToRank(collComm, team, memberId, *rankId);
}

HcclResult HcclTeamRankToMember(HcclComm comm, HcommTeamHandle team, uint32_t rankId, uint32_t* memberId)
{
    CHK_PTR_NULL(team);
    CHK_PTR_NULL(memberId);
    CollComm* collComm = nullptr;
    CHK_RET(GetTeamQueryCollComm(comm, collComm));
    return HcclTeamMgr::GetInstance().RankToMember(collComm, team, rankId, *memberId);
}
