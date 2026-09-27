/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_TEAM_H
#define HCOMM_TEAM_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "hcomm_res_defs.h"
#include "hcomm_team_defs.h"
#include "hcomm_result_defs.h"
#include "hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct {
    CommAbiHeader header;
    uint32_t memberNum;
    uint32_t selfMemberId;
    const uint32_t*
        worldMemberIds; /*worldMemberIds在创建worldteam的时候是nullptr，创建subteam的时候是subteam的成员在worldteam中的memberId数组，长度是memberNum*/
    uint32_t netLayer;  /* 用户希望使用的网络层，0表示默认选择 */
    CommProtocol protocol; /* 用户希望使用的通信协议，-1表示保留协议类型, 1个team仅支持一个协议 */
    CommEngine engine; /* 通信引擎类型，建链时使用并持久化到 HcommTeam.engine */
    HcommTeamSyncMemRequirement requirement;
    uint32_t reserved[5];
} HcommTeamCreateDesc;

typedef struct {
    CommAbiHeader header;
    uint32_t* channelNumPerMember; /* 长度为memberNum，表示每个成员的channel数量 */
    ChannelHandle** channelsByMemberId;
    uint32_t memberNum;
    uint32_t reserved[8];
} HcommTeamBindChannelsDesc;

typedef struct {
    CommAbiHeader header;
    CommMem* remoteMems;
    uint32_t remoteMemNum;
    uint32_t reserved[5];
} HcommTeamBindSyncMemDesc;

static const uint32_t HCOMM_TEAM_CREATE_DESC_MAGIC_WORD = 0x0f0f0f10U;
static const uint32_t HCOMM_TEAM_CREATE_DESC_VERSION = 1U;

static const uint32_t HCOMM_TEAM_BIND_CHANNELS_DESC_MAGIC_WORD = 0x0f0f0f11U;
static const uint32_t HCOMM_TEAM_BIND_CHANNELS_DESC_VERSION = 1U;

static const uint32_t HCOMM_TEAM_BIND_SYNCMEM_DESC_MAGIC_WORD = 0x0f0f0f13U;
static const uint32_t HCOMM_TEAM_BIND_SYNCMEM_DESC_VERSION = 1U;

static inline HcommResult HcommTeamCreateDescInit(HcommTeamCreateDesc* desc)
{
    const HcommResult hcommEPointer = HCOMM_E_PTR;

    if (desc == nullptr) {
        return hcommEPointer;
    }

    (void)memset_s(desc, sizeof(HcommTeamCreateDesc), 0xFF, sizeof(HcommTeamCreateDesc));
    desc->header.version = HCOMM_TEAM_CREATE_DESC_VERSION;
    desc->header.magicWord = HCOMM_TEAM_CREATE_DESC_MAGIC_WORD;
    desc->header.size = sizeof(HcommTeamCreateDesc);
    desc->header.reserved = 0;
    desc->memberNum = 0;
    desc->selfMemberId = 0;
    desc->worldMemberIds = nullptr;
    desc->netLayer = 0;
    desc->protocol = COMM_PROTOCOL_RESERVED;
    desc->engine = COMM_ENGINE_RESERVED;
    desc->requirement.signalCount = 0;
    desc->requirement.counterCount = 0;
    desc->requirement.barrierCount = 0;
    const uint32_t reqReservedCount = sizeof(desc->requirement.reserved) / sizeof(desc->requirement.reserved[0]);
    for (uint32_t i = 0; i < reqReservedCount; ++i) {
        desc->requirement.reserved[i] = 0;
    }
    const uint32_t reservedCount = sizeof(desc->reserved) / sizeof(desc->reserved[0]);
    for (uint32_t i = 0; i < reservedCount; ++i) {
        desc->reserved[i] = 0;
    }

    return 0;
}

static inline HcommResult HcommTeamBindChannelsDescInit(HcommTeamBindChannelsDesc* desc)
{
    const HcommResult hcommEPointer = HCOMM_E_PTR;

    if (desc == nullptr) {
        return hcommEPointer;
    }

    (void)memset_s(desc, sizeof(HcommTeamBindChannelsDesc), 0xFF, sizeof(HcommTeamBindChannelsDesc));
    desc->header.version = HCOMM_TEAM_BIND_CHANNELS_DESC_VERSION;
    desc->header.magicWord = HCOMM_TEAM_BIND_CHANNELS_DESC_MAGIC_WORD;
    desc->header.size = sizeof(HcommTeamBindChannelsDesc);
    desc->header.reserved = 0;
    desc->memberNum = 0;
    desc->channelNumPerMember = nullptr;
    desc->channelsByMemberId = nullptr;
    const uint32_t reservedCount = sizeof(desc->reserved) / sizeof(desc->reserved[0]);
    for (uint32_t i = 0; i < reservedCount; ++i) {
        desc->reserved[i] = 0;
    }

    return 0;
}

static inline HcommResult HcommTeamBindSyncMemDescInit(HcommTeamBindSyncMemDesc* desc)
{
    const HcommResult hcommEPointer = HCOMM_E_PTR;

    if (desc == nullptr) {
        return hcommEPointer;
    }

    (void)memset_s(desc, sizeof(HcommTeamBindSyncMemDesc), 0xFF, sizeof(HcommTeamBindSyncMemDesc));
    desc->header.version = HCOMM_TEAM_BIND_SYNCMEM_DESC_VERSION;
    desc->header.magicWord = HCOMM_TEAM_BIND_SYNCMEM_DESC_MAGIC_WORD;
    desc->header.size = sizeof(HcommTeamBindSyncMemDesc);
    desc->header.reserved = 0;
    desc->remoteMems = nullptr;
    desc->remoteMemNum = 0;
    const uint32_t reservedCount = sizeof(desc->reserved) / sizeof(desc->reserved[0]);
    for (uint32_t i = 0; i < reservedCount; ++i) {
        desc->reserved[i] = 0;
    }

    return 0;
}

/* ===== team 生命周期 ===== */
HcommResult HcommTeamCreate(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* syncMemSize);
HcommResult HcommTeamDestroy(HcommTeamHandle team);

/* ===== team 窗口操作 ===== */
HcommResult HcommTeamWindowRegister(void* devLegacySymWin, HcclCommSymWindow* handle);
HcommResult HcommTeamWindowDeregister(HcclCommSymWindow handle);
/* 登记本端窗口注册信息（用户 VA/size + 本端层槽位），UpdateWindowRemoteMemByRank 据此回填本端槽位 */
HcommResult HcommTeamWindowSetSelfInfo(
    HcclCommSymWindow handle, void* selfVa, uint64_t selfSize, const uint32_t* selfSlots, uint32_t selfSlotNum);

/* ===== team 资源绑定 ===== */
HcommResult HcommTeamBindChannels(HcommTeamHandle team, const HcommTeamBindChannelsDesc* desc);
HcommResult HcommTeamBindRemoteSyncMem(HcommTeamHandle team, const HcommTeamBindSyncMemDesc* remoteDesc);
HcommResult HcommTeamBindUbSymmetricWindow(
    HcclCommSymWindow handle, HcommTeamHandle lsaTeam, uint32_t netLayer, const CommMem* memberMems, uint32_t memberNum,
    void* baseVa, size_t stride, size_t userSize);
HcommResult HcommTeamUpdateWindowRemoteMemByRank(
    HcclCommSymWindow handle, const uint32_t* sizes, uint32_t sizeNum, const uint32_t* slots, uint32_t slotNum,
    const CommMem* remoteMem);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCOMM_TEAM_H
