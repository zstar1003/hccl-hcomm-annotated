/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_TEAM_ENTITY_DEFS_H
#define HCOMM_TEAM_ENTITY_DEFS_H

#include <cstdint>
#include "hcomm_res_defs.h"
#include "hcomm_team_defs.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/* HcommTeam / HcommWindow 结构体的 ABI 头部常量 */
static const uint32_t HCOMM_TEAM_MAGIC_WORD = 0x0f0f0f20U;
static const uint32_t HCOMM_TEAM_VERSION = 1U;
static const uint32_t HCOMM_WINDOW_MAGIC_WORD = 0x0f0f0f21U;
static const uint32_t HCOMM_WINDOW_VERSION = 2U;

typedef struct {
    CommAbiHeader header;
    struct {
        uint64_t baseRemoteMemAddr; /* 远端内存addr数组基地址，按(worldTeamAccumulateId[netLayer] + worldTeamId) *
                                       sizeof(uint64_t)偏移访问 */
        uint64_t windowSize;        /* window大小 */
        uint32_t* worldTeamAccumulateId; /* 数组，表示sum(worldTeamSizePerNetLayer[0..netLayerNum-1]) */
        uint32_t netLayerNum;            /* netLayer的数量 */
        uint32_t reserved[8];
    } netWin;

    struct {
        uint64_t baseVa;
        uint64_t stride;
        uint64_t userSize;
        uint32_t reserved[8];
    } lsaWin;

    uint64_t legacySymWindow; // SymmetricWindow结构体device侧指针
    uint32_t reserved[8];
} HcommWindow;

typedef struct {
    CommMem* remoteMems; // 从远端交换过来的内存,
                         // 用作signal/barrier等，如果memberId是selfMemberId，则是本地内存，否则是远端内存
    uint32_t remoteMemsNum;
    CommMem
        shadowMem; // 申请一块(signalCount + counterCount + barrierCount) * sizeof(uint64_t) * memberNum的device侧内存
    HcommTeamSyncMemRequirement syncMemReq;
    uint64_t syncMemSize; // (signalCount + counterCount + barrierCount) * sizeof(uint64_t) * memberNum
    uint32_t reserved[5];
} HcommTeamSyncMem;

// hcommTeamRes存放worldTeam的资源，使用subTeam来寻找hcommTeamRes
typedef struct {
    CommAbiHeader header;
    CommEngine engine; /* COMM_ENGINE_AICPU_TS / CCU 等 */
    uint32_t memberNum;
    uint32_t selfMemberId;
    uint64_t channelsBaseAddr; /* 连续ChannelEntity数组的基地址，按(channelCntAccumulatePerMember[peer] + channelIdx) *
                                  sizeof(ChannelEntity)偏移访问 */
    uint32_t* channelCntAccumulatePerMember; /* 长度为memberNum，表示sum(channelNumPerMember[0..peer-1]) */
    uint32_t netLayer;                       /* 用户希望使用的网络层，0表示默认选择 */
    uint32_t* worldTeamIds;
    HcommTeamSyncMem syncMem;
    uint32_t reserved[8];
} HcommTeam;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCOMM_TEAM_ENTITY_DEFS_H
