/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TEAM_H
#define HCCL_TEAM_H
#include <stdint.h>
#include <stddef.h>
#include "hcomm_res_defs.h"
#include "hcomm_team_defs.h"
#include "hccl_types.h"
#include "hccl_channel.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct {
    CommAbiHeader header;
    const uint32_t* rankIds; /* 用户希望填写的rankids */
    uint32_t rankNum;
    uint32_t selfRankId;
    uint32_t netLayer; /* 用户希望使用的网络层，0表示默认选择 */
    CommProtocol protocol; /* 用户希望使用的通信协议，-1表示保留协议类型, 1个team仅支持一个协议 */
    HcommTeamSyncMemRequirement requirement;
    CommEngine engine;                                                 /* 通信引擎类型 */
    uint32_t notifyNum;                                                /* channel所需的notify数量 */
    uint32_t channelCnt;                                               /* 用户可自行指定channel个数 */
    uint32_t isSharedQueue;                                            /* 是否共享队列 */
    char sharedQueueTag[HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN]; /* 共享队列标签 */
    uint32_t reserved[8];
} HcclTeamCreateDesc;

static const uint32_t HCCL_TEAM_CREATE_DESC_MAGIC_WORD = 0x0fcf0f14U;
static const uint32_t HCCL_TEAM_CREATE_DESC_VERSION = 2U;

static inline HcclResult HcclTeamCreateDescInit(HcclTeamCreateDesc* desc)
{
    if (desc == NULL) {
        return HCCL_E_PTR;
    }

    (void)memset_s(desc, sizeof(HcclTeamCreateDesc), 0xFF, sizeof(HcclTeamCreateDesc));
    desc->header.version = HCCL_TEAM_CREATE_DESC_VERSION;
    desc->header.magicWord = HCCL_TEAM_CREATE_DESC_MAGIC_WORD;
    desc->header.size = sizeof(HcclTeamCreateDesc);
    desc->header.reserved = 0;
    desc->rankIds = NULL;
    desc->rankNum = 0;
    desc->selfRankId = 0;
    desc->netLayer = 0;
    desc->protocol = COMM_PROTOCOL_RESERVED;
    desc->requirement.signalCount = 0;
    desc->requirement.counterCount = 0;
    desc->requirement.barrierCount = 0;
    desc->engine = COMM_ENGINE_RESERVED;
    desc->notifyNum = 0;
    desc->channelCnt = 0;
    desc->isSharedQueue = 0;
    desc->sharedQueueTag[0] = '\0';
    for (uint32_t i = 0; i < 8; ++i) {
        desc->reserved[i] = 0;
    }

    return HCCL_SUCCESS;
}

/**
 * @brief Create a team for HCCL communication.
 *
 * @param comm A pointer identifying the initialized communication resource.
 * @param desc A pointer identifying the team creation description.
 * @param team A pointer identifying the created team handle.
 * @return HcclResult
 * @see HcclTeamDestroy()
 */
extern HcclResult HcclTeamCreate(HcclComm comm, const HcclTeamCreateDesc* desc, HcommTeamHandle* team);

/**
 * @brief Destroy a team for HCCL communication.
 *
 * @param team A pointer identifying the team handle to be destroyed.
 * @return HcclResult
 * @see HcclTeamCreate()
 */
extern HcclResult HcclTeamDestroy(HcommTeamHandle team);

/**
 * @brief Get the prebuilt LSA world team of a communicator.
 *
 * @param comm A pointer identifying the initialized communication resource.
 * @param lsaTeam A pointer to the returned LSA world team handle. The handle is owned by comm.
 * @return HcclResult
 */
extern HcclResult HcclTeamGetLsaTeam(HcclComm comm, HcommTeamHandle* lsaTeam);

/**
 * @brief Convert a team member ID to its communicator rank ID.
 *
 * @param comm A pointer identifying the initialized communication resource.
 * @param team A team handle owned by comm.
 * @param memberId The member ID in team.
 * @param rankId A pointer to the returned communicator rank ID.
 * @return HcclResult
 */
extern HcclResult HcclTeamMemberToRank(HcclComm comm, HcommTeamHandle team, uint32_t memberId, uint32_t* rankId);

/**
 * @brief Convert a communicator rank ID to its member ID in a team.
 *
 * @param comm A pointer identifying the initialized communication resource.
 * @param team A team handle owned by comm.
 * @param rankId The communicator rank ID.
 * @param memberId A pointer to the returned member ID in team.
 * @return HcclResult
 */
extern HcclResult HcclTeamRankToMember(HcclComm comm, HcommTeamHandle team, uint32_t rankId, uint32_t* memberId);
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_TEAM_H
