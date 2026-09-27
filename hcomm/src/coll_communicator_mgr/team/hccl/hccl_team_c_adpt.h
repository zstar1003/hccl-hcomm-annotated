/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TEAM_C_ADPT_H
#define HCCL_TEAM_C_ADPT_H

#include <stdint.h>
#include <string>
#include <vector>

#include "hccl/hccl_res.h"
#include "hccl_team.h"
#include "hccl_team_mgr.h"
#include "hcomm_team_defs.h"
#include "hcomm_res_defs.h"

namespace hccl {
/* team window 内部注册内存使用的保留前缀，禁止用户在 HcclCommMemReg 中使用 */
constexpr const char* HCCL_TEAM_SYNCMEM_TAG_PREFIX = "__hccl_team_syncmem__";

/* HcclTeamCreate 跨段共享的中间状态。全程引用传递，禁用拷贝，避免 vector 深拷贝。 */
struct ChannelsCreateCtx {
    std::vector<uint32_t> rankIds; // 当前 team 的 memberId→rankId 映射，下标=memberId
    uint32_t memberNum{0};         // = rankIds.size()
    uint32_t selfMemberId{0};
    HcommTeamHandle worldTeam{nullptr};
    std::vector<uint32_t> worldRankIds; // worldTeam 的 memberId→rankId 映射
    uint32_t worldMemberNum{0};         // = worldRankIds.size()
    uint32_t worldSelfMemberId{0};
    std::vector<uint32_t> curToWorld; // curToWorld[memberId] = worldMemberId
    std::string syncMemTag;
    std::vector<HcclMemHandle> memHandles;
    std::vector<std::vector<ChannelHandle>> channelsByMember;
    std::vector<CommMem> syncMemRemoteMems; // 长度=memberNum，下标=memberId
};

/* ===== 对内工具函数 ===== */
// 从 HcclTeamMgr 取 team 的 rankIds（memberId→rankId），推算 memberNum 与 selfMemberId 填入 ctx。
HcclResult GetTeamMemberInfo(HcommTeamHandle team, uint32_t selfRank, ChannelsCreateCtx& ctx);
// 取 worldTeam 的 syncMem/memHandles、worldTeam 维度信息并计算 curToWorld 映射填入 ctx。
HcclResult GetWorldTeamContext(HcommTeamHandle team, uint32_t selfRank, ChannelsCreateCtx& ctx);
} // namespace hccl

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @note 职责：集合通信的通信域HcclTeam管理的C接口声明（暂未对外的接口）
 */

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCCL_TEAM_C_ADPT_H
