/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_team_c_adpt.h"
#include "hcomm_team.h"
#include "hcomm_team_mgr.h"
#include "hcomm_result_defs.h"
#include "hcomm_team_entity_defs.h"
#include "log.h"

using namespace hcomm;

HcommResult HcommTeamCreate(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* syncMemSize)
{
    CHK_PRT_RET(
        (desc == nullptr || team == nullptr || syncMemSize == nullptr), HCCL_ERROR("[%s] nullptr parameter", __func__),
        HCOMM_E_PTR);
    CHK_PRT_RET(
        (desc->memberNum == 0 || desc->memberNum == 1), HCCL_ERROR("[%s] memberNum cannot be 0 or 1", __func__),
        HCOMM_E_PARA);
    CHK_PRT_RET(
        desc->selfMemberId >= desc->memberNum,
        HCCL_ERROR("[%s] selfMemberId[%u] >= memberNum[%u]", __func__, desc->selfMemberId, desc->memberNum),
        HCOMM_E_PARA);
    CHK_PRT_RET(
        (desc->requirement.signalCount != 0 || desc->requirement.counterCount != 0),
        HCCL_ERROR(
            "[%s] signalCount[%u] and counterCount[%u] must be 0", __func__, desc->requirement.signalCount,
            desc->requirement.counterCount),
        HCOMM_E_PARA);
    CHK_PRT_RET(
        desc->protocol != COMM_PROTOCOL_UB_MEM && desc->requirement.barrierCount == 0,
        HCCL_ERROR("[%s] barrierCount must be >= 1 for protocol[%d]", __func__, desc->protocol), HCOMM_E_PARA);
    CHK_PRT_RET(
        desc->protocol == COMM_PROTOCOL_UB_MEM && desc->requirement.barrierCount != 0,
        HCCL_ERROR("[%s] UB Memory team does not use barrier sync memory", __func__), HCOMM_E_PARA);
    CHK_PRT_RET(
        worldTeam != nullptr && desc->worldMemberIds == nullptr,
        HCCL_ERROR("[%s] sub team worldMemberIds is null", __func__), HCOMM_E_PTR);
    return HcommTeamMgr::GetInstance().TeamCreate(worldTeam, desc, team, syncMemSize);
}

HcommResult HcommTeamDestroy(HcommTeamHandle team)
{
    CHK_PRT_RET(team == nullptr, HCCL_ERROR("[%s] team is nullptr", __func__), HCOMM_E_PTR);
    return HcommTeamMgr::GetInstance().TeamDestroy(team);
}

HcommResult HcommTeamBindChannels(HcommTeamHandle team, const HcommTeamBindChannelsDesc* desc)
{
    if (team == nullptr || desc == nullptr || desc->channelNumPerMember == nullptr
        || desc->channelsByMemberId == nullptr) {
        HCCL_ERROR("[%s] nullptr parameter", __func__);
        return HCOMM_E_PTR;
    }
    return HcommTeamMgr::GetInstance().BindChannels(team, desc);
}

HcommResult HcommTeamBindRemoteSyncMem(HcommTeamHandle team, const HcommTeamBindSyncMemDesc* remoteDesc)
{
    CHK_PRT_RET(
        (team == nullptr || remoteDesc == nullptr), HCCL_ERROR("[%s] nullptr parameter", __func__), HCOMM_E_PTR);
    CHK_PRT_RET(remoteDesc->remoteMems == nullptr, HCCL_ERROR("[%s] remoteMems is nullptr", __func__), HCOMM_E_PTR);
    CHK_PRT_RET(remoteDesc->remoteMemNum == 0, HCCL_ERROR("[%s] remoteMemNum is zero", __func__), HCOMM_E_PARA);
    return HcommTeamMgr::GetInstance().BindSyncMem(team, remoteDesc);
}

HcommResult HcommTeamBindUbSymmetricWindow(
    HcclCommSymWindow handle, HcommTeamHandle lsaTeam, uint32_t netLayer, const CommMem* memberMems, uint32_t memberNum,
    void* baseVa, size_t stride, size_t userSize)
{
    CHK_PRT_RET(
        handle == nullptr || lsaTeam == nullptr || memberMems == nullptr || baseVa == nullptr,
        HCCL_ERROR("[%s] nullptr parameter", __func__), HCOMM_E_PTR);
    CHK_PRT_RET(
        memberNum == 0 || stride == 0 || userSize == 0,
        HCCL_ERROR(
            "[%s] invalid layout, memberNum[%u], stride[%zu], userSize[%zu]", __func__, memberNum, stride, userSize),
        HCOMM_E_PARA);
    return HcommTeamMgr::GetInstance().BindUbSymmetricWindow(
        handle, lsaTeam, netLayer, memberMems, memberNum, baseVa, stride, userSize);
}

HcommResult HcommTeamWindowRegister(void* devLegacySymWin, HcclCommSymWindow* handle)
{
    CHK_PRT_RET(devLegacySymWin == nullptr, HCCL_ERROR("[%s] nullptr parameter", __func__), HCOMM_E_PTR);
    CHK_PRT_RET(handle == nullptr, HCCL_ERROR("[%s] nullptr parameter", __func__), HCOMM_E_PTR);
    return HcommTeamMgr::GetInstance().WindowRegister(devLegacySymWin, handle);
}

HcommResult HcommTeamWindowDeregister(HcclCommSymWindow handle)
{
    CHK_PRT_RET(handle == nullptr, HCCL_ERROR("[%s] nullptr parameter", __func__), HCOMM_E_PTR);
    return HcommTeamMgr::GetInstance().WindowDeregister(handle);
}

HcommResult HcommTeamWindowSetSelfInfo(
    HcclCommSymWindow handle, void* selfVa, uint64_t selfSize, const uint32_t* selfSlots, uint32_t selfSlotNum)
{
    CHK_PRT_RET(handle == nullptr, HCCL_ERROR("[%s] nullptr parameter", __func__), HCOMM_E_PTR);
    CHK_PRT_RET(
        (selfVa == nullptr || selfSize == 0) && (selfSlots == nullptr || selfSlotNum == 0),
        HCCL_ERROR(
            "[%s] nothing to set, selfVa[%p] selfSize[%llu] selfSlotNum[%u]", __func__, selfVa, selfSize, selfSlotNum),
        HCOMM_E_PARA);
    return HcommTeamMgr::GetInstance().SetWindowSelfInfo(handle, selfVa, selfSize, selfSlots, selfSlotNum);
}

HcommResult HcommTeamUpdateWindowRemoteMemByRank(
    HcclCommSymWindow handle, const uint32_t* sizes, uint32_t sizeNum, const uint32_t* slots, uint32_t slotNum,
    const CommMem* remoteMem)
{
    CHK_PRT_RET(
        (handle == nullptr || remoteMem == nullptr), HCCL_ERROR("[%s] nullptr parameter", __func__), HCOMM_E_PTR);
    CHK_PRT_RET(
        slotNum == 0 || slots == nullptr || sizes == nullptr || sizeNum == 0,
        HCCL_ERROR("[%s] no layer slots or invalid sizes", __func__), HCOMM_E_PARA);
    return HcommTeamMgr::GetInstance().UpdateWindowRemoteMemByRank(handle, sizes, sizeNum, slots, slotNum, *remoteMem);
}

HcommResult HcommTeamGetNetLayer(HcommTeamHandle team, uint32_t* netLayer)
{
    CHK_PRT_RET((team == nullptr || netLayer == nullptr), HCCL_ERROR("[%s] nullptr parameter", __func__), HCOMM_E_PTR);
    return HcommTeamMgr::GetInstance().GetNetLayer(team, netLayer);
}

HcommResult HcommTeamGetEngine(HcommTeamHandle team, CommEngine* engine)
{
    CHK_PRT_RET((team == nullptr || engine == nullptr), HCCL_ERROR("[%s] nullptr parameter", __func__), HCOMM_E_PTR);
    return HcommTeamMgr::GetInstance().GetEngine(team, engine);
}
