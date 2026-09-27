/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp_tlv_hdc_manager.h"
#include "unified_platform/ccu/ccu_device/ccu_res_specs_legacy.h"
#include "unified_platform/ccu/ccu_device/ccu_component/ccu_component.h"
#include "unified_platform/ccu/ccu_device/ccu_res_batch_allocator_legacy.h"
#include "unified_platform/ccu/ccu_context/ccu_context_mgr_imp.h"

#include "ccu_comp.h"

namespace Hccl {

void* HccpTlvHdcManager::GetTlvHandle(s32 deviceLogicId)
{
    (void)deviceLogicId;
    return (void*)0x12345678;
}

CcuComponent& CcuComponent::GetInstance(const int32_t deviceLogicId)
{
    static CcuComponent ccuComponent;
    return ccuComponent;
}

CcuComponent::~CcuComponent() {}

CcuResSpecifications& CcuResSpecifications::GetInstance(const int32_t deviceLogicId)
{
    static CcuResSpecifications ccuResSpecification;
    return ccuResSpecification;
}

HcclResult CcuResSpecifications::GetGsaNum(const uint8_t dieId, uint32_t& gsaNum) const
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetInstructionNum(const uint8_t dieId, uint32_t& instrNum) const
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetXnNum(const uint8_t dieId, uint32_t& xnNum) const
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetLoopEngineNum(const uint8_t dieId, uint32_t& loopNum) const
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetMissionNum(const uint8_t dieId, uint32_t& missionNum) const
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetMsNum(const uint8_t dieId, uint32_t& msNum) const
{
    return HcclResult::HCCL_SUCCESS;
}

CtxMgrImp& CtxMgrImp::GetInstance(s32 deviceLogicId)
{
    static CtxMgrImp contextManager;
    return contextManager;
}

CtxMgrImp::CtxMgrImp() {}

CtxMgrImp::~CtxMgrImp() {}

CcuResBatchAllocator& CcuResBatchAllocator::GetInstance(const int32_t deviceLogicId)
{
    static CcuResBatchAllocator ccuResBatchAllocator;
    return ccuResBatchAllocator;
}

HcclResult CcuResSpecifications::GetCkeNum(const uint8_t dieId, uint32_t& ckeNum) const
{
    return HcclResult::HCCL_SUCCESS;
}

void CcuResSpecifications::Init() {}

void CcuResSpecifications::Deinit() {}

CcuVersion CcuResSpecifications::GetCcuVersion() const { return CcuVersion::INVALID; }

HcclResult CcuResSpecifications::GetMissionKey(const uint8_t dieId, uint32_t& missionKey) const
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuResSpecifications::GetXnBaseAddr(const uint8_t dieId, uint64_t& xnBaseAddr) const
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

void CcuComponent::Init() {}

void CcuComponent::Deinit() {}

HcclResult CcuComponent::GetCcuResourceSpaceBufInfo(const uint8_t dieId, uint64_t& addr, uint64_t& size) const
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult
CcuComponent::GetCcuResourceSpaceTokenInfo(const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue) const
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult
CcuComponent::GetCcuResourceSpaceTokenInfoForLocal(const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue) const
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult
CcuComponent::AllocChannels(const uint8_t dieId, const ChannelPara& channelPara, std::vector<ChannelInfo>& channelInfos)
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuComponent::ConfigChannel(const uint8_t dieId, const ChannelCfg& cfg)
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuComponent::ReleaseChannel(const uint8_t dieId, const uint32_t channelId)
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuComponent::GetLoopChannelId(const uint8_t srcDieId, const uint8_t dstDieId, uint32_t& channelId) const
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuComponent::AllocRes(
    const uint8_t dieId, const ResType resType, const uint32_t num, const bool consecutive, vector<ResInfo>& resInfos)
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult
CcuComponent::ReleaseRes(const uint8_t dieId, const ResType resType, const uint32_t startId, const uint32_t num)
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuComponent::AllocIns(const uint8_t dieId, const uint32_t num, ResInfo& insInfo)
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuComponent::ReleaseIns(const uint8_t dieId, const ResInfo& insInfo)
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuComponent::AllocCke(const uint8_t dieId, const uint32_t num, vector<ResInfo>& ckeInfos)
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuComponent::ReleaseCke(const uint8_t dieId, const vector<ResInfo>& ckeInfos)
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuComponent::AllocXn(const uint8_t dieId, const uint32_t num, vector<ResInfo>& xnInfos)
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuComponent::ReleaseXn(const uint8_t dieId, const vector<ResInfo>& xnInfos)
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuComponent::CleanDieCkes(const uint8_t dieId) const { return HcclResult::HCCL_E_NOT_SUPPORT; }

HcclResult CcuComponent::SetTaskKill() { return HcclResult::HCCL_E_NOT_SUPPORT; }

HcclResult CcuComponent::SetTaskKillDone() { return HcclResult::HCCL_E_NOT_SUPPORT; }

HcclResult CcuComponent::CleanTaskKillState() const { return HcclResult::HCCL_E_NOT_SUPPORT; }

const std::array<bool, MAX_CCU_IODIE_NUM>& CcuComponent::GetDieEnableFlags() const { return dieEnableFlags; }

uint32_t CcuComponent::GetInsConsecutiveRemainSize(const uint8_t dieId) const { return 0; }

void CcuResBatchAllocator::Init() {}

void CcuResBatchAllocator::Deinit() {}

HcclResult CcuResBatchAllocator::AllocResHandle(const CcuResReq& resReq, CcuResHandle& resHandle)
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuResBatchAllocator::ReleaseResHandle(const CcuResHandle& handle) { return HcclResult::HCCL_E_NOT_SUPPORT; }

HcclResult CcuResBatchAllocator::GetResource(const CcuResHandle& handle, CcuResRepository& ccuResRepo)
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuResBatchAllocator::QueryRemainRes(uint8_t dieId, ResType resType, uint32_t& remainNum) const
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult CcuResBatchAllocator::GetAllocatableMaxBlockResNum(ResType resType, uint8_t dieId, uint32_t& num) const
{
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

void CtxMgrImp::Init() {}

void CtxMgrImp::Deinit() {}

} // namespace Hccl
