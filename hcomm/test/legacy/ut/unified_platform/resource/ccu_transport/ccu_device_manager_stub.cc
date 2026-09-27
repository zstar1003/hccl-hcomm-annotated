/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_device_manager.h"

namespace Hccl {

HcclResult CcuAllocChannels(
    const int32_t deviceLogicId, const CcuChannelPara& ccuChannelPara, std::vector<CcuChannelInfo>& ccuChannelInfos)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuReleaseChannel(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t ccuChannelId)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeviceManager::GetCcuResourceSpaceBufInfo(
    const int32_t deviceLogicId, const uint8_t dieId, uint64_t& addr, uint64_t& size)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeviceManager::GetCcuResourceSpaceTokenInfo(
    const int32_t deviceLogicId, const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeviceManager::ConfigChannel(const int32_t deviceLogicId, const uint8_t dieId, ChannelCfg& cfg)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeviceManager::GetLoopChannelId(
    const int32_t deviceLogicId, const uint8_t srcDieId, const uint8_t dstDieId, uint32_t& channIdx)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuDeviceManager::GetResource(const int32_t deviceLogicId, const CcuResHandle handle, CcuResRepository& ccuResRepo)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeviceManager::AllocResHandle(const int32_t deviceLogicId, const CcuResReq resReq, CcuResHandle& handle)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeviceManager::ReleaseResHandle(const int32_t deviceLogicId, const CcuResHandle handle)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuDeviceManager::AllocIns(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, ResInfo& insInfo)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeviceManager::ReleaseIns(const int32_t deviceLogicId, const uint8_t dieId, ResInfo& insInfo)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeviceManager::AllocCke(
    const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& ckeInfos)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuDeviceManager::ReleaseCke(const int32_t deviceLogicId, const uint8_t dieId, std::vector<ResInfo>& ckeInfos)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeviceManager::AllocXn(
    const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, vector<ResInfo>& xnInfos)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeviceManager::ReleaseXn(const int32_t deviceLogicId, const uint8_t dieId, vector<ResInfo>& xnInfos)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeviceManager::GetMissionKey(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& missionKey)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeviceManager::GetInstructionNum(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& instrNum)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeviceManager::GetXnBaseAddr(const uint32_t devLogicId, const uint8_t dieId, uint64_t& xnBaseAddr)
{
    return HcclResult::HCCL_SUCCESS;
}

std::string ResInfo::Describe() const { return StringFormat("ResInfo[startId=%u, num=%u]", startId, num); }

}; // namespace Hccl
