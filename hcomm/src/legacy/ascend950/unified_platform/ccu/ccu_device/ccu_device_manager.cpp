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

#include "hccl_common_v2.h"
#include "rdma_handle_manager.h"

#include "ccu_component.h"
#include "ccu_res_specs_legacy.h"
#include "ccu_res_batch_allocator_legacy.h"

namespace Hccl {

HcclResult CcuAllocChannels(
    const int32_t deviceLogicId, const CcuChannelPara& ccuChannelPara, std::vector<CcuChannelInfo>& ccuChannelInfos)
{
    HCCL_INFO(
        "[%s] new allocation request: deviceLogicId[%d], ipAddr[%s], "
        "channelnum[%u], jettyNum[%u], sqSize[%u].",
        __func__, deviceLogicId, ccuChannelPara.ipAddr.Describe().c_str(), ccuChannelPara.channelNum,
        ccuChannelPara.jettyNum, ccuChannelPara.sqSize);
    // 入参校验拦截
    CHK_PRT_RET(
        (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR(
            "[CcuAllocChannels]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId,
            MAX_MODULE_DEVICE_NUM),
        HcclResult::HCCL_E_PARA);
    TRY_CATCH_RETURN(
        const uint32_t devPhyId = HrtGetDevicePhyIdByUserDevId(deviceLogicId);
        auto& rdmaHandleMgr = RdmaHandleManager::GetInstance();
        const RdmaHandle rdmaHandle = rdmaHandleMgr.GetByIp(devPhyId, ccuChannelPara.ipAddr);
        const auto& dieIdAndFuncId = rdmaHandleMgr.GetDieAndFuncId(rdmaHandle);
        const uint8_t dieId = dieIdAndFuncId.first; ChannelPara para{}; // TRY_CATCH_RETURN 宏内不能直接在{}传参
        para.feId = dieIdAndFuncId.second; para.jettyNum = ccuChannelPara.jettyNum; para.sqSize = ccuChannelPara.sqSize;
        return CcuComponent::GetInstance(deviceLogicId).AllocChannels(dieId, para, ccuChannelInfos););
}

HcclResult CcuReleaseChannel(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t ccuChannelId)
{
    HCCL_INFO(
        "[%s] new release request: deviceLogicId[%d], dieId[%u], "
        "ccuChannelId[%u].",
        __func__, deviceLogicId, dieId, ccuChannelId);
    // 入参校验拦截
    CHK_PRT_RET(
        (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR(
            "[CcuReleaseChannel]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId,
            MAX_MODULE_DEVICE_NUM),
        HcclResult::HCCL_E_PARA);
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId).ReleaseChannel(dieId, ccuChannelId););
}

HcclResult CcuGetChannelSpecNum(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& channelNum)
{
    HCCL_INFO(
        "[CcuGetChannelSpecNum] Input params: deviceLogicId[%d], dieId[%u], channelNum[%u]", deviceLogicId, dieId,
        channelNum);
    // 入参校验拦截
    CHK_PRT_RET(
        (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR(
            "[CcuGetChannelSpecNum]deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId,
            MAX_MODULE_DEVICE_NUM),
        HcclResult::HCCL_E_PARA);
    TRY_CATCH_RETURN(return CcuResSpecifications::GetInstance(deviceLogicId).GetChannelNum(dieId, channelNum););
}

bool CcuIsInited(const int32_t deviceLogicId)
{
    HCCL_INFO("[CcuIsInited] Input params: deviceLogicId[%d]", deviceLogicId);
    CHK_PRT_RET(
        (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR(
            "[CcuIsInited] deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
        false);
    return CcuComponent::GetInstance(deviceLogicId).IsInited();
}

HcclResult CcuSetTaskKill(const int32_t deviceLogicId)
{
    HCCL_INFO("[CcuSetTaskKill] Input params: deviceLogicId[%d]", deviceLogicId);
    // 入参校验拦截
    CHK_PRT_RET(
        (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR(
            "[CcuSetTaskKill] deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId,
            MAX_MODULE_DEVICE_NUM),
        HcclResult::HCCL_E_PARA);
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId).SetTaskKill(););
}

HcclResult CcuSetTaskKillDone(const int32_t deviceLogicId)
{
    HCCL_INFO("[CcuSetTaskKillDone] Input params: deviceLogicId[%d]", deviceLogicId);
    // 入参校验拦截
    CHK_PRT_RET(
        (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR(
            "[CcuSetTaskKillDone] deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId,
            MAX_MODULE_DEVICE_NUM),
        HcclResult::HCCL_E_PARA);
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId).SetTaskKillDone(););
}

HcclResult CcuCleanTaskKillState(const int32_t deviceLogicId)
{
    HCCL_INFO("[CcuCleanTaskKillState] Input params: deviceLogicId[%d]", deviceLogicId);
    // 入参校验拦截
    CHK_PRT_RET(
        (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR(
            "[CcuCleanTaskKillState] deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId,
            MAX_MODULE_DEVICE_NUM),
        HcclResult::HCCL_E_PARA);
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId).CleanTaskKillState(););
}

HcclResult CcuCleanDieCkes(const int32_t deviceLogicId, const uint8_t dieId)
{
    HCCL_INFO("[CcuCleanDieCkes] Input params: deviceLogicId[%d], dieId[%u]", deviceLogicId, dieId);
    // 入参校验拦截
    CHK_PRT_RET(
        (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR(
            "[CcuCleanDieCkes] deviceLogicId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId,
            MAX_MODULE_DEVICE_NUM),
        HcclResult::HCCL_E_PARA);
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId).CleanDieCkes(dieId););
}

HcclResult CcuDeviceManager::GetCcuVersion(const int32_t deviceLogicId, CcuVersion& ccuVersion)
{
    TRY_CATCH_RETURN(ccuVersion = CcuResSpecifications::GetInstance(deviceLogicId).GetCcuVersion(););

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuDeviceManager::GetCcuResourceSpaceBufInfo(
    const int32_t deviceLogicId, const uint8_t dieId, uint64_t& addr, uint64_t& size)
{
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId).GetCcuResourceSpaceBufInfo(dieId, addr, size););
}

HcclResult CcuDeviceManager::GetCcuResourceSpaceTokenInfo(
    const int32_t deviceLogicId, const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue)
{
    TRY_CATCH_RETURN(
        return CcuComponent::GetInstance(deviceLogicId).GetCcuResourceSpaceTokenInfo(dieId, tokenId, tokenValue););
}

HcclResult CcuDeviceManager::GetCcuResourceSpaceTokenInfoForLocal(
    const int32_t deviceLogicId, const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue)
{
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId)
                         .GetCcuResourceSpaceTokenInfoForLocal(dieId, tokenId, tokenValue););
}

HcclResult CcuDeviceManager::ConfigChannel(const int32_t deviceLogicId, const uint8_t dieId, ChannelCfg& cfg)
{
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId).ConfigChannel(dieId, cfg););
}

HcclResult CcuDeviceManager::GetLoopChannelId(
    const int32_t deviceLogicId, const uint8_t srcDieId, const uint8_t dstDieId, uint32_t& channIdx)
{
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId).GetLoopChannelId(srcDieId, dstDieId, channIdx););
}

HcclResult
CcuDeviceManager::GetResource(const int32_t deviceLogicId, const CcuResHandle handle, CcuResRepository& ccuResRepo)
{
    TRY_CATCH_RETURN(return CcuResBatchAllocator::GetInstance(deviceLogicId).GetResource(handle, ccuResRepo););
}

HcclResult CcuDeviceManager::AllocResHandle(const int32_t deviceLogicId, const CcuResReq resReq, CcuResHandle& handle)
{
    TRY_CATCH_RETURN(return CcuResBatchAllocator::GetInstance(deviceLogicId).AllocResHandle(resReq, handle););
}

HcclResult CcuDeviceManager::ReleaseResHandle(const int32_t deviceLogicId, const CcuResHandle handle)
{
    TRY_CATCH_RETURN(return CcuResBatchAllocator::GetInstance(deviceLogicId).ReleaseResHandle(handle););
}

HcclResult
CcuDeviceManager::AllocIns(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, ResInfo& insInfo)
{
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId).AllocIns(dieId, num, insInfo););
}

HcclResult CcuDeviceManager::ReleaseIns(const int32_t deviceLogicId, const uint8_t dieId, ResInfo& insInfo)
{
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId).ReleaseIns(dieId, insInfo););
}

HcclResult CcuDeviceManager::AllocCke(
    const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& ckeInfos)
{
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId).AllocCke(dieId, num, ckeInfos););
}

HcclResult
CcuDeviceManager::ReleaseCke(const int32_t deviceLogicId, const uint8_t dieId, std::vector<ResInfo>& ckeInfos)
{
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId).ReleaseCke(dieId, ckeInfos););
}

HcclResult CcuDeviceManager::AllocXn(
    const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, vector<ResInfo>& xnInfos)
{
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId).AllocXn(dieId, num, xnInfos););
}

HcclResult CcuDeviceManager::ReleaseXn(const int32_t deviceLogicId, const uint8_t dieId, vector<ResInfo>& xnInfos)
{
    TRY_CATCH_RETURN(return CcuComponent::GetInstance(deviceLogicId).ReleaseXn(dieId, xnInfos););
}

HcclResult CcuDeviceManager::GetMissionKey(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& missionKey)
{
    TRY_CATCH_RETURN(return CcuResSpecifications::GetInstance(deviceLogicId).GetMissionKey(dieId, missionKey););
}

HcclResult CcuDeviceManager::GetInstructionNum(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& instrNum)
{
    TRY_CATCH_RETURN(return CcuResSpecifications::GetInstance(deviceLogicId).GetInstructionNum(dieId, instrNum););
}

HcclResult CcuDeviceManager::GetXnBaseAddr(const uint32_t devLogicId, const uint8_t dieId, uint64_t& xnBaseAddr)
{
    TRY_CATCH_RETURN(return CcuResSpecifications::GetInstance(devLogicId).GetXnBaseAddr(dieId, xnBaseAddr););
}

HcclResult CheckDieValid(
    const char* funcName, const int32_t devLogicId, const uint8_t dieId,
    const std::array<bool, MAX_CCU_IODIE_NUM>& dieEnableFlags)
{
    CHK_PRT_RET(
        dieId >= MAX_CCU_IODIE_NUM,
        HCCL_ERROR(
            "[%s] failed, dieId[%u] is invalid, should be in [0-%u), devLogicId[%d].", funcName, dieId,
            MAX_CCU_IODIE_NUM, devLogicId),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        !dieEnableFlags[dieId],
        HCCL_WARNING("[%s] failed, dieId[%u] is disable, devLogicId[%d].", funcName, dieId, devLogicId),
        HcclResult::HCCL_E_PARA);

    return HcclResult::HCCL_SUCCESS;
}

}; // namespace Hccl
