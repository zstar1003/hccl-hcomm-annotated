/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_dev_mgr_mock.h"
#include "../hccl_api_base_test.h"

#define private public
#define protected public

#include "ccu_device_pub.h"
#include "ccu_dev_mgr_imp.h"

#include "ccu_comp.h"
#include "src/base_comm/resources/ccu/ccu_device/ccu_res_specs.h"
#include "src/base_comm/resources/ccu/ccu_device/ccu_res_batch_allocator.h"

#include "eid_info_mgr.h"

#undef protected
#undef private

namespace hcomm {

static CcuResult MockCcuGetDieEnableInfo(int32_t deviceLogicId, uint8_t dieId, bool& enableFlag)
{
    CHK_PRT_RET(
        dieId >= CCU_MAX_IODIE_NUM,
        HCCL_ERROR(
            "[%s] failed, dieId[%u] is invalid, shoudle be in [0-%u), devLogicId[%d].", __func__, dieId,
            CCU_MAX_IODIE_NUM, deviceLogicId),
        CcuResult::CCU_E_PARA);
    const auto& dieEnableFlags = CcuComponent::GetInstance(deviceLogicId).GetDieEnableFlags();

    enableFlag = dieEnableFlags[dieId];
    return CcuResult::CCU_SUCCESS;
}

static HcclResult MockAllocResHandle(const int32_t deviceLogicId, const CcuResReq resReq, CcuResHandle& handle)
{
    return CcuResBatchAllocator::GetInstance(deviceLogicId).AllocResHandle(resReq, handle);
}

static HcclResult
MockGetLoopChannelId(const int32_t deviceLogicId, const uint8_t srcDieId, const uint8_t dstDieId, uint32_t& channIdx)
{
    return CcuComponent::GetInstance(deviceLogicId).GetLoopChannelId(srcDieId, dstDieId, channIdx);
}

static HcclResult MockGetResource(const int32_t deviceLogicId, const CcuResHandle handle, CcuResRepository& ccuResRepo)
{
    return CcuResBatchAllocator::GetInstance(deviceLogicId).GetResource(handle, ccuResRepo);
}

static HcclResult MockReleaseResHandle(const int32_t deviceLogicId, const CcuResHandle handle)
{
    return CcuResBatchAllocator::GetInstance(deviceLogicId).ReleaseResHandle(handle);
}

static HcclResult
MockQueryRemainRes(const int32_t deviceLogicId, const uint8_t dieId, const ResType& internalType, uint32_t& remainNum)
{
    return CcuResBatchAllocator::GetInstance(deviceLogicId).QueryRemainRes(dieId, internalType, remainNum);
}

static HcclResult MockAllocIns(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, ResInfo& insInfo)
{
    return CcuComponent::GetInstance(deviceLogicId).AllocIns(dieId, num, insInfo);
}

static HcclResult MockReleaseIns(const int32_t deviceLogicId, const uint8_t dieId, const ResInfo& insInfo)
{
    return CcuComponent::GetInstance(deviceLogicId).ReleaseIns(dieId, insInfo);
}

static HcclResult
MockAllocCke(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& ckeInfos)
{
    return CcuComponent::GetInstance(deviceLogicId).AllocCke(dieId, num, ckeInfos);
}

static HcclResult MockReleaseCke(const int32_t deviceLogicId, const uint8_t dieId, const std::vector<ResInfo>& ckeInfos)
{
    return CcuComponent::GetInstance(deviceLogicId).ReleaseCke(dieId, ckeInfos);
}

static HcclResult
MockAllocXn(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& xnInfos)
{
    return CcuComponent::GetInstance(deviceLogicId).AllocXn(dieId, num, xnInfos);
}

static HcclResult MockReleaseXn(const int32_t deviceLogicId, const uint8_t dieId, const std::vector<ResInfo>& xnInfos)
{
    return CcuComponent::GetInstance(deviceLogicId).ReleaseXn(dieId, xnInfos);
}

static HcclResult MockGetMissionKey(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& missionKey)
{
    return CcuResSpecifications::GetInstance(deviceLogicId).GetMissionKey(dieId, missionKey);
}

static HcclResult MockGetInstructionNum(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& instrNum)
{
    return CcuResSpecifications::GetInstance(deviceLogicId).GetInstructionNum(dieId, instrNum);
}

static HcclResult MockGetLoopEngineNum(int32_t deviceLogicId, uint8_t dieId, uint32_t& num)
{
    return CcuResBatchAllocator::GetInstance(deviceLogicId).GetAllocatableMaxBlockResNum(ResType::LOOP, dieId, num);
}

static HcclResult MockGetMsNum(int32_t deviceLogicId, uint8_t dieId, uint32_t& num)
{
    return CcuResBatchAllocator::GetInstance(deviceLogicId).GetAllocatableMaxBlockResNum(ResType::MS, dieId, num);
}

static HcclResult MockGetCkeNum(int32_t deviceLogicId, uint8_t dieId, uint32_t& num)
{
    return CcuResBatchAllocator::GetInstance(deviceLogicId).GetAllocatableMaxBlockResNum(ResType::CKE, dieId, num);
}

static HcclResult MockGetXnNum(int32_t deviceLogicId, uint8_t dieId, uint32_t& num)
{
    return CcuResBatchAllocator::GetInstance(deviceLogicId).GetAllocatableMaxBlockResNum(ResType::XN, dieId, num);
}

static HcclResult MockGetGsaNum(int32_t deviceLogicId, uint8_t dieId, uint32_t& num)
{
    return CcuResBatchAllocator::GetInstance(deviceLogicId).GetAllocatableMaxBlockResNum(ResType::GSA, dieId, num);
}

static HcclResult MockGetMissionNum(int32_t deviceLogicId, uint8_t dieId, uint32_t& num)
{
    return CcuResSpecifications::GetInstance(deviceLogicId).GetMissionNum(dieId, num);
}

static HcclResult MockGetXnBaseAddr(const int32_t devLogicId, const uint8_t dieId, uint64_t& xnBaseAddr)
{
    return CcuResSpecifications::GetInstance(devLogicId).GetXnBaseAddr(dieId, xnBaseAddr);
}

static HcclResult MockConfigChannel(const int32_t deviceLogicId, const uint8_t dieId, ChannelCfg& cfg)
{
    return CcuComponent::GetInstance(deviceLogicId).ConfigChannel(dieId, cfg);
}

static HcclResult
MockGetCcuResourceSpaceBufInfo(const int32_t deviceLogicId, const uint8_t dieId, uint64_t& addr, uint64_t& size)
{
    return CcuComponent::GetInstance(deviceLogicId).GetCcuResourceSpaceBufInfo(dieId, addr, size);
}

static HcclResult MockGetCcuResourceSpaceTokenInfo(
    const int32_t deviceLogicId, const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue)
{
    return CcuComponent::GetInstance(deviceLogicId).GetCcuResourceSpaceTokenInfo(dieId, tokenId, tokenValue);
}

static HcclResult MockGetCcuVersion(const int32_t deviceLogicId, CcuVersion& ccuVersion)
{
    ccuVersion = CcuResSpecifications::GetInstance(deviceLogicId).GetCcuVersion();
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult MockCcuAllocChannels(
    const int32_t deviceLogicId, const CcuChannelPara& ccuChannelPara, std::vector<CcuChannelInfo>& ccuChannelInfos)
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(ccuChannelPara.commAddr, ipAddr)); // 为了打印信息暂时添加
    HCCL_INFO(
        "[%s] new allocation request: deviceLogicId[%d], ipAddr[%s], "
        "channelnum[%u], jettyNum[%u], sqSize[%u].",
        __func__, deviceLogicId, ipAddr.Describe().c_str(), ccuChannelPara.channelNum, ccuChannelPara.jettyNum,
        ccuChannelPara.sqSize);

    uint32_t devPhyId{0};
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devPhyId));

    DevEidInfo eidInfo{};
    CHK_RET(EidInfoMgr::GetInstance(devPhyId).GetEidInfoByAddr(ccuChannelPara.commAddr, eidInfo));
    const uint8_t dieId = static_cast<uint8_t>(eidInfo.dieId);
    const uint32_t feId = eidInfo.funcId;
    ChannelPara para{};
    para.feId = feId;
    para.jettyNum = ccuChannelPara.jettyNum;
    para.sqSize = ccuChannelPara.sqSize;

    return CcuComponent::GetInstance(deviceLogicId).AllocChannels(dieId, para, ccuChannelInfos);
}

HcclResult MockCcuReleaseChannel(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t ccuChannelId)
{
    return CcuComponent::GetInstance(deviceLogicId).ReleaseChannel(dieId, ccuChannelId);
}

} // namespace hcomm

void MockCcuDevMgr()
{
    MOCKER(hcomm::CcuGetDieEnableInfo).stubs().will(invoke(hcomm::MockCcuGetDieEnableInfo));
    MOCKER(hcomm::CcuDevMgrImp::AllocResHandle).stubs().will(invoke(hcomm::MockAllocResHandle));
    MOCKER(hcomm::CcuDevMgrImp::GetLoopChannelId).stubs().will(invoke(hcomm::MockGetLoopChannelId));
    MOCKER(hcomm::CcuDevMgrImp::GetResource).stubs().will(invoke(hcomm::MockGetResource));
    MOCKER(hcomm::CcuDevMgrImp::ReleaseResHandle).stubs().will(invoke(hcomm::MockReleaseResHandle));
    MOCKER(hcomm::CcuDevMgrImp::QueryRemainRes).stubs().will(invoke(hcomm::MockQueryRemainRes));
    MOCKER(hcomm::CcuDevMgrImp::AllocIns).stubs().will(invoke(hcomm::MockAllocIns));
    MOCKER(hcomm::CcuDevMgrImp::ReleaseIns).stubs().will(invoke(hcomm::MockReleaseIns));
    MOCKER(hcomm::CcuDevMgrImp::AllocCke).stubs().will(invoke(hcomm::MockAllocCke));
    MOCKER(hcomm::CcuDevMgrImp::ReleaseCke).stubs().will(invoke(hcomm::MockReleaseCke));
    MOCKER(hcomm::CcuDevMgrImp::AllocXn).stubs().will(invoke(hcomm::MockAllocXn));
    MOCKER(hcomm::CcuDevMgrImp::ReleaseXn).stubs().will(invoke(hcomm::MockReleaseXn));
    MOCKER(hcomm::CcuDevMgrImp::GetMissionKey).stubs().will(invoke(hcomm::MockGetMissionKey));
    MOCKER(hcomm::CcuDevMgrImp::GetResSpecsInstructionNum).stubs().will(invoke(hcomm::MockGetInstructionNum));
    MOCKER(hcomm::CcuDevMgrImp::GetAllocatableMaxLoopEngineNum).stubs().will(invoke(hcomm::MockGetLoopEngineNum));
    MOCKER(hcomm::CcuDevMgrImp::GetAllocatableMaxMsNum).stubs().will(invoke(hcomm::MockGetMsNum));
    MOCKER(hcomm::CcuDevMgrImp::GetAllocatableMaxCkeNum).stubs().will(invoke(hcomm::MockGetCkeNum));
    MOCKER(hcomm::CcuDevMgrImp::GetAllocatableMaxXnNum).stubs().will(invoke(hcomm::MockGetXnNum));
    MOCKER(hcomm::CcuDevMgrImp::GetAllocatableMaxGsaNum).stubs().will(invoke(hcomm::MockGetGsaNum));
    MOCKER(hcomm::CcuDevMgrImp::GetResSpecsMissionNum).stubs().will(invoke(hcomm::MockGetMissionNum));
    MOCKER(hcomm::CcuDevMgrImp::GetXnBaseAddr).stubs().will(invoke(hcomm::MockGetXnBaseAddr));
    MOCKER(hcomm::CcuDevMgrImp::ConfigChannel).stubs().will(invoke(hcomm::MockConfigChannel));
    MOCKER(hcomm::CcuDevMgrImp::GetCcuResourceSpaceBufInfo).stubs().will(invoke(hcomm::MockGetCcuResourceSpaceBufInfo));
    MOCKER(hcomm::CcuDevMgrImp::GetCcuResourceSpaceTokenInfo)
        .stubs()
        .will(invoke(hcomm::MockGetCcuResourceSpaceTokenInfo));
    MOCKER(hcomm::CcuDevMgrImp::GetCcuVersion).stubs().will(invoke(hcomm::MockGetCcuVersion));
    MOCKER(hcomm::CcuAllocChannels).stubs().will(invoke(hcomm::MockCcuAllocChannels));
    MOCKER(hcomm::CcuReleaseChannel).stubs().will(invoke(hcomm::MockCcuReleaseChannel));
}
