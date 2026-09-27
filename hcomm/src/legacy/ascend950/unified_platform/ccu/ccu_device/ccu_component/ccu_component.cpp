/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>

#include "ccu_component.h"

#include "exception_util.h"
#include "hccl_common_v2.h"
#include "ccu_api_exception.h"
#include "orion_adapter_rts.h"
#include "internal_exception.h"
#include "rdma_handle_manager.h"

#include "ccu_eid_info.h"
#include "ccu_res_specs_legacy.h"

#include "ccu_channel_mgr_v1.h"
#include "hccp_ctx_tp.h"
#include "env_config_v2.h"
#include "hccp_tlv_hdc_manager.h"

namespace Hccl {

namespace {
    constexpr uint8_t CCU_MAX_MISSION_NUM = 16;
} // namespace

constexpr uint16_t INVAILD_LOOP_CHANNEL_ID = 0xFFFF;

// 设置为0，分配数量由channelMgr决定，v1 默认1个
constexpr uint32_t LOOP_CHANNEL_USE_JETTY = 0;
constexpr uint32_t LOOP_CHANNEL_USE_SQSIZE = 16;

// 环回获取TP信息超时等待10s
constexpr uint32_t LOOP_CHANNEL_WAIT_TIMEOUT_MS = 10000;
// 环回获取TP信息间隔1ms
constexpr u32 ONE_MILLISECOND_OF_USLEEP = 1000;

// 清理CKE批量申请大小
constexpr u32 MAX_CKE_DATA_ARRAY_SIZE = 8;

// 环境是A+X时，配置die0的MS交织粒度为1<<7 = 128
constexpr uint32_t MSID_CONFIG_AX_MAINBOARD = 7;
constexpr TpProtocol LOOP_JETTY_PROTOCOL = TpProtocol::TP; // 环回使用TP避免被环境link down阻塞

// 与 TpManager 缓存键一致：须与 RequestNewTpInfo / Deinit::ReleaseTpInfo 使用同一套 qos 与环回标志
static RaUbGetTpInfoParam MakeCcuLoopRaUbGetTpInfoParam(const IpAddress& locAddr, const IpAddress& rmtAddr)
{
    RaUbGetTpInfoParam p{};
    p.locAddr = locAddr;
    p.rmtAddr = rmtAddr;
    p.tpProtocol = LOOP_JETTY_PROTOCOL;
    p.qos = 0U;
    p.slLevelCount = 0U;
    p.loopFirstTpLowestSl = true;
    p.ccuLoopbackGetTpInfo = true;
    return p;
}

CcuComponent& CcuComponent::GetInstance(const int32_t deviceLogicId)
{
    static CcuComponent ccuComponent[MAX_MODULE_DEVICE_NUM + 1];

    if (deviceLogicId < 0 || static_cast<uint32_t>(deviceLogicId) > MAX_MODULE_DEVICE_NUM) {
        THROW<InvalidParamsException>(
            "[CcuComponent][%s] failed, devLogicId[%d] should be less "
            "than %u.",
            __func__, deviceLogicId, MAX_MODULE_DEVICE_NUM);
    }

    ccuComponent[deviceLogicId].devLogicId = deviceLogicId;
    return ccuComponent[deviceLogicId];
}

void CcuComponent::PrintCcuMissionStatus(int32_t devLogicId) const
{
    uint16_t status = 0;
    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        if (!dieEnableFlags[dieId]) {
            HCCL_WARNING("[%s]devLogicId[%d], dieId[%u] is not enable, skip.", __func__, devLogicId, dieId);
            continue;
        }
        std::string missionStatus;
        for (uint8_t missionId = 0; missionId < CCU_MAX_MISSION_NUM; missionId++) {
            status = Hccl::CcuErrorHandler::GetCcuMissionContext(devLogicId, dieId, missionId).GetStatus();
            missionStatus += "missionId[" + std::to_string(missionId) + "]=status[" + std::to_string(status) + "] ";
        }
        HCCL_RUN_INFO("Init devLogicId[%d], dieId[%d], content[%s]", devLogicId, dieId, missionStatus.c_str());
    }
}

void CcuComponent::Init()
{
    std::unique_lock<std::shared_mutex> _lock(innerMutex);

    if (ifInit) {
        return;
    }

    devPhyId = HrtGetDevicePhyIdByUserDevId(devLogicId);
    CheckDiesEnable();
    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        CleanDieCkes(dieId);
    }
    CreateCcuRmaBuffer();
    CreateResourceManagers();
    CreateLoopChannels();
    ConfigMsIdToken();
    // 打印最初的mission状态
    PrintCcuMissionStatus(devLogicId);
    SetTaskKill();
    SetTaskKillDone();
    // 打印taskKill恢复后的mission状态
    PrintCcuMissionStatus(devLogicId);
    ifInit = true;
}
// 资源清理
void CcuComponent::Deinit()
{
    std::unique_lock<std::shared_mutex> _lock(innerMutex);
    ReleaseJettyRes();

    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        CleanDieCkes(dieId);
    }

    for (const auto& item : tpAttrInfoMap) {
        const auto& ipAddr = item.first;
        const auto& tpAttrInfo = item.second;
        const auto& tpInfoIter = tpInfoMap.find(ipAddr);
        if (tpInfoIter != tpInfoMap.end() && tpInfoIter->second.tpHandle != 0) {
            (void)TpManager::GetInstance(devLogicId).ReleaseTpAttr(tpInfoIter->second.tpHandle, tpAttrInfo);
        }
    }
    tpAttrInfoMap.clear();

    for (const auto& item : tpInfoMap) {
        const auto& ipAddr = item.first;
        const auto& tpInfo = item.second;
        (void)TpManager::GetInstance(devLogicId).ReleaseTpInfo(MakeCcuLoopRaUbGetTpInfoParam(ipAddr, ipAddr), tpInfo);
    }

    createdOutParamMap.clear();
    importedOutParamMap.clear();
    tpInfoMap.clear();
    psnMap.clear();

    loopFeIpAddrMap.clear();
    ccuRmaBufferMap.clear();
    localCcuRmaBufferMap.clear();
    additionalCcuRmaBufferMap.clear();
    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        channelMgrs[dieId] = nullptr;
        resAllocators[dieId] = nullptr;
        loopChannelIds[dieId] = INVAILD_LOOP_CHANNEL_ID;
    }

    ifInit = false;
}

void CcuComponent::CheckDiesEnable()
{
    ccuVersion = CcuResSpecifications::GetInstance(devLogicId).GetCcuVersion();
    HCCL_INFO(
        "[CcuComponent][%s] ccu version[%s], devLogicId[%d].", __func__, ccuVersion.Describe().c_str(), devLogicId);

    std::array<bool, MAX_CCU_IODIE_NUM> dieDrvEnableFlags{false, false};
    bool allDieDisable = true;
    const auto& ccuResSpecs = CcuResSpecifications::GetInstance(devLogicId);
    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        dieEnableFlags[dieId] = false;
        (void)ccuResSpecs.GetDieEnableFlag(dieId, dieDrvEnableFlags[dieId]);
        ChooseLoopEid(dieDrvEnableFlags[dieId], dieId);
        allDieDisable = allDieDisable && !dieEnableFlags[dieId];
        if (!dieEnableFlags[dieId]) { // 调用接口失败时不会改变dieEnableFlags[i]
            HCCL_WARNING("[CcuComponent][%s] devLogicId[%d], dieId[%u] is not usable.", __func__, devLogicId, dieId);
            continue;
        }

        HCCL_INFO("[CcuComponent][%s] devLogicId[%d] die[%u] is usable.", __func__, devLogicId, dieId);
    }

    if (allDieDisable) {
        THROW<CcuApiException>(
            "[CcuComponent][%s] failed, because all dies are "
            "disabled, devLogicId[%d].",
            __func__, devLogicId);
    }
}

static HcclResult FindOneUsableEid(const uint32_t devLogicId, const uint8_t dieId, uint32_t& feId, IpAddress& ipAddr)
{
    std::vector<HrtDevEidInfo> eidInfoList;
    auto ret = CcuEidInfo::GetInstance(devLogicId).GetEidInfo(devLogicId, eidInfoList);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_WARNING("[CcuComponent][%s] failed, devLogicId[%u], dieId[%u].", __func__, devLogicId, dieId), ret);

    std::string name;
    bool findFlag = false;
    u32 devPhyId = HrtGetDevicePhyIdByUserDevId(devLogicId);

    // 如果无法查询设备是否为uboe设备，报错退出
    CHK_RET(HrtGetUboeFlagEnable(devPhyId));

    auto& rdmaHandleMgr = RdmaHandleManager::GetInstance();
    // 当前结论，需要选择可以申请到Tp handle的eid
    for (auto& eidInfo : eidInfoList) {
        // 如果是UBOE设备，则跳过
        if (HrtCheckUboeSupported(eidInfo.devFeature) || (eidInfo.dieId != dieId)) {
            continue;
        }
        const RdmaHandle rdmaHandle = rdmaHandleMgr.GetByIp(devPhyId, eidInfo.ipAddress);
        const bool rtpEnable = rdmaHandleMgr.GetRtpEnable(rdmaHandle);
        if (rtpEnable) {
            feId = eidInfo.funcId;
            ipAddr = eidInfo.ipAddress;
            name = eidInfo.name;
            HCCL_RUN_INFO(
                "[%s] rtpEnable[%d] dieId[%u] choose:"
                "name[%s] feId[%u] ipAddr[%s], devLogicId[%u]",
                __func__, rtpEnable, dieId, name.c_str(), feId, ipAddr.Describe().c_str(), devLogicId);
            findFlag = true;
            break;
        }
    }

    if (!findFlag) {
        HCCL_RUN_INFO(
            "[CcuComponent][%s] dieId[%u] doesn't have usable func ID, "
            "devLogicId[%u].",
            __func__, dieId, devLogicId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    HCCL_INFO(
        "[CcuComponent][%s] dieId[%u] choose: name[%s] feId[%u] ipAddr[%s], "
        "devLogicId[%u].",
        __func__, dieId, name.c_str(), feId, ipAddr.Describe().c_str(), devLogicId);

    return HcclResult::HCCL_SUCCESS;
}

void CcuComponent::ChooseLoopEid(bool& dieDrvEnableFlag, uint8_t dieId)
{
    if (!dieDrvEnableFlag) {
        return;
    }

    uint32_t feId = 0;
    IpAddress ipAddr = IpAddress();
    if (FindOneUsableEid(devLogicId, dieId, feId, ipAddr) != HcclResult::HCCL_SUCCESS) {
        HCCL_WARNING(
            "[CcuComponent][%s] failed to find feId eid, but passed, "
            "devLogicId[%d], dieId[%u].",
            __func__, devLogicId, dieId);
        return;
    }

    loopFeIpAddrMap[dieId] = {feId, ipAddr};
    dieEnableFlags[dieId] = dieDrvEnableFlag;
    HCCL_INFO("[CcuComponent][%s] die[%u] is enable", __func__, dieId);
}

HcclResult CcuComponent::GetLoopFeIpByDieId(const uint8_t dieId, uint32_t& feId, IpAddress& ipAddr)
{
    const auto& dieIter = loopFeIpAddrMap.find(dieId);
    CHK_PRT_RET(
        dieIter == loopFeIpAddrMap.end(),
        HCCL_WARNING(
            "[CcuComponent][%s] failed, dieId[%u] doesn't have usable loop feId, "
            "devLogicId[%d].",
            __func__, dieId, devLogicId),
        HcclResult::HCCL_E_NOT_FOUND);

    const auto& feIdIpAddr = dieIter->second;
    feId = feIdIpAddr.first;
    ipAddr = feIdIpAddr.second;

    return HcclResult::HCCL_SUCCESS;
}

void CcuComponent::CreateCcuRmaBuffer()
{
    auto& ccuResSpecs = CcuResSpecifications::GetInstance(devLogicId);
    auto& rdmaHandleMgr = RdmaHandleManager::GetInstance();
    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        if (!dieEnableFlags[dieId]) {
            continue;
        }

        uint32_t feId = 0;
        IpAddress ipAddr{};
        if (GetLoopFeIpByDieId(dieId, feId, ipAddr) != HcclResult::HCCL_SUCCESS) {
            continue;
        }

        uint64_t ccuResAddr = 0;
        (void)ccuResSpecs.GetResourceAddr(dieId, ccuResAddr);
        if (ccuResAddr == 0) {
            HCCL_WARNING(
                "[CcuComponent][%s] failed, ccu resource space address[0] is invalid, "
                "devLogicId[%d] dieId[%u]",
                __func__, devLogicId, dieId);
            continue;
        }

        const auto rdmaHandle = rdmaHandleMgr.GetByIp(devPhyId, ipAddr);
        CHECK_NULLPTR(
            rdmaHandle, StringFormat(
                            "[CcuComponent][%s] failed, rdmaHandle is nullptr, "
                            "devLogicId[%d] dieId[%u]",
                            __func__, devLogicId, dieId));

        std::array<CcuMemInfo, CCU_MEM_INFO_SIZE> memInfoList{};
        uint32_t count{0};
        ccuResSpecs.GetCcuMemInfoList(dieId, memInfoList.data(), count);
        for (uint32_t i = 0; i < count; i++) {
            if (memInfoList[i].memVa == ccuResAddr) {
                const auto ccuBuffer = std::make_shared<Buffer>(ccuResAddr, memInfoList[i].memSize);
                ccuRmaBufferMap.emplace(dieId, std::make_unique<LocalUbRmaBuffer>(ccuBuffer, rdmaHandle));
            } else {
                const auto ccuBuffer = std::make_shared<Buffer>(memInfoList[i].memVa, memInfoList[i].memSize);
                additionalCcuRmaBufferMap.emplace_back(std::make_unique<LocalUbRmaBuffer>(ccuBuffer, rdmaHandle));
            }
        }
        const auto ccuBuffer = std::make_shared<Buffer>(ccuResAddr, CCU_RESOURCE_SIZE);
        // 本端专用的buffer，具有整块内存的权限
        localCcuRmaBufferMap.emplace(dieId, std::make_unique<LocalUbRmaBuffer>(ccuBuffer, rdmaHandle));
    }
}

inline std::unique_ptr<CcuChannelMgr> CreateChannelMgrByVersion(
    const CcuVersion version, const uint32_t devLogicId, const uint8_t dieId, const uint32_t devPhyId)
{
    switch (version) {
        case CcuVersion::CCU_V1:
            return std::make_unique<CcuChannelMgrV1>(devLogicId, dieId, devPhyId);
        default:
            break;
    }

    return nullptr;
}

void CcuComponent::CreateResourceManagers()
{
    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        if (!dieEnableFlags[dieId]) {
            continue;
        }

        std::unique_ptr<CcuChannelMgr> channelMgrPtr
            = CreateChannelMgrByVersion(ccuVersion, devLogicId, dieId, devPhyId);
        CHECK_NULLPTR(
            channelMgrPtr, StringFormat(
                               "[CcuComponent][%s] failed, ccu driver version[%s] is not expected, "
                               "devLogicId[%d] dieId[%u].",
                               __func__, ccuVersion.Describe().c_str(), devLogicId, dieId));

        channelMgrs[dieId] = std::move(channelMgrPtr);
        resAllocators[dieId] = std::make_unique<CcuResAllocator>(devLogicId, dieId);
    }
}

void CcuComponent::CreateLoopChannels()
{
    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        loopChannelIds[dieId] = INVAILD_LOOP_CHANNEL_ID;
    }

    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        // 失败抛异常处理，jetty资源跟随数据结构析构释放
        CHK_RET_THROW(
            InternalException,
            StringFormat("[CcuComponent][%s] failed, devLogicId[%d], dieId[%u].", __func__, devLogicId, dieId),
            CreateLoopChannel(dieId, loopChannelIds[dieId]));

        HCCL_INFO(
            "[CcuComponent][%s] succeed, loop channel id[%u], "
            "devLogicId[%d], dieId[%u].",
            __func__, loopChannelIds[dieId], devLogicId, dieId);
    }
}

HcclResult CcuComponent::CreateLoopChannel(const uint8_t dieId, uint32_t& channelId)
{
    if (!dieEnableFlags[dieId]) {
        HCCL_WARNING(
            "[CcuComponent][%s] passed, dieId[%u] is not enabled, skip. "
            "devLogicId[%d].",
            __func__, dieId, devLogicId);
        return HcclResult::HCCL_SUCCESS;
    }

    // 对于单p或单die场景，可能设备或die不会配置eid，按成功处理不阻塞用例
    uint32_t feId = 0;
    IpAddress ipAddr{};
    if (GetLoopFeIpByDieId(dieId, feId, ipAddr) != HcclResult::HCCL_SUCCESS) {
        channelId = INVAILD_LOOP_CHANNEL_ID;
        HCCL_WARNING(
            "[CcuComponent][%s] failed but passed, dieId[%u] doesn't have loop feId, "
            "devLogicId[%d].",
            __func__, dieId, devLogicId);
        return HcclResult::HCCL_SUCCESS;
    }

    std::vector<ChannelInfo> channelInfos; // 按jetty组分配
    const ChannelPara channelPara{feId, LOOP_CHANNEL_USE_JETTY, LOOP_CHANNEL_USE_SQSIZE};
    auto ret = channelMgrs[dieId]->Alloc(channelPara, channelInfos);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuComponent][%s] failed to alloc channel, "
            "devLogicId[%d], dieId[%u].",
            __func__, devLogicId, dieId),
        ret);

    const auto& channelInfo = channelInfos[0]; // 环回只使用1个channel
    ret = CreateAndImportLoopJettys(dieId, ipAddr, channelInfo.jettyInfos);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuComponent][%s] failed to create or import loop jettys, "
            "devLogicId[%d], dieId[%u].",
            __func__, devLogicId, dieId),
        ret);

    ret = ConfigLoopChannel(dieId, ipAddr, channelInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuComponent][%s] failed to config the loop channel, "
            "devLogicId[%d], dieId[%u].",
            __func__, devLogicId, dieId),
        ret);

    channelId = channelInfo.channelId;
    return HcclResult::HCCL_SUCCESS;
}

JettyImportCfg GetJettyImportCfg(const TpInfo& tpInfo, const uint32_t& psn)
{
    const TpHandle tpHandle = tpInfo.tpHandle;
    HCCL_INFO("[CcuComponent][%s] loop channel use tp handle[%llu] psn[%u].", __func__, tpHandle, psn);

    JettyImportCfg cfg = {};
    cfg.localTpHandle = tpHandle;
    cfg.remoteTpHandle = tpHandle;
    cfg.localPsn = psn;
    cfg.remotePsn = psn;
    cfg.protocol = LOOP_JETTY_PROTOCOL;
    return cfg;
}

HcclResult CcuComponent::CreateAndImportLoopJettys(
    const uint8_t dieId, const IpAddress& ipAddr, const vector<JettyInfo>& jettyInfos)
{
    Hccl::CqCreateInfo cqInfo{};
    auto& rdmaHandleMgr = RdmaHandleManager::GetInstance();
    const auto rdmaHandle = rdmaHandleMgr.GetByIp(devPhyId, ipAddr);
    const auto jfcHandle = rdmaHandleMgr.GetJfcHandle(rdmaHandle, cqInfo, HrtUbJfcMode::CCU_POLL);

    const auto& rmaBufferIter = localCcuRmaBufferMap.find(dieId);
    CHK_PRT_RET(
        rmaBufferIter == localCcuRmaBufferMap.end(),
        HCCL_WARNING(
            "[CcuComponent][%s] failed, ccu rma buffer of die[%u] is not existed, "
            "devLogicId[%d].",
            __func__, dieId, devLogicId),
        HcclResult::HCCL_E_NOT_FOUND);

    const auto& ccuRmaBuffer = rmaBufferIter->second;
    const auto ccuBufTokenValue = ccuRmaBuffer->GetTokenValue();

    const auto& tpInfo = GetTpInfo(ipAddr);
    uint8_t errTimeout = 0;
    CHK_RET(GetLoopJettyTimeout(ipAddr, tpInfo.tpHandle, errTimeout));

    auto& createdVec = createdOutParamMap[dieId];
    auto& importedVec = importedOutParamMap[dieId];

    // 与 Next 对齐：环回 Jetty qos 直接用 GetTpInfo(loopFirstTpLowestSl) 映射的 mappedJettyPriority
    const uint8_t loopJettyQos = tpInfo.hasMappedJettyPriority ?
                                     static_cast<uint8_t>(tpInfo.mappedJettyPriority & 0xFU) :
                                     static_cast<uint8_t>(UB_QOS_DEFAULT);

    for (const auto& jettyInfo : jettyInfos) {
        const auto jettyMode = HrtJettyMode::CCU_CCUM_CACHE; // 当前仅支持该模式
        HrtRaUbCreateJettyParam req{
            jfcHandle,
            jfcHandle,
            ccuBufTokenValue,
            0,
            jettyMode,
            jettyInfo.taJettyId,
            jettyInfo.sqBufVa,
            jettyInfo.sqBufSize,
            jettyInfo.wqeBBStartId,
            jettyInfo.sqDepth,
            errTimeout};
        req.qos = loopJettyQos;
        auto createdOutParam = HrtRaUbCreateJetty(rdmaHandle, req);
        createdVec.emplace_back(createdOutParam);

        const auto psn = GetPsn(ipAddr);
        const auto jettyImportCfg = GetJettyImportCfg(tpInfo, psn);
        const auto importedOutParam = RaUbTpImportJetty(
            rdmaHandle, createdOutParam.key, createdOutParam.keySize, ccuBufTokenValue, jettyImportCfg);
        importedVec.emplace_back(ImportOutParamPair{rdmaHandle, importedOutParam});
    }

    return HcclResult::HCCL_SUCCESS;
}

TpInfo CcuComponent::RequestNewTpInfo(const IpAddress& srcIpAddr, const IpAddress& dstIpAddr) const
{
    TpInfo tpInfo{};

    auto& tpManager = TpManager::GetInstance(devLogicId);
    // 与 Next `MakeLoopGetTpInfoParam` 对齐：环回与通信域 hcclQos 解耦；SL 由 GetTpAttr.slBitmap + loopFirstTpLowestSl
    // 决定
    const RaUbGetTpInfoParam loopParam = MakeCcuLoopRaUbGetTpInfoParam(srcIpAddr, dstIpAddr);

    const auto timeout = std::chrono::milliseconds(LOOP_CHANNEL_WAIT_TIMEOUT_MS);
    const auto startTime = std::chrono::steady_clock::now();
    auto ret = tpManager.GetTpInfo(loopParam, tpInfo);
    while (ret == HcclResult::HCCL_E_AGAIN) {
        ret = tpManager.GetTpInfo(loopParam, tpInfo);
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            THROW<InternalException>(
                "[CcuComponent][%s] failed, get tp info "
                "timeout[%d ms], devLogicId[%d].",
                __func__, timeout, devLogicId);
        }
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<InternalException>(
            "[CcuComponent][%s] failed, ret[%d], "
            "devLogicId[%d].",
            __func__, static_cast<int>(ret), devLogicId);
    }

    return tpInfo;
}

TpInfo CcuComponent::GetTpInfo(const IpAddress& ipAddr)
{
    const auto& srcIter = tpInfoMap.find(ipAddr);
    // 优先使用已经创建过的tpHandle
    if (srcIter == tpInfoMap.end()) {
        const auto& tpInfo = RequestNewTpInfo(ipAddr, ipAddr);
        tpInfoMap[ipAddr] = tpInfo;
        return tpInfo;
    }

    return srcIter->second;
}

TpAttrInfo CcuComponent::GetLoopTpAttr(const IpAddress& ipAddr, const TpHandle tpHandle)
{
    const auto& srcIter = tpAttrInfoMap.find(ipAddr);
    if (srcIter != tpAttrInfoMap.end()) {
        return srcIter->second;
    }

    auto& rdmaHandleMgr = RdmaHandleManager::GetInstance();
    const auto rdmaHandle = rdmaHandleMgr.GetByIp(devPhyId, ipAddr);

    constexpr uint32_t kTpAttrRetryTimesInitBit = 0U;
    constexpr uint32_t kTpAttrAtBit = 1U;
    constexpr uint32_t TP_ATTR_BITMAP = (1U << kTpAttrRetryTimesInitBit) | (1U << kTpAttrAtBit);
    const GetTpAttrParam tpAttrParam = {tpHandle, TP_ATTR_BITMAP};

    TpAttrInfo tpAttrInfo{};
    auto& tpMgr = TpManager::GetInstance(devLogicId);
    const auto timeout = std::chrono::milliseconds(LOOP_CHANNEL_WAIT_TIMEOUT_MS);
    const auto startTime = std::chrono::steady_clock::now();

    HcclResult ret = tpMgr.GetTpAttr(tpAttrParam, tpAttrInfo, rdmaHandle);
    while (ret == HcclResult::HCCL_E_AGAIN) {
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            THROW<InternalException>(
                "[CcuComponent][%s] failed, get tp attr "
                "timeout[%d ms], devLogicId[%d].",
                __func__, timeout, devLogicId);
        }
        ret = tpMgr.GetTpAttr(tpAttrParam, tpAttrInfo, rdmaHandle);
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<InternalException>(
            "[CcuComponent][%s] failed, ret[%u], "
            "devLogicId[%d].",
            __func__, ret, devLogicId);
    }

    tpAttrInfoMap[ipAddr] = tpAttrInfo;
    return tpAttrInfo;
}

HcclResult CcuComponent::GetLoopJettyTimeout(const IpAddress& ipAddr, const TpHandle tpHandle, uint8_t& errTimeout)
{
    const auto tpAttrInfo = GetLoopTpAttr(ipAddr, tpHandle);
    // CTP 协议不感知 TP 建链，跳过 GetTpTotalTimeout（对齐 DevUbConnection::GetTimeOut 的 CTP 分支），
    // tpTimeOutMs 保持 0，由 TpManager::CalcTaTimeout 内部按 CTP 规则直接使用 taTimeOut_
    uint32_t tpTimeOutMs = 0;
    if (LOOP_JETTY_PROTOCOL != TpProtocol::CTP) {
        CHK_RET(TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs));
    }
    errTimeout = TpManager::CalcTaTimeout(LOOP_JETTY_PROTOCOL, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs);
    return HcclResult::HCCL_SUCCESS;
}

inline uint32_t GetRandomNum()
{
    uint32_t randNum = std::rand();
    return randNum;
}

uint32_t CcuComponent::GetPsn(const IpAddress& ipAddr)
{
    const auto& srcIter = psnMap.find(ipAddr);
    if (srcIter == psnMap.end()) {
        const auto psn = GetRandomNum();
        psnMap[ipAddr] = psn;
        return psn;
    }

    return srcIter->second;
}

HcclResult CcuComponent::ConfigLoopChannel(const uint8_t dieId, const IpAddress& ipAddr, const ChannelInfo& channelInfo)
{
    HCCL_INFO("[CcuComponent][%s] Create loop channel with another die's address, my dieId[%u]", __func__, dieId);
    auto rmaBufferIter = ccuRmaBufferMap.find(1 - dieId); // 需要配置另一die的rma buffer
    if (rmaBufferIter == ccuRmaBufferMap.end()) {
        HCCL_WARNING(
            "[CcuComponent][%s] Another die is not enable, create loop channel with my die[%u]", __func__, dieId);
        rmaBufferIter = ccuRmaBufferMap.find(dieId);
    }
    CHK_PRT_RET(
        rmaBufferIter == ccuRmaBufferMap.end(),
        HCCL_WARNING(
            "[CcuComponent][%s] failed, ccu rma buffer of die[%u] is not existed, "
            "devLogicId[%d].",
            __func__, dieId, devLogicId),
        HcclResult::HCCL_E_NOT_FOUND);

    const auto& ccuRmaBuffer = rmaBufferIter->second;
    const auto ccuBufTokenValue = ccuRmaBuffer->GetTokenValue();

    ChannelCfg cfg{};
    cfg.channelId = channelInfo.channelId;
    cfg.remoteEid = ipAddr.GetReverseEid();
    HCCL_INFO("[CcuComponent::ConfigLoopChannel] remoteEid=%s", cfg.remoteEid.Describe().c_str());
    cfg.tpn = importedOutParamMap[dieId][0].second.tpn;

    cfg.remoteCcuVa = ccuRmaBuffer->GetBuf()->GetAddr();
    cfg.memTokenId = ccuRmaBuffer->GetTokenId();
    cfg.memTokenValue = ccuBufTokenValue;

    const auto& jettyInfos = channelInfo.jettyInfos;
    const auto& createdVec = createdOutParamMap[dieId];
    const uint32_t jettyNum = jettyInfos.size();
    for (uint32_t i = 0; i < jettyNum; i++) {
        cfg.jettyCfgs.emplace_back(
            JettyCfg{jettyInfos[i].jettyCtxId, createdVec[i].dbVa, createdVec[i].dbTokenId, ccuBufTokenValue});
    }

    CHK_PTR_NULL(channelMgrs[dieId]);
    return channelMgrs[dieId]->Config(cfg);
}

void CcuComponent::ConfigMsIdToken()
{
    bool isAX = CcuResSpecifications::GetInstance(devLogicId).GetAXFlag();
    auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(devLogicId);
    CHECK_NULLPTR(
        tlvHandle, StringFormat("[CcuComponent][%s] tlvHandle is nullptr, devLogicId[%d]", __func__, devLogicId));

    struct CustomChannelInfoIn inBuff {};
    struct CustomChannelInfoOut outBuff {};

    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        const auto& dieIter = localCcuRmaBufferMap.find(dieId);
        if (dieIter == localCcuRmaBufferMap.end()) {
            HCCL_WARNING(
                "[CcuComponent][%s] failed but passed, ccu rma buffer of die[%u] "
                "is not existed, devLogicId[%d].",
                __func__, dieId, devLogicId);
            continue;
        }
        const auto& ccuRmaBuffer = dieIter->second;
        const uint32_t tokenId = ccuRmaBuffer->GetTokenId();
        const uint32_t tokenValue = ccuRmaBuffer->GetTokenValue();
        uint32_t msId = 0;
        CHK_RET_THROW(
            InternalException,
            StringFormat("[CcuComponent][%s] failed, devLogicId[%d], dieId[%u].", __func__, devLogicId, dieId),
            CcuResSpecifications::GetInstance(devLogicId).GetMsId(dieId, msId));

        inBuff.op = CcuOpcodeType::CCU_U_OP_SET_MSID_TOKEN;
        inBuff.offsetStartIdx = 0;
        inBuff.data.dataInfo.udieIdx = dieId;

        if (isAX && dieId == 0) { // A+X环境，给udie0配置新的交织粒度
            msId = MSID_CONFIG_AX_MAINBOARD;
        }
        inBuff.data.dataInfo.dataArray[0].baseinfo.msId = msId;
        inBuff.data.dataInfo.dataArray[0].baseinfo.tokenId = tokenId;
        inBuff.data.dataInfo.dataArray[0].baseinfo.tokenValue = tokenValue;

        HrtRaTlvRequestForCustomChannel(
            tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));

        HCCL_INFO("[CcuComponent][%s] config MS ID token success, dieId[%u], msid[%u]", __func__, dieId, msId);
    }
}

HcclResult CcuComponent::GetCcuResourceSpaceBufInfo(const uint8_t dieId, uint64_t& addr, uint64_t& size) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));

    auto res = ccuRmaBufferMap.find(dieId);
    CHK_PRT_RET(
        res == ccuRmaBufferMap.end(),
        HCCL_WARNING(
            "[CcuComponent][%s] failed, ccu rma buffer of die[%u] is not existed, "
            "devLogicId[%d].",
            __func__, dieId, devLogicId),
        HcclResult::HCCL_E_NOT_FOUND);

    const auto rawBuffer = res->second->GetBuf();
    addr = static_cast<uint64_t>(rawBuffer->GetAddr());
    size = static_cast<uint64_t>(rawBuffer->GetSize());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuComponent::GetCcuResourceSpaceTokenInfoForLocal(const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));

    auto res = localCcuRmaBufferMap.find(dieId);
    CHK_PRT_RET(
        res == localCcuRmaBufferMap.end(),
        HCCL_WARNING(
            "[CcuComponent][%s] failed, ccu rma buffer of die[%u] is not existed, "
            "devLogicId[%d].",
            __func__, dieId, devLogicId),
        HcclResult::HCCL_E_NOT_FOUND);

    const auto& ccuRmaBuffer = res->second;
    tokenId = static_cast<uint64_t>(ccuRmaBuffer->GetTokenId());
    tokenValue = static_cast<uint64_t>(ccuRmaBuffer->GetTokenValue());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuComponent::GetCcuResourceSpaceTokenInfo(const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue) const
{
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));

    auto res = ccuRmaBufferMap.find(dieId);
    CHK_PRT_RET(
        res == ccuRmaBufferMap.end(),
        HCCL_WARNING(
            "[CcuComponent][%s] failed, ccu rma buffer of die[%u] is not existed, "
            "devLogicId[%d].",
            __func__, dieId, devLogicId),
        HcclResult::HCCL_E_NOT_FOUND);

    const auto& ccuRmaBuffer = res->second;
    tokenId = static_cast<uint64_t>(ccuRmaBuffer->GetTokenId());
    tokenValue = static_cast<uint64_t>(ccuRmaBuffer->GetTokenValue());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuComponent::AllocChannels(const uint8_t dieId, const ChannelPara& channelPara, std::vector<ChannelInfo>& channelInfos)
{
    std::shared_lock<std::shared_mutex> lock(innerMutex);
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));

    CHK_PTR_NULL(channelMgrs[dieId]);
    auto ret = channelMgrs[dieId]->Alloc(channelPara, channelInfos);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuComponent][%s] failed, feId[%u], devLogicId[%d], dieId[%u].", __func__, channelPara.feId, devLogicId,
            dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ConfigChannel(const uint8_t dieId, const ChannelCfg& cfg)
{
    std::shared_lock<std::shared_mutex> lock(innerMutex);
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));

    uint32_t channelId = cfg.channelId;
    CHK_PRT_RET(
        channelId == loopChannelIds[dieId],
        HCCL_WARNING(
            "[CcuComponent][%s] failed, refused to config loop channel[%u], "
            "devLogicId[%d], dieId[%u].",
            __func__, channelId, devLogicId, dieId),
        HcclResult::HCCL_E_PARA);

    CHK_PTR_NULL(channelMgrs[dieId]);
    auto ret = channelMgrs[dieId]->Config(cfg);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuComponent][%s] failed, channelId[%u], devLogicId[%d], dieId[%u].", __func__, channelId, devLogicId,
            dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ReleaseChannel(const uint8_t dieId, const uint32_t channelId)
{
    std::shared_lock<std::shared_mutex> lock(innerMutex);
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    CHK_PRT_RET(
        channelId == loopChannelIds[dieId],
        HCCL_WARNING(
            "[CcuComponent][%s] failed, refused to release loop channel[%u], "
            "devLogicId[%d], dieId[%u].",
            __func__, channelId, devLogicId, dieId),
        HcclResult::HCCL_E_PARA);

    CHK_PTR_NULL(channelMgrs[dieId]);
    auto ret = channelMgrs[dieId]->Release(channelId);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuComponent][%s] failed, channelId[%u], devLogicId[%d], dieId[%u].", __func__, channelId, devLogicId,
            dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::GetLoopChannelId(const uint8_t srcDieId, const uint8_t dstDieId, uint32_t& channelId) const
{
    channelId = INVAILD_LOOP_CHANNEL_ID; // 允许die未启用时查询环回channelId
    CHK_RET(CheckDieValid(__func__, devLogicId, srcDieId, {true, true}));
    CHK_RET(CheckDieValid(__func__, devLogicId, dstDieId, {true, true}));

    CHK_PRT_RET(
        loopChannelIds[srcDieId] == INVAILD_LOOP_CHANNEL_ID, // 环回channel每个die共用1个
        HCCL_WARNING(
            "[CcuComponent][%s] failed, invalid loop channel id, "
            "devLogicId[%d], srcDieId[%u].",
            __func__, devLogicId, srcDieId),
        HcclResult::HCCL_E_INTERNAL);

    channelId = loopChannelIds[srcDieId];
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::AllocRes(
    const uint8_t dieId, const ResType resType, const uint32_t num, const bool consecutive, vector<ResInfo>& resInfos)
{
    std::shared_lock<std::shared_mutex> lock(innerMutex);
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));

    CHK_PTR_NULL(resAllocators[dieId]);
    auto ret = resAllocators[dieId]->Alloc(resType, num, consecutive, resInfos);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuComponent][%s] failed, resType[%s], num[%u], devLogicId[%d], dieId[%u].", __func__,
            resType.Describe().c_str(), num, devLogicId, dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuComponent::ReleaseRes(const uint8_t dieId, const ResType resType, const uint32_t startId, const uint32_t num)
{
    std::shared_lock<std::shared_mutex> lock(innerMutex);
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));

    CHK_PTR_NULL(resAllocators[dieId]);
    auto ret = resAllocators[dieId]->Release(resType, startId, num);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuComponent][%s] failed, resType[%s], startId[%u], num[%u], "
            "devLogicId[%d], dieId[%u].",
            __func__, resType.Describe().c_str(), startId, num, devLogicId, dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

uint32_t CcuComponent::GetInsConsecutiveRemainSize(const uint8_t dieId) const
{
    std::shared_lock<std::shared_mutex> lock(innerMutex);
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    if (resAllocators[dieId] == nullptr)
        return 0;
    return resAllocators[dieId]->GetConsecutiveRemainSize(ResType::INS);
}

HcclResult CcuComponent::AllocIns(const uint8_t dieId, const uint32_t num, ResInfo& insInfo)
{
    std::shared_lock<std::shared_mutex> lock(innerMutex);
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    CHK_PTR_NULL(resAllocators[dieId]);

    vector<ResInfo> resInfos;
    auto ret = resAllocators[dieId]->Alloc(ResType::INS, num, true, resInfos);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuComponent][%s] failed, num[%u], devLogicId[%d], dieId[%u].", __func__, num, devLogicId, dieId),
        ret);

    insInfo = resInfos[0]; // 申请连续资源只会有一份
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ReleaseIns(const uint8_t dieId, const ResInfo& insInfo)
{
    std::shared_lock<std::shared_mutex> lock(innerMutex);
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    CHK_PTR_NULL(resAllocators[dieId]);

    auto ret = resAllocators[dieId]->Release(ResType::INS, insInfo.startId, insInfo.num);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuComponent][%s] failed, resInfo[%s], devLogicId[%d], dieId[%u].", __func__, insInfo.Describe().c_str(),
            devLogicId, dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::AllocCke(const uint8_t dieId, const uint32_t num, vector<ResInfo>& ckeInfos)
{
    std::shared_lock<std::shared_mutex> lock(innerMutex);
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    CHK_PTR_NULL(resAllocators[dieId]);

    auto ret = resAllocators[dieId]->Alloc(ResType::CKE, num, false, ckeInfos);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuComponent][%s] failed, num[%u], devLogicId[%d], dieId[%u].", __func__, num, devLogicId, dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ReleaseCke(const uint8_t dieId, const vector<ResInfo>& ckeInfos)
{
    std::shared_lock<std::shared_mutex> lock(innerMutex);
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    CHK_PTR_NULL(resAllocators[dieId]);

    for (auto& ckeInfo : ckeInfos) {
        auto ret = resAllocators[dieId]->Release(ResType::CKE, ckeInfo.startId, ckeInfo.num);
        CHK_PRT_RET(
            ret != HcclResult::HCCL_SUCCESS,
            HCCL_WARNING(
                "[CcuComponent][%s] failed, resInfo[%s], devLogicId[%d], dieId[%u].", __func__,
                ckeInfo.Describe().c_str(), devLogicId, dieId),
            ret);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::AllocXn(const uint8_t dieId, const uint32_t num, vector<ResInfo>& xnInfos)
{
    std::shared_lock<std::shared_mutex> lock(innerMutex);
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    CHK_PTR_NULL(resAllocators[dieId]);

    auto ret = resAllocators[dieId]->Alloc(ResType::XN, num, false, xnInfos);
    CHK_PRT_RET(
        ret != HcclResult::HCCL_SUCCESS,
        HCCL_WARNING(
            "[CcuComponent][%s] failed, num[%u], devLogicId[%d], dieId[%u].", __func__, num, devLogicId, dieId),
        ret);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::ReleaseXn(const uint8_t dieId, const vector<ResInfo>& xnInfos)
{
    std::shared_lock<std::shared_mutex> lock(innerMutex);
    CHK_RET(CheckDieValid(__func__, devLogicId, dieId, dieEnableFlags));
    CHK_PTR_NULL(resAllocators[dieId]);

    for (auto& xnInfo : xnInfos) {
        auto ret = resAllocators[dieId]->Release(ResType::XN, xnInfo.startId, xnInfo.num);
        CHK_PRT_RET(
            ret != HcclResult::HCCL_SUCCESS,
            HCCL_WARNING(
                "[CcuComponent][%s] failed, resInfo[%s], devLogicId[%d], dieId[%u].", __func__,
                xnInfo.Describe().c_str(), devLogicId, dieId),
            ret);
    }

    return HcclResult::HCCL_SUCCESS;
}

// 以下接口用于n秒快恢与TaskException
HcclResult CcuComponent::CleanDieCkes(const uint8_t dieId) const
{
    CHK_PRT_RET(
        dieId >= MAX_CCU_IODIE_NUM,
        HCCL_WARNING(
            "[CcuComponent][%s] failed, dieId[%u] is invalid, should be in [0-%u), devLogicId[%d].", __func__, dieId,
            MAX_CCU_IODIE_NUM, devLogicId),
        HcclResult::HCCL_E_PARA);

    if (!dieEnableFlags[dieId]) {
        return HcclResult::HCCL_SUCCESS;
    }

    auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(devLogicId);
    CHK_PTR_NULL(tlvHandle);
    CustomChannelInfoIn inBuff{};
    CustomChannelInfoOut outBuff{};

    // 设置操作码和数据
    uint32_t ckeNum = 0;
    CHK_RET(CcuResSpecifications::GetInstance(devLogicId).GetCkeNum(dieId, ckeNum));
    HCCL_INFO("[CcuComponent][CleanAllCke]Nsrecovery devLogicId[%d], dieId[%u] ckeNum[%u].", devLogicId, dieId, ckeNum);

    inBuff.op = CcuOpcodeType::CCU_U_OP_SET_CKE;
    inBuff.data.dataInfo.udieIdx = dieId;
    // 接口限制，目前方案每次最多清理8个cke，超过8个时分多次清理
    for (uint32_t startIdx = 0; startIdx < ckeNum; startIdx += MAX_CKE_DATA_ARRAY_SIZE) {
        inBuff.data.dataInfo.dataArraySize = std::min(ckeNum - startIdx, MAX_CKE_DATA_ARRAY_SIZE);
        inBuff.data.dataInfo.dataLen = sizeof(CcuDataByte8) * inBuff.data.dataInfo.dataArraySize;
        inBuff.offsetStartIdx = startIdx;
        HrtRaTlvRequestForCustomChannel(
            tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
    }

    return HcclResult::HCCL_SUCCESS;
}

void CcuComponent::SetProcess(CcuOpcodeType opCode) const
{
    auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(devLogicId);
    CHECK_NULLPTR(
        tlvHandle, StringFormat("[CcuComponent][%s] tlvHandle is nullptr, devLogicId[%d]", __func__, devLogicId));

    struct CustomChannelInfoIn inBuff;
    struct CustomChannelInfoOut outBuff;

    inBuff.op = opCode;
    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        if (!dieEnableFlags[dieId]) {
            HCCL_WARNING(
                "[CcuComponent::SetProcess] devLogicId[%d], dieId[%u] is not enable,"
                "skip SetProcess.",
                devLogicId, dieId);
            continue;
        }
        HCCL_INFO("[CcuComponent::SetProcess] devLogicId[%d], dieId[%u] start.", devLogicId, dieId);
        inBuff.data.dataInfo.udieIdx = dieId;
        HrtRaTlvRequestForCustomChannel(
            tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
    }
}

HcclResult CcuComponent::SetTaskKill()
{
    std::lock_guard<std::mutex> _lock(taskKillMutex_);

    if (status == CcuTaskKillStatus::INVALID) {
        status = CcuTaskKillStatus::INIT;
    }

    if (status == CcuTaskKillStatus::TASK_KILL) {
        HCCL_INFO("No need to set task kill, state = %u, devLogicId = %u", status, devLogicId);
        return HcclResult::HCCL_SUCCESS;
    }

    if (status != CcuTaskKillStatus::INIT) {
        HCCL_ERROR(
            "[CcuComponent][%s] failed, cannot be invoked in the current state, "
            "state = %u, devLogicId = %d.",
            __func__, status, devLogicId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    SetProcess(CcuOpcodeType::CCU_U_OP_SET_TASKKILL);
    status = CcuTaskKillStatus::TASK_KILL;
    HCCL_INFO("[CcuComponent][%s] success, state = %u, devLogicId = %d.", __func__, status, devLogicId);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::SetTaskKillDone()
{
    std::lock_guard<std::mutex> _lock(taskKillMutex_);

    if (status == CcuTaskKillStatus::INVALID) {
        HCCL_ERROR(
            "[CcuComponent][%s] failed, cannot be invoked in the current state, "
            "state = %u, devLogicId = %d.",
            __func__, status, devLogicId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (status == CcuTaskKillStatus::INIT) {
        HCCL_INFO("No need to set task kill done, state = %u, devLogicId = %u", status, devLogicId);
        return HcclResult::HCCL_SUCCESS;
    }

    if (status != CcuTaskKillStatus::TASK_KILL) {
        HCCL_ERROR(
            "[CcuComponent][%s] failed, cannot be invoked in the current state, "
            "state = %u, devLogicId = %d.",
            __func__, status, devLogicId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    SetProcess(CcuOpcodeType::CCU_U_OP_CLEAN_TASKKILL_STATE);
    status = CcuTaskKillStatus::INIT;
    HCCL_INFO("[CcuComponent][%s] success, state = %u, devLogicId = %d", __func__, status, devLogicId);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuComponent::CleanTaskKillState() const
{
    SetProcess(CcuOpcodeType::CCU_U_OP_CLEAN_TASKKILL_STATE);
    return HcclResult::HCCL_SUCCESS;
}

const std::array<bool, MAX_CCU_IODIE_NUM>& CcuComponent::GetDieEnableFlags() const { return dieEnableFlags; }

CcuComponent::~CcuComponent() { DECTOR_TRY_CATCH("CcuComponent", ReleaseJettyRes()); }

void CcuComponent::ReleaseJettyRes()
{
    UnimportAllJetty();
    DestroyAllJetty();
    // HrtRaUbLocalMemReg 跟随 LocalUbRmaBuffer 析构时释放
    // 环回channel不需要手动释放，channelMgr跟随CcuComponent释放
}

void CcuComponent::UnimportAllJetty()
{
    // tpInfo不需要主动释放，因为CcuComponent生命周期与TpManager一致
    for (auto& importedVec : importedOutParamMap) {
        for (auto& paramPair : importedVec.second) {
            const auto rdmaHandle = paramPair.first;
            const auto remoteJettyHandle = paramPair.second.handle;
            if (rdmaHandle != nullptr && remoteJettyHandle != 0) {
                paramPair.second.handle = 0;
                HrtRaUbUnimportJetty(rdmaHandle, remoteJettyHandle);
            }
        }
    }

    importedOutParamMap.clear();
}

void CcuComponent::DestroyAllJetty()
{
    for (auto& createdVec : createdOutParamMap) {
        for (auto& param : createdVec.second) {
            const auto jettyHandle = param.handle;
            if (jettyHandle != 0) {
                param.handle = 0;
                HrtRaUbDestroyJetty(jettyHandle);
            }
        }
    }
    createdOutParamMap.clear();
}

}; // namespace Hccl
