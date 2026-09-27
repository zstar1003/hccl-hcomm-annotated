/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rdma_handle_manager.h"
#include <mutex>

#include "socket_handle_manager.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "log.h"
#include "tokenInfo_manager.h"

namespace Hccl {

RdmaHandleManager::RdmaHandleManager()
{
    rdmaHandleMap.resize(MAX_DEVICE_NUM);
    for (u32 i = 0; i < rdmaHandleMap.size(); ++i) {
        rdmaHandleMap[i].resize(LINK_PROTO_TYPE_NUM);
    }
}

RdmaHandleManager::~RdmaHandleManager() { DECTOR_TRY_CATCH("RdmaHandleManager", DestroyAll()); }

RdmaHandleManager& RdmaHandleManager::GetInstance()
{
    static RdmaHandleManager rdmaHandleManager;
    return rdmaHandleManager;
}

RdmaHandle RdmaHandleManager::Create(u32 devPhyId, const PortData& localPort)
{
    RaInterface intf{};
    intf.address = localPort.GetAddr();
    intf.phyId = devPhyId;

    HCCL_INFO("RdmaHandleManager::Create, devPhyId[%u], localPort[%s]", devPhyId, localPort.Describe().c_str());

    HrtNetworkMode netMode = HrtNetworkMode::HDC;
    if (localPort.GetType() == PortDeploymentType::HOST_NET) {
        netMode = HrtNetworkMode::PEER;
    }

    RdmaHandle rdmaHandle = HrtRaRdmaInit(netMode, intf);
    rdmaHandleMap[devPhyId][localPort.GetProto()][localPort.GetAddr()] = rdmaHandle;
    netWorkModeMap[rdmaHandle] = netMode;
    activeHandles_.insert(rdmaHandle);
    return rdmaHandle;
}

RdmaHandle RdmaHandleManager::Create(
    u32 devPhyId, const LinkProtoType& localProtocolType, const IpAddress& localIp, PortDeploymentType type)
{
    RaInterface intf{};
    intf.address = localIp;
    intf.phyId = devPhyId;

    HCCL_INFO(
        "RdmaHandleManager::Create, devPhyId[%u], LinkProtoType[%s], localIp[%s]", devPhyId,
        localProtocolType.Describe().c_str(), localIp.GetIpStr().c_str());

    HrtNetworkMode netMode = HrtNetworkMode::HDC;
    if (type == PortDeploymentType::HOST_NET) {
        netMode = HrtNetworkMode::PEER;
    }
    RdmaHandle rdmaHandle = HrtRaRdmaInit(netMode, intf);
    rdmaHandleMap[devPhyId][localProtocolType][localIp] = rdmaHandle;
    netWorkModeMap[rdmaHandle] = netMode;
    tokenInfoMap[rdmaHandle] = std::make_unique<TokenInfoManager>(devPhyId, rdmaHandle);
    activeHandles_.insert(rdmaHandle);
    return rdmaHandle;
}

RdmaHandle RdmaHandleManager::Get(u32 devPhyId, const PortData& localPort, LinkProtocol linkProtocol)
{
    std::lock_guard<std::mutex> lock(managerMutex);

    LinkProtoType localProto = localPort.GetProto();
    if (devPhyId > rdmaHandleMap.size() - 1 || localProto == LinkProtoType::HCCS_PCIE) {
        return nullptr;
    }

    IpAddress localIp = localPort.GetAddr();
    if (linkProtocol == LinkProtocol::UBOE) {
        IpAddress eidAddress;
        UboeIpv4ToEid(localIp, eidAddress, devPhyId);
        localIp = eidAddress;
    }

    RdmaHandle res = rdmaHandleMap[devPhyId][localProto][localIp];
    if (res == nullptr) {
        if (localProto == LinkProtoType::RDMA) {
            res = Create(devPhyId, localPort);
        } else if (localProto == LinkProtoType::UB) {
            HrtRaUbCtxInitParam in(HrtNetworkMode::HDC, devPhyId, localIp);
            res = HrtRaUbCtxInit(in);
            rdmaHandleMap[devPhyId][localProto][localIp] = res;
            tokenInfoMap[res] = std::make_unique<TokenInfoManager>(devPhyId, res);
            activeHandles_.insert(res);
            HCCL_INFO(
                "Create one rdmahandle [%p], devPhyId [%u], portAddr [%s]", res, devPhyId,
                localPort.GetAddr().Describe().c_str());
        }
    }

    HCCL_INFO(
        "[RdmaHandleManager::Get] one rdmahandle [%p], devPhyId [%u], portAddr [%s]", res, devPhyId,
        localPort.GetAddr().Describe().c_str());
    return res;
}

RdmaHandle RdmaHandleManager::GetByAddr(
    u32 devPhyId, const LinkProtoType& localProtocolType, IpAddress& localIp, PortDeploymentType type)
{
    std::lock_guard<std::mutex> lock(managerMutex);

    if (devPhyId > rdmaHandleMap.size() - 1 || localProtocolType == LinkProtoType::HCCS_PCIE) {
        return nullptr;
    }

    RdmaHandle res = rdmaHandleMap[devPhyId][localProtocolType][localIp];
    if (res == nullptr) {
        if (localProtocolType == LinkProtoType::RDMA) {
            res = Create(devPhyId, localProtocolType, localIp, type);
        } else if (localProtocolType == LinkProtoType::UB) {
            HrtNetworkMode mode
                = (type == Hccl::PortDeploymentType::DEV_NET) ? HrtNetworkMode::HDC : HrtNetworkMode::PEER;
            HrtRaUbCtxInitParam in(mode, devPhyId, localIp);
            res = HrtRaUbCtxInit(in);
            rdmaHandleMap[devPhyId][localProtocolType][localIp] = res;
            tokenInfoMap[res] = std::make_unique<TokenInfoManager>(devPhyId, res);
            activeHandles_.insert(res);
            HCCL_INFO(
                "Create one rdmahandle [%p], devPhyId [%u], portAddr [%s]", res, devPhyId, localIp.Describe().c_str());
        }
    }
    HCCL_INFO(
        "[RdmaHandleManager::GetByAddr] one rdmahandle [%p], devPhyId [%u], portAddr [%s]", res, devPhyId,
        localIp.Describe().c_str());
    return res;
}

RdmaHandle RdmaHandleManager::GetByIp(u32 devPhyId, const IpAddress& localIp)
{
    std::lock_guard<std::mutex> lock(managerMutex);

    if (devPhyId > rdmaHandleMap.size() - 1) {
        HCCL_ERROR(
            "[RdmaHandleManager][GetByIp]devPhyId[%u] is invalid, "
            "should be less than [%zu]",
            devPhyId, rdmaHandleMap.size());
        return nullptr;
    }

    // only support UB type
    RdmaHandle res = rdmaHandleMap[devPhyId][LinkProtoType::UB][localIp];
    if (res == nullptr) {
        HrtRaUbCtxInitParam in(HrtNetworkMode::HDC, devPhyId, localIp);
        res = HrtRaUbCtxInit(in);
        rdmaHandleMap[devPhyId][LinkProtoType::UB][localIp] = res;
        tokenInfoMap[res] = std::make_unique<TokenInfoManager>(devPhyId, res);
        activeHandles_.insert(res);
        HCCL_INFO("Create one rdmahandle [%p], devPhyId [%u], ipAddr [%s]", res, devPhyId, localIp.Describe().c_str());
    }

    return res;
}

bool RdmaHandleManager::FindCachedJfcHandle(
    RdmaHandle rdmaHandle, HrtUbJfcMode jfcMode, JfcHandle& handle, CqCreateInfo& cqInfo)
{
    auto outerIt = jfcHandleMap.find(rdmaHandle);
    if (outerIt == jfcHandleMap.end()) {
        return false;
    }
    auto innerIt = outerIt->second.find(jfcMode);
    if (innerIt == outerIt->second.end()) {
        return false;
    }
    handle = innerIt->second;
    cqInfo = cqInfoMap[innerIt->second];
    return true;
}

JfcHandle RdmaHandleManager::GetJfcHandle(RdmaHandle rdmaHandle, CqCreateInfo& cqInfo, HrtUbJfcMode jfcMode)
{
    std::lock_guard<std::mutex> lock(managerMutex);

    if (rdmaHandle == nullptr) {
        THROW<InvalidParamsException>("[RdmaHandleManager][GetJfcHandle]rdmaHandle is nullptr, please check input.");
    }

    if (jfcMode != HrtUbJfcMode::STARS_POLL && jfcMode != HrtUbJfcMode::CCU_POLL && jfcMode != HrtUbJfcMode::NORMAL) {
        THROW<InvalidParamsException>(
            "[RdmaHandleManager][GetJfcHandle]jfcMode[%s] is not STARS_POLL or CCU_POLL or NORMAL, "
            "please check input.",
            jfcMode.Describe().c_str());
    }

    JfcHandle cachedHandle{};
    if (FindCachedJfcHandle(rdmaHandle, jfcMode, cachedHandle, cqInfo)) {
        return cachedHandle;
    }
    JfcHandle newHandle = HrtRaUbCreateJfc(rdmaHandle, cqInfo, jfcMode);
    JfcHandle& ref = jfcHandleMap[rdmaHandle][jfcMode]; // 引用缓存
    ref = newHandle;
    cqInfoMap[ref] = cqInfo;
    return ref;
}

JfcHandle RdmaHandleManager::GetJfcHandleAndCqInfo(RdmaHandle rdmaHandle, CqCreateInfo& cqInfo, HrtUbJfcMode jfcMode)
{
    std::lock_guard<std::mutex> lock(managerMutex);

    if (rdmaHandle == nullptr) {
        THROW<InvalidParamsException>(
            "[RdmaHandleManager][GetJfcHandleAndCqInfo]rdmaHandle is nullptr, please check input.");
    }

    if (jfcMode != HrtUbJfcMode::USER_CTL) {
        THROW<InvalidParamsException>(
            "[RdmaHandleManager][GetJfcHandleAndCqInfo]jfcMode[%s] is not USER_CTL, "
            "please check input.",
            jfcMode.Describe().c_str());
    }

    JfcHandle cachedHandle{};
    if (FindCachedJfcHandle(rdmaHandle, jfcMode, cachedHandle, cqInfo)) {
        return cachedHandle;
    }
    JfcHandle newHandle = HrtRaUbCreateJfcUserCtl(rdmaHandle, cqInfo);
    JfcHandle& ref = jfcHandleMap[rdmaHandle][jfcMode]; // 引用缓存
    ref = newHandle;
    cqInfoMap[ref] = cqInfo;
    return ref;
}

std::pair<uint32_t, uint32_t> RdmaHandleManager::GetDieAndFuncId(RdmaHandle rdmaHandle)
{
    std::lock_guard<std::mutex> lock(managerMutex);

    if (rdmaHandle == nullptr) {
        THROW<InvalidParamsException>("[RdmaHandleManager][GetDieAndFuncId]rdmaHandle is nullptr, please check input.");
    }

    if (DieAndFuncIdMap.find(rdmaHandle) != DieAndFuncIdMap.end()) {
        return DieAndFuncIdMap[rdmaHandle];
    }

    DieAndFuncIdMap[rdmaHandle] = HraGetDieAndFuncId(rdmaHandle);
    return DieAndFuncIdMap[rdmaHandle];
}

bool RdmaHandleManager::GetRtpEnable(RdmaHandle rdmaHandle)
{
    std::lock_guard<std::mutex> lock(managerMutex);

    if (rdmaHandle == nullptr) {
        THROW<InvalidParamsException>("[RdmaHandleManager][GetRtpEnable]rdmaHandle is nullptr, please check input.");
    }

    if (RtpEnableMap.find(rdmaHandle) != RtpEnableMap.end()) {
        return RtpEnableMap[rdmaHandle];
    }

    RtpEnableMap[rdmaHandle] = HraGetRtpEnable(rdmaHandle);
    HCCL_RUN_INFO("[%s] GetRtpEnable return[%d]", __func__, RtpEnableMap[rdmaHandle]);
    return RtpEnableMap[rdmaHandle];
}

std::pair<TokenIdHandle, uint32_t>
RdmaHandleManager::GetTokenIdInfo(RdmaHandle rdmaHandle, const BufferKey<uintptr_t, u64>& bufKey)
{
    std::lock_guard<std::mutex> lock(managerMutex);

    if (rdmaHandle == nullptr) {
        THROW<InvalidParamsException>("[RdmaHandleManager::%s]rdmaHandle is nullptr, please check input.", __func__);
    }

    if (tokenInfoMap.find(rdmaHandle) == tokenInfoMap.end()) {
        THROW<InvalidParamsException>("[RdmaHandleManager::%s]tokenInfoManager is nullptr, please check.", __func__);
    }

    std::pair<TokenIdHandle, uint32_t> tokenInfo = tokenInfoMap[rdmaHandle]->GetTokenInfo(bufKey);

    HCCL_INFO(
        "[RdmaHandleManager::%s] Addr[%llu] Size[%llu] rdmahandle[%p]", __func__, bufKey.Addr(), bufKey.Size(),
        rdmaHandle);
    return tokenInfo;
}

void RdmaHandleManager::PutTokenIdInfo(
    RdmaHandle rdmaHandle, const BufferKey<uintptr_t, u64>& bufKey, TokenIdHandle tokenIdHandle)
{
    std::lock_guard<std::mutex> lock(managerMutex);

    if (rdmaHandle == nullptr) {
        HCCL_WARNING("[RdmaHandleManager::%s]rdmaHandle is nullptr", __func__);
        return;
    }

    if (tokenInfoMap.find(rdmaHandle) == tokenInfoMap.end()) {
        HCCL_WARNING("[RdmaHandleManager::%s]tokenInfoManager is nullptr", __func__);
        return;
    }

    tokenInfoMap[rdmaHandle]->PutTokenInfo(bufKey, tokenIdHandle);
    HCCL_DEBUG(
        "[RdmaHandleManager::%s] Addr[%llu] Size[%llu] rdmahandle[%p]", __func__, bufKey.Addr(), bufKey.Size(),
        rdmaHandle);
}

constexpr u32 UB_HANDLE_INDEX = 3;
constexpr u32 RDMA_HANDLE_INDEX = 2;

void RdmaHandleManager::DestroyAll()
{
    if (destroyed.load()) {
        return;
    }
    std::lock_guard<std::mutex> lock(managerMutex);
    if (destroyed.load()) {
        return;
    }
    destroyed.store(true);
    HCCL_INFO("[RdmaHandleManager::%s] destroy all", __func__);
    activeHandles_.clear();
    for (auto& handleIter : jfcHandleMap) {
        for (auto& modeIter : handleIter.second) {
            DECTOR_TRY_CATCH("jfc handle destroy", HrtRaUbDestroyJfc(handleIter.first, modeIter.second));
        }
    }

    for (u32 i = 0; i < rdmaHandleMap.size(); ++i) {
        for (u32 j = 0; j < rdmaHandleMap[i].size(); ++j) {
            for (auto& handleIter : rdmaHandleMap[i][j]) {
                if (j == RDMA_HANDLE_INDEX && handleIter.second != nullptr) {
                    DECTOR_TRY_CATCH(
                        "rdma handle deinit", HrtRaRdmaDeInit(handleIter.second, netWorkModeMap[handleIter.second]));
                }
                if (j == UB_HANDLE_INDEX && handleIter.second != nullptr) {
                    if (tokenInfoMap[handleIter.second] != nullptr) {
                        DECTOR_TRY_CATCH("token id handle destroy", tokenInfoMap[handleIter.second]->Destroy());
                    }
                    DECTOR_TRY_CATCH("ub handle destroy", HrtRaUbCtxDestroy(handleIter.second));
                }
            }
            rdmaHandleMap[i][j].clear();
        }
    }

    rdmaHandleMap.clear();
    DieAndFuncIdMap.clear();
    RtpEnableMap.clear();
    jfcHandleMap.clear();
    cqInfoMap.clear();
    netWorkModeMap.clear();
}

HcclResult
GetEidByAnyEidInfo(s32 deviceLogicId, const HrtDevEidInfo& eidInfo, const IpAddress& ipV4Address, IpAddress& eidAddress)
{
    // 根据eidInfo初始化rdmaHandle
    HrtRaUbCtxInitParam in(HrtNetworkMode::HDC, HrtGetDevicePhyIdByUserDevId(deviceLogicId), eidInfo.ipAddress);
    RdmaHandle rdmaHandle = HrtRaUbCtxInit(in);

    // 调用ra_get_eid_by_ip转换ipAddress为eid
    vector<IpAddress> eidAddrList{};
    CHK_RET(HrtRaGetEidByIp(rdmaHandle, {ipV4Address}, eidAddrList));
    if (eidAddrList.empty()) {
        HCCL_WARNING(
            "[RdmaHandleManager::%s] Get Eid failed, deviceLogicId=%d, ipV4Address=%s", __func__, deviceLogicId,
            ipV4Address.Describe().c_str());
        return HCCL_E_NOT_FOUND;
    }
    eidAddress = eidAddrList.front();
    return HCCL_SUCCESS;
}

/* 将IPV4转为EID
    1、基于IPV4 IpAddress查询uboeIpv4EidMap，如果存在直接返回
    2、不存在时，调用hccp接口根据IPV4 IpAddress查询EID，并将其保存到uboeIpv4EidMap中
*/
void RdmaHandleManager::UboeIpv4ToEid(const IpAddress& ipV4Address, IpAddress& eidAddress, u32 devPhyId)
{
    // 如果无法查询设备是否为uboe设备，直接退出
    if (HrtGetUboeFlagEnable(devPhyId) != HCCL_SUCCESS) {
        return;
    }
    HCCL_INFO("[UboeIpv4ToEid] begin, ipV4Address[%s]", ipV4Address.Describe().c_str());
    auto it = uboeIpv4EidMap.find(ipV4Address);
    if (it != uboeIpv4EidMap.end()) {
        eidAddress = it->second;
        HCCL_INFO("[UboeIpv4ToEid] uboeIpv4EidMap find, eidAddress[%s]", it->second.Describe().c_str());
        return;
    }

    s32 deviceLogicId = HrtGetDevice();
    HRaInfo info(HrtNetworkMode::HDC, HrtGetDevicePhyIdByUserDevId(deviceLogicId));
    vector<HrtDevEidInfo> eidInfoList = HrtRaGetDevEidInfoList(info);
    if (eidInfoList.empty()) {
        HCCL_WARNING("[RdmaHandleManager::%s] Get EidInfoList empty, deviceLogicId=%d", __func__, deviceLogicId);
        return;
    }
    HCCL_INFO(
        "[RdmaHandleManager::%s] Get EidInfo success, deviceLogicId=%d, eidInfo size=%u", __func__, deviceLogicId,
        eidInfoList.size());

    for (const auto& eidInfo : eidInfoList) {
        if (HrtCheckUboeSupported(eidInfo.devFeature)
            && GetEidByAnyEidInfo(deviceLogicId, eidInfo, ipV4Address, eidAddress) == HCCL_SUCCESS) {
            // 存储eid到AddressInfo
            HCCL_INFO("[UboeIpv4ToEid] success, eidAddress[%s]", eidAddress.Describe().c_str());
            uboeIpv4EidMap.insert(std::make_pair(ipV4Address, eidAddress));
            return;
        }
    }
    HCCL_WARNING("[RdmaHandleManager::%s] Get EidInfo failed, deviceLogicId=%d", __func__, deviceLogicId);
}

HcclResult RdmaHandleManager::GetEidByIpv4Addr(const IpAddress& addr, IpAddress& eidAddr)
{
    auto it = uboeIpv4EidMap.find(addr);
    if (it == uboeIpv4EidMap.end()) {
        HCCL_WARNING("[RdmaHandleManager::%s] Find Eid failed, addr[%s]", __func__, addr.Describe().c_str());
        return HCCL_E_PARA;
    }
    eidAddr = it->second;
    return HCCL_SUCCESS;
}

void RdmaHandleManager::CollectHandlesToCleanup(u32 devPhyId, std::vector<HandleInfo>& handlesToCleanup)
{
    for (u32 j = 0; j < rdmaHandleMap[devPhyId].size(); ++j) {
        for (auto& handleIter : rdmaHandleMap[devPhyId][j]) {
            if (handleIter.second != nullptr) {
                handlesToCleanup.push_back({handleIter.second, j});
            }
        }
        rdmaHandleMap[devPhyId][j].clear();
    }
}

void RdmaHandleManager::CleanupJfcHandles(RdmaHandle handle)
{
    auto jfcIt = jfcHandleMap.find(handle);
    if (jfcIt != jfcHandleMap.end()) {
        for (auto& modeIter : jfcIt->second) {
            DECTOR_TRY_CATCH("jfc handle destroy", HrtRaUbDestroyJfc(handle, modeIter.second));
            cqInfoMap.erase(modeIter.second);
        }
        jfcHandleMap.erase(jfcIt);
    }
}

void RdmaHandleManager::CleanupRdmaHandleEntry(RdmaHandle handle)
{
    auto modeIt = netWorkModeMap.find(handle);
    if (modeIt != netWorkModeMap.end()) {
        DECTOR_TRY_CATCH("rdma handle deinit", HrtRaRdmaDeInit(handle, modeIt->second));
        netWorkModeMap.erase(modeIt);
    }
}

void RdmaHandleManager::CleanupUbHandleEntry(RdmaHandle handle)
{
    auto tokenIt = tokenInfoMap.find(handle);
    if (tokenIt != tokenInfoMap.end()) {
        if (tokenIt->second != nullptr) {
            DECTOR_TRY_CATCH("token id handle destroy", tokenIt->second->Destroy());
        }
        DECTOR_TRY_CATCH("ub handle destroy", HrtRaUbCtxDestroy(handle));
        tokenInfoMap.erase(tokenIt);
    } else {
        DECTOR_TRY_CATCH("ub handle destroy", HrtRaUbCtxDestroy(handle));
    }
}

void RdmaHandleManager::CleanupAuxiliaryMaps(RdmaHandle handle)
{
    DieAndFuncIdMap.erase(handle);
    RtpEnableMap.erase(handle);
}

void RdmaHandleManager::CleanupSingleHandle(const HandleInfo& info)
{
    RdmaHandle handle = info.handle;
    CleanupJfcHandles(handle);
    if (info.protoIndex == RDMA_HANDLE_INDEX) {
        CleanupRdmaHandleEntry(handle);
    }
    if (info.protoIndex == UB_HANDLE_INDEX) {
        CleanupUbHandleEntry(handle);
    }
    CleanupAuxiliaryMaps(handle);
}

void RdmaHandleManager::DeInit(u32 devPhyId)
{
    HCCL_INFO("[RdmaHandleManager][%s] DeInit[%u]", __func__, devPhyId);

    std::lock_guard<std::mutex> lock(managerMutex);
    if (destroyed.load()) {
        return;
    }
    if (devPhyId >= rdmaHandleMap.size()) {
        HCCL_INFO("[RdmaHandleManager][%s] devPhyId[%u] is out of range", __func__, devPhyId);
        return;
    }

    std::vector<HandleInfo> handlesToCleanup;
    CollectHandlesToCleanup(devPhyId, handlesToCleanup);

    for (auto& info : handlesToCleanup) {
        activeHandles_.erase(info.handle);
    }

    for (auto& info : handlesToCleanup) {
        CleanupSingleHandle(info);
    }
}

bool RdmaHandleManager::IsHandleValid(RdmaHandle handle)
{
    if (handle == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(managerMutex);
    if (destroyed.load()) {
        return false;
    }
    return activeHandles_.find(handle) != activeHandles_.end();
}

} // namespace Hccl
