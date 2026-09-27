/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RDMA_HANDLE_MANAGER_H
#define HCCLV2_RDMA_HANDLE_MANAGER_H

#include <mutex>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "port.h"
#include "orion_adapter_hccp.h"
#include "ip_address.h"
#include "buffer_key.h"
#include "tokenInfo_manager.h"

namespace Hccl {

struct LinkProtoTypelHash {
    size_t operator()(LinkProtoType t) const { return static_cast<size_t>(t); }
};

struct HandleInfo {
    RdmaHandle handle;
    u32 protoIndex;
};

class RdmaHandleManager {
public:
    static RdmaHandleManager& GetInstance();

    ~RdmaHandleManager();

    RdmaHandle
    GetByAddr(u32 devPhyId, const LinkProtoType& localProtocolType, IpAddress& localIp, PortDeploymentType type);
    RdmaHandle Get(u32 devPhyId, const PortData& localPort, LinkProtocol linkProtocol);
    RdmaHandle GetByIp(u32 devPhyId, const IpAddress& localIp); // only support ccu create loop channel
    JfcHandle GetJfcHandle(RdmaHandle rdmaHandle, CqCreateInfo& cqInfo, HrtUbJfcMode jfcMode);
    JfcHandle GetJfcHandleAndCqInfo(RdmaHandle rdmaHandle, CqCreateInfo& cqInfo, HrtUbJfcMode jfcMode);
    std::pair<uint32_t, uint32_t> GetDieAndFuncId(RdmaHandle rdmaHandle);
    bool GetRtpEnable(RdmaHandle rdmaHandle);

    std::pair<TokenIdHandle, uint32_t> GetTokenIdInfo(RdmaHandle rdmaHandle, const BufferKey<uintptr_t, u64>& bufKey);
    void PutTokenIdInfo(RdmaHandle rdmaHandle, const BufferKey<uintptr_t, u64>& bufKey, TokenIdHandle tokenIdHandle);
    RdmaHandleManager(const RdmaHandleManager& rdmaHandleManager) = delete;
    RdmaHandleManager& operator=(const RdmaHandleManager& rdmaHandleManager) = delete;
    void DestroyAll();
    void DeInit(u32 devPhyId);
    bool IsHandleValid(RdmaHandle handle);

    HcclResult GetEidByIpv4Addr(const IpAddress& addr, IpAddress& eidAddr);
    void UboeIpv4ToEid(const IpAddress& ipV4Address, IpAddress& eidAddress, u32 devPhyId);

private:
    std::mutex managerMutex;
    std::atomic<bool> destroyed{false};

    // key: devicePhyId, ConnectProtoType, portAddr, 目前仅限于device侧
    std::vector<std::vector<std::unordered_map<IpAddress, RdmaHandle>>> rdmaHandleMap;

    std::unordered_map<RdmaHandle, std::unordered_map<HrtUbJfcMode, JfcHandle, EnumClassHash>> jfcHandleMap;

    std::unordered_map<JfcHandle, CqCreateInfo> cqInfoMap;

    std::unordered_map<RdmaHandle, std::pair<uint32_t, uint32_t>> DieAndFuncIdMap;

    std::unordered_map<RdmaHandle, bool> RtpEnableMap;

    std::unordered_map<RdmaHandle, std::unique_ptr<TokenInfoManager>> tokenInfoMap;

    std::unordered_map<RdmaHandle, HrtNetworkMode> netWorkModeMap;

    std::unordered_map<IpAddress, IpAddress> uboeIpv4EidMap;

    std::unordered_set<RdmaHandle> activeHandles_;

    RdmaHandleManager();

    bool FindCachedJfcHandle(RdmaHandle rdmaHandle, HrtUbJfcMode jfcMode, JfcHandle& handle, CqCreateInfo& cqInfo);

    RdmaHandle Create(u32 devPhyId, const PortData& localPort);
    RdmaHandle
    Create(u32 devPhyId, const LinkProtoType& localProtocolType, const IpAddress& localIp, PortDeploymentType type);

    void CollectHandlesToCleanup(u32 devPhyId, std::vector<HandleInfo>& handlesToCleanup);
    void CleanupJfcHandles(RdmaHandle handle);
    void CleanupRdmaHandleEntry(RdmaHandle handle);
    void CleanupUbHandleEntry(RdmaHandle handle);
    void CleanupAuxiliaryMaps(RdmaHandle handle);
    void CleanupSingleHandle(const HandleInfo& info);
};

} // namespace Hccl

#endif // HCCLV2_RDMA_HANDLE_MANAGER_H
