/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ts_roce_endpoint.h"
#include "log.h"
#include "hccl_net_dev.h"
#include "aicpu_ts_roce_reged_mem_mgr.h"
#include "adapter_rts_common.h"
#include "hccl_network.h"
#include "network_manager_pub.h"
#include <exception>

namespace hcomm {

AicpuTsRoceEndpoint::AicpuTsRoceEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

AicpuTsRoceEndpoint::~AicpuTsRoceEndpoint()
{
    ctxHandle_ = nullptr;
    // 析构顺序保持：先释放serverSocketContext，再释放共享 netDev
    if (serverSocketContext_.has_value()) {
        serverSocketContext_->ReleaseListenSocketRefs();
    }
    ReleaseSharedNetDev();
}

bool AicpuTsRoceEndpoint::IsCtxHandleValid() const
{
    // 本类 ctxHandle_ 来自 NetworkManager::GetRdmaHandleByIpAddr（生命周期由 NetworkManager 的
    // nicSocketMap 管理），不进入 RdmaHandleManager::activeHandles_，因此不能用
    // RdmaHandleManager::IsHandleValid 校验（对不在册句柄恒返回 false）；判空即为有效性校验。
    return ctxHandle_ != nullptr;
}

std::mutex& AicpuTsRoceEndpoint::NetDevMapMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<uint32_t, AicpuTsNetDevSlot>& AicpuTsRoceEndpoint::GetNetDevMap()
{
    static std::unordered_map<uint32_t, AicpuTsNetDevSlot> netDevMap;
    return netDevMap;
}

HcclResult AicpuTsRoceEndpoint::AcquireSharedNetDev(uint32_t devicePhyId, const HcclNetDevInfos& info)
{
    std::lock_guard<std::mutex> lk(NetDevMapMutex());
    auto& netDevMap = GetNetDevMap();
    const auto it = netDevMap.find(devicePhyId);
    if (it != netDevMap.end()) {
        it->second.refCount++;
        netDev_ = it->second.netDev;
        netDevRefPhyId_ = devicePhyId;
        HCCL_INFO(
            "[AicpuTsRoceEndpoint][%s] reuse HcclNetDev for devicePhyId[%u], ref[%u]", __func__, devicePhyId,
            it->second.refCount);
        return HCCL_SUCCESS;
    }

    HcclNetDev netDev = nullptr;
    const HcclResult ret = HcclNetDevOpen(&info, &netDev);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[AicpuTsRoceEndpoint][%s] HcclNetDevOpen failed, ret[%d]", __func__, ret);
        return ret;
    }
    netDevMap[devicePhyId] = AicpuTsNetDevSlot{netDev, 1U};
    netDev_ = netDev;
    netDevRefPhyId_ = devicePhyId;
    return HCCL_SUCCESS;
}

void AicpuTsRoceEndpoint::ReleaseSharedNetDev()
{
    if (netDevRefPhyId_ == UINT32_MAX) {
        return;
    }
    const uint32_t key = netDevRefPhyId_;
    netDevRefPhyId_ = UINT32_MAX;
    HcclNetDev toClose = nullptr;
    {
        std::lock_guard<std::mutex> lk(NetDevMapMutex());
        auto& netDevMap = GetNetDevMap();
        const auto it = netDevMap.find(key);
        if (it == netDevMap.end()) {
            HCCL_ERROR("[AicpuTsRoceEndpoint][ReleaseSharedNetDev] missing slot for devicePhyId[%u]", key);
        } else {
            if (it->second.refCount > 0U) {
                it->second.refCount--;
            }
            if (it->second.refCount == 0U) {
                toClose = it->second.netDev;
                (void)netDevMap.erase(it);
            }
        }
    }
    netDev_ = nullptr;
    if (toClose != nullptr) {
        // 监听引用状态迁移至 serverSocketContext_：未释放监听引用时不额外停止 NicSocketHandle
        const bool hasListenRef = serverSocketContext_.has_value() && serverSocketContext_->HasListenSocketRef();
        if (!hasListenRef) {
            ReleaseNicSocketHandle(toClose);
        }
        HCCL_INFO("[AicpuTsRoceEndpoint][ReleaseSharedNetDev] closing HcclNetDev for devicePhyId[%u]", key);
        const HcclResult ret = HcclNetDevClose(toClose);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[AicpuTsRoceEndpoint][ReleaseSharedNetDev] HcclNetDevClose failed, ret[%d]", ret);
        }
    }
}

void AicpuTsRoceEndpoint::ReleaseNicSocketHandle(HcclNetDev netDev) const
{
    auto* netDevCtx = static_cast<hccl::NetDevContext*>(netDev);
    if (netDevCtx == nullptr) {
        return;
    }
    const hccl::HcclIpAddress localIp = netDevCtx->GetLocalIp();
    const HcclResult ret = hccl::NetworkManager::GetInstance(netDevCtx->GetLogicId()).StopNicSocketHandle(localIp);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING(
            "[AicpuTsRoceEndpoint][%s] StopNicSocketHandle failed, ip[%s], ret[%d]", __func__,
            localIp.GetReadableAddress(), ret);
    }
}

HcclResult AicpuTsRoceEndpoint::AcquireRdmaContext(uint32_t devPhyId, const EndpointDesc& endpointDesc)
{
    HcclNetDevInfos info;
    info.addr.protoType = HCCL_PROTO_TYPE_ROCE;
    CHK_RET(CommAddrTypeToHcclAddressType(endpointDesc.commAddr.type, info.addr.type));
    if (endpointDesc.commAddr.type == COMM_ADDR_TYPE_IP_V4) {
        info.addr.addr = endpointDesc.commAddr.addr;
    } else {
        info.addr.addr6 = endpointDesc.commAddr.addr6;
    }
    info.netdevDeployment = HCCL_NETDEV_DEPLOYMENT_DEVICE;
    info.devicePhyId = static_cast<int32_t>(devPhyId);
    HcclResult ret = AcquireSharedNetDev(devPhyId, info);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    auto* netDevCtx = static_cast<hccl::NetDevContext*>(netDev_);
    if (netDevCtx == nullptr) {
        ReleaseSharedNetDev();
        return HCCL_E_PTR;
    }
    const hccl::HcclIpAddress ipAddr = netDevCtx->GetLocalIp();
    RdmaHandle rdmaHandle = nullptr;
    ret = hccl::NetworkManager::GetInstance(netDevCtx->GetLogicId()).GetRdmaHandleByIpAddr(ipAddr, rdmaHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s]call trace: hcclRet -> %d", __func__, ret);
        ReleaseSharedNetDev();
        return ret;
    }
    ctxHandle_ = rdmaHandle;
    if (ctxHandle_ == nullptr) {
        HCCL_ERROR(
            "[%s]errNo[0x%016llx]ptr [ctxHandle_] is nullptr, return HCCL_E_PTR", __func__,
            HCCL_ERROR_CODE(HCCL_E_PTR));
        ReleaseSharedNetDev();
        return HCCL_E_PTR;
    }
    HCCL_INFO(
        "AicpuTsRoceEndpoint::%s success, devPhyId[%u], ipAddr[%s], ctxHandle[%p]", __func__, devPhyId,
        ipAddr.GetReadableAddress(), ctxHandle_);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceEndpoint::Init()
{
    HCCL_INFO("[%s] localEndpoint protocol[%d]", __func__, endpointDesc_.protocol);

    if (endpointDesc_.loc.locType != ENDPOINT_LOC_TYPE_DEVICE) {
        HCCL_INFO("[AicpuTsRoceEndpoint][%s] AicpuTsRoceEndpoint not support host", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    s32 devId = 0;
    CHK_RET(hrtGetDevice(&devId));
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(devId, devPhyId));

    HcclResult ret = AcquireRdmaContext(devPhyId, endpointDesc_);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    serverSocketContext_.emplace(netDev_, netDevRefPhyId_);
    try {
        regedMemMgr_ = std::make_shared<AicpuTsRoceRegedMemMgr>(netDev_, ctxHandle_);
    } catch (std::exception& e) {
        HCCL_ERROR("[%s]Failed, exception caught:%s", __func__, e.what());
        ctxHandle_ = nullptr;
        serverSocketContext_.reset();
        ReleaseSharedNetDev();
        return HCCL_E_PTR;
    }

    return HCCL_SUCCESS;
}

} // namespace hcomm
