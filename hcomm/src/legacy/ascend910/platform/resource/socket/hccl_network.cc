/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_types.h>
#include "dlhal_function.h"
#include "dlra_function.h"
#include "sal_pub.h"
#include "adapter_hccp.h"
#include "network_manager_pub.h"
#include "externalinput_pub.h"
#include "hccl_network.h"

namespace hccl {
HcclResult
NetDevContext::Init(NicType nicType, s32 devicePhyId, s32 deviceLogicId, HcclIpAddress localIp, HcclIpAddress backupIp)
{
    devicePhyId_ = devicePhyId;
    deviceLogicId_ = deviceLogicId;
    localIp_ = localIp;
    backupIp_ = backupIp;
    nicType_ = nicType;

    if (nicType == NicType::VNIC_TYPE || nicType == NicType::DEVICE_NIC_TYPE) {
        nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    } else {
        nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_HOST;
    }

    if (static_cast<s32>(devicePhyId) == HOST_DEVICE_ID) {
        deviceLogicId_ = 0;
    }

    if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
        CHK_RET(NetworkManager::GetInstance(deviceLogicId_).StartHostNet(localIp, hostSocketHandle_));
    }

    return HCCL_SUCCESS;
}

HcclResult NetDevContext::GetinfoConfig(const HcclNetDevInfos* info)
{
    CHK_PTR_NULL(info);
    devicePhyId_ = info->devicePhyId;
    isBackup_ = info->isBackup;
    u32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceIndexByPhyId(devicePhyId_, deviceLogicId));
    deviceLogicId_ = deviceLogicId;
    CHK_RET(ConvertIP(info->addr));
    netDevDeployment_ = info->netdevDeployment;
    protoType_ = info->addr.protoType;

    HCCL_INFO("[NetDevContext][InitV2] netDevDeployment_ = [%u], protoType_ = [%u]", netDevDeployment_, protoType_);
    return HCCL_SUCCESS;
}

HcclResult NetDevContext::ConvertIP(const HcclAddress address)
{
    s32 family = AF_INET;
    HcclInAddr temp;
    if (address.type == HCCL_ADDR_TYPE_IP_V4) {
        family = AF_INET;
        temp.addr = address.addr;
    } else if (address.type == HCCL_ADDR_TYPE_IP_V6) {
        family = AF_INET6;
        temp.addr6 = address.addr6;
    } else {
        HCCL_ERROR(
            "[NetDevContext][InitV2]this addrType [%u] is not supported, please check the configuration.",
            address.type);
        return HCCL_E_PARA;
    }
    HcclIpAddress localIptemp(family, temp);
    localIp_ = localIptemp;
    return HCCL_SUCCESS;
}

// 初始化进程和设备
HcclResult NetDevContext::InitV2(const HcclNetDevInfos* info)
{
    CHK_PTR_NULL(info);
    CHK_RET(GetinfoConfig(info));
    // 需要保存新的协议类型
    if (netDevDeployment_ == HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE) {
        switch (protoType_) {
            case HCCL_PROTO_TYPE_ROCE: {
                nicType_ = NicType::DEVICE_NIC_TYPE;
                nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_DEVICE;
                bool rdmaFlag = !GetExternalInputHcclIsTcpMode();
                if (!rdmaFlag) {
                    HCCL_ERROR(
                        "[NetDevContext][InitV2]rdmaFlag and protoType are not equal, please check the configuration.");
                    return HCCL_E_PARA;
                }
                NetworkMode netMode;
                NetworkManager::GetInstance(deviceLogicId_).GetNetworkMode(netMode);
                NotifyTypeT notifyType;
                NetworkManager::GetInstance(deviceLogicId_).GetNotifyType(notifyType);
                CHK_RET(NetworkManager::GetInstance(deviceLogicId_)
                            .CreateRdmaHandle(localIp_, isBackup_, netMode, notifyType, netDevDeployment_));
                CHK_RET(NetworkManager::GetInstance(deviceLogicId_).CreateNicSocketHandle(localIp_));
                NetworkManager::GetInstance(deviceLogicId_).GetRdmaHandleByIpAddr(localIp_, handle_);

                CHK_PTR_NULL(handle_);
                HCCL_INFO("[NetDevContext][InitV2]Deployment is device and proto is roce");
                break;
            }
            case HCCL_PROTO_TYPE_BUS: {
                nicType_ = NicType::VNIC_TYPE;
                nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_DEVICE;
                CHK_RET(NetworkManager::GetInstance(deviceLogicId_).CreateVnicSocketHandle(localIp_));
                RaResourceInfo raResourceInfo;
                NetworkManager::GetInstance(deviceLogicId_).GetRaResourceInfo(raResourceInfo);
                IpSocket& sock = raResourceInfo.vnicSocketMap[localIp_];
                handle_ = sock.nicSocketHandle;
                CHK_PTR_NULL(handle_);
                HCCL_INFO("[NetDevContext][InitV2]Deployment is device and proto is bus");
                break;
            }

            case HCCL_PROTO_TYPE_TCP: {
                nicType_ = NicType::DEVICE_NIC_TYPE;
                nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_DEVICE;
                bool rdmaFlag = !GetExternalInputHcclIsTcpMode();
                if (rdmaFlag) {
                    HCCL_ERROR("[NetDevContext][InitV2]rdmaFlag is ERROR, please check the configuration.");
                    return HCCL_E_PARA;
                }
                CHK_RET(NetworkManager::GetInstance(deviceLogicId_).CreateNicSocketHandle(localIp_));
                NetworkManager::GetInstance(deviceLogicId_).GetNicHandleByIpAddr(localIp_, handle_);
                CHK_PTR_NULL(handle_);
                HCCL_INFO("[NetDevContext][InitV2]Deployment is device and proto is tcp");
                break;
            }

            default: // 保留
                HCCL_ERROR(
                    "[NetDevContext][DeinitV2]this prototype [%u] is not supported in device mode, please check the "
                    "configuration.",
                    protoType_);
                return HCCL_E_NOT_SUPPORT;
        }
    } else if (netDevDeployment_ == HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_HOST) {
        switch (protoType_) {
            case HCCL_PROTO_TYPE_TCP: {
                nicType_ = NicType::HOST_NIC_TYPE;
                nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_HOST;
                CHK_RET(NetworkManager::GetInstance(deviceLogicId_).CreateHostSocketHandle(localIp_, handle_));
                RaResourceInfo raResourceInfo;
                NetworkManager::GetInstance(deviceLogicId_).GetRaResourceInfo(raResourceInfo);
                IpSocket& sock = raResourceInfo.hostNetSocketMap[localIp_];
                handle_ = sock.nicSocketHandle;
                CHK_PTR_NULL(handle_);
                HCCL_INFO("[NetDevContext][InitV2]Deployment is host and proto is tcp");
                break;
            }
            case HCCL_PROTO_TYPE_ROCE: {
                nicType_ = NicType::HOST_NIC_TYPE;
                nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_HOST;
                NetworkMode netMode = NETWORK_PEER_ONLINE;
                NotifyTypeT notifyType = NOTIFY;
                CHK_RET(NetworkManager::GetInstance(deviceLogicId_)
                            .CreateRdmaHandle(localIp_, isBackup_, netMode, notifyType, netDevDeployment_));
                NetworkManager::GetInstance(deviceLogicId_).GetRdmaHandleByIpAddr(localIp_, handle_);
                CHK_PTR_NULL(handle_);
                HCCL_INFO("[NetDevContext][InitV2]Deployment is host and proto is roce");
                break;
            }
            default: // 保留
                HCCL_ERROR(
                    "[NetDevContext][DeinitV2]this prototype [%u] is not supported in host mode, please check the "
                    "configuration.",
                    protoType_);
                return HCCL_E_NOT_SUPPORT;
        }
    } else {
        HCCL_ERROR(
            "[NetDevContext][DeinitV2]this Deployment [%u] is not supported, please check the configuration.",
            netDevDeployment_);
        return HCCL_E_NOT_SUPPORT;
        // 保留
    }

    return HCCL_SUCCESS;
}

HcclResult NetDevContext::Deinit()
{
    if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
        CHK_RET(NetworkManager::GetInstance(deviceLogicId_).StopHostNet(hostSocketHandle_, localIp_));
    }
    return HCCL_SUCCESS;
}

HcclResult NetDevContext::DeinitV2()
{
    if (netDevDeployment_ == HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE) {
        switch (protoType_) {
            case HCCL_PROTO_TYPE_ROCE:
                CHK_RET(NetworkManager::GetInstance(deviceLogicId_).StopRdmaHandle(localIp_, netDevDeployment_));
                break;
            case HCCL_PROTO_TYPE_BUS:
                CHK_RET(NetworkManager::GetInstance(deviceLogicId_).StopVnicSocketHandle(localIp_));
                break;
            case HCCL_PROTO_TYPE_TCP:
                CHK_RET(NetworkManager::GetInstance(deviceLogicId_).StopNicSocketHandle(localIp_));
                break;
            default: // 保留
                HCCL_ERROR(
                    "[NetDevContext][DeinitV2]this prototype [%u] is not supported in host mode, please check the "
                    "configuration.",
                    protoType_);
                return HCCL_E_NOT_SUPPORT;
        }
    } else if (netDevDeployment_ == HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_HOST) {
        switch (protoType_) {
            case HCCL_PROTO_TYPE_TCP:
                CHK_RET(NetworkManager::GetInstance(deviceLogicId_).StopHostSocketHandle(localIp_));
                break;
            case HCCL_PROTO_TYPE_ROCE:
                CHK_RET(NetworkManager::GetInstance(deviceLogicId_).StopRdmaHandle(localIp_, netDevDeployment_));
                break;
            default: // 保留
                HCCL_ERROR(
                    "[NetDevContext][DeinitV2]this prototype [%u] is not supported in device mode, please check the "
                    "configuration.",
                    protoType_);
                return HCCL_E_NOT_SUPPORT;
        }
    } else {
        HCCL_ERROR(
            "[NetDevContext][DeinitV2]this Deployment [%u] is not supported, please check the configuration.",
            netDevDeployment_);
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

void NetDevContext::SetTlsStatus(TlsStatus tlsStatus)
{
    tlsStatus_ = tlsStatus;
    HCCL_INFO("[NetDevContext][SetTlsStatus]devicePhyId[%d], set tlsStatus[%d]", devicePhyId_, tlsStatus);
    return;
}

void NetDevContext::SetIsNotNeedGetTlsStatus(bool isNotNeedGetTlsStatus)
{
    isNotNeedGetTlsStatus_ = isNotNeedGetTlsStatus;
    return;
}
} // namespace hccl

HcclResult
HcclNetInit(NICDeployment nicDeploy, s32 devicePhyId, s32 deviceLogicId, bool enableWhitelistFlag, bool hasBackup)
{
    CHK_RET(hccl::DlRaFunction::GetInstance().DlRaFunctionInit());
    if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        CHK_RET(hccl::DlHalFunction::GetInstance().DlHalFunctionInit());
        bool isHostUseDevNic;
        CHK_RET(IsHostUseDevNic(isHostUseDevNic));
        u32 tempDevicePhyId = hasBackup ? static_cast<u32>(devicePhyId) : hccl::DEFAULT_PHY_ID;
        HCCL_DEBUG(
            "[%s]start NetworkManager Init, deviceLogicId[%u], devicePhyId[%u], nicDeploy[%d], hasBackup[%d],"
            " tempDevicePhyId[%u]",
            __func__, deviceLogicId, devicePhyId, nicDeploy, hasBackup, tempDevicePhyId);
        CHK_RET(hccl::NetworkManager::GetInstance(deviceLogicId)
                    .Init(
                        NICDeployment::NIC_DEPLOYMENT_DEVICE, enableWhitelistFlag, tempDevicePhyId, isHostUseDevNic,
                        hasBackup));
    } else {
        CHK_RET(hccl::NetworkManager::GetInstance(deviceLogicId)
                    .Init(NICDeployment::NIC_DEPLOYMENT_HOST, enableWhitelistFlag, devicePhyId));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclNetDeInit(NICDeployment nicDeploy, [[maybe_unused]] s32 devicePhyId, s32 deviceLogicId, bool hasBackup)
{
    CHK_RET(hccl::NetworkManager::GetInstance(deviceLogicId).DeInit(nicDeploy, false, hasBackup));
    return HCCL_SUCCESS;
}

HcclResult HcclNetOpenDev(
    HcclNetDevCtx* netDevCtx, NicType nicType, s32 devicePhyId, s32 deviceLogicId, hccl::HcclIpAddress localIp,
    hccl::HcclIpAddress backupIp)
{
    CHK_PTR_NULL(netDevCtx);

    hccl::NetDevContext* pNetDevCtx = new (std::nothrow) hccl::NetDevContext();
    CHK_PTR_NULL(pNetDevCtx);

    HcclResult ret = pNetDevCtx->Init(nicType, devicePhyId, deviceLogicId, localIp, backupIp);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Init][Port]Init fail. ret[%u]", ret);
        delete pNetDevCtx;
        pNetDevCtx = nullptr;
        return ret;
    }

    *netDevCtx = pNetDevCtx;

    return HCCL_SUCCESS;
}

void HcclNetCloseDev(HcclNetDevCtx netDevCtx)
{
    if (netDevCtx == nullptr) {
        HCCL_ERROR("[HcclNetCloseDev] netDevCtx is nullptr");
        return;
    }
    hccl::NetDevContext* pNetDevCtx = static_cast<hccl::NetDevContext*>(netDevCtx);

    HcclResult ret = pNetDevCtx->Deinit();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[DeInit][Port]DeInit fail. ret[%u]", ret);
    }

    delete pNetDevCtx;
}

HcclResult HcclNetDevGetNicType(HcclNetDevCtx netDevCtx, NicType* nicType)
{
    CHK_PTR_NULL(netDevCtx);
    CHK_PTR_NULL(nicType);
    hccl::NetDevContext* pNetDevCtx = static_cast<hccl::NetDevContext*>(netDevCtx);

    *nicType = pNetDevCtx->GetNicType();
    return HCCL_SUCCESS;
}

HcclResult HcclNetDevGetLocalIp(HcclNetDevCtx netDevCtx, hccl::HcclIpAddress& localIp)
{
    CHK_PTR_NULL(netDevCtx);
    hccl::NetDevContext* pNetDevCtx = static_cast<hccl::NetDevContext*>(netDevCtx);

    localIp = pNetDevCtx->GetLocalIp();
    return HCCL_SUCCESS;
}

HcclResult HcclNetDevGetProtoType(HcclNetDevCtx netDevCtx, u32& proto)
{
    CHK_PTR_NULL(netDevCtx);
    hccl::NetDevContext* pNetDevCtx = static_cast<hccl::NetDevContext*>(netDevCtx);

    proto = (u32)pNetDevCtx->GetProtoType();
    return HCCL_SUCCESS;
}

HcclResult HcclNetDevSetProtoType(HcclNetDevCtx netDevCtx, u32 proto)
{
    CHK_PTR_NULL(netDevCtx);
    hccl::NetDevContext* pNetDevCtx = static_cast<hccl::NetDevContext*>(netDevCtx);

    pNetDevCtx->SetProtoType(proto);
    return HCCL_SUCCESS;
}

HcclResult HcclNetDevGetPortStatus(HcclNetDevCtx netDevCtx, bool& portStatus)
{
    CHK_PTR_NULL(netDevCtx);
    hccl::NetDevContext* pNetDevCtx = static_cast<hccl::NetDevContext*>(netDevCtx);
    u32 devicePhyId = static_cast<u32>(pNetDevCtx->GetPhyId());
    RdmaHandle rdmaHandle = nullptr;
    enum PortStatus status;
    CHK_RET(HrtRaRdmaGetHandle(devicePhyId, rdmaHandle));
    CHK_RET(hrtRaRdevGetPortStatus(rdmaHandle, &status));
    portStatus = (status == PORT_STATUS_ACTIVE);
    HCCL_RUN_INFO("[HcclNetDevGetPortStatus]devicePhysicID_[%u], portStatus_[%d]", devicePhyId, portStatus);
    return HCCL_SUCCESS;
}

HcclResult HcclNetDevGetTlsStatus(HcclNetDevCtx netDevCtx, TlsStatus* tlsStatus)
{
    CHK_PTR_NULL(netDevCtx);
    CHK_PTR_NULL(tlsStatus);

    hccl::NetDevContext* pNetDevCtx = static_cast<hccl::NetDevContext*>(netDevCtx);
    std::lock_guard<std::mutex> lock(pNetDevCtx->mu_);
    // tls开关状态多个通信域只需要查询一次，后续一直使用第一次查询结果
    if (pNetDevCtx->IsNotNeedGetTlsStatus()) {
        *tlsStatus = pNetDevCtx->GettlsStatus();
        return HCCL_SUCCESS;
    }

    u32 devicePhyId = static_cast<u32>(pNetDevCtx->GetPhyId());
    struct RaInfo raInfo = {};
    raInfo.mode = static_cast<int>(pNetDevCtx->GetNicDeployment());
    raInfo.phyId = devicePhyId;
    bool tlsEnable = false;
    HcclResult ret = HrtRaGetTlsEnable(&raInfo, &tlsEnable);
    if (ret == HCCL_E_NOT_SUPPORT) {
        pNetDevCtx->SetTlsStatus(TlsStatus::UNKNOWN);
    } else if (tlsEnable) {
        pNetDevCtx->SetTlsStatus(TlsStatus::ENABLE);
    } else {
        pNetDevCtx->SetTlsStatus(TlsStatus::DISABLE);
    }
    *tlsStatus = pNetDevCtx->GettlsStatus();
    pNetDevCtx->SetIsNotNeedGetTlsStatus(true);
    return ret;
}
