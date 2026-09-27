/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_net_dev_v2.h"
#include "net_device.h"
#include "inner_net_dev_manager.h"
#include "inner_net_dev.h"
#include "log.h"

using namespace Hccl;
/**
 * @brief 将 HcclNetDevInfos 转换为 NetDevInfo
 * @param src 输入的 HcclNetDevInfos 结构体
 * @param[out] dst 输出的 NetDevInfo 结构体
 * @return HcclResult 转换结果（成功返回HCCL_SUCCESS，失败返回对应错误码）
 */
static Hccl::LinkProtoType ConvertHcclProtoToLinkProto(HcclProtoType hcclProto)
{
    switch (hcclProto) {
        case HCCL_PROTO_TYPE_BUS:
            // 设备间总线直连协议 -> UB（统一总线，涵盖华为UB系列）
            return Hccl::LinkProtoType::UB;
        case HCCL_PROTO_TYPE_TCP:
            // TCP协议 -> 直接映射
            return Hccl::LinkProtoType::TCP;
        case HCCL_PROTO_TYPE_ROCE:
            // RoCE（基于RDMA的以太网协议）-> RDMA
            return Hccl::LinkProtoType::RDMA;
        case HCCL_PROTO_TYPE_UB_CTP:
        case HCCL_PROTO_TYPE_UBC_TP:
        case HCCL_PROTO_TYPE_UB_RTP:
            // 华为统一总线系列协议 -> UB
            return Hccl::LinkProtoType::UB;
        case HCCL_PROTO_TYPE_RESERVED:
            return Hccl::LinkProtoType::HCCS_PCIE;
        default:
            return Hccl::LinkProtoType::HCCS_PCIE;
    }
}

static Hccl::PortDeploymentType ConvertDeploymentType(HcclNetDevDeployment type)
{
    if (type == HCCL_NETDEV_DEPLOYMENT_DEVICE) {
        return Hccl::PortDeploymentType::DEV_NET;
    } else if (type == HCCL_NETDEV_DEPLOYMENT_HOST) {
        return Hccl::PortDeploymentType::HOST_NET;
    } else {
        return Hccl::PortDeploymentType::P2P;
    }
}

static HcclResult ConvertToNetDevInfo(const HcclNetDevInfos& src, Hccl::NetDevInfo& dst)
{
    dst.devId = src.devicePhyId;
    dst.protoType = ConvertHcclProtoToLinkProto(src.addr.protoType);
    dst.type = ConvertDeploymentType(src.netdevDeployment);
    if (dst.type == Hccl::PortDeploymentType::P2P) {
        HCCL_ERROR("Invalid deployment type (devPhyId: %d)", src.devicePhyId);
        return HCCL_E_PARA;
    }
    try {
        if (src.addr.type == HCCL_ADDR_TYPE_IP_V4) {
            // 转换IPv4地址：从 struct in_addr 到 IpAddress
            dst.addr = Hccl::IpAddress(src.addr.addr.s_addr);
        } else if (src.addr.type == HCCL_ADDR_TYPE_IP_V6) {
            // IPv6地址转换：先将in6_addr转换为字符串，再构造IpAddress
            char ipv6Str[INET6_ADDRSTRLEN];
            const char* result = inet_ntop(AF_INET6, &src.addr.addr6, ipv6Str, INET6_ADDRSTRLEN);
            if (result == nullptr) {
                HCCL_ERROR("Invalid result (devPhyId: %d)", src.devicePhyId);
                return HCCL_E_PARA;
            }
            dst.addr = Hccl::IpAddress(ipv6Str, AF_INET6);
        } else {
            HCCL_ERROR("Invalid address type(devPhyId: %d)", src.devicePhyId);
            return HCCL_E_PARA;
        }
    } catch (const std::exception& e) {
        // 捕获IpAddress构造中的异常（如无效地址、不支持的协议族）
        HCCL_ERROR("Failed to convert (devPhyId: %d)", src.devicePhyId);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclNetDevOpenV2(const HcclNetDevInfos* info, HcclNetDev* netDev)
{
    CHK_PTR_NULL(info);
    CHK_PTR_NULL(netDev);

    Hccl::NetDevInfo pltInfo;
    HcclResult ret = ConvertToNetDevInfo(*info, pltInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("ConvertToNetDevInfo fail, devPhyId[%d]", info->devicePhyId);
        return ret;
    }

    Hccl::InnerNetDevManager* netDevMgr = &Hccl::InnerNetDevManager::GetInstance();
    if (netDevMgr == nullptr) {
        HCCL_ERROR("InnerNetDevManager::GetInstance() fail, devPhyId[%d]", info->devicePhyId);
        return HCCL_E_PTR;
    }

    HcclNetDevice* hcclNetDev = nullptr;
    ret = netDevMgr->AddDevice(pltInfo, hcclNetDev);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("AddDevice fail, devPhyId[%d]", info->devicePhyId);
        return ret;
    }
    *netDev = static_cast<HcclNetDev>(hcclNetDev);
    HCCL_INFO("HcclNetDevOpenV2: successfully opened netDev[%p]!", *netDev);
    return HCCL_SUCCESS;
}

HcclResult HcclNetDevCloseV2(HcclNetDev netDev)
{
    HCCL_INFO("[HcclNetDevCloseV2] netDev[%p].", netDev);
    CHK_PTR_NULL(netDev);
    Hccl::InnerNetDevManager* netDevMgr = &Hccl::InnerNetDevManager::GetInstance();
    CHK_PTR_NULL(netDevMgr);
    return netDevMgr->DeleteDevice(static_cast<HcclNetDevice*>(netDev));
}

HcclResult HcclNetDevGetAddrV2(const HcclNetDev netDev, HcclAddress* addr)
{
    CHK_PTR_NULL(netDev);
    CHK_PTR_NULL(addr);

    auto ipAddr = static_cast<HcclNetDevice*>(netDev)->GetNetDevInfo().addr;
    CHK_PTR_NULL(&ipAddr);
    if (ipAddr.GetFamily() == AF_INET) {
        addr->type = HCCL_ADDR_TYPE_IP_V4;
        addr->addr = ipAddr.GetBinaryAddress().addr;
        unsigned char* ipv4Addr = reinterpret_cast<unsigned char*>(&addr->addr);
        HCCL_DEBUG("IPv4 Address: %u.%u.%u.%u", ipv4Addr[0], ipv4Addr[1], ipv4Addr[2], ipv4Addr[3]);
    } else if (ipAddr.GetFamily() == AF_INET6) {
        addr->type = HCCL_ADDR_TYPE_IP_V6;
        addr->addr6 = ipAddr.GetBinaryAddress().addr6;
        unsigned short* ipv6Addr = reinterpret_cast<unsigned short*>(&addr->addr6);
        HCCL_DEBUG(
            "IPv6 Address: %04x:%04x:%04x:%04x:%04x:%04x:%04x:%04x", ipv6Addr[0], ipv6Addr[1], ipv6Addr[2], ipv6Addr[3],
            ipv6Addr[4], ipv6Addr[5], ipv6Addr[6], ipv6Addr[7]);
    } else {
        HCCL_ERROR(
            "HcclNetDevGetAddrV2 fail, devPhyId[%u]", static_cast<HcclNetDevice*>(netDev)->GetNetDevInfo().devId);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclNetDevGetBusAddrV2(HcclDeviceId dstDevId, HcclAddress* busAddr)
{
    (void)dstDevId;
    (void)busAddr;
    HCCL_ERROR("HcclNetDevGetBusAddrV2: failed to get bus address for device[%d], busAddr[%p].", dstDevId, busAddr);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclNetDevGetNicAddrV2(int32_t devicePhyId, HcclAddress** addr, uint32_t* addrNum)
{
    (void)devicePhyId;
    (void)addr;
    (void)addrNum;
    HCCL_ERROR(
        "HcclNetDevGetNicAddrV2: failed to get NIC address for devicePhyId[%d], addr[%p], addrNum[%p].", devicePhyId,
        addr, addrNum);
    return HCCL_E_NOT_SUPPORT;
}
