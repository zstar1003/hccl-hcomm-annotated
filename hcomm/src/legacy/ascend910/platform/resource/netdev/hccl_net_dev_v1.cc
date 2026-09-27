/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_network.h"
#include <sys/socket.h>
#include "hccl/base.h"
#include "log.h"
#include "adapter_rts_common.h"
#include "dlra_function.h"
#include "network_manager_pub.h"
#include "dlhal_function.h"
#include "adapter_hccp_common.h"
#include "hccl_net_dev_v1.h"

using namespace std;

HcclResult HcclNetDevOpenV1(const HcclNetDevInfos* info, HcclNetDev* netDev)
{
    CHK_PTR_NULL(netDev);
    CHK_PTR_NULL(info);
    static bool flag = false;
    if (UNLIKELY(flag == false)) {
        CHK_RET(hccl::DlHalFunction::GetInstance().DlHalFunctionInit());
        CHK_RET(hccl::DlRaFunction::GetInstance().DlRaFunctionInit());
    }
    // 拉起进程 对比HcclNetInit
    u32 deviceLogicId;
    bool hasBackup = info->isBackup; // 从外部判断是否有备份
    CHK_RET(hrtGetDeviceIndexByPhyId(info->devicePhyId, deviceLogicId));
    switch (info->netdevDeployment) {
        case HCCL_NETDEV_DEPLOYMENT_HOST: {
            CHK_RET(hccl::NetworkManager::GetInstance(deviceLogicId)
                        .InitV2(NICDeployment::NIC_DEPLOYMENT_HOST, hasBackup, info->devicePhyId));
            break;
        }
        case HCCL_NETDEV_DEPLOYMENT_DEVICE: {
            bool isHostUseDevNic;
            CHK_RET(IsHostUseDevNic(isHostUseDevNic));
            u32 tempDevicePhyId = hasBackup ? static_cast<u32>(info->devicePhyId) : hccl::DEFAULT_PHY_ID;
            HCCL_DEBUG(
                "[%s]start HcclNetDevOpen, deviceLogicId[%u], devicePhyId[%u], nicDeploy[%d], hasBackup[%d],"
                " tempDevicePhyId[%u]",
                __func__, deviceLogicId, info->devicePhyId, info->netdevDeployment, hasBackup, tempDevicePhyId);
            CHK_RET(hccl::NetworkManager::GetInstance(deviceLogicId)
                        .InitV2(NICDeployment::NIC_DEPLOYMENT_DEVICE, hasBackup, tempDevicePhyId, isHostUseDevNic));
            break;
        }
        default: {
            HCCL_ERROR("[HcclNetDevOpen]No Such HcclNetDevDeployment: %d", info->netdevDeployment);
            *netDev = nullptr;
            return HCCL_E_NOT_SUPPORT;
        }
    }
    // 实现设备

    hccl::NetDevContext* pNetDevCtx = new (std::nothrow) hccl::NetDevContext();
    CHK_PTR_NULL(pNetDevCtx);

    HcclResult ret = pNetDevCtx->InitV2(info);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclNetDevOpen][InitV2]Init fail. ret[%u]", ret);
        delete pNetDevCtx;
        pNetDevCtx = nullptr;
        *netDev = nullptr;
        return ret;
    }

    *netDev = pNetDevCtx;
    return HCCL_SUCCESS;
}

HcclResult HcclNetDevCloseV1(HcclNetDev netDev)
{
    // 先销毁设备
    CHK_PTR_NULL(netDev);

    hccl::NetDevContext* pNetDevCtx = static_cast<hccl::NetDevContext*>(netDev);
    bool isBackup = pNetDevCtx->GetIsBackup();
    HcclResult ret = pNetDevCtx->DeinitV2();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclNetDevClose] NetDev Close fail. ret[%u]", ret);
    }

    // 再销毁进程
    switch (pNetDevCtx->GetNetDevDeployment()) {
        case HCCL_NETDEV_DEPLOYMENT_HOST: {
            HcclResult ret_temp = hccl::NetworkManager::GetInstance(pNetDevCtx->GetLogicId())
                                      .DeInitV2(NICDeployment::NIC_DEPLOYMENT_HOST, isBackup, false);
            if (ret_temp != HCCL_SUCCESS) {
                HCCL_ERROR("[HcclNetDevClose] DEPLOYMENT_HOST close fail");
                delete pNetDevCtx;
                return ret_temp;
            }
            break;
        }
        case HCCL_NETDEV_DEPLOYMENT_DEVICE: {
            HcclResult ret_temp = hccl::NetworkManager::GetInstance(pNetDevCtx->GetLogicId())
                                      .DeInitV2(NICDeployment::NIC_DEPLOYMENT_DEVICE, isBackup, false);
            if (ret_temp != HCCL_SUCCESS) {
                HCCL_ERROR("[HcclNetDevClose] DEPLOYMENT_DEVICE close fail");
                delete pNetDevCtx;
                return ret_temp;
            }
            break;
        }
        default:
            HCCL_ERROR("[HcclNetDevClose]No Such HcclNetDevDeployment: %d", pNetDevCtx->GetNetDevDeployment());
            delete pNetDevCtx;
            return HCCL_E_NOT_SUPPORT;
    }

    delete pNetDevCtx;
    return ret;
}

HcclResult HcclNetDevGetAddrV1(HcclNetDev netDev, HcclAddress* addr)
{
    CHK_PTR_NULL(netDev);
    CHK_PTR_NULL(addr);

    hccl::NetDevContext* pNetDevCtx = static_cast<hccl::NetDevContext*>(netDev);
    hccl::HcclIpAddress ipAddr = pNetDevCtx->GetLocalIp();
    addr->protoType = pNetDevCtx->GetProtoType();
    hccl::NetworkManager::GetInstance(pNetDevCtx->GetLogicId()).HcclIpAddressConvertHcclAddr(addr, &ipAddr);
    return HCCL_SUCCESS;
}

HcclResult HcclNetDevGetBusAddrV1(HcclDeviceId dstDevId, HcclAddress* busAddr)
{
    CHK_PTR_NULL(busAddr);

    s32 localDeviceLogicId;
    u32 localDeviceId;
    CHK_RET(hrtGetDevice(&localDeviceLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(localDeviceLogicId), localDeviceId));
    // 先创建进程
    bool isHostUseDevNic;
    CHK_RET(IsHostUseDevNic(isHostUseDevNic));
    u32 tempDevicePhyId = hccl::DEFAULT_PHY_ID;
    HCCL_DEBUG(
        "[%s]HcclNetDevGetBusAddr, deviceLogicId[%u], devicePhyId[%u], nicDeploy[%d], hasBackup[%d],"
        " tempDevicePhyId[%u]",
        __func__, localDeviceLogicId, dstDevId.devicePhyId, NICDeployment::NIC_DEPLOYMENT_DEVICE, false,
        tempDevicePhyId);
    CHK_RET(hccl::NetworkManager::GetInstance(localDeviceLogicId)
                .InitV2(NICDeployment::NIC_DEPLOYMENT_DEVICE, false, tempDevicePhyId, isHostUseDevNic));

    // 参考 Heartbeat::GetConnInfo
    hccl::HcclIpAddress vnicIP(localDeviceId);
    if (dstDevId.superDeviceId != SUPER_DEVICE_ID_INVALID) {
        CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
            localDeviceId, DeviceIdType::DEVICE_ID_TYPE_SDID, dstDevId.superDeviceId, vnicIP));
    } else {
        CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
            localDeviceId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, dstDevId.devicePhyId, vnicIP));
    }

    hccl::NetworkManager::GetInstance(localDeviceLogicId).HcclIpAddressConvertHcclAddr(busAddr, &vnicIP);
    busAddr->protoType = HCCL_PROTO_TYPE_BUS;
    HCCL_INFO("[HcclNetDevGetBusAddr] vnicIP [%s] ", vnicIP.GetReadableAddress());
    // 销毁进程
    CHK_RET(hccl::NetworkManager::GetInstance(localDeviceLogicId)
                .DeInitV2(NICDeployment::NIC_DEPLOYMENT_DEVICE, false, false));

    return HCCL_SUCCESS;
}

HcclResult HcclNetDevGetNicAddrV1(int32_t devicePhyId, HcclAddress** addr, uint32_t* addrNum)
{
    CHK_PTR_NULL(addrNum);
    CHK_PTR_NULL(addr);

    u32 deviceLogicId;
    CHK_RET(hrtGetDeviceIndexByPhyId(devicePhyId, deviceLogicId));

    // 先创建进程
    bool isHostUseDevNic;
    CHK_RET(IsHostUseDevNic(isHostUseDevNic));
    u32 tempDevicePhyId = hccl::DEFAULT_PHY_ID;
    HCCL_DEBUG(
        "[%s]HcclNetDevGetBusAddr, deviceLogicId[%u], devicePhyId[%u], nicDeploy[%d], hasBackup[%d],"
        " tempDevicePhyId[%u]",
        __func__, deviceLogicId, devicePhyId, NICDeployment::NIC_DEPLOYMENT_DEVICE, false, tempDevicePhyId);
    CHK_RET(hccl::NetworkManager::GetInstance(deviceLogicId)
                .InitV2(NICDeployment::NIC_DEPLOYMENT_DEVICE, false, tempDevicePhyId, isHostUseDevNic));
    CHK_RET(hccl::NetworkManager::GetInstance(deviceLogicId).GetNicIp(devicePhyId, addr, addrNum));

    // 销毁进程
    CHK_RET(
        hccl::NetworkManager::GetInstance(deviceLogicId).DeInitV2(NICDeployment::NIC_DEPLOYMENT_DEVICE, false, false));
    return HCCL_SUCCESS;
}
