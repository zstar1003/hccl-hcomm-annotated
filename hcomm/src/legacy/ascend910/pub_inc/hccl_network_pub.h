/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_NETWORK_PUB_H
#define HCCL_NETWORK_PUB_H

#include "hccl_common.h"
#include "hccl_ip_address.h"

using HcclNetDevCtx = void*;

struct HcclNetDevInfo {
    s32 devicePhyId;
    s32 deviceLogicId;
    u32 superDeviceId;
    u32 rsvd;
};

enum class NicType { VNIC_TYPE = 0, DEVICE_NIC_TYPE, HOST_NIC_TYPE };

enum class TlsStatus {
    UNKNOWN = -1, // 不支持查询
    DISABLE = 0,  //  未使能
    ENABLE,       //  使能
};

HcclResult HcclNetInit(
    NICDeployment nicDeploy, s32 devicePhyId, s32 deviceLogicId, bool enableWhitelistFlag, bool hasBackup = false);
HcclResult HcclNetDeInit(NICDeployment nicDeploy, s32 devicePhyId, s32 deviceLogicId, bool hasBackup = false);

HcclResult HcclNetOpenDev(
    HcclNetDevCtx* netDevCtx, NicType nicType, s32 devicePhyId, s32 deviceLogicId, hccl::HcclIpAddress localIp,
    hccl::HcclIpAddress backupIp = hccl::HcclIpAddress(0));
void HcclNetCloseDev(HcclNetDevCtx netDevCtx);

HcclResult HcclNetDevGetNicType(HcclNetDevCtx netDevCtx, NicType* nicType);
HcclResult HcclNetDevGetLocalIp(HcclNetDevCtx netDevCtx, hccl::HcclIpAddress& localIp);
HcclResult HcclNetDevGetProtoType(HcclNetDevCtx netDevCtx, u32& proto);
HcclResult HcclNetDevSetProtoType(HcclNetDevCtx netDevCtx, u32 proto);
HcclResult HcclNetDevGetPortStatus(HcclNetDevCtx netDevCtx, bool& portStatus);
HcclResult HcclNetDevGetTlsStatus(HcclNetDevCtx netDevCtx, TlsStatus* tlsStatus);
#endif
