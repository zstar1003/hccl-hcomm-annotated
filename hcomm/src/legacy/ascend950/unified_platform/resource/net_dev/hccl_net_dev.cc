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
#include "hccl_net_dev.h"

using namespace std;

HcclResult HcclNetDevOpen(const HcclNetDevInfos* info, HcclNetDev* netDev) { return HcclNetDevOpenV2(info, netDev); }

HcclResult HcclNetDevClose(HcclNetDev netDev) { return HcclNetDevCloseV2(netDev); }

HcclResult HcclNetDevGetAddr(HcclNetDev netDev, HcclAddress* addr) { return HcclNetDevGetAddrV2(netDev, addr); }

HcclResult HcclNetDevGetBusAddr(HcclDeviceId dstDevId, HcclAddress* busAddr)
{
    return HcclNetDevGetBusAddrV2(dstDevId, busAddr);
}

HcclResult HcclNetDevGetNicAddr(int32_t devicePhyId, HcclAddress** addr, uint32_t* addrNum)
{
    return HcclNetDevGetNicAddrV2(devicePhyId, addr, addrNum);
}
