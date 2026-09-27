/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_NET_DEV_V2_H
#define HCCL_NET_DEV_V2_H

#include <stdint.h>
#include <arpa/inet.h>
#include <hccl/hccl_types.h>
#include "hccl_net_dev_defs.h"
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

HcclResult HcclNetDevOpenV2(const HcclNetDevInfos* info, HcclNetDev* netDev);
HcclResult HcclNetDevCloseV2(HcclNetDev netDev);
HcclResult HcclNetDevGetAddrV2(const HcclNetDev netDev, HcclAddress* addr);
HcclResult HcclNetDevGetBusAddrV2(HcclDeviceId dstDevId, HcclAddress* busAddr);
HcclResult HcclNetDevGetNicAddrV2(int32_t devicePhyId, HcclAddress** addr, uint32_t* addrNum);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_NET_DEV_V2_H
