/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_NET_DEV_V1_H
#define HCCL_NET_DEV_V1_H

#include "hccl_types.h"
#include "hccl_net_dev_defs.h"
#include <stdint.h>

#ifndef OPEN_BUILD_PROJECT
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult __attribute__((weak)) HcclNetDevOpenV1(const HcclNetDevInfos* info, HcclNetDev* netDev);
HcclResult __attribute__((weak)) HcclNetDevCloseV1(HcclNetDev netDev);
HcclResult __attribute__((weak)) HcclNetDevGetAddrV1(HcclNetDev netDev, HcclAddress* addr);
HcclResult __attribute__((weak)) HcclNetDevGetBusAddrV1(HcclDeviceId dstDevId, HcclAddress* busAddr);
HcclResult __attribute__((weak)) HcclNetDevGetNicAddrV1(int32_t devicePhyId, HcclAddress** addr, uint32_t* addrNum);

#ifdef __cplusplus
}
#endif
#endif // __cplusplus

#endif
