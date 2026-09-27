/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_NET_DEV_H
#define HCCL_NET_DEV_H

#include <stdint.h>
#include <arpa/inet.h>
#include <hccl/hccl_types.h>
#include "hccl_net_dev_defs.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief 打开网络设备
 * @param[in] info 设备初始化配置信息
 * @param[out] netDev 返回的网络设备句柄
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclNetDevOpen(const HcclNetDevInfos* info, HcclNetDev* netDev);

/**
 * @brief 关闭网络设备
 * @param[in] netDev 要关闭的设备句柄
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclNetDevClose(HcclNetDev netDev);

/**
 * @brief 获取网络设备地址信息
 * @param[in] netDev 网络设备句柄
 * @param[out] addr 返回的地址信息结构
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclNetDevGetAddr(HcclNetDev netDev, HcclAddress* addr);

/**
 * @brief 通过设备物理ID获取总线地址
 * @param[in] dstDevId 设备物理ID或者超节点物理ID
 * @param[out] busAddr 返回的总线地址信息
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclNetDevGetBusAddr(HcclDeviceId dstDevId, HcclAddress* busAddr);
/**
 * @brief 通过设备物理ID获取nic地址
 * @param[in] devicePhyId 设备物理ID
 * @param[out] addr 返回的地址信息数组
 * @param[out] addrNum 返回的地址数量
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclNetDevGetNicAddr(int32_t devicePhyId, HcclAddress** addr, uint32_t* addrNum);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
