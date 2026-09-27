/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_NET_DEV_DEFS_H
#define HCCL_NET_DEV_DEFS_H

#include <stdint.h>
#include <arpa/inet.h>

/* 网络设备句柄 */
typedef void* HcclNetDev;
constexpr uint32_t SUPER_DEVICE_ID_INVALID = 0xFFFFFFFF;
/**
 * @enum HcclNetDevDeployment
 * @brief 网络设备部署位置枚举
 */
typedef enum {
    HCCL_NETDEV_DEPLOYMENT_RESERVED = -1, ///< 保留部署模式
    HCCL_NETDEV_DEPLOYMENT_HOST = 0,      ///< 网络设备部署在主机侧
    HCCL_NETDEV_DEPLOYMENT_DEVICE = 1     ///< 网络设备部署在设备侧
} HcclNetDevDeployment;

/**
 * @enum HcclProtoType
 * @brief 网络传输协议类型枚举
 */
typedef enum {
    HCCL_PROTO_TYPE_RESERVED = -1,                    ///< 保留协议类型
    HCCL_PROTO_TYPE_BUS = 0,                          ///< 设备间总线直连协议
    HCCL_PROTO_TYPE_TCP = 1,                          ///< 标准TCP协议
    HCCL_PROTO_TYPE_ROCE = 2,                         ///< RDMA over Converged Ethernet
    HCCL_PROTO_TYPE_UB_CTP = 3,                       ///< 华为统一总线UB_CTP
    HCCL_PROTO_TYPE_UBC_CTP = HCCL_PROTO_TYPE_UB_CTP, ///< 兼容旧协议名UBC_CTP
    HCCL_PROTO_TYPE_UBC_TP = 4,                       ///< 华为统一总线UBC_TP
    HCCL_PROTO_TYPE_UB_RTP = 5,                       ///< 华为统一总线UB_RTP
    HCCL_PROTO_TYPE_UBG_TP = HCCL_PROTO_TYPE_UB_RTP   ///< 兼容旧协议名UBG_TP
} HcclProtoType;

/**
 * @enum HcclAddressType
 * @brief 设备地址类型枚举
 */
typedef enum {
    HCCL_ADDR_TYPE_RESERVED = -1, ///< 保留地址类型
    HCCL_ADDR_TYPE_IP_V4 = 0,     ///< IPv4地址类型
    HCCL_ADDR_TYPE_IP_V6 = 1,     ///< IPv6地址类型
} HcclAddressType;

/**
 * @struct HcclAddress
 * @brief 网络设备地址描述结构体
 * @var protoType - 使用的传输协议类型
 * @var type     - 地址类型（IPv4/IPv6）
 * @var addr     - IPv4地址存储（当type=IP_V4时有效）
 * @var addr6    - IPv6地址存储（当type=IP_V6时有效）
 */
typedef struct {
    HcclProtoType protoType{HCCL_PROTO_TYPE_RESERVED};
    HcclAddressType type{HCCL_ADDR_TYPE_RESERVED};
    union {
        struct in_addr addr;   ///< IPv4地址结构
        struct in6_addr addr6; ///< IPv6地址结构
    };
} HcclAddress;

/**
 * @struct HcclNetDevInfos
 * @brief 网络设备信息结构体
 * @var nicDeployment - 网络设备部署位置
 * @var devicePhyId   - 物理设备ID（需与硬件拓扑匹配）
 * @var addr          - 网络地址配置信息
 */
typedef struct {
    HcclNetDevDeployment netdevDeployment{HCCL_NETDEV_DEPLOYMENT_RESERVED};
    int32_t devicePhyId{-1};
    HcclAddress addr;
    bool isBackup{false}; // 仅支持NETDEV_DEPLOYMENT_DEVICE配置，默认false
} HcclNetDevInfos;

typedef struct {
    uint32_t superDeviceId{SUPER_DEVICE_ID_INVALID};
    int32_t devicePhyId{-1};
} HcclDeviceId;
#endif
