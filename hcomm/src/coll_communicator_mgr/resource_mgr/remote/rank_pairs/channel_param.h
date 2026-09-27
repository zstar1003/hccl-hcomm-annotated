/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHANNEL_PARAM_H
#define CHANNEL_PARAM_H

#include "hccl_mem_defs.h"
#include "hccl_common.h"
#include "transport_pub.h"
#include "hccl/hccl_res.h"
#include "aicpu_operator_pub.h"

constexpr u64 P2P_NOTIFY_MAX_NUM = 66;
// 独立算子同步资源
struct HcclChannelP2p {
    HcclMem* remoteUserMem = nullptr;                  // 远端用户内存
    HcclMem remoteHcclbuffer;                          // 远端用户cclbuffer
    u32 remoteUserMemCount = 0;                        // 远端用户内存数量
    HcclSignalInfo localIpcSignal[P2P_NOTIFY_MAX_NUM]; // localnotify
    HcclSignalInfo remoteIpcSignal[P2P_NOTIFY_MAX_NUM];
    hccl::TransportAttr transportAttr;
    u32 qos;
};

struct HcclChannelRoce {
    // 将固定数组改为指针
    MemDetails* remoteUserHostMem = nullptr;   // 远端用户主机内存
    MemDetails* remoteUserDeviceMem = nullptr; // 远端用户设备内存
    MemDetails localHcclbuffer;                // 本端用户cclbuffer
    MemDetails remoteHcclbuffer;               // 远端用户cclbuffer
    u32 remoteUserHostMemCount = 0;            // 远端用户主机内存数量
    u32 remoteUserDeviceMemCount = 0;          // 远端用户设备内存数量
    u64 notifyValue{0};
    u32 notifyValueKey{0};
    u32 singleQPNotifyNum{0};
    u64 localNotifyList{0};
    u64 remoteNotifyList{0};
    u32 remoteNotifyKey{0};
    s64 chipId{LLONG_MAX};
    HcclQpInfoV2 QpInfo[RDMA_QP_MAX_NUM];
    u32 qpsPerConnection{1};
};

struct HcclIndOpChannelRemoteResV2 {
    u64 p2pNotifyNum = 0;        // 用于linkp2p添加notify信息
    u64 roceNotifyNum = 0;       // 主链路：用于linkroce添加notify信息
    u64 qpNum = 0;               // 主链路：QP计数，支持多个，用于linkroce添加QP信息
    u32 remoteRank = 0;          // 远端rankId
    u32 remoteWorldRank = 0;     // 远端rankWorldId
    bool isUsedRdma = false;     // 是否使用RDMA，对应Roce和P2p
    HcclChannelP2p channelP2p;   // P2p资源
    HcclChannelRoce channelRoce; // Roce资源
};

struct HcclIndOpChannelRemoteResV3 {
    char hcomId[HCOMID_MAX_LENGTH];                     // 通信域ID 最大长度待修改
    char channelTag[TAG_MAX_LENGTH];                    // channelTag 最大长度待修改
    CommEngine engine;                                  // 通信引擎类型
    u32 localUserRank;                                  // 本地rankId
    u32 multiQpThreshold;                               // 多QP每个QP分担数据量最小阈值
    void* channelList;                                  // device侧channelList地址
    u32 listNum = 0;                                    // 建链channel的总数量
    HcclIndOpChannelRemoteResV2* remoteResV2 = nullptr; // 不同remoteRank建链的资源
};

struct DevAicpuChannelConfig {
    // 如要新增配置类字段，在此处添加
};

struct HcclChannelUrmaRes {
    char hcomId[HCOMID_MAX_LENGTH];      // 通信域ID 最大长度待修改
    void* channelList;                   // 反序列后返回给host侧的device侧handle地址
    u32 listNum = 0;                     // 建链channel的总数量
    void* uniqueIdAddr;                  // 序列化后device侧地址
    u32 uniqueIdSize{0};                 // 序列化后总地址长度
    void* channelSizeAddr;               // 存放序列化后device channel size的指针
    u32* remoteRankList;                 // 序列化后返回给host侧的device侧rankList地址
    u32* remoteRankId;                   // 记录每个channel的对端rank
    s32 deviceLogicId{0};                // 基础通信使用
    u32 deviceType{0};                   // 基础通信使用
    DevAicpuChannelConfig channelConfig; // 收编channel配置类变量
    void* ctxList{nullptr};              // device侧ctx指针数组，非null时kernel写ctx->deviceChannel
};

struct HcommRoceChannelRes {
    void* localMem = nullptr;  // device 上 RoceMemDetails[localMemCount]
    void* remoteMem = nullptr; // device 上 RoceMemDetails[remoteMemCount]
    u32 localMemCount = 0;
    u32 remoteMemCount = 0;
    s64 chipId{LLONG_MAX};
    HcclQpInfoV2 QpInfo[RDMA_QP_MAX_NUM];
    u32 qpsPerConnection{1};
    void* remoteNotifyAddr = nullptr;    // 对端 NOTIFY_SRC_MEM 地址 (Fence Read 读取源)
    u32 remoteNotifyKey = 0;             // 对端 notify rkey
    void* localDataNotifyAddr = nullptr; // 本端datanotify地址
    u32 localDataNotifyKey = 0;          // 本端dataNotify key值
    u32 notifySize = sizeof(u32);        // notify 大小
    HcclSignalInfo localDataSignal{};    // 本端dataNotify signal
};

// for A2/A3 endpoint with transport
struct HcclChannelHccsRes {
    char channelTag[TAG_MAX_LENGTH]; // channelTag 最大长度待修改
    HcclChannelP2p channelP2p;       // P2p资源
    u64 p2pNotifyNum{0};             // 用于linkp2p添加notify信息
    u32 deviceType{0};               // 基础通信使用
    u32 remoteRank{0};               // 远端rankId
    u32 localRank{0};
    u32 remoteDevicePhyId{0}; // 远端 PhyId
    u32 localDevicePhyId{0};
    s32 localDeviceLogicId{0};
    hccl::MachineType machineType{hccl::MachineType::MACHINE_RESERVED_TYPE}; // client或者server
    u32 localBufSize{0};
    u32 remoteBufSize{0};
    HcclMemEx* localBufMem{nullptr};
    HcclMemEx* remoteBufMem{nullptr};
};

struct HcommDeviceInfo {
    s32 deviceLogicId{0};
    u32 devicePhyId{0};
    u32 deviceType{0};
};

struct HcommChannelRes {
    void* channelList;             // 反序列后返回给host侧的device侧handle地址
    u32 listNum = 0;               // 建链channel的总数量
    void* channelDataListAddr;     // device 上 listNum 个指针，每项指向该 channel 的序列化 device 内存
    void* channelDataSizeListAddr; // device 上 listNum 个 u32，每项为对应 channel 序列化字节数
    void* channelTypeListAddr; // device 上 listNum 个 u32，每项为 hcomm::HcommChannelKind 数值（见 channel.h）
    HcommDeviceInfo deviceInfo;
    void* ctxList{nullptr}; // device侧ctx指针数组，非null时kernel写ctx->deviceChannel
};

#endif
