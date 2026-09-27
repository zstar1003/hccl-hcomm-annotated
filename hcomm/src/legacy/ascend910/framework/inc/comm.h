/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_H
#define COMM_H

#include "hccl_common.h"
#include "common.h"
#include "hccl_socket.h"

static constexpr uint32_t HCCL_ALG_MESH = 0b1U;
static constexpr uint32_t HCCL_ALG_SWITCH = (HCCL_ALG_MESH << 1U);
static constexpr uint32_t HCCL_ALG_RING = (HCCL_ALG_MESH << 2U);
static constexpr uint32_t HCCL_ALG_PAIRWISE = (HCCL_ALG_MESH << 3U);
enum class HcclTopoLevel {
    HCCL_TOPO_L0 = 0,
    HCCL_TOPO_L1,
    HCCL_TOPO_MAX,
};

namespace hccl {
enum class HcclCommState {
    IDLE = 0, // 初始化完成，未使用
    BUILDING, // 正在使用，且在建链
    INUSE,    // 正在使用，建链完成或未建链
    RESERVED
};

inline const char* HcclCommStateToString(HcclCommState state)
{
    switch (state) {
        case HcclCommState::IDLE:
            return "IDLE";
        case HcclCommState::BUILDING:
            return "BUILDING";
        case HcclCommState::INUSE:
            return "INUSE";
        case HcclCommState::RESERVED:
            return "RESERVED";
        default:
            return "UNKNOWN";
    }
}

using HcclCommConnections = struct HcclCommConnectionsDef {
    bool isRoot{false};
    std::shared_ptr<HcclSocket> agentConnection{nullptr};
    std::map<u32, std::shared_ptr<HcclSocket>> serverConnections;
};

using HcclSocketPortConfig = struct HcclSocketPortConfigDef {
    // devPortSwitchOn 用于判断是否开启了用户配置的端口（通过环境变量配置的端口范围或者通过ranktable指定的端口）。
    // devPortSwitchOn开启时，将启用独立的vnic端口；即nic和vnic使用的端口可能不一致。
    bool devPortSwitchOn{false};
    std::pair<std::shared_ptr<HcclSocket>, HcclNetDevCtx> devNicListen{nullptr, nullptr};  // 抢占的device nic socket
    std::pair<std::shared_ptr<HcclSocket>, HcclNetDevCtx> devVnicListen{nullptr, nullptr}; // 抢占的device vnic socket
    std::pair<std::shared_ptr<HcclSocket>, HcclNetDevCtx> backupDevNicListen{
        nullptr, nullptr}; // 抢占的backup nic socket
};

using HcclCommParams = struct TagHCCLCollectiveParams {
    /**
    通信域的基本构建信息，通信域标识、节点数及本节点的编号
    通信域通过如下条件构建:
    1.用户在某个计算实体(rank)内调用hcclGetUniqueId作为本通信域的id
    2.将此id发往通信域的其它计算实体(rank)
    3.用户指定本comm对应的device, 本comm实例对应的device将会是用户set的device
    4.每个计算实体根据id, rank和total_ranks创建通信域
    */
    HcclRootInfo id; /* * 用于标识不同的通信域 */
    u32 rank;        /* * 用于标识通信域内不同节点 */
    u32 userRank;
    u32 totalRanks; /* * 用于指示通信域内的节点总数, rank范围[0, totalRanks-1] */
    s32 logicDevId;
    std::string serverId;
    DevType deviceType; // 芯片类型信息
    HcomProfilingMode profilingMode;
    std::string profilingOption;
    bool profilingInitiated;
    HcclComm commHandle;
    bool isHeterogComm;
    bool hcomGroupNicInit; // 在子group中对应world group NIC初始化标识
    CommAttr attr;
    WorkMode commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    std::string identifier;
    std::string cclBuffName;
    u32 ranktableCrc;
    HcclCommConnections commConnections;
    HcclSocketPortConfig commPortConfig;
    TagHCCLCollectiveParams()
        : id{0},
          rank(INVALID_VALUE_RANKID),
          userRank(INVALID_VALUE_RANKID),
          totalRanks(0xFFFFFFFF),
          logicDevId(-1),
          deviceType(DevType::DEV_TYPE_COUNT),
          profilingMode(HcomProfilingMode::PROFILING_CLOSE),
          profilingInitiated(false),
          commHandle(nullptr),
          isHeterogComm(false),
          hcomGroupNicInit(false),
          identifier(""),
          cclBuffName(""),
          ranktableCrc(0)
    {}
};

using WorldGroupInfo = struct worldGroupInfo {
    bool inlineReduceSwitchOn;
    DevType deviceType;
    s32 deviceLogicId;
    bool profilingInitiated;
    std::string serverId;
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> phyIdNicInfoMap;
    std::vector<RankInfo> worldRankInfoList;
    std::vector<u32> ranksPort;
    std::vector<u32> vnicRanksPort;
    bool devPortSwitchOn{false};
    bool useSuperPodMode;
    worldGroupInfo()
        : inlineReduceSwitchOn(true),
          deviceType(DevType::DEV_TYPE_COUNT),
          deviceLogicId(-1),
          profilingInitiated(false),
          useSuperPodMode(false)
    {}
};
} // namespace hccl
#endif // COMM_H
