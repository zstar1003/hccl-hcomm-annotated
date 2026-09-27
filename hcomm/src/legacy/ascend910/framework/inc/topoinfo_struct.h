/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_STRUCT_H
#define TOPOINFO_STRUCT_H

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include "hccl/base.h"
#include "hccl_common.h"
#include "adapter_pub.h"
#include "externalinput_pub.h"
#include "hccl_network_pub.h"

namespace hccl {
using NetworkInfo_t = struct tagNetworkInfo {
    std::string ethName;       // [DEPRECATED]网卡名, 用于获取PCIe总线信息, 网卡在device侧时无效
    HcclIpAddress ipAddr;      // [DEPRECATED]空字符串则表示没有配ip, 不得参与节点间通信
    u32 networkPort{0};        // [DEPRECATED]port接口
    HcclIpAddress refIp;       // [DEPRECATED]reference ip
    u32 planeID{INVALID_UINT}; // [DEPRECATED]该网卡对应的网络平面ID, 用于控制网卡之间的建链
};

// [DEPRECATED] type ServerInfo_t
using ServerInfo_t = struct tagServerInfo {
    std::string serverId;                   // 集群内服务器唯一标识
    std::vector<NetworkInfo_t> networkInfo; // [DEPRECATED]网卡信息
    bool operator==(const tagServerInfo& that) { return this->serverId == that.serverId; }
};

using DeviceInfo_t = struct tagDeviceInfo {
    s32 devicePhyId;                             // 服务器内device唯一标识
    DevType deviceType{DevType::DEV_TYPE_NOSOC}; // 服务器内device类型
    std::vector<HcclIpAddress> deviceIp;         // device 对应的网卡ip
    std::vector<HcclIpAddress> backupDeviceIp;   // 同一卡另一个device的网卡ip，应用于重执行借轨场景
    HcclIpAddress refIp;                         // [DEPRECATED]device 对应的ref ip
    u32 port{HCCL_INVALID_PORT};
    u32 vnicPort{HCCL_INVALID_PORT};
    u32 backupPort{HCCL_INVALID_PORT};
    NICDeployment nicDeploy{NICDeployment::NIC_DEPLOYMENT_DEVICE};
    u32 proto;
};

using TransportInfo_t = struct tagTransportInfo {
    u32 dstRankId;
    TransportType transportType;
};

using RankInfo_t = struct tagRankInfo {
    u32 rankId = 0xFFFFFFFF;                    // rank 标识，userRank,cloud时hcom计算填入
    u32 localRank = 0xFFFFFFFF;                 // 本server内rank号
    std::string serverId;                       // 集群内服务器唯一标识
    u32 serverIdx = INVALID_UINT;               // Server在ranktable中的自然顺序（用户指定）
    u32 superDeviceId = INVALID_UINT;           // 超节点device id，超节点内唯一
    std::string superPodId;                     // 超节点标识
    u32 superPodIdx = INVALID_UINT;             // SuperPod在ranktable中的自然顺序（用户指定）
    HcclIpAddress hostIp;                       // 本server的host ip，用于host rdma通信
    u32 hostPort = HCCL_INVALID_PORT;           // 本rank进行host socket通信使用的端口
    u32 nodeId = INVALID_UINT;                  // 离线编译逻辑ranktable 和NumaConfig中的node id相同
    s32 itemId = INVALID_UINT;                  // 离线编译逻辑ranktable 和NumaConfig中的item id相同
    std::string groupName;                      // [DEPRECATED]group名称
    std::string podName;                        // [DEPRECATED]容器名称
    DeviceInfo_t deviceInfo;                    // 设备信息
    std::vector<TransportInfo_t> transportInfo; // [DEPRECATED]本rank与其余rank的数据传输信息（抽象信息）
    s32 bindDeviceId = INVALID_INT;             // 绑定的device id
    TlsStatus tlsStatus = TlsStatus::UNKNOWN;   // TLS开关状态
    std::string originalSuperPodId;             // 划分逻辑超节点前的原超节点ID，来源为用户配置
};

// 对外的ranktable格式
using RankTable_t = struct tagRankTable {
    u32 deviceNum{0};                                              // 当前通信域内device数量
    u32 serverNum{0};                                              // 集群内服务器总数
    u32 superPodNum{0};                                            // 集群超节点总数
    u32 groupNum{0};                                               // [DEPRECATED]集群内group总数
    NICDeployment nicDeploy{NICDeployment::NIC_DEPLOYMENT_DEVICE}; // 网卡挂载位置 0:host 1:device
    u32 nicNum{0};                                                 // [DEPRECATED]参数平面网卡数量
    std::vector<std::string> nicNames;                             // [DEPRECATED]参数平面网卡名称
    u32 rankNum{0};                                                // 通信域内的rank总数
    std::vector<RankInfo_t> rankList;                              // 通信域内所有的rank信息
    std::vector<ServerInfo_t> serverList; // [DEPRECATED]通信域内服务器信息 目前cloud和lab网卡在device场景置空
    std::string collectiveId;             // 通信域ID
    std::string version;                  // rankTable版本信息
    std::string mode;                     // [DEPRECATED]通讯方式tcp/rdma
};

using RoleTableNodeInfo = struct RoleTableNodeInfoTag {
    u32 id;
    std::string serverId;
    HcclIpAddress ipAddr;
    u32 port;
    u32 rankId;
    HcclIpAddress hostIp;
    u32 hostPort{HCCL_INVALID_PORT};
    s32 devicePhyId;
    RoleTableNodeInfoTag()
        : id(INVALID_UINT),
          port(INVALID_UINT),
          rankId(INVALID_UINT),
          hostPort(INVALID_UINT),
          devicePhyId(INVALID_INT)
    {}
};

using RoleTableInfo = struct RoleTableInfoTag {
    std::vector<RoleTableNodeInfo> servers;
    std::vector<RoleTableNodeInfo> clients;
};

constexpr uint32_t ROOTINFO_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t ROOTINFO_PORT_MAX_LENGTH = 16;
using HcclRootHandle = struct HcclRootHandleDef {
    char ip[IP_ADDRESS_BUFFER_LEN];
    uint32_t port;
    NICDeployment nicDeploy;
    char identifier[ROOTINFO_INDENTIFIER_MAX_LENGTH];
    u32 rankId;
};

using HcclRankHandle = HcclRootHandle;

// scalable 多root建链时，单个 root 的完整信息：agent服务地址 + root间mesh全互联监听端口。
using HcclScalableRootHandle = struct HcclScalableRootHandleDef {
    HcclRootHandle rootHandle;            // 本root的agent服务地址（ip/port/identifier/rankId）
    uint32_t meshPort{HCCL_INVALID_PORT}; // root间mesh全互联监听端口
};

// root间mesh全互联时，对端root的连接信息（ip + mesh端口），由HcclScalableRootHandle提炼得到。
using RootMeshInfo = struct RootMeshInfoDef {
    char ip[IP_ADDRESS_BUFFER_LEN]{0};
    uint32_t meshPort{HCCL_INVALID_PORT};
};

using GroupLeader_t = struct tagGroupLeaderInfo {
    u32 grpLeaderNum{0};
    std::vector<HcclRankHandle> GroupLeaderList;
};

const std::string PROP_STEP = "step";
const std::string PROP_RANK_NUM = "rank_num";
const std::string PROP_DEV_NUM = "device_num";
const std::string PROP_SRV_NUM = "server_num";
const std::string PROP_DEV_INFO = "dev_info";
const std::string PROP_RANK_LIST = "rank_list";
const std::string PROP_RANK_ID = "rank_id";
const std::string PROP_DEST_RANK = "dest_rank";
const std::string PROP_SERVER_ID = "server_id";
const std::string PROP_SERVER_LIST = "server_list";
const std::string PROP_DEV_ID = "device_id";
const std::string PROP_DEV_TYPE = "device_type";
const std::string PROP_DEV_IP = "device_ip";
const std::string PROP_DEV_NIC_PORT = "device_port";
const std::string PROP_DEV_VNIC_PORT = "device_vnic_port";
const std::string PROP_BACKUP_DEV_PORT = "backup_device_port";
const std::string PROP_BACKUP_DEV_IP = "backup_device_ip";
const std::string PROP_HOST_IP = "host_ip";
const std::string PROP_DEPLOY_MODE = "deploy_mode";
const std::string PROP_TRANS_INFO = "trans_info";
const std::string PROP_TRANS_TYPE = "trans_type";
const std::string PROP_TLS_STATUS = "tls_status";

const std::string PROP_SERVER_COUNT = "server_count";
const std::string PROP_DEVICE = "device";
const std::string PROP_STATUS = "status";
const std::string PROP_VERSION = "version";

const std::string PROP_SUPER_POD_LIST = "super_pod_list";
const std::string PROP_SUPER_POD_ID = "super_pod_id";
const std::string PROP_SUPER_DEVICE_ID = "super_device_id";
const std::string PROP_SUPER_POD_NUM = "super_pod_num";

const std::string PROP_NETWORK_INFO_LIST = "network_info_list";
const std::string PROP_NETWORK_ETHNAME = "eth_name";
const std::string PROP_NETWORK_IPADDR = "ip_addr";
const std::string PROP_NETWORK_NETWORKPORT = "network_port";
const std::string PROP_NETWORK_REFIP = "ref_ip";
const std::string PROP_NETWORK_PLANEID = "plane_id";

const std::string PROP_NETWORK_IDENTIFIER = "identifier";
const std::string PROP_GROUP_LEADER_LIST = "group_leader_list";
const std::string PROP_GROUP_CLUSTER_LIST = "group_cluster_list";

const std::string TOPO_DETECT_TAG = "topo_detect_default_tag";
} // namespace hccl
#endif /* __TOPINFO_STRUCT_H__ */
