/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_ascend_cluster_parser.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <runnerdb/db_sim_runner_common.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <system_error>

namespace fs = std::filesystem;

#include "topo_cluster_dumper.h"
#include "sim_common_macro.h"
#include "sim_log.h"
#include "sim_common_api.h"
#include "sim_yaml_config.h"
#include "sim_ip_address.h"
#include "db_sim_runner_ops.h"

namespace {
// 辅助函数：将单个十六进制字符转为对应数值
static uint8_t hex_char_to_val(char c)
{
    if (c >= '0' && c <= '9') {
        return static_cast<int>(c) - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return 10 + (static_cast<int>(c) - 'a');
    }
    if (c >= 'A' && c <= 'F') {
        return 10 + (static_cast<int>(c) - 'A');
    }
    return 0; // 非法字符默认返回0
}

// 核心函数：34位十六进制EID字符串转16字节uint8_t数组
// 参数：eid_str - 输入字符串(如"0x00000000000000000000000000000018")
//       eid_out - 输出数组(长度必须为16)
void eid_hex_to_uint8(const char* eid_str, uint8_t* eid_out)
{
    // 1. 跳过0x前缀，指向有效十六进制数据
    const char* hex_data = eid_str + 2;

    // 2. 清空输出数组（初始化全0）
    memset(eid_out, 0, 16);

    // 3. 每2个字符转换为1个uint8_t字节，共转换16字节
    for (int i = 0; i < 16; i++) {
        // 高4位 + 低4位 组合成1个字节
        uint8_t high = hex_char_to_val(hex_data[2 * i]);
        uint8_t low = hex_char_to_val(hex_data[2 * i + 1]);
        eid_out[i] = (high << 4) | low;
    }
}

// EID 字符串转 IPv4 字符串
std::string eidToIP(const std::string& eid)
{
    // 1. 取最后 8 个十六进制字符（IPv4 所在位置）
    std::string ipv4_hex = eid.substr(eid.size() - 8, 8);

    // 2. 每 2 个字符转成一个 IP 段
    uint8_t o1 = (uint8_t)strtoul(ipv4_hex.substr(0, 2).c_str(), nullptr, 16);
    uint8_t o2 = (uint8_t)strtoul(ipv4_hex.substr(2, 2).c_str(), nullptr, 16);
    uint8_t o3 = (uint8_t)strtoul(ipv4_hex.substr(4, 2).c_str(), nullptr, 16);
    uint8_t o4 = (uint8_t)strtoul(ipv4_hex.substr(6, 2).c_str(), nullptr, 16);

    // 3. 拼接成 IP 字符串
    return std::to_string(o1) + "." + std::to_string(o2) + "." + std::to_string(o3) + "." + std::to_string(o4);
}
} // namespace
/*
 * @brief 初始化集群拓扑数据
 *
 * @param clusterDir 集群拓扑文件目录
 * @return HcclVmResult 状态码
 *
 * @note 该函数会解析clusterDir目录下的所有文件，生成IR数据，初始化至DB层。
 * 初始化静态DB表数据：Server/Host/Device/Ccu/Port/EndPoint/EndPointPortMapping
 */
HcclVmResult AscendClusterTopoParser::InitClusterTopo(const std::string& clusterDir)
{
    HCCL_VM_DEBUG("Enter InitClusterTopo: {}", clusterDir);
    // 1. 解析集群拓扑文件
    ClusterTopoParser parser;
    ParseStatus status = parser.ParseClusterDir(clusterDir);

    if (status != ParseStatus::OK) {
        HCCL_VM_ERROR("Failed to parse cluster directory (status={:d})", static_cast<int>(status));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 2. 生成IR数据
    network_ = parser.GetNetwork();

    HCCL_VM_DEBUG("Parse completed successfully!");
    HCCL_VM_DEBUG("Network version: {}", network_.version);
    HCCL_VM_DEBUG("SuperPod count: {}", network_.GetSuperPodCount());
    HCCL_VM_DEBUG("Total device count: {}", network_.GetTotalDeviceCount());
    HCCL_VM_DEBUG("Total link count: {}", network_.GetTotalLinkCount());

    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    ClusterTopoDumper::DumpToFile(network_, InstallPath::ResolveToInstallRoot("data/" + outputFile_));

    // 3. 初始化IR数据，保存至DB层
    if (InitClusterStaticTopoData() != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Failed to init cluster static topo data");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

/*
 * @brief 初始化集群通信域表项
 *
 * @param topoMeta 集群拓扑元数据
 * @return HcclVmResult 状态码
 *
 * @note 该函数会解析topoMeta，生成IR数据，初始化至DB层。
 * 初始化本次算子的动态DB数据：Rank/Link等
 * 更新静态表字段：Device.logic_id
 */
HcclVmResult AscendClusterTopoParser::ParseRanktableAndInitCommDomain(const std::string& ranktablePath)
{
    HCCL_VM_DEBUG("Enter ParseRanktableAndInitCommDomain, ranktablePath: {}", ranktablePath);

    TopoMeta topoMeta;
    if (ParseRanktable(ranktablePath, topoMeta) != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("parse rank table failed");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    return InitCommunicationDomain(topoMeta, true);
}

/*
 * @brief 初始化集群通信域表项
 *
 * @param topoMeta 集群拓扑元数据
 * @return HcclVmResult 状态码
 *
 * @note 该函数会解析topoMeta，生成IR数据，初始化至DB层。
 * 初始化本次算子的动态DB数据：Rank/Link等
 * 更新静态表字段：Device.logic_id
 */
HcclVmResult AscendClusterTopoParser::InitCommunicationDomain(const TopoMeta& topoMeta, bool withRanktable)
{
    HCCL_VM_DEBUG("Enter InitCommunicationDomain");

    // 第一步：初始化所有 server 上的 Device（logic_id = physical_id，保持不变）
    for (uint32_t superPodIdx = 0; superPodIdx < topoMeta.size(); ++superPodIdx) {
        auto superPod = topoMeta[superPodIdx];
        for (uint32_t serverIdx = 0; serverIdx < superPod.size(); ++serverIdx) {
            auto server = superPod[serverIdx];
            auto serverKey = sim::GetServerKeyById(superPodIdx, serverIdx);
            for (uint32_t deviceIdx = 0; deviceIdx < server.size(); ++deviceIdx) {
                // logic_id = physical_id = deviceIdx，按顺序递增
                // todo: 暂时不考虑容器场景，userId == logicDevId
                auto userId = deviceIdx;
                if (sim::UpdateDeviceLogicId(serverKey, deviceIdx, deviceIdx, userId) != ACL_SUCCESS) {
                    HCCL_VM_ERROR("update device logic id by serverKey {:d} failed", serverKey);
                    return HcclVmResult::HCCL_SIM_E_INTERNAL;
                }
            }
        }
    }

    // 第二步：按 rankId % deviceCount 的逻辑创建 Rank 表
    if (InitDynamicModelData(topoMeta) != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("init dynamic model data failed");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 通信域未初始化，则需创建ranktable.json文件
    if (status_ == HvmClusterStatus::COMM_DOMAIN_UNINIT && !withRanktable) {
        // 根据topoMeta生成ranktable.json文件
        if (CreateRankTableFile(topoMeta) != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("create rank table file failed");
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
    }

    status_ = HvmClusterStatus::COMM_DOMAIN_INIT_DONE;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AscendClusterTopoParser::InitDynamicModelData(const TopoMeta& topoMeta)
{
    // 应用通过 aclrtSetDevice(rank % deviceCount) 选择 device，Rank 表需与之匹配
    uint32_t rankId = 0;
    for (uint32_t superPodIdx = 0; superPodIdx < topoMeta.size(); ++superPodIdx) {
        auto superPod = topoMeta[superPodIdx];
        for (uint32_t serverIdx = 0; serverIdx < superPod.size(); ++serverIdx) {
            auto server = superPod[serverIdx];
            auto serverKey = sim::GetServerKeyById(superPodIdx, serverIdx);
            uint32_t deviceCount = server.size();
            for (uint32_t deviceIdx = 0; deviceIdx < deviceCount; ++deviceIdx) {
                // rankId % deviceCount 是应用实际选择的 logicDevId
                uint32_t targetLogicDevId = rankId % deviceCount;
                // 查找该 server 上 logic_id == targetLogicDevId 的 Device
                auto devRet = RunnerDB::GetOneByPred<sim::Device>([serverKey, targetLogicDevId](const sim::Device& d) {
                    return d.server_id == serverKey && d.logic_id == targetLogicDevId;
                });
                if (!devRet.second) {
                    HCCL_VM_ERROR("device not found: serverKey={}, logicDevId={}", serverKey, targetLogicDevId);
                    return HcclVmResult::HCCL_SIM_E_INTERNAL;
                }
                auto deviceKey = devRet.first.id;

                // 创建 Rank 表记录
                sim::Rank rank;
                rank.rank_id = rankId;
                rank.device_id = deviceKey;
                rank.state = 1;
                RunnerDB::Add<sim::Rank>(rank);
                HCCL_VM_DEBUG(
                    "Init Rank: rankId={}, deviceKey={}, logicDevId={}, phyDevId={}", rankId, deviceKey,
                    devRet.first.logic_id, devRet.first.physical_id);

                // 初始化该 device 关联的 CCU 资源
                auto deviceAllCcu = RunnerDB::GetByPred<sim::Ccu>([deviceKey](const sim::Ccu& d) {
                    return d.device_id == deviceKey;
                });
                if (deviceAllCcu.empty()) {
                    HCCL_VM_ERROR("get device all ccu failed");
                    return HcclVmResult::HCCL_SIM_E_INTERNAL;
                }
                for (const auto& ccu : deviceAllCcu) {
                    InitCcuResource(ccu.id);
                }

                rankId++;
            }
        }
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AscendClusterTopoParser::BuildLevelList(
    const Server& server, int srcDevPhyId, const std::set<int>& commDomainLocalIds, uint32_t spIdx, uint32_t srvIdx,
    json& levelList)
{
    const Device* device = server.GetDevice(srcDevPhyId);
    if (!device) {
        HCCL_VM_ERROR("cannot find device by phyDevId {:d} in superpod{:d}/server{:d}", srcDevPhyId, spIdx, srvIdx);
        return HCCL_SIM_E_NOT_FOUND;
    }

    auto linksByLayer = server.GetLinksByLayer(0);
    std::map<int, std::vector<const Link*>> layerLinks;
    for (const auto* link : linksByLayer) {
        if (link->sideA.localId == srcDevPhyId || link->sideB.localId == srcDevPhyId) {
            layerLinks[link->netLayer].push_back(link);
        }
    }
    auto linksByLayer1 = server.GetLinksByLayer(1);
    for (const auto* link : linksByLayer1) {
        if (link->sideA.localId == srcDevPhyId || link->sideB.localId == srcDevPhyId) {
            layerLinks[link->netLayer].push_back(link);
        }
    }

    for (const auto& [netLayer, links] : layerLinks) {
        if (netLayer > 1) {
            continue;
        }

        // Separate P2P and P2N links at this netLayer
        std::vector<const Link*> p2pLinks, p2nLinks;
        for (const auto* link : links) {
            if (link->IsPeer2Peer()) {
                p2pLinks.push_back(link);
            } else if (link->IsPeer2Net()) {
                p2nLinks.push_back(link);
            }
        }

        // Process PEER2PEER links from topo
        if (!p2pLinks.empty()) {
            json levelEntry;
            levelEntry["net_addr"] = "";
            levelEntry["net_type"] = "TOPO_FILE_DESC";
            levelEntry["net_instance_id"] = "superPod" + std::to_string(spIdx) + "_rack" + std::to_string(srvIdx);
            levelEntry["net_layer"] = netLayer;

            json rankAddrList = json::array();
            for (const auto* link : p2pLinks) {
                const LinkPortRef* srcPort = nullptr;
                const LinkPortRef* dstPort = nullptr;
                if (link->sideA.localId == srcDevPhyId) {
                    srcPort = &link->sideA;
                    dstPort = &link->sideB;
                } else if (link->sideB.localId == srcDevPhyId) {
                    srcPort = &link->sideB;
                    dstPort = &link->sideA;
                } else {
                    continue;
                }

                int peerId = (link->sideA.localId == srcDevPhyId) ? link->sideB.localId : link->sideA.localId;
                if (commDomainLocalIds.find(peerId) == commDomainLocalIds.end()) {
                    continue;
                }

                json addrEntry;
                std::string eid = srcPort->eid;
                if (eid.size() > 2 && eid[0] == '0' && (eid[1] == 'x' || eid[1] == 'X')) {
                    eid = eid.substr(2);
                }
                addrEntry["addr"] = eid;
                addrEntry["addr_type"] = "EID";
                addrEntry["plane_id"] = "planeA";
                addrEntry["ports"] = srcPort->portIds;

                AddLinkInfo(srcPort, dstPort, netLayer, link->protocols);

                rankAddrList.push_back(addrEntry);
            }

            levelEntry["rank_addr_list"] = rankAddrList;
            levelList.push_back(levelEntry);
        }

        // Process PEER2NET from rootinfo portGroups
        // topo.json port_list may combine multiple port_group entries into one edge;
        // ranktable must use rootinfo's individual portGroup entries with correct EIDs.
        if (!p2nLinks.empty()) {
            std::map<std::string, std::vector<std::string>> eidToPorts;
            for (const auto& port : device->ports) {
                if (port.layer == netLayer && port.linkType == LinkType::PEER2NET && !port.eid.empty()) {
                    eidToPorts[port.eid].push_back(port.portId);
                }
            }

            if (!eidToPorts.empty()) {
                json levelEntry;
                levelEntry["net_addr"] = "";
                levelEntry["net_type"] = "CLOS";
                levelEntry["net_instance_id"] = "az0";
                levelEntry["net_layer"] = netLayer;

                json rankAddrList = json::array();
                for (const auto& [eid, ports] : eidToPorts) {
                    json addrEntry;
                    std::string cleanEid = eid;
                    if (cleanEid.size() > 2 && cleanEid[0] == '0' && (cleanEid[1] == 'x' || cleanEid[1] == 'X')) {
                        cleanEid = cleanEid.substr(2);
                    }
                    addrEntry["addr"] = cleanEid;
                    addrEntry["addr_type"] = "EID";
                    addrEntry["plane_id"] = "plane0";
                    addrEntry["ports"] = ports;

                    LinkPortRef srcRef;
                    srcRef.localId = srcDevPhyId;
                    srcRef.eid = eid;
                    srcRef.portIds = ports;
                    AddLinkInfo(&srcRef, nullptr, netLayer, p2nLinks[0]->protocols);

                    rankAddrList.push_back(addrEntry);
                }

                levelEntry["rank_addr_list"] = rankAddrList;
                levelList.push_back(levelEntry);
            }
        }
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

json AscendClusterTopoParser::BuildRankEntry(
    const Server& server, int srcDevPhyId, const std::set<int>& commDomainLocalIds, uint32_t spIdx, uint32_t srvIdx,
    uint32_t rankId)
{
    json rankEntry;
    rankEntry["device_id"] = srcDevPhyId;
    rankEntry["local_id"] = srcDevPhyId;
    rankEntry["rank_id"] = rankId;
    rankEntry["level_list"] = json::array();

    BuildLevelList(server, srcDevPhyId, commDomainLocalIds, spIdx, srvIdx, rankEntry["level_list"]);

    return rankEntry;
}

HcclVmResult AscendClusterTopoParser::CreateRankTableFile(const TopoMeta& topoMeta)
{
    json rankTable;
    rankTable["rank_count"] = ShmGetPhyDeviceTotalCount(topoMeta);
    rankTable["rank_list"] = json::array();
    rankTable["version"] = "2.0";

    uint32_t rankId = 0;

    for (uint32_t spIdx = 0; spIdx < topoMeta.size(); ++spIdx) {
        if (spIdx >= network_.superPods.size()) {
            HCCL_VM_ERROR("superpod index {:d} out of range", spIdx);
            return HCCL_SIM_E_NOT_FOUND;
        }
        const SuperPod& superPod = network_.superPods[spIdx];
        const auto& superPodMeta = topoMeta[spIdx];

        for (uint32_t srvIdx = 0; srvIdx < superPodMeta.size(); ++srvIdx) {
            if (srvIdx >= superPod.servers.size()) {
                HCCL_VM_ERROR("server index {:d} out of range in superpod {:d}", srvIdx, spIdx);
                return HCCL_SIM_E_NOT_FOUND;
            }
            const Server& server = superPod.servers[srvIdx];
            const auto& serverMeta = superPodMeta[srvIdx];

            std::set<int> commDomainLocalIds(serverMeta.begin(), serverMeta.end());

            for (uint32_t deviceIdx = 0; deviceIdx < serverMeta.size(); ++deviceIdx) {
                uint32_t logicDevId = rankId % serverMeta.size();
                PhyDeviceId phyDevId = serverMeta[logicDevId];
                rankTable["rank_list"].push_back(
                    BuildRankEntry(server, phyDevId, commDomainLocalIds, spIdx, srvIdx, rankId));
                rankId++;
            }
        }
    }

    fs::path dataDir = fs::path(InstallPath::ResolveToInstallRoot("data"));
    std::error_code ec2;
    fs::create_directories(dataDir, ec2);
    std::string outputPath = (dataDir / "ranktable.json").string();
    std::ofstream ofs(outputPath);
    if (!ofs.is_open()) {
        HCCL_VM_ERROR("failed to open ranktable.json for writing");
        return HcclVmResult::HCCL_SIM_E_OPEN_FILE_FAILURE;
    }

    ofs << rankTable.dump(4) << std::endl;
    ofs.close();

    setenv("RANK_TABLE_FILE", outputPath.c_str(), 1);

    HCCL_VM_DEBUG("ranktable.json generated with {:d} ranks", rankId);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AscendClusterTopoParser::InitClusterStaticTopoData()
{
    for (const auto& superPod : network_.superPods) {
        for (const auto& server : superPod.servers) {
            if (server.hostEid.empty()) {
                HCCL_VM_ERROR("server host eid is empty");
                return HcclVmResult::HCCL_SIM_E_INTERNAL;
            }
            auto serverKey = InitServer(superPod.superPodId, server);
            serverIdx2Key_[superPod.superPodId][server.serverId] = serverKey;
            for (const auto& device : server.devices) {
                InitDevice(serverKey, device);
            }
        }
    }
    HCCL_VM_DEBUG("InitClusterStaticTopoData completed successfully!");
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

uint64_t AscendClusterTopoParser::InitServer(uint32_t superPodId, const Server& server)
{
    HCCL_VM_DEBUG("Enter InitServer: {}, eid= {}", server.serverId, server.hostEid.c_str());
    sim::Server serverDb;
    serverDb.pod_id = superPodId;
    serverDb.server_id = server.serverId;
    auto serverKey = RunnerDB::Add<sim::Server>(serverDb);

    sim::Host host;
    host.server_id = serverKey;
    strcpy(host.ip_addr, eidToIP(server.hostEid).c_str());
    auto hostKey = RunnerDB::Add<sim::Host>(host);

    return serverKey;
}

uint64_t AscendClusterTopoParser::InitCcu(uint64_t deviceKey, uint32_t dieId)
{
    sim::Ccu ccu{};
    ccu.device_id = deviceKey;
    ccu.die_id = dieId;
    ccu.status = 0;
    return RunnerDB::Add<sim::Ccu>(ccu);
}

uint64_t AscendClusterTopoParser::InitCcuResource(uint64_t ccuKey)
{
    sim::CcuResource ccuRes{};
    ccuRes.ccu_id = ccuKey;
    return RunnerDB::Add<sim::CcuResource>(ccuRes);
}

uint64_t AscendClusterTopoParser::InitPort(uint64_t deviceKey, const Port& portIn)
{
    sim::Port portDb{};
    portDb.device_id = deviceKey;
    if (portIn.dieId >= 0) {
        portDb.die_id = static_cast<uint32_t>(portIn.dieId);
    }
    strcpy(portDb.name, portIn.portId.c_str());
    auto portKey = RunnerDB::Add<sim::Port>(portDb);

    sim::EndPoint endPoint{};
    endPoint.device_id = deviceKey;
    if (portIn.dieId >= 0) {
        endPoint.die_id = static_cast<uint32_t>(portIn.dieId);
    }
    if (portIn.layer == 0) {
        endPoint.func_id = 2;
    } else {
        endPoint.func_id = 3;
    }
    endPoint.type = 0;
    eid_hex_to_uint8(portIn.eid.c_str(), endPoint.eid);
    strcpy(endPoint.ip_addr, eidToIP(portIn.eid).c_str());
    auto endPointKey = RunnerDB::Add<sim::EndPoint>(endPoint);

    sim::EndPointPortMapping endPointPortMapping{};
    endPointPortMapping.port_id = portKey;
    endPointPortMapping.endpoint_id = endPointKey;
    auto endPointPortMappingKey = RunnerDB::Add<sim::EndPointPortMapping>(endPointPortMapping);
    return portKey;
}

uint64_t AscendClusterTopoParser::InitDevice(uint64_t serverKey, const Device& deviceIn)
{
    sim::Device deviceDb{};
    deviceDb.server_id = serverKey;
    deviceDb.physical_id = deviceIn.localId;
    deviceDb.overflow_mode = 0;
    if (!deviceIn.socVersion.empty()) {
        // todo: 从yaml文件中获取soc版本
        strncpy(deviceDb.soc_version, deviceIn.socVersion.c_str(), sizeof(deviceDb.soc_version) - 1);
        deviceDb.soc_version[sizeof(deviceDb.soc_version) - 1] = '\0';
    }
    deviceDb.status = 0;
    auto deviceKey = RunnerDB::Add<sim::Device>(deviceDb);

    // 初始化ccu资源
    InitCcu(deviceKey, 0);
    InitCcu(deviceKey, 1);

    for (const auto& port : deviceIn.ports) {
        InitPort(deviceKey, port);
    }
    return deviceKey;
}

HcclVmResult AscendClusterTopoParser::AddLinkInfo(
    const LinkPortRef* srcPort, const LinkPortRef* dstPort, uint32_t netLayer, const std::vector<Protocol>& protocols)
{
    (void)protocols;
    uint64_t srcEndPointId = 0;
    if (srcPort != nullptr && srcPort->localId != -1 && !srcPort->eid.empty()) {
        std::string srcIp = eidToIP(srcPort->eid);
        auto ret = RunnerDB::GetOneByPred<sim::EndPoint>([&srcIp](const sim::EndPoint& d) {
            return strcmp(d.ip_addr, srcIp.c_str()) == 0;
        });
        if (!ret.second) {
            HCCL_VM_ERROR("cannot find endPoint by ip {}", srcIp);
            return HCCL_SIM_E_NOT_FOUND;
        }
        srcEndPointId = ret.first.id;
        // 更新Endpoint status
        RunnerDB::Update<sim::EndPoint>(srcEndPointId, [](sim::EndPoint& ep) {
            ep.status = 1;
        });
    }

    uint64_t dstEndPointId = 0;
    if (dstPort != nullptr && dstPort->localId != -1 && !dstPort->eid.empty()) {
        std::string dstIp = eidToIP(dstPort->eid);
        auto ret = RunnerDB::GetOneByPred<sim::EndPoint>([&dstIp](const sim::EndPoint& d) {
            return strcmp(d.ip_addr, dstIp.c_str()) == 0;
        });
        if (!ret.second) {
            HCCL_VM_ERROR("cannot find endPoint by ip {}", dstIp);
            return HCCL_SIM_E_NOT_FOUND;
        }
        dstEndPointId = ret.first.id;
        // 更新Endpoint status
        RunnerDB::Update<sim::EndPoint>(dstEndPointId, [](sim::EndPoint& ep) {
            ep.status = 1;
        });
    }

    sim::Link link{};
    link.local_endpoint_id = srcEndPointId;
    link.remote_endpoint_id = dstEndPointId;
    link.net_layer = netLayer;
    link.type = 0;
    auto linkKey = RunnerDB::Add<sim::Link>(link);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// 解析用户提供的ranktable文件，得到topoMeta信息，并初始化通信域相关数据库表项
HcclVmResult AscendClusterTopoParser::ParseRanktable(const std::string& ranktablePath, TopoMeta& topoMeta)
{
    std::ifstream ifs(ranktablePath);
    if (!ifs.is_open()) {
        HCCL_VM_ERROR("failed to open ranktable file: {}", ranktablePath);
        return HcclVmResult::HCCL_SIM_E_OPEN_FILE_FAILURE;
    }

    json rankTable;
    try {
        rankTable = json::parse(ifs);
    } catch (const json::exception& e) {
        HCCL_VM_ERROR("failed to parse ranktable json: {}", e.what());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    ifs.close();

    if (!rankTable.contains("rank_list") || !rankTable["rank_list"].is_array()) {
        HCCL_VM_ERROR("rank_list not found or not array in ranktable");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    std::map<uint32_t, std::map<uint32_t, ServerMeta>> spSrvDevices;

    for (const auto& rank : rankTable["rank_list"]) {
        uint32_t localId = rank.value("local_id", 0);
        const auto& levelList = rank.value("level_list", json::array());

        uint32_t spIdx = 0;
        uint32_t srvIdx = 0;
        bool found = false;

        for (const auto& level : levelList) {
            if (level.value("net_layer", -1) == 0) {
                std::string netInstanceId = level.value("net_instance_id", "");
                if (sscanf(netInstanceId.c_str(), "superPod%u_rack%u", &spIdx, &srvIdx) == 2) {
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            for (const auto& level : levelList) {
                const auto& addrList = level.value("rank_addr_list", json::array());
                if (!addrList.empty()) {
                    std::string addr = addrList[0].value("addr", "");
                    std::string eidKey = "0x" + addr;
                    auto it = network_.eidIndex.find(eidKey);
                    if (it != network_.eidIndex.end() && !it->second.empty()) {
                        spIdx = static_cast<uint32_t>(std::get<0>(it->second[0]));
                        srvIdx = static_cast<uint32_t>(std::get<1>(it->second[0]));
                        found = true;
                        break;
                    }
                }
            }
        }

        if (!found) {
            HCCL_VM_WARN("rank local_id={} cannot determine superpod/server, skip", localId);
            continue;
        }

        spSrvDevices[spIdx][srvIdx].push_back(localId);
    }

    topoMeta.clear();
    if (spSrvDevices.empty()) {
        HCCL_VM_ERROR("no valid device found in ranktable");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    uint32_t maxSpIdx = spSrvDevices.rbegin()->first;
    for (uint32_t sp = 0; sp <= maxSpIdx; ++sp) {
        SuperPodMeta spMeta;
        auto spIt = spSrvDevices.find(sp);
        if (spIt != spSrvDevices.end()) {
            uint32_t maxSrvIdx = spIt->second.rbegin()->first;
            for (uint32_t srv = 0; srv <= maxSrvIdx; ++srv) {
                ServerMeta srvMeta;
                auto srvIt = spIt->second.find(srv);
                if (srvIt != spIt->second.end()) {
                    srvMeta = srvIt->second;
                }
                spMeta.push_back(srvMeta);
            }
        }
        topoMeta.push_back(spMeta);
    }

    if (!network_.superPods.empty() && !network_.superPods[0].servers.empty()
        && !network_.superPods[0].servers[0].devices.empty()) {
        HCCL_VM_ERROR("network superpord is error");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    HCCL_VM_DEBUG("parsed ranktable: topoMetaSize={}", topoMeta.size());
    return HcclVmResult::HCCL_SIM_SUCCESS;
}
