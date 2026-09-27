/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <json.hpp>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>

#include "topo_cluster_parser.h"
#include "sim_log.h"

using json = nlohmann::json;

ClusterTopoParser::ClusterTopoParser() {}

ClusterTopoParser::~ClusterTopoParser() {}

ParseStatus ClusterTopoParser::ParseClusterDir(const std::string& clusterDir)
{
    network_ = Network();

    ParseStatus status = ScanSuperPodDirs(clusterDir);
    if (status != ParseStatus::OK) {
        return status;
    }

    std::string serversInfoPath = clusterDir + "/servers_info.json";
    ParseStatus siStatus = ParseServersInfoJson(serversInfoPath);
    if (siStatus != ParseStatus::OK) {
        HCCL_VM_WARN("failed to parse servers_info.json");
    }

    return network_.BuildIndexes();
}

ParseStatus ClusterTopoParser::ScanSuperPodDirs(const std::string& clusterDir)
{
    DIR* dir = opendir(clusterDir.c_str());
    if (!dir) {
        return ParseStatus::FILE_NOT_FOUND;
    }

    struct dirent* entry;
    std::vector<std::string> superpodDirs;

    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }

        std::string fullPath = clusterDir + "/" + name;
        struct stat st;
        if (stat(fullPath.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            if (name.find("superpod") != std::string::npos || name.find("SuperPod") != std::string::npos) {
                superpodDirs.push_back(fullPath);
            }
        }
    }
    closedir(dir);

    std::sort(superpodDirs.begin(), superpodDirs.end());

    for (size_t i = 0; i < superpodDirs.size(); ++i) {
        SuperPod superpod;
        superpod.superPodId = static_cast<int>(i);

        std::string dirName = superpodDirs[i].substr(superpodDirs[i].find_last_of('/') + 1);
        size_t usPos = dirName.find('_');
        if (usPos != std::string::npos) {
            std::string numStr = dirName.substr(usPos + 1);
            try {
                superpod.superPodId = std::stoi(numStr);
            } catch (...) {
                HCCL_VM_ERROR("Convert numStr[{}] to int error", numStr.c_str());
            }
        }

        ParseStatus status = ScanServerDirs(superpodDirs[i], superpod);
        if (status != ParseStatus::OK) {
            return status;
        }

        network_.superPods.push_back(superpod);
    }

    if (!network_.superPods.empty() && network_.version.empty()) {
        network_.version = "2.0";
    }

    return ParseStatus::OK;
}

ParseStatus ClusterTopoParser::ScanServerDirs(const std::string& superpodDir, SuperPod& superpod)
{
    DIR* dir = opendir(superpodDir.c_str());
    if (!dir) {
        return ParseStatus::FILE_NOT_FOUND;
    }

    struct dirent* entry;
    std::vector<std::string> serverDirs;

    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }

        std::string fullPath = superpodDir + "/" + name;
        struct stat st;
        if (stat(fullPath.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            if (name.find("server") != std::string::npos || name.find("Server") != std::string::npos) {
                serverDirs.push_back(fullPath);
            }
        }
    }
    closedir(dir);

    std::sort(serverDirs.begin(), serverDirs.end());

    for (size_t i = 0; i < serverDirs.size(); ++i) {
        Server server;
        server.serverId = static_cast<int>(i);

        std::string dirName = serverDirs[i].substr(serverDirs[i].find_last_of('/') + 1);
        size_t usPos = dirName.find('_');
        if (usPos != std::string::npos) {
            std::string numStr = dirName.substr(usPos + 1);
            try {
                server.serverId = std::stoi(numStr);
            } catch (...) {
                HCCL_VM_ERROR("Convert numStr[{}] to int error", numStr.c_str());
            }
        }

        ParseStatus status = ParseServer(serverDirs[i], server);
        if (status != ParseStatus::OK) {
            return status;
        }

        superpod.servers.push_back(server);
    }

    return ParseStatus::OK;
}

ParseStatus ClusterTopoParser::ParseServer(const std::string& serverDir, Server& server)
{
    std::string topoPath = FindFileBySuffix(serverDir, "topo");
    std::string rootinfoPath = FindFileBySuffix(serverDir, "rootinfo");

    if (topoPath.empty()) {
        HCCL_VM_WARN("no topo.json found in {}", serverDir.c_str());
    }
    if (rootinfoPath.empty()) {
        HCCL_VM_WARN("no rootinfo.json found in {}", serverDir.c_str());
    }

    if (!topoPath.empty()) {
        ParseStatus status = ParseTopoJson(topoPath, server);
        if (status != ParseStatus::OK) {
            return status;
        }
    }

    if (!rootinfoPath.empty()) {
        ParseStatus status = ParseRootinfoJson(rootinfoPath, server);
        if (status != ParseStatus::OK) {
            return status;
        }
    }

    if (!topoPath.empty() && !rootinfoPath.empty()) {
        MergeTopoAndRootinfo(server);
    }

    return ParseStatus::OK;
}

std::string ClusterTopoParser::FindFileBySuffix(const std::string& dir, const std::string& suffix)
{
    DIR* dp = opendir(dir.c_str());
    if (!dp) {
        return "";
    }

    struct dirent* entry;
    std::string result;

    while ((entry = readdir(dp)) != nullptr) {
        std::string name = entry->d_name;
        if (name.find("switch_description") != std::string::npos) {
            continue;
        }
        if (name.size() > 5 && name.substr(name.size() - 5) == ".json") {
            if (name.find(suffix) != std::string::npos) {
                result = dir + "/" + name;
                break;
            }
        }
    }
    closedir(dp);
    return result;
}

ParseStatus ClusterTopoParser::ParseTopoJson(const std::string& topoPath, Server& server)
{
    std::ifstream ifs(topoPath);
    if (!ifs.is_open()) {
        return ParseStatus::FILE_NOT_FOUND;
    }

    try {
        json j;
        ifs >> j;

        if (j.contains("version") && j["version"].is_string()) {
            network_.version = j["version"].get<std::string>();
        }

        if (j.contains("hardware_type") && j["hardware_type"].is_string()) {
            server.hardwareType = j["hardware_type"].get<std::string>();
        }

        if (j.contains("peer_list") && j["peer_list"].is_array()) {
            for (const auto& peer : j["peer_list"]) {
                if (peer.contains("local_id")) {
                    Device dev;
                    dev.localId = peer["local_id"].get<int>();
                    server.devices.push_back(dev);
                }
            }
        }

        if (j.contains("edge_list") && j["edge_list"].is_array()) {
            for (const auto& edge : j["edge_list"]) {
                Link link;

                if (edge.contains("net_layer")) {
                    link.netLayer = edge["net_layer"].get<int>();
                }
                if (edge.contains("link_type")) {
                    link.linkType = ParseLinkType(edge["link_type"].get<std::string>());
                }
                if (edge.contains("topo_type")) {
                    link.topoType = ParseTopoType(edge["topo_type"].get<std::string>());
                }
                if (edge.contains("topo_instance_id")) {
                    link.topoInstanceId = edge["topo_instance_id"].get<int>();
                }
                if (edge.contains("topo_attr")) {
                    link.topoAttr = edge["topo_attr"].get<std::string>();
                }

                if (edge.contains("local_a")) {
                    link.sideA.localId = edge["local_a"].get<int>();
                }
                if (edge.contains("local_a_ports") && edge["local_a_ports"].is_array()) {
                    for (const auto& p : edge["local_a_ports"]) {
                        link.sideA.portIds.push_back(p.get<std::string>());
                    }
                }

                if (edge.contains("local_b")) {
                    link.sideB.localId = edge["local_b"].get<int>();
                }
                if (edge.contains("local_b_ports") && edge["local_b_ports"].is_array()) {
                    for (const auto& p : edge["local_b_ports"]) {
                        link.sideB.portIds.push_back(p.get<std::string>());
                    }
                }

                if (edge.contains("protocols") && edge["protocols"].is_array()) {
                    for (const auto& proto : edge["protocols"]) {
                        link.protocols.push_back(ParseProtocol(proto.get<std::string>()));
                    }
                }

                if (edge.contains("position")) {
                    link.position = ParsePosition(edge["position"].get<std::string>());
                }

                std::ostringstream linkIdBuilder;
                linkIdBuilder << "L" << link.netLayer << "_" << (link.IsPeer2Peer() ? "P2P" : "P2N") << "_"
                              << link.sideA.localId << "_" << link.sideB.localId << "_I" << link.topoInstanceId;
                link.linkId = linkIdBuilder.str();

                server.links.push_back(link);
            }
        }
    } catch (const json::exception& e) {
        HCCL_VM_ERROR("JSON parse error in {}: {}", topoPath.c_str(), e.what());
        return ParseStatus::PARSE_ERROR;
    }

    return ParseStatus::OK;
}

ParseStatus ClusterTopoParser::ParseRootinfoJson(const std::string& rootinfoPath, Server& server)
{
    std::ifstream ifs(rootinfoPath);
    if (!ifs.is_open()) {
        return ParseStatus::FILE_NOT_FOUND;
    }

    try {
        json j;
        ifs >> j;

        if (j.contains("version") && j["version"].is_string() && network_.version.empty()) {
            network_.version = j["version"].get<std::string>();
        }

        if (j.contains("rank_list") && j["rank_list"].is_array()) {
            for (const auto& rank : j["rank_list"]) {
                int localId = -1;
                int deviceId = -1;

                if (rank.contains("local_id")) {
                    localId = rank["local_id"].get<int>();
                }
                if (rank.contains("device_id")) {
                    deviceId = rank["device_id"].get<int>();
                }

                Device* dev = nullptr;
                for (auto& d : server.devices) {
                    if (d.localId == localId) {
                        dev = &d;
                        break;
                    }
                }

                if (!dev) {
                    Device newDev;
                    newDev.localId = localId;
                    newDev.deviceId = deviceId;
                    server.devices.push_back(newDev);
                    dev = &server.devices.back();
                } else {
                    dev->deviceId = deviceId;
                }

                if (rank.contains("level_list") && rank["level_list"].is_array()) {
                    for (const auto& level : rank["level_list"]) {
                        int netLayer = 0;
                        std::string netInstanceId;

                        if (level.contains("net_layer")) {
                            netLayer = level["net_layer"].get<int>();
                        }
                        if (level.contains("net_instance_id")) {
                            netInstanceId = level["net_instance_id"].get<std::string>();
                        }

                        if (server.netInstanceId.empty() && !netInstanceId.empty()) {
                            server.netInstanceId = netInstanceId;
                        }

                        if (level.contains("rank_addr_list") && level["rank_addr_list"].is_array()) {
                            for (const auto& addr : level["rank_addr_list"]) {
                                std::string eid;
                                std::vector<std::string> portIds;
                                std::string planeId;

                                if (addr.contains("addr")) {
                                    eid = addr["addr"].get<std::string>();
                                }
                                if (addr.contains("ports") && addr["ports"].is_array()) {
                                    for (const auto& p : addr["ports"]) {
                                        portIds.push_back(p.get<std::string>());
                                    }
                                }
                                if (addr.contains("plane_id")) {
                                    planeId = addr["plane_id"].get<std::string>();
                                }

                                if (portIds.empty()) {
                                    Port port;
                                    port.layer = netLayer;
                                    port.eid = eid;
                                    port.planeId = planeId;
                                    dev->ports.push_back(port);
                                } else if (portIds.size() == 1) {
                                    Port port;
                                    port.layer = netLayer;
                                    port.eid = eid;
                                    port.portId = portIds[0];
                                    port.dieId = ExtractDieIdFromPortId(portIds[0]);
                                    port.planeId = planeId;
                                    dev->ports.push_back(port);
                                } else {
                                    for (const auto& pid : portIds) {
                                        Port port;
                                        port.layer = netLayer;
                                        port.eid = eid;
                                        port.portId = pid;
                                        port.dieId = ExtractDieIdFromPortId(pid);
                                        port.planeId = planeId;
                                        dev->ports.push_back(port);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } catch (const json::exception& e) {
        HCCL_VM_ERROR("JSON parse error in {}: {}", rootinfoPath.c_str(), e.what());
        return ParseStatus::PARSE_ERROR;
    }

    return ParseStatus::OK;
}

void ClusterTopoParser::MergeTopoAndRootinfo(Server& server)
{
    std::map<int, std::map<std::string, std::string>> devicePortGroupEid;

    for (const auto& dev : server.devices) {
        std::map<std::string, std::string> portGroupEid;
        std::map<std::string, std::vector<std::string>> eidPortGroups;

        for (const auto& port : dev.ports) {
            if (!port.eid.empty() && !port.portId.empty()) {
                eidPortGroups[port.eid].push_back(port.portId);
            }
        }

        for (const auto& [eid, pids] : eidPortGroups) {
            std::string key;
            std::vector<std::string> sortedPids = pids;
            std::sort(sortedPids.begin(), sortedPids.end());
            for (const auto& pid : sortedPids) {
                if (!key.empty()) {
                    key += ",";
                }
                key += pid;
            }
            portGroupEid[key] = eid;
        }

        devicePortGroupEid[dev.localId] = portGroupEid;
    }

    for (auto& link : server.links) {
        auto fillEid = [&](LinkPortRef& side) {
            if (side.portIds.empty()) {
                return;
            }

            auto devIt = devicePortGroupEid.find(side.localId);
            if (devIt == devicePortGroupEid.end()) {
                return;
            }

            std::string key;
            std::vector<std::string> sortedPids = side.portIds;
            std::sort(sortedPids.begin(), sortedPids.end());
            for (const auto& pid : sortedPids) {
                if (!key.empty()) {
                    key += ",";
                }
                key += pid;
            }

            auto eidIt = devIt->second.find(key);
            if (eidIt != devIt->second.end()) {
                side.eid = eidIt->second;
            } else if (sortedPids.size() > 1) {
                // Fallback: subset matching for PEER2NET links.
                // When port_list (topo) differs from port_group (rootinfo),
                // the link's portIds may be a superset of a rootinfo portGroup.
                // Assign the first matching portGroup's EID.
                std::set<std::string> linkPortSet(sortedPids.begin(), sortedPids.end());
                for (const auto& [rootinfoKey, rootinfoEid] : devIt->second) {
                    std::istringstream iss(rootinfoKey);
                    std::string token;
                    bool allFound = true;
                    while (std::getline(iss, token, ',')) {
                        if (linkPortSet.find(token) == linkPortSet.end()) {
                            allFound = false;
                            break;
                        }
                    }
                    if (allFound) {
                        side.eid = rootinfoEid;
                        break;
                    }
                }
            }
        };

        fillEid(link.sideA);
        fillEid(link.sideB);
    }

    for (auto& dev : server.devices) {
        std::map<int, LinkType> layerLinkType;
        std::map<int, TopoType> layerTopoType;
        std::map<int, int> layerTopoInstanceId;
        std::map<int, Position> layerPosition;
        std::map<int, std::vector<Protocol>> layerProtocols;

        for (const auto& link : server.links) {
            bool matchA = (link.sideA.localId == dev.localId);
            bool matchB = (link.sideB.localId == dev.localId);
            if (!matchA && !matchB) {
                continue;
            }

            int layer = link.netLayer;
            if (layerLinkType.find(layer) == layerLinkType.end()) {
                layerLinkType[layer] = link.linkType;
            }
            if (layerTopoType.find(layer) == layerTopoType.end()) {
                layerTopoType[layer] = link.topoType;
            }
            if (layerTopoInstanceId.find(layer) == layerTopoInstanceId.end()) {
                layerTopoInstanceId[layer] = link.topoInstanceId;
            }
            if (layerPosition.find(layer) == layerPosition.end()) {
                layerPosition[layer] = link.position;
            }
            if (layerProtocols.find(layer) == layerProtocols.end() && !link.protocols.empty()) {
                layerProtocols[layer] = link.protocols;
            }
        }

        for (auto& port : dev.ports) {
            int layer = port.layer;

            bool exactMatch = false;
            for (const auto& link : server.links) {
                if (link.netLayer != layer) {
                    continue;
                }

                bool matchA = (link.sideA.localId == dev.localId);
                bool matchB = (link.sideB.localId == dev.localId);
                if (!matchA && !matchB) {
                    continue;
                }

                if (matchA) {
                    for (const auto& pId : link.sideA.portIds) {
                        if (pId == port.portId) {
                            port.linkType = link.linkType;
                            port.topoType = link.topoType;
                            port.topoInstanceId = link.topoInstanceId;
                            port.position = link.position;
                            if (port.protocols.empty()) {
                                port.protocols = link.protocols;
                            }
                            exactMatch = true;
                            break;
                        }
                    }
                }
                if (exactMatch) {
                    break;
                }

                if (matchB) {
                    for (const auto& pId : link.sideB.portIds) {
                        if (pId == port.portId) {
                            port.linkType = link.linkType;
                            port.topoType = link.topoType;
                            port.topoInstanceId = link.topoInstanceId;
                            port.position = link.position;
                            if (port.protocols.empty()) {
                                port.protocols = link.protocols;
                            }
                            exactMatch = true;
                            break;
                        }
                    }
                }
                if (exactMatch) {
                    break;
                }
            }

            if (!exactMatch) {
                if (layerLinkType.count(layer)) {
                    port.linkType = layerLinkType[layer];
                }
                if (layerTopoType.count(layer)) {
                    port.topoType = layerTopoType[layer];
                }
                if (layerTopoInstanceId.count(layer)) {
                    port.topoInstanceId = layerTopoInstanceId[layer];
                }
                if (layerPosition.count(layer)) {
                    port.position = layerPosition[layer];
                }
                if (layerProtocols.count(layer) && port.protocols.empty()) {
                    port.protocols = layerProtocols[layer];
                }
            }
        }

        std::set<std::string> existingPortIds;
        for (const auto& port : dev.ports) {
            existingPortIds.insert(port.portId);
        }

        for (const auto& link : server.links) {
            bool matchA = (link.sideA.localId == dev.localId);
            bool matchB = (link.sideB.localId == dev.localId);
            if (!matchA && !matchB) {
                continue;
            }

            const auto& portIds = matchA ? link.sideA.portIds : link.sideB.portIds;
            for (const auto& pid : portIds) {
                if (existingPortIds.count(pid)) {
                    continue;
                }

                Port newPort;
                newPort.portId = pid;
                newPort.layer = link.netLayer;
                newPort.dieId = ExtractDieIdFromPortId(pid);
                newPort.eid = "";
                newPort.linkType = link.linkType;
                newPort.planeId = "plane_0";
                newPort.topoType = link.topoType;
                newPort.topoInstanceId = link.topoInstanceId;
                newPort.position = link.position;
                newPort.protocols = link.protocols;

                dev.ports.push_back(newPort);
                existingPortIds.insert(pid);
            }
        }
    }
}

int ClusterTopoParser::ExtractDieIdFromPortId(const std::string& portId)
{
    size_t slashPos = portId.find('/');
    if (slashPos != std::string::npos && slashPos > 0) {
        try {
            return std::stoi(portId.substr(0, slashPos));
        } catch (...) {
            HCCL_VM_ERROR("ExtractDieIdFromPortId error");
            return -1;
        }
    }
    return -1;
}

ParseStatus ClusterTopoParser::ParseServersInfoJson(const std::string& serversInfoPath)
{
    std::ifstream ifs(serversInfoPath);
    if (!ifs.is_open()) {
        HCCL_VM_WARN("servers_info.json not found: {}", serversInfoPath.c_str());
        return ParseStatus::FILE_NOT_FOUND;
    }

    try {
        json j;
        ifs >> j;

        if (j.contains("version") && j["version"].is_string() && network_.version.empty()) {
            network_.version = j["version"].get<std::string>();
        }

        if (j.contains("server_count") && j["server_count"].is_number()) {
            network_.serverCount = j["server_count"].get<int>();
        }

        if (j.contains("server_ip_list") && j["server_ip_list"].is_array()) {
            const auto& ipList = j["server_ip_list"];
            int serverGlobalIdx = 0;

            for (size_t spIdx = 0; spIdx < network_.superPods.size(); ++spIdx) {
                auto& superpod = network_.superPods[spIdx];
                for (size_t srvIdx = 0; srvIdx < superpod.servers.size(); ++srvIdx) {
                    auto& server = superpod.servers[srvIdx];
                    if (serverGlobalIdx < static_cast<int>(ipList.size())) {
                        const auto& entry = ipList[serverGlobalIdx];
                        if (entry.contains("addr") && entry["addr"].is_string()) {
                            server.hostEid = entry["addr"].get<std::string>();
                        }
                        std::string socVersion;
                        if (entry.contains("soc_version") && entry["soc_version"].is_string()) {
                            socVersion = entry["soc_version"].get<std::string>();
                        }
                        for (auto& dev : server.devices) {
                            if (!socVersion.empty()) {
                                dev.socVersion = socVersion;
                                dev.devType = ParseDevType(socVersion);
                            }
                        }
                    }
                    serverGlobalIdx++;
                }
            }
        }
    } catch (const json::exception& e) {
        HCCL_VM_ERROR("JSON parse error in {}: {}", serversInfoPath.c_str(), e.what());
        return ParseStatus::PARSE_ERROR;
    }

    return ParseStatus::OK;
}
