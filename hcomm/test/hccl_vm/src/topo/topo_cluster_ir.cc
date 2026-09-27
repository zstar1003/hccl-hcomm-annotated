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

#include "topo_cluster_ir.h"

LinkType ParseLinkType(const std::string& str)
{
    if (str == "PEER2PEER") {
        return LinkType::PEER2PEER;
    }
    if (str == "PEER2NET") {
        return LinkType::PEER2NET;
    }
    return LinkType::UNKNOWN;
}

std::string LinkTypeToString(LinkType type)
{
    switch (type) {
        case LinkType::PEER2PEER:
            return "PEER2PEER";
        case LinkType::PEER2NET:
            return "PEER2NET";
        default:
            return "UNKNOWN";
    }
}

TopoType ParseTopoType(const std::string& str)
{
    if (str == "1DMESH") {
        return TopoType::MESH_1D;
    }
    if (str == "2DMESH") {
        return TopoType::MESH_2D;
    }
    if (str == "CLOS") {
        return TopoType::CLOS;
    }
    if (str == "RING") {
        return TopoType::RING;
    }
    return TopoType::UNKNOWN;
}

std::string TopoTypeToString(TopoType type)
{
    switch (type) {
        case TopoType::MESH_1D:
            return "1DMESH";
        case TopoType::MESH_2D:
            return "2DMESH";
        case TopoType::CLOS:
            return "CLOS";
        case TopoType::RING:
            return "RING";
        default:
            return "UNKNOWN";
    }
}

Protocol ParseProtocol(const std::string& str)
{
    if (str == "UB_CTP") {
        return Protocol::UB_CTP;
    }
    if (str == "UB_MEM") {
        return Protocol::UB_MEM;
    }
    if (str == "UB_TP") {
        return Protocol::UB_TP;
    }
    if (str == "ROCE") {
        return Protocol::ROCE;
    }
    return Protocol::UNKNOWN;
}

std::string ProtocolToString(Protocol proto)
{
    switch (proto) {
        case Protocol::UB_CTP:
            return "UB_CTP";
        case Protocol::UB_MEM:
            return "UB_MEM";
        case Protocol::UB_TP:
            return "UB_TP";
        case Protocol::ROCE:
            return "ROCE";
        default:
            return "UNKNOWN";
    }
}

Position ParsePosition(const std::string& str)
{
    if (str == "DEVICE") {
        return Position::DEVICE;
    }
    if (str == "HOST") {
        return Position::HOST;
    }
    return Position::UNKNOWN;
}

std::string PositionToString(Position pos)
{
    switch (pos) {
        case Position::DEVICE:
            return "DEVICE";
        case Position::HOST:
            return "HOST";
        default:
            return "UNKNOWN";
    }
}

SimDevType ParseDevType(const std::string& str)
{
    if (str == "ascend950" || str == "ASCEND950" || str == "950") {
        return SimDevType::DEV_TYPE_950;
    }
    if (str == "ascend910B" || str == "ASCEND910B" || str == "910B") {
        return SimDevType::DEV_TYPE_910B;
    }
    if (str == "ascend910C" || str == "ASCEND910C" || str == "910C") {
        return SimDevType::DEV_TYPE_910C;
    }
    if (str == "ascend910D" || str == "ASCEND910D" || str == "910D") {
        return SimDevType::DEV_TYPE_910D;
    }
    return SimDevType::UNKNOWN;
}

std::string DevTypeToString(SimDevType type)
{
    switch (type) {
        case SimDevType::DEV_TYPE_950:
            return "ASCEND950";
        case SimDevType::DEV_TYPE_910B:
            return "ASCEND910B";
        case SimDevType::DEV_TYPE_910C:
            return "ASCEND910C";
        case SimDevType::DEV_TYPE_910D:
            return "ASCEND910D";
        default:
            return "UNKNOWN";
    }
}

std::vector<const Port*> Device::GetPortsByLayer(int layer) const
{
    std::vector<const Port*> result;
    for (const auto& p : ports) {
        if (p.layer == layer) {
            result.push_back(&p);
        }
    }
    return result;
}

std::vector<const Port*> Device::GetPortsByLinkType(LinkType type) const
{
    std::vector<const Port*> result;
    for (const auto& p : ports) {
        if (p.linkType == type) {
            result.push_back(&p);
        }
    }
    return result;
}

std::vector<const Port*> Device::GetPortsByPlane(const std::string& pid) const
{
    std::vector<const Port*> result;
    for (const auto& p : ports) {
        if (p.planeId == pid) {
            result.push_back(&p);
        }
    }
    return result;
}

const Port* Device::GetPort(const std::string& pid) const
{
    for (const auto& p : ports) {
        if (p.portId == pid) {
            return &p;
        }
    }
    return nullptr;
}

const Device* Server::GetDevice(int localId) const
{
    for (const auto& d : devices) {
        if (d.localId == localId) {
            return &d;
        }
    }
    return nullptr;
}

const Link* Server::GetLink(const std::string& lid) const
{
    for (const auto& l : links) {
        if (l.linkId == lid) {
            return &l;
        }
    }
    return nullptr;
}

std::vector<const Link*> Server::GetLinksByLayer(int layer) const
{
    std::vector<const Link*> result;
    for (const auto& l : links) {
        if (l.netLayer == layer) {
            result.push_back(&l);
        }
    }
    return result;
}

std::vector<const Link*> Server::GetLinksByType(LinkType type) const
{
    std::vector<const Link*> result;
    for (const auto& l : links) {
        if (l.linkType == type) {
            result.push_back(&l);
        }
    }
    return result;
}

std::vector<const Link*> Server::GetDeviceLinks(int localId) const
{
    std::vector<const Link*> result;
    for (const auto& l : links) {
        if (l.sideA.localId == localId || l.sideB.localId == localId) {
            result.push_back(&l);
        }
    }
    return result;
}

const Server* SuperPod::GetServer(int serverId) const
{
    for (const auto& s : servers) {
        if (s.serverId == serverId) {
            return &s;
        }
    }
    return nullptr;
}

int SuperPod::GetTotalDeviceCount() const
{
    int count = 0;
    for (const auto& s : servers) {
        count += s.GetDeviceCount();
    }
    return count;
}

const SuperPod* Network::GetSuperPod(int superPodId) const
{
    for (const auto& sp : superPods) {
        if (sp.superPodId == superPodId) {
            return &sp;
        }
    }
    return nullptr;
}

const Server* Network::GetServer(int superPodId, int serverId) const
{
    const SuperPod* sp = GetSuperPod(superPodId);
    if (sp) {
        return sp->GetServer(serverId);
    }
    return nullptr;
}

const Device* Network::GetDeviceByGlobalId(int deviceId) const
{
    auto it = deviceIndex.find(deviceId);
    if (it != deviceIndex.end() && !it->second.empty()) {
        int spId = it->second[0].first;
        int srvId = it->second[0].second;
        const Server* srv = GetServer(spId, srvId);
        if (srv) {
            return srv->GetDevice(deviceId);
        }
    }
    return nullptr;
}

std::vector<const Device*> Network::GetAllDevices() const
{
    std::vector<const Device*> result;
    for (const auto& sp : superPods) {
        for (const auto& srv : sp.servers) {
            for (const auto& dev : srv.devices) {
                result.push_back(&dev);
            }
        }
    }
    return result;
}

std::vector<const Port*> Network::GetAllPortsByLayer(int layer) const
{
    std::vector<const Port*> result;
    for (const auto& sp : superPods) {
        for (const auto& srv : sp.servers) {
            for (const auto& dev : srv.devices) {
                auto ports = dev.GetPortsByLayer(layer);
                result.insert(result.end(), ports.begin(), ports.end());
            }
        }
    }
    return result;
}

std::vector<const Port*> Network::GetAllPortsByLinkType(LinkType type) const
{
    std::vector<const Port*> result;
    for (const auto& sp : superPods) {
        for (const auto& srv : sp.servers) {
            for (const auto& dev : srv.devices) {
                auto ports = dev.GetPortsByLinkType(type);
                result.insert(result.end(), ports.begin(), ports.end());
            }
        }
    }
    return result;
}

std::vector<const Link*> Network::GetAllLinksByType(LinkType type) const
{
    std::vector<const Link*> result;
    for (const auto& sp : superPods) {
        for (const auto& srv : sp.servers) {
            auto links = srv.GetLinksByType(type);
            result.insert(result.end(), links.begin(), links.end());
        }
    }
    return result;
}

int Network::GetTotalDeviceCount() const
{
    int count = 0;
    for (const auto& sp : superPods) {
        count += sp.GetTotalDeviceCount();
    }
    return count;
}

int Network::GetTotalLinkCount() const
{
    int count = 0;
    for (const auto& sp : superPods) {
        for (const auto& srv : sp.servers) {
            count += srv.GetLinkCount();
        }
    }
    return count;
}

ParseStatus Network::BuildIndexes()
{
    deviceIndex.clear();
    eidIndex.clear();
    for (const auto& sp : superPods) {
        for (const auto& srv : sp.servers) {
            for (const auto& dev : srv.devices) {
                deviceIndex[dev.deviceId].push_back({sp.superPodId, srv.serverId});
                for (const auto& port : dev.ports) {
                    if (!port.eid.empty()) {
                        eidIndex[port.eid].push_back({sp.superPodId, srv.serverId, dev.localId});
                    }
                }
            }
        }
    }
    return ParseStatus::OK;
}
