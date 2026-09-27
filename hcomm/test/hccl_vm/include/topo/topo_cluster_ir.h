/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_CLUSTER_IR_H
#define TOPO_CLUSTER_IR_H

#include <map>
#include <string>
#include <tuple>
#include <vector>

enum class LinkType { UNKNOWN, PEER2PEER, PEER2NET };

enum class TopoType { UNKNOWN, MESH_1D, MESH_2D, CLOS, RING };

enum class Protocol { UNKNOWN, UB_CTP, UB_MEM, UB_TP, ROCE };

enum class Position { UNKNOWN, DEVICE, HOST };

enum class SimDevType { UNKNOWN, DEV_TYPE_950, DEV_TYPE_910B, DEV_TYPE_910C, DEV_TYPE_910D };

enum class ParseStatus { OK, FILE_NOT_FOUND, PARSE_ERROR, INVALID_FORMAT };

LinkType ParseLinkType(const std::string& str);
std::string LinkTypeToString(LinkType type);

TopoType ParseTopoType(const std::string& str);
std::string TopoTypeToString(TopoType type);

Protocol ParseProtocol(const std::string& str);
std::string ProtocolToString(Protocol proto);

Position ParsePosition(const std::string& str);
std::string PositionToString(Position pos);

SimDevType ParseDevType(const std::string& str);
std::string DevTypeToString(SimDevType type);

struct Port {
    std::string portId;
    int dieId;
    int layer;
    std::string eid;
    LinkType linkType;
    std::string planeId;
    std::vector<Protocol> protocols;
    TopoType topoType;
    int topoInstanceId;
    Position position;

    Port()
        : portId(""),
          dieId(-1),
          layer(0),
          eid(""),
          linkType(LinkType::UNKNOWN),
          planeId(""),
          protocols(),
          topoType(TopoType::UNKNOWN),
          topoInstanceId(0),
          position(Position::UNKNOWN)
    {}
};

struct Device {
    int localId;
    int deviceId;
    SimDevType devType;
    std::string socVersion;
    std::vector<Port> ports;

    Device() : localId(-1), deviceId(-1), devType(SimDevType::UNKNOWN), socVersion(""), ports() {}

    std::vector<const Port*> GetPortsByLayer(int layer) const;
    std::vector<const Port*> GetPortsByLinkType(LinkType type) const;
    std::vector<const Port*> GetPortsByPlane(const std::string& pid) const;
    const Port* GetPort(const std::string& pid) const;
};

struct LinkPortRef {
    int localId;
    std::string eid;
    std::vector<std::string> portIds;

    LinkPortRef() : localId(-1), portIds() {}
};

struct Link {
    std::string linkId;
    int netLayer;
    LinkType linkType;
    TopoType topoType;
    int topoInstanceId;
    std::string topoAttr;
    LinkPortRef sideA;
    LinkPortRef sideB;
    std::vector<Protocol> protocols;
    Position position;

    Link()
        : linkId(""),
          netLayer(0),
          linkType(LinkType::UNKNOWN),
          topoType(TopoType::UNKNOWN),
          topoInstanceId(0),
          topoAttr(""),
          sideA(),
          sideB(),
          protocols(),
          position(Position::UNKNOWN)
    {}

    bool IsPeer2Peer() const { return linkType == LinkType::PEER2PEER; }
    bool IsPeer2Net() const { return linkType == LinkType::PEER2NET; }
};

struct Server {
    int serverId;
    std::string hardwareType;
    std::string netInstanceId;
    std::string hostEid;
    std::vector<Device> devices;
    std::vector<Link> links;

    Server() : serverId(-1), hardwareType(""), netInstanceId(""), hostEid(""), devices(), links() {}

    const Device* GetDevice(int localId) const;
    const Link* GetLink(const std::string& lid) const;
    std::vector<const Link*> GetLinksByLayer(int layer) const;
    std::vector<const Link*> GetLinksByType(LinkType type) const;
    std::vector<const Link*> GetDeviceLinks(int localId) const;
    int GetDeviceCount() const { return static_cast<int>(devices.size()); }
    int GetLinkCount() const { return static_cast<int>(links.size()); }
};

struct SuperPod {
    int superPodId;
    std::vector<Server> servers;

    SuperPod() : superPodId(-1), servers() {}

    const Server* GetServer(int serverId) const;
    int GetServerCount() const { return static_cast<int>(servers.size()); }
    int GetTotalDeviceCount() const;
};

struct Network {
    std::string version;
    int serverCount;
    std::vector<SuperPod> superPods;
    std::map<int, std::vector<std::pair<int, int>>> deviceIndex;
    std::map<std::string, std::vector<std::tuple<int, int, int>>> eidIndex;

    Network() : version(""), serverCount(0), superPods(), deviceIndex(), eidIndex() {}

    const SuperPod* GetSuperPod(int superPodId) const;
    const Server* GetServer(int superPodId, int serverId) const;
    const Device* GetDeviceByGlobalId(int deviceId) const;
    std::vector<const Device*> GetAllDevices() const;
    std::vector<const Port*> GetAllPortsByLayer(int layer) const;
    std::vector<const Port*> GetAllPortsByLinkType(LinkType type) const;
    std::vector<const Link*> GetAllLinksByType(LinkType type) const;
    int GetSuperPodCount() const { return static_cast<int>(superPods.size()); }
    int GetTotalDeviceCount() const;
    int GetTotalLinkCount() const;

    ParseStatus BuildIndexes();
};

#endif // TOPO_CLUSTER_IR_H
