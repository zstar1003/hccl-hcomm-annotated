/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <iostream>

#include "topo_cluster_dumper.h"
#include "sim_log.h"

std::string ClusterTopoDumper::Indent(int level) { return std::string(level * 2, ' '); }

void ClusterTopoDumper::DumpToFile(const Network& network, const std::string& filePath)
{
    std::ofstream ofs(filePath);
    if (!ofs.is_open()) {
        HCCL_VM_ERROR("cannot open file for writing: {}", filePath.c_str());
        return;
    }
    DumpToStream(network, ofs);
    ofs.close();
    HCCL_VM_INFO("IR data dumped to: {}", filePath.c_str());
}

void ClusterTopoDumper::DumpToStream(const Network& network, std::ostream& os) { DumpNetwork(network, os, 0); }

void ClusterTopoDumper::DumpNetwork(const Network& network, std::ostream& os, int indent)
{
    os << Indent(indent) << "Network {\n";
    os << Indent(indent + 1) << "version: \"" << network.version << "\"\n";
    os << Indent(indent + 1) << "serverCount: " << network.serverCount << "\n";
    os << Indent(indent + 1) << "superPodCount: " << network.GetSuperPodCount() << "\n";
    os << Indent(indent + 1) << "totalDeviceCount: " << network.GetTotalDeviceCount() << "\n";
    os << Indent(indent + 1) << "totalLinkCount: " << network.GetTotalLinkCount() << "\n";
    os << "\n";

    for (size_t i = 0; i < network.superPods.size(); ++i) {
        os << Indent(indent + 1) << "--- SuperPod[" << i << "] ---\n";
        DumpSuperPod(network.superPods[i], os, indent + 1);
        if (i + 1 < network.superPods.size()) {
            os << "\n";
        }
    }

    os << Indent(indent) << "}\n";
}

void ClusterTopoDumper::DumpSuperPod(const SuperPod& superpod, std::ostream& os, int indent)
{
    os << Indent(indent) << "SuperPod {\n";
    os << Indent(indent + 1) << "superPodId: " << superpod.superPodId << "\n";
    os << Indent(indent + 1) << "serverCount: " << superpod.GetServerCount() << "\n";
    os << Indent(indent + 1) << "totalDeviceCount: " << superpod.GetTotalDeviceCount() << "\n";
    os << "\n";

    for (size_t i = 0; i < superpod.servers.size(); ++i) {
        os << Indent(indent + 1) << "--- Server[" << i << "] ---\n";
        DumpServer(superpod.servers[i], os, indent + 1);
        if (i + 1 < superpod.servers.size()) {
            os << "\n";
        }
    }

    os << Indent(indent) << "}\n";
}

void ClusterTopoDumper::DumpServer(const Server& server, std::ostream& os, int indent)
{
    os << Indent(indent) << "Server {\n";
    os << Indent(indent + 1) << "serverId: " << server.serverId << "\n";
    os << Indent(indent + 1) << "hardwareType: \"" << server.hardwareType << "\"\n";
    os << Indent(indent + 1) << "netInstanceId: \"" << server.netInstanceId << "\"\n";
    os << Indent(indent + 1) << "hostEid: \"" << server.hostEid << "\"\n";
    os << Indent(indent + 1) << "deviceCount: " << server.GetDeviceCount() << "\n";
    os << Indent(indent + 1) << "linkCount: " << server.GetLinkCount() << "\n";
    os << "\n";

    os << Indent(indent + 1) << "=== Devices ===\n";
    for (size_t i = 0; i < server.devices.size(); ++i) {
        os << Indent(indent + 1) << "--- Device[" << i << "] ---\n";
        DumpDevice(server.devices[i], os, indent + 1);
    }

    os << "\n";
    os << Indent(indent + 1) << "=== Links ===\n";
    for (size_t i = 0; i < server.links.size(); ++i) {
        os << Indent(indent + 1) << "--- Link[" << i << "] ---\n";
        DumpLink(server.links[i], os, indent + 1);
    }

    os << Indent(indent) << "}\n";
}

void ClusterTopoDumper::DumpDevice(const Device& device, std::ostream& os, int indent)
{
    os << Indent(indent) << "Device {\n";
    os << Indent(indent + 1) << "localId: " << device.localId << "\n";
    os << Indent(indent + 1) << "deviceId: " << device.deviceId << "\n";
    os << Indent(indent + 1) << "devType: " << DevTypeToString(device.devType) << "\n";
    os << Indent(indent + 1) << "socVersion: \"" << device.socVersion << "\"\n";
    os << Indent(indent + 1) << "portCount: " << device.ports.size() << "\n";

    for (size_t i = 0; i < device.ports.size(); ++i) {
        os << Indent(indent + 1) << "--- Port[" << i << "] ---\n";
        DumpPort(device.ports[i], os, indent + 1);
    }

    os << Indent(indent) << "}\n";
}

void ClusterTopoDumper::DumpPort(const Port& port, std::ostream& os, int indent)
{
    os << Indent(indent) << "Port {\n";
    os << Indent(indent + 1) << "portId: \"" << port.portId << "\"\n";
    os << Indent(indent + 1) << "dieId: " << port.dieId << "\n";
    os << Indent(indent + 1) << "layer: " << port.layer << "\n";
    os << Indent(indent + 1) << "eid: \"" << port.eid << "\"\n";
    os << Indent(indent + 1) << "linkType: " << LinkTypeToString(port.linkType) << "\n";
    os << Indent(indent + 1) << "planeId: \"" << port.planeId << "\"\n";

    os << Indent(indent + 1) << "protocols: [";
    for (size_t i = 0; i < port.protocols.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << ProtocolToString(port.protocols[i]);
    }
    os << "]\n";

    os << Indent(indent + 1) << "topoType: " << TopoTypeToString(port.topoType) << "\n";
    os << Indent(indent + 1) << "topoInstanceId: " << port.topoInstanceId << "\n";
    os << Indent(indent + 1) << "position: " << PositionToString(port.position) << "\n";
    os << Indent(indent) << "}\n";
}

void ClusterTopoDumper::DumpLink(const Link& link, std::ostream& os, int indent)
{
    os << Indent(indent) << "Link {\n";
    os << Indent(indent + 1) << "linkId: \"" << link.linkId << "\"\n";
    os << Indent(indent + 1) << "netLayer: " << link.netLayer << "\n";
    os << Indent(indent + 1) << "linkType: " << LinkTypeToString(link.linkType) << "\n";
    os << Indent(indent + 1) << "topoType: " << TopoTypeToString(link.topoType) << "\n";
    os << Indent(indent + 1) << "topoInstanceId: " << link.topoInstanceId << "\n";
    os << Indent(indent + 1) << "topoAttr: \"" << link.topoAttr << "\"\n";

    os << Indent(indent + 1) << "sideA:\n";
    DumpLinkPortRef(link.sideA, os, indent + 2);

    os << Indent(indent + 1) << "sideB:\n";
    DumpLinkPortRef(link.sideB, os, indent + 2);

    os << Indent(indent + 1) << "protocols: [";
    for (size_t i = 0; i < link.protocols.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << ProtocolToString(link.protocols[i]);
    }
    os << "]\n";

    os << Indent(indent + 1) << "position: " << PositionToString(link.position) << "\n";
    os << Indent(indent) << "}\n";
}

void ClusterTopoDumper::DumpLinkPortRef(const LinkPortRef& ref, std::ostream& os, int indent)
{
    os << Indent(indent) << "LinkPortRef {\n";
    os << Indent(indent + 1) << "localId: " << ref.localId << "\n";
    os << Indent(indent + 1) << "eid: \"" << ref.eid << "\"\n";
    os << Indent(indent + 1) << "portIds: [";
    for (size_t i = 0; i < ref.portIds.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << "\"" << ref.portIds[i] << "\"";
    }
    os << "]\n";
    os << Indent(indent) << "}\n";
}
