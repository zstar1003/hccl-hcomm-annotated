/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "edge_info.h"
#include "json_parser.h"
#include "invalid_params_exception.h"
#include "exception_util.h"
#include "rank_table_report_macro.h"

namespace Hccl {

const unordered_map<string, LinkProtocol> EdgeInfo::strToLinkProtocol = (unordered_map<string, LinkProtocol>{
    {"UB_CTP", LinkProtocol::UB_CTP},
    {"UB_TP", LinkProtocol::UB_TP},
    {"ROCE", LinkProtocol::ROCE},
    {"HCCS", LinkProtocol::HCCS},
    {"PCIE", LinkProtocol::PCIE},
    {"TCP", LinkProtocol::TCP},
    {"UB_MEM", LinkProtocol::UB_MEM},
    {"UBOE", LinkProtocol::UBOE},
    {"UB_RTP", LinkProtocol::UB_RTP},
    {"UBG", LinkProtocol::UB_RTP}});

const unordered_map<std::string, TopoType> EdgeInfo::strToTopoType = (unordered_map<string, TopoType>{
    {"CLOS", TopoType::CLOS},
    {"1DMESH", TopoType::MESH_1D},
    {"2DMESH", TopoType::MESH_2D},
    {"A3_SERVER", TopoType::A3_SERVER},
    {"A2_AX_SERVER", TopoType::A2_AX_SERVER}});

const unordered_map<string, LinkType> EdgeInfo::strToLinkType
    = (unordered_map<string, LinkType>{{"PEER2PEER", LinkType::PEER2PEER}, {"PEER2NET", LinkType::PEER2NET}});

const unordered_map<string, AddrPosition> EdgeInfo::strToAddrPosition
    = (unordered_map<string, AddrPosition>{{"DEVICE", AddrPosition::DEVICE}, {"HOST", AddrPosition::HOST}});

void EdgeInfo::Deserialize(const nlohmann::json& edgeInfoJson)
{
    std::string msgNetLayer = "[EdgeInfo::Deserialize] error occurs when parser object of propName \"net_layer\"";
    TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msgNetLayer, (netLayer = GetJsonPropertyUInt(edgeInfoJson, "net_layer")), edgeInfoJson,
        "net_layer", "0 ~ UINT32_MAX");
    if (netLayer > MAX_VALUE_LEVEL) {
        THROW<InvalidParamsException>(StringFormat(
            "[EdgeInfo::%s] netLayer value[%u] is out of range[0, %u].", __func__, netLayer, MAX_VALUE_LEVEL));
    }

    DeserializeProtocol(edgeInfoJson);

    if (edgeInfoJson.contains("topo_type")) {
        std::string topoTypeStr;
        std::string msgtopoType = "[EdgeInfo::Deserialize] error occurs when parser object of propName \"topo_type\"";
        TRY_CATCH_THROW_REPORT_TOPO(
            InvalidParamsException, msgtopoType, (topoTypeStr = GetJsonProperty(edgeInfoJson, "topo_type")),
            edgeInfoJson, "topo_type", "non-empty string");
        topoType = GetTopoType(topoTypeStr);
    } else {
        HCCL_WARNING("[EdgeInfo::%s] topo_type not found, [default]topo_type=TopoType::CLOS", __func__);
        topoType = TopoType::CLOS; // topo_type字段不存在时，取默认值CLOS
    }

    std::string msgtopoInstIdType
        = "[EdgeInfo::Deserialize] error occurs when parser object of propName \"topo_instance_id\"";
    if (edgeInfoJson.contains("topo_instance_id")) {
        TRY_CATCH_THROW_REPORT_TOPO(
            InvalidParamsException, msgtopoInstIdType,
            (topoInstId = GetJsonPropertyUInt(edgeInfoJson, "topo_instance_id")), edgeInfoJson, "topo_instance_id",
            "0 ~ UINT32_MAX");
    } else {
        HCCL_WARNING("[EdgeInfo::%s] topo_instance_id not found, [default]topo_instance_id=0", __func__);
        topoInstId = 0;
    }

    // 解析localA 和 localB
    DeserializeEndpoint(edgeInfoJson);
}

void EdgeInfo::DeserializeProtocol(const nlohmann::json& edgeInfoJson)
{
    nlohmann::json jsonProtocols;
    std::string msgProtocols
        = "[EdgeInfo::DeserializeProtocol] error occurs when parser object of propName \"protocols\"";
    TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msgProtocols, (GetJsonPropertyList(edgeInfoJson, "protocols", jsonProtocols)),
        edgeInfoJson, "protocols", "array");
    for (auto& protocolEle : jsonProtocols) {
        auto protocolStr = protocolEle.get<std::string>();
        LinkProtocol protocol = GetLinkProtocol(protocolStr);
        if (protocols.count(protocol) == 0) {
            protocols.emplace(protocol);
        } else {
            HCCL_WARNING("[EdgeInfo::%s] repeat member[%s] in \"protocols\"", __func__, protocolStr.c_str());
        }
    }

    if (protocols.empty()) {
        THROW<InvalidParamsException>("[EdgeInfo::%s] \"protocols\" is empty", __func__);
    }
}

void EdgeInfo::DeserializeEndpoint(const nlohmann::json& edgeInfoJson)
{
    std::string linkTypeStr;
    std::string msglinkType
        = "[EdgeInfo::DeserializeEndpoint] error occurs when parser object of propName \"link_type\"";
    TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msglinkType, (linkTypeStr = GetJsonProperty(edgeInfoJson, "link_type")), edgeInfoJson,
        "link_type", "non-empty string");
    linkType = GetLinkType(linkTypeStr);

    std::string msgLocalA = "[EdgeInfo::DeserializeEndpoint] error occurs when parser object of propName \"local_a\"";
    TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msgLocalA, (localA = GetJsonPropertyUInt(edgeInfoJson, "local_a")), edgeInfoJson,
        "local_a", "0 ~ UINT32_MAX");

    DeserializePort(edgeInfoJson, "local_a_ports", localAPorts);
    if (localAPorts.empty()) {
        THROW<InvalidParamsException>("[EdgeInfo::%s] local_a_ports can not be empty", __func__);
    }

    if (linkType == LinkType::PEER2PEER) {
        std::string msgLocalB
            = "[EdgeInfo::DeserializeEndpoint] error occurs when parser object of propName \"local_b\"";
        TRY_CATCH_THROW_REPORT_TOPO(
            InvalidParamsException, msgLocalB, (localB = GetJsonPropertyUInt(edgeInfoJson, "local_b")), edgeInfoJson,
            "local_b", "0 ~ UINT32_MAX");

        if (localA == localB) { // localA 和 localB 不能是同一个点
            THROW<InvalidParamsException>(
                "[EdgeInfo::%s] local_a and local_b can not be the same Endpoint id[%u].", __func__, localA);
        }

        DeserializePort(edgeInfoJson, "local_b_ports", localBPorts);
        if (localBPorts.empty()) {
            THROW<InvalidParamsException>("[EdgeInfo::%s] local_b_ports can not be empty when PEER2PEER", __func__);
        }
    } else {
        if (edgeInfoJson.contains("local_b") || edgeInfoJson.contains("local_b_ports")) {
            HCCL_WARNING("[EdgeInfo::%s] local_b and local_b_ports are not needed when PEER2NET", __func__);
        }
    }

    if (edgeInfoJson.contains("position")) {
        string positionStr;
        std::string msgPosition
            = "[EdgeInfo::DeserializeEndpoint] error occurs when parser object of propName \"position\"";
        TRY_CATCH_THROW_REPORT_TOPO(
            InvalidParamsException, msgPosition, (positionStr = GetJsonProperty(edgeInfoJson, "position")),
            edgeInfoJson, "position", "non-empty string");
        position = GetAddrPosition(positionStr);
    } else {
        HCCL_WARNING("[EdgeInfo::%s] position not found, [default]position=DEVICE", __func__);
        position = AddrPosition::DEVICE;
    }
}

void EdgeInfo::DeserializePort(const nlohmann::json& edgeInfoJson, std::string propName, std::set<std::string>& ports)
{
    nlohmann::json jsonPorts;
    std::string msgPort
        = StringFormat("[EdgeInfo::%s] error occurs when parser object of propName \"%s\"", __func__, propName.c_str());
    TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msgPort, (GetJsonPropertyList(edgeInfoJson, propName.c_str(), jsonPorts)), edgeInfoJson,
        propName.c_str(), "array");
    if (jsonPorts.empty() || jsonPorts.size() > MAX_PORTS_SIZE) {
        THROW<InvalidParamsException>(
            "[EdgeInfo::%s] ports[%s].size=[%zu] out of range[1, %u]", __func__, propName.c_str(), jsonPorts.size(),
            MAX_PORTS_SIZE);
    }
    for (auto& portEle : jsonPorts) {
        string port = portEle.get<string>();
        if (!port.empty() && port.size() <= PORT_MAX_LENGTH) {
            if (ports.count(port) == 0) {
                ports.emplace(port);
            } else {
                HCCL_WARNING("[EdgeInfo::%s] Repeat port:[%s]", __func__, port.c_str());
            }
        } else {
            THROW<InvalidParamsException>(
                "[EdgeInfo::%s] Invalid port[%s], length[%zu] out of range[1, %u]", __func__, port.c_str(), port.size(),
                PORT_MAX_LENGTH);
        }
    }
}

bool EdgeInfo::operator==(const EdgeInfo& other) const
{
    return netLayer == other.netLayer && linkType == other.linkType && protocols == other.protocols
           && topoType == other.topoType && topoInstId == other.topoInstId && CompareEndpoints(other)
           && position == other.position;
}

// 比较EndpointA和B
bool EdgeInfo::CompareEndpoints(const EdgeInfo& other) const
{
    // 无论什么情况，A=other.A && B=other.B时，可视为相同的连接关系
    if (localA == other.localA && localB == other.localB && localAPorts == other.localAPorts
        && localBPorts == other.localBPorts) {
        return true;
    }

    // 当连接类型是PEER2PEER时，A=other.B && B=other.A时，可视为相同的连接关系
    if (linkType == other.linkType && linkType == LinkType::PEER2PEER && localA == other.localB
        && localAPorts == other.localBPorts && localB == other.localA && localBPorts == other.localAPorts) {
        return true;
    }

    return false;
}

LinkProtocol EdgeInfo::GetLinkProtocol(string str) const
{
    if (strToLinkProtocol.count(str) == 0) {
        THROW<InvalidParamsException>(
            "[EdgeInfo::%s] string['%s'] is not type of LinkProtocol.", __func__, str.c_str());
    }
    return strToLinkProtocol.at(str);
}

TopoType EdgeInfo::GetTopoType(std::string topoTypeStr) const
{
    if (topoTypeStr.empty()) {
        return TopoType::CLOS; // 不填写时，取默认值
    }
    if (strToTopoType.count(topoTypeStr) == 0) {
        THROW<InvalidParamsException>(
            "[EdgeInfo::%s] string['%s'] is not type of TopoType.", __func__, topoTypeStr.c_str());
    }
    return strToTopoType.at(topoTypeStr);
}

LinkType EdgeInfo::GetLinkType(std::string linkTypeStr) const
{
    if (strToLinkType.count(linkTypeStr) == 0) {
        THROW<InvalidParamsException>(
            "[EdgeInfo::%s] string['%s'] is not type of LinkType.", __func__, linkTypeStr.c_str());
    }
    return strToLinkType.at(linkTypeStr);
}

AddrPosition EdgeInfo::GetAddrPosition(string str) const
{
    if (str.empty()) {
        HCCL_WARNING("[EdgeInfo::%s] position is null, [default]position=DEVICE", __func__);
        return AddrPosition::DEVICE; //  默认取值为DEVICE
    }
    if (strToAddrPosition.count(str) == 0) {
        THROW<InvalidParamsException>(StringFormat("string ['%s'] is not type of AddrPosition.", str.c_str()));
    }
    return strToAddrPosition.at(str);
}

std::string EdgeInfo::Describe() const
{
    stringstream protocolStr;
    protocolStr << "[";
    for (auto it = protocols.begin(); it != protocols.end(); ++it) {
        if (it != protocols.begin()) {
            protocolStr << ", ";
        }
        protocolStr << it->Describe();
    }
    protocolStr << "]";

    string localAPortsStr = DescribePorts(localAPorts);
    string localBPortsStr = DescribePorts(localBPorts);

    std::string description = "EdgeInfo{";
    description += StringFormat("netLayer=%u", netLayer);
    description += StringFormat("topoType=%s", topoType.Describe().c_str());
    description += StringFormat(", topoInstanceId=%u", topoInstId);
    description += StringFormat(", protocols=%s", protocolStr.str().c_str());
    description += StringFormat(", linkType=%s", linkType.Describe().c_str());
    description += StringFormat(", localA=%u", localA);
    description += StringFormat(", localAPortsStr=%s", localAPortsStr.c_str());
    description += StringFormat(", localB=%u", localB);
    description += StringFormat(", localBPortsStr=%s", localBPortsStr.c_str());
    description += StringFormat(", position=%s", position.Describe().c_str());
    description += "}";
    return description;
}

std::string EdgeInfo::DescribePorts(std::set<std::string> ports) const
{
    stringstream portsStr;
    portsStr << "[";
    for (auto it = ports.begin(); it != ports.end(); ++it) {
        if (it != ports.begin()) {
            portsStr << ", ";
        }
        portsStr << *it;
    }
    portsStr << "]";
    return portsStr.str();
}

void EdgeInfo::GetBinStream(BinaryStream& binaryStream) const
{
    binaryStream << netLayer << static_cast<u32>(linkType) << static_cast<u32>(topoType) << topoInstId;
    binaryStream << protocols.size();
    for (LinkProtocol protocol : protocols) {
        binaryStream << static_cast<u32>(protocol);
    }

    binaryStream << localA << localB;

    binaryStream << localAPorts.size();
    for (string port : localAPorts) {
        binaryStream << port;
    }

    binaryStream << localBPorts.size();
    for (string port : localBPorts) {
        binaryStream << port;
    }

    binaryStream << static_cast<u32>(position);
}

EdgeInfo::EdgeInfo(BinaryStream& binaryStream)
{
    binaryStream >> netLayer;
    u32 linkTypeTmp;
    binaryStream >> linkTypeTmp;
    linkType = static_cast<LinkType::Value>(linkTypeTmp);
    u32 topoTypeTmp;
    binaryStream >> topoTypeTmp;
    topoType = static_cast<TopoType::Value>(topoTypeTmp);
    binaryStream >> topoInstId;
    size_t protocolSize;
    binaryStream >> protocolSize;
    protocols.clear();
    for (size_t i = 0; i < protocolSize; i++) {
        u32 protocolTmp;
        binaryStream >> protocolTmp;
        LinkProtocol protocol = static_cast<LinkProtocol::Value>(protocolTmp);
        protocols.emplace(protocol);
    }

    binaryStream >> localA >> localB;

    size_t localAPortsSize;
    binaryStream >> localAPortsSize;
    localAPorts.clear();
    for (size_t i = 0; i < localAPortsSize; i++) {
        string port;
        binaryStream >> port;
        localAPorts.emplace(port);
    }

    size_t localBPortsSize;
    binaryStream >> localBPortsSize;
    localBPorts.clear();
    for (size_t i = 0; i < localBPortsSize; i++) {
        string port;
        binaryStream >> port;
        localBPorts.emplace(port);
    }

    u32 positionTmp;
    binaryStream >> positionTmp;
    position = static_cast<AddrPosition::Value>(positionTmp);
}

} // namespace Hccl
