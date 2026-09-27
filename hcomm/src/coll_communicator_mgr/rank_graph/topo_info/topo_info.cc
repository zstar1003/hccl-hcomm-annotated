/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_info.h"
#include "json_parser.h"
#include "invalid_params_exception.h"
#include "exception_util.h"
#include "rank_table_report_macro.h"

namespace Hccl {

using namespace std;

void TopoInfo::Deserialize(const nlohmann::json& topoInfoJson)
{
    std::string msgVersion = "[TopoInfo::Deserialize] error occurs when parser object of propName \"version\"";
    std::string msgPc = "[TopoInfo::Deserialize] error occurs when parser object of propName \"peer_count\"";
    std::string msgEc = "[TopoInfo::Deserialize] error occurs when parser object of propName \"edge_count\"";
    TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msgVersion, (version = GetJsonProperty(topoInfoJson, "version")), topoInfoJson,
        "version", "non-empty string");
    TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msgPc, (peerCount = GetJsonPropertyUInt(topoInfoJson, "peer_count")), topoInfoJson,
        "peer_count", "0 ~ UINT32_MAX");
    TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msgEc, (edgeCount = GetJsonPropertyUInt(topoInfoJson, "edge_count")), topoInfoJson,
        "edge_count", "0 ~ UINT32_MAX");

    if (version != "2.0") {
        HCCL_ERROR("[TopoInfo::%s] failed with version[%s] is not \"2.0\".", __func__, version.c_str());
        THROW<InvalidParamsException>(StringFormat(
            "[TopoInfo::%s] failed with version[%s] is not \"2.0\" in topo file.", __func__, version.c_str()));
    }

    if (peerCount == 0 || peerCount > MAX_PEER_COUNT) {
        THROW<InvalidParamsException>(
            "[TopoInfo::%s] the range for the prop peer_count is [1, %u] while peer_count is %u", __func__,
            MAX_PEER_COUNT, peerCount);
    }

    if (edgeCount == 0) {
        HCCL_WARNING("[TopoInfo::%s]: edge_count is zero", __func__);
    }

    if (peerCount == 0 || peerCount > MAX_PEER_COUNT) {
        THROW<InvalidParamsException>(
            "[TopoInfo::%s] the range for the prop peer_count is [1, 65] while peer_count is %u", __func__, peerCount);
    }

    DeserializePeers(topoInfoJson);
    DeserializeEdges(topoInfoJson);
}

void TopoInfo::DeserializePeers(const nlohmann::json& topoInfoJson)
{
    nlohmann::json peerJsons;
    std::string msgPl = "[TopoInfo::DeserializePeers] error occurs when parser object of propName \"peer_list\"";
    TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msgPl, (GetJsonPropertyList(topoInfoJson, "peer_list", peerJsons)), topoInfoJson,
        "peer_list", "array");
    for (auto& peerJson : peerJsons) {
        PeerInfo peer;
        peer.Deserialize(peerJson);

        if (idSet.count(peer.localId) > 0) {
            THROW<InvalidParamsException>(
                StringFormat("[TopoInfo::%s] in peers exist duplicate localId = %u.", __func__, peer.localId));
        }

        peers.emplace_back(peer);
        idSet.insert(peer.localId);
    }

    if (peerCount != peers.size()) {
        THROW<InvalidParamsException>(StringFormat(
            "[TopoInfo::%s] Value of peer_count[%u] is inconsistent with the size of the peer_list[%zu].", __func__,
            peerCount, peers.size()));
    }
}

void TopoInfo::VerifyEdges(EdgeInfo& edge)
{
    if (idSet.count(edge.localA) == 0 || idSet.count(edge.localB) == 0) {
        THROW<InvalidParamsException>(StringFormat(
            "[TopoInfo::%s] endpoint localId [%u] or [%u] is not exist in peers[%zu].", __func__, edge.localA,
            edge.localB, idSet.size()));
    }
    //  检查edge.netLayer这一层级是否存在，不存在则初始化
    if (edges.find(edge.netLayer) == edges.end()) {
        edges[edge.netLayer] = vector<EdgeInfo>();
    }
    //  判断edge.netLayer该层级是否存在重复edge
    if (find(edges[edge.netLayer].begin(), edges[edge.netLayer].end(), edge) != edges[edge.netLayer].end()) {
        THROW<InvalidParamsException>(StringFormat(
            "[TopoInfo::%s] exist duplicate edges. Location information:{edge.netLayer=%u, edge.linkType=%s, "
            "edge.topoType=%s, edge.topoInstanceId=%u, localA=%u, localB=%u}",
            __func__, edge.netLayer, edge.linkType.Describe().c_str(), edge.topoType.Describe().c_str(),
            edge.topoInstId, edge.localA, edge.localB));
    }

    edges[edge.netLayer].emplace_back(edge);
}

void TopoInfo::DeserializeEdges(const nlohmann::json& topoInfoJson)
{
    nlohmann::json edgeJsons;
    std::string msgPe = "[TopoInfo::DeserializeEdges] error occurs when parser object of propName \"edge_list\"";
    TRY_CATCH_THROW_REPORT_TOPO(
        InvalidParamsException, msgPe, (GetJsonPropertyList(topoInfoJson, "edge_list", edgeJsons)), topoInfoJson,
        "edge_list", "array");
    if (edgeJsons.empty()) {
        if (edgeCount != 0) {
            THROW<InvalidParamsException>(StringFormat(
                "[TopoInfo::%s] Value of edge_count[%u] is inconsistent with the size of edge_list[0].", __func__,
                edgeCount));
        } else {
            HCCL_WARNING("[TopoInfo::%s] edge count is zero", __func__);
            return;
        }
    }
    for (auto& edgeJson : edgeJsons) {
        EdgeInfo edge;
        edge.Deserialize(edgeJson);
        VerifyEdges(edge);
    }

    size_t sumEdge = 0;
    for (const auto& entry : edges) {
        sumEdge += entry.second.size();
    }
    if (sumEdge != edgeCount) {
        THROW<InvalidParamsException>(StringFormat(
            "[TopoInfo::%s] Value of edge_count[%u] is inconsistent with the size of edge_list[%zu].", __func__,
            edgeCount, sumEdge));
    }
}

string TopoInfo::Describe() const
{
    string description = "TopoInfo{";
    description += StringFormat("version=%s", version.c_str());
    description += StringFormat(", peer_count=%u", peerCount);
    description += StringFormat(", edge_count=%u", edgeCount);
    description += StringFormat(", Peers.size=%u", peers.size());
    description += StringFormat(", Edges.size=%u", edges.size());
    description += "}";
    return description;
}

void TopoInfo::Dump() const
{
    HCCL_DEBUG("TopoInfo Dump:");
    HCCL_DEBUG("%s", Describe().c_str());
    HCCL_DEBUG("peers:");
    for (const auto& peer : peers) {
        HCCL_DEBUG("%s", peer.Describe().c_str());
    }
    HCCL_DEBUG("edges:");
    for (const auto& itor : edges) {
        HCCL_DEBUG("netLayer[%u]:", itor.first);
        for (const auto& edge : itor.second) {
            HCCL_DEBUG("    %s", edge.Describe().c_str());
        }
    }
}

TopoInfo::TopoInfo(BinaryStream& binaryStream)
{
    binaryStream >> version >> peerCount >> edgeCount;
    size_t peersSize = 0;
    binaryStream >> peersSize;
    for (u32 i = 0; i < peersSize; i++) {
        PeerInfo peer(binaryStream);
        peers.emplace_back(peer);
    }
    size_t edgesSize = 0;
    binaryStream >> edgesSize;

    HCCL_INFO(
        "[TopoInfo] version is [%s], peerCount is [%u], edgeCount is [%u], peers size is [%u], edges size is [%u]",
        version.c_str(), peerCount, edgeCount, peers.size(), edgesSize);
    for (u32 i = 0; i < edgesSize; i++) {
        u32 edgeInfoIndex = 0;
        binaryStream >> edgeInfoIndex; // key
        size_t edgeSize = 0;
        binaryStream >> edgeSize;
        HCCL_INFO("[TopoInfo] edges key is [%u], value size is [%u]", edgeInfoIndex, edgeSize);
        for (u32 j = 0; j < edgeSize; j++) { // value
            EdgeInfo edge(binaryStream);
            edges[edgeInfoIndex].emplace_back(edge);
        }
    }
}

void TopoInfo::GetBinStream(BinaryStream& binaryStream) const
{
    binaryStream << version << peerCount << edgeCount;
    binaryStream << peers.size();
    for (auto& it : peers) {
        it.GetBinStream(binaryStream);
    }
    binaryStream << edges.size();
    HCCL_INFO(
        "[TopoInfo::GetBinStream] version is [%s], peerCount is [%u], edgeCount is [%u], peers size is [%u], edges "
        "size is [%u]",
        version.c_str(), peerCount, edgeCount, peers.size(), edges.size());
    for (auto& it : edges) {
        binaryStream << it.first;
        binaryStream << it.second.size();
        HCCL_INFO("[TopoInfo::GetBinStream] edges key is [%u], value size is [%u]", it.first, it.second.size());
        for (auto& edge : it.second) {
            edge.GetBinStream(binaryStream);
        }
    }
}

} // namespace Hccl
