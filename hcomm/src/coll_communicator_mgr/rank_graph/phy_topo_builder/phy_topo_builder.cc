/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "phy_topo_builder.h"
#include "log.h"
#include "json_parser.h"
#include "exception_util.h"
#include "null_ptr_exception.h"
#include "invalid_params_exception.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"

namespace Hccl {

PhyTopoBuilder& PhyTopoBuilder::GetInstance()
{
    static PhyTopoBuilder phyTopoBuilder;
    return phyTopoBuilder;
}

// 根据nodeType和localId创建节点
std::shared_ptr<PhyTopo::Node> CreateNode(const PhyTopo::Node::NodeType nodeType, const LocalId localId)
{
    if (nodeType == PhyTopo::Node::NodeType::PEER) {
        return std::make_shared<PhyTopo::Peer>(localId);
    } else {
        return std::make_shared<PhyTopo::Fabric>();
    }
}

void PhyTopoBuilder::Build(const std::string& topoPath)
{
    std::lock_guard<std::mutex> lock(phyTopoMutex);

    if (PhyTopo::GetInstance()->IsInitFinished()) {
        HCCL_INFO("PhyTopo has been initialized and does not need to be rebuilt");
        return;
    }

    if (topoPath.empty()) {
        RPT_INPUT_ERR(
            true, "EI0004", std::vector<std::string>({"ranktable_path", "error_reason"}),
            std::vector<std::string>(
                {"Please check the path configuration of the topo json file.",
                 "The rankTable file path does not exist, the permission is insufficient, or the JSON format is "
                 "incorrect."}));
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] topo path %s is not a valid real path.", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CONFIG.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), topoPath.c_str());
        THROW<InvalidParamsException>("[PhyTopoBuilder::%s] Topo path is empty.", __func__);
    }

    HCCL_DEBUG("[PhyTopoBuilder::%s]Start to build physic topo.", __func__);

    auto topoInfo = LoadTopoInfo(topoPath);
    // 根据topoInfo，按netLayer构造Graph
    for (const auto& iter : topoInfo->edges) {
        auto netLayer = iter.first;
        auto graph = CreateGraph(iter.second);
        PhyTopo::GetInstance()->AddTopoGraph(netLayer, graph);
        HCCL_DEBUG("[PhyTopoBuilder::%s]Build netLayer[%u] topo graph success.", __func__, netLayer);
    }

    // 遍历peerList，把peerList中的localId构造peer节点，加到layer-0的graph中
    auto graph = PhyTopo::GetInstance()->GetTopoGraph(0);
    if (graph == nullptr) {
        HCCL_INFO("[PhyTopoBuilder::%s] layer0 graph is nullptr, build now", __func__);
        graph = make_shared<Graph<PhyTopo::Node, PhyTopo::Link>>();
        PhyTopo::GetInstance()->AddTopoGraph(0, graph);
    }
    for (const auto& iter : topoInfo->peers) {
        // 判断当前layer0的graph有没有这个节点
        if (!graph->HasNode(iter.localId)) {
            auto node = CreateNode(PhyTopo::Node::NodeType::PEER, iter.localId);
            graph->AddNode(iter.localId, node);
        }
    }

    PhyTopo::GetInstance()->InitFinish();

    HCCL_DEBUG("[PhyTopoBuilder::%s]build physic topo success.", __func__);
    PhyTopo::GetInstance()->Dump();
}

std::shared_ptr<TopoInfo> PhyTopoBuilder::LoadTopoInfo(const std::string& topoPath)
{
    std::shared_ptr<TopoInfo> topoInfo = std::make_shared<TopoInfo>();
    // 检查是否为非法路径以及size的大小。
    struct stat fileStat;
    if (stat(topoPath.c_str(), &fileStat) != 0) {
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] Get file stat failed, file path:%s, errno:%d, error: %s",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CONFIG.c_str(),
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), topoPath.c_str(), errno, strerror(errno));
        THROW<InvalidParamsException>(
            "[PhyTopoBuilder][LoadTopoInfo]Get file stat failed, file path:%s, errno:%d, error: %s", topoPath.c_str(),
            errno, strerror(errno));
    }

    u64 topoFileSize = static_cast<u64>(fileStat.st_size);
    if (topoFileSize > SUPPORT_MAX_TOPOFILE_SIZE || topoFileSize <= 0) {
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] topoFileSize size: %llu exceeds max allowed size (%u bytes)",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CONFIG.c_str(),
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), topoFileSize, SUPPORT_MAX_TOPOFILE_SIZE);
        THROW<InvalidParamsException>(StringFormat(
            "[PhyTopoBuilder][LoadTopoInfo]file %s size (%llu bytes) exceeds max allowed size (%u bytes)",
            topoPath.c_str(), topoFileSize, SUPPORT_MAX_TOPOFILE_SIZE));
    }

    JsonParser topoParser;
    topoParser.ParseFile(topoPath, *topoInfo);
    topoInfo_ = topoInfo;
    return topoInfo;
}

shared_ptr<PhyTopo::Link> CreatePeer2PeerLink(const LinkParams& params)
{
    LinkType linkType = LinkType::PEER2PEER;
    auto srcIface
        = std::make_shared<PhyTopo::ConnInterface>(params.localAPorts, params.position, linkType, params.protocols);
    auto dstIface
        = std::make_shared<PhyTopo::ConnInterface>(params.localBPorts, params.position, linkType, params.protocols);
    PhyTopo::LinkAttributes linkAttrs;
    linkAttrs.linktype = linkType;
    linkAttrs.protocols = params.protocols;
    auto link = std::make_shared<PhyTopo::Link>(
        params.srcNode, params.dstNode, linkAttrs, params.topoType, params.topoInstanceId);
    link->SetSourceIface(srcIface);
    params.srcNode->AddConnInterface(srcIface);
    link->SetTargetIface(dstIface);
    params.dstNode->AddConnInterface(dstIface);
    return {link};
}

shared_ptr<PhyTopo::Link> CreatePeer2NetLink(const LinkParams& params)
{
    LinkType linkType = LinkType::PEER2NET;
    auto srcIface
        = std::make_shared<PhyTopo::ConnInterface>(params.localAPorts, params.position, linkType, params.protocols);
    // fabric 没有ports
    std::set<std::string> ports{};
    auto dstIface = std::make_shared<PhyTopo::ConnInterface>(ports, params.position, linkType, params.protocols);
    PhyTopo::LinkAttributes linkAttrs;
    linkAttrs.linktype = linkType;
    linkAttrs.protocols = params.protocols;
    auto link = std::make_shared<PhyTopo::Link>(
        params.srcNode, params.dstNode, linkAttrs, params.topoType, params.topoInstanceId);
    link->SetSourceIface(srcIface);
    params.srcNode->AddConnInterface(srcIface);
    link->SetTargetIface(dstIface);
    params.dstNode->AddConnInterface(dstIface);
    return {link};
}

// 根据链路类型建链
shared_ptr<PhyTopo::Link> CreateLink(const LinkType linkType, const LinkParams& params)
{
    if (linkType == LinkType::PEER2PEER) {
        return CreatePeer2PeerLink(params);
    } else {
        return CreatePeer2NetLink(params);
    }
}

NodeId GetNodeId(const PhyTopo::Node::NodeType nodeType, LocalId localId)
{
    switch (nodeType) {
        case PhyTopo::Node::NodeType::PEER:
            return PhyTopo::Peer::GetId(localId);
        case PhyTopo::Node::NodeType::FABRIC:
            return PhyTopo::Fabric::GetId();
        default:
            THROW<InvalidParamsException>(StringFormat("[PhyTopoBuilder]Invalid NodeType."));
            return 0;
    }
}

std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>>
PhyTopoBuilder::CreateGraph(const std::vector<EdgeInfo>& edges) const
{
    auto graph = std::make_shared<Graph<PhyTopo::Node, PhyTopo::Link>>();

    for (const auto& edgeInfo : edges) {
        std::shared_ptr<PhyTopo::Node> nodeA;
        std::shared_ptr<PhyTopo::Node> nodeB;

        // 生成 nodeAId，localA 始终为 PEER 类型
        NodeId nodeAId = GetNodeId(PhyTopo::Node::NodeType::PEER, edgeInfo.localA);
        // 获取或创建 nodeA
        if (!graph->HasNode(nodeAId)) {
            nodeA = CreateNode(PhyTopo::Node::NodeType::PEER, edgeInfo.localA);
            graph->AddNode(nodeAId, nodeA);
            HCCL_DEBUG("[PhyTopoBuilder::%s] Add node[%llu] success.", __func__, nodeAId);
        } else {
            // 使用 TraverseNode 查找 nodeA
            graph->TraverseNode([&nodeA, nodeAId](NodeId id, const std::shared_ptr<PhyTopo::Node>& n) {
                if (id == nodeAId) {
                    nodeA = n;
                }
            });
            if (nodeA == nullptr) {
                HCCL_ERROR("[PhyTopoBuilder::CreateGraph] nodeAId [%llu] not found.", nodeAId);
                THROW<NullPtrException>(
                    StringFormat("[PhyTopoBuilder::CreateGraph] Unable to find node [%llu].", nodeAId));
            }
        }

        // 根据 linkType 生成 nodeBId
        PhyTopo::Node::NodeType nodeBType;
        if (edgeInfo.linkType == LinkType::PEER2PEER) {
            nodeBType = PhyTopo::Node::NodeType::PEER;
        } else {
            nodeBType = PhyTopo::Node::NodeType::FABRIC;
        }

        NodeId nodeBId = GetNodeId(nodeBType, edgeInfo.localB);
        // 获取或创建 nodeB
        if (!graph->HasNode(nodeBId)) {
            nodeB = CreateNode(nodeBType, edgeInfo.localB);
            graph->AddNode(nodeBId, nodeB);
            HCCL_DEBUG("[PhyTopoBuilder::%s] Add node[%llu] success.", __func__, nodeBId);
        } else {
            // 使用 TraverseNode 查找 nodeB
            graph->TraverseNode([&nodeB, nodeBId](NodeId id, const std::shared_ptr<PhyTopo::Node>& n) {
                if (id == nodeBId) {
                    nodeB = n;
                }
            });
            if (nodeB == nullptr) {
                HCCL_ERROR("[PhyTopoBuilder::CreateGraph] nodeBId [%llu] not found.", nodeBId);
                THROW<NullPtrException>(
                    StringFormat("[PhyTopoBuilder::CreateGraph] Unable to find node [%llu].", nodeBId));
            }
        }

        // 双向分别创建link
        LinkParams abLinkParams{
            nodeA,
            nodeB,
            edgeInfo.localAPorts,
            edgeInfo.localBPorts,
            edgeInfo.position,
            edgeInfo.topoType,
            edgeInfo.topoInstId,
            edgeInfo.protocols};
        auto abLinks = CreateLink(edgeInfo.linkType, abLinkParams);
        graph->AddEdge(nodeAId, nodeBId, abLinks);
        LinkParams baLinkParams{
            nodeB,
            nodeA,
            edgeInfo.localBPorts,
            edgeInfo.localAPorts,
            edgeInfo.position,
            edgeInfo.topoType,
            edgeInfo.topoInstId,
            edgeInfo.protocols};
        auto baLinks = CreateLink(edgeInfo.linkType, baLinkParams);
        graph->AddEdge(nodeBId, nodeAId, baLinks);
    }

    return graph;
}

void PhyTopoBuilder::RecoverBuild(const TopoInfo& topoInfo)
{
    std::lock_guard<std::mutex> lock(phyTopoMutex);

    // PhyTopo::GetInstance()->ResetInstance();
    if (PhyTopo::GetInstance()->IsInitFinished()) {
        return;
    }
    // 根据topoInfo，按netLayer构造Graph
    for (const auto& iter : topoInfo.edges) {
        auto netLayer = iter.first;
        auto graph = CreateGraph(iter.second);
        PhyTopo::GetInstance()->AddTopoGraph(netLayer, graph);
        HCCL_DEBUG("[PhyTopoBuilder::%s]Build netLayer[%u] topo graph success.", __func__, netLayer);
    }

    PhyTopo::GetInstance()->InitFinish();

    HCCL_DEBUG("[PhyTopoBuilder::%s]build physic topo success.", __func__);
    PhyTopo::GetInstance()->Dump();
}

std::shared_ptr<TopoInfo> PhyTopoBuilder::GetTopoInfo() const { return topoInfo_; }

} // namespace Hccl
