/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include "rank_graph_builder.h"
#include "detour_service.h"
#include "json_parser.h"
#include "phy_topo_builder.h"

namespace Hccl {

using namespace std;

// peer 与 fabric 之间的 hop 数（peer 经 fabric 一跳到达 = 2 hop）
constexpr u32 PEER2NET_LINK_HOP = 2;

unique_ptr<RankGraph> RankGraphBuilder::Build(const string& ranktableM, const string& topoPath, RankId myRank)
{
    PhyTopoBuilder::GetInstance().Build(topoPath);
    topoInfo_ = PhyTopoBuilder::GetInstance().GetTopoInfo();

    JsonParser rankTableParser;
    RankTableInfo rankTableInfo;
    rankTableParser.ParseString(ranktableM, rankTableInfo);
    rankTable_ = make_unique<RankTableInfo>(rankTableInfo);

    this->myRank_ = myRank;
    BuildRankGraph();

    HCCL_INFO("[RankGraphBuilder] Build VirtualTopo success!");
    rankGraph_->Dump();
    return std::move(rankGraph_);
}

unique_ptr<RankGraph> RankGraphBuilder::Build(const RankTableInfo& ranktable, const string& topoPath, RankId myRank)
{
    PhyTopoBuilder::GetInstance().Build(topoPath);
    topoInfo_ = PhyTopoBuilder::GetInstance().GetTopoInfo();
    rankTable_ = make_unique<RankTableInfo>(ranktable);

    myRank_ = myRank;
    BuildRankGraph();

    HCCL_INFO("[RankGraphBuilder] Build VirtualTopo success!");
    rankGraph_->Dump();
    return std::move(rankGraph_);
}

const RankLevelInfo& RankGraphBuilder::GetRankLevelInfoByNetLayer(const NewRankInfo& rankInfo, u32 netLayer) const
{
    auto it = std::find_if(
        rankInfo.rankLevelInfos.begin(), rankInfo.rankLevelInfos.end(), [netLayer](const RankLevelInfo& levelInfo) {
            return levelInfo.netLayer == netLayer;
        });
    if (it == rankInfo.rankLevelInfos.end()) {
        THROW<InvalidParamsException>(StringFormat(
            "[RankGraphBuilder][GetRankLevelInfoByNetLayer] rankId[%u] netLayer[%u] does not exist in ranktable.",
            rankInfo.rankId, netLayer));
    }
    return *it;
}

std::vector<shared_ptr<PhyTopo::Link>> GetPeer2NetPhyLinks(u32 netLayer, LocalId localId)
{
    const shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> phyGraph = PhyTopo::GetInstance()->GetTopoGraph(netLayer);
    if (phyGraph == nullptr) {
        THROW<InvalidParamsException>(StringFormat(
            "[RankGraphBuilder][GetPhyLink] localId[%d] is not level[%u] in"
            " topo.json, not match rankTable.",
            localId, netLayer));
    }
    std::vector<shared_ptr<PhyTopo::Link>> links;
    phyGraph->TraverseEdge(PhyTopo::Peer::GetId(localId), [&](shared_ptr<PhyTopo::Link> link) {
        if (link != nullptr) {
            links.push_back(link);
        }
    });

    if (links.empty()) {
        THROW<InvalidParamsException>(
            StringFormat("[RankGraphBuilder][GetPhyLink] SourceNode localId[%d] edge does not exist.", localId));
    }
    return links;
}

void RankGraphBuilder::AddPeer2NetLink(
    const u32 netLayer, const string& netInstId, RankId rankId, const AddressInfo& addrInfo,
    const shared_ptr<NetInstance::Fabric>& fabNode, const vector<shared_ptr<PhyTopo::Link>>& links)
{
    for (shared_ptr<PhyTopo::Link> link : links) {
        if (link->GetSourceIFace() == nullptr) {
            continue;
        }
        std::set<std::string> ports = link->GetSourceIFace()->GetPorts();
        std::set<std::string> rankGraphPorts;
        std::set_intersection(
            ports.begin(), ports.end(), addrInfo.ports.begin(), addrInfo.ports.end(),
            std::inserter(rankGraphPorts, rankGraphPorts.begin()));

        if (rankGraphPorts.empty()) {
            // 该地址在topo里没有对应边
            continue;
        }
        // 获取topoInstId topoType
        u32 topoInstId = link->GetTopoInstId();
        auto topoType = link->GetTopoType();

        // 构造 RankGraph 的 PeerIface
        shared_ptr<NetInstance::ConnInterface> peerIface = make_shared<NetInstance::ConnInterface>(
            addrInfo.addr, rankGraphPorts, link->GetSourceIFace()->GetPos(), LinkType::PEER2NET,
            link->GetLinkProtocols(), topoType, topoInstId);
        // 获取 rankId 对应 PeerNode
        shared_ptr<NetInstance::Peer> peerNode = peers_.at(rankId);
        peerNode->AddConnInterface(netLayer, peerIface);

        // 构造 peer2netLink 和 net2peerLink 两条link
        shared_ptr<NetInstance::Link> peer2netLink = make_shared<NetInstance::Link>(
            peerNode, fabNode, peerIface, nullptr, LinkType::PEER2NET, link->GetLinkProtocols(), LinkDirection::BOTH,
            PEER2NET_LINK_HOP);
        shared_ptr<NetInstance::Link> net2peerLink = make_shared<NetInstance::Link>(
            fabNode, peerNode, nullptr, peerIface, LinkType::PEER2NET, link->GetLinkProtocols(), LinkDirection::BOTH,
            PEER2NET_LINK_HOP);

        // 插入 link
        tempNetInsts_[netLayer][netInstId]->AddLink(peer2netLink);
        tempNetInsts_[netLayer][netInstId]->AddLink(net2peerLink);

        // 将rank插入到当前netInstance对应的topoInstance中
        tempNetInsts_[netLayer][netInstId]->UpdateTopoInst(topoInstId, topoType, rankId);

        // 只打印当前卡的rank_id和eid对应关系
        if (rankId == myRank_) {
            HCCL_RUN_INFO(
                "[RankGraphBuilder][AddPeer2NetLink] Add Peer2NetLink Net2PeerLink success. level[%u] "
                "netInstId[%s] rankId[%u] planeId[%s] AddrStr[%s],topoInstId[%u],topoType[%u]",
                netLayer, netInstId.c_str(), rankId, fabNode->GetPlaneId().c_str(), addrInfo.addr.Describe().c_str(),
                topoInstId, topoType);
        }
    }
}

void RankGraphBuilder::AddFabricInfo(u32 netLayer)
{
    auto netInst = rankGraph_->GetNetInstanceByRankId(netLayer, myRank_);
    if (netInst == nullptr) {
        THROW<NullPtrException>(
            StringFormat("[RankGraphBuilder][AddFabricInfo] rankGraph->GetNetInstanceByRankId is nullptr"));
    }

    if (netInst->GetNetType() != NetType::CLOS) {
        THROW<NotSupportException>(
            StringFormat("[RankGraphBuilder][AddFabricInfo] NetInstance is not CLOS, not support add fabric."));
    }
    set<RankId> inRanks = netInst->GetRankIds();
    string netInstId = netInst->GetNetInstId();
    const auto& myLevelInfo = GetRankLevelInfoByNetLayer(rankTable_->ranks[myRank_], netLayer);
    // 根据planeId确认Fabric个数，每个fabricId对应一个planeId
    std::map<PlaneId, FabricId> planeId2Node = GetFabricsFromAddrInfo(myLevelInfo.rankAddrs);

    if (planeId2Node.size() == 0) {
        HCCL_WARNING(
            "[RankGraphBuilder][AddFabricInfo] current rankId[%d] netLayer[%u] group no net plane", myRank_, netLayer);
        return;
    }
    vector<shared_ptr<NetInstance::Fabric>> fabNodes(planeId2Node.size(), nullptr);

    // 遍历每一个rankId，每个rankId都增加 peer2net 和 net2peer 两条链路
    for (RankId srcRankId : inRanks) {
        const auto& srcLevelInfo = GetRankLevelInfoByNetLayer(rankTable_->ranks[srcRankId], netLayer);
        const vector<AddressInfo>& addrs = srcLevelInfo.rankAddrs;
        // rankId对应的物理逻辑localId
        LocalId localId = rankGraph_->GetLocalId(srcRankId);
        // 从物理拓扑图中找出 localId在 netLayer 中所有的peer2Net的边。
        std::vector<shared_ptr<PhyTopo::Link>> links = GetPeer2NetPhyLinks(netLayer, localId);
        // 遍历ranktable中的addr，有几个addr就有几条peer2net的边
        for (AddressInfo addrInfo : addrs) {
            if (addrInfo.addr == IpAddress()) {
                continue;
            }

            if (planeId2Node.count(addrInfo.planeId) == 0) {
                continue;
            }
            FabricId fabId = planeId2Node[addrInfo.planeId];
            // 若 fabNodes[fabId] 不存在则创建 如果存在则获取fabNode
            shared_ptr<NetInstance::Fabric> fabNode;
            if (fabNodes[fabId] == nullptr) {
                fabNode = make_shared<NetInstance::Fabric>(fabId, addrInfo.planeId);
                tempNetInsts_[netLayer][netInstId]->AddNode(fabNode);
                fabNodes[fabId] = fabNode;
            } else {
                fabNode = fabNodes[fabId];
            }
            // 插入peer和fabric的peer2net和net2peer两条link
            AddPeer2NetLink(netLayer, netInstId, srcRankId, addrInfo, fabNode, links);
        }
    }

    HCCL_DEBUG(
        "[RankGraphBuilder][AddFabricInfo] netLayer [%u] netInstId[%s] Add Fabric Info success!", netLayer,
        netInstId.c_str());
}

void RankGraphBuilder::AddTopoDescFabricInfo()
{
    // 1. 获取物理拓扑图
    auto phyTopoGraph = PhyTopo::GetInstance()->GetTopoGraph(0);
    if (phyTopoGraph == nullptr) {
        THROW<NullPtrException>(StringFormat("[RankGraphBuilder][AddTopoDescFabricInfo] phyTopoGraph is nullptr"));
    }
    HCCL_INFO("[RankGraphBuilder][AddTopoDescFabricInfo] Successfully retrieved phyTopoGraph");

    // 2. 获取当前 NetInstance
    NetInstance* innerNetInstance = rankGraph_->GetNetInstanceByRankId(0, myRank_);
    if (innerNetInstance == nullptr) {
        THROW<NullPtrException>(
            StringFormat("[RankGraphBuilder][AddTopoDescFabricInfo] rankGraph->GetNetInstanceByRankId is nullptr"));
    }
    std::string netInstId = innerNetInstance->GetNetInstId();
    std::set<RankId> rankIds = innerNetInstance->GetRankIds();

    // 存储所有fabric节点，key为topoInstId
    std::map<u32, std::shared_ptr<NetInstance::Fabric>> fabNodes;

    auto peer = rankGraph_->GetPeer(rankGraph_->GetMyRank());
    auto localDeviceId = peer->GetDeviceId();

    // 3. 遍历所有rank节点，根据topoInstId创建fabric节点
    for (RankId rankId : rankIds) {
        LocalId localId = rankGraph_->GetLocalId(rankId);
        auto peer2netEdges = phyTopoGraph->GetEdges(localId, PhyTopo::Fabric::GetId());

        HCCL_RUN_INFO(
            "[RankGraphBuilder][AddTopoDescFabricInfo] Processing rank %d (localId: %u), found %zu peer2net edges",
            rankId, localId, peer2netEdges.size());

        for (const auto& link : peer2netEdges) {
            u32 topoInstId = link->GetTopoInstId();
            auto topoType = link->GetTopoType();

            // 创建Fabric节点
            if (fabNodes.find(topoInstId) == fabNodes.end()) {
                auto fabNodePtr = std::make_shared<NetInstance::Fabric>(topoInstId);
                innerNetInstance->AddNode(fabNodePtr);
                fabNodes[topoInstId] = fabNodePtr;
                HCCL_INFO(
                    "[RankGraphBuilder][AddTopoDescFabricInfo] Created new Fabric node for topoInstId: %u", topoInstId);
            }

            // 获取 peer 节点
            auto peerNode = peers_.at(rankId);

            // 构造连接接口
            auto peerIfaces = ConstructConnIFromPhyTopoConnIAndPortMap(
                link->GetSourceIFace(), peerNode->GetPortAddrMapLayer0(), topoType, topoInstId, localDeviceId);

            for (const auto& iface : peerIfaces) {
                peerNode->AddConnInterface(0, iface);
            }
            auto fabNodePtr = fabNodes[topoInstId];
            // 构造 peer2netLink 和 net2peerLink（双向）
            for (const auto& iface : peerIfaces) {
                auto peer2netLink = std::make_shared<NetInstance::Link>(
                    peerNode, fabNodePtr, iface, nullptr, LinkType::PEER2NET, link->GetLinkProtocols(),
                    LinkDirection::BOTH, PEER2NET_LINK_HOP);

                auto net2peerLink = std::make_shared<NetInstance::Link>(
                    fabNodePtr, peerNode, nullptr, iface, LinkType::PEER2NET, link->GetLinkProtocols(),
                    LinkDirection::BOTH, PEER2NET_LINK_HOP);

                // 插入 link
                tempNetInsts_[0][netInstId]->AddLink(peer2netLink);
                tempNetInsts_[0][netInstId]->AddLink(net2peerLink);
                tempNetInsts_[0][netInstId]->UpdateTopoInst(topoInstId, topoType, rankId);
                HCCL_RUN_INFO(
                    "[RankGraphBuilder][AddTopoDescFabricInfo] netLayer0 rankId[%u] netInstId[%s] Add Fabric "
                    "Info success!",
                    rankId, netInstId.c_str());
            }
        }
    }
    HCCL_INFO("[RankGraphBuilder][AddTopoDescFabricInfo] Successfully completed fabric link construction");
}

std::map<PlaneId, FabricId> GetFabricsFromAddrInfo(const std::vector<AddressInfo>& rankAddrs)
{
    std::map<PlaneId, FabricId> planeId2FabricId;
    for (const auto& addrInfo : rankAddrs) {
        if (planeId2FabricId.count(addrInfo.planeId) == 0) {
            FabricId fabId = planeId2FabricId.size();
            planeId2FabricId[addrInfo.planeId] = fabId;
        }
    }
    return planeId2FabricId;
}

void RankGraphBuilder::CheckNetLayerFromPhyTopo(const u32 netLayer) const
{
    if (!PhyTopo::GetInstance()->IsNetLayerExisted(netLayer)) {
        THROW<InvalidParamsException>(StringFormat(
            "[RankGraphBuilder][CheckNetLayerFromPhyTopo]"
            "netLayer[%u] not exist in topo.",
            netLayer));
    }
}

// 根据ranktable构造添加peers和NetInstances, NetInstance添加nodes和links(peer2net)
// 1. 创建NetInstance ( 每个NetInstance 添加 Rank， Node， Link)；
// 2. RankGraph中添加NetInstance， Peer， Fabric，
void RankGraphBuilder::BuildFromRankTable()
{
    // 保存NetInstance指针以便后续执行Add操作
    tempNetInsts_.resize(MAX_NET_LAYER); // 为了方便修改RankGraph的NetInstance，共享指针。

    // 遍历rankTable每一个rank, virtualTopo添加Peers
    for (const auto& rankInfo : rankTable_->ranks) {
        updaterFor64Plus1_.SaveReplaceInfo(rankInfo); // 暂存备份替换信息
        RankId rankId = rankInfo.rankId;
        shared_ptr<NetInstance::Peer> peer = make_shared<NetInstance::Peer>(
            rankId, rankInfo.localId, rankInfo.replacedLocalId, rankInfo.deviceId, rankInfo.devicePort,
            rankInfo.hostPort);
        rankGraph_->AddPeer(peer);
        peers_.emplace(rankId, peer); // rankid2peer

        // 构造当前rank的每个LevelInfo所在NetInstance, 添加 RankId 和 Peer
        for (const auto& levelInfo : rankInfo.rankLevelInfos) {
            // 校验netLayer是否在topo中
            CheckNetLayerFromPhyTopo(levelInfo.netLayer);
            // rankLevelInfo.level、id对应NetInstance，若不存在则创建
            auto curNetInstance = GetOrCreateNetInstance(
                levelInfo.netLayer, levelInfo.netInstId, levelInfo.netType, tempNetInsts_, rankGraph_.get());
            if (curNetInstance == nullptr) {
                continue;
            }
            // NetInstance add Peer
            curNetInstance->AddRankId(rankId);
            curNetInstance->AddNode(peer);
            // Peer add NetInstance
            peer->AddNetInstance(curNetInstance);
            if (levelInfo.netLayer == 0) {
                peer->SetPortPortAddrMapLayer0(levelInfo.portAddrMap);
                if (rankId == myRank_) {
                    // 记录本rank的level0是否为无UB兜底层（显式标记字段，见rank_info_detect_client）
                    level0PcieFallback_ = levelInfo.pcieFallback;
                }
            }
            HCCL_DEBUG(
                "[RankGraphBuilder][BuildFromRankTable] rankLevelInfo : rankId[%d] level[%u] "
                "netInstId[%s] fabricType[%s].",
                rankId, levelInfo.netLayer, levelInfo.netInstId.c_str(), levelInfo.netType.Describe().c_str());
        }
    }

    // 对 myrank 所在每个level的NetInstance 添加 Fabrics 和 links(peer2net)
    set<u32> myLevels = rankGraph_->GetLevels(myRank_);
    HCCL_DEBUG("myRank netType: level size %u", myLevels.size());
    for (u32 level : myLevels) {
        if (level == 0) {
            AddTopoDescFabricInfo();
        } else {
            AddFabricInfo(level);
        }
    }

    // 初始化innerRanks
    rankGraph_->InitInnerRanks();

    HCCL_DEBUG("[RankGraphBuilder][BuildFromRankTable] Build VirtualTopo from RankTable success!");
}

void RankGraphBuilder::SetEndpointDesc()
{
    std::shared_ptr<NetInstance::Peer> peer = peers_[myRank_];
    CHK_PRT_THROW(
        peer == nullptr, HCCL_ERROR("[RankGraphBuilder::%s] fail", __func__), NullPtrException, "peer is null");
    // 获取 peer 的 Iface
    std::set<u32> layers = peer->GetLevels();
    for (const auto& layer : layers) {
        auto ifacesVec = peer->GetIfacesByLayer(layer);
        for (const auto& iface : ifacesVec) {
            const auto& protocols = iface->GetLinkProtocols();
            for (const auto& protocol : protocols) {
                EndpointDesc desc{};

                HcclResult ret = GetCommAddr(desc.commAddr, iface->GetAddr());
                CHK_PRT_THROW(
                    ret != HCCL_SUCCESS, HCCL_ERROR("[RankGraphBuilder::%s] fail", __func__), InternalException,
                    "GetCommAddr fail");

                desc.protocol = LinkProtocolToCommProtocol(protocol);
                desc.loc.locType = AddrPositionToEndpointLoc(iface->GetPos());

                HCCL_INFO(
                    "[RankGraphBuilder::SetEndpointDesc] local type[%d] protocol[%d]", desc.loc.locType, desc.protocol);

                peer->SetEndpointToIface(desc.commAddr, desc.protocol, iface);
            }
        }
    }
}

std::shared_ptr<NetInstance> RankGraphBuilder::GetNetInstance(const RankLevelInfo& levelInfo)
{
    auto it = tempNetInsts_[levelInfo.netLayer].find(levelInfo.netInstId);
    if (it == tempNetInsts_[levelInfo.netLayer].end()) {
        return nullptr;
    }
    // 若NetInstance存在, type不一致则报错
    NetType netType = it->second->GetNetType();
    if (netType != levelInfo.netType) {
        HCCL_WARNING(
            "[CreateNetInstance]FabType [%s] and [%s] no match", netType.Describe().c_str(),
            levelInfo.netType.Describe().c_str());
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<NetInstance> RankGraphBuilder::CreateNetInstance(const RankLevelInfo& levelInfo)
{
    std::shared_ptr<NetInstance> netInst;
    if (levelInfo.netType == NetType::TOPO_FILE_DESC) {
        netInst = std::make_shared<InnerNetInstance>(levelInfo.netLayer, levelInfo.netInstId);
    } else if (levelInfo.netType == NetType::CLOS) {
        netInst = std::make_shared<ClosNetInstance>(levelInfo.netLayer, levelInfo.netInstId);
    } else {
        THROW<NotSupportException>(
            StringFormat("[RankGraphBuilder][CreateNetInstance] netType: %s is not support", levelInfo.netType));
    }
    return netInst;
}

// 从phytopo和ranktable中读取数据共同构建peer2peer的边。
void RankGraphBuilder::BuildPeer2PeerLinks()
{
    auto phyTopoGraph = PhyTopo::GetInstance()->GetTopoGraph(0);
    if (phyTopoGraph == nullptr) {
        THROW<NullPtrException>(StringFormat("[RankGraphBuilder][BuildPeer2PeerLinks] phyTopoGraph is nullptr"));
    }
    // 遍历innerNetInstance中的每两个rankId之间是否存在边，存在则添加peer2peerlink
    NetInstance* innerNetInstance = rankGraph_->GetNetInstanceByRankId(0, myRank_);
    if (innerNetInstance == nullptr) {
        THROW<NullPtrException>(StringFormat("[RankGraphBuilder][BuildPeer2PeerLinks] innerNetInstance is nullptr"));
    }
    set<RankId> rankIds = innerNetInstance->GetRankIds();

    auto peer = rankGraph_->GetPeer(rankGraph_->GetMyRank());
    auto localDeviceId = peer->GetDeviceId();
    for (const auto srcRankId : rankIds) {
        for (const auto dstRankId : rankIds) {
            if (srcRankId == dstRankId) {
                continue;
            }

            // 得到phyTopoGraph中对应的localId
            LocalId srcLocalId = rankGraph_->GetLocalId(srcRankId);
            LocalId dstLocalId = rankGraph_->GetLocalId(dstRankId);
            if (srcLocalId == BACKUP_LOCAL_ID || dstLocalId == BACKUP_LOCAL_ID) {
                continue;
            }

            std::vector<shared_ptr<PhyTopo::Link>> phyLinks
                = GetPeer2PeerPhyLinks(phyTopoGraph, srcLocalId, dstLocalId);
            // 根据ports在ranktable找对对应的地址，几个地址就有几条link。

            shared_ptr<NetInstance::Peer> srcPeer = peers_.at(srcRankId);
            shared_ptr<NetInstance::Peer> dstPeer = peers_.at(dstRankId);

            for (shared_ptr<PhyTopo::Link> phyLink : phyLinks) {
                auto sourceIfaces = ConstructConnIFromPhyTopoConnIAndPortMap(
                    phyLink->GetSourceIFace(), srcPeer->GetPortAddrMapLayer0(), phyLink->GetTopoType(),
                    phyLink->GetTopoInstId(), localDeviceId);
                auto targetIfaces = ConstructConnIFromPhyTopoConnIAndPortMap(
                    phyLink->GetTargetIFace(), dstPeer->GetPortAddrMapLayer0(), phyLink->GetTopoType(),
                    phyLink->GetTopoInstId(), localDeviceId);
                if (sourceIfaces.empty() || targetIfaces.empty()) {
                    // 没有可用的接口。
                    HCCL_WARNING(
                        "[RankGraphBuilder][BuildPeer2PeerLinks] no available interface, "
                        "srcRankId[%u] dstRankId[%u].",
                        srcRankId, dstRankId);
                    continue;
                }
                srcPeer->AddConnInterfaces(0, sourceIfaces);
                dstPeer->AddConnInterfaces(0, targetIfaces);
                std::vector<shared_ptr<NetInstance::Link>> links
                    = ConstructLinks(srcPeer, dstPeer, sourceIfaces, targetIfaces, phyLink);
                for (auto link : links) {
                    innerNetInstance->AddLink(link);
                }
            }
        }
    }

    // 无UB场景兜底：ranktable携带兜底level0（pcie_fallback标记，按server划分）时，
    // topo文件中的UB边因无地址无法建链，此处为inner域（同server）内rank两两合成PCIE链路（vnic地址），
    // 使数据面走PCIe P2P传输；跨server流量由上层网络（RoCE等）承载

    if (level0PcieFallback_) {
        BuildPcieFallbackLinks(innerNetInstance);
    }
}

// 无UB场景兜底：inner域内rank两两合成PCIE链路，地址取各卡vnic IP，端口固定d2h
void RankGraphBuilder::BuildPcieFallbackLinks(NetInstance* innerNetInstance)
{
    set<RankId> rankIds = innerNetInstance->GetRankIds();
    auto peer = rankGraph_->GetPeer(rankGraph_->GetMyRank());
    auto localDeviceId = peer->GetDeviceId();

    for (const auto srcRankId : rankIds) {
        for (const auto dstRankId : rankIds) {
            if (srcRankId == dstRankId) {
                continue;
            }
            shared_ptr<NetInstance::Peer> srcPeer = peers_.at(srcRankId);
            shared_ptr<NetInstance::Peer> dstPeer = peers_.at(dstRankId);

            LocalId srcLocalId = rankGraph_->GetLocalId(srcRankId);
            LocalId dstLocalId = rankGraph_->GetLocalId(dstRankId);
            if (srcLocalId == BACKUP_LOCAL_ID || dstLocalId == BACKUP_LOCAL_ID) {
                continue;
            }

            // 从本卡视角查询对端卡的vnic IP，查询失败说明PCIe不可达，跳过该对rank
            IpAddress srcIp;
            IpAddress dstIp;
            bool vnicQueryOk = false;
            try {
                HrtRaSocketGetVnicIpInfos(
                    localDeviceId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, srcPeer->GetDeviceId(), srcIp);
                HrtRaSocketGetVnicIpInfos(
                    localDeviceId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, dstPeer->GetDeviceId(), dstIp);
                vnicQueryOk = true;
            } catch (const HcclException& e) {
                HCCL_ERROR(
                    "[RankGraphBuilder][BuildPcieFallbackLinks] get vnic ip failed, skip pcie fallback link, "
                    "srcRankId[%u] dstRankId[%u], reason[%s].",
                    srcRankId, dstRankId, e.what());
            }
            if (!vnicQueryOk) {
                continue;
            }

            std::set<string> pciePorts = {"d2h"};
            std::set<LinkProtocol> pcieProtocol = {LinkProtocol::PCIE};
            auto srcIface = make_shared<NetInstance::ConnInterface>(
                srcIp, pciePorts, AddrPosition::DEVICE, LinkType::PEER2NET, pcieProtocol, TopoType::MESH_1D, 0);
            auto dstIface = make_shared<NetInstance::ConnInterface>(
                dstIp, pciePorts, AddrPosition::DEVICE, LinkType::PEER2NET, pcieProtocol, TopoType::MESH_1D, 0);

            srcPeer->AddConnInterfaces(0, {srcIface});
            dstPeer->AddConnInterfaces(0, {dstIface});
            auto link = make_shared<NetInstance::Link>(
                srcPeer, dstPeer, srcIface, dstIface, LinkType::PEER2PEER, pcieProtocol);
            innerNetInstance->AddLink(link);
            // 兜底链路统一注册为单个MESH_1D topoInst，供selector层CalcLevel0TopoShape计算level0形状
            innerNetInstance->UpdateTopoInst(0, TopoType::MESH_1D, srcRankId);
            innerNetInstance->UpdateTopoInst(0, TopoType::MESH_1D, dstRankId);
            HCCL_INFO(
                "[RankGraphBuilder][BuildPcieFallbackLinks] add pcie fallback link, srcRankId[%u] dstRankId[%u], "
                "srcIp[%s] dstIp[%s].",
                srcRankId, dstRankId, srcIp.Describe().c_str(), dstIp.Describe().c_str());
        }
    }
}

void RankGraphBuilder::UpdateTopoInstForMyRankOnly()
{
    auto innerNetInstance = rankGraph_->GetNetInstanceByRankId(0, myRank_);
    if (innerNetInstance == nullptr) {
        THROW<NullPtrException>(
            StringFormat("[RankGraphBuilder][UpdateTopoInstForMyRankOnly] innerNetInstance is nullptr"));
    }

    auto netInstId = innerNetInstance->GetNetInstId();
    set<RankId> rankIds = innerNetInstance->GetRankIds();
    auto phyTopoGraph = PhyTopo::GetInstance()->GetTopoGraph(0);
    if (phyTopoGraph == nullptr) {
        THROW<NullPtrException>(
            StringFormat("[RankGraphBuilder][UpdateTopoInstForMyRankOnly] phyTopoGraph is nullptr"));
    }

    // 无UB场景兜底：topo文件中的UB边不生效，topoInst已由BuildPcieFallbackLinks按PCIE全互连注册为
    // 单个MESH_1D实例；此处若继续按phytopo的UB边注册，会得到与兜底level0规模不一致的topoInst，
    // 导致selector层CalcLevel0TopoShape校验失败
    if (level0PcieFallback_) {
        return;
    }

    if (rankIds.size() == 1) {
        // 单卡场景直接返回1DMESH
        RankId singleId = *rankIds.begin();
        tempNetInsts_[0][netInstId]->UpdateTopoInst(0, TopoType::MESH_1D, singleId);
        return;
    }

    for (const auto srcRankId : rankIds) {
        for (const auto dstRankId : rankIds) {
            // 只处理涉及 myRank_ 的边
            if (srcRankId != myRank_ && dstRankId != myRank_) {
                continue;
            }

            LocalId srcLocalId = rankGraph_->GetLocalId(srcRankId);
            LocalId dstLocalId = rankGraph_->GetLocalId(dstRankId);

            std::vector<shared_ptr<PhyTopo::Link>> phyLinks
                = GetPeer2PeerPhyLinks(phyTopoGraph, srcLocalId, dstLocalId);

            for (shared_ptr<PhyTopo::Link> phyLink : phyLinks) {
                u32 topoInstId = phyLink->GetTopoInstId();
                auto topoType = phyLink->GetTopoType();
                tempNetInsts_[0][netInstId]->UpdateTopoInst(topoInstId, topoType, dstRankId);
            }
        }
    }
}

std::vector<std::shared_ptr<NetInstance::ConnInterface>> ConstructConnIFromPhyTopoConnIAndPortMap(
    std::shared_ptr<PhyTopo::ConnInterface> phyConnIFace,
    const std::map<std::string, std::vector<IpAddress>>& portAddrMap, const TopoType topoType, const u32 topoInstId,
    u32 localDeviceId)
{
    std::vector<std::shared_ptr<NetInstance::ConnInterface>> netConnIFaces;
    std::set<string> phyPorts = phyConnIFace->GetPorts();
    std::map<IpAddress, std::set<string>> addr2Ports;
    for (auto port : phyPorts) {
        if (*(phyConnIFace->GetLinkProtocols().begin()) == LinkProtocol::PCIE) {
            IpAddress tempIp;
            HrtRaSocketGetVnicIpInfos(localDeviceId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, localDeviceId, tempIp);
            auto it = addr2Ports.find(tempIp);
            if (it == addr2Ports.end()) {
                std::set<std::string> newPorts;
                newPorts.insert("d2h");
                addr2Ports[tempIp] = newPorts;
            } else {
                it->second.insert("d2h");
            }
        } else {
            auto itPort = portAddrMap.find(port);
            if (itPort == portAddrMap.end()) {
                HCCL_WARNING(
                    "[RankGraphBuilder][ConstructConnIFromPhyTopoConnIAndPortMap] topo uses port [%s] but cannot find "
                    "addrs in "
                    "ranktable.",
                    port.c_str());
                continue;
            }
            for (auto addr : itPort->second) {
                auto it = addr2Ports.find(addr);
                if (it == addr2Ports.end()) {
                    std::set<std::string> newPorts;
                    newPorts.insert(port);
                    addr2Ports[addr] = newPorts;
                } else {
                    it->second.insert(port);
                }
            }
        }
    }

    for (auto it = addr2Ports.begin(); it != addr2Ports.end(); ++it) {
        auto linkType = *(phyConnIFace->GetLinkProtocols().begin()) == LinkProtocol::PCIE ? LinkType::PEER2NET :
                                                                                            LinkType::PEER2PEER;
        shared_ptr<NetInstance::ConnInterface> netConnIFace = make_shared<NetInstance::ConnInterface>(
            it->first, it->second, phyConnIFace->GetPos(), linkType, phyConnIFace->GetLinkProtocols(), topoType,
            topoInstId);
        netConnIFaces.push_back(netConnIFace);
    }
    return netConnIFaces;
}

std::vector<shared_ptr<NetInstance::Link>> ConstructLinks(
    shared_ptr<NetInstance::Peer> srcPeer, shared_ptr<NetInstance::Peer> dstPeer,
    std::vector<std::shared_ptr<NetInstance::ConnInterface>> sourceIfaces,
    std::vector<std::shared_ptr<NetInstance::ConnInterface>> targetIfaces, shared_ptr<PhyTopo::Link> phyLink)
{
    std::vector<shared_ptr<NetInstance::Link>> links;
    for (auto sourceIFace : sourceIfaces) {
        for (auto targetIFace : targetIfaces) {
            shared_ptr<NetInstance::Link> link = make_shared<NetInstance::Link>(
                srcPeer, dstPeer, sourceIFace, targetIFace, LinkType::PEER2PEER, phyLink->GetLinkProtocols());
            links.push_back(link);
        }
    }
    return links;
}

std::vector<std::shared_ptr<PhyTopo::Link>> GetPeer2PeerPhyLinks(
    std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> phyTopoGraph, LocalId srcLocalId, LocalId dstLocalId)
{
    std::vector<shared_ptr<PhyTopo::Link>> links;
    if (!phyTopoGraph->HasNode(srcLocalId) || !phyTopoGraph->HasNode(dstLocalId)) {
        HCCL_WARNING(
            "[RankGraphBuilder][BuildFromPhytopo] srcLocalId[%u] dstLocalId[%u] not exist in phyTopoGraph.", srcLocalId,
            dstLocalId);
        return links;
    }
    // 得到phyTopoGraph对应的NodeId
    NodeId srcNodeId = PhyTopo::Peer::GetId(srcLocalId);
    NodeId dstNodeId = PhyTopo::Peer::GetId(dstLocalId);

    phyTopoGraph->TraverseEdge(srcNodeId, dstNodeId, [&](shared_ptr<PhyTopo::Link> link) {
        if (link != nullptr) {
            links.push_back(link);
        }
    });
    if (links.empty()) {
        HCCL_WARNING(
            "[RankGraphBuilder][GetPeer2PeerPhyLinks] srcLocalId[%u] dstLocalId[%u] edge does not exist.", srcLocalId,
            dstLocalId);
    }
    return links;
}

void RankGraphBuilder::CheckMyRankInRankTable() const
{
    if (myRank_ >= static_cast<s32>(rankTable_->rankCount)) {
        THROW<InvalidParamsException>(StringFormat(
            "[RankGraphBuilder][CheckMyRankInRankTable]"
            "myRank[%d] is not in rankTable rankCount[%u].",
            myRank_, rankTable_->rankCount));
    }
}

void RankGraphBuilder::BuildRankGraph()
{
    // 创建VirtualTopo
    rankGraph_ = make_unique<RankGraph>(myRank_);

    // 校验myRank在rankTable中
    CheckMyRankInRankTable();

    // 根据ranktable构造添加peers和NetInstances, 每个NetInstance添加nodes和links(peer2net)
    BuildFromRankTable();

    // 根据phytopo构造添加InnerGroup中的links(peer2peer), 不包括备份节点
    BuildPeer2PeerLinks();

    // 使用备份D时需要修改虚拟拓扑
    updaterFor64Plus1_.UpdateRankGraph(rankGraph_.get(), rankTable_.get());

    // 为myrank的peer2peer更新topoInst
    UpdateTopoInstForMyRankOnly();

    // 添加绕路 绕路获取
    DetourService::GetInstance().InsertDetourLinks(rankGraph_.get(), rankTable_.get());

    // 设置endpoint
    SetEndpointDesc();

    // 构造完成
    rankGraph_->InitFinish();
}

std::unique_ptr<RankTableInfo> RankGraphBuilder::GetRankTableInfo() { return move(rankTable_); }

std::shared_ptr<TopoInfo> RankGraphBuilder::GetTopoInfo() { return topoInfo_; }

unique_ptr<RankGraph>
RankGraphBuilder::RecoverBuild(const RankTableInfo& rankTableInfo, const TopoInfo& topoInfo, RankId myRank)
{
    topoInfo_ = std::make_shared<TopoInfo>(topoInfo);
    PhyTopoBuilder::GetInstance().RecoverBuild(*topoInfo_);

    rankTable_ = make_unique<RankTableInfo>(rankTableInfo);
    HCCL_INFO(
        "[%s] RankTable[%s] RankTableInfo[%s]", __func__, rankTable_->Describe().c_str(),
        rankTableInfo.Describe().c_str());

    this->myRank_ = myRank;
    BuildRankGraph();

    HCCL_INFO("[RankGraphBuilder] Build VirtualTopo success!");
    rankGraph_->Dump();
    return std::move(rankGraph_);
}

} // namespace Hccl
