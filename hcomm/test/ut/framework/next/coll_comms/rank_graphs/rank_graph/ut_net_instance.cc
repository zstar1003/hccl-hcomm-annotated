/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <utility>
#include <set>
#include "net_instance.h"
#include "not_support_exception.h"
#include "invalid_params_exception.h"

using namespace Hccl;

class NetInstanceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NetInstanceTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "NetInstanceTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in NetInstanceTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in NetInstanceTest TearDown" << std::endl;
    }
};

TEST_F(NetInstanceTest, ut_NetInstance_Node_When_Normal_Expect_SUCCESS)
{
    s32 rankId = 0;
    s32 localId = 0;
    u32 groupLevel = 0;
    DeviceId deviceId = 0;
    constexpr u32 devPort = 60001;
    string netInstId = "test";
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/0"};
    std::set<LinkProtocol> protocals = {LinkProtocol::UB_CTP, LinkProtocol::UB_TP};
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocals);
    std::shared_ptr<NetInstance::Node> node
        = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId, devPort);
    node->AddConnInterface(0, connInterface);
    node->AddConnInterface(0, connInterface);
}

TEST_F(NetInstanceTest, FabGroup_Peer_test)
{
    s32 rankId = 0;
    s32 localId = 0;
    u32 groupLevel = 0;
    DeviceId deviceId = 0;
    constexpr u32 devPort = 60001;
    string netInstId = "test";
    std::shared_ptr<NetInstance::Peer> peer
        = std::make_shared<NetInstance::Peer>(rankId, localId, localId, localId, devPort);
    EXPECT_EQ(nullptr, peer->GetNetInstance(1));
    EXPECT_EQ(nullptr, peer->GetNetInstance(groupLevel));
    std::shared_ptr<NetInstance> fabGroup = std::make_shared<InnerNetInstance>(groupLevel, netInstId);
    peer->AddNetInstance(fabGroup);
    std::shared_ptr<NetInstance> fabGroup1 = std::make_shared<InnerNetInstance>(groupLevel, netInstId);
    EXPECT_THROW(peer->AddNetInstance(fabGroup1), InvalidParamsException);
    EXPECT_NE(nullptr, peer->GetNetInstance(groupLevel));
    peer->GetType();
    EXPECT_EQ(NetInstance::Peer::NodeType::PEER, peer->GetType());
    EXPECT_EQ(0, peer->GetNodeId());
    EXPECT_EQ(0, peer->GetLocalId());
    EXPECT_EQ(0, peer->GetRankId());
    peer->GetLevels();
    EXPECT_EQ(0, peer->GetNodeId());
    peer->Describe();
}

TEST_F(NetInstanceTest, FabGroup_Fabric_test)
{
    s32 fabricId = 0;
    PlaneId planeId = "planaA";
    std::shared_ptr<NetInstance::Fabric> fabric = std::make_shared<NetInstance::Fabric>(fabricId, planeId);
    EXPECT_EQ(planeId, fabric->GetPlaneId());
    EXPECT_EQ(0x100000000, fabric->GetNodeId());
    fabric->Describe();
}

TEST_F(NetInstanceTest, FabGroup_Link_test)
{
    s32 rankId = 0;
    s32 localId = 0;
    s32 rankId1 = 1;
    s32 localId1 = 1;
    u64 srcNodeId = 0;
    u64 dstNodeId = 1;
    constexpr u32 devPort = 60001;
    constexpr u32 devPort1 = 60002;
    shared_ptr<NetInstance::Node> source
        = std::make_shared<NetInstance::Peer>(rankId, localId, localId, localId, devPort);
    shared_ptr<NetInstance::Node> target
        = std::make_shared<NetInstance::Peer>(rankId1, localId1, localId1, localId1, devPort1);
    IpAddress inputAddr(0);
    std::set<LinkProtocol> protocals = {LinkProtocol::UB_CTP};
    std::set<std::string> ports = {"0/0"};
    shared_ptr<NetInstance::ConnInterface> sourceIface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocals);
    shared_ptr<NetInstance::ConnInterface> targetIface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::DEVICE, LinkType::PEER2PEER, protocals);
    shared_ptr<NetInstance::Link> link
        = std::make_shared<NetInstance::Link>(source, target, sourceIface, targetIface, LinkType::PEER2PEER, protocals);
    EXPECT_EQ(protocals, link->GetLinkProtocols());
    EXPECT_EQ(LinkDirection::BOTH, link->GetLinkDirection());
    EXPECT_EQ(LinkType::PEER2PEER, link->GetType());
    EXPECT_EQ(1, link->GetHop());
    EXPECT_NE(nullptr, link->GetSourceNode());
    EXPECT_NE(nullptr, link->GetTargetNode());
    EXPECT_NE(nullptr, link->GetSourceIface());
    EXPECT_NE(nullptr, link->GetTargetIface());
    link->Describe();

    shared_ptr<NetInstance::Link> link1
        = std::make_shared<NetInstance::Link>(source, target, sourceIface, targetIface, LinkType::PEER2PEER, protocals);
    EXPECT_EQ(false, link == link1);
    EXPECT_EQ(true, link != link1);
}

TEST_F(NetInstanceTest, fabGroup_inner_test)
{
    u32 level = 0;
    std::string netInstId = "InnerGroup";
    NetType fabType = NetType::TOPO_FILE_DESC;

    InnerNetInstance fabGroup = InnerNetInstance(level, netInstId);

    EXPECT_EQ(fabGroup.GetNetLayer(), level);
    EXPECT_EQ(fabGroup.GetNetInstId(), netInstId);
    EXPECT_EQ(fabGroup.GetNetType(), fabType);
}

TEST_F(NetInstanceTest, fabGroup_clos_test)
{
    u32 level = 1;
    std::string netInstId = "ClosGroup";
    NetType fabType = NetType::CLOS;

    ClosNetInstance fabGroup = ClosNetInstance(level, netInstId);

    EXPECT_EQ(fabGroup.GetNetLayer(), level);
    EXPECT_EQ(fabGroup.GetNetInstId(), netInstId);
    EXPECT_EQ(fabGroup.GetNetType(), fabType);
}

TEST_F(NetInstanceTest, fabGroup_add_rankId_test)
{
    u32 level = 0;
    std::string netInstId = "InnerGroup";
    RankId rankId = 27;

    InnerNetInstance fabGroup = InnerNetInstance(level, netInstId);

    fabGroup.AddRankId(rankId);
    EXPECT_EQ(fabGroup.GetRankIds().size(), 1);
    EXPECT_EQ(*fabGroup.GetRankIds().cbegin(), rankId);
}

TEST_F(NetInstanceTest, fabGroup_add_peer_test)
{
    u32 level = 0;
    std::string netInstId = "InnerGroup";
    RankId rankId = 27;
    LocalId localId = 27;
    DeviceId deviceId = 27;
    constexpr u32 devPort = 60001;

    NetInstance::Peer peer = NetInstance::Peer(rankId, localId, localId, deviceId, devPort);
    std::shared_ptr<NetInstance::Peer> peerPtr = std::make_shared<NetInstance::Peer>(peer);

    InnerNetInstance fabGroup = InnerNetInstance(level, netInstId);

    fabGroup.AddRankId(rankId);
    fabGroup.AddNode(peerPtr);

    NodeId nodeId = peer.GetNodeId();
    EXPECT_EQ(fabGroup.HasNode(nodeId), true);
}

TEST_F(NetInstanceTest, fabGroup_add_fabric_test_v1)
{
    u32 level = 1;
    std::string netInstId = "ClosGroup";

    s32 fabricId = 1;
    PlaneId planeId = "planaA";
    NetInstance::Fabric fabric = NetInstance::Fabric(fabricId, planeId);
    std::shared_ptr<NetInstance::Fabric> fabricPtr = std::make_shared<NetInstance::Fabric>(fabric);
    ClosNetInstance fabGroup = ClosNetInstance(level, netInstId);
    fabGroup.AddNode(fabricPtr);
    NodeId nodeId = fabric.GetNodeId();
    EXPECT_EQ(fabGroup.HasNode(nodeId), true);
}

TEST_F(NetInstanceTest, fabGroup_add_fabric_test_v2)
{
    u32 level = 0;
    std::string netInstId = "InnerGroup";
    RankId rankId = 27;
    LocalId localId = 27;
    s32 fabricId = 1;
    PlaneId planeId = "planaA";
    NetInstance::Fabric fabric = NetInstance::Fabric(fabricId, planeId);
    std::shared_ptr<NetInstance::Fabric> fabricPtr = std::make_shared<NetInstance::Fabric>(fabric);

    std::shared_ptr<InnerNetInstance> fabGroup = std::make_shared<InnerNetInstance>(level, netInstId);
    EXPECT_NE(fabGroup, nullptr);
}

NetInstance::Link
InitBaseLink(std::shared_ptr<NetInstance::Node> srcNodePtr, std::shared_ptr<NetInstance::Node> dstNodePtr, u32 hop = 1)
{
    IpAddress srcAddr = IpAddress(0);
    IpAddress dstAddr = IpAddress(0);
    AddrPosition addrPos = AddrPosition::DEVICE;
    LinkType linkType = LinkType::PEER2PEER;
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    LinkDirection direction = LinkDirection::BOTH;
    std::set<std::string> ports = {"0/0"};

    NetInstance::ConnInterface srcIf = NetInstance::ConnInterface(srcAddr, ports, addrPos, linkType, protocols);

    NetInstance::ConnInterface dstIf = NetInstance::ConnInterface(dstAddr, ports, addrPos, linkType, protocols);

    NetInstance::Link link = NetInstance::Link(
        srcNodePtr, dstNodePtr, std::make_shared<NetInstance::ConnInterface>(srcIf),
        std::make_shared<NetInstance::ConnInterface>(dstIf), linkType, protocols, direction, hop);

    return link;
}

TEST_F(NetInstanceTest, fabGroup_add_link_test_v1)
{
    u32 level = 0;
    std::string netInstId = "InnerGroup";

    RankId srcRankId = 27;
    LocalId srcLocalId = 27;
    RankId dstRankId = 53;
    LocalId dstLocalId = 53;
    constexpr u32 devPort = 60001;

    NetInstance::Peer srcPeer = NetInstance::Peer(srcRankId, srcLocalId, srcLocalId, srcLocalId, devPort);
    NetInstance::Peer dstPeer = NetInstance::Peer(dstRankId, dstLocalId, dstLocalId, srcLocalId, devPort);

    std::shared_ptr<NetInstance::Peer> srcPeerPtr = std::make_shared<NetInstance::Peer>(srcPeer);
    std::shared_ptr<NetInstance::Peer> dstPeerPtr = std::make_shared<NetInstance::Peer>(dstPeer);
    NetInstance::Link link = InitBaseLink(srcPeerPtr, dstPeerPtr);
    InnerNetInstance fabGroup = InnerNetInstance(level, netInstId);

    fabGroup.AddRankId(srcRankId);
    fabGroup.AddRankId(dstRankId);

    fabGroup.AddNode(srcPeerPtr);
    fabGroup.AddNode(dstPeerPtr);

    NodeId srcNodeId = srcPeerPtr->GetNodeId();
    NodeId dstNodeId = dstPeerPtr->GetNodeId();
    EXPECT_EQ(fabGroup.HasNode(srcNodeId), true);
    EXPECT_EQ(fabGroup.HasNode(dstNodeId), true);

    std::shared_ptr<NetInstance::Link> linkPtr = std::make_shared<NetInstance::Link>(link);
    fabGroup.AddLink(linkPtr);

    EXPECT_EQ(fabGroup.GetPaths(srcRankId, dstRankId).size(), 1);
    EXPECT_EQ(fabGroup.GetPaths(dstRankId, srcRankId).size(), 0);
}

TEST_F(NetInstanceTest, fabGroup_add_link_test_v2)
{
    u32 level = 0;
    std::string netInstId = "InnerGroup";

    RankId srcRankId = 27;
    LocalId srcLocalId = 27;
    RankId dstRankId = 53;
    LocalId dstLocalId = 53;
    constexpr u32 devPort = 60001;

    NetInstance::Peer srcPeer = NetInstance::Peer(srcRankId, srcLocalId, srcLocalId, srcLocalId, devPort);
    NetInstance::Peer dstPeer = NetInstance::Peer(dstRankId, dstLocalId, dstLocalId, dstLocalId, devPort);

    std::shared_ptr<NetInstance::Peer> srcPeerPtr = std::make_shared<NetInstance::Peer>(srcPeer);
    std::shared_ptr<NetInstance::Peer> dstPeerPtr = std::make_shared<NetInstance::Peer>(dstPeer);

    NetInstance::Link link = InitBaseLink(srcPeerPtr, dstPeerPtr);
    std::shared_ptr<NetInstance::Link> linkPtr = std::make_shared<NetInstance::Link>(link);

    InnerNetInstance fabGroup = InnerNetInstance(level, netInstId);

    fabGroup.AddRankId(srcRankId);
    fabGroup.AddRankId(dstRankId);

    fabGroup.AddNode(srcPeerPtr);
    fabGroup.AddNode(dstPeerPtr);

    NodeId srcNodeId = srcPeerPtr->GetNodeId();
    NodeId dstNodeId = dstPeerPtr->GetNodeId();

    EXPECT_EQ(fabGroup.HasNode(srcNodeId), true);
    EXPECT_EQ(fabGroup.HasNode(dstNodeId), true);

    fabGroup.AddLink(linkPtr);

    EXPECT_EQ(fabGroup.GetPaths(srcRankId, dstRankId).size(), 1);
    EXPECT_EQ(fabGroup.GetPaths(dstRankId, srcRankId).size(), 0);

    std::shared_ptr<NetInstance::Link> sameLinkPtr = std::make_shared<NetInstance::Link>(link);
    fabGroup.AddLink(sameLinkPtr);

    EXPECT_EQ(fabGroup.GetPaths(srcRankId, dstRankId).size(), 1);
    EXPECT_EQ(fabGroup.GetPaths(dstRankId, srcRankId).size(), 0);
}

std::vector<std::tuple<RankId, RankId, NodeId, NodeId, NetInstance::Link>>
InitInnerLinks(std::vector<std::tuple<RankId, NodeId, NetInstance::Peer>>& peers)
{
    std::vector<std::tuple<RankId, RankId, NodeId, NodeId, NetInstance::Link>> links;

    for (int i = 0; i < peers.size(); i++) {
        for (int j = 0; j < peers.size(); j++) {
            if (i == j)
                continue;

            RankId srcRankId = get<0>(peers[i]);
            RankId dstRankId = get<0>(peers[j]);

            NodeId srcPeerId = get<1>(peers[i]);
            NodeId dstPeerId = get<1>(peers[j]);

            NetInstance::Link link = InitBaseLink(
                std::make_shared<NetInstance::Peer>(get<2>(peers[i])),
                std::make_shared<NetInstance::Peer>(get<2>(peers[j])));

            links.push_back(std::make_tuple(srcRankId, dstRankId, srcPeerId, dstPeerId, link));
        }
    }

    return links;
}

std::vector<std::tuple<RankId, RankId, NodeId, NodeId, NetInstance::Link>> InitClosLinks(
    std::vector<std::tuple<RankId, NodeId, NetInstance::Peer>>& peers,
    std::vector<std::pair<NodeId, NetInstance::Fabric>>& fabrics)
{
    std::vector<std::tuple<RankId, RankId, NodeId, NodeId, NetInstance::Link>> links;

    for (int i = 0; i < peers.size(); i++) {
        RankId rankId = get<0>(peers[i]);
        NodeId peerId = get<1>(peers[i]);

        for (int index = 0; index < fabrics.size() / 2; index++) {
            NodeId fabricNodeId = fabrics[index].first;

            NetInstance::Link fromLink = InitBaseLink(
                std::make_shared<NetInstance::Peer>(get<2>(peers[i])),
                std::make_shared<NetInstance::Fabric>(fabrics[index].second));

            NetInstance::Link toLink = InitBaseLink(
                std::make_shared<NetInstance::Fabric>(fabrics[index].second),
                std::make_shared<NetInstance::Peer>(get<2>(peers[i])));

            links.push_back(std::make_tuple(rankId, rankId, peerId, fabricNodeId, fromLink));
            links.push_back(std::make_tuple(rankId, rankId, fabricNodeId, peerId, toLink));
        }
    }

    return links;
}

std::unique_ptr<NetInstance> InitFullFabGroup(
    u32 level, std::string netInstId, NetType fabType, std::vector<RankId> rankIds, int fabricSize, int fabricIdBase)
{
    std::vector<std::tuple<RankId, NodeId, NetInstance::Peer>> peers;
    for (auto& rankId : rankIds) {
        LocalId localId = static_cast<LocalId>(rankId);
        NetInstance::Peer peer = NetInstance::Peer(rankId, localId, localId, localId, 60001);
        NodeId peerId = peer.GetNodeId();

        peers.push_back(std::make_tuple(rankId, peerId, peer));
    }

    std::vector<std::pair<NodeId, NetInstance::Fabric>> fabrics;
    for (int i = 0; i < fabricSize; i++) {
        FabricId fabricId = fabricIdBase + i;
        NetInstance::Fabric fabric = NetInstance::Fabric(fabricId, "planeA");
        NodeId nodeId = fabric.GetNodeId();
        fabrics.push_back(std::make_pair(nodeId, fabric));
    }

    std::vector<std::tuple<RankId, RankId, NodeId, NodeId, NetInstance::Link>> links;
    if (fabType == NetType::CLOS) {
        links = InitClosLinks(peers, fabrics);
    } else if (fabType == NetType::TOPO_FILE_DESC) {
        links = InitInnerLinks(peers);
    } else {
        return nullptr;
    }

    std::unique_ptr<NetInstance> fabGroupPtr = nullptr;
    if (fabType == NetType::CLOS) {
        fabGroupPtr.reset(new ClosNetInstance(level, netInstId));
    } else if (fabType == NetType::TOPO_FILE_DESC) {
        fabGroupPtr.reset(new InnerNetInstance(level, netInstId));
    } else {
        return nullptr;
    }

    for (auto& rankId : rankIds) {
        fabGroupPtr->AddRankId(rankId);
    }

    for (auto& peer : peers) {
        fabGroupPtr->AddNode(std::make_shared<NetInstance::Peer>(get<2>(peer)));
    }

    for (auto& fabric : fabrics) {
        fabGroupPtr->AddNode(std::make_shared<NetInstance::Fabric>(fabric.second));
    }

    for (auto& link : links) {
        fabGroupPtr->AddLink(std::make_shared<NetInstance::Link>(std::get<4>(link)));
    }

    std::cout << fabGroupPtr->Describe() << std::endl;

    return std::move(fabGroupPtr);
}

std::pair<std::shared_ptr<NetInstance::Node>, std::shared_ptr<NetInstance::Node>> GetBothNodes(NetInstance::Link link)
{
    u64 srcNodeId = 0;
    u64 dstNodeId = 0;
    std::shared_ptr<NetInstance::Node> srcNode = link.GetSourceNode();
    std::shared_ptr<NetInstance::Node> dstNode = link.GetTargetNode();

    return std::make_pair(srcNode, dstNode);
}

TEST_F(NetInstanceTest, fabGroup_inner_get_paths_v1)
{
    u32 level = 0;
    std::string netInstId = "InnerGroup";
    NetType fabType = NetType::TOPO_FILE_DESC;
    std::vector<RankId> rankIds = {0, 1, 2, 3, 4, 5, 6};
    RankId wrongRankId = 27;

    std::unique_ptr<NetInstance> fabGroupPtr = InitFullFabGroup(level, netInstId, fabType, rankIds, 0, 0);

    EXPECT_NE(fabGroupPtr, nullptr);

    for (int i = 0; i < rankIds.size(); i++) {
        for (int j = 0; j < rankIds.size(); j++) {
            if (i == j)
                continue;
            RankId srcRankId = rankIds[i];
            RankId dstRankId = rankIds[j];

            vector<NetInstance::Path> paths = fabGroupPtr->GetPaths(srcRankId, dstRankId);
            EXPECT_EQ(paths.size(), 1);
            EXPECT_EQ(paths[0].links.size(), 1);

            auto testNodeId = GetBothNodes(paths[0].links[0]);

            EXPECT_EQ(std::dynamic_pointer_cast<NetInstance::Peer>(testNodeId.first)->GetRankId(), srcRankId);
            EXPECT_EQ(std::dynamic_pointer_cast<NetInstance::Peer>(testNodeId.second)->GetRankId(), dstRankId);
        }
    }
}

TEST_F(NetInstanceTest, fabGroup_clos_get_paths_v1)
{
    u32 level = 1;
    std::string netInstId = "ClosGroup";
    NetType fabType = NetType::CLOS;
    std::vector<RankId> rankIds = {0, 1, 2, 3, 4, 5, 6};
    RankId wrongRankId = 27;
    int fabricSize = 4;
    int fabricIdBase = 10;

    std::unique_ptr<NetInstance> fabGroupPtr
        = InitFullFabGroup(level, netInstId, fabType, rankIds, fabricSize, fabricIdBase);

    EXPECT_NE(fabGroupPtr, nullptr);

    for (int i = 0; i < rankIds.size(); i++) {
        for (int j = 0; j < rankIds.size(); j++) {
            if (i == j)
                continue;
            RankId srcRankId = rankIds[i];
            RankId dstRankId = rankIds[j];

            vector<NetInstance::Path> paths = fabGroupPtr->GetPaths(srcRankId, dstRankId);
            EXPECT_EQ(paths.size(), fabricSize / 2);

            for (auto path : paths) {
                EXPECT_EQ(path.links.size(), 2);

                auto testFromNodes = GetBothNodes(path.links[0]);
                auto testToNodes = GetBothNodes(path.links[1]);

                RankId testSrcRankId = std::dynamic_pointer_cast<NetInstance::Peer>(testFromNodes.first)->GetRankId();
                NodeId testFromfabircId = (testFromNodes.second)->GetNodeId();
                NodeId testToFabricId = (testToNodes.first)->GetNodeId();
                RankId testDstRankId = std::dynamic_pointer_cast<NetInstance::Peer>(testToNodes.second)->GetRankId();

                EXPECT_EQ(testSrcRankId, srcRankId);
                EXPECT_EQ(testDstRankId, dstRankId);
                EXPECT_EQ(testFromfabircId, testToFabricId);
            }
        }
    }
}

TEST_F(NetInstanceTest, UT_GetIfacesByLayer_When_Valid_Return_HCCl_SUCCESS)
{
    s32 rankId = 0;
    s32 localId = 0;
    u32 groupLevel = 0;
    DeviceId deviceId = 0;
    constexpr u32 devPort = 60001;
    string netInstId = "test";
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/0"};
    std::set<LinkProtocol> protocals = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocals);
    EXPECT_EQ(connInterface->GetTopoType(), TopoType::CLOS);
    EXPECT_EQ(connInterface->GetTopoInstId(), 0);
    std::shared_ptr<NetInstance::Node> node
        = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId, devPort);
    node->AddConnInterface(0, connInterface);

    auto ifaces = node->GetIfacesByLayer(0);
    EXPECT_EQ(ifaces.size(), 1);
    EXPECT_EQ(ifaces[0], connInterface);
}
