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
#include "graph.h"
#include "phy_topo.h"
#include "topo_common_types.h"
#include <set>

using namespace Hccl;

class PhyTopoConnInterfaceTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(PhyTopoConnInterfaceTest, St_PhyTopoConnInterface_When_ValidParameter_Expect_ConstructSuccess)
{
    std::set<std::string> ports = {"0/1"};
    AddrPosition pos = AddrPosition::DEVICE;
    LinkType linkType = LinkType::PEER2PEER;
    std::set<LinkProtocol> linkProtocol = {LinkProtocol::UB_CTP};
    PhyTopo::ConnInterface conn(ports, pos, linkType, linkProtocol);
    EXPECT_EQ(conn.GetPorts(), ports);
    EXPECT_EQ(conn.GetPos(), pos);
    EXPECT_EQ(conn.GetLinkType(), linkType);
    EXPECT_EQ(conn.GetLinkProtocols(), linkProtocol);
}

TEST_F(PhyTopoConnInterfaceTest, St_PhyTopoConnInterfaceOperatorEqual_When_PortsDisorder_Expect_ReturnTrue)
{
    std::set<std::string> ports1 = {"0/1", "0/2"};
    std::set<std::string> ports2 = {"0/2", "0/1"};
    AddrPosition pos = AddrPosition::DEVICE;
    LinkType linkType = LinkType::PEER2PEER;
    std::set<LinkProtocol> linkProtocol = {LinkProtocol::UB_CTP};
    PhyTopo::ConnInterface conn(ports1, pos, linkType, linkProtocol);
    PhyTopo::ConnInterface conn1(ports2, pos, linkType, linkProtocol);
    EXPECT_EQ(conn == conn1, true);
}

TEST_F(PhyTopoConnInterfaceTest, St_PhyTopoConnInterfaceOperatorNotEqual_When_ParametersNotSame_Expect_ReturnTrue)
{
    std::set<std::string> ports1 = {"0/1", "0/2"};
    std::set<std::string> ports2 = {"0/3", "0/1"};
    AddrPosition pos = AddrPosition::DEVICE;
    LinkType linkType = LinkType::PEER2PEER;
    std::set<LinkProtocol> linkProtocol = {LinkProtocol::UB_CTP};
    PhyTopo::ConnInterface conn(ports1, pos, linkType, linkProtocol);
    PhyTopo::ConnInterface conn1(ports2, pos, linkType, linkProtocol);
    EXPECT_EQ(conn != conn1, true);
}
class PhyTopoNodeTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(PhyTopoNodeTest, St_PhyTopoNode_When_ValidParameter_Expect_ReturnIsNodeType)
{
    PhyTopo::Node::NodeType nodeType = PhyTopo::Node::NodeType::PEER;
    PhyTopo::Node node(nodeType);
    EXPECT_EQ(node.GetType(), nodeType);
}

class PhyTopoPeerTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(PhyTopoPeerTest, St_PhyTopoNode_When_ValidParameter_Expect_ReturnIsPeer)
{
    LocalId localId = 0;
    PhyTopo::Peer peer(localId);
    EXPECT_EQ(peer.GetLocalId(), localId);
    EXPECT_EQ(peer.GetType(), PhyTopo::Node::NodeType::PEER);
}

TEST_F(PhyTopoPeerTest, St_PhyTopoNode_When_ValidParameter_Expect_ReturnIsLocalId)
{
    LocalId localId = 0;
    PhyTopo::Peer peer(localId);
    EXPECT_EQ(peer.GetLocalId(), localId);
    EXPECT_EQ(peer.GetType(), PhyTopo::Node::NodeType::PEER);
}

TEST_F(PhyTopoPeerTest, St_PhyTopoNode_When_ValidParameter_Expect_ReturnIsInputLocalId)
{
    LocalId localId = 0;
    NodeId nodeId = 0;
    EXPECT_EQ(PhyTopo::Peer::GetId(localId), nodeId);
}

class PhyTopoFabricTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(PhyTopoFabricTest, St_PhyTopoNode_When_ValidParameter_Expect_ReturnIsNodeType)
{
    PhyTopo::Fabric fabric;
    EXPECT_EQ(fabric.GetType(), PhyTopo::Node::NodeType::FABRIC);
}

TEST_F(PhyTopoFabricTest, St_PhyTopoNode_When_ValidParameter_Expect_ReturnIsNodeId)
{
    NodeId nodeId = 0;
    nodeId |= (1ULL << 32);
    EXPECT_EQ(PhyTopo::Fabric::GetId(), nodeId);
}

class PhyTopoLinkTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(PhyTopoLinkTest, St_PhyTopoLink_When_ValidParameter_Expect_ConstructSuccess)
{
    std::shared_ptr<PhyTopo::Node> sourceNode = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    std::shared_ptr<PhyTopo::Node> targetNode = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    PhyTopo::LinkAttributes linkAttrs;
    linkAttrs.linktype = LinkType::PEER2PEER;
    linkAttrs.protocols = {LinkProtocol::UB_CTP};
    TopoType topoType = TopoType::CLOS;
    u32 topoInstId = 0;
    PhyTopo::Link link(sourceNode, targetNode, linkAttrs, topoType, topoInstId);

    EXPECT_EQ(link.GetType(), linkAttrs.linktype);
    EXPECT_EQ(link.GetLinkProtocols(), linkAttrs.protocols);
    EXPECT_EQ(link.GetSourceNode(), sourceNode);
    EXPECT_EQ(link.GetTargetNode(), targetNode);
    EXPECT_EQ(link.GetTopoType(), topoType);
    EXPECT_EQ(link.GetTopoInstId(), topoInstId);
}

TEST_F(PhyTopoLinkTest, St_PhyTopoLink_When_ValidInterface_Expect_ReturnIsSourceIface)
{
    std::shared_ptr<PhyTopo::Node> sourceNode = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    std::shared_ptr<PhyTopo::Node> targetNode = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    PhyTopo::LinkAttributes linkAttrs;
    linkAttrs.linktype = LinkType::PEER2PEER;
    linkAttrs.protocols = {LinkProtocol::UB_CTP};
    TopoType topoType = TopoType::CLOS;
    u32 topoInstId = 0;
    PhyTopo::Link link(sourceNode, targetNode, linkAttrs, topoType, topoInstId);

    std::set<std::string> ports = {"0/1"};
    AddrPosition pos = AddrPosition::DEVICE;
    std::shared_ptr<PhyTopo::ConnInterface> sourceIface
        = std::make_shared<PhyTopo::ConnInterface>(ports, pos, linkAttrs.linktype, linkAttrs.protocols);
    link.SetTargetIface(sourceIface);
    EXPECT_EQ(link.GetTargetIFace(), sourceIface);
}

TEST_F(PhyTopoLinkTest, St_PhyTopoLink_When_ValidInterface_Expect_ReturnIsTargetIface)
{
    std::shared_ptr<PhyTopo::Node> sourceNode = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    std::shared_ptr<PhyTopo::Node> targetNode = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    PhyTopo::LinkAttributes linkAttrs;
    linkAttrs.linktype = LinkType::PEER2PEER;
    linkAttrs.protocols = {LinkProtocol::UB_CTP};
    TopoType topoType = TopoType::CLOS;
    u32 topoInstId = 0;
    PhyTopo::Link link(sourceNode, targetNode, linkAttrs, topoType, topoInstId);
    std::set<std::string> ports = {"0/1"};
    AddrPosition pos = AddrPosition::DEVICE;
    std::shared_ptr<PhyTopo::ConnInterface> targetIface
        = std::make_shared<PhyTopo::ConnInterface>(ports, pos, linkAttrs.linktype, linkAttrs.protocols);
    link.SetTargetIface(targetIface);
    EXPECT_EQ(link.GetTargetIFace(), targetIface);
}

TEST_F(PhyTopoLinkTest, St_PhyTopoLink_When_LinkDirectionSet_Expect_ReturnIsBOTH)
{
    std::shared_ptr<PhyTopo::Node> sourceNode = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    std::shared_ptr<PhyTopo::Node> targetNode = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    PhyTopo::LinkAttributes linkAttrs;
    linkAttrs.linktype = LinkType::PEER2PEER;
    linkAttrs.protocols = {LinkProtocol::UB_CTP};
    TopoType topoType = TopoType::CLOS;
    u32 topoInstId = 0;
    PhyTopo::Link link(sourceNode, targetNode, linkAttrs, topoType, topoInstId);
    EXPECT_EQ(link.GetLinkDirection(), LinkDirection::BOTH);
}

TEST_F(PhyTopoLinkTest, St_PhyTopoLink_When_HopSet_Expect_ReturnIs1)
{
    std::shared_ptr<PhyTopo::Node> sourceNode = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    std::shared_ptr<PhyTopo::Node> targetNode = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    PhyTopo::LinkAttributes linkAttrs;
    linkAttrs.linktype = LinkType::PEER2PEER;
    linkAttrs.protocols = {LinkProtocol::UB_CTP};
    TopoType topoType = TopoType::CLOS;
    u32 topoInstId = 0;
    PhyTopo::Link link(sourceNode, targetNode, linkAttrs, topoType, topoInstId);
    EXPECT_EQ(link.GetHop(), 1);
}

class PhyTopoTest : public ::testing::Test {
protected:
    PhyTopo phyTopo;
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(PhyTopoTest, St_AddTopoGraph_When_TopoNotNullAndNetLayerNotExist_Expect_ReturnTopo)
{
    std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> topo = std::make_shared<Graph<PhyTopo::Node, PhyTopo::Link>>();
    u32 netLayer = 1;
    phyTopo.AddTopoGraph(netLayer, topo);
    EXPECT_EQ(phyTopo.GetTopoGraph(netLayer), topo);
}

TEST_F(PhyTopoTest, St_GetTopoGraph_When_NetLayerExist_Expect_ReturnTopo)
{
    std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> topo1
        = std::make_shared<Graph<PhyTopo::Node, PhyTopo::Link>>();
    std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> topo2
        = std::make_shared<Graph<PhyTopo::Node, PhyTopo::Link>>();
    u32 netLayer1 = 1;
    u32 netLayer2 = 2;
    phyTopo.AddTopoGraph(netLayer1, topo1);
    phyTopo.AddTopoGraph(netLayer2, topo2);
    EXPECT_EQ(phyTopo.GetTopoGraph(netLayer1), topo1);
    EXPECT_EQ(phyTopo.GetTopoGraph(netLayer2), topo2);
}

TEST_F(PhyTopoTest, St_GetTopoGraph_When_NetLayerNoExist_Expect_ReturnNullptr)
{
    std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> topo1
        = std::make_shared<Graph<PhyTopo::Node, PhyTopo::Link>>();
    u32 netLayer1 = 1;
    u32 netLayer2 = 2;
    phyTopo.AddTopoGraph(netLayer1, topo1);
    EXPECT_EQ(phyTopo.GetTopoGraph(netLayer2), nullptr);
}
