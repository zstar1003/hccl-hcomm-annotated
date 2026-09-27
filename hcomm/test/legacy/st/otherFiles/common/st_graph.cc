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

using namespace Hccl;

class GraphTest : public ::testing::Test {
protected:
    Graph<PhyTopo::Node, PhyTopo::Link>* graph;

    void SetUp() override { graph = new Graph<PhyTopo::Node, PhyTopo::Link>(); }

    void TearDown() override
    {
        delete graph;
        graph = nullptr;
    }
};

TEST_F(GraphTest, St_HasNode_When_NodeAlreadyExist_Expect_ReturnTrue)
{
    NodeId nodeId = 1;
    std::shared_ptr<PhyTopo::Node> node = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    graph->AddNode(nodeId, node);
    EXPECT_TRUE(graph->HasNode(nodeId));
}

TEST_F(GraphTest, St_HasNode_When_NodeNotExist_Expect_ReturnTrue)
{
    NodeId nodeId = 1;
    NodeId testNodeId = 2;
    std::shared_ptr<PhyTopo::Node> node = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    graph->AddNode(nodeId, node);
    EXPECT_TRUE(graph->HasNode(nodeId));
    EXPECT_FALSE(graph->HasNode(testNodeId));
}

TEST_F(GraphTest, St_AddNode_When_NodeIsNotExist_Expect_Success)
{
    NodeId nodeId1 = 1;
    NodeId nodeId2 = 2;
    NodeId nodeId3 = 3;

    std::shared_ptr<PhyTopo::Node> node1 = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    std::shared_ptr<PhyTopo::Node> node2 = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    std::shared_ptr<PhyTopo::Node> node3 = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::FABRIC);

    graph->AddNode(nodeId1, node1);
    graph->AddNode(nodeId2, node2);
    graph->AddNode(nodeId3, node3);

    int fabricNodeNum = 0;
    int peerNodeNum = 0;
    int nodeNum = 0;

    graph->TraverseNode([&](const std::shared_ptr<PhyTopo::Node>& node) {
        ++nodeNum;
        if (node->GetType() == PhyTopo::Node::NodeType::FABRIC) {
            ++fabricNodeNum;
        } else {
            ++peerNodeNum;
        }
    });

    EXPECT_EQ(nodeNum, 3);
    EXPECT_EQ(peerNodeNum, 2);
    EXPECT_EQ(fabricNodeNum, 1);
}

TEST_F(GraphTest, St_AddEdge_When_BothNodesExistAndEdgeIsNotNull_Expect_Success)
{
    NodeId nodeId1 = 1;
    NodeId nodeId2 = 2;
    NodeId nodeId3 = 3;

    std::shared_ptr<PhyTopo::Node> node1 = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    std::shared_ptr<PhyTopo::Node> node2 = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    std::shared_ptr<PhyTopo::Node> node3 = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::FABRIC);

    PhyTopo::LinkAttributes linkAttrs;
    linkAttrs.linktype = LinkType::PEER2PEER;
    linkAttrs.protocols = {LinkProtocol::UB_CTP};

    TopoType topoType = TopoType::CLOS;
    u32 topoInstId = 0;

    std::shared_ptr<PhyTopo::Link> edge1
        = std::make_shared<PhyTopo::Link>(node1, node2, linkAttrs, topoType, topoInstId);
    std::shared_ptr<PhyTopo::Link> edge2
        = std::make_shared<PhyTopo::Link>(node1, node3, linkAttrs, topoType, topoInstId);
    std::shared_ptr<PhyTopo::Link> edge3
        = std::make_shared<PhyTopo::Link>(node2, node3, linkAttrs, topoType, topoInstId);

    graph->AddNode(nodeId1, node1);
    graph->AddNode(nodeId2, node2);
    graph->AddNode(nodeId3, node3);
    graph->AddEdge(nodeId1, nodeId2, edge1);
    graph->AddEdge(nodeId1, nodeId3, edge2);
    graph->AddEdge(nodeId2, nodeId3, edge3);

    int fabricNodeNum = 0;
    int peerNodeNum = 0;
    int edgeNum = 0;

    graph->TraverseEdge(nodeId1, [&](const std::shared_ptr<PhyTopo::Link>& edge) {
        ++edgeNum;
        if (edge->GetTargetNode()->GetType() == PhyTopo::Node::NodeType::FABRIC) {
            ++fabricNodeNum;
        } else {
            ++peerNodeNum;
        }

        if (edge->GetSourceNode()->GetType() == PhyTopo::Node::NodeType::PEER) {
            ++peerNodeNum;
        }
    });

    EXPECT_EQ(edgeNum, 2);
    EXPECT_EQ(peerNodeNum, 3);
    EXPECT_EQ(fabricNodeNum, 1);

    fabricNodeNum = 0;
    peerNodeNum = 0;
    edgeNum = 0;

    graph->TraverseEdge(nodeId1, nodeId2, [&](const std::shared_ptr<PhyTopo::Link>& edge) {
        ++edgeNum;
        if (edge->GetTargetNode()->GetType() == PhyTopo::Node::NodeType::FABRIC) {
            ++fabricNodeNum;
        } else {
            ++peerNodeNum;
        }

        if (edge->GetSourceNode()->GetType() == PhyTopo::Node::NodeType::PEER) {
            ++peerNodeNum;
        }
    });

    EXPECT_EQ(edgeNum, 1);
    EXPECT_EQ(peerNodeNum, 2);
    EXPECT_EQ(fabricNodeNum, 0);

    fabricNodeNum = 0;
    peerNodeNum = 0;
    edgeNum = 0;

    graph->TraverseEdge(nodeId1, nodeId3, [&](const std::shared_ptr<PhyTopo::Link>& edge) {
        ++edgeNum;
        if (edge->GetTargetNode()->GetType() == PhyTopo::Node::NodeType::FABRIC) {
            ++fabricNodeNum;
        } else {
            ++peerNodeNum;
        }

        if (edge->GetSourceNode()->GetType() == PhyTopo::Node::NodeType::PEER) {
            ++peerNodeNum;
        }
    });

    EXPECT_EQ(edgeNum, 1);
    EXPECT_EQ(peerNodeNum, 1);
    EXPECT_EQ(fabricNodeNum, 1);
}

TEST_F(GraphTest, St_TraverseNode_When_NodeExist_Expect_Success)
{
    NodeId nodeId1 = 1;
    NodeId nodeId2 = 2;
    NodeId nodeId3 = 3;

    std::shared_ptr<PhyTopo::Node> node1 = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    std::shared_ptr<PhyTopo::Node> node2 = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    std::shared_ptr<PhyTopo::Node> node3 = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::FABRIC);

    graph->AddNode(nodeId1, node1);
    graph->AddNode(nodeId2, node2);
    graph->AddNode(nodeId3, node3);

    int fabricNodeNum = 0;
    int peerNodeNum = 0;
    int nodeNum = 0;

    graph->TraverseNode([&](const std::shared_ptr<PhyTopo::Node>& node) {
        ++nodeNum;
        if (node->GetType() == PhyTopo::Node::NodeType::FABRIC) {
            ++fabricNodeNum;
        } else {
            ++peerNodeNum;
        }
    });

    EXPECT_EQ(nodeNum, 3);
    EXPECT_EQ(peerNodeNum, 2);
    EXPECT_EQ(fabricNodeNum, 1);
}

TEST_F(GraphTest, St_TraverseEdge_When_EdgeExist_Expect_Success)
{
    NodeId nodeId1 = 1;
    NodeId nodeId2 = 2;
    NodeId nodeId3 = 3;

    std::shared_ptr<PhyTopo::Node> node1 = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    std::shared_ptr<PhyTopo::Node> node2 = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::PEER);
    std::shared_ptr<PhyTopo::Node> node3 = std::make_shared<PhyTopo::Node>(PhyTopo::Node::NodeType::FABRIC);
    PhyTopo::LinkAttributes linkAttrs;
    linkAttrs.linktype = LinkType::PEER2PEER;
    linkAttrs.protocols = {LinkProtocol::UB_CTP};
    TopoType topoType = TopoType::CLOS;
    u32 topoInstId = 0;

    std::shared_ptr<PhyTopo::Link> edge1
        = std::make_shared<PhyTopo::Link>(node1, node2, linkAttrs, topoType, topoInstId);
    std::shared_ptr<PhyTopo::Link> edge2
        = std::make_shared<PhyTopo::Link>(node1, node3, linkAttrs, topoType, topoInstId);
    std::shared_ptr<PhyTopo::Link> edge3
        = std::make_shared<PhyTopo::Link>(node2, node3, linkAttrs, topoType, topoInstId);

    graph->AddNode(nodeId1, node1);
    graph->AddNode(nodeId2, node2);
    graph->AddNode(nodeId3, node3);
    graph->AddEdge(nodeId1, nodeId2, edge1);
    graph->AddEdge(nodeId1, nodeId3, edge2);
    graph->AddEdge(nodeId2, nodeId3, edge3);

    int fabricNodeNum = 0;
    int peerNodeNum = 0;
    int edgeNum = 0;

    graph->TraverseEdge(nodeId1, [&](const std::shared_ptr<PhyTopo::Link>& edge) {
        ++edgeNum;
        if (edge->GetTargetNode()->GetType() == PhyTopo::Node::NodeType::FABRIC) {
            ++fabricNodeNum;
        } else {
            ++peerNodeNum;
        }

        if (edge->GetSourceNode()->GetType() == PhyTopo::Node::NodeType::PEER) {
            ++peerNodeNum;
        }
    });

    EXPECT_EQ(edgeNum, 2);
    EXPECT_EQ(peerNodeNum, 3);
    EXPECT_EQ(fabricNodeNum, 1);

    fabricNodeNum = 0;
    peerNodeNum = 0;
    edgeNum = 0;

    graph->TraverseEdge(nodeId1, nodeId2, [&](const std::shared_ptr<PhyTopo::Link>& edge) {
        ++edgeNum;
        if (edge->GetTargetNode()->GetType() == PhyTopo::Node::NodeType::FABRIC) {
            ++fabricNodeNum;
        } else {
            ++peerNodeNum;
        }

        if (edge->GetSourceNode()->GetType() == PhyTopo::Node::NodeType::PEER) {
            ++peerNodeNum;
        }
    });

    EXPECT_EQ(edgeNum, 1);
    EXPECT_EQ(peerNodeNum, 2);
    EXPECT_EQ(fabricNodeNum, 0);

    fabricNodeNum = 0;
    peerNodeNum = 0;
    edgeNum = 0;

    graph->TraverseEdge(nodeId1, nodeId3, [&](const std::shared_ptr<PhyTopo::Link>& edge) {
        ++edgeNum;
        if (edge->GetTargetNode()->GetType() == PhyTopo::Node::NodeType::FABRIC) {
            ++fabricNodeNum;
        } else {
            ++peerNodeNum;
        }

        if (edge->GetSourceNode()->GetType() == PhyTopo::Node::NodeType::PEER) {
            ++peerNodeNum;
        }
    });

    EXPECT_EQ(edgeNum, 1);
    EXPECT_EQ(peerNodeNum, 1);
    EXPECT_EQ(fabricNodeNum, 1);
}
