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
#include <iostream>
#include <unistd.h>

#define private public
#define protected public

#include "detour_service.h"
#include "phy_topo.h"
#include "phy_topo_builder.h"
#include "rank_graph_test_data_builder.h"
#include "rank_gph.h"
#include "rank_graph_builder.h"

#undef private
#undef protected

using namespace Hccl;

class RankGraphBuilderTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankGraphBuilderTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RankGraphBuilderTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in RankGraphBuilderTest SetUP" << std::endl;
        GlobalMockObject::reset();
        PhyTopo::GetInstance()->Clear();
    }

    virtual void TearDown()
    {
        PhyTopo::GetInstance()->Clear();
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in RankGraphBuilderTest TearDown" << std::endl;
    }
};

TEST_F(RankGraphBuilderTest, Ut_Build_When_Normal_Expect_Success)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    TopoInfo topoInfo = test::MakeTwoPeerClosTopo();
    std::unique_ptr<RankGraph> rankGraph = rankGraphBuilder.RecoverBuild(test::MakeRankTable2p(), topoInfo, 0);
    EXPECT_NE(nullptr, rankGraph);
    ASSERT_NE(nullptr, rankGraphBuilder.topoInfo_);
    EXPECT_EQ(topoInfo.Describe(), rankGraphBuilder.topoInfo_->Describe());
    auto path1 = rankGraph->GetPaths(0, 0, 1);
    EXPECT_NE(1, path1.size());
    auto path2 = rankGraph->GetPaths(0, 1, 0);
    EXPECT_NE(1, path2.size());
}

TEST_F(RankGraphBuilderTest, Ut_BuildRankGraph_When_Normal_Expect_Success)
{
    // when
    MOCKER_CPP(&RankGraphBuilder::BuildFromRankTable).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RankGraph::InitInnerRanks).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::BuildPeer2PeerLinks).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&DetourService::InsertDetourLinks).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::UpdateTopoInstForMyRankOnly).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::SetEndpointDesc).stubs();
    // then
    RankGraphBuilder rankGraphBuilder;
    Hccl::RankTableInfo tmpRankTable;
    tmpRankTable.rankCount = 3;
    rankGraphBuilder.rankTable_ = make_unique<Hccl::RankTableInfo>(tmpRankTable);
    rankGraphBuilder.myRank_ = 0;
    EXPECT_NO_THROW(rankGraphBuilder.BuildRankGraph());
}

TEST_F(RankGraphBuilderTest, Ut_Build_When_1pRankTable_Expect_Success)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(test::MakeRankTable1p(), test::MakeOnePeerTopo(true), 0);
    EXPECT_NE(nullptr, rankGraph);
    auto path1 = rankGraph->GetPaths(0, 0, 1);
    EXPECT_EQ(0, path1.size());
    auto path2 = rankGraph->GetPaths(0, 1, 0);
    EXPECT_EQ(0, path2.size());
    vector<u32> subRankIds = {0};
    auto subRankGraph = rankGraph->CreateSubRankGraph(subRankIds);
    EXPECT_EQ(1, subRankGraph->GetInnerRankSize());
}

TEST_F(RankGraphBuilderTest, Ut_Build_When_OnePTopoFileWithoutEdge_Expect_Success)
{
    PhyTopo::GetInstance()->Clear();
    auto graph = std::make_shared<Graph<PhyTopo::Node, PhyTopo::Link>>();
    graph->AddNode(PhyTopo::Peer::GetId(0), std::make_shared<PhyTopo::Peer>(0));
    PhyTopo::GetInstance()->AddTopoGraph(0, graph);
    PhyTopo::GetInstance()->InitFinish();
    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(test::MakeRankTable1p(), test::MakeOnePeerTopo(false), 0);
    EXPECT_NE(nullptr, rankGraph);
    auto rankSize = rankGraph->GetRankSize();
    EXPECT_EQ(rankSize, 1);
    auto peer = rankGraph->GetPeer(rankGraph->GetMyRank());
    ASSERT_NE(peer, nullptr);
    EXPECT_EQ(peer->GetLocalId(), 0);
}

TEST_F(RankGraphBuilderTest, Ut_BuildFromRankTable_When_NetLayerInconsistent_Expect_InvalidParamsException)
{
    // 校验BuildFromRankTable的Add(RankId, Peer)
    // when
    MOCKER_CPP(&PhyTopoBuilder::Build).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&RankGraph::InitInnerRanks).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&DetourService::InsertDetourLinks).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::BuildPeer2PeerLinks).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&RankGraphBuilder::AddFabricInfo).stubs().will(ignoreReturnValue());
    // then
    RankGraphBuilder rankGraphBuilder;
    RankTableInfo rankTable = test::MakeRankTable2p();
    rankTable.ranks[0].rankLevelInfos.emplace_back(test::MakeRankLevel(
        2, "missing-layer", NetType::CLOS,
        {
            test::MakeAddress("192.168.200.1", {"0/1"}),
        }));
    EXPECT_THROW(rankGraphBuilder.Build(rankTable, "topo.json", 0), InvalidParamsException);
}

TEST_F(RankGraphBuilderTest, ut_Build_When_4pRankTable_Expect_Success)
{
    PhyTopo::GetInstance()->Clear();
    RankGraphBuilder rankGraphBuilder;
    std::unique_ptr<RankGraph> rankGraph
        = rankGraphBuilder.RecoverBuild(test::MakeRankTable4pForBuilder(), test::MakeFourPeerBuilderTopo(), 0);
    EXPECT_NE(nullptr, rankGraph);
    std::vector<std::string> netIds = {"az0-rack0", "az0", "all"};
    for (s32 rankId = 0; rankId < 3; rankId++) {
        for (u32 netLayer = 0; netLayer < 3; netLayer++) {
            const NetInstance* fabGroup = rankGraph->GetNetInstanceByRankId(netLayer, rankId);
            EXPECT_EQ(netIds[netLayer], fabGroup->GetNetInstId());

            EXPECT_EQ(true, fabGroup->HasNode(NetInstance::Peer(rankId, 0, 0, 0).GetLocalId()));
            if (netLayer == 0) {
                EXPECT_EQ(NetType::TOPO_FILE_DESC, fabGroup->GetNetType());
            } else if (netLayer == 1) {
                EXPECT_EQ(NetType::CLOS, fabGroup->GetNetType());
            } else if (netLayer == 2) {
                EXPECT_EQ(NetType::CLOS, fabGroup->GetNetType());
            }
        }
    }

    std::vector<NetInstance::Path> pathsLayer0 = rankGraph->GetPaths(0, 0, 1);
    EXPECT_EQ(1, pathsLayer0.size());
    // GetPaths检查边0 - 1的level1的边 peer0->net0、net0->peer1、peer0->net1、net1->peer1
    std::vector<NetInstance::Path> pathsLayer1 = rankGraph->GetPaths(1, 0, 1);
    std::vector<std::string> ips1 = {"IpAddress[AF=v4, addr=192.168.101.1]", "IpAddress[AF=v4, addr=124.112.1.1]"};
    std::vector<std::string> ips2 = {"IpAddress[AF=v4, addr=192.168.101.11]", "IpAddress[AF=v4, addr=124.112.1.4]"};
    EXPECT_EQ(1, pathsLayer1.size());
    EXPECT_EQ(2, pathsLayer1[0].links.size());

    EXPECT_EQ(2, pathsLayer1[0].links[0].GetHop());
    EXPECT_EQ(LinkType::PEER2NET, pathsLayer1[0].links[0].GetType());
    EXPECT_EQ(std::set<Hccl::LinkProtocol>{LinkProtocol::UB_CTP}, pathsLayer1[0].links[0].GetLinkProtocols());
    EXPECT_EQ(0, pathsLayer1[0].links[0].GetSourceNode()->GetNodeId());
    EXPECT_EQ(4294967296, pathsLayer1[0].links[0].GetTargetNode()->GetNodeId());
    EXPECT_EQ(htonl(0xC0A86501), pathsLayer1[0].links[0].GetSourceIface()->GetAddr().GetBinaryAddress().addr.s_addr);

    EXPECT_EQ(2, pathsLayer1[0].links[1].GetHop());
    EXPECT_EQ(LinkType::PEER2NET, pathsLayer1[0].links[1].GetType());
    EXPECT_EQ(std::set<Hccl::LinkProtocol>{LinkProtocol::UB_CTP}, pathsLayer1[0].links[0].GetLinkProtocols());
    EXPECT_EQ(4294967296, pathsLayer1[0].links[1].GetSourceNode()->GetNodeId());
    EXPECT_EQ(1, pathsLayer1[0].links[1].GetTargetNode()->GetNodeId());
    EXPECT_EQ(htonl(0xC0A8650B), pathsLayer1[0].links[1].GetTargetIface()->GetAddr().GetBinaryAddress().addr.s_addr);

    std::vector<NetInstance::Path> pathsLayer2 = rankGraph->GetPaths(2, 2, 3);
    EXPECT_EQ(1, pathsLayer1.size());
    EXPECT_EQ(2, pathsLayer1[0].links.size());
}

TEST_F(RankGraphBuilderTest, Ut_RankGraphBuilderRecoverBuild_When_Invalid_Expect_InvalidParamsException)
{
    RankTableInfo rankTableInfo;
    TopoInfo topoInfo;
    RankGraphBuilder rankGraphBuilder;
    EXPECT_THROW(rankGraphBuilder.RecoverBuild(rankTableInfo, topoInfo, 0), InvalidParamsException);
}
