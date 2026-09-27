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
#include "rank_gph.h"
#include "rank_graph_builder.h"
#include "phy_topo_builder.h"
#include "detour_service.h"

using namespace Hccl;

class RankGraphTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        if (setenv("HCCL_DETOUR", "detour:1", 1) == -1) {
            perror("setenv");
        }

        // 获取环境变量
        char* env_var = getenv("HCCL_DETOUR");
        if (env_var != nullptr) {
            std::cout << "HCCL_DETOUR: " << env_var << std::endl;
        } else {
            std::cout << "HCCL_DETOUR is not set" << std::endl;
        }

        std::cout << "VirtTopoBuilderTest SetUP" << std::endl;
    }

    static void TearDownTestCase()
    {
        if (unsetenv("HCCL_DETOUR") == -1) {
            perror("unsetenv");
        }
        char* env_var = getenv("HCCL_DETOUR");
        if (env_var != nullptr) {
            std::cout << "HCCL_DETOUR: " << env_var << std::endl;
        } else {
            std::cout << "HCCL_DETOUR has been successfully unset" << std::endl;
        }

        std::cout << "VirtTopoBuilderTest TearDown" << std::endl;
    }

    virtual void SetUp() { std::cout << "A Test case in RankGraphTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RankGraphTest TearDown" << std::endl;
    }
    RankId myRank = 0;
};

std::shared_ptr<NetInstance::Peer> createPeer(int rankId = 0, int localId = 0, DeviceId deviceId = 0)
{
    return std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
}

TEST_F(RankGraphTest, st_AddPeer_When_Normal_Expect_SUCCESS)
{
    RankGraph rankGraph(myRank);
    auto peer = createPeer();
    EXPECT_NO_THROW(rankGraph.AddPeer(peer));
    EXPECT_EQ(false, rankGraph.HasRank(3));
    EXPECT_EQ(true, rankGraph.HasRank(peer->GetRankId()));
}

TEST_F(RankGraphTest, st_AddPeer_When_NullPeer_Expect_NullPtrException)
{
    RankGraph rankGraph(myRank);
    EXPECT_THROW(rankGraph.AddPeer(nullptr), NullPtrException);
}

TEST_F(RankGraphTest, st_AddPeer_When_Init_Finshed_Expect_InternalException)
{
    RankGraph rankGraph(myRank);
    rankGraph.InitFinish();
    auto peer = createPeer();
    EXPECT_THROW(rankGraph.AddPeer(peer), InternalException);
}

TEST_F(RankGraphTest, st_AddNetInstance_When_Normal_Expect_SUCCESS)
{
    s32 netLayer = 0;
    string netInstId = "test";
    RankGraph rankGraph(myRank);
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    EXPECT_NO_THROW(rankGraph.AddNetInstance(netInstance));
    auto netInstResp = rankGraph.GetNetInstanceByNetInstId(netLayer, netInstId);
    EXPECT_NE(nullptr, netInstResp);
}

TEST_F(RankGraphTest, st_AddNetInstance_When_NullPeer_Expect_NullPtrException)
{
    RankGraph rankGraph(myRank);
    EXPECT_THROW(rankGraph.AddNetInstance(nullptr), NullPtrException);
}

TEST_F(RankGraphTest, st_AddNetInstance_When_Init_Finshed_Expect_InternalException)
{
    RankGraph rankGraph(myRank);
    rankGraph.InitFinish();
    auto netInstance = std::make_shared<InnerNetInstance>(0, "test");
    EXPECT_THROW(rankGraph.AddNetInstance(netInstance), InternalException);
}

TEST_F(RankGraphTest, st_AddNetInstance_When_Not_Include_NetInstance_RankId_Expect_InvalidParamsException)
{
    RankGraph rankGraph(myRank);
    auto netInstance = std::make_shared<InnerNetInstance>(0, "test");
    netInstance->AddRankId(1);
    EXPECT_THROW(rankGraph.AddNetInstance(netInstance), InvalidParamsException);
}

TEST_F(RankGraphTest, st_InitInnerRanks_When_NullPeer_Expect_NullPtrException)
{
    RankGraph rankGraph(myRank);
    EXPECT_THROW(rankGraph.InitInnerRanks(), NullPtrException);
}

TEST_F(RankGraphTest, st_InitInnerRanks_When_Normal_Expect_SUCCESS)
{
    RankGraph rankGraph(myRank);
    std::shared_ptr<NetInstance> netInstance = std::make_shared<InnerNetInstance>(0, "test");
    auto peer = createPeer(myRank, 0, 0);
    peer->AddNetInstance(netInstance);
    rankGraph.AddNetInstance(netInstance);
    rankGraph.AddPeer(peer);
    EXPECT_NO_THROW(rankGraph.InitInnerRanks());
}

std::shared_ptr<RankGraph> create4pRankGraph(RankId myRank)
{
    RankGraph rankGraph(myRank);
    std::shared_ptr<NetInstance> netInstLayer0_1 = std::make_shared<InnerNetInstance>(0, "layer0_1");
    std::shared_ptr<NetInstance> netInstLayer0_2 = std::make_shared<InnerNetInstance>(0, "layer0_2");
    std::shared_ptr<NetInstance> netInstLayer1_1 = std::make_shared<ClosNetInstance>(1, "layer1_1");
    std::shared_ptr<NetInstance> netInstLayer1_2 = std::make_shared<ClosNetInstance>(1, "layer1_2");
    std::shared_ptr<NetInstance> netInstLayer2 = std::make_shared<ClosNetInstance>(2, "layer2");
    auto peer0 = createPeer(myRank, 0, 0);
    auto peer1 = createPeer(1, 1, 1);
    auto peer2 = createPeer(2, 2, 2);
    auto peer3 = createPeer(3, 3, 3);
    auto peer4 = createPeer(4, 4, 4);
    peer0->AddNetInstance(netInstLayer0_1);
    peer0->AddNetInstance(netInstLayer1_1);
    peer0->AddNetInstance(netInstLayer2);
    peer1->AddNetInstance(netInstLayer0_1);
    peer1->AddNetInstance(netInstLayer1_1);
    peer2->AddNetInstance(netInstLayer0_2);
    peer2->AddNetInstance(netInstLayer1_1);
    peer2->AddNetInstance(netInstLayer2);
    peer3->AddNetInstance(netInstLayer0_2);
    peer4->AddNetInstance(netInstLayer1_2);

    netInstLayer0_1->AddRankId(peer0->GetRankId());
    netInstLayer0_1->AddRankId(peer1->GetRankId());
    netInstLayer0_2->AddRankId(peer2->GetRankId());
    netInstLayer0_2->AddRankId(peer3->GetRankId());
    netInstLayer1_1->AddRankId(peer0->GetRankId());
    netInstLayer1_1->AddRankId(peer1->GetRankId());
    netInstLayer1_1->AddRankId(peer2->GetRankId());
    netInstLayer2->AddRankId(peer0->GetRankId());
    netInstLayer2->AddRankId(peer2->GetRankId());
    netInstLayer1_2->AddRankId(peer4->GetRankId());

    rankGraph.AddPeer(peer0);
    rankGraph.AddPeer(peer1);
    rankGraph.AddPeer(peer2);
    rankGraph.AddPeer(peer3);
    rankGraph.AddPeer(peer4);

    rankGraph.AddNetInstance(netInstLayer0_1);
    rankGraph.AddNetInstance(netInstLayer0_2);
    rankGraph.AddNetInstance(netInstLayer1_1);
    rankGraph.AddNetInstance(netInstLayer1_2);
    rankGraph.AddNetInstance(netInstLayer2);
    return std::make_shared<RankGraph>(rankGraph);
}

TEST_F(RankGraphTest, st_GetLayerRanks_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    EXPECT_EQ(4, rankGraph->GetLayerRanks(0));
    EXPECT_EQ(4, rankGraph->GetLayerRanks(1));
    EXPECT_EQ(2, rankGraph->GetLayerRanks(2));
}

TEST_F(RankGraphTest, st_GetLocalInstRanks_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    std::vector<u32> rankListLayer0;
    std::vector<u32> rankListLayer1;
    std::vector<u32> rankListLayer2;
    u32 rankNumLayer0;
    u32 rankNumLayer1;
    u32 rankNumLayer2;
    rankGraph->GetLocalInstRanks(0, rankListLayer0, rankNumLayer0);
    rankGraph->GetLocalInstRanks(1, rankListLayer1, rankNumLayer1);
    rankGraph->GetLocalInstRanks(2, rankListLayer2, rankNumLayer2);

    vector<u32> expectedRankList0 = {0, 1};
    vector<u32> expectedRankList1 = {0, 1, 2};
    vector<u32> expectedRankList2 = {0, 2};

    EXPECT_EQ(expectedRankList0.size(), rankNumLayer0);
    for (size_t i = 0; i < rankListLayer0.size(); ++i) {
        EXPECT_EQ(rankListLayer0[i], rankListLayer0[i]);
    }

    EXPECT_EQ(expectedRankList1.size(), rankNumLayer1);
    for (size_t i = 0; i < rankListLayer1.size(); ++i) {
        EXPECT_EQ(rankListLayer1[i], rankListLayer1[i]);
    }

    EXPECT_EQ(expectedRankList2.size(), rankNumLayer2);
    for (size_t i = 0; i < rankListLayer2.size(); ++i) {
        EXPECT_EQ(rankListLayer2[i], rankListLayer2[i]);
    }
}

TEST_F(RankGraphTest, st_GetLocalInstSize_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    EXPECT_EQ(2, rankGraph->GetLocalInstSize(0));
    EXPECT_EQ(3, rankGraph->GetLocalInstSize(1));
    EXPECT_EQ(2, rankGraph->GetLocalInstSize(2));
}

TEST_F(RankGraphTest, st_GetNetType_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    EXPECT_EQ(NetType::TOPO_FILE_DESC, rankGraph->GetNetType(0));
    EXPECT_EQ(NetType::CLOS, rankGraph->GetNetType(1));
    EXPECT_EQ(NetType::CLOS, rankGraph->GetNetType(2));
}

TEST_F(RankGraphTest, st_GetNetInstanceList_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    std::vector<u32> instSizeListLayer0;
    std::vector<u32> instSizeListLayer1;
    std::vector<u32> instSizeListLayer2;
    u32 listSizeLayer0;
    u32 listSizeLayer1;
    u32 listSizeLayer2;
    rankGraph->GetNetInstanceList(0, instSizeListLayer0, listSizeLayer0);
    rankGraph->GetNetInstanceList(1, instSizeListLayer1, listSizeLayer1);
    rankGraph->GetNetInstanceList(2, instSizeListLayer2, listSizeLayer2);

    vector<u32> expectedRankList0 = {2, 2};
    vector<u32> expectedRankList1 = {3, 1};
    vector<u32> expectedRankList2 = {2};

    EXPECT_EQ(expectedRankList0.size(), listSizeLayer0);
    for (size_t i = 0; i < instSizeListLayer0.size(); ++i) {
        EXPECT_EQ(instSizeListLayer0[i], instSizeListLayer0[i]);
    }

    EXPECT_EQ(expectedRankList1.size(), listSizeLayer1);
    for (size_t i = 0; i < instSizeListLayer1.size(); ++i) {
        EXPECT_EQ(instSizeListLayer1[i], instSizeListLayer1[i]);
    }

    EXPECT_EQ(expectedRankList2.size(), listSizeLayer2);
    for (size_t i = 0; i < instSizeListLayer2.size(); ++i) {
        EXPECT_EQ(instSizeListLayer2[i], instSizeListLayer2[i]);
    }
}

TEST_F(RankGraphTest, st_IsSymmetric_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    EXPECT_EQ(true, rankGraph->IsSymmetric(0));
    EXPECT_EQ(false, rankGraph->IsSymmetric(1));
    EXPECT_EQ(true, rankGraph->IsSymmetric(2));
    EXPECT_THROW(rankGraph->IsSymmetric(3), NullPtrException);
}

TEST_F(RankGraphTest, st_CreateSubRankGraph_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    vector<u32> subRankIds = {0, 2};
    std::unique_ptr<RankGraph> subRankGraph = rankGraph->CreateSubRankGraph(subRankIds);
    subRankGraph->Dump();
    EXPECT_EQ(1, subRankGraph->GetLocalInstSize(0));
    EXPECT_EQ(2, subRankGraph->GetLocalInstSize(1));
    EXPECT_EQ(2, subRankGraph->GetLocalInstSize(2));
}
