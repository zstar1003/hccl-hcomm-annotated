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
#include "ranktable_builder.h"
#include "internal_exception.h"

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

// **单个测试用例命名**：`Ut_<API名称>_When_<测试条件>_Expect_<预期行为>`（大驼峰命名）
// 例如：`ut_HcclGetCommName_When_Normal_Expect_ReturnIsHCCL_SUCCESS` 用例名应该要按照这个格式吧

std::shared_ptr<NetInstance::Peer> createPeer(int rankId = 0, int localId = 0, DeviceId deviceId = 0)
{
    return std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
}

TEST_F(RankGraphTest, ut_AddPeer_When_Normal_Expect_SUCCESS)
{
    RankGraph rankGraph(myRank);
    auto peer = createPeer();
    EXPECT_NO_THROW(rankGraph.AddPeer(peer));
    EXPECT_EQ(false, rankGraph.HasRank(3));
    EXPECT_EQ(true, rankGraph.HasRank(peer->GetRankId()));
}

TEST_F(RankGraphTest, ut_AddPeer_When_NullPeer_Expect_NullPtrException)
{
    RankGraph rankGraph(myRank);
    EXPECT_THROW(rankGraph.AddPeer(nullptr), NullPtrException);
}

TEST_F(RankGraphTest, ut_AddPeer_When_Init_Finshed_Expect_InternalException)
{
    RankGraph rankGraph(myRank);
    rankGraph.InitFinish();
    auto peer = createPeer();
    EXPECT_THROW(rankGraph.AddPeer(peer), InternalException);
}

TEST_F(RankGraphTest, ut_AddNetInstance_When_Normal_Expect_SUCCESS)
{
    s32 netLayer = 0;
    string netInstId = "test";
    RankGraph rankGraph(myRank);
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    EXPECT_NO_THROW(rankGraph.AddNetInstance(netInstance));
    auto netInstResp = rankGraph.GetNetInstanceByNetInstId(netLayer, netInstId);
    EXPECT_NE(nullptr, netInstResp);
}

TEST_F(RankGraphTest, ut_AddNetInstance_When_NullPeer_Expect_NullPtrException)
{
    RankGraph rankGraph(myRank);
    EXPECT_THROW(rankGraph.AddNetInstance(nullptr), NullPtrException);
}

TEST_F(RankGraphTest, ut_AddNetInstance_When_Init_Finshed_Expect_InternalException)
{
    RankGraph rankGraph(myRank);
    rankGraph.InitFinish();
    auto netInstance = std::make_shared<InnerNetInstance>(0, "test");
    EXPECT_THROW(rankGraph.AddNetInstance(netInstance), InternalException);
}

TEST_F(RankGraphTest, ut_AddNetInstance_When_Not_Include_NetInstance_RankId_Expect_InvalidParamsException)
{
    RankGraph rankGraph(myRank);
    auto netInstance = std::make_shared<InnerNetInstance>(0, "test");
    netInstance->AddRankId(1);
    EXPECT_THROW(rankGraph.AddNetInstance(netInstance), InvalidParamsException);
}

TEST_F(RankGraphTest, ut_Dump_When_LinkHasLongDescription_Expect_LogComplete)
{
    RankGraph rankGraph(myRank);
    auto netInstance = std::make_shared<ClosNetInstance>(0, "test");
    auto peer = createPeer(0, 0, 0);
    auto fabric = std::make_shared<NetInstance::Fabric>(0, "plane0");
    netInstance->AddRankId(peer->GetRankId());
    netInstance->AddNode(peer);
    netInstance->AddNode(fabric);
    rankGraph.AddPeer(peer);
    rankGraph.AddNetInstance(netInstance);

    std::set<std::string> ports = {"0/0"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP, LinkProtocol::UB_MEM};
    auto sourceIface = std::make_shared<NetInstance::ConnInterface>(
        IpAddress("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffe"), ports, AddrPosition::DEVICE, LinkType::PEER2PEER,
        protocols);
    auto targetIface = std::make_shared<NetInstance::ConnInterface>(
        IpAddress("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"), ports, AddrPosition::DEVICE, LinkType::PEER2PEER,
        protocols);
    sourceIface->SetLocalDieId(111);
    targetIface->SetLocalDieId(222);
    netInstance->AddLink(
        std::make_shared<NetInstance::Link>(peer, fabric, sourceIface, nullptr, LinkType::PEER2NET, protocols));
    netInstance->AddLink(
        std::make_shared<NetInstance::Link>(fabric, peer, nullptr, targetIface, LinkType::PEER2NET, protocols));

    extern s32 log_level_get_stub();
    extern void log_level_set_stub(s32 logLevel);
    const s32 originalLogLevel = log_level_get_stub();
    log_level_set_stub(DLOG_DEBUG);
    testing::internal::CaptureStdout();
    rankGraph.Dump();
    const std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(originalLogLevel);

    EXPECT_NE(output.find("NetInstance::Link[srcId=0, dstId=4294967296, type=LinkType::PEER2NET"), std::string::npos);
    EXPECT_NE(output.find("srcIface=Iface"), std::string::npos);
    EXPECT_NE(output.find("localDieId=111"), std::string::npos);
    EXPECT_NE(output.find("NetInstance::Link[srcId=4294967296, dstId=0, type=LinkType::PEER2NET"), std::string::npos);
    EXPECT_NE(output.find("dstIface=Iface"), std::string::npos);
    EXPECT_NE(output.find("localDieId=222"), std::string::npos);
}
TEST_F(RankGraphTest, ut_InitInnerRanks_When_NullPeer_Expect_NullPtrException)
{
    RankGraph rankGraph(myRank);
    EXPECT_THROW(rankGraph.InitInnerRanks(), NullPtrException);
}

TEST_F(RankGraphTest, ut_InitInnerRanks_When_Normal_Expect_SUCCESS)
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

    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/0", "0/2"};
    std::set<LinkProtocol> protocals = {LinkProtocol::UB_CTP, LinkProtocol::UB_TP};
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocals);
    std::shared_ptr<NetInstance::Node> node = std::make_shared<NetInstance::Peer>(myRank, localId, localId, deviceId);
    connInterface->SetLocalDieId(1);
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
    peer0->AddConnInterface(0, connInterface);
    peer1->AddConnInterface(0, connInterface);
    peer2->AddConnInterface(0, connInterface);
    peer3->AddConnInterface(0, connInterface);
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

std::shared_ptr<RankGraph> create4pclosRankGraph(RankId myRank)
{
    RankGraph rankGraph(myRank);
    constexpr u32 layer = 0;
    constexpr u32 mesh01TopoInstId = 0;
    constexpr u32 mesh23TopoInstId = 1;
    constexpr u32 closTopoInstId = 2;

    auto netInstMixPcie = std::make_shared<InnerNetInstance>(layer, "mixpcie");
    auto peer0 = createPeer(myRank, 0, 0);
    auto peer1 = createPeer(1, 1, 1);
    auto peer2 = createPeer(2, 2, 2);
    auto peer3 = createPeer(3, 3, 3);
    auto fabric = std::make_shared<NetInstance::Fabric>(0, "plane0");

    netInstMixPcie->AddRankId(peer0->GetRankId());
    netInstMixPcie->AddRankId(peer1->GetRankId());
    netInstMixPcie->AddRankId(peer2->GetRankId());
    netInstMixPcie->AddRankId(peer3->GetRankId());

    netInstMixPcie->AddNode(peer0);
    netInstMixPcie->AddNode(peer1);
    netInstMixPcie->AddNode(peer2);
    netInstMixPcie->AddNode(peer3);
    netInstMixPcie->AddNode(fabric);

    peer0->AddNetInstance(netInstMixPcie);
    peer1->AddNetInstance(netInstMixPcie);
    peer2->AddNetInstance(netInstMixPcie);
    peer3->AddNetInstance(netInstMixPcie);

    rankGraph.AddPeer(peer0);
    rankGraph.AddPeer(peer1);
    rankGraph.AddPeer(peer2);
    rankGraph.AddPeer(peer3);
    rankGraph.AddNetInstance(netInstMixPcie);

    std::set<std::string> meshPorts = {"0/0"};
    std::set<LinkProtocol> meshProtocols = {LinkProtocol::UB_TP};
    auto makeMeshIface = [&](u32 topoInstId) {
        return std::make_shared<NetInstance::ConnInterface>(
            IpAddress(0), meshPorts, AddrPosition::DEVICE, LinkType::PEER2PEER, meshProtocols, TopoType::MESH_1D,
            topoInstId);
    };
    auto addMeshLink = [&](NetInstance* netInstance, const std::shared_ptr<NetInstance::Peer>& srcPeer,
                           const std::shared_ptr<NetInstance::Peer>& dstPeer, u32 topoInstId) {
        auto srcIface = makeMeshIface(topoInstId);
        auto dstIface = makeMeshIface(topoInstId);
        srcPeer->AddConnInterface(layer, srcIface);
        dstPeer->AddConnInterface(layer, dstIface);
        netInstance->AddLink(std::make_shared<NetInstance::Link>(
            srcPeer, dstPeer, srcIface, dstIface, LinkType::PEER2PEER, meshProtocols));
    };
    addMeshLink(netInstMixPcie.get(), peer0, peer1, mesh01TopoInstId);
    addMeshLink(netInstMixPcie.get(), peer1, peer0, mesh01TopoInstId);
    addMeshLink(netInstMixPcie.get(), peer2, peer3, mesh23TopoInstId);
    addMeshLink(netInstMixPcie.get(), peer3, peer2, mesh23TopoInstId);

    std::set<std::string> ports = {"1/0"};
    std::set<LinkProtocol> protocols = {LinkProtocol::PCIE};
    auto makeIface = [&](u32 topoInstId) {
        return std::make_shared<NetInstance::ConnInterface>(
            IpAddress(0), ports, AddrPosition::DEVICE, LinkType::PEER2NET, protocols, TopoType::CLOS, topoInstId);
    };

    auto rank0ToFabricIface = makeIface(closTopoInstId);
    auto fabricToRank0Iface = makeIface(closTopoInstId);
    auto rank1ToFabricIface = makeIface(closTopoInstId);
    auto fabricToRank1Iface = makeIface(closTopoInstId);
    auto rank2ToFabricIface = makeIface(closTopoInstId);
    auto fabricToRank2Iface = makeIface(closTopoInstId);
    auto rank3ToFabricIface = makeIface(closTopoInstId);
    auto fabricToRank3Iface = makeIface(closTopoInstId);
    peer0->AddConnInterface(layer, rank0ToFabricIface);
    peer0->AddConnInterface(layer, fabricToRank0Iface);
    peer1->AddConnInterface(layer, rank1ToFabricIface);
    peer1->AddConnInterface(layer, fabricToRank1Iface);
    peer2->AddConnInterface(layer, rank2ToFabricIface);
    peer2->AddConnInterface(layer, fabricToRank2Iface);
    peer3->AddConnInterface(layer, rank3ToFabricIface);
    peer3->AddConnInterface(layer, fabricToRank3Iface);

    netInstMixPcie->UpdateTopoInst(mesh01TopoInstId, TopoType::MESH_1D, peer0->GetRankId());
    netInstMixPcie->UpdateTopoInst(mesh01TopoInstId, TopoType::MESH_1D, peer1->GetRankId());
    netInstMixPcie->UpdateTopoInst(mesh23TopoInstId, TopoType::MESH_1D, peer2->GetRankId());
    netInstMixPcie->UpdateTopoInst(mesh23TopoInstId, TopoType::MESH_1D, peer3->GetRankId());
    netInstMixPcie->UpdateTopoInst(closTopoInstId, TopoType::CLOS, peer0->GetRankId());
    netInstMixPcie->UpdateTopoInst(closTopoInstId, TopoType::CLOS, peer1->GetRankId());
    netInstMixPcie->UpdateTopoInst(closTopoInstId, TopoType::CLOS, peer2->GetRankId());
    netInstMixPcie->UpdateTopoInst(closTopoInstId, TopoType::CLOS, peer3->GetRankId());
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(peer0, fabric, rank0ToFabricIface, nullptr, LinkType::PEER2NET, protocols));
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(fabric, peer0, nullptr, fabricToRank0Iface, LinkType::PEER2NET, protocols));
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(peer1, fabric, rank1ToFabricIface, nullptr, LinkType::PEER2NET, protocols));
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(fabric, peer1, nullptr, fabricToRank1Iface, LinkType::PEER2NET, protocols));
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(peer2, fabric, rank2ToFabricIface, nullptr, LinkType::PEER2NET, protocols));
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(fabric, peer2, nullptr, fabricToRank2Iface, LinkType::PEER2NET, protocols));
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(peer3, fabric, rank3ToFabricIface, nullptr, LinkType::PEER2NET, protocols));
    netInstMixPcie->AddLink(
        std::make_shared<NetInstance::Link>(fabric, peer3, nullptr, fabricToRank3Iface, LinkType::PEER2NET, protocols));
    return std::make_shared<RankGraph>(rankGraph);
}

TEST_F(RankGraphTest, ut_GetLayerRanks_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    EXPECT_EQ(4, rankGraph->GetLayerRanks(0));
    EXPECT_EQ(4, rankGraph->GetLayerRanks(1));
    EXPECT_EQ(2, rankGraph->GetLayerRanks(2));
}

TEST_F(RankGraphTest, ut_GetLocalInstRanks_When_Normal_Expect_SUCCESS)
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

TEST_F(RankGraphTest, ut_GetLocalInstSize_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    EXPECT_EQ(2, rankGraph->GetLocalInstSize(0));
    EXPECT_EQ(3, rankGraph->GetLocalInstSize(1));
    EXPECT_EQ(2, rankGraph->GetLocalInstSize(2));
}

TEST_F(RankGraphTest, ut_GetNetType_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    EXPECT_EQ(NetType::TOPO_FILE_DESC, rankGraph->GetNetType(0));
    EXPECT_EQ(NetType::CLOS, rankGraph->GetNetType(1));
    EXPECT_EQ(NetType::CLOS, rankGraph->GetNetType(2));
}

TEST_F(RankGraphTest, ut_GetNetInstanceList_When_Normal_Expect_SUCCESS)
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

TEST_F(RankGraphTest, ut_IsSymmetric_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    EXPECT_EQ(true, rankGraph->IsSymmetric(0));
    EXPECT_EQ(false, rankGraph->IsSymmetric(1));
    EXPECT_EQ(true, rankGraph->IsSymmetric(2));
    EXPECT_THROW(rankGraph->IsSymmetric(3), NullPtrException);
}

TEST_F(RankGraphTest, ut_CreateSubRankGraph_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pclosRankGraph(myRank);
    constexpr u32 layer = 0;
    constexpr u32 missingTopoInstId = 0;
    constexpr u32 nullTopoInstId = 1;
    constexpr u32 noParentTopoInstId = 2;
    constexpr u32 parentTopoInstId = 3;

    vector<u32> subRankIds = {0, 1, 2, 3};
    std::unique_ptr<RankGraph> subRankGraph = rankGraph->CreateSubRankGraph(subRankIds);
    EXPECT_EQ(4, subRankGraph->GetLocalInstSize(layer));
}

TEST_F(RankGraphTest, ut_GetEndpointNum_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    uint32_t num = 0;
    uint32_t topoInstId = 0;
    HcclResult ret = rankGraph->GetEndpointNum(0, topoInstId, &num);
    EXPECT_EQ(num, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankGraphTest, ut_GetEndpointNum_When_nullPeer_HCCL_E_PTR)
{
    RankGraph rankGraph(0);
    uint32_t num = 0;
    uint32_t topoInstId = 0;
    HcclResult ret = rankGraph.GetEndpointNum(0, topoInstId, &num);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(RankGraphTest, ut_GetEndpointDesc_When_nullPeer_HCCL_E_PTR)
{
    RankGraph rankGraph(0);
    uint32_t descNum = 1;
    uint32_t topoInstId = 0;

    EndpointDesc* endPointDesc = new EndpointDesc[descNum];
    HcclResult ret = rankGraph.GetEndpointDesc(0, topoInstId, &descNum, endPointDesc);
    delete[] endPointDesc;
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(RankGraphTest, ut_GetEndpointDesc_When_Normal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    uint32_t layer = 0;
    uint32_t num = 2;
    uint32_t topoInstId = 0;
    HcclResult ret = rankGraph->GetEndpointNum(0, topoInstId, &num);
    EXPECT_EQ(num, 2);

    uint32_t descNum = num;
    EndpointDesc* endPointDesc = new EndpointDesc[descNum];
    ret = rankGraph->GetEndpointDesc(layer, topoInstId, &descNum, endPointDesc);
    for (uint32_t i = 0; i < num; ++i) {
        EXPECT_NE(endPointDesc[i].commAddr.type, COMM_ADDR_TYPE_RESERVED);
        EXPECT_NE(endPointDesc[i].loc.locType, ENDPOINT_LOC_TYPE_RESERVED);
        EXPECT_NE(endPointDesc[i].protocol, COMM_PROTOCOL_RESERVED);
    }
    delete[] endPointDesc;

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankGraphTest, ut_AddGroupLinks_When_1pNormal_Expect_SUCCESS)
{
    auto rankGraph = create4pRankGraph(myRank);
    vector<u32> subRankIds = {0};
    std::unique_ptr<RankGraph> subRankGraph = rankGraph->CreateSubRankGraph(subRankIds);
    EXPECT_EQ(1, subRankGraph->GetLocalInstSize(0));
}

TEST_F(RankGraphTest, ut_GetNetInstanceByNetInstId_When_NotExisted_Expect_ReturnNullptr)
{
    RankGraph rankGraph(myRank);
    auto result = rankGraph.GetNetInstanceByNetInstId(0, "not_existed");
    EXPECT_EQ(nullptr, result);

    s32 netLayer = 0;
    string netInstId = "test";
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    EXPECT_NO_THROW(rankGraph.AddNetInstance(netInstance));
    auto result2 = rankGraph.GetNetInstanceByNetInstId(99, "test");
    EXPECT_EQ(nullptr, result2);
    auto result3 = rankGraph.GetNetInstanceByNetInstId(0, "not_existed");
    EXPECT_EQ(nullptr, result3);
}

TEST_F(RankGraphTest, ut_GetNetInstanceByNetInstId_When_NotExisted_ConstVersion_Expect_ReturnNullptr)
{
    RankGraph rankGraph(myRank);
    const RankGraph& constRankGraph = rankGraph;
    auto result = constRankGraph.GetNetInstanceByNetInstId(0, "not_existed");
    EXPECT_EQ(nullptr, result);

    s32 netLayer = 0;
    string netInstId = "test";
    auto netInstance = std::make_shared<InnerNetInstance>(netLayer, netInstId);
    EXPECT_NO_THROW(rankGraph.AddNetInstance(netInstance));
    auto result2 = constRankGraph.GetNetInstanceByNetInstId(99, "test");
    EXPECT_EQ(nullptr, result2);
    auto result3 = constRankGraph.GetNetInstanceByNetInstId(0, "not_existed");
    EXPECT_EQ(nullptr, result3);
}

TEST_F(RankGraphTest, ut_NeedUpdateTopoInstForSubGraph_When_TopoInstIsNull_Expect_ReturnFalse)
{
    u32 level = 0;
    std::string netInstId = "test";
    InnerNetInstance netInstance(level, netInstId);
    u32 topoInstId = 0;
    netInstance.topoInsts_.emplace(topoInstId, nullptr);

    EXPECT_EQ(false, NeedUpdateTopoInstForSubGraph(&netInstance, topoInstId, myRank));
}
