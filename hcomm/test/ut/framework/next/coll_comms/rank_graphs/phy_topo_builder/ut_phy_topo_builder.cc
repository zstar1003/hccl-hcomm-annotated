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
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/mokc.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include "invalid_params_exception.h"
#include "rank_graph_test_data_builder.h"
#include "types.h"

#define private public
#include "phy_topo_builder.h"
#include "topo_info.h"

#undef private

using namespace Hccl;

std::shared_ptr<TopoInfo> LoadTopoInfoStub(PhyTopoBuilder* This, const std::string& topoPath)
{
    (void)This;
    (void)topoPath;
    return std::make_shared<TopoInfo>(test::MakeTopoInfo(
        {0, 1, 2}, {
                       test::MakeEdge(
                           0, LinkType::PEER2PEER, 0, {"0/0"}, 1, {"0/1"}, LinkProtocol::UB_CTP, AddrPosition::DEVICE,
                           0, TopoType::MESH_1D),
                       test::MakeEdge(
                           0, LinkType::PEER2PEER, 1, {"0/1"}, 2, {"0/2"}, LinkProtocol::UB_CTP, AddrPosition::DEVICE,
                           0, TopoType::MESH_1D),
                       test::MakeEdge(
                           0, LinkType::PEER2NET, 1, {"0/1"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0,
                           TopoType::MESH_1D),
                       test::MakeEdge(
                           1, LinkType::PEER2NET, 0, {"1/3", "1/4", "0/3", "0/4"}, 0, {}, LinkProtocol::UB_MEM,
                           AddrPosition::DEVICE, 1, TopoType::MESH_1D),
                   }));
}

std::shared_ptr<TopoInfo> LoadTopoInfoWithDiffProtocols(PhyTopoBuilder* This, const std::string& topoPath)
{
    (void)This;
    (void)topoPath;
    return std::make_shared<TopoInfo>(test::MakeTopoInfo(
        {0, 1, 2}, {
                       test::MakeEdge(
                           0, LinkType::PEER2PEER, 1, {"0/1"}, 2, {"0/2"}, LinkProtocol::UB_CTP, AddrPosition::DEVICE,
                           0, TopoType::MESH_1D),
                       test::MakeEdge(0, LinkType::PEER2PEER, 1, {"0/1"}, 0, {"0/1"}, LinkProtocol::UB_MEM),
                       test::MakeEdge(
                           0, LinkType::PEER2NET, 0, {"0/1"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0,
                           TopoType::MESH_1D),
                       test::MakeEdge(
                           0, LinkType::PEER2NET, 1, {"0/1"}, 0, {}, LinkProtocol::TCP, AddrPosition::DEVICE, 0,
                           TopoType::MESH_1D),
                   }));
}

std::shared_ptr<TopoInfo> LoadTopoInfoWithRepeatEdge(PhyTopoBuilder* This, const std::string& topoPath)
{
    (void)This;
    (void)topoPath;
    return std::make_shared<TopoInfo>(test::MakeTopoInfo(
        {0, 1}, {
                    test::MakeEdge(
                        0, LinkType::PEER2PEER, 0, {"0/1"}, 1, {"0/2"}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0,
                        TopoType::MESH_1D),
                    test::MakeEdge(
                        0, LinkType::PEER2PEER, 0, {"0/1"}, 1, {"0/2"}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0,
                        TopoType::MESH_1D),
                }));
}

std::unique_ptr<PhyTopo> PhyTopoBuilderBuildStub(const std::string& topoPath)
{
    if (topoPath.empty()) {
        THROW<InvalidParamsException>("[PhyTopoBuilder::%s]Topo path is empty.", __func__);
    }

    HCCL_DEBUG("[PhyTopoBuilder::%s]Start to build physic topo.", __func__);
    std::unique_ptr<PhyTopo> phyTopo = std::make_unique<PhyTopo>();
    PhyTopoBuilder phyTopoBuilder;
    auto topoInfo = phyTopoBuilder.LoadTopoInfo(topoPath);
    // 根据topoInfo，按netLayer构造Graph
    for (const auto& iter : topoInfo->edges) {
        auto netLayer = iter.first;
        auto graph = phyTopoBuilder.CreateGraph(iter.second);
        phyTopo->AddTopoGraph(netLayer, graph);
        HCCL_DEBUG("[PhyTopoBuilder::%s]Build netLayer[%u] topo graph success.", __func__, netLayer);
    }
    return phyTopo;
}

int StatLargeTopoFile(const char* path, struct stat* fileStat)
{
    (void)path;
    if (fileStat == nullptr) {
        return -1;
    }
    fileStat->st_size = SUPPORT_MAX_TOPOFILE_SIZE + 1U;
    return 0;
}

class PhyTopoBuilderTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PhyTopoBuilderTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "PhyTopoBuilderTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in PhyTopoBuilderTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in PhyTopoBuilderTest TearDown" << std::endl;
    }
};

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_EmptyTopoPath_Expect_ThrowInvalidParamsException)
{
    std::string topoPath = "";
    EXPECT_THROW(PhyTopoBuilderBuildStub(topoPath), InvalidParamsException);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_InValidTopoPath_Expect_ThrowInvalidParamsException)
{
    std::string topoPath = "../topo_stub";
    EXPECT_THROW(PhyTopoBuilderBuildStub(topoPath), InvalidParamsException);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_MaxTopoSizeFile_Expect_ThrowInvalidParamsException)
{
    std::string topoPath = "./large_topo_stub";
    MOCKER(stat).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(StatLargeTopoFile));
    EXPECT_THROW(PhyTopoBuilderBuildStub(topoPath), InvalidParamsException);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_ErrorFilePath_Expect_ThrowInvalidParamsException)
{
    std::string topoPath = "llt/ace/comop/hccl/none_file";
    EXPECT_THROW(PhyTopoBuilderBuildStub(topoPath), InvalidParamsException);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_ValidTopoPath_Expect_ReturnEdgeNum)
{
    std::string topoPath = "./topo_stub";
    MOCKER_CPP(&PhyTopoBuilder::LoadTopoInfo).stubs().will(invoke(LoadTopoInfoStub));
    std::unique_ptr<PhyTopo> phyTopo = PhyTopoBuilderBuildStub(topoPath);
    auto graph = phyTopo->GetTopoGraph(0);
    size_t totalEdgeNum = 0;

    // 遍历所有源节点
    for (const auto& srcEntry : graph->edges) {
        const auto& dstMap = srcEntry.second;
        // 遍历该源节点下的所有目标节点
        for (const auto& dstEntry : dstMap) {
            const auto& edgesVec = dstEntry.second; // 该源->目标的所有边
            totalEdgeNum += edgesVec.size();
        }
    }

    EXPECT_EQ(totalEdgeNum, 6);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_EdgeRepeat_Expect_ReturnEdgeNum)
{
    std::string topoPath = "./test_topo_stub";
    MOCKER_CPP(&PhyTopoBuilder::LoadTopoInfo).stubs().will(invoke(LoadTopoInfoWithRepeatEdge));
    std::unique_ptr<PhyTopo> phyTopo = PhyTopoBuilderBuildStub(topoPath);
    auto graph = phyTopo->GetTopoGraph(0);
    ASSERT_NE(graph, nullptr);

    size_t totalEdgeNum = 0;
    for (const auto& srcEntry : graph->edges) {
        for (const auto& dstEntry : srcEntry.second) {
            totalEdgeNum += dstEntry.second.size();
        }
    }
    EXPECT_EQ(totalEdgeNum, 4);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_DiffProtocols_Expect_ReturnEdgeNum)
{
    std::string topoPath = "./topo_stub";
    MOCKER_CPP(&PhyTopoBuilder::LoadTopoInfo).stubs().will(invoke(LoadTopoInfoWithDiffProtocols));
    std::unique_ptr<PhyTopo> phyTopo = PhyTopoBuilderBuildStub(topoPath);
    auto graph = phyTopo->GetTopoGraph(0);
    size_t totalEdgeNum = 0;

    // 遍历所有源节点
    for (const auto& srcEntry : graph->edges) {
        const auto& dstMap = srcEntry.second;
        // 遍历该源节点下的所有目标节点
        for (const auto& dstEntry : dstMap) {
            const auto& edgesVec = dstEntry.second; // 该源->目标的所有边
            totalEdgeNum += edgesVec.size();
        }
    }
    EXPECT_EQ(totalEdgeNum, 8);
}
