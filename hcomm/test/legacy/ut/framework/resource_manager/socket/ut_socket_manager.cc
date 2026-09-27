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
#define private public
#define protected public
#include "socket_manager.h"
#include "socket_handle_manager.h"
#include "communicator_impl.h"
#include "ranktable_stub_clos.h"
#include "preempt_port_manager_v2.h"
#include "host_socket_handle_manager.h"
#include "phy_topo.h"
#include "rank_graph_builder.h"
#undef protected
#undef private

using namespace Hccl;

class SocketManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SocketManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "SocketManagerTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        hccpSocketHandle = new int(0);
        MOCKER_CPP(&SocketHandleManager::Create)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(hccpSocketHandle));
        MOCKER_CPP(&SocketHandleManager::Get)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(hccpSocketHandle));
        MOCKER_CPP(&PreemptPortManager::ListenPreempt).stubs().will(ignoreReturnValue());
        SetLinks();

        std::cout << "A Test case in SocketManagerTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete hccpSocketHandle;
        std::cout << "A Test case in SocketManagerTest TearDown" << std::endl;
    }

    IpAddress GetAnIpAddress(RankId rankId = 1)
    {
        IpAddress ipAddress(StringFormat("%u.0.0.0", rankId));
        return ipAddress;
    }

    void SetLinks()
    {
        links.clear();
        for (u32 i = 0; i < 4; i++) {
            PortDeploymentType portDeploymentType = PortDeploymentType::DEV_NET;
            LinkProtocol linkProtocol = LinkProtocol::UB_CTP;
            RankId localRankId = 0;
            RankId remoteRankId = 3 - i;

            LinkData tmpLink(
                portDeploymentType, linkProtocol, localRankId, remoteRankId, GetAnIpAddress(localRankId),
                GetAnIpAddress(remoteRankId));
            links.push_back(tmpLink);
        }
    }

    void* hccpSocketHandle;
    IpAddress localIp;
    IpAddress remoteIp;
    vector<LinkData> links;
    CommunicatorImpl impl;
    u32 localRank = 0;
    u32 devicePhyId = 0;
    u32 listenPort = 60001;
};

TEST_F(SocketManagerTest, batch_create_sockets_should_ok)
{
    // Given
    MOCKER_CPP(&SocketManager::BatchAddWhiteList).stubs();

    // when

    // then
    SocketManager socketMgr(impl, localRank, devicePhyId, listenPort);
    socketMgr.BatchCreateSockets(links);
    auto& serverSocketMap = SocketManager::GetServerSocketMap();
    for (const auto& sock : serverSocketMap) {
        EXPECT_EQ(sock.second->socketStatus, SocketStatus::LISTENING);
    }

    for (const auto& sock : socketMgr.connectedSocketMap) {
        if (sock.first.role == SocketRole::CLIENT) {
            EXPECT_EQ(sock.second->socketStatus, SocketStatus::CONNECT_STARTING);
        }
        std::cout << sock.first.remoteRank << " " << sock.second->socketStatus << std::endl;
    }
}

TEST_F(SocketManagerTest, batch_server_listen_async_conect_sockets_should_ok)
{
    MOCKER_CPP(&SocketManager::BatchAddWhiteList).stubs();
    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
    auto link = links[0];
    Hccl::SocketConfig socketConfig(link.GetRemoteRankId(), link, "test");
    socketMgr.ServerListen(socketConfig);
    auto& serverSocketMap = SocketManager::GetServerSocketMap();
    for (const auto& sock : serverSocketMap) {
        EXPECT_EQ(sock.second->socketStatus, SocketStatus::LISTENING);
    }
    socketMgr.ConnectSockets(socketConfig);
    for (const auto& sock : socketMgr.connectedSocketMap) {
        if (sock.first.role == SocketRole::CLIENT) {
            EXPECT_EQ(sock.second->socketStatus, SocketStatus::CONNECT_STARTING);
        }
        std::cout << sock.first.remoteRank << " " << sock.second->socketStatus << std::endl;
    }
}

TEST_F(SocketManagerTest, test_ServerDeInit_and_GetServerListenSocket)
{
    MOCKER_CPP(&SocketManager::BatchAddWhiteList).stubs();
    SocketManager socketMgr(impl, localRank, devicePhyId, listenPort);
    socketMgr.BatchCreateSockets(links);
    for (auto& link : links) {
        auto portData = link.GetLocalPort();
        socketMgr.ServerDeInit(portData);
        socketMgr.GetServerListenSocket(portData);
    }
}

TEST_F(SocketManagerTest, Ut_ServerInitAll_Skip_Init_When_Env_not_Config)
{
    EnvHostNicConfig envConfig;
    EnvHostNicConfig& fakeEnvConfig = envConfig;
    fakeEnvConfig.hcclDeviceSocketPortRange = CfgField<std::vector<SocketPortRange>>{
        "HCCL_NPU_SOCKET_PORT_RANGE", {}, [](const std::string& s) -> std::vector<SocketPortRange> {
            return CastSocketPortRange(s, "HCCL_NPU_SOCKET_PORT_RANGE");
        }};
    fakeEnvConfig.hcclDeviceSocketPortRange.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));
    NewRankInfo rankInfo;
    EXPECT_NO_THROW(SocketManager::ServerInitAll(rankInfo));
}

TEST_F(SocketManagerTest, Ut_ServerInitAll_Skip_Init_When_Env_Config)
{
    EnvHostNicConfig envConfig;
    EnvHostNicConfig& fakeEnvConfig = envConfig;
    fakeEnvConfig.hcclDeviceSocketPortRange = CfgField<std::vector<SocketPortRange>>{
        "HCCL_NPU_SOCKET_PORT_RANGE", {{16666, 18888}}, [](const std::string& s) -> std::vector<SocketPortRange> {
            return CastSocketPortRange(s, "HCCL_NPU_SOCKET_PORT_RANGE");
        }};
    fakeEnvConfig.hcclDeviceSocketPortRange.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));

    string topoFilePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/communicator/topo2pclos.json"};
    MOCKER_CPP(&CommunicatorImpl::GetTopoFilePath).stubs().will(returnValue(topoFilePath));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    RankGraphBuilder rankGraphBuilder;
    unique_ptr<RankGraph> graph = rankGraphBuilder.Build(RankTable2pClos, topoFilePath, 0);
    EXPECT_NE(nullptr, graph);
    NewRankInfo rankInfo = rankGraphBuilder.GetRankTableInfo()->ranks[0];
    EXPECT_NO_THROW(SocketManager::ServerInitAll(rankInfo));
}

TEST_F(SocketManagerTest, test_BatchCreateSockets_with_SocketConfig)
{
    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
    auto link = links[0];
    Hccl::SocketConfig socketConfig(link.GetRemoteRankId(), link, "test");
    socketMgr.BatchCreateSockets(socketConfig);
    socketMgr.GetConnectedSocket(socketConfig);
}

TEST_F(SocketManagerTest, test_CheckServerPortListening_When_Port_Inconsistent_Expected_False)
{
    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
    auto link = links[0];
    auto& serverSocketMap = SocketManager::GetServerSocketMap();
    serverSocketMap[link.GetLocalPort()] = std::make_shared<Socket>(
        hccpSocketHandle, link.GetLocalPort().GetAddr(), 60001, link.GetRemotePort().GetAddr(), "test",
        SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    bool isListen = socketMgr.CheckServerPortListening(link.GetLocalPort(), 60002);
    EXPECT_FALSE(isListen);
}

TEST_F(SocketManagerTest, Ut_GetDeviceListenPort_When_EnvConfigured_Expect_UsePortRangeMin)
{
    EnvHostNicConfig envConfig;
    EnvHostNicConfig& fakeEnvConfig = envConfig;
    fakeEnvConfig.hcclDeviceSocketPortRange = CfgField<std::vector<SocketPortRange>>{
        "HCCL_NPU_SOCKET_PORT_RANGE", {{60001, 60010}}, [](const std::string& s) -> std::vector<SocketPortRange> {
            return CastSocketPortRange(s, "HCCL_NPU_SOCKET_PORT_RANGE");
        }};
    fakeEnvConfig.hcclDeviceSocketPortRange.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));

    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
    IpAddress addr("1.0.0.0");
    u32 port = socketMgr.GetDeviceListenPort(localRank, addr);
    EXPECT_EQ(port, 60001u);
    GlobalMockObject::verify();
}

TEST_F(SocketManagerTest, Ut_GetDeviceListenPort_When_EnvNotConfigured_Expect_UseDefaultPort)
{
    EnvHostNicConfig envConfig;
    EnvHostNicConfig& fakeEnvConfig = envConfig;
    fakeEnvConfig.hcclDeviceSocketPortRange = CfgField<std::vector<SocketPortRange>>{
        "HCCL_NPU_SOCKET_PORT_RANGE", {}, [](const std::string& s) -> std::vector<SocketPortRange> {
            return CastSocketPortRange(s, "HCCL_NPU_SOCKET_PORT_RANGE");
        }};
    fakeEnvConfig.hcclDeviceSocketPortRange.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));

    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
    IpAddress addr("1.0.0.0");
    u32 port = socketMgr.GetDeviceListenPort(localRank, addr);
    EXPECT_EQ(port, DEFAULT_VALUE_TCPPORT);
    GlobalMockObject::verify();
}

TEST_F(SocketManagerTest, Ut_GetDeviceListenPort_When_PortInMap_Expect_UseMapPort)
{
    EnvHostNicConfig envConfig;
    EnvHostNicConfig& fakeEnvConfig = envConfig;
    fakeEnvConfig.hcclDeviceSocketPortRange = CfgField<std::vector<SocketPortRange>>{
        "HCCL_NPU_SOCKET_PORT_RANGE", {{60001, 60010}}, [](const std::string& s) -> std::vector<SocketPortRange> {
            return CastSocketPortRange(s, "HCCL_NPU_SOCKET_PORT_RANGE");
        }};
    fakeEnvConfig.hcclDeviceSocketPortRange.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));

    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
    IpAddress addr("1.0.0.0");
    RankIpPortMapPtr portMap = std::make_shared<RankIpPortMap>();
    (*portMap)[localRank][addr] = 20000;
    HcclResult ret = socketMgr.SetDeviceServerListenPortMap(portMap);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 port = socketMgr.GetDeviceListenPort(localRank, addr);
    EXPECT_EQ(port, 20000u);
    GlobalMockObject::verify();
}

// 构造含 4 个 rank（0/1/2/3）的 RankIpPortMapPtr，每个 rank 关联一个 IpAddress+端口
static RankIpPortMapPtr BuildRankIpPortMap(u32 basePort)
{
    RankIpPortMapPtr portMap = std::make_shared<RankIpPortMap>();
    for (u32 rank = 0; rank < 4; ++rank) {
        IpAddress addr(StringFormat("%u.0.0.0", rank));
        (*portMap)[rank][addr] = basePort + rank;
    }
    return portMap;
}

// 共享指针不深拷贝并返回 HCCL_SUCCESS
TEST_F(SocketManagerTest, Ut_SetDeviceServerListenPortMap_When_ValidSharedPtr_Expect_SharedNoCopy)
{
    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
    RankIpPortMapPtr portMap = BuildRankIpPortMap(60000);

    long useCountBefore = portMap.use_count();
    HcclResult ret = socketMgr.SetDeviceServerListenPortMap(portMap);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 共享指针不深拷贝：内部 rankListenPortMap_ 与传入指针指向同一对象
    EXPECT_EQ(socketMgr.rankListenPortMap_.get(), portMap.get());
    // use_count 增加（外部 + 内部）
    EXPECT_EQ(portMap.use_count(), useCountBefore + 1);
    GlobalMockObject::verify();
}

// 传入空 shared_ptr 返回 HCCL_E_PTR
TEST_F(SocketManagerTest, Ut_SetDeviceServerListenPortMap_When_NullPtr_Expect_ReturnEPtr)
{
    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
    // rankListenPortMap_ 初始为空
    EXPECT_EQ(socketMgr.rankListenPortMap_, nullptr);

    RankIpPortMapPtr nullPortMap = nullptr;
    HcclResult ret = socketMgr.SetDeviceServerListenPortMap(nullPortMap);
    // CHK_PTR_NULL 判空宏返回 HCCL_E_PTR
    EXPECT_EQ(ret, HCCL_E_PTR);
    // rankListenPortMap_ 未被赋值，保持原值（nullptr）
    EXPECT_EQ(socketMgr.rankListenPortMap_, nullptr);
    GlobalMockObject::verify();
}

// 先 Set 非空，再 Set nullptr，判空宏提前返回且不被置空
TEST_F(SocketManagerTest, Ut_SetDeviceServerListenPortMap_When_SetNullAfterValid_Expect_KeepOriginal)
{
    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
    // 1. 先 Set 一次非空 map
    RankIpPortMapPtr portMap = BuildRankIpPortMap(60000);
    HcclResult ret = socketMgr.SetDeviceServerListenPortMap(portMap);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(socketMgr.rankListenPortMap_.get(), portMap.get());

    // 2. 调用 SetDeviceServerListenPortMap(nullptr)
    RankIpPortMapPtr nullPortMap = nullptr;
    ret = socketMgr.SetDeviceServerListenPortMap(nullPortMap);
    // 3. 返回 HCCL_E_PTR
    EXPECT_EQ(ret, HCCL_E_PTR);
    // 4. rankListenPortMap_ 仍为步骤1设置的值（判空宏提前返回，未被置空）
    EXPECT_EQ(socketMgr.rankListenPortMap_.get(), portMap.get());
    GlobalMockObject::verify();
}

// 经 out 参数返回深拷贝子集且返回 HCCL_SUCCESS
TEST_F(SocketManagerTest, Ut_GetSubCommDeviceServerListenPortMap_When_ValidRankIds_Expect_DeepCopySubSet)
{
    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
    // 设置含 rank 0,1,2,3 的 map
    RankIpPortMapPtr portMap = BuildRankIpPortMap(60000);
    HcclResult ret = socketMgr.SetDeviceServerListenPortMap(portMap);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 取 rank 0,2 的子集
    RankIpPortMapPtr subMap = nullptr;
    std::vector<u32> rankIds = {0, 2};
    ret = socketMgr.GetSubCommDeviceServerListenPortMap(rankIds, subMap);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // subMap 非空
    EXPECT_NE(subMap, nullptr);
    ASSERT_EQ(subMap->size(), 2u);
    // subRankId 0 -> 原 rank 0 的端口；subRankId 1 -> 原 rank 2 的端口
    IpAddress addr0(StringFormat("%u.0.0.0", 0));
    IpAddress addr2(StringFormat("%u.0.0.0", 2));
    EXPECT_EQ(subMap->at(0u).at(addr0), 60000u);
    EXPECT_EQ(subMap->at(1u).at(addr2), 60002u);

    // 子/父隔离：subMap 是深拷贝，修改 subMap 不影响父 map
    (*subMap)[0u][addr0] = 99999;
    EXPECT_EQ((*portMap)[0u][addr0], 60000u);
    GlobalMockObject::verify();
}

// 空父 map（未 Set）时降级返回 HCCL_SUCCESS（subMap 为空 map，后续走默认端口兜底）
TEST_F(SocketManagerTest, Ut_GetSubCommDeviceServerListenPortMap_When_NullParentMap_Expect_ReturnSuccessWithEmptySubMap)
{
    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
    // 1. 验证内部 rankListenPortMap_ 为空（未 Set）
    EXPECT_EQ(socketMgr.rankListenPortMap_, nullptr);

    RankIpPortMapPtr subMap = nullptr;
    std::vector<u32> rankIds = {0, 2};
    // 2. 调用 GetSubCommDeviceServerListenPortMap
    HcclResult ret = socketMgr.GetSubCommDeviceServerListenPortMap(rankIds, subMap);
    // 3. 空父 map 降级返回 HCCL_SUCCESS（subMap 为空 map，后续走默认端口兜底）
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(subMap, nullptr);
    EXPECT_TRUE(subMap->empty());
    GlobalMockObject::verify();
}

// 默认端口缓存命中，第二次未命中查询不重算
TEST_F(SocketManagerTest, Ut_GetDeviceListenPort_When_PortRangeConfigured_Expect_CacheHitOnSecondCall)
{
    EnvHostNicConfig envConfig;
    EnvHostNicConfig& fakeEnvConfig = envConfig;
    fakeEnvConfig.hcclDeviceSocketPortRange = CfgField<std::vector<SocketPortRange>>{
        "HCCL_NPU_SOCKET_PORT_RANGE", {{60001, 60010}}, [](const std::string& s) -> std::vector<SocketPortRange> {
            return CastSocketPortRange(s, "HCCL_NPU_SOCKET_PORT_RANGE");
        }};
    fakeEnvConfig.hcclDeviceSocketPortRange.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).expects(mockcpp::once()).will(returnValue(fakeEnvConfig));

    SocketManager socketMgr(localRank, devicePhyId, devicePhyId, "tmp");
    // 缓存初始为 0
    EXPECT_EQ(socketMgr.defaultListenPort_.load(std::memory_order_relaxed), 0u);

    IpAddress addr("1.0.0.0");
    u32 rankId = 5; // rank 5 不存在（未命中查询）

    // 第一次调用（未命中）：计算并缓存 defaultListenPort_
    u32 port1 = socketMgr.GetDeviceListenPort(rankId, addr);
    EXPECT_EQ(port1, 60001u); // portRanges[0].min = 60001
    EXPECT_EQ(socketMgr.defaultListenPort_.load(std::memory_order_relaxed), 60001u);

    // 第二次调用（仍未命中）：走缓存不重算，返回值与首次相同
    u32 port2 = socketMgr.GetDeviceListenPort(rankId, addr);
    EXPECT_EQ(port2, 60001u);
    // GetHostNicConfig 仅被调用 1 次（expects(once())），第二次走缓存未调 EnvConfig
    EXPECT_EQ(socketMgr.defaultListenPort_.load(std::memory_order_relaxed), 60001u);

    // map 未回写（未命中不回写 map）
    EXPECT_EQ(socketMgr.rankListenPortMap_, nullptr);
    GlobalMockObject::verify();
}
