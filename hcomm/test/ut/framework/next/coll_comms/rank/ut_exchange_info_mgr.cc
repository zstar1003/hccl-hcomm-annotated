/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "rank_graph_interface.h"
#include "rank_graph_v2.h"
#include "hcomm_c_adpt.h"
#include <hccl/hccl_comm.h>
#include "channel_process.h"
#include "base_config_legacy.h"
#define private public
#include "my_rank.h"
#undef private
#include "hccl_comm_pub.h"
#include "llt_hccl_stub_rank_graph.h"
#include "rank_consistency_checker_v2.h"

using namespace hccl;

class ExchangeInfoMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ExchangeInfoMgrTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "ExchangeInfoMgrTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in ExchangeInfoMgrTest SetUP" << std::endl;
        rankIpPortMap = std::make_shared<std::unordered_map<u32, std::unordered_map<Hccl::IpAddress, u32>>>();
        (*rankIpPortMap)[0][Hccl::IpAddress("1.0.0.0")] = 16666;
        (*rankIpPortMap)[1][Hccl::IpAddress("2.0.0.0")] = 16666;
        (*rankIpPortMap)[2][Hccl::IpAddress("0.0.0.0")] = 16666;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in ExchangeInfoMgrTest TearDown" << std::endl;
    }

    Hccl::RankIpPortMapPtr rankIpPortMap;
};

static HcclResult InitCollComm(std::shared_ptr<hccl::hcclComm> hcclCommPtr)
{
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    void* commV2 = (void*)0x2000;
    uint32_t rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclOpExpansionMode = 1;
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;
    return hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
}

TEST_F(ExchangeInfoMgrTest, Ut_WaitAllAsyncComplete_When_AllOk_Expect_Success)
{
    // mock Socket::GetAsyncStatus返回OK
    MOCKER_CPP(&Hccl::Socket::GetAsyncStatus).stubs().will(returnValue(Hccl::SocketStatus(Hccl::SocketStatus::OK)));

    aclrtBinHandle binHandle;
    CommConfig config;
    ManagerCallbacks callbacks;
    void* rankGraphPtr = (void*)0x114514;
    std::shared_ptr<RankGraph> rankGraph = std::make_shared<RankGraphV2>(rankGraphPtr);
    MyRank myRank(binHandle, 0, config, callbacks, rankGraph.get(), rankIpPortMap);

    // 构造socket列表（指针值仅用于mock匹配，不实际调用）
    std::vector<Hccl::Socket*> sockets = {(Hccl::Socket*)0x1, (Hccl::Socket*)0x2};
    std::vector<u32> remoteRanks = {1, 2};
    ExchangeInfoMgr exchangeInfoMgr;
    HcclResult ret = exchangeInfoMgr.WaitAllAsyncComplete(sockets, remoteRanks);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ExchangeInfoMgrTest, Ut_BatchExchange_When_NewRankConsistent_Expect_Success)
{
    HcclResult ret = HCCL_SUCCESS;
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>();
    ASSERT_EQ(InitCollComm(hcclCommPtr), HCCL_SUCCESS);
    hccl::CollComm* collComm = hcclCommPtr->GetCollComm();
    hccl::MyRank* myRank = collComm->GetMyRank();
    CollCommConfigConsistency& collCommConfigConsistency = myRank->GetCollCommConfigConsistency();
    std::vector<u8> localData = {0xDE, 0xAD, 0xBE, 0xEF};
    ret = collCommConfigConsistency.AddExchangeInfo(localData.data(), localData.size());
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // mock Socket异步接口：GetAsyncStatus返回OK
    MOCKER_CPP(&Hccl::Socket::GetAsyncStatus).stubs().will(returnValue(Hccl::SocketStatus(Hccl::SocketStatus::OK)));
    // mock SendAsync/RecvAsync成功
    MOCKER_CPP(&Hccl::Socket::SendAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::RecvAsync).stubs().will(returnValue(HCCL_SUCCESS));
    // mock 超时配置
    MOCKER_CPP(&Hccl::EnvSocketConfig::GetLinkTimeOut).stubs().will(returnValue((s32)1));
    MOCKER_CPP(&RankConsistencyCheckerV2::CompareCheckFrameV2).stubs().will(returnValue(HCCL_SUCCESS));

    HcclChannelDesc channelDescs[1];
    channelDescs[0].remoteRank = 1;
    HcommChannelDesc hcommDesc;
    hcommDesc.socket = (HcommSocket)0x1; // 非空socket
    hcommDesc.role = HCOMM_SOCKET_ROLE_CLIENT;
    std::vector<HcommChannelDesc> hcommDescVec;
    hcommDescVec.push_back(hcommDesc);
    ExchangeInfoMgr exchangeInfoMgr;
    std::vector<std::pair<u32, u32>> newChannels;
    newChannels.emplace_back(std::make_pair(0, 1));
    ret = exchangeInfoMgr.BatchExchangeAndCheckConsistency(
        channelDescs, hcommDescVec, 1, newChannels, collCommConfigConsistency, COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// host 网卡场景，使用 Peer 模式同步收发接口，模拟交换信息成功
TEST_F(ExchangeInfoMgrTest, Ut_BatchExchange_When_NewRankConsistent_Expect_Success_On_Cpu)
{
    // 初始化通信域
    HcclResult ret = HCCL_SUCCESS;
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>();
    ASSERT_EQ(InitCollComm(hcclCommPtr), HCCL_SUCCESS);

    // 添加交换信息
    hccl::CollComm* collComm = hcclCommPtr->GetCollComm();
    hccl::MyRank* myRank = collComm->GetMyRank();
    CollCommConfigConsistency& collCommConfigConsistency = myRank->GetCollCommConfigConsistency();
    std::vector<u8> localData = {0xDE, 0xAD, 0xBE, 0xEF};
    ret = collCommConfigConsistency.AddExchangeInfo(localData.data(), localData.size());
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // mock socket接口
    MOCKER_CPP(&Hccl::Socket::GetAsyncStatus).stubs().will(returnValue(Hccl::SocketStatus(Hccl::SocketStatus::OK)));
    MOCKER_CPP(&Hccl::Socket::Send).stubs().will(returnValue(true));
    MOCKER_CPP(&Hccl::Socket::Recv).stubs().will(returnValue(true));
    // mock 超时配置
    MOCKER_CPP(&Hccl::EnvSocketConfig::GetLinkTimeOut).stubs().will(returnValue((s32)1));
    // mock 比对结果
    MOCKER_CPP(&RankConsistencyCheckerV2::CompareCheckFrameV2).stubs().will(returnValue(HCCL_SUCCESS));

    // 构造 hcclDesc
    HcclChannelDesc channelDescs[1];
    channelDescs[0].remoteRank = 1;
    channelDescs[0].localEndpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST; // host 网卡场景
    // 构造 hcommDescs
    HcommChannelDesc hcommDesc;
    hcommDesc.socket = (HcommSocket)0x1; // 非空socket
    hcommDesc.role = HCOMM_SOCKET_ROLE_CLIENT;
    std::vector<HcommChannelDesc> hcommDescVec;
    hcommDescVec.push_back(hcommDesc);
    // 构造 channels
    std::vector<std::pair<u32, u32>> newChannels;
    newChannels.emplace_back(std::make_pair(0, 1));

    // 交换信息
    ExchangeInfoMgr exchangeInfoMgr;
    ret = exchangeInfoMgr.BatchExchangeAndCheckConsistency(
        channelDescs, hcommDescVec, 1, newChannels, collCommConfigConsistency, COMM_ENGINE_CPU);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ============================================================================================
// ExchangeAsyncDataPhase 单元测试
// 对应业务代码：src/coll_communicator_mgr/resource_mgr/local/my_rank/exchange_info_mgr.cc
// 该接口为 ExchangeUserInfoAsync 内部提取的两阶段异步收发私有方法：
//   isServerRecv=true  -> SERVER 先 Recv、CLIENT 先 Send（第一阶段，防死锁）
//   isServerRecv=false -> SERVER 再 Send、CLIENT 再 Recv（第二阶段）
// 通过 mock Socket::SendAsync/RecvAsync 验证收发方向与越界长度校验
// ============================================================================================

// 正常路径：第一阶段 isServerRecv=true，SERVER 角色 Recv、CLIENT 角色 Send
TEST_F(ExchangeInfoMgrTest, Ut_ExchangeAsyncDataPhase_When_ServerRecvFirst_Expect_Success)
{
    // 构造本地交换数据，使 GetExchangeInfoBuf 在 Send 分支返回非空 buffer
    CollCommConfigConsistency collCommConfigConsistency;
    std::vector<u8> localData = {0xAA, 0xBB, 0xCC, 0xDD};
    ASSERT_EQ(collCommConfigConsistency.AddExchangeInfo(localData.data(), localData.size()), HCCL_SUCCESS);

    // mock Socket 异步收发接口（非虚成员函数，MOCKER_CPP 经 API hook 拦截）
    MOCKER_CPP(&Hccl::Socket::SendAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::RecvAsync).stubs().will(returnValue(HCCL_SUCCESS));

    // 构造 2 个对端：rank1=SERVER(Recv)、rank2=CLIENT(Send)
    std::vector<Hccl::Socket*> sockets = {(Hccl::Socket*)0x1, (Hccl::Socket*)0x2};
    std::vector<HcommSocketRole> roles = {HCOMM_SOCKET_ROLE_SERVER, HCOMM_SOCKET_ROLE_CLIENT};
    std::vector<u32> remoteRanks = {1, 2};
    std::vector<u32> remoteExchangeInfoLens = {4, 4};
    u32 localExchangeInfoLen = 4;
    std::vector<std::vector<u8>> remoteUserDatas(sockets.size());

    ExchangeInfoMgr exchangeInfoMgr;
    HcclResult ret = exchangeInfoMgr.ExchangeAsyncDataPhase(
        sockets, roles, remoteRanks, remoteUserDatas, remoteExchangeInfoLens, localExchangeInfoLen,
        collCommConfigConsistency, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // SERVER 角色应完成 Recv，remoteUserDatas[0] 应被 resize
    EXPECT_EQ(remoteUserDatas[0].size(), 4ULL);
}

// 正常路径：第二阶段 isServerRecv=false，SERVER 角色 Send、CLIENT 角色 Recv
TEST_F(ExchangeInfoMgrTest, Ut_ExchangeAsyncDataPhase_When_ServerSendSecond_Expect_Success)
{
    // 构造本地交换数据
    CollCommConfigConsistency collCommConfigConsistency;
    std::vector<u8> localData = {0x11, 0x22, 0x33};
    ASSERT_EQ(collCommConfigConsistency.AddExchangeInfo(localData.data(), localData.size()), HCCL_SUCCESS);

    // mock Socket 异步收发接口
    MOCKER_CPP(&Hccl::Socket::SendAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::RecvAsync).stubs().will(returnValue(HCCL_SUCCESS));

    // 构造 2 个对端：rank1=SERVER(Send)、rank2=CLIENT(Recv)
    std::vector<Hccl::Socket*> sockets = {(Hccl::Socket*)0x1, (Hccl::Socket*)0x2};
    std::vector<HcommSocketRole> roles = {HCOMM_SOCKET_ROLE_SERVER, HCOMM_SOCKET_ROLE_CLIENT};
    std::vector<u32> remoteRanks = {1, 2};
    std::vector<u32> remoteExchangeInfoLens = {3, 3};
    u32 localExchangeInfoLen = 3;
    std::vector<std::vector<u8>> remoteUserDatas(sockets.size());

    ExchangeInfoMgr exchangeInfoMgr;
    HcclResult ret = exchangeInfoMgr.ExchangeAsyncDataPhase(
        sockets, roles, remoteRanks, remoteUserDatas, remoteExchangeInfoLens, localExchangeInfoLen,
        collCommConfigConsistency, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // CLIENT 角色应完成 Recv，remoteUserDatas[1] 应被 resize
    EXPECT_EQ(remoteUserDatas[1].size(), 3ULL);
}

// 边界条件：remoteExchangeInfoLens 超过 HCCL_EXCHANGE_INFO_LEN(2048)，应返回 HCCL_E_PARA
TEST_F(ExchangeInfoMgrTest, Ut_ExchangeAsyncDataPhase_When_LenExceedsMax_Expect_ParaError)
{
    CollCommConfigConsistency collCommConfigConsistency;
    std::vector<u8> localData = {0x01};
    ASSERT_EQ(collCommConfigConsistency.AddExchangeInfo(localData.data(), localData.size()), HCCL_SUCCESS);

    // mock Socket 异步收发接口（本用例不应实际触发收发）
    MOCKER_CPP(&Hccl::Socket::SendAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::RecvAsync).stubs().will(returnValue(HCCL_SUCCESS));

    // SERVER 角色 + isServerRecv=true -> 走 Recv 分支，触发长度越界校验
    std::vector<Hccl::Socket*> sockets = {(Hccl::Socket*)0x1};
    std::vector<HcommSocketRole> roles = {HCOMM_SOCKET_ROLE_SERVER};
    std::vector<u32> remoteRanks = {1};
    // 长度超过 HCCL_EXCHANGE_INFO_LEN 上限(2048)，应被 CHK_PRT_RET 拦截并返回 HCCL_E_PARA
    std::vector<u32> remoteExchangeInfoLens = {2049}; // 2049 > 2048
    u32 localExchangeInfoLen = 1;
    std::vector<std::vector<u8>> remoteUserDatas(sockets.size());

    ExchangeInfoMgr exchangeInfoMgr;
    HcclResult ret = exchangeInfoMgr.ExchangeAsyncDataPhase(
        sockets, roles, remoteRanks, remoteUserDatas, remoteExchangeInfoLens, localExchangeInfoLen,
        collCommConfigConsistency, true);
    EXPECT_EQ(ret, HCCL_E_PARA);
}
