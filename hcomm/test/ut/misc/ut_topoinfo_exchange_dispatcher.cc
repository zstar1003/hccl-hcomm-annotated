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
#include <cerrno>
#include <cstring>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <unistd.h>

#define private public
#include "topoinfo_exchange_dispatcher.h"
#undef private
#include "network_manager_pub.h"
#include "topoinfo_exchange_server.h"
#include "hccl_socket.h"
#include "adapter_hccp.h"

using namespace std;
using namespace hccl;

class TopoExchangeDispatcherTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TopoExchangeDispatcherTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "TopoExchangeDispatcherTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

#if 0
TEST_F(TopoExchangeDispatcherTest, ut_processOneSendEvent_sendDataFail)
{
    // create dispatcher
    HcclIpAddress hostIP;
    u32 hostPort;
    std::vector<HcclIpAddress> whitelist;
    HcclNetDevCtx netDevCtx;
    auto socketPtr = std::make_shared<HcclSocket>(netDevCtx, 5);
    std::string identifier;
    TopoInfoExchangeServer topoServer(hostIP, hostPort, whitelist, netDevCtx, socketPtr, identifier);
    TopoInfoExchangeDispatcher workers(&topoServer);

    // create epoll_event
    TopoInfoExchangeDispatcher::SendState txS;
    TopoInfoExchangeDispatcher::FdContext fdCtx;
    auto socket = std::make_shared<HcclSocket>(netDevCtx, 5);
    FdHandle fdHandle;
    socket->fdHandle_ = fdHandle;
    fdCtx.txState = txS;
    fdCtx.socket = socket;
    workers.fdHandleToFdContextMap_.emplace(socket->fdHandle_, fdCtx);

    MOCKER_CPP(&TopoInfoExchangeDispatcher::SendState::Send)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(1));

    HcclResult ret = HCCL_SUCCESS;
    ret = workers.ProcessOneSendEvent(1, socket->fdHandle_);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}

TEST_F(TopoExchangeDispatcherTest, ut_processOneSendEvent_sendDataSuccess)
{
    // create dispatcher
    HcclIpAddress hostIP;
    u32 hostPort;
    std::vector<HcclIpAddress> whitelist;
    HcclNetDevCtx netDevCtx;
    auto socketPtr = std::make_shared<HcclSocket>(netDevCtx, 5);
    std::string identifier;
    TopoInfoExchangeServer topoServer(hostIP, hostPort, whitelist, netDevCtx, socketPtr, identifier);
    TopoInfoExchangeDispatcher workers(&topoServer);

    // create epoll_event
    TopoInfoExchangeDispatcher::SendState txS;
    TopoInfoExchangeDispatcher::FdContext fdCtx;
    auto socket = std::make_shared<HcclSocket>(netDevCtx, 5);
    FdHandle fdHandle;
    socket->fdHandle_ = fdHandle;
    fdCtx.txState = txS;
    fdCtx.socket = socket;
    workers.fdHandleToFdContextMap_.emplace(socket->fdHandle_, fdCtx);

    MOCKER_CPP(&TopoInfoExchangeDispatcher::SendState::Send)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(0));

    MOCKER_CPP(&TopoInfoExchangeDispatcher::SendState::IsOk)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(true));

    MOCKER(hrtRaCtlEventHandle)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));;

    HcclResult ret = HCCL_SUCCESS;
    ret = workers.ProcessOneSendEvent(1, socket->fdHandle_);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(TopoExchangeDispatcherTest, ut_processOneSendEvent_sendAgain)
{
    // create dispatcher
    HcclIpAddress hostIP;
    u32 hostPort;
    std::vector<HcclIpAddress> whitelist;
    HcclNetDevCtx netDevCtx;
    auto socketPtr = std::make_shared<HcclSocket>(netDevCtx, 5);
    std::string identifier;
    TopoInfoExchangeServer topoServer(hostIP, hostPort, whitelist, netDevCtx, socketPtr, identifier);
    TopoInfoExchangeDispatcher workers(&topoServer);

    // create epoll_event
    TopoInfoExchangeDispatcher::SendState txS;
    TopoInfoExchangeDispatcher::FdContext fdCtx;
    auto socket = std::make_shared<HcclSocket>(netDevCtx, 5);
    FdHandle fdHandle;
    socket->fdHandle_ = fdHandle;
    fdCtx.txState = txS;
    fdCtx.socket = socket;
    workers.fdHandleToFdContextMap_.emplace(socket->fdHandle_, fdCtx);

    MOCKER_CPP(&TopoInfoExchangeDispatcher::SendState::Send)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(0));

    MOCKER_CPP(&TopoInfoExchangeDispatcher::SendState::IsOk)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(false));

    MOCKER(hrtRaCtlEventHandle)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_E_NETWORK));;

    HcclResult ret = HCCL_SUCCESS;
    ret = workers.ProcessOneSendEvent(1, socket->fdHandle_);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}
#endif

TEST_F(TopoExchangeDispatcherTest, ut_sendHelper_ISendSuccess)
{
    // create dispatcher
    HcclIpAddress hostIP;
    u32 hostPort;
    std::vector<HcclIpAddress> whitelist;
    HcclNetDevCtx netDevCtx;
    auto socket = std::make_shared<HcclSocket>(netDevCtx, 5);
    std::string identifier;
    TopoInfoExchangeServer topoServer(hostIP, hostPort, whitelist, netDevCtx, socket, identifier);
    TopoInfoExchangeDispatcher workers(&topoServer);
    // set params
    auto socketPtr = std::make_shared<HcclSocket>(netDevCtx, 5);
    char* buf = reinterpret_cast<char*>(0x1234);
    size_t dataLen = 100;
    size_t sendedLen = 80;
    TopoInfoExchangeDispatcher::SendState txS;

    MOCKER_CPP(&HcclSocket::ISend).stubs().with(mockcpp::any()).will(returnValue(0));

    HcclResult ret = HCCL_SUCCESS;
    ret = txS.SendHelper(socketPtr, buf, dataLen, sendedLen);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(TopoExchangeDispatcherTest, ut_sendHelper_networkError)
{
    // create dispatcher
    HcclIpAddress hostIP;
    u32 hostPort;
    std::vector<HcclIpAddress> whitelist;
    HcclNetDevCtx netDevCtx;
    auto socket = std::make_shared<HcclSocket>(netDevCtx, 5);
    std::string identifier;
    TopoInfoExchangeServer topoServer(hostIP, hostPort, whitelist, netDevCtx, socket, identifier);
    TopoInfoExchangeDispatcher workers(&topoServer);
    // set params
    auto socketPtr = std::make_shared<HcclSocket>(netDevCtx, 5);
    char* buf = reinterpret_cast<char*>(0x1234);
    size_t dataLen = 100;
    size_t sendedLen = 80;
    TopoInfoExchangeDispatcher::SendState txS;

    MOCKER_CPP(&HcclSocket::ISend).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_NETWORK));

    HcclResult ret = HCCL_SUCCESS;
    ret = txS.SendHelper(socketPtr, buf, dataLen, sendedLen);
    EXPECT_EQ(ret, HCCL_E_TCP_TRANSFER);
    GlobalMockObject::verify();
}

TEST_F(TopoExchangeDispatcherTest, ut_sendHelper_internalError)
{
    // create dispatcher
    HcclIpAddress hostIP;
    u32 hostPort;
    std::vector<HcclIpAddress> whitelist;
    HcclNetDevCtx netDevCtx;
    auto socket = std::make_shared<HcclSocket>(netDevCtx, 5);
    std::string identifier;
    TopoInfoExchangeServer topoServer(hostIP, hostPort, whitelist, netDevCtx, socket, identifier);
    TopoInfoExchangeDispatcher workers(&topoServer);
    // set params
    auto socketPtr = std::make_shared<HcclSocket>(netDevCtx, 5);
    char* buf = reinterpret_cast<char*>(0x1234);
    size_t dataLen = 100;
    size_t sendedLen = 80;
    TopoInfoExchangeDispatcher::SendState txS;

    MOCKER_CPP(&HcclSocket::ISend).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    HcclResult ret = HCCL_SUCCESS;
    ret = txS.SendHelper(socketPtr, buf, dataLen, sendedLen);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}

TEST_F(TopoExchangeDispatcherTest, ut_Teardown)
{
    // create dispatcher
    HcclIpAddress hostIP;
    u32 hostPort;
    std::vector<HcclIpAddress> whitelist;
    HcclNetDevCtx netDevCtx;
    auto socket = std::make_shared<HcclSocket>(netDevCtx, 5);
    std::string identifier;
    TopoInfoExchangeServer topoServer(hostIP, hostPort, whitelist, netDevCtx, socket, identifier);

    HcclResult ret = HCCL_SUCCESS;
    ret = topoServer.Teardown();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::map<u32, std::shared_ptr<HcclSocket>> connectSockets;
    ret = topoServer.GetConnections(connectSockets);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = topoServer.StopSocketListen(whitelist, hostPort);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = topoServer.StopNetwork(whitelist, hostPort);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

#if 0
TEST_F(TopoExchangeDispatcherTest, ut_processOneSendEvent_sendDataOnceFail)
{
    // create dispatcher
    HcclIpAddress hostIP;
    u32 hostPort;
    std::vector<HcclIpAddress> whitelist;
    HcclNetDevCtx netDevCtx;
    auto socketPtr = std::make_shared<HcclSocket>(netDevCtx, 5);
    std::string identifier;
    TopoInfoExchangeServer topoServer(hostIP, hostPort, whitelist, netDevCtx, socketPtr, identifier);
    TopoInfoExchangeDispatcher workers(&topoServer);

    // create epoll_event
    TopoInfoExchangeDispatcher::SendState txS;
    TopoInfoExchangeDispatcher::FdContext fdCtx;
    auto socket = std::make_shared<HcclSocket>(netDevCtx, 5);
    FdHandle fdHandle;
    socket->fdHandle_ = fdHandle;
    txS.rankId = 1;
    fdCtx.txState = txS;
    fdCtx.socket = socket;
    workers.fdHandleToFdContextMap_.emplace(socket->fdHandle_, fdCtx);

    MOCKER_CPP(&TopoInfoExchangeDispatcher::SendState::Send)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(1));

    HcclResult ret = HCCL_SUCCESS;
    ret = workers.SendOnce();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}

TEST_F(TopoExchangeDispatcherTest, ut_broadcastgroupleaderinfo)
{
    MOCKER_CPP(&TopoInfoExchangeDispatcher::SendState::Send)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(0));
 
    MOCKER_CPP(&TopoInfoExchangeDispatcher::SendState::IsOk)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(true));
 
    MOCKER(hrtRaCtlEventHandle)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
 
    MOCKER(hrtRaCreateEventHandle)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
 
    MOCKER(hrtRaDestroyEventHandle)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
 
    // create dispatcher
    HcclIpAddress hostIP;
    u32 hostPort;
    std::vector<HcclIpAddress> whitelist;
    HcclNetDevCtx netDevCtx;
    auto socketPtr = std::make_shared<HcclSocket>(netDevCtx, 5);
 
 
    std::string identifier;
    TopoInfoExchangeServer topoServer(hostIP, hostPort, whitelist, netDevCtx, socketPtr, identifier);
    TopoInfoExchangeDispatcher workers(&topoServer);
 
    auto socket = std::make_shared<HcclSocket>(netDevCtx, 5);
    std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets;
    connectSockets.insert(std::pair<std::string, std::shared_ptr<HcclSocket>>("1", socket));
 
    GroupLeader_t groupLeader;
    groupLeader.grpLeaderNum = 1;
    vector<HcclRankHandle> rankVec(1);
    rankVec[0].port = 60016;
    rankVec[0].rankId = 0;
    groupLeader.GroupLeaderList.assign(rankVec.begin(), rankVec.end());
 
    HcclResult ret = workers.BroadcastGroupLeaderInfo(connectSockets, groupLeader);
    EXPECT_EQ(ret, HCCL_SUCCESS);
 
    GlobalMockObject::verify();
}

TEST_F(TopoExchangeDispatcherTest, ut_broadcastgroupleaderPortinfo)
{
    MOCKER_CPP(&TopoInfoExchangeDispatcher::SendState::Send)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(0));
 
    MOCKER_CPP(&TopoInfoExchangeDispatcher::SendState::IsOk)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(true));
 
    MOCKER(hrtRaCtlEventHandle)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
 
    MOCKER(hrtRaCreateEventHandle)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
 
    MOCKER(hrtRaDestroyEventHandle)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
 
    // create dispatcher
    HcclIpAddress hostIP;
    u32 hostPort;
    std::vector<HcclIpAddress> whitelist;
    HcclNetDevCtx netDevCtx;
    auto socketPtr = std::make_shared<HcclSocket>(netDevCtx, 5);
 
 
    std::string identifier;
    TopoInfoExchangeServer topoServer(hostIP, hostPort, whitelist, netDevCtx, socketPtr, identifier);
    TopoInfoExchangeDispatcher workers(&topoServer);
 
    auto socket = std::make_shared<HcclSocket>(netDevCtx, 5);
    std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets;
    connectSockets.insert(std::pair<std::string, std::shared_ptr<HcclSocket>>("1", socket));
 
    GroupLeader_t groupLeader;
    groupLeader.grpLeaderNum = 1;
    vector<HcclRankHandle> rankVec(1);
    rankVec[0].port = 60016;
    rankVec[0].rankId = 0;
    groupLeader.GroupLeaderList.assign(rankVec.begin(), rankVec.end());
 
    HcclResult ret = workers.BroadcastGroupLeaderInfo(connectSockets, groupLeader);
    EXPECT_EQ(ret, HCCL_SUCCESS);
 
    GlobalMockObject::verify();
}

#endif
