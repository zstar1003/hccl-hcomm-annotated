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
#include <functional>
#define private public
#include "base_config_legacy.h"
#include "env_config_v2.h"
#include "rank_info_dispatcher.h"
#include "socket.h"
#include "rank_info_detect_service.h"
#include "rank_info_detect_client.h"
#undef private

using namespace std;
using namespace Hccl;

class RankInfoDispatcherTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankInfoDispatcherTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "RankInfoDispatcherTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(RankInfoDispatcherTest, St_ProcessOneSendEvent_When_Input_Right_Expect_Send_Ok)
{
    // when
    MOCKER_CPP(&RankInfoDispatcher::SendState::Send).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&RankInfoDispatcher::SendState::IsOk).stubs().with(mockcpp::any()).will(returnValue(true));

    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);
    RankInfoDispatcher::SendState txS;
    RankInfoDispatcher::FdContext fdCtx;
    fdCtx.txState = txS;
    fdCtx.socket = socket;
    workers.fdHandleToFdContextMap_.emplace(socket->fdHandle, fdCtx);

    // check
    EXPECT_NO_THROW(workers.ProcessOneSendEvent(1, socket->fdHandle));
    EXPECT_EQ(workers.stop_, false);
}

TEST_F(RankInfoDispatcherTest, St_ProcessOneSendEvent_When_Input_Error_Expect_Stop_True)
{
    // when
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    EXPECT_NO_THROW(workers.ProcessOneSendEvent(1, socket->fdHandle));
    EXPECT_EQ(workers.stop_, true);

    // when
    MOCKER_CPP(&RankInfoDispatcher::SendState::Send)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(false))
        .then(returnValue(true));
    RankInfoDispatcher::SendState txS;
    RankInfoDispatcher::FdContext fdCtx;
    fdCtx.txState = txS;
    fdCtx.socket = socket;
    workers.fdHandleToFdContextMap_.emplace(socket->fdHandle, fdCtx);

    // check
    EXPECT_NO_THROW(workers.ProcessOneSendEvent(1, socket->fdHandle));
    EXPECT_EQ(workers.stop_, true);

    // when
    MOCKER_CPP(&RankInfoDispatcher::SendState::IsOk).stubs().with(mockcpp::any()).will(returnValue(false));

    // check
    EXPECT_NO_THROW(workers.ProcessOneSendEvent(1, socket->fdHandle));
    EXPECT_EQ(workers.stop_, true);
}

TEST_F(RankInfoDispatcherTest, St_SendHeader_When_ISend_Ok_Expect_Return_True)
{
    // when
    MOCKER(HrtRaSocketNonBlockSend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(true));

    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    RankInfoDispatcher::SendState txS;
    char* buf;
    size_t dataLen = 100;
    size_t sendedLen = 80;
    EXPECT_EQ(txS.SendHelper(socket, buf, dataLen, sendedLen), true);
}

TEST_F(RankInfoDispatcherTest, St_SendHeader_When_ISend_False_Expect_Return_False)
{
    // when
    MOCKER(HrtRaSocketNonBlockSend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(false));

    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    RankInfoDispatcher::SendState txS;
    char* buf;
    size_t dataLen = 100;
    size_t sendedLen = 80;
    EXPECT_EQ(txS.SendHelper(socket, buf, dataLen, sendedLen), false);
}

TEST_F(RankInfoDispatcherTest, St_SendState_Send_When_ISend_Ok_Expect_SendHeader)
{
    // when
    MOCKER(HrtRaSocketNonBlockSend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(true));

    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    RankInfoDispatcher::SendState txS;
    EXPECT_EQ(txS.Send(socket), true);
}

TEST_F(RankInfoDispatcherTest, St_SendState_Send_When_ISend_Ok_Expect_SendBody)
{
    // when
    MOCKER(HrtRaSocketNonBlockSend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(true));

    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    RankInfoDispatcher::SendState txS;
    txS.headerSended = 1;
    txS.headerLen = 1;
    txS.bodyLen = 1;
    txS.bodySended = 0;
    EXPECT_EQ(txS.Send(socket), true);
}

TEST_F(RankInfoDispatcherTest, St_SendState_Send_When_ISend_False_Expect_Return_False)
{
    // when
    MOCKER(HrtRaSocketNonBlockSend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(false));

    // when
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    RankInfoDispatcher::SendState txS;
    EXPECT_EQ(txS.Send(socket), false);
}

TEST_F(RankInfoDispatcherTest, St_ProcessSend_When_Send_Again_Expect_Return_TimeOut)
{
    // when
    MOCKER(HrtRaSocketNonBlockSend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(false));
    MOCKER_CPP(&RankInfoDispatcher::SendOnce).stubs().with().will(ignoreReturnValue());
    u32 eventsNum1 = 1;
    MOCKER(HrtRaWaitEventHandle)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(eventsNum1))
        .will(returnValue(HCCL_SUCCESS));

    EnvSocketConfig envConfig;
    EnvSocketConfig& fakeEnvConfig = envConfig;
    fakeEnvConfig.linkTimeOut = CfgField<s32>{"HCCL_CONNECT_TIMEOUT", s32(1), Str2T<s32>};
    fakeEnvConfig.linkTimeOut.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetSocketConfig).stubs().will(returnValue(fakeEnvConfig));

    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    workers.rankNum_ = 1;
    EXPECT_THROW(workers.ProcessSend(), TimeoutException);
}

TEST_F(RankInfoDispatcherTest, St_ProcessSend_When_EventsNum_Error_Expect_Return_TimeOut)
{
    // when
    MOCKER(HrtRaSocketNonBlockSend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(false));
    MOCKER_CPP(&RankInfoDispatcher::SendOnce).stubs().with().will(ignoreReturnValue());
    u32 eventsNum = 0;
    MOCKER(HrtRaWaitEventHandle)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(eventsNum))
        .will(returnValue(HCCL_E_NETWORK));

    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    workers.rankNum_ = 1;
    EXPECT_THROW(workers.ProcessSend(), InvalidParamsException);
}

TEST_F(RankInfoDispatcherTest, St_SendOnce_When_InputValue_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDispatcher::SendState::Send).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&RankInfoDispatcher::SendState::IsOk).stubs().with(mockcpp::any()).will(returnValue(true));

    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);
    RankInfoDispatcher::SendState txS;
    RankInfoDispatcher::FdContext fdCtx;
    fdCtx.txState = txS;
    fdCtx.socket = socket;
    workers.fdHandleToFdContextMap_.emplace(socket->fdHandle, fdCtx);

    EXPECT_NO_THROW(workers.SendOnce());
    EXPECT_EQ(workers.sendDoneCount_, 1);
}

TEST_F(RankInfoDispatcherTest, St_SendOnce_When_Input_Error_Expect_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDispatcher::SendState::Send)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(false))
        .then(returnValue(true));
    MOCKER_CPP(&RankInfoDispatcher::SendState::IsOk).stubs().with(mockcpp::any()).will(returnValue(false));

    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);
    RankInfoDispatcher::SendState txS;
    RankInfoDispatcher::FdContext fdCtx;
    fdCtx.txState = txS;
    fdCtx.socket = socket;
    workers.fdHandleToFdContextMap_.emplace(socket->fdHandle, fdCtx);

    EXPECT_THROW(workers.SendOnce(), InvalidParamsException);
    EXPECT_NO_THROW(workers.SendOnce());
    EXPECT_EQ(workers.sendDoneCount_, 0);
}

TEST_F(RankInfoDispatcherTest, St_CleanResource_When_Input_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDispatcher::WakeWoker).stubs().with().will(ignoreReturnValue());

    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    EXPECT_NO_THROW(workers.CleanResource());
    EXPECT_EQ(workers.stop_, true);
}

TEST_F(RankInfoDispatcherTest, St_PrepareResource_When_Input_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDispatcher::InitWorkerThread).stubs().with().will(ignoreReturnValue());

    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    std::unordered_map<std::string, std::shared_ptr<Socket>> connectSockets;
    connectSockets["0"] = socket;
    std::string failedAgentIdList;
    RankTableInfo clusterInfo;
    RankInfoDetectClient client(0, 1, 0, socket);
    client.ConstructSingleRank(clusterInfo);
    EXPECT_NO_THROW(workers.PrepareResource(connectSockets, clusterInfo, failedAgentIdList, 0));
    EXPECT_EQ(workers.fdHandleToFdContextMap_.size(), 1);
}

TEST_F(RankInfoDispatcherTest, St_PrepareResource_When_Input_Expect_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDispatcher::InitWorkerThread).stubs().with().will(ignoreReturnValue());

    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    std::unordered_map<std::string, std::shared_ptr<Socket>> connectSockets;
    connectSockets["abc"] = socket;
    RankTableInfo clusterInfo;
    RankInfoDetectClient client(0, 1, 0, socket);
    client.ConstructSingleRank(clusterInfo);
    std::string failedAgentIdList;
    EXPECT_THROW(workers.PrepareResource(connectSockets, clusterInfo, failedAgentIdList, 0), InvalidParamsException);
    EXPECT_EQ(workers.fdHandleToFdContextMap_.size(), 0);
}

TEST_F(RankInfoDispatcherTest, St_GetTask_When_Input_Null_Expect_Return_false)
{
    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    RankInfoDispatcher::WorkerTask workTask;
    EXPECT_EQ(workers.GetTask(workTask), false);
}

TEST_F(RankInfoDispatcherTest, St_InitWorkerThread_When_Input_Expect_Return_NO_THROW)
{
    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    workers.rankNum_ = 1;
    EXPECT_NO_THROW(workers.InitWorkerThread());
}

TEST_F(RankInfoDispatcherTest, St_WorkerWait_When_Input_Expect_Return_NO_THROW)
{
    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    workers.ready_ = true;
    EXPECT_NO_THROW(workers.WorkerWait(0));
}

TEST_F(RankInfoDispatcherTest, St_BroadcastRankTable_When_Input_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDispatcher::PrepareResource).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&RankInfoDispatcher::ProcessSend).stubs().with().will(ignoreReturnValue());

    // then
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    std::unordered_map<std::string, std::shared_ptr<Socket>> connectSockets;
    connectSockets["0"] = socket;
    std::string failedAgentIdList;
    RankTableInfo clusterInfo;
    RankInfoDetectClient client(0, 1, 0, socket);
    client.ConstructSingleRank(clusterInfo);
    EXPECT_NO_THROW(workers.BroadcastRankTable(connectSockets, clusterInfo, failedAgentIdList, 0));
}
