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
#include "host_socket_handle_manager.h"
#include "whitelist.h"
#include "hccp_peer_manager.h"
#include "dev_type.h"

#include "socket.h"
#include <string>
#include <unordered_map>
#include "new_rank_info.h"
#include "rank_table_info.h"
#include "json_parser.h"
#include "root_handle_v2.h"
#include "internal_exception.h"
#include "timeout_exception.h"
#include "socket_exception.h"
#include "null_ptr_exception.h"
#include "rank_info_dispatcher.h"
#include "rank_info_detect_service.h"
#include "orion_adapter_rts.h"
#include "env_config/env_config_v2.h"
#include "base_config_legacy.h"
#include "socket_agent.h"
#include "adapter_error_manager_pub.h"
#undef private

using namespace Hccl;

namespace {
std::string g_serviceCapturedErrorCode;
void StubServiceRptInputErr(std::string error_code, std::vector<std::string> key, std::vector<std::string> value)
{
    (void)key;
    (void)value;
    g_serviceCapturedErrorCode = error_code;
}
} // namespace

class RankInfoDetectServiceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankInfoDetectService tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RankInfoDetectService tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in RankInfoDetectService SetUP" << std::endl;
        // 初始化模拟 Socket 句柄
        hccpSocketHandle = new int(0);
        MOCKER(HrtRaSocketInit).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(hccpSocketHandle));

        // 1. 构造测试所需参数
        u32 devPhyId_ = 0;
        IpAddress hostIp_ = GetAnIpAddress();
        u32 hostPort_ = 60007;
        IpAddress remoteIp_ = IpAddress("127.0.0.1");
        std::string serverSocketTag_ = "rank_info_test_server";

        // 2. 创建服务器 Socket（用于传入被测试类构造函数）
        auto serverSocket_ = std::make_shared<Socket>(
            hccpSocketHandle, hostIp_, hostPort_, remoteIp_, serverSocketTag_, SocketRole::SERVER,
            NicType::DEVICE_NIC_TYPE);

        MOCKER_CPP(&HccpPeerManager::Init).stubs().with(mockcpp::any());
        MOCKER_CPP(&HccpPeerManager::DeInit).stubs().with(mockcpp::any());
        MOCKER_CPP(&HostSocketHandleManager::Create)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(hccpSocketHandle));

        serverSocket_->Listen(); // 启动监听（模拟真实场景）

        // 4. 初始化被测试对象
        rankInfoDetectService_ = new RankInfoDetectService(devPhyId_, serverSocket_, "test", {});
    }

    virtual void TearDown()
    {
        // 验证所有打桩的预期是否满足（mockcpp 核心校验）
        GlobalMockObject::verify();
        // 释放动态资源
        delete hccpSocketHandle;
        delete rankInfoDetectService_;
        std::cout << "A Test case in RankInfoDetectService TearDown" << std::endl;
    }

    IpAddress GetAnIpAddress() { return IpAddress("1.0.0.0"); }

    // 辅助函数：配置 1s 建链超时并 mock HostSocketHandleManager::Get
    void SetupMockSocketAndTimeoutConfig()
    {
        static EnvSocketConfig envSocketConfig;
        EnvSocketConfig& fakeEnvSocketConfig = envSocketConfig;
        fakeEnvSocketConfig.linkTimeOut = CfgField<s32>{"HCCL_CONNECT_TIMEOUT", s32(1), Str2T<s32>};
        fakeEnvSocketConfig.linkTimeOut.isParsed = true;
        MOCKER_CPP(&EnvConfig::GetSocketConfig).stubs().will(returnValue(fakeEnvSocketConfig));

        MOCKER_CPP(&HostSocketHandleManager::Get)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(hccpSocketHandle));
    }

    // 辅助函数：生成模拟的 RankInfo 消息（用于 GetRankTable 测试）
    std::string GenMockRankInfoMsg(const std::string& rankTableStr, u32 step)
    {
        // 消息格式：[rankTableStr][step(4字节)]
        std::string msg = rankTableStr;
        msg.append(reinterpret_cast<const char*>(&step), sizeof(step));
        return msg;
    }

    // 成员变量
    void* hccpSocketHandle;
    RankInfoDetectService* rankInfoDetectService_;
};

TEST_F(RankInfoDetectServiceTest, Ut_RankInfoDetectService_When_Init_Expect_Success)
{
    auto res1 = rankInfoDetectService_->devPhyId_;
    EXPECT_EQ(0, res1);

    EXPECT_NE(nullptr, rankInfoDetectService_->serverSocket_);

    auto res2 = rankInfoDetectService_->serverSocket_->isListening;
    EXPECT_EQ(true, res2);

    auto res3 = rankInfoDetectService_->serverSocket_->listenPort;
    EXPECT_EQ(60007, res3);
}

TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_Timeout_Expect_fail)
{
    SetupMockSocketAndTimeoutConfig();

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT));

    EXPECT_THROW(rankInfoDetectService_->GetConnections(), InternalException);

    auto res1 = rankInfoDetectService_->connSockets_.size();
    EXPECT_EQ(0, res1);
}

TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_Timeout_Expect_NoEI0016Report)
{
    SetupMockSocketAndTimeoutConfig();

    // 持续 CONNECTING，模拟对端一直未建链直至超时
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::CONNECTING));

    g_serviceCapturedErrorCode.clear();
    MOCKER(RptInputErr).stubs().will(invoke(StubServiceRptInputErr));

    EXPECT_THROW(rankInfoDetectService_->GetConnections(), InternalException);
    // 耗时耗尽分支已收敛为不再单独上报 EI0016，避免与超时 EI0015 重复
    EXPECT_NE(g_serviceCapturedErrorCode, "EI0016");
}

TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_Normal_Expect_Success)
{
    MOCKER_CPP(&HostSocketHandleManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(hccpSocketHandle));

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK));

    u32 rankSize = 1;
    void* msg = &rankSize;
    u64 msgLen = sizeof(u32);
    u64& revMsgLen = msgLen;
    MOCKER_CPP(&SocketAgent::RecvMsg).stubs().with(outBound(msg), outBound(revMsgLen)).will(returnValue(true));

    rankInfoDetectService_->GetConnections();

    auto res1 = rankInfoDetectService_->connSockets_.size();
    EXPECT_EQ(1, res1);

    for (auto& iter : rankInfoDetectService_->connSockets_) {
        EXPECT_NE(nullptr, iter.second);
    }
}

TEST_F(RankInfoDetectServiceTest, Ut_GetRankTable_When_Normal_Expect_Success)
{
    u32 port1 = 12345;
    std::string socketKey1 = std::to_string(port1);

    IpAddress hostIp = rankInfoDetectService_->serverSocket_->GetLocalIp();
    std::string tag1 = RANK_INFO_DETECT_TAG + "_" + socketKey1;
    auto mockSocket1 = std::make_shared<Socket>(
        hccpSocketHandle, hostIp, port1, hostIp, tag1, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    rankInfoDetectService_->connSockets_[socketKey1] = mockSocket1;

    auto res0 = rankInfoDetectService_->connSockets_.size();
    EXPECT_EQ(1, res0);

    for (auto& iter : rankInfoDetectService_->connSockets_) {
        EXPECT_NE(nullptr, iter.second);
    }

    // 构造一个ranktableinfo
    RankTableInfo localRankTable;
    localRankTable.version = "1.0";
    localRankTable.rankCount = 1;
    NewRankInfo rankInfo{};
    rankInfo.rankId = 0;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo{});
    localRankTable.ranks.emplace_back(rankInfo);

    // 把构造好的ranktableinfo转为vector<char> rankInfoMsg
    // 消息格式: [ranktable数据(n字节)][step(4字节)]
    BinaryStream binaryStream;
    localRankTable.GetBinStream(true, binaryStream);
    binaryStream << rankInfoDetectService_->currentStep_;

    // 字节流转换为vector<char>格式
    vector<char> rankInfoMsg;
    binaryStream.Dump(rankInfoMsg);

    // 取rankInfoMsg的size
    u32 rankInfoSize = rankInfoMsg.size();
    u64 expectLen = rankInfoSize;

    // 取rankInfoMsg的data() （const char *）
    MOCKER(aclrtMallocHostWithCfg).stubs().will(returnValue(1));
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue((void*)rankInfoMsg.data()));
    char* rankInfoMsgToSend = rankInfoMsg.data();
    void* msg = rankInfoMsg.data();
    u64 msgLen = rankInfoMsg.size();
    u64& revMsgLen = msgLen;
    MOCKER_CPP(&SocketAgent::RecvMsg).stubs().with(outBoundP(msg, msgLen), outBound(revMsgLen)).will(returnValue(true));

    EXPECT_NO_THROW(rankInfoDetectService_->GetRankTable());
}

TEST_F(RankInfoDetectServiceTest, Ut_BroadcastRankTable_When_Normal_Expect_Success)
{
    MOCKER_CPP(&RankInfoDispatcher::BroadcastRankTable)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(true));

    EXPECT_NO_THROW(rankInfoDetectService_->BroadcastRankTable());

    IpAddress localIp, remoteIp;
    SocketHandle socketHandle;
    std::string tag = "test";
    rankInfoDetectService_->connSockets_[tag] = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    EXPECT_NO_THROW(rankInfoDetectService_->BroadcastRankTable());
}

TEST_F(RankInfoDetectServiceTest, Ut_Disconnect_When_Normal_Expect_Success)
{
    u32 port1 = 12345;
    u32 port2 = 12346;
    std::string socketKey1 = std::to_string(port1);
    std::string socketKey2 = std::to_string(port2);

    IpAddress hostIp = rankInfoDetectService_->serverSocket_->GetLocalIp();
    std::string tag1 = RANK_INFO_DETECT_TAG + "_" + socketKey1;
    std::string tag2 = RANK_INFO_DETECT_TAG + "_" + socketKey2;
    auto mockSocket1 = std::make_shared<Socket>(
        hccpSocketHandle, hostIp, port1, hostIp, tag1, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    auto mockSocket2 = std::make_shared<Socket>(
        hccpSocketHandle, hostIp, port2, hostIp, tag2, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    rankInfoDetectService_->connSockets_[socketKey1] = mockSocket1;
    rankInfoDetectService_->connSockets_[socketKey2] = mockSocket2;

    auto res1 = rankInfoDetectService_->connSockets_.size();
    EXPECT_EQ(2, res1);

    for (auto& iter : rankInfoDetectService_->connSockets_) {
        EXPECT_NE(nullptr, iter.second);
    }

    rankInfoDetectService_->Disconnect();

    auto res2 = rankInfoDetectService_->connSockets_.size();
    EXPECT_EQ(0, res2);
}

TEST_F(RankInfoDetectServiceTest, Ut_ParseRankTable_When_Normal_Expect_Success)
{
    RankTableInfo localRankTable;
    localRankTable.version = "1.0";
    localRankTable.rankCount = 1;
    NewRankInfo rankInfo{};
    rankInfo.rankId = 0;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo{});
    localRankTable.ranks.emplace_back(rankInfo);

    // 把构造好的ranktableinfo转为vector<char> rankInfoMsg
    // 消息格式: [ranktable数据(n字节)][step(4字节)]
    BinaryStream binaryStream;
    localRankTable.GetBinStream(true, binaryStream);
    binaryStream << rankInfoDetectService_->currentStep_;

    // 字节流转换为vector<char>格式
    vector<char> rankInfoMsg;
    binaryStream.Dump(rankInfoMsg);

    // Mock UpdateRankTable方法
    MOCKER_CPP(&RankTableInfo::UpdateRankTable).expects(exactly(1)).with(mockcpp::any()).will(returnValue(true));

    // 6. 执行测试
    EXPECT_NO_THROW(rankInfoDetectService_->ParseRankTable(rankInfoMsg));
}

bool RecvRemoteAgentIdStub(RankInfoDetectService* thisPtr, SocketAgent& connSocketAgent, std::string& agentId)
{
    static int nextRank = 0;
    agentId = std::to_string(nextRank);
    nextRank++;
    return true;
}

TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_Last_Fail_Expect_Print_Connected_Socket)
{
    MOCKER_CPP(&HostSocketHandleManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(hccpSocketHandle));

    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        // start time
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        // execute time
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(10))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(20))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(30))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(40))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(50))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(60))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(70))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(80))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(90))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(100))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(100000))));

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        // 第一次连接中成功
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        // 第二次成功
        .then(returnValue((SocketStatus)SocketStatus::OK))
        // 第三次超时后成功
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        // 第四次异常
        .then(throws(InternalException("Internal Error Happens")));

    u32 rankSize = 4;
    void* msg = &rankSize;
    u64 msgLen = sizeof(u32);
    u64& revMsgLen = msgLen;
    MOCKER_CPP(&SocketAgent::RecvMsg).stubs().with(outBound(msg), outBound(revMsgLen)).will(returnValue(true));

    string agentId{"0"};
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId)
        .stubs()
        .with(mockcpp::any(), outBound(agentId))
        .will(invoke(RecvRemoteAgentIdStub));

    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));

    rankInfoDetectService_->GetConnections();
}

TEST_F(RankInfoDetectServiceTest, Ut_GetConnections_When_Sudden_Fail)
{
    MOCKER_CPP(&HostSocketHandleManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(hccpSocketHandle));

    MOCKER_CPP(&Socket::GetStatus)
        .stubs()
        // 第一次连接中成功
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::OK))
        // 第二次先超时，后失败
        .then(returnValue((SocketStatus)SocketStatus::CONNECTING))
        .then(returnValue((SocketStatus)SocketStatus::TIMEOUT))
        .then(throws(InternalException("Internal Error Happens")));

    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        // start time
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        // execute time
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(20))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(40))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(60))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(80))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(100))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(100000))));

    u32 rankSize = 4;
    void* msg = &rankSize;
    u64 msgLen = sizeof(u32);
    u64& revMsgLen = msgLen;
    MOCKER_CPP(&SocketAgent::RecvMsg).stubs().with(outBound(msg), outBound(revMsgLen)).will(returnValue(true));

    string agentId{"0"};
    MOCKER_CPP(&RankInfoDetectService::RecvRemoteAgentId)
        .stubs()
        .with(mockcpp::any(), outBound(agentId))
        .will(invoke(RecvRemoteAgentIdStub));

    MOCKER_CPP(&RankInfoDetectService::RecvRemoteRankSize)
        .stubs()
        .with(mockcpp::any(), outBound(rankSize))
        .will(returnValue(true));

    rankInfoDetectService_->GetConnections();

    auto res1 = rankInfoDetectService_->connSockets_.size();
    EXPECT_EQ(1, res1);
}
