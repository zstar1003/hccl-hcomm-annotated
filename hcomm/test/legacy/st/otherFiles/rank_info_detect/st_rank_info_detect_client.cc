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
#include "hccp_peer_manager.h"
#include "timeout_exception.h"
#include "orion_adapter_rts.h"
#include "invalid_params_exception.h"
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
#include "rank_info_detect_client.h"
#include "orion_adapter_rts.h"
#include "env_config_v2.h"
#include "base_config_legacy.h"
#include "json_parser.h"
#undef private

using namespace Hccl;

std::string filePath = "llt/ace/comop/hccl/orion/st/otherFiles/rank_info_detect/rootInfo.json";

class RankInfoDetectClientTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankInfoDetectClientTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RankInfoDetectClientTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in RankInfoDetectClientTest SetUP" << std::endl;
        socketHandle = new int(0);
        MOCKER(HrtRaSocketInit).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(socketHandle));
        MOCKER_CPP(&HccpPeerManager::Init).stubs().with(mockcpp::any());
        MOCKER_CPP(&HccpPeerManager::DeInit).stubs().with(mockcpp::any());
        IpAddress serverIp = IpAddress("10.0.0.10");
        u32 hostPort = 60001;
        IpAddress hostIp_ = IpAddress("192.168.1.8");
        u32 rankSize_ = 1;
        u32 devPhyId_ = 0;
        u32 rankId_ = 0;
        std::string clientSocketTag = "rank_info_test_server";

        auto clientSocket_ = std::make_shared<Socket>(
            socketHandle, hostIp_, hostPort, serverIp, clientSocketTag, SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
        SocketAgent socketAgent_ = SocketAgent(clientSocket_.get());
        rankInfoDetectClient_ = new RankInfoDetectClient(devPhyId_, rankSize_, rankId_, clientSocket_);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete socketHandle;
        delete rankInfoDetectClient_;
        std::cout << "A Test case in RankInfoDetectServiceTest TearDown" << std::endl;
    }

    RankInfoDetectClient* rankInfoDetectClient_;
    nlohmann::json presetParseJson_;     // 模拟ParseFileToJson的返回结果
    nlohmann::json presetLocalDevJson_;  // 模拟GetLocalDevInfoJson的返回结果
    nlohmann::json presetRankTableJson_; // 模拟GetLocalRankTableJson的返回结果
    SocketHandle socketHandle;
};

TEST_F(RankInfoDetectClientTest, st_CheckStatus_When_Normal_Expect_Success)
{
    MOCKER_CPP(&Socket::GetStatus).stubs().then(returnValue((SocketStatus)SocketStatus::OK));

    EXPECT_NO_THROW(rankInfoDetectClient_->CheckStatus());
}

TEST_F(RankInfoDetectClientTest, st_SendAgentIdAndRankSize_When_Normal_Expect_Success)
{
    MOCKER(HrtRaSocketBlockSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(true));

    EXPECT_NO_THROW(rankInfoDetectClient_->SendAgentIdAndRankSize());
}

TEST_F(RankInfoDetectClientTest, st_ConstructSingleRank_When_Normal_Expect_Success)
{
    RankTableInfo localRankTable;

    EXPECT_NO_THROW(rankInfoDetectClient_->ConstructSingleRank(localRankTable));

    EXPECT_EQ(localRankTable.version, "2.0");
    EXPECT_EQ(localRankTable.rankCount, 1U);
    EXPECT_EQ(localRankTable.ranks.size(), 1U);
    const NewRankInfo& actualRankInfo = localRankTable.ranks[0];
    EXPECT_EQ(actualRankInfo.rankId, 0);
    EXPECT_EQ(actualRankInfo.rankLevelInfos.size(), 1U);
}

TEST_F(RankInfoDetectClientTest, st_ConstructRankTable_When_Normal_Expect_Success)
{
    rankInfoDetectClient_->rankSize_ = 2;
    RankTableInfo localRankTable;
    std::string testJsonPath = "llt/ace/comop/hccl/orion/st/otherFiles/rank_info_detect/rootInfo.json";

    MOCKER(realpath)
        .stubs()
        .with(mockcpp::any(), outBoundP(const_cast<char*>(testJsonPath.c_str()), testJsonPath.size() + 1))
        .will(returnValue(const_cast<char*>(testJsonPath.c_str())));

    EXPECT_NO_THROW(rankInfoDetectClient_->ConstructRankTable(localRankTable));

    EXPECT_EQ(localRankTable.version, "2.0");
    EXPECT_EQ(localRankTable.rankCount, 2U);
}

TEST_F(RankInfoDetectClientTest, st_RecvRankTable_When_Normal_Expect_Success)
{
    RankTableInfo localRankTable;
    localRankTable.version = "1.0";
    localRankTable.rankCount = 1;
    NewRankInfo rankInfo{};
    rankInfo.rankId = 0;
    rankInfo.rankLevelInfos.emplace_back(RankLevelInfo{});
    localRankTable.ranks.emplace_back(rankInfo);

    BinaryStream binaryStream;
    localRankTable.GetBinStream(true, binaryStream);
    binaryStream << rankInfoDetectClient_->currentStep_;
std:
    string temp = "";
    binaryStream << temp;

    // 字节流转换为vector<char>格式
    vector<char> rankInfoMsg;
    binaryStream.Dump(rankInfoMsg);

    // 取rankInfoMsg的size
    u32 rankInfoSize = rankInfoMsg.size();
    u64 expectLen = rankInfoSize;

    // 取rankInfoMsg的data() （const char *）
    MOCKER(aclrtMallocHostWithCfg).stubs().will(returnValue(1));
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue((void*)rankInfoMsg.data()));
    MOCKER(HrtFreeHost).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    void* msg = rankInfoMsg.data();
    u64 msgLen = rankInfoMsg.size();
    u64& revMsgLen = msgLen;
    MOCKER_CPP(&SocketAgent::RecvMsg).stubs().with(outBound(msg), outBound(revMsgLen)).will(returnValue(true));

    MOCKER_CPP(&RankInfoDetectClient::VerifyRankTable).stubs().will(ignoreReturnValue());

    EXPECT_NO_THROW(rankInfoDetectClient_->RecvRankTable());
}
