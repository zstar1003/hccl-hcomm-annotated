/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <iostream>
#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "externalinput.h"
#include "llt_hccl_stub_pub.h"

#define private public
#include "opretry_connection_pub.h"
#include "opretry_connection.h"

using namespace hccl;

class OpRetryConnTest : public testing::Test {
public:
    static void SetUpTestCase()
    {
        UseRealPortAndName(true);
        std::cout << "OpRetryConnTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "OpRetryConnTest TearDown" << std::endl;
        UseRealPortAndName(false);
    }

    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        std::cout << "A Test TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

struct UTGenWhitelist {
    UTGenWhitelist(const std::string filePath, const std::string whitelist) : f(filePath), wl(whitelist) {}
    ~UTGenWhitelist() { DelFile(); }

    bool GenJson()
    {
        try {
            nlohmann::json rankTableJson = nlohmann::json::parse(wl);
            std::ofstream out(f, std::ofstream::out);
            out << rankTableJson;
        } catch (...) {
            return false;
        }

        return true;
    }

    void DelFile() { unlink(f.c_str()); }

    std::string f;
    std::string wl;
};

TEST_F(OpRetryConnTest, ut_opretry_connection_setget)
{
    bool enable = true;
    OpRetryConnectionPub::SetOpRetryConnEnable(enable);
    EXPECT_EQ(OpRetryConnectionPub::IsOpRetryConnEnable(), enable);

    enable = false;
    OpRetryConnectionPub::SetOpRetryConnEnable(enable);
    EXPECT_EQ(OpRetryConnectionPub::IsOpRetryConnEnable(), enable);

    // 恢复
    OpRetryConnectionPub::SetOpRetryConnEnable(true);
}

TEST_F(OpRetryConnTest, ut_opretry_connection_default_root)
{
    OpRetryConnection conn;
    u32 rankId = 0;
    u32 rankSize = 1;
    u32 serverPort = 50001;
    HcclIpAddress hostIp(std::string("127.0.0.1"));
    HcclIpAddress localIp(std::string("127.0.0.1"));

    EXPECT_EQ(conn.Init(rankId, rankSize, hostIp, serverPort, 0, localIp), HCCL_SUCCESS);
    EXPECT_EQ(conn.DeInit(), HCCL_SUCCESS);
}

TEST_F(OpRetryConnTest, ut_opretry_connection)
{
    OpRetryConnection conn;
    u32 rankId = 0;
    u32 rankSize = 1;
    u32 rootRank = 0;
    u32 serverPort = 50001;
    HcclIpAddress hostIp(std::string("127.0.0.1"));
    HcclIpAddress localIp(std::string("127.0.0.1"));

    EXPECT_EQ(conn.Init(rankId, rankSize, hostIp, serverPort, 0, localIp, rootRank), HCCL_SUCCESS);
    EXPECT_EQ(conn.DeInit(), HCCL_SUCCESS);
}

TEST_F(OpRetryConnTest, ut_invalid_params)
{
    u32 rankId = 2;
    u32 rankSize = 1;
    u32 rootRank = 0;
    u32 serverPort = 50001;
    HcclIpAddress hostIp(std::string("0.0.0.0"));
    std::string groupName = "invalidGroup";

    OpRetryServerInfo serverInfo = {hostIp, 50001, 0};
    OpRetryAgentInfo agentInfo = {rankId, 0, hostIp, hostIp};

    // 关闭特性，所以直接返回OK
    OpRetryConnectionPub::SetOpRetryConnEnable(false);
    EXPECT_EQ(OpRetryConnectionPub::Init(groupName, rankSize, serverInfo, agentInfo, rootRank), HCCL_SUCCESS);

    bool isRoot = false;
    std::shared_ptr<HcclSocket> agent = nullptr;
    std::map<u32, std::shared_ptr<HcclSocket>> server;
    EXPECT_EQ(OpRetryConnectionPub::GetConns(groupName, isRoot, agent, server), HCCL_SUCCESS);
    EXPECT_EQ(agent, nullptr);
    EXPECT_EQ(server.empty(), true);

    OpRetryConnectionPub::SetOpRetryConnEnable(true);
    EXPECT_EQ(OpRetryConnectionPub::Init(groupName, rankSize, serverInfo, agentInfo, rootRank), HCCL_E_PARA);

    OpRetryConnection conn;
    EXPECT_EQ(conn.Init(groupName, rankSize, serverInfo, agentInfo, rootRank), HCCL_E_PARA);
}

TEST_F(OpRetryConnTest, ut_opretry_connection_whitelist)
{
    OpRetryConnection conn;
    u32 rankId = 0;
    u32 rankSize = 1;
    u32 rootRank = 0;
    u32 serverPort = 50001;
    HcclIpAddress hostIp(std::string("127.0.0.1"));
    HcclIpAddress localIp(std::string("127.0.0.1"));

    // 设置白名单
    std::string file = "wl.json";
    setenv("HCCL_WHITELIST_DISABLE", "0", 1);
    setenv("HCCL_WHITELIST_FILE", file.c_str(), 1);
    std::string whitelist = R"(
        { "host_ip":["127.0.0.1"], "device_ip":[]}
    )";

    UTGenWhitelist genWl(file, whitelist);
    ASSERT_EQ(genWl.GenJson(), true);

    ASSERT_EQ(ParseHcclWhitelistSwitch(), HCCL_SUCCESS);
    ASSERT_EQ(ParseHcclWhitelistFilePath(), HCCL_SUCCESS);

    EXPECT_EQ(conn.Init(rankId, rankSize, hostIp, serverPort, 0, localIp, rootRank), HCCL_SUCCESS);
    EXPECT_EQ(conn.DeInit(), HCCL_SUCCESS);

    unsetenv("HCCL_WHITELIST_DISABLE");
    unsetenv("HCCL_WHITELIST_FILE");
    ASSERT_EQ(ParseHcclWhitelistSwitch(), HCCL_SUCCESS);
    ASSERT_EQ(ParseHcclWhitelistFilePath(), HCCL_SUCCESS);
}

TEST_F(OpRetryConnTest, ut_opretry_connection_static_init)
{
    u32 rankId = 0;
    u32 rankSize = 1;
    u32 rootRank = 0;
    HcclIpAddress hostIp(std::string("127.0.0.1"));
    HcclIpAddress localIp(std::string("127.0.0.1"));
    std::string groupName = "Test_group";
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    OpRetryServerInfo serverInfo = {hostIp, 50001, 0};
    OpRetryAgentInfo agentInfo = {rankId, 0, localIp, localIp};

    EXPECT_EQ(OpRetryConnectionPub::Init(groupName, rankSize, serverInfo, agentInfo, rootRank), HCCL_SUCCESS);

    bool isRoot = false;
    std::shared_ptr<HcclSocket> agent = nullptr;
    std::map<u32, std::shared_ptr<HcclSocket>> server;
    EXPECT_EQ(OpRetryConnectionPub::GetConns(groupName, isRoot, agent, server), HCCL_SUCCESS);

    EXPECT_EQ(isRoot, true);
    EXPECT_NE(agent, nullptr);
    size_t expectSeverLink = 1;
    EXPECT_EQ(server.size(), expectSeverLink);

    OpRetryConnectionPub::DeInit(groupName);
}

TEST_F(OpRetryConnTest, ut_opretry_connect_error)
{
    u32 rankId = 0;
    u32 rankSize = 1;
    u32 rootRank = 0;
    HcclIpAddress hostIp(std::string("127.0.0.1"));
    HcclIpAddress localIp(std::string("127.0.0.1"));
    std::string group = "invalidGroup";

    OpRetryServerInfo serverInfo = {hostIp, 50001, 0};
    OpRetryAgentInfo agentInfo = {rankId, 0, localIp, localIp};

    // Client侧重连两次，第一次是重试，第二次直接失败
    MOCKER(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(1));
    MOCKER_CPP(&OpRetryConnection::Connect).stubs().will(returnValue(HCCL_E_AGAIN)).then(returnValue(HCCL_E_INTERNAL));
    EXPECT_NE(OpRetryConnectionPub::Init(group, rankSize, serverInfo, agentInfo, rootRank), HCCL_SUCCESS);
    OpRetryConnectionPub::DeInit(group);
    GlobalMockObject::verify();

    // Connect重试成功，但是后续执行失败
    MOCKER(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(1));
    MOCKER_CPP(&OpRetryConnection::Connect).stubs().will(returnValue(HCCL_E_AGAIN)).then(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryConnection::WaitAcceptFinish).stubs().will(returnValue(HCCL_E_INTERNAL));
    EXPECT_NE(OpRetryConnectionPub::Init(group, rankSize, serverInfo, agentInfo, rootRank), HCCL_SUCCESS);
    OpRetryConnectionPub::DeInit(group);
    GlobalMockObject::verify();

    // mocker内部接口失败
    MOCKER(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(1));
    MOCKER_CPP(&OpRetryConnection::RecvAckInfo)
        .stubs()
        .will(returnValue(HCCL_E_AGAIN))
        .then(returnValue(HCCL_E_INTERNAL));
    EXPECT_NE(OpRetryConnectionPub::Init(group, rankSize, serverInfo, agentInfo, rootRank), HCCL_SUCCESS);
    OpRetryConnectionPub::DeInit(group);
    GlobalMockObject::verify();
}

TEST_F(OpRetryConnTest, ut_opretry_wait_error)
{
    OpRetryConnection op;
    op.backgroudThreadStop_ = true;
    HcclResult ret = op.WaitAcceptFinish();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}
