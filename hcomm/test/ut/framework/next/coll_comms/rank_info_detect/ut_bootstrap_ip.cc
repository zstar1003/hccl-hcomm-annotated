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
#include <cstdlib>
#include <string>

#define private public
#include "rank_info_detect.h"
#include "host_socket_handle_manager.h"
#include "socket.h"
#include "whitelist.h"
#include "hccp_peer_manager.h"
#include "dev_type.h"
#include "orion_adapter_rts.h"
#include "invalid_params_exception.h"
#include "host_ip_not_found_exception.h"
#include "null_ptr_exception.h"
#include "internal_exception.h"
#include "env_config/env_config_v2.h"
#include "env_func.h"
#include "bootstrap_ip.h"
#undef private
#include "env_config_stub.h"
#include "env_config/env_config_v2.h"

using namespace std;
using namespace Hccl;

namespace Hccl {
std::vector<IpAddress> GetHostSocketWhitelist();
}

namespace {
EnvHostNicConfig MakeHostNicConfig(bool whitelistDisable, bool setIfFields = true)
{
    EnvHostNicConfig fakeEnvConfig;
    fakeEnvConfig.whitelistDisable = CfgField<bool>{"HCCL_WHITELIST_DISABLE", whitelistDisable, CastBin2Bool};
    fakeEnvConfig.whitelistDisable.isParsed = true;
    fakeEnvConfig.hcclWhiteListFile = CfgField<std::string>{"HCCL_WHITELIST_FILE", "unused", Str2T<std::string>};
    fakeEnvConfig.hcclWhiteListFile.isParsed = true;
    if (setIfFields) {
        fakeEnvConfig.hcclIfIp.isParsed = true;
        fakeEnvConfig.hcclSocketIfName.isParsed = true;
    }
    return fakeEnvConfig;
}

std::vector<IpAddress> ReturnLocalHostWhitelist() { return {IpAddress("127.0.0.1")}; }
} // namespace

class GetBootstrapIpTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GetBootstrapIpTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "GetBootstrapIpTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in GetBootstrapIpTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in GetBootstrapIpTest TearDown" << std::endl;
    }
};

TEST_F(GetBootstrapIpTest, Ut_GetBootstrapIp_When_hostIfInfos_Empty_Expect_THROW)
{
    // when
    std::vector<std::pair<std::string, IpAddress>> hostIfInfos;
    MOCKER(HrtGetHostIf).stubs().with(mockcpp::any()).will(returnValue(hostIfInfos));

    // check
    EXPECT_THROW(GetBootstrapIp(1), InternalException);
}

TEST_F(GetBootstrapIpTest, Ut_GetAllValidHostIfInfos_When_ifInfos_Empty_Expect_THROW)
{
    // when
    std::vector<std::pair<std::string, IpAddress>> hostIfInfos;
    hostIfInfos.push_back(std::make_pair("lo", IpAddress("0.0.0.0")));
    MOCKER(HrtGetHostIf).stubs().with(mockcpp::any()).will(returnValue(hostIfInfos));
    auto fakeEnvConfig = MakeHostNicConfig(false, false);
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));
    MOCKER(GetHostSocketWhitelist).stubs().with().will(invoke(ReturnLocalHostWhitelist));

    // check
    EXPECT_THROW(GetBootstrapIp(2), InternalException);
}

TEST_F(GetBootstrapIpTest, Ut_FindLocalHostIp_When_ifInfos_lo_Expect_Right_Ip)
{
    // when
    std::vector<std::pair<std::string, IpAddress>> hostIfInfos;
    hostIfInfos.push_back(std::make_pair("lo", IpAddress("127.0.0.1")));
    MOCKER(HrtGetHostIf).stubs().with(mockcpp::any()).will(returnValue(hostIfInfos));
    auto fakeEnvConfig = MakeHostNicConfig(false);
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));
    MOCKER(GetHostSocketWhitelist).stubs().with().will(invoke(ReturnLocalHostWhitelist));

    // check
    EXPECT_EQ(GetBootstrapIp(3), IpAddress("127.0.0.1"));

    // 已获取过则直接使用ip
    EXPECT_EQ(GetBootstrapIp(3), IpAddress("127.0.0.1"));
}

TEST_F(GetBootstrapIpTest, Ut_FindLocalHostIp_When_ifInfos_docker_Expect_Right)
{
    // when
    std::vector<std::pair<std::string, IpAddress>> hostIfInfos;
    hostIfInfos.push_back(std::make_pair("docker", IpAddress("127.0.0.1")));
    MOCKER(HrtGetHostIf).stubs().with(mockcpp::any()).will(returnValue(hostIfInfos));
    auto fakeEnvConfig = MakeHostNicConfig(false);
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));
    MOCKER(GetHostSocketWhitelist).stubs().with().will(invoke(ReturnLocalHostWhitelist));

    // check
    EXPECT_EQ(GetBootstrapIp(4), IpAddress("127.0.0.1"));
}

TEST_F(GetBootstrapIpTest, Ut_FindLocalHostIp_When_ifInfos_normal_Expect_Right)
{
    // when
    std::vector<std::pair<std::string, IpAddress>> hostIfInfos;
    hostIfInfos.push_back(std::make_pair("normal", IpAddress("127.0.0.1")));
    MOCKER(HrtGetHostIf).stubs().with(mockcpp::any()).will(returnValue(hostIfInfos));
    auto fakeEnvConfig = MakeHostNicConfig(false);
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));
    MOCKER(GetHostSocketWhitelist).stubs().with().will(invoke(ReturnLocalHostWhitelist));

    // check
    EXPECT_EQ(GetBootstrapIp(5), IpAddress("127.0.0.1"));
}

TEST_F(GetBootstrapIpTest, Ut_FindLocalHostIp_When_ifInfos_error_Expect_Right)
{
    // when
    std::vector<std::pair<std::string, IpAddress>> hostIfInfos;
    MOCKER(HrtGetHostIf).stubs().with(mockcpp::any()).will(returnValue(hostIfInfos));
    auto fakeEnvConfig = MakeHostNicConfig(true);
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));

    // check
    EXPECT_THROW(GetBootstrapIp(6), InternalException);
}

TEST_F(GetBootstrapIpTest, Ut_FindLocalHostIp_When_Config_HCCL_IF_IP_Expect_Right_Ip)
{
    // when
    std::vector<std::pair<std::string, IpAddress>> hostIfInfos;
    hostIfInfos.push_back(std::make_pair("lo", IpAddress("127.0.0.2")));
    MOCKER(HrtGetHostIf).stubs().with(mockcpp::any()).will(returnValue(hostIfInfos));
    auto fakeEnvConfig = MakeHostNicConfig(true);
    fakeEnvConfig.hcclIfIp = CfgField<IpAddress>{"HCCL_IF_IP", IpAddress("127.0.0.2"), Str2T<IpAddress>};
    fakeEnvConfig.hcclIfIp.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));

    // check
    EXPECT_EQ(GetBootstrapIp(7), IpAddress("127.0.0.2"));
}

TEST_F(GetBootstrapIpTest, Ut_FindLocalHostIp_When_Config_HCCL_SOCKET_IFNAME_Expect_Right_Ip)
{
    // when
    std::vector<std::pair<std::string, IpAddress>> hostIfInfos;
    hostIfInfos.push_back(std::make_pair("lo", IpAddress("127.0.0.3")));
    MOCKER(HrtGetHostIf).stubs().with(mockcpp::any()).will(returnValue(hostIfInfos));
    auto fakeEnvConfig = MakeHostNicConfig(true);
    fakeEnvConfig.hcclSocketIfName = CfgField<SocketIfName>{
        "HCCL_SOCKET_IFNAME", SocketIfName(std::vector<std::string>{"lo"}, false, false), CastSocketIfName};
    fakeEnvConfig.hcclSocketIfName.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));

    // check
    EXPECT_EQ(GetBootstrapIp(8), IpAddress("127.0.0.3"));
}

TEST_F(GetBootstrapIpTest, Ut_FindLocalHostIp_When_Config_HCCL_SOCKET_IFNAME_Expect_THROW)
{
    // when
    std::vector<std::pair<std::string, IpAddress>> hostIfInfos;
    hostIfInfos.push_back(std::make_pair("lo", IpAddress("127.0.0.3")));
    MOCKER(HrtGetHostIf).stubs().with(mockcpp::any()).will(returnValue(hostIfInfos));
    auto fakeEnvConfig = MakeHostNicConfig(true);
    fakeEnvConfig.hcclSocketIfName = CfgField<SocketIfName>{
        "HCCL_SOCKET_IFNAME", SocketIfName(std::vector<std::string>{"po"}, false, false), CastSocketIfName};
    fakeEnvConfig.hcclSocketIfName.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));

    // check
    EXPECT_THROW(GetBootstrapIp(9), InternalException);
}
