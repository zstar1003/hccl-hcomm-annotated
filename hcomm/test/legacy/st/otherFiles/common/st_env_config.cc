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
#include "env_config_stub.h"

#define private public
#define protected public
#include "env_config_v2.h"
#include "orion_adapter_rts.h"
#include <stdexcept>
#include <climits>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include "invalid_params_exception.h"
#include "env_func.h"
#undef private
#undef protected

using namespace Hccl;

std::map<std::string, std::string> envCfgMap = defaultEnvCfgMap;

char* getenv_stub(const char* __name)
{
    char* ret = const_cast<char*>(envCfgMap[std::string(__name)].c_str());
    return ret;
}

class EnvConfigTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "EnvConfigTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "EnvConfigTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in EnvConfigTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in EnvConfigTest TearDown" << std::endl;
    }

    bool CmpIpAddress(const IpAddress& ip1, const IpAddress& ip2)
    {
        if (ip1.GetFamily() != ip2.GetFamily()) {
            return false;
        }
        if (ip1.GetFamily() == AF_INET) {
            return (ip1.GetBinaryAddress().addr.s_addr == ip2.GetBinaryAddress().addr.s_addr);
        } else {
            auto biAddr1 = ip1.GetBinaryAddress();
            auto biAddr2 = ip2.GetBinaryAddress();
            return (memcmp(&biAddr1.addr6, &biAddr2.addr6, sizeof(biAddr1.addr6)) == 0);
        }
    }

    bool CmpSocketIfName(const SocketIfName& fiName1, const SocketIfName& fiName2)
    {
        return (fiName1.configIfNames == fiName2.configIfNames) && (fiName1.searchNot == fiName2.searchNot)
               && (fiName1.searchExact == fiName2.searchExact);
    }

    void ResetEnvCfgMap()
    {
        envCfgMap.clear();
        envCfgMap = defaultEnvCfgMap;
    }

    void MockFunc()
    {
        MOCKER(getenv).stubs().with(mockcpp::any()).will(invoke(getenv_stub));

        char c = '1';
        MOCKER(realpath).stubs().with(mockcpp::any()).will(returnValue(&c));

        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A));
    }

private:
    void GenFile(const std::string& filePath, const std::string fileContent)
    {
        try {
            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << fileContent;
        } catch (...) {
            std::cout << filePath << " generate failed!" << std::endl;
            return;
        }
        std::cout << filePath << " generated." << std::endl;
    }

    void DelFile(const std::string& filePath)
    {
        int res = unlink(filePath.c_str());
        if (res == -1) {
            std::cout << filePath << " delete failed!" << std::endl;
            return;
        }
        std::cout << filePath << " deleted." << std::endl;
    }
};

TEST_F(EnvConfigTest, parse_env_config)
{
    // 使用真实的单例EnvConfig，提升覆盖率
    // 由于是单例，只能使用一个测试用例
    ResetEnvCfgMap();

    MockFunc();

    EnvConfig::GetInstance().GetHostNicConfig();
    EnvConfig::GetInstance().GetSocketConfig();
    EnvConfig::GetInstance().GetRtsConfig();
    EnvConfig::GetInstance().GetRdmaConfig();
    EnvConfig::GetInstance().GetAlgoConfig();
    EnvConfig::GetInstance().GetLogConfig();
    EnvConfig::GetInstance().GetDetourConfig();
}

TEST_F(EnvConfigTest, parse_env_config_should_success)
{
    ResetEnvCfgMap();

    MockFunc();

    try {
        EnvConfigStub envCfg;
        EXPECT_EQ(CmpIpAddress(envCfg.GetHostNicConfig().GetControlIfIp(), IpAddress("10.10.10.1")), true);
        EXPECT_EQ(envCfg.GetHostNicConfig().GetIfBasePort(), 50000);
        EXPECT_EQ(envCfg.GetHostNicConfig().GetWhitelistDisable(), false);
        EXPECT_EQ(envCfg.GetHostNicConfig().GetWhiteListFile(), "");
        EXPECT_EQ(envCfg.GetSocketConfig().GetSocketFamily(), AF_INET6);
        EXPECT_EQ(envCfg.GetSocketConfig().GetLinkTimeOut(), 200);
        EXPECT_EQ(envCfg.GetRtsConfig().GetExecTimeOut(), 1768);
        EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaTrafficClass(), 100);
        EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaServerLevel(), 3);
        EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaTimeOut(), 6);
        EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaRetryCnt(), 5);
        EXPECT_EQ(envCfg.GetAlgoConfig().GetPrimQueueGenName(), "AllReduceRing");
        std::map<OpType, std::vector<HcclAlgoType>> algoMap
            = {{OpType::ALLREDUCE,
                {HcclAlgoType::HCCL_ALGO_TYPE_NA, HcclAlgoType::HCCL_ALGO_TYPE_RING,
                 HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT}}};
        EXPECT_EQ(envCfg.GetAlgoConfig().GetAlgoConfig(), algoMap);
        EXPECT_EQ(envCfg.GetAlgoConfig().GetBuffSize(), 200 * 1024 * 1024);
        EXPECT_EQ(envCfg.GetLogConfig().GetEntryLogEnable(), true);
        EXPECT_EQ(envCfg.GetLogConfig().GetCannVersion(), "");
        EXPECT_EQ(envCfg.GetDetourConfig().GetDetourType(), HcclDetourType::HCCL_DETOUR_ENABLE_2P);
    } catch (...) {
    }
}
/*
TEST_F(EnvConfigTest, parse_env_config_should_success2)
{
    ResetEnvCfgMap();
    envCfgMap["HCCL_NPU_NET_PROTOCOL"] = "TCP";
    envCfgMap["HCCL_SOCKET_FAMILY"] = "AF_INET";
    envCfgMap["LD_LIBRARY_PATH"] = "/temp:/latest";

    MOCKER(getenv)
    .stubs()
    .with(mockcpp::any())
    .will(invoke(getenv_stub));

    char c = '1';
    MOCKER(realpath)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(&c));

    MOCKER(HrtGetDeviceType)
    .stubs()
    .will(returnValue((DevType)DevType::DEV_TYPE_910A3));

    EnvConfigStub envCfg;

    EXPECT_EQ(CmpIpAddress(envCfg.GetHostNicConfig().GetControlIfIp(), IpAddress("10.10.10.1")), true);
    EXPECT_EQ(envCfg.GetHostNicConfig().GetIfBasePort(), 50000);
    EXPECT_EQ(CmpSocketIfName(envCfg.GetHostNicConfig().GetSocketIfName(), SocketIfName({{"eth0", "endvnic"}, true,
true})), true); EXPECT_EQ(envCfg.GetHostNicConfig().GetWhitelistDisable(), false);
    EXPECT_EQ(envCfg.GetHostNicConfig().GetWhiteListFile(), "");
    EXPECT_EQ(envCfg.GetSocketConfig().GetSocketFamily(), AF_INET);
    EXPECT_EQ(envCfg.GetSocketConfig().GetLinkTimeOut(), 200);
    EXPECT_EQ(envCfg.GetRtsConfig().GetExecTimeOut(), 1800);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaTrafficClass(), 100);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaServerLevel(), 3);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaTimeOut(), 6);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaRetryCnt(), 5);
    EXPECT_EQ(envCfg.GetAlgoConfig().GetPrimQueueGenName(), "AllReduceRing");
    EXPECT_EQ(envCfg.GetAlgoConfig().GetAlgoConfig(), vector<HcclAlgoType>({HcclAlgoType::HCCL_ALGO_TYPE_RING,
HcclAlgoType::HCCL_ALGO_TYPE_RING, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT}));
    EXPECT_EQ(envCfg.GetAlgoConfig().GetBuffSize(), 200*1024*1024);
    EXPECT_EQ(envCfg.GetAlgoConfig().GetOpExpansionMode(), OpExpansionMode::AI_CPU);
    EXPECT_EQ(envCfg.GetLogConfig().GetEntryLogEnable(), true);
    EXPECT_EQ(envCfg.GetLogConfig().GetCannVersion(), "");
}

TEST_F(EnvConfigTest, parse_env_config_should_success3)
{
    ResetEnvCfgMap();
    envCfgMap["HCCL_NPU_NET_PROTOCOL"] = "";
    envCfgMap["HCCL_ALGO"] = "";

    MockFunc();

    EnvConfigStub envCfg;

    EXPECT_EQ(CmpIpAddress(envCfg.GetHostNicConfig().GetControlIfIp(), IpAddress("10.10.10.1")), true);
    EXPECT_EQ(envCfg.GetHostNicConfig().GetIfBasePort(), 50000);
    EXPECT_EQ(CmpSocketIfName(envCfg.GetHostNicConfig().GetSocketIfName(), SocketIfName({{"eth0", "endvnic"}, true,
true})), true); EXPECT_EQ(envCfg.GetHostNicConfig().GetWhitelistDisable(), false);
    EXPECT_EQ(envCfg.GetHostNicConfig().GetWhiteListFile(), "");
    EXPECT_EQ(envCfg.GetSocketConfig().GetSocketFamily(), AF_INET6);
    EXPECT_EQ(envCfg.GetSocketConfig().GetLinkTimeOut(), 200);
    EXPECT_EQ(envCfg.GetRtsConfig().GetExecTimeOut(), 1768);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaTrafficClass(), 100);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaServerLevel(), 3);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaTimeOut(), 6);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaRetryCnt(), 5);
    EXPECT_EQ(envCfg.GetAlgoConfig().GetPrimQueueGenName(), "AllReduceRing");
    EXPECT_EQ(envCfg.GetAlgoConfig().GetAlgoConfig(), vector<HcclAlgoType>({HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT,
HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT}));
    EXPECT_EQ(envCfg.GetAlgoConfig().GetBuffSize(), 200*1024*1024);
    EXPECT_EQ(envCfg.GetAlgoConfig().GetOpExpansionMode(), OpExpansionMode::AI_CPU);
    EXPECT_EQ(envCfg.GetLogConfig().GetEntryLogEnable(), true);
    EXPECT_EQ(envCfg.GetLogConfig().GetCannVersion(), "");
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_SOCKET_IFNAME_should_fail)
{
    ResetEnvCfgMap();
    envCfgMap["HCCL_SOCKET_IFNAME"] = "^=eth0,,endvnic";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_INTRA_PCIE_ENABLE_should_fail)
{
    ResetEnvCfgMap();
    envCfgMap["HCCL_INTRA_PCIE_ENABLE"] = "true";
    envCfgMap["HCCL_INTRA_ROCE_ENABLE"] = "true";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_NPU_NET_PROTOCOL_should_fail)
{
    ResetEnvCfgMap();
    char proto[NPU_NET_PROTOCOL_MAX_LEN + 1] = {};
    std::fill_n(proto, NPU_NET_PROTOCOL_MAX_LEN, 1);
    // longer than NPU_NET_PROTOCOL_MAX_LEN
    envCfgMap["HCCL_NPU_NET_PROTOCOL"] = std::string(proto);

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_NPU_NET_PROTOCOL_proto_should_fail)
{
    ResetEnvCfgMap();
    // not TCP or RDMA
    envCfgMap["HCCL_NPU_NET_PROTOCOL"] = "PCIE";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_ALGO_should_fail)
{
    ResetEnvCfgMap();
    envCfgMap["HCCL_ALGO"] = ":NA;level1:";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_ALGO_level_should_fail)
{
    ResetEnvCfgMap();
    // not defined level
    envCfgMap["HCCL_ALGO"] = "level0:NA;level4:ring";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_ALGO_algo_should_fail)
{
    ResetEnvCfgMap();
    // not defined algo
    envCfgMap["HCCL_ALGO"] = "level0:somealgo;level3:ring";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_ALGO_duplicate_level_should_fail)
{
    ResetEnvCfgMap();
    // duplicate level
    envCfgMap["HCCL_ALGO"] = "level0:NA;level0:ring";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_ALGO_level_num_should_fail)
{
    ResetEnvCfgMap();
    // too many levels
    envCfgMap["HCCL_ALGO"] = "level0:NA;level0:NA;level0:NA;level0:NA;level0:NA";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_SOCKET_FAMILY_should_fail)
{
    ResetEnvCfgMap();
    envCfgMap["HCCL_SOCKET_FAMILY"] = "AF_INET4";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_WHITELIST_FILE_should_fail)
{
    ResetEnvCfgMap();
    char path[PATH_MAX + 1] = {};
    std::fill_n(path, PATH_MAX, 1);
    envCfgMap["HCCL_WHITELIST_FILE"] = std::string(path);

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}
*/

// 临时方案
TEST_F(EnvConfigTest, parse_env_config_socketIFName_test)
{
    std::string input = "=eth0,endvnic";
    EXPECT_NO_THROW(CastSocketIfName(input));
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_DETOUR_test)
{
    std::string input = "detour:0";
    EXPECT_EQ(CastDetourType(input), HcclDetourType::HCCL_DETOUR_DISABLE);
    input = "detour:1";
    EXPECT_EQ(CastDetourType(input), HcclDetourType::HCCL_DETOUR_ENABLE_2P);
    input = "detour:2";
    EXPECT_THROW(CastDetourType(input), NotSupportException);
    input = "detour:3";
    EXPECT_THROW(CastDetourType(input), NotSupportException);
    input = "!";
    EXPECT_THROW(CastDetourType(input), NotSupportException);
}

TEST_F(EnvConfigTest, Ut_CastSocketPortRange_When_Config_Auto_Expect_Right)
{
    std::vector<SocketPortRange> rangs;
    SocketPortRange autoSocketPortRange = {HCCL_SOCKET_PORT_RANGE_AUTO, HCCL_SOCKET_PORT_RANGE_AUTO};
    rangs.push_back(autoSocketPortRange);
    EXPECT_EQ(CastSocketPortRange(HCCL_AUTO_PORT_CONFIG, "envName"), rangs);
}

TEST_F(EnvConfigTest, Ut_CastSocketPortRange_When_Config_Whitespace_Expect_Erase_Return_OK)
{
    std::vector<SocketPortRange> rangs;
    SocketPortRange autoSocketPortRange = {60000, 60050};
    rangs.push_back(autoSocketPortRange);
    EXPECT_EQ(CastSocketPortRange(" 60000-60050 ", "envName"), rangs);
}

TEST_F(EnvConfigTest, Ut_CastSocketPortRange_When_Config_More_Expect_Return_OK)
{
    std::vector<SocketPortRange> rangs;
    rangs.push_back(SocketPortRange{50000, 50000});
    rangs.push_back(SocketPortRange{60000, 60050});
    rangs.push_back(SocketPortRange{60100, 60260});
    EXPECT_EQ(CastSocketPortRange("50000,60000-60050, 60100-60260", "envName"), rangs);
}

TEST_F(EnvConfigTest, Ut_CastSocketPortRange_When_Config_Bound_Error_Expect_Throw)
{
    EXPECT_THROW(CastSocketPortRange("50000,60050-60000, 60100-60260", "envName"), InvalidParamsException);
    EXPECT_THROW(CastSocketPortRange("50000,60000-60150,60100-60260", "envName"), InvalidParamsException);
}

TEST_F(EnvConfigTest, Ut_CastSocketPortRange_When_Config_Invalid_Expect_Throw)
{
    EXPECT_THROW(CastSocketPortRange("60000-60050,0,60100-60260", "envName"), InvalidParamsException);
    EXPECT_THROW(CastSocketPortRange("50000,60000-60050,0-0", "envName"), InvalidParamsException);
    EXPECT_THROW(CastSocketPortRange("65536", "envName"), InvalidParamsException);
}

TEST_F(EnvConfigTest, test_parse_HCCL_DFS_CONFIG_shoud_return_SUCCESS)
{
    EXPECT_EQ(CastDfsConfig("").taskExceptionEnable, true);
    EXPECT_EQ(CastDfsConfig("task_exception:off").taskExceptionEnable, false);
    EXPECT_EQ(CastDfsConfig("task_exception:on").taskExceptionEnable, true);
    EXPECT_EQ(CastDfsConfig(" ").taskExceptionEnable, true);
    EXPECT_EQ(CastDfsConfig("task_exception:off ").taskExceptionEnable, false);
    EXPECT_EQ(CastDfsConfig(" task_exception : off ").taskExceptionEnable, false);
}

TEST_F(EnvConfigTest, test_parse_HCCL_DFS_CONFIG_task_shoud_exception)
{
    EXPECT_THROW(CastDfsConfig("!"), InvalidParamsException);
    EXPECT_THROW(CastDfsConfig(":"), InvalidParamsException);
    EXPECT_THROW(CastDfsConfig(","), InvalidParamsException);
    EXPECT_THROW(CastDfsConfig("task_exception:off,"), InvalidParamsException);
    EXPECT_THROW(CastDfsConfig(",task_exception:off"), InvalidParamsException);
    EXPECT_THROW(CastDfsConfig(",task_exception,:,off,"), InvalidParamsException);
    EXPECT_THROW(CastDfsConfig("task_exception : off, "), InvalidParamsException);
    EXPECT_THROW(CastDfsConfig("task_exception : off, "), InvalidParamsException);
    EXPECT_THROW(
        CastDfsConfig("task_exception:on,connection_fault_detction_time:30,cluster_heartbeat:on,"
                      "stuck_detection:on"),
        InvalidParamsException);
    EXPECT_THROW(CastDfsConfig("hccl_task_exception:on"), InvalidParamsException);
}
