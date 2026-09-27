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

#include <cstdio>
#include <cstdlib>

#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>
#include "externalinput_pub.h"
#include "externalinput.h"
#include "adapter_rts.h"
#include "env_config.h"
#include "aicpu_indop_env.h"
#include "config.h"

#define private public
#define protected public
#include "comm_config_pub.h"
#include "comm_configer.h"
#include "hccl_communicator.h"
#include "coll_comm_config.h"
#undef protected
#undef private
using namespace std;
using namespace hccl;

class CommConfigTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--CommConfigTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--CommConfigTest TearDown--\033[0m" << std::endl; }
    virtual void SetUp()
    {
        setenv("HCCL_DFS_CONFIG", "connection_fault_detection_time:0", 1);
        InitEnvParam();
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(CommConfigTest, utCommConfig_load)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");

    EXPECT_EQ(commConfig.GetConfigBufferSize(), 200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE);
    EXPECT_EQ(commConfig.GetConfigDeterministic(), 0);

    HcclCommConfig userConfig;
    HcclCommConfigInit(&userConfig);

    userConfig.hcclBufferSize = 300;
    userConfig.hcclDeterministic = 1;
    userConfig.hcclChannelSqDepth = 128;
    strcpy_s(userConfig.hcclCommName, COMM_NAME_MAX_LENGTH, "Comm1");

    HcclResult ret = commConfig.Load(&userConfig);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(commConfig.GetConfigBufferSize(), 300 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE);
    EXPECT_EQ(commConfig.GetConfigDeterministic(), 1);
    EXPECT_EQ(commConfig.GetConfigCommName(), "Comm1");
    EXPECT_EQ(commConfig.GetConfigSqDepth(), 128U);

    CommConfig legacyCommConfig("legacy_comm_ID");
    HcclCommConfig legacyUserConfig;
    HcclCommConfigInit(&legacyUserConfig);
    auto* legacyConfigInfo = reinterpret_cast<CommConfigInfo*>(legacyUserConfig.reserved);
    legacyConfigInfo->version = CommConfigVersion::COMM_CONFIG_VERSION_TEN;
    legacyUserConfig.hcclChannelSqDepth = 128U;

    ret = legacyCommConfig.Load(&legacyUserConfig);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(legacyCommConfig.GetConfigSqDepth(), HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET);
    GlobalMockObject::verify();
}

#if 0

TEST_F(CommConfigTest, utCommConfig_magicword_verify)
{
    MOCKER(GetExternalInputCCLBuffSize)
    .stubs()
    .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic)
    .stubs()
    .will(returnValue(false));

    CommConfig commConfig("comm_ID");
    CommConfigInfo configInfo = { sizeof(CommConfigHandle), COMM_CONFIG_MAGIC_WORD, 1, { 0 } };
    CommConfigHandle configHandle = { configInfo, 200, 0 };

    HcclResult ret = commConfig.CheckMagicWord(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    configHandle.info = { sizeof(configHandle), 0, 1, { 0 } };
    ret = commConfig.CheckMagicWord(configHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

#endif

TEST_F(CommConfigTest, utCommConfig_version_compatibility_v0)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");
    CommConfigInfo configInfo = {sizeof(CommConfigHandle), COMM_CONFIG_MAGIC_WORD, 0, {0}};
    CommConfigHandle configHandle = {configInfo, 300, 1};

    HcclResult ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(commConfig.GetConfigBufferSize(), 200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE);
    EXPECT_EQ(commConfig.GetConfigDeterministic(), 0);
    EXPECT_EQ(commConfig.GetConfigCommName(), "comm_ID");
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, utCommConfig_version_compatibility_v1)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");
    CommConfigInfo configInfo = {sizeof(CommConfigHandle), COMM_CONFIG_MAGIC_WORD, 1, {0}};
    CommConfigHandle configHandle = {configInfo, 300, 1, "comm_ID", "should_not_be_loaded", 0, 132, 4};

    HcclResult ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(commConfig.GetConfigBufferSize(), 300 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE);
    EXPECT_EQ(commConfig.GetConfigDeterministic(), 1);
    EXPECT_EQ(commConfig.GetConfigCommName(), "comm_ID");
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, utCommConfig_default_env_config)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");
    CommConfigInfo configInfo = {sizeof(CommConfigHandle), COMM_CONFIG_MAGIC_WORD, 1, {0}};
    CommConfigHandle configHandle
        = {configInfo, HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET, HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET};

    HcclResult ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(commConfig.GetConfigBufferSize(), 200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE);
    EXPECT_EQ(commConfig.GetConfigDeterministic(), 0);
    GlobalMockObject::verify();
}

ExternalInput g_externalInput;

TEST_F(CommConfigTest, utCommConfig_op_expansion)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    CommConfig commConfig("comm_ID");
    CommConfigInfo configInfo = {sizeof(CommConfigHandle), COMM_CONFIG_MAGIC_WORD, 4, {0}};
    CommConfigHandle configHandle = {configInfo, 300, 1, "comm_ID", "Unspecified", 3, 132, 4};

    HcclResult ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(commConfig.GetConfigAivMode(), true);
    EXPECT_EQ(commConfig.GetConfigDeterministic(), 1);
    EXPECT_EQ(configHandle.info.version, 4);
    EXPECT_EQ(configHandle.opExpansionMode, 3);

    configHandle.opExpansionMode = 1;
    ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    configHandle.opExpansionMode = 2;
    ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    configHandle.opExpansionMode = 3;
    ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    configHandle.opExpansionMode = 4;
    ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(commConfig.GetConfigIsOnlyAivMode(), true);

    g_externalInput.aicpuUnfold = false;
    ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, utCommConfig_op_expansion_v0)
{
    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    CommConfig commConfig("comm_ID");
    CommConfigInfo configInfo = {sizeof(CommConfigHandle), COMM_CONFIG_MAGIC_WORD, 4, {0}};
    CommConfigHandle configHandle = {configInfo, 300, 1, "comm_ID", "Unspecified", 3, 132, 4};
    g_externalInput.hcclDeterministic == true;
    configHandle.opExpansionMode = 3;
    configHandle.deterministic = 1;
    HcclResult ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(commConfig.GetConfigAivMode(), true);
    EXPECT_EQ(commConfig.GetConfigDeterministic(), 1);
    ret = commConfig.SetConfigOpExpansionMode(configHandle);
    configHandle.opExpansionMode = 0;
    ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    configHandle.opExpansionMode = 1;
    ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    configHandle.opExpansionMode = 2;
    ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    configHandle.opExpansionMode = 999;
    ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, utCommConfig_deterministic_strcit)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministicV2).stubs().will(returnValue(0));

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    CommConfig commConfig("comm_ID");
    CommConfigInfo configInfo = {sizeof(CommConfigHandle), COMM_CONFIG_MAGIC_WORD, 1, {0}};
    CommConfigHandle configHandle = {configInfo, 300, 2, "comm_ID", "should_not_be_loaded", 0, 132, 4};

    HcclResult ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(commConfig.GetConfigBufferSize(), 300 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE);
    EXPECT_EQ(commConfig.GetConfigDeterministic(), 2);
    EXPECT_EQ(commConfig.GetConfigCommName(), "comm_ID");
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, Ut_GetAicpuUnfoldConfig_When_SetConfigOpExpansionMode_Aicpu_A3_ReturnIsHCCL_SUCCESS)
{
    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    CommConfig commConfig("comm_ID");
    CommConfigInfo configInfo = {sizeof(CommConfigHandle), COMM_CONFIG_MAGIC_WORD, 4, {0}};
    CommConfigHandle configHandle = {configInfo, 300, 1, "comm_ID", "Unspecified", 3, 132, 4};
    configHandle.opExpansionMode = 2;
    HcclResult ret = commConfig.SetConfigOpExpansionMode(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    RankTable_t rankTable;
    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(2);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 2;
    rankTable.serverNum = 2;
    aclrtSetDevice(0);

    HcclRtStream opStream;
    rtStream_t stream;
    HcclCommunicator communicator(commConfig);
    bool flag = communicator.GetAicpuUnfoldConfig();
    EXPECT_EQ(flag, true);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, utCommConfig_deterministic_strcit_910_93_success)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministicV2).stubs().will(returnValue(0));

    // 确定性计算配置为规约保序支持A2/A3场景
    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    CommConfig commConfig("comm_ID");
    CommConfigInfo configInfo = {sizeof(CommConfigHandle), COMM_CONFIG_MAGIC_WORD, 1, {0}};
    CommConfigHandle configHandle = {configInfo, 300, 2, "comm_ID", "should_not_be_loaded", 0, 132, 4};

    HcclResult ret = commConfig.SetConfigByVersion(configHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(commConfig.GetConfigDeterministic(), 2);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, CheckRankIpFamily_ValidIPv4_Success)
{
    std::vector<RankInfo_t> rankList(2);

    HcclIpAddress ip1(0x7f000001);
    rankList[0].hostIp = ip1;
    rankList[0].serverId = "server1";

    HcclIpAddress ip2(0x7f000002);
    rankList[1].hostIp = ip2;
    rankList[1].serverId = "server2";

    HcclResult ret = CheckRankIpFamily(rankList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommConfigTest, CheckRankIpFamily_ValidIPv6_Success)
{
    std::vector<RankInfo_t> rankList(2);

    HcclIpAddress ip1("::1");
    rankList[0].hostIp = ip1;
    rankList[0].serverId = "server1";

    HcclIpAddress ip2("::2");
    rankList[1].hostIp = ip2;
    rankList[1].serverId = "server2";

    HcclResult ret = CheckRankIpFamily(rankList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommConfigTest, CheckRankIpFamily_InvalidHostIpFamily_ReturnParaError)
{
    std::vector<RankInfo_t> rankList(1);

    HcclIpAddress ip1(0x7f000001);
    rankList[0].hostIp = ip1;
    rankList[0].serverId = "server1";

    rankList[0].hostIp.family = 100;

    HcclResult ret = CheckRankIpFamily(rankList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CommConfigTest, CheckRankIpFamily_InconsistentHostIpFamily_ReturnParaError)
{
    std::vector<RankInfo_t> rankList(2);

    HcclIpAddress ip1(0x7f000001);
    rankList[0].hostIp = ip1;
    rankList[0].serverId = "server1";

    HcclIpAddress ip2("::1");
    rankList[1].hostIp = ip2;
    rankList[1].serverId = "server2";

    HcclResult ret = CheckRankIpFamily(rankList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CommConfigTest, CheckRankIpFamily_InvalidDeviceIpFamily_ReturnParaError)
{
    std::vector<RankInfo_t> rankList(1);

    HcclIpAddress ip1(0x7f000001);
    rankList[0].hostIp = ip1;
    rankList[0].serverId = "server1";

    HcclIpAddress deviceIp(0x7f000002);
    deviceIp.family = 100;
    rankList[0].deviceInfo.deviceIp.push_back(deviceIp);

    HcclResult ret = CheckRankIpFamily(rankList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CommConfigTest, CheckRankIpFamily_InconsistentDeviceIpFamily_ReturnParaError)
{
    std::vector<RankInfo_t> rankList(2);

    HcclIpAddress ip1(0x7f000001);
    rankList[0].hostIp = ip1;
    rankList[0].serverId = "server1";

    HcclIpAddress deviceIp1(0x7f000002);
    rankList[0].deviceInfo.deviceIp.push_back(deviceIp1);

    HcclIpAddress ip2(0x7f000003);
    rankList[1].hostIp = ip2;
    rankList[1].serverId = "server2";

    HcclIpAddress deviceIp2("::1");
    rankList[1].deviceInfo.deviceIp.push_back(deviceIp2);

    HcclResult ret = CheckRankIpFamily(rankList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(CommConfigTest, CheckRankIpFamily_EmptyDeviceIp_Success)
{
    std::vector<RankInfo_t> rankList(1);

    HcclIpAddress ip1(0x7f000001);
    rankList[0].hostIp = ip1;
    rankList[0].serverId = "server1";

    HcclResult ret = CheckRankIpFamily(rankList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommConfigTest, CheckRankIpFamily_InvalidDeviceIp_Success)
{
    std::vector<RankInfo_t> rankList(1);

    HcclIpAddress ip1(0x7f000001);
    rankList[0].hostIp = ip1;
    rankList[0].serverId = "server1";

    HcclIpAddress invalidIpDevice;
    rankList[0].deviceInfo.deviceIp.push_back(invalidIpDevice);

    HcclResult ret = CheckRankIpFamily(rankList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommConfigTest, Check_taskexception_enable)
{
    bool taskExceptionEnable = true;
    hcomm::SetTaskExceptionEnable(taskExceptionEnable);
    EXPECT_EQ(hcomm::GetTaskExceptionEnable(), taskExceptionEnable);
}

TEST_F(CommConfigTest, ApplyHcclCommConfig_NullConfig_ReturnSuccess)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");
    uint32_t opExpansionMode = 999;
    HcclResult ret = ApplyHcclCommConfig(nullptr, commConfig, opExpansionMode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(opExpansionMode, 0u);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, ApplyHcclCommConfig_ValidConfig_ReturnSuccess)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclOpExpansionMode = 3;
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;

    uint32_t opExpansionMode = 0;
    HcclResult ret = ApplyHcclCommConfig(&config, commConfig, opExpansionMode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(opExpansionMode, 3u);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, ApplyHcclCommConfig_InvalidTrafficClass_ReturnParaError)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclRdmaTrafficClass = 3;

    uint32_t opExpansionMode = 0;
    HcclResult ret = ApplyHcclCommConfig(&config, commConfig, opExpansionMode);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, ApplyHcclCommConfig_InvalidServiceLevel_ReturnParaError)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclRdmaTrafficClass = 0;
    config.hcclRdmaServiceLevel = 8;

    uint32_t opExpansionMode = 0;
    HcclResult ret = ApplyHcclCommConfig(&config, commConfig, opExpansionMode);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, ApplyHcclCommConfig_ValidTrafficClassAndServiceLevel_ReturnSuccess)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclRdmaTrafficClass = 4;
    config.hcclRdmaServiceLevel = 3;

    uint32_t opExpansionMode = 0;
    HcclResult ret = ApplyHcclCommConfig(&config, commConfig, opExpansionMode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(commConfig.GetConfigTrafficClass(), 4u);
    EXPECT_EQ(commConfig.GetConfigServiceLevel(), 3u);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, ApplyHcclCommConfig_InvalidQos_ReturnParaError)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;

    typedef struct {
        size_t size;
        uint32_t magicWord;
        uint32_t version;
        uint64_t reserved;
    } configInfo_t;
    configInfo_t* info = (configInfo_t*)&config;
    info->version = 10;
    config.hcclQos = 8;

    uint32_t opExpansionMode = 0;
    HcclResult ret = ApplyHcclCommConfig(&config, commConfig, opExpansionMode);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, ApplyHcclCommConfig_ValidQos_ReturnSuccess)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;

    typedef struct {
        size_t size;
        uint32_t magicWord;
        uint32_t version;
        uint64_t reserved;
    } configInfo_t;
    configInfo_t* info = (configInfo_t*)&config;
    info->version = 10;
    config.hcclQos = 5;

    uint32_t opExpansionMode = 0;
    HcclResult ret = ApplyHcclCommConfig(&config, commConfig, opExpansionMode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(commConfig.GetConfigHcclQos(), 5u);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, ApplyHcclCommConfig_QosVersionBelow10_SkipQos)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;

    typedef struct {
        size_t size;
        uint32_t magicWord;
        uint32_t version;
        uint64_t reserved;
    } configInfo_t;
    configInfo_t* info = (configInfo_t*)&config;
    info->version = 9;
    config.hcclQos = 5;

    uint32_t opExpansionMode = 0;
    HcclResult ret = ApplyHcclCommConfig(&config, commConfig, opExpansionMode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, ApplyHcclCommConfig_TcNotMultipleOf4_ReturnParaError)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclRdmaTrafficClass = 5;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;

    uint32_t opExpansionMode = 0;
    HcclResult ret = ApplyHcclCommConfig(&config, commConfig, opExpansionMode);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, ApplyHcclCommConfig_QosNotSetWithHighVersion_ReturnSuccess)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;

    typedef struct {
        size_t size;
        uint32_t magicWord;
        uint32_t version;
        uint64_t reserved;
    } configInfo_t;
    configInfo_t* info = (configInfo_t*)&config;
    info->version = 10;
    config.hcclQos = HCCL_COMM_QOS_CONFIG_NOT_SET;

    uint32_t opExpansionMode = 0;
    HcclResult ret = ApplyHcclCommConfig(&config, commConfig, opExpansionMode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, SetConfigExecTimeout_NotSet_Expect_UseEnvDefault)
{
    MOCKER(GetExternalInputCCLBuffSize)
        .stubs()
        .will(returnValue(static_cast<u64>(200 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE)));

    MOCKER(GetExternalInputHcclDeterministic).stubs().will(returnValue(false));

    CommConfig commConfig("comm_ID");
    CommConfigHandle config;
    config.execTimeOut = COMM_EXECTIMEOUT_CONFIG_NOT_SET;

    HcclResult ret = commConfig.SetConfigExecTimeout(config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(commConfig.execTimeOutSetByConfig_, false);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, CommConfiger_GetCommConfigRetryHoldTime_EmptyIdentifier_Expect_Default)
{
    auto& configer = CommConfiger::GetInstance();
    u32 result = configer.GetCommConfigRetryHoldTime("");
    u32 expected = GetExternalInputRetryHoldTime();
    EXPECT_EQ(result, expected);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, CommConfiger_GetCommConfigRetryHoldTime_NotFoundIdentifier_Expect_Default)
{
    auto& configer = CommConfiger::GetInstance();
    u32 result = configer.GetCommConfigRetryHoldTime("nonexistent_identifier_test");
    u32 expected = GetExternalInputRetryHoldTime();
    EXPECT_EQ(result, expected);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, CommConfiger_GetCommConfigRetryHoldTime_ValidIdentifier_Expect_ConfigValue)
{
    auto& configer = CommConfiger::GetInstance();
    const std::string identifier = "test_retry_hold_time_identifier";
    CommConfig config("test_comm");
    configer.SetCommConfig(config, identifier);

    u32 result = configer.GetCommConfigRetryHoldTime(identifier);
    u32 expected = config.GetConfigRetryHoldTime();
    EXPECT_EQ(result, expected);

    configer.UnRegisterToCommConfiger(identifier);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, CommConfiger_GetCommConfigRetryIntervalTime_EmptyIdentifier_Expect_Default)
{
    auto& configer = CommConfiger::GetInstance();
    u32 result = configer.GetCommConfigRetryIntervalTime("");
    u32 expected = GetExternalInputRetryIntervalTime();
    EXPECT_EQ(result, expected);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, CommConfiger_GetCommConfigRetryIntervalTime_NotFoundIdentifier_Expect_Default)
{
    auto& configer = CommConfiger::GetInstance();
    u32 result = configer.GetCommConfigRetryIntervalTime("nonexistent_identifier_test");
    u32 expected = GetExternalInputRetryIntervalTime();
    EXPECT_EQ(result, expected);
    GlobalMockObject::verify();
}

TEST_F(CommConfigTest, CommConfiger_GetCommConfigRetryIntervalTime_ValidIdentifier_Expect_ConfigValue)
{
    auto& configer = CommConfiger::GetInstance();
    const std::string identifier = "test_retry_interval_identifier";
    CommConfig config("test_comm");
    configer.SetCommConfig(config, identifier);

    u32 result = configer.GetCommConfigRetryIntervalTime(identifier);
    u32 expected = config.GetConfigRetryIntervalTime();
    EXPECT_EQ(result, expected);

    configer.UnRegisterToCommConfiger(identifier);
    GlobalMockObject::verify();
}
