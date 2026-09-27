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
#include <thread>

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
#include "env_config_v2.h"
#include "env_func.h"
#include "bootstrap_ip.h"
#include "preempt_port_manager_v2.h"
#include "rank_info_detect.h"
#include "env_config_v2.h"
#undef private

using namespace std;
using namespace Hccl;

string whitelistFilePath = "llt/ace/comop/hccl/orion/st/otherFiles/rank_info_detect/whitelist.json";

const u32 RANKINFO_DETECT_SERVER_STATUS_IDLE = 0;
const u32 RANKINFO_DETECT_SERVER_STATUS_RUNING = 1;
const u32 RANKINFO_DETECT_SERVER_STATUS_ERROR = 2;

class RankInfoDetectTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankInfoDetectTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RankInfoDetectTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtRaSocketInit).stubs().with(mockcpp::any(), mockcpp::any()).will(ignoreReturnValue());
        MOCKER_CPP(&HccpPeerManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());
        MOCKER_CPP(&HccpPeerManager::DeInit).stubs().with(mockcpp::any()).will(ignoreReturnValue());
        SocketHandle hostSocketHandle;
        MOCKER_CPP(&HostSocketHandleManager::Create)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(hostSocketHandle));
        MOCKER(HrtRaSocketWhiteListAdd)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(ignoreReturnValue());
        MOCKER(HrtGetDevice).stubs().with().will(returnValue(0));
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<DevId>(0)));
        MOCKER(HrtRaInit).stubs().with(mockcpp::any()).will(ignoreReturnValue());
        MOCKER(HrtRaDeInit).stubs().with(mockcpp::any()).will(ignoreReturnValue());
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));
        std::vector<std::pair<std::string, IpAddress>> hostIfInfos;
        hostIfInfos.push_back(std::make_pair("lo", IpAddress("127.0.0.1")));
        MOCKER(HrtGetHostIf).stubs().with(mockcpp::any()).will(returnValue(hostIfInfos));
        MOCKER(HrtRaSocketTryListenOneStart).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(true));
        MOCKER(HrtGetDeviceCount).stubs().with().will(returnValue(8));
        MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
        MOCKER(HrtResetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
        std::cout << "A Test case in RankInfoDetectTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RankInfoDetectTest TearDown" << std::endl;
    }
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
};

TEST_F(RankInfoDetectTest, St_SetupServer_When_Invalid_Ip_Expect_THROW)
{
    // when
    MOCKER(GetBootstrapIp).stubs().with(mockcpp::any()).will(returnValue(IpAddress()));
    MOCKER_CPP(&RankInfoDetect::GetHostListenPort).stubs().with().will(returnValue(60000));
    SocketHandle socketHandle;
    MOCKER_CPP(&RankInfoDetect::GetHostSocketHandle).stubs().with().will(returnValue(socketHandle));
    MOCKER_CPP(&RankInfoDetect::SetupRankInfoDetectService).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&RankInfoDetect::GetRootHandle).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // check
    shared_ptr<RankInfoDetect> rankInfoDetect = make_shared<RankInfoDetect>();
    HcclRootHandleV2 outRootHandle;
    EXPECT_THROW(rankInfoDetect->SetupServer(outRootHandle), InternalException);
}

TEST_F(RankInfoDetectTest, St_GetHostSocketHandle_When_Whitelist_Expect_AddHost_NO_THROW)
{
    // when
    MOCKER(HrtGetDeviceCount).stubs().with().will(returnValue(8));
    MOCKER(HrtRaSocketWhiteListAdd).stubs().with(mockcpp::any(), mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtRaSocketSetWhiteListStatus).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    EnvHostNicConfig envConfig;
    EnvHostNicConfig& fakeEnvConfig = envConfig;
    fakeEnvConfig.whitelistDisable = CfgField<bool>{"HCCL_WHITELIST_DISABLE", false, CastBin2Bool};
    fakeEnvConfig.whitelistDisable.isParsed = true;
    fakeEnvConfig.hcclWhiteListFile
        = CfgField<std::string>{"HCCL_WHITELIST_FILE", whitelistFilePath, Str2T<std::string>};
    fakeEnvConfig.hcclWhiteListFile.isParsed = true;
    fakeEnvConfig.hcclIfIp.isParsed = true;
    fakeEnvConfig.hcclSocketIfName.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));

    // check
    EXPECT_NO_THROW(GetBootstrapIp(0));
    shared_ptr<RankInfoDetect> rankInfoDetect = make_shared<RankInfoDetect>();
    HcclRootHandleV2 outRootHandle;
    EXPECT_NO_THROW(rankInfoDetect->GetHostSocketHandle());
}

TEST_F(RankInfoDetectTest, St_ClientInit_When_InpSt_Expect_NO_THROW)
{
    // check
    RankInfoDetect rankInfoDetect;
    HcclRootHandleV2 handle{};
    string ip = "1.1.1.1";
    memcpy_s(handle.ip, ip.size(), ip.c_str(), ip.size());
    EXPECT_NO_THROW(rankInfoDetect.ClientInit(handle));
}

TEST_F(RankInfoDetectTest, St_ServerInit_When_Invalid_Port_Expect_ListenPreempt)
{
    // when
    MOCKER_CPP(&PreemptPortManager::ListenPreempt)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(throws(InternalException("aaa")));

    // check
    shared_ptr<RankInfoDetect> rankInfoDetect = make_shared<RankInfoDetect>();
    rankInfoDetect->hostPort_ = HCCL_INVALID_PORT;
    EXPECT_THROW(rankInfoDetect->ServerInit(), InternalException);
}

TEST_F(RankInfoDetectTest, St_SetupAgent_When_InpSt_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDetectClient::Setup).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // check
    RankInfoDetect rankInfoDetect;
    HcclRootHandleV2 rootHandle;
    EXPECT_NO_THROW(rankInfoDetect.SetupAgent(0, 0, rootHandle));
}

TEST_F(RankInfoDetectTest, St_SetupRankInfoDetectService_When_InpSt_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDetectService::Setup).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // check
    RankInfoDetect rankInfoDetect;
    HcclRootHandleV2 rootHandle;
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    EXPECT_NO_THROW(rankInfoDetect.SetupRankInfoDetectService(ErrContextPub{}, socket, 0, 0, "test", {}));
}

TEST_F(RankInfoDetectTest, St_SetupRankInfoDetectService_When_Setup_Fail_Expect_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDetectService::Setup).stubs().with(mockcpp::any()).will(throws(InternalException("aaa")));

    // check
    RankInfoDetect rankInfoDetect;
    HcclRootHandleV2 rootHandle;
    u32 listenPort = 50;
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    EXPECT_NO_THROW(rankInfoDetect.SetupRankInfoDetectService(ErrContextPub{}, socket, 0, 0, "test", {}));
    EXPECT_EQ(rankInfoDetect.g_detectServerStatus_.Find(listenPort).first->second, RANKINFO_DETECT_SERVER_STATUS_ERROR);
}

TEST_F(RankInfoDetectTest, St_GetHostListenPort_When_InpSt_Expect_NO_THROW)
{
    // check
    RankInfoDetect rankInfoDetect;
    EXPECT_EQ(rankInfoDetect.GetHostListenPort(), 60000); // HOST_CONTROL_BASE_PORT
}

TEST_F(RankInfoDetectTest, St_GetHostListenPort_When_Config_PORT_RANGE_Expect_Right)
{
    // when
    EnvHostNicConfig envConfig;
    EnvHostNicConfig& fakeEnvConfig = envConfig;
    std::vector<SocketPortRange> range;
    range.push_back(SocketPortRange{50000, 50001});
    fakeEnvConfig.hcclHostSocketPortRange = CfgField<std::vector<SocketPortRange>>{
        "HCCL_HOST_SOCKET_PORT_RANGE", range, [](const std::string& s) -> std::vector<SocketPortRange> {
            return CastSocketPortRange(s, "HCCL_HOST_SOCKET_PORT_RANGE");
        }};
    fakeEnvConfig.hcclHostSocketPortRange.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));

    // check
    RankInfoDetect rankInfoDetect;
    EXPECT_EQ(rankInfoDetect.GetHostListenPort(), HCCL_INVALID_PORT);
}

TEST_F(RankInfoDetectTest, St_GetHostListenPort_When_Config_PORT_BASE_Expect_Right)
{
    // when
    EnvHostNicConfig envConfig;
    EnvHostNicConfig& fakeEnvConfig = envConfig;
    fakeEnvConfig.hcclIfBasePort = CfgField<u32>{"HCCL_IF_BASE_PORT", 40000, Str2T<u32>};
    fakeEnvConfig.hcclIfBasePort.isParsed = true;
    fakeEnvConfig.hcclHostSocketPortRange.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));

    // check
    RankInfoDetect rankInfoDetect;
    EXPECT_EQ(rankInfoDetect.GetHostListenPort(), 40000);
}

TEST_F(RankInfoDetectTest, St_GetRootHandle_When_InpSt_Expect_NO_THROW)
{
    // check
    RankInfoDetect rankInfoDetect;
    HcclRootHandleV2 rootHandle;
    EXPECT_NO_THROW(rankInfoDetect.GetRootHandle(rootHandle));
}

TEST_F(RankInfoDetectTest, St_WaitComplete_When_InpSt_Expect_Right)
{
    // when
    const u32 RANKINFO_DETECT_SERVER_STATUS_IDLE = 0;
    const u32 RANKINFO_DETECT_SERVER_STATUS_RUNING = 1;
    const u32 RANKINFO_DETECT_SERVER_STATUS_ERROR = 2;
    RankInfoDetect::g_detectServerStatus_[5000] = RANKINFO_DETECT_SERVER_STATUS_ERROR;
    RankInfoDetect::g_detectServerStatus_[6000] = RANKINFO_DETECT_SERVER_STATUS_IDLE;
    RankInfoDetect::g_detectServerStatus_[4000] = RANKINFO_DETECT_SERVER_STATUS_RUNING;

    // check
    RankInfoDetect rankInfoDetect;
    HcclRootHandleV2 rootHandle;
    EXPECT_THROW(rankInfoDetect.WaitComplete(5000), InternalException);
    EXPECT_NO_THROW(rankInfoDetect.WaitComplete(6000));

    // when
    EnvSocketConfig envConfig;
    EnvSocketConfig& fakeEnvConfig = envConfig;
    fakeEnvConfig.linkTimeOut = CfgField<s32>{"HCCL_CONNECT_TIMEOUT", s32(0.1), Str2T<s32>};
    fakeEnvConfig.linkTimeOut.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetSocketConfig).stubs().will(returnValue(fakeEnvConfig));

    // check
    EXPECT_THROW(rankInfoDetect.WaitComplete(4000), TimeoutException);
}
