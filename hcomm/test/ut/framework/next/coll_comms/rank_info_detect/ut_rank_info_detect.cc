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
#include <stdexcept>
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
#include "internal_exception.h"
#include "network_api_exception.h"
#include "null_ptr_exception.h"
#include "env_config/env_config_v2.h"
#include "env_func.h"
#include "bootstrap_ip.h"
#include "preempt_port_manager_v2.h"
#include "rank_info_detect.h"
#undef private
#include "env_config_stub.h"
#include "env_config/env_config_v2.h"

using namespace std;
using namespace Hccl;

namespace Hccl {
std::vector<IpAddress> GetHostSocketWhitelist();
}

namespace {
std::vector<IpAddress> ReturnLocalHostWhitelist() { return {IpAddress("127.0.0.1")}; }

void FillHostWhiteList(Whitelist* whitelist, std::vector<IpAddress>& whiteList)
{
    (void)whitelist;
    whiteList = ReturnLocalHostWhitelist();
}

} // namespace

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
        MOCKER(HrtGetDevice).stubs().with().will(returnValue(0));
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<DevId>(0)));
        MOCKER(HrtRaInit).stubs().with(mockcpp::any()).will(ignoreReturnValue());
        MOCKER(HrtRaDeInit).stubs().with(mockcpp::any()).will(ignoreReturnValue());
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((Hccl::DevType)Hccl::DevType::DEV_TYPE_950));
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

    void MockSetupServerWhitelistResult(HcclResult ret)
    {
        MOCKER(GetBootstrapIp).stubs().with(mockcpp::any()).will(returnValue(IpAddress("127.0.0.1")));
        MOCKER_CPP(&RankInfoDetect::GetHostListenPort).stubs().with().will(returnValue(60000));
        MOCKER_CPP(&RankInfoDetect::ServerInit).stubs().with().will(returnValue(std::shared_ptr<Socket>{}));
        MOCKER_CPP(&RankInfoDetect::GetRootHandle).stubs().with(mockcpp::any()).will(ignoreReturnValue());
        MOCKER_CPP(&RankInfoDetect::GetHandleAndAddHostSocketWhitelist).stubs().with().will(returnValue(ret));
    }

    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle;
    std::string tag = "test";
};

TEST_F(RankInfoDetectTest, Ut_SetupServer_When_Invalid_Ip_Expect_THROW)
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

TEST_F(RankInfoDetectTest, Ut_SetupServer_When_WhitelistReturnsEPtr_Expect_NullPtrException)
{
    MockSetupServerWhitelistResult(HCCL_E_PTR);

    RankInfoDetect rankInfoDetect;
    HcclRootHandleV2 rootHandle{};
    EXPECT_THROW(rankInfoDetect.SetupServer(rootHandle), NullPtrException);
}

TEST_F(RankInfoDetectTest, Ut_SetupServer_When_WhitelistReturnsENetwork_Expect_NetworkApiException)
{
    MockSetupServerWhitelistResult(HCCL_E_NETWORK);

    RankInfoDetect rankInfoDetect;
    HcclRootHandleV2 rootHandle{};
    EXPECT_THROW(rankInfoDetect.SetupServer(rootHandle), NetworkApiException);
}

TEST_F(RankInfoDetectTest, Ut_SetupServer_When_WhitelistReturnsEInternal_Expect_InternalException)
{
    MockSetupServerWhitelistResult(HCCL_E_INTERNAL);

    RankInfoDetect rankInfoDetect;
    HcclRootHandleV2 rootHandle{};
    EXPECT_THROW(rankInfoDetect.SetupServer(rootHandle), InternalException);
}

TEST_F(RankInfoDetectTest, Ut_GetHostSocketHandle_When_WhitelistEnabled_Expect_DeferTagConfiguration)
{
    // when
    MOCKER(HrtGetDeviceCount).stubs().with().will(returnValue(8));
    MOCKER(HrtRaSocketWhiteListAdd).stubs().with(mockcpp::any(), mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtRaSocketSetWhiteListStatus).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    EnvHostNicConfig envConfig;
    EnvHostNicConfig& fakeEnvConfig = envConfig;
    fakeEnvConfig.whitelistDisable = CfgField<bool>{"HCCL_WHITELIST_DISABLE", false, CastBin2Bool};
    fakeEnvConfig.whitelistDisable.isParsed = true;
    fakeEnvConfig.hcclWhiteListFile = CfgField<std::string>{"HCCL_WHITELIST_FILE", "unused", Str2T<std::string>};
    fakeEnvConfig.hcclWhiteListFile.isParsed = true;
    fakeEnvConfig.hcclIfIp.isParsed = true;
    fakeEnvConfig.hcclSocketIfName.isParsed = true;
    MOCKER_CPP(&Hccl::EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));
    MOCKER(GetHostSocketWhitelist).stubs().with().will(invoke(ReturnLocalHostWhitelist));
    MOCKER_CPP(&Whitelist::GetHostWhiteList).stubs().with(mockcpp::any()).will(invoke(FillHostWhiteList));

    // check
    EXPECT_NO_THROW(GetBootstrapIp(0));
    shared_ptr<RankInfoDetect> rankInfoDetect = make_shared<RankInfoDetect>();
    HcclRootHandleV2 outRootHandle;
    EXPECT_NO_THROW(rankInfoDetect->GetHostSocketHandle());
    EXPECT_EQ(rankInfoDetect->hostSocketWlist_.size(), 1U);
    EXPECT_TRUE(rankInfoDetect->wlistInfo_.empty());
}

TEST_F(RankInfoDetectTest, Ut_AddHostSocketWhitelist_When_TagInputsReady_Expect_CorrectTag)
{
    RankInfoDetect rankInfoDetect;
    rankInfoDetect.identifier_ = "test_identifier";
    rankInfoDetect.hostPort_ = 60001;
    std::vector<IpAddress> whitelist{IpAddress("127.0.0.1")};

    MOCKER(HrtRaSocketWhiteListAdd).stubs().with(mockcpp::any(), mockcpp::any()).will(ignoreReturnValue());
    EXPECT_EQ(rankInfoDetect.AddHostSocketWhitelist(socketHandle, whitelist), HCCL_SUCCESS);
    ASSERT_EQ(rankInfoDetect.wlistInfo_.size(), 1U);
    EXPECT_EQ(rankInfoDetect.wlistInfo_[0].tag, "rank_info_detect_default_tag_test_identifier_60001");
}

TEST_F(RankInfoDetectTest, Ut_AddHostSocketWhitelist_When_HrtThrowsStdException_Expect_E_INTERNAL)
{
    RankInfoDetect rankInfoDetect;
    rankInfoDetect.identifier_ = "test_identifier";
    rankInfoDetect.hostPort_ = 60001;
    std::vector<IpAddress> whitelist{IpAddress("127.0.0.1")};

    MOCKER(HrtRaSocketWhiteListAdd)
        .expects(once())
        .with(mockcpp::any(), mockcpp::any())
        .will(throws(std::runtime_error("add whitelist failed")));
    EXPECT_EQ(rankInfoDetect.AddHostSocketWhitelist(socketHandle, whitelist), HCCL_E_INTERNAL);
}

TEST_F(RankInfoDetectTest, Ut_GetHandleAndAddHostSocketWhitelist_When_WhitelistEmpty_Expect_ReturnDirectly)
{
    RankInfoDetect rankInfoDetect;
    rankInfoDetect.wlistInfo_.push_back(RaSocketWhitelist{});

    MOCKER_CPP(&HostSocketHandleManager::Get).expects(never());
    MOCKER(HrtRaSocketWhiteListAdd).expects(never());

    EXPECT_EQ(rankInfoDetect.GetHandleAndAddHostSocketWhitelist(), HCCL_SUCCESS);
    EXPECT_TRUE(rankInfoDetect.wlistInfo_.empty());
}

TEST_F(RankInfoDetectTest, Ut_GetHandleAndAddHostSocketWhitelist_When_SocketHandleNull_Expect_E_PTR)
{
    RankInfoDetect rankInfoDetect;
    rankInfoDetect.devPhyId_ = 0;
    rankInfoDetect.hostIp_ = IpAddress("127.0.0.1");
    rankInfoDetect.identifier_ = "test_identifier";
    rankInfoDetect.hostPort_ = 60001;
    rankInfoDetect.hostSocketWlist_ = {IpAddress("127.0.0.1")};
    rankInfoDetect.wlistInfo_.push_back(RaSocketWhitelist{});

    SocketHandle nullHandle = nullptr;
    MOCKER_CPP(&HostSocketHandleManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(nullHandle));
    MOCKER(HrtRaSocketWhiteListAdd).expects(never());

    EXPECT_EQ(rankInfoDetect.GetHandleAndAddHostSocketWhitelist(), HCCL_E_PTR);
    EXPECT_TRUE(rankInfoDetect.wlistInfo_.empty());
}

TEST_F(RankInfoDetectTest, Ut_ClientInit_When_Input_Expect_NO_THROW)
{
    // check
    RankInfoDetect rankInfoDetect{};
    HcclRootHandleV2 handle{};
    string ip = "1.1.1.1";
    memcpy_s(handle.ip, ip.size(), ip.c_str(), ip.size());
    EXPECT_NO_THROW(rankInfoDetect.ClientInit(handle));
}

TEST_F(RankInfoDetectTest, Ut_ServerInit_When_Invalid_Port_Expect_ListenPreempt)
{
    // when
    MOCKER_CPP(&PreemptPortManager::ListenPreempt)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(throws(InternalException("aaa")));

    // check
    shared_ptr<RankInfoDetect> rankInfoDetect = make_shared<RankInfoDetect>();
    rankInfoDetect->hostPort_ = Hccl::HCCL_INVALID_PORT;
    EXPECT_THROW(rankInfoDetect->ServerInit(), std::exception);
}

TEST_F(RankInfoDetectTest, Ut_SetupAgent_When_Input_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDetectClient::Setup).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&RankInfoDetectClient::TearDown).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // check
    RankInfoDetect rankInfoDetect;
    HcclRootHandleV2 rootHandle;
    EXPECT_NO_THROW(rankInfoDetect.SetupAgent(0, 0, rootHandle));
}

TEST_F(RankInfoDetectTest, Ut_SetupRankInfoDetectService_When_Input_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDetectService::Setup).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // check
    RankInfoDetect rankInfoDetect;
    HcclRootHandleV2 rootHandle;
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE);
    EXPECT_NO_THROW(rankInfoDetect.SetupRankInfoDetectService(ErrContextPub{}, socket, 0, 0, "test", {}));
}

TEST_F(RankInfoDetectTest, Ut_SetupRankInfoDetectService_When_Setup_Fail_Expect_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDetectService::Setup).stubs().with(mockcpp::any()).will(throws(InternalException("aaa")));

    // check
    RankInfoDetect rankInfoDetect;
    HcclRootHandleV2 rootHandle;
    u32 listenPort = 50;
    std::shared_ptr<Socket> socket = std::make_shared<Socket>(
        socketHandle, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE);

    EXPECT_NO_THROW(rankInfoDetect.SetupRankInfoDetectService(ErrContextPub{}, socket, 0, 0, "test", {}));
    EXPECT_EQ(rankInfoDetect.g_detectServerStatus_.Find(listenPort).first->second, RANKINFO_DETECT_SERVER_STATUS_ERROR);
}

TEST_F(RankInfoDetectTest, Ut_GetHostListenPort_When_Input_Expect_NO_THROW)
{
    // check
    RankInfoDetect rankInfoDetect;
    EXPECT_EQ(rankInfoDetect.GetHostListenPort(), Hccl::HCCL_INVALID_PORT);
}

TEST_F(RankInfoDetectTest, Ut_ServerInit_When_AutoPort_Expect_Right)
{
    MOCKER_CPP(&Socket::Listen, bool(Socket::*)(u32 & port))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(false)) // 第1次调用: 60000 端口失败
        .then(returnValue(true)); // 第2次调用: 60001 端口成功

    // Mock Socket::GetListenPort 返回实际监听端口
    MOCKER_CPP(&Socket::GetListenPort).stubs().will(returnValue(static_cast<u32>(60001)));

    RankInfoDetect rankInfoDetect;
    u32 listenPort = rankInfoDetect.GetHostListenPort();

    EXPECT_EQ(listenPort, Hccl::HCCL_INVALID_PORT);

    rankInfoDetect.hostPort_ = listenPort;
    shared_ptr<Socket> serverSocket;
    EXPECT_NO_THROW(serverSocket = rankInfoDetect.ServerInit());

    EXPECT_TRUE(serverSocket != nullptr);
    EXPECT_EQ(rankInfoDetect.hostPort_, 60001);
}

TEST_F(RankInfoDetectTest, Ut_ServerInit_When_AllDefaultPortsOccupied_Expect_THROW)
{
    MOCKER_CPP(&Socket::Listen, bool(Socket::*)(u32 & port)).stubs().with(mockcpp::any()).will(returnValue(false));

    RankInfoDetect rankInfoDetect;
    u32 listenPort = rankInfoDetect.GetHostListenPort();
    EXPECT_EQ(listenPort, Hccl::HCCL_INVALID_PORT);
    rankInfoDetect.hostPort_ = listenPort;
    EXPECT_THROW(rankInfoDetect.ServerInit(), InvalidParamsException);
}

TEST_F(RankInfoDetectTest, Ut_GetHostListenPort_When_Config_PORT_RANGE_Expect_Right)
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
    MOCKER_CPP(&Hccl::EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));

    // check
    RankInfoDetect rankInfoDetect;
    EXPECT_EQ(rankInfoDetect.GetHostListenPort(), Hccl::HCCL_INVALID_PORT);
}

TEST_F(RankInfoDetectTest, Ut_GetHostListenPort_When_Config_PORT_BASE_Expect_Right)
{
    // when
    EnvHostNicConfig envConfig;
    EnvHostNicConfig& fakeEnvConfig = envConfig;
    fakeEnvConfig.hcclIfBasePort = CfgField<u32>{"HCCL_IF_BASE_PORT", 40000, Str2T<u32>};
    fakeEnvConfig.hcclIfBasePort.isParsed = true;
    fakeEnvConfig.hcclHostSocketPortRange.isParsed = true;
    MOCKER_CPP(&Hccl::EnvConfig::GetHostNicConfig).stubs().will(returnValue(fakeEnvConfig));

    // check
    RankInfoDetect rankInfoDetect;
    EXPECT_EQ(rankInfoDetect.GetHostListenPort(), 40000);
}

TEST_F(RankInfoDetectTest, Ut_GetRootHandle_When_Input_Expect_NO_THROW)
{
    // check
    RankInfoDetect rankInfoDetect;
    HcclRootHandleV2 rootHandle;
    EXPECT_NO_THROW(rankInfoDetect.GetRootHandle(rootHandle));
}

TEST_F(RankInfoDetectTest, Ut_WaitComplete_When_Input_Expect_Right)
{
    // when
    const u32 RANKINFO_DETECT_SERVER_STATUS_IDLE = 0;
    const u32 RANKINFO_DETECT_SERVER_STATUS_RUNING = 1;
    RankInfoDetect::g_detectServerStatus_[5000] = RANKINFO_DETECT_SERVER_STATUS_ERROR;
    RankInfoDetect::g_detectServerStatus_[6000] = RANKINFO_DETECT_SERVER_STATUS_IDLE;
    RankInfoDetect::g_detectServerStatus_[4000] = RANKINFO_DETECT_SERVER_STATUS_RUNING;

    // check
    RankInfoDetect rankInfoDetect;
    HcclRootHandleV2 rootHandle;
    EXPECT_THROW(rankInfoDetect.WaitComplete(5000, RANKINFO_DETECT_SERVER_STATUS_IDLE), InternalException);
    EXPECT_NO_THROW(rankInfoDetect.WaitComplete(6000, RANKINFO_DETECT_SERVER_STATUS_IDLE));

    // when
    EnvSocketConfig envConfig;
    EnvSocketConfig& fakeEnvConfig = envConfig;
    fakeEnvConfig.linkTimeOut = CfgField<s32>{"HCCL_CONNECT_TIMEOUT", s32(0.1), Str2T<s32>};
    fakeEnvConfig.linkTimeOut.isParsed = true;
    MOCKER_CPP(&Hccl::EnvConfig::GetSocketConfig).stubs().will(returnValue(fakeEnvConfig));

    // check
    EXPECT_THROW(rankInfoDetect.WaitComplete(4000, RANKINFO_DETECT_SERVER_STATUS_IDLE), TimeoutException);
}
