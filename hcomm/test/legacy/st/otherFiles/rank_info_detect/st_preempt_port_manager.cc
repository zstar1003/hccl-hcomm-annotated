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

#include <string>
#define private public
#define protected public
#include "hccl_common_v2.h"
#include "preempt_port_manager_v2.h"
#undef private
#undef protected

using namespace Hccl;
using namespace std;

class HcclPreemptPortManagerV2Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclPreemptPortManagerV2Test SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "HcclPreemptPortManagerV2Test TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() {}

    virtual void TearDown() { GlobalMockObject::verify(); }

    PreemptPortManager& ppm = PreemptPortManager::GetInstance(0);
    PreemptPortManager& ppm1 = PreemptPortManager::GetInstance(-1);
    PreemptPortManager& ppm2 = PreemptPortManager::GetInstance(MAX_MODULE_DEVICE_NUM);
};

TEST_F(HcclPreemptPortManagerV2Test, St_GetRangeStr_When_InputValue_Expect_NO_THROW)
{
    // when
    std::vector<SocketPortRange> portRange;
    SocketPortRange range = {50000, 50000};
    portRange.push_back(range);

    // check
    EXPECT_NO_THROW(ppm.GetRangeStr(portRange));
}

TEST_F(HcclPreemptPortManagerV2Test, St_ListenPreempt_When_InputValue_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&PreemptPortManager::PreemptPortInRange).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.01"};
    std::vector<SocketPortRange> portRangeList;
    SocketPortRange range = {50000, 50005};
    portRangeList.push_back(range);
    u32 usePort = 50000;
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    // check
    EXPECT_NO_THROW(ppm.ListenPreempt(listenSocket, portRangeList, usePort));
}

TEST_F(HcclPreemptPortManagerV2Test, St_PreemptPortInRange_When_Exist_IP_Expect_Use_Occupied)
{
    // when
    MOCKER_CPP(&Socket::Listen, bool(Socket::*)(u32 & port)).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.01"};
    std::vector<SocketPortRange> portRangeList;
    SocketPortRange range = {50000, 50005};
    portRangeList.push_back(range);
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    u32 usePort = 0;

    // check
    IpPortRef hostPortRef;
    hostPortRef.insert({localIp.GetIpStr().c_str(), std::make_pair(50000, Referenced())});
    ppm.preemptSockets_[HrtNetworkMode::PEER] = hostPortRef;
    EXPECT_NO_THROW(ppm.PreemptPortInRange(listenSocket, HrtNetworkMode::PEER, portRangeList, usePort));
    EXPECT_EQ(usePort, 50000);
}

TEST_F(HcclPreemptPortManagerV2Test, St_PreemptPortInRange_When_New_IP_Expect_HCCL_E_PARA)
{
    // when
    MOCKER_CPP(&Socket::Listen, bool(Socket::*)(u32 & port)).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.01"};
    std::vector<SocketPortRange> portRange;
    SocketPortRange range = {50000, 50005};
    portRange.push_back(range);
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    u32 usePort = 0;

    // check
    ppm.preemptSockets_[HrtNetworkMode::PEER] = IpPortRef();
    EXPECT_THROW(
        ppm.PreemptPortInRange(listenSocket, HrtNetworkMode::PEER, portRange, usePort), InvalidParamsException);
}

TEST_F(HcclPreemptPortManagerV2Test, St_IsAlreadyListening_When_Ref_0_Expect_false)
{
    // then
    IpPortRef hostPortRef;
    hostPortRef.insert({"10.10.10.02", std::make_pair(5000, Referenced())});

    // check
    EXPECT_EQ(ppm.IsAlreadyListening(hostPortRef, "10.10.10.02", 5000), false);
}

TEST_F(HcclPreemptPortManagerV2Test, St_IsAlreadyListening_When_Right_Expect_true)
{
    // then
    Referenced ref;
    ref.Ref();
    IpPortRef hostPortRef;
    hostPortRef.insert({"10.10.10.02", std::make_pair(5000, ref)});

    // check
    EXPECT_EQ(ppm.IsAlreadyListening(hostPortRef, "10.10.10.02", 5000), true);
}

TEST_F(HcclPreemptPortManagerV2Test, St_ReleasePreempt_When_InputValue_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&PreemptPortManager::IsAlreadyListening).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.1"};
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    IpPortRef hostPortRef;
    Referenced ref;
    ref.refCount = 1;
    hostPortRef.insert({localIp.GetIpStr().c_str(), std::make_pair(5000, ref)});

    // check
    EXPECT_NO_THROW(ppm.ReleasePreempt(hostPortRef, listenSocket, HrtNetworkMode::PEER));
    EXPECT_EQ(hostPortRef.find(localIp.GetIpStr().c_str()), hostPortRef.end());
}

TEST_F(HcclPreemptPortManagerV2Test, St_ReleasePreempt_When_Ref_Count_2_Expect_NO_ERASE)
{
    // when
    MOCKER_CPP(&PreemptPortManager::IsAlreadyListening).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.1"};
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    IpPortRef hostPortRef;
    Referenced ref;
    ref.refCount = 2;
    hostPortRef.insert({localIp.GetIpStr().c_str(), std::make_pair(5000, ref)});

    // check
    EXPECT_NO_THROW(ppm.ReleasePreempt(hostPortRef, listenSocket, HrtNetworkMode::PEER));
    Referenced& outRef = hostPortRef[localIp.GetIpStr().c_str()].second;
    EXPECT_EQ(outRef.refCount, 1);
}

TEST_F(HcclPreemptPortManagerV2Test, St_ReleasePreempt_When_Ref_Count_ERROR_Expect_THROW)
{
    // when
    MOCKER_CPP(&PreemptPortManager::IsAlreadyListening).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.1"};
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    IpPortRef hostPortRef;
    Referenced ref;
    ref.refCount = 0;
    hostPortRef.insert({localIp.GetIpStr().c_str(), std::make_pair(5000, ref)});

    // check
    EXPECT_THROW(ppm.ReleasePreempt(hostPortRef, listenSocket, HrtNetworkMode::PEER), InvalidParamsException);
}

TEST_F(HcclPreemptPortManagerV2Test, St_Release_When_InputValue_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&PreemptPortManager::ReleasePreempt).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&IpAddress::InitBinaryAddr).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    IpAddress remoteIp{"10.10.10.10"};
    IpAddress localIp{"10.10.10.1"};
    SocketHandle socketHandle;
    std::string tag = "test";
    std::shared_ptr<Socket> listenSocket = std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    // check
    EXPECT_NO_THROW(ppm.Release(listenSocket));
}
