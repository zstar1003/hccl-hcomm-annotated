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
#include "hccl_common.h"
#include "preempt_port_manager.h"
#undef private
#undef protected

using namespace hccl;
using namespace std;

class HcclPreemptPortManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclPreemptPortManagerTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "HcclPreemptPortManagerTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() {}
    virtual void TearDown() { GlobalMockObject::verify(); }
};

TEST_F(HcclPreemptPortManagerTest, ut_GetInstance_unnormal_case)
{
    {
        PreemptPortManager ppm;
        std::vector<HcclSocketPortRange> portRange;
        HcclSocketPortRange range = {50000, 50000};
        portRange.push_back(range);
        ppm.GetRangeStr(portRange);
    }
    PreemptPortManager& ppm1 = PreemptPortManager::GetInstance(HOST_DEVICE_ID);
    PreemptPortManager& ppm2 = PreemptPortManager::GetInstance(MAX_MODULE_DEVICE_NUM);
    PreemptPortManager& ppm = PreemptPortManager::GetInstance(0);
}

TEST_F(HcclPreemptPortManagerTest, ut_ListenPreempt)
{
    MOCKER_CPP(&PreemptPortManager::PreemptPortInRange).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    PreemptPortManager& ppm = PreemptPortManager::GetInstance(0);

    HcclIpAddress remoteIp{};
    std::shared_ptr<HcclSocket> listenSocket(
        new (std::nothrow) HcclSocket("my tag", nullptr, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    std::vector<HcclSocketPortRange> portRangeList;
    HcclSocketPortRange range = {50000, 50005};
    portRangeList.push_back(range);
    u32 usePort = 50000;
    HcclResult ret;
    ret = ppm.ListenPreempt(listenSocket, portRangeList, usePort);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclPreemptPortManagerTest, ut_PreemptPortInRange_reuse)
{
    MOCKER_CPP(&HcclSocket::Listen, HcclResult(HcclSocket::*)(u32 port))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclIpAddress remoteIp{"10.10.10.10"};
    HcclIpAddress localIp{"10.10.10.01"};
    std::vector<HcclSocketPortRange> portRangeList;
    HcclSocketPortRange range = {50000, 50005};
    portRangeList.push_back(range);
    u32 usePort = 50000;
    std::shared_ptr<HcclSocket> listenSocket(
        new (std::nothrow) HcclSocket("my tag", nullptr, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    listenSocket->localIp_ = localIp;

    PreemptPortManager& ppm = PreemptPortManager::GetInstance(0);

    IpPortRef hostPortRef;
    hostPortRef.insert({localIp.GetReadableAddress(), std::make_pair(50000, Referenced())});

    HcclResult ret;
    ret = ppm.PreemptPortInRange(hostPortRef, listenSocket, NICDeployment::NIC_DEPLOYMENT_HOST, portRangeList, usePort);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclPreemptPortManagerTest, ut_PreemptPortInRange_nouse1)
{ // PreemptPortInRange nouse success
    MOCKER_CPP(&HcclSocket::Listen, HcclResult(HcclSocket::*)(u32 port))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclIpAddress remoteIp{"10.10.10.10"};
    HcclIpAddress localIp{"10.10.10.01"};
    std::vector<HcclSocketPortRange> portRange;
    HcclSocketPortRange range = {50000, 50000};
    portRange.push_back(range);
    u32 usePort = 50000;
    std::shared_ptr<HcclSocket> listenSocket(
        new (std::nothrow) HcclSocket("my tag", nullptr, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));

    listenSocket->localPort_ = 50000;
    listenSocket->localIp_ = localIp;

    PreemptPortManager& ppm = PreemptPortManager::GetInstance(0);

    IpPortRef hostPortRef;
    hostPortRef.insert({"10.10.10.02", std::make_pair(5000, Referenced())});

    HcclResult ret
        = ppm.PreemptPortInRange(hostPortRef, listenSocket, NICDeployment::NIC_DEPLOYMENT_HOST, portRange, usePort);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclPreemptPortManagerTest, ut_PreemptPortInRange_nouse2)
{ // PreemptPortInRange nouse HCCL_E_MEMORY
    MOCKER_CPP(&HcclSocket::Listen, HcclResult(HcclSocket::*)(u32 port))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_PARA));

    HcclIpAddress remoteIp{"10.10.10.10"};

    std::vector<HcclSocketPortRange> portRange;
    HcclSocketPortRange range = {50000, 50000};
    portRange.push_back(range);
    u32 usePort = 50000;
    std::shared_ptr<HcclSocket> listenSocket(
        new (std::nothrow) HcclSocket("my tag", nullptr, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    HcclIpAddress localIp{"10.10.10.01"};
    listenSocket->localIp_ = localIp;

    PreemptPortManager& ppm = PreemptPortManager::GetInstance(0);

    IpPortRef hostPortRef;
    hostPortRef.insert({"10.10.10.02", std::make_pair(5000, Referenced())});

    HcclResult ret;
    ret = ppm.PreemptPortInRange(hostPortRef, listenSocket, NICDeployment::NIC_DEPLOYMENT_HOST, portRange, usePort);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

TEST_F(HcclPreemptPortManagerTest, ut_PreemptPortInRange_nouse3)
{
    MOCKER_CPP(&HcclSocket::Listen, HcclResult(HcclSocket::*)(u32 port))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_UNAVAIL));

    HcclIpAddress remoteIp{"10.10.10.10"};
    HcclIpAddress localIp{"10.10.10.1"};
    std::vector<HcclSocketPortRange> portRange;
    HcclSocketPortRange range = {50000, 50000};
    portRange.push_back(range);
    u32 usePort = 50000;
    std::shared_ptr<HcclSocket> listenSocket(
        new (std::nothrow) HcclSocket("my tag", nullptr, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    listenSocket->localIp_ = localIp;

    PreemptPortManager& ppm = PreemptPortManager::GetInstance(0);

    IpPortRef hostPortRef;
    hostPortRef.insert({"10.10.10.02", std::make_pair(5000, Referenced())});

    HcclResult ret;
    ret = ppm.PreemptPortInRange(hostPortRef, listenSocket, NICDeployment::NIC_DEPLOYMENT_HOST, portRange, usePort);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
    GlobalMockObject::verify();
}

TEST_F(HcclPreemptPortManagerTest, ut_IsAlreadyListening)
{
    IpPortRef hostPortRef;
    hostPortRef.insert({"10.10.10.02", std::make_pair(5000, Referenced())});
    PreemptPortManager& ppm = PreemptPortManager::GetInstance(0);
    bool ret = true;
    ret = ppm.IsAlreadyListening(hostPortRef, "10.10.10.02", 5000);
    EXPECT_EQ(ret, false);
}

TEST_F(HcclPreemptPortManagerTest, ut_ReleasePreempt)
{
    MOCKER_CPP(&PreemptPortManager::IsAlreadyListening).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&HcclSocket::DeInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclIpAddress remoteIp{"10.10.10.10"};
    std::shared_ptr<HcclSocket> listenSocket(
        new (std::nothrow) HcclSocket("my tag", nullptr, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    HcclIpAddress localIp{"10.10.10.1"};
    listenSocket->localIp_ = localIp;
    listenSocket->localPort_ = 50000;

    IpPortRef hostPortRef;
    Referenced ref;
    ref.refCount = 1;
    hostPortRef.insert({localIp.GetReadableAddress(), std::make_pair(5000, ref)});
    HcclResult ret = HCCL_SUCCESS;
    PreemptPortManager& ppm = PreemptPortManager::GetInstance(0);
    ret = ppm.ReleasePreempt(hostPortRef, listenSocket, NICDeployment::NIC_DEPLOYMENT_HOST);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclPreemptPortManagerTest, ut_Release)
{
    MOCKER_CPP(&PreemptPortManager::ReleasePreempt).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclIpAddress remoteIp{"10.10.10.10"};
    std::shared_ptr<HcclSocket> listenSocket(
        new (std::nothrow) HcclSocket("my tag", nullptr, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    HcclResult ret = HCCL_SUCCESS;
    PreemptPortManager& ppm = PreemptPortManager::GetInstance(0);
    ret = ppm.Release(listenSocket);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}
