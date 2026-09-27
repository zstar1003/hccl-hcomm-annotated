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
#include <sys/time.h>
#define private public
#include "heartbeat.h"
#include "notify_pool_impl.h"
#undef private
#include "llt_hccl_stub_pub.h"
#include "llt_hccl_stub_gdr.h"
#include "hccl_comm_pub.h"
#include "network_manager_pub.h"
#include "transport_ibverbs.h"
#include "externalinput.h"
#include "dispatcher_pub.h"
#include "opexecounter_pub.h"
#include "dlra_function.h"
#include "detect_connect_anomalies.h"
#include "hccl_network.h"
#include "env_config.h"
using namespace std;
using namespace hccl;

class DetectConnectionAnomaliesSt : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--HcomParamCheck SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--HcomParamCheck TearDown--\033[0m" << std::endl; }
    virtual void SetUp()
    {
        s32 portNum = 7;
        setenv("HCCL_DFS_CONFIG", "connection_fault_detection_time:20", 1);
        InitEnvParam();
        MOCKER(GetExternalInputDfsConnectionFaultDetectionTime).stubs().will(returnValue(1));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        unsetenv("HCCL_DFS_CONFIG");
        std::cout << "A Test TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

#if 0

TEST_F(DetectConnectionAnomaliesSt, addIpQueue1)
{
    RankInfo localRankInfo;
    RankInfo remoteRankInfo;
    NicType nicType = NicType::VNIC_TYPE;
    localRankInfo.deviceType = DevType::DEV_TYPE_910B;
    HcclIpAddress localIp("127.0.0.1");
    HcclIpAddress remoteIpIp("127.0.0.2");
    localRankInfo.deviceVnicIp = localIp;
    remoteRankInfo.deviceVnicIp = remoteIpIp;
    remoteRankInfo.nicIp.push_back(remoteIpIp);
    localRankInfo.nicIp.push_back(localIp);
    MOCKER(GetExternalInputDfsConnectionFaultDetectionTime)
    .stubs()
    .will(returnValue(1));
    DetectConnectionAnomalies::GetInstance(0).broadCastTime = 1;
    DetectConnectionAnomalies::GetInstance(0).AddIpQueue(localRankInfo, remoteRankInfo, nicType, 0);

    nicType = NicType::DEVICE_NIC_TYPE;
    localRankInfo.nicIp.push_back(localIp);
    remoteRankInfo.nicIp.push_back(remoteIpIp);
    DetectConnectionAnomalies::GetInstance(0).AddIpQueue(localRankInfo, remoteRankInfo, nicType, 0);
	GlobalMockObject::verify();
}

TEST_F(DetectConnectionAnomaliesSt, addIpQueue2)
{
    RankInfo localRankInfo;
    RankInfo remoteRankInfo;
    NicType nicType = NicType::DEVICE_NIC_TYPE;
    localRankInfo.deviceType = DevType::DEV_TYPE_910B;
    HcclIpAddress localIp("127.0.0.1");
    HcclIpAddress remoteIpIp("127.0.0.1");
    localRankInfo.nicIp.push_back(localIp);
    remoteRankInfo.nicIp.push_back(remoteIpIp);

    localRankInfo.deviceVnicIp = localIp;
    remoteRankInfo.deviceVnicIp = remoteIpIp;
    MOCKER(GetExternalInputDfsConnectionFaultDetectionTime)
    .stubs()
    .will(returnValue(1));
    DetectConnectionAnomalies::GetInstance(1).broadCastTime = 1;
    DetectConnectionAnomalies::GetInstance(1).AddIpQueue(localRankInfo, remoteRankInfo, nicType, 1);

    std::shared_ptr<HcclSocket> newSocket(new (std::nothrow)HcclSocket("detect",
        nullptr, localIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    std::map<HcclIpAddress, std::shared_ptr<HcclSocket>> socketMap;
    socketMap.emplace(localIp, newSocket);

    NetDevContext ctx1;
    ctx1.Init(nicType, 0, 0, localIp);

    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    netDevCtxMap.emplace(localIp, reinterpret_cast<HcclNetDevCtx>(&ctx1));
    DetectConnectionAnomalies::GetInstance(1).threadExit_ = true;
    DetectConnectionAnomalies::GetInstance(1).isInitThread_ = true;
    HcclResult  ret = DetectConnectionAnomalies::GetInstance(1).Detect();
    GlobalMockObject::verify();
}

#endif

TEST_F(DetectConnectionAnomaliesSt, GetStatus)
{
    MOCKER_CPP(&HcclSocket::GetStatus).stubs().with(mockcpp::any()).will(returnValue(HcclSocketStatus::SOCKET_OK));
    HcclIpAddress ipAddr("192.168.1.1");
    HcclIpAddress localIpAddr("192.168.1.2");
    std::shared_ptr<HcclSocket> newSocket(
        new (std::nothrow) HcclSocket("detect", nullptr, localIpAddr, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    ErrInfo errInfo;

    errInfo.localRankInfo.superPodId = 1;
    errInfo.localRankInfo.devicePhyId = 0;
    errInfo.localRankInfo.serverId = "111";
    errInfo.remoteRankInfo.serverId = "222";
    errInfo.remoteRankInfo.superPodId = 1;
    errInfo.remoteRankInfo.devicePhyId = 1;
    MOCKER(GetExternalInputDfsConnectionFaultDetectionTime).stubs().will(returnValue(1));
    HcclResult ret = DetectConnectionAnomalies::GetInstance(2).GetStatus(errInfo, newSocket);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    MOCKER(GetExternalInputDfsConnectionFaultDetectionTime).stubs().will(returnValue(1));
    MOCKER_CPP(&HcclSocket::GetStatus).stubs().with(mockcpp::any()).will(returnValue(HcclSocketStatus::SOCKET_ERROR));
    ret = DetectConnectionAnomalies::GetInstance(2).GetStatus(errInfo, newSocket);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
    GlobalMockObject::verify();
}

TEST_F(DetectConnectionAnomaliesSt, ConstructErrorInfo)
{
    HcclIpAddress ipAddr("192.168.1.1");
    HcclIpAddress localIpAddr("192.168.1.2");
    RankInfo localRankInfo;
    RankInfo remoteRankInfo;
    localRankInfo.devicePhyId = 0;
    localRankInfo.superPodId = "0";
    remoteRankInfo.devicePhyId = 1;
    remoteRankInfo.superPodId = "0";
    localRankInfo.serverId = "111";
    remoteRankInfo.serverId = "222";

    std::shared_ptr<HcclSocket> newSocket(
        new (std::nothrow) HcclSocket("detect", nullptr, localIpAddr, 0, HcclSocketRole::SOCKET_ROLE_SERVER));

    HcclResult ret
        = DetectConnectionAnomalies::GetInstance(3).ConstructErrorInfo(newSocket, localRankInfo, remoteRankInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DetectConnectionAnomaliesSt, CreateDetectNicLinks)
{
    HcclIpAddress ipAddr("192.168.1.1");
    HcclIpAddress localIpAddr("192.168.1.2");
    std::shared_ptr<HcclSocket> newSocket(
        new (std::nothrow) HcclSocket("detect", nullptr, localIpAddr, 0, HcclSocketRole::SOCKET_ROLE_SERVER));

    NicType nicType = NicType::DEVICE_NIC_TYPE;
    DetectConnectionAnomalies::GetInstance(4).broadCastTime = 1;
    MOCKER_CPP(&HcclSocket::Accept).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_TIMEOUT));
    ErrInfo errInfo;
    errInfo.localRankInfo.devicePhyId = 0;
    errInfo.deviceLogicId = 0;
    errInfo.localRankInfo.nicIp.push_back(localIpAddr);
    errInfo.localRankInfo.deviceNicPort = 16677;
    errInfo.remoteRankInfo.nicIp.push_back(ipAddr);
    MOCKER(GetExternalInputDfsConnectionFaultDetectionTime).stubs().will(returnValue(1));
    HcclResult ret = DetectConnectionAnomalies::GetInstance(4).CreateDetectNicLinks(errInfo);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    DetectConnectionAnomalies::GetInstance(4).Deinit();

    MOCKER_CPP(&HcclSocket::Listen, HcclResult(HcclSocket::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocket::Send, HcclResult(HcclSocket::*)(const void*, u64))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    DetectInfo info;
    SendInfo sendInfo;
    errInfo.localRankInfo.nicIp.push_back(ipAddr);
    DetectConnectionAnomalies::GetInstance(4).recvErrorInfoMap_["127.0.0.2"] = info;
    DetectConnectionAnomalies::GetInstance(4).sendErrorInfoMap_["127.0.0.2"] = sendInfo;
    DetectConnectionAnomalies::GetInstance(4).listenNicVec_.push_back(newSocket);
    DetectConnectionAnomalies::GetInstance(4).threadExit_ = true;
    ret = DetectConnectionAnomalies::GetInstance(4).CreateDetectNicLinks(errInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DetectConnectionAnomalies::GetInstance(4).Deinit();
    GlobalMockObject::verify();
}

TEST_F(DetectConnectionAnomaliesSt, CreateDetectVnicLinks)
{
    HcclIpAddress ipAddr("192.168.1.1");
    HcclIpAddress localIpAddr("192.168.1.2");
    std::shared_ptr<HcclSocket> newSocket(
        new (std::nothrow) HcclSocket("detect", nullptr, localIpAddr, 0, HcclSocketRole::SOCKET_ROLE_SERVER));

    NicType nicType = NicType::VNIC_TYPE;
    DetectConnectionAnomalies::GetInstance(5).threadExit_ = true;

    MOCKER_CPP(&HcclSocket::Listen, HcclResult(HcclSocket::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocket::Accept).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_TIMEOUT));
    MOCKER_CPP(&HcclSocket::Send, HcclResult(HcclSocket::*)(const void*, u64))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    ErrInfo errInfo;
    errInfo.localRankInfo.devicePhyId = 1;
    errInfo.deviceLogicId = 1;
    errInfo.localRankInfo.nicIp.push_back(localIpAddr);
    errInfo.localRankInfo.deviceNicPort = 16677;
    errInfo.remoteRankInfo.nicIp.push_back(ipAddr);
    DetectInfo info;
    SendInfo sendInfo;
    DetectConnectionAnomalies::GetInstance(5).recvErrorInfoMap_["127.0.0.1"] = info;
    DetectConnectionAnomalies::GetInstance(5).sendErrorInfoMap_["127.0.0.1"] = sendInfo;
    DetectConnectionAnomalies::GetInstance(5).listenVnicVec_.push_back(newSocket);
    DetectConnectionAnomalies::GetInstance(5).broadCastTime = 1;

    MOCKER_CPP(&DetectConnectionAnomalies::AddWhiteList).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = DetectConnectionAnomalies::GetInstance(5).CreateDetectVnicLinks(errInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DetectConnectionAnomalies::GetInstance(5).threadExit_ = false;
    GlobalMockObject::verify();
}

#if 0

TEST_F(DetectConnectionAnomaliesSt, addIpQueue3)
{
    RankInfo localRankInfo;
    RankInfo remoteRankInfo;
    NicType nicType = NicType::VNIC_TYPE;
    localRankInfo.deviceType = DevType::DEV_TYPE_910B;
    HcclIpAddress localIp("127.0.0.1");
    HcclIpAddress remoteIpIp("127.0.0.2");
    localRankInfo.nicIp.push_back(localIp);
    localRankInfo.deviceVnicIp = localIp;
    remoteRankInfo.deviceVnicIp = remoteIpIp;
    setenv("HCCL_DFS_CONFIG", "connection_fault_detection_time:0", 1);
    DetectConnectionAnomalies::GetInstance(6).broadCastTime = 1;
    DetectConnectionAnomalies::GetInstance(6).AddIpQueue(localRankInfo, remoteRankInfo, nicType, 0);
    unsetenv("HCCL_DFS_CONFIG");

    DetectConnectionAnomalies::GetInstance(33).AddIpQueue(localRankInfo, remoteRankInfo, nicType, 1);
}

TEST_F(DetectConnectionAnomaliesSt, addIpQueue4)
{
    RankInfo localRankInfo;
    RankInfo remoteRankInfo;
    NicType nicType = NicType::DEVICE_NIC_TYPE;
    localRankInfo.deviceType = DevType::DEV_TYPE_910B;
    HcclIpAddress localIp("127.0.0.6");
    HcclIpAddress remoteIpIp("127.0.0.7");
    localRankInfo.nicIp.push_back(localIp);
    remoteRankInfo.nicIp.push_back(remoteIpIp);

    std::shared_ptr<HcclSocket> newSocket(new (std::nothrow)HcclSocket("detect",
        nullptr, localIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));


    DetectConnectionAnomalies::GetInstance(7).threadExit_ = true;
    DetectConnectionAnomalies::GetInstance(7).broadCastTime = 1;

    ErrInfo errInfo;
    errInfo.localRankInfo = localRankInfo;
    errInfo.remoteRankInfo = remoteRankInfo;

    localRankInfo.devicePhyId = 0;
    errInfo.deviceLogicId = 0;
    errInfo.localRankInfo.nicIp.push_back(localIp);
    errInfo.localRankInfo.deviceNicPort = 16677;
    errInfo.remoteRankInfo.nicIp.push_back(remoteIpIp);

    DetectConnectionAnomalies::GetInstance(7).AddIpQueue(localRankInfo, remoteRankInfo, nicType, 2);
    DetectConnectionAnomalies::GetInstance(7).isNeedNic_ = true;
    DetectConnectionAnomalies::GetInstance(7).ipNictypeQueue_.push(errInfo);
    HcclResult  ret = DetectConnectionAnomalies::GetInstance(7).GetIpQueue();
    DetectConnectionAnomalies::GetInstance(7).Deinit();
}

#endif

TEST_F(DetectConnectionAnomaliesSt, AddWhiteList)
{
    MOCKER_CPP(&HcclSocket::AddWhiteList).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclIpAddress ipAddr("192.168.1.1");
    HcclIpAddress localIpAddr("192.168.1.2");
    NicType nicType = NicType::DEVICE_NIC_TYPE;
    std::string tag = "detect_0";

    std::shared_ptr<HcclSocket> newSocket(
        new (std::nothrow) HcclSocket("detect", nullptr, localIpAddr, 0, HcclSocketRole::SOCKET_ROLE_SERVER));

    DetectConnectionAnomalies::GetInstance(8).isNeedNic_ = true;
    DetectConnectionAnomalies::GetInstance(8).uniqueIps_.insert(ipAddr);

    HcclResult ret = DetectConnectionAnomalies::GetInstance(8).AddWhiteList(newSocket, nicType, tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DetectConnectionAnomalies::GetInstance(8).Deinit();
    GlobalMockObject::verify();
}

TEST_F(DetectConnectionAnomaliesSt, Connect_test)
{
    MOCKER_CPP(&HcclSocket::Connect).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&NetworkManager::StartHostNet).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&NetworkManager::StopHostNet).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&DetectConnectionAnomalies::GetStatus)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_TCP_CONNECT));

    struct ErrInfo errInfo;
    errInfo.nicType = NicType::VNIC_TYPE;
    errInfo.localRankInfo.devicePhyId = 0;
    errInfo.deviceLogicId = 0;

    HcclIpAddress ipAddr("192.168.1.3");
    HcclIpAddress localIpAddr("192.168.1.4");
    std::shared_ptr<HcclSocket> newSocket(
        new (std::nothrow) HcclSocket("detect", nullptr, localIpAddr, 0, HcclSocketRole::SOCKET_ROLE_SERVER));

    HcclResult ret = DetectConnectionAnomalies::GetInstance(9).CreateClient(errInfo);
    EXPECT_EQ(ret, HCCL_E_TCP_CONNECT);
    DetectConnectionAnomalies::GetInstance(9).Deinit();
    GlobalMockObject::verify();
}
