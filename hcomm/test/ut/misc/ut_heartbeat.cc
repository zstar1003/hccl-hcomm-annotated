/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <sys/time.h>

#define private public
#include "heartbeat.h"
#undef private

#include "llt_hccl_stub_pub.h"
#include "llt_hccl_stub_gdr.h"
#include "hccl_comm_pub.h"
#include "network_manager_pub.h"
#include "transport_ibverbs.h"
#include "externalinput.h"
#include "opexecounter_pub.h"
#include "dispatcher_pub.h"
#include "dlra_function.h"
#include "env_config.h"

using namespace std;
using namespace hccl;

HcclResult stub_GetRaResourceInfo(NetworkManager* that, RaResourceInfo& raResourceInfo)
{
    static bool initialized = false;
    static RaResourceInfo fake_raResourceInfo;
    static int fake_handle = 1;
    HcclIpAddress ipAddr = HcclIpAddress(1684515008);
    if (!initialized) {
        IpSocket tmpIpSocket;
        tmpIpSocket.nicSocketHandle = &fake_handle;
        for (int i = 0; i < 8; i++) {
            fake_raResourceInfo.vnicSocketMap[ipAddr] = tmpIpSocket;
            fake_raResourceInfo.nicSocketMap[ipAddr] = tmpIpSocket;
        }
    }
    raResourceInfo = fake_raResourceInfo;
    return HCCL_SUCCESS;
}

HcclResult stub_GetRaResourceInfo_RdmaHandle(NetworkManager* that, RaResourceInfo& raResourceInfo)
{
    static bool initialized = false;
    static RaResourceInfo fake_raResourceInfo;
    static int fake_handle = 1;
    HcclIpAddress ipAddr = HcclIpAddress(1684515008);
    if (!initialized) {
        IpSocket tmpIpSocket;
        tmpIpSocket.nicRdmaHandle = &fake_handle;
        for (int i = 0; i < 8; i++) {
            fake_raResourceInfo.vnicSocketMap[ipAddr] = tmpIpSocket;
            fake_raResourceInfo.nicSocketMap[ipAddr] = tmpIpSocket;
        }
    }
    raResourceInfo = fake_raResourceInfo;
    return HCCL_SUCCESS;
}

s32 stub_complete_hrtRaSocketNonBlockSendHB(const FdHandle fdHandle, const void* data, u64 size, u64* sent_size)
{
    *sent_size = size;
    return 0;
}

s32 stub_complete_hrtRaSocketNonBlockRecvHB(const FdHandle fdHandle, void* data, u64 size, u64* recvSize)
{
    static u32 count = 0;
    if (count++ % 5 != 0) {
        *recvSize = size;
        count = 0;
    }
    return 0;
}

HcclResult stub_complete_hrtRaBlockGetSockets(u32 role, struct SocketInfoT conn[], u32 num)
{
    static std::vector<int> fdHandle;
    for (int i = 0; i < num; i++) {
        fdHandle.push_back(0);
        conn[i].fdHandle = &fdHandle[fdHandle.size() - 1];
        conn[i].status = CONNECT_OK;
    }
    return HCCL_SUCCESS;
}

HcclResult stub_complete_hrtRaNonBlockGetSockets(u32 role, struct SocketInfoT conn[], u32 num, u32* connectedNum)
{
    static std::vector<int> fdHandle;
    for (int i = 0; i < num; i++) {
        fdHandle.push_back(0);
        conn[i].fdHandle = &fdHandle[fdHandle.size() - 1];
        conn[i].status = CONNECT_OK;
    }
    *connectedNum = num;
    return HCCL_SUCCESS;
}

extern s32 stub_SocketManagerTest_hrtRaGetSockets(u32 role, struct SocketInfoT conn[], u32 num, u32* connectedNum);

class HeartBeatTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        DlRaFunction::GetInstance().DlRaFunctionInit();
        std::cout << "HeartBeatTest SetUP" << std::endl;
    }
    static void TearDownTestCase() { std::cout << "HeartBeatTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        MOCKER(hrtRaSocketWhiteListAdd).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtRaSocketWhiteListDel).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtRaSocketBatchConnect).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtRaBlockGetSockets).stubs().will(invoke(stub_complete_hrtRaBlockGetSockets));
        MOCKER(hrtRaNonBlockGetSockets).stubs().will(invoke(stub_complete_hrtRaNonBlockGetSockets));
        MOCKER(hrtRaSocketBatchClose).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtRaSocketNonBlockSend).stubs().will(invoke(stub_complete_hrtRaSocketNonBlockSendHB));
        MOCKER(hrtRaSocketNonBlockRecv).stubs().will(invoke(stub_complete_hrtRaSocketNonBlockRecvHB));
        MOCKER_CPP(&HcclSocket::AddWhiteList).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HcclSocket::DelWhiteList).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&NetworkManager::GetRaResourceInfo).stubs().will(invoke(stub_GetRaResourceInfo));
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtRaGetSockets).stubs().will(invoke(stub_SocketManagerTest_hrtRaGetSockets));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(HeartBeatTest, ut_ReferenceMapTest)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 count = 0;
    ReferenceMap<u32, u32> testMap;
    count = testMap.insert(1, 1);
    EXPECT_EQ(count, 1);
    count = testMap.insert(1, 2);
    EXPECT_EQ(count, 2);
    count = testMap.insert(1, 3);
    EXPECT_EQ(count, 3);
    count = testMap.insert(1, 4);
    EXPECT_EQ(count, 4);
    count = testMap.erase(1);
    EXPECT_EQ(count, 3);
    count = testMap.erase(1);
    EXPECT_EQ(count, 2);
    count = testMap.erase(1);
    EXPECT_EQ(count, 1);
    count = testMap.erase(1);
    EXPECT_EQ(count, 0);
    count = testMap.erase(1);
    EXPECT_EQ(count, 0);
    count = testMap.insert(2, 1);
    EXPECT_EQ(count, 1);
    count = testMap.insert(2, 2);
    EXPECT_EQ(count, 2);
    count = testMap.insert(2, 1);
    EXPECT_EQ(count, 3);
    testMap.clear();
    count = testMap.insert(1, 1);
    EXPECT_EQ(count, 1);
    ret = testMap.ref(1);
    EXPECT_EQ(testMap.count(1), 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = testMap.ref(4);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = testMap.unref(4);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = testMap.unref(1);
    EXPECT_EQ(testMap.count(1), 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    count = testMap.insert(2, 2);
    EXPECT_EQ(count, 1);
    count = testMap.insert(3, 3);
    EXPECT_EQ(count, 1);
    testMap[4] = 4;
    EXPECT_EQ(testMap[4], 4);
    for (auto iter = testMap.begin(); iter != testMap.end(); iter++) {
        EXPECT_EQ(iter->first, iter->second);
    }
}

TEST_F(HeartBeatTest, ut_RingBufferTest)
{
    auto rb = RingBuffer();
    HcclResult ret = HCCL_SUCCESS;
    ret = rb.Init(10);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rb.Size(), 0);
    u8 src1[5] = {0, 1, 2, 3, 4};
    ret = rb.PushSeg(src1, 5);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rb.Size(), 5);
    u8 src2[5] = {5, 6, 7, 8, 9};
    ret = rb.PushSeg(src2, 5);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rb.Size(), 10);
    u8 dst[10] = {0};
    ret = rb.GetSeg(dst, 10);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (u32 i = 0; i < 10; i++) {
        EXPECT_EQ(dst[i], i);
    }
    ret = rb.PopSeg(10);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rb.Size(), 0);
}

TEST_F(HeartBeatTest, ut_HeartBeatTest1)
{
    struct RaInitConfig config;
    RaInit(&config);
    std::vector<RankInfo> rankInfos1;
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos1.push_back(rankInfo);
    }
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos1.push_back(rankInfo);
    }
    DevType type = DevType::DEV_TYPE_910_93;

    Heartbeat::GetInstance(0).RegisterRanks(type, rankInfos1[3], rankInfos1, 0, false, "test1");
    Heartbeat::GetInstance(0).UnRegisterRanks("test1");
}

TEST_F(HeartBeatTest, ut_HeartBeatTest2)
{
    std::vector<RankInfo> rankInfos1;
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos1.push_back(rankInfo);
    }
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos1.push_back(rankInfo);
    }
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos1.push_back(rankInfo);
    }
    DevType type = DevType::DEV_TYPE_910B;
    Heartbeat::GetInstance(0).RegisterRanks(type, rankInfos1[3], rankInfos1, 0, false, "test1");
    std::this_thread::sleep_for(std::chrono::milliseconds(LLT_SOCKET_SLEEP_MILLISECONDS));
    Heartbeat::GetInstance(0).UnRegisterRanks("test1");

    std::vector<RankInfo> rankInfos2;
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos2.push_back(rankInfo);
    }
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos2.push_back(rankInfo);
    }
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos2.push_back(rankInfo);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(LLT_SOCKET_SLEEP_MILLISECONDS));
    Heartbeat::GetInstance(0).RegisterRanks(type, rankInfos2[3], rankInfos2, 0, false, "test2");
    std::this_thread::sleep_for(std::chrono::milliseconds(LLT_SOCKET_SLEEP_MILLISECONDS));
    Heartbeat::GetInstance(0).UnRegisterRanks("test2");
}

TEST_F(HeartBeatTest, ut_HeartBeatTest3)
{
    std::vector<RankInfo> rankInfos1;
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos1.push_back(rankInfo);
    }
    DevType type = DevType::DEV_TYPE_910B;
    Heartbeat::GetInstance(0).RegisterRanks(type, rankInfos1[3], rankInfos1, 0, false, "test1");

    std::vector<RankInfo> rankInfos2;
    for (size_t i = 0; i < 4; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos2.push_back(rankInfo);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(LLT_SOCKET_SLEEP_MILLISECONDS));
    Heartbeat::GetInstance(0).RegisterRanks(type, rankInfos2[3], rankInfos2, 0, false, "test2");
    std::this_thread::sleep_for(std::chrono::milliseconds(LLT_SOCKET_SLEEP_MILLISECONDS));
    Heartbeat::GetInstance(0).UnRegisterRanks("test2");
    std::this_thread::sleep_for(std::chrono::milliseconds(LLT_SOCKET_SLEEP_MILLISECONDS));
    Heartbeat::GetInstance(0).UnRegisterRanks("test1");
}

TEST_F(HeartBeatTest, ut_HeartBeatTest4)
{
    struct RaInitConfig config;
    RaInit(&config);
    std::vector<RankInfo> rankInfos1;
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.serverId = "0.0.0.0";
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos1.push_back(rankInfo);
    }
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.serverId = "1.0.0.0";
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos1.push_back(rankInfo);
    }

    DevType type = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(type)).will(returnValue(HCCL_SUCCESS));

    Heartbeat::GetInstance(0).RegisterRanks(type, rankInfos1[3], rankInfos1, 0, false, "test1");
    Heartbeat::GetInstance(0).UnRegisterRanks("test1");
    GlobalMockObject::verify();

    type = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(type)).will(returnValue(HCCL_SUCCESS));

    Heartbeat::GetInstance(0).RegisterRanks(type, rankInfos1[3], rankInfos1, 0, false, "test1");
    Heartbeat::GetInstance(0).UnRegisterRanks("test1");
}

TEST_F(HeartBeatTest, ut_HeartBeatTest5)
{
    struct RaInitConfig config;
    RaInit(&config);

    std::vector<RankInfo> rankInfos2;
    RankInfo rankInfo;
    rankInfo.worldRank = 0;
    rankInfo.serverId = "0.0.0.0";
    rankInfo.nicIp.push_back(HcclIpAddress(1684515008));

    DevType type = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(type)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetPairDevicePhyId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceIndexByPhyId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Heartbeat::IsEnableBackupLink).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&Heartbeat::GetConnectRank).stubs().with(mockcpp::any()).will(returnValue(0));

    Heartbeat::GetInstance(0).Init(rankInfo, false, false, 0);
    Heartbeat::GetInstance(0).DeInit();

    rankInfo.devicePhyId = 0;
    Heartbeat::GetInstance(0).Init(rankInfo, false, false, 0);
    Heartbeat::GetInstance(0).DeInit();

    std::vector<RankInfo> rankInfos1;
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.serverId = "0.0.0.0";
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos1.push_back(rankInfo);
    }
    rankInfo.backupNicIp.push_back(HcclIpAddress(1684515008));
    MOCKER_CPP(&Heartbeat::Init).stubs().with(mockcpp::any()).will(returnValue(0));
    Heartbeat::GetInstance(0).RegisterRanks(type, rankInfo, rankInfos1, 0, false, "test1");
    Heartbeat::GetInstance(0).UnRegisterRanks("test1");

    Heartbeat::GetInstance(0).ProcessCqeErrInfo();

    GlobalMockObject::verify();
}

// TEST_F(HeartBeatTest, ut_TransformIpStr2NumFailed)
// {
//     MOCKER(TransformIpStr2Num)
//     .stubs()
//     .will(returnValue(HCCL_E_INTERNAL));

//     RankInfo rankinfo;
//     rankinfo.serverId = "0.0.0.0";

//     UIDType uid = Heartbeat::GetInstance(0).GetUId(rankinfo);
//     std::string str = Heartbeat::GetInstance(0).FormatUId(uid);
// }

// TEST_F(HeartBeatTest, ut_TransformIntStr2NumFailed)
// {
//     MOCKER(TransformIpStr2Num)
//     .stubs()
//     .will(returnValue(HCCL_E_INTERNAL));

//     RankInfo rankinfo;
//     rankinfo.serverId = "0.0.0.0";

//     UIDType uid = Heartbeat::GetInstance(0).GetUId(rankinfo);
//     std::string str = Heartbeat::GetInstance(0).FormatUId(uid);
// }

HcclResult stub_ISend(HcclSocket* that, void* data, u64 size, u64& compSize)
{
    compSize = size;
    return HCCL_SUCCESS;
}

HcclResult stub_ISend_nocomp(HcclSocket* that, void* data, u64 size, u64& compSize)
{
    compSize = 0;
    return HCCL_SUCCESS;
}

TEST_F(HeartBeatTest, ut_SendFrame)
{
    MOCKER_CPP(&HcclSocket::ISend).stubs().will(invoke(stub_ISend));
    HcclResult ret = HCCL_SUCCESS;

    RankInfo rankInfo;
    rankInfo.serverId = "127.0.0.1";
    rankInfo.devicePhyId = 0;
    UIDType src = Heartbeat::GetInstance(0).GetUId(rankInfo);
    UIDType dst = Heartbeat::GetInstance(0).GetUId(rankInfo);
    UIDType crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    UIDType informer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    HeartBeatFrame bf(src, dst, crimer, informer, HeartBeatStatus::HEARTBEAT_OK);
    ConnInfo conninfo;
    conninfo.sendBuffer.push(bf);

    rankInfo.devicePhyId = 1;
    dst = Heartbeat::GetInstance(0).GetUId(rankInfo);
    rankInfo.devicePhyId = 2;
    crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    rankInfo.devicePhyId = 3;
    informer = Heartbeat::GetInstance(0).GetUId(rankInfo);

    Heartbeat::GetInstance(0).rankId2SocketMap_.insert(dst, conninfo);
    ret = Heartbeat::GetInstance(0).SendFrame(dst, crimer, informer, HeartBeatStatus::HEARTBEAT_LOST);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = Heartbeat::GetInstance(0).SendFrame(dst, crimer, informer, HeartBeatStatus::HEARTBEAT_LOST);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    MOCKER_CPP(&HcclSocket::ISend).stubs().will(invoke(stub_ISend_nocomp));
    ret = Heartbeat::GetInstance(0).SendFrame(dst, crimer, informer, HeartBeatStatus::HEARTBEAT_LOST);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    Heartbeat::GetInstance(0).rankId2SocketMap_.erase(dst);
}

TEST_F(HeartBeatTest, ut_TestErrRankQueue)
{
    MOCKER_CPP(&HcclSocket::ISend).stubs().will(invoke(stub_ISend));
    HcclResult ret = HCCL_SUCCESS;

    RankInfo rankInfo;
    rankInfo.devicePhyId = 1;
    UIDType dst = Heartbeat::GetInstance(0).GetUId(rankInfo);

    ConnInfo conninfo;
    std::shared_ptr<HcclSocket> newSocket(
        new (std::nothrow) HcclSocket("my tag", nullptr, HcclIpAddress(), 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    conninfo.socket = newSocket;
    Heartbeat::GetInstance(0).rankId2SocketMap_.insert(dst, conninfo);

    rankInfo.devicePhyId = 10;
    UIDType tmp = Heartbeat::GetInstance(0).GetUId(rankInfo);
    Heartbeat::GetInstance(0).errRankQueue_.push(tmp);
    Heartbeat::GetInstance(0).ProcessExceptionEvent();

    Heartbeat::GetInstance(0).rankId2SocketMap_.erase(dst);

    GlobalMockObject::verify();
    GlobalMockObject::verify();
}

HcclResult stub_hrtRaGetCqeErrInfo_value_status_0(unsigned int phy_id, struct CqeErrInfo* info)
{
    info->qpn = 10;
    info->status = 0;
    struct timeval tv;
    gettimeofday(&tv, NULL);

    info->time = tv;
    return HCCL_SUCCESS;
}

HcclResult stub_hrtRaGetCqeErrInfo_value(unsigned int phy_id, struct CqeErrInfo* info)
{
    info->qpn = 1;
    info->status = 12;
    struct timeval tv;
    gettimeofday(&tv, NULL);

    info->time = tv;
    return HCCL_SUCCESS;
}

HcclResult stub_hrtRaGetCqeErrInfoList_value(RdmaHandle handle, struct CqeErrInfo* infolist, unsigned int* num)
{
    *num = 65;
    infolist[0].qpn = 1;
    infolist[0].status = 12;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    infolist[0].time = tv;
    return HCCL_SUCCESS;
}

HcclResult stub_GetIdentifierByQpn_value(std::pair<u32, u32> qpnPair, std::string& identifier)
{
    qpnPair.second = 1;
    identifier = "comm_id_1";
    return HCCL_SUCCESS;
}

HcclResult stub_GetRemoteRankByQpn_value(std::pair<u32, u32> qpnPair, u32& remoteRank)
{
    qpnPair.second = 1;
    remoteRank = 24;
    return HCCL_SUCCESS;
}

struct tm* stub_localtime_value(const time_t* time)
{
    tm* aa;
    return aa;
}

HcclResult stub_GetRemoteIpAddrByQpn_value(u32 qpn, HcclIpAddress& ip)
{
    qpn = 1;

    return HCCL_SUCCESS;
}

TEST_F(HeartBeatTest, ut_ProcessCqeErrInfo_1)
{
    TransportIbverbs::g_flag = true;

    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value));

    MOCKER(localtime).stubs().will(invoke(stub_localtime_value));

    MOCKER_CPP(&Heartbeat::SetStatus).stubs();

    Heartbeat::GetInstance(0).ProcessCqeErrInfo();

    TransportIbverbs::g_flag = false;
}

TEST_F(HeartBeatTest, ut_ProcessCqeErrInfo_001)
{
    TransportIbverbs::g_flag = true;

    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value));

    Heartbeat::GetInstance(0).ProcessCqeErrInfo();

    TransportIbverbs::g_flag = false;
}

TEST_F(HeartBeatTest, ut_ProcessCqeErrInfo_002)
{
    TransportIbverbs::g_flag = true;

    MOCKER(hrtRaGetCqeErrInfoList).expects(atMost(0)).will(returnValue(HCCL_SUCCESS));

    Heartbeat::GetInstance(0).ProcessCqeErrInfo();
    TransportIbverbs::g_flag = false;
}

TEST_F(HeartBeatTest, ust_ProcessCqeErrInfo_003)
{
    TransportIbverbs::g_flag = true;

    MOCKER(hrtRaGetCqeErrInfoList).expects(atMost(1)).will(returnValue(HCCL_E_INTERNAL));

    Heartbeat::GetInstance(0).ProcessCqeErrInfo();
    TransportIbverbs::g_flag = false;
}

TEST_F(HeartBeatTest, ut_ProcessCqeErrInfo_004)
{
    TransportIbverbs::g_flag = true;

    MOCKER(hrtRaGetCqeErrInfoList).expects(atMost(1)).will(invoke(stub_hrtRaGetCqeErrInfo_value));

    Heartbeat::GetInstance(0).ProcessCqeErrInfo();
    TransportIbverbs::g_flag = false;
}

TEST_F(HeartBeatTest, ut_ProcessCqeErrInfo_005)
{
    TransportIbverbs::g_flag = true;

    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value));

    Heartbeat::GetInstance(0).ProcessCqeErrInfo();

    TransportIbverbs::g_flag = false;
}

TEST_F(HeartBeatTest, ut_ProcessCqeErrInfo_006)
{
    TransportIbverbs::g_flag = true;

    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value));

    Heartbeat::GetInstance(0).ProcessCqeErrInfo();

    TransportIbverbs::g_flag = false;
}

TEST_F(HeartBeatTest, ut_ProcessCqeErrInfo_007)
{
    TransportIbverbs::g_flag = true;

    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value));

    // MOCKER(TransformIpNum2Str)
    // .stubs()
    // .will(returnValue(HCCL_E_INTERNAL));

    Heartbeat::GetInstance(0).ProcessCqeErrInfo();

    TransportIbverbs::g_flag = false;
}

TEST_F(HeartBeatTest, ut_ProcessCqeErrInfo_008)
{
    // 开启重执行分支 && 注入 status = 0，提前退出
    MOCKER_CPP(&Heartbeat::GetRetryEnable).stubs().with(mockcpp::any()).will(returnValue(true));
    TransportIbverbs::g_flag = true;

    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value_status_0));
    Heartbeat::GetInstance(0).ProcessCqeErrInfo();

    GlobalMockObject::verify();
    TransportIbverbs::g_flag = false;
}

TEST_F(HeartBeatTest, ut_ProcessCqeErrInfo_009)
{
    // 未开启重执行分支
    MOCKER_CPP(&Heartbeat::GetRetryEnable).stubs().with(mockcpp::any()).will(returnValue(false));

    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value));
    Heartbeat::GetInstance(0).ProcessCqeErrInfo();

    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_ProcessCqeErrInfo_010)
{
    // 开启重执行分支 && 注入 status = 0，提前退出
    MOCKER_CPP(&Heartbeat::GetRetryEnable).stubs().with(mockcpp::any()).will(returnValue(true));
    TransportIbverbs::g_flag = true;

    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value));
    Heartbeat::GetInstance(0).ProcessCqeErrInfo();

    GlobalMockObject::verify();
    TransportIbverbs::g_flag = false;
}

TEST_F(HeartBeatTest, ut_SaveQpnForOpRetry_001)
{
    // 提前 return 覆盖
    ErrCqeInfo info;
    info.cqeInfo.status = 0;
    info.linkInfo.remoteRank = 24;

    Heartbeat::GetInstance(0).SaveQpnForOpRetry(info);
    Heartbeat::GetInstance(0).SaveQpnForOpRetry(info);
    info.cqeInfo.status = 12;
    Heartbeat::GetInstance(0).SaveQpnForOpRetry(info);

    info.cqeInfo.status = 12;
    Heartbeat::GetInstance(0).SaveQpnForOpRetry(info);
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_SaveQpnForOpRetry_002)
{
    std::map<u32, std::set<ErrCqeInfo>> infomap;
    ErrCqeInfo errqpn;
    errqpn.qpn = 54;
    ErrCqeInfo infoExt;
    CqeInfo info1;
    info1.status = 12;
    u32 dstRank = 24;
    errqpn.cqeInfo = info1;
    infoExt.linkInfo.remoteRank = dstRank;
    infoExt.cqeInfo = info1;
    infomap[dstRank] = {errqpn};
    /*rankMapForRetryAgent =  {
     *       comm_id_1 : { 24 : <54, time,12,remoteIp>> }
     *   }
     */
    Heartbeat::GetInstance(0).rankMapForRetryAgent.insert({"comm_id_1", {infomap}});
    // 有通信域，有remote rank
    ErrCqeInfo info2;
    info2.cqeInfo.status = 12;
    info2.linkInfo.remoteRank = 24;
    Heartbeat::GetInstance(0).SaveQpnForOpRetry(info2);
    // 有通信域，没有remote rank
    info2.cqeInfo.status = 12;
    info2.linkInfo.remoteRank = 2;
    Heartbeat::GetInstance(0).SaveQpnForOpRetry(info2);
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_CheckErrorCqe)
{
    HcclResult result = HCCL_SUCCESS;
    Heartbeat::GetInstance(0).CheckErrorCqe("test1", result);
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_CheckErrorCqe1)
{
    std::set<ErrCqeInfo> s1;
    s1.insert(ErrCqeInfo());
    Heartbeat::GetInstance(0).remoteIpMap.insert(std::make_pair("test1", s1));
    HcclResult result = HCCL_SUCCESS;
    Heartbeat::GetInstance(0).CheckErrorCqe("test1", result);
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_keyEvents)
{
    std::vector<RankInfo> rankInfos1;
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.serverId = "0.0.0.0";
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos1.push_back(rankInfo);
    }
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.serverId = "1.0.0.0";
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos1.push_back(rankInfo);
    }
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.serverId = "2.0.0.0";
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos1.push_back(rankInfo);
    }

    std::vector<RankInfo> rankInfos2;
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.serverId = "0.0.0.0";
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos2.push_back(rankInfo);
    }
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.serverId = "3.0.0.0";
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos2.push_back(rankInfo);
    }
    for (size_t i = 0; i < 8; i++) {
        RankInfo rankInfo;
        rankInfo.worldRank = i;
        rankInfo.devicePhyId = i;
        rankInfo.serverId = "4.0.0.0";
        rankInfo.nicIp.push_back(HcclIpAddress(1684515008));
        rankInfos2.push_back(rankInfo);
    }
    UIDType src0 = Heartbeat::GetInstance(0).GetUId(rankInfos1[0]);
    UIDType dst0 = Heartbeat::GetInstance(0).GetUId(rankInfos1[0]);
    UIDType src1 = Heartbeat::GetInstance(0).GetUId(rankInfos1[1]);
    UIDType dst1 = Heartbeat::GetInstance(0).GetUId(rankInfos1[1]);

    Heartbeat::GetInstance(0).SetStatus(src0, dst0, HeartBeatStatus::HEARTBEAT_CQE_ERR, false);
    Heartbeat::GetInstance(0).SetStatus(src1, dst1, HeartBeatStatus::HEARTBEAT_LOST, false);

    s32 INPUT_TIMEOUT = 180;
    std::string setTimeOutValue = to_string(INPUT_TIMEOUT);
    HcclResult ret = SetHccLExecTimeOut(setTimeOutValue.c_str(), HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_SET_BY_ENV);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    std::vector<std::string> tmpEventStr = Heartbeat::GetInstance(0).GetErrStatusVec();

    UIDType src = Heartbeat::GetInstance(0).GetUId(rankInfos1[2]);
    UIDType dst = Heartbeat::GetInstance(0).GetUId(rankInfos1[2]);
    Heartbeat::GetInstance(0).SetStatus(src, dst, HeartBeatStatus::HEARTBEAT_CQE_ERR, false);
    INPUT_TIMEOUT = 1800;
    setTimeOutValue = to_string(INPUT_TIMEOUT);
    ret = SetHccLExecTimeOut(setTimeOutValue.c_str(), HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_SET_BY_ENV);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    tmpEventStr = Heartbeat::GetInstance(0).GetErrStatusVec();
}

TEST_F(HeartBeatTest, ut_SutckDetection1)
{
    auto counterStat = CounterStat();
    Heartbeat::GetInstance(0).InitStuckDetection(counterStat);
    uint64_t cnt = counterStat.couterPrintInter - 1;
    counterStat.isNeedDetect = true;
    Heartbeat::GetInstance(0).StuckDetection(cnt, counterStat);
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_SutckDetection2)

{
    std::pair<int32_t, int32_t> counter(0, 0);
    auto counterStat = CounterStat();
    Heartbeat::GetInstance(0).InitStuckDetection(counterStat);
    uint64_t cnt = counterStat.couterPrintInter - 1;
    counterStat.isFirst = false;
    counterStat.isNeedDetect = true;
    Heartbeat::GetInstance(0).StuckDetection(cnt, counterStat);
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_transport_ibv_stop_test)
{
    DispatcherPub* dispatcher;
    const std::unique_ptr<NotifyPool> notifyPool;
    MachinePara machinePara;
    std::chrono::milliseconds timeout;
    TransportIbverbs transportIbverbs(dispatcher, notifyPool, machinePara, timeout);
    MOCKER(hrtRaQpBatchModify).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    auto ret = transportIbverbs.Stop();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HeartBeatTest, ut_transport_ibv_Resume_test)
{
    DispatcherPub* dispatcher;
    const std::unique_ptr<NotifyPool> notifyPool;
    MachinePara machinePara;
    std::chrono::milliseconds timeout;
    TransportIbverbs transportIbverbs(dispatcher, notifyPool, machinePara, timeout);
    MOCKER(hrtRaQpBatchModify).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    auto ret = transportIbverbs.Resume();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HeartBeatTest, ut_GetQpnErrForOpRetryAgent_Null)
{
    // 测试 GetQpnErr 返回 false
    MOCKER_CPP(&Heartbeat::GetRetryEnable).stubs().with(mockcpp::any()).will(returnValue(true));
    TransportIbverbs::g_flag = true;
    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value));

    Heartbeat::GetInstance(0).ProcessCqeErrInfo();

    std::set<tuple<u32, u32, u32>> qpErrSet;
    Heartbeat::GetInstance(0).GetQpnErr("comm_id_2", qpErrSet);
    bool isExistQPErr = (qpErrSet.size() > 0);
    EXPECT_EQ(isExistQPErr, false);

    TransportIbverbs::g_flag = false;
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_GetQpnErrForOpRetryAgent)
{
    // 开启重执行分支
    // qpmapForOpRetry 中没有指定通信域的 kv pair
    MOCKER_CPP(&Heartbeat::GetRetryEnable).stubs().with(mockcpp::any()).will(returnValue(true));
    TransportIbverbs::g_flag = true;
    Heartbeat::GetInstance(0).deviceLogicId_ = 0;
    // 调用hrtRaGetCqeErrInfo 是会注入一个Rdma Err
    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value));
    // 调用hrtRaGetCqeErrInfoList 是会注入一个Rdma Err
    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfoList_value));

    // 使 rdmahandle 不为 nullptr
    MOCKER_CPP(&NetworkManager::GetRaResourceInfo).stubs().will(invoke(stub_GetRaResourceInfo_RdmaHandle));
    Heartbeat::GetInstance(0).nicIp_ = HcclIpAddress(1684515008);
    Heartbeat::GetInstance(0).ProcessCqeErrInfo();

    std::set<std::tuple<u32, u32, u32>> qpErrSet;
    Heartbeat::GetInstance(0).GetQpnErr("comm_id_1", qpErrSet);
    bool isExistQPErr = (qpErrSet.size() > 0);
    EXPECT_EQ(isExistQPErr, true);

    TransportIbverbs::g_flag = false;
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_BroadcastCqeErr)
{
    // 开启重执行分支
    // qpmapForOpRetry 中没有指定通信域的 kv pair
    MOCKER_CPP(&Heartbeat::GetRetryEnable).stubs().with(mockcpp::any()).will(returnValue(true));
    TransportIbverbs::g_flag = true;
    // 调用hrtRaGetCqeErrInfo 是会注入一个Rdma Err
    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value));

    Heartbeat::GetInstance(0).ProcessCqeErrInfo();
    Heartbeat::GetInstance(0).ProcessCqeErrInfo();
    Heartbeat::GetInstance(0).BroadcastCqeErr("comm_id_1");

    TransportIbverbs::g_flag = false;
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_ClearAllCqeErr)
{
    // 开启重执行分支
    // qpmapForOpRetry 中没有指定通信域的 kv pair
    MOCKER_CPP(&Heartbeat::GetRetryEnable).stubs().with(mockcpp::any()).will(returnValue(true));
    TransportIbverbs::g_flag = true;
    // 调用hrtRaGetCqeErrInfo 是会注入一个Rdma Err
    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value));

    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfoList_value));
    Heartbeat::GetInstance(0).ProcessCqeErrInfo();
    Heartbeat::GetInstance(0).ClearAllCqeErr("comm_id_1");

    TransportIbverbs::g_flag = false;
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_ClearCqeErr1)
{
    // 清除不存在的通信域中的数据
    MOCKER_CPP(&Heartbeat::GetRetryEnable).stubs().with(mockcpp::any()).will(returnValue(true));
    TransportIbverbs::g_flag = true;
    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value));

    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfoList_value));
    Heartbeat::GetInstance(0).ProcessCqeErrInfo();
    Heartbeat::GetInstance(0).ClearCqeErr("comm_id_1", 24);

    TransportIbverbs::g_flag = false;
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_ClearCqeErr2)
{
    // 清除不存在的通信域中的dstRank
    MOCKER_CPP(&Heartbeat::GetRetryEnable).stubs().with(mockcpp::any()).will(returnValue(true));
    TransportIbverbs::g_flag = true;
    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value));

    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfoList_value));
    Heartbeat::GetInstance(0).ProcessCqeErrInfo();
    Heartbeat::GetInstance(0).ClearCqeErr("comm_id_1", 23);

    TransportIbverbs::g_flag = false;
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_ClearCqeErr3)
{
    MOCKER_CPP(&Heartbeat::GetRetryEnable).stubs().with(mockcpp::any()).will(returnValue(true));
    TransportIbverbs::g_flag = true;
    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfo_value));

    MOCKER(hrtRaGetCqeErrInfoList).stubs().will(invoke(stub_hrtRaGetCqeErrInfoList_value));
    Heartbeat::GetInstance(0).ProcessCqeErrInfo();
    Heartbeat::GetInstance(0).ClearCqeErr("comm_id_1", 24);

    TransportIbverbs::g_flag = false;
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_ClearCqeErr5)
{
    struct timeval time;
    uint32_t status = 0;
    HcclIpAddress remoteIp;
    CqeInfo cqeInfo(time, status, remoteIp);
    LinkInfo linkInfo;
    std::map<u32, std::set<ErrCqeInfo>> rankExtendMap;
    rankExtendMap[5] = {ErrCqeInfo(cqeInfo, linkInfo, 12)};
    Heartbeat::GetInstance(0).rankMapForRetryAgent.insert(std::make_pair("commid_1", rankExtendMap));
    auto ret = Heartbeat::GetInstance(0).ClearCqeErr("commid_1", 5);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rankExtendMap[6] = {ErrCqeInfo(cqeInfo, linkInfo, 14), ErrCqeInfo(cqeInfo, linkInfo, 13)};
    Heartbeat::GetInstance(0).rankMapForRetryAgent.insert(std::make_pair("commid_2", rankExtendMap));
    ret = Heartbeat::GetInstance(0).ClearCqeErr("commid_2", 6, 13);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
TEST_F(HeartBeatTest, ut_IsKeyEvent_OpretryNotSupport)
{
    // 构造异常心跳帧
    RankInfo rankInfo;
    rankInfo.serverId = "127.0.0.1";
    rankInfo.devicePhyId = 0;
    UIDType src = Heartbeat::GetInstance(0).GetUId(rankInfo);
    UIDType dst = Heartbeat::GetInstance(0).GetUId(rankInfo);
    UIDType crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    UIDType informer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    HeartBeatFrame bf(src, dst, crimer, informer, HeartBeatStatus::HEARTBEAT_OPRETRY_NOT_SUPPORT);

    double timeout = static_cast<double>(68);
    MOCKER(GetExternalInputHcclExecTimeOut).stubs().will(returnValue(timeout));
    HcclUs curTime;
    EXPECT_EQ(Heartbeat::GetInstance(0).IsKeyEvent(bf, curTime), true);
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_MakeErrMsg_OpretryNotSupport)
{
    // 构造异常心跳帧
    RankInfo rankInfo;
    rankInfo.serverId = "127.0.0.1";
    rankInfo.devicePhyId = 0;
    UIDType src = Heartbeat::GetInstance(0).GetUId(rankInfo);
    UIDType dst = Heartbeat::GetInstance(0).GetUId(rankInfo);
    UIDType crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    UIDType informer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    HeartBeatFrame bf(src, dst, crimer, informer, HeartBeatStatus::HEARTBEAT_OPRETRY_NOT_SUPPORT);

    std::queue<HeartBeatFrame> keyEvents;
    keyEvents.push(bf);
    std::vector<std::string> errStatusVec;

    Heartbeat::GetInstance(0).MakeErrMsg(keyEvents, errStatusVec);
}

TEST_F(HeartBeatTest, ut_GetHostPort)
{
    MOCKER(GetExternalInputHcclIfBasePort).stubs().will(returnValue(HCCL_INVALID_PORT));
    s32 devicePhyId = 0;
    EXPECT_EQ(Heartbeat::GetInstance(0).GetHostPort(devicePhyId), 60008);
    GlobalMockObject::verify();

    MOCKER(GetExternalInputHcclIfBasePort).stubs().will(returnValue(16666));
    EXPECT_EQ(Heartbeat::GetInstance(0).GetHostPort(devicePhyId), 16674);
    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_cluster_heart_switch_off)
{
    setenv("HCCL_DFS_CONFIG", "cluster_heartbeat:off", 1);

    HcclResult ret = HCCL_SUCCESS;
    ret = InitEnvParam();
    EXPECT_EQ(GetExternalInputHcclHeartBeatEnable(), false);
    unsetenv("HCCL_DFS_CONFIG");

    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_cluster_heart_switch_on)
{
    setenv("HCCL_DFS_CONFIG", "cluster_heartbeat:on", 1);

    HcclResult ret = HCCL_SUCCESS;
    ret = InitEnvParam();
    EXPECT_EQ(GetExternalInputHcclHeartBeatEnable(), true);
    unsetenv("HCCL_DFS_CONFIG");

    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_cluster_heart_env_config)
{
    setenv("HCCL_DFS_CONFIG", "cluster_heartbeat:OFF", 1);

    HcclResult ret = HCCL_SUCCESS;
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputHcclHeartBeatEnable(), false);
    unsetenv("HCCL_DFS_CONFIG");

    setenv("HCCL_DFS_CONFIG", "cluster_heartbeat:Off", 1);

    ret = HCCL_SUCCESS;
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputHcclHeartBeatEnable(), false);
    unsetenv("HCCL_DFS_CONFIG");

    setenv("HCCL_DFS_CONFIG", "cluster_heartbeat:on,cluster_heartbeat:off", 1);

    ret = HCCL_SUCCESS;
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputHcclHeartBeatEnable(), true);
    unsetenv("HCCL_DFS_CONFIG");

    setenv("HCCL_DFS_CONFIG", "cluster_heartbeat:On", 1);

    ret = HCCL_SUCCESS;
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputHcclHeartBeatEnable(), true);
    unsetenv("HCCL_DFS_CONFIG");

    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_stuck_detection_env_config)
{
    setenv("HCCL_DFS_CONFIG", "stuck_detection:on, cluster_heartbeat:OFF", 1);

    HcclResult ret = HCCL_SUCCESS;
    ret = InitEnvParam();
    EXPECT_EQ(GetExternalInputStuckDetect(), true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    unsetenv("HCCL_DFS_CONFIG");

    setenv("HCCL_DFS_CONFIG", "stuck_detection:off, cluster_heartbeat:OFF", 1);

    ret = HCCL_SUCCESS;
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputStuckDetect(), false);
    unsetenv("HCCL_DFS_CONFIG");

    GlobalMockObject::verify();
}

TEST_F(HeartBeatTest, ut_SaveOpInfo)
{
    // SetStatus 函数覆盖
    RankInfo localRank;
    localRank.serverId = "1.1.1.0";
    localRank.devicePhyId = 0;
    UIDType uid_ = Heartbeat::GetInstance(0).GetUId(localRank);
    Heartbeat::GetInstance(0).SetStatus(uid_, uid_, HeartBeatStatus::HEARTBEAT_INCONSISTENT, false);

    OpInfoTagQueueFrame frame;
    frame.opInfoTagQueue[0].opInfoNum = 1;

    // SaveOpInfo 函数覆盖
    OpInfoDesc opInfo;
    OpInfoDesc remoteopInfo;
    opInfo.isValid = false;
    frame.opInfoTagQueue[0].opInfoList[0] = opInfo;
    Heartbeat::GetInstance(0).SaveOpInfo(frame, uid_);
    opInfo.isValid = true;
    frame.opInfoTagQueue[0].opInfoList[0] = opInfo;
    Heartbeat::GetInstance(0).SaveOpInfo(frame, uid_);

    // CheckIsSameOp 函数覆盖
    // 覆盖场景:Send Recv 不匹配
    opInfo.opType = HcclCMDType::HCCL_CMD_SEND;
    remoteopInfo.opType = HcclCMDType::HCCL_CMD_SEND;
    InconsistentType status = InconsistentType::NO_INCONSISTENT;
    HcclResult ret = HCCL_SUCCESS;
    ret = Heartbeat::GetInstance(0).CheckIsSameOp(opInfo, remoteopInfo, status);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 覆盖场景:Recv Send 不匹配
    opInfo.opType = HcclCMDType::HCCL_CMD_RECEIVE;
    remoteopInfo.opType = HcclCMDType::HCCL_CMD_RECEIVE;
    ret = Heartbeat::GetInstance(0).CheckIsSameOp(opInfo, remoteopInfo, status);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 覆盖场景:其他算子 不匹配
    opInfo.opType = HcclCMDType::HCCL_CMD_SCATTER;
    remoteopInfo.opType = HcclCMDType::HCCL_CMD_REDUCE;
    ret = Heartbeat::GetInstance(0).CheckIsSameOp(opInfo, remoteopInfo, status);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 覆盖场景:算子下发 数据类型不匹配
    opInfo.opType = HcclCMDType::HCCL_CMD_SEND;
    remoteopInfo.opType = HcclCMDType::HCCL_CMD_RECEIVE;
    opInfo.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    remoteopInfo.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    ret = Heartbeat::GetInstance(0).CheckIsSameOp(opInfo, remoteopInfo, status);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 覆盖场景:算子下发 数据量不匹配
    opInfo.opType = HcclCMDType::HCCL_CMD_SCATTER;
    remoteopInfo.opType = HcclCMDType::HCCL_CMD_SCATTER;
    opInfo.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    remoteopInfo.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    opInfo.count = 1;
    remoteopInfo.count = 2;
    ret = Heartbeat::GetInstance(0).CheckIsSameOp(opInfo, remoteopInfo, status);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 覆盖场景:算子下发 匹配场景
    opInfo.opType = HcclCMDType::HCCL_CMD_SCATTER;
    remoteopInfo.opType = HcclCMDType::HCCL_CMD_SCATTER;
    opInfo.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    remoteopInfo.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    opInfo.count = 1;
    remoteopInfo.count = 1;
    ret = Heartbeat::GetInstance(0).CheckIsSameOp(opInfo, remoteopInfo, status);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 覆盖场景:算子下发 不匹配场景
    opInfo.opType = HcclCMDType::HCCL_CMD_SCATTER;
    remoteopInfo.opType = HcclCMDType::HCCL_CMD_SCATTER;
    opInfo.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    remoteopInfo.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    opInfo.count = 1;
    remoteopInfo.count = 1;
    opInfo.isValid = true;
    remoteopInfo.isValid = true;
    opInfo.index = 1;
    remoteopInfo.index = 1;
    std::string identifier = "test";
    Heartbeat::GetInstance(0).AddOpInfo(identifier, opInfo, identifier);

    frame.opInfoTagQueue[0].opInfoNum = 1;
    frame.opInfoTagQueue[0].opInfoList[0] = remoteopInfo;
    Heartbeat::GetInstance(0).SaveOpInfo(frame, uid_);
    Heartbeat::GetInstance(0).GetOneOpInfo(identifier, remoteopInfo);
    Heartbeat::GetInstance(0).CheckRecvOpInfoList();

    // 覆盖场景:算子下发 匹配场景
    remoteopInfo.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    frame.opInfoTagQueue[0].opInfoList[0] = remoteopInfo;
    Heartbeat::GetInstance(0).SaveOpInfo(frame, uid_);
    Heartbeat::GetInstance(0).GetOneOpInfo(identifier, remoteopInfo);
    Heartbeat::GetInstance(0).CheckRecvOpInfoList();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 覆盖 GetSendOpInfoList 内 memcpy_s 拷贝 identifier 及返回值校验分支
TEST_F(HeartBeatTest, ut_GetSendOpInfoList)
{
    setenv("HCCL_DFS_CONFIG", "inconsistent_check:on", 1);
    (void)InitEnvParam();

    OpInfoDesc opInfo;
    opInfo.isValid = true;
    std::string tag = "ut_test_op";
    Heartbeat& heartbeat = Heartbeat::GetInstance(0);
    heartbeat.opInfoQueueForSend_.clear();
    heartbeat.opInfoQueueForSend_.push_back(std::make_pair(tag, opInfo));

    OpInfoTagQueueFrame frame;
    memset_s(&frame, sizeof(frame), 0, sizeof(frame));
    heartbeat.GetSendOpInfoList(frame);

    EXPECT_STREQ(frame.opInfoTagQueue[0].identifier, tag.c_str());
    EXPECT_EQ(frame.opInfoTagQueue[0].opInfoNum, 1U);

    unsetenv("HCCL_DFS_CONFIG");
    GlobalMockObject::verify();
}
