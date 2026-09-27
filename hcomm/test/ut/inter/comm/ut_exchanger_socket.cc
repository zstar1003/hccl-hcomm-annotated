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
#include <stdio.h>
#include <mockcpp/mockcpp.hpp>
#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "hccl_comm_pub.h"
#include "llt_hccl_stub_pub.h"
#include "llt_hccl_stub.h"
#include "dlra_function.h"
#include "sal.h"
#define private public
#define protected public
#include "exchanger_socket_pub.h"
#include "exchanger_network_pub.h"
#undef private
#undef protected
#include <externalinput_pub.h>
#include "network_manager_pub.h"
#include "tsd/tsd_client.h"
#include "dltdt_function.h"

using namespace std;
using namespace hccl;

HcclResult stub_GetRaResourceInfo_exchangerSocketTest(NetworkManager* that, RaResourceInfo& raResourceInfo)
{
    static bool initialized = false;
    static RaResourceInfo fake_raResourceInfo;
    static int fake_handle = 1;
    if (!initialized) {
        IpSocket tmpIpSocket;
        fake_raResourceInfo.vnicSocketHandle = &fake_handle;
        tmpIpSocket.nicSocketHandle = &fake_handle;
        HcclIpAddress ipAddr(1684515008);
        for (int i = 0; i < 8; i++) {
            fake_raResourceInfo.nicSocketMap[ipAddr] = tmpIpSocket;
        }
    }
    raResourceInfo = fake_raResourceInfo;
    return HCCL_SUCCESS;
}

s32 stub_exchangerSocketTest_hrtRaSocketNonBlockSendHB(
    const FdHandle fdHandle, const void* data, u64 size, u64* sent_size)
{
    *sent_size = size;
    return 0;
}

32 stub_exchangerSocketTest_hrtRaSocketNonBlockRecvHB(const FdHandle fdHandle, void* data, u64 size, u64* recvSize)
{
    static u32 count = 0;
    if (count++ % 5 != 0) {
        *recvSize = size;
        count = 0;
    }
    return 0;
}

HcclResult stub_exchangerSocketTest_hrtRaBlockGetSockets(u32 role, struct SocketInfoT conn[], u32 num)
{
    static std::vector<int> fdHandle;
    for (int i = 0; i < num; i++) {
        fdHandle.push_back(0);
        conn[i].fd_handle = &fdHandle[fdHandle.size() - 1];
        conn[i].status = CONNECT_OK;
    }
    return HCCL_SUCCESS;
}

void get_ranks_8server_1dev_exchangerSocketTest(std::vector<RankInfo>& rank_vector)
{
    RankInfo tmp_para_0;
    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicIp.push_back(HcclIpAddress("192.168.0.11"));
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_1;
    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverIdx = 1;
    tmp_para_1.serverId = "10.0.1.10";
    tmp_para_1.nicIp.push_back(HcclIpAddress("192.168.0.12"));
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_2;
    tmp_para_2.userRank = 2;
    tmp_para_2.devicePhyId = 2;
    tmp_para_2.deviceType = DevType::DEV_TYPE_910;
    tmp_para_2.serverIdx = 2;
    tmp_para_2.serverId = "10.0.2.10";
    tmp_para_2.nicIp.push_back(HcclIpAddress("192.168.0.13"));
    tmp_para_2.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_3;
    tmp_para_3.userRank = 3;
    tmp_para_3.devicePhyId = 3;
    tmp_para_3.deviceType = DevType::DEV_TYPE_910;
    tmp_para_3.serverIdx = 3;
    tmp_para_3.serverId = "10.0.3.10";
    tmp_para_3.nicIp.push_back(HcclIpAddress("192.168.0.14"));
    tmp_para_3.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_4;
    tmp_para_4.userRank = 4;
    tmp_para_4.devicePhyId = 4;
    tmp_para_4.deviceType = DevType::DEV_TYPE_910;
    tmp_para_4.serverIdx = 4;
    tmp_para_4.serverId = "10.0.4.10";
    tmp_para_4.nicIp.push_back(HcclIpAddress("192.168.0.15"));
    tmp_para_4.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_5;
    tmp_para_5.userRank = 5;
    tmp_para_5.devicePhyId = 5;
    tmp_para_5.deviceType = DevType::DEV_TYPE_910;
    tmp_para_5.serverIdx = 5;
    tmp_para_5.serverId = "10.0.5.10";
    tmp_para_5.nicIp.push_back(HcclIpAddress("192.168.0.16"));
    tmp_para_5.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_6;
    tmp_para_6.userRank = 6;
    tmp_para_6.devicePhyId = 6;
    tmp_para_6.deviceType = DevType::DEV_TYPE_910;
    tmp_para_6.serverIdx = 6;
    tmp_para_6.serverId = "10.0.6.10";
    tmp_para_6.nicIp.push_back(HcclIpAddress("192.168.0.17"));
    tmp_para_6.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_7;
    tmp_para_7.userRank = 7;
    tmp_para_7.devicePhyId = 7;
    tmp_para_7.deviceType = DevType::DEV_TYPE_910;
    tmp_para_7.serverIdx = 7;
    tmp_para_7.serverId = "10.0.7.10";
    tmp_para_7.nicIp.push_back(HcclIpAddress("192.168.0.18"));
    tmp_para_7.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    rank_vector.push_back(tmp_para_0);
    rank_vector.push_back(tmp_para_1);
    rank_vector.push_back(tmp_para_2);
    rank_vector.push_back(tmp_para_3);
    rank_vector.push_back(tmp_para_4);
    rank_vector.push_back(tmp_para_5);
    rank_vector.push_back(tmp_para_6);
    rank_vector.push_back(tmp_para_7);
    return;
}

class ExchangerSocketTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--ExchangerSocket SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--ExchangerSocket TearDown--\033[0m" << std::endl; }
    virtual void SetUp()
    {
        MOCKER(hrtRaSocketWhiteListAdd).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtRaSocketWhiteListDel).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtRaSocketBatchConnect).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtRaBlockGetSockets).stubs().will(invoke(stub_exchangerSocketTest_hrtRaBlockGetSockets));
        MOCKER(hrtRaSocketBatchClose).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtRaSocketNonBlockSend).stubs().will(invoke(stub_exchangerSocketTest_hrtRaSocketNonBlockSendHB));
        MOCKER(hrtRaSocketNonBlockRecv).stubs().will(invoke(stub_exchangerSocketTest_hrtRaSocketNonBlockRecvHB));

        MOCKER_CPP(&NetworkManager::GetRaResourceInfo).stubs().will(invoke(stub_GetRaResourceInfo_exchangerSocketTest));
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(ExchangerSocketTest, ut_ExchangerSocket_init)
{
    std::string identifier = "my tag";
    u32 userRank = 8;
    u32 userRankSize = 8;
    std::vector<s32> deviceIds;
    deviceIds.push_back(1);
    deviceIds.push_back(2);

    std::vector<u32> userRanks;
    userRanks.push_back(1);
    userRanks.push_back(2);
    std::string tag = "my tag";
    std::vector<RankInfo> rankVector;
    get_ranks_8server_1dev_exchangerSocketTest(rankVector);
    std::map<u32, NetworkInfo> rankIpMap;
    bool isHaveCpuRank_;
    for (auto& rank : userRanks) {
        for (auto& rankInfo : rankVector) {
            if (rankInfo.userRank == rank) {
                NetworkInfo info;
                info.ip = rankInfo.hostIp;
                info.port = HOST_PARA_BASE_PORT + rankInfo.devicePhyId;
                rankIpMap.insert(std::make_pair(rank, info));
                break;
            }
        }
    }
    ExchangerSocket Socket(identifier, userRank, userRankSize, deviceIds, userRanks, tag, rankIpMap);
    HcclResult ret = Socket.Init();
    EXPECT_EQ(ret, HCCL_E_PARA);
    std::vector<SocketInfoT> socketInfoVer;
    SocketInfoT socketInfo;
    socketInfo.socket_handle = (void*)0x00000001;
    socketInfoVer.push_back(socketInfo);
    ret = Socket.GetSockets(0, userRanks, socketInfoVer);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = Socket.GetSockets(1, userRanks, socketInfoVer);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = Socket.ClientConnect(userRanks);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string message = "test";

    ret = Socket.Send(userRank, message);
    EXPECT_EQ(ret, HCCL_E_PARA);
    const char* str = "buff";
    const u8* sendBuf = (u8*)str;
    u8* recvBuf = (u8*)str;
    ret = Socket.Send(userRank, sendBuf, 4);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = Socket.Recv(userRank, recvBuf, 4);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = Socket.Recv(userRank, message);
    EXPECT_EQ(ret, HCCL_E_PARA);

    u32 size = 2;
    ret = Socket.Send(size, message);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = Socket.Send(size, sendBuf, 4);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = Socket.Recv(size, recvBuf, 4);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = Socket.Recv(size, message);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
