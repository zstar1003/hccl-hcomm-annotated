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
#include <iostream>
#include <vector>
#include <algorithm>

// 假设SimpleVaAllocator在这些头文件中
#define private public
#define protected public
#include "symmetric_memory.h" // 替换为实际的头文件
#undef private
#undef protected

using namespace std;
using namespace hccl;

class SymmetricMemoryAgentTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SymmetricMemoryAgentTest Testcase SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "SymmetricMemoryAgentTest Testcase TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "A SymmetricMemoryAgentTest SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A SymmetricMemoryAgentTest TearDown" << std::endl;
    }
};

void get_ranks_1server_3dev(std::vector<RankInfo>& rank_vector)
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
    tmp_para_1.serverIdx = 0;
    tmp_para_1.serverId = "10.0.0.10";
    tmp_para_1.nicIp.push_back(HcclIpAddress("192.168.0.12"));
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_2;

    tmp_para_2.userRank = 2;
    tmp_para_2.devicePhyId = 2;
    tmp_para_2.deviceType = DevType::DEV_TYPE_910;
    tmp_para_2.serverIdx = 0;
    tmp_para_2.serverId = "10.0.0.10";
    tmp_para_2.nicIp.push_back(HcclIpAddress("192.168.0.13"));
    tmp_para_2.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    rank_vector.push_back(tmp_para_0);
    rank_vector.push_back(tmp_para_1);
    rank_vector.push_back(tmp_para_2);
    return;
}

TEST_F(SymmetricMemoryAgentTest, ut_ExchangeInfo_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    MOCKER_CPP(hrtRaGetSingleSocketVnicIpInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::WaitLinkEstablish).stubs().will(returnValue(HCCL_SUCCESS));
    std::unique_ptr<HcclSocketManager> socketManager;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    HcclIpAddress localIPs(0x01);
    std::vector<RankInfo> rank_vector;
    get_ranks_1server_3dev(rank_vector);
    SymmetricMemoryAgent symmetricMemoryAgent(
        socketManager, 0, 0, localIPs, rank_vector, 0, true, "SymmetricMemoryAgentTest");
    EXPECT_EQ(symmetricMemoryAgent.Init(), HCCL_SUCCESS);

    HcclResult ret = HCCL_SUCCESS;
    u64 len = PACKET_TOTAL_LEN;
    Packet dataPkt;
    dataPkt.rankId = 1;
    dataPkt.type = MsgType::MSG_TYPE_DATA;
    int a = 1;
    memcpy_s(dataPkt.data, PACKET_DATA_MAX_LEN, &a, sizeof(int));
    MOCKER_CPP(&HcclSocket::IRecv)
        .stubs()
        .with(outBoundP(reinterpret_cast<void*>(&dataPkt), sizeof(Packet)), mockcpp::any(), outBound(len))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocket::Send, HcclResult(HcclSocket::*)(const void*, u64))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(1));

    int input_temp = 0;
    std::vector<int> output_temp(3, -1);
    ret = symmetricMemoryAgent.ExchangeInfo((void*)&input_temp, (void*)output_temp.data(), sizeof(int));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (u32 i = 0; i < 2; i++) {
        EXPECT_EQ(output_temp[i], i);
    }
}

TEST_F(SymmetricMemoryAgentTest, ut_Init_When_RankSize_Is_One_Expect_ReturnHCCL_E_PARA)
{
    std::unique_ptr<HcclSocketManager> socketManager;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    HcclIpAddress localIPs(0x01);
    std::vector<RankInfo> rank_vector(1);
    SymmetricMemoryAgent symmetricMemoryAgent(
        socketManager, 0, 0, localIPs, rank_vector, 0, true, "SymmetricMemoryAgentTest");
    EXPECT_EQ(symmetricMemoryAgent.Init(), HCCL_E_PARA);
}

TEST_F(SymmetricMemoryAgentTest, ut_ExchangeInfo_When_Para_Is_Error_Expect_ReturnHCCL_E_PARA)
{
    MOCKER_CPP(hrtRaGetSingleSocketVnicIpInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::WaitLinkEstablish).stubs().will(returnValue(HCCL_SUCCESS));
    std::unique_ptr<HcclSocketManager> socketManager;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    HcclIpAddress localIPs(0x01);
    std::vector<RankInfo> rank_vector;
    get_ranks_1server_3dev(rank_vector);
    SymmetricMemoryAgent symmetricMemoryAgent(
        socketManager, 0, 0, localIPs, rank_vector, 0, true, "SymmetricMemoryAgentTest");
    EXPECT_EQ(symmetricMemoryAgent.Init(), HCCL_SUCCESS);

    HcclResult ret = HCCL_SUCCESS;
    int input_temp = 0;
    std::vector<int> output_temp(3, -1);
    ret = symmetricMemoryAgent.ExchangeInfo(nullptr, (void*)output_temp.data(), sizeof(int));
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = symmetricMemoryAgent.ExchangeInfo((void*)&input_temp, nullptr, sizeof(int));
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = symmetricMemoryAgent.ExchangeInfo((void*)&input_temp, (void*)output_temp.data(), 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = symmetricMemoryAgent.ExchangeInfo((void*)&input_temp, (void*)output_temp.data(), PACKET_DATA_MAX_LEN + 1);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(SymmetricMemoryAgentTest, ut_ExchangeInfo_When_Link_Is_Failed_Expect_ReturnHCCL_E_INTERNAL)
{
    MOCKER_CPP(hrtRaGetSingleSocketVnicIpInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&SymmetricMemoryAgent::EstablishSockets).stubs().will(returnValue(HCCL_SUCCESS));
    std::unique_ptr<HcclSocketManager> socketManager;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    HcclIpAddress localIPs(0x01);
    std::vector<RankInfo> rank_vector;
    get_ranks_1server_3dev(rank_vector);
    SymmetricMemoryAgent symmetricMemoryAgent(
        socketManager, 0, 0, localIPs, rank_vector, 0, true, "SymmetricMemoryAgentTest");
    EXPECT_EQ(symmetricMemoryAgent.Init(), HCCL_SUCCESS);

    HcclResult ret = HCCL_SUCCESS;
    int input_temp = 0;
    std::vector<int> output_temp(3, -1);
    ret = symmetricMemoryAgent.ExchangeInfo((void*)&input_temp, (void*)output_temp.data(), sizeof(int));
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(SymmetricMemoryAgentTest, ut_ExchangeInfo_When_TimeOut_Expect_ReturnHCCL_E_TCP_TRANSFER)
{
    MOCKER_CPP(hrtRaGetSingleSocketVnicIpInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::WaitLinkEstablish).stubs().will(returnValue(HCCL_SUCCESS));
    std::unique_ptr<HcclSocketManager> socketManager;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    HcclIpAddress localIPs(0x01);
    std::vector<RankInfo> rank_vector;
    get_ranks_1server_3dev(rank_vector);
    SymmetricMemoryAgent symmetricMemoryAgent(
        socketManager, 0, 0, localIPs, rank_vector, 0, true, "SymmetricMemoryAgentTest");
    EXPECT_EQ(symmetricMemoryAgent.Init(), HCCL_SUCCESS);

    HcclResult ret = HCCL_SUCCESS;
    u64 len = PACKET_TOTAL_LEN;
    Packet dataPkt;
    dataPkt.rankId = 1;
    dataPkt.type = MsgType::MSG_TYPE_DATA;
    int a = 1;
    memcpy_s(dataPkt.data, PACKET_DATA_MAX_LEN, &a, sizeof(int));
    MOCKER_CPP(&HcclSocket::IRecv)
        .stubs()
        .with(outBoundP(reinterpret_cast<void*>(&dataPkt), sizeof(Packet)), mockcpp::any(), outBound(len))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocket::Send, HcclResult(HcclSocket::*)(const void*, u64))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(0));

    int input_temp = 0;
    std::vector<int> output_temp(3, -1);
    ret = symmetricMemoryAgent.ExchangeInfo((void*)&input_temp, (void*)output_temp.data(), sizeof(int));
    EXPECT_EQ(ret, HCCL_E_TCP_TRANSFER);
}

TEST_F(SymmetricMemoryAgentTest, ut_ExchangeInfo_When_RecvIsFailed_Expect_ReturnHCCL_E_TCP_TRANSFER)
{
    MOCKER_CPP(hrtRaGetSingleSocketVnicIpInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::WaitLinkEstablish).stubs().will(returnValue(HCCL_SUCCESS));
    std::unique_ptr<HcclSocketManager> socketManager;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    HcclIpAddress localIPs(0x01);
    std::vector<RankInfo> rank_vector;
    get_ranks_1server_3dev(rank_vector);
    SymmetricMemoryAgent symmetricMemoryAgent(
        socketManager, 0, 0, localIPs, rank_vector, 0, true, "SymmetricMemoryAgentTest");
    EXPECT_EQ(symmetricMemoryAgent.Init(), HCCL_SUCCESS);

    HcclResult ret = HCCL_SUCCESS;
    u64 len = PACKET_TOTAL_LEN;
    Packet dataPkt;
    dataPkt.rankId = 1;
    dataPkt.type = MsgType::MSG_TYPE_DATA;
    int a = 1;
    memcpy_s(dataPkt.data, PACKET_DATA_MAX_LEN, &a, sizeof(int));
    MOCKER_CPP(&HcclSocket::IRecv).stubs().will(returnValue(HCCL_E_TCP_TRANSFER));
    MOCKER_CPP(&HcclSocket::Send, HcclResult(HcclSocket::*)(const void*, u64))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(1));

    int input_temp = 0;
    std::vector<int> output_temp(3, -1);
    ret = symmetricMemoryAgent.ExchangeInfo((void*)&input_temp, (void*)output_temp.data(), sizeof(int));
    EXPECT_EQ(ret, HCCL_E_TCP_TRANSFER);
}

TEST_F(SymmetricMemoryAgentTest, ut_ExchangeInfo_When_SendIsFailed_Expect_ReturnHCCL_E_TCP_TRANSFER)
{
    MOCKER_CPP(hrtRaGetSingleSocketVnicIpInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::WaitLinkEstablish).stubs().will(returnValue(HCCL_SUCCESS));
    std::unique_ptr<HcclSocketManager> socketManager;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    HcclIpAddress localIPs(0x01);
    std::vector<RankInfo> rank_vector;
    get_ranks_1server_3dev(rank_vector);
    SymmetricMemoryAgent symmetricMemoryAgent(
        socketManager, 0, 0, localIPs, rank_vector, 0, true, "SymmetricMemoryAgentTest");
    EXPECT_EQ(symmetricMemoryAgent.Init(), HCCL_SUCCESS);

    HcclResult ret = HCCL_SUCCESS;
    u64 len = PACKET_TOTAL_LEN;
    Packet dataPkt;
    dataPkt.rankId = 1;
    dataPkt.type = MsgType::MSG_TYPE_DATA;
    int a = 1;
    memcpy_s(dataPkt.data, PACKET_DATA_MAX_LEN, &a, sizeof(int));
    MOCKER_CPP(&HcclSocket::IRecv)
        .stubs()
        .with(outBoundP(reinterpret_cast<void*>(&dataPkt), sizeof(Packet)), mockcpp::any(), outBound(len))
        .will(returnValue(HCCL_E_TCP_TRANSFER));
    MOCKER_CPP(&HcclSocket::Send, HcclResult(HcclSocket::*)(const void*, u64))
        .stubs()
        .will(returnValue(HCCL_E_TCP_TRANSFER));
    MOCKER_CPP(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(1));

    int input_temp = 0;
    std::vector<int> output_temp(3, -1);
    ret = symmetricMemoryAgent.ExchangeInfo((void*)&input_temp, (void*)output_temp.data(), sizeof(int));
    EXPECT_EQ(ret, HCCL_E_TCP_TRANSFER);
}
