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
#include <thread>
#include <chrono>

#include "env_config.h"

#define private public
#define protected public
#include "detect_connect_anomalies.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

class DetectConnectAnomaliesTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DetectConnectAnomaliesTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "DetectConnectAnomaliesTest TearDown" << std::endl; }
    virtual void SetUp()
    {
        dca_ = &DetectConnectionAnomalies::GetInstance(0);
        dca_->isInitThread_ = false;
        dca_->threadExit_ = true;
        dca_->isCreateLink_ = false;
        dca_->isCreateNicLink_ = false;
        dca_->isNeedNic_ = false;
        dca_->getIpNictypeQueue_ = nullptr;
        dca_->detectVnicThread_ = nullptr;
        dca_->detectNicThread_ = nullptr;
        std::cout << "A Test SetUp" << std::endl;
    }
    virtual void TearDown()
    {
        dca_->threadExit_ = false;
        if (dca_->getIpNictypeQueue_ != nullptr && dca_->getIpNictypeQueue_->joinable()) {
            dca_->getIpNictypeQueue_->join();
            dca_->getIpNictypeQueue_ = nullptr;
        }
        if (dca_->detectVnicThread_ != nullptr && dca_->detectVnicThread_->joinable()) {
            dca_->detectVnicThread_->join();
            dca_->detectVnicThread_ = nullptr;
        }
        if (dca_->detectNicThread_ != nullptr && dca_->detectNicThread_->joinable()) {
            dca_->detectNicThread_->join();
            dca_->detectNicThread_ = nullptr;
        }
        for (auto& t : dca_->linkClientThreads_) {
            if (t != nullptr && t->joinable()) {
                t->join();
            }
        }
        dca_->linkClientThreads_.clear();
        dca_->isInitThread_ = false;
        dca_->threadExit_ = true;
        dca_->isCreateLink_ = false;
        dca_->isCreateNicLink_ = false;
        dca_->isNeedNic_ = false;
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
    DetectConnectionAnomalies* dca_ = nullptr;
};

TEST_F(DetectConnectAnomaliesTest, Detect_Success_FirstCall_CreateThread)
{
    EXPECT_FALSE(dca_->isInitThread_);
    EXPECT_TRUE(dca_->threadExit_);
    EXPECT_EQ(dca_->getIpNictypeQueue_, nullptr);

    HcclResult ret = dca_->Detect();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(dca_->isInitThread_);
    EXPECT_NE(dca_->getIpNictypeQueue_, nullptr);
    EXPECT_TRUE(dca_->getIpNictypeQueue_->joinable());
}

TEST_F(DetectConnectAnomaliesTest, Detect_AlreadyInitialized_NoNewThread)
{
    dca_->Detect();
    EXPECT_TRUE(dca_->isInitThread_);
    auto* firstThread = dca_->getIpNictypeQueue_.get();
    ASSERT_NE(firstThread, nullptr);

    HcclResult ret = dca_->Detect();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(dca_->isInitThread_);
    EXPECT_EQ(dca_->getIpNictypeQueue_.get(), firstThread);
}

TEST_F(DetectConnectAnomaliesTest, Detect_ThreadExitFalse_NoThreadCreated)
{
    dca_->threadExit_ = false;

    HcclResult ret = dca_->Detect();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(dca_->isInitThread_);
    EXPECT_EQ(dca_->getIpNictypeQueue_, nullptr);
}

TEST_F(DetectConnectAnomaliesTest, Detect_BothConditionsFalse_NoThreadCreated)
{
    dca_->isInitThread_ = true;
    dca_->threadExit_ = false;

    HcclResult ret = dca_->Detect();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(dca_->isInitThread_);
    EXPECT_EQ(dca_->getIpNictypeQueue_, nullptr);
}

TEST_F(DetectConnectAnomaliesTest, Detect_ThreadCreatedRunsDetectMonitor)
{
    HcclResult ret = dca_->Detect();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(dca_->getIpNictypeQueue_, nullptr);

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    EXPECT_TRUE(dca_->threadExit_);
    EXPECT_TRUE(dca_->isInitThread_);
}

TEST_F(DetectConnectAnomaliesTest, Detect_MultipleCallsOnlyOneThread)
{
    for (int i = 0; i < 5; i++) {
        HcclResult ret = dca_->Detect();
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    EXPECT_TRUE(dca_->isInitThread_);
    EXPECT_NE(dca_->getIpNictypeQueue_, nullptr);
    EXPECT_TRUE(dca_->getIpNictypeQueue_->joinable());
}

TEST_F(DetectConnectAnomaliesTest, CreateServers_ThreadExitFalse_ReturnSuccess)
{
    dca_->threadExit_ = false;
    ErrInfo errInfo;
    HcclResult ret = dca_->CreateServers(errInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(dca_->isCreateLink_);
    EXPECT_EQ(dca_->detectVnicThread_, nullptr);
    EXPECT_EQ(dca_->detectNicThread_, nullptr);
}

TEST_F(DetectConnectAnomaliesTest, CreateServers_CreateVnicThread_Success)
{
    ErrInfo errInfo;
    MOCKER_CPP(&DetectConnectionAnomalies::CreateDetectVnicLinks)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = dca_->CreateServers(errInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(dca_->isCreateLink_);
    ASSERT_NE(dca_->detectVnicThread_, nullptr);
    ASSERT_TRUE(dca_->detectVnicThread_->joinable());
    dca_->detectVnicThread_->join();
    dca_->detectVnicThread_ = nullptr;

    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomaliesTest, CreateServers_CreateNicThread_Success)
{
    dca_->isCreateLink_ = true;
    dca_->isNeedNic_ = true;
    ErrInfo errInfo;

    MOCKER_CPP(&DetectConnectionAnomalies::CreateDetectNicLinks)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = dca_->CreateServers(errInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(dca_->isCreateNicLink_);
    ASSERT_NE(dca_->detectNicThread_, nullptr);
    ASSERT_TRUE(dca_->detectNicThread_->joinable());
    dca_->detectNicThread_->join();
    dca_->detectNicThread_ = nullptr;

    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomaliesTest, CreateServers_VnicAlreadyCreated_SkipVnic)
{
    dca_->isCreateLink_ = true;
    ErrInfo errInfo;

    HcclResult ret = dca_->CreateServers(errInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(dca_->detectVnicThread_, nullptr);
}

TEST_F(DetectConnectAnomaliesTest, CreateClients_Success)
{
    ErrInfo errInfo;
    std::vector<std::unique_ptr<std::thread>> linkClientThreads;

    MOCKER_CPP(&DetectConnectionAnomalies::CreateClient).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = dca_->CreateClients(errInfo, linkClientThreads);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(linkClientThreads.size(), 1u);
    ASSERT_TRUE(linkClientThreads[0]->joinable());
    linkClientThreads[0]->join();

    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomaliesTest, CreateClient_Success_ClientSocketSaved)
{
    ErrInfo errInfo;
    errInfo.nicType = NicType::VNIC_TYPE;
    errInfo.localRankInfo.devicePhyId = 0;
    errInfo.deviceLogicId = HOST_DEVICE_ID; // 跳过 hrtSetDevice/hrtResetDevice
    errInfo.localRankInfo.serverId = "test_server_id";
    errInfo.remoteRankInfo.serverId = "remote_server_id";

    HcclIpAddress localIp("192.168.1.1");
    HcclIpAddress remoteIp("192.168.1.2");
    errInfo.localRankInfo.deviceVnicIp = localIp;
    errInfo.remoteRankInfo.deviceVnicIp = remoteIp;
    errInfo.remoteRankInfo.deviceVnicPort = 16677;

    // 预设置 vnicCtx_ 为非空，跳过 HcclNetOpenDev
    dca_->vnicCtx_ = reinterpret_cast<HcclNetDevCtx>(0x1);

    // Mock GetExternalInputDfsConnectionFaultDetectionTime
    MOCKER(GetExternalInputDfsConnectionFaultDetectionTime).stubs().will(returnValue(1));

    // Mock HcclSocket::Init
    MOCKER_CPP(&HcclSocket::Init).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // Mock HcclSocket::Connect
    MOCKER_CPP(&HcclSocket::Connect).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // Mock HcclSocket::GetStatus 返回 SOCKET_OK，使 GetStatus 立即返回成功
    MOCKER_CPP(&HcclSocket::GetStatus).stubs().with(mockcpp::any()).will(returnValue(HcclSocketStatus::SOCKET_OK));

    // 记录原始 clientSockets_ 大小
    size_t originalSize = dca_->clientSockets_.size();

    // 设置 threadExit_ = false 让 while 循环快速退出
    dca_->threadExit_ = false;
    dca_->broadCastTime = 0;

    // 执行 CreateClient，覆盖 lines 506-509
    HcclResult ret = dca_->CreateClient(errInfo);

    // 验证返回成功
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 验证 clientSocket 被正确保存到 clientSockets_
    EXPECT_EQ(dca_->clientSockets_.size(), originalSize + 1);
    EXPECT_NE(dca_->clientSockets_.back(), nullptr);

    // 清理
    dca_->clientSockets_.clear();
    dca_->vnicCtx_ = nullptr;

    GlobalMockObject::verify();
}
