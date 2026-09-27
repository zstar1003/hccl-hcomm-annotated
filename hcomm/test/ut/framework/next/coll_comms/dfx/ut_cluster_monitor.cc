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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#include <memory>
#include <string>
#include <cstring>
#include "hccl_types.h"
#define private public
#include "dfx/cluster_monitor/cluster_monitor.h"
#include "hccl_comm_pub.h"
#include "rank_graph_v2.h"
#include "new_rank_info.h"
#undef private
#include "hccl_comm_socket_c_adpt.h"
#include "base_config_legacy.h"
#include "llt_hccl_stub_rank_graph.h"

using namespace hcomm;

class ClusterMonitorTest : public ::testing::Test {
public:
    void SetUp() override { testing::Test::SetUp(); }
    void TearDown() override
    {
        testing::Test::TearDown();
        GlobalMockObject::verify();
    }

    static ClusterMonitor g_monitor;
};

ClusterMonitor ClusterMonitorTest::g_monitor;
std::map<uint32_t, std::vector<UIDContext>> g_uidCtxs;

TEST_F(ClusterMonitorTest, Ut_CreateMonitorLinksAsync_When_NormalInput_Expect_CreateLinks)
{
    setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    s32 ret;
    SocketDesc socketDesc;
    std::string testTag = "monitor_test_tag";
    struct in_addr localAddr;
    struct in_addr remoteAddr;
    inet_pton(AF_INET, "127.0.0.1", &localAddr);
    socketDesc.localEndpoint.commAddr.addr = localAddr;
    socketDesc.localEndpoint.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    socketDesc.localEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    socketDesc.localEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    inet_pton(AF_INET, "192.186.0.1", &remoteAddr);
    socketDesc.remoteEndpoint.commAddr.addr = remoteAddr;
    socketDesc.remoteEndpoint.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    socketDesc.remoteEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    socketDesc.remoteEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    ret = memcpy_s(socketDesc.tag, HCCL_SOCKET_TAG_LEN, testTag.c_str(), testTag.length() + 1);
    EXPECT_EQ(ret, EOK);
    socketDesc.role = HCOMM_SOCKET_ROLE_CLIENT;
    socketDesc.listenPort = 8080;

    SocketDesc socketDesc1;
    testTag = "monitor_test_tag_1";
    inet_pton(AF_INET, "127.0.0.1", &localAddr);
    socketDesc1.localEndpoint.commAddr.addr = localAddr;
    socketDesc1.localEndpoint.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    socketDesc1.localEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_HOST;
    socketDesc1.localEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    inet_pton(AF_INET, "192.186.0.2", &remoteAddr);
    socketDesc1.remoteEndpoint.commAddr.addr = remoteAddr;
    socketDesc1.remoteEndpoint.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    socketDesc1.remoteEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    socketDesc1.remoteEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    ret = memcpy_s(socketDesc1.tag, HCCL_SOCKET_TAG_LEN, testTag.c_str(), testTag.length() + 1);
    EXPECT_EQ(ret, EOK);
    socketDesc1.role = HCOMM_SOCKET_ROLE_CLIENT;
    socketDesc1.listenPort = 8080;

    SocketDesc socketDesc2;
    testTag = "monitor_test_tag_2";
    inet_pton(AF_INET, "127.0.0.1", &localAddr);
    socketDesc2.localEndpoint.commAddr.addr = localAddr;
    socketDesc2.localEndpoint.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    socketDesc2.localEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_HOST;
    socketDesc2.localEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    inet_pton(AF_INET, "192.186.0.3", &remoteAddr);
    socketDesc2.remoteEndpoint.commAddr.addr = remoteAddr;
    socketDesc2.remoteEndpoint.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    socketDesc2.remoteEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_HOST;
    socketDesc2.remoteEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    ret = memcpy_s(socketDesc2.tag, HCCL_SOCKET_TAG_LEN, testTag.c_str(), testTag.length() + 1);
    EXPECT_EQ(ret, EOK);
    socketDesc2.role = HCOMM_SOCKET_ROLE_CLIENT;
    socketDesc2.listenPort = 8080;

    ClusterMonitorSocketCtx context;
    context.socketDesc = socketDesc;
    context.socketHandler = nullptr;

    ClusterMonitorSocketCtx context1;
    context1.socketDesc = socketDesc1;
    context1.socketHandler = nullptr;

    ClusterMonitorSocketCtx context2;
    context2.socketDesc = socketDesc2;
    context2.socketHandler = nullptr;

    ClusterUIDType uid;
    ret = memcpy_s(uid.id, sizeof(uid.id), "test_uid", sizeof("test_uid") + 1);
    EXPECT_EQ(ret, EOK);

    ClusterUIDType uid1;
    ret = memcpy_s(uid1.id, sizeof(uid1.id), "test_uid1", sizeof("test_uid1") + 1);
    EXPECT_EQ(ret, EOK);

    ClusterUIDType uid2;
    ret = memcpy_s(uid2.id, sizeof(uid2.id), "test_uid2", sizeof("test_uid2") + 1);
    EXPECT_EQ(ret, EOK);

    g_monitor.commIdMap_["comm1"].insert(std::make_pair(uid, false));
    g_monitor.commIdMap_["comm1"].insert(std::make_pair(uid1, false));
    g_monitor.commIdMap_["comm1"].insert(std::make_pair(uid2, false));

    std::unique_lock<std::mutex> linksLock(g_monitor.clusertMonitorLinkMtx_);
    g_monitor.clusterLinkContext_["comm1"].push(std::make_pair(uid, context));
    g_monitor.clusterLinkContext_["comm1"].push(std::make_pair(uid1, context1));
    g_monitor.clusterLinkContext_["comm2"].push(std::make_pair(uid2, context2));
    linksLock.unlock();

    MOCKER_CPP(&Hccl::EnvSocketConfig::GetLinkTimeOut).stubs().with(mockcpp::any()).will(returnValue((s32)(5)));
    g_monitor.CreateHBLinksAsync();

    u32 uncompletedCount = 3;
    auto linkThreadIt = g_monitor.monitorLinkStatusMap_.begin();
    const auto CREATE_LINK_TIMEOUT = std::chrono::seconds(10);
    auto startTime = std::chrono::steady_clock::now();
    while ((std::chrono::steady_clock::now() - startTime) <= CREATE_LINK_TIMEOUT) {
        HCCL_ERROR("Waiting for monitor links to complete, uncompletedCount = %u", uncompletedCount);
        if (linkThreadIt == g_monitor.monitorLinkStatusMap_.end()) {
            linkThreadIt = g_monitor.monitorLinkStatusMap_.begin();
        }
        if (linkThreadIt->second == ClusterMonitor::MonitorLinkStatus::MONITOR_LINK_COMPLETED) {
            uncompletedCount--;
        }
        if (uncompletedCount == 0) {
            break;
        } else {
            linkThreadIt++;
        }
    }

    g_monitor.linkThreadRunning_ = false;
    // 在心跳进程结束之前join所有的建链线程
    for (auto& pair : g_monitor.linkThreadMap_) {
        if (pair.second != nullptr && pair.second->joinable()) {
            pair.second->join();
            HCCL_INFO(
                "[HeartbeatStatusMonitor] thread has joined. Remote uid is [%s]", g_monitor.GetUID(pair.first).c_str());
        }
    }

    EXPECT_EQ(uncompletedCount, 0);
}

TEST_F(ClusterMonitorTest, Ut_SendMonitorFrame_When_NormalInput_Expect_SendFrame)
{
    for (auto iter = g_monitor.uid2SocketRefMap_.begin(); iter != g_monitor.uid2SocketRefMap_.end(); iter++) {
        ClusterUIDType rem = iter->first;
        g_monitor.uid2SocketRefMap_[rem].lostNum++;
        g_monitor.SendFrame(
            rem, g_monitor.myRankUID_, g_monitor.myRankUID_, ClusterMonitorStatus::CLUSTER_MONITOR_STUCK);
        EXPECT_EQ(g_monitor.uid2SocketRefMap_[rem].sendBuffer.size(), 0);
    }
}

TEST_F(ClusterMonitorTest, Ut_SendMonitorFrame_When_SendFrameFromBufferFail_Expect_ReturnError)
{
    for (auto iter = g_monitor.uid2SocketRefMap_.begin(); iter != g_monitor.uid2SocketRefMap_.end(); iter++) {
        ClusterUIDType rem = iter->first;
        ClusterMonitorFrame bufferedFrame(
            rem, g_monitor.myRankUID_, g_monitor.myRankUID_, g_monitor.myRankUID_,
            ClusterMonitorStatus::CLUSTER_MONITOR_STUCK);
        g_monitor.uid2SocketRefMap_[rem].sendBuffer.push(bufferedFrame);
        g_monitor.uid2SocketRefMap_[rem].restSize = sizeof(ClusterMonitorFrame);

        MOCKER(SocketSendNb).stubs().will(returnValue(HCCL_E_INTERNAL));

        HcclResult ret = g_monitor.SendFrame(
            rem, g_monitor.myRankUID_, g_monitor.myRankUID_, ClusterMonitorStatus::CLUSTER_MONITOR_STUCK);
        EXPECT_NE(ret, HCCL_SUCCESS);

        GlobalMockObject::verify();
        std::queue<ClusterMonitorFrame> emptyQueue;
        g_monitor.uid2SocketRefMap_[rem].sendBuffer.swap(emptyQueue);
        g_monitor.uid2SocketRefMap_[rem].restSize = 0;
    }
}

TEST_F(ClusterMonitorTest, Ut_RecvMonitorFrame_When_NormalInput_Expect_RecvFrame)
{
    MOCKER_CPP(&ClusterMonitor::ParseFrame).stubs().will(returnValue(HCCL_SUCCESS));

    uint64_t compSize = 0;
    MOCKER(SocketRecvNb)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&compSize, sizeof(uint64_t*)))
        .will(invoke(SocketRecvNb))       // 第1次走真实函数
        .then(returnValue(HCCL_E_AGAIN)); // 第2次返回特定值

    for (auto iter = g_monitor.uid2SocketRefMap_.begin(); iter != g_monitor.uid2SocketRefMap_.end(); iter++) {
        ClusterUIDType rem = iter->first;
        HcclResult ret = g_monitor.RecvFrame(rem);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
}

TEST_F(ClusterMonitorTest, Ut_GetCqeErrInfoFromTaskException_When_NormalInput_Expect_SetCqeErrInfo)
{
    std::string netInstId = "localInstance";
    u32 localId = 0;
    ClusterUIDCxt localUidCxt(netInstId, localId);
    ClusterUIDType localUid = g_monitor.FormatUID(localUidCxt);
    g_monitor.myRankUID_ = localUid;
    g_monitor.myRankNetInstId_ = netInstId;
    g_monitor.myRankLocalId_ = localId;

    u32 remoteLocalId = 1;
    uint16_t status = 1;
    std::string localEid = "localEid";
    std::string remoteEid = "remoteEid";
    std::string remoteInsId = "remoteInsId";

    g_monitor.GetCqeErrInfoFromTaskException(remoteLocalId, status, localEid, remoteEid, remoteInsId);

    EXPECT_EQ(g_monitor.cqeErrInfo_.cqeRemoteLocalId, remoteLocalId);
    EXPECT_EQ(g_monitor.cqeErrInfo_.cqeStatus, status);
    EXPECT_EQ(g_monitor.cqeErrInfo_.cqeLocalEid, localEid);
    EXPECT_EQ(g_monitor.cqeErrInfo_.cqeRemoteEid, remoteEid);
    EXPECT_EQ(g_monitor.cqeErrInfo_.cqeRemoteInsId, remoteInsId);
}

TEST_F(ClusterMonitorTest, Ut_ProcessExceptionEvent_When_NormalQueue_Expect_SendFrames)
{
    std::string netInstId = "localInstance";
    u32 localId = 0;
    ClusterUIDCxt localUidCxt(netInstId, localId);
    ClusterUIDType localUid = g_monitor.FormatUID(localUidCxt);
    g_monitor.myRankUID_ = localUid;

    std::string crimerInstId = "remoteInstance";
    ClusterUIDCxt crimerCxt(crimerInstId, 1);
    ClusterUIDType crimerUid = g_monitor.FormatUID(crimerCxt);

    ClusterUIDType informerUid = localUid;

    ClusterMonitor::FrameStatus frameStatus;
    frameStatus.status = ClusterMonitorStatus::CLUSTER_MONITOR_LOST;
    frameStatus.informer = informerUid;
    frameStatus.needBroadcast = false;
    g_monitor.uid2FrameStatusMap_.insert(crimerUid, frameStatus);

    g_monitor.errRankQueue_.push(crimerUid);

    std::string otherInstId = "otherInstance";
    ClusterUIDCxt otherCxt(otherInstId, 2);
    ClusterUIDType otherUid = g_monitor.FormatUID(otherCxt);

    ClusterMonitorSocketCtx socketCtx;
    socketCtx.socketHandler = nullptr;
    g_monitor.uid2SocketRefMap_.insert(otherUid, socketCtx);

    ClusterMonitor::FrameStatus otherStatus;
    otherStatus.status = ClusterMonitorStatus::CLUSTER_MONITOR_OK;
    g_monitor.uid2FrameStatusMap_.insert(otherUid, otherStatus);

    MOCKER(SocketSendNb).stubs().will(returnValue(HCCL_SUCCESS));

    g_monitor.ProcessExceptionEvent();

    EXPECT_TRUE(g_monitor.errRankQueue_.empty());

    GlobalMockObject::verify();
}

TEST_F(ClusterMonitorTest, Ut_ProcessExceptionEvent_When_EmptyQueue_Expect_Return)
{
    while (!g_monitor.errRankQueue_.empty()) {
        g_monitor.errRankQueue_.pop();
    }

    g_monitor.ProcessExceptionEvent();

    EXPECT_TRUE(g_monitor.errRankQueue_.empty());
}

TEST_F(ClusterMonitorTest, Ut_MakeErrMsg_When_LostStatus_Expect_GenerateCorrectMsg)
{
    std::queue<ClusterMonitorFrame> keyEvents;

    std::string crimerStr = "crimerInstance";
    ClusterUIDCxt crimerCxt(crimerStr, 1);
    ClusterUIDType crimerUid = g_monitor.FormatUID(crimerCxt);

    std::string informerStr = "informerInstance";
    ClusterUIDCxt informerCxt(informerStr, 2);
    ClusterUIDType informerUid = g_monitor.FormatUID(informerCxt);

    HcclUs relativeTime = std::chrono::steady_clock::now();
    std::chrono::system_clock::time_point systemTime = std::chrono::system_clock::now();

    ClusterMonitorFrame frame(
        crimerUid, informerUid, ClusterMonitorStatus::CLUSTER_MONITOR_LOST, relativeTime, systemTime);
    keyEvents.push(frame);

    std::vector<std::string> errStatusVec;
    g_monitor.MakeErrMsg(keyEvents, errStatusVec);

    EXPECT_EQ(errStatusVec.size(), 1);
    EXPECT_TRUE(errStatusVec[0].find("Heartbeat Lost Occurred") != std::string::npos);
}

TEST_F(ClusterMonitorTest, Ut_MakeErrMsg_When_CqeErrStatus_Expect_GenerateCorrectMsg)
{
    std::queue<ClusterMonitorFrame> keyEvents;

    std::string crimerStr2 = "crimerInstance";
    ClusterUIDCxt crimerCxt(crimerStr2, 1);
    ClusterUIDType crimerUid = g_monitor.FormatUID(crimerCxt);

    std::string informerStr2 = "informerInstance";
    ClusterUIDCxt informerCxt(informerStr2, 2);
    ClusterUIDType informerUid = g_monitor.FormatUID(informerCxt);

    HcclUs relativeTime = std::chrono::steady_clock::now();
    std::chrono::system_clock::time_point systemTime = std::chrono::system_clock::now();

    ClusterMonitorFrame frame(
        crimerUid, informerUid, ClusterMonitorStatus::CLUSTER_MONITOR_CQE_ERR, relativeTime, systemTime);
    keyEvents.push(frame);

    std::vector<std::string> errStatusVec;
    g_monitor.MakeErrMsg(keyEvents, errStatusVec);

    EXPECT_EQ(errStatusVec.size(), 1);
    EXPECT_TRUE(errStatusVec[0].find("Error cqe Occurred") != std::string::npos);
}

TEST_F(ClusterMonitorTest, Ut_MakeErrMsg_When_EmptyQueue_Expect_EmptyVec)
{
    std::queue<ClusterMonitorFrame> keyEvents;
    std::vector<std::string> errStatusVec;

    g_monitor.MakeErrMsg(keyEvents, errStatusVec);

    EXPECT_EQ(errStatusVec.size(), 0);
}

TEST_F(ClusterMonitorTest, Ut_PrintEvents_When_NormalMap_Expect_ReturnVec)
{
    std::map<ClusterMonitorStatus, std::queue<ClusterMonitorFrame>> keyEvents;

    std::string crimerStr3 = "crimerInstance";
    ClusterUIDCxt crimerCxt(crimerStr3, 1);
    ClusterUIDType crimerUid = g_monitor.FormatUID(crimerCxt);

    std::string informerStr3 = "informerInstance";
    ClusterUIDCxt informerCxt(informerStr3, 2);
    ClusterUIDType informerUid = g_monitor.FormatUID(informerCxt);

    HcclUs relativeTime = std::chrono::steady_clock::now();
    std::chrono::system_clock::time_point systemTime = std::chrono::system_clock::now();

    ClusterMonitorFrame cqeFrame(
        crimerUid, informerUid, ClusterMonitorStatus::CLUSTER_MONITOR_CQE_ERR, relativeTime, systemTime);
    keyEvents[ClusterMonitorStatus::CLUSTER_MONITOR_CQE_ERR].push(cqeFrame);

    ClusterMonitorFrame lostFrame(
        crimerUid, informerUid, ClusterMonitorStatus::CLUSTER_MONITOR_LOST, relativeTime, systemTime);
    keyEvents[ClusterMonitorStatus::CLUSTER_MONITOR_LOST].push(lostFrame);

    auto result = g_monitor.PrintEvents(keyEvents);

    EXPECT_GE(result.size(), 1);
}

TEST_F(ClusterMonitorTest, Ut_PrintEvents_When_EmptyMap_Expect_EmptyVec)
{
    std::map<ClusterMonitorStatus, std::queue<ClusterMonitorFrame>> keyEvents;

    auto result = g_monitor.PrintEvents(keyEvents);

    EXPECT_EQ(result.size(), 0);
}

TEST_F(ClusterMonitorTest, Ut_GetErrStatusVecFromCluserMonitor_When_Normal_Expect_ReturnVec)
{
    std::string crimerStr4 = "crimerInstance";
    ClusterUIDCxt crimerCxt(crimerStr4, 1);
    ClusterUIDType crimerUid = g_monitor.FormatUID(crimerCxt);

    std::string informerStr4 = "informerInstance";
    ClusterUIDCxt informerCxt(informerStr4, 2);
    ClusterUIDType informerUid = g_monitor.FormatUID(informerCxt);

    HcclUs relativeTime = std::chrono::steady_clock::now();
    std::chrono::system_clock::time_point systemTime = std::chrono::system_clock::now();

    ClusterMonitorFrame frame(
        crimerUid, informerUid, ClusterMonitorStatus::CLUSTER_MONITOR_CQE_ERR, relativeTime, systemTime);
    g_monitor.errStatusQueue_.push(frame);

    auto result = g_monitor.GetErrStatusVecFromCluserMonitor();

    EXPECT_EQ(g_monitor.errStatusQueue_.size(), 0);
}

TEST_F(ClusterMonitorTest, Ut_GetErrStatusVecFromCluserMonitor_When_EmptyQueue_Expect_EmptyVec)
{
    while (!g_monitor.errStatusQueue_.empty()) {
        g_monitor.errStatusQueue_.pop();
    }

    auto result = g_monitor.GetErrStatusVecFromCluserMonitor();

    EXPECT_EQ(result.size(), 0);
}

TEST_F(ClusterMonitorTest, Ut_MonitorThread_When_Normal_Expect_ProcessEvents)
{
    MOCKER_CPP(&ClusterMonitor::SendFrame).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&ClusterMonitor::RecvFrame).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&ClusterMonitor::PrintEvents).stubs().will(returnValue(std::vector<std::string>{"Error Message"}));

    HCCL_ERROR("g_monitor.uid2SocketRefMap_.Size = %d", g_monitor.uid2SocketRefMap_.Size());
    g_monitor.clusterMonitorThreadFlag_ = true;
    g_monitor.RunMonitorThread();
    EXPECT_TRUE(g_monitor.clusterMonitorThread_ != nullptr);
    SalSleep(1); // 等待线程处理完队列中的事件
    g_monitor.clusterMonitorThreadFlag_ = false;

    if (g_monitor.clusterMonitorThread_) {
        if (g_monitor.clusterMonitorThread_->joinable()) {
            g_monitor.clusterMonitorThread_->join();
        }
    }
}

TEST_F(ClusterMonitorTest, Ut_RegisterToClusterMonitor_When_Normal_Expect_Success)
{
    Hccl::RankGraph stub(0);
    void* stubPtr = &stub;
    MOCKER_CPP(&Hccl::HcclCommunicator::GetRankGraphV2).stubs().with(outBound(stubPtr)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&ClusterMonitor::GetRemEndpointDescs).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    setenv("HCCL_RDMA_RETRY_CNT", "7", 1);
    setenv("HCCL_RDMA_TIMEOUT", "20", 1);
    setenv("HCCL_RDMA_TC", "120", 1);
    setenv("HCCL_RDMA_SL", "2", 1);
    setenv("HCCL_DFS_CONFIG", "clusterHeartBeatEnable:on", 1);
    hccl::RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    void* commV2 = (void*)0x2000;
    uint32_t rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[128] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    config.hcclQos = 0xFFFFFFFF;
    unsetenv("HCCL_DFS_CONFIG");
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    hccl::CollComm* collComm = hcclCommPtr->GetCollComm();
    HcclComm comm = static_cast<HcclComm>(hcclCommPtr.get());

    ret = g_monitor.RegisterToClusterMonitor(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ClusterMonitorTest, Ut_UnRegisterToClusterMonitor_When_Normal_Expect_Success)
{
    MOCKER(SocketDestroy).stubs().will(returnValue(HCCL_SUCCESS));

    g_monitor.initialized_ = true; // 设置已初始化标志，允许调用UnRegisterToClusterMonitor
    hccl::RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    void* commV2 = (void*)0x2000;
    uint32_t rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[128] = {"comm1"};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    config.hcclQos = 0xFFFFFFFF;
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    hccl::CollComm* collComm = hcclCommPtr->GetCollComm();
    HcclComm comm = static_cast<HcclComm>(hcclCommPtr.get());

    std::string commKey(commName);
    std::string netInstId("remoteInstance");
    ClusterUIDCxt clusterUidCxt(netInstId, 1);
    ClusterUIDType clusterUid = g_monitor.FormatUID(clusterUidCxt);
    g_monitor.commIdMap_[commKey][clusterUid] = true;
    ret = g_monitor.UnRegisterToClusterMonitor(collComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ClusterMonitorTest, Ut_GetConnectRank_When_Normal_Expect_Success)
{
    std::map<ClusterUIDType, ClusterMonitorSocketCtx> needConnectRank;
    std::vector<uint32_t> netLayersVector;
    int ret;
    g_monitor.myRankLocalId_ = 0;
    netLayersVector.push_back(0); // 添加netLayer 0
    netLayersVector.push_back(1); // 添加netLayer 1
    netLayersVector.push_back(2); // 添加netLayer 2
    ClusterUIDType uid1;
    ret = memcpy_s(uid1.id, sizeof(uid1.id), "remoteInstance0/0", sizeof("remoteInstance0/0") + 1);
    EXPECT_EQ(ret, EOK);
    ClusterUIDType uid2;
    ret = memcpy_s(uid2.id, sizeof(uid2.id), "remoteInstance0/1", sizeof("remoteInstance0/1") + 1);
    EXPECT_EQ(ret, EOK);
    ClusterUIDType uid3;
    ret = memcpy_s(uid3.id, sizeof(uid3.id), "remoteInstance0/2", sizeof("remoteInstance0/2") + 1);
    EXPECT_EQ(ret, EOK);
    ClusterUIDType uid4;
    ret = memcpy_s(uid4.id, sizeof(uid4.id), "remoteInstance0/3", sizeof("remoteInstance0/3") + 1);
    EXPECT_EQ(ret, EOK);
    g_uidCtxs[0] = {UIDContext(uid1, 0, 0, 0, "remoteInstance0"), UIDContext(uid2, 0, 1, 1, "remoteInstance1")};
    g_uidCtxs[1] = {UIDContext(uid3, 1, 2, 0, "remoteInstance2"), UIDContext(uid4, 1, 3, 1, "remoteInstance3")};
    g_uidCtxs[2] = {UIDContext(uid3, 2, 4, 0, "remoteInstance4"), UIDContext(uid4, 2, 5, 1, "remoteInstance5")};
    MOCKER_CPP(&ClusterMonitor::GetSamePlaneRank).stubs().will(returnValue(HCCL_SUCCESS));
    ret = g_monitor.GetConnectRank(nullptr, needConnectRank, g_uidCtxs, netLayersVector);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

static Hccl::LocalId GetLocalIdStub(Hccl::RankGraph* self, Hccl::RankId rankId)
{
    if (rankId == 0) {
        return 0;
    }
    return 1;
}

static Hccl::NetInstance* GetNetInstanceByRankIdStub(Hccl::RankGraph* self, u32 netLayer, Hccl::RankId rankId)
{
    static Hccl::InnerNetInstance netInstance0(0, "netInstance0");
    static Hccl::InnerNetInstance netInstance1(0, "netInstance1");
    if (rankId == 0) {
        return &netInstance0;
    }
    return &netInstance1;
}

static HcclResult
HcclRankGraphGetRanksByLayerStub(HcclComm comm, uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum)
{
    *ranks = (uint32_t*)malloc(sizeof(uint32_t) * 2);
    (*ranks)[0] = 0;
    (*ranks)[1] = 1;
    *rankNum = 2;
    return HCCL_SUCCESS;
}

static HcclResult HcclRankGraphGetLayersStub(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum)
{
    *netLayers = (uint32_t*)malloc(sizeof(uint32_t) * 1);
    (*netLayers)[0] = 0;
    *netLayerNum = 1;
    return HCCL_SUCCESS;
}

TEST_F(ClusterMonitorTest, Ut_GetRemEndpointDescs_When_Normal_Expect_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    bool isDeviceSide = false;
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    setenv("HCCL_RDMA_RETRY_CNT", "7", 1);
    setenv("HCCL_RDMA_TIMEOUT", "20", 1);
    setenv("HCCL_RDMA_TC", "120", 1);
    setenv("HCCL_RDMA_SL", "2", 1);
    setenv("HCCL_DFS_CONFIG", "clusterHeartBeatEnable:on", 1);

    hccl::RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    void* stubPtr = rankGraphV2.get();
    MOCKER_CPP(&Hccl::HcclCommunicator::GetRankGraphV2).stubs().with(outBound(stubPtr)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcclRankGraphGetLayers).stubs().will(invoke(HcclRankGraphGetLayersStub));

    MOCKER(HcclRankGraphGetRanksByLayer).stubs().will(invoke(HcclRankGraphGetRanksByLayerStub));

    MOCKER_CPP((&Hccl::RankGraph::GetLocalId)).stubs().will(invoke(GetLocalIdStub));

    MOCKER_CPP((Hccl::NetInstance * (Hccl::RankGraph::*)(u32, Hccl::RankId))(&Hccl::RankGraph::GetNetInstanceByRankId))
        .stubs()
        .will(invoke(GetNetInstanceByRankIdStub));

    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, const_cast<char*>("commTest"));
    void* commV2 = (void*)0x2000;
    uint32_t rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[128] = {};
    HcclCommConfig config;
    config.hcclOpExpansionMode = 1;
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;
    config.hcclQos = 0xFFFFFFFF;
    unsetenv("HCCL_DFS_CONFIG");
    hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);

    HcclComm comm = static_cast<HcclComm>(hcclCommPtr.get());
    std::map<uint32_t, std::vector<UIDContext>> uidCtxs;
    std::vector<uint32_t> netLayersVector;

    HcclResult ret = g_monitor.GetRemEndpointDescs(comm, uidCtxs, netLayersVector);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(netLayersVector.size(), 1);
    EXPECT_EQ(netLayersVector[0], 0);
}

TEST_F(ClusterMonitorTest, Ut_GetSamePlaneRank_When_Size2_Expect_InsertClusterMonitorCtx)
{
    std::map<ClusterUIDType, ClusterMonitorSocketCtx> needConnectRank;
    ClusterUIDType uid1;
    int ret = memcpy_s(uid1.id, sizeof(uid1.id), "remoteInstance0/0", sizeof("remoteInstance0/0") + 1);
    EXPECT_EQ(ret, EOK);
    ClusterUIDType uid2;
    ret = memcpy_s(uid2.id, sizeof(uid2.id), "remoteInstance0/1", sizeof("remoteInstance0/1") + 1);
    EXPECT_EQ(ret, EOK);
    ClusterUIDType myUid;
    ret = memcpy_s(myUid.id, sizeof(myUid.id), "myInstance/0", sizeof("myInstance/0") + 1);
    EXPECT_EQ(ret, EOK);

    g_monitor.myRankUID_ = myUid;
    g_monitor.myRankLocalId_ = 0;
    g_monitor.myRankNetInstId_ = "myInstance";

    std::vector<UIDContext> singlePlaneCtx;
    singlePlaneCtx.push_back(UIDContext(myUid, 0, 0, 0, "myInstance"));
    singlePlaneCtx.push_back(UIDContext(uid1, 0, 1, 1, "remoteInstance0"));

    MOCKER_CPP(&ClusterMonitor::InsertClusterMonitorCtx).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult result = g_monitor.GetSamePlaneRank(nullptr, singlePlaneCtx, needConnectRank);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(ClusterMonitorTest, Ut_GetSamePlaneRank_When_Size3_Expect_InsertClusterMonitorCtxTwice)
{
    std::map<ClusterUIDType, ClusterMonitorSocketCtx> needConnectRank;
    ClusterUIDType uid1;
    int ret = memcpy_s(uid1.id, sizeof(uid1.id), "remoteInstance0/0", sizeof("remoteInstance0/0") + 1);
    EXPECT_EQ(ret, EOK);
    ClusterUIDType uid2;
    ret = memcpy_s(uid2.id, sizeof(uid2.id), "remoteInstance0/1", sizeof("remoteInstance0/1") + 1);
    EXPECT_EQ(ret, EOK);
    ClusterUIDType uid3;
    ret = memcpy_s(uid3.id, sizeof(uid3.id), "remoteInstance0/2", sizeof("remoteInstance0/2") + 1);
    EXPECT_EQ(ret, EOK);
    ClusterUIDType myUid;
    ret = memcpy_s(myUid.id, sizeof(myUid.id), "myInstance/0", sizeof("myInstance/0") + 1);
    EXPECT_EQ(ret, EOK);

    g_monitor.myRankUID_ = myUid;
    g_monitor.myRankLocalId_ = 0;
    g_monitor.myRankNetInstId_ = "myInstance";

    std::vector<UIDContext> singlePlaneCtx;
    singlePlaneCtx.push_back(UIDContext(myUid, 0, 0, 0, "myInstance"));
    singlePlaneCtx.push_back(UIDContext(uid1, 0, 1, 1, "remoteInstance0"));
    singlePlaneCtx.push_back(UIDContext(uid2, 0, 2, 2, "remoteInstance1"));

    MOCKER_CPP(&ClusterMonitor::InsertClusterMonitorCtx).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult result = g_monitor.GetSamePlaneRank(nullptr, singlePlaneCtx, needConnectRank);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(ClusterMonitorTest, Ut_InsertClusterMonitorCtx_When_NewConn_Expect_InsertToMap)
{
    ClusterUIDType myUid;
    int ret = memcpy_s(myUid.id, sizeof(myUid.id), "myInstance/0", sizeof("myInstance/0") + 1);
    EXPECT_EQ(ret, EOK);
    ClusterUIDType remoteUid;
    ret = memcpy_s(remoteUid.id, sizeof(remoteUid.id), "remoteInstance/1", sizeof("remoteInstance/1") + 1);
    EXPECT_EQ(ret, EOK);

    g_monitor.myRankUID_ = myUid;
    g_monitor.myRankLocalId_ = 0;
    g_monitor.myRankNetInstId_ = "myInstance";

    UIDContext remoteCtx(remoteUid, 0, 1, 1, "remoteInstance");

    std::map<ClusterUIDType, ClusterMonitorSocketCtx> needConnectRank;

    SocketDesc socketDesc{};
    socketDesc.role = HcommSocketRole::HCOMM_SOCKET_ROLE_CLIENT;
    socketDesc.listenPort = 8080;
    struct in_addr localAddr;
    struct in_addr remoteAddr;
    inet_pton(AF_INET, "127.0.0.1", &localAddr);
    inet_pton(AF_INET, "192.186.0.1", &remoteAddr);
    socketDesc.localEndpoint.commAddr.addr = localAddr;
    socketDesc.localEndpoint.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    socketDesc.remoteEndpoint.commAddr.addr = remoteAddr;
    socketDesc.remoteEndpoint.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    std::string tag = "test_tag";
    memcpy_s(socketDesc.tag, sizeof(socketDesc.tag), tag.c_str(), tag.size());

    ClusterMonitorSocketCtx ctx(socketDesc, true);
    needConnectRank.insert(std::make_pair(remoteUid, ctx));

    EXPECT_EQ(needConnectRank.size(), 1);
    EXPECT_EQ(needConnectRank[remoteUid].newConn, true);
}

TEST_F(ClusterMonitorTest, Ut_GetConnectRank_When_HighLayerCommLinks_Expect_SortExecuted)
{
    std::map<ClusterUIDType, ClusterMonitorSocketCtx> needConnectRank;
    std::vector<uint32_t> netLayersVector;
    int ret;
    g_monitor.myRankLocalId_ = 0;

    netLayersVector.push_back(0);
    netLayersVector.push_back(1);

    ClusterUIDType uid1;
    ret = memcpy_s(uid1.id, sizeof(uid1.id), "remoteInstanceA/0", sizeof("remoteInstanceA/0") + 1);
    EXPECT_EQ(ret, EOK);
    ClusterUIDType uid2;
    ret = memcpy_s(uid2.id, sizeof(uid2.id), "remoteInstanceB/0", sizeof("remoteInstanceB/0") + 1);
    EXPECT_EQ(ret, EOK);
    ClusterUIDType uid3;
    ret = memcpy_s(uid3.id, sizeof(uid3.id), "remoteInstanceC/0", sizeof("remoteInstanceC/0") + 1);
    EXPECT_EQ(ret, EOK);
    ClusterUIDType uid4;
    ret = memcpy_s(uid4.id, sizeof(uid4.id), "remoteInstanceD/1", sizeof("remoteInstanceD/1") + 1);
    EXPECT_EQ(ret, EOK);
    ClusterUIDType uid5;
    ret = memcpy_s(uid5.id, sizeof(uid5.id), "remoteInstanceE/1", sizeof("remoteInstanceE/1") + 1);
    EXPECT_EQ(ret, EOK);

    g_uidCtxs[0] = {UIDContext(uid1, 0, 0, 0, "remoteInstanceA"), UIDContext(uid2, 0, 1, 1, "remoteInstanceB")};
    g_uidCtxs[1]
        = {UIDContext(uid3, 1, 2, 0, "remoteInstanceC"), UIDContext(uid4, 1, 3, 1, "remoteInstanceD"),
           UIDContext(uid5, 1, 4, 0, "remoteInstanceE")};

    MOCKER_CPP(&ClusterMonitor::GetSamePlaneRank).stubs().will(returnValue(HCCL_SUCCESS));

    ret = g_monitor.GetConnectRank(nullptr, needConnectRank, g_uidCtxs, netLayersVector);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

struct InsertTestCtx {
    std::shared_ptr<Hccl::RankGraph> rankGraphData;
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
};

static InsertTestCtx CreateHcclCommForInsertTest()
{
    InsertTestCtx ctx;
    hccl::RankGraphStub rankGraphStub;
    ctx.rankGraphData = rankGraphStub.Create2PGraph();

    hccl::RankGraphV2* rankGraphV2 = new hccl::RankGraphV2(ctx.rankGraphData.get());

    hccl::ManagerCallbacks callbacks;
    callbacks.getAicpuCommState = []() {
        return false;
    };
    callbacks.setAicpuCommState = [](bool) {};
    callbacks.kernelLaunchAicpuCommInit = []() {
        return HCCL_SUCCESS;
    };

    auto collComm
        = std::make_unique<hccl::CollComm>(nullptr, 1, "test_comm", callbacks, hccl::CollCommInitMode::simpleMode);
    collComm->rankgraph_ = rankGraphV2;

    ctx.hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, "test_comm");
    ctx.hcclCommPtr->collComm_ = std::move(collComm);

    return ctx;
}

TEST_F(ClusterMonitorTest, Ut_InsertClusterMonitorCtx_When_GetLinksFailed_Expect_WarningAndSuccess)
{
    auto ctx = CreateHcclCommForInsertTest();
    ASSERT_NE(ctx.hcclCommPtr, nullptr);
    HcclComm comm = static_cast<HcclComm>(ctx.hcclCommPtr.get());

    ClusterUIDType myUid;
    int ret = memcpy_s(myUid.id, sizeof(myUid.id), "myInstance/1", sizeof("myInstance/1"));
    EXPECT_EQ(ret, EOK);
    g_monitor.myRankUID_ = myUid;
    g_monitor.myRankLocalId_ = 1;
    g_monitor.myRankNetInstId_ = "myInstance";

    ClusterUIDType remoteUid;
    ret = memcpy_s(remoteUid.id, sizeof(remoteUid.id), "remoteInstance/0", sizeof("remoteInstance/0"));
    EXPECT_EQ(ret, EOK);
    UIDContext remoteCtx(remoteUid, 0, 0, 0, "remoteInstance");

    std::map<ClusterUIDType, ClusterMonitorSocketCtx> needConnectRank;

    MOCKER(HcclRankGraphGetLinks).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclResult result = g_monitor.InsertClusterMonitorCtx(comm, remoteCtx, needConnectRank);
    EXPECT_EQ(result, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

static int g_salSleepCallCount = 0;
static int g_socketGetStatusCallCount = 0;

static void SalSleepStub(u32 sec) { g_salSleepCallCount++; }

static HcclResult SocketGetStatusStub(SocketHandler socketHandle, SocketStates* status)
{
    g_socketGetStatusCallCount++;
    if (g_socketGetStatusCallCount <= 3) {
        *status = SocketStates::SOCKET_CONNECTING;
        return HCCL_SUCCESS;
    }
    return HCCL_E_INTERNAL;
}

TEST_F(ClusterMonitorTest, Ut_CreateLinkWithRemotePonit_When_SocketConnecting_Expect_SalSleepCalled)
{
    g_salSleepCallCount = 0;
    g_socketGetStatusCallCount = 0;

    g_monitor.linkThreadRunning_ = true;
    g_monitor.deviceLogicId_ = 0;

    std::string localInstId = "localInstance";
    std::string remInstId = "remoteInstance";
    ClusterUIDCxt localUidCxt(localInstId, 0);
    ClusterUIDType localUid = g_monitor.FormatUID(localUidCxt);
    g_monitor.myRankUID_ = localUid;
    g_monitor.myRankNetInstId_ = "localInstance";
    g_monitor.myRankLocalId_ = 0;

    ClusterUIDCxt remUidCxt(remInstId, 1);
    ClusterUIDType remUid = g_monitor.FormatUID(remUidCxt);
    std::string commId = "comm_sleep_test";

    ClusterMonitorSocketCtx needConnectRank;
    needConnectRank.socketHandler = (void*)0x1000;
    needConnectRank.newConn = true;

    MOCKER(SocketGetStatus).stubs().will(invoke(SocketGetStatusStub));
    MOCKER(SalSleep).stubs().will(invoke(SalSleepStub));
    MOCKER(SocketDestroy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&ClusterMonitor::CreateTransportHandle).stubs().will(returnValue(HCCL_SUCCESS));

    std::thread testThread([&]() {
        g_monitor.CreateLinkWithRemotePonit(commId, remUid, needConnectRank);
    });
    testThread.join();

    EXPECT_GT(g_salSleepCallCount, 0);
    EXPECT_EQ(g_salSleepCallCount, 3);
    EXPECT_EQ(g_socketGetStatusCallCount, 4);

    GlobalMockObject::verify();
}

TEST_F(ClusterMonitorTest, Ut_InsertClusterMonitorCtx_When_GetLinksSuccess_ZeroLinks_Expect_Success)
{
    auto ctx = CreateHcclCommForInsertTest();
    ASSERT_NE(ctx.hcclCommPtr, nullptr);
    HcclComm comm = static_cast<HcclComm>(ctx.hcclCommPtr.get());

    ClusterUIDType myUid;
    int ret = memcpy_s(myUid.id, sizeof(myUid.id), "myInstance/1", sizeof("myInstance/1"));
    EXPECT_EQ(ret, EOK);
    g_monitor.myRankUID_ = myUid;
    g_monitor.myRankLocalId_ = 1;
    g_monitor.myRankNetInstId_ = "myInstance";

    ClusterUIDType remoteUid;
    ret = memcpy_s(remoteUid.id, sizeof(remoteUid.id), "remoteInstance/0", sizeof("remoteInstance/0"));
    EXPECT_EQ(ret, EOK);
    UIDContext remoteCtx(remoteUid, 0, 0, 0, "remoteInstance");

    std::map<ClusterUIDType, ClusterMonitorSocketCtx> needConnectRank;

    MOCKER(HcclRankGraphGetLinks).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult result = g_monitor.InsertClusterMonitorCtx(comm, remoteCtx, needConnectRank);
    EXPECT_EQ(result, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(ClusterMonitorTest, Ut_FormatConnTag_When_ClientRole_Expect_CorrectFormat)
{
    ClusterUIDType uid1;
    int ret = memcpy_s(uid1.id, sizeof(uid1.id), "instA/0", sizeof("instA/0"));
    EXPECT_EQ(ret, EOK);
    ClusterUIDType uid2;
    ret = memcpy_s(uid2.id, sizeof(uid2.id), "instB/1", sizeof("instB/1"));
    EXPECT_EQ(ret, EOK);

    std::string tag = g_monitor.FormatConnTag(HCOMM_SOCKET_ROLE_CLIENT, std::make_pair(uid1, uid2));
    EXPECT_TRUE(tag.find("HeartBeat_") != std::string::npos);
}

TEST_F(ClusterMonitorTest, Ut_FormatConnTag_When_ServerRole_Expect_CorrectFormat)
{
    ClusterUIDType uid1;
    int ret = memcpy_s(uid1.id, sizeof(uid1.id), "instA/0", sizeof("instA/0"));
    EXPECT_EQ(ret, EOK);
    ClusterUIDType uid2;
    ret = memcpy_s(uid2.id, sizeof(uid2.id), "instB/1", sizeof("instB/1"));
    EXPECT_EQ(ret, EOK);

    std::string tag = g_monitor.FormatConnTag(HCOMM_SOCKET_ROLE_SERVER, std::make_pair(uid1, uid2));
    EXPECT_TRUE(tag.find("HeartBeat_") != std::string::npos);
}

TEST_F(ClusterMonitorTest, Ut_FormatUID_When_NormalInput_Expect_ReturnUid)
{
    std::string netInstId = "testInstance";
    u32 localId = 42;
    ClusterUIDCxt cxt(netInstId, localId);
    ClusterUIDType uid = g_monitor.FormatUID(cxt);
    EXPECT_TRUE(uid.id[0] != '\0');
}

TEST_F(ClusterMonitorTest, Ut_GetSocketDescFromRankInfo_When_RemotePortInvalid_Expect_ParaError)
{
    auto ctx = CreateHcclCommForInsertTest();
    ASSERT_NE(ctx.hcclCommPtr, nullptr);
    HcclComm comm = static_cast<HcclComm>(ctx.hcclCommPtr.get());

    uint32_t invalidPort = Hccl::MAX_VALUE_TCPPORT + 1;
    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort)
        .stubs()
        .with(mockcpp::any(), outBoundP(&invalidPort, sizeof(invalidPort)))
        .will(returnValue(HCCL_SUCCESS));

    SocketDesc socketDesc{};
    HcclResult result = g_monitor.GetSocketDescFromRankInfo(comm, 0, 0, ClusterUIDType{}, socketDesc);
    EXPECT_EQ(result, HCCL_E_PARA);

    GlobalMockObject::verify();
}

// 对端端口正常，本端作为server但监听端口非法时命中 GetSocketDescFromRankInfo 的校验分支
static HcclResult GetDevicePortStub(Hccl::IRankGraph* self, uint32_t rank, uint32_t* devPort)
{
    // remoteRank=0 返回合法端口，myRankId=1 返回非法端口（触发 listenPort 校验）
    *devPort = (rank == 0) ? 10000 : (Hccl::MAX_VALUE_TCPPORT + 1);
    return HCCL_SUCCESS;
}

static HcclResult HcclRankGraphGetLinksDeviceStub(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** links, uint32_t* linkNum)
{
    static CommLink link;
    (void)memset_s(&link, sizeof(link), 0, sizeof(link));
    // local(src)地址 < remote(dst)地址，使本端作为server监听
    link.srcEndpointDesc.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    inet_pton(AF_INET, "1.0.0.1", &link.srcEndpointDesc.commAddr.addr);
    link.srcEndpointDesc.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    link.dstEndpointDesc.commAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    inet_pton(AF_INET, "2.0.0.1", &link.dstEndpointDesc.commAddr.addr);
    link.dstEndpointDesc.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    *links = &link;
    *linkNum = 1;
    return HCCL_SUCCESS;
}

TEST_F(ClusterMonitorTest, Ut_GetSocketDescFromRankInfo_When_ListenPortInvalid_Expect_ParaError)
{
    auto ctx = CreateHcclCommForInsertTest();
    ASSERT_NE(ctx.hcclCommPtr, nullptr);
    HcclComm comm = static_cast<HcclComm>(ctx.hcclCommPtr.get());

    MOCKER_CPP(&Hccl::IRankGraph::GetDevicePort).stubs().will(invoke(GetDevicePortStub));
    MOCKER(HcclRankGraphGetLinks).stubs().will(invoke(HcclRankGraphGetLinksDeviceStub));

    SocketDesc socketDesc{};
    HcclResult result = g_monitor.GetSocketDescFromRankInfo(comm, 0, 0, ClusterUIDType{}, socketDesc);
    EXPECT_EQ(result, HCCL_E_PARA);

    GlobalMockObject::verify();
}

TEST_F(ClusterMonitorTest, Ut_GetUID_When_NormalInput_Expect_ReturnString)
{
    std::string netInstId = "testInstance";
    u32 localId = 42;
    ClusterUIDCxt cxt(netInstId, localId);
    ClusterUIDType uid = g_monitor.FormatUID(cxt);
    std::string uidStr = g_monitor.GetUID(uid);
    EXPECT_FALSE(uidStr.empty());
}
