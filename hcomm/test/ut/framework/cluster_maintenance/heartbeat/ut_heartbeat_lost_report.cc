/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <cstring>
#include <thread>
#include <chrono>

#define private public
#define protected public
#include "heartbeat.h"
#include "env_config.h"
#include "sal_pub.h"
#include "adapter_rts_common.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

static bool g_recvCalled = false;

static InconsistentCheckMode StubGetExternalInconsistentCheckSwitch() { return InconsistentCheckMode::OFF; }

static HcclResult StubRecvFrame(Heartbeat*, UIDType&)
{
    g_recvCalled = true;
    return HCCL_SUCCESS;
}

static HcclResult StubSendFrame(Heartbeat*, UIDType&, UIDType&, UIDType&, HeartBeatStatus) { return HCCL_SUCCESS; }

class HeartbeatLostReportTest : public testing::Test {
protected:
    void SetUp() override
    {
        hb_ = &Heartbeat::GetInstance(0);
        uidSelf_ = {};
        memcpy(uidSelf_.id, "self", 5);
        hb_->uid_ = uidSelf_;

        uidRem_ = {};
        memcpy(uidRem_.id, "remote", 7);

        hb_->initialized_ = true;
        hb_->lostThreshold_ = HCCL_LOST_THRESHOLD;
        hb_->deviceLogicId_ = static_cast<u32>(-1);
        hb_->isPaused_ = false;
        g_recvCalled = false;
    }
    void TearDown() override
    {
        hb_->rankId2SocketMap_.erase(uidRem_);
        hb_->rankId2SocketMap_.erase(uidSelf_);
        hb_->rankId2StatusMap_.erase(uidRem_);
        hb_->rankId2StatusMap_.erase(uidSelf_);
        GlobalMockObject::verify();
    }

    void StartMonitorAndWaitRecv()
    {
        hb_->startSendRecvTask_ = true;
        std::thread monitorThread(&Heartbeat::HeartbeatStatusMonitor, hb_);

        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!g_recvCalled && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        hb_->startSendRecvTask_ = false;

        if (monitorThread.joinable()) {
            monitorThread.join();
        }
    }

    Heartbeat* hb_;
    UIDType uidSelf_;
    UIDType uidRem_;
};

TEST_F(HeartbeatLostReportTest, ParseFrame_OK_Resets_LostReportCnt)
{
    ConnInfo info;
    info.lostReportCnt = 3;
    hb_->rankId2SocketMap_.insert(uidRem_, info);

    HeartBeatFrame frame;
    frame.src = uidRem_;
    frame.dst = uidSelf_;
    frame.status = HeartBeatStatus::HEARTBEAT_OK;

    hb_->ParseFrame(frame, uidRem_);
    EXPECT_EQ(hb_->rankId2SocketMap_[uidRem_].lostReportCnt, 0u);
}

TEST_F(HeartbeatLostReportTest, ParseFrame_STUCK_Resets_LostReportCnt)
{
    ConnInfo info;
    info.lostReportCnt = 5;
    hb_->rankId2SocketMap_.insert(uidRem_, info);

    HeartBeatFrame frame;
    frame.src = uidRem_;
    frame.dst = uidSelf_;
    frame.status = HeartBeatStatus::HEARTBEAT_STUCK;

    hb_->ParseFrame(frame, uidRem_);
    EXPECT_EQ(hb_->rankId2SocketMap_[uidRem_].lostReportCnt, 0u);
}

TEST_F(HeartbeatLostReportTest, ParseFrameWithOpCheck_OK_Resets_LostReportCnt)
{
    ConnInfo info;
    info.lostReportCnt = 3;
    hb_->rankId2SocketMap_.insert(uidRem_, info);

    HeartBeatFrameWithOpCheck frame;
    frame.src = uidRem_;
    frame.dst = uidSelf_;
    frame.status = HeartBeatStatus::HEARTBEAT_OK;

    hb_->ParseFrameWithOpCheck(frame, uidRem_);
    EXPECT_EQ(hb_->rankId2SocketMap_[uidRem_].lostReportCnt, 0u);
}

TEST_F(HeartbeatLostReportTest, ParseFrameWithOpCheck_STUCK_Resets_LostReportCnt)
{
    ConnInfo info;
    info.lostReportCnt = 5;
    hb_->rankId2SocketMap_.insert(uidRem_, info);

    HeartBeatFrameWithOpCheck frame;
    frame.src = uidRem_;
    frame.dst = uidSelf_;
    frame.status = HeartBeatStatus::HEARTBEAT_STUCK;

    hb_->ParseFrameWithOpCheck(frame, uidRem_);
    EXPECT_EQ(hb_->rankId2SocketMap_[uidRem_].lostReportCnt, 0u);
}

TEST_F(HeartbeatLostReportTest, MonitorLoop_LostNumExceedsThreshold_IncrementsLostReportCnt)
{
    MOCKER(GetExternalInconsistentCheckSwitch).stubs().will(invoke(StubGetExternalInconsistentCheckSwitch));
    MOCKER(hrtSetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtResetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(SetThreadName).stubs();
    MOCKER_CPP(&Heartbeat::RecvFrame).stubs().will(invoke(StubRecvFrame));
    MOCKER_CPP(&Heartbeat::SendFrame).stubs().will(invoke(StubSendFrame));
    MOCKER_CPP(&Heartbeat::CheckSnapshotStatus).stubs();
    MOCKER_CPP(&Heartbeat::StuckDetection).stubs();
    MOCKER_CPP(&Heartbeat::ProcessExceptionEvent).stubs();
    MOCKER_CPP(&Heartbeat::DelErrorSocket).stubs();
    MOCKER_CPP(&Heartbeat::ProcessCqeErrInfo).stubs();
    MOCKER_CPP(&Heartbeat::CreateHBLinksAsync).stubs();
    MOCKER_CPP(&Heartbeat::CheckRecvOpInfoList).stubs();
    MOCKER_CPP(&Heartbeat::InitStuckDetection).stubs();
    MOCKER_CPP(&Heartbeat::GetSendOpInfoList).stubs();
    MOCKER_CPP(&Heartbeat::SetStatus).stubs();

    ConnInfo connInfo;
    connInfo.lostNum = hb_->lostThreshold_;
    connInfo.lostReportCnt = 0;
    hb_->rankId2SocketMap_.insert(uidRem_, connInfo);
    hb_->rankId2SocketMap_.insert(uidSelf_, ConnInfo());

    StartMonitorAndWaitRecv();

    EXPECT_TRUE(g_recvCalled);
    EXPECT_GE(hb_->rankId2SocketMap_[uidRem_].lostReportCnt, 1u);
}

TEST_F(HeartbeatLostReportTest, MonitorLoop_ExponentialBackoff_PreventsIncrement)
{
    MOCKER(GetExternalInconsistentCheckSwitch).stubs().will(invoke(StubGetExternalInconsistentCheckSwitch));
    MOCKER(hrtSetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtResetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(SetThreadName).stubs();
    MOCKER_CPP(&Heartbeat::RecvFrame).stubs().will(invoke(StubRecvFrame));
    MOCKER_CPP(&Heartbeat::SendFrame).stubs().will(invoke(StubSendFrame));
    MOCKER_CPP(&Heartbeat::CheckSnapshotStatus).stubs();
    MOCKER_CPP(&Heartbeat::StuckDetection).stubs();
    MOCKER_CPP(&Heartbeat::ProcessExceptionEvent).stubs();
    MOCKER_CPP(&Heartbeat::DelErrorSocket).stubs();
    MOCKER_CPP(&Heartbeat::ProcessCqeErrInfo).stubs();
    MOCKER_CPP(&Heartbeat::CreateHBLinksAsync).stubs();
    MOCKER_CPP(&Heartbeat::CheckRecvOpInfoList).stubs();
    MOCKER_CPP(&Heartbeat::InitStuckDetection).stubs();
    MOCKER_CPP(&Heartbeat::GetSendOpInfoList).stubs();
    MOCKER_CPP(&Heartbeat::SetStatus).stubs();

    ConnInfo connInfo;
    connInfo.lostNum = hb_->lostThreshold_;
    connInfo.lostReportCnt = 1;
    hb_->rankId2SocketMap_.insert(uidRem_, connInfo);
    hb_->rankId2SocketMap_.insert(uidSelf_, ConnInfo());

    StartMonitorAndWaitRecv();

    EXPECT_TRUE(g_recvCalled);
    EXPECT_EQ(hb_->rankId2SocketMap_[uidRem_].lostReportCnt, 1u);
}
