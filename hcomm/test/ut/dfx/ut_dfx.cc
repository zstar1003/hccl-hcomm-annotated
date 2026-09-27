/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <stdexcept>
#include <cstring>

#include "exception_callback_mgr.h"
#include "aicpu_daemon_service.h"
#include "daemon_func.h"
#include "exception_handle.h"
#include "hcomm/hcomm_exception.h"

using namespace hcomm;

/* ===================== halCqReportRecv stub for CQE tests ================== */
static drvError_t g_halCqReportRecvResult = DRV_ERROR_NONE;
static uint32_t g_reportCqeNum = 0;
static rtLogicCqReport_t g_cqeReport{};

static void ResetCqReportRecvStub()
{
    g_halCqReportRecvResult = DRV_ERROR_NONE;
    g_reportCqeNum = 0;
    g_cqeReport = {};
}

drvError_t halCqReportRecv(uint32_t devId, struct halReportRecvInfo* info)
{
    (void)devId;
    info->report_cqe_num = g_reportCqeNum;
    if (g_reportCqeNum > 0 && info->cqe_addr != nullptr) {
        auto* reports = reinterpret_cast<rtLogicCqReport_t*>(info->cqe_addr);
        reports[0] = g_cqeReport;
    }
    return g_halCqReportRecvResult;
}

/* ===================== FakeThread for CQE tests ============================ */
class FakeThread : public hccl::Thread {
public:
    void SetStreamLite(Hccl::StreamLite* sl) { streamLite_ = sl; }
    void* GetStreamLitePtr() const override { return streamLite_; }
    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult DeInit() override { return HCCL_SUCCESS; }
    std::string& GetUniqueId() override { return uniqueId_; }
    uint32_t GetNotifyNum() const override { return 0; }
    hccl::LocalNotify* GetNotify(uint32_t index) const override { return nullptr; }
    HcclResult SupplementNotify(uint32_t notifyNum) override { return HCCL_SUCCESS; }
    bool IsDeviceA5() const override { return false; }
    hccl::Stream* GetStream() const override { return nullptr; }
    void LaunchTask() const override {}
    void TryLaunchTask() const override {}
    HcclResult LocalNotifyRecord(uint32_t notifyId) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyWait(uint32_t notifyId) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyRecord(ThreadHandle dstThread, uint32_t dstNotifyIdx) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyWait(uint32_t notifyIdx, uint32_t timeOut) const override { return HCCL_SUCCESS; }
    HcclResult LocalCopy(void* dst, const void* src, uint64_t sizeByte) const override { return HCCL_SUCCESS; }
    HcclResult LocalReduce(
        void* dst, const void* src, uint64_t sizeByte, HcommDataType dataType, HcommReduceOp reduceOp) const override
    {
        return HCCL_SUCCESS;
    }
    bool GetMaster() const override { return false; }
    void SetIsMaster(bool isMaster) override { isMaster_ = isMaster; }

private:
    Hccl::StreamLite* streamLite_ = nullptr;
    std::string uniqueId_;
    bool isMaster_ = false;
};

/* ======================= ExceptionCallbackMgr UT ========================= */

class ExceptionCallbackMgrTest : public testing::Test {
protected:
    void TearDown() override
    {
        ExceptionCallbackMgr::GetInstance().Unregister(cb1_);
        ExceptionCallbackMgr::GetInstance().Unregister(cb2_);
        ExceptionCallbackMgr::GetInstance().Unregister(throwingCb_);
    }

    static void cb1(const HcommExceptionInfo* info, void* userData)
    {
        auto* called = static_cast<std::atomic<int>*>(userData);
        called->fetch_add(1, std::memory_order_relaxed);
        lastInfo = *info;
    }
    static void cb2(const HcommExceptionInfo* info, void* userData)
    {
        auto* called = static_cast<std::atomic<int>*>(userData);
        called->fetch_add(1, std::memory_order_relaxed);
    }
    static void throwingCb(const HcommExceptionInfo* info, void* userData)
    {
        (void)info;
        (void)userData;
        throw std::runtime_error("intentional test exception");
    }

    static inline HcommExceptionInfo lastInfo{};
    static inline HcommExceptionCallback cb1_ = cb1;
    static inline HcommExceptionCallback cb2_ = cb2;
    static inline HcommExceptionCallback throwingCb_ = throwingCb;
};

TEST_F(ExceptionCallbackMgrTest, RegisterNullptrReturnsError)
{
    EXPECT_NE(ExceptionCallbackMgr::GetInstance().Register(nullptr, nullptr), HCCL_SUCCESS);
}

TEST_F(ExceptionCallbackMgrTest, RegisterAndNotifyAll)
{
    std::atomic<int> called{0};
    EXPECT_EQ(ExceptionCallbackMgr::GetInstance().Register(cb1_, &called), HCCL_SUCCESS);

    HcommExceptionInfo info{};
    info.taskId = 42;
    info.channel = 0xDEAD;
    ExceptionCallbackMgr::GetInstance().NotifyAll(info);

    EXPECT_EQ(called.load(), 1);
    EXPECT_EQ(lastInfo.taskId, 42u);
    EXPECT_EQ(lastInfo.channel, 0xDEADu);
}

TEST_F(ExceptionCallbackMgrTest, RegisterMultipleCallbacksAllNotified)
{
    std::atomic<int> called1{0};
    std::atomic<int> called2{0};
    EXPECT_EQ(ExceptionCallbackMgr::GetInstance().Register(cb1_, &called1), HCCL_SUCCESS);
    EXPECT_EQ(ExceptionCallbackMgr::GetInstance().Register(cb2_, &called2), HCCL_SUCCESS);

    HcommExceptionInfo info{};
    ExceptionCallbackMgr::GetInstance().NotifyAll(info);

    EXPECT_EQ(called1.load(), 1);
    EXPECT_EQ(called2.load(), 1);
}

TEST_F(ExceptionCallbackMgrTest, UnregisterRemovesCallback)
{
    std::atomic<int> called{0};
    EXPECT_EQ(ExceptionCallbackMgr::GetInstance().Register(cb1_, &called), HCCL_SUCCESS);
    EXPECT_EQ(ExceptionCallbackMgr::GetInstance().Unregister(cb1_), HCCL_SUCCESS);

    HcommExceptionInfo info{};
    ExceptionCallbackMgr::GetInstance().NotifyAll(info);

    EXPECT_EQ(called.load(), 0);
}

TEST_F(ExceptionCallbackMgrTest, UnregisterNullptrReturnsError)
{
    EXPECT_NE(ExceptionCallbackMgr::GetInstance().Unregister(nullptr), HCCL_SUCCESS);
}

TEST_F(ExceptionCallbackMgrTest, RegisterSameCallbackTwiceUpdatesUserData)
{
    std::atomic<int> called1{0};
    std::atomic<int> called2{0};
    EXPECT_EQ(ExceptionCallbackMgr::GetInstance().Register(cb1_, &called1), HCCL_SUCCESS);
    EXPECT_EQ(ExceptionCallbackMgr::GetInstance().Register(cb1_, &called2), HCCL_SUCCESS);

    HcommExceptionInfo info{};
    ExceptionCallbackMgr::GetInstance().NotifyAll(info);

    EXPECT_EQ(called1.load(), 0);
    EXPECT_EQ(called2.load(), 1);
}

TEST_F(ExceptionCallbackMgrTest, ThrowingCallbackDoesNotBlockOthers)
{
    std::atomic<int> called1{0};
    std::atomic<int> called2{0};
    EXPECT_EQ(ExceptionCallbackMgr::GetInstance().Register(cb1_, &called1), HCCL_SUCCESS);
    EXPECT_EQ(ExceptionCallbackMgr::GetInstance().Register(throwingCb_, nullptr), HCCL_SUCCESS);
    EXPECT_EQ(ExceptionCallbackMgr::GetInstance().Register(cb2_, &called2), HCCL_SUCCESS);

    HcommExceptionInfo info{};
    EXPECT_NO_THROW(ExceptionCallbackMgr::GetInstance().NotifyAll(info));

    EXPECT_EQ(called1.load(), 1);
    EXPECT_EQ(called2.load(), 1);
}

/* ======================= AicpuDaemonService UT =========================== */

class FakeDaemonFunc : public Hccl::DaemonFunc {
public:
    void Call() override { called++; }
    int called = 0;
};

class AicpuDaemonServiceTest : public testing::Test {
protected:
    void TearDown() override
    {
        if (fakeFunc != nullptr) {
            Hccl::AicpuDaemonService::GetInstance().Unregister(fakeFunc);
        }
    }
    FakeDaemonFunc* fakeFunc = nullptr;
};

TEST_F(AicpuDaemonServiceTest, RegisterNullptrDoesNotCrash)
{
    EXPECT_NO_THROW(Hccl::AicpuDaemonService::GetInstance().Register(nullptr));
}

TEST_F(AicpuDaemonServiceTest, UnregisterNullptrDoesNotCrash)
{
    EXPECT_NO_THROW(Hccl::AicpuDaemonService::GetInstance().Unregister(nullptr));
}

TEST_F(AicpuDaemonServiceTest, RegisterAndUnregisterNormal)
{
    fakeFunc = new FakeDaemonFunc();
    EXPECT_NO_THROW(Hccl::AicpuDaemonService::GetInstance().Register(fakeFunc));
    EXPECT_NO_THROW(Hccl::AicpuDaemonService::GetInstance().Unregister(fakeFunc));
    delete fakeFunc;
    fakeFunc = nullptr;
}

TEST_F(AicpuDaemonServiceTest, RegisterSameFuncTwiceIsIdempotent)
{
    fakeFunc = new FakeDaemonFunc();
    Hccl::AicpuDaemonService::GetInstance().Register(fakeFunc);
    Hccl::AicpuDaemonService::GetInstance().Register(fakeFunc);
    Hccl::AicpuDaemonService::GetInstance().Unregister(fakeFunc);
    delete fakeFunc;
    fakeFunc = nullptr;
}

/* ========================= ExceptionHandle UT ============================ */

class ExceptionHandleTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& eh = ExceptionHandle::GetInstance();
        eh.threadsPrinted_.clear();
        eh.sqCqeErrorSet_.clear();
    }
    void TearDown() override
    {
        auto& eh = ExceptionHandle::GetInstance();
        eh.threadsPrinted_.clear();
        eh.sqCqeErrorSet_.clear();
    }
};

TEST_F(ExceptionHandleTest, GetInstanceReturnsSameInstance)
{
    auto& a = ExceptionHandle::GetInstance();
    auto& b = ExceptionHandle::GetInstance();
    EXPECT_EQ(&a, &b);
}

TEST_F(ExceptionHandleTest, GetSqeIdCombinesTaskIdAndStreamId)
{
    uint32_t sqeId = ExceptionHandle::GetSqeId(0x1234, 0x5678);
    EXPECT_EQ(sqeId, 0x12345678u);
}

TEST_F(ExceptionHandleTest, GetSqeIdWithZeroValues)
{
    uint32_t sqeId = ExceptionHandle::GetSqeId(0, 0);
    EXPECT_EQ(sqeId, 0u);
}

TEST_F(ExceptionHandleTest, GetSqeIdMaxValues)
{
    uint32_t sqeId = ExceptionHandle::GetSqeId(0xFFFF, 0xFFFF);
    EXPECT_EQ(sqeId, 0xFFFFFFFFu);
}

TEST_F(ExceptionHandleTest, GetSqeIdTaskIdInHighBits)
{
    uint32_t sqeId = ExceptionHandle::GetSqeId(1, 0);
    EXPECT_EQ(sqeId, 0x00010000u);
}

TEST_F(ExceptionHandleTest, GetSqeIdStreamIdInLowBits)
{
    uint32_t sqeId = ExceptionHandle::GetSqeId(0, 1);
    EXPECT_EQ(sqeId, 0x00000001u);
}

TEST_F(ExceptionHandleTest, SwitchCqeErrCodeToHcclErrCodeUrmaError)
{
    auto& eh = ExceptionHandle::GetInstance();
    EXPECT_EQ(eh.SwitchCqeErrCodeToHcclErrCode(0x5), static_cast<uint32_t>(HCCL_E_INTERNAL));
}

TEST_F(ExceptionHandleTest, SwitchCqeErrCodeToHcclErrCodeDefaultReturnsRoceTransfer)
{
    auto& eh = ExceptionHandle::GetInstance();
    EXPECT_EQ(eh.SwitchCqeErrCodeToHcclErrCode(0x0), static_cast<uint32_t>(HCCL_E_ROCE_TRANSFER));
    EXPECT_EQ(eh.SwitchCqeErrCodeToHcclErrCode(0x1), static_cast<uint32_t>(HCCL_E_ROCE_TRANSFER));
    EXPECT_EQ(eh.SwitchCqeErrCodeToHcclErrCode(0xFF), static_cast<uint32_t>(HCCL_E_ROCE_TRANSFER));
}

TEST_F(ExceptionHandleTest, ClearStreamStateRemovesAllEntries)
{
    auto& eh = ExceptionHandle::GetInstance();
    eh.threadsPrinted_[100] = 0x1234;
    eh.sqCqeErrorSet_.insert(100);

    eh.ClearStreamState(100);

    EXPECT_EQ(eh.threadsPrinted_.count(100), 0u);
    EXPECT_EQ(eh.sqCqeErrorSet_.count(100), 0u);
}

TEST_F(ExceptionHandleTest, ClearStreamStateDoesNotAffectOtherSqIds)
{
    auto& eh = ExceptionHandle::GetInstance();
    eh.threadsPrinted_[100] = 0x1111;
    eh.threadsPrinted_[200] = 0x2222;
    eh.sqCqeErrorSet_.insert(100);
    eh.sqCqeErrorSet_.insert(200);

    eh.ClearStreamState(100);

    EXPECT_EQ(eh.threadsPrinted_.count(100), 0u);
    EXPECT_EQ(eh.threadsPrinted_.count(200), 1u);
    EXPECT_EQ(eh.sqCqeErrorSet_.count(100), 0u);
    EXPECT_EQ(eh.sqCqeErrorSet_.count(200), 1u);
}

TEST_F(ExceptionHandleTest, ClearStreamStateForNonExistentSqIdIsNoop)
{
    auto& eh = ExceptionHandle::GetInstance();
    EXPECT_NO_THROW(eh.ClearStreamState(999));
}

TEST_F(ExceptionHandleTest, FillExceptionInfoWithNullThreadReturns)
{
    auto& eh = ExceptionHandle::GetInstance();
    HcommExceptionInfo info{};
    rtLogicCqReport_t cqe{};
    cqe.errorCode = 0x05;
    cqe.errorType = 0x3F;
    cqe.sqeType = 9;

    EXPECT_EQ(eh.FillExceptionInfo(info, nullptr, 42, cqe), HCCL_E_PTR);
    EXPECT_EQ(info.taskId, 0u);
}

TEST_F(ExceptionHandleTest, FillExceptionInfoPopulatesFields)
{
    auto& eh = ExceptionHandle::GetInstance();
    HcommExceptionInfo info{};
    rtLogicCqReport_t cqe{};
    cqe.errorCode = 0x012345;
    cqe.errorType = 0x3F;
    cqe.sqeType = 9;

    uint64_t fakeThreadAddr = 0xDEADBEEF;
    hccl::Thread* fakeThread = reinterpret_cast<hccl::Thread*>(fakeThreadAddr);

    EXPECT_EQ(eh.FillExceptionInfo(info, fakeThread, 42, cqe), HCCL_SUCCESS);

    EXPECT_EQ(info.thread, fakeThreadAddr);
    EXPECT_EQ(info.channel, 0u);
    EXPECT_EQ(info.taskId, 42u);
    EXPECT_EQ(info.retCode, static_cast<uint32_t>(HCCL_E_ROCE_TRANSFER));
    EXPECT_EQ(info.expandInfo.type, HCOMM_EXCEPTION_STARS);
    EXPECT_EQ(info.expandInfo.detail.starsInfo.starsErrcode, 0x3Fu);
    EXPECT_EQ(info.expandInfo.detail.starsInfo.sqeType, 9u);
    EXPECT_EQ(info.expandInfo.detail.starsInfo.statusMerged, 0x45u);
}

TEST_F(ExceptionHandleTest, FillExceptionInfoUrmaErrorCode)
{
    auto& eh = ExceptionHandle::GetInstance();
    HcommExceptionInfo info{};
    rtLogicCqReport_t cqe{};
    cqe.errorCode = 0x05;
    cqe.errorType = 0x3F;
    cqe.sqeType = 9;

    uint64_t fakeThreadAddr = 0x1000;
    hccl::Thread* fakeThread = reinterpret_cast<hccl::Thread*>(fakeThreadAddr);

    EXPECT_EQ(eh.FillExceptionInfo(info, fakeThread, 1, cqe), HCCL_SUCCESS);

    EXPECT_EQ(info.retCode, static_cast<uint32_t>(HCCL_E_INTERNAL));
}

TEST_F(ExceptionHandleTest, CallWithNoThreadsDoesNotCrash)
{
    auto& eh = ExceptionHandle::GetInstance();
    EXPECT_NO_THROW(eh.Call());
}

TEST_F(ExceptionHandleTest, CallWithEmptyCallbacksSkipsCqeQuery)
{
    auto& eh = ExceptionHandle::GetInstance();
    ExceptionCallbackMgr::GetInstance().Unregister(nullptr);
    EXPECT_NO_THROW(eh.Call());
}

TEST_F(ExceptionHandleTest, HandleExceptionCqeWithNoThreadsReturnsSuccess)
{
    auto& eh = ExceptionHandle::GetInstance();
    HcclResult ret = eh.HandleExceptionCqe();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ExceptionHandleTest, CheckRepeatBySqeIdWithNullStreamLiteReturnsError)
{
    auto& eh = ExceptionHandle::GetInstance();
    EXPECT_EQ(eh.CheckRepeatBySqeId(nullptr, 0, 1, 2), HCCL_E_PTR);
}

TEST_F(ExceptionHandleTest, CheckExceptionCqeWithNullThreadReturnsError)
{
    auto& eh = ExceptionHandle::GetInstance();
    EXPECT_EQ(eh.CheckExceptionCqe(nullptr, 0), HCCL_E_PTR);
}

TEST_F(ExceptionHandleTest, ClearStreamStateAfterSqeErrorAllowsRecheck)
{
    auto& eh = ExceptionHandle::GetInstance();
    eh.sqCqeErrorSet_.insert(100);
    EXPECT_EQ(eh.sqCqeErrorSet_.count(100), 1u);

    eh.ClearStreamState(100);
    EXPECT_EQ(eh.sqCqeErrorSet_.count(100), 0u);
}

TEST_F(ExceptionHandleTest, FillExceptionInfoWithZeroErrorCode)
{
    auto& eh = ExceptionHandle::GetInstance();
    HcommExceptionInfo info{};
    rtLogicCqReport_t cqe{};
    cqe.errorCode = 0x00;
    cqe.errorType = 0x00;
    cqe.sqeType = 0;

    uint64_t fakeThreadAddr = 0x2000;
    hccl::Thread* fakeThread = reinterpret_cast<hccl::Thread*>(fakeThreadAddr);

    EXPECT_EQ(eh.FillExceptionInfo(info, fakeThread, 10, cqe), HCCL_SUCCESS);

    EXPECT_EQ(info.thread, fakeThreadAddr);
    EXPECT_EQ(info.channel, 0u);
    EXPECT_EQ(info.taskId, 10u);
    EXPECT_EQ(info.retCode, static_cast<uint32_t>(HCCL_E_ROCE_TRANSFER));
    EXPECT_EQ(info.expandInfo.detail.starsInfo.starsErrcode, 0u);
    EXPECT_EQ(info.expandInfo.detail.starsInfo.sqeType, 0u);
    EXPECT_EQ(info.expandInfo.detail.starsInfo.statusMerged, 0u);
}

TEST_F(ExceptionHandleTest, FillExceptionInfoWithMaxErrorCode)
{
    auto& eh = ExceptionHandle::GetInstance();
    HcommExceptionInfo info{};
    rtLogicCqReport_t cqe{};
    cqe.errorCode = 0xFFFFFFFF;
    cqe.errorType = 0xFF;
    cqe.sqeType = 0xFF;

    uint64_t fakeThreadAddr = 0x3000;
    hccl::Thread* fakeThread = reinterpret_cast<hccl::Thread*>(fakeThreadAddr);

    EXPECT_EQ(eh.FillExceptionInfo(info, fakeThread, 0xFFFF, cqe), HCCL_SUCCESS);

    EXPECT_EQ(info.retCode, static_cast<uint32_t>(HCCL_E_ROCE_TRANSFER));
    EXPECT_EQ(info.expandInfo.detail.starsInfo.starsErrcode, 0xFFu);
    EXPECT_EQ(info.expandInfo.detail.starsInfo.statusMerged, 0xFFu);
}

TEST_F(ExceptionHandleTest, GetSqeIdReconstructsFullTaskId)
{
    uint32_t taskId = 0xABCD;
    uint32_t streamId = 0x1234;
    uint32_t sqeId = ExceptionHandle::GetSqeId(taskId, streamId);
    EXPECT_EQ(sqeId, 0xABCD1234u);
    EXPECT_EQ((sqeId >> 16) & 0xFFFF, taskId);
    EXPECT_EQ(sqeId & 0xFFFF, streamId);
}

/* =================== ExceptionHandle CQE coverage UT ====================== */

class ExceptionHandleCqeTest : public testing::Test {
protected:
    static constexpr uint32_t STREAM_ID = 10;
    static constexpr uint32_t SQ_ID = 20;
    static constexpr uint32_t PHY_ID = 0;
    static constexpr uint32_t CQ_ID = 30;

    void SetUp() override
    {
        auto& eh = ExceptionHandle::GetInstance();
        eh.threadsPrinted_.clear();
        eh.sqCqeErrorSet_.clear();
        ResetCqReportRecvStub();
        streamLite_ = std::make_unique<Hccl::StreamLite>(STREAM_ID, SQ_ID, PHY_ID, CQ_ID);
        fakeThread_.SetStreamLite(streamLite_.get());
    }
    void TearDown() override
    {
        auto& eh = ExceptionHandle::GetInstance();
        eh.threadsPrinted_.clear();
        eh.sqCqeErrorSet_.clear();
        ResetCqReportRecvStub();
        ExceptionCallbackMgr::GetInstance().Unregister(cqeCb_);
    }

    static void cqeCb(const HcommExceptionInfo* info, void* userData)
    {
        auto* called = static_cast<std::atomic<int>*>(userData);
        called->fetch_add(1, std::memory_order_relaxed);
        lastCqeInfo = *info;
    }

    static inline HcommExceptionInfo lastCqeInfo{};
    static inline HcommExceptionCallback cqeCb_ = cqeCb;
    std::unique_ptr<Hccl::StreamLite> streamLite_;
    FakeThread fakeThread_;
};

/* ---------- CheckRepeatBySqeId with non-null streamLite ---------- */

TEST_F(ExceptionHandleCqeTest, CheckRepeatBySqeIdFirstCallReturnsSuccess)
{
    auto& eh = ExceptionHandle::GetInstance();
    EXPECT_EQ(eh.CheckRepeatBySqeId(streamLite_.get(), 0, 1, 2), HCCL_SUCCESS);
}

TEST_F(ExceptionHandleCqeTest, CheckRepeatBySqeIdDuplicateReturnsError)
{
    auto& eh = ExceptionHandle::GetInstance();
    EXPECT_EQ(eh.CheckRepeatBySqeId(streamLite_.get(), 0, 1, 2), HCCL_SUCCESS);
    EXPECT_EQ(eh.CheckRepeatBySqeId(streamLite_.get(), 0, 1, 2), HCCL_E_AGAIN);
}

TEST_F(ExceptionHandleCqeTest, CheckRepeatBySqeIdDifferentSqeIdReturnsSuccess)
{
    auto& eh = ExceptionHandle::GetInstance();
    EXPECT_EQ(eh.CheckRepeatBySqeId(streamLite_.get(), 0, 1, 2), HCCL_SUCCESS);
    EXPECT_EQ(eh.CheckRepeatBySqeId(streamLite_.get(), 0, 3, 4), HCCL_SUCCESS);
}

/* ---------- ReceiveCqeReport ---------- */

TEST_F(ExceptionHandleCqeTest, ReceiveCqeReportReturnsDefaultWhenNoCqe)
{
    auto& eh = ExceptionHandle::GetInstance();
    g_halCqReportRecvResult = DRV_ERROR_NONE;
    g_reportCqeNum = 0;
    rtLogicCqReport_t cqe{};
    EXPECT_EQ(eh.ReceiveCqeReport(0, streamLite_.get(), cqe), dfx::CqeStatus::kDefault);
}

TEST_F(ExceptionHandleCqeTest, ReceiveCqeReportReturnsTimeOut)
{
    auto& eh = ExceptionHandle::GetInstance();
    g_halCqReportRecvResult = DRV_ERROR_WAIT_TIMEOUT;
    g_reportCqeNum = 0;
    rtLogicCqReport_t cqe{};
    EXPECT_EQ(eh.ReceiveCqeReport(0, streamLite_.get(), cqe), dfx::CqeStatus::kCqeTimeOut);
}

TEST_F(ExceptionHandleCqeTest, ReceiveCqeReportReturnsInnerError)
{
    auto& eh = ExceptionHandle::GetInstance();
    g_halCqReportRecvResult = DRV_ERROR_INNER_ERR;
    g_reportCqeNum = 0;
    rtLogicCqReport_t cqe{};
    EXPECT_EQ(eh.ReceiveCqeReport(0, streamLite_.get(), cqe), dfx::CqeStatus::kCqeInnerError);
}

TEST_F(ExceptionHandleCqeTest, ReceiveCqeReportReturnsException)
{
    auto& eh = ExceptionHandle::GetInstance();
    g_halCqReportRecvResult = DRV_ERROR_NONE;
    g_reportCqeNum = 1;
    g_cqeReport = {};
    g_cqeReport.errorType = RT_STARS_EXIST_ERROR;
    g_cqeReport.sqeType = DFX_SQE_TYPE_UDMA;
    g_cqeReport.taskId = 0x1234;
    g_cqeReport.streamId = 0x5678;
    rtLogicCqReport_t cqe{};
    EXPECT_EQ(eh.ReceiveCqeReport(0, streamLite_.get(), cqe), dfx::CqeStatus::kCqeException);
    EXPECT_EQ(cqe.taskId, 0x1234u);
    EXPECT_EQ(cqe.streamId, 0x5678u);
}

/* ---------- CheckExceptionCqe with non-null thread ---------- */

TEST_F(ExceptionHandleCqeTest, CheckExceptionCqeWithSqInErrorSetReturnsSuccess)
{
    auto& eh = ExceptionHandle::GetInstance();
    eh.sqCqeErrorSet_.insert(SQ_ID);
    EXPECT_EQ(eh.CheckExceptionCqe(&fakeThread_, 0), HCCL_SUCCESS);
}

TEST_F(ExceptionHandleCqeTest, CheckExceptionCqeWithNullStreamLiteReturnsError)
{
    auto& eh = ExceptionHandle::GetInstance();
    FakeThread emptyThread;
    emptyThread.SetStreamLite(nullptr);
    EXPECT_EQ(eh.CheckExceptionCqe(&emptyThread, 0), HCCL_E_PTR);
}

TEST_F(ExceptionHandleCqeTest, CheckExceptionCqeDefaultReturnsSuccess)
{
    auto& eh = ExceptionHandle::GetInstance();
    g_halCqReportRecvResult = DRV_ERROR_NONE;
    g_reportCqeNum = 0;
    EXPECT_EQ(eh.CheckExceptionCqe(&fakeThread_, 0), HCCL_SUCCESS);
}

TEST_F(ExceptionHandleCqeTest, CheckExceptionCqeTimeOutReturnsSuccessDueToSqeType)
{
    auto& eh = ExceptionHandle::GetInstance();
    g_halCqReportRecvResult = DRV_ERROR_WAIT_TIMEOUT;
    g_reportCqeNum = 0;
    EXPECT_EQ(eh.CheckExceptionCqe(&fakeThread_, 0), HCCL_SUCCESS);
}

TEST_F(ExceptionHandleCqeTest, CheckExceptionCqeInnerErrorAddsToErrorSet)
{
    auto& eh = ExceptionHandle::GetInstance();
    g_halCqReportRecvResult = DRV_ERROR_INNER_ERR;
    g_reportCqeNum = 0;
    EXPECT_EQ(eh.CheckExceptionCqe(&fakeThread_, 0), HCCL_E_INTERNAL);
    EXPECT_EQ(eh.sqCqeErrorSet_.count(SQ_ID), 1u);
}

TEST_F(ExceptionHandleCqeTest, CheckExceptionCqeExceptionWithNonUdmaSqeTypeReturnsSuccess)
{
    auto& eh = ExceptionHandle::GetInstance();
    g_halCqReportRecvResult = DRV_ERROR_NONE;
    g_reportCqeNum = 1;
    g_cqeReport = {};
    g_cqeReport.errorType = RT_STARS_EXIST_ERROR;
    g_cqeReport.sqeType = 0;
    EXPECT_EQ(eh.CheckExceptionCqe(&fakeThread_, 0), HCCL_SUCCESS);
}

TEST_F(ExceptionHandleCqeTest, CheckExceptionCqeExceptionWithRepeatReturnsSuccess)
{
    auto& eh = ExceptionHandle::GetInstance();
    g_halCqReportRecvResult = DRV_ERROR_NONE;
    g_reportCqeNum = 1;
    g_cqeReport = {};
    g_cqeReport.errorType = RT_STARS_EXIST_ERROR;
    g_cqeReport.sqeType = DFX_SQE_TYPE_UDMA;
    g_cqeReport.taskId = 1;
    g_cqeReport.streamId = 2;
    eh.threadsPrinted_[SQ_ID] = ExceptionHandle::GetSqeId(1, 2);
    EXPECT_EQ(eh.CheckExceptionCqe(&fakeThread_, 0), HCCL_SUCCESS);
}

TEST_F(ExceptionHandleCqeTest, CheckExceptionCqeExceptionNoErrorBitsReturnsSuccess)
{
    auto& eh = ExceptionHandle::GetInstance();
    g_halCqReportRecvResult = DRV_ERROR_NONE;
    g_reportCqeNum = 1;
    g_cqeReport = {};
    g_cqeReport.errorType = 0;
    g_cqeReport.sqeType = DFX_SQE_TYPE_UDMA;
    g_cqeReport.taskId = 1;
    g_cqeReport.streamId = 2;
    EXPECT_EQ(eh.CheckExceptionCqe(&fakeThread_, 0), HCCL_SUCCESS);
}

TEST_F(ExceptionHandleCqeTest, CheckExceptionCqeExceptionDetectedNotifiesCallback)
{
    auto& eh = ExceptionHandle::GetInstance();
    g_halCqReportRecvResult = DRV_ERROR_NONE;
    g_reportCqeNum = 1;
    g_cqeReport = {};
    g_cqeReport.errorType = RT_STARS_EXIST_ERROR;
    g_cqeReport.sqeType = DFX_SQE_TYPE_UDMA;
    g_cqeReport.taskId = 0x1234;
    g_cqeReport.streamId = 0x5678;
    g_cqeReport.errorCode = 0x05;

    std::atomic<int> called{0};
    lastCqeInfo = {};
    EXPECT_EQ(ExceptionCallbackMgr::GetInstance().Register(cqeCb_, &called), HCCL_SUCCESS);

    HcclResult ret = eh.CheckExceptionCqe(&fakeThread_, 0);
    EXPECT_EQ(ret, HCCL_E_ROCE_TRANSFER);
    EXPECT_EQ(called.load(), 1);
    EXPECT_EQ(lastCqeInfo.taskId, 0x1234u);
    EXPECT_EQ(lastCqeInfo.thread, reinterpret_cast<uint64_t>(&fakeThread_));
    EXPECT_EQ(lastCqeInfo.retCode, static_cast<uint32_t>(HCCL_E_INTERNAL));
    EXPECT_EQ(lastCqeInfo.expandInfo.type, HCOMM_EXCEPTION_STARS);
    EXPECT_EQ(lastCqeInfo.expandInfo.detail.starsInfo.starsErrcode, static_cast<uint32_t>(RT_STARS_EXIST_ERROR));
    EXPECT_EQ(lastCqeInfo.expandInfo.detail.starsInfo.sqeType, static_cast<uint8_t>(DFX_SQE_TYPE_UDMA));
}

/* ================ C API: HcommExceptionRegister/UnregisterCallback UT ===== */

class ExceptionMgrCAdptTest : public testing::Test {
protected:
    void TearDown() override { HcommExceptionUnregisterCallback(testCb); }

    static void testCb(const HcommExceptionInfo* info, void* userData)
    {
        auto* called = static_cast<std::atomic<int>*>(userData);
        if (called != nullptr) {
            called->fetch_add(1, std::memory_order_relaxed);
        }
        if (info != nullptr) {
            lastInfo = *info;
        }
    }

    static inline HcommExceptionInfo lastInfo{};
};

TEST_F(ExceptionMgrCAdptTest, RegisterNullptrCallbackReturnsError)
{
    EXPECT_NE(HcommExceptionRegisterCallback(nullptr, nullptr), 0);
}

TEST_F(ExceptionMgrCAdptTest, UnregisterNullptrCallbackReturnsError)
{
    EXPECT_NE(HcommExceptionUnregisterCallback(nullptr), 0);
}

TEST_F(ExceptionMgrCAdptTest, RegisterValidCallbackReturnsSuccess)
{
    std::atomic<int> called{0};
    EXPECT_EQ(HcommExceptionRegisterCallback(testCb, &called), 0);
}

TEST_F(ExceptionMgrCAdptTest, RegisterAndNotifyViaCallback)
{
    std::atomic<int> called{0};
    HcommExceptionRegisterCallback(testCb, &called);

    HcommExceptionInfo info{};
    info.taskId = 99;
    info.channel = 0xCAFE;
    ExceptionCallbackMgr::GetInstance().NotifyAll(info);

    EXPECT_EQ(called.load(), 1);
    EXPECT_EQ(lastInfo.taskId, 99u);
    EXPECT_EQ(lastInfo.channel, 0xCAFEu);
}

TEST_F(ExceptionMgrCAdptTest, UnregisterStopsCallback)
{
    std::atomic<int> called{0};
    HcommExceptionRegisterCallback(testCb, &called);
    HcommExceptionUnregisterCallback(testCb);

    HcommExceptionInfo info{};
    ExceptionCallbackMgr::GetInstance().NotifyAll(info);

    EXPECT_EQ(called.load(), 0);
}

TEST_F(ExceptionMgrCAdptTest, RegisterSameCallbackTwiceUpdatesUserData)
{
    std::atomic<int> called1{0};
    std::atomic<int> called2{0};
    HcommExceptionRegisterCallback(testCb, &called1);
    HcommExceptionRegisterCallback(testCb, &called2);

    HcommExceptionInfo info{};
    ExceptionCallbackMgr::GetInstance().NotifyAll(info);

    EXPECT_EQ(called1.load(), 0);
    EXPECT_EQ(called2.load(), 1);
}

TEST_F(ExceptionMgrCAdptTest, RegisterUnregisterReregisterWorks)
{
    std::atomic<int> called{0};
    HcommExceptionRegisterCallback(testCb, &called);
    HcommExceptionUnregisterCallback(testCb);

    HcommExceptionInfo info1{};
    ExceptionCallbackMgr::GetInstance().NotifyAll(info1);
    EXPECT_EQ(called.load(), 0);

    HcommExceptionRegisterCallback(testCb, &called);
    HcommExceptionInfo info2{};
    ExceptionCallbackMgr::GetInstance().NotifyAll(info2);
    EXPECT_EQ(called.load(), 1);
}
