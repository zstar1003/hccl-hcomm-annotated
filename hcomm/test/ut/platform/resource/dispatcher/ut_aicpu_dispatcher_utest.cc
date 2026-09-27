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

#ifndef private
#define private public
#define protected public
#include "dispatcher_aicpu.h"
#endif
#include "profiling_manager.h"
#include "dlprof_function.h"
#include "profiler_manager.h"
#include "externalinput.h"
#include "adapter_rts.h"
#include "rts_notify.h"
#include "queue_notify_manager.h"
#include "llt_hccl_stub_mc2.h"
#include "profiling_manager_device.h"

#undef private
#undef protected
#include "hccl_dispatcher_ctx.h"
#include "dispatcher_ctx.h"

using namespace hccl;

extern HcclResult CommTaskPrepare(char* key, uint32_t keyLen);
extern HcclResult CommTaskLaunch(ThreadHandle* threads, uint32_t threadNum);

class DispatcherAiCpu_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DispatcherAiCpu_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "DispatcherAiCpu_UT TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        dispatcherAiCpu->aicpuInfo_.devType = DevType::DEV_TYPE_910B;
        dispatcherAiCpu->aicpuInfo_.devId = 1;
        dispatcherAiCpu->aicpuInfo_.ssid = 1;
        dispatcherAiCpu->Init();

        streamInfo.actualStreamId = 1;
        streamInfo.sqId = 1;
        streamInfo.sqDepth = 100;
        streamInfo.sqBaseAddr = static_cast<void*>(sq_addr);
        streamInfo.logicCqId = 1;
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "DispatcherAiCpu_UT Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "DispatcherAiCpu_UT Test TearDown" << std::endl;
    }

    std::unique_ptr<DispatcherAiCpu> dispatcherAiCpu
        = std::unique_ptr<DispatcherAiCpu>(new (std::nothrow) DispatcherAiCpu(1));
    HcclComStreamInfo streamInfo;

    static u8 sq_addr[HCCL_SQE_SIZE * HCCL_SQE_MAX_CNT];
};

u8 DispatcherAiCpu_UT::sq_addr[HCCL_SQE_SIZE * HCCL_SQE_MAX_CNT] = {0};
// 补充覆盖率
#if 0
TEST_F(DispatcherAiCpu_UT, ut_DispatcherAiCpuLaunchTaskEx)
{
    u32 streamNum = 7;
    Stream stream(streamInfo, true);
    std::vector<Stream> subStreams;
    subStreams.resize(streamNum);

    for (u32 i = 0; i < subStreams.size(); i++) {
        streamInfo.actualStreamId = i;
        streamInfo.sqId = i;
        subStreams[i] = Stream(streamInfo, false);
    }

    MOCKER_CPP(&DispatcherAiCpu::LaunchTask, HcclResult(DispatcherAiCpu::*)(hccl::Stream &, bool))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL))
        .then(returnValue(HCCL_SUCCESS))
        .then(returnValue(HCCL_E_INTERNAL))
        .then(returnValue(HCCL_SUCCESS));
    auto ret1 = dispatcherAiCpu->LaunchTasksEx(stream, subStreams);
    EXPECT_EQ(ret1, HCCL_E_INTERNAL);

    auto ret2 = dispatcherAiCpu->LaunchTasksEx(stream, subStreams);
    EXPECT_EQ(ret2, HCCL_E_INTERNAL);

    auto ret = dispatcherAiCpu->LaunchTasksEx(stream, subStreams);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;

    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);
    // 测试初始化是否成功
    HcclSqeContext *sqeContext = stream.GetSqeContextPtr();
    auto &buff = sqeContext->buffer;
    HCCL_ERROR("buf sqHead[%u] sqTail[%u]", buff.sqHead, buff.sqTail);
    EXPECT_EQ(buff.sqHead, sqHead);
    EXPECT_EQ(buff.sqTail, sqTail);

    ret = dispatcherAiCpu->AddRetryPreamble(stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
#endif

TEST_F(DispatcherAiCpu_UT, ut_DispatcherAiCpuSignalRecord)
{
    HcclSignalInfo notifyInfo;
    std::unique_ptr<QueueNotifyManager> queueNotifyManager = nullptr;
    queueNotifyManager.reset(new (std::nothrow) QueueNotifyManager());
    EXPECT_NE(queueNotifyManager, nullptr);
    auto ret = queueNotifyManager->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<LocalNotify> signalAux = nullptr;
    std::shared_ptr<LocalNotify> signalMain = nullptr;
    std::string tag = "aicpu_signal_test";
    std::vector<std::shared_ptr<LocalNotify>> notifys(2, nullptr);
    ret = queueNotifyManager->Alloc(tag, 2, notifys, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    signalMain = notifys[0];
    signalAux = notifys[1];
    signalMain->GetNotifyData(notifyInfo);

    u32 notifyId = static_cast<u32>(notifyInfo.resId);

    HcclRtNotify signal;
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    u32 userRank = 1;
    u64 offset = 0;
    s32 stage = 0;
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    u64 signalAddr = INVALID_U64;
    ret = dispatcherAiCpu->SignalRecord(signal, stream, userRank, offset, stage, true, signalAddr, notifyId);
    EXPECT_EQ(HCCL_SUCCESS, ret);

    ret = dispatcherAiCpu->SignalRecord(signal, stream, userRank, offset, stage, false, signalAddr, notifyId);
    EXPECT_EQ(HCCL_SUCCESS, ret);

    GlobalMockObject::verify();
}

TEST_F(DispatcherAiCpu_UT, ut_DispatcherAiCpuSignalWait)
{
    HcclSignalInfo notifyInfo;
    std::unique_ptr<QueueNotifyManager> queueNotifyManager = nullptr;
    queueNotifyManager.reset(new (std::nothrow) QueueNotifyManager());
    EXPECT_NE(queueNotifyManager, nullptr);
    auto ret = queueNotifyManager->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<LocalNotify> signalAux = nullptr;
    std::shared_ptr<LocalNotify> signalMain = nullptr;
    std::string tag = "aicpu_signal_test";
    std::vector<std::shared_ptr<LocalNotify>> notifys(2, nullptr);
    ret = queueNotifyManager->Alloc(tag, 2, notifys, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    signalMain = notifys[0];
    signalAux = notifys[1];

    signalMain->GetNotifyData(notifyInfo);

    u32 notifyId = static_cast<u32>(notifyInfo.resId);
    HcclRtNotify signal;
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    u32 userRank = 0;
    u32 remoteRank = 1;
    u64 offset = 0;
    s32 stage = 0;
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    ret = dispatcherAiCpu->SignalWait(signal, stream, userRank, remoteRank, stage, true, notifyId);
    EXPECT_EQ(HCCL_SUCCESS, ret);

    ret = dispatcherAiCpu->SignalWait(signal, stream, userRank, remoteRank, stage, false, notifyId);
    EXPECT_EQ(HCCL_SUCCESS, ret);

    GlobalMockObject::verify();
}

TEST_F(DispatcherAiCpu_UT, ut_DispatcherAiCpuMemcpy)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    u64* ptr = new u64(1);
    DeviceMem dst(ptr, 0x80, false);
    DeviceMem src(ptr, 0x80, false);
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);
    auto ret = dispatcherAiCpu->MemcpyAsync(dst, src, stream);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    delete ptr;

    GlobalMockObject::verify();
}

TEST_F(DispatcherAiCpu_UT, ut_DispatcherAiCpuInlineReduceAsync)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    DeviceMem dst = DeviceMem::alloc(80);
    DeviceMem src = DeviceMem::alloc(80);
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    u64 dataCount = 20;
    auto ret = dispatcherAiCpu->InlineReduceAsync(
        src.ptr(), dataCount, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM, stream, dst.ptr());
    EXPECT_EQ(HCCL_SUCCESS, ret);
    GlobalMockObject::verify();
}

TEST_F(DispatcherAiCpu_UT, ut_DispatcherAiCpuReduceAsync)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    DeviceMem dst = DeviceMem::alloc(0x19000000);
    DeviceMem src = DeviceMem::alloc(0x19000000);

    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    u64 dataCount = 0x6400000;

    auto ret = dispatcherAiCpu->ReduceAsync(
        src.ptr(), dst.ptr(), dataCount, HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, stream,
        HcclReduceType::HCCL_INLINE_REDUCE);

    EXPECT_EQ(HCCL_SUCCESS, ret);
    GlobalMockObject::verify();
}

TEST_F(DispatcherAiCpu_UT, ut_DispatcherAiCpuTbeReduce_RdmaSend)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    u32 dbIndex = 0;
    u64 dbInfo = 0;
    u32 userRank;
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    RdmaTaskInfo taskInfo;
    taskInfo.remoteRank = userRank;

    auto ret = dispatcherAiCpu->RdmaSend(dbIndex, dbInfo, stream, taskInfo);
    EXPECT_EQ(HCCL_SUCCESS, ret);

    DeviceMem dst = DeviceMem::alloc(0x19000000);
    DeviceMem src1 = DeviceMem::alloc(0x19000000);
    DeviceMem src2 = DeviceMem::alloc(0x19100000);

    u64 dataCount = 0x6400000;

    auto ret2 = dispatcherAiCpu->TbeReduceAsync(
        src1.ptr(), src2.ptr(), dataCount, HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, stream,
        dst.ptr());
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret2);

    GlobalMockObject::verify();
}

TEST_F(DispatcherAiCpu_UT, ut_DispatcherProfilingRdmaSend)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 5;
    u32 dbIndex = 0;
    u64 dbInfo = 0;
    u32 userRank;

    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    RdmaTaskInfo taskInfo;
    taskInfo.remoteRank = userRank;

    auto ret = dispatcherAiCpu->RdmaSend(dbIndex, dbInfo, stream, taskInfo);
    EXPECT_EQ(HCCL_SUCCESS, ret);

    dispatcherAiCpu->aicpuInfo_.devType = DevType::DEV_TYPE_910_93;
    ret = dispatcherAiCpu->RdmaSend(dbIndex, dbInfo, stream, taskInfo);
    EXPECT_EQ(HCCL_SUCCESS, ret);

    ret = dfx::ProfilingManager::ReportTaskInfo(stream.id(), stream.GetSqeContextPtr());
    GlobalMockObject::verify();
}

int32_t AdprofReportBatchAdditionalInfo(uint32_t agingFlag, ConstVoidPtr data, uint32_t length) { return 0; }

int32_t MsprofReportBatchAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length) { return 0; }

TEST_F(DispatcherAiCpu_UT, ut_AdprofReportBatchAdditionalInfo)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 5;
    u32 dbIndex = 0;
    u64 dbInfo = 0;
    u32 userRank;

    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    RdmaTaskInfo taskInfo;
    taskInfo.remoteRank = userRank;

    auto ret = dispatcherAiCpu->RdmaSend(dbIndex, dbInfo, stream, taskInfo);
    EXPECT_EQ(HCCL_SUCCESS, ret);

    dispatcherAiCpu->aicpuInfo_.devType = DevType::DEV_TYPE_910_93;
    ret = dispatcherAiCpu->RdmaSend(dbIndex, dbInfo, stream, taskInfo);
    EXPECT_EQ(HCCL_SUCCESS, ret);

    ret = dfx::ProfilingManager::ReportTaskInfo(stream.id(), stream.GetSqeContextPtr());
    GlobalMockObject::verify();
}

// 覆盖 ReportTaskInfo 主循环并命中批量上报后的 memset 清理分支
TEST_F(DispatcherAiCpu_UT, ut_ReportTaskInfo_MemsetClearBranch)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 1;
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    // 打开L1 profiling开关, 否则 ReportTaskInfo 会在循环前直接返回
    dfx::ProfilingManager::SetProL1On(true);
    // 只有1个sqe, 使循环内唯一一次迭代满足 idx == (endIdx - 1), 进入批量上报后的清理分支
    stream.sqeContext_->buffer.sqeCnt = 1;
    stream.sqeContext_->buffer.tailSqeIdx = 1;

    auto ret = dfx::ProfilingManager::ReportTaskInfo(stream.id(), stream.GetSqeContextPtr());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(dfx::ProfilingManager::GetProfL1State(), true);

    // 恢复开关, 避免影响其他用例
    dfx::ProfilingManager::SetProL1On(false);
    GlobalMockObject::verify();
}

TEST_F(DispatcherAiCpu_UT, ut_DispatcherAiCpuGetPrivateMember)
{
    // TEST V2
    dispatcherAiCpu->aicpuInfo_.devType = DevType::DEV_TYPE_310P1;
    auto ret = dispatcherAiCpu->Init();

    EXPECT_EQ(HCCL_SUCCESS, ret);

    u32 notifyId = 32768;
    HcclRtNotify signal;
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    u32 userRank = 0;
    u32 remoteRank = 1;
    u64 offset = 0;
    s32 stage = 0;
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    ret = dispatcherAiCpu->SignalWait(signal, stream, userRank, remoteRank, stage, false, notifyId);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(DispatcherAiCpu_UT, ut_DispatcherAiCpuLaunchTask)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 50;
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    MOCKER(QuerySqStatusByType).stubs().will(returnValue(HCCL_SUCCESS));
    dispatcherAiCpu->dfxTimeOutConfig_.sqFullWaitTimeOut = 2;
    auto ret = dispatcherAiCpu->LaunchTask(stream, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DispatcherAiCpu_UT, ut_aicpu_prof_taskInfoReport)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    MOCKER(QuerySqStatusByType).stubs().will(returnValue(HCCL_SUCCESS));

    dispatcherAiCpu->dfxTimeOutConfig_.sqFullWaitTimeOut = 2;
    auto ret = dispatcherAiCpu->LaunchTask(stream, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    stream.sqeContext_->buffer.sqeCnt = 10;
    stream.sqeContext_->buffer.tailSqeIdx = 10;

    ret = dispatcherAiCpu->LaunchTask(stream, true);

    stream.sqeContext_->buffer.sqTail = 50;
    ret = dispatcherAiCpu->LaunchTask(stream, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DispatcherAiCpu_UT, ut_launchTask_rtsqfull)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    MOCKER(QuerySqStatusByType).stubs().will(returnValue(HCCL_SUCCESS));

    stream.sqeContext_->buffer.sqeCnt = 10;
    stream.sqeContext_->buffer.tailSqeIdx = 10;

    Stream stream1(streamInfo, true);
    SqCqeContext sqeCqeCtx1;
    sqeCqeCtx1.sqContext.inited = false;
    stream1.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx1);
    dispatcherAiCpu->SaveStreamInfo(stream1);

    dispatcherAiCpu->dfxTimeOutConfig_.sqFullWaitTimeOut = 2;
    auto ret = dispatcherAiCpu->LaunchTask(stream, true);
    EXPECT_EQ(ret, HCCL_E_AGAIN);
}

TEST_F(DispatcherAiCpu_UT, ut_DispatcherAiCpu_StreamSync)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    stream.sqeContext_->buffer.sqeCnt = 10;
    stream.sqeContext_->buffer.tailSqeIdx = 10;

    dispatcherAiCpu->dfxTimeOutConfig_.sqeTimeOutTimeOut = 0;
    auto ret = dispatcherAiCpu->StreamSync(stream);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);

    dispatcherAiCpu->dfxTimeOutConfig_.sqeTimeOutTimeOut = 1000000000U;
    Stream stream1(streamInfo, true);
    SqCqeContext sqeCqeCtx1;
    sqeCqeCtx1.sqContext.inited = false;
    stream1.InitSqAndCqeContext(0, 0, &sqeCqeCtx1);

    stream1.sqeContext_->buffer.sqeCnt = 0;
    stream1.sqeContext_->buffer.tailSqeIdx = 0;

    ret = dispatcherAiCpu->StreamSync(stream1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DispatcherAiCpu_UT, ut_GetStreamSqeBufferAddr_2048)
{
    // sqe cnt != 0 && tailSqeIdx == 2048
    uint32_t sqHead = 0;
    uint32_t sqTail = 3;
    u64* ptr = new u64(1);
    DeviceMem dst(ptr, 0x80, false);
    DeviceMem src(ptr, 0x80, false);
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    stream.sqeContext_->buffer.sqeCnt = 3;
    stream.sqeContext_->buffer.tailSqeIdx = 2048;

    auto ret = dispatcherAiCpu->MemcpyAsync(dst, src, stream);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    delete ptr;

    GlobalMockObject::verify();
}

TEST_F(DispatcherAiCpu_UT, ut_aicpu_fine_granularity)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 3;
    u64 addr = 0x12345678;
    u64 value = 1;
    u64 valueaddr = (u64)&value;
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);
    stream.sqeContext_->buffer.sqeCnt = 3;
    stream.sqeContext_->buffer.tailSqeIdx = 1024;

    auto ret = dispatcherAiCpu->WriteValue(stream, addr, valueaddr);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    bool reset = false;

    ret = dispatcherAiCpu->WaitValue(stream, addr, valueaddr, reset);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(DispatcherAiCpu_UT, ut_DispatcherAiCpu_SignalRecord)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    u32* dstPtr = new u32(0);
    u32* srcPtr = new u32(0);
    DeviceMem dst(dstPtr, 4, false);
    DeviceMem src(srcPtr, 4, false);
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);
    auto ret = dispatcherAiCpu->SignalRecord(dst, src, stream, 0, LinkType::LINK_HCCS_SW, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete dstPtr;
    delete srcPtr;

    GlobalMockObject::verify();
}

// 覆盖 SaveStreamInfo: 新流插入streamMap_
TEST_F(DispatcherAiCpu_UT, ut_SaveStreamInfo_NewStream)
{
    HcclComStreamInfo info = {};
    info.actualStreamId = 10;
    info.sqId = 10;
    info.sqDepth = 100;
    info.sqBaseAddr = static_cast<void*>(sq_addr);
    info.logicCqId = 10;
    Stream stream(info, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(0, 0, &sqeCqeCtx);

    EXPECT_EQ(dispatcherAiCpu->streamMap_.size(), 0u);
    dispatcherAiCpu->SaveStreamInfo(stream);
    EXPECT_EQ(dispatcherAiCpu->streamMap_.size(), 1u);
    EXPECT_NE(dispatcherAiCpu->streamMap_.find(10), dispatcherAiCpu->streamMap_.end());
}

// 覆盖 SaveStreamInfo: 重复流不重复插入
TEST_F(DispatcherAiCpu_UT, ut_SaveStreamInfo_DuplicateStream)
{
    HcclComStreamInfo info = {};
    info.actualStreamId = 10;
    info.sqId = 10;
    info.sqDepth = 100;
    info.sqBaseAddr = static_cast<void*>(sq_addr);
    info.logicCqId = 10;
    Stream stream(info, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(0, 0, &sqeCqeCtx);

    dispatcherAiCpu->SaveStreamInfo(stream);
    EXPECT_EQ(dispatcherAiCpu->streamMap_.size(), 1u);

    dispatcherAiCpu->SaveStreamInfo(stream);
    EXPECT_EQ(dispatcherAiCpu->streamMap_.size(), 1u);
}

// 覆盖 LaunchAllTasks: 空streamMap_直接返回成功
TEST_F(DispatcherAiCpu_UT, ut_LaunchAllTasks_EmptyMap)
{
    EXPECT_EQ(dispatcherAiCpu->streamMap_.size(), 0u);
    auto ret = dispatcherAiCpu->LaunchAllTasks();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 覆盖 LaunchAllTasks: 多流全部成功 (sqeCnt=0, LaunchTask直接返回)
TEST_F(DispatcherAiCpu_UT, ut_LaunchAllTasks_MultipleStreams)
{
    HcclComStreamInfo info1 = {};
    info1.actualStreamId = 1;
    info1.sqId = 1;
    info1.sqDepth = 100;
    info1.sqBaseAddr = static_cast<void*>(sq_addr);
    info1.logicCqId = 1;
    Stream stream1(info1, true);
    SqCqeContext sqeCqeCtx1;
    sqeCqeCtx1.sqContext.inited = false;
    stream1.InitSqAndCqeContext(0, 0, &sqeCqeCtx1);

    HcclComStreamInfo info2 = {};
    info2.actualStreamId = 2;
    info2.sqId = 2;
    info2.sqDepth = 100;
    info2.sqBaseAddr = static_cast<void*>(sq_addr);
    info2.logicCqId = 2;
    Stream stream2(info2, true);
    SqCqeContext sqeCqeCtx2;
    sqeCqeCtx2.sqContext.inited = false;
    stream2.InitSqAndCqeContext(0, 0, &sqeCqeCtx2);

    dispatcherAiCpu->SaveStreamInfo(stream1);
    dispatcherAiCpu->SaveStreamInfo(stream2);
    EXPECT_EQ(dispatcherAiCpu->streamMap_.size(), 2u);

    auto ret = dispatcherAiCpu->LaunchAllTasks();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 覆盖 LaunchAllTasks: 某流LaunchTask失败，错误码向上传播
TEST_F(DispatcherAiCpu_UT, ut_LaunchAllTasks_FailedLaunch)
{
    HcclComStreamInfo info1 = {};
    info1.actualStreamId = 1;
    info1.sqId = 1;
    info1.sqDepth = 100;
    info1.sqBaseAddr = static_cast<void*>(sq_addr);
    info1.logicCqId = 1;
    Stream stream1(info1, true);
    SqCqeContext sqeCqeCtx1;
    sqeCqeCtx1.sqContext.inited = false;
    stream1.InitSqAndCqeContext(0, 0, &sqeCqeCtx1);
    stream1.sqeContext_->buffer.sqeCnt = 200; // > sqDepth(100), 触发HCCL_E_PTR
    stream1.sqeContext_->buffer.tailSqeIdx = 200;

    dispatcherAiCpu->SaveStreamInfo(stream1);

    auto ret = dispatcherAiCpu->LaunchAllTasks();
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 覆盖 LaunchTask: RTSQ满时遍历streamMap_启动其他流 (成功路径)
TEST_F(DispatcherAiCpu_UT, ut_LaunchTask_IterateOtherStreams)
{
    // 主流: sqTail=100, sqDepth=100, RTSQ看起来已满
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    Stream stream(streamInfo, true); // actualStreamId=1
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);
    stream.sqeContext_->buffer.sqeCnt = 2;
    stream.sqeContext_->buffer.tailSqeIdx = 2;

    // 另一个流: actualStreamId=2, 与主流不同, 会被LaunchTask调用
    HcclComStreamInfo info2 = {};
    info2.actualStreamId = 2;
    info2.sqId = 2;
    info2.sqDepth = 100;
    info2.sqBaseAddr = static_cast<void*>(sq_addr);
    info2.logicCqId = 2;
    Stream stream2(info2, true);
    SqCqeContext sqeCqeCtx2;
    sqeCqeCtx2.sqContext.inited = false;
    stream2.InitSqAndCqeContext(0, 0, &sqeCqeCtx2);
    dispatcherAiCpu->SaveStreamInfo(stream2);

    // Query后head推进到98, while条件不满足, 退出循环
    uint32_t queryHead = 98;
    MOCKER(QuerySqStatusByType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(queryHead))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(ConfigSqStatusByType).stubs().will(returnValue(HCCL_SUCCESS));

    dispatcherAiCpu->dfxTimeOutConfig_.sqFullWaitTimeOut = 0;
    auto ret = dispatcherAiCpu->LaunchTask(stream, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

// 覆盖 WaitRtsq: RTSQ满时遍历streamMap_启动其他流 (成功路径)
TEST_F(DispatcherAiCpu_UT, ut_WaitRtsq_IterateOtherStreams)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    Stream stream(streamInfo, true); // actualStreamId=1
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    // 另一个流: actualStreamId=2, sqeCnt=0, LaunchTask直接返回成功
    HcclComStreamInfo info2 = {};
    info2.actualStreamId = 2;
    info2.sqId = 2;
    info2.sqDepth = 100;
    info2.sqBaseAddr = static_cast<void*>(sq_addr);
    info2.logicCqId = 2;
    Stream stream2(info2, true);
    SqCqeContext sqeCqeCtx2;
    sqeCqeCtx2.sqContext.inited = false;
    stream2.InitSqAndCqeContext(0, 0, &sqeCqeCtx2);
    dispatcherAiCpu->SaveStreamInfo(stream2);

    // Query后head推进到95, 为5个SQE腾出空间
    uint32_t queryHead = 95;
    MOCKER(QuerySqStatusByType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(queryHead))
        .will(returnValue(HCCL_SUCCESS));

    dispatcherAiCpu->dfxTimeOutConfig_.sqFullWaitTimeOut = 0;
    auto ret = dispatcherAiCpu->WaitRtsq(stream, 5, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

// 覆盖 WaitRtsq: 非阻塞模式下RTSQ满时直接返回
TEST_F(DispatcherAiCpu_UT, ut_WaitRtsq_NonBlockReturn)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;
    Stream stream(streamInfo, true);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    MOCKER(QuerySqStatusByType).stubs().will(returnValue(HCCL_SUCCESS));

    auto ret = dispatcherAiCpu->WaitRtsq(stream, 10, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(DispatcherAiCpu_UT, ut_SetSqeTimeOut_ExceedsMax_Expect_ClampedToMax)
{
    u64 largeTimeout = dispatcherAiCpu->notifyMaxWaitTime_ + 1;
    dispatcherAiCpu->SetSqeTimeOut(largeTimeout);
    EXPECT_EQ(dispatcherAiCpu->dfxTimeOutConfig_.sqeTimeOutTimeOut, dispatcherAiCpu->notifyMaxWaitTime_);
    GlobalMockObject::verify();
}

TEST_F(DispatcherAiCpu_UT, ut_SetSqeTimeOut_WithinMax_Expect_SetToInput)
{
    u64 normalTimeout = 100;
    if (normalTimeout > dispatcherAiCpu->notifyMaxWaitTime_) {
        normalTimeout = dispatcherAiCpu->notifyMaxWaitTime_ > 0 ? dispatcherAiCpu->notifyMaxWaitTime_ - 1 : 0;
    }
    dispatcherAiCpu->SetSqeTimeOut(normalTimeout);
    EXPECT_EQ(dispatcherAiCpu->dfxTimeOutConfig_.sqeTimeOutTimeOut, normalTimeout);
    GlobalMockObject::verify();
}
