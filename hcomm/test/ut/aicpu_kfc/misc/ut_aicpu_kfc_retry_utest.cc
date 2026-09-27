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
#endif
#include "common/aicpu_kfc_def.h"
#include "framework/aicpu_communicator.h"
#include "framework/aicpu_kfc_rpc_server.h"
#include "framework/aicpu_kfc_deprecated_process.h"
#include "framework/aicpu_kfc_process.h"
#include "framework/aicpu_hdc.h"
#include "aicpu_kfc/aicpu_kfc_interface.h"
#include "algorithm/aicpu_allreduce.h"
#include "algorithm/task_orchestrator.h"
#include "algorithm/aicpu_dmy_cal_allreduce.h"
#include "dlhal_function.h"
#include "utils/aicpu_hdc_utils.h"
#include "dfx/aicpu_executor_tracer.h"
#include "llt_aicpu_kfc_stub_mc2.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

constexpr u32 h2dBufferSize = sizeof(KfcExecControl);
constexpr u32 d2hBufferSize = sizeof(KfcExecStatus);

class MC2AicpuRetry_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MC2AicpuRetry_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "MC2AicpuRetry_UT TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        g_stubDevType = DevType::DEV_TYPE_910B;
        MOCKER(halGetDeviceInfo).stubs().with(mockcpp::any()).will(invoke(StubhalGetDeviceInfo));

        DlHalFunction::GetInstance().DlHalFunctionInit();
        hrtSetDevice(0);
        set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_910B));
        std::cout << "MC2AicpuRetry_UT Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        ResetMC2Context();
        GlobalMockObject::verify();
        std::cout << "MC2AicpuRetry_UT Test TearDown" << std::endl;
    }
};

#define init_kfc_args(initTask)                                                                     \
    StubHccCommRes commRes;                                                                         \
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();                              \
    AicpuKfcRpcServer::RpcMsgBody msgBody;                                                          \
    (void)memset_s(&msgBody, sizeof(msgBody), 0, sizeof(msgBody));                                  \
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);                                          \
                                                                                                    \
    std::shared_ptr<hccl::HDCommunicate> h2dTransfer;                                               \
    std::shared_ptr<hccl::HDCommunicate> d2hTransfer;                                               \
    h2dTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, h2dBufferSize)); \
    d2hTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, d2hBufferSize)); \
    h2dTransfer->InitHost();                                                                        \
    d2hTransfer->InitHost();                                                                        \
    paramTask.kfcControlTransferH2DParams = h2dTransfer->GetCommunicateParams();                    \
    paramTask.kfcStatusTransferD2HParams = d2hTransfer->GetCommunicateParams();                     \
    paramTask.config.retryEnable = 1;                                                               \
    initTask.context = uint64_t(&paramTask);

#define init_tiling_data(tilingData)                   \
    tilingData.commOrder = 0;                          \
    tilingData.commType = HCCL_CMD_ALLREDUCE;          \
    tilingData.dataType = HCCL_DATA_TYPE_FP16;         \
    tilingData.turnNum = 1;                            \
    tilingData.sendCnt = 256;                          \
    tilingData.totalCnt = 1;                           \
    tilingData.rspPolicy = 0;                          \
    tilingData.waitPolicy = 0;                         \
    tilingData.taskType = HCCL_KFC_TASK_HCCL_ONLY_EXE; \
    tilingData.useBufferType = 0;                      \
    tilingData.debugMode = MC2_DEBUG_WAIT_COMM;        \
    tilingData.preparePosition = TASK_PREPARE_HOST;    \
    tilingData.notifyOff = 0;

#define init_kfc_task(kfcTask)              \
    u64* a = (u64*)malloc(1024 * 1024);     \
    u64* b = (u64*)malloc(1024 * 1024);     \
    kfcTask.inputA = uint64_t(a);           \
    kfcTask.outputC = 0;                    \
    kfcTask.commOut = uint64_t(b);          \
    kfcTask.context = uint64_t(&paramTask); \
    kfcTask.tilingData = uint64_t(&tilingData);

#define deinit() \
    free(a);     \
    free(b);

uint8_t sqBuffer[64 * 32 * 64];
drvError_t halSqCpQueryStub_1(uint32_t devId, struct halSqCqQueryInfo* info)
{
    if (info == nullptr) {
        return DRV_ERROR_INNER_ERR;
    }
    static u32 counter = 1;
    auto queryinfo = *info;
    switch (queryinfo.prop) {
        case DRV_SQCQ_PROP_SQ_HEAD: {
            info->value[0] = 1;
            return DRV_ERROR_NONE;
        }
        case DRV_SQCQ_PROP_SQ_DEPTH: {
            info->value[0] = 4096;
            return DRV_ERROR_NONE;
        }
        case DRV_SQCQ_PROP_SQ_TAIL: {
            info->value[0] = counter;
            counter++;
            return DRV_ERROR_NONE;
        };
        case DRV_SQCQ_PROP_SQ_BASE: {
            uint8_t* buffer = sqBuffer;
            info->value[0] = reinterpret_cast<uintptr_t>(buffer) & 0xFFFFFFFF;
            info->value[1] = reinterpret_cast<uintptr_t>(buffer) >> 32;
        }
        default:
            return DRV_ERROR_NONE;
    }
}

drvError_t halSqCpQueryStub_2(uint32_t devId, struct halSqCqQueryInfo* info)
{
    if (info == nullptr) {
        return DRV_ERROR_INNER_ERR;
    }
    static u32 counter = 1;
    auto queryinfo = *info;
    switch (queryinfo.prop) {
        case DRV_SQCQ_PROP_SQ_HEAD: {
            info->value[0] = 2;
            return DRV_ERROR_NONE;
        }
        case DRV_SQCQ_PROP_SQ_DEPTH: {
            info->value[0] = 4096;
            return DRV_ERROR_NONE;
        }
        case DRV_SQCQ_PROP_SQ_TAIL: {
            info->value[0] = counter;
            counter++;
            counter++;
            return DRV_ERROR_NONE;
        };
        case DRV_SQCQ_PROP_SQ_BASE: {
            uint8_t* buffer = sqBuffer;
            info->value[0] = reinterpret_cast<uintptr_t>(buffer) & 0xFFFFFFFF;
            info->value[1] = reinterpret_cast<uintptr_t>(buffer) >> 32;
        }
        default:
            return DRV_ERROR_NONE;
    }
}

TEST_F(MC2AicpuRetry_UT, allreduce_fp16_normal)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);

    KFCTask kfcTask;
    init_kfc_task(kfcTask);

    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));

    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));

    d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
    EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kEnd);
    EXPECT_EQ(response.execStatus.kfcError, KfcError::kNone);

    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_fp16_retry_success_when_op_runing)
{
    MOCKER_CPP(&HcclCommAicpu::GenTaskExceptionInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    thread threadHandle([&] {
        request.kfcCmd = KfcCommand::kStopLaunch;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kRuning)
                && (response.execStatus.kfcStatus != KfcStatus::kNull)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStoplaunch);

        request.kfcCmd = KfcCommand::kStopExec;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if (response.execStatus.kfcStatus != KfcStatus::kStoplaunch) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStopExec);

        request.kfcCmd = KfcCommand::kRetry;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kStopExec)
                && (response.execStatus.kfcStatus != KfcStatus::kRuning)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kEnd);
    });

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);

    KFCTask kfcTask;
    init_kfc_task(kfcTask);

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any()).will(invoke(halSqCpQueryStub_2));

    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(&TaskOrchestrator::WaitFinishWhileLoop)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_SUSPENDING))
        .then(returnValue(HCCL_SUCCESS));

    usleep(10000);

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));

    threadHandle.join();
    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_fp16_retry_when_op_end)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    thread threadHandle([&] {
        request.kfcCmd = KfcCommand::kStopLaunch;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kRuning)
                && (response.execStatus.kfcStatus != KfcStatus::kNull)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStoplaunch);

        request.kfcCmd = KfcCommand::kStopExec;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if (response.execStatus.kfcStatus != KfcStatus::kStoplaunch) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kEnd);
    });

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);

    KFCTask kfcTask;
    init_kfc_task(kfcTask);

    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    usleep(10000);

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));

    threadHandle.join();
    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_fp16_not_support_retry_for_inplace)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    thread threadHandle([&] {
        request.kfcCmd = KfcCommand::kStopLaunch;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kRuning)
                && (response.execStatus.kfcStatus != KfcStatus::kNull)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStoplaunch);

        request.kfcCmd = KfcCommand::kStopExec;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if (response.execStatus.kfcStatus != KfcStatus::kStoplaunch) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kError);
    });

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);

    KFCTask kfcTask;
    init_kfc_task(kfcTask);
    kfcTask.inputA = uint64_t(a);
    kfcTask.commOut = uint64_t(a); // 输入内存和输出内存一致，不能进行retry

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any()).will(invoke(halSqCpQueryStub_2));

    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(&TaskOrchestrator::WaitFinishWhileLoop)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_SUSPENDING))
        .then(returnValue(HCCL_SUCCESS));

    usleep(10000);

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(HCCL_E_INTERNAL, RunAicpuRpcSrvLaunch(&kfcTask));

    threadHandle.join();
    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_fp16_not_support_retry_for_disable_retry)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    paramTask.config.retryEnable = 0; // 关闭op retry
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    thread threadHandle([&] {
        request.kfcCmd = KfcCommand::kStopLaunch;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kRuning)
                && (response.execStatus.kfcStatus != KfcStatus::kNull)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStoplaunch);

        request.kfcCmd = KfcCommand::kStopExec;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if (response.execStatus.kfcStatus != KfcStatus::kStoplaunch) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kError);
    });

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);

    KFCTask kfcTask;
    init_kfc_task(kfcTask);

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any()).will(invoke(halSqCpQueryStub_2));

    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(&TaskOrchestrator::WaitFinishWhileLoop)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_SUSPENDING))
        .then(returnValue(HCCL_SUCCESS));

    usleep(10000);

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(HCCL_E_INTERNAL, RunAicpuRpcSrvLaunch(&kfcTask));

    threadHandle.join();
    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_fp16_stop)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    thread threadHandle([&] {
        request.kfcCmd = KfcCommand::kExit;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kRuning)
                && (response.execStatus.kfcStatus != KfcStatus::kNull)) {
                break;
            }
        }
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if (response.execStatus.kfcStatus != KfcStatus::kStoplaunch) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kError);
    });

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);

    KFCTask kfcTask;
    init_kfc_task(kfcTask);

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any()).will(invoke(halSqCpQueryStub_2));

    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(&TaskOrchestrator::WaitFinishWhileLoop)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_SUSPENDING))
        .then(returnValue(HCCL_SUCCESS));

    usleep(10000);

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(HCCL_E_INTERNAL, RunAicpuRpcSrvLaunch(&kfcTask));

    threadHandle.join();
    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_fp16_stop_when_wait_stop_exec)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    thread threadHandle([&] {
        request.kfcCmd = KfcCommand::kStopLaunch;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kRuning)
                && (response.execStatus.kfcStatus != KfcStatus::kNull)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStoplaunch);

        request.kfcCmd = KfcCommand::kExit;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if (response.execStatus.kfcStatus != KfcStatus::kStoplaunch) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kError);
    });

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);

    KFCTask kfcTask;
    init_kfc_task(kfcTask);

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any()).will(invoke(halSqCpQueryStub_2));

    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(&TaskOrchestrator::WaitFinishWhileLoop)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_SUSPENDING))
        .then(returnValue(HCCL_SUCCESS));

    usleep(10000);

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(HCCL_E_INTERNAL, RunAicpuRpcSrvLaunch(&kfcTask));

    threadHandle.join();
    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_fp16_stop_when_wait_retry)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    thread threadHandle([&] {
        request.kfcCmd = KfcCommand::kStopLaunch;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kRuning)
                && (response.execStatus.kfcStatus != KfcStatus::kNull)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStoplaunch);

        request.kfcCmd = KfcCommand::kStopExec;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if (response.execStatus.kfcStatus != KfcStatus::kStoplaunch) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStopExec);

        request.kfcCmd = KfcCommand::kExit;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kStopExec)
                && (response.execStatus.kfcStatus != KfcStatus::kRuning)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kError);
    });

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);

    KFCTask kfcTask;
    init_kfc_task(kfcTask);

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any()).will(invoke(halSqCpQueryStub_2));

    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(&TaskOrchestrator::WaitFinishWhileLoop)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_SUSPENDING))
        .then(returnValue(HCCL_SUCCESS));

    usleep(10000);

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(HCCL_E_INTERNAL, RunAicpuRpcSrvLaunch(&kfcTask));

    threadHandle.join();
    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_fp16_retry_launch_fail)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    thread threadHandle([&] {
        request.kfcCmd = KfcCommand::kStopLaunch;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kRuning)
                && (response.execStatus.kfcStatus != KfcStatus::kNull)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStoplaunch);

        request.kfcCmd = KfcCommand::kStopExec;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if (response.execStatus.kfcStatus != KfcStatus::kStoplaunch) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStopExec);

        request.kfcCmd = KfcCommand::kRetry;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kStopExec)
                && (response.execStatus.kfcStatus != KfcStatus::kRuning)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kError);
    });

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);

    KFCTask kfcTask;
    init_kfc_task(kfcTask);

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any()).will(invoke(halSqCpQueryStub_2));

    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(&TaskOrchestrator::WaitFinishWhileLoop)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_SUSPENDING))
        .then(returnValue(HCCL_SUCCESS));

    MOCKER(&AicpuKfcDeprecatedProcess::RetryLaunchHcclOp)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    usleep(10000);

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(HCCL_E_INTERNAL, RunAicpuRpcSrvLaunch(&kfcTask));

    threadHandle.join();
    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_fp16_retry_reset_sq_fail)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    thread threadHandle([&] {
        request.kfcCmd = KfcCommand::kStopLaunch;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kRuning)
                && (response.execStatus.kfcStatus != KfcStatus::kNull)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStoplaunch);

        request.kfcCmd = KfcCommand::kStopExec;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if (response.execStatus.kfcStatus != KfcStatus::kStoplaunch) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStopExec);

        request.kfcCmd = KfcCommand::kRetry;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kStopExec)
                && (response.execStatus.kfcStatus != KfcStatus::kRuning)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kError);
    });

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);

    KFCTask kfcTask;
    init_kfc_task(kfcTask);

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any()).will(invoke(halSqCpQueryStub_2));

    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(&TaskOrchestrator::WaitFinishWhileLoop)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_SUSPENDING))
        .then(returnValue(HCCL_SUCCESS));

    MOCKER(&AicpuKfcProcess::ResetSqBuff).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    usleep(10000);

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(HCCL_E_INTERNAL, RunAicpuRpcSrvLaunch(&kfcTask));

    threadHandle.join();
    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_fp16_init_opexec_status_fail)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    thread threadHandle([&] {
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kRuning)
                && (response.execStatus.kfcStatus != KfcStatus::kNull)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kError);
    });

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);

    KFCTask kfcTask;
    init_kfc_task(kfcTask);

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any()).will(invoke(halSqCpQueryStub_2));

    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(&TaskOrchestrator::WaitFinishWhileLoop)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_SUSPENDING))
        .then(returnValue(HCCL_SUCCESS));

    MOCKER(&AicpuHdcUtils::InitOpExecStatus).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    usleep(10000);

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(HCCL_E_INTERNAL, RunAicpuRpcSrvLaunch(&kfcTask));

    threadHandle.join();
    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_fp16_task_exec_fail)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    thread threadHandle([&] {
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kRuning)
                && (response.execStatus.kfcStatus != KfcStatus::kNull)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kError);
    });

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);

    KFCTask kfcTask;
    init_kfc_task(kfcTask);

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any()).will(invoke(halSqCpQueryStub_2));

    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(&TaskOrchestrator::WaitFinishWhileLoop)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_TIMEOUT))
        .then(returnValue(HCCL_SUCCESS));

    MOCKER(&AicpuKfcProcess::ResetSqBuff).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    usleep(10000);

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(HCCL_E_TIMEOUT, RunAicpuRpcSrvLaunch(&kfcTask));

    threadHandle.join();
    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_fp16_retry_when_sdma_taskexception)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    thread threadHandle([&] {
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus == KfcStatus::kRuning)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kRuning);

        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kRuning)
                && (response.execStatus.kfcStatus != KfcStatus::kStoplaunch)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStopExec);

        request.kfcCmd = KfcCommand::kRetry;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kStopExec)
                && (response.execStatus.kfcStatus != KfcStatus::kRuning)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kEnd);
    });

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);

    KFCTask kfcTask;
    init_kfc_task(kfcTask);

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any()).will(invoke(halSqCpQueryStub_2));

    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(&TaskOrchestrator::WaitFinishWhileLoop)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_SUSPENDING))
        .then(returnValue(HCCL_SUCCESS));

    MOCKER(&TaskOrchestrator::IsTaskExceptionForHccs)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(true))
        .then(returnValue(true))
        .then(returnValue(false));

    usleep(10000);

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));

    threadHandle.join();
    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_fp16_retry_wait_stop_exec_timeout)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);
    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));
    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    thread threadHandle([&] {
        request.kfcCmd = KfcCommand::kStopLaunch;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kRuning)
                && (response.execStatus.kfcStatus != KfcStatus::kNull)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStoplaunch);
    });

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);

    KFCTask kfcTask;
    init_kfc_task(kfcTask);

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any()).will(invoke(halSqCpQueryStub_2));

    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(&HcclCommAicpu::HcclGetWaitStopExecCmdTimeout).stubs().with(mockcpp::any()).will(returnValue(1));

    MOCKER(&TaskOrchestrator::WaitFinishWhileLoop)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_SUSPENDING))
        .then(returnValue(HCCL_SUCCESS));

    usleep(10000);

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(HCCL_E_INTERNAL, RunAicpuRpcSrvLaunch(&kfcTask));

    threadHandle.join();
    deinit();
}

TEST_F(MC2AicpuRetry_UT, ut_AicpuHcclProcess_function)
{
    AicpuComContext* ctx = AicpuGetComContext();
    HcclOpExecFSM state;
    KfcError errorCode;
    AicpuKfcDeprecatedProcess Process;
    AicpuKfcRpcServer rpc;
    AivAicpuOpParam opParams;
    uint32_t beginSqePos;
    uint32_t endSqePos;
    HcclOpExecFSM fsmState;
    MOCKER(&AicpuKfcDeprecatedProcess::LaunchHcclOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_PARA));
    HcclResult ret = Process.HcclOpExecFsmLaunchProcess(ctx, state, errorCode, opParams, beginSqePos, endSqePos);
    EXPECT_EQ(ret, HCCL_E_PARA);

    MOCKER(&AicpuHdcUtils::GetOpExecCtrlCmd).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_PARA));
    ret = HcclOpExecFsmStoppingProcess(ctx, state, errorCode);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = Process.HcclOpExecFsmWaitRetryProcess(ctx, state, errorCode);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = HcclOpExecFsmStoppedProcess(ctx, state, errorCode, beginSqePos, opParams, beginSqePos, endSqePos);
    EXPECT_EQ(ret, HCCL_E_PARA);
    KfcStatus opstate;
    MOCKER(&AicpuHdcUtils::SetOpExecStatus).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_PARA));
    ret = UpdateOpExecStatus(ctx, fsmState, opstate, errorCode, beginSqePos);
    EXPECT_EQ(ret, HCCL_E_PARA);

    GlobalMockObject::verify();
    KfcCommand cmd = KfcCommand::kExit;
    MOCKER(&AicpuHdcUtils::GetOpExecCtrlCmd)
        .stubs()
        .with(mockcpp::any(), outBound(cmd))
        .will(returnValue(HCCL_SUCCESS));
    ret = HcclOpExecFsmStoppedProcess(ctx, state, errorCode, beginSqePos, opParams, beginSqePos, endSqePos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    cmd = KfcCommand::kNone;
    MOCKER(&AicpuHdcUtils::GetOpExecCtrlCmd)
        .stubs()
        .with(mockcpp::any(), outBound(cmd))
        .will(returnValue(HCCL_SUCCESS));
    ret = HcclOpExecFsmStoppingProcess(ctx, state, errorCode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/* 91093 aicpu 新流程 */

TEST_F(MC2AicpuRetry_UT, ut_InitProcess_BatchSendRecv_hcclCommAicpu)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommAicpu comm;

    AlgResourceResponse algResource;
    Stream stream1(StreamType::STREAM_TYPE_OFFLINE);
    Stream stream2(StreamType::STREAM_TYPE_OFFLINE);
    algResource.slaveStreams.emplace_back(stream1);
    algResource.slaveStreams.emplace_back(stream2);
    HcclOpExecFSM state;
    KfcError errorCode;
    std::string newTag = "test";

    MOCKER(&AicpuHdcUtils::InitOpExecStatus).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(
        &HcclCommAicpu::InitBatchSendRecvOpId, HcclResult(HcclCommAicpu::*)(const OpParam&, AlgResourceResponse&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    comm.retryEnable_ = true;
    OpParam param;
    param.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    ret = comm.HcclOpExecFsmInitProcess(newTag, param, algResource, state, errorCode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

HcclResult QuerySqStatusByTypeStub(uint32_t devId, uint32_t sqId, drvSqCqPropType_t type, uint32_t& outVal)
{
    if (type == DRV_SQCQ_PROP_SQ_TAIL) {
        outVal = 1;
    } else if (type == DRV_SQCQ_PROP_SQ_HEAD) {
        outVal = 0;
    }
    return HCCL_SUCCESS;
}

TEST_F(MC2AicpuRetry_UT, ut_WaitEndProcess_hcclCommAicpu)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommAicpu comm;

    AlgResourceResponse algResource;
    HcclOpExecFSM state;
    KfcError errorCode;
    uint32_t retryCnt = 0;
    std::string tag = "";

    MOCKER(QuerySqStatusByType).stubs().with(mockcpp::any()).will(invoke(QuerySqStatusByTypeStub));

    MOCKER_CPP(&Stream::ClearLocalBuff).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    KfcCommand cmd = KfcCommand::kStopLaunch;
    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().with(mockcpp::any(), outBound(cmd)).will(returnValue(HCCL_SUCCESS));

    comm.retryEnable_ = true;
    cmd = KfcCommand::kStopLaunch;
    OpParam param;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    ret = comm.HcclOpExecFsmWaitEndProcess(param, algResource, state, errorCode, retryCnt, tag, 0);
    EXPECT_EQ(ret, HCCL_E_SUSPENDING);

    cmd = KfcCommand::kExit;
    ret = comm.HcclOpExecFsmWaitEndProcess(param, algResource, state, errorCode, retryCnt, tag, 0);
    EXPECT_EQ(ret, HCCL_E_SUSPENDING);

    comm.dfxExtendInfo_.pollStatus == PollStatus::kStopAsException;
    ret = comm.HcclOpExecFsmWaitEndProcess(param, algResource, state, errorCode, retryCnt, tag, 0);
    EXPECT_EQ(ret, HCCL_E_SUSPENDING);

    comm.dfxExtendInfo_.cqeStatus != dfx::CqeStatus::kDefault;
    ret = comm.HcclOpExecFsmWaitEndProcess(param, algResource, state, errorCode, retryCnt, tag, 0);
    EXPECT_EQ(ret, HCCL_E_SUSPENDING);
}

TEST_F(MC2AicpuRetry_UT, allreduce_backGround_stop)
{
    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    AicpuComContext* ctx = AicpuGetComContext();
    ctx->dfxExtendInfo.commandToBackGroud = CommandToBackGroud::kDefault;
    KfcExecControl request;

    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    request.bgCmd = BackgroundCommand::kStop;
    h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);

    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);
    tilingData.taskType = 0;
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    KFCTask kfcTask;
    init_kfc_task(kfcTask);
    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    StubSqeBuffer sqeBufferStub;
    EXPECT_NE(HCCL_SUCCESS, RunAicpuRpcSrvLaunch(&kfcTask));
    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_ns_stoplaunch)
{
    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));
    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    thread threadHandle([&] {
        request.kfcCmd = KfcCommand::NsStopLaunch;
        h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
        while (true) {
            d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
            if ((response.execStatus.kfcStatus != KfcStatus::kRuning)
                && (response.execStatus.kfcStatus != KfcStatus::kNull)) {
                break;
            }
        }
        EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStoplaunch);
    });
    HcclKFCTilingData tilingData = {0};
    init_tiling_data(tilingData);
    tilingData.taskType = 0;
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    KFCTask kfcTask;
    init_kfc_task(kfcTask);
    MOCKER(&AicpuDispatcher::LaunchTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    usleep(10000);
    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(AICPUSUSPENDING_ERROR, RunAicpuRpcSrvLaunch(&kfcTask));
    threadHandle.join();
    deinit();
}

TEST_F(MC2AicpuRetry_UT, allreduce_ns_stopexec_clear)
{
    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));
    AicpuComContext* ctx = AicpuGetComContext();
    ctx->isStopLaunch = true;
    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    request.kfcCmd = KfcCommand::NsStopExec;
    h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
    dfx_tracer::AicpuExecutorTracer::KfcCommandHandle(ctx);
    d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
    EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kStopExec);
    ctx->isStopLaunch = true;
    request.kfcCmd = KfcCommand::NsClear;
    h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
    dfx_tracer::AicpuExecutorTracer::KfcCommandHandle(ctx);
    d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
    EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kClear);
}

TEST_F(MC2AicpuRetry_UT, allreduce_ns_stopexec_clear_end)
{
    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));
    AicpuComContext* ctx = AicpuGetComContext();
    ctx->isStopLaunch = false;
    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    request.kfcCmd = KfcCommand::NsStopExec;
    h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
    dfx_tracer::AicpuExecutorTracer::KfcCommandHandle(ctx);
    d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
    EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kEnd);
    request.kfcCmd = KfcCommand::NsClear;
    h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
    dfx_tracer::AicpuExecutorTracer::KfcCommandHandle(ctx);
    d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
    EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kEnd);
}

TEST_F(MC2AicpuRetry_UT, allreduce_ns_stopexec_clear_error)
{
    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));
    MOCKER(halTsdrvCtl).stubs().with(mockcpp::any()).will(returnValue(DRV_ERROR_NOT_SUPPORT));
    AicpuComContext* ctx = AicpuGetComContext();
    ctx->isStopLaunch = true;
    KfcExecControl request;
    memset_s(&request, sizeof(KfcExecControl), 0, sizeof(KfcExecControl));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    request.kfcCmd = KfcCommand::NsStopExec;
    h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
    dfx_tracer::AicpuExecutorTracer::KfcCommandHandle(ctx);
    d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
    EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kError);
    ctx->isStopLaunch = true;
    request.kfcCmd = KfcCommand::NsClear;
    h2dTransfer->Put(0, sizeof(KfcExecControl), (u8*)&request);
    dfx_tracer::AicpuExecutorTracer::KfcCommandHandle(ctx);
    d2hTransfer->Get(0, sizeof(KfcExecStatus), (u8*)&response);
    EXPECT_EQ(response.execStatus.kfcStatus, KfcStatus::kError);
}
