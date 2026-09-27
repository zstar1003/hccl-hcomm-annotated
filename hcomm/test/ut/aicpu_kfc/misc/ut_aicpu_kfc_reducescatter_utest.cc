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
#include "framework/aicpu_kfc_rpc_server.h"
#include "aicpu_kfc/aicpu_kfc_interface.h"
#include "algorithm/aicpu_reduce_scatter.h"
#include "dlhal_function.h"
#include "llt_aicpu_kfc_stub_mc2.h"
#undef private
#undef protected

using namespace std;
using namespace HcclApi;
using namespace hccl;

class MC2Reducescatter_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MC2Reducescatter_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "MC2Reducescatter_UT TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        g_stubDevType = DevType::DEV_TYPE_910B;
        MockGetSendRecvCnt();
        MOCKER(halGetDeviceInfo).stubs().with(mockcpp::any()).will(invoke(StubhalGetDeviceInfo));
        MOCKER(QuerySqStatusByType).stubs().will(returnValue(HCCL_SUCCESS));
        DlHalFunction::GetInstance().DlHalFunctionInit();
        hrtSetDevice(0);
        set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_910B));
        std::cout << "MC2Reducescatter_UT Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        ResetMC2Context();
        GlobalMockObject::verify();
        std::cout << "MC2Reducescatter_UT Test TearDown" << std::endl;
    }
};

#define init_kfc_args(initTask)                                                                              \
    StubHccCommRes commRes;                                                                                  \
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();                                       \
    AicpuKfcRpcServer::RpcMsgBody msgBody;                                                                   \
    (void)memset_s(&msgBody, sizeof(msgBody), 0, sizeof(msgBody));                                           \
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);                                                   \
                                                                                                             \
    std::shared_ptr<hccl::HDCommunicate> h2dTransfer;                                                        \
    std::shared_ptr<hccl::HDCommunicate> d2hTransfer;                                                        \
    h2dTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl))); \
    d2hTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));  \
    h2dTransfer->InitHost();                                                                                 \
    d2hTransfer->InitHost();                                                                                 \
    paramTask.kfcControlTransferH2DParams = h2dTransfer->GetCommunicateParams();                             \
    paramTask.kfcStatusTransferD2HParams = d2hTransfer->GetCommunicateParams();                              \
    paramTask.config.retryEnable = 0;                                                                        \
    initTask.context = uint64_t(&paramTask);

TEST_F(MC2Reducescatter_UT, reducescatter_fp16)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 1;
    tilingData.commType = HCCL_CMD_REDUCE_SCATTER;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.turnNum = 2;
    tilingData.sendCnt = 1024;
    tilingData.rspPolicy = 1;
    tilingData.waitPolicy = 1;
    tilingData.preparePosition = TASK_PREPARE_HOST;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.inputA = uint64_t(a);
    kfcTask.outputC = uint64_t(a);
    kfcTask.commOut = uint64_t(a);
    kfcTask.workSpace = uint64_t(a);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    MOCKER(memcpy_s).stubs().will(returnValue(EOK));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    free(a);
}

#if 1
TEST_F(MC2Reducescatter_UT, reducescatter_bf16)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 1;
    tilingData.commType = HCCL_CMD_REDUCE_SCATTER;
    tilingData.dataType = HCCL_DATA_TYPE_BFP16;
    tilingData.turnNum = 2;
    tilingData.sendCnt = 1024;
    tilingData.rspPolicy = 1;
    tilingData.waitPolicy = 1;
    tilingData.preparePosition = TASK_PREPARE_HOST;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.inputA = uint64_t(a);
    kfcTask.outputC = uint64_t(a);
    kfcTask.commOut = uint64_t(a);
    kfcTask.workSpace = uint64_t(a);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    MOCKER(memcpy_s).stubs().will(returnValue(EOK));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    free(a);
}

TEST_F(MC2Reducescatter_UT, reducescatter_fp16Determinisitic_less_winsize)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 1;
    tilingData.commType = HCCL_CMD_REDUCE_SCATTER;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.turnNum = 2;
    tilingData.sendCnt = 2048;
    tilingData.rspPolicy = 1;
    tilingData.waitPolicy = 1;
    tilingData.preparePosition = TASK_PREPARE_HOST;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(2048 * 2048);
    kfcTask.inputA = uint64_t(a);
    kfcTask.outputC = uint64_t(a);
    kfcTask.commOut = uint64_t(a);
    kfcTask.workSpace = uint64_t(a);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    MOCKER(memcpy_s).stubs().will(returnValue(EOK));

    AicpuComContext* ctx = AicpuGetComContext();
    ctx->determinism = true;
    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    ctx->determinism = false;
    free(a);
}

TEST_F(MC2Reducescatter_UT, reducescatter_fp16Determinisitic_test_equal_winsize)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 1;
    tilingData.commType = HCCL_CMD_REDUCE_SCATTER;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.turnNum = 2;
    tilingData.sendCnt = 104857600;
    tilingData.rspPolicy = 1;
    tilingData.waitPolicy = 1;
    tilingData.preparePosition = TASK_PREPARE_HOST;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(2048 * 2048);
    kfcTask.inputA = uint64_t(a);
    kfcTask.outputC = uint64_t(a);
    kfcTask.commOut = uint64_t(a);
    kfcTask.workSpace = uint64_t(a);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    MOCKER(memcpy_s).stubs().will(returnValue(EOK));

    AicpuComContext* ctx = AicpuGetComContext();
    ctx->determinism = true;
    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    ctx->determinism = false;
    free(a);
}

TEST_F(MC2Reducescatter_UT, reducescatter_fp16Determinisitic_test_greater_winsize)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 1;
    tilingData.commType = HCCL_CMD_REDUCE_SCATTER;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.turnNum = 2;
    tilingData.sendCnt = 209715200;
    tilingData.rspPolicy = 1;
    tilingData.waitPolicy = 1;
    tilingData.preparePosition = TASK_PREPARE_HOST;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(2048 * 2048);
    kfcTask.inputA = uint64_t(a);
    kfcTask.outputC = uint64_t(a);
    kfcTask.commOut = uint64_t(a);
    kfcTask.workSpace = uint64_t(a);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    MOCKER(memcpy_s).stubs().will(returnValue(EOK));

    AicpuComContext* ctx = AicpuGetComContext();
    ctx->determinism = true;
    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    ctx->determinism = false;
    free(a);
}

TEST_F(MC2Reducescatter_UT, reducescatter_doublering)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);
    g_stubDevType = DevType::DEV_TYPE_910_93;

    halChipInfo info = {"Ascend", "910_9381", "0"};
    MOCKER(halGetChipInfo)
        .stubs()
        .with(mockcpp::any(), outBoundP(&info, sizeof(halChipInfo)))
        .will(returnValue(DRV_ERROR_NONE));

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 1;
    tilingData.commType = HCCL_CMD_REDUCE_SCATTER;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.turnNum = 2;
    tilingData.sendCnt = 1024;
    tilingData.rspPolicy = 1;
    tilingData.waitPolicy = 1;
    tilingData.commAlg = 2;
    tilingData.preparePosition = TASK_PREPARE_HOST;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.inputA = uint64_t(a);
    kfcTask.outputC = uint64_t(a);
    kfcTask.commOut = uint64_t(a);
    kfcTask.workSpace = uint64_t(a);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    MOCKER(memcpy_s).stubs().will(returnValue(EOK));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    free(a);
}

TEST_F(MC2Reducescatter_UT, reducescatter_switch)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);
    g_stubDevType = DevType::DEV_TYPE_910_93;

    halChipInfo info = {"Ascend", "910_9381", "0"};
    MOCKER(halGetChipInfo)
        .stubs()
        .with(mockcpp::any(), outBoundP(&info, sizeof(halChipInfo)))
        .will(returnValue(DRV_ERROR_NONE));

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 1;
    tilingData.commType = HCCL_CMD_REDUCE_SCATTER;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.turnNum = 2;
    tilingData.sendCnt = 1024;
    tilingData.rspPolicy = 1;
    tilingData.waitPolicy = 1;
    tilingData.commAlg = 3;
    tilingData.preparePosition = TASK_PREPARE_HOST;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.inputA = uint64_t(a);
    kfcTask.outputC = uint64_t(a);
    kfcTask.commOut = uint64_t(a);
    kfcTask.workSpace = uint64_t(a);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    MOCKER(memcpy_s).stubs().will(returnValue(EOK));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    free(a);
}

TEST_F(MC2Reducescatter_UT, reducescatter_fp16_only_aicpu_commorder0)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 0;
    tilingData.commType = HCCL_CMD_REDUCE_SCATTER;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.turnNum = 2;
    tilingData.sendCnt = 1024;
    tilingData.rspPolicy = 1;
    tilingData.waitPolicy = 1;
    tilingData.debugMode = 4;
    tilingData.useBufferType = 1;
    tilingData.preparePosition = TASK_PREPARE_HOST;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.inputA = uint64_t(a);
    kfcTask.outputC = uint64_t(a);
    kfcTask.commOut = uint64_t(a);
    kfcTask.workSpace = uint64_t(a);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    MOCKER(memcpy_s).stubs().will(returnValue(EOK));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    free(a);
}

TEST_F(MC2Reducescatter_UT, reducescatter_fp16_commorder0)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 0;
    tilingData.commType = HCCL_CMD_REDUCE_SCATTER;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.turnNum = 2;
    tilingData.sendCnt = 1024;
    tilingData.rspPolicy = 1;
    tilingData.waitPolicy = 1;
    tilingData.useBufferType = 1;
    tilingData.preparePosition = TASK_PREPARE_HOST;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.inputA = uint64_t(a);
    kfcTask.outputC = uint64_t(a);
    kfcTask.commOut = uint64_t(a);
    kfcTask.workSpace = uint64_t(a);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    MOCKER(memcpy_s).stubs().will(returnValue(EOK));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    free(a);
}

TEST_F(MC2Reducescatter_UT, reducescatter_fp16_unfold) // 单reducescatter不带计算 aicpu展开
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 0;
    tilingData.commType = HCCL_CMD_REDUCE_SCATTER;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.turnNum = 1;
    tilingData.sendCnt = 256;
    tilingData.totalCnt = 1;
    tilingData.rspPolicy = 0;
    tilingData.waitPolicy = 0;
    tilingData.taskType = HCCL_KFC_TASK_HCCL_ONLY_EXE;
    tilingData.useBufferType = 0;
    tilingData.preparePosition = TASK_PREPARE_HOST;

    KFCTask kfcTask;
    kfcTask.inputA = 0x124080012000;
    kfcTask.outputC = 0;
    kfcTask.commOut = 0x124080013000;
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    auto ctx = AicpuGetComContext();
    AicpuKfcRpcServer rpc;
    rpc.Init(ctx->workSpaceAddr, ctx->notifyOff, ctx->notifyBeginCnt, &kfcTask);
    AivAicpuOpParam msg;
    rpc.CheckRcvAddrMsg(&msg, 0);
    EXPECT_EQ(msg.sendBuffer, 0x124080012000);
    EXPECT_EQ(msg.recvBuffer, 0x124080013000);
}

// 覆盖率用例
TEST_F(MC2Reducescatter_UT, ut_RunAlgorithm)
{
    AicpuComContext* ctx = AicpuGetComContext();
    AicpuReduceScatter rs(ctx);
    rs.rankNum_ = 2;
    rs.RunAlgorithm(HCCL_REDUCE_SUM, nullptr, nullptr, 1, HCCL_DATA_TYPE_FP16);

    ctx->commAlg = CommAlgType::COMM_ALG_RESERVED;
    rs.RunAlgorithm(HCCL_REDUCE_SUM, nullptr, nullptr, 1, HCCL_DATA_TYPE_FP16);
}

#endif
