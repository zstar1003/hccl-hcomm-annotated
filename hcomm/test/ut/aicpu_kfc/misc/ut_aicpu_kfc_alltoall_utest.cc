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
#include "dlhal_function.h"
#include "llt_aicpu_kfc_stub_mc2.h"
#undef private
#undef protected
#include "mc2_handler_pub.h"

using namespace std;
using namespace HcclApi;
using namespace hccl;

class MC2AllToAll_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MC2AllToAll_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "MC2AllToAll_UT TearDown" << std::endl; }
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
        std::cout << "MC2AllToAll_UT Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        ResetMC2Context();
        GlobalMockObject::verify();
        std::cout << "MC2AllToAll_UT Test TearDown" << std::endl;
    }
};

#define init_kfc_args(initTask)                                                                              \
    StubHccCommRes commRes;                                                                                  \
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();                                       \
    AicpuKfcRpcServer::RpcMsgBody msgBody;                                                                   \
    memset_s(&msgBody, sizeof(AicpuKfcRpcServer::RpcMsgBody), 0, sizeof(AicpuKfcRpcServer::RpcMsgBody));     \
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

TEST_F(MC2AllToAll_UT, alltoall_mc2Api)
{
    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    tilingData.hasCommOut = true;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    AicpuComContext* ctx = AicpuGetComContext();
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLTOALL;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[0].dataCnt = 15000000;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 1;
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finilze
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    EXPECT_EQ(1, ctx->msgPosForKernel);
    EXPECT_EQ(1, ctx->curTurnCntForKernel);
    free(a);
}

TEST_F(MC2AllToAll_UT, alltoall_mc2Api_repeat)
{
    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_KERNEL;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    AicpuComContext* ctx = AicpuGetComContext();
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLTOALL;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[0].dataCnt = 400;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 2;
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finilze
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    EXPECT_EQ(1, ctx->msgPosForKernel);
    EXPECT_EQ(2, ctx->curTurnCntForKernel);
    free(a);
}

TEST_F(MC2AllToAll_UT, alltoall_mc2Api_win)
{
    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    tilingData.useBufferType = MC2_BUFFER_TYPE_WINDOW_IN;
    tilingData.hasCommOut = true;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    AicpuComContext* ctx = AicpuGetComContext();
    AicpuComRankInfo* selfRankInfo = &ctx->rankInfo[ctx->rankId];
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLTOALL;
    hcclMsgArea->sendMsgList[0].sendBuffer = selfRankInfo->window;
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[0].dataCnt = 400;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 1;
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finilze
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    EXPECT_EQ(1, ctx->msgPosForKernel);
    EXPECT_EQ(1, ctx->curTurnCntForKernel);
    free(a);
}

TEST_F(MC2AllToAll_UT, alltoall_win_repeat)
{
    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    tilingData.useBufferType = MC2_BUFFER_TYPE_WINDOW_IN;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    AicpuComContext* ctx = AicpuGetComContext();
    AicpuComRankInfo* selfRankInfo = &ctx->rankInfo[ctx->rankId];
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLTOALL;
    hcclMsgArea->sendMsgList[0].sendBuffer = selfRankInfo->window;
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[0].dataCnt = 400;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 2;
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finilze
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    EXPECT_EQ(1, ctx->msgPosForKernel);
    EXPECT_EQ(2, ctx->curTurnCntForKernel);
    free(a);
}

TEST_F(MC2AllToAll_UT, alltoall_mc2Api_prepare_timeout)
{
    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    tilingData.hasCommOut = true;
    tilingData.debugMode = MC2_DEBUG_PREPARE_TIMEOUT;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    AicpuComContext* ctx = AicpuGetComContext();
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLTOALL;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[0].dataCnt = 15000000;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 1;
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finilze
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(HCCL_E_TIMEOUT, RunAicpuRpcSrvLaunch(&kfcTask));
    EXPECT_EQ(0, ctx->msgPosForKernel);
    free(a);
}

TEST_F(MC2AllToAll_UT, alltoall_mc2Api_finalize_timeout)
{
    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    tilingData.hasCommOut = true;
    tilingData.debugMode = MC2_DEBUG_FINALIZE_TIMEOUT;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    AicpuComContext* ctx = AicpuGetComContext();
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLTOALL;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[0].dataCnt = 15000000;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 1;
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finilze
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(HCCL_E_TIMEOUT, RunAicpuRpcSrvLaunch(&kfcTask));
    EXPECT_EQ(1, ctx->msgPosForKernel);
    free(a);
}

TEST_F(MC2AllToAll_UT, alltoall_mc2Api_profiling)
{
    MOCKER(AdprofCheckFeatureIsOn).stubs().will(returnValue(1));
    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    tilingData.hasCommOut = true;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    AicpuComContext* ctx = AicpuGetComContext();
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLTOALL;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[0].dataCnt = 15000000;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 1;
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finilze
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    EXPECT_EQ(1, ctx->msgPosForKernel);
    free(a);
    log_level_set_stub(3);
}

TEST_F(MC2AllToAll_UT, alltoall_mc2_fine_granularity)
{
    HcclDispatcher dispatcher_ = nullptr;
    Stream mainstream;
    u32 roundIdx = 0;
    Mc2HandlerPub mc2HandlerPub;
    EXPECT_EQ(0, mc2HandlerPub.Mc2WaitValue(dispatcher_, mainstream, nullptr, roundIdx));
    EXPECT_EQ(0, mc2HandlerPub.Mc2WriteValue(dispatcher_, mainstream, nullptr));

    Mc2Handler mc2Handler;
    mc2Handler.repeatCnt = 0;
    EXPECT_EQ(1, mc2HandlerPub.Mc2WaitValue(dispatcher_, mainstream, &mc2Handler, roundIdx));
    mc2Handler.repeatCnt = 2;
    mc2Handler.rankSize = 2;
    mc2Handler.stepSize = 2;
    roundIdx = 1;
    MOCKER(HcclDispatcherWaitValue).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(0, mc2HandlerPub.Mc2WaitValue(dispatcher_, mainstream, &mc2Handler, roundIdx));
}
