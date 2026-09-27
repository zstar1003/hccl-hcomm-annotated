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

using namespace std;
using namespace HcclApi;
using namespace hccl;

class MC2Allgather_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MC2Allgather_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "MC2Allgather_UT TearDown" << std::endl; }
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
        std::cout << "MC2Allgather_UT Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        ResetMC2Context();
        GlobalMockObject::verify();
        std::cout << "MC2Allgather_UT Test TearDown" << std::endl;
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

TEST_F(MC2Allgather_UT, allgather_fp16)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commType = HCCL_CMD_ALLGATHER;
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

TEST_F(MC2Allgather_UT, allgather_bf16)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commType = HCCL_CMD_ALLGATHER;
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

TEST_F(MC2Allgather_UT, allgather_fp16LargeData)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commType = HCCL_CMD_ALLGATHER;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.turnNum = 8;
    tilingData.sendCnt = 100 * 1024 * 1024;
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

TEST_F(MC2Allgather_UT, allgather_doublering)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);
    g_stubDevType = DevType::DEV_TYPE_910_93;

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    auto ctx = AicpuGetComContext();
    ctx->devType = DevType::DEV_TYPE_910_93;

    HcclKFCTilingData tilingData = {0};
    tilingData.commType = HCCL_CMD_ALLGATHER;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.turnNum = 8;
    tilingData.sendCnt = 200;
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

TEST_F(MC2Allgather_UT, allgather_chipTypeDc)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commType = HCCL_CMD_MAX;
    tilingData.dataType = HCCL_DATA_TYPE_BFP16;
    tilingData.turnNum = 2;
    tilingData.sendCnt = 1024;
    tilingData.rspPolicy = 1;
    tilingData.waitPolicy = 1;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.inputA = uint64_t(a);
    kfcTask.outputC = uint64_t(a);
    kfcTask.commOut = uint64_t(a);
    kfcTask.workSpace = uint64_t(a);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    AicpuComContext* ctx = AicpuGetComContext();
    ctx->devType = DevType::DEV_TYPE_310P1;

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(1, RunAicpuRpcSrvLaunch(&kfcTask));
    free(a);
}

TEST_F(MC2Allgather_UT, allgather_hcclKfcTaskHcclOnlyExe)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commType = HCCL_CMD_MAX;
    tilingData.dataType = HCCL_DATA_TYPE_BFP16;
    tilingData.turnNum = 2;
    tilingData.sendCnt = 1024;
    tilingData.rspPolicy = 1;
    tilingData.waitPolicy = 1;
    tilingData.taskType = HCCL_KFC_TASK_HCCL_ONLY_EXE;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.inputA = uint64_t(a);
    kfcTask.outputC = uint64_t(a);
    kfcTask.commOut = uint64_t(a);
    kfcTask.workSpace = uint64_t(a);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(1, RunAicpuRpcSrvLaunch(&kfcTask));
    free(a);
}

TEST_F(MC2Allgather_UT, allgather_mc2Api)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    tilingData.taskType = HCCL_KFC_TASK_HCC_RES_INIT;

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
    (void)memset_s(hcclMsgArea, sizeof(HcclMsgAreaForTest), 0, sizeof(HcclMsgAreaForTest));
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLGATHER;
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 1;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
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

TEST_F(MC2Allgather_UT, allgather_mc2Api_turn2)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

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
    (void)memset_s(hcclMsgArea, sizeof(HcclMsgAreaForTest), 0, sizeof(HcclMsgAreaForTest));
    // commit 1
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLGATHER;
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 1;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // commit 2
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_ALLGATHER;
    hcclMsgArea->sendMsgList[1].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].repeatCnt = 1;
    hcclMsgArea->sendMsgList[1].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[1].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));
    // finilze
    hcclMsgArea->sendMsgList[2].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[2].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[2].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[2]));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    EXPECT_EQ(2, ctx->msgPosForKernel);
    EXPECT_EQ(1, ctx->curTurnCntForKernel);
    free(a);
}

TEST_F(MC2Allgather_UT, allgather_mc2Api_repeat)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

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
    (void)memset_s(hcclMsgArea, sizeof(HcclMsgAreaForTest), 0, sizeof(HcclMsgAreaForTest));
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLGATHER;
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 2;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
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

TEST_F(MC2Allgather_UT, allgather_mc2Api_combine)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

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
    (void)memset_s(hcclMsgArea, sizeof(HcclMsgAreaForTest), 0, sizeof(HcclMsgAreaForTest));
    // commit 1
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLREDUCE;
    hcclMsgArea->sendMsgList[0].opType = HCCL_REDUCE_SUM;
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 1;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // commit 2
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_ALLGATHER;
    hcclMsgArea->sendMsgList[1].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].repeatCnt = 1;
    hcclMsgArea->sendMsgList[1].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[1].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));
    // commit 3
    hcclMsgArea->sendMsgList[2].commType = HCCL_CMD_REDUCE_SCATTER;
    hcclMsgArea->sendMsgList[2].opType = HCCL_REDUCE_SUM;
    hcclMsgArea->sendMsgList[2].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[2].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[2].repeatCnt = 1;
    hcclMsgArea->sendMsgList[2].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[2].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[2].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[2]));
    // finilze
    hcclMsgArea->sendMsgList[3].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[3].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[3].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[3]));

    StubSqeBuffer sqeBufferStub;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    EXPECT_EQ(3, ctx->msgPosForKernel);
    EXPECT_EQ(1, ctx->curTurnCntForKernel);
    free(a);
}
