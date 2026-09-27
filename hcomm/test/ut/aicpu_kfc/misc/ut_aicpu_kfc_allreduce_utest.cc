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
#include "algorithm/aicpu_allreduce.h"
#include "algorithm/task_orchestrator.h"
#include "algorithm/aicpu_dmy_cal_allreduce.h"
#include "dlhal_function.h"
#include "llt_aicpu_kfc_stub_mc2.h"
#undef private
#undef protected

using namespace std;
using namespace HcclApi;
using namespace hccl;

class MC2AicpuAllreduce_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MC2AicpuAllreduce_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "MC2AicpuAllreduce_UT TearDown" << std::endl; }
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
        std::cout << "MC2AicpuAllreduce_UT Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        ResetMC2Context();
        GlobalMockObject::verify();
        std::cout << "MC2AicpuAllreduce_UT Test TearDown" << std::endl;
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

TEST_F(MC2AicpuAllreduce_UT, allreduce_fp16_with_profiling)
{
    MOCKER(AdprofCheckFeatureIsOn).stubs().will(returnValue(1));

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 1;
    tilingData.commType = HCCL_CMD_ALLREDUCE;
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
    log_level_set_stub(3);
}

TEST_F(MC2AicpuAllreduce_UT, allreduce_bfp16)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 1;
    tilingData.commType = HCCL_CMD_ALLREDUCE;
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

TEST_F(MC2AicpuAllreduce_UT, allreduce_fp16Deterministic)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 1;
    tilingData.commType = HCCL_CMD_ALLREDUCE;
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

TEST_F(MC2AicpuAllreduce_UT, allreduce_fp16_only_aicpu)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 1;
    tilingData.commType = HCCL_CMD_ALLREDUCE;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.turnNum = 2;
    tilingData.sendCnt = 1024;
    tilingData.rspPolicy = 1;
    tilingData.waitPolicy = 1;
    tilingData.debugMode = 4;
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

TEST_F(MC2AicpuAllreduce_UT, allreduce_fp16_only_aicpu_commorder0)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 0;
    tilingData.commType = HCCL_CMD_ALLREDUCE;
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

TEST_F(MC2AicpuAllreduce_UT, allreduce_fp16_commorder0)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 0;
    tilingData.commType = HCCL_CMD_ALLREDUCE;
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

TEST_F(MC2AicpuAllreduce_UT, allreduce_fp16_unfold) // 单allreduce不带计算 aicpu展开
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.commOrder = 0;
    tilingData.commType = HCCL_CMD_ALLREDUCE;
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

TEST_F(MC2AicpuAllreduce_UT, allreduce_RunAllReduceOneShot4Stream) // 111
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);
    AicpuComContext* ctx = AicpuGetComContext();
    ctx->rankNum = 1;
    ctx->rankId = 0;

    AicpuAllreduce allreduce(ctx);
    u64 dataCount = 4;
    HcclDataType dataType = HCCL_DATA_TYPE_FP16;
    u64 sendBuffer[dataCount] = {1, 2, 3, 4};
    u64 recvBuffer[dataCount] = {1, 2, 3, 4};
    HcclResult ret = allreduce.RunAllReduceOneShot4Stream(
        HCCL_REDUCE_SUM, static_cast<void*>(sendBuffer), static_cast<void*>(recvBuffer), dataCount, dataType);
    EXPECT_EQ(ret, 0);
}

TEST_F(MC2AicpuAllreduce_UT, allreduce_RunAllReduceSlice) // 222
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);
    AicpuComContext* ctx = AicpuGetComContext();
    ctx->rankNum = 1;
    ctx->rankId = 0;

    AicpuAllreduce allreduce(ctx);
    u64 dataCount = 4;
    HcclDataType dataType = HCCL_DATA_TYPE_FP16;
    u8 outputPtr[dataCount] = {1, 1, 1, 1};
    u8 inputPtr[dataCount] = {1, 1, 1, 1};
    u64 sliceSize[dataCount] = {1, 1, 1, 1};
    u64 dataSlice[dataCount] = {0, 0, 0, 0};

    allreduce.RunAllReduceSlice(outputPtr, inputPtr, sliceSize, dataSlice, HCCL_REDUCE_SUM, dataType);
    EXPECT_EQ(dataSlice[0], 0);
}

TEST_F(MC2AicpuAllreduce_UT, allreduce_RunAllReduceSliceWin2Win) // 333
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);
    AicpuComContext* ctx = AicpuGetComContext();
    ctx->rankNum = 1;
    ctx->rankId = 0;

    AicpuAllreduce allreduce(ctx);
    u64 dataCount = 4;
    HcclDataType dataType = HCCL_DATA_TYPE_FP16;
    u8 outputPtr[dataCount] = {1, 1, 1, 1};
    u64 sliceSize[dataCount] = {1, 1, 1, 1};
    u64 dataSlice[dataCount] = {0, 0, 0, 0};

    allreduce.RunAllReduceSliceWin2Win(outputPtr, sliceSize, dataSlice, HCCL_REDUCE_SUM, dataType);
    EXPECT_EQ(dataSlice[0], 0);
}

TEST_F(MC2AicpuAllreduce_UT, allreduce_PrepareRingSlice)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);
    AicpuComContext* ctx = AicpuGetComContext();
    ctx->rankNum = 2;

    AicpuAllreduce allreduce(ctx);
    u64 dataCount = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP16;
    std::vector<std::vector<u32>> ringOrders = allreduce.GetRingOrders();
    std::vector<std::vector<Slice>> orderedRingSlices;
    HcclResult ret = allreduce.PrepareRingSlice(ringOrders, dataCount, dataType, orderedRingSlices);
    EXPECT_EQ(ret, 0);

    std::vector<u64> dataSizes(ctx->rankNum, 0);
    allreduce.GetDataSizes16K(dataSizes, dataCount * 2);
}

// 补充覆盖率用例
TEST_F(MC2AicpuAllreduce_UT, allreduce_RunAlgorithm)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);
    AicpuComContext* ctx = AicpuGetComContext();
    ctx->rankNum = 1;
    ctx->unitSize = 1024;

    ctx->commOpType = CC_EXE_ONE_SHOT_8_STREAM;
    AicpuAllreduce allreduce(ctx);
    allreduce.RunAlgorithm(HCCL_REDUCE_SUM, nullptr, nullptr, 0, HCCL_DATA_TYPE_FP16, 0, nullptr);

    ctx->commOpType = CC_EXE_ONE_SHOT_1_STREAM;
    allreduce.RunAlgorithm(HCCL_REDUCE_SUM, nullptr, nullptr, 0, HCCL_DATA_TYPE_FP16, 0, nullptr);

    ctx->commOpType = CC_EXE_TWO_SHOT_1_STREAM;
    allreduce.RunAlgorithm(HCCL_REDUCE_SUM, nullptr, nullptr, 0, HCCL_DATA_TYPE_FP16, 0, nullptr);

    ctx->commOpType = CC_EXE_ONE_SHOT_HD;
    allreduce.RunAlgorithm(HCCL_REDUCE_SUM, nullptr, nullptr, 0, HCCL_DATA_TYPE_FP16, 0, nullptr);

    ctx->commOpType = CC_EXE_ONE_SHOT_SINGLE_RING;
    allreduce.RunAlgorithm(HCCL_REDUCE_SUM, nullptr, nullptr, 0, HCCL_DATA_TYPE_FP16, 0, nullptr);

    ctx->commOpType = CC_EXE_ONE_SHOT_SINGLE_RING;
    allreduce.RunAlgorithm(HCCL_REDUCE_SUM, nullptr, nullptr, 0, HCCL_DATA_TYPE_FP16, 0, nullptr);

    ctx->commOpType = CC_EXE_TWO_SHOT_8_STREAM;
    allreduce.RunAlgorithm(HCCL_REDUCE_SUM, nullptr, nullptr, 0, HCCL_DATA_TYPE_FP16, 0, nullptr);

    ctx->useBufferType = MC2_BUFFER_TYPE_WINDOW_IN;
    allreduce.RunAlgorithm(HCCL_REDUCE_SUM, nullptr, nullptr, 0, HCCL_DATA_TYPE_FP16, 0, nullptr);
}

// 补充覆盖率用例
TEST_F(MC2AicpuAllreduce_UT, allreduce_RunAllReduce_fail)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);
    AicpuComContext* ctx = AicpuGetComContext();
    ctx->rankNum = 0;
    ctx->unitSize = 1024;

    AicpuAllreduce allreduce(ctx);
    HcclResult ret = allreduce.RunAllReduce(HCCL_REDUCE_SUM, nullptr, nullptr, 0, HCCL_DATA_TYPE_FP16);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
}

// 补充覆盖率用例
TEST_F(MC2AicpuAllreduce_UT, allreduce_RunAllReduceTwoShot1Stream)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);
    AicpuComContext* ctx = AicpuGetComContext();
    ctx->rankNum = 0;
    ctx->unitSize = 1024;

    AicpuAllreduce allreduce(ctx);
    HcclResult ret = allreduce.RunAllReduceTwoShot1Stream(HCCL_REDUCE_SUM, nullptr, nullptr, 0, HCCL_DATA_TYPE_FP16);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 补充覆盖率用例
TEST_F(MC2AicpuAllreduce_UT, allreduce_RunAllReduceAL)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);
    AicpuComContext* ctx = AicpuGetComContext();
    ctx->commLen = 1024 * 1024;
    ctx->unitSize = 1024;

    AicpuDmyCalAllreduce allreduce(ctx);
    HcclResult ret = allreduce.RunAlgorithm(HCCL_REDUCE_SUM, nullptr, nullptr, 0, HCCL_DATA_TYPE_FP16, 0, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(MC2AicpuAllreduce_UT, retryEnale_false)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    auto ctx = AicpuGetComContext();
    bool res = ctx->retryEnable;
    EXPECT_EQ(false, res);
}

TEST_F(MC2AicpuAllreduce_UT, retryEnale_true)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 1);

    KFCResInitTask initTask;
    init_kfc_args(initTask);
    paramTask.config.retryEnable = 1;
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    auto ctx = AicpuGetComContext();
    bool res = ctx->retryEnable;
    EXPECT_EQ(true, res);
}

TEST_F(MC2AicpuAllreduce_UT, allreduce_mc2Api)
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
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLREDUCE;
    hcclMsgArea->sendMsgList[0].opType = HCCL_REDUCE_SUM;
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

TEST_F(MC2AicpuAllreduce_UT, allreduce_mc2Api_turn2)
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
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLREDUCE;
    hcclMsgArea->sendMsgList[0].opType = HCCL_REDUCE_SUM;
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 1;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // commit 2
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_ALLREDUCE;
    hcclMsgArea->sendMsgList[1].opType = HCCL_REDUCE_SUM;
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

TEST_F(MC2AicpuAllreduce_UT, allreduce_mc2Api_repeat)
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
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLREDUCE;
    hcclMsgArea->sendMsgList[0].opType = HCCL_REDUCE_SUM;
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
