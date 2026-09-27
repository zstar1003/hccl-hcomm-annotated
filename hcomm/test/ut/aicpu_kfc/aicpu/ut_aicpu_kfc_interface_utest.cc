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
#include "transport_ibverbs_pub.h"
#include "common/aicpu_kfc_def.h"
#include "common/aicpu_hccl_common.h"
#include "algorithm/task_orchestrator.h"
#include "aicpu_kfc/aicpu_kfc_interface.h"
#include "framework/aicpu_hccl_process.h"
#include "framework/aicpu_kfc_process.h"
#include "dlhal_function.h"
#include "llt_aicpu_kfc_stub_mc2.h"
#include "aicpu_kfc/common/aicpu_kfc_tiling_utils.h"
#include "hccl_aicpu_utils.h"
#include "framework/aicpu_kfc_rpc_server.h"
#include "framework/aicpu_kfc_rpc_serverv2.h"
#include "framework/aicpu_kfc_batchwrite_process.h"
#include "aicpu_kfc/common/aicpu_kfc_utils.h"
#undef private
#undef protected

using namespace std;
using namespace HcclApi;
using namespace hccl;
namespace {
uint8_t sqBuffer[64 * 32 * 64];
drvError_t StubhalCqReportRecv(uint32_t devId, struct halReportRecvInfo* info)
{
    info->report_cqe_num = 1U;
    info->type == DRV_LOGIC_TYPE;
    rtLogicCqReport_t* reportinfo = reinterpret_cast<rtLogicCqReport_t*>(info->cqe_addr);
    reportinfo->errorType = 32;
    return drvError_t(0);
}

drvError_t StubhalSqCqQuery(uint32_t devId, struct halSqCqQueryInfo* info)
{
    if (info == nullptr) {
        return DRV_ERROR_INNER_ERR;
    }
    auto queryinfo = *info;
    switch (queryinfo.prop) {
        case DRV_SQCQ_PROP_SQ_HEAD: {
            info->value[0] = 0;
            return DRV_ERROR_NONE;
        }
        case DRV_SQCQ_PROP_SQ_DEPTH: {
            info->value[0] = 4096;
            return DRV_ERROR_NONE;
        }
        case DRV_SQCQ_PROP_SQ_TAIL: {
            info->value[0] = 1;
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
} // namespace
class MC2AicpuInterface_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MC2AicpuInterface_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "MC2AicpuInterface_UT TearDown" << std::endl; }
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
        std::cout << "MC2AicpuInterface_UT Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        ResetMC2Context();
        GlobalMockObject::verify();
        std::cout << "MC2AicpuInterface_UT Test TearDown" << std::endl;
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
    initTask.context = uint64_t(&paramTask);

class MC2AicpuInterfaceV2_UT : public MC2AicpuInterface_UT {
protected:
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        g_stubDevType = DevType::DEV_TYPE_910B;
        MockGetSendRecvCnt();
        MOCKER(halGetDeviceInfo).stubs().with(mockcpp::any()).will(invoke(StubhalGetDeviceInfo));
        std::cout << "MC2AicpuInterfaceV2_UT Test SetUP" << std::endl;
    }
};

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcResInit_nullptr) { EXPECT_EQ(HCCL_E_PARA, RunAicpuKfcResInit(nullptr)); }

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcResInit_hcomUnEqual)
{
    StubHccCommRes commRes;
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();
    KFCResInitTask initTask;
    initTask.context = uint64_t(&paramTask);

    AicpuComContext* ctx = AicpuGetComContext();
    ctx->alreadyInit = true;
    strcpy(ctx->hcomId, "hcom1\0");

    EXPECT_EQ(AC_ERROR_INVALID_PARAM, RunAicpuKfcResInit(&initTask));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcResInit_ok)
{
    dlog_setlevel(HCCL, DLOG_DEBUG, 0);
    KFCResInitTask initTask;
    init_kfc_args(initTask) EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvLaunch_argsNullptr)
{
    EXPECT_EQ(HCCL_E_PARA, RunAicpuRpcSrvLaunch(nullptr));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvGroupLaunch_argsNullptr)
{
    EXPECT_EQ(HCCL_E_PARA, RunAicpuRpcSrvGroupLaunch(nullptr));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvLaunch_tilingDataNullptr)
{
    KFCTask kfcTask;
    kfcTask.tilingData = 0;
    EXPECT_EQ(HCCL_E_PARA, RunAicpuRpcSrvLaunch(&kfcTask));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvGroupLaunch_tilingDataNullptr)
{
    KFCTask kfcTask;
    kfcTask.tilingData = 0;
    EXPECT_EQ(HCCL_E_PARA, RunAicpuRpcSrvGroupLaunch(&kfcTask));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvLaunch_ctxNullptr)
{
    HccCommResParamTask paramTask;
    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_HOST;
    KFCTask kfcTask;
    kfcTask.inputA = 0xa000;
    kfcTask.outputC = 0xb000;
    kfcTask.commOut = 0xc000;
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.workSpace = 0xd000;
    kfcTask.tilingData = uint64_t(&tilingData);

    EXPECT_EQ(HCCL_E_PARA, RunAicpuRpcSrvLaunch(&kfcTask));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvGroupLaunch_ctxNullptr)
{
    HccCommResParamTask paramTask;
    KFCGroupTilingData tilingData;
    tilingData.groupNum = 0;
    KFCTask kfcTask;
    int a = 0;
    kfcTask.inputA = 0xa000;
    kfcTask.outputC = uint64_t(&a);
    kfcTask.commOut = 0xc000;
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.workSpace = 0xd000;
    kfcTask.tilingData = uint64_t(&tilingData);

    EXPECT_EQ(HCCL_E_PARA, RunAicpuRpcSrvGroupLaunch(&kfcTask));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvLaunch_debugModeOnlyCube)
{
    KFCResInitTask initTask;
    init_kfc_args(initTask) EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_HOST;
    tilingData.debugMode = MC2_DEBUG_ONLY_CUBE;
    KFCTask kfcTask;
    kfcTask.inputA = 0xa000;
    kfcTask.outputC = 0xb000;
    kfcTask.commOut = 0xc000;
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.workSpace = 0xd000;
    kfcTask.tilingData = uint64_t(&tilingData);

    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvGroupLaunch_debugModeOnlyCube)
{
    KFCResInitTask initTask;
    init_kfc_args(initTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    KFCGroupTilingData tilingData;
    tilingData.groupNum = 1;
    tilingData.msg[0].preparePosition = TASK_PREPARE_HOST;
    tilingData.msg[0].debugMode = MC2_DEBUG_ONLY_CUBE;
    KFCTask kfcTask;
    u64 a = 0;
    kfcTask.inputA = 0xa000;
    kfcTask.outputC = uint64_t(&a);
    kfcTask.commOut = 0xc000;
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.workSpace = 0xd000;
    kfcTask.tilingData = uint64_t(&tilingData);

    EXPECT_EQ(0, RunAicpuRpcSrvGroupLaunch(&kfcTask));
}

TEST_F(MC2AicpuInterfaceV2_UT, RunAicpuRpcSrvLaunch_debugModeWaitComm)
{
    // 模拟产生了异常cq
    MOCKER(halCqReportRecv).stubs().with(mockcpp::any()).will(invoke(StubhalCqReportRecv));
    MOCKER(halSqCqQuery).stubs().with(mockcpp::any()).will(invoke(StubhalSqCqQuery));
    MOCKER(AicpuSqeContext::QuerySqeInfoByTaskId).stubs().will(returnValue(HCCL_SUCCESS));
    StubHccCommRes commRes;
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();
    AicpuKfcRpcServer::RpcMsgBody msgBody;
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);
    std::shared_ptr<hccl::HDCommunicate> h2dTransfer;
    std::shared_ptr<hccl::HDCommunicate> d2hTransfer;
    h2dTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    d2hTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    h2dTransfer->InitHost();
    d2hTransfer->InitHost();
    paramTask.kfcControlTransferH2DParams = h2dTransfer->GetCommunicateParams();
    paramTask.kfcStatusTransferD2HParams = d2hTransfer->GetCommunicateParams();
    paramTask.config.retryEnable = 0;

    KFCResInitTask initTask;
    initTask.context = uint64_t(&paramTask);

    EXPECT_EQ(0, RunAicpuKfcResInitStub(&initTask));

    HcclKFCTilingData tilingData = {0};
    memset_s(&tilingData, sizeof(HcclKFCTilingData), 0, sizeof(HcclKFCTilingData));
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
    MOCKER(&TaskOrchestrator::WaitFinishWhileLoop)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL))
        .then(returnValue(HCCL_SUCCESS));
    StubSqeBuffer stub;
    // 有异常cq，直接返回失败
    AicpuComContext* ctx = AicpuGetComContext();
    ctx->dfxExtendInfo.commandToBackGroud = CommandToBackGroud::kDefault;
    EXPECT_EQ(4, RunAicpuRpcSrvLaunch(&kfcTask));

    // 执行下一个算子时报错
    EXPECT_EQ(4, RunAicpuRpcSrvLaunch(&kfcTask));
    free(a);
    GlobalMockObject::verify();
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvLaunch_debugModeTimeTaken)
{
    StubHccCommRes commRes;
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();
    AicpuKfcRpcServer::RpcMsgBody msgBody;
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);
    std::shared_ptr<hccl::HDCommunicate> h2dTransfer;
    std::shared_ptr<hccl::HDCommunicate> d2hTransfer;

    h2dTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    d2hTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));

    h2dTransfer->InitHost();
    d2hTransfer->InitHost();
    paramTask.kfcControlTransferH2DParams = h2dTransfer->GetCommunicateParams();
    paramTask.kfcStatusTransferD2HParams = d2hTransfer->GetCommunicateParams();
    paramTask.config.retryEnable = 0;

    KFCResInitTask initTask;
    initTask.context = uint64_t(&paramTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.debugMode = MC2_DEBUG_TIME_TAKEN;
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

// 覆盖率用例
TEST_F(MC2AicpuInterface_UT, ut_GetMsgTypeString)
{
    AicpuComContext* ctx = AicpuGetComContext();
    AicpuKfcRpcServer rpc;
    rpc.GetMsgTypeString(RANK_ADDR);
    rpc.GetMsgTypeString(RANK_WORK);
    rpc.GetMsgTypeString(RANK_ADD_AND_WORK);
    rpc.GetMsgTypeString(RANK_TAIL_TIME);
    rpc.GetMsgTypeString(RANK_MSG_END);
}

// 覆盖率用例
TEST_F(MC2AicpuInterface_UT, ut_GetSendRecvOff1)
{
    HcclKFCTilingData tilingData = {0};
    tilingData.commAlg = 0;
    tilingData.turnNum = 1;
    tilingData.tailNum = 0;
    tilingData.sendOff = 1;
    tilingData.tailSendOff = 1;
    tilingData.recvOff = 1;
    tilingData.tailRecvOff = 1;
    AicpuKfcRpcServer rpc;
    rpc.tilingData_ = &tilingData;
    rpc.genTaskNum_ = 2;
    rpc.GetSendOff();
    rpc.GetRecvOff();
}

// 覆盖率用例
TEST_F(MC2AicpuInterface_UT, ut_GetSendRecvOff2)
{
    HcclKFCTilingData tilingData = {0};
    tilingData.commAlg = 0;
    tilingData.turnNum = 1;
    tilingData.tailNum = 0;
    tilingData.sendOff = 1;
    tilingData.tailSendOff = 1;
    tilingData.recvOff = 1;
    tilingData.tailRecvOff = 1;
    AicpuKfcRpcServer rpc;
    rpc.tilingData_ = &tilingData;
    rpc.genTaskNum_ = 0;
    rpc.GetSendOff();
    rpc.GetRecvOff();
}

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcSrvLaunchTestPrintKFC)
{
    HccCommResParamTask paramTask;
    HcclKFCTilingData tilingData = {0};
    KFCTask kfcTask;
    kfcTask.inputA = 0xa000;
    kfcTask.outputC = 0xb000;
    kfcTask.commOut = 0xc000;
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.workSpace = 0xd000;
    kfcTask.tilingData = uint64_t(&tilingData);
    AicpuKfcUtils::PrintKFCTask(kfcTask);
}

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcSrvLaunchTestPrintHccl)
{
    HccCommResParamTask paramTask;
    HcclKFCTilingData tilingData = {0};
    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 1;
    desc.hasFfts = 1;
    desc.tilingOff = 11;
    desc.isDyn = 0;
    AicpuKfcUtils::PrintHcclCommParamDesc(desc);
}

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcSrvLaunchTestNullArgs) { EXPECT_EQ(1, RunAicpuKfcSrvLaunch(nullptr)); }

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcSrvLaunchTestNullDesc)
{
    HccCommResParamTask paramTask;
    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 1;
    desc.hasFfts = 1;
    desc.tilingOff = 11;
    desc.isDyn = 0;
    auto hexDescAddr = 0x1711;
    void* desPtr = reinterpret_cast<void*>(hexDescAddr);
    HcclKFCTilingData tilingData = {0};
    tilingData.commType = HCCL_CMD_ALLGATHER;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.turnNum = 2;
    tilingData.sendCnt = 16;
    tilingData.rspPolicy = 1;
    tilingData.waitPolicy = 1;
    tilingData.sendArgIndex = 1;
    tilingData.recvArgIndex = 1;
    tilingData.commOutArgIndex = 1;
    tilingData.hasCommOut = 1;
    tilingData.preparePosition = TASK_PREPARE_HOST;
    u64* a = (u64*)malloc(128 * 128);
    KFCTask kfcTask;
    kfcTask.inputA = uint64_t(a);
    kfcTask.outputC = uint64_t(a);
    kfcTask.commOut = uint64_t(a);
    kfcTask.workSpace = uint64_t(a);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);
    EXPECT_EQ(1, RunAicpuKfcSrvLaunch(nullptr));
    free(a);
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvLaunch_Mc2api_Finalize)
{
    StubHccCommRes commRes;
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();
    AicpuKfcRpcServer::RpcMsgBody msgBody;
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);

    std::shared_ptr<hccl::HDCommunicate> h2dTransfer;
    std::shared_ptr<hccl::HDCommunicate> d2hTransfer;
    h2dTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    d2hTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    h2dTransfer->InitHost();
    d2hTransfer->InitHost();
    paramTask.kfcControlTransferH2DParams = h2dTransfer->GetCommunicateParams();
    paramTask.kfcStatusTransferD2HParams = d2hTransfer->GetCommunicateParams();
    paramTask.config.retryEnable = 0;

    KFCResInitTask initTask;
    initTask.context = uint64_t(&paramTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_KERNEL;

    KFCTask kfcTask;
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    StubSqeBuffer sqeBufferStub;
    AicpuComContext* ctx = AicpuGetComContext();
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvLaunch_PreparePositionFail)
{
    StubHccCommRes commRes;
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();
    AicpuKfcRpcServer::RpcMsgBody msgBody;
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);

    std::shared_ptr<hccl::HDCommunicate> h2dTransfer;
    std::shared_ptr<hccl::HDCommunicate> d2hTransfer;
    h2dTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    d2hTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    h2dTransfer->InitHost();
    d2hTransfer->InitHost();
    paramTask.kfcControlTransferH2DParams = h2dTransfer->GetCommunicateParams();
    paramTask.kfcStatusTransferD2HParams = d2hTransfer->GetCommunicateParams();
    paramTask.config.retryEnable = 0;

    KFCResInitTask initTask;
    initTask.context = uint64_t(&paramTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_RESERVED;

    KFCTask kfcTask;
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    StubSqeBuffer sqeBufferStub;
    AicpuComContext* ctx = AicpuGetComContext();
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    EXPECT_EQ(1, RunAicpuRpcSrvLaunch(&kfcTask));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvLaunch_Mc2api_Timeout)
{
    StubHccCommRes commRes;
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();
    AicpuKfcRpcServer::RpcMsgBody msgBody;
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);

    std::shared_ptr<hccl::HDCommunicate> h2dTransfer;
    std::shared_ptr<hccl::HDCommunicate> d2hTransfer;
    h2dTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    d2hTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    h2dTransfer->InitHost();
    d2hTransfer->InitHost();
    paramTask.kfcControlTransferH2DParams = h2dTransfer->GetCommunicateParams();
    paramTask.kfcStatusTransferD2HParams = d2hTransfer->GetCommunicateParams();
    paramTask.config.retryEnable = 0;

    KFCResInitTask initTask;
    initTask.context = uint64_t(&paramTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_KERNEL;

    KFCTask kfcTask;
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    StubSqeBuffer sqeBufferStub;
    AicpuComContext* ctx = AicpuGetComContext();
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    (void)memset_s(hcclMsgArea, sizeof(HcclMsgAreaForTest), 0, sizeof(HcclMsgAreaForTest));
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[0].valid = ~HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    EXPECT_EQ(9, RunAicpuRpcSrvLaunch(&kfcTask));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvLaunch_Mc2api_nullptr)
{
    StubHccCommRes commRes;
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();
    AicpuKfcRpcServer::RpcMsgBody msgBody;
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);

    std::shared_ptr<hccl::HDCommunicate> h2dTransfer;
    std::shared_ptr<hccl::HDCommunicate> d2hTransfer;
    h2dTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    d2hTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    h2dTransfer->InitHost();
    d2hTransfer->InitHost();
    paramTask.kfcControlTransferH2DParams = h2dTransfer->GetCommunicateParams();
    paramTask.kfcStatusTransferD2HParams = d2hTransfer->GetCommunicateParams();
    paramTask.config.retryEnable = 0;

    KFCResInitTask initTask;
    initTask.context = uint64_t(&paramTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_KERNEL;

    KFCTask kfcTask;
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    StubSqeBuffer sqeBufferStub;
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
    hcclMsgArea->sendMsgList[0].sendBuffer = 0;
    hcclMsgArea->sendMsgList[0].recvBuffer = 0;
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finilze
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));
    EXPECT_EQ(1, RunAicpuRpcSrvLaunch(&kfcTask));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvLaunch_Mc2api_debugModePrint)
{
    StubHccCommRes commRes;
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();
    AicpuKfcRpcServer::RpcMsgBody msgBody;
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);

    KFCResInitTask initTask;
    initTask.context = uint64_t(&paramTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData;
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    tilingData.debugMode = MC2_DEBUG_PRINT_MSG;

    KFCTask kfcTask;
    u64* a = (u64*)malloc(1024 * 1024);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    StubSqeBuffer sqeBufferStub;
    AicpuComContext* ctx = AicpuGetComContext();
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    memset_s(hcclMsgArea, sizeof(HcclMsgAreaForTest), 0, sizeof(HcclMsgAreaForTest));
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_ALLGATHER;
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsgArea->sendMsgList[0].dataCnt = 16;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 1;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finilze
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));

    tilingData.debugMode = MC2_DEBUG_PRINT_BUFF;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    EXPECT_EQ(0, RunAicpuRpcSrvLaunch(&kfcTask));
    free(a);
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvLaunch_Mc2api_errorDataTypeOPTYPE)
{
    StubHccCommRes commRes;
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();
    AicpuKfcRpcServer::RpcMsgBody msgBody;
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);

    std::shared_ptr<hccl::HDCommunicate> h2dTransfer;
    std::shared_ptr<hccl::HDCommunicate> d2hTransfer;
    h2dTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    d2hTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    h2dTransfer->InitHost();
    d2hTransfer->InitHost();
    paramTask.kfcControlTransferH2DParams = h2dTransfer->GetCommunicateParams();
    paramTask.kfcStatusTransferD2HParams = d2hTransfer->GetCommunicateParams();
    paramTask.config.retryEnable = 0;

    KFCResInitTask initTask;
    initTask.context = uint64_t(&paramTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_KERNEL;

    KFCTask kfcTask;
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    StubSqeBuffer sqeBufferStub;
    u64* a = (u64*)malloc(1024 * 1024);
    AicpuComContext* ctx = AicpuGetComContext();
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    // commit
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_REDUCE_SCATTER;
    hcclMsgArea->sendMsgList[0].opType = HCCL_REDUCE_PROD;
    hcclMsgArea->sendMsgList[0].hcclDataType = HCCL_DATA_TYPE_INT64;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].repeatCnt = 1;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].recvBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finilze
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));
    EXPECT_EQ(1, RunAicpuRpcSrvLaunch(&kfcTask));
    free(a);
}

TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvLaunch_Mc2api_errorChipType)
{
    StubHccCommRes commRes;
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();
    AicpuKfcRpcServer::RpcMsgBody msgBody;
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);

    std::shared_ptr<hccl::HDCommunicate> h2dTransfer;
    std::shared_ptr<hccl::HDCommunicate> d2hTransfer;
    h2dTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    d2hTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    h2dTransfer->InitHost();
    d2hTransfer->InitHost();
    paramTask.kfcControlTransferH2DParams = h2dTransfer->GetCommunicateParams();
    paramTask.kfcStatusTransferD2HParams = d2hTransfer->GetCommunicateParams();
    paramTask.config.retryEnable = 0;

    KFCResInitTask initTask;
    initTask.context = uint64_t(&paramTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    HcclKFCTilingData tilingData = {0};
    tilingData.preparePosition = TASK_PREPARE_KERNEL;

    KFCTask kfcTask;
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);

    StubSqeBuffer sqeBufferStub;
    u64* a = (u64*)malloc(1024 * 1024);
    AicpuComContext* ctx = AicpuGetComContext();
    ctx->devType = DevType::DEV_TYPE_910_93;
    EXPECT_EQ(1, RunAicpuRpcSrvLaunch(&kfcTask));
    free(a);
}

struct TestTilingData {
    uint32_t version;
    uint32_t commCnt;
    Mc2ServerCfg serverCfg;
    Mc2HcommCfg cfg1;
    Mc2HcommCfg cfg2;
};
struct ArgsInput {
    uint64_t inputDesc;
    void* context1;
    void* context2;
    void* workspace;
    TestTilingData* tilingData;
};

struct ArgsInputForHost {
    uint64_t inputDesc;
    void* context1;
    void* context2;
    void* workspace;
    HcclKFCTilingData* tilingData;
};

struct ArgsGroupInputForHost {
    uint64_t inputDesc;
    void* context1;
    void* context2;
    void* context3;
    KFCGroupTilingDataAuto* tilingData;
    void* context5;
    void* workspace;
    void* context4;
};

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcSrvLaunch_Mc2api_1)
{
    HccCommResParamTask paramTask;
    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 4;
    desc.isDyn = 0;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(CommKfcParamDesc));
    TestTilingData tilingData;
    tilingData.version = 2;
    tilingData.commCnt = 1;
    tilingData.serverCfg.debugMode = 0;
    MOCKER(AicpuKfcProcess::AicpuRunRpcServerForMC2).stubs().will(returnValue((u32)HCCL_SUCCESS));
    ArgsInput curArgs = {inputDesc, nullptr, nullptr, nullptr, &tilingData};
    void* args[sizeof(curArgs) / sizeof(void*)];
    memcpy(args, &curArgs, sizeof(curArgs));
    EXPECT_EQ(1, RunAicpuKfcSrvLaunch(args));

    tilingData.serverCfg.debugMode = MC2_DEBUG_ONLY_CUBE;
    EXPECT_EQ(0, RunAicpuKfcSrvLaunch(args));

    tilingData.version = 1;
    MOCKER(AicpuKfcProcess::RunRpcServerApi).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(RunAicpuKfcSrvLaunch(args), HCCL_E_PARA);

    tilingData.version = 99;
    EXPECT_EQ(RunAicpuKfcSrvLaunch(args), HCCL_E_PARA);
}

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcSrvLaunch_Mc2api_2)
{
    MOCKER(AdprofCheckFeatureIsOn).stubs().will(returnValue(1));
    HccCommResParamTask paramTask;
    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 2;
    desc.hasFfts = 0;
    desc.tilingOff = 4;
    desc.isDyn = 0;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(CommKfcParamDesc));
    TestTilingData tilingData;

    tilingData.serverCfg.debugMode = MC2_DEBUG_TIME_TAKEN;
    tilingData.commCnt = 2;
    tilingData.version = 2;
    HcclOpResParam context1;
    memset(&context1, 0, sizeof(HcclOpResParam));
    HcclOpResParam context2;
    memset(&context2, 0, sizeof(HcclOpResParam));
    int workspace = 0;
    MOCKER(AicpuKfcProcess::AicpuRunRpcServerForMC2).stubs().will(returnValue((u32)HCCL_SUCCESS));
    ArgsInput curArgs = {inputDesc, (void*)&context1, (void*)&context2, (void*)&workspace, &tilingData};
    void* args[sizeof(curArgs) / sizeof(void*)];
    memcpy(args, &curArgs, sizeof(curArgs));
    EXPECT_EQ(0, RunAicpuKfcSrvLaunch(args));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcSrvLaunch_Mc2api_3)
{
    HccCommResParamTask paramTask;
    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 4;
    desc.isDyn = 0;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(CommKfcParamDesc));
    auto hexDescAddr = 0x1711;
    void* desPtr = reinterpret_cast<void*>(hexDescAddr);
    HcclKFCTilingData tilingData = {0};
    tilingData.commType = HCCL_CMD_ALLGATHER;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.turnNum = 2;
    tilingData.sendCnt = 16;
    tilingData.rspPolicy = 1;
    tilingData.waitPolicy = 1;
    tilingData.sendArgIndex = 0;
    tilingData.recvArgIndex = 0;
    tilingData.commOutArgIndex = 1;
    tilingData.hasCommOut = 1;
    tilingData.preparePosition = TASK_PREPARE_HOST;
    u64* a = (u64*)malloc(128 * 128);
    KFCTask kfcTask;
    kfcTask.inputA = uint64_t(a);
    kfcTask.outputC = uint64_t(a);
    kfcTask.commOut = uint64_t(a);
    kfcTask.workSpace = uint64_t(a);
    kfcTask.context = uint64_t(&paramTask);
    kfcTask.tilingData = uint64_t(&tilingData);
    MOCKER(RunAicpuRpcSrvLaunch).stubs().will(returnValue(0));
    int workspace = 0;
    ArgsInputForHost curArgs = {inputDesc, (void*)&paramTask, (void*)a, (void*)&workspace, &tilingData};
    void* args[sizeof(curArgs) / sizeof(void*)];
    memcpy(args, &curArgs, sizeof(curArgs));
    EXPECT_EQ(0, RunAicpuKfcSrvLaunch(args));
    free(a);
}

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcSrvLaunch_Mc2api_4)
{
    HccCommResParamTask paramTask;
    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 1;
    desc.hasFfts = 1;
    desc.tilingOff = 4;
    desc.isDyn = 23;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(CommKfcParamDesc));
    auto hexDescAddr = 0x1711;
    void* desPtr = reinterpret_cast<void*>(hexDescAddr);
    HcclKFCTilingData tilingData = {0};
    tilingData.commType = HCCL_CMD_ALLREDUCE;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.sendArgIndex = 0;
    tilingData.recvArgIndex = 4;
    tilingData.preparePosition = TASK_PREPARE_HOST;
    KFCGroupTilingDataAuto kfcTilingData;
    kfcTilingData.msg[0] = tilingData;
    kfcTilingData.groupNum = 1;
    kfcTilingData.groupTilingMagicNum = 99;
    u64* a = (u64*)malloc(128 * 128);
    *a = 0;
    MOCKER(RunAicpuRpcSrvLaunch).stubs().will(returnValue(0));
    int workspace = 0;

    ArgsGroupInputForHost curArgs
        = {inputDesc, (void*)&paramTask, (void*)a, (void*)a, &kfcTilingData, (void*)a, (void*)&workspace, (void*)a};
    void* args[sizeof(curArgs) / sizeof(void*)];
    memcpy(args, &curArgs, sizeof(curArgs));
    EXPECT_EQ(0, RunAicpuKfcSrvLaunch(args));
    free(a);
}

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcSrvLaunch_tilingDataNullptr)
{
    HccCommResParamTask paramTask;
    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 1;
    desc.hasFfts = 1;
    desc.tilingOff = 4;
    desc.isDyn = 23;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(CommKfcParamDesc));
    auto hexDescAddr = 0x1711;
    void* desPtr = reinterpret_cast<void*>(hexDescAddr);
    HcclKFCTilingData tilingData = {0};
    tilingData.commType = HCCL_CMD_ALLREDUCE;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.sendArgIndex = 0;
    tilingData.recvArgIndex = 4;
    tilingData.preparePosition = TASK_PREPARE_HOST;
    KFCGroupTilingDataAuto kfcTilingData;
    kfcTilingData.msg[0] = tilingData;
    kfcTilingData.groupNum = 0;
    kfcTilingData.groupTilingMagicNum = 99;
    u64* a = (u64*)malloc(128 * 128);
    *a = 0;
    MOCKER(RunAicpuRpcSrvLaunch).stubs().will(returnValue(0));
    int workspace = 0;

    ArgsGroupInputForHost curArgs
        = {inputDesc, (void*)&paramTask, (void*)a, (void*)a, &kfcTilingData, (void*)a, (void*)&workspace, (void*)a};
    void* args[sizeof(curArgs) / sizeof(void*)];
    memcpy(args, &curArgs, sizeof(curArgs));
    EXPECT_EQ(HCCL_E_PARA, RunAicpuKfcSrvLaunch(args));
    free(a);
}

TEST_F(MC2AicpuInterface_UT, RunAicpuInnerRpcSrvGroupLaunch_launchFailed)
{
    HccCommResParamTask paramTask;
    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 1;
    desc.hasFfts = 1;
    desc.tilingOff = 4;
    desc.isDyn = 23;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(CommKfcParamDesc));
    auto hexDescAddr = 0x1711;
    void* desPtr = reinterpret_cast<void*>(hexDescAddr);
    HcclKFCTilingData tilingData = {0};
    tilingData.commType = HCCL_CMD_ALLREDUCE;
    tilingData.dataType = HCCL_DATA_TYPE_FP16;
    tilingData.sendArgIndex = 0;
    tilingData.recvArgIndex = 4;
    tilingData.preparePosition = TASK_PREPARE_HOST;
    KFCGroupTilingDataAuto kfcTilingData;
    kfcTilingData.msg[0] = tilingData;
    kfcTilingData.groupNum = 1;
    kfcTilingData.groupTilingMagicNum = 99;
    u64* a = (u64*)malloc(128 * 128);
    *a = 0;
    MOCKER(RunAicpuRpcSrvLaunch).stubs().will(returnValue(1));
    int workspace = 0;

    ArgsGroupInputForHost curArgs
        = {inputDesc, (void*)&paramTask, (void*)a, (void*)a, &kfcTilingData, (void*)a, (void*)&workspace, (void*)a};
    void* args[sizeof(curArgs) / sizeof(void*)];
    memcpy(args, &curArgs, sizeof(curArgs));
    EXPECT_EQ(HCCL_E_PARA, RunAicpuKfcSrvLaunch(args));
    free(a);
}

// 覆盖率用例
TEST_F(MC2AicpuInterface_UT, ReadApiValidMsg_1)
{
    HcclMsgForTest rMsg;
    HcclMsgForTest msg;
    memset(&rMsg, 0, sizeof(HcclMsgForTest));
    memset(&msg, 0, sizeof(HcclMsgForTest));
    msg.valid = HCCL_MSG_VALID_MASK;
    msg.xorCheck = 0;
    bool reset = false;
    AicpuKfcRpcServer server;
    EXPECT_EQ(false, server.ReadApiValidMsg((HcclApi::HcclMsg*)&rMsg, (HcclApi::HcclMsg*)&msg, reset));
}

struct ArgsInputApi {
    uint64_t inputDesc;
    void* context1;
    void* context2;
    void* workspace;
    Mc2InitTilingInner* tilingData;
};
#if 0
TEST_F(MC2AicpuInterface_UT, RunAicpuRpcSrvLaunch_Mc2api_debugModePrintBuff)
{
    StubHccCommRes commRes;
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();
    AicpuKfcRpcServer::RpcMsgBody msgBody;
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);

    KFCResInitTask initTask;
    initTask.context = uint64_t(&paramTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    u64* a = (u64*)malloc(1024*1024);
    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 4;
    desc.isDyn = 0;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(CommKfcParamDesc));
    Mc2InitTilingInner tilingData;
    std::memset(&tilingData, 0, sizeof(Mc2InitTilingInner));
    tilingData.version = 100;
    tilingData.debugMode = MC2_DEBUG_PRINT_BUFF;
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    MOCKER(AicpuHcclProcess::AicpuGetInnerDevType).stubs().will(returnValue(DevType::DEV_TYPE_910B));
    MOCKER(AicpuKfcProcess::AddTaskForHcclMsg).stubs().will(returnValue(HCCL_SUCCESS));

    StubSqeBuffer sqeBufferStub;
    AicpuComContext *ctx = AicpuGetComContext();
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest *hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest *>(newAddr);
    memset_s(hcclMsgArea, sizeof(HcclMsgAreaForTest), 0, sizeof(HcclMsgAreaForTest));
    // commit
    HcclMsgForTest *hcclMsg = &(hcclMsgArea->sendMsgList[0]);
    hcclMsg->version = 1;
    HcclMsgV1ForTest *hcclMsg0 = reinterpret_cast<HcclMsgV1ForTest *>(hcclMsg);
    hcclMsg0->commType = HCCL_CMD_ALLGATHER;
    hcclMsg0->hcclDataType = HCCL_DATA_TYPE_FP16;
    hcclMsg0->valid = HCCL_MSG_VALID_MASK;
    hcclMsg0->repeatCnt = 1;
    hcclMsg0->dataCnt = 400;
    hcclMsg0->sendBuffer = uint64_t(a);
    hcclMsg0->recvBuffer = uint64_t(a);
    hcclMsg0->xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finilze
    hcclMsg = &(hcclMsgArea->sendMsgList[1]);
    hcclMsg->version = 1;
    HcclMsgV1ForTest *hcclMsg1 = reinterpret_cast<HcclMsgV1ForTest *>(hcclMsg);
    hcclMsg1->commType = HCCL_CMD_FINALIZE;
    hcclMsg1->valid = HCCL_MSG_VALID_MASK;
    hcclMsg1->xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));

    ArgsInputApi curArgs = {inputDesc, (void *)&paramTask, (void *)&paramTask, nullptr, &tilingData};
    void *args[sizeof(curArgs)/ sizeof(void *)];
    memcpy(args, &curArgs, sizeof(curArgs));
    EXPECT_EQ(0, RunAicpuKfcSrvLaunch(args));
    free(a);
}
#endif
TEST_F(MC2AicpuInterface_UT, RunAicpuKfcSrvLaunch_Mc2api_5)
{
    MOCKER(AdprofCheckFeatureIsOn).stubs().will(returnValue(1));
    HccCommResParamTask paramTask;
    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 2;
    desc.hasFfts = 0;
    desc.tilingOff = 4;
    desc.isDyn = 0;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(CommKfcParamDesc));
    Mc2InitTilingInner tilingData;
    tilingData.version = 100;
    tilingData.mc2HcommCnt = 2;
    tilingData.debugMode = 0;
    tilingData.preparePosition = 0;
    MOCKER(AicpuHcclProcess::AicpuGetInnerDevType).stubs().will(returnValue(DevType::DEV_TYPE_910_93));
    MOCKER(AicpuKfcProcess::AicpuRunRpcServerForMC2V2).stubs().will(returnValue((u32)HCCL_SUCCESS));
    ArgsInputApi curArgs = {inputDesc, (void*)&paramTask, (void*)&paramTask, nullptr, &tilingData};
    void* args[sizeof(curArgs) / sizeof(void*)];
    memcpy(args, &curArgs, sizeof(curArgs));
    EXPECT_EQ(0, RunAicpuKfcSrvLaunch(args));

    tilingData.debugMode = MC2_DEBUG_ONLY_CUBE;
    EXPECT_EQ(0, RunAicpuKfcSrvLaunch(args));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcSrvLaunch_Mc2api_6)
{
    StubHccCommRes commRes;
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();
    AicpuKfcRpcServer::RpcMsgBody msgBody;
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);
    std::shared_ptr<hccl::HDCommunicate> h2dTransfer;
    std::shared_ptr<hccl::HDCommunicate> d2hTransfer;
    h2dTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, 8192));
    d2hTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, 8192));
    h2dTransfer->InitHost();
    d2hTransfer->InitHost();
    paramTask.kfcControlTransferH2DParams = h2dTransfer->GetCommunicateParams();
    paramTask.kfcStatusTransferD2HParams = d2hTransfer->GetCommunicateParams();
    paramTask.config.retryEnable = 0;

    KFCResInitTask initTask;
    initTask.context = uint64_t(&paramTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    StubSqeBuffer sqeBufferStub;
    AicpuComContext* ctx = AicpuGetComContext();
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));

    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 2;
    desc.hasFfts = 0;
    desc.tilingOff = 4;
    desc.isDyn = 0;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(CommKfcParamDesc));
    Mc2InitTilingInner tilingData;
    tilingData.version = 100;
    tilingData.mc2HcommCnt = 2;
    tilingData.debugMode = 0;
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    MOCKER(AicpuHcclProcess::AicpuGetInnerDevType).stubs().will(returnValue(DevType::DEV_TYPE_910B));
    MOCKER(AicpuKfcProcess::RunRpcServerApi).stubs().will(returnValue(HCCL_SUCCESS));
    ArgsInputApi curArgs = {inputDesc, (void*)&paramTask, (void*)&paramTask, nullptr, &tilingData};
    void* args[sizeof(curArgs) / sizeof(void*)];
    memcpy(args, &curArgs, sizeof(curArgs));
    EXPECT_EQ(0, RunAicpuKfcSrvLaunch(args));

    tilingData.debugMode = MC2_DEBUG_ONLY_CUBE;
    EXPECT_EQ(0, RunAicpuKfcSrvLaunch(args));
}

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcSrvLaunch_Mc2api_7)
{
    HccCommResParamTask paramTask;
    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 2;
    desc.hasFfts = 0;
    desc.tilingOff = 4;
    desc.isDyn = 0;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(CommKfcParamDesc));
    Mc2InitTilingInner tilingData;
    tilingData.version = 100;
    tilingData.mc2HcommCnt = 2;
    tilingData.debugMode = 0;
    tilingData.preparePosition = 0;
    MOCKER(AicpuHcclProcess::AicpuGetInnerDevType).stubs().will(returnValue(DevType::DEV_TYPE_910_93));
    MOCKER(AicpuKfcProcess::AicpuRunRpcServerForMC2V2).stubs().will(returnValue((u32)HCCL_SUCCESS));
    ArgsInputApi curArgs = {inputDesc, (void*)&paramTask, nullptr, nullptr, &tilingData};
    void* args[sizeof(curArgs) / sizeof(void*)];
    memcpy(args, &curArgs, sizeof(curArgs));
    EXPECT_EQ(RunAicpuKfcSrvLaunch(args), 0U);
}

struct BatchWriteItem {
    uint64_t localBuf;
    uint64_t remoteBuf;
    uint64_t count;
    uint32_t dataType;
    uint32_t remoteRankId;
};

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcSrvLaunch_Mc2api_8)
{
    MOCKER(AdprofCheckFeatureIsOn).stubs().will(returnValue(1));
    StubHccCommRes commRes;
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();
    AicpuKfcRpcServer::RpcMsgBody msgBody;
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);
    std::shared_ptr<hccl::HDCommunicate> h2dTransfer;
    std::shared_ptr<hccl::HDCommunicate> d2hTransfer;
    h2dTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, 8192));
    d2hTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, 8192));
    h2dTransfer->InitHost();
    d2hTransfer->InitHost();
    paramTask.kfcControlTransferH2DParams = h2dTransfer->GetCommunicateParams();
    paramTask.kfcStatusTransferD2HParams = d2hTransfer->GetCommunicateParams();
    paramTask.config.retryEnable = 0;

    KFCResInitTask initTask;
    initTask.context = uint64_t(&paramTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    StubSqeBuffer sqeBufferStub;

    AicpuComContext* ctx = AicpuGetComContext();
    ctx->multiServerFlag = true;
    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }

    BatchWriteItem item0;
    item0.localBuf = 0;
    item0.remoteBuf = 0;
    item0.count = 1;
    item0.dataType = static_cast<uint32_t>(HcclDataType::HCCL_DATA_TYPE_INT32);
    item0.remoteRankId = 0;

    BatchWriteItem item1;
    item1.localBuf = 0;
    item1.remoteBuf = 0;
    item1.count = 1;
    item1.dataType = static_cast<uint32_t>(HcclDataType::HCCL_DATA_TYPE_INT32);
    item1.remoteRankId = 1;

    BatchWriteItem item2;
    item2.localBuf = 0;
    item2.remoteBuf = 0;
    item2.count = 1;
    item2.dataType = static_cast<uint32_t>(HcclDataType::HCCL_DATA_TYPE_INT32);
    item2.remoteRankId = 1;

    // 计算BatchWriteItem对象的大小
    std::size_t size = sizeof(BatchWriteItem);

    // 将item0和item1的数据复制到a指向的内存块中
    u64* a = (u64*)malloc(1024 * 2048);
    std::memcpy(a, &item0, size);
    std::memcpy(a + (size / sizeof(u64)), &item1, size);
    std::memcpy(a + (size / sizeof(u64)) * 2, &item2, size);

    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    (void)memset_s(hcclMsgArea, sizeof(HcclMsgAreaForTest), 0, sizeof(HcclMsgAreaForTest));
    // batch write
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_BATCH_WRITE;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].dataCnt = 3;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finalize
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));

    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 4;
    desc.isDyn = 0;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(CommKfcParamDesc));
    Mc2InitTilingInner tilingData;
    std::memset(&tilingData, 0, sizeof(Mc2InitTilingInner));
    tilingData.version = 100;
    tilingData.mc2HcommCnt = 2;
    tilingData.debugMode = 0;
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    MOCKER(AicpuHcclProcess::AicpuGetInnerDevType).stubs().will(returnValue(DevType::DEV_TYPE_910B));
    MOCKER(AicpuKfcProcess::AddTaskForHcclMsg).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(
        HcclAicpuUtils::PostSend, HcclResult(
                                      const AicpuComContext&, u32, struct std::vector<hccl::Transport::Buffer>&,
                                      struct std::vector<hccl::Transport::Buffer>&, bool isWrite))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclAicpuUtils::GetCpuId).stubs().will(returnValue(2));
    MOCKER(AicpuGetComContext).stubs().will(returnValue(ctx));
    u64 res = 0;
    MOCKER(GetCurCpuTimestamp).stubs().will(returnValue(res));
    MOCKER(HcclAicpuUtils::GetBlockNum).stubs().will(returnValue(1U));
    ArgsInputApi curArgs = {inputDesc, (void*)&paramTask, (void*)&paramTask, nullptr, &tilingData};
    void* args[sizeof(curArgs) / sizeof(void*)];
    memcpy(args, &curArgs, sizeof(curArgs));
    EXPECT_EQ(0, RunAicpuKfcSrvLaunch(args));
    free(a);
    GlobalMockObject::verify();
}

BatchWriteItem CreateBatchWriteItem(uint32_t remoteRankId)
{
    BatchWriteItem item;
    item.localBuf = 0;
    item.remoteBuf = 0;
    item.count = 1;
    item.dataType = static_cast<uint32_t>(HcclDataType::HCCL_DATA_TYPE_INT32);
    item.remoteRankId = remoteRankId;
    return item;
}

static std::mutex MockMtx;
static std::map<std::thread::id, int32_t> threadValues;
static int32_t g_cpuId = 2;
int32_t threadAwareStub()
{
    std::lock_guard<std::mutex> lock(MockMtx);
    std::thread::id tid = std::this_thread::get_id();
    if (threadValues.find(tid) == threadValues.end()) {
        threadValues[tid] = g_cpuId++;
        HCCL_INFO("set thread id %ld cpuId %d", *reinterpret_cast<uint64_t*>(&tid), threadValues[tid]);
    }
    HCCL_INFO(
        "thread %ld get cpuId %d, addr %p.", *reinterpret_cast<uint64_t*>(&tid), threadValues[tid], &threadValues[tid]);
    return threadValues[tid];
}

TEST_F(MC2AicpuInterface_UT, RunAicpuKfcSrvLaunch_Mc2api_9)
{
    MOCKER(AdprofCheckFeatureIsOn).stubs().will(returnValue(1));
    StubHccCommRes commRes;
    HccCommResParamTask paramTask = commRes.StubHccCommResParamTask();
    AicpuKfcRpcServer::RpcMsgBody msgBody;
    paramTask.mc2WorkSpace.workSpace = uint64_t(&msgBody);
    std::shared_ptr<hccl::HDCommunicate> h2dTransfer;
    std::shared_ptr<hccl::HDCommunicate> d2hTransfer;
    h2dTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, 8192));
    d2hTransfer.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, 8192));
    h2dTransfer->InitHost();
    d2hTransfer->InitHost();
    paramTask.kfcControlTransferH2DParams = h2dTransfer->GetCommunicateParams();
    paramTask.kfcStatusTransferD2HParams = d2hTransfer->GetCommunicateParams();
    paramTask.config.retryEnable = 0;

    KFCResInitTask initTask;
    initTask.context = uint64_t(&paramTask);
    EXPECT_EQ(0, RunAicpuKfcResInit(&initTask));

    StubSqeBuffer sqeBufferStub;

    AicpuComContext* ctx = AicpuGetComContext();
    ctx->multiServerFlag = true;

    u64 newAddr = ctx->workSpaceAddr;
    if (newAddr & 0x1ff) {
        newAddr = (newAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    int32_t itemNum = 32;
    std::size_t size = sizeof(BatchWriteItem);

    u64* a = reinterpret_cast<u64*>(malloc(itemNum * size));
    for (uint32_t i = 0; i < itemNum; ++i) {
        BatchWriteItem item = CreateBatchWriteItem(i); // remoteRankId = i
        std::memcpy(reinterpret_cast<void*>(a) + i * size, &item, size);
    }

    HcclMsgAreaForTest* hcclMsgArea = reinterpret_cast<HcclMsgAreaForTest*>(newAddr);
    (void)memset_s(hcclMsgArea, sizeof(HcclMsgAreaForTest), 0, sizeof(HcclMsgAreaForTest));
    // batch write
    hcclMsgArea->sendMsgList[0].commType = HCCL_CMD_BATCH_WRITE;
    hcclMsgArea->sendMsgList[0].sendBuffer = uint64_t(a);
    hcclMsgArea->sendMsgList[0].dataCnt = itemNum;
    hcclMsgArea->sendMsgList[0].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[0].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[0]));
    // finalize
    hcclMsgArea->sendMsgList[1].commType = HCCL_CMD_FINALIZE;
    hcclMsgArea->sendMsgList[1].valid = HCCL_MSG_VALID_MASK;
    hcclMsgArea->sendMsgList[1].xorCheck = GenXorStub(&(hcclMsgArea->sendMsgList[1]));

    CommKfcParamDesc desc;
    desc.version = 1;
    desc.itemNum = 1;
    desc.hasFfts = 0;
    desc.tilingOff = 4;
    desc.isDyn = 0;
    uint64_t inputDesc;
    std::memcpy(&inputDesc, &desc, sizeof(CommKfcParamDesc));
    Mc2InitTilingInner tilingData;
    std::memset(&tilingData, 0, sizeof(Mc2InitTilingInner));
    tilingData.version = 100;
    tilingData.mc2HcommCnt = 2;
    tilingData.debugMode = 0;
    tilingData.preparePosition = TASK_PREPARE_KERNEL;
    MOCKER(AicpuHcclProcess::AicpuGetInnerDevType).stubs().will(returnValue(DevType::DEV_TYPE_910B));
    MOCKER(AicpuKfcProcess::AddTaskForHcclMsg).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(
        HcclAicpuUtils::PostSend, HcclResult(
                                      const AicpuComContext&, u32, struct std::vector<hccl::Transport::Buffer>&,
                                      struct std::vector<hccl::Transport::Buffer>&, bool isWrite))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuGetComContext).stubs().will(returnValue(ctx));
    u64 res = 0;
    MOCKER(GetCurCpuTimestamp).stubs().will(returnValue(res));
    InitMultiThreadSharedCtx(0);
    ArgsInputApi curArgs = {inputDesc, (void*)&paramTask, (void*)&paramTask, nullptr, &tilingData};
    void* args[sizeof(curArgs) / sizeof(void*)];

    memcpy(args, &curArgs, sizeof(curArgs));

    std::vector<std::thread> threads;
    MOCKER(HcclAicpuUtils::GetCpuId).stubs().will(invoke(threadAwareStub));

    uint32_t threadCnt = 3U;
    MOCKER(HcclAicpuUtils::GetBlockNum).stubs().will(returnValue(threadCnt));
    for (uint32_t i = 0U; i < threadCnt; ++i) {
        threads.emplace_back([&]() {
            EXPECT_EQ(0, RunAicpuKfcSrvLaunch(args));
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    free(a);
    GlobalMockObject::verify();
}

TEST_F(MC2AicpuInterface_UT, PrintHcclMsgArea)
{
    HcclMsgAreaForMultiQueForTest msgArea;
    (void)memset_s(&msgArea, sizeof(msgArea), 0, sizeof(msgArea));
    HcclMsgV1ForTest* msg = reinterpret_cast<HcclMsgV1ForTest*>(&(msgArea.sendMsgList[0][0]));
    msg->commType = HCCL_CMD_BATCH_WRITE;
    AicpuKfcUtils::PrintAllHcclMsgAreaForMulti((HcclApi::HcclMsgArea*)&msgArea, true);
    AicpuKfcUtils::PrintAllHcclMsgAreaForMulti((HcclApi::HcclMsgArea*)&msgArea, false);
    AicpuKfcUtils::PrintAllHcclMsgAreaForMulti((HcclApi::HcclMsgArea*)nullptr, true);
    AicpuKfcUtils::PrintAllHcclMsgAreaForMulti((HcclApi::HcclMsgArea*)nullptr, false);
}
