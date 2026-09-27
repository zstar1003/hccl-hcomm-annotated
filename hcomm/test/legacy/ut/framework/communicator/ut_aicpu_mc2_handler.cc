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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <thread>
#include "kfc.h"
#include "internal_exception.h"
#define private public
#define protected public
#include "aicpu_utils.h"
#include "aicpu_mc2_handler.h"
#include "task_exception_func.h"
#include "communicator_impl_lite_manager.h"
#include "rtsq_a5.h"
#undef private
#undef protected

using namespace Hccl;

class AicpuMc2HandlerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AicpuMc2HandlerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "AicpuMc2HandlerTest TearDown" << std::endl; }

protected:
    void SetUp() override
    {
        // 初始化代码
        kernelParam = new HcclKernelParamLite();
        communicatorImplLite = new CommunicatorImplLite(0);
        kernelParam->envConfig.taskExceptionEnable = true;
        kernelParam->comm.idIndex = 0;
        kernelParam->comm.devType = DevType::DEV_TYPE_950;
        kernelParam->op.algOperator.opMode = OpMode::OPBASE;

        MOCKER_CPP(&AicpuUtils::Init).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqDepth)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<u32>(AC_SQE_MAX_CNT)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(1)));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();

        std::cout << "A Test case in AicpuMc2HandlerTest SetUp" << std::endl;
    }

    void TearDown() override
    {
        // 清理代码
        GlobalMockObject::verify();
        delete kernelParam;
        delete communicatorImplLite;
        std::cout << "A Test case in AicpuMc2HandlerTest TearDown" << std::endl;
    }

    AicpuMc2Handler handler;
    CommunicatorImplLite* communicatorImplLite;
    HcclKernelParamLite* kernelParam;
    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

// HcclGetCommHandleByCtx
TEST_F(AicpuMc2HandlerTest, Ut_HcclGetCommHandleByCtx_When_GetCommIsNull_Expect_ReturnError)
{
    auto* ctx = reinterpret_cast<void*>(kernelParam);
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    MOCKER_CPP(&CommunicatorImplLiteMgr::Get)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<Hccl::CommunicatorImplLite*>(nullptr)));
    EXPECT_EQ(HCCL_E_PTR, handler.HcclGetCommHandleByCtx(ctx, &comm));
    GlobalMockObject::verify();
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclGetCommHandleByCtx_When_OpModeIsOffload_Expect_ReturnError)
{
    kernelParam->op.algOperator.opMode = OpMode::OFFLOAD;
    auto* ctx = reinterpret_cast<void*>(kernelParam);
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    MOCKER_CPP(&CommunicatorImplLiteMgr::Get).stubs().with(mockcpp::any()).will(returnValue(communicatorImplLite));
    EXPECT_EQ(HCCL_E_PARA, handler.HcclGetCommHandleByCtx(ctx, &comm));
    kernelParam->op.algOperator.opMode = OpMode::OPBASE;
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclGetCommHandleByCtx_When_CommIsUsed_Expect_ReturnTimeout)
{
    auto* ctx = reinterpret_cast<void*>(kernelParam);
    communicatorImplLite->SetIsUsed(true);
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    MOCKER_CPP(&CommunicatorImplLiteMgr::Get).stubs().with().will(returnValue(communicatorImplLite));
    EXPECT_EQ(HCCL_E_TIMEOUT, handler.HcclGetCommHandleByCtx(ctx, &comm));
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclGetCommHandleByCtx_When_CommIsFree_Expect_ReturnSuccess)
{
    auto* ctx = reinterpret_cast<void*>(kernelParam);
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    MOCKER_CPP(&CommunicatorImplLite::UpdateRes).stubs();
    MOCKER_CPP(&CommunicatorImplLiteMgr::Get).stubs().with().will(returnValue(communicatorImplLite));
    MOCKER_CPP(&CommunicatorImplLite::SetDfxOpInfo).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::UpdateHDCommnicate).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::RegisterRtsqCallback).stubs().will(ignoreReturnValue());
    EXPECT_EQ(HCCL_SUCCESS, handler.HcclGetCommHandleByCtx(ctx, &comm));
}

// HcclReleaseComm
TEST_F(AicpuMc2HandlerTest, Ut_HcclReleaseComm_When_ValidParams_Expect_ReturnSuccess)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    handler.HcclReleaseComm(comm);
    EXPECT_EQ(false, communicatorImplLite->IsUsed());
}

// HcclGetTaskStatus
extern "C" {
drvError_t halCqReportRecv(uint32_t devId, struct halReportRecvInfo* info) { return DRV_ERROR_NONE; }
}

int Mocker_GetReporterInfo_Normal(
    TaskExceptionFunc* This, const StreamLite* curStream, std::shared_ptr<halReportRecvInfo> recvInfo)
{
    recvInfo->type = static_cast<drvSqCqType_t>(DRV_LOGIC_TYPE);
    recvInfo->tsId = 0;
    recvInfo->report_cqe_num = 1;
    recvInfo->stream_id = 0;
    recvInfo->cqId = 0;
    recvInfo->timeout = 0;      // 不设置超时时间，非阻塞
    recvInfo->task_id = 0xFFFF; // 接收所有类型
    recvInfo->cqe_num = 100;    // 单次接收的最大cqe数量
    constexpr uint32_t cqeSize = 100;
    rtLogicCqReport_t* tmpCqeAddr = reinterpret_cast<rtLogicCqReport_t*>(recvInfo->cqe_addr);
    tmpCqeAddr[0] = {1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0};
    tmpCqeAddr->errorType = 0;
    return 0;
}

int Mocker_GetReporterInfo_Error(
    TaskExceptionFunc* This, const StreamLite* curStream, std::shared_ptr<halReportRecvInfo> recvInfo)
{
    recvInfo->type = static_cast<drvSqCqType_t>(DRV_LOGIC_TYPE);
    recvInfo->tsId = 0;
    recvInfo->report_cqe_num = 1;
    recvInfo->stream_id = 0;
    recvInfo->cqId = 0;
    recvInfo->timeout = 0;      // 不设置超时时间，非阻塞
    recvInfo->task_id = 0xFFFF; // 接收所有类型
    recvInfo->cqe_num = 100;    // 单次接收的最大cqe数量
    constexpr uint32_t cqeSize = 100;
    rtLogicCqReport_t* tmpCqeAddr = reinterpret_cast<rtLogicCqReport_t*>(recvInfo->cqe_addr);
    tmpCqeAddr[0] = {1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0};
    tmpCqeAddr->errorType = 1;
    return 0;
}

int Mocker_GetReporterInfo_Continue(
    TaskExceptionFunc* This, const StreamLite* curStream, std::shared_ptr<halReportRecvInfo> recvInfo)
{
    recvInfo->type = static_cast<drvSqCqType_t>(DRV_LOGIC_TYPE);
    recvInfo->tsId = 0;
    recvInfo->report_cqe_num = 1;
    recvInfo->stream_id = 0;
    recvInfo->cqId = 0;
    recvInfo->timeout = 0;      // 不设置超时时间，非阻塞
    recvInfo->task_id = 0xFFFF; // 接收所有类型
    recvInfo->cqe_num = 100;    // 单次接收的最大cqe数量
    constexpr uint32_t cqeSize = 100;
    rtLogicCqReport_t* tmpCqeAddr = reinterpret_cast<rtLogicCqReport_t*>(recvInfo->cqe_addr);
    tmpCqeAddr[0] = {1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0};
    tmpCqeAddr->errorType = 0;
    return 1;
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclGetTaskStatus_When_TaskHasException_Expect_ReturnErrorStatus)
{
    MOCKER_CPP(&TaskExceptionFunc::GetReporterInfo)
        .stubs()
        .with(mockcpp::any())
        .will(invoke(Mocker_GetReporterInfo_Error));

    vector<char> masterBuff = {
        0x00, 0x00, 0x00, 0x00, // id
        0x00, 0x00, 0x00, 0x01, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x01  // cqId
    };
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    HcclTaskStatus status = HcclTaskStatus::HCCL_NORMAL_STATUS;

    communicatorImplLite->GetStreamLiteMgr()->streams.push_back(std::make_unique<StreamLite>(masterBuff));
    EXPECT_EQ(HCCL_SUCCESS, handler.HcclGetTaskStatus(comm, &status));
    EXPECT_EQ(HcclTaskStatus::HCCL_CQE_ERROR, status);
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclGetTaskStatus_When_TaskIsNormal_Expect_ReturnNormalStatus)
{
    MOCKER_CPP(&TaskExceptionFunc::GetReporterInfo)
        .stubs()
        .with(mockcpp::any())
        .will(invoke(Mocker_GetReporterInfo_Normal));

    vector<char> masterBuff = {
        0x00, 0x00, 0x00, 0x00, // id
        0x00, 0x00, 0x00, 0x01, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x01  // cqId
    };

    communicatorImplLite->GetStreamLiteMgr()->streams.push_back(std::make_unique<StreamLite>(masterBuff));

    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    HcclTaskStatus status = HcclTaskStatus::HCCL_NORMAL_STATUS;
    EXPECT_EQ(HCCL_SUCCESS, handler.HcclGetTaskStatus(comm, &status));
    EXPECT_EQ(HcclTaskStatus::HCCL_NORMAL_STATUS, status);
}

// HcclCheckFinishByStream
TEST_F(AicpuMc2HandlerTest, Ut_HcclCheckFinishByStream_When_StreamLiteMgrIsNull_Expect_ReturnError)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    MOCKER_CPP_VIRTUAL(communicatorImplLite, &CommunicatorImplLite::GetStreamLiteMgr)
        .stubs()
        .will(returnValue((StreamLiteMgr*)nullptr));
    EXPECT_EQ(HCCL_E_PTR, handler.HcclCheckFinishByStream(comm));
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclCheckFinishByStream_When_MasterIsNull_Expect_ReturnError)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    MOCKER_CPP(&StreamLiteMgr::GetMaster).stubs().with(mockcpp::any()).will(returnValue((StreamLite*)nullptr));
    EXPECT_EQ(HCCL_E_PTR, handler.HcclCheckFinishByStream(comm));
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclCheckFinishByStream_When_StreamIsFinished_Expect_ReturnSuccess)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    vector<char> masterBuff = {
        0x00, 0x00, 0x00, 0x00, // id
        0x00, 0x00, 0x00, 0x01, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x01  // cqId
    };
    vector<char> slaveBuff = {
        0x00, 0x00, 0x00, 0x01, // id
        0x00, 0x00, 0x00, 0x02, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x02  // cqId
    };
    StreamLite master(masterBuff);
    StreamLite slave(slaveBuff);
    MOCKER_CPP(&StreamLiteMgr::GetMaster).stubs().with().will(returnValue(&master));
    MOCKER_CPP(&StreamLiteMgr::GetSlave).stubs().with(mockcpp::any()).will(returnValue(&slave));

    halSqCqQueryInfo queryInfo;
    queryInfo.tsId = 0;
    queryInfo.sqId = 0;
    queryInfo.cqId = 0;
    queryInfo.type = DRV_NORMAL_TYPE;
    queryInfo.prop = DRV_SQCQ_PROP_SQ_BASE;
    queryInfo.value[0] = 0;
    queryInfo.value[1] = 0;

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any(), outBoundP(&queryInfo, sizeof(queryInfo))).will(returnValue(0));
    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(u32(0)));
    MOCKER_CPP(&RtsqA5::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(u32(0)));
    EXPECT_EQ(HCCL_SUCCESS, handler.HcclCheckFinishByStream(comm));
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclCheckFinishByStream_When_StreamIsRunning_Expect_ReturnUnavail)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    vector<char> masterBuff = {
        0x00, 0x00, 0x00, 0x00, // id
        0x00, 0x00, 0x00, 0x01, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x01  // cqId
    };
    vector<char> slaveBuff = {
        0x00, 0x00, 0x00, 0x01, // id
        0x00, 0x00, 0x00, 0x02, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x02  // cqId
    };
    StreamLite master(masterBuff);
    StreamLite slave(slaveBuff);
    MOCKER_CPP(&StreamLiteMgr::GetMaster).stubs().with().will(returnValue(&master));
    MOCKER_CPP(&StreamLiteMgr::GetSlave).stubs().with(mockcpp::any()).will(returnValue(&slave));

    halSqCqQueryInfo queryInfo;
    queryInfo.tsId = 0;
    queryInfo.sqId = 0;
    queryInfo.cqId = 0;
    queryInfo.type = DRV_NORMAL_TYPE;
    queryInfo.prop = DRV_SQCQ_PROP_SQ_BASE;
    queryInfo.value[0] = 0;
    queryInfo.value[1] = 0;

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any(), outBoundP(&queryInfo, sizeof(queryInfo))).will(returnValue(0));
    MOCKER_CPP(&RtsqA5::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(u32(0)));
    MOCKER_CPP(&RtsqA5::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(u32(8)));
    EXPECT_EQ(HCCL_E_UNAVAIL, handler.HcclCheckFinishByStream(comm));
}

// HcclPrintTaskExceptionAllComm
TEST_F(AicpuMc2HandlerTest, Ut_HcclPrintTaskExceptionAllComm_When_CommunicatorImplLiteMgrIsNull_Expect_Return)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    MOCKER_CPP(&CommunicatorImplLiteMgr::GetAll)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(vector<CommunicatorImplLite*>()));
    handler.HcclPrintTaskExceptionAllComm(comm);
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclPrintTaskExceptionAllComm_When_StreamLiteMgrIsNull_Expect_Return)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    vector<CommunicatorImplLite*> v;
    v.push_back(communicatorImplLite);
    MOCKER_CPP(&CommunicatorImplLiteMgr::GetAll).stubs().with(mockcpp::any()).will(returnValue(v));
    MOCKER_CPP_VIRTUAL(communicatorImplLite, &CommunicatorImplLite::GetStreamLiteMgr)
        .stubs()
        .will(returnValue((StreamLiteMgr*)nullptr));
    handler.HcclPrintTaskExceptionAllComm(comm);
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclPrintTaskExceptionAllComm_When_MasterIsNull_Expect_Return)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    vector<CommunicatorImplLite*> v;
    v.push_back(communicatorImplLite);
    MOCKER_CPP(&CommunicatorImplLiteMgr::GetAll).stubs().with(mockcpp::any()).will(returnValue(v));
    vector<char> masterBuff = {
        0x00, 0x00, 0x00, 0x00, // id
        0x00, 0x00, 0x00, 0x01, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x01  // cqId
    };

    StreamLite master(masterBuff);
    MOCKER_CPP(&StreamLiteMgr::SizeOfSlaves).stubs().will(returnValue(static_cast<u32>(0)));
    MOCKER_CPP(&StreamLiteMgr::GetMaster).stubs().will(returnValue((StreamLite*)nullptr));
    handler.HcclPrintTaskExceptionAllComm(comm);
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclPrintTaskExceptionAllComm_When_RtsqIsNull_Expect_Return)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    vector<CommunicatorImplLite*> v;
    v.push_back(communicatorImplLite);
    MOCKER_CPP(&CommunicatorImplLiteMgr::GetAll).stubs().with(mockcpp::any()).will(returnValue(v));
    vector<char> masterBuff = {
        0x00, 0x00, 0x00, 0x00, // id
        0x00, 0x00, 0x00, 0x01, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x01  // cqId
    };
    vector<char> slaveBuff = {
        0x00, 0x00, 0x00, 0x01, // id
        0x00, 0x00, 0x00, 0x02, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x02  // cqId
    };

    StreamLite master(masterBuff);
    StreamLite slave(slaveBuff);
    MOCKER_CPP(&StreamLiteMgr::GetMaster).stubs().will(returnValue(&master));
    MOCKER_CPP(&StreamLiteMgr::GetSlave).stubs().will(returnValue(&slave));
    handler.HcclPrintTaskExceptionAllComm(comm);
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclPrintTaskExceptionAllComm_When_ReportIsInvalid_Expect_Return)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    vector<CommunicatorImplLite*> v;
    v.push_back(communicatorImplLite);
    MOCKER_CPP(&CommunicatorImplLiteMgr::GetAll).stubs().with(mockcpp::any()).will(returnValue(v));
    vector<char> masterBuff = {
        0x00, 0x00, 0x00, 0x00, // id
        0x00, 0x00, 0x00, 0x01, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x01  // cqId
    };
    vector<char> slaveBuff = {
        0x00, 0x00, 0x00, 0x01, // id
        0x00, 0x00, 0x00, 0x02, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x02  // cqId
    };
    StreamLite master(masterBuff);
    StreamLite slave(slaveBuff);
    MOCKER_CPP(&StreamLiteMgr::GetMaster).stubs().will(returnValue(&master));
    MOCKER_CPP(&StreamLiteMgr::GetSlave).stubs().will(returnValue(&slave));
    MOCKER_CPP(&TaskExceptionFunc::GetReporterInfo).stubs().will(invoke(Mocker_GetReporterInfo_Continue));
    handler.HcclPrintTaskExceptionAllComm(comm);
}

// HcclLaunchCcoreWait
TEST_F(AicpuMc2HandlerTest, Ut_HcclLaunchCcoreWait_When_ValidParams_Expect_ReturnSuccess)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    vector<char> masterBuff = {
        0x00, 0x00, 0x00, 0x00, // id
        0x00, 0x00, 0x00, 0x01, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x01  // cqId
    };
    vector<char> slaveBuff = {
        0x00, 0x00, 0x00, 0x01, // id
        0x00, 0x00, 0x00, 0x02, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x02  // cqId
    };
    StreamLite master(masterBuff);
    StreamLite slave(slaveBuff);
    halSqCqQueryInfo queryInfo;
    queryInfo.tsId = 0;
    queryInfo.sqId = 0;
    queryInfo.cqId = 0;
    queryInfo.type = DRV_NORMAL_TYPE;
    queryInfo.prop = DRV_SQCQ_PROP_SQ_BASE;
    queryInfo.value[0] = 0;
    queryInfo.value[1] = 0;

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any(), outBoundP(&queryInfo, sizeof(queryInfo))).will(returnValue(0));
    MOCKER_CPP(&StreamLiteMgr::GetMaster).stubs().with().will(returnValue(&master));
    auto rtsq = static_cast<RtsqA5*>(master.GetRtsq());
    MOCKER_CPP_VIRTUAL(*rtsq, &RtsqA5::LaunchTask).stubs().will(ignoreReturnValue());

    EXPECT_EQ(HCCL_SUCCESS, handler.HcclLaunchCcoreWait(comm, 0, 0, 0, false));
}

// HcclLaunchCcorePost
TEST_F(AicpuMc2HandlerTest, Ut_HcclLaunchCcorePost_When_ThrowException_Expect_ReturnError)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    vector<char> masterBuff = {
        0x00, 0x00, 0x00, 0x00, // id
        0x00, 0x00, 0x00, 0x01, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x01  // cqId
    };
    vector<char> slaveBuff = {
        0x00, 0x00, 0x00, 0x01, // id
        0x00, 0x00, 0x00, 0x02, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x02  // cqId
    };
    StreamLite master(masterBuff);
    StreamLite slave(slaveBuff);
    halSqCqQueryInfo queryInfo;
    queryInfo.tsId = 0;
    queryInfo.sqId = 0;
    queryInfo.cqId = 0;
    queryInfo.type = DRV_NORMAL_TYPE;
    queryInfo.prop = DRV_SQCQ_PROP_SQ_BASE;
    queryInfo.value[0] = 0;
    queryInfo.value[1] = 0;

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any(), outBoundP(&queryInfo, sizeof(queryInfo))).will(returnValue(0));
    MOCKER_CPP(&StreamLiteMgr::GetMaster).stubs().with().will(returnValue(&master));
    auto rtsq = static_cast<RtsqA5*>(master.GetRtsq());
    MOCKER_CPP_VIRTUAL(*rtsq, &RtsqA5::LaunchTask).stubs().will(ignoreReturnValue());
    MOCKER_CPP_VIRTUAL(*rtsq, &RtsqA5::CCoreNotifyRecord).stubs().will(throws(InternalException("")));
    EXPECT_THROW(handler.HcclLaunchCcorePost(comm, 0, 0, 0), std::exception);
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclLaunchCcorePost_When_ValidParams_Expect_ReturnSuccess)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    vector<char> masterBuff = {
        0x00, 0x00, 0x00, 0x00, // id
        0x00, 0x00, 0x00, 0x01, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x01  // cqId
    };
    vector<char> slaveBuff = {
        0x00, 0x00, 0x00, 0x01, // id
        0x00, 0x00, 0x00, 0x02, // sqId
        0x00, 0x00, 0x00, 0x01, // devPhyId
        0x00, 0x00, 0x00, 0x02  // cqId
    };
    StreamLite master(masterBuff);
    StreamLite slave(slaveBuff);
    halSqCqQueryInfo queryInfo;
    queryInfo.tsId = 0;
    queryInfo.sqId = 0;
    queryInfo.cqId = 0;
    queryInfo.type = DRV_NORMAL_TYPE;
    queryInfo.prop = DRV_SQCQ_PROP_SQ_BASE;
    queryInfo.value[0] = 0;
    queryInfo.value[1] = 0;

    MOCKER(halSqCqQuery).stubs().with(mockcpp::any(), outBoundP(&queryInfo, sizeof(queryInfo))).will(returnValue(0));
    MOCKER_CPP(&StreamLiteMgr::GetMaster).stubs().with().will(returnValue(&master));
    auto rtsq = static_cast<RtsqA5*>(master.GetRtsq());
    MOCKER_CPP_VIRTUAL(*rtsq, &RtsqA5::LaunchTask).stubs().will(ignoreReturnValue());
    EXPECT_EQ(HCCL_SUCCESS, handler.HcclLaunchCcorePost(comm, 0, 0, 0));
}

// HcclLaunchOp
TEST_F(AicpuMc2HandlerTest, Ut_HcclLaunchOp_When_ThrowException_Expect_ReturnError)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    HcclOpData data;
    data.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    data.dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    data.outputDataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    data.dataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    data.dataCount = 536870912;
    data.reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    data.input = 0x1000000;
    data.output = 0x2000000;
    kernelParam->op.algOperator.opType = OP_TYPE_MAP.at(HCCL_CMD_ALLREDUCE);
    AicpuUtils::GetInstance().kernelParam_ = kernelParam;
    AicpuUtils::GetInstance().kernelParamMap_[0] = AicpuUtils::GetInstance().kernelParam_;
    MOCKER_CPP(&CommunicatorImplLite::UpdateLocBuffer).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::GetInsQueue)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(std::make_shared<InsQueue>()));
    MOCKER_CPP(&InsExecutor::ExecuteV82).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&ProfilingReporterLite::ReportAllTasks).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::SetDfxOpInfo).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::UpdateHDCommnicate).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::RegisterRtsqCallback).stubs().will(ignoreReturnValue());

    MOCKER_CPP(&AicpuUtils::RecoverKernelParam).stubs().will(throws(InternalException("")));
    EXPECT_THROW(handler.HcclLaunchOp(comm, &data), std::exception);
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclLaunchOp_When_KernelParamIsNull_Expect_ReturnError)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    AicpuUtils::GetInstance().kernelParam_ = nullptr;
    AicpuUtils::GetInstance().kernelParamMap_ = {};
    HcclOpData data;
    data.opType = HCCL_CMD_ALLREDUCE;
    data.dataType = HCCL_DATA_TYPE_INT16;
    data.outputDataType = HCCL_DATA_TYPE_INT16;
    data.dataCount = 536870912;
    data.reduceOp = HCCL_REDUCE_MIN;
    data.input = 0x1000000;
    data.output = 0x2000000;
    EXPECT_EQ(HCCL_E_PTR, handler.HcclLaunchOp(comm, &data));
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclLaunchOp_When_OpTypeMismatch_Expect_ReturnError)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    HcclOpData data;
    data.opType = HCCL_CMD_ALLREDUCE;
    data.dataType = HCCL_DATA_TYPE_INT16;
    data.outputDataType = HCCL_DATA_TYPE_INT16;
    data.dataCount = 536870912;
    data.reduceOp = HCCL_REDUCE_MIN;
    data.input = 0x1000000;
    data.output = 0x2000000;
    kernelParam->op.algOperator.opType = OP_TYPE_MAP.at(HCCL_CMD_ALLREDUCE);
    AicpuUtils::GetInstance().kernelParam_ = kernelParam;
    AicpuUtils::GetInstance().kernelParamMap_[0] = AicpuUtils::GetInstance().kernelParam_;
    MOCKER_CPP(&CommunicatorImplLite::UpdateLocBuffer).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::GetInsQueue)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(std::make_shared<InsQueue>()));
    MOCKER_CPP(&InsExecutor::ExecuteV82).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&ProfilingReporterLite::ReportAllTasks).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    EXPECT_EQ(HCCL_E_PARA, handler.HcclLaunchOp(comm, &data));
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclLaunchOp_When_ALLREDUCE_Expect_ReturnSuccess)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    HcclOpData data;
    data.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    data.dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    data.outputDataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    data.dataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    data.dataCount = 536870912;
    data.reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    data.input = 0x1000000;
    data.output = 0x2000000;
    kernelParam->op.algOperator.opType = OP_TYPE_MAP.at(HCCL_CMD_ALLREDUCE);
    AicpuUtils::GetInstance().kernelParam_ = kernelParam;
    AicpuUtils::GetInstance().kernelParamMap_[0] = AicpuUtils::GetInstance().kernelParam_;
    MOCKER_CPP(&CommunicatorImplLite::UpdateLocBuffer).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::GetInsQueue)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(std::make_shared<InsQueue>()));
    MOCKER_CPP(&InsExecutor::ExecuteV82).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&ProfilingReporterLite::ReportAllTasks).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::SetDfxOpInfo).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::UpdateHDCommnicate).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::RegisterRtsqCallback).stubs().will(ignoreReturnValue());
    EXPECT_EQ(HCCL_SUCCESS, handler.HcclLaunchOp(comm, &data));
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclLaunchOp_When_ALLTOALL_Expect_ReturnSuccess)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    HcclOpData data;
    data.opType = HcclCMDType::HCCL_CMD_ALLTOALL;
    data.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    data.outputDataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    data.dataCount = 536870912;
    data.input = 0x1000000;
    data.output = 0x2000000;
    data.all2AllDataDes.sendType = HcclDataType::HCCL_DATA_TYPE_INT8;
    data.all2AllDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_INT8;
    data.all2AllDataDes.sendCount = 1;
    data.all2AllDataDes.recvCount = 1;
    kernelParam->op.algOperator.opType = OP_TYPE_MAP.at(HCCL_CMD_ALLTOALL);
    AicpuUtils::GetInstance().kernelParam_ = kernelParam;
    AicpuUtils::GetInstance().kernelParamMap_[0] = AicpuUtils::GetInstance().kernelParam_;
    MOCKER_CPP(&CommunicatorImplLite::UpdateLocBuffer).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::GetInsQueue)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(std::make_shared<InsQueue>()));
    MOCKER_CPP(&InsExecutor::ExecuteV82).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&ProfilingReporterLite::ReportAllTasks).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::SetDfxOpInfo).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::UpdateHDCommnicate).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::RegisterRtsqCallback).stubs().will(ignoreReturnValue());
    EXPECT_EQ(HCCL_SUCCESS, handler.HcclLaunchOp(comm, &data));
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclLaunchOp_When_ALLTOALLV_Expect_ReturnSuccess)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    HcclOpData data;
    data.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    data.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    data.outputDataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    data.dataCount = 536870912;
    data.input = 0x1000000;
    data.output = 0x2000000;
    data.all2AllVDataDes.sendType = HcclDataType::HCCL_DATA_TYPE_INT8;
    data.all2AllVDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_INT8;
    u64 sendCounts[4] = {100, 100, 100, 100};
    u64 sdispls[4] = {0, 100, 200, 300};
    u64 recvCounts[4] = {100, 100, 100, 100};
    u64 rdispls[4] = {0, 100, 200, 300};
    data.all2AllVDataDes.sendCounts = (void*)sendCounts;
    data.all2AllVDataDes.sdispls = (void*)sdispls;
    data.all2AllVDataDes.recvCounts = (void*)recvCounts;
    data.all2AllVDataDes.rdispls = (void*)rdispls;
    kernelParam->op.algOperator.opType = OP_TYPE_MAP.at(HCCL_CMD_ALLTOALLV);
    AicpuUtils::GetInstance().kernelParam_ = kernelParam;
    AicpuUtils::GetInstance().kernelParamMap_[0] = AicpuUtils::GetInstance().kernelParam_;
    MOCKER_CPP(&CommunicatorImplLite::UpdateLocBuffer).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::GetInsQueue)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(std::make_shared<InsQueue>()));
    MOCKER_CPP(&InsExecutor::ExecuteV82).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&ProfilingReporterLite::ReportAllTasks).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::SetDfxOpInfo).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::UpdateHDCommnicate).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::RegisterRtsqCallback).stubs().will(ignoreReturnValue());
    EXPECT_EQ(HCCL_SUCCESS, handler.HcclLaunchOp(comm, &data));
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclLaunchOp_When_ALLTOALLVC_Expect_ReturnSuccess)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    HcclOpData data;
    data.opType = HcclCMDType::HCCL_CMD_ALLTOALLVC;
    data.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    data.outputDataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    data.dataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    data.dataCount = 536870912;
    data.input = 0x1000000;
    data.output = 0x2000000;
    data.all2AllVCDataDes.sendType = HcclDataType::HCCL_DATA_TYPE_INT8;
    data.all2AllVCDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_INT8;
    uint64_t sendCountMatrixTmp = 0;
    data.all2AllVCDataDes.sendCountMatrix = &sendCountMatrixTmp;
    kernelParam->op.algOperator.opType = OP_TYPE_MAP.at(HCCL_CMD_ALLTOALLVC);
    AicpuUtils::GetInstance().kernelParam_ = kernelParam;
    AicpuUtils::GetInstance().kernelParamMap_[0] = AicpuUtils::GetInstance().kernelParam_;
    MOCKER_CPP(&CommunicatorImplLite::UpdateLocBuffer).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::GetInsQueue)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(std::make_shared<InsQueue>()));
    MOCKER_CPP(&InsExecutor::ExecuteV82).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&ProfilingReporterLite::ReportAllTasks).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::SetDfxOpInfo).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::UpdateHDCommnicate).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::RegisterRtsqCallback).stubs().will(ignoreReturnValue());
    EXPECT_EQ(HCCL_SUCCESS, handler.HcclLaunchOp(comm, &data));
}

TEST_F(AicpuMc2HandlerTest, Ut_HcclLaunchOp_When_REDUCE_SCATTER_V_Expect_ReturnSuccess)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    vector<u64> counts{300, 200};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }
    HcclOpData data;
    data.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
    data.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    data.outputDataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    data.dataCount = 536870912;
    data.input = 0x1000000;
    data.output = 0x2000000;
    data.reduceOp = HCCL_REDUCE_SUM;
    data.vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    data.vDataDes.counts = (void*)counts.data();
    data.vDataDes.displs = (void*)displs.data();
    kernelParam->op.algOperator.opType = OP_TYPE_MAP.at(HCCL_CMD_REDUCE_SCATTER_V);
    AicpuUtils::GetInstance().kernelParam_ = kernelParam;
    AicpuUtils::GetInstance().kernelParamMap_[0] = AicpuUtils::GetInstance().kernelParam_;
    MOCKER_CPP(&CommunicatorImplLite::UpdateLocBuffer).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::GetInsQueue)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(std::make_shared<InsQueue>()));
    MOCKER_CPP(&InsExecutor::ExecuteV82).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&ProfilingReporterLite::ReportAllTasks).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::SetDfxOpInfo).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::UpdateHDCommnicate).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::RegisterRtsqCallback).stubs().will(ignoreReturnValue());
    EXPECT_EQ(HCCL_SUCCESS, handler.HcclLaunchOp(comm, &data));
}
