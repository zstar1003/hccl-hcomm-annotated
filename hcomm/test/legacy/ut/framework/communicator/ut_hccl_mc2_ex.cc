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
#define private public
#define protected public
#include "kfc.h"
#include "internal_exception.h"
#include "hccl_mc2_ex.h"
#include "aicpu_utils.h"
#include "task_exception_func.h"
#include "communicator_impl_lite_manager.h"
#include "rtsq_a5.h"
#undef protected
#undef private

using namespace Hccl;

class HcclMc2ExTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclMc2ExTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HcclMc2ExTest TearDown" << std::endl; }

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

        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqDepth)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<u32>(AC_SQE_MAX_CNT)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();
        std::cout << "A Test case in HcclMc2ExTest SetUp" << std::endl;
    }

    void TearDown() override
    {
        // 清理代码
        GlobalMockObject::verify();
        delete kernelParam;
        delete communicatorImplLite;
        std::cout << "A Test case in HcclMc2ExTest TearDown" << std::endl;
    }

    CommunicatorImplLite* communicatorImplLite;
    HcclKernelParamLite* kernelParam;
    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

// check nullptr
TEST_F(HcclMc2ExTest, Ut_HcclGetCommHandleByCtx_When_CtxIsNull_Expect_ReturnError)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    EXPECT_EQ(HCCL_E_PTR, ::HcclGetCommHandleByCtx(nullptr, &comm));
}

TEST_F(HcclMc2ExTest, Ut_HcclGetCommHandleByCtx_When_CommIsNull_Expect_ReturnError)
{
    auto* ctx = reinterpret_cast<void*>(kernelParam);
    EXPECT_EQ(HCCL_E_PTR, ::HcclGetCommHandleByCtx(ctx, nullptr));
}

TEST_F(HcclMc2ExTest, Ut_HcclGetTaskStatus_When_CommIsNull_Expect_ReturnError)
{
    HcclTaskStatus status = HcclTaskStatus::HCCL_NORMAL_STATUS;
    EXPECT_EQ(HCCL_E_PTR, ::HcclGetTaskStatus(nullptr, &status));
}

TEST_F(HcclMc2ExTest, Ut_HcclGetTaskStatus_When_StatusIsNull_Expect_ReturnError)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    EXPECT_EQ(HCCL_E_PTR, ::HcclGetTaskStatus(comm, nullptr));
}

TEST_F(HcclMc2ExTest, Ut_HcclCheckFinishByStream_When_CommIsNull_Expect_ReturnError)
{
    EXPECT_EQ(HCCL_E_PTR, ::HcclCheckFinishByStream(nullptr));
}

TEST_F(HcclMc2ExTest, Ut_HcclPrintTaskExceptionAllComm_When_CommIsNull_Expect_Return)
{
    ::HcclPrintTaskExceptionAllComm(nullptr);
}

TEST_F(HcclMc2ExTest, Ut_HcclLaunchCcoreWait_When_CommIsNull_Expect_ReturnError)
{
    EXPECT_EQ(HCCL_E_PTR, ::HcclLaunchCcoreWait(nullptr, 0, 0, 0, false));
}

TEST_F(HcclMc2ExTest, Ut_HcclLaunchCcorePost_When_CommIsNull_Expect_ReturnError)
{
    EXPECT_EQ(HCCL_E_PTR, ::HcclLaunchCcorePost(nullptr, 0, 0, 0));
}

TEST_F(HcclMc2ExTest, Ut_HcclLaunchOp_When_CommIsNull_Expect_ReturnError)
{
    HcclOpData data;
    data.opType = HCCL_CMD_ALLREDUCE;
    data.dataType = HCCL_DATA_TYPE_INT16;
    data.dataCount = 536870912;
    data.reduceOp = HCCL_REDUCE_MIN;

    data.input = 0x1000000;
    data.output = 0x2000000;
    EXPECT_EQ(HCCL_E_PTR, ::HcclLaunchOp(nullptr, &data));
}

TEST_F(HcclMc2ExTest, Ut_HcclLaunchOp_When_DataIsNull_Expect_ReturnError)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    EXPECT_EQ(HCCL_E_PTR, ::HcclLaunchOp(comm, nullptr));
}

// HcclGetCommHandleByCtx
TEST_F(HcclMc2ExTest, Ut_HcclGetCommHandleByCtx_When_CommIsFree_Expect_ReturnSuccess)
{
    auto* ctx = reinterpret_cast<void*>(kernelParam);
    void* comm = reinterpret_cast<void*>(communicatorImplLite);

    MOCKER_CPP(&AicpuUtils::Init).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImplLite::UpdateRes).stubs();
    MOCKER_CPP(&CommunicatorImplLiteMgr::Get).stubs().with().will(returnValue(communicatorImplLite));
    MOCKER_CPP(&CommunicatorImplLite::SetDfxOpInfo).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::UpdateHDCommnicate).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImplLite::RegisterRtsqCallback).stubs().will(ignoreReturnValue());
    EXPECT_EQ(HCCL_SUCCESS, ::HcclGetCommHandleByCtx(ctx, &comm));
}

// HcclReleaseComm
TEST_F(HcclMc2ExTest, Ut_HcclReleaseComm_When_ValidParams_Expect_ReturnSuccess)
{
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    ::HcclReleaseComm(comm);
    EXPECT_EQ(false, communicatorImplLite->IsUsed());
}

// HcclGetTaskStatus
HcclResult Mocker_HcclGetTaskStatus_NORMAL(void* comm, HcclTaskStatus* status)
{
    halReportRecvInfo recvInfo;
    recvInfo.type = static_cast<drvSqCqType_t>(DRV_LOGIC_TYPE);
    recvInfo.tsId = 0;
    recvInfo.report_cqe_num = 0;
    recvInfo.stream_id = 0;
    recvInfo.cqId = 0;
    recvInfo.timeout = 0;      // 不设置超时时间，非阻塞
    recvInfo.task_id = 0xFFFF; // 接收所有类型
    recvInfo.cqe_num = 100;    // 单次接收的最大cqe数量
    constexpr uint32_t cqeSize = 100;
    rtLogicCqReport_t tmp = {1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0}; // cqe byte size
    tmp.errorType = 0;
    if (TaskExceptionFunc::GetInstance().IsExceptionCqe(tmp)) {
        *status = HcclTaskStatus::HCCL_CQE_ERROR;
        return HCCL_SUCCESS;
    }
    *status = HcclTaskStatus::HCCL_NORMAL_STATUS;
    return HCCL_SUCCESS;
}

HcclResult Mocker_HcclGetTaskStatus_ERROR(void* comm, HcclTaskStatus* status)
{
    halReportRecvInfo recvInfo;
    recvInfo.type = static_cast<drvSqCqType_t>(DRV_LOGIC_TYPE);
    recvInfo.tsId = 0;
    recvInfo.report_cqe_num = 0;
    recvInfo.stream_id = 0;
    recvInfo.cqId = 0;
    recvInfo.timeout = 0;      // 不设置超时时间，非阻塞
    recvInfo.task_id = 0xFFFF; // 接收所有类型
    recvInfo.cqe_num = 100;    // 单次接收的最大cqe数量
    constexpr uint32_t cqeSize = 100;
    rtLogicCqReport_t tmp = {1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0}; // cqe byte size
    tmp.errorType = 1;
    if (TaskExceptionFunc::GetInstance().IsExceptionCqe(tmp)) {
        *status = HcclTaskStatus::HCCL_CQE_ERROR;
        return HCCL_SUCCESS;
    }
    *status = HcclTaskStatus::HCCL_NORMAL_STATUS;
    return HCCL_SUCCESS;
}

TEST_F(HcclMc2ExTest, Ut_HcclGetTaskStatus_When_TaskHasException_Expect_ReturnErrorStatus)
{
    MOCKER(HcclGetTaskStatus).stubs().will(invoke(Mocker_HcclGetTaskStatus_ERROR));
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    HcclTaskStatus status = HcclTaskStatus::HCCL_NORMAL_STATUS;
    EXPECT_EQ(HCCL_SUCCESS, ::HcclGetTaskStatus(comm, &status));
    EXPECT_EQ(HcclTaskStatus::HCCL_CQE_ERROR, status);
}

TEST_F(HcclMc2ExTest, Ut_HcclGetTaskStatus_When_TaskIsNormal_Expect_ReturnNormalStatus)
{
    MOCKER(HcclGetTaskStatus).stubs().will(invoke(Mocker_HcclGetTaskStatus_NORMAL));
    void* comm = reinterpret_cast<void*>(communicatorImplLite);
    HcclTaskStatus status = HcclTaskStatus::HCCL_NORMAL_STATUS;
    EXPECT_EQ(HCCL_SUCCESS, ::HcclGetTaskStatus(comm, &status));
    EXPECT_EQ(HcclTaskStatus::HCCL_NORMAL_STATUS, status);
}

// HcclCheckFinishByStream
TEST_F(HcclMc2ExTest, Ut_HcclCheckFinishByStream_When_StreamIsFinished_Expect_ReturnSuccess)
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
    EXPECT_EQ(HCCL_SUCCESS, ::HcclCheckFinishByStream(comm));
}

TEST_F(HcclMc2ExTest, Ut_HcclCheckFinishByStream_When_StreamIsRunning_Expect_ReturnUnavail)
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
    EXPECT_EQ(HCCL_E_UNAVAIL, ::HcclCheckFinishByStream(comm));
}

// HcclLaunchCcoreWait
TEST_F(HcclMc2ExTest, Ut_HcclLaunchCcoreWait_When_ValidParams_Expect_ReturnSuccess)
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
    MOCKER_CPP_VIRTUAL(*rtsq, &RtsqA5::CCoreNotifyWait).stubs().will(ignoreReturnValue());
    MOCKER_CPP_VIRTUAL(*rtsq, &RtsqA5::LaunchTask).stubs().will(ignoreReturnValue());

    EXPECT_EQ(HCCL_SUCCESS, ::HcclLaunchCcoreWait(comm, 0, 0, 0, false));
}

// HcclLaunchCcorePost
TEST_F(HcclMc2ExTest, Ut_HcclLaunchCcorePost_When_ValidParams_Expect_ReturnSuccess)
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
    MOCKER_CPP_VIRTUAL(*rtsq, &RtsqA5::CCoreNotifyRecord).stubs().will(ignoreReturnValue());
    MOCKER_CPP_VIRTUAL(*rtsq, &RtsqA5::LaunchTask).stubs().will(ignoreReturnValue());
    EXPECT_EQ(HCCL_SUCCESS, ::HcclLaunchCcorePost(comm, 0, 0, 0));
}
