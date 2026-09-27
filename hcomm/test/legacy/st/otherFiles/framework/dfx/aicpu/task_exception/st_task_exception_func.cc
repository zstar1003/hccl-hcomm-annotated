/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "task_exception_func.h"
#include "rtsq_base.h"
#undef private
#undef protected

using namespace std;
using namespace Hccl;

class TaskExceptionFuncTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TaskExceptionFuncTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "TaskExceptionFuncTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in TaskExceptionFuncTest SetUp" << std::endl; }

    virtual void TearDown()
    {
        std::cout << "A Test case in TaskExceptionFuncTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

TEST_F(TaskExceptionFuncTest, SetDevId_ShouldSetDevId_WhenCalled)
{
    TaskExceptionFunc& instance = TaskExceptionFunc::GetInstance();
    instance.SetDevId(12345);
    EXPECT_EQ(instance.GetDevId(), 12345);
}

TEST_F(TaskExceptionFuncTest, RegisterCallback_ShouldRegisterCallback_WhenCalled)
{
    TaskExceptionFunc& instance = TaskExceptionFunc::GetInstance();
    instance.RegisterCallback([](const rtLogicCqReport_t* report) {
        // Do something with the report
    });
}

TEST_F(TaskExceptionFuncTest, Register_ShouldRegisterStreamLite_WhenCalled)
{
    // Mock StreamLite
    MOCKER_CPP(&RtsqBase::QuerySqBaseAddr).stubs().will(returnValue(static_cast<u64>(0)));
    MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().will(returnValue(static_cast<u32>(0)));
    std::vector<char> uniqueId{'0', '0', '0'};
    StreamLite streamLite(uniqueId);

    TaskExceptionFunc& instance = TaskExceptionFunc::GetInstance();
    instance.Register(&streamLite);
}

TEST_F(TaskExceptionFuncTest, UnRegister_ShouldUnRegisterStreamLite_WhenCalled)
{
    // Mock StreamLite
    MOCKER_CPP(&RtsqBase::QuerySqBaseAddr).stubs().will(returnValue(static_cast<u64>(0)));
    MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().will(returnValue(static_cast<u32>(0)));
    std::vector<char> uniqueId{'0', '0', '0'};
    StreamLite streamLite(uniqueId);

    TaskExceptionFunc& instance = TaskExceptionFunc::GetInstance();
    instance.Register(&streamLite);
    instance.UnRegister(&streamLite);
}

TEST_F(TaskExceptionFuncTest, StringLogicCqReportInfo_ShouldReturnCorrectString_WhenCalled)
{
    TaskExceptionFunc& instance = TaskExceptionFunc::GetInstance();

    rtLogicCqReport_t report;
    report.streamId = 1;
    report.taskId = 2;
    report.errorCode = 3;
    report.errorType = 0b0;
    report.sqeType = 5;
    report.sqId = 6;
    report.sqHead = 7;
    report.matchFlag = 8;
    report.dropFlag = 9;
    report.errorBit = 10;
    report.accError = 11;
    EXPECT_EQ(
        instance.StringLogicCqReportInfo(report), "streamId :1 taskId :2 errorCode :3 errorType :0 sqeType :5 sqId :6 "
                                                  "sqHead :7 matchFlag :0 dropFlag :1 errorBit :0 accError :1");

    report.errorType = 0b11;
    EXPECT_EQ(
        instance.StringLogicCqReportInfo(report),
        "streamId :1 taskId :2 errorCode :3 errorType :3(bus error) sqeType :5 sqId :6 sqHead :7 matchFlag :0 dropFlag "
        ":1 errorBit :0 accError :1");

    report.errorType = 0b101;
    EXPECT_EQ(
        instance.StringLogicCqReportInfo(report), "streamId :1 taskId :2 errorCode :3 errorType :5(rsv) sqeType :5 "
                                                  "sqId :6 sqHead :7 matchFlag :0 dropFlag :1 errorBit :0 accError :1");

    report.errorType = 0b1001;
    EXPECT_EQ(
        instance.StringLogicCqReportInfo(report),
        "streamId :1 taskId :2 errorCode :3 errorType :9(sqe error) sqeType :5 sqId :6 sqHead :7 matchFlag :0 dropFlag "
        ":1 errorBit :0 accError :1");

    report.errorType = 0b10001;
    EXPECT_EQ(
        instance.StringLogicCqReportInfo(report),
        "streamId :1 taskId :2 errorCode :3 errorType :17(res conflict error) sqeType :5 sqId :6 sqHead :7 matchFlag "
        ":0 dropFlag :1 errorBit :0 accError :1");

    report.errorType = 0b100001;
    EXPECT_EQ(
        instance.StringLogicCqReportInfo(report),
        "streamId :1 taskId :2 errorCode :3 errorType :33(pre_p/post_p error) sqeType :5 sqId :6 sqHead :7 matchFlag "
        ":0 dropFlag :1 errorBit :0 accError :1");

    report.errorType = 0b1;
    report.errorCode = 0x4;
    EXPECT_EQ(
        instance.StringLogicCqReportInfo(report),
        "streamId :1 taskId :2 errorCode :4(Transaction Retry Counter Exceeded) errorType :1(exception) sqeType :5 "
        "sqId :6 sqHead :7 matchFlag :0 dropFlag :1 errorBit :0 accError :1");

    report.errorCode = 0xF;
    EXPECT_EQ(
        instance.StringLogicCqReportInfo(report),
        "streamId :1 taskId :2 errorCode :15(Reserved) errorType :1(exception) sqeType :5 sqId :6 sqHead :7 matchFlag "
        ":0 dropFlag :1 errorBit :0 accError :1");
}

TEST_F(TaskExceptionFuncTest, getTrailingZeros_ShouldReturnCorrectCount_WhenCalled)
{
    TaskExceptionFunc& instance = TaskExceptionFunc::GetInstance();
    EXPECT_EQ(instance.GetTrailingZeros(16), 4);
}

TEST_F(TaskExceptionFuncTest, IsExceptionCqe_ShouldReturnTrue_WhenErrorTypeIsException)
{
    TaskExceptionFunc& instance = TaskExceptionFunc::GetInstance();
    rtLogicCqReport_t report;
    report.errorType = 0x3F;
    EXPECT_TRUE(instance.IsExceptionCqe(report));
}

TEST_F(TaskExceptionFuncTest, IsExceptionCqe_ShouldReturnFalse_WhenErrorTypeIsNotException)
{
    TaskExceptionFunc& instance = TaskExceptionFunc::GetInstance();
    rtLogicCqReport_t report;
    report.errorType = 0x00;
    EXPECT_FALSE(instance.IsExceptionCqe(report));
}

extern "C" {
drvError_t halCqReportRecv(uint32_t devId, struct halReportRecvInfo* info) { return DRV_ERROR_NONE; }
}

TEST_F(TaskExceptionFuncTest, ShouldContinueOnWaitTimeout)
{
    // Mock StreamLite
    MOCKER_CPP(&RtsqBase::QuerySqBaseAddr).stubs().will(returnValue(static_cast<u64>(0)));
    MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().will(returnValue(static_cast<u32>(0)));
    std::vector<char> uniqueId{'0', '0', '0'};
    StreamLite streamLite(uniqueId);

    TaskExceptionFunc& instance = TaskExceptionFunc::GetInstance();
    instance.Register(&streamLite);
    MOCKER(halCqReportRecv)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<error_t>((int)DRV_ERROR_WAIT_TIMEOUT)));
    instance.Call();
}

TEST_F(TaskExceptionFuncTest, ShouldContinueOnNonNoneError)
{
    // Mock StreamLite
    MOCKER_CPP(&RtsqBase::QuerySqBaseAddr).stubs().will(returnValue(static_cast<u64>(0)));
    MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().will(returnValue(static_cast<u32>(0)));
    std::vector<char> uniqueId{'0', '0', '0'};
    StreamLite streamLite(uniqueId);

    TaskExceptionFunc& instance = TaskExceptionFunc::GetInstance();
    instance.Register(&streamLite);
    MOCKER(halCqReportRecv)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<error_t>((int)DRV_ERROR_NO_DEVICE)));
    instance.Call();
}
