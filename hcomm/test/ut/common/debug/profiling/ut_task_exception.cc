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
#include <stdio.h>
#include <sys/time.h>
#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include <runtime/rt_error_codes.h>
#define private public
#define protected public
#include "task_exception_handler_pub.h"
#include "heartbeat.h"
#undef protected
#undef private
#include "sal.h"
#include "externalinput.h"
#include "aicpu_operator_pub.h"
#include "alg_profiling.h"
#include "adapter_error_manager_pub.h"

using namespace std;
using namespace hccl;

extern array<map<s32, GetAicpuTaskExceptionCallBack>, MAX_MODULE_DEVICE_NUM> g_communicatorCallbackMap;
extern array<bool, MAX_MODULE_DEVICE_NUM> g_commHadCallbackArray;
extern std::mutex g_commHadCallbackArrayMutex;

class TaskExceptionErrMsgFlagTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    virtual void SetUp()
    {
        TaskExceptionHandler::errMsgFlag_.store(false);
        g_commHadCallbackArray.fill(false);
        g_communicatorCallbackMap.fill({});
    }
    virtual void TearDown()
    {
        TaskExceptionHandler::errMsgFlag_.store(false);
        g_commHadCallbackArray.fill(false);
        g_communicatorCallbackMap.fill({});
    }
};

HcclResult stub_hrtGetStreamAvailableNum(u32& maxStrCount)
{
    maxStrCount = 16;
    return HCCL_SUCCESS;
}

void stub_RptInputErr_print(std::string error_code, std::vector<std::string> key, std::vector<std::string> value)
{
    if (error_code == "EI0012") {
        printf(
            "Execution_Error_SDMA(EI0012): SDMA memory copy task exception occurred. Remote rank: [%s]. Base "
            "information: [%s]. "
            "Task information: [%s]. Communicator information: [%s].\n",
            value.size() >= 1 ? value[0].c_str() : "0",
            value.size() >= 2 ?
                value[1].c_str() :
                "streamID:[0], taskID[0], taskType:[SDMA], tag:[test_tag], AlgType(level 0-1-2):[ring-ring-ring].",
            "src:[0x1000], dst:[0x2000], size:[1024], op:[ALLREDUCE], data type:[FLOAT32], link type:[OnChip], remote "
            "rank:[0]",
            "group:[testGroup], user define information[testUdi], rankSize[8], rankId[0 0]");
        printf("        Possible cause: The DMA data transfer fails due to a hardware fault or link error.\n"
               "        Solution: 1. Check whether the network link is abnormal during the execution.\n"
               "2. Check whether a process in the cluster exits before an error is reported. If yes, locate the cause "
               "of the process exit.\n"
               "3. Check whether the input/output memory size is correct and whether the memory or communicator is "
               "released prematurely.\n");
    }
    fflush(stdout);
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_ErrMsgFlag_InitialValue_IsFalse)
{
    EXPECT_FALSE(TaskExceptionHandler::errMsgFlag_.load());
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_ErrMsgFlag_Exchange_FirstCallReturnsFalse)
{
    bool oldValue = TaskExceptionHandler::errMsgFlag_.exchange(true);
    EXPECT_FALSE(oldValue);
    EXPECT_TRUE(TaskExceptionHandler::errMsgFlag_.load());
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_ErrMsgFlag_Exchange_SecondCallReturnsTrue)
{
    TaskExceptionHandler::errMsgFlag_.exchange(true);
    bool oldValue = TaskExceptionHandler::errMsgFlag_.exchange(true);
    EXPECT_TRUE(oldValue);
    EXPECT_TRUE(TaskExceptionHandler::errMsgFlag_.load());
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_ErrMsgFlag_Exchange_OnlyFirstCallEntersBlock)
{
    int enterCount = 0;
    if (!TaskExceptionHandler::errMsgFlag_.exchange(true)) {
        enterCount++;
    }
    if (!TaskExceptionHandler::errMsgFlag_.exchange(true)) {
        enterCount++;
    }
    if (!TaskExceptionHandler::errMsgFlag_.exchange(true)) {
        enterCount++;
    }
    EXPECT_EQ(enterCount, 1);
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_PrintAicpuErrorMessage_NotifyWait_ReportsEI0002)
{
    u32 deviceId = 0;
    s32 streamId = 100;

    auto callback = [=]() -> ErrorMessageReport {
        ErrorMessageReport msg;
        strncpy(msg.tag, "test_tag_notify_wait", TAG_MAX_LENGTH - 1);
        msg.taskType = TaskType::TASK_NOTIFY_WAIT;
        msg.remoteUserRank = 1;
        msg.streamId = streamId;
        return msg;
    };

    RegisterGetAicpuTaskExceptionCallBack(streamId, deviceId, callback);

    rtExceptionInfo exceptionInfo;
    exceptionInfo.deviceid = deviceId;
    exceptionInfo.streamid = streamId;
    exceptionInfo.taskid = 0;
    exceptionInfo.retcode = 0;
    memset(&exceptionInfo.expandInfo, 0, sizeof(exceptionInfo.expandInfo));

    bool isExistAicpuError = false;

    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    TaskExceptionHandler::PrintAicpuErrorMessage(&exceptionInfo, isExistAicpuError);

    EXPECT_TRUE(isExistAicpuError);
    EXPECT_TRUE(TaskExceptionHandler::errMsgFlag_.load());

    GlobalMockObject::verify();
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_PrintAicpuErrorMessage_Sdma_ReportsEI0012)
{
    u32 deviceId = 0;
    s32 streamId = 101;

    auto callback = [=]() -> ErrorMessageReport {
        ErrorMessageReport msg;
        strncpy(msg.tag, "test_tag_sdma", TAG_MAX_LENGTH - 1);
        msg.taskType = TaskType::TASK_SDMA;
        msg.remoteUserRank = 2;
        msg.streamId = streamId;
        return msg;
    };

    RegisterGetAicpuTaskExceptionCallBack(streamId, deviceId, callback);

    rtExceptionInfo exceptionInfo;
    exceptionInfo.deviceid = deviceId;
    exceptionInfo.streamid = streamId;
    exceptionInfo.taskid = 0;
    exceptionInfo.retcode = 0;
    memset(&exceptionInfo.expandInfo, 0, sizeof(exceptionInfo.expandInfo));

    bool isExistAicpuError = false;

    MOCKER(RptInputErr).stubs().will(invoke(stub_RptInputErr_print));

    TaskExceptionHandler::PrintAicpuErrorMessage(&exceptionInfo, isExistAicpuError);

    EXPECT_TRUE(isExistAicpuError);
    EXPECT_TRUE(TaskExceptionHandler::errMsgFlag_.load());

    GlobalMockObject::verify();
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_PrintAicpuErrorMessage_ErrMsgFlagTrue_NoReport)
{
    TaskExceptionHandler::errMsgFlag_.store(true);

    u32 deviceId = 0;
    s32 streamId = 103;

    auto callback = [=]() -> ErrorMessageReport {
        ErrorMessageReport msg;
        strncpy(msg.tag, "test_tag_notify_wait", TAG_MAX_LENGTH - 1);
        msg.taskType = TaskType::TASK_NOTIFY_WAIT;
        msg.remoteUserRank = 1;
        msg.streamId = streamId;
        return msg;
    };

    RegisterGetAicpuTaskExceptionCallBack(streamId, deviceId, callback);

    rtExceptionInfo exceptionInfo;
    exceptionInfo.deviceid = deviceId;
    exceptionInfo.streamid = streamId;
    exceptionInfo.taskid = 0;
    exceptionInfo.retcode = 0;
    memset(&exceptionInfo.expandInfo, 0, sizeof(exceptionInfo.expandInfo));

    bool isExistAicpuError = false;

    MOCKER(RptInputErr).expects(never());

    TaskExceptionHandler::PrintAicpuErrorMessage(&exceptionInfo, isExistAicpuError);

    EXPECT_TRUE(isExistAicpuError);

    GlobalMockObject::verify();
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_PrintAicpuErrorMessage_AlreadyReported_ReturnsEarly)
{
    u32 deviceId = 0;
    g_commHadCallbackArray[deviceId] = true;

    s32 streamId = 104;
    auto callback = []() -> ErrorMessageReport {
        ErrorMessageReport msg;
        strncpy(msg.tag, "test_tag", TAG_MAX_LENGTH - 1);
        return msg;
    };
    RegisterGetAicpuTaskExceptionCallBack(streamId, deviceId, callback);

    rtExceptionInfo exceptionInfo;
    exceptionInfo.deviceid = deviceId;
    exceptionInfo.streamid = streamId;
    exceptionInfo.taskid = 0;
    exceptionInfo.retcode = 0;
    memset(&exceptionInfo.expandInfo, 0, sizeof(exceptionInfo.expandInfo));

    bool isExistAicpuError = false;

    TaskExceptionHandler::PrintAicpuErrorMessage(&exceptionInfo, isExistAicpuError);

    EXPECT_FALSE(isExistAicpuError);
    EXPECT_FALSE(TaskExceptionHandler::errMsgFlag_.load());
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_PrintAicpuErrorMessage_NoCallback_NoError)
{
    u32 deviceId = 0;
    s32 streamId = 999;

    rtExceptionInfo exceptionInfo;
    exceptionInfo.deviceid = deviceId;
    exceptionInfo.streamid = streamId;
    exceptionInfo.taskid = 0;
    exceptionInfo.retcode = 0;
    memset(&exceptionInfo.expandInfo, 0, sizeof(exceptionInfo.expandInfo));

    bool isExistAicpuError = false;

    TaskExceptionHandler::PrintAicpuErrorMessage(&exceptionInfo, isExistAicpuError);

    EXPECT_FALSE(isExistAicpuError);
    EXPECT_FALSE(TaskExceptionHandler::errMsgFlag_.load());
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_PrintAicpuErrorMessage_EmptyTag_NoError)
{
    u32 deviceId = 0;
    s32 streamId = 105;

    auto callback = []() -> ErrorMessageReport {
        ErrorMessageReport msg;
        msg.tag[0] = '\0';
        msg.taskType = TaskType::TASK_NOTIFY_WAIT;
        return msg;
    };

    RegisterGetAicpuTaskExceptionCallBack(streamId, deviceId, callback);

    rtExceptionInfo exceptionInfo;
    exceptionInfo.deviceid = deviceId;
    exceptionInfo.streamid = streamId;
    exceptionInfo.taskid = 0;
    exceptionInfo.retcode = 0;
    memset(&exceptionInfo.expandInfo, 0, sizeof(exceptionInfo.expandInfo));

    bool isExistAicpuError = false;

    TaskExceptionHandler::PrintAicpuErrorMessage(&exceptionInfo, isExistAicpuError);

    EXPECT_FALSE(isExistAicpuError);
    EXPECT_FALSE(TaskExceptionHandler::errMsgFlag_.load());
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_PrintAicpuErrorMessage_SecondCall_ErrMsgFlagBlocksReport)
{
    u32 deviceId = 0;
    s32 streamId1 = 110;
    s32 streamId2 = 111;

    auto callback1 = [=]() -> ErrorMessageReport {
        ErrorMessageReport msg;
        strncpy(msg.tag, "test_tag_notify_wait", TAG_MAX_LENGTH - 1);
        msg.taskType = TaskType::TASK_NOTIFY_WAIT;
        msg.remoteUserRank = 1;
        msg.streamId = streamId1;
        return msg;
    };

    auto callback2 = [=]() -> ErrorMessageReport {
        ErrorMessageReport msg;
        strncpy(msg.tag, "test_tag_sdma", TAG_MAX_LENGTH - 1);
        msg.taskType = TaskType::TASK_SDMA;
        msg.remoteUserRank = 2;
        msg.streamId = streamId2;
        return msg;
    };

    RegisterGetAicpuTaskExceptionCallBack(streamId1, deviceId, callback1);
    RegisterGetAicpuTaskExceptionCallBack(streamId2, deviceId, callback2);

    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    rtExceptionInfo exceptionInfo1;
    exceptionInfo1.deviceid = deviceId;
    exceptionInfo1.streamid = streamId1;
    exceptionInfo1.taskid = 0;
    exceptionInfo1.retcode = 0;
    memset(&exceptionInfo1.expandInfo, 0, sizeof(exceptionInfo1.expandInfo));

    bool isExistAicpuError1 = false;
    TaskExceptionHandler::PrintAicpuErrorMessage(&exceptionInfo1, isExistAicpuError1);
    EXPECT_TRUE(isExistAicpuError1);
    EXPECT_TRUE(TaskExceptionHandler::errMsgFlag_.load());

    g_commHadCallbackArray[deviceId] = false;

    rtExceptionInfo exceptionInfo2;
    exceptionInfo2.deviceid = deviceId;
    exceptionInfo2.streamid = streamId2;
    exceptionInfo2.taskid = 0;
    exceptionInfo2.retcode = 0;
    memset(&exceptionInfo2.expandInfo, 0, sizeof(exceptionInfo2.expandInfo));

    bool isExistAicpuError2 = false;
    TaskExceptionHandler::PrintAicpuErrorMessage(&exceptionInfo2, isExistAicpuError2);
    EXPECT_TRUE(isExistAicpuError2);

    GlobalMockObject::verify();
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_DealExceptionTask_NotifyWait_SetsErrMsgFlag)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    MOCKER(hrtGetStreamAvailableNum).stubs().will(invoke(stub_hrtGetStreamAvailableNum));

    HcclResult ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_notify_wait_tag";
    TaskType taskType = TaskType::TASK_NOTIFY_WAIT;
    TaskParaNotify taskParaNotify;
    AlgType algType = AlgType::Reserved();

    u32 index = 0;
    TaskInfo taskInfo(streamID, taskID, tag, taskType, algType, index, taskParaNotify);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_FALSE(TaskExceptionHandler::errMsgFlag_.load());

    rtExceptionInfo exceptionInfo;
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = streamID;
    exceptionInfo.taskid = taskID;
    exceptionInfo.retcode = 0;
    memset(&exceptionInfo.expandInfo, 0, sizeof(exceptionInfo.expandInfo));

    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    bool result = TaskExceptionHandler::DealExceptionTask(&exceptionInfo);
    EXPECT_TRUE(result);
    EXPECT_TRUE(TaskExceptionHandler::errMsgFlag_.load());

    taskExceptionHandler.Flush();
    GlobalMockObject::verify();
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_DealExceptionTask_Sdma_SetsErrMsgFlag)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    MOCKER(hrtGetStreamAvailableNum).stubs().will(invoke(stub_hrtGetStreamAvailableNum));

    HcclResult ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 streamID = 0;
    u32 taskID = 1;
    std::string tag = "test_sdma_tag";
    TaskType taskType = TaskType::TASK_SDMA;
    TaskParaDMA taskParaDMA;
    AlgType algType = AlgType::Reserved();

    u32 index = 0;
    TaskInfo taskInfo(streamID, taskID, tag, taskType, algType, index, taskParaDMA);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_FALSE(TaskExceptionHandler::errMsgFlag_.load());

    rtExceptionInfo exceptionInfo;
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = streamID;
    exceptionInfo.taskid = taskID;
    exceptionInfo.retcode = 0;
    memset(&exceptionInfo.expandInfo, 0, sizeof(exceptionInfo.expandInfo));

    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    bool result = TaskExceptionHandler::DealExceptionTask(&exceptionInfo);
    EXPECT_TRUE(result);
    EXPECT_TRUE(TaskExceptionHandler::errMsgFlag_.load());

    taskExceptionHandler.Flush();
    GlobalMockObject::verify();
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_DealExceptionTask_ErrMsgFlagTrue_NoReport)
{
    TaskExceptionHandler::errMsgFlag_.store(true);

    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    MOCKER(hrtGetStreamAvailableNum).stubs().will(invoke(stub_hrtGetStreamAvailableNum));

    HcclResult ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 streamID = 0;
    u32 taskID = 2;
    std::string tag = "test_notify_wait_tag_2";
    TaskType taskType = TaskType::TASK_NOTIFY_WAIT;
    TaskParaNotify taskParaNotify;
    AlgType algType = AlgType::Reserved();

    u32 index = 0;
    TaskInfo taskInfo(streamID, taskID, tag, taskType, algType, index, taskParaNotify);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rtExceptionInfo exceptionInfo;
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = streamID;
    exceptionInfo.taskid = taskID;
    exceptionInfo.retcode = 0;
    memset(&exceptionInfo.expandInfo, 0, sizeof(exceptionInfo.expandInfo));

    MOCKER(RptInputErr).expects(never());

    bool result = TaskExceptionHandler::DealExceptionTask(&exceptionInfo);
    EXPECT_TRUE(result);
    EXPECT_TRUE(TaskExceptionHandler::errMsgFlag_.load());

    taskExceptionHandler.Flush();
    GlobalMockObject::verify();
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_DealExceptionTask_ReduceInline_SetsErrMsgFlag)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    MOCKER(hrtGetStreamAvailableNum).stubs().will(invoke(stub_hrtGetStreamAvailableNum));

    HcclResult ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 streamID = 0;
    u32 taskID = 3;
    std::string tag = "test_reduce_inline_tag";
    TaskType taskType = TaskType::TASK_REDUCE_INLINE;
    TaskParaReduce taskParaReduce;
    AlgType algType = AlgType::Reserved();

    u32 index = 0;
    TaskInfo taskInfo(streamID, taskID, tag, taskType, algType, index, taskParaReduce);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_FALSE(TaskExceptionHandler::errMsgFlag_.load());

    rtExceptionInfo exceptionInfo;
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = streamID;
    exceptionInfo.taskid = taskID;
    exceptionInfo.retcode = 0;
    memset(&exceptionInfo.expandInfo, 0, sizeof(exceptionInfo.expandInfo));

    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    bool result = TaskExceptionHandler::DealExceptionTask(&exceptionInfo);
    EXPECT_TRUE(result);
    EXPECT_TRUE(TaskExceptionHandler::errMsgFlag_.load());

    taskExceptionHandler.Flush();
    GlobalMockObject::verify();
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_DealExceptionTask_SecondCall_ErrMsgFlagBlocksReport)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    MOCKER(hrtGetStreamAvailableNum).stubs().will(invoke(stub_hrtGetStreamAvailableNum));

    HcclResult ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 streamID = 0;
    u32 taskID1 = 10;
    u32 taskID2 = 11;
    std::string tag1 = "test_notify_wait_first";
    std::string tag2 = "test_sdma_second";
    TaskParaNotify taskParaNotify;
    TaskParaDMA taskParaDMA;
    AlgType algType = AlgType::Reserved();

    u32 index = 0;
    TaskType taskType1 = TaskType::TASK_NOTIFY_WAIT;
    TaskInfo taskInfo1(streamID, taskID1, tag1, taskType1, algType, index, taskParaNotify);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    TaskType taskType2 = TaskType::TASK_SDMA;
    TaskInfo taskInfo2(streamID, taskID2, tag2, taskType2, algType, index, taskParaDMA);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    rtExceptionInfo exceptionInfo1;
    exceptionInfo1.deviceid = 0;
    exceptionInfo1.streamid = streamID;
    exceptionInfo1.taskid = taskID1;
    exceptionInfo1.retcode = 0;
    memset(&exceptionInfo1.expandInfo, 0, sizeof(exceptionInfo1.expandInfo));

    bool result1 = TaskExceptionHandler::DealExceptionTask(&exceptionInfo1);
    EXPECT_TRUE(result1);
    EXPECT_TRUE(TaskExceptionHandler::errMsgFlag_.load());

    rtExceptionInfo exceptionInfo2;
    exceptionInfo2.deviceid = 0;
    exceptionInfo2.streamid = streamID;
    exceptionInfo2.taskid = taskID2;
    exceptionInfo2.retcode = 0;
    memset(&exceptionInfo2.expandInfo, 0, sizeof(exceptionInfo2.expandInfo));

    bool result2 = TaskExceptionHandler::DealExceptionTask(&exceptionInfo2);
    EXPECT_FALSE(result2);

    taskExceptionHandler.Flush();
    GlobalMockObject::verify();
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_DealExceptionCtx_NotifyWait_SetsErrMsgFlag)
{
    MOCKER(GetExternalInputTaskExceptionSwitch).stubs().will(returnValue(1));
    MOCKER(hrtGetStreamAvailableNum).stubs().will(invoke(stub_hrtGetStreamAvailableNum));

    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_ctx_notify_wait";
    TaskParaNotify taskParaNotify;
    TaskType taskType = TaskType::TASK_NOTIFY_WAIT;

    ret = taskExceptionHandler.Save(streamID, taskID, taskType, taskParaNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    TaskParaDMA taskParaDMA;
    taskType = TaskType::TASK_RDMA;
    ret = taskExceptionHandler.Save(streamID, taskID, taskType, taskParaDMA);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = taskExceptionHandler.Save(streamID, taskID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_FALSE(TaskExceptionHandler::errMsgFlag_.load());

    rtExceptionExpandInfo_t expandInfo;
    expandInfo.u.fftsPlusInfo.contextId = 0;
    expandInfo.type = tagRtExceptionExpandType::RT_EXCEPTION_FFTS_PLUS;
    rtExceptionInfo exceptionInfo;
    exceptionInfo.streamid = streamID;
    exceptionInfo.expandInfo = expandInfo;
    exceptionInfo.deviceid = 0;
    exceptionInfo.taskid = taskID;

    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    TaskExceptionHandler::DealExceptionCtx(&exceptionInfo);
    EXPECT_TRUE(TaskExceptionHandler::errMsgFlag_.load());

    taskExceptionHandler.Flush();
    GlobalMockObject::verify();
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_DealExceptionCtx_ErrMsgFlagTrue_NoReport)
{
    TaskExceptionHandler::errMsgFlag_.store(true);

    MOCKER(GetExternalInputTaskExceptionSwitch).stubs().will(returnValue(1));
    MOCKER(hrtGetStreamAvailableNum).stubs().will(invoke(stub_hrtGetStreamAvailableNum));

    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_ctx_notify_wait_blocked";
    TaskParaNotify taskParaNotify;
    TaskType taskType = TaskType::TASK_NOTIFY_WAIT;

    ret = taskExceptionHandler.Save(streamID, taskID, taskType, taskParaNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    TaskParaDMA taskParaDMA;
    taskType = TaskType::TASK_RDMA;
    ret = taskExceptionHandler.Save(streamID, taskID, taskType, taskParaDMA);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = taskExceptionHandler.Save(streamID, taskID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rtExceptionExpandInfo_t expandInfo;
    expandInfo.u.fftsPlusInfo.contextId = 0;
    expandInfo.type = tagRtExceptionExpandType::RT_EXCEPTION_FFTS_PLUS;
    rtExceptionInfo exceptionInfo;
    exceptionInfo.streamid = streamID;
    exceptionInfo.expandInfo = expandInfo;
    exceptionInfo.deviceid = 0;
    exceptionInfo.taskid = taskID;

    MOCKER(RptInputErr).expects(never());

    TaskExceptionHandler::DealExceptionCtx(&exceptionInfo);

    taskExceptionHandler.Flush();
    GlobalMockObject::verify();
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_DealExceptionOp_Timeout_SetsErrMsgFlag)
{
    MOCKER(hrtGetStreamAvailableNum).stubs().will(invoke(stub_hrtGetStreamAvailableNum));

    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_op_timeout";
    AlgType algType = AlgType::Reserved();

    std::vector<uint32_t> descData = {32, 1};
    size_t descBufLen = 128;
    ret = taskExceptionHandler.Save(streamID, taskID, descData.data(), descBufLen);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_FALSE(TaskExceptionHandler::errMsgFlag_.load());

    rtExceptionInfo exceptionInfo;
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = streamID;
    exceptionInfo.taskid = taskID;
    exceptionInfo.retcode = ACL_ERROR_RT_FFTS_PLUS_TIMEOUT;
    memset(&exceptionInfo.expandInfo, 0, sizeof(exceptionInfo.expandInfo));

    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    bool result = TaskExceptionHandler::DealExceptionOp(&exceptionInfo);
    EXPECT_TRUE(result);
    EXPECT_TRUE(TaskExceptionHandler::errMsgFlag_.load());

    taskExceptionHandler.Flush();
    GlobalMockObject::verify();
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_DealExceptionOp_NonTimeout_NoReportButSetsFlag)
{
    MOCKER(hrtGetStreamAvailableNum).stubs().will(invoke(stub_hrtGetStreamAvailableNum));

    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_op_non_timeout";
    AlgType algType = AlgType::Reserved();

    std::vector<uint32_t> descData = {32, 1};
    size_t descBufLen = 128;
    ret = taskExceptionHandler.Save(streamID, taskID, descData.data(), descBufLen);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_FALSE(TaskExceptionHandler::errMsgFlag_.load());

    rtExceptionInfo exceptionInfo;
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = streamID;
    exceptionInfo.taskid = taskID;
    exceptionInfo.retcode = 0;
    memset(&exceptionInfo.expandInfo, 0, sizeof(exceptionInfo.expandInfo));

    MOCKER(RptInputErr).expects(never());

    bool result = TaskExceptionHandler::DealExceptionOp(&exceptionInfo);
    EXPECT_TRUE(result);
    EXPECT_TRUE(TaskExceptionHandler::errMsgFlag_.load());

    taskExceptionHandler.Flush();
    GlobalMockObject::verify();
}

TEST_F(TaskExceptionErrMsgFlagTest, Ut_DealExceptionOp_ErrMsgFlagTrue_NoReport)
{
    TaskExceptionHandler::errMsgFlag_.store(true);

    MOCKER(hrtGetStreamAvailableNum).stubs().will(invoke(stub_hrtGetStreamAvailableNum));

    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_op_blocked";
    AlgType algType = AlgType::Reserved();

    std::vector<uint32_t> descData = {32, 1};
    size_t descBufLen = 128;
    ret = taskExceptionHandler.Save(streamID, taskID, descData.data(), descBufLen);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rtExceptionInfo exceptionInfo;
    exceptionInfo.deviceid = 0;
    exceptionInfo.streamid = streamID;
    exceptionInfo.taskid = taskID;
    exceptionInfo.retcode = ACL_ERROR_RT_FFTS_PLUS_TIMEOUT;
    memset(&exceptionInfo.expandInfo, 0, sizeof(exceptionInfo.expandInfo));

    MOCKER(RptInputErr).expects(never());

    bool result = TaskExceptionHandler::DealExceptionOp(&exceptionInfo);
    EXPECT_TRUE(result);

    taskExceptionHandler.Flush();
    GlobalMockObject::verify();
}
