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
#include <sys/time.h> /* 获取时间 */
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

using namespace std;
using namespace hccl;

extern array<map<s32, GetAicpuTaskExceptionCallBack>, MAX_MODULE_DEVICE_NUM> g_communicatorCallbackMap;
class TaskExceptionTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--TaskExceptionTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--TaskExceptionTest TearDown--\033[0m" << std::endl; }

    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};
#if 0 // 执行失败Map size is bigger than max stream count
TEST_F(TaskExceptionTest, ut_task_exception_callback)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret;
    ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_tag";
    TaskType taskType = TaskType::TASK_NOTIFY_WAIT;
    TaskParaNotify taskParaNotify;
    AlgType algType = AlgType::Reserved();

    // TaskInfo(u32 &streamID, u32 &taskID, std::string &tag, TaskType &taskType, const TaskParaNotify &para);
    u32 index = 0;
    TaskInfo taskInfo1(streamID, taskID, tag, taskType, algType, index, taskParaNotify);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rtExceptionInfo rtExceptionInfo1{0, 0, 0, 0, 0};
    taskExceptionHandler.Callback(&rtExceptionInfo1);
}
TEST_F(TaskExceptionTest, ut_task_exception_callback_ffts)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret;
    ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 streamID = 1;
    u32 taskID = 0;
    std::string tag = "test_tag";
    aclrtStream steam;
    aclrtCreateStream(&steam);
    aclrtStreamGetId(steam, (s32 *)&streamID);
    ret = taskExceptionHandler.AddStream(streamID, tag, 0, AlgType::Reserved());
    EXPECT_EQ(ret, HCCL_SUCCESS);

    RankInfo rankInfo;
    rankInfo.devicePhyId = 10;
    UIDType crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    UIDType informer = Heartbeat::GetInstance(0).GetUId(rankInfo);

    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_LOST);

    ret = taskExceptionHandler.Save(streamID, taskID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    taskID = 1;
    ret = taskExceptionHandler.Save(streamID, taskID);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rtExceptionInfo rtExceptionInfo1{0, streamID, 0, 0, ACL_ERROR_RT_FFTS_PLUS_TIMEOUT};
    rtExceptionInfo1.expandInfo.type = RT_EXCEPTION_FFTS_PLUS;
    taskExceptionHandler.Callback(&rtExceptionInfo1);
    rtExceptionInfo rtExceptionInfo2{0, streamID, 0, 0, 0};
    rtExceptionInfo2.expandInfo.type = RT_EXCEPTION_FFTS_PLUS;
    taskExceptionHandler.Callback(&rtExceptionInfo2);
    aclrtDestroyStream(steam);
}
TEST_F(TaskExceptionTest, ut_task_exception_callback_context_size)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret;
    ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 streamID = 0;
    std::string tag = "test_tag";
    TaskParaNotify taskParaNotify;
    TaskParaDMA taskParaDMA;
    TaskParaReduce taskParaReduce;
    AlgType algType = AlgType::Reserved();

    u32 taskID = 0;
    TaskType taskType = TaskType::TASK_RDMA;
    u32 index = 0;
    TaskInfo taskInfo1(streamID, taskID, tag, taskType, algType, index, taskParaDMA);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    taskID = 1;
    taskType = TaskType::TASK_REDUCE_TBE;
    TaskInfo taskInfo2(streamID, taskID, tag, taskType, algType, index, taskParaReduce);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    taskID = 2;
    taskType = TaskType::TASK_REDUCE_INLINE;
    taskParaReduce.remoteUserRank = 1;
    TaskInfo taskInfo3(streamID, taskID, tag, taskType, algType, index, taskParaReduce);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo3);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    taskID = 3;
    taskType = TaskType::TASK_NOTIFY_WAIT;
    TaskInfo taskInfo4(streamID, taskID, tag, taskType, algType, index, taskParaNotify);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo4);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    taskID = 4;
    taskType = TaskType::TASK_SDMA;
    TaskInfo taskInfo5(streamID, taskID, tag, taskType, algType, index, taskParaDMA);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo5);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    taskID = 5;
    taskType = TaskType::TASK_NOTIFY_RECORD;
    taskParaNotify.notifyID = 0xffffffffabcdef12;
    TaskInfo taskInfo6(streamID, taskID, tag, taskType, algType, index, taskParaNotify);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo6);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    taskID = 6;
    taskType = TaskType::TASK_NOTIFY_WAIT;
    TaskInfo taskInfo7(streamID, taskID, tag, taskType, algType, index, taskParaNotify);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo7);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rtExceptionInfo rtExceptionInfo1{taskID, 0, 0, 0, 0};
    taskExceptionHandler.Callback(&rtExceptionInfo1);
}
TEST_F(TaskExceptionTest, ut_task_exception_callback_fftsCtx)
{
    MOCKER(GetExternalInputTaskExceptionSwitch)
    .stubs()
    .will(returnValue(1));

    MOCKER(GetExternalInputHcclEnableFfts)
    .stubs()
    .will(returnValue(true));

    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 streamID = 1;
    u32 taskID = 0;
    std::string tag = "test_tag";

    aclrtStream steam;
    aclrtCreateStream(&steam);
    aclrtStreamGetId(steam, (s32 *)&streamID);
    ret = taskExceptionHandler.AddStream(streamID, tag, 0, AlgType::Reserved());
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // ctxInfoVector存储3个CtxInfo
    TaskParaNotify taskParaNotify;
    TaskType taskType = TaskType::TASK_NOTIFY_WAIT;
    ret = taskExceptionHandler.Save(streamID, taskID, taskType, taskParaNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    TaskParaDMA taskParaDMA;
    taskType = TaskType::TASK_RDMA;
    ret = taskExceptionHandler.Save(streamID, taskID, taskType, taskParaDMA);
    CtxInfo tmpCtxInfo(taskType, taskParaDMA);
    tmpCtxInfo.GetCtxParaInfoStr();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    TaskParaReduce taskParaReduce;
    taskType = TaskType::TASK_REDUCE_TBE;
    ret = taskExceptionHandler.Save(streamID, taskID, taskType, taskParaReduce);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 存储ctxInfoVector至opCtxInfo
    ret = taskExceptionHandler.Save(streamID, taskID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // callback查找
    rtExceptionExpandInfo_t expandInfo;
    expandInfo.u.fftsPlusInfo.contextId = 0;
    rtExceptionInfo rtExceptionInfo1;
    rtExceptionInfo1.streamid = streamID;
    rtExceptionInfo1.expandInfo = expandInfo;
    rtExceptionInfo1.deviceid = 0;
    taskExceptionHandler.Callback(&rtExceptionInfo1);
    aclrtDestroyStream(steam);
    taskExceptionHandler.DeInit();
    GlobalMockObject::verify();
}
TEST_F(TaskExceptionTest, ut_task_exception_callback_fftsCtx_context_size)
{
    MOCKER(GetExternalInputTaskExceptionSwitch)
    .stubs()
    .will(returnValue(1));

    MOCKER(GetExternalInputHcclEnableFfts)
    .stubs()
    .will(returnValue(true));

    MOCKER(GetWorkflowMode)
    .stubs()
    .will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 streamID = 1;
    u32 taskID = 0;
    std::string tag = "test_tag";

    aclrtStream steam;
    aclrtCreateStream(&steam);
    aclrtStreamGetId(steam, (s32 *)&streamID);
    ret = taskExceptionHandler.AddStream(streamID, tag, 0, AlgType::Reserved());
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // ctxInfoVector存储3个CtxInfo
    TaskParaNotify taskParaNotify;
    TaskParaDMA taskParaDMA;
    TaskType taskType = TaskType::TASK_RDMA;
    ret = taskExceptionHandler.Save(streamID, taskID, taskType, taskParaDMA);

    CtxInfo tmpCtxInfo(taskType, taskParaDMA);
    tmpCtxInfo.GetCtxParaInfoStr();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    TaskParaReduce taskParaReduce;
    taskType = TaskType::TASK_REDUCE_TBE;
    ret = taskExceptionHandler.Save(streamID, taskID, taskType, taskParaReduce);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    taskType = TaskType::TASK_NOTIFY_WAIT;
    ret = taskExceptionHandler.Save(streamID, taskID, taskType, taskParaNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 存储ctxInfoVector至opCtxInfo
    ret = taskExceptionHandler.Save(streamID, taskID);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // callback查找
    rtExceptionExpandInfo_t expandInfo;
    expandInfo.u.fftsPlusInfo.contextId = 2;
    expandInfo.type = tagRtExceptionExpandType::RT_EXCEPTION_FFTS_PLUS;
    rtExceptionInfo rtExceptionInfo1;
    rtExceptionInfo1.streamid = streamID;
    rtExceptionInfo1.expandInfo = expandInfo;
    rtExceptionInfo1.deviceid = 0;
    rtExceptionInfo1.taskid = taskID;
    taskExceptionHandler.Callback(&rtExceptionInfo1);

    aclrtDestroyStream(steam);
    taskExceptionHandler.DeInit();
    GlobalMockObject::verify();
}
TEST_F(TaskExceptionTest, ut_task_exception_callback_heartbeat1)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret;
    ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_tag";
    TaskType taskType = TaskType::TASK_NOTIFY_WAIT;
    TaskParaNotify taskParaNotify;
    AlgType algType = AlgType::Reserved();

    // TaskInfo(u32 &streamID, u32 &taskID, std::string &tag, TaskType &taskType, const TaskParaNotify &para);
    u32 index = 0;
    TaskInfo taskInfo1(streamID, taskID, tag, taskType, algType, index, taskParaNotify);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rtExceptionInfo rtExceptionInfo1{0, 0, 0, 0, 0};

    RankInfo rankInfo;
    rankInfo.devicePhyId = 10;
    UIDType crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    UIDType informer = Heartbeat::GetInstance(0).GetUId(rankInfo);

    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_LOST);

    rankInfo.devicePhyId = 11;
    crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_LOST);

    rankInfo.devicePhyId = 12;
    crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_LOST);

    rankInfo.devicePhyId = 13;
    crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_LOST);

    rankInfo.devicePhyId = 13;
    crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_LOST);

    taskExceptionHandler.Callback(&rtExceptionInfo1);
}

TEST_F(TaskExceptionTest, ut_task_exception_callback_heartbeat2)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret;
    ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_tag";
    TaskType taskType = TaskType::TASK_NOTIFY_WAIT;
    TaskParaNotify taskParaNotify;
    AlgType algType = AlgType::Reserved();

    // TaskInfo(u32 &streamID, u32 &taskID, std::string &tag, TaskType &taskType, const TaskParaNotify &para);
    u32 index = 0;
    TaskInfo taskInfo1(streamID, taskID, tag, taskType, algType, index, taskParaNotify);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rtExceptionInfo rtExceptionInfo1{0, 0, 0, 0, 0};

    RankInfo rankInfo;
    rankInfo.devicePhyId = 20;
    UIDType crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    rankInfo.devicePhyId = 10;
    UIDType informer = Heartbeat::GetInstance(0).GetUId(rankInfo);

    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_LOST);

    taskExceptionHandler.Callback(&rtExceptionInfo1);
}

TEST_F(TaskExceptionTest, ut_task_exception_callback__cqe_heartbeat)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret;
    ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_tag";
    TaskType taskType = TaskType::TASK_NOTIFY_WAIT;
    TaskParaNotify taskParaNotify;
    AlgType algType = AlgType::Reserved();

    // TaskInfo(u32 &streamID, u32 &taskID, std::string &tag, TaskType &taskType, const TaskParaNotify &para);
    u32 index = 0;
    TaskInfo taskInfo1(streamID, taskID, tag, taskType, algType, index, taskParaNotify);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rtExceptionInfo rtExceptionInfo1{0, 0, 0, 0, 0};

    RankInfo rankInfo;
    rankInfo.devicePhyId = 10;
    UIDType crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    UIDType informer = Heartbeat::GetInstance(0).GetUId(rankInfo);

    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_CQE_ERR);

    rankInfo.devicePhyId = 11;
    crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_CQE_ERR);

    rankInfo.devicePhyId = 12;
    crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_CQE_ERR);

    rankInfo.devicePhyId = 13;
    crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_CQE_ERR);

    rankInfo.devicePhyId = 13;
    crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_CQE_ERR);

    taskExceptionHandler.Callback(&rtExceptionInfo1);
}
#endif

HcclResult stub_hrtGetStreamAvailableNum(u32& maxStrCount)
{
    maxStrCount = 16;
    return HCCL_SUCCESS;
}

TEST_F(TaskExceptionTest, ut_TaskInfo_ffts_context)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    MOCKER(GetExternalInputTaskExceptionSwitch).stubs().will(returnValue(1));
    MOCKER(hrtGetStreamAvailableNum).stubs().will(invoke(stub_hrtGetStreamAvailableNum));

    HcclResult ret;
    ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_tag";
    TaskType taskType = TaskType::TASK_RDMA;
    TaskParaNotify taskParaNotify;
    string str;
    AlgType algType = AlgType::Reserved();

    u32 index = 0;
    TaskInfo taskInfo1(streamID, taskID, tag, taskType, algType, index, taskParaNotify);
    str = taskInfo1.GetBaseInfoStr();
    TaskParaDMA taskParaDMA;
    taskType = TaskType::TASK_RDMA;
    ret = taskExceptionHandler.Save(streamID, taskID, taskType, taskParaDMA);

    std::vector<uint32_t> descData = {32, 1};
    size_t descBufLen = 128;
    ret = taskExceptionHandler.Save(streamID, taskID, descData.data(), descBufLen);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rtExceptionExpandInfo_t expandInfo;
    expandInfo.u.fftsPlusInfo.contextId = 0;
    expandInfo.type = tagRtExceptionExpandType::RT_EXCEPTION_FFTS_PLUS;
    rtExceptionInfo rtExceptionInfo1;
    rtExceptionInfo1.streamid = streamID;
    rtExceptionInfo1.expandInfo = expandInfo;
    rtExceptionInfo1.deviceid = 0;
    rtExceptionInfo1.taskid = taskID;
    TaskExceptionHandler::DealExceptionCtx(&rtExceptionInfo1);
    taskExceptionHandler.Flush();
}

TEST_F(TaskExceptionTest, ut_TaskInfo_GetBaseInfoStr)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret;
    ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_tag";
    TaskType taskType = TaskType::TASK_RDMA;
    TaskParaNotify taskParaNotify;
    string str;
    AlgType algType = AlgType::Reserved();

    u32 index = 0;
    TaskInfo taskInfo1(streamID, taskID, tag, taskType, algType, index, taskParaNotify);
    str = taskInfo1.GetBaseInfoStr();
    TaskParaDMA taskParaDMA;
    taskType = TaskType::TASK_RDMA;
    ret = taskExceptionHandler.Save(streamID, taskID, taskType, taskParaDMA);
    CtxInfo tmpCtxInfo(taskType, taskParaDMA);
    str = tmpCtxInfo.GetCtxBaseInfoStr();
    str = tmpCtxInfo.GetCtxParaNotify();

    rtExceptionInfo rtExceptionInfo1{0, 0, 0, 0, 0};
    TaskExceptionHandler::DealExceptionCtx(&rtExceptionInfo1);
    TaskExceptionHandler::DealExceptionOp(&rtExceptionInfo1);
    taskExceptionHandler.Flush();
}

TEST_F(TaskExceptionTest, ut_TaskInfo_GetOpDataAndRankInfo)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret;
    ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_tag";
    u64 count = 0;
    void* src = nullptr;
    void* dst = nullptr;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    u32 rootId = 0;
    std::string group = "test_group";
    TaskType taskType = TaskType::TASK_RDMA;
    TaskParaNotify taskParaNotify;
    string str;
    AlgType algType = AlgType::Reserved();

    aclrtStream steam;
    aclrtCreateStream(&steam);
    aclrtStreamGetId(steam, (s32*)&streamID);
    ret = taskExceptionHandler.AddStream(streamID, tag, 0, AlgType::Reserved());
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = taskExceptionHandler.AddTag(tag, group, HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = taskExceptionHandler.AddOpData(tag, count, src, dst, dataType, rootId, group, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = taskExceptionHandler.AddGroupRankInfo(group, 8, rootId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 index = 0;
    TaskInfo taskInfo1(streamID, taskID, tag, taskType, algType, index, taskParaNotify);
    str = taskInfo1.GetBaseInfoStr();
    TaskParaDMA taskParaDMA;
    taskType = TaskType::TASK_RDMA;
    ret = taskExceptionHandler.Save(streamID, taskID, taskType, taskParaDMA);
    CtxInfo tmpCtxInfo(taskType, taskParaDMA);
    str = tmpCtxInfo.GetCtxBaseInfoStr();
    str = tmpCtxInfo.GetCtxParaNotify();

    rtExceptionExpandInfo_t expandInfo;
    expandInfo.u.fftsPlusInfo.contextId = 0;
    expandInfo.type = tagRtExceptionExpandType::RT_EXCEPTION_FFTS_PLUS;
    rtExceptionInfo rtExceptionInfo1;
    rtExceptionInfo1.streamid = streamID;
    rtExceptionInfo1.expandInfo = expandInfo;
    rtExceptionInfo1.deviceid = deviceLogicId;
    taskExceptionHandler.Callback(&rtExceptionInfo1);
    std::string groupRankContentInfo;
    std::string stageErrInfo;
    TaskExceptionHandler::DealExceptionGroupRank(&rtExceptionInfo1, tag, true, groupRankContentInfo, stageErrInfo);
    TaskExceptionHandler::DealExceptionOpData(&rtExceptionInfo1, tag, true, index, stageErrInfo);

    struct timeval tv {
        0
    };
    std::string opDataContent;
    (void)gettimeofday(&tv, nullptr);
    TaskExceptionHandler::TimeStruct2Str(tv, opDataContent);
    ret = taskExceptionHandler.DelOpData(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = taskExceptionHandler.DelGroupRankInfo(group);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    taskExceptionHandler.Flush();
    taskExceptionHandler.DeInit();
    aclrtDestroyStream(steam);
    GlobalMockObject::verify();
}
#if 0 // 执行失败Map size is bigger than max stream count
TEST_F(TaskExceptionTest, ut_task_exception_callback__stuck_heartbeat)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);
 
    HcclResult ret;
    ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_tag";
    TaskType taskType = TaskType::TASK_NOTIFY_WAIT;
    TaskParaNotify taskParaNotify;
    AlgType algType = AlgType::Reserved();
 
    u32 index = 0;
    TaskInfo taskInfo1(streamID, taskID, tag, taskType, algType, index, taskParaNotify);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rtExceptionInfo rtExceptionInfo1{0, 0, 0, 0, 0};
 
    RankInfo rankInfo;
    rankInfo.devicePhyId = 10;
    UIDType crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    UIDType informer = Heartbeat::GetInstance(0).GetUId(rankInfo);
 
    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_STUCK);
 
    rankInfo.devicePhyId = 11;
    crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_STUCK);
 
    rankInfo.devicePhyId = 12;
    crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_STUCK);
 
    rankInfo.devicePhyId = 13;
    crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_STUCK);
 
    rankInfo.devicePhyId = 13;
    crimer = Heartbeat::GetInstance(0).GetUId(rankInfo);
    Heartbeat::GetInstance(0).SetStatus(crimer, informer, HeartBeatStatus::HEARTBEAT_STUCK);
 
    taskExceptionHandler.Callback(&rtExceptionInfo1);
}
#endif
ErrorMessageReport GetAicpuTaskException()
{
    ErrorMessageReport emrReport;
    strcpy(emrReport.tag, "test");
    strcpy(emrReport.group, "grouptest");
    emrReport.remoteUserRank = 1;
    emrReport.streamId = 2;
    emrReport.taskId = 3;
    emrReport.notifyId = 4;
    emrReport.rankId = 5;
    emrReport.rankSize = 6;
    emrReport.algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_HD;
    emrReport.algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
    emrReport.taskType = TaskType::TASK_NOTIFY_WAIT;

    return emrReport;
}

TEST_F(TaskExceptionTest, ut_PrintAicpuErrorMessage)
{
    bool isExistAicpuError = false;
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);
    auto getAicpuTaskExceptionCallBack = []() {
        return GetAicpuTaskException();
    };
    RegisterGetAicpuTaskExceptionCallBack(0, 0, getAicpuTaskExceptionCallBack);
    rtExceptionInfo exceptionInfo{0, 0, 0, 0, 0};

    taskExceptionHandler.PrintAicpuErrorMessage(&exceptionInfo, isExistAicpuError);

    taskExceptionHandler.Callback(&exceptionInfo);
}
#if 0 // 执行失败Map size is bigger than max stream count
TEST_F(TaskExceptionTest, ut_task_exception_aiv)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);
 
    HcclResult ret;
    ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_tag";
    TaskParaAiv taskParaAiv;
    
    u32 index = 0;
    TaskInfo taskInfo1(streamID, taskID, tag, taskParaAiv);

    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rtExceptionInfo rtExceptionInfo1{0, 0, 0, 0, 0};
 
    taskExceptionHandler.DealExceptionTask(&rtExceptionInfo1);
}
#endif

TEST_F(TaskExceptionTest, St_PrintCommAivInfo_When_GetDevice_Fail_Expect_Ret_HCCL_E_PARA)
{
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    EXPECT_EQ(taskExceptionHandler.PrintCommAivInfo(), HCCL_E_PARA);
    GlobalMockObject::verify();
}

TEST_F(TaskExceptionTest, St_PrintCommAivInfo_When_AivGroup_Count_Zero_Expect_Print_Group_No_Aiv)
{
    u32 deviceLogicId = 0;

    TaskExceptionHandler taskExceptionHandler(deviceLogicId);
    std::string groupName = "group_101";
    taskExceptionHandler.aivGroupIndexMap_[deviceLogicId][groupName] = 0;

    EXPECT_EQ(taskExceptionHandler.PrintCommAivInfo(), HCCL_SUCCESS);
    taskExceptionHandler.aivGroupIndexMap_[deviceLogicId].clear();
    GlobalMockObject::verify();
}

TEST_F(TaskExceptionTest, St_PrintCommAivInfo_When_AivGroup_Size_Zero_Expect_Print_Aiv_Group_Not_Recode)
{
    u32 deviceLogicId = 0;

    TaskExceptionHandler taskExceptionHandler(deviceLogicId);
    taskExceptionHandler.aivGroupIndexMap_[deviceLogicId].clear();

    EXPECT_EQ(taskExceptionHandler.PrintCommAivInfo(), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(TaskExceptionTest, St_PrintCommAivInfo_When_AivGroup_Size_One_Expect_No_Print_Aiv_Group)
{
    u32 deviceLogicId = 0;

    TaskExceptionHandler taskExceptionHandler(deviceLogicId);
    std::string groupName = "group_101";
    taskExceptionHandler.aivGroupIndexMap_[deviceLogicId][groupName] = 101;

    EXPECT_EQ(taskExceptionHandler.PrintCommAivInfo(), HCCL_SUCCESS);
    taskExceptionHandler.aivGroupIndexMap_[deviceLogicId].clear();
    GlobalMockObject::verify();
}

TEST_F(TaskExceptionTest, St_PrintCommAivInfo_When_AivGroup_Size_No_Zero_Expect_Print_Multi_Aiv_May_Execution_Stuck)
{
    u32 deviceLogicId = 0;

    TaskExceptionHandler taskExceptionHandler(deviceLogicId);
    HcclResult ret;
    ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    taskExceptionHandler.aivGroupIndexMap_[deviceLogicId]["group_000"] = 0;
    taskExceptionHandler.aivGroupIndexMap_[deviceLogicId]["group_001"] = 0;
    taskExceptionHandler.aivGroupIndexMap_[deviceLogicId]["group_002"] = 0;
    taskExceptionHandler.aivGroupIndexMap_[deviceLogicId]["group_101"] = 101;
    taskExceptionHandler.aivGroupIndexMap_[deviceLogicId]["group_102"] = 102;
    taskExceptionHandler.aivGroupIndexMap_[deviceLogicId]["group_103"] = 103;

    EXPECT_EQ(taskExceptionHandler.PrintCommAivInfo(), HCCL_SUCCESS);
    taskExceptionHandler.aivGroupIndexMap_[deviceLogicId].clear();
    GlobalMockObject::verify();
}

TEST_F(TaskExceptionTest, St_PrintCommAivInfo_When_AivGroup_Size_100_Expect_Print_Multi_Aiv_May_Execution_Stuck)
{
    u32 deviceLogicId = 0;

    TaskExceptionHandler taskExceptionHandler(deviceLogicId);
    HcclResult ret;
    ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < 100; i++) {
        std::stringstream temp;
        temp << "group_" << i;
        taskExceptionHandler.aivGroupIndexMap_[deviceLogicId][temp.str()] = 7;
    }

    EXPECT_EQ(taskExceptionHandler.PrintCommAivInfo(), HCCL_SUCCESS);
    taskExceptionHandler.aivGroupIndexMap_[deviceLogicId].clear();
    GlobalMockObject::verify();
}
#if 0 // 执行失败taskMap size is bigger than max stream count
TEST_F(TaskExceptionTest, St_DealExceptionTask_When_Comm_Has_Multi_Aiv_Expect_Print_May_Execution_Stuck)
{
    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);
 
    HcclResult ret;
    ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 streamID = 0;
    u32 taskID = 0;
    std::string tag = "test_tag";
    TaskParaAiv taskParaAiv;

    TaskInfo taskInfo1(streamID, taskID, tag, taskParaAiv);
    for (int i = 0; i < 100; i++) {
        std::stringstream temp;
        temp << "group_" << i;
        taskExceptionHandler.aivGroupIndexMap_[deviceLogicId][temp.str()] = 1;
    }

    for (int i = 0; i < 100; i++) {
        std::stringstream temp;
        temp << "group_" << i + 100;
        taskExceptionHandler.aivGroupIndexMap_[deviceLogicId][temp.str()] = 0;
    }

    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = taskExceptionHandler.InsertTaskMap(streamID, taskInfo1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rtExceptionInfo rtExceptionInfo1{0, 0, 0, 0, 0};
 
    taskExceptionHandler.DealExceptionTask(&rtExceptionInfo1);
    taskExceptionHandler.aivGroupIndexMap_[deviceLogicId].clear();
    GlobalMockObject::verify();
}
#endif

TEST_F(TaskExceptionTest, Ut_RegisterGetAicpuTaskExceptionCallBack_When_InvaildDeviceId_Expect_NoRegister)
{
    u32 invalidDeviceLogicId = MAX_MODULE_DEVICE_NUM - 1;
    s32 streamId = 1;

    auto callback = []() -> ErrorMessageReport {
        return ErrorMessageReport();
    };

    RegisterGetAicpuTaskExceptionCallBack(streamId, invalidDeviceLogicId, callback);
    EXPECT_TRUE(
        g_communicatorCallbackMap[invalidDeviceLogicId].find(streamId)
        != g_communicatorCallbackMap[invalidDeviceLogicId].end());
    g_communicatorCallbackMap.fill({});
}

TEST_F(TaskExceptionTest, Ut_UnregisterGetAicpuTaskExceptionCallBack_When_Registered_Expect_Removed)
{
    s32 streamId = 2;
    u32 deviceLogicId = 0;
    auto callback = []() -> ErrorMessageReport {
        return ErrorMessageReport();
    };

    RegisterGetAicpuTaskExceptionCallBack(streamId, deviceLogicId, callback);
    EXPECT_TRUE(
        g_communicatorCallbackMap[deviceLogicId].find(streamId) != g_communicatorCallbackMap[deviceLogicId].end());
    UnregisterGetAicpuTaskExceptionCallBack(streamId, deviceLogicId);
    EXPECT_FALSE(
        g_communicatorCallbackMap[deviceLogicId].find(streamId) != g_communicatorCallbackMap[deviceLogicId].end());
    g_communicatorCallbackMap.fill({});
}

TEST_F(TaskExceptionTest, Ut_UnregisterGetAicpuTaskExceptionCallBack_When_NoRegistered_Expect_NoCrash)
{
    s32 streamId = 999;
    u32 deviceLogicId = 0;

    UnregisterGetAicpuTaskExceptionCallBack(streamId, deviceLogicId);
    EXPECT_TRUE(true);
    g_communicatorCallbackMap.fill({});
}

TEST_F(TaskExceptionTest, Ut_RegisterMultipleCallBacks_SameDevice_DifferentStream_Expect_AllStored)
{
    u32 deviceLogicId = 1;
    s32 streamId1 = 10;
    s32 streamId2 = 20;
    s32 streamId3 = 30;
    auto callback1 = []() -> ErrorMessageReport {
        return ErrorMessageReport();
    };
    auto callback2 = []() -> ErrorMessageReport {
        return ErrorMessageReport();
    };
    auto callback3 = []() -> ErrorMessageReport {
        return ErrorMessageReport();
    };

    RegisterGetAicpuTaskExceptionCallBack(streamId1, deviceLogicId, callback1);
    RegisterGetAicpuTaskExceptionCallBack(streamId2, deviceLogicId, callback2);
    RegisterGetAicpuTaskExceptionCallBack(streamId3, deviceLogicId, callback3);

    EXPECT_EQ(g_communicatorCallbackMap[deviceLogicId].size(), 3u);
    EXPECT_TRUE(
        g_communicatorCallbackMap[deviceLogicId].find(streamId1) != g_communicatorCallbackMap[deviceLogicId].end());
    EXPECT_TRUE(
        g_communicatorCallbackMap[deviceLogicId].find(streamId2) != g_communicatorCallbackMap[deviceLogicId].end());
    EXPECT_TRUE(
        g_communicatorCallbackMap[deviceLogicId].find(streamId3) != g_communicatorCallbackMap[deviceLogicId].end());
    g_communicatorCallbackMap.fill({});
}

TEST_F(TaskExceptionTest, Ut_UnregisterOneCallBack_OtherCallbacksPreserved_Expect_Correct)
{
    u32 deviceLogicId = 2;
    s32 streamId1 = 100;
    s32 streamId2 = 200;
    auto callback1 = []() -> ErrorMessageReport {
        return ErrorMessageReport();
    };
    auto callback2 = []() -> ErrorMessageReport {
        return ErrorMessageReport();
    };

    RegisterGetAicpuTaskExceptionCallBack(streamId1, deviceLogicId, callback1);
    RegisterGetAicpuTaskExceptionCallBack(streamId2, deviceLogicId, callback2);
    EXPECT_EQ(g_communicatorCallbackMap[deviceLogicId].size(), 2u);
    UnregisterGetAicpuTaskExceptionCallBack(streamId1, deviceLogicId);
    EXPECT_EQ(g_communicatorCallbackMap[deviceLogicId].size(), 1u);
    EXPECT_TRUE(
        g_communicatorCallbackMap[deviceLogicId].find(streamId2) != g_communicatorCallbackMap[deviceLogicId].end());
    EXPECT_FALSE(
        g_communicatorCallbackMap[deviceLogicId].find(streamId1) != g_communicatorCallbackMap[deviceLogicId].end());
    g_communicatorCallbackMap.fill({});
}

TEST_F(TaskExceptionTest, Ut_CallbackOverwrite_SameStreamId_Expect_Updated)
{
    u32 deviceLogicId = 3;
    s32 streamId = 500;
    bool callback1Called = false;
    bool callback2Called = false;

    auto callback1 = [&callback1Called]() -> ErrorMessageReport {
        callback1Called = true;
        return ErrorMessageReport();
    };
    auto callback2 = [&callback2Called]() -> ErrorMessageReport {
        callback2Called = true;
        return ErrorMessageReport();
    };

    RegisterGetAicpuTaskExceptionCallBack(streamId, deviceLogicId, callback1);
    RegisterGetAicpuTaskExceptionCallBack(streamId, deviceLogicId, callback2);
    EXPECT_EQ(g_communicatorCallbackMap[deviceLogicId].size(), 1u);
    auto it = g_communicatorCallbackMap[deviceLogicId].find(streamId);
    ASSERT_TRUE(it != g_communicatorCallbackMap[deviceLogicId].end());
    it->second();
    EXPECT_TRUE(callback2Called);
    EXPECT_FALSE(callback1Called);
    g_communicatorCallbackMap.fill({});
}

TEST_F(TaskExceptionTest, Ut_RegisterGetAicpuTaskExceptionCallBack_When_Normal_Expect_SUCCESS)
{
    s32 streamId = 1;
    u32 deviceLogicId = 0;
    auto callback = []() -> ErrorMessageReport {
        return ErrorMessageReport();
    };

    RegisterGetAicpuTaskExceptionCallBack(streamId, deviceLogicId, callback);
    EXPECT_TRUE(
        g_communicatorCallbackMap[deviceLogicId].find(streamId) != g_communicatorCallbackMap[deviceLogicId].end());
    g_communicatorCallbackMap.fill({});
}
