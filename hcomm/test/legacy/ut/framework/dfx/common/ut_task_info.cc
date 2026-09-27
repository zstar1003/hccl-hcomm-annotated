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
#include <mockcpp/mokc.h>

#include "buffer.h"

#define private public
#define protected public
#include "task_info.h"
#undef private
#undef protected

using namespace std;
using namespace Hccl;

class TaskInfoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TaskInfoTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "TaskInfoTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in TaskInfoTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in TaskInfoTest TearDown" << std::endl;
    }

    TaskInfo InitTaskInfo()
    {
        TaskParam taskParam{};
        shared_ptr<DfxOpInfo> dfxOpInfo = make_shared<DfxOpInfo>();
        return TaskInfo{0, 0, 0, taskParam, dfxOpInfo};
    }
};

TEST_F(TaskInfoTest, test_get_alg_type_name)
{
    TaskInfo taskInfo = InitTaskInfo();

    taskInfo.dfxOpInfo_->algType_ = AlgType{AlgType::RING}.Describe();
    EXPECT_EQ(taskInfo.GetAlgTypeName(), "AlgType::RING");

    taskInfo.dfxOpInfo_->algType_ = AlgType{AlgType::MULTI_RING}.Describe();
    EXPECT_EQ(taskInfo.GetAlgTypeName(), "AlgType::MULTI_RING");

    taskInfo.dfxOpInfo_->algType_ = AlgType{AlgType::MESH}.Describe();
    EXPECT_EQ(taskInfo.GetAlgTypeName(), "AlgType::MESH");

    taskInfo.dfxOpInfo_->algType_ = AlgType{AlgType::RECURSIVE_HD}.Describe();
    EXPECT_EQ(taskInfo.GetAlgTypeName(), "AlgType::RECURSIVE_HD");

    taskInfo.dfxOpInfo_->algType_ = AlgType{AlgType::BINARY_HD}.Describe();
    EXPECT_EQ(taskInfo.GetAlgTypeName(), "AlgType::BINARY_HD");

    taskInfo.dfxOpInfo_->algType_ = AlgType{AlgType::PAIR_WISE}.Describe();
    EXPECT_EQ(taskInfo.GetAlgTypeName(), "AlgType::PAIR_WISE");

    taskInfo.dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
    EXPECT_EQ(taskInfo.GetAlgTypeName(), "NULL");
}

TEST_F(TaskInfoTest, test_get_task_concise_name)
{
    TaskInfo taskInfo = InitTaskInfo();

    taskInfo.taskParam_.taskType = TaskParamType::TASK_SDMA;
    EXPECT_EQ(taskInfo.GetTaskConciseName(), "M");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_RDMA;
    EXPECT_EQ(taskInfo.GetTaskConciseName(), "RS");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_SEND_PAYLOAD;
    EXPECT_EQ(taskInfo.GetTaskConciseName(), "SP");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_REDUCE_INLINE;
    EXPECT_EQ(taskInfo.GetTaskConciseName(), "IR");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_REDUCE_TBE;
    EXPECT_EQ(taskInfo.GetTaskConciseName(), "R");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_NOTIFY_RECORD;
    EXPECT_EQ(taskInfo.GetTaskConciseName(), "NR");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    EXPECT_EQ(taskInfo.GetTaskConciseName(), "NW");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_SEND_NOTIFY;
    EXPECT_EQ(taskInfo.GetTaskConciseName(), "SN");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_WRITE_WITH_NOTIFY;
    EXPECT_EQ(taskInfo.GetTaskConciseName(), "WN");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY;
    EXPECT_EQ(taskInfo.GetTaskConciseName(), "WRN");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_CCU;
    EXPECT_EQ(taskInfo.GetTaskConciseName(), "CCU");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_AICPU_KERNEL;
    EXPECT_EQ(taskInfo.GetTaskConciseName(), "AIK");
}

TEST_F(TaskInfoTest, test_get_notify_info)
{
    TaskInfo taskInfo = InitTaskInfo();

    taskInfo.taskParam_.taskType = TaskParamType::TASK_REDUCE_TBE;
    EXPECT_EQ(taskInfo.GetNotifyInfo(), "/");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_RDMA;
    taskInfo.taskParam_.taskPara.DMA.notifyID = UINT64_MAX;
    EXPECT_EQ(taskInfo.GetNotifyInfo(), "/");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskInfo.taskParam_.taskPara.Notify.notifyID = 222;
    EXPECT_EQ(taskInfo.GetNotifyInfo(), "222");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_NOTIFY_RECORD;
    taskInfo.taskParam_.taskPara.Notify.notifyID = 333;
    EXPECT_EQ(taskInfo.GetNotifyInfo(), "333");
}

TEST_F(TaskInfoTest, test_get_base_info)
{
    TaskInfo taskInfo = InitTaskInfo();

    taskInfo.streamId_ = 1;
    taskInfo.taskId_ = 7;
    taskInfo.taskParam_.taskType = TaskParamType::TASK_SDMA;
    EXPECT_EQ(taskInfo.GetBaseInfo(), "streamID(sqId):[1], taskID(sqeId):[7], taskType:[TaskParamType::TASK_SDMA]");

    taskInfo.dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
    EXPECT_EQ(taskInfo.GetBaseInfo(), "");
}

TEST_F(TaskInfoTest, test_get_concise_base_info)
{
    TaskInfo taskInfo = InitTaskInfo();

    taskInfo.taskParam_.taskType = TaskParamType::TASK_REDUCE_TBE;
    taskInfo.remoteRank_ = UINT32_MAX;
    EXPECT_EQ(taskInfo.GetConciseBaseInfo(), "R(/)");

    taskInfo.taskParam_.taskType = TaskParamType::TASK_NOTIFY_RECORD;
    taskInfo.remoteRank_ = 3;
    taskInfo.taskParam_.taskPara.Notify.notifyID = 123;
    EXPECT_EQ(taskInfo.GetConciseBaseInfo(), "NR(3,123)");
}

TEST_F(TaskInfoTest, test_get_para_ccu)
{
    TaskInfo taskInfo = InitTaskInfo();
    taskInfo.taskParam_.taskType = TaskParamType::TASK_CCU;
    EXPECT_EQ(taskInfo.GetParaInfo(), "TaskParamType::TASK_CCU");
}

TEST_F(TaskInfoTest, test_get_para_dma)
{
    TaskInfo taskInfo = InitTaskInfo();
    taskInfo.taskParam_.taskType = TaskParamType::TASK_RDMA;
    taskInfo.remoteRank_ = 3;
    ParaDMA paraDMA{(void*)0xaaaa, (void*)0xbbbb, 0xa, 123, 1, DfxLinkType::ONCHIP};
    taskInfo.taskParam_.taskPara.DMA = paraDMA;
    EXPECT_EQ(
        taskInfo.GetParaInfo(), "src:[0xaaaa], dst:[0xbbbb], size:[0xa], notify id:[123], link "
                                "type:[DfxLinkType::ONCHIP], remote rank:[3]");
}

TEST_F(TaskInfoTest, test_get_para_reduce)
{
    TaskInfo taskInfo = InitTaskInfo();
    taskInfo.taskParam_.taskType = TaskParamType::TASK_REDUCE_TBE;
    taskInfo.remoteRank_ = UINT32_MAX;
    ParaReduce paraReduce{
        (void*)0xaaaa,
        (void*)0xbbbb,
        0xa,
        123,
        1,
        DfxLinkType::HCCS,
        HcclReduceOp::HCCL_REDUCE_SUM,
        HcclDataType::HCCL_DATA_TYPE_INT32};
    taskInfo.taskParam_.taskPara.Reduce = paraReduce;
    EXPECT_EQ(
        taskInfo.GetParaInfo(), "src:[0xaaaa], dst:[0xbbbb], size:[0xa], notify id:[123], op:[0], data "
                                "type:[2], link type:[DfxLinkType::HCCS], remote rank:[local]");
}

TEST_F(TaskInfoTest, test_get_para_notify)
{
    TaskInfo taskInfo = InitTaskInfo();
    taskInfo.taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskInfo.remoteRank_ = 3;
    taskInfo.taskParam_.taskPara.Notify.notifyID = 123;
    taskInfo.taskParam_.taskPara.Notify.value = 0xa;
    EXPECT_EQ(taskInfo.GetParaInfo(), "notify id:[123], value:[10], remote rank:[3]");
}

TEST_F(TaskInfoTest, test_GetIndopBaseInfo)
{
    TaskInfo taskInfo = InitTaskInfo();
    taskInfo.streamId_ = 1;
    taskInfo.taskId_ = 7;
    taskInfo.taskParam_.taskType = TaskParamType::TASK_SDMA;
    EXPECT_EQ(
        taskInfo.GetIndopBaseInfo(), "streamID(sqId):[1], taskID(sqeId):[7], taskType:[TaskParamType::TASK_SDMA]");
}

TEST_F(TaskInfoTest, test_GetIndopDataInfo)
{
    TaskInfo taskInfo = InitTaskInfo();
    taskInfo.dfxOpInfo_->opIndex_ = 1;
    taskInfo.dfxOpInfo_->algTag_ = "allreduce_test";
    taskInfo.dfxOpInfo_->op_.dataCount = 1024;
    taskInfo.dfxOpInfo_->op_.oldReduceOp = static_cast<u32>(ReduceOp::SUM);
    taskInfo.dfxOpInfo_->op_.oldDataType = static_cast<u32>(DataType::INT32);
    taskInfo.dfxOpInfo_->op_.inputAddr = 0x1;
    taskInfo.dfxOpInfo_->op_.inputSize = 11;
    taskInfo.dfxOpInfo_->op_.outputAddr = 0x2;
    taskInfo.dfxOpInfo_->op_.outputSize = 22;
    EXPECT_EQ(
        taskInfo.GetIndopDataInfo(),
        "opIndex[1], algTag[allreduce_test], count[1024], reduceType[ReduceOp::SUM], dataType[DataType::INT32], "
        "input: ptr[0x1] size[11], output: ptr[0x2] size[22]");

    taskInfo.dfxOpInfo_ = nullptr;
    EXPECT_EQ(taskInfo.GetIndopDataInfo(), "");
}

TEST_F(TaskInfoTest, Ut_Describe_When_DfxOpInfoNullptr_Expect_ReturnNullptrDesc)
{
    TaskParam taskParam{};
    TaskInfo taskInfo{0, 0, 0, taskParam, nullptr};
    std::string desc = taskInfo.Describe();
    EXPECT_NE(desc.find("dfxOpInfo:[nullptr]"), std::string::npos);
    EXPECT_NE(desc.find("streamId(sqId):[0]"), std::string::npos);
    EXPECT_NE(desc.find("taskId(sqeId):[0]"), std::string::npos);
}

TEST_F(TaskInfoTest, Ut_Describe_When_DfxOpInfoValid_Expect_ReturnFullDesc)
{
    TaskInfo taskInfo = InitTaskInfo();
    taskInfo.streamId_ = 1;
    taskInfo.taskId_ = 2;
    taskInfo.dfxOpInfo_->tag_ = "test_tag";
    std::string desc = taskInfo.Describe();
    EXPECT_NE(desc.find("dfxOpInfo:[DfxOpInfo:"), std::string::npos);
    EXPECT_NE(desc.find("streamId(sqId):[1]"), std::string::npos);
    EXPECT_NE(desc.find("taskId(sqeId):[2]"), std::string::npos);
}

TEST_F(TaskInfoTest, Ut_GetOpInfo_When_DfxOpInfoNullptr_Expect_ReturnEmpty)
{
    TaskParam taskParam{};
    TaskInfo taskInfo{0, 0, 0, taskParam, nullptr};
    EXPECT_EQ(taskInfo.GetOpInfo(), "");
}

TEST_F(TaskInfoTest, Ut_GetOpInfo_When_DfxOpInfoValid_Expect_ReturnOpInfo)
{
    TaskInfo taskInfo = InitTaskInfo();
    taskInfo.dfxOpInfo_->commIndex_ = 5;
    taskInfo.dfxOpInfo_->op_.opType = OpType::ALLREDUCE;
    taskInfo.dfxOpInfo_->op_.dataCount = 100;
    std::string result = taskInfo.GetOpInfo();
    EXPECT_NE(result.find("commIndex[5]"), std::string::npos);
    EXPECT_NE(result.find("count[100]"), std::string::npos);
}

TEST_F(TaskInfoTest, Ut_PrintTaskLog_When_TaskTypeSdma_Expect_NoCrash)
{
    TaskParam taskParam{};
    taskParam.taskType = TaskParamType::TASK_SDMA;
    taskParam.taskPara.DMA.src = reinterpret_cast<void*>(0x1234);
    taskParam.taskPara.DMA.dst = reinterpret_cast<void*>(0x5678);
    taskParam.taskPara.DMA.size = 1024;
    taskParam.taskPara.DMA.notifyID = 999;
    PrintTaskLog(0, 0, taskParam, 0);
    SUCCEED();
}

TEST_F(TaskInfoTest, Ut_PrintTaskLog_When_TaskTypeReduce_Expect_NoCrash)
{
    TaskParam taskParam{};
    taskParam.taskType = TaskParamType::TASK_REDUCE_INLINE;
    taskParam.taskPara.Reduce.src = reinterpret_cast<void*>(0x1234);
    taskParam.taskPara.Reduce.dst = reinterpret_cast<void*>(0x5678);
    taskParam.taskPara.Reduce.size = 1024;
    taskParam.taskPara.Reduce.notifyID = 999;
    taskParam.taskPara.Reduce.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    taskParam.taskPara.Reduce.reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    PrintTaskLog(0, 0, taskParam, 0);
    SUCCEED();
}

TEST_F(TaskInfoTest, Ut_PrintTaskLog_When_TaskTypeNotify_Expect_NoCrash)
{
    TaskParam taskParam{};
    taskParam.taskType = TaskParamType::TASK_NOTIFY_RECORD;
    taskParam.taskPara.Notify.notifyID = 888;
    PrintTaskLog(0, 0, taskParam, 0);
    SUCCEED();
}
