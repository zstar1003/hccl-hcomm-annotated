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

#define private public
#define protected public
#include "string_util.h"
#include "buffer.h"
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

    taskInfo.dfxOpInfo_->algType_ = AlgType::RING;
    EXPECT_EQ(taskInfo.GetAlgTypeName(), "AlgType::RING");

    taskInfo.dfxOpInfo_->algType_ = AlgType::MULTI_RING;
    EXPECT_EQ(taskInfo.GetAlgTypeName(), "AlgType::MULTI_RING");

    taskInfo.dfxOpInfo_->algType_ = AlgType::MESH;
    EXPECT_EQ(taskInfo.GetAlgTypeName(), "AlgType::MESH");

    taskInfo.dfxOpInfo_->algType_ = AlgType::RECURSIVE_HD;
    EXPECT_EQ(taskInfo.GetAlgTypeName(), "AlgType::RECURSIVE_HD");

    taskInfo.dfxOpInfo_->algType_ = AlgType::BINARY_HD;
    EXPECT_EQ(taskInfo.GetAlgTypeName(), "AlgType::BINARY_HD");

    taskInfo.dfxOpInfo_->algType_ = AlgType::PAIR_WISE;
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
    taskInfo.dfxOpInfo_->tag_ = "tag_name";
    taskInfo.dfxOpInfo_->algType_ = AlgType::MESH;
    string res = taskInfo.GetBaseInfo();
    cout << res << endl;
    EXPECT_NO_THROW(taskInfo.GetBaseInfo());

    taskInfo.dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
    EXPECT_NO_THROW(taskInfo.GetBaseInfo());
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
    ParaDMA paraDMA{(void*)0xaaaa, (void*)0xbbbb, 0xa, 0xaaaabbbbcccc, DfxLinkType::ONCHIP};
    taskInfo.taskParam_.taskPara.DMA = paraDMA;
    EXPECT_NO_THROW(taskInfo.GetParaInfo());
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
        0xaaaabbbbcccc,
        DfxLinkType::HCCS,
        HcclReduceOp::HCCL_REDUCE_SUM,
        HcclDataType::HCCL_DATA_TYPE_INT32};
    taskInfo.taskParam_.taskPara.Reduce = paraReduce;
    EXPECT_NO_THROW(taskInfo.GetParaInfo());
}

TEST_F(TaskInfoTest, test_get_para_notify)
{
    TaskInfo taskInfo = InitTaskInfo();
    taskInfo.taskParam_.taskType = TaskParamType::TASK_NOTIFY_WAIT;
    taskInfo.remoteRank_ = 3;
    taskInfo.taskParam_.taskPara.Notify.notifyID = 0xaaaabbbbcccc;
    taskInfo.taskParam_.taskPara.Notify.value = 0xa;
    EXPECT_NO_THROW(taskInfo.GetParaInfo());
}

TEST_F(TaskInfoTest, test_get_op_info)
{
    TaskInfo taskInfo = InitTaskInfo();

    taskInfo.dfxOpInfo_->commIndex_ = 3;
    taskInfo.dfxOpInfo_->op_.dataCount = 0xaaaabbbbcccc;
    taskInfo.dfxOpInfo_->op_.reduceOp = ReduceOp::SUM;
    taskInfo.dfxOpInfo_->op_.dataType = DataType::UINT64;
    EXPECT_NO_THROW(taskInfo.GetOpInfo());

    taskInfo.dfxOpInfo_->op_.inputMem = make_shared<Buffer>(0x111122223333, 0);
    taskInfo.dfxOpInfo_->op_.outputMem = make_shared<Buffer>(0xaaaabbbbcccc, 0);
    EXPECT_NO_THROW(taskInfo.GetOpInfo());

    taskInfo.dfxOpInfo_ = shared_ptr<DfxOpInfo>(nullptr);
    EXPECT_NO_THROW(taskInfo.GetOpInfo());
}
