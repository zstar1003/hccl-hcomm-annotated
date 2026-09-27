/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include "gtest/gtest.h"
#include "stream_pub.h"
#include "topoinfo_struct.h"
#include "log.h"
#include "checker_def.h"
#include "topo_meta.h"
#include "testcase_utils.h"
#include "hccl_aiv.h"
#include "checker.h"
#include "dispatcher.h"
using namespace hccl;
using namespace checker;

std::vector<string> allReduceAlgName = {
    "AllReduceMeshOneshotLoopExecutor",    "AllReduceMeshAivSmallCountExecutor", "AllReduceMeshOpbaseLoopExecutor",
    "AllReduceMeshOpbasePipelineExecutor", "AllReduceMeshSmallCountExecutor",    "AllReduceRingExecutor",
};

class AllReduceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CheckOpSemanticsTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CheckOpSemanticsTest tear down." << std::endl; }

    virtual void SetUp()
    {
        const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string caseName
            = "analysis_result_" + std::string(test_info->test_case_name()) + "_" + std::string(test_info->name());
        Checker::SetDumpFileName(caseName);
        MOCKER(ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    }

    virtual void TearDown()
    {
        Checker::SetDumpFileName("analysis_result");
        GlobalMockObject::verify();
        // 这边每个case执行完成需要清理所有的环境变量，如果有新增的环境变量，需要在这个函数中进行清理
        ClearHcclEnv();
    }
};

TEST_F(AllReduceTest, allreduce_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);

    setenv("HCCL_DETERMINISTIC", "true", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_test_loop)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    setenv("HCCL_DETERMINISTIC", "true", 1);

    HcclResult ret;
    for (int i = 1; i <= 2; i++) {
        for (int j = 1; j <= 8; j++) {
            Checker checker;
            TopoMeta topoMeta;
            gen.GenTopoMeta(topoMeta, 1, i, j);
            ret = checker.Check(checkerOpParam, topoMeta);
            printf("serverNum = %d, rankNum = %d, ret is %d\n", i, j, ret);
            EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
        }
    }
}

TEST_F(AllReduceTest, allreduce_test_offload_loop)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    setenv("HCCL_DETERMINISTIC", "true", 1);
    HcclResult ret;
    for (int i = 1; i <= 2; i++) {
        for (int j = 1; j <= 8; j++) {
            Checker checker;
            TopoMeta topoMeta;
            gen.GenTopoMeta(topoMeta, 1, i, j);
            ret = checker.Check(checkerOpParam, topoMeta);
            EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
        }
    }
}

TEST_F(AllReduceTest, allreduce_executor_offload_AllReduceMeshOneshotLoopExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceMeshOneshotLoopExecutor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_executor_offload_AllReduceMeshSmallCountExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);
    setenv("HCCL_DETERMINISTIC", "true", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceMeshSmallCountExecutor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// 跑AllReduceReduceBcast
TEST_F(AllReduceTest, allreduce_AllReduceReduceBcast)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    setenv("HCCL_DETERMINISTIC", "false", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "AllReduceMeshSmallCountExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck(); // AllReduceReduceBcast算法以mesh方式做原子Reduce操作，导致内存冲突检测误报
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_executor_offload_AllReduceRingExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceRingExecutor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_allReduceComm_executor_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.algName = "AllReduceComm";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_executor_test_loop)
{
    MOCKER(ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    for (int i = 0; i < allReduceAlgName.size(); i++) {
        checkerOpParam.algName = allReduceAlgName[i];
        Checker checker;
        HcclResult ret;
        checker.CloseRankMemCheck();
        ret = checker.Check(checkerOpParam, topoMeta);
        if (ret == HcclResult::HCCL_SUCCESS) {
            std::cout << allReduceAlgName[i] + " run succeeded" << std::endl;
        } else {
            std::cout << allReduceAlgName[i] + " run failed" << std::endl;
        }
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
}

TEST_F(AllReduceTest, allreduce_AllReduceMeshMidCountLoopExecutor_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.algName = "AllReduceMeshMidCountLoopExecutor";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_AllReduceSingleExecutor_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.algName = "AllReduceSingleExecutor";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910A_offload_AllReduceMeshExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4); // 等下改成8，看看会不会报错

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceMeshExecutor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910A_offload_AllReduceRingExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceRingExecutor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910A_offload_AllReduceRingExecutor_NSLB)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 3, 8);
    setenv("HCCL_ALGO", "level0:ring;level1:H-D_R", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceRingExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910A_offload_AllReduceComm)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{2, 5}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceComm";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910A_opbase_AllReduceMeshExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceMeshExecutor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910A_opbase_AllReduceRingExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceRingExecutor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910A_opbase_AllReduceComm)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{2, 5}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceComm";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_opbase_AllReduceFastDoubleRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceFastDoubleRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_offload_AllReduceFastDoubleRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceFastDoubleRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_opbase_AlignedAllReduceDoubleRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AlignedAllReduceDoubleRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_offload_AlignedAllReduceDoubleRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AlignedAllReduceDoubleRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_opbase_AllReduceRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_opbase_NHRoneshot)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);
    setenv("HCCL_ALGO", "level0:ring;level1:NHR", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT64;
    checkerOpParam.algName = "AllReduceFastDoubleRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    checker.EnableGraphicDump();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_opbase_NHRoneshot_reduce_min)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);
    setenv("HCCL_ALGO", "level0:ring;level1:NHR", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MIN;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT64;
    checkerOpParam.algName = "AllReduceFastDoubleRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    checker.EnableGraphicDump();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_opbase_NHRoneshot_reduce_max)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);
    setenv("HCCL_ALGO", "level0:ring;level1:NHR", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MAX;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT64;
    checkerOpParam.algName = "AllReduceFastDoubleRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    checker.EnableGraphicDump();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_opbase_NHRoneshot_reduce_prod)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);
    setenv("HCCL_ALGO", "level0:ring;level1:NHR", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_PROD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT64;
    checkerOpParam.algName = "AllReduceFastDoubleRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    checker.EnableGraphicDump();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_AllReduceRingFor91093Executor_NHR_tbe)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 8);
    setenv("HCCL_ALGO", "level0:ring;level1:NHR", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 4 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT64;
    checkerOpParam.algName = "AllReduceFastDoubleRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_AllReduceRingFor91093Executor_NHR)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);
    setenv("HCCL_ALGO", "level0:ring;level1:NHR", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 4 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "AllReduceRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_opbase_AllReduceComm)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceComm";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_offload_AllReduceComm)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceComm";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_offload_AllReduceDoubleRingConcurrentExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    setenv("HCCL_CONCURRENT_ENABLE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_310P3_opbase_AllReduceRing)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_310P3_opbase_AllReduceDoublingDirect)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.DataDes.count = 16;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_310P3_opbase_AllReduceDoublingDirect_bigcount)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.DataDes.count = 1000000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_310P3_opbase_AllReduceDoubling)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.DataDes.count = 16;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceDoubling";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_opbase_AllReduceFastDoubleRingFor91093Executor_superPod)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceFastDoubleRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_level1_NB)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);

    setenv("HCCL_ALGO", "level0:ring;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_level1_NHR)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);

    setenv("HCCL_ALGO", "level0:ring;level1:NHR", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_level1_AHC)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);

    setenv("HCCL_ALGO", "level0:ring;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_310P3_offload_AllReduceRing)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_310P3V_opbase_AllReduceRing)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceRing";
    checkerOpParam.is310P3V = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_310P3V_offload_AllReduceRing)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceRing";
    checkerOpParam.is310P3V = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_910_93_retry_inplace_presync_case_level0)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 134217728;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AlignedAllReduceDoubleRingFor91093Executor";
    checkerOpParam.algOpContext.opRetryHandler.isInplacePreSync = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    // PreSync会memcopy一段无效数据以占满所有端口，从而达到检测断链的效果，所以数据本身没有被使用，是允许的。
    // 且PreSync不执行任何reduce scatter算法。
}

TEST_F(AllReduceTest, allreduce_910_93_retry_inplace_presync_case_level1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 8, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 134217728;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AlignedAllReduceDoubleRingFor91093Executor";
    checkerOpParam.algOpContext.opRetryHandler.isInplacePreSync = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    // PreSync会memcopy一段无效数据以占满所有端口，从而达到检测断链的效果，所以数据本身没有被使用，是允许的。
    // 且PreSync不执行任何reduce scatter算法。
}

TEST_F(AllReduceTest, allreduce_910_93_retry_inplace_presync_case_level2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 134217728;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AlignedAllReduceDoubleRingFor91093Executor";
    checkerOpParam.algOpContext.opRetryHandler.isInplacePreSync = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    // PreSync会memcopy一段无效数据以占满所有端口，从而达到检测断链的效果，所以数据本身没有被使用，是允许的。
    // 且PreSync不执行任何reduce scatter算法。
}

TEST_F(AllReduceTest, all_reduce_smallcount_multiserver_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;

    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_ax_4server_16p)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_superpod_asym_gcd)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 1000000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_superpod_asym_gcd_graph)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{
        {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_mix_AllReduceMixExecutor)
{
    RankTable_For_LLT gen;
    /***** GenTopoMeta Start *****/
    TopoMeta topoMeta;
    SuperPodMeta superPodMeta910B;
    ServerMeta serverMeta910B;
    u32 rankNum910B = 16;
    for (u32 k = 0; k < rankNum910B; k++) {
        serverMeta910B.push_back(k);
    }
    superPodMeta910B.push_back(serverMeta910B);
    SuperPodMeta superPodMeta91093;
    ServerMeta serverMeta91093;
    u32 rankNum91093 = 16;
    for (u32 k = 0; k < rankNum91093; k++) {
        serverMeta91093.push_back(k);
    }
    superPodMeta91093.push_back(serverMeta91093);

    topoMeta.push_back(superPodMeta910B);
    topoMeta.push_back(superPodMeta91093);
    /***** GenTopoMeta End *****/

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910B);
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910_93);

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_mix_AllReduceMixExecutor_deter)
{
    setenv("HCCL_DETERMINISTIC", "true", 1);
    RankTable_For_LLT gen;

    /***** GenTopoMeta Start *****/
    TopoMeta topoMeta;
    SuperPodMeta superPodMeta910B;
    ServerMeta serverMeta910B;
    u32 rankNum910B = 16;
    for (u32 k = 0; k < rankNum910B; k++) {
        serverMeta910B.push_back(k);
    }
    superPodMeta910B.push_back(serverMeta910B);
    SuperPodMeta superPodMeta91093;
    ServerMeta serverMeta91093;
    u32 rankNum91093 = 16;
    for (u32 k = 0; k < rankNum91093; k++) {
        serverMeta91093.push_back(k);
    }
    superPodMeta91093.push_back(serverMeta91093);

    topoMeta.push_back(superPodMeta910B);
    topoMeta.push_back(superPodMeta91093);
    /***** GenTopoMeta End *****/

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910B);
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910_93);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_mix_AllReduceMixExecutor_deter_comm)
{
    setenv("HCCL_DETERMINISTIC", "true", 1);
    RankTable_For_LLT gen;

    /***** GenTopoMeta Start *****/
    TopoMeta topoMeta;
    SuperPodMeta superPodMeta910B;
    ServerMeta serverMeta910B;
    u32 rankNum910B = 1;
    for (u32 k = 0; k < rankNum910B; k++) {
        serverMeta910B.push_back(k);
    }
    superPodMeta910B.push_back(serverMeta910B);
    SuperPodMeta superPodMeta91093;
    ServerMeta serverMeta91093;
    u32 rankNum91093 = 1;
    for (u32 k = 0; k < rankNum91093; k++) {
        serverMeta91093.push_back(k);
    }
    superPodMeta91093.push_back(serverMeta91093);

    topoMeta.push_back(superPodMeta910B);
    topoMeta.push_back(superPodMeta91093);
    /***** GenTopoMeta End *****/

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910B);
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910_93);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_aiv_a2_AllReduceMeshAivExecutor)
{
    MOCKER(ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 4000000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.aicpuUnfoldMode = true;
    checkerOpParam.algName = "AllReduceMeshAivExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_aiv_a3_graph_AllReduceMeshAivSmallCountExecutor)
{
    MOCKER(ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_AllReduceRingFor91093Executor_NHR)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:NHR;level2:NHR", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_AllReduceRingFor91093Executor_NB)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:NB;level2:NB", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_AllReduceRingFor91093Executor_Ring)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 2);

    setenv("HCCL_ALGO", "level0:NA;level1:ring;level2:ring", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_aiv_ax_AllReduceSmallCountAivRdmaExecutor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(HcclD2DMemcpyAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceSmallCountAivRdmaExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
}

TEST_F(AllReduceTest, allreduce_aiv_ax_AllReduceMidCountAivRdmaExecutor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(HcclD2DMemcpyAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceMidCountAivRdmaExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
}

TEST_F(AllReduceTest, allreduce_A3_2Server1Rank_nb_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 1);

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_91093_AllReduceRingZerocopyExecutor_multisuperpod)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllReduceRingZerocopyExecutor";
    checkerOpParam.isZeroCopy = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_91093_AllReduceRingZerocopyExecutor_singlesuperpod)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllReduceRingZerocopyExecutor";
    checkerOpParam.isZeroCopy = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_91093_AllReduceRingZerocopyExecutor_singlesuperpod_AHC)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 4);

    setenv("HCCL_ALGO", "level1:AHC", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllReduceRingZerocopyExecutor";
    checkerOpParam.isZeroCopy = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_91093_AllReduceAivExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 4);

    setenv("HCCL_ALGO", "level1:AHC", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllReduceMeshAivExecutor";
    checkerOpParam.isZeroCopy = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
TEST_F(AllReduceTest, allreduce_A2_2Server_offload_AllReduceMeshGraphPipelineExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);
    setenv("HCCL_ALGO", "level1:pipeline", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "AllReduceMeshGraphPipelineExecutor";

    Checker checker;
    checker.CloseRankMemCheck();
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_A2_2Server_offload_per7_AllReduceMeshGraphPipelineExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 7);
    setenv("HCCL_ALGO", "level1:pipeline", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "AllReduceMeshGraphPipelineExecutor";

    Checker checker;
    checker.CloseRankMemCheck();
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_A2_3Server_offload_AllReduceMeshGraphPipelineExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 3, 8);
    setenv("HCCL_ALGO", "level1:pipeline", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "AllReduceMeshGraphPipelineExecutor";

    Checker checker;
    checker.CloseRankMemCheck();
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_A2_1Server8Rank_order_preserved)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "AllReduceOrderPreservedExecutor";

    Checker checker;
    HcclResult ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_A2_1Server16Rank_order_preserved)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 190;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "AllReduceOrderPreservedExecutor";

    Checker checker;
    HcclResult ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_A2_1Server2Rank_mesh_opbase)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 512 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "AllReduceMeshOpbaseLoopExecutor";

    Checker checker;
    HcclResult ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_aiv_determinstic_small_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);

    setenv("HCCL_DETERMINISTIC", "true", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AIV", 1);
    MOCKER(ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceAivDeterSmallExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_aiv_determinstic_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);

    setenv("HCCL_DETERMINISTIC", "true", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AIV", 1);
    MOCKER(ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 8 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceAivDeterExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_aiv_determinstic_mid_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);

    setenv("HCCL_DETERMINISTIC", "true", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AIV", 1);
    MOCKER(ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllReduceAivDeterSmallExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_aiv_determinstic_priority_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    setenv("HCCL_DETERMINISTIC", "true", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AIV", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

constexpr int BIG_COUNT = 128 - 1;
void AllReduceDeterCheckerFor910B(int serverNum, int intraRankNum, int count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, serverNum, intraRankNum);
    setenv("HCCL_DETERMINISTIC", "true", 1);
    setenv("HCCL_ALGO", "level0:NA;level1:ring", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = count;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceDeterPipelineExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, all_reduce_executor_deter_big_count_two_servers)
{
    AllReduceDeterCheckerFor910B(2, 8, 2 * 8 * BIG_COUNT);
}

TEST_F(AllReduceTest, all_reduce_executor_deter_big_count_normal)
{
    AllReduceDeterCheckerFor910B(3, 4, 3 * 4 * BIG_COUNT);
}

TEST_F(AllReduceTest, all_reduce_executor_deter_big_count_not_power_of_two_intraRanks)
{
    AllReduceDeterCheckerFor910B(4, 3, 4 * 3 * BIG_COUNT);
}

TEST_F(AllReduceTest, all_reduce_executor_deter_big_count_ax)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
    setenv("HCCL_DETERMINISTIC", "true", 1);
    setenv("HCCL_ALGO", "level0:NA;level1:ring", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = BIG_COUNT * 16;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceDeterPipelineExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, all_reduce_executor_deter_very_big_count)
{
    AllReduceDeterCheckerFor910B(2, 3, 1024 * 1024 * 30);
}

TEST_F(AllReduceTest, allreduce_A2_1Server16Rank_midcount_deterministic)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
    setenv("HCCL_DETERMINISTIC", "true", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "AllReduceMeshOpbaseMidCountDeterministicExecutor";

    Checker checker;
    HcclResult ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_A2_2Server8Rank_midcount_deterministic)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);
    setenv("HCCL_DETERMINISTIC", "true", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "AllReduceMeshOpbaseMidCountDeterministicExecutor";

    Checker checker;
    HcclResult ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_A3_AllReducemidcountExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllREDUCE";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllReduceMidCountFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, AllReduceOrderPreservedFor91093Executor1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);
    // setenv("HCCL_DETERMINISTIC", "STRICT", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllREDUCE";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceOrderPreservedFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, AllReduceOrderPreservedFor91093Executor2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 4);
    // setenv("HCCL_DETERMINISTIC", "STRICT", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllREDUCE";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceOrderPreservedFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, AllReduceOrderPreservedFor91093Executor3)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 8);
    // setenv("HCCL_DETERMINISTIC", "STRICT", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllREDUCE";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceOrderPreservedFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_order_preserved_multi_node_single_rank_910b)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "AllReduceOrderPreservedExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_order_preserved_multi_node_single_rank_91093_opbase)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 4, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllReduceOrderPreservedFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_order_preserved_multi_node_single_rank_91093_offload)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 4, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllReduceOrderPreservedFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_order_preserved_multi_node_single_rank_small_count_910b)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 8, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 64;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "AllReduceOrderPreservedExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_order_preserved_multi_node_single_rank_small_count_91093)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 4, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllReduceOrderPreservedFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_order_preserved_multi_node_single_rank_fp16_910b)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 512;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "AllReduceOrderPreservedExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllReduceTest, allreduce_order_preserved_multi_node_16server_single_rank_910b)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 16, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 2048;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "AllReduceOrderPreservedExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
