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
#include "gtest/gtest.h"
#include "topoinfo_struct.h"
#include "log.h"
#include "checker_def.h"
#include "topo_meta.h"
#include "testcase_utils.h"
#include "coll_reduce_scatter_executor.h"
#include "hccl_aiv.h"
#include "checker.h"
using namespace checker;
using namespace hccl;

std::vector<string> reduceScatterAlgName = {
    "ReduceScatterDeterExecutor",
    "ReduceScatterMeshExecutor",
    "ReduceScatterMeshOpbasePipelineExecutor",
    "ReduceScatterRingExecutor",
    "ReduceScatterMeshAivFor91093Executor",
};

class ReduceScatterTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceScatterTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "ReduceScatterTest tear down." << std::endl; }

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

TEST_F(ReduceScatterTest, reduce_scatter_executor_310P_opbase_ReduceScatterSlimRingExecutorINT8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterSlimRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_310P_opbase_ReduceScatterSlimRingExecutorFP32)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterSlimRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
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

TEST_F(ReduceScatterTest, reduce_scatter_getaivexecpara)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "ReduceScatterMeshAivExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
}

TEST_F(ReduceScatterTest, reduce_scatter_test_loop)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
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
            EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
        }
    }
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_opbase_ReduceScatterDeterExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 10000000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterDeterExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_offload_ReduceScatterMeshExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);
    setenv("HCCL_DETERMINISTIC", "true", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterMeshExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_offload_ReduceScatterMeshGraphExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 4096;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterMeshGraphExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_offload_1Server_ReduceScatterMeshGraphExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 4096;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_offload_2Server_ReduceScatterMeshGraphPipelineExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 4096;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_offload_2Server_per7_ReduceScatterMeshGraphPipelineExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 7);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_offload_3Server_ReduceScatterMeshGraphPipelineExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 3, 8);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 4096;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_offload_4Server_ReduceScatterMeshGraphPipelineExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 8);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 4096;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_offload_4Server_per7_ReduceScatterMeshGraphPipelineExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 7);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 4096;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_offload_8Server_ReduceScatterMeshGraphPipelineExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 8, 8);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 4096;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_offload_NHR_ReduceScatterMeshGraphExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);
    setenv("HCCL_ALGO", "level1:NHR", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 4096;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterMeshGraphExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_offload_ReduceScatterRingExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterRingExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_ReduceScatterRingExecutor_NSLB)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 3, 8);
    setenv("HCCL_ALGO", "level0:ring;level1:H-D_R", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterRingExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// 仅测试用，不同算法限制不同
TEST_F(ReduceScatterTest, reduce_scatter_executor_test_loop)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    for (int i = 0; i < reduceScatterAlgName.size(); i++) {
        checkerOpParam.algName = reduceScatterAlgName[i];
        Checker checker;
        HcclResult ret;
        checker.CloseRankMemCheck();
        ret = checker.Check(checkerOpParam, topoMeta);
        if (ret == HcclResult::HCCL_SUCCESS) {
            std::cout << reduceScatterAlgName[i] + " run succeeded" << std::endl;
        } else {
            std::cout << reduceScatterAlgName[i] + " run failed" << std::endl;
        }
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
}

TEST_F(ReduceScatterTest, reduce_scatter_ReduceScatterComm_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterComm";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_ReduceScatterSingleExecutor_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    checkerOpParam.algName = "ReduceScatterSingleExecutor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910A_offload_ReduceScatterMeshExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterMeshExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910A_offload_ReduceScatterRingExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterRingExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910A_offload_ReduceScatterComm)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{2, 5}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterComm";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910A_opbase_ReduceScatterMeshExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterMeshExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910A_opbase_ReduceScatterRingExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterRingExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910A_opbase_ReduceScatterComm)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{2, 5}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterComm";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// 适配910_93算法
TEST_F(ReduceScatterTest, reduce_scatter_910_93_opbase_ReduceScatterFastDoubleRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);
    setenv("HCCL_DETERMINISTIC", "true", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterFastDoubleRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_opbase_ReduceScatterFastDoubleRingFor91093Executor_AvoidCceRewrite)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 2; // not CCE reduce aligned(32B)
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT64;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterFastDoubleRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_opbase_ReduceScatterFastDoubleRingFor91093Executor_tbe)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);
    setenv("HCCL_DETERMINISTIC", "true", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 4 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT64;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterFastDoubleRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_offload_ReduceScatterFastDoubleRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterFastDoubleRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_opbase_AlignedReduceScatterDoubleRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AlignedReduceScatterDoubleRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_opbase_MultiSuperPod_2_6_2_ReduceScatterDoubleRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 6, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_opbase_MultiSuperPod_4_3_2_ReduceScatterDoubleRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 4, 3, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_offload_AlignedReduceScatterDoubleRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AlignedReduceScatterDoubleRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_offload_ReduceScatterDoubleRingConcurrentExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    setenv("HCCL_CONCURRENT_ENABLE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_opbase_ReduceScatterDoubleRingConcurrentExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    setenv("HCCL_CONCURRENT_ENABLE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_310P3_opbase_ReduceScatterRing)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_310P3_offload_ReduceScatterRing)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_opbase_accuracy_double_pod)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_retry_inplace_presync_case_level0)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 134217728 * 2;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AlignedReduceScatterDoubleRingFor91093Executor";
    checkerOpParam.algOpContext.opRetryHandler.isInplacePreSync = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    // PreSync会memcopy一段无效数据以占满所有端口，从而达到检测断链的效果，所以数据本身没有被使用，是允许的。
    // 且PreSync不执行任何reduce scatter算法。
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_retry_inplace_presync_case_level1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 8, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 134217728 * 2;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AlignedReduceScatterDoubleRingFor91093Executor";
    checkerOpParam.algOpContext.opRetryHandler.isInplacePreSync = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    // PreSync会memcopy一段无效数据以占满所有端口，从而达到检测断链的效果，所以数据本身没有被使用，是允许的。
    // 且PreSync不执行任何reduce scatter算法。
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_retry_inplace_presync_case_level2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 134217728 * 2;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AlignedReduceScatterDoubleRingFor91093Executor";
    checkerOpParam.algOpContext.opRetryHandler.isInplacePreSync = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    // PreSync会memcopy一段无效数据以占满所有端口，从而达到检测断链的效果，所以数据本身没有被使用，是允许的。
    // 且PreSync不执行任何reduce scatter算法。
}
// 适配910_93算法
TEST_F(ReduceScatterTest, reduce_scatter_smallcount_multiserver_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
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

TEST_F(ReduceScatterTest, reduce_scatter_910_93_offload_ReduceScatterMeshAtomic)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterMeshAtomic";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_offload_ReduceScatterSemiRingExecutor_inlineReduce_singleServer)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8); // rankSize = 16, 14, 12, 10, 8, 6, 4

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "ReduceScatterSemiRingExecutor";
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_offload_ReduceScatterSemiRingExecutor_inlineReduce_singleServer_oddNum)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 55;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "ReduceScatterSemiRingExecutor";
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_offload_ReduceScatterSemiRingExecutor_tbeReduce)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 12); // rankSize = 16, 14, 12, 10, 8, 6, 4

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT64;
    checkerOpParam.algName = "ReduceScatterSemiRingExecutor";
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_ax_4server_16p)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
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

TEST_F(ReduceScatterTest, reduce_scatter_superpod_asym_gcd)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{
        {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
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

TEST_F(ReduceScatterTest, reduce_scatter_superpod_asym_gcd_graph)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
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

TEST_F(ReduceScatterTest, reduce_scatter_mix_ReduceScatterMixExecutor)
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
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910B);
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910_93);

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_mix_ReduceScatterMixExecutor_deter)
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
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
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

TEST_F(ReduceScatterTest, reduce_scatter_mix_ReduceScatterMixExecutor_deter_comm)
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
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
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

TEST_F(ReduceScatterTest, reduce_scatter_aiv_a3_ReduceScatterMeshAivSmallCountExecutor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.aicpuUnfoldMode = true;
    checkerOpParam.algName = "ReduceScatterMeshAivSmallCountExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_aiv_a3_ReduceScatterMeshAivExecutor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.aicpuUnfoldMode = true;
    checkerOpParam.algName = "ReduceScatterMeshAivExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_aiv_a3_ReduceScatterMeshAivFor91093Executor1)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.aicpuUnfoldMode = true;
    checkerOpParam.algName = "ReduceScatterMeshAivFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_order_preserved_multi_node_single_rank_910b)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    // 4节点 × 1 rank/节点
    gen.GenTopoMeta(topoMeta, 1, 4, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatterOrderPreserved";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "ReduceScatterOrderPreservedExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_order_preserved_multi_node_single_rank_91093)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    // 4节点 × 1 rank/节点
    gen.GenTopoMeta(topoMeta, 4, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatterOrderPreserved";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "ReduceScatterOrderPreservedFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_order_preserved_multi_node_single_rank_91093_offload)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 4, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatterOrderPreserved";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "ReduceScatterOrderPreservedFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_order_preserved_multi_node_8server_single_rank_910b)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 8, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatterOrderPreserved";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 2048;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "ReduceScatterOrderPreservedExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_order_preserved_multi_node_single_rank_small_count_910b)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 8, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatterOrderPreserved";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 64;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "ReduceScatterOrderPreservedExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_order_preserved_multi_node_single_rank_fp16_910b)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatterOrderPreserved";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 512;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "ReduceScatterOrderPreservedExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_order_preserved_multi_node_single_rank_int32_91093)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 4, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatterOrderPreserved";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "ReduceScatterOrderPreservedFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_aiv_a3_ReduceScatterMeshAivFor91093Executor2)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 64 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.aicpuUnfoldMode = true;
    checkerOpParam.algName = "ReduceScatterMeshAivFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_smallcount_rhd_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_ReduceScatterRingFor91093Executor_NHR)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:NHR;level2:NHR", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "ReduceScatterRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_ReduceScatterRingFor91093Executor_NB)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:NB;level2:NB", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "ReduceScatterRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_A3_2Server1Rank_nb_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 1);

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
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

TEST_F(ReduceScatterTest, reduce_scatter_aiv_a2_ReduceScatterAivRdmaExecutor_2_servers)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.aicpuUnfoldMode = false;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    // 目前机内部分的AIV kernel不支持执行，所以暂不检查是否成功
}

TEST_F(ReduceScatterTest, reduce_scatter_aiv_ReduceScatterMeshAivSmallCountExecutor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.aicpuUnfoldMode = true;
    checkerOpParam.algName = "ReduceScatterMeshAivSmallCountExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    // GlobalMockObject::verify();
}

TEST_F(ReduceScatterTest, reduce_scatter_aiv_a2_ReduceScatterAivRdmaExecutor_3_servers)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 3, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.aicpuUnfoldMode = false;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    // 机内部分的AIV kernel不支持执行，所以不检查是否成功
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_opbase_zcopy_ReduceScatterRingZerocopyExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterRingZerocopyExecutor";
    checkerOpParam.isZeroCopy = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_opbase_zcopy_ReduceScatterRingZerocopyExecutor_sixserver)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 6);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterRingZerocopyExecutor";
    checkerOpParam.isZeroCopy = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_opbase_zcopy_ReduceScatterRingZerocopyExecutor_AHC)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 4);

    setenv("HCCL_ALGO", "level1:AHC", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterRingZerocopyExecutor";
    checkerOpParam.isZeroCopy = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_opbase_zcopy_ReduceScatterRingZerocopyExchangeExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterRingZerocopyExchangeExecutor";
    checkerOpParam.isZeroCopy = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_A2_1Server8Rank_order_preserved)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.aicpuUnfoldMode = true;

    setenv("HCCL_DETERMINISTIC", "strict", 1);
    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    Checker checker;
    HcclResult ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_A2_1server16Rank_order_preserved)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "ReduceScatterOrderPreservedExecutor";

    Checker checker;
    HcclResult ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_A2_2Server16Rank_order_preserved)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "ReduceScatterOrderPreservedExecutor";

    Checker checker;
    HcclResult ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
TEST_F(ReduceScatterTest, reduce_scatter_910_93_opbase_2die_hccs_sio)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "ReduceScatterHccsSioExecutor";
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_910_93_offload_2die_hccs_sio)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "ReduceScatterHccsSioExecutor";
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_aiv_mesh_small_deter)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    setenv("HCCL_DETERMINISTIC", "true", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AIV", 1);

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 1000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterAivDeterSmallExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_aiv_mesh_deter)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    setenv("HCCL_DETERMINISTIC", "true", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AIV", 1);

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 8 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterAivDeterExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_aiv_mesh_mid_deter)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    setenv("HCCL_DETERMINISTIC", "true", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AIV", 1);

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterAivDeterSmallExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_aiv_deter_priority_test)
{
    setenv("HCCL_DETERMINISTIC", "true", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AIV", 1);

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

constexpr int BIG_COUNT = 2 * 1024 * 1024 + 1;
void ReduceScatterDeterCheckerFor910B(int serverNum, int intraRankNum, int count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, serverNum, intraRankNum);
    setenv("HCCL_DETERMINISTIC", "true", 1);
    setenv("HCCL_ALGO", "level0:NA;level1:ring", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = count;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterDeterPipelineExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_deter_big_count_two_servers)
{
    ReduceScatterDeterCheckerFor910B(2, 8, BIG_COUNT);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_deter_big_count_normal)
{
    ReduceScatterDeterCheckerFor910B(4, 4, BIG_COUNT);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_deter_big_count_not_power_of_two_num_servers)
{
    ReduceScatterDeterCheckerFor910B(3, 8, BIG_COUNT);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_deter_big_count_not_power_of_two_intraRanks)
{
    ReduceScatterDeterCheckerFor910B(4, 3, BIG_COUNT);
}

TEST_F(ReduceScatterTest, reduce_scatter_executor_deter_big_count_ax)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
    setenv("HCCL_DETERMINISTIC", "true", 1);
    setenv("HCCL_ALGO", "level0:NA;level1:ring", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = BIG_COUNT;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterDeterPipelineExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, ReduceScatterOrderPreservedFor91093Executor1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);
    // setenv("HCCL_DETERMINISTIC", "STRICT", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterOrderPreservedFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, ReduceScatterOrderPreservedFor91093Executor2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 4);
    // setenv("HCCL_DETERMINISTIC", "STRICT", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterOrderPreservedFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, ReduceScatterOrderPreservedFor91093Executor3)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 8);
    // setenv("HCCL_DETERMINISTIC", "STRICT", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterOrderPreservedFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterTest, ReduceScatterMeshOpbaseSmallCountDeterministicExecutor_NB_3Server)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 3, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:NB;level2:NB", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 8;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "ReduceScatterMeshOpbaseSmallCountDeterministicExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// Case 1: DR + RING + 多server; 4块等大(偶数), numLoopTotal=5
TEST_F(ReduceScatterTest, reduce_scatter_910_93_pipeline_DR_2pod_2server_4blk_equal)
{
    constexpr u32 p = 2, s = 2, r = 8, rankSize = p * s * r;
    constexpr u32 buffSizeMB = 1, unitSize = 4;
    constexpr u64 countDataPerLoop = buffSizeMB * 1024 * 1024 / rankSize / 2 / 128 * 128 / unitSize; // 4096

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, p, s, r);

    setenv("HCCL_BUFFSIZE", std::to_string(buffSizeMB).c_str(), 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 4 * countDataPerLoop; // 16384
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    unsetenv("HCCL_BUFFSIZE");
}

// Case 2: DR + 3SuperPod + 4rank + BFP16/MAX; 3块(奇数) 尾块小落bufferA, numLoopTotal=4
TEST_F(ReduceScatterTest, reduce_scatter_910_93_pipeline_DR_3pod_4rank_bfp16_max_3blk_smallA)
{
    constexpr u32 p = 3, s = 1, r = 4, rankSize = p * s * r;
    constexpr u32 buffSizeMB = 1, unitSize = 2;
    constexpr u64 countDataPerLoop = buffSizeMB * 1024 * 1024 / rankSize / 2 / 128 * 128 / unitSize; // 21824
    constexpr u64 numFullBlocks = 2;
    constexpr u64 countDataLastLoop = countDataPerLoop / 4; // 5456

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, p, s, r);

    setenv("HCCL_BUFFSIZE", std::to_string(buffSizeMB).c_str(), 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = numFullBlocks * countDataPerLoop + countDataLastLoop; // 49104
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MAX;
    checkerOpParam.algName = "ReduceScatterPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    unsetenv("HCCL_BUFFSIZE");
}

// Case 3: DR + FP32/MIN; 4块(偶数) 尾块小落bufferB, numLoopTotal=5
TEST_F(ReduceScatterTest, reduce_scatter_910_93_pipeline_DR_fp32_min_4blk_smallB)
{
    constexpr u32 p = 2, s = 1, r = 8, rankSize = p * s * r;
    constexpr u32 buffSizeMB = 1, unitSize = 4;
    constexpr u64 countDataPerLoop = buffSizeMB * 1024 * 1024 / rankSize / 2 / 128 * 128 / unitSize; // 8192
    constexpr u64 numFullBlocks = 3;
    constexpr u64 countDataLastLoop = countDataPerLoop / 4; // 2048

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, p, s, r);

    setenv("HCCL_BUFFSIZE", std::to_string(buffSizeMB).c_str(), 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = numFullBlocks * countDataPerLoop + countDataLastLoop; // 26624
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MIN;
    checkerOpParam.algName = "ReduceScatterPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    unsetenv("HCCL_BUFFSIZE");
}

// Case 4: DR + force RING L1/L2 + INT8/SUM; 覆盖默认不覆盖的Level1/Level2 RING路径, numLoopTotal=4
TEST_F(ReduceScatterTest, reduce_scatter_910_93_pipeline_DR_force_ring_int8_sum)
{
    constexpr u32 p = 2, s = 2, r = 8, rankSize = p * s * r;
    constexpr u32 buffSizeMB = 1, unitSize = 1;
    constexpr u64 countDataPerLoop = buffSizeMB * 1024 * 1024 / rankSize / 2 / 128 * 128 / unitSize; // 16384

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, p, s, r);

    setenv("HCCL_BUFFSIZE", std::to_string(buffSizeMB).c_str(), 1);
    setenv("HCCL_ALGO", "level1:ring;level2:ring", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 3 * countDataPerLoop; // 49152
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    unsetenv("HCCL_ALGO");
    unsetenv("HCCL_BUFFSIZE");
}

// Case 5: DR + force NB L1/L2 + INT16/MAX; 覆盖Level1/Level2 NB路径, numLoopTotal=4
TEST_F(ReduceScatterTest, reduce_scatter_910_93_pipeline_DR_force_nb_int16_max)
{
    constexpr u32 p = 2, s = 2, r = 8, rankSize = p * s * r;
    constexpr u32 buffSizeMB = 1, unitSize = 2;
    constexpr u64 countDataPerLoop = buffSizeMB * 1024 * 1024 / rankSize / 2 / 128 * 128 / unitSize; // 8192

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, p, s, r);

    setenv("HCCL_BUFFSIZE", std::to_string(buffSizeMB).c_str(), 1);
    setenv("HCCL_ALGO", "level1:NB;level2:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 3 * countDataPerLoop; // 24576
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MAX;
    checkerOpParam.algName = "ReduceScatterPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    unsetenv("HCCL_ALGO");
    unsetenv("HCCL_BUFFSIZE");
}

// Case 6: SR + 3SuperPod + 奇数rank(7) + BFP16/SUM; Level1默认RING(server=1) + Level2默认NHR, numLoopTotal=4
TEST_F(ReduceScatterTest, reduce_scatter_910_93_pipeline_SR_3pod_7rank_bfp16_sum)
{
    constexpr u32 p = 3, s = 1, r = 7, rankSize = p * s * r;
    constexpr u32 buffSizeMB = 1, unitSize = 2;
    constexpr u64 countDataPerLoop = buffSizeMB * 1024 * 1024 / rankSize / 2 / 128 * 128 / unitSize; // 12480

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, p, s, r);

    setenv("HCCL_BUFFSIZE", std::to_string(buffSizeMB).c_str(), 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 3 * countDataPerLoop; // 37440
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    unsetenv("HCCL_BUFFSIZE");
}

// Case 7: SR + 2pod + 3server + rank/server=1边界 + INT32/SUM; L0退化, L1默认RING(3server) + L2默认NHR, numLoopTotal=4
TEST_F(ReduceScatterTest, reduce_scatter_910_93_pipeline_SR_2pod_3server_1rank_int32_sum)
{
    constexpr u32 p = 2, s = 3, r = 1, rankSize = p * s * r;
    constexpr u32 buffSizeMB = 1, unitSize = 4;
    constexpr u64 countDataPerLoop = buffSizeMB * 1024 * 1024 / rankSize / 2 / 128 * 128 / unitSize; // 21824

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, p, s, r);

    setenv("HCCL_BUFFSIZE", std::to_string(buffSizeMB).c_str(), 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 3 * countDataPerLoop; // 65472
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    unsetenv("HCCL_BUFFSIZE");
}

// Case 8: DR + 小数据量退化串行 + FP16/SUM; numBlockTotal=1, Pipeline内部单块无并行重叠
TEST_F(ReduceScatterTest, reduce_scatter_910_93_pipeline_DR_one_block_degenerate_fp16_sum)
{
    constexpr u32 p = 2, s = 1, r = 8;
    constexpr u32 buffSizeMB = 200;

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, p, s, r);

    setenv("HCCL_BUFFSIZE", std::to_string(buffSizeMB).c_str(), 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 8192;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    unsetenv("HCCL_BUFFSIZE");
}
