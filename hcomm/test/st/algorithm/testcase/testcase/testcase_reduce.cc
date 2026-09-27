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

#include <vector>
#include <iostream>

#include "topoinfo_struct.h"
#include "log.h"
#include "checker_def.h"
#include "topo_meta.h"
#include "testcase_utils.h"
#include "checker.h"
using namespace checker;

class ReduceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "ReduceTest tear down." << std::endl; }

    virtual void SetUp()
    {
        const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string caseName
            = "analysis_result_" + std::string(test_info->test_case_name()) + "_" + std::string(test_info->name());
        Checker::SetDumpFileName(caseName);
    }

    virtual void TearDown()
    {
        Checker::SetDumpFileName("analysis_result");
        // GlobalMockObject::verify();
        // 这边每个case执行完成需要清理所有的环境变量，如果有新增的环境变量，需要在这个函数中进行清理
        ClearHcclEnv();
    }
};

TEST_F(ReduceTest, reduce_test_opbase_910B_loop)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    HcclResult ret;
    for (int i = 1; i <= 2; i++) {
        for (int j = 1; j <= 8; j++) {
            Checker checker;
            TopoMeta topoMeta;
            gen.GenTopoMeta(topoMeta, 1, i, j);
            checker.CloseRankMemCheck();
            ret = checker.Check(checkerOpParam, topoMeta);
            EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
        }
    }
}

TEST_F(ReduceTest, reduce_test_offload_910B_loop)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    HcclResult ret;
    for (int i = 1; i <= 2; i++) {
        for (int j = 1; j <= 8; j++) {
            Checker checker;
            TopoMeta topoMeta;
            gen.GenTopoMeta(topoMeta, 1, i, j);
            checker.CloseRankMemCheck();
            ret = checker.Check(checkerOpParam, topoMeta);
            EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
        }
    }
}

TEST_F(ReduceTest, reduce_test_offload_910A)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    HcclResult ret;

    Checker checker;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceTest, reduce_test_opbase_910A)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    HcclResult ret;
    Checker checker;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceTest, reduce_test_opbase_910_93)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    HcclResult ret;
    Checker checker;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceTest, reduce_test_opbase_offload_93_2pods)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    HcclResult ret;
    Checker checker;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 8);
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceTest, reduce_test_opbase_ReduceComm)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.algName = "ReduceComm";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    HcclResult ret;
    Checker checker;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceTest, reduce_test_opbase_ReduceMeshExecutor)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.algName = "ReduceMeshExecutor";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    HcclResult ret;
    Checker checker;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceTest, reduce_test_opbase_ReduceRingPlusHd)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.algName = "ReduceRingPlusHd";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    HcclResult ret;
    Checker checker;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceTest, reduce_test_opbase_ReduceRingPlusHd_singleRank)
{
    RankTable_For_LLT gen;

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.algName = "ReduceRingPlusHd";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    HcclResult ret;
    Checker checker;

    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceTest, reduce_test_opbase_ReduceSingleExecutor)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.algName = "ReduceSingleExecutor";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    HcclResult ret;
    Checker checker;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 1);
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceTest, reduce_test_opbase_910_93_2superpod_ReduceMeshExecutor_postSync)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    checkerOpParam.aicpuUnfoldMode = true;
    checkerOpParam.algOpContext.opRetryHandler.isPostSync = true;

    HcclResult ret;
    Checker checker;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 1);
    ret = checker.Check(checkerOpParam, topoMeta);
}

TEST_F(ReduceTest, reduce_test_opbase_910_93_2superpod_4servers_4ranks_Reduce)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 4, 4);

    setenv("HCCL_INTER_HCCS_DISABLE", "TRUE", 1);
    setenv("HCCL_ALGO", "level1:ring", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 8192;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.root = 0;

    HcclResult ret;
    Checker checker;

    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceTest, reduce_superpod_asym_gcd)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 10000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceTest, reduce_test_ReduceRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 5, 1, 3);

    setenv("HCCL_ALGO", "level0:NA;level1:ring;level2:NB", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 8192;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "ReduceRingFor91093Executor";
    checkerOpParam.root = 4;

    HcclResult ret;
    Checker checker;

    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceTest, reduce_test_opbase_910B_Reduce_NSLB)
{
    RankTable_For_LLT gen;
    setenv("HCCL_ALGO", "level0:ring;level1:ring;level2:ring", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.algName = "ReduceRingPlusHd";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    HcclResult ret;
    Checker checker;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
