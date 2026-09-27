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
#include <stdlib.h>

#include "stream_pub.h"
#include "topoinfo_struct.h"
#include "log.h"
#include "checker_def.h"
#include "topo_meta.h"
#include "testcase_utils.h"
#include "checker.h"
#include "dispatcher.h"
using namespace checker;
using namespace hccl;

using namespace std;

class AHCTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CheckOpSemanticsTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CheckOpSemanticsTest tear down." << std::endl; }

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

// all reduce case
TEST_F(AHCTest, all_reduce_910_93_1spod_2server_same_8_ring_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

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

TEST_F(AHCTest, all_reduce_910_93_1spod_2server_same_8_ring_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    setenv("HCCL_ALGO", "level0:ring;level1:AHC_BROKE", 1);

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

TEST_F(AHCTest, all_reduce_910_93_2spod_1server_same_8_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

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

TEST_F(AHCTest, all_reduce_910_93_2spod_1server_same_8_default_test) // default case
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_reduce_910_93_2spod_1server_same_8_default_broke_test) // default case
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_reduce_910_93_2spod_server_diff_1_3_test) // default case
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}}, {{0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 8;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_reduce_910_93_2spod_server_diff_1_3_broke_test) // default case
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}}, {{0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 8;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_reduce_910_93_2spod_server_diff_1_7_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_reduce_910_93_2spod_server_diff_1_7_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_reduce_910_93_4spod_1server_test) // default case
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}}, {{0}}, {{0}}, {{0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, allreduce_910_93_2spod_server_diff_2_3_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1}}, {{0, 1}, {0, 1}, {0, 1}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, allreduce_910_93_2spod_server_diff_2_3_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1}}, {{0, 1}, {0, 1}, {0, 1}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, allreduce_910_93_2spod_server_diff_2_4_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}}, {{0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, allreduce_910_93_2spod_server_diff_2_4_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}}, {{0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, allreduce_910_93_2spod_server_diff_2_8_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, allreduce_910_93_2spod_server_diff_2_8_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_reduce_910_93_2spod_server_diff_3_5_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_reduce_910_93_2spod_server_diff_3_5_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, allreduce_910_93_2spod_server_diff_3_7_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, allreduce_910_93_2spod_server_diff_3_7_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, allreduce_910_93_2spod_server_diff_4_6_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, allreduce_910_93_2spod_server_diff_4_6_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_reduce_910_93_2spod_server_diff_4_8_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_reduce_910_93_2spod_server_diff_4_8_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// reduce scatter case
TEST_F(AHCTest, reduce_scatter_910_93_1spod_2server_same_8_ring_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    setenv("HCCL_ALGO", "level0:ring;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "ReduceScatterFastDoubleRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_1spod_2server_same_8_ring_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    setenv("HCCL_ALGO", "level0:ring;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "ReduceScatterRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_1spod_2server_same_8_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "ReduceScatterRingFor91093Executor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_1spod_2server_same_8_default_test) // default case
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_1spod_2server_same_8_broke_default_test) // default case
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_1_3_test) // default case
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}}, {{0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 8;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_1_3_broke_test) // default case
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}}, {{0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 8;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_1_7_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_1_7_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_2_3_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1}}, {{0, 1}, {0, 1}, {0, 1}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_2_3_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1}}, {{0, 1}, {0, 1}, {0, 1}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_2_4_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}}, {{0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_2_4_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}}, {{0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_2_8_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_2_8_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_3_5_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_3_5_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_3_7_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_3_7_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_4_6_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_4_6_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_4_8_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, reduce_scatter_910_93_2spod_server_diff_4_8_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// all gather
TEST_F(AHCTest, all_gather_910_93_1spod_2server_same_8_ring_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    setenv("HCCL_ALGO", "level0:ring;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AlignedAllGatherDoubleRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_1spod_2server_same_8_ring_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    setenv("HCCL_ALGO", "level0:ring;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_1server_same_8_test) // default case
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_1server_same_8_broke_test) // default case
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 800;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_1_3_test) // default case
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}}, {{0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 8;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_1_3_broke_test) // default case
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}}, {{0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 8;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_1_7_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_1_7_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_2_3_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1}}, {{0, 1}, {0, 1}, {0, 1}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_2_3_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1}}, {{0, 1}, {0, 1}, {0, 1}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_2_4_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}}, {{0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_2_4_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}}, {{0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_2_8_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_2_8_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_3_5_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_3_5_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_3_7_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_3_7_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_4_6_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_4_6_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_4_8_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AHCTest, all_gather_910_93_2spod_server_diff_4_8_broke_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}, {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 80;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
