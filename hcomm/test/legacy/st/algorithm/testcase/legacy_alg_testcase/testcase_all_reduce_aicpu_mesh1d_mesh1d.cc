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
#include <vector>
#include <iostream>
#include <string>
#include "coll_service_stub.h"
#include "checker.h"
#include "testcase_utils.h"
#include "topo_meta.h"

namespace checker {

constexpr u64 K = 1024;
constexpr u64 M = 1024 * K;
constexpr u64 G = 1024 * M;

class AllReduceAICPUMesh1dMesh1dTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AllReduce AICPU ParrallelMesh1DNHR test set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AllReduce AICPU ParrallelMesh1DNHR test tear down" << std::endl; }

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
        GlobalMockObject::verify();
        ClearHcclEnv();
    }

    void RunAllReduceTest(
        TopoMeta& topoMeta, CheckerOpMode opMode, CheckerReduceOp reduceType, u64 dataCount, CheckerDataType dataType)
    {
        setenv("HCCL_IODIE_NUM", "2", 1);
        setenv("HCCL_BUFFSIZE", "200", 1);

        CheckerOpParam checkerOpParam;
        checkerOpParam.opType = CheckerOpType::ALLREDUCE;
        checkerOpParam.tag = "AllReduce";
        checkerOpParam.opMode = opMode;
        checkerOpParam.reduceType = reduceType;
        checkerOpParam.devtype = CheckerDevType::DEV_TYPE_950;
        checkerOpParam.DataDes.count = dataCount;
        checkerOpParam.DataDes.dataType = dataType;
        checkerOpParam.algName = "InsAllReduceFourTemplateMesh1DNHR";

        Checker checker;
        HcclResult ret = checker.CheckA5Aicpu(checkerOpParam, topoMeta);
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
};

TEST_F(AllReduceAICPUMesh1dMesh1dTest, allreduce_aicpu_case_test_2x2_rank_ParallelMesh1DNHR_data_0)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    RunAllReduceTest(
        topoMeta, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, 100, CheckerDataType::DATA_TYPE_INT32);
}

TEST_F(AllReduceAICPUMesh1dMesh1dTest, allreduce_aicpu_case_test_2x2_rank_ParallelMesh1DNHR_data_1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    RunAllReduceTest(
        topoMeta, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, 4096, CheckerDataType::DATA_TYPE_INT32);
}

TEST_F(AllReduceAICPUMesh1dMesh1dTest, allreduce_aicpu_case_test_2x2_rank_ParallelMesh1DNHR_smalldata)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    RunAllReduceTest(topoMeta, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, 1, CheckerDataType::DATA_TYPE_INT32);
}

TEST_F(AllReduceAICPUMesh1dMesh1dTest, allreduce_aicpu_case_test_2x3_rank_ParallelMesh1DNHR)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}};
    RunAllReduceTest(topoMeta, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, 17, CheckerDataType::DATA_TYPE_INT8);
}

TEST_F(AllReduceAICPUMesh1dMesh1dTest, allreduce_aicpu_case_test_2x5_rank_ParallelMesh1DNHR)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}}};
    RunAllReduceTest(
        topoMeta, CheckerOpMode::OFFLOAD, CheckerReduceOp::REDUCE_SUM, 131, CheckerDataType::DATA_TYPE_FP16);
}

TEST_F(AllReduceAICPUMesh1dMesh1dTest, allreduce_aicpu_case_test_3x3_rank_ParallelMesh1DNHR)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    RunAllReduceTest(
        topoMeta, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, 100, CheckerDataType::DATA_TYPE_FP16);
}

TEST_F(AllReduceAICPUMesh1dMesh1dTest, allreduce_aicpu_case_test_3_mul_3_rank_ParallelMesh1DNHR)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};

    RunAllReduceTest(
        topoMeta, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, 100, CheckerDataType::DATA_TYPE_FP16);
}
TEST_F(AllReduceAICPUMesh1dMesh1dTest, allreduce_aicpu_case_test_2_mul_4_rank_ParallelMesh1DNHR_bigdata)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}};

    RunAllReduceTest(
        topoMeta, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, 210 * 1024 * 1024,
        CheckerDataType::DATA_TYPE_FP16);
}

TEST_F(AllReduceAICPUMesh1dMesh1dTest, allreduce_aicpu_case_test_2_mul_8_rank_ParallelMesh1DNHR_1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};

    RunAllReduceTest(
        topoMeta, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, 100, CheckerDataType::DATA_TYPE_FP16);
}

TEST_F(AllReduceAICPUMesh1dMesh1dTest, allreduce_aicpu_case_test_2_mul_8_rank_ParallelMesh1DNHR_2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};

    RunAllReduceTest(
        topoMeta, CheckerOpMode::OPBASE, CheckerReduceOp::REDUCE_SUM, 1025, CheckerDataType::DATA_TYPE_FP16);
}
} // namespace checker
