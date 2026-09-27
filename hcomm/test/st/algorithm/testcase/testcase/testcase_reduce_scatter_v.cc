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
#include "coll_reduce_scatter_v_executor.h"
#include "checker.h"
using namespace checker;

class ReduceScatterVTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceScatterVTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "ReduceScatterVTest tear down." << std::endl; }
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
        // 这边每个case执行完成需要清理所有的环境变量，如果有新增的环境变量，需要在这个函数中进行清理
        ClearHcclEnv();
    }
};

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_opbase_ReduceScatterRing)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    vector<u64> counts;
    vector<u64> displs;
    u64 displacement = 0;
    const u64 count = 100;
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    for (u32 i = 0; i < rankNum; i++) {
        counts.emplace_back(count);
        displs.emplace_back(displacement);
        displacement += count;
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshOpbaseExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_opbase_ReduceScatterRing_varying_size)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    vector<u64> counts{100, 200, 300, 400};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshOpbaseExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_opbase_ReduceScatterRing_varying_size_2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    vector<u64> counts{400, 300, 200, 100};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshOpbaseExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_opbase_ReduceScatterRing_varying_size_3)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    vector<u64> counts{400, 300};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshOpbaseExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_opbase_ReduceScatterRing_varying_size_4)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 5);

    vector<u64> counts{100, 200, 300, 200, 100};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshOpbaseExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_opbase_ReduceScatterRing_varying_size_large_size)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    vector<u64> counts{40000000, 30000000, 20000000, 100};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshOpbaseExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_opbase_ReduceScatterRing_varying_size_tbe)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    vector<u64> counts{400, 300, 200, 100};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshOpbaseExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_opbase_ReduceScatterRing_varying_size_6)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 6);

    vector<u64> counts{100, 200, 300, 300, 200, 100};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshOpbaseExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_opbase_ReduceScatterRing_varying_size_7)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 5);

    vector<u64> counts{100, 200, 300, 400, 300, 200, 100};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshOpbaseExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_opbase_ReduceScatterRing_varying_size_8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 5);

    vector<u64> counts{100, 200, 300, 400, 400, 300, 200, 100};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshOpbaseExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_opbase_ReduceScatterRing_varying_size_containes_zero)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    vector<u64> counts{400, 0, 200, 0};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshOpbaseExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_offload_ReduceScatterV_varying_size_1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    vector<u64> counts{400, 0, 200, 0, 400, 0, 200, 0};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_offload_ReduceScatterRing_varying_size_2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    vector<u64> counts{100, 200, 300, 400, 400, 300, 200, 100};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_offload_ReduceScatterRing_varying_size_3)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 7);

    vector<u64> counts{100, 200, 300, 400, 400, 300, 200};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MAX;
    checkerOpParam.algName = "ReduceScatterVMeshExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_offload_ReduceScatterRing_varying_size_4)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 7);

    vector<u64> counts{10000000, 20000000, 30000000, 40000000, 40000000, 30000000, 20000000};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_offload_ReduceScatterRing_varying_size_5)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    vector<u64> counts{100, 200, 300, 400, 400, 300, 200, 100};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_offload_ReduceScatterRing_varying_size_6)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    vector<u64> counts{100, 200, 300, 400, 400, 300, 200, 100};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_MAX;
    checkerOpParam.algName = "ReduceScatterVMeshExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_offload_ReduceScatterRing_varying_size_7)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    vector<u64> counts{100, 200};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_offload_ReduceScatterRing_one_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterSingleExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_opbase_ReduceScatterRing_one_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterSingleExecutor";

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_ReduceScatterVMeshAivSmallCountExecutor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 5);

    vector<u64> counts{100, 200, 300, 400, 400, 300, 200, 100};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVMeshAivSmallCountExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_910B_ReduceScatterVAIVBigCountExecutor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 5);

    vector<u64> counts{100, 200, 300, 400, 400, 300, 200, 100};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVAIVBigCountExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
}

// 双环场景：rankNum不要设置成奇数，否则可能误走单Ring逻辑
TEST_F(ReduceScatterVTest, reduce_scatter_v_91093_CollReduceScatterVFastDoubleRingFor91093Executor_multiSuperPod)
{
    const u32 podNum = 3;
    const u32 serverNum = 1;
    const u32 rankNum = 4; // even
    const u32 rankSize = podNum * serverNum * rankNum;
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, podNum, serverNum, rankNum);

    constexpr u32 HCCL_BUFFSIZE = 1;
    setenv("HCCL_BUFFSIZE", std::to_string(HCCL_BUFFSIZE).c_str(), 1);

    constexpr u64 BASE_COUNT = HCCL_BUFFSIZE * 1024 * 1024 / rankSize + 1024; // total > HCCL_BUFFSIZE
    vector<u64> counts;
    for (u32 i = 0; i < rankSize; ++i) {
        counts.emplace_back(BASE_COUNT - i * 1000);
    }
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "ReduceScatterVFastDoubleRingFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    unsetenv("HCCL_BUFFSIZE");
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_91093_CollReduceScatterVFastDoubleRingFor91093Executor_multiServer)
{
    const u32 podNum = 1;
    const u32 serverNum = 2;
    const u32 rankNum = 8; // even
    const u32 rankSize = podNum * serverNum * rankNum;
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, podNum, serverNum, rankNum);

    vector<u64> counts;
    for (u32 i = 0; i < rankSize; ++i) {
        counts.emplace_back((i + 1) * 100);
    }
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.algName = "ReduceScatterVFastDoubleRingFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = false;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_91093_CollReduceScatterVFastDoubleRingFor91093Executor_singleServer)
{
    const u32 podNum = 1;
    const u32 serverNum = 1;
    const u32 rankNum = 8; // even
    const u32 rankSize = podNum * serverNum * rankNum;
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, podNum, serverNum, rankNum);

    vector<u64> counts;
    for (u32 i = 0; i < rankSize; ++i) {
        counts.emplace_back((i + 1) * 100);
    }
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD; // take over semi-ring
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_UINT64;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVFastDoubleRingFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_91093_CollReduceScatterVFastDoubleRingFor91093Executor_AvoidCceRewrite)
{
    const u32 podNum = 1;
    const u32 serverNum = 1;
    const u32 rankNum = 4; // even
    const u32 rankSize = podNum * serverNum * rankNum;
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, podNum, serverNum, rankNum);

    vector<u64> counts;
    for (u32 i = 0; i < rankSize; ++i) {
        counts.emplace_back(i + 1); // not CCE_REDUCE_ALIGN_SIZE(32B) aligned, avoid cce rewrite
    }
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT64;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVFastDoubleRingFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// 单环场景：rankNum不要设置成偶数，否则可能误走DoubleRing逻辑
TEST_F(ReduceScatterVTest, reduce_scatter_v_91093_ReduceScatterVRingFor91093Executor_multiSuperPod)
{
    const u32 podNum = 3;
    const u32 serverNum = 1;
    const u32 rankNum = 5;
    const u32 rankSize = podNum * serverNum * rankNum;
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, podNum, serverNum, rankNum);

    vector<u64> counts;
    for (u32 i = 0; i < rankSize; ++i) {
        counts.emplace_back((i + 1) * 100);
    }
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_UINT64;
    checkerOpParam.algName = "ReduceScatterVRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_91093_ReduceScatterVRingFor91093Executor_multiServer)
{
    const u32 podNum = 1;
    const u32 serverNum = 4;
    const u32 rankNum = 3;
    const u32 rankSize = podNum * serverNum * rankNum;
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, podNum, serverNum, rankNum);

    vector<u64> counts;
    for (u32 i = 0; i < rankSize; ++i) {
        counts.emplace_back((i + 1) * 101);
    }
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_PROD;
    checkerOpParam.algName = "ReduceScatterVRingFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_91093_ReduceScatterVRingFor91093Executor_singleServer)
{
    const u32 podNum = 1;
    const u32 serverNum = 1;
    const u32 rankNum = 7;
    const u32 rankSize = podNum * serverNum * rankNum;
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, podNum, serverNum, rankNum);

    constexpr u32 HCCL_BUFFSIZE = 1; // MB
    setenv("HCCL_BUFFSIZE", std::to_string(HCCL_BUFFSIZE).c_str(), 1);

    constexpr u64 BASE_COUNT = HCCL_BUFFSIZE * 1024 * 1024 / rankSize * 2;
    vector<u64> counts;
    for (u32 i = 0; i < rankSize; ++i) {
        counts.emplace_back(BASE_COUNT + i); // > HCCL_BUFFSIZE
    }
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_PROD;
    checkerOpParam.algName = "ReduceScatterVRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    unsetenv("HCCL_BUFFSIZE");
}

// 双环场景：rankNum不要设置成奇数，否则可能误走单Ring逻辑
TEST_F(ReduceScatterVTest, reduce_scatter_v_91093_AlignedReduceScatterVDoubleRingFor91093Executor_multiSuperPod)
{
    const u32 podNum = 2;
    const u32 serverNum = 1;
    const u32 rankNum = 6;
    const u32 rankSize = podNum * serverNum * rankNum;
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, podNum, serverNum, rankNum);

    vector<u64> counts;
    for (u32 i = 0; i < rankSize; ++i) {
        counts.emplace_back((i + 1) * 100);
    }

    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AlignedReduceScatterVDoubleRingFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_91093_AlignedReduceScatterVDoubleRingFor91093Executor_multiServer)
{
    const u32 podNum = 1;
    const u32 serverNum = 2;
    const u32 rankNum = 6;
    const u32 rankSize = podNum * serverNum * rankNum;
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, podNum, serverNum, rankNum);

    vector<u64> counts;
    for (u32 i = 0; i < rankSize; ++i) {
        counts.emplace_back(200); // fix count
    }
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "AlignedReduceScatterVDoubleRingFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTest, reduce_scatter_v_91093_AlignedReduceScatterVDoubleRingFor91093Executor_singleServer)
{
    const u32 podNum = 1;
    const u32 serverNum = 1;
    const u32 rankNum = 12;
    const u32 rankSize = podNum * serverNum * rankNum;
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, podNum, serverNum, rankNum);

    vector<u64> counts;
    for (u32 i = 0; i < rankSize; ++i) {
        counts.emplace_back((i + 1) * 100);
    }
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER_V;
    checkerOpParam.tag = "ReduceScatterV";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD; // take over semi-ring
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "AlignedReduceScatterVDoubleRingFor91093Executor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
