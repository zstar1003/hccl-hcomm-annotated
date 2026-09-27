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
#include "coll_reduce_scatter_v_executor.h"
#include "checker.h"
using namespace checker;

class ReduceScatterVTestFor310P : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceScatterVTestFor310P set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "ReduceScatterVTestFor310P tear down." << std::endl; }
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

TEST_F(ReduceScatterVTestFor310P, reduce_scatter_v_310P3_opbase_ReduceScatterRing)
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
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVFor310PRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTestFor310P, reduce_scatter_v_310P3_opbase_ReduceScatterRing_varying_size)
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
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVFor310PRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTestFor310P, reduce_scatter_v_310P3_opbase_ReduceScatterRing_varying_size_2)
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
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVFor310PRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTestFor310P, reduce_scatter_v_310P3_opbase_ReduceScatterRing_varying_size_3)
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
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVFor310PRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTestFor310P, reduce_scatter_v_310P3_opbase_ReduceScatterRing_varying_size_4)
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
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVFor310PRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTestFor310P, reduce_scatter_v_310P3_opbase_ReduceScatterRing_varying_size_large_size)
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
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVFor310PRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTestFor310P, reduce_scatter_v_310P3_opbase_ReduceScatterRing_varying_size_tbe)
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
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVFor310PRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTestFor310P, reduce_scatter_v_310P3_opbase_ReduceScatterRing_varying_size_6)
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
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVFor310PRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTestFor310P, reduce_scatter_v_310P3_opbase_ReduceScatterRing_varying_size_7)
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
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVFor310PRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTestFor310P, reduce_scatter_v_310P3_opbase_ReduceScatterRing_varying_size_8)
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
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVFor310PRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(ReduceScatterVTestFor310P, reduce_scatter_v_310P3_opbase_ReduceScatterRing_varying_size_containes_zero)
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
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "ReduceScatterVFor310PRing";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
