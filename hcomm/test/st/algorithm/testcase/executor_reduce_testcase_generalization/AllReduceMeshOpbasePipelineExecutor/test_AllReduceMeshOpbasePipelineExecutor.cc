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
#include <tuple>
#include <iostream>

#include "topoinfo_struct.h"
#include "log.h"
#include "checker_def.h"
#include "topo_meta.h"
#include "testcase_utils.h"
#include "coll_native_executor_base.h"

#include "checker.h"
using namespace checker;

class AllReduceGeneralizationTest : public ::testing::TestWithParam<std::tuple<int, CheckerDataType, vector<int>>> {
public:
    static void SetUpTestCase() { std::cout << "AllReduceMeshOpbasePipelineExecutor set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AllReduceMeshOpbasePipelineExecutor tear down." << std::endl; }

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

TEST_P(AllReduceGeneralizationTest, all_reduce_a_test)
{
    const auto& mytuple = GetParam();
    // 打印一下mytuple当前的参数
    std::cout << "checkerOpParam.DataDes.count : " << std::get<0>(mytuple) << std::endl;
    std::cout << "checkerOpParam.DataDes.dataType : " << std::get<1>(mytuple) << std::endl;
    std::cout << "checkerOpParam.TopoMeta : " << std::get<2>(mytuple)[0] << " , " << std::get<2>(mytuple)[1] << " , "
              << std::get<2>(mytuple)[2] << std::endl;

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, std::get<2>(mytuple)[0], std::get<2>(mytuple)[1], std::get<2>(mytuple)[2]);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.algName = "AllReduceMeshOpbasePipelineExecutor";
    u64 datacount = std::get<0>(mytuple);
    checkerOpParam.DataDes.count = datacount / SIZE_TABLE[std::get<1>(mytuple)];
    if (datacount > 1000000008) {
        setenv("HCCL_BUFFSIZE", "4096", 1);
    };
    checkerOpParam.DataDes.dataType = std::get<1>(mytuple);
    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    AllReduceCaseTest, AllReduceGeneralizationTest,
    testing::Combine(
        testing::Values(800, 1000000008, 5000000008),
        testing::Values(
            CheckerDataType::DATA_TYPE_FP32, CheckerDataType::DATA_TYPE_INT8, CheckerDataType::DATA_TYPE_BFP16),
        testing::ValuesIn(std::vector<std::vector<int>>{{1, 1, 16}, {1, 4, 16}, {1, 2, 8}, {1, 2, 7}})));
