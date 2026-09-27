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

class RunReduceMeshExecutorA2Test :
    public ::testing::TestWithParam<
        std::tuple<uint64_t, CheckerDataType, vector<int>, CheckerOpMode, CheckerDevType, std::string, int>> {
public:
    static void SetUpTestCase() { std::cout << "RunReduceMeshExecutorA2Test set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RunReduceMeshExecutorA2Test tear down." << std::endl; }

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
        ClearHcclEnv();
    }

protected:
    int GetRootValue(int rankSize, int rootParam)
    {
        switch (rootParam) {
            case 0:
                return 0;
                break;
            case 1:
                return rankSize - 1;
                break;
            case 2:
                return rankSize / 2;
                break;
            case 3:
                return std::min(rankSize / 2 + 1, rankSize - 1);
                break;
            default:
                std::cout << "The root node is invalid values." << std::endl;
                return -1;
                break;
        }
    }
};

TEST_P(RunReduceMeshExecutorA2Test, Test_ReduceMesh_A2)
{
    const auto& settingTuple = GetParam();
    uint64_t dataSize = std::get<0>(settingTuple);
    CheckerDataType dataType = std::get<1>(settingTuple);
    const std::vector<int>& topo = std::get<2>(settingTuple);
    CheckerOpMode opMode = std::get<3>(settingTuple);
    CheckerDevType devType = std::get<4>(settingTuple);
    const std::string& hcclAlgo = std::get<5>(settingTuple);
    int rootIndex = std::get<6>(settingTuple);

    if (dataSize == 5000000008ull) {
        setenv("HCCL_BUFFSIZE", "4096", 1);
    }

    if (!hcclAlgo.empty()) {
        std::string hcclAlgoEnv = "level0:NA;level1:" + hcclAlgo;
        setenv("HCCL_ALGO", hcclAlgoEnv.c_str(), 1);
    }

    TopoMeta topoMeta;
    RankTable_For_LLT gen;
    gen.GenTopoMeta(topoMeta, topo[0], topo[1], topo[2]);
    u32 rankSize = GetRankNumFormTopoMeta(topoMeta);
    RankId root = GetRootValue(rankSize, rootIndex);

    std::cout << "--- dataSize=" << dataSize << ", dataType=" << dataType << ", opMode=" << opMode << ", topo={"
              << topo[0] << "," << topo[1] << "," << topo[2] << "}" << ", hcclAlgo=" << hcclAlgo << ", root = " << root
              << std::endl;

    CheckerOpParam checkerOpParam;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.algName = "ReduceMeshExecutor";
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    checkerOpParam.DataDes.dataType = dataType;
    checkerOpParam.DataDes.count = dataSize / SIZE_TABLE[dataType];
    checkerOpParam.opMode = opMode;
    checkerOpParam.devtype = devType;
    checkerOpParam.root = root;

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    ReduceMeshExecutor_A2, RunReduceMeshExecutorA2Test,
    testing::Combine(
        testing::Values(800ull /*, 1000000008ull, 5000000008ull*/),
        testing::Values(
            CheckerDataType::DATA_TYPE_FP32, CheckerDataType::DATA_TYPE_INT8, CheckerDataType::DATA_TYPE_BFP16,
            CheckerDataType::DATA_TYPE_INT64),
        testing::ValuesIn(std::vector<std::vector<int>>{{1, 2, 8}, {1, 2, 7} /*, {1, 1, 16} {1, 4, 16}*/}),
        testing::Values(CheckerOpMode::OPBASE, CheckerOpMode::OFFLOAD), testing::Values(CheckerDevType::DEV_TYPE_910B),
        testing::Values("ring"), testing::Values(0, 1, 2, 3)));

class RunReduceMeshExecutorAlgoA2Test :
    public ::testing::TestWithParam<
        std::tuple<uint64_t, CheckerDataType, vector<int>, CheckerOpMode, CheckerDevType, std::string, bool>> {
public:
    static void SetUpTestCase() { std::cout << "RunReduceMeshExecutorAlgoA2Test set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RunReduceMeshExecutorAlgoA2Test tear down." << std::endl; }

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
        ClearHcclEnv();
    }
};

TEST_P(RunReduceMeshExecutorAlgoA2Test, Test_ReduceMesh_Algo_A2)
{
    const auto& settingTuple = GetParam();
    uint64_t dataSize = std::get<0>(settingTuple);
    CheckerDataType dataType = std::get<1>(settingTuple);
    const std::vector<int>& topo = std::get<2>(settingTuple);
    CheckerOpMode opMode = std::get<3>(settingTuple);
    CheckerDevType devType = std::get<4>(settingTuple);
    const std::string& hcclAlgo = std::get<5>(settingTuple);
    bool enableAnypath = std::get<6>(settingTuple);

    std::cout << "--- dataSize=" << dataSize << ", dataType=" << dataType << ", opMode=" << opMode << ", topo={"
              << topo[0] << "," << topo[1] << "," << topo[2] << "}" << ", hcclAlgo=" << hcclAlgo << ", root = 0"
              << std::endl;

    if (dataSize == 5000000008ull) {
        setenv("HCCL_BUFFSIZE", "4096", 1);
    }

    if (enableAnypath) {
        setenv("HCCL_CONCURRENT_ENABLE", "1", 1);
    }

    if (!hcclAlgo.empty()) {
        std::string hcclAlgoEnv = "level0:NA;level1:" + hcclAlgo;
        setenv("HCCL_ALGO", hcclAlgoEnv.c_str(), 1);
    }

    TopoMeta topoMeta;
    RankTable_For_LLT gen;
    gen.GenTopoMeta(topoMeta, topo[0], topo[1], topo[2]);

    CheckerOpParam checkerOpParam;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.algName = "ReduceMeshExecutor";
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;

    checkerOpParam.DataDes.dataType = dataType;
    checkerOpParam.DataDes.count = dataSize / SIZE_TABLE[dataType];
    checkerOpParam.opMode = opMode;
    checkerOpParam.devtype = devType;
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    ReduceMeshExecutor_Algo_A2, RunReduceMeshExecutorAlgoA2Test,
    testing::Combine(
        testing::Values(1000000008ull), testing::Values(CheckerDataType::DATA_TYPE_FP32),
        testing::ValuesIn(std::vector<std::vector<int>>{{1, 2, 8} /*, {1, 4, 16}*/}),
        testing::Values(CheckerOpMode::OPBASE), testing::Values(CheckerDevType::DEV_TYPE_910B),
        testing::Values("H-D_R"), // Reduce 仅支持 ring HDR 算法
        testing::Values(true)));

class RunReduceMeshExecutorReduceOpA2Test :
    public ::testing::TestWithParam<std::tuple<
        uint64_t, CheckerDataType, vector<int>, CheckerOpMode, CheckerDevType, std::string, CheckerReduceOp>> {
public:
    static void SetUpTestCase() { std::cout << "RunReduceMeshExecutorReduceOpA2Test set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RunReduceMeshExecutorReduceOpA2Test tear down." << std::endl; }

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
        ClearHcclEnv();
    }
};

TEST_P(RunReduceMeshExecutorReduceOpA2Test, Test_ReduceMesh_ReduceOp_A2)
{
    const auto& settingTuple = GetParam();
    uint64_t dataSize = std::get<0>(settingTuple);
    CheckerDataType dataType = std::get<1>(settingTuple);
    const std::vector<int>& topo = std::get<2>(settingTuple);
    CheckerOpMode opMode = std::get<3>(settingTuple);
    CheckerDevType devType = std::get<4>(settingTuple);
    const std::string& hcclAlgo = std::get<5>(settingTuple);
    CheckerReduceOp reduceOp = std::get<6>(settingTuple);

    std::cout << "--- dataSize=" << dataSize << ", dataType=" << dataType << ", opMode=" << opMode << ", topo={"
              << topo[0] << "," << topo[1] << "," << topo[2] << "}" << ", hcclAlgo=" << hcclAlgo << ", root = 0"
              << std::endl;

    if (dataSize == 5000000008ull) {
        setenv("HCCL_BUFFSIZE", "4096", 1);
    }

    if (!hcclAlgo.empty()) {
        std::string hcclAlgoEnv = "level0:NA;level1:" + hcclAlgo;
        setenv("HCCL_ALGO", hcclAlgoEnv.c_str(), 1);
    }

    TopoMeta topoMeta;
    RankTable_For_LLT gen;
    gen.GenTopoMeta(topoMeta, topo[0], topo[1], topo[2]);

    CheckerOpParam checkerOpParam;
    checkerOpParam.tag = "Reduce";
    checkerOpParam.opType = CheckerOpType::REDUCE;
    checkerOpParam.algName = "ReduceMeshExecutor";

    checkerOpParam.DataDes.dataType = dataType;
    checkerOpParam.DataDes.count = dataSize / SIZE_TABLE[dataType];
    checkerOpParam.opMode = opMode;
    checkerOpParam.devtype = devType;
    checkerOpParam.reduceType = reduceOp;
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    ReduceMeshExecutor_ReduceOp_A2, RunReduceMeshExecutorReduceOpA2Test,
    testing::Combine(
        testing::Values(800ull), testing::Values(CheckerDataType::DATA_TYPE_FP32),
        testing::ValuesIn(std::vector<std::vector<int>>{{1, 2, 7} /*, {1, 4, 16}*/}),
        testing::Values(CheckerOpMode::OPBASE), testing::Values(CheckerDevType::DEV_TYPE_910B),
        testing::Values("H-D_R"),
        testing::Values(
            CheckerReduceOp::REDUCE_SUM, CheckerReduceOp::REDUCE_PROD, CheckerReduceOp::REDUCE_MAX,
            CheckerReduceOp::REDUCE_MIN)));
