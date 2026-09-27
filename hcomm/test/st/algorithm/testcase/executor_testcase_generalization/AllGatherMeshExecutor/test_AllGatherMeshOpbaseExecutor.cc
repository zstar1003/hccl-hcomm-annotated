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

class RunAllGatherMeshOpbaseExecutorA2Test :
    public ::testing::TestWithParam<
        std::tuple<uint64_t, CheckerDataType, vector<int>, CheckerOpMode, CheckerDevType, std::string>> {
public:
    static void SetUpTestCase() { std::cout << "RunAllGatherMeshOpbaseExecutorTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RunAllGatherMeshOpbaseExecutorTest tear down." << std::endl; }

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

TEST_P(RunAllGatherMeshOpbaseExecutorA2Test, Test_AllGatherMeshOpbase_A2)
{
    const auto& settingTuple = GetParam();
    uint64_t dataSize = std::get<0>(settingTuple);
    CheckerDataType dataType = std::get<1>(settingTuple);
    const std::vector<int>& topo = std::get<2>(settingTuple);
    CheckerOpMode opMode = std::get<3>(settingTuple);
    CheckerDevType devType = std::get<4>(settingTuple);
    const std::string& hcclAlgo = std::get<5>(settingTuple);

    std::cout << "--- dataCount=" << dataSize << ", dataType=" << dataType << ", --- opMode=" << opMode
              << ", --- topo={" << topo[0] << ", " << topo[1] << ", " << topo[2] << "}" << ", --- hcclAlgo=" << hcclAlgo
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
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.algName = "AllGatherMeshOpbaseExecutor";

    checkerOpParam.DataDes.dataType = dataType;
    checkerOpParam.DataDes.count = dataSize / SIZE_TABLE[dataType];
    ;
    checkerOpParam.opMode = opMode;
    checkerOpParam.devtype = devType;

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMeshOpbaseExecutor_A2, RunAllGatherMeshOpbaseExecutorA2Test,
    testing::Combine(
        testing::Values(800ull /*, 1000000008ull, 5000000008ull*/),
        testing::Values(
            CheckerDataType::DATA_TYPE_FP32, CheckerDataType::DATA_TYPE_INT8, CheckerDataType::DATA_TYPE_BFP16,
            CheckerDataType::DATA_TYPE_INT64),
        testing::ValuesIn(std::vector<std::vector<int>>{{1, 1, 8}}), // MeshOpbase 不支持多 Mesh 拓扑
        testing::Values(CheckerOpMode::OPBASE),                      // 图模式不支持  CheckerOpMode::OFFLOAD
        testing::Values(CheckerDevType::DEV_TYPE_910B), testing::Values("NB")));

class RunAllGatherMeshOpbaseExecutorAlgoA2Test :
    public ::testing::TestWithParam<
        std::tuple<uint64_t, CheckerDataType, vector<int>, CheckerOpMode, CheckerDevType, std::string, bool>> {
public:
    static void SetUpTestCase() { std::cout << "RunAllGatherMeshOpbaseExecutorAlgoTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RunAllGatherMeshOpbaseExecutorAlgoTest tear down." << std::endl; }

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

TEST_P(RunAllGatherMeshOpbaseExecutorAlgoA2Test, Test_AllGatherMesh_Algo_A2)
{
    const auto& settingTuple = GetParam();
    uint64_t dataSize = std::get<0>(settingTuple);
    CheckerDataType dataType = std::get<1>(settingTuple);
    const std::vector<int>& topo = std::get<2>(settingTuple);
    CheckerOpMode opMode = std::get<3>(settingTuple);
    CheckerDevType devType = std::get<4>(settingTuple);
    const std::string& hcclAlgo = std::get<5>(settingTuple);
    bool enableAnypath = std::get<6>(settingTuple);

    std::cout << "--- dataCount=" << dataSize << ", dataType=" << dataType << ", --- opMode=" << opMode
              << ", --- topo={" << topo[0] << ", " << topo[1] << ", " << topo[2] << "}" << ", --- hcclAlgo=" << hcclAlgo
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
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.algName = "AllGatherMeshOpbaseExecutor";

    checkerOpParam.DataDes.dataType = dataType;
    checkerOpParam.DataDes.count = dataSize / SIZE_TABLE[dataType];
    ;
    checkerOpParam.opMode = opMode;
    checkerOpParam.devtype = devType;

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMeshOpbaseExecutor_Algo_A2, RunAllGatherMeshOpbaseExecutorAlgoA2Test,
    testing::Combine(
        testing::Values(800ull /*, 1000000008ull, 5000000008ull*/), testing::Values(CheckerDataType::DATA_TYPE_FP32),
        testing::ValuesIn(std::vector<std::vector<int>>{{1, 1, 8}}),
        testing::Values(CheckerOpMode::OPBASE), // 图模式不支持  CheckerOpMode::OFFLOAD
        testing::Values(CheckerDevType::DEV_TYPE_910B), testing::Values("ring"), testing::Values(true)));
