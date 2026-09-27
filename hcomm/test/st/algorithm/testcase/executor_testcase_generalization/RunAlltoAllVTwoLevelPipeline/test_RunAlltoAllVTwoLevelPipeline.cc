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
#include <string>
#include <tuple>
#include <iostream>

#include "topoinfo_struct.h"
#include "log.h"
#include "checker_def.h"
#include "topo_meta.h"
#include "testcase_utils.h"
#include "checker.h"
using namespace checker;

class RunAlltoAllVTwoLevelPipelineTest :
    public testing::TestWithParam<
        std::tuple<uint64_t, CheckerDataType, CheckerOpMode, std::vector<int>, std::string, bool>> {
public:
    static void SetUpTestCase() { std::cout << "RunAlltoAllVTwoLevelPipelineTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RunAlltoAllVTwoLevelPipelineTest tear down." << std::endl; }

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

TEST_P(RunAlltoAllVTwoLevelPipelineTest, alltoallv_test_910B)
{
    const auto& mytuple = GetParam();

    const uint64_t dataSize = std::get<0>(mytuple);
    const CheckerDataType dataType = std::get<1>(mytuple);
    const CheckerOpMode opMode = std::get<2>(mytuple);
    const std::vector<int>& topoVec = std::get<3>(mytuple);
    const std::string& hccl_algo = std::get<4>(mytuple);
    bool enable_anypath = std::get<5>(mytuple);
    const uint64_t dataCount = dataSize / SIZE_TABLE[dataType];

    cout << "dataSize=" << dataSize << ", dataType=" << dataType << ", opMode=" << opMode << ", topo={" << topoVec[0]
         << "," << topoVec[1] << "," << topoVec[2] << "}, hccl_algo=" << hccl_algo
         << ", enable_anypath=" << enable_anypath << endl;

    if (dataSize == 5000000008ull)
        setenv("HCCL_BUFFSIZE", "4096", 1);

    if (enable_anypath)
        setenv("HCCL_CONCURRENT_ENABLE", "1", 1);

    if (!hccl_algo.empty()) {
        std::string hccl_algo_env = "level0:NA;level1:" + hccl_algo;
        setenv("HCCL_ALGO", hccl_algo_env.c_str(), 1);
    }

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, topoVec[0], topoVec[1], topoVec[2]);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAllV";
    checkerOpParam.opMode = opMode;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVTwoLevelPipeline";

    checkerOpParam.All2AllDataDes.sendType = dataType;
    checkerOpParam.All2AllDataDes.recvType = dataType;
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);

    GenAllToAllVParams(
        rankNum, dataCount, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

std::vector<std::vector<int>> topoVecs1 = {{1, 1, 16}, {1, 4, 16}, {1, 2, 8}, {1, 2, 7}};
INSTANTIATE_TEST_SUITE_P(
    TestWithCombine1m, RunAlltoAllVTwoLevelPipelineTest,
    testing::Combine(
        testing::Values(800ull /* , 1000000008ull, 5000000008ull */),
        testing::Values(
            CheckerDataType::DATA_TYPE_FP32, CheckerDataType::DATA_TYPE_INT8, CheckerDataType::DATA_TYPE_BFP16,
            CheckerDataType::DATA_TYPE_INT64),
        testing::Values(OPBASE), testing::ValuesIn(topoVecs1), testing::Values("pipeline"), testing::Values(false)));
