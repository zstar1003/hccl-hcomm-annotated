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
#include "topo_matcher.h"
#include "alltoall_operator.h"
#include "hccl_aiv.h"
#include "checker.h"
using namespace checker;

using namespace hccl;

class A3AllToAllPipelineTest :
    public testing::TestWithParam<std::tuple<
        CheckerOpType, std::vector<int>, int, int, CheckerDataType, CheckerOpMode, std::string, bool, bool>> {
protected:
    static void SetUpTestCase() { std::cout << "A3AllToAllPipelineTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "A3AllToAllPipelineTest tear down." << std::endl; }

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

TEST_P(A3AllToAllPipelineTest, A3_alltoall_pipeline_test)
{
    const auto& mytuple = GetParam();

    const CheckerOpType opType = std::get<0>(mytuple);
    const std::vector<int>& topoVec = std::get<1>(mytuple);
    const int bufSize = std::get<2>(mytuple);
    const int dataCount = std::get<3>(mytuple);
    const CheckerDataType dataType = std::get<4>(mytuple);
    const CheckerOpMode opMode = std::get<5>(mytuple);
    const std::string& hccl_algo = std::get<6>(mytuple);
    const bool aicpu_mode = std::get<7>(mytuple);
    const bool hccs_disable = std::get<8>(mytuple);

    cout << "opType=" << opType << ", topo={" << topoVec[0] << "," << topoVec[1] << "," << topoVec[2]
         << "}, bufSize=" << bufSize << ", dataCount=" << dataCount << ", dataType=" << dataType
         << ", opMode=" << opMode << ", hccl_algo=" << hccl_algo << ", aicpu_mode=" << aicpu_mode << endl;

    std::string sbuf = std::to_string(bufSize);
    setenv("HCCL_BUFFSIZE", sbuf.c_str(), 1);

    if (!hccl_algo.empty()) {
        std::string hccl_algo_env = "level0:NA;level1:" + hccl_algo;
        setenv("HCCL_ALGO", hccl_algo_env.c_str(), 1);
    }

    if (aicpu_mode) {
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    } else {
        setenv("HCCL_OP_EXPANSION_MODE", "HOST", 1);
    }

    if (hccs_disable) {
        setenv("HCCL_INTER_HCCS_DISABLE", "TRUE", 1);
    }

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, topoVec[0], topoVec[1], topoVec[2]);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = opType;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = opMode;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.aicpuUnfoldMode = aicpu_mode;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    if (opType == CheckerOpType::ALLTOALLV) {
        checkerOpParam.All2AllDataDes.sendType = dataType;
        checkerOpParam.All2AllDataDes.recvType = dataType;
        GenAllToAllVParams(
            rankNum, dataCount, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
            checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);
    } else {
        checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(dataCount, rankNum);
        checkerOpParam.All2AllDataDes.sendType = dataType;
        checkerOpParam.All2AllDataDes.recvType = dataType;
        checkerOpParam.All2AllDataDes.sendCount = dataCount;
    }

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(A3AllToAllPipelineTest, A3_alltoall_pipeline_test_multi_server)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}, {{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllVTwoLevelPipeline";

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.sendCount = 100;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

std::vector<std::vector<int>> topoVecs1 = {{1, 1, 4}, {1, 4, 1}, {1, 4, 4}, {4, 4, 1}, {2, 2, 2}};
INSTANTIATE_TEST_SUITE_P(
    TestWithCombine1, A3AllToAllPipelineTest,
    testing::Combine(
        testing::Values(CheckerOpType::ALLTOALLV), testing::ValuesIn(topoVecs1), testing::Values(200),
        testing::Values(100), testing::Values(CheckerDataType::DATA_TYPE_FP32), testing::Values(OPBASE),
        testing::Values("pipeline"), testing::Values(true), testing::Values(true)));

std::vector<std::vector<int>> topoVecs2 = {{2, 2, 2}};
INSTANTIATE_TEST_SUITE_P(
    TestWithCombine2, A3AllToAllPipelineTest,
    testing::Combine(
        testing::Values(CheckerOpType::ALLTOALL, CheckerOpType::ALLTOALLV, CheckerOpType::ALLTOALLVC),
        testing::ValuesIn(topoVecs2), testing::Values(200), testing::Values(100),
        testing::Values(CheckerDataType::DATA_TYPE_FP32), testing::Values(OPBASE), testing::Values("pipeline"),
        testing::Values(true), testing::Values(true)));

std::vector<std::vector<int>> topoVecs3 = {{2, 2, 2}};
INSTANTIATE_TEST_SUITE_P(
    TestWithCombine3, A3AllToAllPipelineTest,
    testing::Combine(
        testing::Values(CheckerOpType::ALLTOALLV), testing::ValuesIn(topoVecs3), testing::Values(10, 100),
        testing::Values(100, 1024000), testing::Values(CheckerDataType::DATA_TYPE_FP32), testing::Values(OPBASE),
        testing::Values("pipeline"), testing::Values(true), testing::Values(true)));

std::vector<std::vector<int>> topoVecs4 = {{2, 2, 2}};
INSTANTIATE_TEST_SUITE_P(
    TestWithCombine4, A3AllToAllPipelineTest,
    testing::Combine(
        testing::Values(CheckerOpType::ALLTOALLV), testing::ValuesIn(topoVecs4), testing::Values(200),
        testing::Values(100), testing::Values(CheckerDataType::DATA_TYPE_FP32), testing::Values(OPBASE, OFFLOAD),
        testing::Values("", "pipeline"), testing::Values(false, true), testing::Values(false, true)));
