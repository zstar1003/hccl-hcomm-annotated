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

class BroadCastRingFor91093GeneralizationTest :
    public ::testing::TestWithParam<std::tuple<
        u64, CheckerDataType, CheckerOpMode, vector<int>, int, std::vector<std::vector<std::vector<unsigned int>>>>> {
public:
    static void SetUpTestCase() { std::cout << "BroadCastRingFor91093GeneralizationTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "BroadCastRingFor91093GeneralizationTest tear down." << std::endl; }

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

TEST_P(BroadCastRingFor91093GeneralizationTest, SymmetricTopo)
{
    const auto& mytuple = GetParam();
    // 打印一下mytuple当前的参数
    // std::cout << "checkerOpParam.DataDes.count : " << std::get<0>(mytuple) << std::endl;
    // std::cout << "checkerOpParam.DataDes.dataType : " << std::get<1>(mytuple) << std::endl;
    // std::cout << "checkerOpParam.opMode : " << std::get<2>(mytuple) << std::endl;
    // std::cout << "checkerOpParam.TopoMeta : " << std::get<3>(mytuple)[0]  << " , " << std::get<3>(mytuple)[1] << " ,
    // " << std::get<2>(mytuple)[3] << std::endl; std::cout << "checkerOpParam.root : " << std::get<4>(mytuple) <<
    // std::endl;

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, std::get<3>(mytuple)[0], std::get<3>(mytuple)[1], std::get<3>(mytuple)[2]);
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);

    setenv("HCCL_ALGO", "level0:NA;level1:NHR", 1);
    u64 dataSize = std::get<0>(mytuple);
    if (dataSize >= 5000000008) {
        setenv("HCCL_BUFFSIZE", "4096", 1);
    };
    u64 datacount = dataSize / SIZE_TABLE[std::get<1>(mytuple)];

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = std::get<2>(mytuple);
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "BroadCastRingFor91093Executor";
    checkerOpParam.DataDes.count = datacount;
    checkerOpParam.DataDes.dataType = std::get<1>(mytuple);
    checkerOpParam.root = GetRootValue(rankNum, std::get<4>(mytuple));

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_P(BroadCastRingFor91093GeneralizationTest, AsymmetricTopo)
{
    const auto& mytuple = GetParam();
    // 打印一下mytuple当前的参数
    // std::cout << "checkerOpParam.DataDes.count : " << std::get<0>(mytuple) << std::endl;
    // std::cout << "checkerOpParam.DataDes.dataType : " << std::get<1>(mytuple) << std::endl;
    // std::cout << "checkerOpParam.opMode : " << std::get<2>(mytuple) << std::endl;
    // std::cout << "checkerOpParam.root : " << std::get<4>(mytuple) << std::endl;

    RankTable_For_LLT gen;
    TopoMeta topoMeta = {std::get<5>(mytuple)[0]};
    u32 rankSize = GetRankNumFormTopoMeta(topoMeta);

    setenv("HCCL_ALGO", "level0:NA;level1:NHR", 1);
    u64 dataSize = std::get<0>(mytuple);
    if (dataSize >= 5000000008) {
        setenv("HCCL_BUFFSIZE", "4096", 1);
    };
    u64 datacount = dataSize / SIZE_TABLE[std::get<1>(mytuple)];

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = std::get<2>(mytuple);
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "BroadCastRingFor91093Executor";
    checkerOpParam.DataDes.count = datacount;
    checkerOpParam.DataDes.dataType = std::get<1>(mytuple);
    checkerOpParam.root = GetRootValue(rankSize, std::get<4>(mytuple));

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    BroadCastTest, BroadCastRingFor91093GeneralizationTest,
    testing::Combine(
        testing::Values(800 /*, 1000000008, 5000000008*/),
        testing::Values(
            CheckerDataType::DATA_TYPE_INT32,
            CheckerDataType::DATA_TYPE_INT8 /*, CheckerDataType::DATA_TYPE_BFP16, CheckerDataType::DATA_TYPE_INT64*/),
        testing::Values(CheckerOpMode::OPBASE, CheckerOpMode::OFFLOAD),
        testing::ValuesIn(std::vector<std::vector<int>>{{1, 1, 8}, {1, 1, 16}, {1, 4, 8}, {1, 4, 16}}),
        testing::Values(0, 1, 2, 3),
        testing::ValuesIn(std::vector<std::vector<std::vector<std::vector<unsigned int>>>>{
            {{{0, 1, 2, 3, 4, 6, 8}, {0, 1, 2, 3, 4, 6, 8}}}})));
