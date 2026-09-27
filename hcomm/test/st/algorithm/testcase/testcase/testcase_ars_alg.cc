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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include <vector>
#include <iostream>

#include "topoinfo_struct.h"
#include "log.h"
#include "checker_def.h"
#include "testcase_utils.h"
#include "hccl_aiv.h"
#include "topo_matcher.h"
#include "checker.h"
using namespace checker;

using namespace hccl;

class arsAlgTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "arsAlgTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "arsAlgTest tear down." << std::endl; }
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

constexpr int NORMAL_DATA_SIZE = 100;
constexpr int SMALL_COUNT_DATA_SIZE = 4 * 1024 * 1024;

void initTopoMeta(TopoMeta& topoMate, int superPods, int servers, const vector<int>& ServerDevice)
{
    for (int i = 0; i < superPods; i++) {
        SuperPodMeta superPodMeta;
        for (int j = 0; j < servers; j++) {
            ServerMeta serverMeta;
            for (int k = 0; k < ServerDevice[j]; k++) {
                serverMeta.push_back((unsigned int)k);
            }
            superPodMeta.push_back(serverMeta);
        }
        topoMate.push_back(superPodMeta);
    }
}

TEST_F(arsAlgTest, allgather_910_93_opbase_CollAllGatherARSFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    constexpr int superPods = 1;
    constexpr int servers = 2;
    initTopoMeta(topoMeta, superPods, servers, std::move(vector<int>{6, 2}));

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherARSFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allgather_910_93_opbase_CollAllGatherARSFor91093Executor_NB_16MB)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    constexpr int superPods = 1;
    constexpr int servers = 2;
    initTopoMeta(topoMeta, superPods, servers, std::move(vector<int>{6, 2}));

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    MOCKER_CPP(&TopoMatcher::GetARSFlag).stubs().with(mockcpp::any()).will(returnValue(true));

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = SMALL_COUNT_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherARSFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allgather_910_93_opbase_CollAllGatherARSFor91093Executor_NHR_16MB)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    constexpr int superPods = 1;
    constexpr int servers = 2;
    initTopoMeta(topoMeta, superPods, servers, std::move(vector<int>{6, 2}));

    setenv("HCCL_ALGO", "level0:NA;level1:NHR", 1);

    MOCKER_CPP(&TopoMatcher::GetARSFlag).stubs().with(mockcpp::any()).will(returnValue(true));

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = SMALL_COUNT_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherARSFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, reduce_scatter_A3_2Server_ReduceScatterARSFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    constexpr int superPods = 1;
    constexpr int servers = 2;
    initTopoMeta(topoMeta, superPods, servers, std::move(vector<int>{6, 2}));

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "ReduceScatterARSFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, reduce_scatter_A3_2Server_ReduceScatterARSFor91093Executor_NB_16MB)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    constexpr int superPods = 1;
    constexpr int servers = 2;
    initTopoMeta(topoMeta, superPods, servers, std::move(vector<int>{6, 2}));

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    MOCKER_CPP(&TopoMatcher::GetARSFlag).stubs().with(mockcpp::any()).will(returnValue(true));

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = SMALL_COUNT_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "ReduceScatterARSFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, reduce_scatter_A3_2Server_ReduceScatterARSFor91093Executor_NHR_16MB)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    constexpr int superPods = 1;
    constexpr int servers = 2;
    initTopoMeta(topoMeta, superPods, servers, std::move(vector<int>{6, 2}));

    setenv("HCCL_ALGO", "level0:NA;level1:NHR", 1);

    MOCKER_CPP(&TopoMatcher::GetARSFlag).stubs().with(mockcpp::any()).will(returnValue(true));

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = SMALL_COUNT_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "ReduceScatterARSFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, reduce_scatter_ARS_NB_4_4_8_8) // AHC (4,4) (8,8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}, {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, reduce_scatter_ARS_NB_4_4_8) // ARS (4,4) (8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}, {{0, 1, 2, 3, 4, 5, 6, 7}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, reduce_scatter_ARS_NB_4_6_8) // AHC (4,6) (8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5}}, {{0, 1, 2, 3, 4, 5, 6, 7}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, reduce_scatter_ARS_NB_4_6_4_6) // ARS (4,6) (4,6)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5}}, {{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allgather_ARS_NB_4_4_8_8) // AHC (4,4)(8,8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}, {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allgather_ARS_NB_4_4_8) // ARS (4,4) (8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}, {{0, 1, 2, 3, 4, 5, 6, 7}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allgather_ARS_NB_4_6_8) // AHC (4,6) (8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5}}, {{0, 1, 2, 3, 4, 5, 6, 7}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allgather_ARS_NB_4_6_4_6) // ARS (4,6)(4,6)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5}}, {{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// allgather 非对称覆盖
TEST_F(arsAlgTest, allgather_ARS_NB_6_2_6_2) // ARS(6,2) (6,2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5}, {0, 1}}, {{0, 1, 2, 3, 4, 5}, {0, 1}}};

    setenv("HCCL_ALGO", "level0:ring;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// reducescatter 非对称覆盖
TEST_F(arsAlgTest, reduce_scatter_ARS_NB_6_2_6_2) // ARS(6,2) (6,2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5}, {0, 1}}, {{0, 1, 2, 3, 4, 5}, {0, 1}}};

    setenv("HCCL_ALGO", "level0:ring;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// allgather 非对称覆盖
TEST_F(arsAlgTest, allgather_ARS_NB_4_2_4_2) // ARS(4,2) (4,2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1}}, {{0, 1, 2, 3}, {0, 1}}};

    setenv("HCCL_ALGO", "level0:ring;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// reducescatter 非对称覆盖
TEST_F(arsAlgTest, reduce_scatter_ARS_NB_4_2_4_2) // ARS(4,2) (4,2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1}}, {{0, 1, 2, 3}, {0, 1}}};

    setenv("HCCL_ALGO", "level0:ring;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allreduce_A3_2Server_AllReduceARSFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    constexpr int superPods = 1;
    constexpr int servers = 2;
    initTopoMeta(topoMeta, superPods, servers, std::move(vector<int>{6, 2}));

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllReduceARSFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allreduce_A3_2Server_AllReduceARSFor91093Executor_NB_16MB)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    constexpr int superPods = 1;
    constexpr int servers = 2;
    initTopoMeta(topoMeta, superPods, servers, std::move(vector<int>{6, 2}));

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    MOCKER_CPP(&TopoMatcher::GetARSFlag).stubs().with(mockcpp::any()).will(returnValue(true));

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = SMALL_COUNT_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllReduceARSFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allreduce_A3_2Server_AllReduceARSFor91093Executor_NHR_16MB)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    constexpr int superPods = 1;
    constexpr int servers = 2;
    initTopoMeta(topoMeta, superPods, servers, std::move(vector<int>{6, 2}));

    setenv("HCCL_ALGO", "level0:NA;level1:NHR", 1);

    MOCKER_CPP(&TopoMatcher::GetARSFlag).stubs().with(mockcpp::any()).will(returnValue(true));

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = SMALL_COUNT_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllReduceARSFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allreduce_ARS_NB_4_4_8_8) // AHC (4,4)(8,8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}, {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allreduce_ARS_NB_4_4_8) // ARS (4,4) (8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}, {{0, 1, 2, 3, 4, 5, 6, 7}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allreduce_ARS_NB_4_6_8) // AHC (4,6) (8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5}}, {{0, 1, 2, 3, 4, 5, 6, 7}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// allreduce 非对称覆盖
TEST_F(arsAlgTest, allreduce_ARS_NB_6_2_6_2) // ARS(6,2) (6,2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5}, {0, 1}}, {{0, 1, 2, 3, 4, 5}, {0, 1}}};

    setenv("HCCL_ALGO", "level0:ring;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allreduce_ARS_NB_4_6_4_6) // ARS (4,6)(4,6)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5}}, {{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// allreduce 非对称覆盖双die不成对
TEST_F(arsAlgTest, allreduce_ARS_NB_6_2_6_2_single_die) // ARS(6,2) (6,2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5}, {0, 2}}, {{0, 1, 2, 3, 4, 5}, {0, 1}}};

    setenv("HCCL_ALGO", "level0:ring;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// allreduce 非对称覆盖
TEST_F(arsAlgTest, allreduce_ARS_NB_4_2_4_2) // ARS(4,2) (4,2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1}}, {{0, 1, 2, 3}, {0, 1}}};

    setenv("HCCL_ALGO", "level0:ring;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allgather_ARS_NB_NOT_DOUBLERING_4_4_8) // ARS (4,4) (8)，非成对die
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 2, 4, 6}, {0, 2, 4, 6}}, {{0, 2, 4, 5, 6, 7, 8, 9}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    checker.EnableTaskPrint();
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(arsAlgTest, allgather_ARS_NB_NOT_DOUBLERING_4_6_4_6) // ARS (4,6)(4,6)，非成对die
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 2, 4, 6}, {0, 2, 4, 5, 6, 7}}, {{0, 2, 4, 6}, {0, 2, 4, 5, 6, 7}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// allreduce 非对称覆盖
TEST_F(arsAlgTest, allreduce_ARS_NB_4_4_4) // ARS(4,4) (4)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}, {{0, 1, 2, 3}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// allgather 非对称覆盖
TEST_F(arsAlgTest, allgather_ARS_NB_4_4_4) // ARS(4,4) (4)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}, {{0, 1, 2, 3}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// reducescatter 非对称覆盖
TEST_F(arsAlgTest, reducescatter_ARS_NB_4_4_4) // ARS(4,4) (4)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}, {{0, 1, 2, 3}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// allreduce 非对称覆盖
TEST_F(arsAlgTest, allreduce_ARS_NB_2_8_6) // ARS(2,8) (6)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1, 2, 3, 4, 5, 6, 7}}, {{0, 1, 2, 3, 4, 5}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// allgather 非对称覆盖
TEST_F(arsAlgTest, allgather_ARS_NB_2_8_6) // ARS(2,8) (6)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1, 2, 3, 4, 5, 6, 7}}, {{0, 1, 2, 3, 4, 5}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// reducescatter 非对称覆盖
TEST_F(arsAlgTest, reducescatter_ARS_NB_2_8_6) // ARS(2,8) (6)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1, 2, 3, 4, 5, 6, 7}}, {{0, 1, 2, 3, 4, 5}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// allreduce 对称覆盖
TEST_F(arsAlgTest, allreduce_ARS_NB_4_4_4_4) // ARS(4,4) (4,4)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}, {{0, 1, 2, 3}, {0, 1, 2, 3}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// allgather 对称覆盖
TEST_F(arsAlgTest, allgather_ARS_NB_4_4_4_4) // ARS(4,4) (4,4)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}, {{0, 1, 2, 3}, {0, 1, 2, 3}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// reducescatter 对称覆盖
TEST_F(arsAlgTest, reducescatter_ARS_NB_4_4_4_4) // ARS(4,4) (4,4)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}, {{0, 1, 2, 3}, {0, 1, 2, 3}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// allreduce 对称覆盖
TEST_F(arsAlgTest, allreduce_ARS_NB_8_8_8_8) // ARS(8,8) (8,8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{
        {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}, {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLREDUCE;
    checkerOpParam.tag = "AllReduce";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// allgather 对称覆盖
TEST_F(arsAlgTest, allgather_ARS_NB_8_8_8_8) // ARS(8,8) (8,8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{
        {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}, {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// reducescatter 对称覆盖
TEST_F(arsAlgTest, reducescatter_ARS_NB_8_8_8_8) // ARS(8,8) (8,8)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{
        {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}, {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::REDUCE_SCATTER;
    checkerOpParam.tag = "ReduceScatter";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = NORMAL_DATA_SIZE;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
