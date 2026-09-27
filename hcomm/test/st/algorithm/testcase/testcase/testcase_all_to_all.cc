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
#include "topo_meta.h"
#include "testcase_utils.h"
#include "topo_matcher.h"
#include "alltoall_operator.h"
#include "hccl_aiv.h"
#include "checker.h"
using namespace checker;

using namespace hccl;

class AllToAllTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AllToAllTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AllToAllTest tear down." << std::endl; }

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

TEST_F(AllToAllTest, alltoall_test_310P3_opbase)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.sendCount = 100;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllTest, alltoall_test_910B_opbase_single_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllSingleExecutor";

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.sendCount = 100;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllTest, alltoall_test_910B_opbase)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.sendCount = 100;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllTest, alltoall_test_910B_offload)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.sendCount = 100;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllTest, alltoall_test_910B_opbase_RunAlltoAllVTwoLevelPipeline)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
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

TEST_F(AllToAllTest, alltoall_test_910B_offload_RunAlltoAllVTwoLevelPipeline)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
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

TEST_F(AllToAllTest, alltoall_test_910B_opbase_RunAlltoAllVFullMesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVFullMesh";

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

TEST_F(AllToAllTest, alltoall_test_910B_offload_RunAlltoAllVFullMesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVFullMesh";

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

TEST_F(AllToAllTest, alltoall_test_910B_opbase_RunAlltoAllVStaged)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVStaged";

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

TEST_F(AllToAllTest, alltoall_test_910B_offload_RunAlltoAllVStaged)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVStaged";

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

TEST_F(AllToAllTest, alltoall_test_910B_opbase_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;

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

TEST_F(AllToAllTest, alltoall_test_910B_offload_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;

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

TEST_F(AllToAllTest, alltoall_test_910_93_opbase_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";

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

TEST_F(AllToAllTest, alltoall_test_910_93_offload_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";
    checkerOpParam.aicpuUnfoldMode = true;

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

TEST_F(AllToAllTest, alltoall_test_910_93_opbase_2superpod_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 9);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";

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

TEST_F(AllToAllTest, alltoall_test_910_93_opbase_RunAlltoAllDirectFullmesh_1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{
        {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
        {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";
    checkerOpParam.aicpuUnfoldMode = true;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.sendCount = 100;

    Checker checker;
    HcclResult ret;
    checker.CloseRankMemCheck();
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllTest, alltoall_test_910_93_opbase_RunAlltoAllDirectFullmesh_interHccsDisable)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 2);

    setenv("HCCL_INTER_HCCS_DISABLE", "true", 1);
    MOCKER_CPP(&TopoMatcher::GetExternalInputInterHccsDisable).stubs().will(returnValue(1));

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";
    checkerOpParam.aicpuUnfoldMode = true;

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

// 跑AlltoAllVDirectFullmesh
TEST_F(AllToAllTest, alltoall_AlltoAllVDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    checkerOpParam.All2AllDataDes.sendCount = 100;
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

TEST_F(AllToAllTest, alltoall_superpod_asym_gcd)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100000, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.sendCount = 100000;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllTest, alltoall_test_91093_opbase_AlltoAllMeshAivExecutor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.aicpuUnfoldMode = true;

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

TEST_F(AllToAllTest, alltoall_test_a2_opbase_AlltoAllStagedAIVRdmaExecutor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.aicpuUnfoldMode = false;
    checkerOpParam.algName = "AlltoAllStagedAIVRdmaExecutor";

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.sendCount = 100;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
}

TEST_F(AllToAllTest, alltoall_test_91093_graph_AlltoAllMeshAivExecutor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.aicpuUnfoldMode = true;
    checkerOpParam.algName = "AlltoAllMeshAivSmallCountExecutor";

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

TEST_F(AllToAllTest, alltoall_test_91093_opbase_AlltoAllMeshAivFor91093Executor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER_CPP(&AlltoAllOperator::IsSatisfyAlltoAllAivCondition).stubs().will(returnValue(true));
    MOCKER(ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.aicpuUnfoldMode = true;

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

TEST_F(AllToAllTest, alltoall_test_910B_opbase_RunAlltoAllTwoLevelPipeline_pingpong)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);
    setenv("HCCL_ALGO", "level0:ring;level1:pipeline", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVTwoLevelPipeline";

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(20971520, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.sendCount = 20971520;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllTest, alltoall_test_910_93_opbase_RunAlltoAllFullMeshSymmetricMemory)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllFullMeshSymmetricMemory";

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.sendCount = 100;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
}

TEST_F(AllToAllTest, RunAlltoAllVPipelineFor91093)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);
    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALL;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT8;

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.sendCount = 100;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
