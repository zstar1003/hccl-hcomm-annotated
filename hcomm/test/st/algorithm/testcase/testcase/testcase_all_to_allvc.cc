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
#include "alltoall_operator.h"
#include "hccl_aiv.h"
#include "checker.h"
using namespace checker;

using namespace hccl;

class AllToAllVCTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AllToAllVCTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AllToAllVCTest tear down." << std::endl; }

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

TEST_F(AllToAllVCTest, alltoallvc_test_910B_opbase_single_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllSingleExecutor";

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, alltoallvc_test_910B_opbase)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, alltoallvc_test_910B_offload)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, alltoallvc_test_910B_opbase_RunAlltoAllVTwoLevelPipeline)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVTwoLevelPipeline";

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, alltoallvc_test_910B_offload_RunAlltoAllVTwoLevelPipeline)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, alltoallvc_test_910B_opbase_RunAlltoAllVFullMesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVFullMesh";

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, alltoallvc_test_910B_offload_RunAlltoAllVFullMesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVFullMesh";

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, alltoallvc_test_910B_opbase_RunAlltoAllVStaged)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVStaged";

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, alltoallvc_test_910B_offload_RunAlltoAllVStaged)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVStaged";

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, alltoallvc_test_910B_opbase_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, alltoallvc_test_910B_offload_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, alltoallvc_test_910_93_opbase_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, alltoallvc_test_910_93_offload_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, alltoallvc_test_91093_offload_AlltoAllMeshAivFor91093Executor)
{
    MOCKER(
        ExecuteKernelLaunch, HcclResult(
                                 const AivOpArgs&, const AivTopoArgs&, const AivResourceArgs&, const AivAlgArgs&,
                                 const ExtraArgsV2&, AivProfilingInfo&))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER_CPP(&AlltoAllOperator::IsSatisfyAlltoAllAivCondition).stubs().will(returnValue(true));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.aicpuUnfoldMode = true;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, alltoallvc_test_910_93_opbase_2superpod_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 9);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVCTest, RunAlltoAllVPipelineFor91093)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 7);
    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLVC;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(100, rankNum);
    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT8;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
