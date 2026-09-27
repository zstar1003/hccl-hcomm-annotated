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
#include "alltoallv_direct_fullmesh_pub.h"
#include "runtime_stub.h"
using namespace checker;

using namespace hccl;

class AllToAllVTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AllToAllVTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AllToAllVTest tear down." << std::endl; }

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

TEST_F(AllToAllVTest, alltoallv_test_310P3_opbase)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_pairwise)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    setenv("HCCL_ALGO", "level0:NA;level1:pairwise", 1);
    setenv("HCCL_INTER_HCCS_DISABLE", "true", 1);
    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_opbase_single_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllSingleExecutor";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_opbase)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_offload)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT32;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_opbase_RunAlltoAllVTwoLevelPipeline)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVTwoLevelPipeline";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_offload_RunAlltoAllVTwoLevelPipeline)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_opbase_RunAlltoAllVFullMesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVFullMesh";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_offload_RunAlltoAllVFullMesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVFullMesh";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_opbase_RunAlltoAllVStaged)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVStaged";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_offload_RunAlltoAllVStaged)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVStaged";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_opbase_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_offload_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910_93_opbase_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910_93_offload_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910_93_opbase_2superpod_RunAlltoAllDirectFullmesh)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 9);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_91093_opbase_AlltoAllMeshAivFor91093Executor)
{
    MOCKER(
        ExecuteKernelLaunch, HcclResult(
                                 const AivOpArgs&, const AivTopoArgs&, const AivResourceArgs&, const AivAlgArgs&,
                                 const ExtraArgsV2&, AivProfilingInfo&))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.aicpuUnfoldMode = true;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_91093_opbase_AlltoAllMeshAivFor91093Executor2)
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
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.aicpuUnfoldMode = true;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_91093_opbase_single_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";
    checkerOpParam.algOpContext.mc2Handler.stepSize = 2;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

TEST_F(AllToAllVTest, alltoallv_test_91093_opbase_single_rank1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";
    checkerOpParam.algOpContext.mc2Handler.stepSize = 1;
    checkerOpParam.algOpContext.mc2Handler.repeatCnt = 1;
    checkerOpParam.algOpContext.mc2Handler.rankSize = 1;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910_93_opbase_2superpod_RunAlltoAllDirectFullmesh_postSync)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 9);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";
    checkerOpParam.aicpuUnfoldMode = true;
    checkerOpParam.algOpContext.opRetryHandler.isPostSync = true;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u64 count = 1048576;
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, count, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    // postSync 会导致 checker的报错：Invalid data is copied
    ret = checker.Check(checkerOpParam, topoMeta);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_opbase_AclGraph_RunAlltoAllDirectFullmesh)
{
    rtStreamCaptureStatus captureStatus = rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(rtStreamGetCaptureInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(0));

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllDirectFullmesh";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_FP16;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 100, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_opbase_RunAlltoAllVContinuousPipeline_2srv_2nodes)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 2);
    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVContinuousPipeline";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT8;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 128, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_opbase_RunAlltoAllVContinuousPipeline_2srv_4nodes)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 4);

    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVContinuousPipeline";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT8;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 1024 * 1024 / 16 - 128, checkerOpParam.All2AllDataDes.sendCounts,
        checkerOpParam.All2AllDataDes.sdispls, checkerOpParam.All2AllDataDes.recvCounts,
        checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    checker.EnableTaskPrint();
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_opbase_RunAlltoAllVContinuousPipeline_1srv_16nodes_ax)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};

    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVContinuousPipeline";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT8;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 128, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    checker.EnableTaskPrint();
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_opbase_RunAlltoAllVContinuousPipeline_2srv_8nodes)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVContinuousPipeline";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT8;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 128, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_opbase_RunAlltoAllVContinuousPipeline_4srv_8nodes)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 8);

    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVContinuousPipeline";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT8;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 128, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, alltoallv_test_910B_opbase_RunAlltoAllVContinuousPipeline_2srv_2nodes_large_count)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 2);

    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "RunAlltoAllVContinuousPipeline";

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT8;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 256 * 1024, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    checker.EnableTaskPrint();
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllToAllVTest, RunAlltoAllVPipelineFor91093)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 16);
    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLTOALLV;
    checkerOpParam.tag = "AllToAll";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;

    checkerOpParam.All2AllDataDes.sendType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.All2AllDataDes.recvType = CheckerDataType::DATA_TYPE_INT8;

    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    GenAllToAllVParams(
        rankNum, 10, checkerOpParam.All2AllDataDes.sendCounts, checkerOpParam.All2AllDataDes.sdispls,
        checkerOpParam.All2AllDataDes.recvCounts, checkerOpParam.All2AllDataDes.rdispls);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
