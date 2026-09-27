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
#include "coll_native_executor_base.h"
#include "hccl_aiv.h"
#include "checker.h"
using namespace checker;
using namespace hccl;

std::vector<string> allGatherAlgName = {
    "AllGatherMeshExecutor", "AllGatherMeshOpbaseExecutor",      "AllGatherMeshOpbasePipelineExecutor",
    "AllGatherRingExecutor", "AllGatherMeshAivFor91093Executor",
};

class AllGatherTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AllGatherTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AllGatherTest tear down." << std::endl; }

    virtual void SetUp()
    {
        const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string caseName
            = "analysis_result_" + std::string(test_info->test_case_name()) + "_" + std::string(test_info->name());
        Checker::SetDumpFileName(caseName);
        MOCKER(ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    }

    virtual void TearDown()
    {
        Checker::SetDumpFileName("analysis_result");
        GlobalMockObject::verify();
        // 这边每个case执行完成需要清理所有的环境变量，如果有新增的环境变量，需要在这个函数中进行清理
        ClearHcclEnv();
    }
};

TEST_F(AllGatherTest, allgather_310P_opbase_AllGatherSlimRingExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherSlimRingFor310PExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_test_loop)
{
    RankTable_For_LLT gen;
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.reduceType = CheckerReduceOp::REDUCE_SUM;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    HcclResult ret;
    for (int i = 1; i <= 2; i++) {
        for (int j = 1; j <= 8; j++) {
            Checker checker;
            TopoMeta topoMeta;
            gen.GenTopoMeta(topoMeta, 1, i, j);
            ret = checker.Check(checkerOpParam, topoMeta);
            EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
        }
    }
}

TEST_F(AllGatherTest, allgather_offload_AllGatherMeshExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherMeshExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_offload_AllGatherMeshGraphExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 4096;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherMeshGraphExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_offload_3Server_AllGatherMeshGraphExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 3, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 4096;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherMeshGraphExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_offload_8Server_AllGatherMeshGraphExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 8, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 4096;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_offload_2Server_AllGatherMeshGraphPipelineExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 4096;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_offload_3Server_AllGatherMeshGraphPipelineExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 3, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 2 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_offload_AllGatherRingExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherRingExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// 仅测试用，不同算法限制不同
TEST_F(AllGatherTest, allgather_executor_test_loop)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    for (int i = 0; i < allGatherAlgName.size(); i++) {
        checkerOpParam.algName = allGatherAlgName[i];
        Checker checker;
        HcclResult ret;
        ret = checker.Check(checkerOpParam, topoMeta);
        if (ret == HcclResult::HCCL_SUCCESS) {
            std::cout << allGatherAlgName[i] + " run succeeded" << std::endl;
        } else {
            std::cout << allGatherAlgName[i] + " run failed" << std::endl;
        }
        EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }
}

TEST_F(AllGatherTest, allgather_AllGatherComm_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    checkerOpParam.algName = "AllGatherComm";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_AllGatherSingleExecutor_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGatherSingleExecutor";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    checkerOpParam.algName = "AllGatherSingleExecutor";
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910A_offload_AllGatherMeshExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherMeshExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910A_offload_AllGatherRingExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherRingExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910A_offload_AllGatherComm)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherComm";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910A_opbase_AllGatherMeshExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherMeshExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910A_opbase_AllGatherRingExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherRingExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910A_opbase_AllGatherComm)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherComm";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910A_opbase_AllGatherRingExecutor_NSLB)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 3, 8);
    setenv("HCCL_ALGO", "level0:ring;level1:H-D_R", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherRingExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910_93_opbase_AlignedAllGatherDoubleRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AlignedAllGatherDoubleRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910_93_offload_AlignedAllGatherDoubleRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AlignedAllGatherDoubleRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910_93_opbase_multiSuperPod_2_6_2_AllGatherRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 6, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910_93_opbase_multiSuperPod_4_3_2_AllGatherRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 4, 3, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910_93_opbase_AllGatherRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910_93_offload_AllGatherDoubleRingConcurrentExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    setenv("HCCL_CONCURRENT_ENABLE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_310P3_opbase_AllGatherFor310PExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherFor310PExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_310P3_offload_AllGatherFor310PExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherFor310PExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_smallcount_multiserver_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 12);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;

    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910_93_offload_AllGatherSemiRingExecutor_singleServer)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherSemiRingExecutor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_910_93_offload_AllGatherSemiRingExecutor_singleServer_oddNum)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 11;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "AllGatherSemiRingExecutor";
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_ax_4server_16p)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_superpod_asym_gcd)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 1000000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_superpod_asym_gcd_graph)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{
        {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_mix_AllGatherMixExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2}}, {{0, 1, 2, 3, 4, 5, 6, 7, 8}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910B);
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910_93);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_mix_AllGatherMixExecutorComm)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0}}, {{0}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910B);
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910_93);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_aiv_a3_AllGatherVMeshAivExecutor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.aicpuUnfoldMode = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_aiv_a3_AllGatherMeshAivFor91093Executor1)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.aicpuUnfoldMode = true;
    checkerOpParam.algName = "AllGatherMeshAivFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_aiv_a3_AllGatherMeshAivFor91093Executor2)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 64 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.aicpuUnfoldMode = true;
    checkerOpParam.algName = "AllGatherMeshAivFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_AllGatherSmallCountExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllGatherSmallCount";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
TEST_F(AllGatherTest, allgather_AllGatherSmallCountExecutor_singleRank)
{
    // hccl::RankTable_t ranktable;
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 3);
    CheckerOpParam testOpParam;
    testOpParam.opType = CheckerOpType::ALLGATHER;
    testOpParam.tag = "AllGather";
    testOpParam.opMode = CheckerOpMode::OPBASE;
    testOpParam.DataDes.count = 100;
    testOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    testOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    testOpParam.algName = "AllGatherSmallCount";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(testOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_aiv_a3__AllGatherMeshAivExecutor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllGatherMeshAivExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
}

TEST_F(AllGatherTest, allgather_AllGatherAivRdmaExecutor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllGatherAivRdmaExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
}

TEST_F(AllGatherTest, allgather_AllGatherRingFor91093Executor_NHR)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:NHR;level2:NHR", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllGatherRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_AllGatherRingFor91093Executor_NB)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:NB;level2:NB", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllGatherRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_A3_2Server1Rank_nb_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 1);

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_A3_2Die_HccsSio_opbase_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_A3_2Die_HccsSio_offload_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.aicpuUnfoldMode = true;
    checkerOpParam.algName = "AllGatherHccsSioExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_91093_AllGatherRingZerocopyExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllGatherRingZerocopyExecutor";
    checkerOpParam.isZeroCopy = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_91093_AllGatherRingZerocopyExecutor_AHC)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 4);

    setenv("HCCL_ALGO", "level1:AHC", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllGatherRingZerocopyExecutor";
    checkerOpParam.isZeroCopy = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_AllGatherMeshAivExecutor)
{
    setenv("HCCL_OP_EXPANSION_MODE", "AIV", 1);

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 16 * 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "AllGatherMeshAivExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
}

TEST_F(AllGatherTest, allgather_AllGatherMeshAivSmallExecutor)
{
    setenv("HCCL_OP_EXPANSION_MODE", "AIV", 1);

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "AllGatherMeshAivSmallCountExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
}

TEST_F(AllGatherTest, allgather_91093_AllGatherRingZerocopyExchangeExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 4, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllGatherRingZerocopyExchangeExecutor";
    checkerOpParam.isZeroCopy = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_aiv_a3_AllGatherMeshAivSmallCountExecutor)
{
    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER(hccl::ExecuteKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.aicpuUnfoldMode = false;
    checkerOpParam.algName = "AllGatherMeshAivSmallCountExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, allgather_91093_AllGathermidcountExecutor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 1024 * 1024; // 64K
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "AllGatherMidCountFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// ==================== Pipeline AllGather (A3, 910_93, CollAllGatherPipelineFor91093Executor) ====================

TEST_F(AllGatherTest, ag_pipeline_DR_2pod_2server_int32_4blk_equal)
{
    // DoubleRing + L1/L2 默认 NHR；4 块等大（偶数），完整预热->稳态->排空
    setenv("HCCL_BUFFSIZE", "1", 1);
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 16384;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, ag_pipeline_DR_3pod_1server_bfp16_3blk_smallA)
{
    // 3 SuperPod；3 块（奇数）尾块小落 BufferA（loopIdx=2, memIdx_==0）
    setenv("HCCL_BUFFSIZE", "1", 1);
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 3, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 49104;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.algName = "AllGatherPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, ag_pipeline_DR_fp32_4blk_smallB)
{
    // FP32；4 块尾块小落 BufferB（loopIdx=3, memIdx_==1），互补用例 2
    setenv("HCCL_BUFFSIZE", "1", 1);
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 26624;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.algName = "AllGatherPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, ag_pipeline_DR_force_ring_int8)
{
    // 强制 RING 模板；覆盖默认不覆盖的 Level1/Level2 RING 路径
    setenv("HCCL_BUFFSIZE", "1", 1);
    setenv("HCCL_ALGO", "level1:ring;level2:ring", 1);
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 49152;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.algName = "AllGatherPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, ag_pipeline_DR_force_nb_int16)
{
    // 强制 NB 模板路径覆盖
    setenv("HCCL_BUFFSIZE", "1", 1);
    setenv("HCCL_ALGO", "level1:NB;level2:NB", 1);
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 24576;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT16;
    checkerOpParam.algName = "AllGatherPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, ag_pipeline_SR_3pod_7rank_bfp16)
{
    // SingleRing：奇数 rank(7) 触发 SINGLE_RING
    setenv("HCCL_BUFFSIZE", "1", 1);
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 3, 1, 7);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 37440;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_BFP16;
    checkerOpParam.algName = "AllGatherPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, ag_pipeline_SR_2pod_3server_1rank_int32)
{
    // rank/server=1 边界：L0 退化（机内仅 1 rank 无 gather），Pipeline 流水主要覆盖 L1/L2
    setenv("HCCL_BUFFSIZE", "1", 1);
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 3, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 65520;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherTest, ag_pipeline_DR_one_block_degenerate_fp16)
{
    // 小数据量单 block，Pipeline 退化为串行时序
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER;
    checkerOpParam.tag = "AllGather";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 8192;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.algName = "AllGatherPipelineFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
