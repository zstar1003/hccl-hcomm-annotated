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
#include "checker.h"
#include "hccl_aiv.h"
using namespace checker;

// using namespace hccl;

class BroadcastTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BroadcastTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "BroadcastTest tear down." << std::endl; }

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

TEST_F(BroadcastTest, broadcast_opbase_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 3);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 300;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_two_servers_opbase_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_offload_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// 除了有数据的server，其他server存在无效的数据拷贝
TEST_F(BroadcastTest, broadcast_two_servers_offload_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_BroadCastComm_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.algName = "BroadCastComm";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;

    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// 跑BroadcastNHRV1
TEST_F(BroadcastTest, broadcast_BroadcastNHRV1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 4);

    setenv("HCCL_ALGO", "level0:NA;level1:NHR_V1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.algName = "BroadCastComm";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_BroadcastNHRV1_singleRank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 3);

    setenv("HCCL_ALGO", "level0:NA;level1:NHR_V1", 1);

    CheckerOpParam testOpParam;
    testOpParam.opType = CheckerOpType::BROADCAST;
    testOpParam.tag = "broadcast";
    testOpParam.algName = "BroadCastComm";
    testOpParam.opMode = CheckerOpMode::OPBASE;
    testOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    testOpParam.DataDes.count = 100;
    testOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    testOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(testOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcastmesh_BroadcastNHRV1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 4);

    setenv("HCCL_ALGO", "level0:fullmesh;level1:NHR_V1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.algName = "BroadCastMeshExecutor";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcastring_BroadcastNHRV1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 8);

    setenv("HCCL_ALGO", "level0:ring;level1:NHR_V1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.algName = "BroadCastRingExecutor";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcastring_910_93_BroadcastNHRV1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 2);

    setenv("HCCL_ALGO", "level0:ring;level1:NHR_V1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.algName = "BroadCastRingFor91093Executor";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

#ifdef ASCEND_310P_DEVICE
TEST_F(BroadcastTest, broadcast_310P_BroadcastNHRV1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);

    setenv("HCCL_ALGO", "level0:ring;level1:NHR_V1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.algName = "BroadcastPlusBroadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P1;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
#endif

TEST_F(BroadcastTest, broadcast_BroadCastRingExecutor_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.algName = "BroadCastRingExecutor";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;

    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_BroadCastRingExecutor_NSLB_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:NHR;level2:ring", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.algName = "BroadCastRingExecutor";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 10000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;

    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// 存在无效的数据拷贝
TEST_F(BroadcastTest, broadcast_BroadCastRingFor91093Executor_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.algName = "BroadCastRingFor91093Executor";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 10000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;

    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_BroadCastRingFor91093Executor_single_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.algName = "BroadCastRingFor91093Executor";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 10000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;

    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_BroadCastSmallCountExecutor_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.algName = "BroadCastSmallCountExecutor";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;

    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_smallcount_multiserver_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;

    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_ax_4server_16p)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 4, 16);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.root = 0;
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_superpod_asym_gcd)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 1000000;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_superpod_asym_gcd_graph)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta{
        {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}, {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.root = 0;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_mix_BroadcastMixExecutor)
{
    RankTable_For_LLT gen;

    /***** GenTopoMeta Start *****/
    TopoMeta topoMeta;
    SuperPodMeta superPodMeta910B;
    ServerMeta serverMeta910B;
    u32 rankNum910B = 16;
    for (u32 k = 0; k < rankNum910B; k++) {
        serverMeta910B.push_back(k);
    }
    superPodMeta910B.push_back(serverMeta910B);
    SuperPodMeta superPodMeta91093;
    ServerMeta serverMeta91093;
    u32 rankNum91093 = 16;
    for (u32 k = 0; k < rankNum91093; k++) {
        serverMeta91093.push_back(k);
    }
    superPodMeta91093.push_back(serverMeta91093);

    topoMeta.push_back(superPodMeta910B);
    topoMeta.push_back(superPodMeta91093);
    /***** GenTopoMeta End *****/

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.root = 0;
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910B);
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910_93);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_mix_BroadcastMixExecutorComm)
{
    RankTable_For_LLT gen;

    /***** GenTopoMeta Start *****/
    TopoMeta topoMeta;
    SuperPodMeta superPodMeta910B;
    ServerMeta serverMeta910B;
    u32 rankNum910B = 1;
    for (u32 k = 0; k < rankNum910B; k++) {
        serverMeta910B.push_back(k);
    }
    superPodMeta910B.push_back(serverMeta910B);
    SuperPodMeta superPodMeta91093;
    ServerMeta serverMeta91093;
    u32 rankNum91093 = 1;
    for (u32 k = 0; k < rankNum91093; k++) {
        serverMeta91093.push_back(k);
    }
    superPodMeta91093.push_back(serverMeta91093);

    topoMeta.push_back(superPodMeta910B);
    topoMeta.push_back(superPodMeta91093);
    /***** GenTopoMeta End *****/

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP32;
    checkerOpParam.root = 0;
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910B);
    checkerOpParam.devTypes.push_back(CheckerDevType::DEV_TYPE_910_93);

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_BroadCastRingFor91093Executor_NHR)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:NHR;level2:NHR", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.root = 0;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "BroadCastRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_BroadCastRingFor91093Executor_NB)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 8);

    setenv("HCCL_ALGO", "level0:NA;level1:NB;level2:NB", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.root = 0;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "BroadCastRingFor91093Executor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_A3_2Server1Rank_nb_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 1);

    setenv("HCCL_ALGO", "level0:NA;level1:NB", 1);
    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OFFLOAD;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.root = 0;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_91093_BroadCastRingZerocopyExecutor_multisuperpod)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 2, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.root = 0;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "BroadCastRingZerocopyExecutor";
    checkerOpParam.isZeroCopy = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_91093_BroadCastRingZerocopyExecutor_singlesuperpod)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.root = 0;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "BroadCastRingZerocopyExecutor";
    checkerOpParam.isZeroCopy = true;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_BroadCastRingFor91093Executor)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 5, 1, 3);

    setenv("HCCL_ALGO", "level0:NA;level1:ring;level2:ring", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 100;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "BroadCastRingFor91093Executor";
    checkerOpParam.root = 4;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BroadcastTest, broadcast_aivsmallcount)
{
    setenv("HCCL_OP_EXPANSION_MODE", "AIV", 1);
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 1;
    checkerOpParam.root = 0;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.algName = "BroadcastMeshAivExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
}

TEST_F(BroadcastTest, broadcast_BroadcastMidCountFor91093Executor)
{
    setenv("HCCL_ALGO", "level1:NHR", 1);
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 2, 1, 4);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BROADCAST;
    checkerOpParam.tag = "broadcast";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.DataDes.count = 10;
    checkerOpParam.root = 0;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_FP16;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.algName = "BroadcastMidCountFor91093Executor"; // BroadCastComm BroadcastMidCountFor91093Executor

    Checker checker;
    HcclResult ret;
    checker.EnableTaskPrint();
    ret = checker.Check(checkerOpParam, topoMeta);
}
