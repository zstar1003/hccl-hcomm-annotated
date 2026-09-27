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
#include "coll_native_executor_base.h"

#include "checker.h"
using namespace checker;

class AllGatherVTestFor310P : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AllGatherVTestFor310P set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AllGatherVTestFor310P tear down." << std::endl; }
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

TEST_F(AllGatherVTestFor310P, all_gather_v_310P3_opbase_varying_size_1)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    vector<u64> counts{1000000, 2000000, 3000000, 4000000};
    vector<u64> displs{0, 1000000, 3000000, 6000000};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER_V;
    checkerOpParam.tag = "AllGatherV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherVFor310PExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherVTestFor310P, all_gather_v_310P3_opbase_varying_size_2)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    vector<u64> counts{400000, 300000, 200000, 100000};
    vector<u64> displs{0, 400000, 700000, 900000};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER_V;
    checkerOpParam.tag = "AllGatherV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherVFor310PExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherVTestFor310P, allgather_v_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 2);

    vector<u64> counts{40000, 30000, 20000, 10000};
    vector<u64> displs{0, 40000, 70000, 90000};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER_V;
    checkerOpParam.tag = "AllGatherV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherVTestFor310P, allgather_v_AllGatherVFor310PExecutor_test)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    vector<u64> counts{400000, 300000, 200000, 100000};
    vector<u64> displs{0, 400000, 700000, 900000};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER_V;
    checkerOpParam.tag = "AllGatherV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherVFor310PExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherVTestFor310P, all_gather_v_310P3_opbase_varying_size_3)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    vector<u64> counts{200000, 400000, 600000, 800000};
    vector<u64> displs{0, 200000, 600000, 1200000};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER_V;
    checkerOpParam.tag = "AllGatherV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherVFor310PExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherVTestFor310P, all_gather_v_310P3_opbase_varying_size_4)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 4);

    vector<u64> counts{100000, 300000, 600000, 1200000};
    vector<u64> displs{0, 100000, 400000, 1000000};

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER_V;
    checkerOpParam.tag = "AllGatherV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherVFor310PExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherVTestFor310P, all_gather_v_310P3_opbase_varying_size_5)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 5);

    vector<u64> counts{10000, 20000, 30000, 20000, 10000};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER_V;
    checkerOpParam.tag = "AllGatherV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherVFor310PExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherVTestFor310P, all_gather_v_310P3_opbase_varying_size_6)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 6);

    vector<u64> counts{10000, 20000, 30000, 30000, 20000, 10000};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER_V;
    checkerOpParam.tag = "AllGatherV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherVFor310PExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(AllGatherVTestFor310P, all_gather_v_310P3_opbase_varying_size_7)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 7);

    vector<u64> counts{10000, 20000, 30000, 40000, 30000, 20000, 10000};
    vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::ALLGATHER_V;
    checkerOpParam.tag = "AllGatherV";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_310P3;
    checkerOpParam.VDataDes.counts = counts;
    checkerOpParam.VDataDes.displs = displs;
    checkerOpParam.VDataDes.dataType = CheckerDataType::DATA_TYPE_INT32;
    checkerOpParam.algName = "AllGatherVFor310PExecutor";

    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
