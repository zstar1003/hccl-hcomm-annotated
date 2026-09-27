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

#include "testcase_utils.h"
#include "checker.h"
#include "env_config.h"

using namespace hccl;
using namespace checker;

class AlgEnvConfigTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CheckOpSemanticsTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CheckOpSemanticsTest tear down." << std::endl; }

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

TEST_F(AlgEnvConfigTest, ut_external_input_env_hccl_algo)
{
    // only for coverage
    HcclResult ret;
    ret = ParseHcclAlgo();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_ALGO", "=level0:NA/level1:pairwise", 1);
    ret = ParseHcclAlgo();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_ALGO", "level0:NA/level1:pairwise", 1);
    ret = ParseHcclAlgo();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_ALGO", "alltoall=level0:NA&level1:pairwise;", 1);
    ret = ParseHcclAlgo();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_ALGO", "alltoall=level0:NA&level1:pairwise", 1);
    ret = ParseHcclAlgo();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_ALGO", "alltoall=level0:NA;level1:pairwise&allreduce=level0:NA;level1:pipeline", 1);
    ret = ParseHcclAlgo();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_ALGO", "level0:NA/level1:pairwise", 1);
    ret = ParseHcclAlgo();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_ALGO", "allreduce=level0:Yes;level1:pairwise", 1);
    ret = ParseHcclAlgo();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_ALGO", "not_op=level0:Yes;level1:pairwise", 1);
    ret = ParseHcclAlgo();
    EXPECT_EQ(ret, HCCL_E_PARA);

    unsetenv("HCCL_ALGO");
}
