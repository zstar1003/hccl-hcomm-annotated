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
#include "alg_template_register.h"
#include "broadcast_nhr_v1_pub.h"
#include "checker.h"

using namespace checker;
using namespace hccl;

class TemplateRegisterTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TemplateRegisterTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "TemplateRegisterTest tear down." << std::endl; }

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

TEST_F(TemplateRegisterTest, template_regist_rereg)
{
    HcclResult ret;
    ret = hccl::AlgTemplateRegistry::Instance().Register(
        TemplateType::TEMPLATE_BROADCAST_NHR_V1, DefaultTemplateCreator<BroadcastNHRV1>);

    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(TemplateRegisterTest, template_regist_outofrange)
{
    HcclResult ret;
    ret = hccl::AlgTemplateRegistry::Instance().Register(
        static_cast<TemplateType>(TEMPLATE_NATIVE_MAX_NUM), DefaultTemplateCreator<BroadcastNHRV1>);

    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(TemplateRegisterTest, template_get_outofrange)
{
    HcclDispatcher dispatcher_;
    auto ret = hccl::AlgTemplateRegistry::Instance().GetAlgTemplate(
        static_cast<TemplateType>(TEMPLATE_NATIVE_MAX_NUM), dispatcher_);

    EXPECT_EQ(ret, nullptr);
}
