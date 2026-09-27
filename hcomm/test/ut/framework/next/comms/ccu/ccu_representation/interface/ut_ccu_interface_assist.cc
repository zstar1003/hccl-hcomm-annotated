/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "ccu_kernel.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_v1.h"

using namespace hcomm;

class AppendToContextTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    static void TearDownTestCase()
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    void SetUp() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    void TearDown() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
};

TEST_F(AppendToContextTest, Ut_AppendToContext_When_NullContext_Expect_ShouldThrowException)
{
    CcuRep::CcuInsGeneratorV1 insGen;
    std::shared_ptr<CcuRep::CcuRepBase> rep = std::make_shared<CcuRep::CcuRepBlock>(&insGen, "test_rep");

    EXPECT_THROW(AppendToContext(nullptr, rep), Hccl::CcuApiException);
}
