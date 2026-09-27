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
#include <mockcpp/mockcpp.hpp>
#include <stdio.h>

#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "llt_hccl_stub_pub.h"
#include "alg_configurator.h"

#define private public
#define protected public
#include "coll_alg_op_registry.h"
#undef private
#undef protected
using namespace std;
using namespace hccl;

class CollRegistryTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--CollRegistryTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--CollRegistryTest TearDown--\033[0m" << std::endl; }
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(CollRegistryTest, get_op_fail)
{
    std::unique_ptr<hcclImpl> pimpl_;
    std::unique_ptr<TopoMatcher> topoMatcher_;
    AlgConfigurator* algConfigurator = nullptr;
    CCLBufferManager cclBufferManager;
    HcclDispatcher dispatcher = nullptr;

    std::unique_ptr<CollAlgOperator> algOperator = CollAlgOpRegistry::Instance().GetAlgOp(
        HcclCMDType::HCCL_CMD_INVALID, algConfigurator, cclBufferManager, dispatcher, topoMatcher_);
    EXPECT_TRUE(algOperator == nullptr);
    GlobalMockObject::verify();
}
