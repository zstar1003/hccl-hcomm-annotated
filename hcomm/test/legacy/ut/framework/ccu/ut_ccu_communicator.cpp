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
#define private public
#define protected public
#include "ccu_ctx.h"
#include "ccu_ins_group.h"
#include "ccu_ctx_mgr.h"
#include "ccu_ins_group.h"
#include "ccu_ins_preprocessor.h"
#include "ccu_registered_ctx_mgr.h"
#include "hierarchical_queue.h"
#include "ccu_communicator.h"
#include "ccu_device_manager.h"
#include "ccu_instruction_all_gather_mesh1d.h"
#include "ccu_context_all_gather_mesh1d.h"
#include "ccu_ctx_signature.h"
#include "env_func.h"
#include "communicator_impl.h"
#undef private
#undef protected

using namespace Hccl;

using namespace std;

class CcuCommunicatorTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CommunicatorImplTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CommunicatorImplTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CommunicatorImplTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CommunicatorImplTest TearDown" << std::endl;
    }
};

TEST_F(CcuCommunicatorTest, should_return_success_when_calling_register)
{
    // when
    MOCKER(HrtGetDevice).stubs().will(returnValue(1));
    CommunicatorImpl* comm;
    std::unique_ptr<CcuCommunicator> communicator = std::make_unique<CcuCommunicator>(comm);
    EXPECT_NE(nullptr, communicator);
    EXPECT_NE(nullptr, communicator->GetCcuResPackMgr());
    EXPECT_NE(nullptr, communicator->GetCcuTransportGrpMgr());
    EXPECT_NE(nullptr, communicator->GetCcuTransportMgr());
    EXPECT_NE(nullptr, communicator->GetCcuJettyMgr());
    EXPECT_EQ(1, communicator->GetDeviceLogicId());
}

TEST_F(CcuCommunicatorTest, should_return_success_when_calling_castbin2uInt)
{
    EXPECT_EQ(CastBin2UInt("1"), 1);
    EXPECT_THROW(CastBin2UInt("3"), InvalidParamsException);
}
