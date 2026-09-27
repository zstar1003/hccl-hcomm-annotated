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
#undef private
#undef protected

using namespace Hccl;

using namespace std;

class CcuResPackMgrTest : public testing::Test {
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

TEST(CcuResPackMgrTest, should_return_success_when_calling_preparealloc)
{
    // check
    CcuResPackMgr ccuResPackMgr;
    u32 expectSize = 3;
    cout << ccuResPackMgr.resPacks.size() << endl;
    EXPECT_NO_THROW(ccuResPackMgr.PrepareAlloc(expectSize));
    EXPECT_EQ(expectSize, ccuResPackMgr.resPacks.size());
    EXPECT_EQ(expectSize, ccuResPackMgr.unConfirmedNum);
}

TEST(CcuResPackMgrTest, should_return_success_when_calling_getrespack)
{
    // check
    CcuResPackMgr ccuResPackMgr;
    u32 expectSize = 3;
    EXPECT_NO_THROW(ccuResPackMgr.PrepareAlloc(expectSize));
    EXPECT_EQ(expectSize, ccuResPackMgr.resPacks.size());
    EXPECT_EQ(expectSize, ccuResPackMgr.unConfirmedNum);

    CcuResPack& ccuResPack = ccuResPackMgr.GetCcuResPack(0);
    EXPECT_EQ(0, ccuResPack.handles.size());
}

TEST(CcuResPackMgrTest, should_return_success_when_calling_firm)
{
    // check
    CcuResPackMgr ccuResPackMgr;
    u32 expectSize = 3;
    EXPECT_NO_THROW(ccuResPackMgr.PrepareAlloc(expectSize));
    EXPECT_EQ(expectSize, ccuResPackMgr.resPacks.size());
    EXPECT_EQ(expectSize, ccuResPackMgr.unConfirmedNum);

    EXPECT_NO_THROW(ccuResPackMgr.Confirm());
    EXPECT_EQ(expectSize, ccuResPackMgr.resPacks.size());
    EXPECT_EQ(0, ccuResPackMgr.unConfirmedNum);
}

TEST(CcuResPackMgrTest, should_return_success_when_calling_fallback)
{
    // check
    CcuResPackMgr ccuResPackMgr;
    u32 expectSize = 3;
    EXPECT_NO_THROW(ccuResPackMgr.PrepareAlloc(expectSize));
    EXPECT_EQ(expectSize, ccuResPackMgr.resPacks.size());
    EXPECT_EQ(expectSize, ccuResPackMgr.unConfirmedNum);

    EXPECT_NO_THROW(ccuResPackMgr.Fallback());
    EXPECT_EQ(0, ccuResPackMgr.resPacks.size());
    EXPECT_EQ(0, ccuResPackMgr.unConfirmedNum);
}
