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
#include "ins_exe_que.h"
#include "port.h"
#undef private
#undef protected

using namespace Hccl;

using namespace std;

class RegisteredCcuCtxMgrTest : public testing::Test {
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

TEST_F(RegisteredCcuCtxMgrTest, should_return_fail_when_calling_hasregistered)
{
    // when
    RegisteredCcuCtxMgr registeredCcuCtxMgr(1);
    CcuCtxSignature ctxSignature;
    ctxSignature.Append("a");
    uintptr_t resPackId = reinterpret_cast<uintptr_t>(&ctxSignature);
    u64 execId = 0;
    bool res = registeredCcuCtxMgr.HasRegistered(ctxSignature, resPackId, execId);

    // check
    EXPECT_EQ(false, res);
    EXPECT_EQ(0, execId);
    EXPECT_EQ(0, registeredCcuCtxMgr.registeredIds.size());
}

TEST_F(RegisteredCcuCtxMgrTest, should_return_success_when_calling_register)
{
    // when
    MOCKER(InsExeQue::RegisterExtendInstruction)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(InsExeQue::DeregisterExtendInstruction)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    RegisteredCcuCtxMgr registeredCcuCtxMgr(1);
    CcuCtxSignature ctxSignature;
    ctxSignature.Append("a");
    uintptr_t resPackId = reinterpret_cast<uintptr_t>(&ctxSignature);
    u64 execId = 0;
    std::unique_ptr<CcuCtxGroup> ccuCtxGroup = std::make_unique<CcuCtxGroup>();
    u64 resExecId = registeredCcuCtxMgr.Register(std::move(ccuCtxGroup), ctxSignature, resPackId, false);

    // check
    EXPECT_EQ(0, resExecId);
    EXPECT_EQ(1, registeredCcuCtxMgr.registeredIds.size());

    u64 execId2 = 0;
    bool res = registeredCcuCtxMgr.HasRegistered(ctxSignature, resPackId, execId2);

    // check
    EXPECT_EQ(true, res);
    EXPECT_EQ(resExecId, execId);
    EXPECT_EQ(1, registeredCcuCtxMgr.registeredIds.size());
}
