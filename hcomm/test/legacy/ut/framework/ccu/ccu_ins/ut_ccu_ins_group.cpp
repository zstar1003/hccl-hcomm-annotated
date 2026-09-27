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
#include "internal_exception.h"
#undef private
#undef protected

using namespace Hccl;

using namespace std;

class CcuInsGroupTest : public testing::Test {
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

TEST_F(CcuInsGroupTest, should_return_success_when_calling_setexecId)
{
    // check
    CcuInsGroup insGroup;
    std::unique_ptr<CcuInstruction> ins = std::make_unique<CcuInstructionAllGatherMesh1D>();
    insGroup.Append(std::move(ins));
    insGroup.SetExecId(100);
    EXPECT_EQ(100, insGroup.GetExecId());
}

TEST_F(CcuInsGroupTest, should_return_success_when_calling_append)
{
    // check
    CcuInsGroup insGroup;
    std::unique_ptr<CcuInstruction> ins = std::make_unique<CcuInstructionAllGatherMesh1D>();
    insGroup.Append(std::move(ins));
    EXPECT_EQ(insGroup.GetCcuInstructions().size(), 1);
}

TEST_F(CcuInsGroupTest, should_return_success_when_calling_describe)
{
    // check
    CcuInsGroup insGroup;
    EXPECT_EQ(insGroup.GetExecId(), 0);
    EXPECT_EQ(insGroup.Describe(), "CcuInsGroup[ccuInstructions_size=0, execId=0]");

    // check2
    std::unique_ptr<CcuInstruction> ins = std::make_unique<CcuInstructionAllGatherMesh1D>();
    insGroup.Append(std::move(ins));
    EXPECT_EQ(insGroup.ccuInstructions.size(), 1);
    EXPECT_EQ(insGroup.Describe(), "CcuInsGroup[ccuInstructions_size=1, execId=0]");
}

TEST_F(CcuInsGroupTest, should_return_success_when_calling_getCtxSignature)
{
    // when
    CcuCtxSignature ctxSignature;
    ctxSignature.Append("a");
    MOCKER(GenerateCcuCtxSignature)
        .stubs()
        .with(outBound(ctxSignature), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    // check
    CcuInsGroup insGroup;
    std::unique_ptr<CcuInstruction> ins = std::make_unique<CcuInstructionAllGatherMesh1D>();
    insGroup.Append(std::move(ins));
    EXPECT_EQ(insGroup.GetCtxSignature(), ctxSignature);
}

TEST_F(CcuInsGroupTest, should_return_success_when_calling_other)
{
    // check
    CcuInsGroup insGroup;
    EXPECT_THROW(insGroup.GetTaskArg(), InternalException);
    EXPECT_THROW(insGroup.GetCtxArg(), NotSupportException);

    std::unique_ptr<CcuInstructionAllGatherMesh1D> ins = std::make_unique<CcuInstructionAllGatherMesh1D>();
    std::vector<LinkData> links;
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    links.push_back(link);
    ins->SetLinks(links);
    std::unique_ptr<CcuInstruction> ins2 = std::move(ins);
    insGroup.Append(std::move(ins2));
    EXPECT_EQ(insGroup.GetLinks().size(), 1);
}
