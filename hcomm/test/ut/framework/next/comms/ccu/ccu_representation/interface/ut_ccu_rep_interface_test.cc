/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_v1.h"
#include "ccu_datatype_v1.h"
#include "ccu_funcblock_v1.h"
#include "ccu_funccall_v1.h"
#include "ccu_loopblock_v1.h"
#include "ccu_loopcall_v1.h"
#include "ccu_condition_v1.h"
#include "ccu_repeat_v1.h"
#include "ccu_interface_assist_v1.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_ins_generator_v2.h"

#include "ccu_api_exception.h"

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <string>
#include <memory>

namespace hcomm {
namespace CcuRep {
    namespace {

        class CcuRepInterfaceTest : public ::testing::Test {
        protected:
            void SetUp() override
            {
                static CcuInsGeneratorV2 insGen;
                MOCKER_CPP(&CcuRepContext::GetInsGenerator)
                    .stubs()
                    .will(returnValue(static_cast<CcuInsGeneratorBase*>(&insGen)));
            }
            void TearDown() override
            {
                mockcpp::GlobalMockObject::verify();
                mockcpp::GlobalMockObject::reset();
            }
        };

        TEST_F(CcuRepInterfaceTest, CcuPhyRes_IdAndDieId)
        {
            CcuPhyRes phyRes;
            phyRes.Reset(10);
            EXPECT_EQ(phyRes.Id(), 10);

            phyRes.SetDieId(5);
            EXPECT_EQ(phyRes.DieId(), 5);
        }

        TEST_F(CcuRepInterfaceTest, CcuPhyRes_ResetWithIdAndDieId)
        {
            CcuVirRes virRes(nullptr);
            virRes.Reset(20, 3);
            EXPECT_EQ(virRes.Id(), 20);
            EXPECT_EQ(virRes.DieId(), 3);
        }

        TEST_F(CcuRepInterfaceTest, CcuVirRes_IdAndDieId)
        {
            CcuRepContext context;
            CcuVirRes virRes(&context);
            virRes.Reset(15);
            virRes.SetDieId(7);
            EXPECT_EQ(virRes.Id(), 15);
            EXPECT_EQ(virRes.DieId(), 7);
        }

        TEST_F(CcuRepInterfaceTest, CcuBuffer_Id)
        {
            CcuRepContext context;
            CcuBuffer buffer(&context);
            buffer.Reset(3);
            buffer.SetDieId(1);
            uint16_t expectedId = 3 + (1 << 15);
            EXPECT_EQ(buffer.Id(), expectedId);
        }

        TEST_F(CcuRepInterfaceTest, CcuBuf_Id)
        {
            CcuRepContext context;
            CcuBuf buf(&context);
            buf.Reset(4);
            buf.SetDieId(2);
            uint16_t expectedId = 4 + (2 << 15);
            EXPECT_EQ(buf.Id(), expectedId);
        }

        TEST_F(CcuRepInterfaceTest, LocalNotify_Constructor)
        {
            CcuRepContext context;
            LocalNotify notify(&context);
            notify.Reset(8);
            EXPECT_EQ(notify.Id(), 8);
        }

        TEST_F(CcuRepInterfaceTest, Executor_Constructor)
        {
            CcuRepContext context;
            Executor executor(&context);
            executor.Reset(12);
            EXPECT_EQ(executor.Id(), 12);
        }

        TEST_F(CcuRepInterfaceTest, Variable_CopyConstructor)
        {
            CcuRepContext context;
            Variable var1(&context);
            var1.Reset(5);

            Variable var2(var1);
            EXPECT_EQ(var2.Id(), 5);
        }

        TEST_F(CcuRepInterfaceTest, Address_CopyConstructor)
        {
            CcuRepContext context;
            Address addr1(&context);
            addr1.Reset(7);

            Address addr2(addr1);
            EXPECT_EQ(addr2.Id(), 7);
        }

        TEST_F(CcuRepInterfaceTest, AppendToContext_NullContext)
        {
            CcuInsGeneratorV1 insGen;
            auto nop = std::make_shared<CcuRepNop>(&insGen);
            EXPECT_THROW(AppendToContext(nullptr, nop), Hccl::CcuApiException);
        }

        TEST_F(CcuRepInterfaceTest, CurrentBlock_NullContext)
        {
            EXPECT_THROW(CurrentBlock(nullptr), Hccl::CcuApiException);
        }

        TEST_F(CcuRepInterfaceTest, SetCurrentBlock_NullContext)
        {
            std::shared_ptr<CcuRepBlock> block = nullptr;
            EXPECT_THROW(SetCurrentBlock(nullptr, block), Hccl::CcuApiException);
        }

        TEST_F(CcuRepInterfaceTest, CreateVariable_NullContext)
        {
            EXPECT_THROW(CreateVariable(nullptr), Hccl::CcuApiException);
        }

        TEST_F(CcuRepInterfaceTest, CcuRepContext_AppendAndCurrentBlock)
        {
            CcuRepContext context;

            auto block1 = context.CurrentBlock();
            EXPECT_NE(block1, nullptr);

            auto repFuncBlock = std::make_shared<CcuRepFuncBlock>(nullptr, "testFunc");
            AppendToContext(&context, repFuncBlock);

            auto block2 = context.CurrentBlock();
            EXPECT_NE(block2, nullptr);
        }

        TEST_F(CcuRepInterfaceTest, FuncBlock_ConstructorAndDestructor)
        {
            CcuRepContext context;
            {
                FuncBlock funcBlock(&context, "testFunc", 1);
                auto currentBlock = CurrentBlock(&context);
                EXPECT_NE(currentBlock, nullptr);
            }
            auto blockAfter = CurrentBlock(&context);
            EXPECT_NE(blockAfter, nullptr);
        }

        TEST_F(CcuRepInterfaceTest, FuncCall_ConstructorWithLabel)
        {
            CcuRepContext context;
            CcuInsGeneratorV1 insGen;
            context.SetInsGenerater(&insGen);
            FuncCall funcCall(&context, "testCall");
            funcCall.AppendToContext();

            auto reps = context.GetRepSequence();
            EXPECT_EQ(reps.size(), 1);
        }

        TEST_F(CcuRepInterfaceTest, FuncCall_ConstructorWithVariable)
        {
            CcuRepContext context;
            CcuInsGeneratorV1 insGen;
            context.SetInsGenerater(&insGen);
            Variable funcAddr(&context);
            FuncCall funcCall(&context, funcAddr);
            funcCall.AppendToContext();

            auto reps = context.GetRepSequence();
            EXPECT_EQ(reps.size(), 1);
        }

        TEST_F(CcuRepInterfaceTest, LoopBlock_ConstructorAndDestructor)
        {
            CcuRepContext context;
            {
                LoopBlock loopBlock(&context, "testLoop");
                auto currentBlock = CurrentBlock(&context);
                EXPECT_NE(currentBlock, nullptr);
            }
            EXPECT_THROW(CurrentBlock(&context), Hccl::CcuApiException);
        }

        TEST_F(CcuRepInterfaceTest, LoopCall_ConstructorAndAppend)
        {
            CcuRepContext context;
            LoopCall loopCall(&context, "testLoopCall");

            EXPECT_EQ(loopCall.GetLabel(), "testLoopCall");
        }

        TEST_F(CcuRepInterfaceTest, CcuRepContext_SetAndGetDieId)
        {
            CcuRepContext context;
            context.SetDieId(3);
            EXPECT_EQ(context.GetDieId(), 3);
        }

        TEST_F(CcuRepInterfaceTest, CcuRepContext_SetAndGetMissionId)
        {
            CcuRepContext context;
            context.SetMissionId(5);
            EXPECT_EQ(context.GetMissionId(), 5);
        }

        TEST_F(CcuRepInterfaceTest, CcuRepContext_SetAndGetMissionKey)
        {
            CcuRepContext context;
            context.SetMissionKey(100);
            EXPECT_EQ(context.GetMissionKey(), 100);
        }

        TEST_F(CcuRepInterfaceTest, Variable_AssignmentOperator)
        {
            CcuRepContext context;
            Variable var1(&context);
            var1.Reset(1);

            Variable var2(&context);
            var2.Reset(2);

            var1 = 100;
        }

        TEST_F(CcuRepInterfaceTest, Address_AssignmentOperator)
        {
            CcuRepContext context;
            Address addr1(&context);
            addr1.Reset(1);

            Address addr2(&context);
            addr2.Reset(2);

            addr1 = 200;
        }

        TEST_F(CcuRepInterfaceTest, Variable_NotEqualOperator)
        {
            CcuRepContext context;
            Variable var(&context);
            var.Reset(0);

            auto rel = var != 10;
            EXPECT_EQ(rel.type, CcuRelationalOperatorType::NOT_EQUAL);
        }

        TEST_F(CcuRepInterfaceTest, Variable_EqualOperator)
        {
            CcuRepContext context;
            Variable var(&context);
            var.Reset(0);

            auto rel = var == 5;
            EXPECT_EQ(rel.type, CcuRelationalOperatorType::EQUAL);
        }

        TEST_F(CcuRepInterfaceTest, FuncBlock_OperatorParentheses)
        {
            CcuRepContext context;
            FuncBlock funcBlock(&context, "testFunc", 1);
        }

        TEST_F(CcuRepInterfaceTest, LoopBlock_OperatorParentheses)
        {
            CcuRepContext context;
            LoopBlock loopBlock(&context, "testLoop");
        }

        TEST_F(CcuRepInterfaceTest, FuncCall_OperatorParentheses)
        {
            CcuRepContext context;
            CcuInsGeneratorV1 insGen;
            context.SetInsGenerater(&insGen);
            FuncCall funcCall(&context, "testCall");
            funcCall();
        }

        TEST_F(CcuRepInterfaceTest, LoopCall_OperatorParentheses)
        {
            CcuRepContext context;
            LoopCall loopCall(&context, "testLoopCall");
        }

        TEST_F(CcuRepInterfaceTest, Condition_EQ_RelationalOperator)
        {
            CcuRepContext context;
            Variable counter(&context);
            counter.Reset(0);

            auto rel = (counter == 10);
            EXPECT_EQ(rel.type, CcuRelationalOperatorType::EQUAL);
        }

        TEST_F(CcuRepInterfaceTest, Repeat_NE_RelationalOperator)
        {
            CcuRepContext context;
            Variable counter(&context);
            counter.Reset(0);

            auto rel = (counter != 10);
            EXPECT_EQ(rel.type, CcuRelationalOperatorType::NOT_EQUAL);
        }

        TEST_F(CcuRepInterfaceTest, Variable_ArithmeticAssignment)
        {
            CcuRepContext context;
            Variable varA(&context);
            varA.Reset(1);
            Variable varB(&context);
            varB.Reset(2);

            varA = varA + varB;
            varA = varA - varB;
            varA = varA * varB;
            varA += varB;
            varA += uint16_t(10);
            varA -= varB;
            varA -= uint16_t(10);
            varA *= varB;
            varA *= uint16_t(10);
        }

        TEST_F(CcuRepInterfaceTest, Address_ArithmeticAssignment)
        {
            CcuRepContext context;
            Address addrA(&context);
            addrA.Reset(1);
            Address addrB(&context);
            addrB.Reset(2);
            Variable varB(&context);
            varB.Reset(3);

            addrA = addrA + varB;
            addrA = addrA + addrB;
            addrA = addrA + uint16_t(10);
            addrA += varB;
        }

        TEST_F(CcuRepInterfaceTest, Variable_VariableAssignment)
        {
            CcuRepContext context;
            Variable varA(&context);
            varA.Reset(1);
            Variable varB(&context);
            varB.Reset(2);

            varA = varB;
        }

        TEST_F(CcuRepInterfaceTest, Address_AddressAndVariableAssignment)
        {
            CcuRepContext context;
            Address addrA(&context);
            addrA.Reset(1);
            Address addrB(&context);
            addrB.Reset(2);
            Variable varB(&context);
            varB.Reset(3);

            addrA = addrB;
            addrA = varB;
        }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
