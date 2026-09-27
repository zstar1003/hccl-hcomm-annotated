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
#include "ccu_ins_generator_v1.h"
#include "ccu_api_exception.h"
#include <gtest/gtest.h>
#include <string>
#include <memory>

namespace hcomm {
namespace CcuRep {
    namespace {

        class CcuRepControlTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
            void TearDown() override {}
        };

        class CcuRepFuncBlockTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        class CcuRepFuncCallTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        class CcuRepJumpTest : public ::testing::Test {
        protected:
            void SetUp() override {}
            CcuInsGeneratorV1 insGen{};
        };

        class CcuRepJumpLabelTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        TEST_F(CcuRepFuncBlockTest, Constructor_Label)
        {
            CcuRepFuncBlock funcBlock(&insGen, "testFunc");
            EXPECT_EQ(funcBlock.Type(), CcuRepType::FUNC_BLOCK);
            EXPECT_EQ(funcBlock.GetLabel(), "testFunc");
        }

        TEST_F(CcuRepFuncBlockTest, Describe)
        {
            CcuRepFuncBlock funcBlock(&insGen, "testFunc");
            std::string desc = funcBlock.Describe();
            EXPECT_NE(desc.find("FuncBlock[testFunc]"), std::string::npos);
        }

        TEST_F(CcuRepFuncBlockTest, SetFuncManager)
        {
            CcuRepFuncBlock funcBlock(&insGen, "testFunc");
            CcuRepReferenceManager funcManager(0);
            funcBlock.SetFuncManager(&funcManager);
            EXPECT_NO_FATAL_FAILURE(funcBlock.SetFuncManager(&funcManager));
        }

        TEST_F(CcuRepFuncBlockTest, DefineInArg_SingleVariable)
        {
            CcuRepFuncBlock funcBlock(&insGen, "testFunc");
            CcuRepContext context;
            Variable var(&context);
            EXPECT_NO_FATAL_FAILURE(funcBlock.DefineInArg(var));
        }

        TEST_F(CcuRepFuncBlockTest, DefineOutArg_SingleVariable)
        {
            CcuRepFuncBlock funcBlock(&insGen, "testFunc");
            CcuRepContext context;
            Variable var(&context);
            EXPECT_NO_FATAL_FAILURE(funcBlock.DefineOutArg(var));
        }

        TEST_F(CcuRepFuncBlockTest, DefineInArg_VariableList)
        {
            CcuRepFuncBlock funcBlock(&insGen, "testFunc");
            CcuRepContext context;
            std::vector<Variable> varList = {Variable(&context)};
            EXPECT_NO_FATAL_FAILURE(funcBlock.DefineInArg(varList));
        }

        TEST_F(CcuRepFuncBlockTest, DefineOutArg_VariableList)
        {
            CcuRepFuncBlock funcBlock(&insGen, "testFunc");
            CcuRepContext context;
            std::vector<Variable> varList = {Variable(&context)};
            EXPECT_NO_FATAL_FAILURE(funcBlock.DefineOutArg(varList));
        }

        TEST_F(CcuRepFuncBlockTest, SetCallLayer_Valid)
        {
            CcuRepFuncBlock funcBlock(&insGen, "testFunc");
            EXPECT_NO_FATAL_FAILURE(funcBlock.SetCallLayer(1));
            EXPECT_EQ(funcBlock.GetCallLayer(), 1);
        }

        TEST_F(CcuRepFuncBlockTest, InstrCount)
        {
            CcuRepFuncBlock funcBlock(&insGen, "testFunc");
            CcuRepContext context;
            Variable var(&context);
            funcBlock.DefineInArg(var);
            funcBlock.DefineOutArg(var);
            uint16_t count = funcBlock.InstrCount();
            EXPECT_GE(count, 2);
        }

        TEST_F(CcuRepFuncCallTest, Constructor_Label)
        {
            CcuRepFuncCall funcCall(&insGen, "testCall");
            EXPECT_EQ(funcCall.Type(), CcuRepType::FUNC_CALL);
            EXPECT_EQ(funcCall.GetLabel(), "testCall");
        }

        TEST_F(CcuRepFuncCallTest, Constructor_FuncAddrVar)
        {
            CcuRepContext context;
            Variable funcAddr(&context);
            CcuRepFuncCall funcCall(&insGen, funcAddr);
            EXPECT_EQ(funcCall.Type(), CcuRepType::FUNC_CALL);
        }

        TEST_F(CcuRepFuncCallTest, Describe)
        {
            CcuRepFuncCall funcCall(&insGen, "testCall");
            std::string desc = funcCall.Describe();
            EXPECT_NE(desc.find("FuncCall[testCall]"), std::string::npos);
        }

        TEST_F(CcuRepFuncCallTest, SetFuncManager)
        {
            CcuRepFuncCall funcCall(&insGen, "testCall");
            CcuRepReferenceManager funcManager(0);
            EXPECT_NO_FATAL_FAILURE(funcCall.SetFuncManager(&funcManager));
        }

        TEST_F(CcuRepFuncCallTest, Reference)
        {
            CcuRepFuncCall funcCall(&insGen, "testCall");
            CcuRepFuncBlock funcBlock(&insGen, "testFunc");
            auto funcBlockPtr = std::make_shared<CcuRepFuncBlock>(funcBlock);
            EXPECT_NO_FATAL_FAILURE(funcCall.Reference(funcBlockPtr));
        }

        TEST_F(CcuRepFuncCallTest, SetInArg_SingleVariable)
        {
            CcuRepFuncCall funcCall(&insGen, "testCall");
            CcuRepContext context;
            Variable var(&context);
            EXPECT_NO_FATAL_FAILURE(funcCall.SetInArg(var));
        }

        TEST_F(CcuRepFuncCallTest, SetOutArg_SingleVariable)
        {
            CcuRepFuncCall funcCall(&insGen, "testCall");
            CcuRepContext context;
            Variable var(&context);
            EXPECT_NO_FATAL_FAILURE(funcCall.SetOutArg(var));
        }

        TEST_F(CcuRepFuncCallTest, SetInArg_VariableList)
        {
            CcuRepFuncCall funcCall(&insGen, "testCall");
            CcuRepContext context;
            std::vector<Variable> varList = {Variable(&context)};
            EXPECT_NO_FATAL_FAILURE(funcCall.SetInArg(varList));
        }

        TEST_F(CcuRepFuncCallTest, SetOutArg_VariableList)
        {
            CcuRepFuncCall funcCall(&insGen, "testCall");
            CcuRepContext context;
            std::vector<Variable> varList = {Variable(&context)};
            EXPECT_NO_FATAL_FAILURE(funcCall.SetOutArg(varList));
        }

        TEST_F(CcuRepFuncCallTest, InstrCount)
        {
            CcuRepFuncCall funcCall(&insGen, "testCall");
            CcuRepContext context;
            Variable var(&context);
            funcCall.SetInArg(var);
            funcCall.SetOutArg(var);
            uint16_t count = funcCall.InstrCount();
            EXPECT_GE(count, 4);
        }

        TEST_F(CcuRepJumpTest, Constructor_Label)
        {
            CcuRepContext context;
            Variable targetInstrId(&context);
            CcuRepJump jump(&insGen, "testJump", targetInstrId);
            EXPECT_EQ(jump.Type(), CcuRepType::JUMP);
        }

        TEST_F(CcuRepJumpTest, Describe)
        {
            CcuRepContext context;
            Variable targetInstrId(&context);
            CcuRepJump jump(&insGen, "testJump", targetInstrId);
            std::string desc = jump.Describe();
            EXPECT_NE(desc.find("Jump To Label[testJump]"), std::string::npos);
        }

        TEST_F(CcuRepJumpTest, Translate_WithJumpLabel)
        {
            CcuRepContext context;
            Variable targetInstrId(&context);
            CcuRepJumpLabel jumpLabel(&insGen, "testLabel");
            auto jumpLabelPtr = std::make_shared<CcuRepJumpLabel>(jumpLabel);

            CcuRepJump jump(&insGen, "testJump", targetInstrId);
            jump.Reference(jumpLabelPtr);

            CcuInstr instr[2] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.reserveXnId = 1;

            jumpLabelPtr->Translate(nullptr, instrPtr, instrId, dep);
            instrId = 0;
            instrPtr = instr;
            bool result = jump.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
            EXPECT_EQ(jump.Translated(), true);
        }

        TEST_F(CcuRepJumpTest, Constructor_JumpNE)
        {
            CcuRepContext context;
            Variable targetInstrId(&context);
            Variable condition(&context);
            CcuRepJumpNE jumpNE(&insGen, "testJumpNE", targetInstrId, condition, condition, 100);
            EXPECT_EQ(jumpNE.Type(), CcuRepType::JUMP_NE);
        }

        TEST_F(CcuRepJumpTest, Describe_JumpNE)
        {
            CcuRepContext context;
            Variable targetInstrId(&context);
            Variable condition(&context);
            CcuRepJumpNE jumpNE(&insGen, "testJumpNE", targetInstrId, condition, condition, 100);
            std::string desc = jumpNE.Describe();
            EXPECT_NE(desc.find("Jump To Label[testJumpNE]"), std::string::npos);
            EXPECT_NE(desc.find("Condition"), std::string::npos);
        }

        TEST_F(CcuRepJumpTest, Translate_JumpNE_WithJumpLabel)
        {
            CcuRepContext context;
            Variable targetInstrId(&context);
            Variable condition(&context);
            CcuRepJumpLabel jumpLabel(&insGen, "testLabel");
            auto jumpLabelPtr = std::make_shared<CcuRepJumpLabel>(jumpLabel);

            CcuRepJumpNE jumpNE(&insGen, "testJumpNE", targetInstrId, condition, condition, 100);
            jumpNE.Reference(jumpLabelPtr);

            CcuInstr instr[2] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.reserveXnId = 1;

            jumpLabelPtr->Translate(nullptr, instrPtr, instrId, dep);
            instrId = 0;
            instrPtr = instr;
            bool result = jumpNE.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
        }

        TEST_F(CcuRepJumpTest, Constructor_JumpEQ)
        {
            CcuRepContext context;
            Variable targetInstrId(&context);
            Variable condition(&context);
            CcuRepJumpEQ jumpEQ(&insGen, "testJumpEQ", targetInstrId, condition, condition, 50);
            EXPECT_EQ(jumpEQ.Type(), CcuRepType::JUMP_EQ);
        }

        TEST_F(CcuRepJumpTest, Describe_JumpEQ)
        {
            CcuRepContext context;
            Variable targetInstrId(&context);
            Variable condition(&context);
            CcuRepJumpEQ jumpEQ(&insGen, "testJumpEQ", targetInstrId, condition, condition, 50);
            std::string desc = jumpEQ.Describe();
            EXPECT_NE(desc.find("Jump To Label[testJumpEQ]"), std::string::npos);
            EXPECT_NE(desc.find("Be equal"), std::string::npos);
        }

        TEST_F(CcuRepJumpTest, Translate_JumpEQ_WithJumpLabel)
        {
            CcuRepContext context;
            Variable targetInstrId(&context);
            Variable condition(&context);
            CcuRepJumpLabel jumpLabel(&insGen, "testLabel");
            auto jumpLabelPtr = std::make_shared<CcuRepJumpLabel>(jumpLabel);

            CcuRepJumpEQ jumpEQ(&insGen, "testJumpEQ", targetInstrId, condition, condition, 50);
            jumpEQ.Reference(jumpLabelPtr);

            CcuInstr instr[5] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.reserveXnId = 1;

            jumpLabelPtr->Translate(nullptr, instrPtr, instrId, dep);
            instrId = 0;
            instrPtr = instr;
            bool result = jumpEQ.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
        }

        TEST_F(CcuRepJumpLabelTest, Constructor_Label)
        {
            CcuRepJumpLabel jumpLabel(&insGen, "testLabel");
            EXPECT_EQ(jumpLabel.Type(), CcuRepType::JUMP_LABEL);
            EXPECT_EQ(jumpLabel.GetLabel(), "testLabel");
        }

        TEST_F(CcuRepJumpLabelTest, Describe)
        {
            CcuRepJumpLabel jumpLabel(&insGen, "testLabel");
            std::string desc = jumpLabel.Describe();
            EXPECT_NE(desc.find("JumpLabel[testLabel]"), std::string::npos);
        }

        TEST_F(CcuRepJumpLabelTest, AppendRep)
        {
            CcuRepJumpLabel jumpLabel(&insGen, "testLabel");
            auto nop = std::make_shared<CcuRepNop>(&insGen);
            EXPECT_NO_FATAL_FAILURE(jumpLabel.Append(nop));
            EXPECT_EQ(jumpLabel.GetReps().size(), 2);
        }

        TEST_F(CcuRepControlTest, CcuRepJumpBase_Reference)
        {
            CcuInsGeneratorV1 insGen;
            CcuRepContext context;
            Variable targetInstrId(&context);
            CcuRepJumpLabel jumpLabel(&insGen, "testLabel");
            auto jumpLabelPtr = std::make_shared<CcuRepJumpLabel>(jumpLabel);

            CcuRepJump jump(&insGen, "testJump", targetInstrId);
            jump.Reference(jumpLabelPtr);

            CcuInstr instr[2] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.reserveXnId = 1;

            jumpLabelPtr->Translate(nullptr, instrPtr, instrId, dep);
            instrId = 0;
            instrPtr = instr;
            bool result = jump.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
            EXPECT_EQ(jump.Translated(), true);
        }

        // TEST_F(CcuRepControlTest, CcuRepTranslator_CheckTypeNullptr_Throw)
        // {
        //     CcuInsGeneratorV1 insGen;
        //     CcuRepReferenceManager funcManager(0);
        //     funcManager.SetRefBlock("nullFunc", nullptr);

        //     CcuRepFuncBlock funcBlock(&insGen, "outerFunc");
        //     auto funcCallPtr = std::make_shared<CcuRepFuncCall>("nullFunc");
        //     funcBlock.Append(funcCallPtr);
        //     funcBlock.SetFuncManager(&funcManager);

        //     CcuInstr instr[128] = {};
        //     CcuInstr *instrPtr = instr;
        //     uint16_t instrId = 0;
        //     TransDep dep = {};
        //     dep.reserveXnId = 1;

        //     EXPECT_THROW(funcBlock.Translate(nullptr, instrPtr, instrId, dep), Hccl::CcuApiException);
        // }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
