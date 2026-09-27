/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_add_v1.h"
#include "ccu_rep_assign_v1.h"
#include "ccu_datatype_v1.h"
#include "ccu_microcode_v1.h"
#include "ccu_rep_base_v1.h"
#include "ccu_rep_type_v1.h"
#include "ccu_operator_v1.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_ins_generator_v2.h"
#include "ccu_rep_mul_v1.h"
#include "ccu_rep_sub_v1.h"
#include <gtest/gtest.h>
#include <string>

namespace hcomm {
namespace CcuRep {
    namespace {

        class CcuRepAddTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            CcuInstr instr{};
            uint16_t instrId{0};
            TransDep dep{};

            void SetUp() override
            {
                memset(&instr, 0, sizeof(instr));
                dep.reserveGsaId = 1;
                dep.reserveXnId = 2;
            }
        };

        TEST_F(CcuRepAddTest, Constructor_AddrPlusVarToAddr)
        {
            Address addrC;
            Address addrA;
            Variable varB;
            addrC.Reset(1);
            addrA.Reset(2);
            varB.Reset(3);
            CcuRepAdd rep(&insGen, addrC, addrA, varB);

            EXPECT_EQ(rep.GetSubType(), AddSubType::ADDR_PLUS_VAR_TO_ADDR);
            EXPECT_EQ(rep.Type(), CcuRepType::ADD);
        }

        TEST_F(CcuRepAddTest, Constructor_AddrPlusAddrToAddr)
        {
            Address addrC;
            Address addrA;
            Address addrB;
            addrC.Reset(1);
            addrA.Reset(2);
            addrB.Reset(3);
            CcuRepAdd rep(&insGen, addrC, addrA, addrB);

            EXPECT_EQ(rep.GetSubType(), AddSubType::ADDR_PLUS_ADDR_TO_ADDR);
            EXPECT_EQ(rep.Type(), CcuRepType::ADD);
        }

        TEST_F(CcuRepAddTest, Constructor_VarPlusVarToVar)
        {
            Variable varC;
            Variable varA;
            Variable varB;
            varC.Reset(1);
            varA.Reset(2);
            varB.Reset(3);
            CcuRepAdd rep(&insGen, varC, varA, varB);

            EXPECT_EQ(rep.GetSubType(), AddSubType::VAR_PLUS_VAR_TO_VAR);
            EXPECT_EQ(rep.Type(), CcuRepType::ADD);
        }

        TEST_F(CcuRepAddTest, Constructor_SelfAddAddress)
        {
            Address addrA;
            Variable offset;
            addrA.Reset(1);
            offset.Reset(2);
            CcuRepAdd rep(&insGen, addrA, offset);

            EXPECT_EQ(rep.GetSubType(), AddSubType::SELF_ADD_ADDRESS);
            EXPECT_EQ(rep.Type(), CcuRepType::ADD);
        }

        TEST_F(CcuRepAddTest, Constructor_SelfAddVariable)
        {
            Variable varA;
            Variable offset;
            varA.Reset(1);
            offset.Reset(2);
            CcuRepAdd rep(&insGen, varA, offset);

            EXPECT_EQ(rep.GetSubType(), AddSubType::SELF_ADD_VARIABLE);
            EXPECT_EQ(rep.Type(), CcuRepType::ADD);
        }

        TEST_F(CcuRepAddTest, Translate_AddrPlusVarToAddr)
        {
            Address addrC;
            Address addrA;
            Variable varB;
            addrC.Reset(5);
            addrA.Reset(3);
            varB.Reset(7);
            CcuRepAdd rep(&insGen, addrC, addrA, varB);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instr.v1.loadGSAXn.gsAdId, addrC.Id());
            EXPECT_EQ(instr.v1.loadGSAXn.gsAmId, addrA.Id());
            EXPECT_EQ(instr.v1.loadGSAXn.xnId, varB.Id());
            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x4);
        }

        TEST_F(CcuRepAddTest, Translate_AddrPlusAddrToAddr)
        {
            Address addrC;
            Address addrA;
            Address addrB;
            addrC.Reset(5);
            addrA.Reset(3);
            addrB.Reset(7);
            CcuRepAdd rep(&insGen, addrC, addrA, addrB);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_EQ(instr.v1.loadGSAGSA.gsAdId, addrC.Id());
            EXPECT_EQ(instr.v1.loadGSAGSA.gsAmId, addrA.Id());
            EXPECT_EQ(instr.v1.loadGSAGSA.gsAnId, addrB.Id());
            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x5);
        }

        TEST_F(CcuRepAddTest, Translate_VarPlusVarToVar)
        {
            Variable varC;
            Variable varA;
            Variable varB;
            varC.Reset(5);
            varA.Reset(3);
            varB.Reset(7);
            CcuRepAdd rep(&insGen, varC, varA, varB);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_EQ(instr.v1.loadXX.xdId, varC.Id());
            EXPECT_EQ(instr.v1.loadXX.xmId, varA.Id());
            EXPECT_EQ(instr.v1.loadXX.xnId, varB.Id());
            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x6);
        }

        TEST_F(CcuRepAddTest, Translate_SelfAddAddress)
        {
            Address addrA;
            Variable offset;
            addrA.Reset(5);
            offset.Reset(7);
            CcuRepAdd rep(&insGen, addrA, offset);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_EQ(instr.v1.loadGSAXn.gsAdId, addrA.Id());
            EXPECT_EQ(instr.v1.loadGSAXn.gsAmId, addrA.Id());
            EXPECT_EQ(instr.v1.loadGSAXn.xnId, offset.Id());
        }

        TEST_F(CcuRepAddTest, Translate_SelfAddVariable)
        {
            Variable varA;
            Variable offset;
            varA.Reset(5);
            offset.Reset(7);
            CcuRepAdd rep(&insGen, varA, offset);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_EQ(instr.v1.loadXX.xdId, varA.Id());
            EXPECT_EQ(instr.v1.loadXX.xmId, varA.Id());
            EXPECT_EQ(instr.v1.loadXX.xnId, offset.Id());
        }

        TEST_F(CcuRepAddTest, Describe_AddrPlusVarToAddr)
        {
            Address addrC;
            Address addrA;
            Variable varB;
            addrC.Reset(1);
            addrA.Reset(2);
            varB.Reset(3);
            CcuRepAdd rep(&insGen, addrC, addrA, varB);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Address[1]"), std::string::npos);
            EXPECT_NE(desc.find("Address[2]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[3]"), std::string::npos);
        }

        TEST_F(CcuRepAddTest, Describe_AddrPlusAddrToAddr)
        {
            Address addrC;
            Address addrA;
            Address addrB;
            addrC.Reset(1);
            addrA.Reset(2);
            addrB.Reset(3);
            CcuRepAdd rep(&insGen, addrC, addrA, addrB);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Address[1]"), std::string::npos);
            EXPECT_NE(desc.find("Address[2]"), std::string::npos);
            EXPECT_NE(desc.find("Address[3]"), std::string::npos);
        }

        TEST_F(CcuRepAddTest, Describe_VarPlusVarToVar)
        {
            Variable varC;
            Variable varA;
            Variable varB;
            varC.Reset(1);
            varA.Reset(2);
            varB.Reset(3);
            CcuRepAdd rep(&insGen, varC, varA, varB);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Variable[1]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[2]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[3]"), std::string::npos);
        }

        TEST_F(CcuRepAddTest, Describe_SelfAddAddress)
        {
            Address addrA;
            Variable offset;
            addrA.Reset(5);
            offset.Reset(7);
            CcuRepAdd rep(&insGen, addrA, offset);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Address[5]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[7]"), std::string::npos);
        }

        TEST_F(CcuRepAddTest, Describe_SelfAddVariable)
        {
            Variable varA;
            Variable offset;
            varA.Reset(5);
            offset.Reset(7);
            CcuRepAdd rep(&insGen, varA, offset);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Variable[5]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[7]"), std::string::npos);
        }

        class CcuRepAssignTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            CcuInstr instr{};
            uint16_t instrId{0};
            TransDep dep{};

            void SetUp() override
            {
                memset(&instr, 0, sizeof(instr));
                dep.reserveGsaId = 1;
                dep.reserveXnId = 2;
            }
        };

        TEST_F(CcuRepAssignTest, Constructor_ImdToVariable)
        {
            Variable varA;
            varA.Reset(1);
            uint64_t immediate = 100;
            CcuRepAssign rep(&insGen, varA, immediate);

            EXPECT_EQ(rep.GetSubType(), AssignSubType::IMD_TO_VARIABLE);
            EXPECT_EQ(rep.Type(), CcuRepType::ASSIGN);
            EXPECT_EQ(rep.GetImmed(), immediate);
        }

        TEST_F(CcuRepAssignTest, Constructor_ImdToAddr)
        {
            Address addrA;
            addrA.Reset(1);
            uint64_t immediate = 200;
            CcuRepAssign rep(&insGen, addrA, immediate);

            EXPECT_EQ(rep.GetSubType(), AssignSubType::IMD_TO_ADDR);
            EXPECT_EQ(rep.Type(), CcuRepType::ASSIGN);
            EXPECT_EQ(rep.GetImmed(), immediate);
        }

        TEST_F(CcuRepAssignTest, Constructor_VarToAddr)
        {
            Address addrA;
            Variable varA;
            addrA.Reset(1);
            varA.Reset(2);
            CcuRepAssign rep(&insGen, addrA, varA);

            EXPECT_EQ(rep.GetSubType(), AssignSubType::VAR_TO_ADDR);
            EXPECT_EQ(rep.Type(), CcuRepType::ASSIGN);
        }

        TEST_F(CcuRepAssignTest, Constructor_AddrToAddr)
        {
            Address addrB;
            Address addrA;
            addrB.Reset(1);
            addrA.Reset(2);
            CcuRepAssign rep(&insGen, addrB, addrA);

            EXPECT_EQ(rep.GetSubType(), AssignSubType::ADDR_TO_ADDR);
            EXPECT_EQ(rep.Type(), CcuRepType::ASSIGN);
        }

        TEST_F(CcuRepAssignTest, Constructor_VarToVar)
        {
            Variable varB;
            Variable varA;
            varB.Reset(1);
            varA.Reset(2);
            CcuRepAssign rep(&insGen, varB, varA);

            EXPECT_EQ(rep.GetSubType(), AssignSubType::VAR_TO_VAR);
            EXPECT_EQ(rep.Type(), CcuRepType::ASSIGN);
        }

        TEST_F(CcuRepAssignTest, Translate_ImdToVariable)
        {
            Variable varA;
            varA.Reset(5);
            uint64_t immediate = 0x12345678ABCD;
            CcuRepAssign rep(&insGen, varA, immediate);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instr.v1.loadImdToXn.xnId, varA.Id());
            EXPECT_EQ(instr.v1.loadImdToXn.immediate, immediate);
            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x3);
        }

        TEST_F(CcuRepAssignTest, Translate_ImdToAddr)
        {
            Address addrA;
            addrA.Reset(5);
            uint64_t immediate = 0xDEADBEEF;
            CcuRepAssign rep(&insGen, addrA, immediate);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_EQ(instr.v1.loadImdToGSA.gsaId, addrA.Id());
            EXPECT_EQ(instr.v1.loadImdToGSA.immediate, immediate);
            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x2);
        }

        TEST_F(CcuRepAssignTest, Translate_VarToAddr)
        {
            Address addrA;
            Variable varA;
            addrA.Reset(5);
            varA.Reset(7);
            CcuRepAssign rep(&insGen, addrA, varA);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_EQ(instr.v1.loadGSAXn.gsAdId, addrA.Id());
            EXPECT_EQ(instr.v1.loadGSAXn.gsAmId, dep.reserveGsaId);
            EXPECT_EQ(instr.v1.loadGSAXn.xnId, varA.Id());
        }

        TEST_F(CcuRepAssignTest, Translate_AddrToAddr)
        {
            Address addrB;
            Address addrA;
            addrB.Reset(5);
            addrA.Reset(7);
            CcuRepAssign rep(&insGen, addrB, addrA);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_EQ(instr.v1.loadGSAGSA.gsAdId, addrB.Id());
            EXPECT_EQ(instr.v1.loadGSAGSA.gsAmId, addrA.Id());
            EXPECT_EQ(instr.v1.loadGSAGSA.gsAnId, dep.reserveGsaId);
        }

        TEST_F(CcuRepAssignTest, Translate_VarToVar)
        {
            Variable varB;
            Variable varA;
            varB.Reset(5);
            varA.Reset(7);
            CcuRepAssign rep(&insGen, varB, varA);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_EQ(instr.v1.loadXX.xdId, varB.Id());
            EXPECT_EQ(instr.v1.loadXX.xmId, varA.Id());
            EXPECT_EQ(instr.v1.loadXX.xnId, dep.reserveXnId);
        }

        TEST_F(CcuRepAssignTest, Describe_ImdToVariable)
        {
            Variable varA;
            varA.Reset(1);
            CcuRepAssign rep(&insGen, varA, 100);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Variable[1]"), std::string::npos);
            EXPECT_NE(desc.find("Value[100]"), std::string::npos);
        }

        TEST_F(CcuRepAssignTest, Describe_ImdToAddr)
        {
            Address addrA;
            addrA.Reset(1);
            CcuRepAssign rep(&insGen, addrA, 200);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Address[1]"), std::string::npos);
            EXPECT_NE(desc.find("Value[200]"), std::string::npos);
        }

        TEST_F(CcuRepAssignTest, Describe_VarToAddr)
        {
            Address addrA;
            Variable varA;
            addrA.Reset(1);
            varA.Reset(2);
            CcuRepAssign rep(&insGen, addrA, varA);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Address[1]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[2]"), std::string::npos);
        }

        TEST_F(CcuRepAssignTest, Describe_AddrToAddr)
        {
            Address addrB;
            Address addrA;
            addrB.Reset(1);
            addrA.Reset(2);
            CcuRepAssign rep(&insGen, addrB, addrA);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Address[1]"), std::string::npos);
            EXPECT_NE(desc.find("Address[2]"), std::string::npos);
        }

        TEST_F(CcuRepAssignTest, Describe_VarToVar)
        {
            Variable varB;
            Variable varA;
            varB.Reset(1);
            varA.Reset(2);
            CcuRepAssign rep(&insGen, varB, varA);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Var[1]"), std::string::npos);
            EXPECT_NE(desc.find("Var[2]"), std::string::npos);
        }

        class CcuOperatorTest : public ::testing::Test {};

        TEST_F(CcuOperatorTest, VariablePlusVariable)
        {
            Variable varA;
            Variable varB;
            varA.Reset(1);
            varB.Reset(2);
            auto op = varA + varB;

            EXPECT_EQ(op.type, CcuArithmeticOperatorType::ADDITION);
            EXPECT_EQ(op.lhs.Id(), varA.Id());
            EXPECT_EQ(op.rhs.Id(), varB.Id());
        }

        TEST_F(CcuOperatorTest, VariablePlusAddress)
        {
            Variable varA;
            Address addrB;
            varA.Reset(1);
            addrB.Reset(2);
            auto op = varA + addrB;

            EXPECT_EQ(op.type, CcuArithmeticOperatorType::ADDITION);
            EXPECT_EQ(op.lhs.Id(), varA.Id());
            EXPECT_EQ(op.rhs.Id(), addrB.Id());
        }

        TEST_F(CcuOperatorTest, AddressPlusVariable)
        {
            Address addrA;
            Variable varB;
            addrA.Reset(1);
            varB.Reset(2);
            auto op = addrA + varB;

            EXPECT_EQ(op.type, CcuArithmeticOperatorType::ADDITION);
            EXPECT_EQ(op.lhs.Id(), varB.Id());
            EXPECT_EQ(op.rhs.Id(), addrA.Id());
        }

        TEST_F(CcuOperatorTest, AddressPlusAddress)
        {
            Address addrA;
            Address addrB;
            addrA.Reset(1);
            addrB.Reset(2);
            auto op = addrA + addrB;

            EXPECT_EQ(op.type, CcuArithmeticOperatorType::ADDITION);
            EXPECT_EQ(op.lhs.Id(), addrA.Id());
            EXPECT_EQ(op.rhs.Id(), addrB.Id());
        }

        TEST_F(CcuOperatorTest, VariableNotEqualImmediate)
        {
            Variable varA;
            varA.Reset(1);
            auto op = varA != 100;

            EXPECT_EQ(op.type, CcuRelationalOperatorType::NOT_EQUAL);
            EXPECT_EQ(op.lhs.Id(), varA.Id());
            EXPECT_EQ(op.rhs, 100);
        }

        TEST_F(CcuOperatorTest, VariableEqualImmediate)
        {
            Variable varA;
            varA.Reset(1);
            auto op = varA == 200;

            EXPECT_EQ(op.type, CcuRelationalOperatorType::EQUAL);
            EXPECT_EQ(op.lhs.Id(), varA.Id());
            EXPECT_EQ(op.rhs, 200);
        }

        class CcuRepAddV2Test : public ::testing::Test {
        protected:
            CcuInstr instr{};
            uint16_t instrId{0};
            TransDep dep{};
            CcuInsGeneratorV2 insGen{};

            void SetUp() override { memset(&instr, 0, sizeof(instr)); }
        };

        TEST_F(CcuRepAddV2Test, Translate_AddrPlusImmToAddr)
        {
            Address addrC, addrA;
            addrC.Reset(5);
            addrA.Reset(3);
            uint16_t imm = 0x40;
            CcuRepAdd rep(&insGen, addrC, addrA, imm);
            EXPECT_EQ(rep.GetSubType(), AddSubType::ADDR_PLUS_IMMED_TO_ADDR);

            CcuInstr* instrPtr = &instr;
            EXPECT_TRUE(rep.Translate(nullptr, instrPtr, instrId, dep));
            EXPECT_EQ(instr.v2.operate.parMode, 0);
            EXPECT_EQ(instr.v2.operate.xdId, addrC.Id());
            EXPECT_EQ(instr.v2.operate.xnId, addrA.Id());
            EXPECT_EQ(instr.v2.operate.xmId, imm);
        }

        TEST_F(CcuRepAddV2Test, Translate_VarPlusVarToAddr)
        {
            Address addrC;
            Variable varA, varB;
            addrC.Reset(5);
            varA.Reset(3);
            varB.Reset(7);
            CcuRepAdd rep(&insGen, addrC, varA, varB);
            EXPECT_EQ(rep.GetSubType(), AddSubType::VAR_PLUS_VAR_TO_ADDR);

            CcuInstr* instrPtr = &instr;
            EXPECT_TRUE(rep.Translate(nullptr, instrPtr, instrId, dep));
            EXPECT_EQ(instr.v2.operate.parMode, 1);
            EXPECT_EQ(instr.v2.operate.xdId, addrC.Id());
            EXPECT_EQ(instr.v2.operate.xnId, varA.Id());
            EXPECT_EQ(instr.v2.operate.xmId, varB.Id());
        }

        TEST_F(CcuRepAddV2Test, Translate_VarPlusImmToAddr)
        {
            Address addrC;
            Variable varA;
            addrC.Reset(5);
            varA.Reset(3);
            uint16_t imm = 0x40;
            CcuRepAdd rep(&insGen, addrC, varA, imm);
            EXPECT_EQ(rep.GetSubType(), AddSubType::VAR_PLUS_IMMED_TO_ADDR);

            CcuInstr* instrPtr = &instr;
            EXPECT_TRUE(rep.Translate(nullptr, instrPtr, instrId, dep));
            EXPECT_EQ(instr.v2.operate.parMode, 0);
            EXPECT_EQ(instr.v2.operate.xdId, addrC.Id());
            EXPECT_EQ(instr.v2.operate.xnId, varA.Id());
            EXPECT_EQ(instr.v2.operate.xmId, imm);
        }

        TEST_F(CcuRepAddV2Test, Translate_AddrPlusImmToVar)
        {
            Variable varC;
            Address addrA;
            varC.Reset(5);
            addrA.Reset(3);
            uint16_t imm = 0x40;
            CcuRepAdd rep(&insGen, varC, addrA, imm);
            EXPECT_EQ(rep.GetSubType(), AddSubType::ADDR_PLUS_IMMED_TO_VAR);

            CcuInstr* instrPtr = &instr;
            EXPECT_TRUE(rep.Translate(nullptr, instrPtr, instrId, dep));
            EXPECT_EQ(instr.v2.operate.parMode, 0);
            EXPECT_EQ(instr.v2.operate.xdId, varC.Id());
            EXPECT_EQ(instr.v2.operate.xnId, addrA.Id());
            EXPECT_EQ(instr.v2.operate.xmId, imm);
        }

        TEST_F(CcuRepAddV2Test, Translate_AddrPlusAddrToVar)
        {
            Variable varC;
            Address addrA, addrB;
            varC.Reset(5);
            addrA.Reset(3);
            addrB.Reset(7);
            CcuRepAdd rep(&insGen, varC, addrA, addrB);
            EXPECT_EQ(rep.GetSubType(), AddSubType::ADDR_PLUS_ADDR_TO_VAR);

            CcuInstr* instrPtr = &instr;
            EXPECT_TRUE(rep.Translate(nullptr, instrPtr, instrId, dep));
            EXPECT_EQ(instr.v2.operate.parMode, 1);
            EXPECT_EQ(instr.v2.operate.xdId, varC.Id());
            EXPECT_EQ(instr.v2.operate.xnId, addrA.Id());
            EXPECT_EQ(instr.v2.operate.xmId, addrB.Id());
        }

        class CcuRepMulTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV2 insGen{};
            void SetUp() override {}
        };

        TEST_F(CcuRepMulTest, Constructor_AllSubTypes)
        {
            Variable varA, varB, varC;
            varA.Reset(1);
            varB.Reset(2);
            varC.Reset(3);
            Address addrA, addrB, addrC;
            addrA.Reset(4);
            addrB.Reset(5);
            addrC.Reset(6);

            CcuRepMul m1(&insGen, varC, varA, varB);
            EXPECT_EQ(m1.GetSubType(), MulSubType::VAR_MUL_VAR_TO_VAR);
            EXPECT_EQ(m1.Type(), CcuRepType::MUL);

            CcuRepMul m2(&insGen, varC, varA, uint16_t(10));
            EXPECT_EQ(m2.GetSubType(), MulSubType::VAR_MUL_IMMED_TO_VAR);

            CcuRepMul m3(&insGen, varA, varB);
            EXPECT_EQ(m3.GetSubType(), MulSubType::SELF_MUL_VAR_VARIABLE);

            CcuRepMul m4(&insGen, varA, uint16_t(10));
            EXPECT_EQ(m4.GetSubType(), MulSubType::SELF_MUL_IMMED_VARIABLE);

            CcuRepMul m5(&insGen, addrC, varA, varB);
            EXPECT_EQ(m5.GetSubType(), MulSubType::VAR_MUL_VAR_TO_ADDR);

            CcuRepMul m6(&insGen, addrC, varA, addrB);
            EXPECT_EQ(m6.GetSubType(), MulSubType::VAR_MUL_ADDR_TO_ADDR);

            CcuRepMul m7(&insGen, addrA, varB);
            EXPECT_EQ(m7.GetSubType(), MulSubType::SELF_MUL_VAR_ADDRESS);

            CcuRepMul m8(&insGen, addrC, addrA, uint16_t(10));
            EXPECT_EQ(m8.GetSubType(), MulSubType::ADDR_MUL_IMMED_TO_ADDR);

            CcuRepMul m9(&insGen, addrC, varA, uint16_t(10));
            EXPECT_EQ(m9.GetSubType(), MulSubType::VAR_MUL_IMMED_TO_ADDR);

            CcuRepMul m10(&insGen, addrA, uint16_t(10));
            EXPECT_EQ(m10.GetSubType(), MulSubType::SELF_MUL_IMMED_ADDRESS);

            CcuRepMul m11(&insGen, varC, addrA, uint16_t(10));
            EXPECT_EQ(m11.GetSubType(), MulSubType::ADDR_MUL_IMMED_TO_VAR);
        }

        TEST_F(CcuRepMulTest, Describe_AllSubTypes)
        {
            Variable varA, varB, varC;
            varA.Reset(1);
            varB.Reset(2);
            varC.Reset(3);
            Address addrA, addrB, addrC;
            addrA.Reset(4);
            addrB.Reset(5);
            addrC.Reset(6);

            EXPECT_NE(CcuRepMul(&insGen, varC, varA, varB).Describe().find("*"), std::string::npos);
            EXPECT_NE(CcuRepMul(&insGen, varC, varA, uint16_t(10)).Describe().find("Immed"), std::string::npos);
            EXPECT_NE(CcuRepMul(&insGen, varA, varB).Describe().find("*="), std::string::npos);
            EXPECT_NE(CcuRepMul(&insGen, varA, uint16_t(10)).Describe().find("*="), std::string::npos);
            EXPECT_NE(CcuRepMul(&insGen, addrC, varA, varB).Describe().find("Address"), std::string::npos);
            EXPECT_NE(CcuRepMul(&insGen, addrC, varA, addrB).Describe().find("Address"), std::string::npos);
            EXPECT_NE(CcuRepMul(&insGen, addrA, varB).Describe().find("*="), std::string::npos);
            EXPECT_NE(CcuRepMul(&insGen, addrC, addrA, uint16_t(10)).Describe().find("Immed"), std::string::npos);
            EXPECT_NE(CcuRepMul(&insGen, addrC, varA, uint16_t(10)).Describe().find("Immed"), std::string::npos);
            EXPECT_NE(CcuRepMul(&insGen, addrA, uint16_t(10)).Describe().find("Immed"), std::string::npos);
            EXPECT_NE(CcuRepMul(&insGen, varC, addrA, uint16_t(10)).Describe().find("Immed"), std::string::npos);
        }

        TEST_F(CcuRepMulTest, Getters)
        {
            Variable varA, varB, varC;
            varA.Reset(1);
            varB.Reset(2);
            varC.Reset(3);
            Address addrA, addrB, addrC;
            addrA.Reset(4);
            addrB.Reset(5);
            addrC.Reset(6);

            CcuRepMul mVarVar(&insGen, varC, varA, varB);
            EXPECT_EQ(mVarVar.GetVarA().Id(), varA.Id());
            EXPECT_EQ(mVarVar.GetVarB().Id(), varB.Id());
            EXPECT_EQ(mVarVar.GetVarC().Id(), varC.Id());

            CcuRepMul mAddrAddr(&insGen, addrC, varA, addrB);
            EXPECT_EQ(mAddrAddr.GetAddrB().Id(), addrB.Id());
            EXPECT_EQ(mAddrAddr.GetAddrC().Id(), addrC.Id());
            EXPECT_EQ(mAddrAddr.GetVarA().Id(), varA.Id());

            CcuRepMul mImmed(&insGen, varC, varA, uint16_t(10));
            EXPECT_EQ(mImmed.GetImmedB(), uint16_t(10));

            CcuRepMul mAddrA(&insGen, addrC, addrA, uint16_t(10));
            EXPECT_EQ(mAddrA.GetAddrA().Id(), addrA.Id());
            EXPECT_EQ(mAddrA.GetAddrC().Id(), addrC.Id());
        }

        class CcuRepSubTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV2 insGen{};
            void SetUp() override {}
        };

        TEST_F(CcuRepSubTest, Constructor_AllSubTypes)
        {
            Variable varA, varB, varC;
            varA.Reset(1);
            varB.Reset(2);
            varC.Reset(3);
            Address addrA, addrB, addrC;
            addrA.Reset(4);
            addrB.Reset(5);
            addrC.Reset(6);

            CcuRepSub s1(&insGen, varC, varA, varB);
            EXPECT_EQ(s1.GetSubType(), MinusSubType::VAR_MINUS_VAR_TO_VAR);
            EXPECT_EQ(s1.Type(), CcuRepType::SUB);

            CcuRepSub s2(&insGen, varC, varA, uint16_t(10));
            EXPECT_EQ(s2.GetSubType(), MinusSubType::VAR_MINUS_IMMED_TO_VAR);

            CcuRepSub s3(&insGen, varA, varB);
            EXPECT_EQ(s3.GetSubType(), MinusSubType::SELF_SUB_VAR_VARIABLE);

            CcuRepSub s4(&insGen, varA, uint16_t(10));
            EXPECT_EQ(s4.GetSubType(), MinusSubType::SELF_SUB_IMMED_VARIABLE);

            CcuRepSub s5(&insGen, addrC, addrA, varB);
            EXPECT_EQ(s5.GetSubType(), MinusSubType::ADDR_MINUS_VAR_TO_ADDR);

            CcuRepSub s6(&insGen, addrC, addrA, uint16_t(10));
            EXPECT_EQ(s6.GetSubType(), MinusSubType::ADDR_MINUS_IMMED_TO_ADDR);

            CcuRepSub s7(&insGen, addrA, varB);
            EXPECT_EQ(s7.GetSubType(), MinusSubType::SELF_SUB_VAR_ADDRESS);

            CcuRepSub s8(&insGen, addrA, uint16_t(10));
            EXPECT_EQ(s8.GetSubType(), MinusSubType::SELF_SUB_IMMED_ADDRESS);

            CcuRepSub s9(&insGen, addrC, varA, uint16_t(10));
            EXPECT_EQ(s9.GetSubType(), MinusSubType::VAR_MINUS_IMMED_TO_ADDR);

            CcuRepSub s10(&insGen, varC, addrA, uint16_t(10));
            EXPECT_EQ(s10.GetSubType(), MinusSubType::ADDR_MINUS_IMMED_TO_VAR);
        }

        TEST_F(CcuRepSubTest, Describe_AllSubTypes)
        {
            Variable varA, varB, varC;
            varA.Reset(1);
            varB.Reset(2);
            varC.Reset(3);
            Address addrA, addrB, addrC;
            addrA.Reset(4);
            addrB.Reset(5);
            addrC.Reset(6);

            EXPECT_NE(CcuRepSub(&insGen, varC, varA, varB).Describe().find("-"), std::string::npos);
            EXPECT_NE(CcuRepSub(&insGen, varC, varA, uint16_t(10)).Describe().find("Immed"), std::string::npos);
            EXPECT_NE(CcuRepSub(&insGen, varA, varB).Describe().find("-="), std::string::npos);
            EXPECT_NE(CcuRepSub(&insGen, varA, uint16_t(10)).Describe().find("Immed"), std::string::npos);
            EXPECT_NE(CcuRepSub(&insGen, addrC, addrA, varB).Describe().find("Variable"), std::string::npos);
            EXPECT_NE(CcuRepSub(&insGen, addrC, addrA, uint16_t(10)).Describe().find("Immed"), std::string::npos);
            EXPECT_NE(CcuRepSub(&insGen, addrA, varB).Describe().find("-="), std::string::npos);
            EXPECT_NE(CcuRepSub(&insGen, addrA, uint16_t(10)).Describe().find("Immed"), std::string::npos);
            EXPECT_NE(CcuRepSub(&insGen, addrC, varA, uint16_t(10)).Describe().find("Immed"), std::string::npos);
            EXPECT_NE(CcuRepSub(&insGen, varC, addrA, uint16_t(10)).Describe().find("Immed"), std::string::npos);
        }

        TEST_F(CcuRepSubTest, Getters)
        {
            Variable varA, varB, varC;
            varA.Reset(1);
            varB.Reset(2);
            varC.Reset(3);
            Address addrA, addrB, addrC;
            addrA.Reset(4);
            addrB.Reset(5);
            addrC.Reset(6);

            CcuRepSub sVarVar(&insGen, varC, varA, varB);
            EXPECT_EQ(sVarVar.GetVarA().Id(), varA.Id());
            EXPECT_EQ(sVarVar.GetVarB().Id(), varB.Id());
            EXPECT_EQ(sVarVar.GetVarC().Id(), varC.Id());

            CcuRepSub sAddrVar(&insGen, addrC, addrA, varB);
            EXPECT_EQ(sAddrVar.GetAddrA().Id(), addrA.Id());
            EXPECT_EQ(sAddrVar.GetAddrC().Id(), addrC.Id());
            EXPECT_EQ(sAddrVar.GetVarB().Id(), varB.Id());

            CcuRepSub sImmed(&insGen, varC, varA, uint16_t(10));
            EXPECT_EQ(sImmed.GetImmedB(), uint16_t(10));

            CcuRepSub sAddrImmed(&insGen, addrA, uint16_t(10));
            EXPECT_EQ(sAddrImmed.GetAddrA().Id(), addrA.Id());
        }

        TEST_F(CcuRepMulTest, Translate_AllSubTypes)
        {
            Variable varA, varB, varC;
            varA.Reset(1);
            varB.Reset(2);
            varC.Reset(3);
            Address addrA, addrB, addrC;
            addrA.Reset(4);
            addrB.Reset(5);
            addrC.Reset(6);

            CcuInstr instr[20] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.reserveGsaId = 1;
            dep.reserveXnId = 2;

            CcuRepMul m1(&insGen, varC, varA, varB);
            EXPECT_TRUE(m1.Translate(nullptr, instrPtr, instrId, dep));

            CcuRepMul m2(&insGen, addrC, varA, varB);
            EXPECT_TRUE(m2.Translate(nullptr, instrPtr, instrId, dep));

            CcuRepMul m3(&insGen, addrA, varB);
            EXPECT_TRUE(m3.Translate(nullptr, instrPtr, instrId, dep));

            CcuRepMul m4(&insGen, varC, addrA, uint16_t(10));
            EXPECT_TRUE(m4.Translate(nullptr, instrPtr, instrId, dep));
        }

        TEST_F(CcuRepSubTest, Translate_AllSubTypes)
        {
            Variable varA, varB, varC;
            varA.Reset(1);
            varB.Reset(2);
            varC.Reset(3);
            Address addrA, addrB, addrC;
            addrA.Reset(4);
            addrB.Reset(5);
            addrC.Reset(6);

            CcuInstr instr[20] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.reserveGsaId = 1;
            dep.reserveXnId = 2;

            CcuRepSub s1(&insGen, varC, varA, varB);
            EXPECT_TRUE(s1.Translate(nullptr, instrPtr, instrId, dep));

            CcuRepSub s2(&insGen, addrC, addrA, varB);
            EXPECT_TRUE(s2.Translate(nullptr, instrPtr, instrId, dep));

            CcuRepSub s3(&insGen, addrA, varB);
            EXPECT_TRUE(s3.Translate(nullptr, instrPtr, instrId, dep));

            CcuRepSub s4(&insGen, varC, addrA, uint16_t(10));
            EXPECT_TRUE(s4.Translate(nullptr, instrPtr, instrId, dep));
        }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
