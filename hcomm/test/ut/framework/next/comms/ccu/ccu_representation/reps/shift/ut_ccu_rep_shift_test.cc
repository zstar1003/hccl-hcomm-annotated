/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_shr.h"
#include "ccu_rep_shl.h"
#include "ccu_datatype_v1.h"
#include "ccu_microcode_v1.h"
#include "ccu_rep_base_v1.h"
#include "ccu_rep_type_v1.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_ins_generator_v2.h"
#include "ccu_api_exception.h"
#include <gtest/gtest.h>
#include <string>

namespace hcomm {
namespace CcuRep {
    namespace {

        class CcuRepShRTest : public ::testing::Test {
        protected:
            CcuInstr instr{};
            uint16_t instrId{0};
            TransDep dep{};
            CcuInsGeneratorV2 insGen{};

            void SetUp() override { memset(&instr, 0, sizeof(instr)); }
        };

        TEST_F(CcuRepShRTest, Constructor_VarEqualsVarShiftVar)
        {
            Variable varD, varN, varM;
            varD.Reset(1);
            varN.Reset(2);
            varM.Reset(3);
            CcuRepShR rep(&insGen, varD, varN, varM);

            EXPECT_EQ(rep.GetShiftSubType(), ShiftSubType::VAR_EQUALS_VAR_SHIFT_VAR);
            EXPECT_EQ(rep.GetShiftType(), ShiftType::LOGICAL_SHIFT);
            EXPECT_EQ(rep.Type(), CcuRepType::SHR);
            EXPECT_EQ(rep.GetVarD().Id(), 1u);
            EXPECT_EQ(rep.GetVarN().Id(), 2u);
            EXPECT_EQ(rep.GetVarM().Id(), 3u);
        }

        TEST_F(CcuRepShRTest, Constructor_VarShiftAssignVar)
        {
            Variable varD, varM;
            varD.Reset(5);
            varM.Reset(7);
            CcuRepShR rep(&insGen, varD, varM);

            EXPECT_EQ(rep.GetShiftSubType(), ShiftSubType::VAR_SHIFT_ASSIGN_VAR);
            EXPECT_EQ(rep.GetShiftType(), ShiftType::LOGICAL_SHIFT);
            EXPECT_EQ(rep.Type(), CcuRepType::SHR);
            EXPECT_EQ(rep.GetVarD().Id(), 5u);
            EXPECT_EQ(rep.GetVarM().Id(), 7u);
        }

        TEST_F(CcuRepShRTest, Constructor_AddrEqualsVarShiftVar)
        {
            Address addrD;
            Variable varN, varM;
            addrD.Reset(10);
            varN.Reset(20);
            varM.Reset(30);
            CcuRepShR rep(&insGen, addrD, varN, varM);

            EXPECT_EQ(rep.GetShiftSubType(), ShiftSubType::ADDR_EQUALS_VAR_SHIFT_VAR);
            EXPECT_EQ(rep.GetShiftType(), ShiftType::LOGICAL_SHIFT);
            EXPECT_EQ(rep.Type(), CcuRepType::SHR);
            EXPECT_EQ(rep.GetAddressD().Id(), 10u);
            EXPECT_EQ(rep.GetVarN().Id(), 20u);
            EXPECT_EQ(rep.GetVarM().Id(), 30u);
        }

        TEST_F(CcuRepShRTest, Constructor_AddrShiftAssignVar)
        {
            Address addrD;
            Variable varM;
            addrD.Reset(15);
            varM.Reset(25);
            CcuRepShR rep(&insGen, addrD, varM);

            EXPECT_EQ(rep.GetShiftSubType(), ShiftSubType::ADDR_SHIFT_ASSIGN_VAR);
            EXPECT_EQ(rep.GetShiftType(), ShiftType::LOGICAL_SHIFT);
            EXPECT_EQ(rep.Type(), CcuRepType::SHR);
            EXPECT_EQ(rep.GetAddressD().Id(), 15u);
            EXPECT_EQ(rep.GetVarM().Id(), 25u);
        }

        TEST_F(CcuRepShRTest, Translate_VarEqualsVarShiftVar)
        {
            Variable varD, varN, varM;
            varD.Reset(5);
            varN.Reset(3);
            varM.Reset(7);
            CcuRepShR rep(&insGen, varD, varN, varM);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x15);
            EXPECT_EQ(instr.v2.operate.xdId, 5u);
            EXPECT_EQ(instr.v2.operate.xnId, 3u);
            EXPECT_EQ(instr.v2.operate.xmId, 7u);
            EXPECT_EQ(instr.v2.operate.shiftType, 0u);
        }

        TEST_F(CcuRepShRTest, Translate_VarShiftAssignVar)
        {
            Variable varD, varM;
            varD.Reset(5);
            varM.Reset(7);
            CcuRepShR rep(&insGen, varD, varM);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x15);
            EXPECT_EQ(instr.v2.operate.xdId, 5u);
            EXPECT_EQ(instr.v2.operate.xnId, 5u);
            EXPECT_EQ(instr.v2.operate.xmId, 7u);
        }

        TEST_F(CcuRepShRTest, Translate_AddrEqualsVarShiftVar)
        {
            Address addrD;
            Variable varN, varM;
            addrD.Reset(5);
            varN.Reset(3);
            varM.Reset(7);
            CcuRepShR rep(&insGen, addrD, varN, varM);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x15);
            EXPECT_EQ(instr.v2.operate.xdId, 5u);
            EXPECT_EQ(instr.v2.operate.xnId, 3u);
            EXPECT_EQ(instr.v2.operate.xmId, 7u);
        }

        TEST_F(CcuRepShRTest, Translate_AddrShiftAssignVar)
        {
            Address addrD;
            Variable varM;
            addrD.Reset(5);
            varM.Reset(7);
            CcuRepShR rep(&insGen, addrD, varM);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x15);
            EXPECT_EQ(instr.v2.operate.xdId, 5u);
            EXPECT_EQ(instr.v2.operate.xnId, 5u);
            EXPECT_EQ(instr.v2.operate.xmId, 7u);
        }

        TEST_F(CcuRepShRTest, Describe_VarEqualsVarShiftVar)
        {
            Variable varD, varN, varM;
            varD.Reset(1);
            varN.Reset(2);
            varM.Reset(3);
            CcuRepShR rep(&insGen, varD, varN, varM);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Variable[1]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[2]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[3]"), std::string::npos);
            EXPECT_NE(desc.find(">>"), std::string::npos);
        }

        TEST_F(CcuRepShRTest, Describe_VarShiftAssignVar)
        {
            Variable varD, varM;
            varD.Reset(5);
            varM.Reset(7);
            CcuRepShR rep(&insGen, varD, varM);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Variable[5]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[7]"), std::string::npos);
            EXPECT_NE(desc.find(">>="), std::string::npos);
        }

        TEST_F(CcuRepShRTest, Describe_AddrEqualsVarShiftVar)
        {
            Address addrD;
            Variable varN, varM;
            addrD.Reset(10);
            varN.Reset(20);
            varM.Reset(30);
            CcuRepShR rep(&insGen, addrD, varN, varM);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Address[10]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[20]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[30]"), std::string::npos);
        }

        TEST_F(CcuRepShRTest, Describe_AddrShiftAssignVar)
        {
            Address addrD;
            Variable varM;
            addrD.Reset(15);
            varM.Reset(25);
            CcuRepShR rep(&insGen, addrD, varM);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Address[15]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[25]"), std::string::npos);
            EXPECT_NE(desc.find(">>="), std::string::npos);
        }

        class CcuRepShRV1Test : public ::testing::Test {
        protected:
            CcuInstr instr{};
            uint16_t instrId{0};
            TransDep dep{};
            CcuInsGeneratorV1 insGen{};

            void SetUp() override { memset(&instr, 0, sizeof(instr)); }
        };

        TEST_F(CcuRepShRV1Test, Translate_V1_ThrowsException)
        {
            Variable varD, varN, varM;
            varD.Reset(1);
            varN.Reset(2);
            varM.Reset(3);

            CcuInstr* instrPtr = &instr;
            EXPECT_THROW(
                {
                    CcuRepShR rep(&insGen, varD, varN, varM);
                    rep.Translate(nullptr, instrPtr, instrId, dep);
                },
                Hccl::CcuApiException);
        }

        class CcuRepShLTest : public ::testing::Test {
        protected:
            CcuInstr instr{};
            uint16_t instrId{0};
            TransDep dep{};
            CcuInsGeneratorV2 insGen{};

            void SetUp() override { memset(&instr, 0, sizeof(instr)); }
        };

        TEST_F(CcuRepShLTest, Constructor_VarEqualsVarShiftVar)
        {
            Variable varD, varN, varM;
            varD.Reset(1);
            varN.Reset(2);
            varM.Reset(3);
            CcuRepShL rep(&insGen, varD, varN, varM);

            EXPECT_EQ(rep.GetShiftSubType(), ShiftSubType::VAR_EQUALS_VAR_SHIFT_VAR);
            EXPECT_EQ(rep.GetShiftType(), ShiftType::LOGICAL_SHIFT);
            EXPECT_EQ(rep.Type(), CcuRepType::SHL);
            EXPECT_EQ(rep.GetVarD().Id(), 1u);
            EXPECT_EQ(rep.GetVarN().Id(), 2u);
            EXPECT_EQ(rep.GetVarM().Id(), 3u);
        }

        TEST_F(CcuRepShLTest, Constructor_VarShiftAssignVar)
        {
            Variable varD, varM;
            varD.Reset(5);
            varM.Reset(7);
            CcuRepShL rep(&insGen, varD, varM);

            EXPECT_EQ(rep.GetShiftSubType(), ShiftSubType::VAR_SHIFT_ASSIGN_VAR);
            EXPECT_EQ(rep.GetShiftType(), ShiftType::LOGICAL_SHIFT);
            EXPECT_EQ(rep.Type(), CcuRepType::SHL);
            EXPECT_EQ(rep.GetVarD().Id(), 5u);
            EXPECT_EQ(rep.GetVarM().Id(), 7u);
        }

        TEST_F(CcuRepShLTest, Constructor_AddrEqualsVarShiftVar)
        {
            Address addrD;
            Variable varN, varM;
            addrD.Reset(10);
            varN.Reset(20);
            varM.Reset(30);
            CcuRepShL rep(&insGen, addrD, varN, varM);

            EXPECT_EQ(rep.GetShiftSubType(), ShiftSubType::ADDR_EQUALS_VAR_SHIFT_VAR);
            EXPECT_EQ(rep.GetShiftType(), ShiftType::LOGICAL_SHIFT);
            EXPECT_EQ(rep.Type(), CcuRepType::SHL);
            EXPECT_EQ(rep.GetAddressD().Id(), 10u);
            EXPECT_EQ(rep.GetVarN().Id(), 20u);
            EXPECT_EQ(rep.GetVarM().Id(), 30u);
        }

        TEST_F(CcuRepShLTest, Constructor_AddrShiftAssignVar)
        {
            Address addrD;
            Variable varM;
            addrD.Reset(15);
            varM.Reset(25);
            CcuRepShL rep(&insGen, addrD, varM);

            EXPECT_EQ(rep.GetShiftSubType(), ShiftSubType::ADDR_SHIFT_ASSIGN_VAR);
            EXPECT_EQ(rep.GetShiftType(), ShiftType::LOGICAL_SHIFT);
            EXPECT_EQ(rep.Type(), CcuRepType::SHL);
            EXPECT_EQ(rep.GetAddressD().Id(), 15u);
            EXPECT_EQ(rep.GetVarM().Id(), 25u);
        }

        TEST_F(CcuRepShLTest, Translate_VarEqualsVarShiftVar)
        {
            Variable varD, varN, varM;
            varD.Reset(5);
            varN.Reset(3);
            varM.Reset(7);
            CcuRepShL rep(&insGen, varD, varN, varM);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x14);
            EXPECT_EQ(instr.v2.operate.xdId, 5u);
            EXPECT_EQ(instr.v2.operate.xnId, 3u);
            EXPECT_EQ(instr.v2.operate.xmId, 7u);
            EXPECT_EQ(instr.v2.operate.shiftType, 0u);
        }

        TEST_F(CcuRepShLTest, Translate_VarShiftAssignVar)
        {
            Variable varD, varM;
            varD.Reset(5);
            varM.Reset(7);
            CcuRepShL rep(&insGen, varD, varM);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x14);
            EXPECT_EQ(instr.v2.operate.xdId, 5u);
            EXPECT_EQ(instr.v2.operate.xnId, 5u);
            EXPECT_EQ(instr.v2.operate.xmId, 7u);
        }

        TEST_F(CcuRepShLTest, Translate_AddrEqualsVarShiftVar)
        {
            Address addrD;
            Variable varN, varM;
            addrD.Reset(5);
            varN.Reset(3);
            varM.Reset(7);
            CcuRepShL rep(&insGen, addrD, varN, varM);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x14);
            EXPECT_EQ(instr.v2.operate.xdId, 5u);
            EXPECT_EQ(instr.v2.operate.xnId, 3u);
            EXPECT_EQ(instr.v2.operate.xmId, 7u);
        }

        TEST_F(CcuRepShLTest, Translate_AddrShiftAssignVar)
        {
            Address addrD;
            Variable varM;
            addrD.Reset(5);
            varM.Reset(7);
            CcuRepShL rep(&insGen, addrD, varM);

            CcuInstr* instrPtr = &instr;
            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x14);
            EXPECT_EQ(instr.v2.operate.xdId, 5u);
            EXPECT_EQ(instr.v2.operate.xnId, 5u);
            EXPECT_EQ(instr.v2.operate.xmId, 7u);
        }

        TEST_F(CcuRepShLTest, Describe_VarEqualsVarShiftVar)
        {
            Variable varD, varN, varM;
            varD.Reset(1);
            varN.Reset(2);
            varM.Reset(3);
            CcuRepShL rep(&insGen, varD, varN, varM);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Variable[1]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[2]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[3]"), std::string::npos);
            EXPECT_NE(desc.find("<<"), std::string::npos);
        }

        TEST_F(CcuRepShLTest, Describe_VarShiftAssignVar)
        {
            Variable varD, varM;
            varD.Reset(5);
            varM.Reset(7);
            CcuRepShL rep(&insGen, varD, varM);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Variable[5]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[7]"), std::string::npos);
            EXPECT_NE(desc.find("<<="), std::string::npos);
        }

        TEST_F(CcuRepShLTest, Describe_AddrEqualsVarShiftVar)
        {
            Address addrD;
            Variable varN, varM;
            addrD.Reset(10);
            varN.Reset(20);
            varM.Reset(30);
            CcuRepShL rep(&insGen, addrD, varN, varM);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Address[10]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[20]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[30]"), std::string::npos);
        }

        TEST_F(CcuRepShLTest, Describe_AddrShiftAssignVar)
        {
            Address addrD;
            Variable varM;
            addrD.Reset(15);
            varM.Reset(25);
            CcuRepShL rep(&insGen, addrD, varM);

            std::string desc = rep.Describe();
            EXPECT_NE(desc.find("Address[15]"), std::string::npos);
            EXPECT_NE(desc.find("Variable[25]"), std::string::npos);
            EXPECT_NE(desc.find("<<="), std::string::npos);
        }

        class CcuRepShLV1Test : public ::testing::Test {
        protected:
            CcuInstr instr{};
            uint16_t instrId{0};
            TransDep dep{};
            CcuInsGeneratorV1 insGen{};

            void SetUp() override { memset(&instr, 0, sizeof(instr)); }
        };

        TEST_F(CcuRepShLV1Test, Translate_V1_ThrowsException)
        {
            Variable varD, varN, varM;
            varD.Reset(1);
            varN.Reset(2);
            varM.Reset(3);

            CcuInstr* instrPtr = &instr;
            EXPECT_THROW(
                {
                    CcuRepShL rep(&insGen, varD, varN, varM);
                    rep.Translate(nullptr, instrPtr, instrId, dep);
                },
                Hccl::CcuApiException);
        }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
