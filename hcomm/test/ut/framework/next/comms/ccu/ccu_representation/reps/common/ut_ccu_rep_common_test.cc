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
#include "ccu_ins_generator_v2.h"
#include "ccu_api_exception.h"
#include <gtest/gtest.h>
#include <string>
#include <memory>

namespace hcomm {
namespace CcuRep {
    namespace {

        class CcuRepCommonBaseTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
            void TearDown() override {}
        };

        class CcuRepBlockTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        class CcuRepLoadTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        class CcuRepLoadArgTest : public ::testing::Test {
        protected:
            void SetUp() override {}
            CcuInsGeneratorV1 insGen{};
        };

        class CcuRepLoadVarTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        class CcuRepNopTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        class CcuRepStoreTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        class CcuRepStoreVarTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        TEST_F(CcuRepBlockTest, Constructor_DefaultLabel)
        {
            CcuRepBlock block(&insGen);
            EXPECT_EQ(block.Type(), CcuRepType::BLOCK);
            EXPECT_EQ(block.GetLabel(), "");
            EXPECT_EQ(block.InstrCount(), 0);
        }

        TEST_F(CcuRepBlockTest, Constructor_WithLabel)
        {
            CcuRepBlock block(&insGen, "testBlock");
            EXPECT_EQ(block.Type(), CcuRepType::BLOCK);
            EXPECT_EQ(block.GetLabel(), "testBlock");
        }

        TEST_F(CcuRepBlockTest, Describe)
        {
            CcuRepBlock block(&insGen, "testBlock");
            std::string desc = block.Describe();
            EXPECT_NE(desc.find("RepBlock"), std::string::npos);
        }

        TEST_F(CcuRepBlockTest, Append_And_GetReps)
        {
            CcuRepBlock block(&insGen, "testBlock");
            auto nop = std::make_shared<CcuRepNop>(&insGen);
            block.Append(nop);
            EXPECT_EQ(block.GetReps().size(), 1);
        }

        TEST_F(CcuRepBlockTest, InstrCount_SingleNop)
        {
            CcuRepBlock block(&insGen, "testBlock");
            auto nop = std::make_shared<CcuRepNop>(&insGen);
            block.Append(nop);
            EXPECT_EQ(block.InstrCount(), 1);
        }

        TEST_F(CcuRepBlockTest, InstrCount_MultipleReps)
        {
            CcuRepBlock block(&insGen, "testBlock");
            block.Append(std::make_shared<CcuRepNop>(&insGen));
            block.Append(std::make_shared<CcuRepNop>(&insGen));
            EXPECT_EQ(block.InstrCount(), 2);
        }

        TEST_F(CcuRepBlockTest, GetRepByInstrId_Found)
        {
            CcuRepBlock block(&insGen, "testBlock");
            auto nop1 = std::make_shared<CcuRepNop>(&insGen);
            auto nop2 = std::make_shared<CcuRepNop>(&insGen);
            block.Append(nop1);
            block.Append(nop2);
            auto rep = block.GetRepByInstrId(0);
            EXPECT_NE(rep, nullptr);
        }

        TEST_F(CcuRepBlockTest, GetRepByInstrId_NotFound)
        {
            CcuRepBlock block(&insGen, "testBlock");
            block.Append(std::make_shared<CcuRepNop>(&insGen));
            auto rep = block.GetRepByInstrId(100);
            EXPECT_EQ(rep, nullptr);
        }

        TEST_F(CcuRepBlockTest, Translate_SetsTranslated)
        {
            CcuInsGeneratorV1 insGen;
            CcuRepBlock block(&insGen, "testBlock");
            block.Append(std::make_shared<CcuRepNop>(&insGen));
            CcuInstr instr[10] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.reserveXnId = 1;
            bool result = block.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
            EXPECT_TRUE(block.Translated());
        }

        TEST_F(CcuRepLoadTest, Constructor)
        {
            CcuRepContext context;
            Variable var(&context);
            var.Reset(1);
            CcuRepLoad load(&insGen, 0x1000, var, 1);
            EXPECT_EQ(load.Type(), CcuRepType::LOAD);
            EXPECT_EQ(load.InstrCount(), 7);
        }

        TEST_F(CcuRepLoadTest, Describe)
        {
            CcuRepContext context;
            Variable var(&context);
            var.Reset(1);
            CcuRepLoad load(&insGen, 0x1000, var, 1);
            std::string desc = load.Describe();
            EXPECT_NE(desc.find("Load"), std::string::npos);
        }

        TEST_F(CcuRepLoadTest, Translate)
        {
            CcuInsGeneratorV1 insGen;
            CcuRepContext context;
            Variable var(&context);
            var.Reset(1);
            CcuRepLoad load(&insGen, 0x1000, var, 1);
            CcuInstr instr[10] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.xnBaseAddr[0] = 0x100000;
            dep.commGsa[0] = 1;
            dep.commGsa[1] = 2;
            dep.commXn[0] = 3;
            dep.commXn[1] = 4;
            dep.commXn[2] = 5;
            dep.reserveChannalId[0] = 6;
            dep.commSignal = 7;
            dep.ccuResSpaceTokenInfo = 0x1000;
            dep.memTokenInfo = 0x2000;
            bool result = load.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
            EXPECT_TRUE(load.Translated());
            EXPECT_EQ(load.StartInstrId(), 0);
            EXPECT_EQ(instrId, 7);
        }

        TEST_F(CcuRepLoadArgTest, Constructor)
        {
            CcuRepContext context;
            Variable var(&context);
            var.Reset(2);
            CcuRepLoadArg loadArg(&insGen, var, 1, 1);
            EXPECT_EQ(loadArg.Type(), CcuRepType::LOAD_ARG);
            EXPECT_EQ(loadArg.InstrCount(), 1);
            EXPECT_EQ(loadArg.GetVarId(), 2);
        }

        TEST_F(CcuRepLoadArgTest, Describe)
        {
            CcuRepContext context;
            Variable var(&context);
            var.Reset(2);
            CcuRepLoadArg loadArg(&insGen, var, 1, 1);
            std::string desc = loadArg.Describe();
            EXPECT_NE(desc.find("Variable[2]"), std::string::npos);
            EXPECT_NE(desc.find("Arg[1]"), std::string::npos);
        }

        TEST_F(CcuRepLoadArgTest, Translate_IsFuncBlock)
        {
            CcuRepContext context;
            Variable var(&context);
            var.Reset(2);
            CcuRepLoadArg loadArg(&insGen, var, 1, 1);
            CcuInstr instr[5] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.isFuncBlock = true;
            dep.loadXnId = 3;
            dep.reserveXnId = 4;
            bool result = loadArg.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
            EXPECT_TRUE(loadArg.Translated());
            EXPECT_EQ(instrId, 1);
        }

        TEST_F(CcuRepLoadArgTest, Translate_NotFuncBlock)
        {
            CcuRepContext context;
            Variable var(&context);
            var.Reset(2);
            CcuRepLoadArg loadArg(&insGen, var, 1, 1);
            CcuInstr instr[5] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.isFuncBlock = false;
            bool result = loadArg.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
            EXPECT_TRUE(loadArg.Translated());
            EXPECT_EQ(instrId, 1);
        }

        TEST_F(CcuRepLoadVarTest, Constructor)
        {
            CcuRepContext context;
            Variable src(&context);
            src.Reset(1);
            Variable var(&context);
            var.Reset(2);
            CcuRepLoadVar loadVar(&insGen, src, var);
            EXPECT_EQ(loadVar.Type(), CcuRepType::LOAD_VAR);
            EXPECT_EQ(loadVar.InstrCount(), 7);
        }

        TEST_F(CcuRepLoadVarTest, Describe)
        {
            CcuRepContext context;
            Variable src(&context);
            src.Reset(1);
            Variable var(&context);
            var.Reset(2);
            CcuRepLoadVar loadVar(&insGen, src, var);
            std::string desc = loadVar.Describe();
            EXPECT_NE(desc.find("Load Var"), std::string::npos);
        }

        TEST_F(CcuRepLoadVarTest, Translate)
        {
            CcuInsGeneratorV1 insGen;
            CcuRepContext context;
            Variable src(&context);
            src.Reset(1);
            Variable var(&context);
            var.Reset(2);
            CcuRepLoadVar loadVar(&insGen, src, var);
            CcuInstr instr[10] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.xnBaseAddr[0] = 0x100000;
            dep.commGsa[0] = 1;
            dep.commGsa[1] = 2;
            dep.commXn[0] = 3;
            dep.commXn[1] = 4;
            dep.commXn[2] = 5;
            dep.reserveChannalId[0] = 6;
            dep.commSignal = 7;
            dep.reserveGsaId = 8;
            dep.ccuResSpaceTokenInfo = 0x1000;
            dep.memTokenInfo = 0x2000;
            bool result = loadVar.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
            EXPECT_TRUE(loadVar.Translated());
            EXPECT_EQ(instrId, 7);
        }

        TEST_F(CcuRepNopTest, Constructor)
        {
            CcuRepNop nop(&insGen);
            EXPECT_EQ(nop.Type(), CcuRepType::NOP);
            EXPECT_EQ(nop.InstrCount(), 1);
        }

        TEST_F(CcuRepNopTest, Describe)
        {
            CcuRepNop nop(&insGen);
            std::string desc = nop.Describe();
            EXPECT_NE(desc.find("Nop"), std::string::npos);
        }

        TEST_F(CcuRepNopTest, Translate)
        {
            CcuInsGeneratorV1 insGen;
            CcuRepNop nop(&insGen);
            CcuInstr instr[5] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 10;
            TransDep dep = {};
            dep.reserveXnId = 1;
            bool result = nop.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
            EXPECT_TRUE(nop.Translated());
            EXPECT_EQ(nop.StartInstrId(), 10);
            EXPECT_EQ(instrId, 11);
        }

        TEST_F(CcuRepStoreTest, Constructor)
        {
            CcuRepContext context;
            Variable var(&context);
            var.Reset(1);
            CcuRepStore store(&insGen, var, 0x2000);
            EXPECT_EQ(store.Type(), CcuRepType::STORE);
            EXPECT_EQ(store.InstrCount(), 7);
        }

        TEST_F(CcuRepStoreTest, Describe)
        {
            CcuRepContext context;
            Variable var(&context);
            var.Reset(1);
            CcuRepStore store(&insGen, var, 0x2000);
            std::string desc = store.Describe();
            EXPECT_NE(desc.find("Store"), std::string::npos);
        }

        TEST_F(CcuRepStoreTest, Translate)
        {
            CcuInsGeneratorV1 insGen;
            CcuRepContext context;
            Variable var(&context);
            var.Reset(1);
            CcuRepStore store(&insGen, var, 0x2000);
            CcuInstr instr[10] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.xnBaseAddr[0] = 0x100000;
            dep.commGsa[0] = 1;
            dep.commGsa[1] = 2;
            dep.commXn[0] = 3;
            dep.commXn[1] = 4;
            dep.commXn[2] = 5;
            dep.reserveChannalId[0] = 6;
            dep.commSignal = 7;
            dep.ccuResSpaceTokenInfo = 0x1000;
            dep.memTokenInfo = 0x2000;
            bool result = store.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
            EXPECT_TRUE(store.Translated());
            EXPECT_EQ(instrId, 7);
        }

        TEST_F(CcuRepStoreVarTest, Constructor)
        {
            CcuRepContext context;
            Variable var(&context);
            var.Reset(1);
            Variable dst(&context);
            dst.Reset(2);

            CcuRepStoreVar storeVarV1(&insGen, var, dst);
            EXPECT_EQ(storeVarV1.Type(), CcuRepType::STORE_VAR);
            EXPECT_EQ(storeVarV1.InstrCount(), 7);

            CcuInsGeneratorV2 insGenV2;
            CcuRepStoreVar storeVarV2(&insGenV2, var, dst);
            EXPECT_EQ(storeVarV2.Type(), CcuRepType::STORE_VAR);
            EXPECT_EQ(storeVarV2.InstrCount(), 3);
        }

        TEST_F(CcuRepStoreVarTest, Describe)
        {
            CcuRepContext context;
            Variable var(&context);
            var.Reset(1);
            Variable dst(&context);
            dst.Reset(2);
            CcuRepStoreVar storeVar(&insGen, var, dst);
            std::string desc = storeVar.Describe();
            EXPECT_NE(desc.find("Store Var"), std::string::npos);
        }

        TEST_F(CcuRepStoreVarTest, Translate)
        {
            CcuInsGeneratorV1 insGen;
            CcuRepContext context;
            Variable var(&context);
            var.Reset(1);
            Variable dst(&context);
            dst.Reset(2);
            CcuRepStoreVar storeVar(&insGen, var, dst);
            CcuInstr instr[10] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.xnBaseAddr[0] = 0x100000;
            dep.commGsa[0] = 1;
            dep.commGsa[1] = 2;
            dep.commXn[0] = 3;
            dep.commXn[1] = 4;
            dep.commXn[2] = 5;
            dep.reserveChannalId[0] = 6;
            dep.commSignal = 7;
            dep.reserveGsaId = 8;
            dep.ccuResSpaceTokenInfo = 0x1000;
            dep.memTokenInfo = 0x2000;
            bool result = storeVar.Translate(nullptr, instrPtr, instrId, dep);
            EXPECT_TRUE(result);
            EXPECT_TRUE(storeVar.Translated());

            auto& transInstr = instr[5].v1.transLocMemToLocMem;
            EXPECT_EQ(transInstr.dstGSAId, dep.commGsa[1]);
            EXPECT_EQ(transInstr.dstXnId, dep.commXn[0]);
            EXPECT_EQ(transInstr.srcGSAId, dep.commGsa[0]);
            EXPECT_EQ(transInstr.srcXnId, dep.commXn[1]);
        }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
