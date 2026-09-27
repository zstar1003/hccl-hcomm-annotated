/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <climits>

#include "ccu_ins_generator_v1.h"
#include "ccu_rep_loc_wait_event.h"
#include "ccu_rep_loc_wait_notify.h"
#include "ccu_rep_loc_record_event.h"
#include "ccu_rep_record_shared_notify.h"
#include "ccu_rep_nop_v1.h"
#include "ccu_rep_rempostsem_v1.h"
#include "ccu_rep_rempostvar_v1.h"
#include "ccu_rep_remwaitsem_v1.h"
#include "ccu_datatype_v1.h"
#include "ccu_rep_context_v1.h"

namespace hcomm {
namespace CcuRep {
    namespace {

        class CcuRepLocWaitEventTest : public ::testing::Test {
        protected:
            void SetUp() override {}
            CcuInsGeneratorV1 insGen{};
        };

        class CcuRepLocWaitNotifyTest : public ::testing::Test {
        protected:
            void SetUp() override {}
            CcuInsGeneratorV1 insGen{};
        };

        class CcuRepLocRecordEventTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        class CcuRepRecordSharedNotifyTest : public ::testing::Test {
        protected:
            void SetUp() override {}
            CcuInsGeneratorV1 insGen{};
        };

        class CcuRepRemPostSemTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        class CcuRepRemPostVarTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        class CcuRepRemWaitSemTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        TEST_F(CcuRepLocWaitEventTest, ConstructorInitializesCorrectly)
        {
            CompletedEvent event;
            CcuRepLocWaitEvent rep(&insGen, event, 0xF, true);

            EXPECT_EQ(rep.Type(), CcuRepType::LOC_WAIT_EVENT);
            EXPECT_EQ(rep.InstrCount(), 1);
            EXPECT_EQ(rep.GetMask(), 0xF);
        }

        TEST_F(CcuRepLocWaitEventTest, ConstructorWithProfilingDisabled)
        {
            CompletedEvent event;
            CcuRepLocWaitEvent rep(&insGen, event, 0xA, false);

            EXPECT_EQ(rep.Type(), CcuRepType::LOC_WAIT_EVENT);
            EXPECT_EQ(rep.InstrCount(), 1);
            EXPECT_FALSE(rep.Translated());
        }

        TEST_F(CcuRepLocWaitEventTest, TranslateWithProfilingEnabled)
        {
            CompletedEvent event;
            CcuRepLocWaitEvent rep(&insGen, event, 0xF, true);

            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 0;
            TransDep dep = {};

            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(rep.StartInstrId(), 0);
            EXPECT_EQ(instrId, 1);
            EXPECT_EQ(instr.header.type, 0x1);
            EXPECT_EQ(instr.header.code, 0x2);
        }

        TEST_F(CcuRepLocWaitEventTest, TranslateWithProfilingDisabled)
        {
            CompletedEvent event;
            CcuRepLocWaitEvent rep(&insGen, event, 0x5, false);

            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 10;
            TransDep dep = {};

            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instrId, 11);
            EXPECT_EQ(instr.header.type, 0x1);
            EXPECT_EQ(instr.header.code, 0x4);
        }

        TEST_F(CcuRepLocWaitEventTest, Describe)
        {
            CompletedEvent event;
            CcuRepLocWaitEvent rep(&insGen, event, 0xABCD, true);

            std::string desc = rep.Describe();

            EXPECT_NE(desc.find("CcuRepLocWaitEvent"), std::string::npos);
            EXPECT_NE(desc.find("mask"), std::string::npos);
        }

        TEST_F(CcuRepLocWaitEventTest, SetAndGetDependencyInfo_SingleBit)
        {
            CompletedEvent event;
            CcuRepLocWaitEvent rep(&insGen, event, 0x3, true);
            std::unordered_map<uint32_t, std::vector<std::shared_ptr<CcuRepBase>>> depInfo;
            auto depRep = std::make_shared<CcuRepNop>(&insGen);
            depInfo[0x1].push_back(depRep);
            rep.SetDependencyInfo(depInfo);
            auto result = rep.GetDependencyInfo(0x1);
            EXPECT_EQ(result.size(), 1U);
            EXPECT_EQ(result[0], depRep);
        }

        TEST_F(CcuRepLocWaitEventTest, GetDependencyInfo_NotFound_ReturnsEmpty)
        {
            CompletedEvent event;
            CcuRepLocWaitEvent rep(&insGen, event, 0x3, true);
            auto result = rep.GetDependencyInfo(0x2);
            EXPECT_TRUE(result.empty());
        }

        class CcuRepContextDepTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        TEST_F(CcuRepContextDepTest, SetDependencyInfo_SingleBitMask)
        {
            CcuRepContext context;
            auto rep = std::make_shared<CcuRepNop>(&insGen);
            context.SetDependencyInfo(1, 0x1, rep);
            auto dep = context.GetDependencyInfo(1);
            EXPECT_EQ(dep.size(), 1U);
            EXPECT_EQ(dep.count(0x1), 1U);
            EXPECT_EQ(dep[0x1].size(), 1U);
            EXPECT_EQ(dep[0x1][0], rep);
        }

        TEST_F(CcuRepContextDepTest, SetDependencyInfo_MultiBitMask_SplitToSingleBitKeys)
        {
            CcuRepContext context;
            auto rep = std::make_shared<CcuRepNop>(&insGen);
            // mask=0x3 (bit0+bit1) 应拆解到 0x1 和 0x2 两个单 bit key，与异常侧按 1<<i 查询对齐
            context.SetDependencyInfo(1, 0x3, rep);
            auto dep = context.GetDependencyInfo(1);
            EXPECT_EQ(dep.size(), 2U);
            EXPECT_EQ(dep.count(0x1), 1U);
            EXPECT_EQ(dep.count(0x2), 1U);
            EXPECT_EQ(dep[0x1].size(), 1U);
            EXPECT_EQ(dep[0x1][0], rep);
            EXPECT_EQ(dep[0x2].size(), 1U);
            EXPECT_EQ(dep[0x2][0], rep);
        }

        TEST_F(CcuRepContextDepTest, SetDependencyInfo_HighBitMask_AllBitsRegistered)
        {
            CcuRepContext context;
            auto rep = std::make_shared<CcuRepNop>(&insGen);
            context.SetDependencyInfo(7, 0xABCD, rep);
            auto dep = context.GetDependencyInfo(7);
            // 0xABCD = 1010 1011 1100 1101，置位 bit: 0,2,3,6,7,8,9,11,13,15 共 10 个
            EXPECT_EQ(dep.size(), 10U);
        }

        TEST_F(CcuRepContextDepTest, SetDependencyInfo_ZeroMask_ReturnsError)
        {
            CcuRepContext context;
            auto rep = std::make_shared<CcuRepNop>(&insGen);
            context.SetDependencyInfo(1, 0, rep);
        }

        TEST_F(CcuRepContextDepTest, GetDependencyInfo_NotFound_ReturnsEmpty)
        {
            CcuRepContext context;
            auto dep = context.GetDependencyInfo(999);
            EXPECT_TRUE(dep.empty());
        }

        TEST_F(CcuRepContextDepTest, SetDependencyInfo_MultipleRepsSameBit_AppendedInOrder)
        {
            CcuRepContext context;
            auto rep1 = std::make_shared<CcuRepNop>(&insGen);
            auto rep2 = std::make_shared<CcuRepNop>(&insGen);
            context.SetDependencyInfo(1, 0x1, rep1);
            context.SetDependencyInfo(1, 0x1, rep2);
            auto dep = context.GetDependencyInfo(1);
            EXPECT_EQ(dep[0x1].size(), 2U);
            EXPECT_EQ(dep[0x1][0], rep1);
            EXPECT_EQ(dep[0x1][1], rep2);
        }

        TEST_F(CcuRepContextDepTest, EraseDependencyInfo_OnlyErasesSpecifiedId)
        {
            CcuRepContext context;
            auto repA = std::make_shared<CcuRepNop>(&insGen);
            auto repB = std::make_shared<CcuRepNop>(&insGen);
            // 模拟多 event 交错：A 注册后 B 注册，清 A 不应影响 B
            context.SetDependencyInfo(1, 0x1, repA);
            context.SetDependencyInfo(2, 0x2, repB);
            context.EraseDependencyInfo(1);
            // B 的依赖应保留
            auto depB = context.GetDependencyInfo(2);
            EXPECT_EQ(depB.size(), 1U);
            EXPECT_EQ(depB.count(0x2), 1U);
            EXPECT_EQ(depB[0x2].size(), 1U);
            EXPECT_EQ(depB[0x2][0], repB);
            // A 的依赖应已清除
            EXPECT_TRUE(context.GetDependencyInfo(1).empty());
        }

        TEST_F(CcuRepContextDepTest, ClearDependencyInfo_RemovesAll)
        {
            CcuRepContext context;
            context.SetDependencyInfo(1, 0x1, std::make_shared<CcuRepNop>(&insGen));
            context.SetDependencyInfo(2, 0x2, std::make_shared<CcuRepNop>(&insGen));
            context.ClearDependencyInfo();
            EXPECT_TRUE(context.GetDependencyInfo(1).empty());
            EXPECT_TRUE(context.GetDependencyInfo(2).empty());
        }

        TEST_F(CcuRepLocWaitNotifyTest, ConstructorInitializesCorrectly)
        {
            LocalNotify notify;
            uint32_t mask = 0xFF;
            CcuRepLocWaitNotify rep(&insGen, notify, mask, true);

            EXPECT_EQ(rep.Type(), CcuRepType::LOC_WAIT_NOTIFY);
            EXPECT_EQ(rep.InstrCount(), 1);
            EXPECT_EQ(rep.GetMask(), 0xFF);
        }

        TEST_F(CcuRepLocWaitNotifyTest, ConstructorWithProfilingDisabled)
        {
            LocalNotify notify;
            uint32_t mask = 0x0F;
            CcuRepLocWaitNotify rep(&insGen, notify, mask, false);

            EXPECT_EQ(rep.Type(), CcuRepType::LOC_WAIT_NOTIFY);
            EXPECT_FALSE(rep.Translated());
        }

        TEST_F(CcuRepLocWaitNotifyTest, TranslateWithProfilingEnabled)
        {
            LocalNotify notify;
            uint32_t mask = 0xF;
            CcuRepLocWaitNotify rep(&insGen, notify, mask, true);

            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 0;
            TransDep dep = {};

            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instrId, 1);
            EXPECT_EQ(instr.header.type, 0x1);
            EXPECT_EQ(instr.header.code, 0x2);
        }

        TEST_F(CcuRepLocWaitNotifyTest, TranslateWithProfilingDisabled)
        {
            LocalNotify notify;
            uint32_t mask = 0xA;
            CcuRepLocWaitNotify rep(&insGen, notify, mask, false);

            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 5;
            TransDep dep = {};

            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instrId, 6);
            EXPECT_EQ(instr.header.type, 0x1);
            EXPECT_EQ(instr.header.code, 0x4);
        }

        TEST_F(CcuRepLocWaitNotifyTest, Describe)
        {
            LocalNotify notify;
            uint32_t mask = 0x1234;
            CcuRepLocWaitNotify rep(&insGen, notify, mask, true);

            std::string desc = rep.Describe();

            EXPECT_NE(desc.find("CcuRepLocWaitNotify"), std::string::npos);
        }

        TEST_F(CcuRepLocRecordEventTest, ConstructorInitializesCorrectly)
        {
            CompletedEvent event;
            CcuRepLocRecordEvent rep(&insGen, event, 0xF);

            EXPECT_EQ(rep.Type(), CcuRepType::LOC_RECORD_EVENT);
            EXPECT_EQ(rep.InstrCount(), 1);
            EXPECT_EQ(rep.GetMask(), 0xF);
        }

        TEST_F(CcuRepLocRecordEventTest, Translate)
        {
            CcuInsGeneratorV1 insGen;
            CompletedEvent event;
            CcuRepLocRecordEvent rep(&insGen, event, 0xABCD);

            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 100;
            TransDep dep = {};

            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(rep.StartInstrId(), 100);
            EXPECT_EQ(instrId, 101);
            EXPECT_EQ(instr.header.type, 0x1);
            EXPECT_EQ(instr.header.code, 0x2);
        }

        TEST_F(CcuRepLocRecordEventTest, Describe)
        {
            CompletedEvent event;
            CcuRepLocRecordEvent rep(&insGen, event, 0xABCD);

            std::string desc = rep.Describe();

            EXPECT_NE(desc.find("CcuRepLocRecordEvent"), std::string::npos);
            EXPECT_NE(desc.find("mask"), std::string::npos);
        }

        TEST_F(CcuRepRecordSharedNotifyTest, ConstructorInitializesCorrectly)
        {
            LocalNotify notify;
            uint16_t mask = 0xFF;
            CcuRepRecordSharedNotify rep(&insGen, notify, mask);

            EXPECT_EQ(rep.Type(), CcuRepType::RECORD_SHARED_NOTIFY);
            EXPECT_EQ(rep.InstrCount(), 1);
            EXPECT_EQ(rep.GetMask(), 0xFF);
        }

        TEST_F(CcuRepRecordSharedNotifyTest, TranslateSameDie)
        {
            LocalNotify notify;
            uint16_t mask = 0xF;
            CcuRepRecordSharedNotify rep(&insGen, notify, mask);

            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 0;
            TransDep dep = {};
            dep.dieId = 0;

            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instrId, 1);
            EXPECT_EQ(instr.header.type, 0x1);
            EXPECT_EQ(instr.header.code, 0x2);
        }

        TEST_F(CcuRepRecordSharedNotifyTest, TranslateDifferentDie)
        {
            LocalNotify notify;
            uint16_t mask = 0xA;
            CcuRepRecordSharedNotify rep(&insGen, notify, mask);

            CcuInstr instr;
            CcuInstr* instrPtr = &instr;
            uint16_t instrId = 5;
            TransDep dep = {};
            dep.dieId = 1;
            dep.reserveCkeId = 2;
            dep.reserveChannalId[1] = 3;

            bool result = rep.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instrId, 6);
            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0xb);
        }

        TEST_F(CcuRepRecordSharedNotifyTest, Describe)
        {
            LocalNotify notify;
            uint16_t mask = 0x1234;
            CcuRepRecordSharedNotify rep(&insGen, notify, mask);

            std::string desc = rep.Describe();

            EXPECT_NE(desc.find("Post"), std::string::npos);
            EXPECT_NE(desc.find("mask"), std::string::npos);
        }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
