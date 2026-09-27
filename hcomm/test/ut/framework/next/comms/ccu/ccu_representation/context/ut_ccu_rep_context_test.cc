/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_context_v1.h"
#include "ccu_rep_loopgroup_bundle_v1.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_types.h"
#include <gtest/gtest.h>
#include <memory>
#include <string>

namespace hcomm {
namespace CcuRep {
    namespace {

        class CcuRepContextTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        TEST_F(CcuRepContextTest, Constructor)
        {
            CcuRepContext context;
            EXPECT_NE(context.CurrentBlock(), nullptr);
            EXPECT_EQ(context.CurrentBlock(), context.CurrentBlock());
        }

        TEST_F(CcuRepContextTest, CurrentBlock_InitiallyMainBlock)
        {
            CcuRepContext context;
            auto block = context.CurrentBlock();
            EXPECT_NE(block, nullptr);
        }

        TEST_F(CcuRepContextTest, SetCurrentBlock)
        {
            CcuRepContext context;
            auto newBlock = std::make_shared<CcuRepBlock>(&insGen, "newBlock");
            context.SetCurrentBlock(newBlock);
            EXPECT_EQ(context.CurrentBlock(), newBlock);
        }

        TEST_F(CcuRepContextTest, Append_SingleRep)
        {
            CcuRepContext context;
            auto nop = std::make_shared<CcuRepNop>(&insGen);
            context.Append(nop);
            EXPECT_EQ(context.GetRepSequence().size(), 1U);
        }

        TEST_F(CcuRepContextTest, Append_MultipleReps)
        {
            CcuRepContext context;
            auto nop1 = std::make_shared<CcuRepNop>(&insGen);
            auto nop2 = std::make_shared<CcuRepNop>(&insGen);
            context.Append(nop1);
            context.Append(nop2);
            EXPECT_EQ(context.GetRepSequence().size(), 2U);
        }

        TEST_F(CcuRepContextTest, GetRepSequence_InitiallyEmpty)
        {
            CcuRepContext context;
            EXPECT_EQ(context.GetRepSequence().size(), 0U);
        }

        TEST_F(CcuRepContextTest, GetRepByInstrId_NotFound)
        {
            CcuRepContext context;
            auto result = context.GetRepByInstrId(100);
            EXPECT_EQ(result, nullptr);
        }

        TEST_F(CcuRepContextTest, DumpReprestation_NoCrash)
        {
            CcuRepContext context;
            EXPECT_NO_FATAL_FAILURE(context.DumpReprestation());
        }

        TEST_F(CcuRepContextTest, DumpReprestation_WithReps)
        {
            CcuRepContext context;
            auto nop = std::make_shared<CcuRepNop>(&insGen);
            context.Append(nop);
            EXPECT_NO_FATAL_FAILURE(context.DumpReprestation());
        }

        TEST_F(CcuRepContextTest, SetAndGetDieId)
        {
            CcuRepContext context;
            uint32_t dieId = 5;
            context.SetDieId(dieId);
            EXPECT_EQ(context.GetDieId(), dieId);
        }

        TEST_F(CcuRepContextTest, SetAndGetMissionId)
        {
            CcuRepContext context;
            uint32_t missionId = 10;
            context.SetMissionId(missionId);
            EXPECT_EQ(context.GetMissionId(), missionId);
        }

        TEST_F(CcuRepContextTest, SetMissionId_OnlySetsOnce)
        {
            CcuRepContext context;
            context.SetMissionId(10);
            context.SetMissionId(20);
            EXPECT_EQ(context.GetMissionId(), 10U);
        }

        TEST_F(CcuRepContextTest, SetAndGetMissionKey)
        {
            CcuRepContext context;
            uint32_t missionKey = 15;
            context.SetMissionKey(missionKey);
            EXPECT_EQ(context.GetMissionKey(), missionKey);
        }

        TEST_F(CcuRepContextTest, GetProfilingInfo_InitiallyEmpty)
        {
            CcuRepContext context;
            EXPECT_EQ(context.GetProfilingInfo().size(), 0U);
        }

        TEST_F(CcuRepContextTest, GetLGProfilingInfo_InitiallyEmpty)
        {
            CcuRepContext context;
            EXPECT_EQ(context.GetLGProfilingInfo().ccuProfilingInfos.size(), 0U);
        }

        TEST_F(CcuRepContextTest, GetWaiteCkeProfilingReps_InitiallyEmpty)
        {
            CcuRepContext context;
            EXPECT_EQ(context.GetWaiteCkeProfilingReps().size(), 0U);
        }

        TEST_F(CcuRepContextTest, CollectProfilingReps_Assign_NotVarToVar)
        {
            CcuRepContext context;
            Variable varA(nullptr);
            CcuRepAssign assign(&insGen, varA, 100);
            assign.subType = AssignSubType::IMD_TO_VARIABLE;
            auto assignPtr = std::make_shared<CcuRepAssign>(assign);
            context.CollectProfilingReps(assignPtr);
            EXPECT_EQ(context.GetLGProfilingInfo().assignProfilingReps.size(), 0U);
        }

        TEST_F(CcuRepContextTest, CollectProfilingReps_Assign_VarToVar)
        {
            CcuRepContext context;
            Variable varA(nullptr);
            Variable varB(nullptr);
            CcuRepAssign assign(&insGen, varB, varA);
            assign.subType = AssignSubType::VAR_TO_VAR;
            auto assignPtr = std::make_shared<CcuRepAssign>(assign);
            context.CollectProfilingReps(assignPtr);
            EXPECT_EQ(context.GetLGProfilingInfo().assignProfilingReps.size(), 1U);
        }

        TEST_F(CcuRepContextTest, CollectProfilingReps_LoopGroup)
        {
            CcuRepContext context;
            CcuLoopGroupCfg grpCfg{};
            Variable parallelParam(nullptr);
            Variable offsetParam(nullptr);
            auto loopGroupPtr = std::make_shared<CcuRepLoopGroupBundle>(&insGen, grpCfg, parallelParam, offsetParam);
            context.CollectProfilingReps(loopGroupPtr);
            EXPECT_EQ(context.GetLGProfilingInfo().lgProfilingReps.size(), 1U);
        }

        TEST_F(CcuRepContextTest, AddSqeProfiling)
        {
            CcuRepContext context;
            context.SetDieId(3);
            context.AddSqeProfiling();
            EXPECT_EQ(context.GetProfilingInfo().size(), 1U);
            EXPECT_EQ(context.GetProfilingInfo()[0].type, static_cast<uint8_t>(CcuProfilinType::CCU_TASK_PROFILING));
            EXPECT_EQ(context.GetProfilingInfo()[0].dieId, 3U);
        }

        TEST_F(CcuRepContextTest, AddProfiling_Simple)
        {
            CcuRepContext context;
            std::string name = "test_profiling";
            uint32_t mask = 0xF;
            int32_t result = context.AddProfiling(name, mask);
            EXPECT_EQ(result, HCCL_SUCCESS);
            EXPECT_EQ(context.GetProfilingInfo().size(), 1U);
            EXPECT_EQ(context.GetProfilingInfo()[0].type, static_cast<uint8_t>(CcuProfilinType::CCU_WAITCKE_PROFILING));
            EXPECT_EQ(context.GetProfilingInfo()[0].name, name);
        }

        TEST_F(CcuRepContextTest, Append_CollectsProfilingReps)
        {
            CcuRepContext context;
            Variable varA(nullptr);
            Variable varB(nullptr);
            CcuRepAssign assign(&insGen, varB, varA);
            assign.subType = AssignSubType::VAR_TO_VAR;
            auto assignPtr = std::make_shared<CcuRepAssign>(assign);
            context.Append(assignPtr);
            EXPECT_EQ(context.GetLGProfilingInfo().assignProfilingReps.size(), 1U);
        }

        TEST_F(CcuRepContextTest, SetCurrentBlock_ToNullBlock)
        {
            CcuRepContext context;
            auto nullBlock = std::make_shared<CcuRepBlock>(&insGen);
            context.SetCurrentBlock(nullBlock);
            EXPECT_EQ(context.CurrentBlock(), nullBlock);
        }

        TEST_F(CcuRepContextTest, GetRepSequence_ReturnsMainBlockReps)
        {
            CcuRepContext context;
            auto nop = std::make_shared<CcuRepNop>(&insGen);
            context.Append(nop);
            const auto& reps = context.GetRepSequence();
            EXPECT_EQ(reps.size(), 1U);
        }

        TEST_F(CcuRepContextTest, MultipleBlocks_SetCurrentBlock)
        {
            CcuRepContext context;
            auto block1 = std::make_shared<CcuRepBlock>(&insGen, "block1");
            auto block2 = std::make_shared<CcuRepBlock>(&insGen, "block2");

            context.SetCurrentBlock(block1);
            auto nop1 = std::make_shared<CcuRepNop>(&insGen);
            context.Append(nop1);

            context.SetCurrentBlock(block2);
            auto nop2 = std::make_shared<CcuRepNop>(&insGen);
            context.Append(nop2);

            EXPECT_EQ(block1->GetReps().size(), 1U);
            EXPECT_EQ(block2->GetReps().size(), 1U);
        }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
