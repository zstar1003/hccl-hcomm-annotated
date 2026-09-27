/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_ins_generator_v1.h"
#include "ccu_ins_generator_v2.h"
#include "ccu_rep_reference_manager_v1.h"
#include "ccu_rep_type_v1.h"
#include "ccu_api_exception.h"
#include <gtest/gtest.h>
#include <string>

namespace hcomm {
namespace CcuRep {

    extern uint32_t GetRelativeInstrId(uint32_t currentInstrId, uint32_t targetInstrId);

    namespace {

        class GetRelativeInstrIdTest : public ::testing::Test {};

        TEST_F(GetRelativeInstrIdTest, ForwardOffset)
        {
            EXPECT_EQ(GetRelativeInstrId(10, 20), 10u);
            EXPECT_EQ(GetRelativeInstrId(0, 1), 1u);
            EXPECT_EQ(GetRelativeInstrId(5, 100), 95u);
        }

        TEST_F(GetRelativeInstrIdTest, EqualIds) { EXPECT_EQ(GetRelativeInstrId(50, 50), 0x10000u); }

        TEST_F(GetRelativeInstrIdTest, BackwardOffset)
        {
            EXPECT_EQ(GetRelativeInstrId(20, 10), 0x10000u - 10u);
            EXPECT_EQ(GetRelativeInstrId(100, 5), 0x10000u - 95u);
        }

        TEST_F(GetRelativeInstrIdTest, WrapAround)
        {
            EXPECT_EQ(GetRelativeInstrId(0xFFFF, 0), 1u);
            EXPECT_EQ(GetRelativeInstrId(0xFFFF, 1), 2u);
            EXPECT_EQ(GetRelativeInstrId(1, 0xFFFF), 0x10000u - 2u);
        }

        class CcuInsGeneratorV2GetInstrCountTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV2 insGen;
        };

        TEST_F(CcuInsGeneratorV2GetInstrCountTest, Nop) { EXPECT_EQ(insGen.GetInstrCount(CcuRepType::NOP), 1u); }

        TEST_F(CcuInsGeneratorV2GetInstrCountTest, Add) { EXPECT_GT(insGen.GetInstrCount(CcuRepType::ADD), 0u); }

        TEST_F(CcuInsGeneratorV2GetInstrCountTest, Assign) { EXPECT_GT(insGen.GetInstrCount(CcuRepType::ASSIGN), 0u); }

        TEST_F(CcuInsGeneratorV2GetInstrCountTest, Jump) { EXPECT_GT(insGen.GetInstrCount(CcuRepType::JUMP), 0u); }

        TEST_F(CcuInsGeneratorV2GetInstrCountTest, Load) { EXPECT_GT(insGen.GetInstrCount(CcuRepType::LOAD), 0u); }

        TEST_F(CcuInsGeneratorV2GetInstrCountTest, Shr) { EXPECT_EQ(insGen.GetInstrCount(CcuRepType::SHR), 1u); }

        TEST_F(CcuInsGeneratorV2GetInstrCountTest, Shl) { EXPECT_EQ(insGen.GetInstrCount(CcuRepType::SHL), 1u); }

        class CcuInsGeneratorV1GetInstrCountTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen;
        };

        TEST_F(CcuInsGeneratorV1GetInstrCountTest, Nop) { EXPECT_EQ(insGen.GetInstrCount(CcuRepType::NOP), 1u); }

        TEST_F(CcuInsGeneratorV1GetInstrCountTest, Add) { EXPECT_GT(insGen.GetInstrCount(CcuRepType::ADD), 0u); }

        TEST_F(CcuInsGeneratorV1GetInstrCountTest, Mul_ThrowsException_WhenNotInTable)
        {
            EXPECT_THROW(insGen.GetInstrCount(CcuRepType::MUL), Hccl::CcuApiException);
        }

        TEST_F(CcuInsGeneratorV1GetInstrCountTest, Shr_ThrowsException_WhenNotInTable)
        {
            EXPECT_THROW(insGen.GetInstrCount(CcuRepType::SHR), Hccl::CcuApiException);
        }

        class CcuRepReferenceManagerTest : public ::testing::Test {
        protected:
            std::unique_ptr<CcuRepReferenceManager> mgr;

            void SetUp() override { mgr = std::make_unique<CcuRepReferenceManager>(0); }
        };

        TEST_F(CcuRepReferenceManagerTest, Constructor_Normal) { EXPECT_NE(mgr, nullptr); }

        TEST_F(CcuRepReferenceManagerTest, GetResReq_Normal)
        {
            CcuResReq req = CcuRepReferenceManager::GetResReq(0);
            EXPECT_GT(req.xnReq[0], 0u);
        }

        TEST_F(CcuRepReferenceManagerTest, GetResReq_DifferentDieId)
        {
            CcuResReq req = CcuRepReferenceManager::GetResReq(1);
            EXPECT_GT(req.xnReq[1], 0u);
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncIn_Normal)
        {
            const auto& funcIn = mgr->GetFuncIn();
            EXPECT_EQ(funcIn.size(), static_cast<size_t>(FUNC_ARG_MAX));
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncOut_Normal)
        {
            const auto& funcOut = mgr->GetFuncOut();
            EXPECT_EQ(funcOut.size(), static_cast<size_t>(FUNC_ARG_MAX));
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncCall_Normal)
        {
            const Variable& funcCall = mgr->GetFuncCall();
            (void)funcCall;
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncRet_Layer0)
        {
            const Variable& ret = mgr->GetFuncRet(0);
            (void)ret;
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncRet_LayerMax)
        {
            const Variable& ret = mgr->GetFuncRet(FUNC_NEST_MAX);
            (void)ret;
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncRet_ExceedMax_ThrowsException)
        {
            EXPECT_THROW(mgr->GetFuncRet(FUNC_NEST_MAX + 1), Hccl::CcuApiException);
        }

        TEST_F(CcuRepReferenceManagerTest, GetRefBlock_NotExist_ThrowsException)
        {
            EXPECT_THROW(mgr->GetRefBlock("nonexistent"), Hccl::CcuApiException);
        }

        TEST_F(CcuRepReferenceManagerTest, SetRefBlock_ThenGetRefBlock_Success)
        {
            CcuInsGeneratorV2 insGen;
            auto block = std::make_shared<CcuRepBlock>(&insGen, "testLabel");
            mgr->SetRefBlock("testLabel", block);
            auto result = mgr->GetRefBlock("testLabel");
            EXPECT_NE(result, nullptr);
        }

        TEST_F(CcuRepReferenceManagerTest, SetRefBlock_Duplicate_ThrowsException)
        {
            CcuInsGeneratorV2 insGen;
            auto block1 = std::make_shared<CcuRepBlock>(&insGen, "label1");
            auto block2 = std::make_shared<CcuRepBlock>(&insGen, "label1");
            mgr->SetRefBlock("label1", block1);
            EXPECT_THROW(mgr->SetRefBlock("label1", block2), Hccl::CcuApiException);
        }

        TEST_F(CcuRepReferenceManagerTest, SetRefBlock_DifferentLabels_Success)
        {
            CcuInsGeneratorV2 insGen;
            auto block1 = std::make_shared<CcuRepBlock>(&insGen, "label1");
            auto block2 = std::make_shared<CcuRepBlock>(&insGen, "label2");
            mgr->SetRefBlock("label1", block1);
            mgr->SetRefBlock("label2", block2);
            EXPECT_NE(mgr->GetRefBlock("label1"), nullptr);
            EXPECT_NE(mgr->GetRefBlock("label2"), nullptr);
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncAddr_NotExist_ThrowsException)
        {
            EXPECT_THROW(mgr->GetFuncAddr("nonexistent"), Hccl::CcuApiException);
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncAddr_WrongType_ThrowsException)
        {
            CcuInsGeneratorV2 insGen;
            auto block = std::make_shared<CcuRepBlock>(&insGen, "notFunc");
            mgr->SetRefBlock("notFunc", block);
            EXPECT_THROW(mgr->GetFuncAddr("notFunc"), Hccl::CcuApiException);
        }

        TEST_F(CcuRepReferenceManagerTest, ClearRepReference_AfterClear_GetThrows)
        {
            CcuInsGeneratorV2 insGen;
            auto block = std::make_shared<CcuRepBlock>(&insGen, "label");
            mgr->SetRefBlock("label", block);
            mgr->ClearRepReference();
            EXPECT_THROW(mgr->GetRefBlock("label"), Hccl::CcuApiException);
        }

        TEST_F(CcuRepReferenceManagerTest, ClearRepReference_ThenSetAgain_Success)
        {
            CcuInsGeneratorV2 insGen;
            auto block1 = std::make_shared<CcuRepBlock>(&insGen, "label");
            mgr->SetRefBlock("label", block1);
            mgr->ClearRepReference();
            auto block2 = std::make_shared<CcuRepBlock>(&insGen, "label");
            mgr->SetRefBlock("label", block2);
            EXPECT_NE(mgr->GetRefBlock("label"), nullptr);
        }

        TEST_F(CcuRepReferenceManagerTest, Dump_DoesNotCrash)
        {
            CcuInsGeneratorV2 insGen;
            auto block = std::make_shared<CcuRepBlock>(&insGen, "dumpLabel");
            mgr->SetRefBlock("dumpLabel", block);
            mgr->Dump();
            SUCCEED();
        }

        TEST_F(CcuRepReferenceManagerTest, GetRes_Normal)
        {
            CcuRepResource res;
            mgr->GetRes(res);
            EXPECT_FALSE(res.variable[0].empty());
        }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
