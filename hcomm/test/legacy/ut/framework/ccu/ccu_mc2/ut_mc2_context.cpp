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
#include "mc2_context.h"
#include "ccu_ctx.h"
#undef private
#undef protected
#include "ccu_ctx_arg_mc2.h"
#include "ccu_task_arg_mc2.h"
#include "ccu_transport.h"
#include "ccu_transport_group.h"
#include "ccu_device_manager.h"
#include "ccu_rep_translator.h"
#include "orion_adapter_rts.h"

using namespace std;
using namespace Hccl;
using namespace CcuRep;

class Mc2ContextTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Mc2ContextTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "Mc2ContextTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in Mc2ContextTest SetUP" << std::endl;
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));
        MOCKER(&CcuDeviceManager::GetLoopChannelId)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(&CcuDeviceManager::GetXnBaseAddr)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(&CcuDeviceManager::GetCcuResourceSpaceTokenInfo)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HcclResult::HCCL_SUCCESS));
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in Mc2ContextTest TearDown" << std::endl;
    }
};

TEST_F(Mc2ContextTest, should_throw_exception_when_algoTemplateInfo_empty)
{
    Mc2Context mc2Context{};

    EXPECT_THROW(mc2Context.Algorithm(), InvalidParamsException);
}

TEST_F(Mc2ContextTest, should_throw_exception_when_die_config_error)
{
    Mc2Context mc2Context{};

    std::map<uint64_t, uint32_t> algoTemplateInfo{{0, 0}};
    mc2Context.SetAlgoTemplateInfo(algoTemplateInfo);              // algoTemplateInfo_设置为非空
    EXPECT_THROW(mc2Context.SetDieNum(3), InvalidParamsException); // dieNum_设置为非法值
}

TEST_F(Mc2ContextTest, should_return_success_when_single_die)
{
    Mc2Context mc2Context{};

    mc2Context.SetDieNum(1);         // 设置为单die
    mc2Context.SetCommAddr(0, 1024); // 设置HBM内存地址
    std::map<uint64_t, uint32_t> algoTemplateInfo{{0, 1}, {1, 2}};
    mc2Context.SetAlgoTemplateInfo(algoTemplateInfo); // algoTemplateInfo_设置为非空

    EXPECT_NO_THROW(mc2Context.Algorithm());

    auto refManager = std::make_shared<CcuRepReferenceManager>(0);
    std::array<uint16_t, 2> channels = {0, 0};
    std::pair<uint64_t, uint64_t> token_info = {0, 0};
    auto translator = CcuRepTranslator(0, 0, refManager, channels, token_info, 0);
    auto instrInfo = translator.Translate(mc2Context.GetRepSequence(), mc2Context.GetInstrId());
}

TEST_F(Mc2ContextTest, should_return_success_when_double_die_0)
{
    Mc2Context mc2Context{};

    mc2Context.SetDieNum(2);         // 设置为双die
    mc2Context.SetDieId(0);          // 设置die id
    mc2Context.SetCommAddr(0, 1024); // 设置HBM内存地址
    std::map<uint64_t, uint32_t> algoTemplateInfo{{0, 1}, {1, 2}};
    mc2Context.SetAlgoTemplateInfo(algoTemplateInfo); // algoTemplateInfo_设置为非空

    EXPECT_NO_THROW(mc2Context.Algorithm());

    auto refManager = std::make_shared<CcuRepReferenceManager>(0);
    std::array<uint16_t, 2> channels = {0, 0};
    std::pair<uint64_t, uint64_t> token_info = {0, 0};
    auto translator = CcuRepTranslator(0, 0, refManager, channels, token_info, 0);
    auto instrInfo = translator.Translate(mc2Context.GetRepSequence(), mc2Context.GetInstrId());
}

TEST_F(Mc2ContextTest, Ut_LoadFuncParamFromMemory_when_iTurn_is_0_Expect_no_throw_exception)
{
    // 前置条件
    Mc2Context mc2Context{};

    mc2Context.SetDieNum(2);         // 设置为双die
    mc2Context.SetDieId(0);          // 设置die id
    mc2Context.SetCommAddr(0, 1024); // 设置HBM内存地址
    std::map<uint64_t, uint32_t> algoTemplateInfo{{0, 1}, {1, 2}};
    mc2Context.SetAlgoTemplateInfo(algoTemplateInfo); // algoTemplateInfo_设置为非空
    uint32_t iTurn = 0;
    constexpr uint32_t CCU_PARAM_NUM_PER_DIE = 32;
    array<CcuRep::Variable, CCU_PARAM_NUM_PER_DIE> param;
}
