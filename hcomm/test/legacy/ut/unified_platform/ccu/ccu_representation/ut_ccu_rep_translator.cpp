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
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/mokc.h>
#define private public
#define protected public
#include "ccu_rep_translator.h"
#include "ccu_rep.h"
#undef protected
#undef private

using namespace Hccl;
using namespace CcuRep;

class CcuRepTranslatorTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuRepTranslatorTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuRepTranslatorTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
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
        std::cout << "A Test case in CcuRepTranslatorTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuRepTranslatorTest TearDown" << std::endl;
    }
};

TEST_F(CcuRepTranslatorTest, get_translator_require_physical_resources_num)
{
    uint8_t dieId = 0;
    CcuResReq translatorResReq = CcuRepTranslator::GetResReq(dieId);
    CcuResReq refMgrResReq = CcuRepReferenceManager::GetResReq(dieId);

    EXPECT_EQ(refMgrResReq.xnReq[dieId], FUNC_IN_MAX + FUNC_OUT_MAX + FUNC_NEST_MAX + 2);
    EXPECT_EQ(translatorResReq.xnReq[dieId], 4);
    EXPECT_EQ(translatorResReq.gsaReq[dieId], 3);
    EXPECT_EQ(translatorResReq.ckeReq[dieId], 2);
}

CcuRepTranslator InitCcuRepResource(CcuRepResource& res)
{
    // 实例化CcuReferenceManager和CcuTranslator，并为其绑定物理资源
    auto refManager = std::make_shared<CcuRepReferenceManager>(0);
    refManager->GetRes(res);
    std::array<uint16_t, 2> channels = {0, 0};
    std::pair<uint64_t, uint64_t> token_info = {0, 0};
    auto translator = CcuRepTranslator(0, 0, refManager, channels, token_info, 0);
    translator.GetRes(res);
    return translator;
}

TEST_F(CcuRepTranslatorTest, translator_bind_phy_resource_success)
{
    CcuRepResource res;
    auto translator = InitCcuRepResource(res);

    EXPECT_EQ(res.variable[0].size(), FUNC_IN_MAX + FUNC_OUT_MAX + FUNC_NEST_MAX + 1 + 1 + 4);
    EXPECT_EQ(res.address[0].size(), 3);
    EXPECT_EQ(res.maskSignal[0].size(), 2);

    // 绑定物理资源
    uint16_t id = 10;
    for (auto& var : res.variable[0]) {
        var.Reset(id);
        id++;
    }
    res.maskSignal[0][0].Reset(5);
    res.address[0][0].Reset(2);

    // 检验translator中资源id
    EXPECT_EQ(translator.var[0].Id(), 10 + FUNC_IN_MAX + FUNC_OUT_MAX + FUNC_NEST_MAX + 2);
    EXPECT_EQ(translator.addr[0].Id(), 2);
    EXPECT_EQ(translator.signal[0].Id(), 5);

    // 构造rep序列
    std::vector<std::shared_ptr<CcuRepBase>> repVec;
    Variable var;
    auto repLoad = std::make_shared<CcuRepLoadArg>(var, 0);
    repVec.push_back(repLoad);
    translator.Translate(repVec, 0);

    // 检验translator中transDep的值是否符合预期
    EXPECT_EQ(translator.transDep.logicalId, 0);
    EXPECT_EQ(translator.transDep.dieId, 0);
    EXPECT_EQ(translator.transDep.reserveXnId, 10 + FUNC_IN_MAX + FUNC_OUT_MAX + FUNC_NEST_MAX + 2);
    EXPECT_EQ(translator.transDep.reserveGsaId, 2);
    EXPECT_EQ(translator.transDep.reserveCkeId, 5);
};

TEST_F(CcuRepTranslatorTest, translate_rep_vec_success)
{
    CcuRepResource res;
    auto translator = InitCcuRepResource(res);

    // 构造rep序列
    std::vector<std::shared_ptr<CcuRepBase>> repVec;
    Variable var;
    auto repLoad = std::make_shared<CcuRepLoadArg>(var, 0);
    auto repLoopCall = std::make_shared<CcuRepLoopCall>("loop");
    auto repLoopBlock = std::make_shared<CcuRepLoopBlock>("loop");
    repVec.push_back(repLoad);
    repVec.push_back(repLoopCall);
    repVec.push_back(repLoopBlock);

    // 翻译Rep序列
    EXPECT_NO_THROW(translator.Translate(repVec, 0));
}

TEST_F(CcuRepTranslatorTest, translate_build_reference_faild)
{
    CcuRepResource res;
    auto translator = InitCcuRepResource(res);

    // 构造rep序列, LoopCall引用的LoopBlock不存在
    std::vector<std::shared_ptr<CcuRepBase>> repVec;
    Variable var;
    auto repLoad = std::make_shared<CcuRepLoadArg>(var, 0);
    auto repLoopCall = std::make_shared<CcuRepLoopCall>("Loop");
    repVec.push_back(repLoad);
    repVec.push_back(repLoopCall);

    // 翻译Rep序列,预期失败
    EXPECT_THROW(translator.Translate(repVec, 0), CcuApiException);

    // 构造rep序列, 存在两个lable相同的LoopBlock
    std::vector<std::shared_ptr<CcuRepBase>> repVec2;
    auto repLoopBlock1 = std::make_shared<CcuRepLoopBlock>("Loop");
    auto repLoopBlock2 = std::make_shared<CcuRepLoopBlock>("Loop");
    repVec2.push_back(repLoad);
    repVec2.push_back(repLoopBlock1);
    repVec2.push_back(repLoopBlock2);
    // 翻译Rep序列，预期失败
    EXPECT_THROW(translator.Translate(repVec2, 0), CcuApiException);
}

TEST_F(CcuRepTranslatorTest, translate_rep_vec_load_args_success)
{
    CcuRepResource res;
    auto translator = InitCcuRepResource(res);

    // 构造rep序列
    std::vector<std::shared_ptr<CcuRepBase>> repVec;
    Variable var;
    auto repLoad0 = std::make_shared<CcuRepLoadArg>(var, 0);
    auto repLoad1 = std::make_shared<CcuRepLoadArg>(var, 1);
    auto repLoopCall = std::make_shared<CcuRepLoopCall>("loop");
    auto repLoopBlock = std::make_shared<CcuRepLoopBlock>("loop");

    repVec.push_back(repLoad0);
    repVec.push_back(repLoopCall);
    repVec.push_back(repLoopBlock);
    repVec.push_back(repLoad1);
    // 输出的微码指令应该按照FuncBlock、LoadArg、FuncCall的顺序, 且第二个load指令对应的微码指令中sqeArgsId应该为0
    EXPECT_NO_THROW(translator.Translate(repVec, 0));
}

TEST_F(CcuRepTranslatorTest, translate_rep_vec_load_args_with_func_block_success)
{
    CcuRepResource res;
    auto translator = InitCcuRepResource(res);

    // 构造rep序列
    std::vector<std::shared_ptr<CcuRepBase>> repVec;
    Variable var;
    auto repLoad0 = std::make_shared<CcuRepLoadArg>(var, 0);
    auto repLoad1 = std::make_shared<CcuRepLoadArg>(var, 1);
    auto repLoopCall = std::make_shared<CcuRepLoopCall>("loop");
    auto repLoopBlock = std::make_shared<CcuRepLoopBlock>("loop");

    repVec.push_back(repLoad0);
    repVec.push_back(repLoopCall);
    repVec.push_back(repLoopBlock);
    repVec.push_back(repLoad1);
    // 输出的微码指令应该按照FuncBlock、LoadArg、FuncCall的顺序, 且第二个load指令对应的微码指令中sqeArgsId应该为0
    EXPECT_NO_THROW(translator.Translate(repVec, 0));
}
