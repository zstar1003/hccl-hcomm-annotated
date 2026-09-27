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

#include "ccu_rep_translator.h"
#include "ccu_rep.h"

using namespace Hccl;
using namespace CcuRep;

class CcuRepStoreTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuRepStoreTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuRepStoreTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CcuRepStoreTest SetUP" << std::endl; }

    virtual void TearDown() { std::cout << "A Test case in CcuRepStoreTest TearDown" << std::endl; }

public:
    const uint16_t INSTR_NUM = 7;
    const uint16_t XN_NUM = 4;
    const uint16_t GSA_NUM = 3;
};

TEST_F(CcuRepStoreTest, rep_store_translate)
{
    uint64_t hbmAddr = 0x12341234;
    Variable var;
    var.Reset(0x123);
    CcuRepStore ccuRepStore(var, hbmAddr);

    EXPECT_EQ(ccuRepStore.Type(), CcuRepType::STORE);

    // 资源初始化
    size_t instrSize = sizeof(CcuInstr) * INSTR_NUM;
    CcuInstr* instr = (CcuInstr*)malloc(instrSize);
    memset_s(instr, instrSize, 0, instrSize);
    uint16_t instrId = 0;
    TransDep transDep{0};
    for (int i = 0; i < XN_NUM - 1; i++) {
        transDep.commXn[i] = i;
    }
    for (int i = 0; i < GSA_NUM - 1; i++) {
        transDep.commGsa[i] = i;
    }
    transDep.commSignal = 0;
    transDep.xnBaseAddr = 0x10000;

    // 翻译STORE
    CcuInstr* instrOri = instr;
    bool translated = ccuRepStore.Translate(instr, instrId, transDep);
    EXPECT_TRUE(translated);
    EXPECT_EQ(instrId, INSTR_NUM); // STORE包含指令数

    free(instrOri);
}
