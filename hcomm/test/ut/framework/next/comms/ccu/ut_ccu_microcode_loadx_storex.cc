/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "ccu_microcode_v1.h"
#include "log.h"

using namespace hcomm;
using namespace hcomm::CcuRep;

class CcuMicroCodeLoadXStoreXTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuMicroCodeLoadXStoreXTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuMicroCodeLoadXStoreXTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CcuMicroCodeLoadXStoreXTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuMicroCodeLoadXStoreXTest TearDown" << std::endl;
    }
};

/* LoadX: 验证全部位段字段按入参正确填充(新映射: xdId=dst, xsId=srcOffset, xso=src, xdo=dstOffset) */
TEST_F(CcuMicroCodeLoadXStoreXTest, LoadX_FillFieldsCorrectly)
{
    CcuInstr instr = {};
    CcuV2::LoadX(&instr, 0x8001, 0x8002, 0x0003, 0x0004, 1, 0x0010, 0xFF00);
    EXPECT_EQ(instr.v2.loadStoreX.xdId, 0x8001);
    EXPECT_EQ(instr.v2.loadStoreX.xsId, 0x0003); // srcOffset 参数填入 xsId 位段
    EXPECT_EQ(instr.v2.loadStoreX.xso, 0x8002);  // src 参数填入 xso 位段
    EXPECT_EQ(instr.v2.loadStoreX.xdo, 0x0004);
    EXPECT_EQ(instr.v2.loadStoreX.oMode, 1u);
    EXPECT_EQ(instr.v2.loadStoreX.setCKEId, 0x0010);
    EXPECT_EQ(instr.v2.loadStoreX.setCKEMask, 0xFF00);
}

/* LoadX: oMode 仅取最低 1 bit(2 -> 0, 3 -> 1) */
TEST_F(CcuMicroCodeLoadXStoreXTest, LoadX_OmodeOnlyLowBit)
{
    CcuInstr instr = {};
    CcuV2::LoadX(&instr, 1, 2, 3, 4, 2, 0, 0);
    EXPECT_EQ(instr.v2.loadStoreX.oMode, 0u);

    CcuV2::LoadX(&instr, 1, 2, 3, 4, 3, 0, 0);
    EXPECT_EQ(instr.v2.loadStoreX.oMode, 1u);
}

/* StoreX: 验证全部位段字段按入参正确填充(新映射: xdId=dstOffset, xsId=src, xso=srcOffset, xdo=dst) */
TEST_F(CcuMicroCodeLoadXStoreXTest, StoreX_FillFieldsCorrectly)
{
    CcuInstr instr = {};
    CcuV2::StoreX(&instr, 0x0005, 0x8006, 0x0007, 0x8008, 0, 0x0020, 0x00FF);
    // 位段映射: xdId <- dstOffset 参数(原样含模式位), xsId <- src, xso <- srcOffset, xdo <- dst
    EXPECT_EQ(instr.v2.loadStoreX.xdId, 0x8008);
    EXPECT_EQ(instr.v2.loadStoreX.xsId, 0x8006);
    EXPECT_EQ(instr.v2.loadStoreX.xso, 0x0007);
    EXPECT_EQ(instr.v2.loadStoreX.xdo, 0x0005);
    EXPECT_EQ(instr.v2.loadStoreX.oMode, 0u);
    EXPECT_EQ(instr.v2.loadStoreX.setCKEId, 0x0020);
    EXPECT_EQ(instr.v2.loadStoreX.setCKEMask, 0x00FF);
}

/* StoreX: oMode 仅取最低 1 bit */
TEST_F(CcuMicroCodeLoadXStoreXTest, StoreX_OmodeOnlyLowBit)
{
    CcuInstr instr = {};
    CcuV2::StoreX(&instr, 1, 2, 3, 4, 5, 0, 0);
    EXPECT_EQ(instr.v2.loadStoreX.oMode, 1u);

    CcuV2::StoreX(&instr, 1, 2, 3, 4, 6, 0, 0);
    EXPECT_EQ(instr.v2.loadStoreX.oMode, 0u);
}

/* LoadX 已注册进指令表: ParseInstrV2 能识别并打印 LoadX 专属字段 */
TEST_F(CcuMicroCodeLoadXStoreXTest, ParseInstr_LoadXRegisteredInTable)
{
    CcuInstr instr = {};
    CcuV2::LoadX(&instr, 0x8001, 0x0002, 0x0003, 0x0004, 1, 0x0010, 0xFF00);
    std::string result = CcuV2::ParseInstrV2(&instr);
    // 未注册时返回空串
    EXPECT_FALSE(result.empty());
    EXPECT_TRUE(result.find("LoadX") != std::string::npos);
}

/* StoreX 已注册进指令表: ParseInstrV2 能识别 */
TEST_F(CcuMicroCodeLoadXStoreXTest, ParseInstr_StoreXRegisteredInTable)
{
    CcuInstr instr = {};
    CcuV2::StoreX(&instr, 0x0005, 0x0006, 0x0007, 0x0008, 0, 0, 0);
    std::string result = CcuV2::ParseInstrV2(&instr);
    EXPECT_FALSE(result.empty());
    EXPECT_TRUE(result.find("StoreX") != std::string::npos);
}

/* ParseLoadX: 反解析字符串包含全部关键字段 */
TEST_F(CcuMicroCodeLoadXStoreXTest, ParseInstr_LoadXContainsAllFields)
{
    CcuInstr instr = {};
    CcuV2::LoadX(&instr, 0x8001, 0x0002, 0x0003, 0x0004, 1, 0x0010, 0xFF00);
    std::string result = CcuV2::ParseInstrV2(&instr);
    EXPECT_TRUE(result.find("xdId[32769]") != std::string::npos); // 0x8001
    EXPECT_TRUE(result.find("xsId[3]") != std::string::npos);     // srcOffset 参数
    EXPECT_TRUE(result.find("xso[2]") != std::string::npos);      // src 参数
    EXPECT_TRUE(result.find("xdo[4]") != std::string::npos);
    EXPECT_TRUE(result.find("oMode[1]") != std::string::npos);
    EXPECT_TRUE(result.find("CKE[16:ff00]") != std::string::npos);
}

/* ParseStoreX: 反解析字符串包含全部关键字段 */
TEST_F(CcuMicroCodeLoadXStoreXTest, ParseInstr_StoreXContainsAllFields)
{
    CcuInstr instr = {};
    CcuV2::StoreX(&instr, 5, 6, 7, 8, 0, 32, 255);
    std::string result = CcuV2::ParseInstrV2(&instr);
    EXPECT_TRUE(result.find("xdId[8]") != std::string::npos); // dstOffset 参数
    EXPECT_TRUE(result.find("xsId[6]") != std::string::npos);
    EXPECT_TRUE(result.find("xso[7]") != std::string::npos);
    EXPECT_TRUE(result.find("xdo[5]") != std::string::npos); // dst 参数
    EXPECT_TRUE(result.find("oMode[0]") != std::string::npos);
    EXPECT_TRUE(result.find("CKE[32:00ff]") != std::string::npos);
}
