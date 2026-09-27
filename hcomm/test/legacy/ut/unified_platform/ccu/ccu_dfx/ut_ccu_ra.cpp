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

#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"
#include "ccu_device_manager.h"

#define private public
#define protected public
#include "ccu_error_handler.h"
#undef private
#undef protected

using namespace std;
using namespace Hccl;

class CcuRaTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuRaTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuRaTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CcuRaTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuRaTest TearDown" << std::endl;
    }
};

TEST_F(CcuRaTest, test_ccu_mission_context)
{
    EXPECT_EQ(sizeof(CcuMissionContext), 64);

    CcuMissionContext missionCtx{};
    missionCtx.part2.value = 0xaaaa;
    missionCtx.part3.value = 0xbbbb;
    missionCtx.part4.value = 0xcccc;
    missionCtx.part5.value = 0xdddd;

    EXPECT_EQ(missionCtx.GetStatus(), 0b0111010101010101);
    EXPECT_EQ(missionCtx.GetCurrentIns(), 0b1110111001100110);
}

TEST_F(CcuRaTest, test_ccu_loop_context)
{
    EXPECT_EQ(sizeof(CcuLoopContext), 64);

    CcuLoopContext loopCtx{};
    loopCtx.part9.value = 0x9999;
    loopCtx.part10.value = 0xaaaa;
    loopCtx.part11.value = 0xbbbb;
    loopCtx.part12.value = 0xcccc;
    loopCtx.part13.value = 0xdddd;
    loopCtx.part14.value = 0xeeee;

    EXPECT_EQ(loopCtx.GetCurrentIns(), 0b1010101010100110);
    EXPECT_EQ(loopCtx.GetCurrentCnt(), 0b0000110111011101);
    EXPECT_EQ(loopCtx.GetAddrStride(), 0b00110011001011101110111011101010);
}

TEST_F(CcuRaTest, test_loop_xm)
{
    EXPECT_EQ(sizeof(LoopXm), 8);

    LoopXm loopXm{};
    loopXm.value = 0xaaaabbbbccccdddd;

    EXPECT_EQ(loopXm.loopCnt, 0b1110111011101);
    EXPECT_EQ(loopXm.gsaStride, 0b11011101110111100110011001100110);
    EXPECT_EQ(loopXm.loopCtxId, 0b01010101);
}

TEST_F(CcuRaTest, test_loop_group_xn)
{
    EXPECT_EQ(sizeof(LoopGroupXn), 8);

    LoopGroupXn loopGroupXn{};
    loopGroupXn.value = 0xaaaabbbbccccdddd;

    EXPECT_EQ(loopGroupXn.loopInsCnt, 0b1011101);
    EXPECT_EQ(loopGroupXn.expandOffset, 0b0101010);
    EXPECT_EQ(loopGroupXn.expandCnt, 0b1010101);
}

TEST_F(CcuRaTest, test_loop_group_xm)
{
    EXPECT_EQ(sizeof(LoopGroupXm), 8);

    LoopGroupXm loopGroupXm{};
    loopGroupXm.value = 0xaaaabbbbccccdddd;

    EXPECT_EQ(loopGroupXm.ckOffset, 0b0111011101);
    EXPECT_EQ(loopGroupXm.msOffset, 0b01100110111);
    EXPECT_EQ(loopGroupXm.gsaOffset, 0b01010101110111011101111001100110);
}

void MockHrtRaTlvRequestForCustomChannelXn(void* tlvHandle, u32 msgType, void* customIn, void* customOut)
{
    uint64_t mockXnVal = 0xaaaabbbbccccdddd;
    CustomChannelInfoOut* mockOutBuff = (CustomChannelInfoOut*)customOut;
    memcpy_s(mockOutBuff->data.dataInfo.dataArray, sizeof(uint64_t), &mockXnVal, sizeof(uint64_t));
}

TEST_F(CcuRaTest, test_get_ccu_xn_value)
{
    MOCKER(HrtRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockHrtRaTlvRequestForCustomChannelXn));

    EXPECT_EQ(CcuErrorHandler::GetCcuXnValue(0, 0, 0), 0xaaaabbbbccccdddd);
}

void MockHrtRaTlvRequestForCustomChannelGSA(void* tlvHandle, u32 msgType, void* customIn, void* customOut)
{
    uint64_t mockGSAVal = 0x1111222233334444;
    CustomChannelInfoOut* mockOutBuff = (CustomChannelInfoOut*)customOut;
    memcpy_s(mockOutBuff->data.dataInfo.dataArray, sizeof(uint64_t), &mockGSAVal, sizeof(uint64_t));
}

TEST_F(CcuRaTest, test_get_ccu_gsa_value)
{
    MOCKER(HrtRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockHrtRaTlvRequestForCustomChannelGSA));

    EXPECT_EQ(CcuErrorHandler::GetCcuGSAValue(0, 0, 0), 0x1111222233334444);
}

void MockHrtRaTlvRequestForCustomChannelCKE(void* tlvHandle, u32 msgType, void* customIn, void* customOut)
{
    uint64_t mockCKEVal = 0x000000000000ffff;
    CustomChannelInfoOut* mockOutBuff = (CustomChannelInfoOut*)customOut;
    memcpy_s(mockOutBuff->data.dataInfo.dataArray, sizeof(uint64_t), &mockCKEVal, sizeof(uint64_t));
}

TEST_F(CcuRaTest, test_get_ccu_cke_value)
{
    MOCKER(HrtRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockHrtRaTlvRequestForCustomChannelCKE));

    EXPECT_EQ(CcuErrorHandler::GetCcuCKEValue(0, 0, 0), 0xffff);
}

void MockHrtRaTlvRequestForCustomChannelMissionContext(void* tlvHandle, u32 msgType, void* customIn, void* customOut)
{
    CcuMissionContext missionCtx{};
    missionCtx.part2.value = 0xaaaa;
    missionCtx.part3.value = 0xbbbb;
    missionCtx.part4.value = 0xcccc;
    missionCtx.part5.value = 0xdddd;
    CustomChannelInfoOut* mockOutBuff = (CustomChannelInfoOut*)customOut;
    memcpy_s(mockOutBuff->data.dataInfo.dataArray, sizeof(CcuMissionContext), &missionCtx, sizeof(CcuMissionContext));
}

TEST_F(CcuRaTest, test_get_ccu_mission_context)
{
    MOCKER(HrtRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockHrtRaTlvRequestForCustomChannelMissionContext));

    auto missionCtx = CcuErrorHandler::GetCcuMissionContext(0, 0, 0);
    EXPECT_EQ(missionCtx.GetStatus(), 0b0111010101010101);
    EXPECT_EQ(missionCtx.GetCurrentIns(), 0b1110111001100110);
}

void MockHrtRaTlvRequestForCustomChannelLoopContext(void* tlvHandle, u32 msgType, void* customIn, void* customOut)
{
    CcuLoopContext loopCtx{};
    loopCtx.part9.value = 0x9999;
    loopCtx.part10.value = 0xaaaa;
    loopCtx.part11.value = 0xbbbb;
    loopCtx.part12.value = 0xcccc;
    loopCtx.part13.value = 0xdddd;
    loopCtx.part14.value = 0xeeee;
    CustomChannelInfoOut* mockOutBuff = (CustomChannelInfoOut*)customOut;
    memcpy_s(mockOutBuff->data.dataInfo.dataArray, sizeof(CcuLoopContext), &loopCtx, sizeof(CcuLoopContext));
}

TEST_F(CcuRaTest, test_get_ccu_loop_context)
{
    MOCKER(HrtRaTlvRequestForCustomChannel)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockHrtRaTlvRequestForCustomChannelLoopContext));

    auto loopCtx = CcuErrorHandler::GetCcuLoopContext(0, 0, 0);
    EXPECT_EQ(loopCtx.GetCurrentIns(), 0b1010101010100110);
    EXPECT_EQ(loopCtx.GetCurrentCnt(), 0b0000110111011101);
    EXPECT_EQ(loopCtx.GetAddrStride(), 0b00110011001011101110111011101010);
}
