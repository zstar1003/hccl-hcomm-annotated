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
#include "tp_manager.h"
#include "hccp.h"
#include "orion_adapter_rts.h"
#include "env_config/env_config_v2.h"

using namespace Hccl;

class TpManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TpManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "TpManagerTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDevicePhyIdByUserDevId).defaults().will(returnValue(static_cast<DevId>(0)));
        MOCKER(RaGetInterfaceVersion).defaults().will(returnValue(static_cast<s32>(-1)));
        TpManager::GetInstance(0).Init();
        std::cout << "A Test case in TpManagerTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in TpManagerTest TearDown" << std::endl;
    }
};

TEST_F(TpManagerTest, tp_manager_get_infos_success)
{
    HcclResult result;
    int32_t devLogicId = 0;

    IpAddress locAddr("3.0.0.1");
    IpAddress rmtAddr("3.0.0.2");
    TpProtocol protocol = TpProtocol::TP;
    TpInfo tpInfo;

    result = TpManager::GetInstance(devLogicId).GetTpInfo({locAddr, rmtAddr, protocol}, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);

    result = TpManager::GetInstance(devLogicId).GetTpInfo({locAddr, rmtAddr, protocol}, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(TpManagerTest, tp_manager_redo_get_infos_success)
{
    HcclResult result;
    int32_t devLogicId = 0;

    IpAddress locAddr("3.0.0.1");
    IpAddress rmtAddr("3.0.0.2");
    TpProtocol protocol = TpProtocol::TP;
    TpInfo tpInfo;

    result = TpManager::GetInstance(devLogicId).GetTpInfo({locAddr, rmtAddr, protocol}, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);

    result = TpManager::GetInstance(devLogicId).ReleaseTpInfo({locAddr, rmtAddr, protocol}, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(TpManagerTest, tp_manager_get_infos_throw)
{
    ReqHandleResult sockEAgain = ReqHandleResult::SOCK_E_AGAIN;
    MOCKER(HrtRaGetAsyncReqResult).stubs().will(returnValue(sockEAgain));

    HcclResult result;
    int32_t devLogicId = 0;

    IpAddress locAddr("4.0.0.1");
    IpAddress rmtAddr("4.0.0.2");
    TpProtocol protocol = TpProtocol::TP;
    TpInfo tpInfo;

    result = TpManager::GetInstance(devLogicId).GetTpInfo({locAddr, rmtAddr, protocol}, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);

    EXPECT_THROW(TpManager::GetInstance(devLogicId).GetTpInfo({locAddr, rmtAddr, protocol}, tpInfo), InternalException);
}

TEST_F(TpManagerTest, tp_manager_get_infos_not_found)
{
    uint32_t errNum = 0;
    RequestHandle reqHandle = 0x12345678;
    MOCKER(RaUbGetTpInfoAsync)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(errNum))
        .will(returnValue(reqHandle));
    HcclResult result;
    int32_t devLogicId = 0;

    IpAddress locAddr("5.0.0.1");
    IpAddress rmtAddr("5.0.0.2");
    TpProtocol protocol = TpProtocol::TP;
    TpInfo tpInfo;

    result = TpManager::GetInstance(devLogicId).GetTpInfo({locAddr, rmtAddr, protocol}, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);

    result = TpManager::GetInstance(devLogicId).GetTpInfo({locAddr, rmtAddr, protocol}, tpInfo);
    EXPECT_EQ(result, HCCL_E_NOT_FOUND);
}

TEST_F(TpManagerTest, tp_manager_redo_get_infos_not_found)
{ // 新版本查询失败后，下一次调用还会尝试寻找tp资源，不会直接按记录报错
    uint32_t errNum = 0;
    RequestHandle reqHandle = 0x12345678;
    MOCKER(RaUbGetTpInfoAsync)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(errNum))
        .will(returnValue(reqHandle));
    HcclResult result;
    int32_t devLogicId = 0;

    IpAddress locAddr("5.0.0.1");
    IpAddress rmtAddr("5.0.0.2");
    TpProtocol protocol = TpProtocol::TP;
    TpInfo tpInfo;

    result = TpManager::GetInstance(devLogicId).GetTpInfo({locAddr, rmtAddr, protocol}, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);

    result = TpManager::GetInstance(devLogicId).GetTpInfo({locAddr, rmtAddr, protocol}, tpInfo);
    EXPECT_EQ(result, HCCL_E_NOT_FOUND);
}

TEST_F(TpManagerTest, St_ReleaseTpInfo_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    HcclResult result;
    int32_t devLogicId = 1;

    IpAddress locAddr("6.0.0.1");
    IpAddress rmtAddr("6.0.0.2");
    TpProtocol protocol = TpProtocol::TP;
    TpInfo tpInfo;

    result = TpManager::GetInstance(devLogicId).ReleaseTpInfo({locAddr, rmtAddr, protocol}, tpInfo);
    EXPECT_EQ(result, HCCL_E_NOT_FOUND);

    result = TpManager::GetInstance(devLogicId).GetTpInfo({locAddr, rmtAddr, protocol}, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);

    TpInfo fakeTpInfo;
    result = TpManager::GetInstance(devLogicId).ReleaseTpInfo({locAddr, rmtAddr, protocol}, fakeTpInfo);
    EXPECT_EQ(result, HCCL_E_NOT_FOUND);

    result = TpManager::GetInstance(devLogicId).GetTpInfo({locAddr, rmtAddr, protocol}, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);

    result = TpManager::GetInstance(devLogicId).ReleaseTpInfo({locAddr, rmtAddr, protocol}, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(TpManagerTest, St_TaHwValueToMs_InvalidGear_ReturnsDefault)
{
    EXPECT_EQ(TpManager::TaHwValueToMs(32), 8000);
    EXPECT_EQ(TpManager::TaHwValueToMs(255), 8000);
}

TEST_F(TpManagerTest, St_FindMinTaHwValue_AllTimeouts_ReturnsCorrectHwValue)
{
    EXPECT_EQ(TpManager::FindMinTaHwValue(100), 0);
    EXPECT_EQ(TpManager::FindMinTaHwValue(700), 8);
    EXPECT_EQ(TpManager::FindMinTaHwValue(5000), 16);
    EXPECT_EQ(TpManager::FindMinTaHwValue(20000), 24);
}

TEST_F(TpManagerTest, St_FindMinTaHwValue_BoundaryValues_ReturnsCorrectHwValue)
{
    EXPECT_EQ(TpManager::FindMinTaHwValue(512), 8);
    EXPECT_EQ(TpManager::FindMinTaHwValue(4000), 16);
    EXPECT_EQ(TpManager::FindMinTaHwValue(8000), 24);
}

TEST_F(TpManagerTest, St_FindMinTaHwValue_ExtremeBoundaries_ReturnsCorrectHwValue)
{
    EXPECT_EQ(TpManager::FindMinTaHwValue(0), 0);
    EXPECT_EQ(TpManager::FindMinTaHwValue(1), 0);
    EXPECT_EQ(TpManager::FindMinTaHwValue(511), 0);
    EXPECT_EQ(TpManager::FindMinTaHwValue(513), 8);
}

TEST_F(TpManagerTest, St_GetTpTotalTimeout_ValidAtGear_ReturnsCorrectTimeout)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 0;
    tpAttrInfo.tpAttr.retryTimesInit = 0;

    uint32_t tpTimeOutMs = 0;
    EXPECT_EQ(TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs), HCCL_SUCCESS);
    EXPECT_EQ(tpTimeOutMs, 16);

    tpAttrInfo.tpAttr.at = 1;
    tpAttrInfo.tpAttr.retryTimesInit = 0;
    EXPECT_EQ(TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs), HCCL_SUCCESS);
    EXPECT_EQ(tpTimeOutMs, 128);

    tpAttrInfo.tpAttr.at = 2;
    tpAttrInfo.tpAttr.retryTimesInit = 0;
    EXPECT_EQ(TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs), HCCL_SUCCESS);
    EXPECT_EQ(tpTimeOutMs, 1000);

    tpAttrInfo.tpAttr.at = 3;
    tpAttrInfo.tpAttr.retryTimesInit = 0;
    EXPECT_EQ(TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs), HCCL_SUCCESS);
    EXPECT_EQ(tpTimeOutMs, 4000);
}

TEST_F(TpManagerTest, St_GetTpTotalTimeout_InvalidAtGear_UsesDefault)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 5;
    tpAttrInfo.tpAttr.retryTimesInit = 0;

    uint32_t tpTimeOutMs = 0;
    EXPECT_EQ(TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs), HCCL_SUCCESS);
    EXPECT_EQ(tpTimeOutMs, 1000);
}

TEST_F(TpManagerTest, St_GetTpTotalTimeout_WithRetryTimes_ReturnsCorrectTimeout)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 2;
    tpAttrInfo.tpAttr.retryTimesInit = 3;

    uint32_t tpTimeOutMs = 0;
    EXPECT_EQ(TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs), HCCL_SUCCESS);
    EXPECT_EQ(tpTimeOutMs, 4000);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Tp_DefaultGeTpTimeout_Expect_Default16)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 0;
    tpAttrInfo.tpAttr.retryTimesInit = 0;
    uint32_t tpTimeOutMs = 0;
    (void)TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs), 16U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Tp_Boundary_8000msEqual_Expect_Upgrade24)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 2;
    tpAttrInfo.tpAttr.retryTimesInit = 7;
    uint32_t tpTimeOutMs = 0;
    (void)TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs), 24U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Tp_DefaultLessTpTimeout_Expect_Upgrade24)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 3;
    tpAttrInfo.tpAttr.retryTimesInit = 2;
    uint32_t tpTimeOutMs = 0;
    (void)TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs), 24U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Tp_InvalidAtGear_Expect_Default16)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 5;
    tpAttrInfo.tpAttr.retryTimesInit = 0;
    uint32_t tpTimeOutMs = 0;
    (void)TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs), 16U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Ctp_DefaultGeTpTimeout_Expect_Default8)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 0;
    tpAttrInfo.tpAttr.retryTimesInit = 0;
    uint32_t tpTimeOutMs = 0;
    (void)TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::CTP, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs), 8U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Ctp_Boundary_4000msEqual_Expect_Default8)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 1;
    tpAttrInfo.tpAttr.retryTimesInit = 30;
    uint32_t tpTimeOutMs = 0;
    (void)TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::CTP, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs), 8U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Ctp_DefaultLessTpTimeout_Expect_Default8)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 3;
    tpAttrInfo.tpAttr.retryTimesInit = 1;
    uint32_t tpTimeOutMs = 0;
    (void)TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    // 归一后 CTP 协议跳过与 TP 总超时的比较，直接使用默认值 8
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::CTP, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs), 8U);
}

// ==================== CalcTaTimeout (Jetty 重载) 系统测试 ====================

TEST_F(TpManagerTest, St_CalcTaTimeout_Jetty_Ctp_TaTimeOutNotSet_Expect_Default8)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::CTP, TpManager::TA_TIMEOUT_NOT_SET, 0U), 8U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Jetty_Ctp_TaTimeOutSet_Expect_InputValue)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::CTP, 16U, 0U), 16U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Jetty_Tp_TaTimeOutNotSet_EnvGeTp_Expect_Default16)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, TpManager::TA_TIMEOUT_NOT_SET, 4000U), 16U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Jetty_Tp_TaTimeOutNotSet_EnvLessTp_Expect_Upgrade24)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, TpManager::TA_TIMEOUT_NOT_SET, 10000U), 24U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Jetty_Tp_TaTimeOutSet_EnvGeTp_Expect_InputValue)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, 24U, 4000U), 24U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Jetty_Tp_TaTimeOutSet_EnvLessTp_Expect_Upgrade)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, 0U, 10000U), 24U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Jetty_Uboe_TaTimeOutNotSet_EnvLessTp_Expect_Upgrade)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::UBOE, TpManager::TA_TIMEOUT_NOT_SET, 10000U), 24U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Jetty_Ubg_TaTimeOutSet_EnvGeTp_Expect_InputValue)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::UBG, 24U, 4000U), 24U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Jetty_Ctp_Boundary_4000msEqual_Expect_Default8)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::CTP, TpManager::TA_TIMEOUT_NOT_SET, 4000U), 8U);
}

TEST_F(TpManagerTest, St_CalcTaTimeout_Jetty_Tp_Boundary_8000msEqual_Expect_Upgrade24)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, TpManager::TA_TIMEOUT_NOT_SET, 8000U), 24U);
}
