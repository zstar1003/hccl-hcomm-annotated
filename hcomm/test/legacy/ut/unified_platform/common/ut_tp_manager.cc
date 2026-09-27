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
// 本文件经 hcomm_res_mgr.h -> resources/endpoints/mgr/endpoint_mgr.h -> endpoint.h -> reged_mem_mgr.h
// 引入 base_comm 头, 其 EXCEPTION_CATCH 宏要求 pub_inc 版 log.h(仅 catch std::exception);
// 若 ascend950/common 版 log.h 先被包含, 其同名宏引用 namespace Hccl 的 HcclException,
// 在 namespace hcomm 中不可见导致编译失败。两版 log.h 的 include guard 同为 LOG_H,
// 先包含者生效, 故在所有业务头之前显式包含 pub_inc 版 log.h。
#include "../../../../../src/legacy/ascend910/pub_inc/log.h"
#include "tp_manager.h"
#include "hccp.h"
#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"
#include "internal_exception.h"
#include "hcomm_res_mgr.h"
#include "env_config/env_config_v2.h"
#include <cstdlib>
#include <string>

using namespace Hccl;

namespace {

class UbTimeoutEnvGuard {
public:
    explicit UbTimeoutEnvGuard(const char* value)
    {
        const char* saved = std::getenv("HCOMM_TA_RTP_UB_TIMEOUT");
        if (saved != nullptr) {
            savedValue_ = saved;
            hadValue_ = true;
        }
        if (value != nullptr) {
            (void)setenv("HCOMM_TA_RTP_UB_TIMEOUT", value, 1);
        } else {
            (void)unsetenv("HCOMM_TA_RTP_UB_TIMEOUT");
        }
        hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    }

    ~UbTimeoutEnvGuard()
    {
        if (hadValue_) {
            (void)setenv("HCOMM_TA_RTP_UB_TIMEOUT", savedValue_.c_str(), 1);
        } else {
            (void)unsetenv("HCOMM_TA_RTP_UB_TIMEOUT");
        }
        hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    }

private:
    bool hadValue_{false};
    std::string savedValue_;
};

void MockDeviceTpAttrAsyncSupport()
{
    u32 tpAttrVersion = 2U;
    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&tpAttrVersion, sizeof(tpAttrVersion)))
        .will(returnValue(0));
    MOCKER(HrtRaSetTpAttrAsync).stubs().will(returnValue(HCCL_SUCCESS));
}

RequestHandle
StubRaUbGetTpInfoAsyncEight(const RdmaHandle, const RaUbGetTpInfoParam&, vector<char_t>& out, uint32_t& num)
{
    num = 1U;
    out.resize(static_cast<size_t>(num) * sizeof(HccpTpInfo));
    auto* list = reinterpret_cast<HccpTpInfo*>(out.data());
    list[0].tpHandle = 0x300ULL;
    return static_cast<RequestHandle>(0x12345678ULL);
}

int StubRaGetTpAttrAsyncUboeSl789Legacy(
    void* ctxHandle, uint64_t tpHandle, uint32_t* attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    static char kLegacyUboeAttrReq{};
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    if (attr != nullptr) {
        (void)memset(attr, 0, sizeof(struct TpAttr));
        attr->slBitmap = (1U << 7U) | (1U << 8U) | (1U << 9U);
        attr->dscpConfigMode = 0U;
        attr->dscp = 12U;
    }
    if (reqHandle != nullptr) {
        *reqHandle = &kLegacyUboeAttrReq;
    }
    return 0;
}

int StubRaGetHccnCfgDscpLegacy(struct RaInfo* info, enum HccnCfgKey key, char* value, unsigned int* valueLen)
{
    (void)info;
    (void)key;
    if (value == nullptr || valueLen == nullptr) {
        return -1;
    }
    const char* cfg = "0:10,1:20,2:30";
    const unsigned int len = static_cast<unsigned int>(strlen(cfg));
    if (*valueLen < len) {
        return -1;
    }
    (void)memcpy(value, cfg, len);
    *valueLen = len;
    return 0;
}

RequestHandle StubRaUbGetTpInfoAsyncTwo(const RdmaHandle, const RaUbGetTpInfoParam&, vector<char_t>& out, uint32_t& num)
{
    num = 2U;
    out.resize(static_cast<size_t>(num) * sizeof(HccpTpInfo));
    auto* list = reinterpret_cast<HccpTpInfo*>(out.data());
    list[0].tpHandle = 0x500ULL;
    list[1].tpHandle = 0x501ULL;
    return static_cast<RequestHandle>(0x12345679ULL);
}

int StubRaGetTpAttrAsyncSl01Legacy(
    void* ctxHandle, uint64_t tpHandle, uint32_t* attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    static char kLegacySl01Req{};
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    if (attr != nullptr) {
        (void)memset(attr, 0, sizeof(struct TpAttr));
        attr->slBitmap = (1U << 0U) | (1U << 1U);
        attr->dscpConfigMode = 1U;
    }
    if (reqHandle != nullptr) {
        *reqHandle = &kLegacySl01Req;
    }
    return 0;
}

} // namespace

class TpManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TpManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "TpManagerTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDevicePhyIdByUserDevId).defaults().will(returnValue(static_cast<DevId>(0)));
        void* rdmaHandle = (void*)0x200;
        MOCKER(HrtRaUbCtxInit).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));
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

TEST_F(TpManagerTest, Ut_ReleaseTpInfo_When_InputValue_Expect_Return_HCCL_SUCCESS)
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

TEST_F(TpManagerTest, Ut_TaHwValueToMs_AllGears_ReturnsCorrectTimeout)
{
    EXPECT_EQ(TpManager::TaHwValueToMs(0), 512);
    EXPECT_EQ(TpManager::TaHwValueToMs(8), 4000);
    EXPECT_EQ(TpManager::TaHwValueToMs(16), 8000);
    EXPECT_EQ(TpManager::TaHwValueToMs(24), 32000);

    EXPECT_EQ(TpManager::TaHwValueToMs(7), 512);
    EXPECT_EQ(TpManager::TaHwValueToMs(15), 4000);
    EXPECT_EQ(TpManager::TaHwValueToMs(23), 8000);
    EXPECT_EQ(TpManager::TaHwValueToMs(31), 32000);
}

TEST_F(TpManagerTest, Ut_TaHwValueToMs_InvalidGear_ReturnsDefault)
{
    EXPECT_EQ(TpManager::TaHwValueToMs(32), 8000);
    EXPECT_EQ(TpManager::TaHwValueToMs(255), 8000);
}

TEST_F(TpManagerTest, Ut_FindMinTaHwValue_AllTimeouts_ReturnsCorrectHwValue)
{
    EXPECT_EQ(TpManager::FindMinTaHwValue(100), 0);
    EXPECT_EQ(TpManager::FindMinTaHwValue(700), 8);
    EXPECT_EQ(TpManager::FindMinTaHwValue(5000), 16);
    EXPECT_EQ(TpManager::FindMinTaHwValue(20000), 24);
}

TEST_F(TpManagerTest, Ut_FindMinTaHwValue_BoundaryValues_ReturnsCorrectHwValue)
{
    EXPECT_EQ(TpManager::FindMinTaHwValue(512), 8);
    EXPECT_EQ(TpManager::FindMinTaHwValue(4000), 16);
    EXPECT_EQ(TpManager::FindMinTaHwValue(8000), 24);
}

TEST_F(TpManagerTest, Ut_FindMinTaHwValue_ExtremeBoundaries_ReturnsCorrectHwValue)
{
    EXPECT_EQ(TpManager::FindMinTaHwValue(0), 0);   // 0ms → GEAR0(0)
    EXPECT_EQ(TpManager::FindMinTaHwValue(1), 0);   // 1ms < 512 → GEAR0(0)
    EXPECT_EQ(TpManager::FindMinTaHwValue(511), 0); // 511ms < 512 → GEAR0(0)
    EXPECT_EQ(TpManager::FindMinTaHwValue(513), 8); // 513ms → GEAR1(8)
}

TEST_F(TpManagerTest, Ut_GetTpTotalTimeout_ValidAtGear_ReturnsCorrectTimeout)
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

TEST_F(TpManagerTest, Ut_GetTpTotalTimeout_InvalidAtGear_UsesDefault)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 5;
    tpAttrInfo.tpAttr.retryTimesInit = 0;

    uint32_t tpTimeOutMs = 0;
    EXPECT_EQ(TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs), HCCL_SUCCESS);
    EXPECT_EQ(tpTimeOutMs, 1000);
}

TEST_F(TpManagerTest, Ut_GetTpTotalTimeout_WithRetryTimes_ReturnsCorrectTimeout)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 2;
    tpAttrInfo.tpAttr.retryTimesInit = 3;

    uint32_t tpTimeOutMs = 0;
    EXPECT_EQ(TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs), HCCL_SUCCESS);
    EXPECT_EQ(tpTimeOutMs, 4000);
}

TEST_F(TpManagerTest, Ut_GetTpTotalTimeout_MaxRetryTimes_ReturnsCorrectTimeout)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 3;
    tpAttrInfo.tpAttr.retryTimesInit = 7;

    uint32_t tpTimeOutMs = 0;
    EXPECT_EQ(TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs), HCCL_SUCCESS);
    EXPECT_EQ(tpTimeOutMs, 32000);
}

TEST_F(TpManagerTest, tp_manager_device_qos_sl_mapping_success)
{
    MockDeviceTpAttrAsyncSupport();

    HcclResult result;
    const int32_t devLogicId = 0;
    IpAddress locAddr("8.0.0.1");
    IpAddress rmtAddr("8.0.0.2");
    const TpProtocol protocol = TpProtocol::TP;
    RaUbGetTpInfoParam param{locAddr, rmtAddr, protocol};
    param.qos = 5U;
    TpInfo tpInfo{};

    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);

    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);

    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_TRUE(tpInfo.hasMappedJettyPriority);
}

TEST_F(TpManagerTest, tp_manager_loop_first_tp_lowest_sl_success)
{
    MockDeviceTpAttrAsyncSupport();

    HcclResult result;
    const int32_t devLogicId = 0;
    IpAddress locAddr("8.0.0.3");
    IpAddress rmtAddr("8.0.0.4");
    const TpProtocol protocol = TpProtocol::TP;
    RaUbGetTpInfoParam param{locAddr, rmtAddr, protocol};
    param.loopFirstTpLowestSl = true;
    param.qos = 2U;
    TpInfo tpInfo{};

    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);

    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);

    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_TRUE(tpInfo.hasMappedJettyPriority);
}

TEST_F(TpManagerTest, tp_manager_qos_cache_by_key_success)
{
    HcclResult result;
    const int32_t devLogicId = 0;
    IpAddress locAddr("8.0.0.5");
    IpAddress rmtAddr("8.0.0.6");
    const TpProtocol protocol = TpProtocol::TP;

    RaUbGetTpInfoParam paramQos0{locAddr, rmtAddr, protocol};
    paramQos0.qos = 0U;
    TpInfo tpInfo0{};

    RaUbGetTpInfoParam paramQos7{locAddr, rmtAddr, protocol};
    paramQos7.qos = 7U;
    TpInfo tpInfo7{};

    result = TpManager::GetInstance(devLogicId).GetTpInfo(paramQos0, tpInfo0);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(paramQos0, tpInfo0);
    EXPECT_EQ(result, HCCL_SUCCESS);

    result = TpManager::GetInstance(devLogicId).GetTpInfo(paramQos7, tpInfo7);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(paramQos7, tpInfo7);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(TpManagerTest, tp_manager_device_uboe_eight_tp_qos7_success)
{
    MockDeviceTpAttrAsyncSupport();
    MOCKER(RaUbGetTpInfoAsync).stubs().will(invoke(StubRaUbGetTpInfoAsyncEight));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncUboeSl789Legacy));

    HcclResult result;
    const int32_t devLogicId = 0;
    IpAddress locAddr("9.0.0.1");
    IpAddress rmtAddr("9.0.0.2");
    RaUbGetTpInfoParam param{locAddr, rmtAddr, TpProtocol::UBOE};
    param.qos = 7U;
    TpInfo tpInfo{};

    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_TRUE(tpInfo.hasMappedJettyPriority);
    EXPECT_EQ(tpInfo.tpHandle, 0x300ULL);
    EXPECT_EQ(tpInfo.mappedJettyPriority, 7U);
}

TEST_F(TpManagerTest, tp_manager_release_tpinfo_qos_key_mismatch)
{
    HcclResult result;
    const int32_t devLogicId = 0;
    IpAddress locAddr("9.0.0.3");
    IpAddress rmtAddr("9.0.0.4");
    RaUbGetTpInfoParam paramQos0{locAddr, rmtAddr, TpProtocol::TP};
    paramQos0.qos = 0U;
    TpInfo tpInfo{};

    result = TpManager::GetInstance(devLogicId).GetTpInfo(paramQos0, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(paramQos0, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);

    RaUbGetTpInfoParam paramQos1{locAddr, rmtAddr, TpProtocol::TP};
    paramQos1.qos = 1U;
    result = TpManager::GetInstance(devLogicId).ReleaseTpInfo(paramQos1, tpInfo);
    EXPECT_EQ(result, HCCL_E_NOT_FOUND);
}

TEST_F(TpManagerTest, tp_manager_device_uboe_dscp_from_hccn_success)
{
    MockDeviceTpAttrAsyncSupport();
    MOCKER(RaUbGetTpInfoAsync).stubs().will(invoke(StubRaUbGetTpInfoAsyncEight));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncUboeSl789Legacy));
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgDscpLegacy));

    const int32_t devLogicId = 0;
    IpAddress locAddr("9.0.0.5");
    IpAddress rmtAddr("9.0.0.6");
    RaUbGetTpInfoParam param{locAddr, rmtAddr, TpProtocol::UBOE};
    param.qos = 2U;
    TpInfo tpInfo{};

    HcclResult result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_TRUE(tpInfo.hasMappedJettyPriority);
    EXPECT_EQ(tpInfo.tpHandle, 0x300ULL);
}

TEST_F(TpManagerTest, tp_manager_device_ctp_with_qos_success)
{
    const int32_t devLogicId = 0;
    IpAddress locAddr("9.0.0.7");
    IpAddress rmtAddr("9.0.0.8");
    RaUbGetTpInfoParam param{locAddr, rmtAddr, TpProtocol::CTP};
    param.qos = 4U;
    TpInfo tpInfo{};

    HcclResult result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_NE(tpInfo.tpHandle, 0U);
}

TEST_F(TpManagerTest, tp_manager_device_two_tp_qos5_success)
{
    MockDeviceTpAttrAsyncSupport();
    MOCKER(RaUbGetTpInfoAsync).stubs().will(invoke(StubRaUbGetTpInfoAsyncTwo));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncSl01Legacy));

    const int32_t devLogicId = 0;
    IpAddress locAddr("9.0.0.9");
    IpAddress rmtAddr("9.0.0.10");
    RaUbGetTpInfoParam param{locAddr, rmtAddr, TpProtocol::TP};
    param.qos = 5U;
    TpInfo tpInfo{};

    HcclResult result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_TRUE(tpInfo.hasMappedJettyPriority);
}

TEST_F(TpManagerTest, tp_manager_get_tpattr_and_release_success)
{
    const int32_t devLogicId = 0;
    IpAddress locAddr("9.0.0.11");
    IpAddress rmtAddr("9.0.0.12");
    RaUbGetTpInfoParam tpParam{locAddr, rmtAddr, TpProtocol::TP};
    tpParam.qos = 1U;
    TpInfo tpInfo{};

    HcclResult result = TpManager::GetInstance(devLogicId).GetTpInfo(tpParam, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(tpParam, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);

    GetTpAttrParam attrParam(tpInfo.tpHandle, (1U << 10U));
    TpAttrInfo tpAttrInfo{};
    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x200);
    result = TpManager::GetInstance(devLogicId).GetTpAttr(attrParam, tpAttrInfo, rdmaHandle);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpAttr(attrParam, tpAttrInfo, rdmaHandle);
    EXPECT_EQ(result, HCCL_SUCCESS);
    result = TpManager::GetInstance(devLogicId).GetTpAttr(attrParam, tpAttrInfo, rdmaHandle);
    EXPECT_EQ(result, HCCL_SUCCESS);
    result = TpManager::GetInstance(devLogicId).ReleaseTpAttr(tpInfo.tpHandle, tpAttrInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(TpManagerTest, tp_manager_release_tpinfo_usecnt_decrement)
{
    const int32_t devLogicId = 0;
    IpAddress locAddr("9.0.0.13");
    IpAddress rmtAddr("9.0.0.14");
    RaUbGetTpInfoParam param{locAddr, rmtAddr, TpProtocol::TP};
    param.qos = 3U;
    TpInfo tpInfo{};

    HcclResult result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
    result = TpManager::GetInstance(devLogicId).ReleaseTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
    result = TpManager::GetInstance(devLogicId).ReleaseTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(TpManagerTest, tp_manager_release_tpinfo_handle_mismatch)
{
    const int32_t devLogicId = 0;
    IpAddress locAddr("9.0.0.15");
    IpAddress rmtAddr("9.0.0.16");
    RaUbGetTpInfoParam param{locAddr, rmtAddr, TpProtocol::TP};
    param.qos = 6U;
    TpInfo tpInfo{};

    HcclResult result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);

    TpInfo wrongTpInfo{};
    wrongTpInfo.tpHandle = tpInfo.tpHandle + 1U;
    result = TpManager::GetInstance(devLogicId).ReleaseTpInfo(param, wrongTpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
    result = TpManager::GetInstance(devLogicId).ReleaseTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Tp_DefaultGeTpTimeout_Expect_Default16)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 0U;
    tpAttrInfo.tpAttr.retryTimesInit = 0U;
    uint32_t tpTimeOutMs = 0;
    (void)TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs), 16U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Tp_Boundary_8000msEqual_Expect_Upgrade24)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 2U;
    tpAttrInfo.tpAttr.retryTimesInit = 7U;
    uint32_t tpTimeOutMs = 0;
    (void)TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs), 24U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Tp_DefaultLessTpTimeout_Expect_Upgrade24)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 3U;
    tpAttrInfo.tpAttr.retryTimesInit = 2U;
    uint32_t tpTimeOutMs = 0;
    (void)TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs), 24U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Tp_InvalidAtGear_Expect_Default16)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 5U;
    tpAttrInfo.tpAttr.retryTimesInit = 0U;
    uint32_t tpTimeOutMs = 0;
    (void)TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs), 16U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Ctp_DefaultGeTpTimeout_Expect_Default8)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 0U;
    tpAttrInfo.tpAttr.retryTimesInit = 0U;
    uint32_t tpTimeOutMs = 0;
    (void)TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::CTP, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs), 8U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Ctp_Boundary_4000msEqual_Expect_Default8)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 1U;
    tpAttrInfo.tpAttr.retryTimesInit = 30U;
    uint32_t tpTimeOutMs = 0;
    (void)TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::CTP, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs), 8U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Ctp_DefaultLessTpTimeout_Expect_Default8)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 3U;
    tpAttrInfo.tpAttr.retryTimesInit = 1U;
    uint32_t tpTimeOutMs = 0;
    (void)TpManager::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs);
    // 归一后 CTP 协议跳过与 TP 总超时的比较，直接使用默认值 8
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::CTP, TpManager::TA_TIMEOUT_NOT_SET, tpTimeOutMs), 8U);
}

TEST_F(TpManagerTest, tp_manager_loop_first_tp_lowest_sl_uboe_success)
{
    MockDeviceTpAttrAsyncSupport();
    MOCKER(RaUbGetTpInfoAsync).stubs().will(invoke(StubRaUbGetTpInfoAsyncEight));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncUboeSl789Legacy));

    const int32_t devLogicId = 0;
    IpAddress locAddr("9.0.0.17");
    IpAddress rmtAddr("9.0.0.18");
    RaUbGetTpInfoParam param{locAddr, rmtAddr, TpProtocol::UBOE};
    param.loopFirstTpLowestSl = true;
    param.qos = 4U;
    TpInfo tpInfo{};

    HcclResult result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_E_AGAIN);
    result = TpManager::GetInstance(devLogicId).GetTpInfo(param, tpInfo);
    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(tpInfo.tpHandle, 0x300ULL);
    EXPECT_EQ(tpInfo.mappedJettyPriority, 7U);
}

// ==================== CalcTaTimeout (Jetty 重载) 单元测试 ====================

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Jetty_Ctp_TaTimeOutNotSet_Expect_Default8)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::CTP, TpManager::TA_TIMEOUT_NOT_SET, 0U), 8U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Jetty_Ctp_TaTimeOutSet_Expect_InputValue)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::CTP, 16U, 0U), 16U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Jetty_Tp_TaTimeOutNotSet_EnvGeTp_Expect_Default16)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, TpManager::TA_TIMEOUT_NOT_SET, 4000U), 16U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Jetty_Tp_TaTimeOutNotSet_EnvLessTp_Expect_Upgrade24)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, TpManager::TA_TIMEOUT_NOT_SET, 10000U), 24U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Jetty_Tp_TaTimeOutSet_EnvGeTp_Expect_InputValue)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, 24U, 4000U), 24U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Jetty_Tp_TaTimeOutSet_EnvLessTp_Expect_Upgrade)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, 0U, 10000U), 24U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Jetty_Uboe_TaTimeOutNotSet_EnvLessTp_Expect_Upgrade)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::UBOE, TpManager::TA_TIMEOUT_NOT_SET, 10000U), 24U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Jetty_Ubg_TaTimeOutSet_EnvGeTp_Expect_InputValue)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::UB_RTP, 24U, 4000U), 24U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Jetty_Ctp_Boundary_4000msEqual_Expect_Default8)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::CTP, TpManager::TA_TIMEOUT_NOT_SET, 4000U), 8U);
}

TEST_F(TpManagerTest, Ut_CalcTaTimeout_Jetty_Tp_Boundary_8000msEqual_Expect_Upgrade24)
{
    EXPECT_EQ(TpManager::CalcTaTimeout(TpProtocol::TP, TpManager::TA_TIMEOUT_NOT_SET, 8000U), 24U);
}
