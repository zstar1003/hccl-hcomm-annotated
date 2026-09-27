/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <string>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "tp_mgr.h"
#include "tp_qos.h"
#include "orion_adpt_utils.h"
#include "orion_adapter_hccp.h"
#include "rdma_handle_manager.h"
#include "hccp.h"
#include "hcomm_res_mgr.h"
#include "env_config/env_config_v2.h"

using namespace hcomm;
using Hccl::HrtRaSetTpAttrAsync;

namespace {

class UbTimeoutEnvGuard {
public:
    explicit UbTimeoutEnvGuard(const char* value)
    {
        SaveEnv("HCOMM_TA_RTP_UB_TIMEOUT", savedUbTimeout_, hadUbTimeout_);

        if (value != nullptr) {
            (void)setenv("HCOMM_TA_RTP_UB_TIMEOUT", value, 1);
        } else {
            (void)unsetenv("HCOMM_TA_RTP_UB_TIMEOUT");
        }
        hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    }

    ~UbTimeoutEnvGuard()
    {
        RestoreEnv("HCOMM_TA_RTP_UB_TIMEOUT", savedUbTimeout_, hadUbTimeout_);
        hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    }

private:
    static void SaveEnv(const char* name, std::string& saved, bool& had)
    {
        const char* value = std::getenv(name);
        if (value != nullptr) {
            saved = value;
            had = true;
        } else {
            had = false;
        }
    }

    static void RestoreEnv(const char* name, const std::string& saved, bool had)
    {
        if (had) {
            (void)setenv(name, saved.c_str(), 1);
        } else {
            (void)unsetenv(name);
        }
    }

    bool hadUbTimeout_{false};
    std::string savedUbTimeout_;
};

void FillIpv4CommAddr(CommAddr& ca, const char* dotted)
{
    ca.type = COMM_ADDR_TYPE_IP_V4;
    EXPECT_EQ(inet_pton(AF_INET, dotted, &ca.addr), 1);
}

GetTpInfoParam MakeParam(const char* loc, const char* rmt, TpProtocol proto, uint32_t qos = 0U)
{
    GetTpInfoParam param;
    FillIpv4CommAddr(param.locAddr, loc);
    FillIpv4CommAddr(param.rmtAddr, rmt);
    param.tpProtocol = proto;
    param.qos = qos;
    return param;
}

HcclResult PollGetTpInfo(TpMgr& mgr, const GetTpInfoParam& param, TpInfo& tpInfo)
{
    for (int i = 0; i < 8; ++i) {
        const HcclResult ret = mgr.GetTpInfo(param, tpInfo);
        if (ret != HCCL_E_AGAIN) {
            return ret;
        }
    }
    return HCCL_E_AGAIN;
}

HcclResult PollGetTpAttr(TpMgr& mgr, const GetTpAttrParam& param, TpAttrInfo& tpAttrInfo, CtxHandle ctxHandle)
{
    for (int i = 0; i < 8; ++i) {
        const HcclResult ret = mgr.GetTpAttr(param, tpAttrInfo, ctxHandle);
        if (ret != HCCL_E_AGAIN) {
            return ret;
        }
    }
    return HCCL_E_AGAIN;
}

static unsigned int StubTpListWriteCnt(unsigned int* num, unsigned int desiredCnt)
{
    if (num == nullptr) {
        return 0U;
    }
    const unsigned int cap = *num;
    return (cap < desiredCnt) ? cap : desiredCnt;
}

int StubRaGetTpAttrAsyncUboe(
    void* ctxHandle, uint64_t tpHandle, uint32_t* attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    static char kUboeTpAttrReq{};
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    if (attr != nullptr) {
        (void)std::memset(attr, 0, sizeof(struct TpAttr));
        attr->slBitmap = 0x7U;
        attr->dscpConfigMode = 0U;
    }
    if (reqHandle != nullptr) {
        *reqHandle = &kUboeTpAttrReq;
    }
    return 0;
}

int StubRaGetTpInfoListAsyncUboeEight(
    void* ctxHandle, struct GetTpCfg* cfg, struct HccpTpInfo infoList[], unsigned int* num, void** reqHandle)
{
    static int kUboeTpListReq = 22378;
    (void)ctxHandle;
    (void)cfg;
    if (infoList != nullptr) {
        infoList[0].tpHandle = 0x100ULL;
    }
    if (num != nullptr) {
        *num = 1U;
    }
    if (reqHandle != nullptr) {
        *reqHandle = &kUboeTpListReq;
    }
    return 0;
}

int StubRaGetTpAttrAsyncUboeSl789(
    void* ctxHandle, uint64_t tpHandle, uint32_t* attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    static char kUboeTpAttrSl789Req{};
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    if (attr != nullptr) {
        (void)std::memset(attr, 0, sizeof(struct TpAttr));
        attr->slBitmap = (1U << 7U) | (1U << 8U) | (1U << 9U);
        attr->dscpConfigMode = 0U;
    }
    if (reqHandle != nullptr) {
        *reqHandle = &kUboeTpAttrSl789Req;
    }
    return 0;
}

int StubRaGetTpInfoListAsyncThree(
    void* ctxHandle, struct GetTpCfg* cfg, struct HccpTpInfo infoList[], unsigned int* num, void** reqHandle)
{
    static int kThreeTpListReq = 33445;
    (void)ctxHandle;
    (void)cfg;
    const unsigned int writeCnt = StubTpListWriteCnt(num, 3U);
    if (infoList != nullptr) {
        for (unsigned int i = 0; i < writeCnt; ++i) {
            infoList[i].tpHandle = 0x200ULL + static_cast<uint64_t>(i);
        }
    }
    if (num != nullptr) {
        *num = writeCnt;
    }
    if (reqHandle != nullptr) {
        *reqHandle = &kThreeTpListReq;
    }
    return 0;
}

int StubRaGetTpAttrAsyncSl123(
    void* ctxHandle, uint64_t tpHandle, uint32_t* attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    static char kSl123Req{};
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    if (attr != nullptr) {
        (void)std::memset(attr, 0, sizeof(struct TpAttr));
        attr->slBitmap = (1U << 1U) | (1U << 2U) | (1U << 3U);
        attr->dscpConfigMode = 1U;
    }
    if (reqHandle != nullptr) {
        *reqHandle = &kSl123Req;
    }
    return 0;
}

int StubRaGetTpAttrAsyncSingleSlBit(
    void* ctxHandle, uint64_t tpHandle, uint32_t* attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    static char kSingleSlReq{};
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    if (attr != nullptr) {
        (void)std::memset(attr, 0, sizeof(struct TpAttr));
        attr->slBitmap = (1U << 5U);
        attr->dscpConfigMode = 1U;
    }
    if (reqHandle != nullptr) {
        *reqHandle = &kSingleSlReq;
    }
    return 0;
}

int StubRaGetTpAttrAsyncTwoSlBits(
    void* ctxHandle, uint64_t tpHandle, uint32_t* attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    static char kTwoSlReq{};
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    if (attr != nullptr) {
        (void)std::memset(attr, 0, sizeof(struct TpAttr));
        attr->slBitmap = (1U << 3U) | (1U << 7U);
        attr->dscpConfigMode = 1U;
    }
    if (reqHandle != nullptr) {
        *reqHandle = &kTwoSlReq;
    }
    return 0;
}

int StubRaGetHccnCfgDscp(struct RaInfo* info, enum HccnCfgKey key, char* value, unsigned int* valueLen)
{
    (void)info;
    (void)key;
    if (value == nullptr || valueLen == nullptr) {
        return -1;
    }
    const char* cfg = "0:10,1:20,2:30";
    const unsigned int len = static_cast<unsigned int>(std::strlen(cfg));
    if (*valueLen < len) {
        return -1;
    }
    (void)std::memcpy(value, cfg, len);
    *valueLen = len;
    return 0;
}

int StubRaGetHccnCfgDscpKeyValue(struct RaInfo* info, enum HccnCfgKey key, char* value, unsigned int* valueLen)
{
    (void)info;
    (void)key;
    if (value == nullptr || valueLen == nullptr) {
        return -1;
    }
    const char* cfg = "2:30,5:40";
    const unsigned int len = static_cast<unsigned int>(std::strlen(cfg));
    if (*valueLen < len) {
        return -1;
    }
    (void)std::memcpy(value, cfg, len);
    *valueLen = len;
    return 0;
}

int StubRaGetHccnCfgInvalidFormat(struct RaInfo* info, enum HccnCfgKey key, char* value, unsigned int* valueLen)
{
    (void)info;
    (void)key;
    if (value == nullptr || valueLen == nullptr) {
        return -1;
    }
    const char* cfg = "0:10,abc";
    const unsigned int len = static_cast<unsigned int>(std::strlen(cfg));
    if (*valueLen < len) {
        return -1;
    }
    (void)std::memcpy(value, cfg, len);
    *valueLen = len;
    return 0;
}

int StubRaGetHccnCfgFail(struct RaInfo* info, enum HccnCfgKey key, char* value, unsigned int* valueLen)
{
    (void)info;
    (void)key;
    (void)value;
    (void)valueLen;
    return -1;
}

int StubRaGetHccnCfgEmpty(struct RaInfo* info, enum HccnCfgKey key, char* value, unsigned int* valueLen)
{
    (void)info;
    (void)key;
    if (value != nullptr && valueLen != nullptr && *valueLen > 0U) {
        value[0] = '\0';
    }
    if (valueLen != nullptr) {
        *valueLen = 0U;
    }
    return 0;
}

// 模拟 RsGetHccnCfg：valueLen = strlen + 1（含结尾 '\0'）
int StubRaGetHccnCfgDscpWithTrailingNull(struct RaInfo* info, enum HccnCfgKey key, char* value, unsigned int* valueLen)
{
    (void)info;
    (void)key;
    if (value == nullptr || valueLen == nullptr) {
        return -1;
    }
    const char* cfg = "4:33";
    const unsigned int strLen = static_cast<unsigned int>(std::strlen(cfg));
    const unsigned int copyLen = strLen + 1U; // 含 '\0'
    if (*valueLen < copyLen) {
        return -1;
    }
    (void)std::memcpy(value, cfg, copyLen);
    *valueLen = copyLen;
    return 0;
}

static uint8_t gCapturedSetTpAttrDscp = 0xFFU;
static uint32_t gCapturedSetTpAttrBitmap = 0U;

HcclResult StubHrtRaSetTpAttrAsyncCaptureDscp(
    Hccl::RdmaHandle handle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr& attr,
    Hccl::RequestHandle& reqHandle)
{
    (void)handle;
    (void)tpHandle;
    gCapturedSetTpAttrBitmap = attrBitmap;
    gCapturedSetTpAttrDscp = attr.dscp;
    reqHandle = 0;
    return HCCL_SUCCESS;
}

int StubRaGetTpAttrAsyncUboeDscpMode0(
    void* ctxHandle, uint64_t tpHandle, uint32_t* attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    static char kUboeDscpMode0Req{};
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    if (attr != nullptr) {
        (void)std::memset(attr, 0, sizeof(struct TpAttr));
        attr->slBitmap = 0x7U;
        attr->dscpConfigMode = 0U;
        attr->dscp = 10U;
    }
    if (reqHandle != nullptr) {
        *reqHandle = &kUboeDscpMode0Req;
    }
    return 0;
}

int StubRaGetTpInfoListAsyncTwo(
    void* ctxHandle, struct GetTpCfg* cfg, struct HccpTpInfo infoList[], unsigned int* num, void** reqHandle)
{
    static int kTwoTpListReq = 44556;
    (void)ctxHandle;
    (void)cfg;
    const unsigned int writeCnt = StubTpListWriteCnt(num, 2U);
    if (infoList != nullptr) {
        for (unsigned int i = 0; i < writeCnt; ++i) {
            infoList[i].tpHandle = 0x400ULL + static_cast<uint64_t>(i);
        }
    }
    if (num != nullptr) {
        *num = writeCnt;
    }
    if (reqHandle != nullptr) {
        *reqHandle = &kTwoTpListReq;
    }
    return 0;
}

int StubRaGetTpAttrAsyncSl01(
    void* ctxHandle, uint64_t tpHandle, uint32_t* attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    static char kSl01Req{};
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    if (attr != nullptr) {
        (void)std::memset(attr, 0, sizeof(struct TpAttr));
        attr->slBitmap = (1U << 0U) | (1U << 1U);
        attr->dscpConfigMode = 1U;
    }
    if (reqHandle != nullptr) {
        *reqHandle = &kSl01Req;
    }
    return 0;
}

} // namespace

class TpMgrTest : public testing::Test {
protected:
    void SetUp() override
    {
        MOCKER_CPP(&Hccl::RdmaHandleManager::GetByIp)
            .stubs()
            .will(returnValue(reinterpret_cast<Hccl::RdmaHandle>(static_cast<uintptr_t>(0x12345678U))));
    }

    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Rtp_WithQos_Expect_Success)
{
    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.1.1", "10.10.1.2", TpProtocol::RTP, 5U);
    TpInfo tpInfo{};
    EXPECT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_NE(tpInfo.tpHandle, 0U);
    EXPECT_TRUE(tpInfo.hasMappedJettyPriority);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_CacheHit_Expect_Success)
{
    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.2.1", "10.10.2.2", TpProtocol::RTP, 3U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_EQ(mgr.GetTpInfo(param, tpInfo), HCCL_SUCCESS);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_LoopFirstTpLowestSl_Expect_Success)
{
    TpMgr& mgr = TpMgr::GetInstance(0);
    GetTpInfoParam param = MakeParam("10.10.3.1", "10.10.3.2", TpProtocol::RTP, 6U);
    param.loopFirstTpLowestSl = true;
    TpInfo tpInfo{};
    EXPECT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_TRUE(tpInfo.hasMappedJettyPriority);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Ctp_Expect_Success)
{
    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.4.1", "10.10.4.2", TpProtocol::CTP, 2U);
    TpInfo tpInfo{};
    EXPECT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Uboe_WithQos_Expect_Success)
{
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncUboe));

    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.5.1", "10.10.5.2", TpProtocol::UBOE, 4U);
    TpInfo tpInfo{};
    EXPECT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Uboe_EightTp_DynamicSl012_Qos0_Expect_LastTp_Sl2)
{
    MOCKER(RaGetTpInfoListAsync).stubs().will(invoke(StubRaGetTpInfoListAsyncUboeEight));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncUboe));

    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.11.1", "10.10.11.2", TpProtocol::UBOE, 0U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_EQ(tpInfo.tpHandle, 0x100ULL);
    EXPECT_EQ(tpInfo.mappedJettyPriority, 2U);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Uboe_EightTp_Sl789_Qos0_Expect_LastTp_Sl9)
{
    MOCKER(RaGetTpInfoListAsync).stubs().will(invoke(StubRaGetTpInfoListAsyncUboeEight));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncUboeSl789));

    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.10.1", "10.10.10.2", TpProtocol::UBOE, 0U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_EQ(tpInfo.tpHandle, 0x100ULL);
    EXPECT_TRUE(tpInfo.hasMappedJettyPriority);
    EXPECT_EQ(tpInfo.mappedJettyPriority, 9U);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Uboe_EightTp_Sl789_Qos7_Expect_FirstTp_Sl7)
{
    MOCKER(RaGetTpInfoListAsync).stubs().will(invoke(StubRaGetTpInfoListAsyncUboeEight));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncUboeSl789));

    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.10.3", "10.10.10.4", TpProtocol::UBOE, 7U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_EQ(tpInfo.tpHandle, 0x100ULL);
    EXPECT_EQ(tpInfo.mappedJettyPriority, 7U);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_DifferentQosKeys_Expect_Both_Success)
{
    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam paramQos0 = MakeParam("10.10.6.1", "10.10.6.2", TpProtocol::RTP, 0U);
    const GetTpInfoParam paramQos7 = MakeParam("10.10.6.1", "10.10.6.2", TpProtocol::RTP, 7U);
    TpInfo tpInfo0{};
    TpInfo tpInfo7{};
    EXPECT_EQ(PollGetTpInfo(mgr, paramQos0, tpInfo0), HCCL_SUCCESS);
    EXPECT_EQ(PollGetTpInfo(mgr, paramQos7, tpInfo7), HCCL_SUCCESS);
    EXPECT_NE(tpInfo0.tpHandle, 0U);
    EXPECT_NE(tpInfo7.tpHandle, 0U);
}

TEST_F(TpMgrTest, Ut_TpMgr_ReleaseTpInfo_AfterGet_Expect_Success)
{
    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.7.1", "10.10.7.2", TpProtocol::RTP, 1U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_EQ(mgr.ReleaseTpInfo(param, tpInfo), HCCL_SUCCESS);
}

TEST_F(TpMgrTest, Ut_TpMgr_ReleaseTpInfo_NotFound_Expect_NotFound)
{
    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.8.1", "10.10.8.2", TpProtocol::RTP, 0U);
    TpInfo tpInfo{};
    EXPECT_EQ(mgr.ReleaseTpInfo(param, tpInfo), HCCL_E_NOT_FOUND);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_UnsupportedProtocol_Expect_NotSupport)
{
    TpMgr& mgr = TpMgr::GetInstance(0);
    GetTpInfoParam param = MakeParam("10.10.9.1", "10.10.9.2", TpProtocol::RTP, 0U);
    param.tpProtocol = TpProtocol::INVALID;
    TpInfo tpInfo{};
    EXPECT_EQ(mgr.GetTpInfo(param, tpInfo), HCCL_E_NOT_SUPPORT);
}

TEST_F(TpMgrTest, GetTpTotalTimeout_ValidAtGear_ReturnsCorrectTimeout)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 0;
    tpAttrInfo.tpAttr.retryTimesInit = 0;

    uint32_t tpTimeOutMs = 0;
    EXPECT_EQ(TpMgr::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs), HCCL_SUCCESS);
    EXPECT_EQ(tpTimeOutMs, 16U);

    tpAttrInfo.tpAttr.at = 3;
    tpAttrInfo.tpAttr.retryTimesInit = 0;
    EXPECT_EQ(TpMgr::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs), HCCL_SUCCESS);
    EXPECT_EQ(tpTimeOutMs, 4000U);
}

TEST_F(TpMgrTest, GetTpTotalTimeout_InvalidAtGear_UsesDefault)
{
    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 5;
    tpAttrInfo.tpAttr.retryTimesInit = 0;

    uint32_t tpTimeOutMs = 0;
    EXPECT_EQ(TpMgr::GetTpTotalTimeout(tpAttrInfo, tpTimeOutMs), HCCL_SUCCESS);
    EXPECT_EQ(tpTimeOutMs, 1000U);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Rtp_ThreeTp_Qos4_Expect_GroupedMapping)
{
    MOCKER(RaGetTpInfoListAsync).stubs().will(invoke(StubRaGetTpInfoListAsyncThree));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncSl123));

    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.12.1", "10.10.12.2", TpProtocol::RTP, 4U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_EQ(tpInfo.tpHandle, 0x200ULL);
    EXPECT_EQ(tpInfo.mappedJettyPriority, 2U);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Uboe_EightTp_SingleSlBit_Qos0_Expect_Sl5)
{
    MOCKER(RaGetTpInfoListAsync).stubs().will(invoke(StubRaGetTpInfoListAsyncUboeEight));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncSingleSlBit));

    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.13.1", "10.10.13.2", TpProtocol::UBOE, 0U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_EQ(tpInfo.tpHandle, 0x100ULL);
    EXPECT_EQ(tpInfo.mappedJettyPriority, 5U);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Uboe_EightTp_TwoSlBits_Qos3_Expect_MappedSl7)
{
    MOCKER(RaGetTpInfoListAsync).stubs().will(invoke(StubRaGetTpInfoListAsyncUboeEight));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncTwoSlBits));

    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.14.1", "10.10.14.2", TpProtocol::UBOE, 3U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_EQ(tpInfo.tpHandle, 0x100ULL);
    EXPECT_EQ(tpInfo.mappedJettyPriority, 7U);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Rtp_SlLevelCountCapsMapping_Expect_Success)
{
    MOCKER(RaGetTpInfoListAsync).stubs().will(invoke(StubRaGetTpInfoListAsyncThree));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncSl123));

    TpMgr& mgr = TpMgr::GetInstance(0);
    GetTpInfoParam param = MakeParam("10.10.15.1", "10.10.15.2", TpProtocol::RTP, 0U);
    param.slLevelCount = 2U;
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_NE(tpInfo.tpHandle, 0U);
    EXPECT_TRUE(tpInfo.hasMappedJettyPriority);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Uboe_DscpFromHccnCfg_Expect_Success)
{
    gCapturedSetTpAttrDscp = 0xFFU;
    gCapturedSetTpAttrBitmap = 0U;
    MOCKER(RaGetTpInfoListAsync).stubs().will(invoke(StubRaGetTpInfoListAsyncUboeEight));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncUboeDscpMode0));
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgDscp));
    MOCKER(HrtRaSetTpAttrAsync).stubs().will(invoke(StubHrtRaSetTpAttrAsyncCaptureDscp));

    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.16.1", "10.10.16.2", TpProtocol::UBOE, 2U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_EQ(tpInfo.tpHandle, 0x100ULL);
    // "0:10,1:20,2:30" → qos=2 命中 dscp=30
    EXPECT_EQ(gCapturedSetTpAttrDscp, 30U);
}

TEST_F(TpMgrTest, Ut_GetDscpByQos_When_Hit_Expect_DscpFromCfg)
{
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgDscp));
    uint8_t dscp = 0U;
    ASSERT_EQ(Hccl::GetDscpByQos(0U, 2U, dscp), HCCL_SUCCESS);
    EXPECT_EQ(dscp, 30U);
}

TEST_F(TpMgrTest, Ut_GetDscpByQos_When_EmptyCfg_Expect_DefaultDscp)
{
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgEmpty));
    uint8_t dscp = 0U;
    ASSERT_EQ(Hccl::GetDscpByQos(0U, 2U, dscp), HCCL_SUCCESS);
    EXPECT_EQ(dscp, Hccl::kUboeDefaultDscp);
}

// valueLen 含尾 '\0' 时，未命中 qos 应回退默认 DSCP，而非误判格式非法
TEST_F(TpMgrTest, Ut_GetDscpByQos_When_TrailingNullAndQosMiss_Expect_DefaultDscp)
{
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgDscpWithTrailingNull));
    uint8_t dscp = 0U;
    ASSERT_EQ(Hccl::GetDscpByQos(0U, 0U, dscp), HCCL_SUCCESS);
    EXPECT_EQ(dscp, Hccl::kUboeDefaultDscp);
}

TEST_F(TpMgrTest, Ut_GetDscpByQos_When_InvalidFormat_Expect_E_PARA)
{
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgInvalidFormat));
    uint8_t dscp = 0U;
    EXPECT_EQ(Hccl::GetDscpByQos(0U, 2U, dscp), HCCL_E_PARA);
}

TEST_F(TpMgrTest, Ut_GetDscpByQos_When_RaGetHccnCfgFail_Expect_E_INTERNAL)
{
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgFail));
    uint8_t dscp = 0U;
    // ret=-1 非 EINVAL/ENOTSUPP，映射为 HCCL_E_INTERNAL（非 NOT_FOUND）
    EXPECT_EQ(Hccl::GetDscpByQos(0U, 2U, dscp), HCCL_E_INTERNAL);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Uboe_DscpKeyValueCfg_Expect_Success)
{
    MOCKER(RaGetTpInfoListAsync).stubs().will(invoke(StubRaGetTpInfoListAsyncUboeEight));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncUboeDscpMode0));
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgDscpKeyValue));

    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.16.3", "10.10.16.4", TpProtocol::UBOE, 5U);
    TpInfo tpInfo{};
    EXPECT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_EQ(tpInfo.tpHandle, 0x100ULL);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Ctp_WithQos_Expect_SuccessWithoutSlCommit)
{
    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.17.1", "10.10.17.2", TpProtocol::CTP, 6U);
    TpInfo tpInfo{};
    EXPECT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_NE(tpInfo.tpHandle, 0U);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Rtp_TwoTp_Qos5_Expect_Mapped)
{
    MOCKER(RaGetTpInfoListAsync).stubs().will(invoke(StubRaGetTpInfoListAsyncTwo));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncSl01));

    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.18.1", "10.10.18.2", TpProtocol::RTP, 5U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_TRUE(tpInfo.hasMappedJettyPriority);
    EXPECT_NE(tpInfo.tpHandle, 0U);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Uboe_LoopFirstTpLowestSl_Expect_FirstTp)
{
    MOCKER(RaGetTpInfoListAsync).stubs().will(invoke(StubRaGetTpInfoListAsyncUboeEight));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncUboeSl789));

    TpMgr& mgr = TpMgr::GetInstance(0);
    GetTpInfoParam param = MakeParam("10.10.19.1", "10.10.19.2", TpProtocol::UBOE, 4U);
    param.loopFirstTpLowestSl = true;
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_EQ(tpInfo.tpHandle, 0x100ULL);
    EXPECT_EQ(tpInfo.mappedJettyPriority, 7U);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpAttr_And_ReleaseTpAttr_Expect_Success)
{
    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam tpParam = MakeParam("10.10.20.1", "10.10.20.2", TpProtocol::RTP, 1U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, tpParam, tpInfo), HCCL_SUCCESS);

    const GetTpAttrParam attrParam(tpInfo.tpHandle, (1U << 10U));
    TpAttrInfo tpAttrInfo{};
    const CtxHandle ctxHandle = reinterpret_cast<CtxHandle>(static_cast<uintptr_t>(0x12345678U));
    ASSERT_EQ(PollGetTpAttr(mgr, attrParam, tpAttrInfo, ctxHandle), HCCL_SUCCESS);
    EXPECT_EQ(mgr.GetTpAttr(attrParam, tpAttrInfo, ctxHandle), HCCL_SUCCESS);
    EXPECT_EQ(mgr.ReleaseTpAttr(tpInfo.tpHandle, tpAttrInfo), HCCL_SUCCESS);
}

TEST_F(TpMgrTest, Ut_TpMgr_ReleaseTpInfo_UseCntDecrement_Expect_Success)
{
    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.21.1", "10.10.21.2", TpProtocol::RTP, 2U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    ASSERT_EQ(mgr.GetTpInfo(param, tpInfo), HCCL_SUCCESS);
    EXPECT_EQ(mgr.ReleaseTpInfo(param, tpInfo), HCCL_SUCCESS);
    EXPECT_EQ(mgr.ReleaseTpInfo(param, tpInfo), HCCL_SUCCESS);
}

TEST_F(TpMgrTest, Ut_TpMgr_ReleaseTpInfo_TpHandleMismatch_Expect_Skip)
{
    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.22.1", "10.10.22.2", TpProtocol::RTP, 3U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    TpInfo wrongTpInfo{};
    wrongTpInfo.tpHandle = tpInfo.tpHandle + 1U;
    EXPECT_EQ(mgr.ReleaseTpInfo(param, wrongTpInfo), HCCL_SUCCESS);
    EXPECT_EQ(mgr.ReleaseTpInfo(param, tpInfo), HCCL_SUCCESS);
}

TEST_F(TpMgrTest, Ut_TpMgr_CalcTaTimeout_When_EnvLessThanTp_Expect_Upgrade)
{
    UbTimeoutEnvGuard envGuard("0");

    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 3U;
    tpAttrInfo.tpAttr.retryTimesInit = 0U;
    uint32_t taTimeOutValue = 0;
    EXPECT_EQ(
        hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(taTimeOutValue),
        HCCL_SUCCESS);
    const uint8_t envTaTimeOut = static_cast<uint8_t>(taTimeOutValue);
    EXPECT_EQ(TpMgr::CalcTaTimeout(TpProtocol::RTP, envTaTimeOut, 4000U), 16U);
}

TEST_F(TpMgrTest, Ut_TpMgr_CalcTaTimeout_When_EnvGreaterThanTp_Expect_EnvValue)
{
    UbTimeoutEnvGuard envGuard("24");

    TpAttrInfo tpAttrInfo{};
    tpAttrInfo.tpAttr.at = 0U;
    tpAttrInfo.tpAttr.retryTimesInit = 0U;
    uint32_t taTimeOutValue = 0;
    EXPECT_EQ(
        hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(taTimeOutValue),
        HCCL_SUCCESS);
    const uint8_t envTaTimeOut = static_cast<uint8_t>(taTimeOutValue);
    EXPECT_EQ(TpMgr::CalcTaTimeout(TpProtocol::RTP, envTaTimeOut, 16U), 24U);
}

TEST_F(TpMgrTest, Ut_TpMgr_ReleaseTpAttr_UseCntDecrement_Expect_Success)
{
    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam tpParam = MakeParam("10.10.23.1", "10.10.23.2", TpProtocol::CTP, 0U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, tpParam, tpInfo), HCCL_SUCCESS);

    const GetTpAttrParam attrParam(tpInfo.tpHandle, (1U << 10U));
    TpAttrInfo tpAttrInfo{};
    const CtxHandle ctxHandle = reinterpret_cast<CtxHandle>(static_cast<uintptr_t>(0x12345678U));
    ASSERT_EQ(PollGetTpAttr(mgr, attrParam, tpAttrInfo, ctxHandle), HCCL_SUCCESS);
    ASSERT_EQ(mgr.GetTpAttr(attrParam, tpAttrInfo, ctxHandle), HCCL_SUCCESS);
    EXPECT_EQ(mgr.ReleaseTpAttr(tpInfo.tpHandle, tpAttrInfo), HCCL_SUCCESS);
    EXPECT_EQ(mgr.ReleaseTpAttr(tpInfo.tpHandle, tpAttrInfo), HCCL_SUCCESS);
}

TEST_F(TpMgrTest, Ut_TpMgr_ReleaseTpInfo_QosKeyMismatch_Expect_NotFound)
{
    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam paramQos0 = MakeParam("10.10.24.1", "10.10.24.2", TpProtocol::RTP, 0U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, paramQos0, tpInfo), HCCL_SUCCESS);

    const GetTpInfoParam paramQos1 = MakeParam("10.10.24.1", "10.10.24.2", TpProtocol::RTP, 1U);
    EXPECT_EQ(mgr.ReleaseTpInfo(paramQos1, tpInfo), HCCL_E_NOT_FOUND);
    EXPECT_EQ(mgr.ReleaseTpInfo(paramQos0, tpInfo), HCCL_SUCCESS);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetInstance_When_PhyIdOverflow_Expect_BackupSlotUsable)
{
    TpMgr& mgr = TpMgr::GetInstance(MAX_MODULE_DEVICE_NUM + 16U);
    const GetTpInfoParam param = MakeParam("10.10.25.1", "10.10.25.2", TpProtocol::CTP, 2U);
    TpInfo tpInfo{};
    EXPECT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_NE(tpInfo.tpHandle, 0U);
}

TEST_F(TpMgrTest, Ut_TpMgr_GetTpInfo_Uboe_ThreeTp_Qos2_Expect_GroupedFirstQos)
{
    MOCKER(RaGetTpInfoListAsync).stubs().will(invoke(StubRaGetTpInfoListAsyncThree));
    MOCKER(RaGetTpAttrAsync).stubs().will(invoke(StubRaGetTpAttrAsyncSl123));

    TpMgr& mgr = TpMgr::GetInstance(0);
    const GetTpInfoParam param = MakeParam("10.10.26.1", "10.10.26.2", TpProtocol::UBOE, 2U);
    TpInfo tpInfo{};
    ASSERT_EQ(PollGetTpInfo(mgr, param, tpInfo), HCCL_SUCCESS);
    EXPECT_TRUE(tpInfo.hasMappedJettyPriority);
    EXPECT_NE(tpInfo.tpHandle, 0U);
}
