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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hcomm_adapter_hccp.h"
#include "hccp_common.h"
#include "hccp_tlv.h"

using namespace hcomm;
using namespace Hccl;

namespace {
constexpr uintptr_t FAKE_HCCP_CONTEXT = 0x12345678U;
constexpr uint32_t FAKE_UBOE_IPV4 = 0xC0A80367U;
constexpr uint8_t FAKE_UBOE_EID_LAST_BYTE = 0x67U;

int RaGetIpByEidSuccessStub(void* ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int* num)
{
    EXPECT_EQ(ctxHandle, reinterpret_cast<void*>(FAKE_HCCP_CONTEXT));
    EXPECT_NE(eid, nullptr);
    EXPECT_NE(ip, nullptr);
    EXPECT_NE(num, nullptr);
    EXPECT_EQ(*num, 1U);
    EXPECT_EQ(eid[0].raw[15], FAKE_UBOE_EID_LAST_BYTE);

    ip[0].family = AF_INET;
    ip[0].ip.addr.s_addr = htonl(FAKE_UBOE_IPV4);
    *num = 1U;
    return 0;
}
} // namespace

class HcommAdapterHccpTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcommAdapterHccpTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "HcommAdapterHccpTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in HcommAdapterHccpTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HcommAdapterHccpTest TearDown" << std::endl;
    }
};

TEST_F(HcommAdapterHccpTest, ut_HccpGetUboeFlagEnable_VersionEnough_Expect_Success)
{
    u32 devPhyId = 1;
    u32 mock_version = GET_UBOE_FLAG_ENABLE_VERSION;
    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mock_version, sizeof(s32)))
        .will(returnValue(0));

    HcclResult ret = HccpGetUboeFlagEnable(devPhyId);

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetUboeFlagEnable_When_VersionNotEnough_Expect_NotSupport)
{
    u32 devPhyId = 1;
    u32 mock_version = GET_UBOE_FLAG_ENABLE_VERSION - 1;
    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mock_version, sizeof(s32)))
        .will(returnValue(0));

    HcclResult ret = HccpGetUboeFlagEnable(devPhyId);

    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetUboeFlagEnable_When_RaGetInterfaceFailed_Expect_E_INTERNAL)
{
    u32 devPhyId = 1;
    MOCKER(RaGetInterfaceVersion).stubs().will(returnValue(-1));
    HcclResult ret = HccpGetUboeFlagEnable(devPhyId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommAdapterHccpTest, ut_HccpCheckUboeSupported_When_DevFeatureBitSet_Expect_True)
{
    u32 devFeature = 1 << UBOE_DEV_FLAG_RIGHT_SHIFT;
    bool result = HccpCheckUboeSupported(devFeature);
    EXPECT_TRUE(result);

    // 测试其他位也有值的情况
    devFeature = (1 << UBOE_DEV_FLAG_RIGHT_SHIFT) | 0xFFFF;
    result = HccpCheckUboeSupported(devFeature);
    EXPECT_TRUE(result);
}

TEST_F(HcommAdapterHccpTest, ut_HccpCheckUboeSupported_When_DevFeatureBitNotSet_Expect_False)
{
    u32 devFeature = 0;
    bool result = HccpCheckUboeSupported(devFeature);
    EXPECT_FALSE(result);

    // 测试只有其他位被设置，但UBOE位未设置
    devFeature = 0xFFFFFFFF & ~(1 << UBOE_DEV_FLAG_RIGHT_SHIFT);
    result = HccpCheckUboeSupported(devFeature);
    EXPECT_FALSE(result);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetIpByEid_When_ValidEid_Expect_Ipv4)
{
    MOCKER(RaGetIpByEid).stubs().will(invoke(RaGetIpByEidSuccessStub));

    CommAddr eidAddr{};
    eidAddr.type = COMM_ADDR_TYPE_EID;
    eidAddr.eid[15] = FAKE_UBOE_EID_LAST_BYTE;
    CommAddr ipAddr{};

    HcclResult ret = HccpGetIpByEid(reinterpret_cast<void*>(FAKE_HCCP_CONTEXT), eidAddr, ipAddr);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ipAddr.type, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(ipAddr.addr.s_addr, htonl(FAKE_UBOE_IPV4));
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetIpByEid_When_RaQueryFails_Expect_E_NETWORK)
{
    MOCKER(RaGetIpByEid).stubs().will(returnValue(-1));

    CommAddr eidAddr{};
    eidAddr.type = COMM_ADDR_TYPE_EID;
    CommAddr ipAddr{};

    HcclResult ret = HccpGetIpByEid(reinterpret_cast<void*>(FAKE_HCCP_CONTEXT), eidAddr, ipAddr);

    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_EQ(ipAddr.type, COMM_ADDR_TYPE_RESERVED);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetCtpEnable_When_CtpExists_Expect_True)
{
    DevBaseAttr attr{};
    attr.ub.priorityInfo[0].tpType.bs.rtp = 1;
    attr.ub.priorityInfo[1].tpType.bs.ctp = 1;
    MOCKER(RaGetDevBaseAttr).stubs().with(mockcpp::any(), outBoundP(&attr, sizeof(attr))).will(returnValue(0));

    bool ctpEnable = false;
    EXPECT_EQ(HccpGetCtpEnable(reinterpret_cast<void*>(FAKE_HCCP_CONTEXT), ctpEnable), HCCL_SUCCESS);
    EXPECT_TRUE(ctpEnable);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetCtpEnable_When_AllRtp_Expect_False)
{
    DevBaseAttr attr{};
    for (uint32_t i = 0; i < MAX_PRIORITY_CNT; ++i) {
        attr.ub.priorityInfo[i].tpType.bs.rtp = 1;
    }
    MOCKER(RaGetDevBaseAttr).stubs().with(mockcpp::any(), outBoundP(&attr, sizeof(attr))).will(returnValue(0));

    bool ctpEnable = true;
    EXPECT_EQ(HccpGetCtpEnable(reinterpret_cast<void*>(FAKE_HCCP_CONTEXT), ctpEnable), HCCL_SUCCESS);
    EXPECT_FALSE(ctpEnable);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetCtpEnable_When_QueryFails_Expect_ENetworkAndFalse)
{
    MOCKER(RaGetDevBaseAttr).stubs().will(returnValue(-1));

    bool ctpEnable = true;
    EXPECT_EQ(HccpGetCtpEnable(reinterpret_cast<void*>(FAKE_HCCP_CONTEXT), ctpEnable), HCCL_E_NETWORK);
    EXPECT_FALSE(ctpEnable);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetCtpEnable_When_CtxHandleNull_Expect_EPtrAndFalse)
{
    bool ctpEnable = true;
    EXPECT_EQ(HccpGetCtpEnable(nullptr, ctpEnable), HCCL_E_PTR);
    EXPECT_FALSE(ctpEnable);
}

// ========== HccpRaTlvRequestForCustomChannel 测试 ==========

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_RaTlvRequestOk_Expect_ReturnHCCL_SUCCESS)
{
    void* tlvHandle = reinterpret_cast<void*>(0x1234);
    char inBuff[8] = {0};
    char outBuff[8] = {0};

    MOCKER(RaTlvRequest)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(0));

    HcclResult ret = HccpRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, outBuff);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_TlvHandleNull_Expect_ReturnHCCL_E_PTR)
{
    char inBuff[8] = {0};
    char outBuff[8] = {0};

    HcclResult ret = HccpRaTlvRequestForCustomChannel(nullptr, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, outBuff);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_CustomInNull_Expect_ReturnHCCL_E_PTR)
{
    void* tlvHandle = reinterpret_cast<void*>(0x1234);
    char outBuff[8] = {0};

    HcclResult ret = HccpRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, nullptr, outBuff);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_CustomOutNull_Expect_ReturnHCCL_E_PTR)
{
    void* tlvHandle = reinterpret_cast<void*>(0x1234);
    char inBuff[8] = {0};

    HcclResult ret = HccpRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_RaTlvRequestFail_Expect_ReturnHCCL_E_NETWORK)
{
    void* tlvHandle = reinterpret_cast<void*>(0x1234);
    char inBuff[8] = {0};
    char outBuff[8] = {0};

    MOCKER(RaTlvRequest)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(-1));

    HcclResult ret = HccpRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, outBuff);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

// ========== HrtRaDumpJettyContext 测试 ==========

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_JettyHandleIsNull_Expect_ReturnPtr)
{
    HcclResult ret = HrtRaDumpJettyContext(nullptr, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_RaCtxGetJettyContextFails_Expect_ReturnInternal)
{
    void* validHandle = reinterpret_cast<void*>(0x1234);
    MOCKER(RaCtxGetJettyContext).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(-1));
    HcclResult ret = HrtRaDumpJettyContext(validHandle, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_ContextLenIsZero_Expect_ReturnInternal)
{
    void* validHandle = reinterpret_cast<void*>(0x1234);
    unsigned int mockLen = 0;
    MOCKER(RaCtxGetJettyContext)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mockLen, sizeof(mockLen)))
        .will(returnValue(0));
    HcclResult ret = HrtRaDumpJettyContext(validHandle, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_ContextLenExceedsMax_Expect_ReturnInternal)
{
    void* validHandle = reinterpret_cast<void*>(0x1234);
    unsigned int mockLen = 600;
    MOCKER(RaCtxGetJettyContext)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mockLen, sizeof(mockLen)))
        .will(returnValue(0));
    HcclResult ret = HrtRaDumpJettyContext(validHandle, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_Normal_Expect_ReturnSuccess)
{
    void* validHandle = reinterpret_cast<void*>(0x1234);
    unsigned int mockLen = 256;
    MOCKER(RaCtxGetJettyContext)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mockLen, sizeof(mockLen)))
        .will(returnValue(0));
    HcclResult ret = HrtRaDumpJettyContext(validHandle, 7);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
