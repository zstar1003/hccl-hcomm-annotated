/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#ifndef private
#define private public
#define protected public
#endif
#include "launch_aicpu.h"
#undef private
#undef protected

#include "adapter_rts_common.h"
#include "hccl_dl.h"

using namespace std;
using namespace hccl;

static aclError stubCacheLastTaskExtendInfoSuccess(const char* tag, size_t tagLen) { return ACL_SUCCESS; }

static aclError stubCacheLastTaskExtendInfoFail(const char* tag, size_t tagLen) { return static_cast<aclError>(2); }

class LaunchAicpuUT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LaunchAicpuUT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "LaunchAicpuUT TearDown" << std::endl; }
};

TEST_F(LaunchAicpuUT, AicpuAclKernelLaunchV2_When_BinHandleIsNull_Expect_ReturnPtrError)
{
    rtStream_t stm = nullptr;
    HcclResult ret = AicpuAclKernelLaunchV2(stm, nullptr, 0, nullptr, "test", true, 0, nullptr, 0, "tag");
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(LaunchAicpuUT, AicpuAclKernelLaunchV2_When_AddrIsNull_Expect_ReturnParaError)
{
    rtStream_t stm = nullptr;
    aclrtBinHandle binHandle = reinterpret_cast<aclrtBinHandle>(1);
    HcclResult ret = AicpuAclKernelLaunchV2(stm, nullptr, 0, binHandle, "test", true, 0, nullptr, 0, "tag");
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(LaunchAicpuUT, AicpuAclKernelLaunchV2_When_SizeIsZero_Expect_ReturnParaError)
{
    rtStream_t stm = nullptr;
    aclrtBinHandle binHandle = reinterpret_cast<aclrtBinHandle>(1);
    u32 addr = 1;
    HcclResult ret = AicpuAclKernelLaunchV2(stm, &addr, 0, binHandle, "test", true, 0, nullptr, 0, "tag");
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(LaunchAicpuUT, AicpuAclKernelLaunchV2_When_GetFunctionFail_Expect_ReturnRuntimeError)
{
    rtStream_t stm = nullptr;
    aclrtBinHandle binHandle = reinterpret_cast<aclrtBinHandle>(1);
    u32 addr = 1;
    u32 size = 10;
    MOCKER(aclrtBinaryGetFunction).stubs().will(returnValue(ACL_ERROR_RT_FEATURE_NOT_SUPPORT));
    HcclResult ret = AicpuAclKernelLaunchV2(stm, &addr, size, binHandle, "test", true, 0, nullptr, 0, "tag");
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
    GlobalMockObject::verify();
}

TEST_F(LaunchAicpuUT, CacheTaskOpInfo_ExtendInfoNotSupported_Expect_ReturnSuccess)
{
    aclrtStream stream = reinterpret_cast<aclrtStream>(1);

    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(ACL_SUCCESS));

    aclrtStreamAttrValue value;
    value.cacheOpInfoSwitch = 1;
    MOCKER(aclrtGetStreamAttribute)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&value, sizeof(value)))
        .will(returnValue(ACL_SUCCESS));

    MOCKER(hrtCacheLastTaskExtendInfo).stubs().will(returnValue(HCCL_E_NOT_SUPPORT));

    HcclResult ret = CacheTaskOpInfo(stream, "test_identify");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(LaunchAicpuUT, CacheTaskOpInfo_ExtendInfoCallSuccess_Expect_ReturnSuccess)
{
    aclrtStream stream = reinterpret_cast<aclrtStream>(1);

    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(ACL_SUCCESS));

    aclrtStreamAttrValue value;
    value.cacheOpInfoSwitch = 1;
    MOCKER(aclrtGetStreamAttribute)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&value, sizeof(value)))
        .will(returnValue(ACL_SUCCESS));

    MOCKER(hrtCacheLastTaskExtendInfo).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = CacheTaskOpInfo(stream, "test_identify");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(LaunchAicpuUT, CacheTaskOpInfo_ExtendInfoCallFailed_Expect_ReturnRuntimeError)
{
    aclrtStream stream = reinterpret_cast<aclrtStream>(1);

    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(ACL_SUCCESS));

    aclrtStreamAttrValue value;
    value.cacheOpInfoSwitch = 1;
    MOCKER(aclrtGetStreamAttribute)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&value, sizeof(value)))
        .will(returnValue(ACL_SUCCESS));

    MOCKER(hrtCacheLastTaskExtendInfo).stubs().will(returnValue(HCCL_E_RUNTIME));

    HcclResult ret = CacheTaskOpInfo(stream, "test_identify");
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
    GlobalMockObject::verify();
}

TEST_F(LaunchAicpuUT, AicpuAclKernelLaunchV2_When_LaunchKernelFail_Expect_ReturnRuntimeError)
{
    rtStream_t stm = nullptr;
    aclrtBinHandle binHandle = reinterpret_cast<aclrtBinHandle>(1);
    u32 addr = 1;
    u32 size = 10;
    MOCKER(aclrtBinaryGetFunction).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(ACL_ERROR_RT_FEATURE_NOT_SUPPORT));
    HcclResult ret = AicpuAclKernelLaunchV2(stm, &addr, size, binHandle, "test", true, 0, nullptr, 0, "tag");
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
    GlobalMockObject::verify();
}

TEST_F(LaunchAicpuUT, AicpuAclKernelLaunchV2_When_CaptureNotSupport_Expect_ReturnSuccess)
{
    rtStream_t stm = reinterpret_cast<rtStream_t>(1);
    aclrtBinHandle binHandle = reinterpret_cast<aclrtBinHandle>(1);
    u32 addr = 1;
    u32 size = 10;
    MOCKER(aclrtBinaryGetFunction).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclmdlRICaptureGetInfo).stubs().will(returnValue(ACL_ERROR_RT_FEATURE_NOT_SUPPORT));
    HcclResult ret = AicpuAclKernelLaunchV2(stm, &addr, size, binHandle, "test", true, 0, nullptr, 0, "tag");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(LaunchAicpuUT, AicpuAclKernelLaunchV2_When_CaptureGetInfoFail_Expect_ReturnRuntimeError)
{
    rtStream_t stm = reinterpret_cast<rtStream_t>(1);
    aclrtBinHandle binHandle = reinterpret_cast<aclrtBinHandle>(1);
    u32 addr = 1;
    u32 size = 10;
    MOCKER(aclrtBinaryGetFunction).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclmdlRICaptureGetInfo).stubs().will(returnValue(static_cast<aclError>(ACL_ERROR_RT_INTERNAL_ERROR)));
    HcclResult ret = AicpuAclKernelLaunchV2(stm, &addr, size, binHandle, "test", true, 0, nullptr, 0, "tag");
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
    GlobalMockObject::verify();
}

TEST_F(LaunchAicpuUT, AicpuAclKernelLaunchV2_When_GetStreamAttrFail_Expect_ReturnRuntimeError)
{
    rtStream_t stm = reinterpret_cast<rtStream_t>(1);
    aclrtBinHandle binHandle = reinterpret_cast<aclrtBinHandle>(1);
    u32 addr = 1;
    u32 size = 10;
    MOCKER(aclrtBinaryGetFunction).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclmdlRICaptureGetInfo).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtGetStreamAttribute).stubs().will(returnValue(static_cast<aclError>(ACL_ERROR_RT_INTERNAL_ERROR)));
    HcclResult ret = AicpuAclKernelLaunchV2(stm, &addr, size, binHandle, "test", true, 0, nullptr, 0, "tag");
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
    GlobalMockObject::verify();
}

TEST_F(LaunchAicpuUT, AicpuAclKernelLaunchV2_When_CacheTaskOpInfoSuccess_Expect_ReturnSuccess)
{
    rtStream_t stm = reinterpret_cast<rtStream_t>(1);
    aclrtBinHandle binHandle = reinterpret_cast<aclrtBinHandle>(1);
    u32 addr = 1;
    u32 size = 10;

    MOCKER(aclrtBinaryGetFunction).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(ACL_SUCCESS));

    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(ACL_SUCCESS));

    aclrtStreamAttrValue value;
    value.cacheOpInfoSwitch = 1;
    MOCKER(aclrtGetStreamAttribute)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&value, sizeof(value)))
        .will(returnValue(ACL_SUCCESS));

    MOCKER(hrtCacheLastTaskExtendInfo).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = AicpuAclKernelLaunchV2(stm, &addr, size, binHandle, "test", true, 0, nullptr, 0, "tag");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(LaunchAicpuUT, hrtCacheLastTaskExtendInfo_DlsymNull_Expect_ReturnNotSupport)
{
    MOCKER(HcclDlsym).stubs().will(returnValue((void*)nullptr));

    HcclResult ret = hrtCacheLastTaskExtendInfo("test_tag", 8);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    GlobalMockObject::verify();
}

TEST_F(LaunchAicpuUT, hrtCacheLastTaskExtendInfo_CallSuccess_Expect_ReturnSuccess)
{
    MOCKER(HcclDlsym).stubs().will(returnValue((void*)stubCacheLastTaskExtendInfoSuccess));

    HcclResult ret = hrtCacheLastTaskExtendInfo("test_tag", 8);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(LaunchAicpuUT, hrtCacheLastTaskExtendInfo_CallFail_Expect_ReturnRuntimeError)
{
    MOCKER(HcclDlsym).stubs().will(returnValue((void*)stubCacheLastTaskExtendInfoFail));

    HcclResult ret = hrtCacheLastTaskExtendInfo("test_tag", 8);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
    GlobalMockObject::verify();
}
