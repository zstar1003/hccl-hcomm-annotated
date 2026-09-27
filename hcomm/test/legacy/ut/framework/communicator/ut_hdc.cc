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
#include "hdc.h"
#include "hdc_pub.h"
#include "hdc_lite.h"
#undef private
#include "ascend_hal.h"
#include "orion_adapter_rts.h"
#include "internal_exception.h"
using namespace Hccl;

static HcclResult fake_hrtHalHostRegister(void* hostPtr, u64 size, u32 flag, u32 devid, void*& devPtr)
{
    devPtr = hostPtr;
    return HCCL_SUCCESS;
}

static HcclResult HrtDrvMemCpyStub(void* dst, uint64_t destMax, const void* src, uint64_t count)
{
    memcpy(dst, src, count);
    return HCCL_SUCCESS;
}

class HdcTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HdcTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HdcTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        memset_s(hostBuf, sizeof(hostBuf), 0, sizeof(hostBuf));
        memset_s(hostCache, sizeof(hostCache), 0, sizeof(hostCache));
        MOCKER(HrtMallocHost)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(static_cast<void*>(hostBuf)))
            .then(returnValue(static_cast<void*>(hostCache)));

        memset_s(devBuf, sizeof(devBuf), 0, sizeof(devBuf));
        memset_s(devCache, sizeof(devCache), 0, sizeof(devCache));
        MOCKER(HrtMalloc)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(static_cast<void*>(devBuf)))
            .then(returnValue(static_cast<void*>(devCache)));

        MOCKER(HrtDrvMemCpy).stubs().with().will(invoke(HrtDrvMemCpyStub));
        MOCKER(halHostRegister).expects(atMost(1)).will(invoke(fake_hrtHalHostRegister));

        std::cout << "A Test case in HdcTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in HdcTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }

    static constexpr u32 buffLen = 8 * 1024;
    char hostBuf[buffLen + 4 * 1024];
    char hostCache[buffLen + 4 * 1024];
    char devBuf[buffLen + 4 * 1024];
    char devCache[buffLen + 4 * 1024];
};

TEST_F(HdcTest, hccl_hdc_init_d2h)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_D2H;

    HDCommunicate hdc(devid, flag, buffLen);
    auto ret = hdc.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HdcTest, hccl_hdc_init_h2d)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_H2D;

    HDCommunicate hdc(devid, flag, buffLen);
    auto ret = hdc.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HdcTest, hccl_hdc_lite_init_d2h)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_D2H;

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicateLite hdcDevice;
    ret = hdcDevice.Init(params);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HdcTest, hccl_hdc_lite_init_h2d)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_H2D;

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicateLite hdcDevice;
    ret = hdcDevice.Init(params);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HdcTest, hccl_hdc_h2d)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_H2D;

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicateLite hdcDevice;
    ret = hdcDevice.Init(params);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 str[256] = {0};
    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = i + 1;
    }

    ret = hdcHost.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 getStr[256] = {0};
    ret = hdcDevice.Get(0, sizeof(getStr), getStr);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr[0]));
}

TEST_F(HdcTest, hccl_hdc_h2d_multi)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_H2D;

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicateLite hdcDevice;
    ret = hdcDevice.Init(params);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 str[256] = {0};
    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = i + 1;
    }

    ret = hdcHost.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 getStr[256] = {0};
    ret = hdcDevice.Get(0, sizeof(getStr), getStr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr[0]));

    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = 2 * i + 1;
    }
    ret = hdcHost.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hdcDevice.Get(0, sizeof(getStr), getStr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr[0]));

    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = 3 * i + 1;
    }
    ret = hdcHost.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hdcDevice.Get(0, sizeof(getStr), getStr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr[0]));
}

TEST_F(HdcTest, hccl_hdc_h2d_multi_psingle_get)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_H2D;

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicateLite hdcDevice;
    ret = hdcDevice.Init(params);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 str[256] = {0};
    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = i + 1;
    }
    ret = hdcHost.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = 2 * i + 1;
    }
    ret = hdcHost.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = 3 * i + 1;
    }
    ret = hdcHost.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 getStr[256] = {0};
    ret = hdcDevice.Get(0, sizeof(getStr), getStr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr[0]));
}

TEST_F(HdcTest, hccl_hdc_h2d_single_pmulti_get)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_H2D;

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicateLite hdcDevice;
    ret = hdcDevice.Init(params);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 str[256] = {0};
    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = i + 1;
    }
    ret = hdcHost.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 getStr1[256] = {0};
    ret = hdcDevice.Get(0, sizeof(getStr1), getStr1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr1[0]));

    u8 getStr2[256] = {0};
    ret = hdcDevice.Get(0, sizeof(getStr2), getStr2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr2[0]));

    u8 getStr3[256] = {0};
    ret = hdcDevice.Get(0, sizeof(getStr3), getStr3);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr3[0]));
}

TEST_F(HdcTest, hccl_hdc_d2h)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_D2H;

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicateLite hdcDevice;
    ret = hdcDevice.Init(params);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 str[256] = {0};
    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = i + 1;
    }

    ret = hdcDevice.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 getStr[256] = {0};
    ret = hdcHost.Get(0, sizeof(getStr), getStr);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr[0]));
}

TEST_F(HdcTest, hccl_hdc_h2d_unSupport_devMemReg)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_H2D;

    MOCKER(halMemCtl).expects(atMost(1)).with(mockcpp::any()).will(returnValue(0));

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicateLite hdcDevice;
    ret = hdcDevice.Init(params);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 str[256] = {0};
    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = i + 1;
    }

    ret = hdcHost.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 getStr[256] = {0};
    ret = hdcDevice.Get(0, sizeof(getStr), getStr);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr[0]));
}

TEST_F(HdcTest, hccl_hdc_d2h_multi)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_D2H;

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicateLite hdcDevice;
    ret = hdcDevice.Init(params);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 str[256] = {0};
    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = i + 1;
    }

    ret = hdcDevice.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 getStr[256] = {0};
    ret = hdcHost.Get(0, sizeof(getStr), getStr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr[0]));

    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = 2 * i + 1;
    }
    ret = hdcDevice.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hdcHost.Get(0, sizeof(getStr), getStr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr[0]));

    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = 3 * i + 1;
    }
    ret = hdcDevice.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hdcHost.Get(0, sizeof(getStr), getStr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr[0]));
}

TEST_F(HdcTest, hccl_hdc_d2h_multi_psingle_get)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_D2H;

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicateLite hdcDevice;
    ret = hdcDevice.Init(params);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 str[256] = {0};
    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = i + 1;
    }
    ret = hdcDevice.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = 2 * i + 1;
    }
    ret = hdcDevice.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = 3 * i + 1;
    }
    ret = hdcDevice.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 getStr[256] = {0};
    ret = hdcHost.Get(0, sizeof(getStr), getStr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr[0]));
}

TEST_F(HdcTest, hccl_hdc_d2h_single_pmulti_get)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_D2H;

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicateLite hdcDevice;
    ret = hdcDevice.Init(params);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 str[256] = {0};
    for (int i = 0; i < sizeof(str) - 1; i++) {
        str[i] = i + 1;
    }
    ret = hdcDevice.Put(0, sizeof(str), str);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 getStr1[256] = {0};
    ret = hdcHost.Get(0, sizeof(getStr1), getStr1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr1[0]));

    u8 getStr2[256] = {0};
    ret = hdcHost.Get(0, sizeof(getStr2), getStr2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr2[0]));

    u8 getStr3[256] = {0};
    ret = hdcHost.Get(0, sizeof(getStr3), getStr3);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(reinterpret_cast<char*>(&str[0]), reinterpret_cast<char*>(&getStr3[0]));
}
