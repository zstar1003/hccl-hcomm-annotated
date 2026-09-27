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

#include "hdc_pub.h"
#include "adapter_hal.h"
#include "adapter_rts.h"
#include "dltdt_function.h"
#include "dlhal_function.h"

using namespace std;
using namespace hccl;

class HcclHDCTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclHDCTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "HcclHDCTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        DlHalFunction::GetInstance().DlHalFunctionInit();
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

HcclResult fake_hrtHalHostRegister(void* hostPtr, u64 size, u32 flag, u32 devid, void*& devPtr)
{
    devPtr = hostPtr;
    return HCCL_SUCCESS;
}

TEST_F(HcclHDCTest, ut_hccl_hdc_init_host_d2h)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_D2H;
    u32 buffLen = 8 * 1024;
    hrtSetDevice(devid);

    MOCKER(hrtHalHostRegister).expects(atMost(1)).will(invoke(fake_hrtHalHostRegister));

    HDCommunicate hdc(devid, flag, buffLen);
    auto ret = hdc.InitHost();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcclHDCTest, ut_hccl_hdc_init_host_h2d)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_H2D;
    u32 buffLen = 8 * 1024;
    hrtSetDevice(devid);

    MOCKER(hrtHalHostRegister).expects(atMost(1)).will(invoke(fake_hrtHalHostRegister));

    HDCommunicate hdc(devid, flag, buffLen);
    auto ret = hdc.InitHost();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcclHDCTest, ut_hccl_hdc_init_device_d2h)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_D2H;
    u32 buffLen = 8 * 1024;
    hrtSetDevice(devid);

    MOCKER(hrtHalHostRegister).expects(atMost(1)).will(invoke(fake_hrtHalHostRegister));

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.InitHost();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicate hdcDevice;
    ret = hdcDevice.InitDevice(params);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcclHDCTest, ut_hccl_hdc_init_device_h2d)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_H2D;
    u32 buffLen = 8 * 1024;
    hrtSetDevice(devid);

    MOCKER(hrtHalHostRegister).expects(atMost(1)).will(invoke(fake_hrtHalHostRegister));

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.InitHost();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicate hdcDevice;
    ret = hdcDevice.InitDevice(params);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

#if 1

TEST_F(HcclHDCTest, ut_hccl_hdc_h2d)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_H2D;
    u32 buffLen = 8 * 1024;
    hrtSetDevice(devid);

    MOCKER(hrtHalHostRegister).expects(atMost(1)).will(invoke(fake_hrtHalHostRegister));

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.InitHost();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicate hdcDevice;
    ret = hdcDevice.InitDevice(params);
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

    GlobalMockObject::verify();
}

TEST_F(HcclHDCTest, ut_hccl_hdc_h2d_multi)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_H2D;
    u32 buffLen = 8 * 1024;
    hrtSetDevice(devid);

    MOCKER(hrtHalHostRegister).expects(atMost(1)).will(invoke(fake_hrtHalHostRegister));

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.InitHost();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicate hdcDevice;
    ret = hdcDevice.InitDevice(params);
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

    GlobalMockObject::verify();
}

TEST_F(HcclHDCTest, ut_hccl_hdc_h2d_multi_put_single_get)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_H2D;
    u32 buffLen = 8 * 1024;
    hrtSetDevice(devid);

    MOCKER(hrtHalHostRegister).expects(atMost(1)).will(invoke(fake_hrtHalHostRegister));

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.InitHost();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicate hdcDevice;
    ret = hdcDevice.InitDevice(params);
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

    GlobalMockObject::verify();
}

TEST_F(HcclHDCTest, ut_hccl_hdc_h2d_single_put_multi_get)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_H2D;
    u32 buffLen = 8 * 1024;
    hrtSetDevice(devid);

    MOCKER(hrtHalHostRegister).expects(atMost(1)).will(invoke(fake_hrtHalHostRegister));

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.InitHost();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicate hdcDevice;
    ret = hdcDevice.InitDevice(params);
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

    GlobalMockObject::verify();
}

#endif

#if 1

TEST_F(HcclHDCTest, ut_hccl_hdc_d2h)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_D2H;
    u32 buffLen = 8 * 1024;
    hrtSetDevice(devid);

    MOCKER(hrtHalHostRegister).expects(atMost(1)).will(invoke(fake_hrtHalHostRegister));

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.InitHost();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicate hdcDevice;
    ret = hdcDevice.InitDevice(params);
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

    GlobalMockObject::verify();
}

TEST_F(HcclHDCTest, ut_hccl_hdc_d2h_multi)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_D2H;
    u32 buffLen = 8 * 1024;
    hrtSetDevice(devid);

    MOCKER(hrtHalHostRegister).expects(atMost(1)).will(invoke(fake_hrtHalHostRegister));

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.InitHost();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicate hdcDevice;
    ret = hdcDevice.InitDevice(params);
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

    GlobalMockObject::verify();
}

TEST_F(HcclHDCTest, ut_hccl_hdc_d2h_multi_put_single_get)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_D2H;
    u32 buffLen = 8 * 1024;
    hrtSetDevice(devid);

    MOCKER(hrtHalHostRegister).expects(atMost(1)).will(invoke(fake_hrtHalHostRegister));

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.InitHost();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicate hdcDevice;
    ret = hdcDevice.InitDevice(params);
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

    GlobalMockObject::verify();
}

TEST_F(HcclHDCTest, ut_hccl_hdc_d2h_single_put_multi_get)
{
    u32 devid = 0;
    u32 flag = HCCL_HDC_TYPE_D2H;
    u32 buffLen = 8 * 1024;
    hrtSetDevice(devid);

    MOCKER(hrtHalHostRegister).expects(atMost(1)).will(invoke(fake_hrtHalHostRegister));

    HDCommunicate hdcHost(devid, flag, buffLen);
    auto ret = hdcHost.InitHost();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct HDCommunicateParams params;
    params = hdcHost.GetCommunicateParams();

    HDCommunicate hdcDevice;
    ret = hdcDevice.InitDevice(params);
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

    GlobalMockObject::verify();
}

#endif
