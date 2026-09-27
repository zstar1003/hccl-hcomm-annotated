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
#include <string>

#include "hccl/base.h"
#include "adapter_rts.h"
#include "device_capacity.h"

using namespace std;
using namespace hccl;

namespace {
HcclResult SetDevType910B(DevType& devType)
{
    devType = DevType::DEV_TYPE_910B;
    return HCCL_SUCCESS;
}
HcclResult SetDevType910(DevType& devType)
{
    devType = DevType::DEV_TYPE_910;
    return HCCL_SUCCESS;
}
HcclResult SetDevType91093(DevType& devType)
{
    devType = DevType::DEV_TYPE_910_93;
    return HCCL_SUCCESS;
}
} // namespace

class DeviceCapacityTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DeviceCapacityTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "DeviceCapacityTest TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "DeviceCapacityTest SetUP per test" << std::endl; }
    virtual void TearDown()
    {
        std::cout << "DeviceCapacityTest TearDown per test" << std::endl;
        GlobalMockObject::verify();
    }
};

// GetBandWidthPerNPU: level=1 且 910B 16p 单server 场景(deviceNumPerAggregation*2)
TEST_F(DeviceCapacityTest, ut_GetBandWidthPerNPU_910B_16p_Expect_Success)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(SetDevType910B));
    float bandWidth = 0.0f;
    HcclResult ret = GetBandWidthPerNPU(1, 16, 8, bandWidth);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_GT(bandWidth, 0.0f);
}

// GetBandWidthPerNPU: 查表命中(level 0, 910)
TEST_F(DeviceCapacityTest, ut_GetBandWidthPerNPU_910_level0_Expect_Success)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(SetDevType910));
    float bandWidth = 0.0f;
    HcclResult ret = GetBandWidthPerNPU(0, 8, 8, bandWidth);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_GT(bandWidth, 0.0f);
}

// IsUseSdidForDeviceId: 910_93 且 superDeviceId 有效
TEST_F(DeviceCapacityTest, ut_IsUseSdidForDeviceId_91093_Expect_True)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(SetDevType91093));
    EXPECT_TRUE(IsUseSdidForDeviceId(1));
}

// GetNotifyMaxWaitTime: 首次调用触发设备类型判断(910_93)
TEST_F(DeviceCapacityTest, ut_GetNotifyMaxWaitTime_91093_Expect_GreaterThanZero)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(SetDevType91093));
    u32 waitTime = GetNotifyMaxWaitTime();
    EXPECT_GT(waitTime, 0u);
}
