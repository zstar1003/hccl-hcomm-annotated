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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hcomm_primitives.h"
#include "host_cpu_roce_channel.h"
#include "hcclCommOp.h"

using namespace hccl;

class UtCpuHcommDpuChannelRegisterDfx : public testing::Test {
protected:
    virtual void SetUp() override {}

    virtual void TearDown() override { GlobalMockObject::verify(); }

    EndpointHandle epHandle = reinterpret_cast<void*>(0x01);
    HcommChannelDesc channelDesc{};
    hcomm::HostCpuRoceChannel channelOnHost{epHandle, channelDesc};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&channelOnHost);
    int32_t res{HCCL_E_RESERVED};
    std::function<HcclResult(const Hccl::TaskParam&, u64)> callback = [](const Hccl::TaskParam&, u64) -> HcclResult {
        return HCCL_SUCCESS;
    };
};

TEST_F(UtCpuHcommDpuChannelRegisterDfx, Ut_HcommDpuChannelRegisterDfx_When_ChannelIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    res = HcommDpuChannelRegisterDfx(0, callback);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtCpuHcommDpuChannelRegisterDfx, Ut_HcommDpuChannelRegisterDfx_When_CallbackIsNull_Expect_ReturnIsHCCL_SUCCESS)
{
    std::function<HcclResult(const Hccl::TaskParam&, u64)> nullCallback{nullptr};
    MOCKER(&hcomm::HostCpuRoceChannel::SetDfxCallback).stubs().will(returnValue(HCCL_SUCCESS));
    res = HcommDpuChannelRegisterDfx(channel, nullCallback);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtCpuHcommDpuChannelRegisterDfx, Ut_HcommDpuChannelRegisterDfx_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER(&hcomm::HostCpuRoceChannel::SetDfxCallback).stubs().will(returnValue(HCCL_SUCCESS));
    res = HcommDpuChannelRegisterDfx(channel, callback);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(
    UtCpuHcommDpuChannelRegisterDfx, Ut_HcommDpuChannelRegisterDfx_When_SetDfxCallbackFails_Expect_ErrorCodePropagated)
{
    MOCKER(&hcomm::HostCpuRoceChannel::SetDfxCallback).stubs().will(returnValue(HCCL_E_INTERNAL));
    res = HcommDpuChannelRegisterDfx(channel, callback);
    EXPECT_EQ(res, HCCL_E_INTERNAL);
}
