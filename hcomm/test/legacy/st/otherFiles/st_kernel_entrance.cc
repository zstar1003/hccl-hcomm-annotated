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
#include "kernel_entrance.h"
#include "communicator_impl_lite_manager.h"
#include "communicator_impl_lite.h"
#undef private

using namespace Hccl;

TEST(KernelEntranceTest, test_hccl_kernel_entrance_with_nullptr)
{
    auto res = HcclKernelEntrance(nullptr);
    EXPECT_EQ(1, res);
}

TEST(KernelEntranceTest, test_hccl_kernel_entrance_with_valid_param)
{
    HcclKernelParamLite param;

    MOCKER_CPP(&CommunicatorImplLite::LoadWithOpBasedMode).stubs().with(mockcpp::any()).will(returnValue(1));

    auto res = HcclKernelEntrance(&param);
    EXPECT_EQ(1, res);
    GlobalMockObject::verify();
}

TEST(UpdateCommKernelEntranceTest, test_hccl_update_comm_kernel_entrance_with_nullptr)
{
    auto res = HcclUpdateCommKernelEntrance(nullptr);
    EXPECT_EQ(1, res);
}

TEST(UpdateCommKernelEntranceTest, test_hccl_update_comm_kernel_entrance_with_valid_param)
{
    HcclKernelParamLite param;

    MOCKER_CPP(&CommunicatorImplLite::UpdateComm).stubs().with(mockcpp::any()).will(returnValue(1));

    auto res = HcclUpdateCommKernelEntrance(&param);
    EXPECT_EQ(0, res);
    GlobalMockObject::verify();
}
