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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#define protected public
#include "hccp_tlv_hdc_manager.h"
#include "orion_adapter_tsd.h"
#include "orion_adapter_rts.h"
#include "socket_handle_manager.h"
#include "rdma_handle_manager.h"
#include "hccp_tlv.h"
#undef protected
#undef private

using namespace testing;
using namespace Hccl;
// Test suite class
class HccpTlvHdcManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HccpTlvHdcManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HccpTlvHdcManagerTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in HccpTlvHdcManagerTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HccpTlvHdcManagerTest TearDown" << std::endl;
    }
};

TEST_F(HccpTlvHdcManagerTest, should_successfully_init_HccpTlvHdcManager)
{
    u32 deviceLogicId = 0;
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<DevId>(0)));
    MOCKER(RaTlvInit).stubs().will(returnValue(0));
    HccpTlvHdcManager::GetInstance().Init(deviceLogicId);
}
