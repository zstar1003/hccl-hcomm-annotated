/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_JETTY_TEST_COMMON_H
#define CCU_JETTY_TEST_COMMON_H

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include "ccu_jetty.h"
#include "rdma_handle_manager.h"
#include "local_ub_rma_buffer.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "ccu_dev_mgr.h"

using namespace Hccl;

class CcuJettyTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuJettyTest tests set up." << std::endl; }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "CcuJettyTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuJettyTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuJettyTest TearDown" << std::endl;
    }
};

inline void MockCcuJetty()
{
    MOCKER(HrtGetDevice).stubs().will(returnValue(10));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(10));
    MOCKER_CPP(&RdmaHandleManager::GetByIp).stubs().will(returnValue((void*)0x12345678));
    MOCKER_CPP(&RdmaHandleManager::GetJfcHandle).stubs().will(returnValue((JfcHandle)0x12345678));
    const auto fakeTokenInfo = std::make_pair<TokenIdHandle, uint32_t>(0x1, 0x1);
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(fakeTokenInfo));
    MOCKER(GetUbToken).stubs().will(returnValue(0x1));
}

#endif // CCU_JETTY_TEST_COMMON_H
