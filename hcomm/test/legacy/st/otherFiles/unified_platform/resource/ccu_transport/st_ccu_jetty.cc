/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_jetty_test_common.h"

TEST_F(CcuJettyTest, St_CreateJetty_When_InterfaceOk_Expect_Return_Ok)
{
    MockCcuJetty();
    IpAddress fakeIp{};
    CcuJettyInfo fakeJettyInfo{};
    CcuJetty ccuJetty(fakeIp, fakeJettyInfo);
    EXPECT_EQ(ccuJetty.CreateJetty(), HcclResult::HCCL_E_AGAIN);
    EXPECT_EQ(ccuJetty.CreateJetty(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ccuJetty.CreateJetty(), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuJettyTest, St_CreateJetty_When_InterfaceFailed_Expect_Return_Error)
{
    MOCKER(RaUbCreateJettyAsync).stubs().will(throws(InternalException("")));
    MockCcuJetty();
    IpAddress fakeIp{};
    CcuJettyInfo fakeJettyInfo{};
    CcuJetty ccuJetty(fakeIp, fakeJettyInfo);
    EXPECT_EQ(ccuJetty.CreateJetty(), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(ccuJetty.CreateJetty(), HcclResult::HCCL_E_INTERNAL);
}
