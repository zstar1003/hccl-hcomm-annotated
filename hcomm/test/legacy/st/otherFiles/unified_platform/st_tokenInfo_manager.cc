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
#include "tokenInfo_manager.h"
#include "orion_adapter_hccp.h"

using namespace Hccl;

// Test fixture for HcclNetDev tests
class HcclTokenInfoManagerTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

// Test case for GetTokenInfo method
TEST(HcclTokenInfoManagerTest, GetTokenInfo)
{
    RdmaHandle rdmahandle;
    TokenInfoManager tokenInfoManager(0, rdmahandle);
    std::pair<TokenIdHandle, uint32_t> tokenInfo{1, 1};
    MOCKER(RaUbAllocTokenIdHandle).stubs().will(returnValue(tokenInfo));
    MOCKER(RaUbFreeTokenIdHandle).stubs();

    std::pair<TokenIdHandle, uint32_t> outTokenInfo = tokenInfoManager.GetTokenInfo(BufferKey<uintptr_t, u64>{0, 0});
    EXPECT_EQ(tokenInfo, outTokenInfo);
    EXPECT_EQ(tokenInfo, outTokenInfo);

    EXPECT_NO_THROW(tokenInfoManager.Destroy());
}
