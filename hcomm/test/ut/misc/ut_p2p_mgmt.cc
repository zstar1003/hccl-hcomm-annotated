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
#include <stdio.h>

#include "hccl_common.h"
#define private public
#include "p2p_mgmt/p2p_mgmt.h"
#undef private

using namespace std;
using namespace hccl;
class P2PMgmtTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--P2PMgmtTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--P2PMgmtTest TearDown--\033[0m" << std::endl; }
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(P2PMgmtTest, ut_BatchEnable_DisableP2P)
{
    HcclResult ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<uint32_t> remoteDevices = {0, 1, 2, 3, 4, 5, 6, 7};

    ret = P2PMgmtPub::EnableP2P(remoteDevices);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = P2PMgmtPub::WaitP2PEnabled(remoteDevices);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = P2PMgmtPub::DisableP2P(remoteDevices);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(P2PMgmtTest, ut_BatchEnable_DisableP2P_1)
{
    HcclResult ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    P2PMgmt::Instance().deviceType_ = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    std::vector<uint32_t> remoteDevices = {0, 1, 2, 3, 4, 5, 6, 7};

    ret = P2PMgmtPub::EnableP2P(remoteDevices);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = P2PMgmtPub::WaitP2PEnabled(remoteDevices);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = P2PMgmtPub::DisableP2P(remoteDevices);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}
