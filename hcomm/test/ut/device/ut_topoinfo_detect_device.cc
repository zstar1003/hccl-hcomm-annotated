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
#include <memory>
#include <vector>
#include <string>

#ifndef private
#define private public
#define protected public
#endif
#include "topoinfo_detect.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

class TopoInfoDetectDeviceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TopoInfoDetectDeviceTest SetUpTestCase" << std::endl; }
    static void TearDownTestCase() { std::cout << "TopoInfoDetectDeviceTest TearDownTestCase" << std::endl; }
    virtual void SetUp() { std::cout << "TopoInfoDetectDeviceTest SetUp" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "TopoInfoDetectDeviceTest TearDown" << std::endl;
    }
};

TEST_F(TopoInfoDetectDeviceTest, Ut_GroupLeaderListen_ReturnIsHCCL_E_NOT_SUPPORT)
{
    TopoInfoDetect topoDetect;
    HcclRankHandle rankHandle;
    std::vector<HcclIpAddress> whitelist;
    HcclResult ret = topoDetect.GroupLeaderListen(rankHandle, whitelist);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TopoInfoDetectDeviceTest, Ut_StartGroupLeaderNetwork_ReturnIsHCCL_E_NOT_SUPPORT)
{
    TopoInfoDetect topoDetect;
    std::vector<HcclIpAddress> whitelist;
    HcclIpAddress hostIP;
    u32 bindPort = 0;
    std::vector<HcclSocketPortRange> portRanges;
    HcclResult ret = topoDetect.StartGroupLeaderNetwork(whitelist, hostIP, bindPort, portRanges);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TopoInfoDetectDeviceTest, Ut_CheckHostNicLinkUp_ReturnIsHCCL_E_NOT_SUPPORT)
{
    const TopoInfoDetect topoDetect;
    const HcclIpAddress hostIP;

    EXPECT_EQ(topoDetect.CheckHostNicLinkUp(hostIP), HCCL_E_NOT_SUPPORT);
}
