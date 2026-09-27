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
#include "hccl_net_dev_v2.h"
#include "orion_adapter_hccp.h"
#include "new/hccl_net_dev_defs.h"

// Test suite class
class HcclNetDevV2Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclNetDevV2Test SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HcclNetDevV2Test TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in HcclNetDevV2Test SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HcclNetDevV2Test TearDown" << std::endl;
    }
};

// 测试HcclNetDevOpenV2
TEST(HcclNetDevV2Test, ReturnsSuccessWhenDeviceAddedSuccessfully)
{
    MOCKER(Hccl::HrtRaRdmaInit, Hccl::RdmaHandle(Hccl::HrtNetworkMode, Hccl::RaInterface&))
        .stubs()
        .will(returnValue(Hccl::RdmaHandle()));
    HcclNetDev netDev = nullptr;
    HcclNetDevInfos info;
    info.devicePhyId = 0;
    info.addr.type = HcclAddressType::HCCL_ADDR_TYPE_IP_V4;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_BUS;
    info.netdevDeployment = HCCL_NETDEV_DEPLOYMENT_HOST;
    EXPECT_EQ(HcclNetDevOpenV2(&info, &netDev), HCCL_SUCCESS);
    EXPECT_EQ(HcclNetDevCloseV2(netDev), HCCL_SUCCESS);
}

TEST(HcclNetDevV2Test, ReturnsSuccessWhenDeviceAddedSuccessfully_1)
{
    MOCKER(Hccl::HrtRaRdmaInit, Hccl::RdmaHandle(Hccl::HrtNetworkMode, Hccl::RaInterface&))
        .stubs()
        .will(returnValue(Hccl::RdmaHandle()));
    HcclNetDev netDev = nullptr;
    HcclNetDevInfos info;
    info.devicePhyId = 0;
    info.addr.type = HcclAddressType::HCCL_ADDR_TYPE_RESERVED;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_BUS;
    info.netdevDeployment = HCCL_NETDEV_DEPLOYMENT_HOST;
    EXPECT_NE(HcclNetDevOpenV2(&info, &netDev), HCCL_SUCCESS);
}

// 测试HcclNetDevGetAddrV2
TEST(HcclNetDevV2Test, ReturnsSuccessWhenAddressIsIPv4)
{
    HcclAddress addr;
    MOCKER(Hccl::HrtRaRdmaInit, Hccl::RdmaHandle(Hccl::HrtNetworkMode, Hccl::RaInterface&))
        .stubs()
        .will(returnValue(Hccl::RdmaHandle()));
    HcclNetDev netDev = nullptr;
    HcclNetDevInfos info;
    info.devicePhyId = 0;
    info.addr.type = HcclAddressType::HCCL_ADDR_TYPE_IP_V6;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_TCP;
    info.netdevDeployment = HCCL_NETDEV_DEPLOYMENT_HOST;
    EXPECT_EQ(HcclNetDevOpenV2(&info, &netDev), HCCL_SUCCESS);
    EXPECT_EQ(HcclNetDevGetAddrV2(netDev, &addr), HCCL_SUCCESS);
    EXPECT_EQ(addr.type, HcclAddressType::HCCL_ADDR_TYPE_IP_V6);
    EXPECT_EQ(HcclNetDevCloseV2(netDev), HCCL_SUCCESS);
}

TEST(HcclNetDevV2Test, HcclNetDevGetNicAddrV2test)
{
    HcclAddress busAddr;
    HcclAddress* addr;
    uint32_t addrNum = 0;
    int32_t devicePhyId = 0;
    EXPECT_EQ(HcclNetDevGetNicAddrV2(devicePhyId, &addr, &addrNum), HCCL_E_NOT_SUPPORT);
}
