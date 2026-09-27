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
#include "inner_net_dev_manager.h"
#include "net_device.h"
#include "hccl_net_dev.h"

// Test fixture for HcclNetDev tests
class HcclNetDevTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        // Initialize NetDevInfo with some default values
        netDevInfo_.rankId = 0;
        netDevInfo_.type = Hccl::PortDeploymentType::DEV_NET;
        netDevInfo_.protoType = Hccl::LinkProtoType::RDMA;
        netDevInfo_.devId = 0;
        netDevInfo_.addr = Hccl::IpAddress("1.0.0.0");

        // Create an HcclNetDev object with the NetDevInfo
        hcclNetDev_ = new Hccl::HcclNetDevice(netDevInfo_);
    }

    void TearDown() override { delete hcclNetDev_; }

    Hccl::NetDevInfo netDevInfo_;
    Hccl::HcclNetDevice* hcclNetDev_;
};

// Test case for GetNetDevInfo method
TEST_F(HcclNetDevTest, GetNetDevInfoReturnsCorrectInfo)
{
    Hccl::NetDevInfo info = hcclNetDev_->GetNetDevInfo();
    EXPECT_EQ(info.addr, netDevInfo_.addr);
}

// Test case for SetInnerNetDev and GetInnerNetDev methods
TEST_F(HcclNetDevTest, SetAndGetInnerNetDevWorksCorrectly)
{
    MOCKER(Hccl::HrtRaRdmaInit).stubs().will(returnValue(Hccl::RdmaHandle()));
    Hccl::NetDevInfo info
        = {0, Hccl::PortDeploymentType::DEV_NET, Hccl::LinkProtoType::RDMA, 0, Hccl::IpAddress("1.0.0.0")};
    Hccl::InnerNetDev netDev(info);
    hcclNetDev_->SetInnerNetDev(&netDev);
    EXPECT_EQ(hcclNetDev_->GetInnerNetDev(), &netDev);
}

// Test case for GetRdmaHandle method when InnerNetDev is set
TEST_F(HcclNetDevTest, GetRdmaHandleReturnsHandleWhenInnerNetDevIsSet)
{
    MOCKER(Hccl::HrtRaRdmaInit).stubs().will(returnValue(Hccl::RdmaHandle()));
    Hccl::NetDevInfo info
        = {0, Hccl::PortDeploymentType::DEV_NET, Hccl::LinkProtoType::RDMA, 0, Hccl::IpAddress("1.0.0.0")};
    Hccl::InnerNetDev netDev(info);
    Hccl::RdmaHandle expectedHandle;
    hcclNetDev_->SetInnerNetDev(&netDev);
}

TEST_F(HcclNetDevTest, HcclNetDevGetBusAddrtest)
{
    HcclAddress busAddr;
    HcclDeviceId testId;
    testId.devicePhyId = 0;
    EXPECT_EQ(HcclNetDevGetBusAddr(testId, &busAddr), HCCL_E_NOT_SUPPORT);
}
