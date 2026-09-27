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
#include "inner_net_dev.h"

using namespace testing;

// Test suite class
class InnerNetDevTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InnerNetDevTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "InnerNetDevTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in InnerNetDevTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in InnerNetDevTest TearDown" << std::endl;
    }
};
/**
 * @tc.name  : InnerNetDev_ShouldInitializeMembers_WhenConstructedWithNetDevInfo
 * @tc.number: InnerNetDev_Test_001
 * @tc.desc  : Test if the InnerNetDev constructor initializes members correctly when constructed with NetDevInfo
 */
TEST_F(InnerNetDevTest, InnerNetDev_ShouldInitializeMembers_WhenConstructedWithNetDevInfo)
{
    MOCKER(Hccl::HrtRaRdmaInit).stubs().will(returnValue(Hccl::RdmaHandle()));
    Hccl::NetDevInfo info
        = {0, Hccl::PortDeploymentType::DEV_NET, Hccl::LinkProtoType::RDMA, 0, Hccl::IpAddress("1.0.0.0")};
    Hccl::InnerNetDev netDev(info);

    EXPECT_EQ(netDev.getRdmaHandle(), Hccl::RdmaHandle());
    EXPECT_EQ(netDev.getUbMode(), Hccl::HrtUbJfcMode());
    EXPECT_EQ(netDev.getDieId(), 0);
    EXPECT_EQ(netDev.getFuncId(), 0);
    EXPECT_EQ(netDev.getTokenHandle(), Hccl::TokenIdHandle());
    EXPECT_EQ(netDev.getTokenId(), 0);
}

/**
 * @tc.name  : InnerNetDev_ShouldSetAndReturnRdmaHandle_WhenSetRdmaHandleCalled
 * @tc.number: InnerNetDev_Test_002
 * @tc.desc  : Test if the InnerNetDev set and return RdmaHandle correctly
 */
TEST_F(InnerNetDevTest, InnerNetDev_ShouldSetAndReturnRdmaHandle_WhenSetRdmaHandleCalled)
{
    MOCKER(Hccl::HrtRaRdmaInit).stubs().will(returnValue(Hccl::RdmaHandle()));
    Hccl::NetDevInfo info
        = {0, Hccl::PortDeploymentType::DEV_NET, Hccl::LinkProtoType::RDMA, 0, Hccl::IpAddress("1.0.0.0")};
    Hccl::InnerNetDev netDev(info);

    Hccl::RdmaHandle handle;
    netDev.setRdmaHandle(handle);
    EXPECT_EQ(netDev.getRdmaHandle(), handle);
}

/**
 * @tc.name  : InnerNetDev_ShouldSetAndReturnUbMode_WhenSetUbModeCalled
 * @tc.number: InnerNetDev_Test_003
 * @tc.desc  : Test if the InnerNetDev set and return UbMode correctly
 */
TEST_F(InnerNetDevTest, InnerNetDev_ShouldSetAndReturnUbMode_WhenSetUbModeCalled)
{
    MOCKER(Hccl::HrtRaRdmaInit).stubs().will(returnValue(Hccl::RdmaHandle()));
    Hccl::NetDevInfo info
        = {0, Hccl::PortDeploymentType::DEV_NET, Hccl::LinkProtoType::RDMA, 0, Hccl::IpAddress("1.0.0.0")};
    Hccl::InnerNetDev netDev(info);

    Hccl::HrtUbJfcMode mode;
    netDev.setUbMode(mode);
    EXPECT_EQ(netDev.getUbMode(), mode);
}

/**
 * @tc.name  : InnerNetDev_ShouldSetAndReturnDieId_WhenSetDieIdCalled
 * @tc.number: InnerNetDev_Test_004
 * @tc.desc  : Test if the InnerNetDev set and return DieId correctly
 */
TEST_F(InnerNetDevTest, InnerNetDev_ShouldSetAndReturnDieId_WhenSetDieIdCalled)
{
    MOCKER(Hccl::HrtRaRdmaInit).stubs().will(returnValue(Hccl::RdmaHandle()));
    Hccl::NetDevInfo info
        = {0, Hccl::PortDeploymentType::DEV_NET, Hccl::LinkProtoType::RDMA, 0, Hccl::IpAddress("1.0.0.0")};
    Hccl::InnerNetDev netDev(info);

    uint32_t dieId = 123;
    netDev.setDieId(dieId);
    EXPECT_EQ(netDev.getDieId(), dieId);
}

/**
 * @tc.name  : InnerNetDev_ShouldSetAndReturnFuncId_WhenSetFuncIdCalled
 * @tc.number: InnerNetDev_Test_005
 * @tc.desc  : Test if the InnerNetDev set and return FuncId correctly
 */
TEST_F(InnerNetDevTest, InnerNetDev_ShouldSetAndReturnFuncId_WhenSetFuncIdCalled)
{
    MOCKER(Hccl::HrtRaRdmaInit).stubs().will(returnValue(Hccl::RdmaHandle()));
    Hccl::NetDevInfo info
        = {0, Hccl::PortDeploymentType::DEV_NET, Hccl::LinkProtoType::RDMA, 0, Hccl::IpAddress("1.0.0.0")};
    Hccl::InnerNetDev netDev(info);

    uint32_t funcId = 456;
    netDev.setFuncId(funcId);
    EXPECT_EQ(netDev.getFuncId(), funcId);
}

/**
 * @tc.name  : InnerNetDev_ShouldSetAndReturnTokenHandle_WhenSetTokenHandleCalled
 * @tc.number: InnerNetDev_Test_006
 * @tc.desc  : Test if the InnerNetDev set and return TokenHandle correctly
 */
TEST_F(InnerNetDevTest, InnerNetDev_ShouldSetAndReturnTokenHandle_WhenSetTokenHandleCalled)
{
    MOCKER(Hccl::HrtRaRdmaInit).stubs().will(returnValue(Hccl::RdmaHandle()));
    Hccl::NetDevInfo info
        = {0, Hccl::PortDeploymentType::DEV_NET, Hccl::LinkProtoType::RDMA, 0, Hccl::IpAddress("1.0.0.0")};
    Hccl::InnerNetDev netDev(info);

    Hccl::TokenIdHandle tokenHandle;
    netDev.setTokenHandle(tokenHandle);
    EXPECT_EQ(netDev.getTokenHandle(), tokenHandle);
}

/**
 * @tc.name  : InnerNetDev_ShouldSetAndReturnTokenId_WhenSetTokenIdCalled
 * @tc.number: InnerNetDev_Test_007
 * @tc.desc  : Test if the InnerNetDev set and return TokenId correctly
 */
TEST_F(InnerNetDevTest, InnerNetDev_ShouldSetAndReturnTokenId_WhenSetTokenIdCalled)
{
    MOCKER(Hccl::HrtRaRdmaInit).stubs().will(returnValue(Hccl::RdmaHandle()));
    Hccl::NetDevInfo info
        = {0, Hccl::PortDeploymentType::DEV_NET, Hccl::LinkProtoType::RDMA, 0, Hccl::IpAddress("1.0.0.0")};
    Hccl::InnerNetDev netDev(info);

    uint32_t tokenId = 789;
    netDev.setTokenId(tokenId);
    EXPECT_EQ(netDev.getTokenId(), tokenId);
}

/**
 * @tc.name  : InnerNetDev_ShouldReturnJfcHandle_WhenGetJfcHandleCalled
 * @tc.number: InnerNetDev_Test_008
 * @tc.desc  : Test if the InnerNetDev set and return JfcHandle correctly
 */
TEST_F(InnerNetDevTest, InnerNetDev_ShouldReturnJfcHandle_WhenGetJfcHandleCalled)
{
    MOCKER(Hccl::HrtRaRdmaInit).stubs().will(returnValue(Hccl::RdmaHandle()));
    Hccl::JfcHandle jfcHandle = 1;
    MOCKER(Hccl::HrtRaUbCreateJfc).stubs().will(returnValue(jfcHandle));
    Hccl::NetDevInfo info
        = {0, Hccl::PortDeploymentType::DEV_NET, Hccl::LinkProtoType::RDMA, 0, Hccl::IpAddress("1.0.0.0")};
    Hccl::InnerNetDev netDev(info);

    netDev.setRdmaHandle(&netDev);
    Hccl::JfcHandle jfcHandleOut = netDev.getUbJfcHandle(Hccl::HrtUbJfcMode::NORMAL);
    EXPECT_EQ(jfcHandle, jfcHandleOut);

    netDev.setRdmaHandle(nullptr);
    EXPECT_THROW(netDev.getUbJfcHandle(Hccl::HrtUbJfcMode::NORMAL), Hccl::NullPtrException);
}
