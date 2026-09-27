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

#ifndef private
#define private public
#define protected public
#endif
#include "hccl_communicator.h"
#include "ccl_buffer_manager.h"
#include "dispatcher.h"
#include "hccl_ip_address.h"
#include "externalinput_pub.h"
#include "common.h"
#undef private
#undef protected
#include "llt_hccl_stub_pub.h"

using namespace std;
using namespace hccl;

class IndOpTransportAllocTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "IndOpTransportAllocTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "IndOpTransportAllocTest TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "IndOpTransportAllocTest Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "IndOpTransportAllocTest Test TearDown" << std::endl;
    }
};

TEST_F(
    IndOpTransportAllocTest, Ut_IndOpTransportAlloc_When_AicpuModeWithUserDeviceMem_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    std::string tag = "test_tag";
    OpCommTransport opCommTransport;
    TransportIOMem transMem;
    transMem.indOpMem.userDeviceMem.push_back(DeviceMem());
    bool isAicpuModeEn = true;

    HcclResult ret = hcclCommunicator->IndOpTransportAlloc(tag, opCommTransport, transMem, isAicpuModeEn);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(IndOpTransportAllocTest, Ut_IndOpTransportAlloc_When_AicpuModeWithUserHostMem_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    std::string tag = "test_tag";
    OpCommTransport opCommTransport;
    TransportIOMem transMem;
    transMem.indOpMem.userHostMem.push_back(HostMem());
    bool isAicpuModeEn = true;

    HcclResult ret = hcclCommunicator->IndOpTransportAlloc(tag, opCommTransport, transMem, isAicpuModeEn);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(IndOpTransportAllocTest, Ut_IndOpTransportAlloc_When_ManagerIsNull_Expect_ReturnHCCL_E_PTR)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    hcclCommunicator->indptOpTransportManager_ = nullptr;
    std::string tag = "test_tag";
    OpCommTransport opCommTransport;
    TransportIOMem transMem;
    bool isAicpuModeEn = false;

    HcclResult ret = hcclCommunicator->IndOpTransportAlloc(tag, opCommTransport, transMem, isAicpuModeEn);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(IndOpTransportAllocTest, Ut_IndOpTransportAlloc_When_AllocFailed_Expect_ReturnHCCL_E_PTR)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());

    CCLBufferManager cclBufferManager;
    std::unique_ptr<HcclSocketManager> socketManager = nullptr;
    HcclDispatcher dispatcher = nullptr;
    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    std::vector<RankInfo> rankInfoList;
    std::vector<u32> nicRanksPort, vnicRanksPort;
    std::vector<HcclIpAddress> devIpAddr;
    HcclIpAddress hostIp, localVnicIp;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;

    hcclCommunicator->indptOpTransportManager_.reset(new (std::nothrow) TransportManager(
        cclBufferManager, socketManager, dispatcher, notifyPool, rankInfoList, 0, "test_tag", 0,
        NICDeployment::NIC_DEPLOYMENT_DEVICE, false, false, false, nicRanksPort, vnicRanksPort, false, devIpAddr,
        hostIp, localVnicIp, netDevCtxMap));

    std::string tag = "test_tag";
    OpCommTransport opCommTransport;
    TransportIOMem transMem;
    bool isAicpuModeEn = false;

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_E_PTR));

    HcclResult ret = hcclCommunicator->IndOpTransportAlloc(tag, opCommTransport, transMem, isAicpuModeEn);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
