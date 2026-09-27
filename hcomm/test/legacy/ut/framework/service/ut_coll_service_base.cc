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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public

#include "coll_service_ai_cpu_impl.h"
#include "communicator_impl.h"
#include "mem_transport_manager.h"
#include "base_config_legacy.h"
#include "ub_mem_transport.h"
#include "internal_exception.h"

#undef protected
#undef private

#include <memory>

using namespace Hccl;

class CollServiceBaseTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CollServiceBaseTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CollServiceBaseTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in CollServiceBaseTest SetUp" << std::endl;
        // 初始化memTransportManager
        comm.InitMemTransportManager();
        // 向transport map添加transport
        unique_ptr<UbMemTransport> transportOpbase
            = make_unique<UbMemTransport>(locRes, attr, linkData, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
        comm.memTransportManager->opTagOpbasedMap[linkData] = std::move(transportOpbase);
        comm.memTransportManager->newOpbasedTransports[linkData] = 0;
        unique_ptr<UbMemTransport> transportOffload
            = make_unique<UbMemTransport>(locRes, attr, linkData, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
        comm.memTransportManager->opTagOffloadMap[opTag][linkData] = std::move(transportOffload);
        comm.memTransportManager->newOffloadTransports[opTag][linkData] = 0;

        // 打桩环境变量超时时间
        EnvSocketConfig envSocketConfig;
        EnvSocketConfig& fakeEnvSocketConfig = envSocketConfig;
        fakeEnvSocketConfig.linkTimeOut = CfgField<s32>{"HCCL_CONNECT_TIMEOUT", s32(1), Str2T<s32>};
        fakeEnvSocketConfig.linkTimeOut.isParsed = true;
        MOCKER_CPP(&EnvConfig::GetSocketConfig).stubs().will(returnValue(fakeEnvSocketConfig));
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in CollServiceBaseTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }

    CommunicatorImpl comm;
    std::string opTag = "test_tag";
    BasePortType portType{PortDeploymentType::DEV_NET, ConnectProtoType::UB};
    LinkData linkData{portType, 0, 1, 0, 1};
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    IpAddress ipAddress{"1.0.0.0"};
    Socket fakeSocket{nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE};
    RdmaHandle rdmaHandle = (void*)0x100;
    bool isRecvFirst = false;
};

TEST_F(CollServiceBaseTest, Ut_WaitOpbasedTransportReady_When_TransportReady_Expect_NoException)
{
    CollServiceAiCpuImpl collService(&comm);
    MOCKER_CPP(&MemTransportManager::IsAllOpbasedTransportReady).stubs().will(returnValue(true));

    EXPECT_NO_THROW(collService.WaitOpbasedTransportReady());
}

TEST_F(CollServiceBaseTest, Ut_WaitOpbasedTransportReady_When_Timeout_Expect_Exception)
{
    CollServiceAiCpuImpl collService(&comm);
    MOCKER_CPP(&MemTransportManager::IsAllOpbasedTransportReady).stubs().will(returnValue(false));

    EXPECT_THROW(collService.WaitOpbasedTransportReady(), InternalException);
}

TEST_F(CollServiceBaseTest, Ut_WaitOffloadTransportReady_When_TransportReady_Expect_NoException)
{
    CollServiceAiCpuImpl collService(&comm);
    MOCKER_CPP(&MemTransportManager::IsAllOffloadTransportReady).stubs().with(mockcpp::any()).will(returnValue(true));

    EXPECT_NO_THROW(collService.WaitOffloadTransportReady(opTag));
}

TEST_F(CollServiceBaseTest, Ut_WaitOffloadTransportReady_When_Timeout_Expect_Exception)
{
    CollServiceAiCpuImpl collService(&comm);
    MOCKER_CPP(&MemTransportManager::IsAllOffloadTransportReady).stubs().with(mockcpp::any()).will(returnValue(false));

    EXPECT_THROW(collService.WaitOffloadTransportReady(opTag), InternalException);
}
