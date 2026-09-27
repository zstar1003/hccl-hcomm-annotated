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

#include "remote_rma_buf_manager.h"
#include "rma_conn_manager.h"
#include "rma_connection.h"
#include "communicator_impl.h"
#include "invalid_params_exception.h"
#include "rdma_handle_manager.h"
#include "internal_exception.h"
using namespace Hccl;
class RemoteRmaBufManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RemoteRmaBufManagerTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RemoteRmaBufManagerTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in RemoteRmaBufManagerTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RemoteRmaBufManagerTest TearDown" << std::endl;
    }
};

TEST_F(RemoteRmaBufManagerTest, create_link_data_is_p2p)
{
    CommunicatorImpl communicator;
    RemoteRmaBufManager remoteRmaBufManager(communicator);
    BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(basePortType, 0, 1, 0, 0);

    unique_ptr<RemoteRmaBuffer> remoteRmaBuffer = remoteRmaBufManager.Create(linkData);
    EXPECT_EQ(RmaType::IPC, remoteRmaBuffer->GetRmaType());
}

TEST_F(RemoteRmaBufManagerTest, create_link_data_is_RDMA)
{
    CommunicatorImpl communicator;
    RemoteRmaBufManager remoteRmaBufManager(communicator);
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData linkData(basePortType, 0, 1, 0, 0);

    void* rdmaHandle = (void*)0x100;

    MOCKER_CPP(&RdmaHandleManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));

    unique_ptr<RemoteRmaBuffer> remoteRmaBuffer = remoteRmaBufManager.Create(linkData);
    EXPECT_EQ(RmaType::RDMA, remoteRmaBuffer->GetRmaType());
}

TEST_F(RemoteRmaBufManagerTest, create_link_data_is_TCP)
{
    CommunicatorImpl communicator;
    RemoteRmaBufManager remoteRmaBufManager(communicator);
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::TCP);
    LinkData linkData(basePortType, 0, 1, 0, 0);

    EXPECT_THROW(remoteRmaBufManager.Create(linkData), InternalException);
}

TEST_F(RemoteRmaBufManagerTest, create_link_data_is_UB)
{
    CommunicatorImpl communicator;
    RemoteRmaBufManager remoteRmaBufManager(communicator);
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(basePortType, 0, 1, 0, 0);

    void* rdmaHandle = (void*)0x300;

    MOCKER_CPP(&RdmaHandleManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));

    unique_ptr<RemoteRmaBuffer> remoteRmaBuffer = remoteRmaBufManager.Create(linkData);
    EXPECT_EQ(RmaType::UB, remoteRmaBuffer->GetRmaType());
}

TEST_F(RemoteRmaBufManagerTest, get_remote_rma_buffer_success)
{
    CommunicatorImpl communicator;
    RemoteRmaBufManager remoteRmaBufManager(communicator);

    void* rdmaHandle = (void*)0x300;
    auto remoteUbRmaBuffer = std::make_unique<RemoteUbRmaBuffer>(rdmaHandle);
    RemoteRmaBuffer* bufRawPtr = remoteUbRmaBuffer.get();
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(basePortType, 0, 1, 0, 0);
    string tag = "test";
    BufferType bufType = BufferType::SCRATCH;

    remoteRmaBufManager.Bind(std::move(remoteUbRmaBuffer), tag, linkData, bufType);

    EXPECT_EQ(remoteRmaBufManager.GetRemoteRmaBuffer(tag, linkData, bufType), bufRawPtr);
}
