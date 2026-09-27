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
#include "communicator_impl.h"
#include "local_rma_buf_manager.h"
#include "rdma_handle_manager.h"
#include "invalid_params_exception.h"
#include "internal_exception.h"
#include "dev_buffer.h"
#include "rma_buffer.h"
#undef protected
#undef private
using namespace Hccl;

class LocalRmaBufManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LocalRmaBufManagerTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "LocalRmaBufManagerTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtIpcSetMemoryName).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
        MOCKER(HrtDevMemAlignWithPage)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
        MOCKER(HrtIpcDestroyMemoryName).stubs().with(mockcpp::any());
        MOCKER(GetUbToken).stubs().will(returnValue(1));

        devBuffer = DevBuffer::Create(0x100, 0x100);
        std::cout << "A Test case in LocalRmaBufManagerTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in LocalRmaBufManagerTest TearDown" << std::endl;
    }

    shared_ptr<DevBuffer> devBuffer;
    BufferType bufferType = BufferType::SCRATCH;
};

TEST_F(LocalRmaBufManagerTest, get_return_null_ptr)
{
    CommunicatorImpl comm;
    LocalRmaBufManager localRmaBufManager(comm);
    BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::RDMA);
    PortData port(0, basePortType, 0, IpAddress());

    auto res = localRmaBufManager.Get("opTag", port, bufferType);
    EXPECT_EQ(nullptr, res);
}

TEST_F(LocalRmaBufManagerTest, reg_invalid_port)
{
    CommunicatorImpl comm;
    LocalRmaBufManager localRmaBufManager(comm);
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::TCP);
    PortData port(0, basePortType, 0, IpAddress());
    string opTag = "optag";

    EXPECT_THROW(localRmaBufManager.Reg(opTag, bufferType, devBuffer, port, LinkProtocol::UB_CTP), InternalException);
}

TEST_F(LocalRmaBufManagerTest, reg_port_ub_first_time_get_then_second_throw)
{
    CommunicatorImpl comm;
    LocalRmaBufManager localRmaBufManager(comm);
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    PortData port(0, basePortType, 0, IpAddress());
    string opTag = "optag";

    void* rdmaHandle = (void*)0x200;

    MOCKER(HrtRaUbCtxInit).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));
    MOCKER_CPP(&RdmaHandleManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));
    RdmaHandleManager::GetInstance().tokenInfoMap[rdmaHandle] = make_unique<TokenInfoManager>(0, rdmaHandle);

    auto res = localRmaBufManager.Reg(opTag, bufferType, devBuffer, port, LinkProtocol::UB_CTP);
    EXPECT_NE(nullptr, res);
    EXPECT_EQ(RmaType::UB, res->GetRmaType());

    auto res2 = localRmaBufManager.Get(opTag, port, bufferType);
    EXPECT_EQ(res, res2);
    localRmaBufManager.Destroy();
}

TEST_F(LocalRmaBufManagerTest, reg_port_ub_first_time_get_then_second_no_throw_aicpu)
{
    MOCKER(HrtUbDevQueryInfo).stubs().with(mockcpp::any(), mockcpp::any());

    CommunicatorImpl comm;
    LocalRmaBufManager localRmaBufManager(comm);
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    PortData port(0, basePortType, 0, IpAddress());
    string opTag = "optag";

    void* rdmaHandle = (void*)0x200;

    MOCKER(HrtRaUbCtxInit).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));
    MOCKER_CPP(&RdmaHandleManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));

    auto res = localRmaBufManager.Reg(opTag, bufferType, devBuffer, port, LinkProtocol::UB_CTP);
    EXPECT_NE(nullptr, res);
    EXPECT_EQ(RmaType::UB, res->GetRmaType());
    auto res2 = localRmaBufManager.Get(opTag, port, bufferType);
    EXPECT_EQ(res, res2);

    // 重复注册逻辑修改，不再抛异常，而是返回注册好的资源
    EXPECT_NO_THROW(localRmaBufManager.Reg(opTag, bufferType, devBuffer, port, LinkProtocol::UB_CTP));
    auto res3 = localRmaBufManager.Get(opTag, port, bufferType);
    EXPECT_EQ(res, res3);
}

// 覆盖 GetAivUrmaBufferTag 纯函数：返回 commId + "_aiv_urma" 后缀，用于隔离 fake 注册与真实 CCL buffer
TEST_F(LocalRmaBufManagerTest, get_aiv_urma_buffer_tag_appends_suffix)
{
    EXPECT_EQ("comm123_aiv_urma", GetAivUrmaBufferTag("comm123"));
    EXPECT_EQ("_aiv_urma", GetAivUrmaBufferTag(""));
}

// 覆盖 Reg UB 分支新增条件 || comm->GetOpAivFeatureFlag()：AIV flag 开启时走 LocalUbRmaBuffer(buffer) 无 rdmaHandle
// 路径
TEST_F(LocalRmaBufManagerTest, reg_port_ub_aiv_flag_on_uses_local_ub_rma_buffer_without_rdma_handle)
{
    MOCKER(HrtUbDevQueryInfo).stubs().with(mockcpp::any(), mockcpp::any());

    CommunicatorImpl comm;
    comm.opExecuteConfig.accState = AcceleratorState::AIV; // 开启 AIV flag
    LocalRmaBufManager localRmaBufManager(comm);
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    PortData port(0, basePortType, 0, IpAddress());
    string opTag = GetAivUrmaBufferTag("aiv_comm");

    // AIV flag on 时不应调用 HrtRaUbCtxInit（走无 rdmaHandle 路径）
    auto res = localRmaBufManager.Reg(opTag, bufferType, devBuffer, port, LinkProtocol::UB_CTP);
    EXPECT_NE(nullptr, res);
    EXPECT_EQ(RmaType::UB, res->GetRmaType());

    // 用隔离后的 opTag 能 Get 到同一对象
    auto res2 = localRmaBufManager.Get(opTag, port, bufferType);
    EXPECT_EQ(res, res2);
    localRmaBufManager.Destroy();
}

// 覆盖 Reg UB 分支条件为 false（AIV/AICPU flag 均关闭）时走带 rdmaHandle 的 else 路径
TEST_F(LocalRmaBufManagerTest, reg_port_ub_aiv_flag_off_uses_rdma_handle_path)
{
    CommunicatorImpl comm;
    // 默认 accState=CCU_MS，AIV/AICPU flag 均为 false
    EXPECT_FALSE(comm.GetOpAivFeatureFlag());
    EXPECT_FALSE(comm.GetOpAiCpuTSFeatureFlag());

    LocalRmaBufManager localRmaBufManager(comm);
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    PortData port(0, basePortType, 0, IpAddress());
    string opTag = "real_ccl_buffer";

    void* rdmaHandle = (void*)0x300;
    MOCKER(HrtRaUbCtxInit).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));
    MOCKER_CPP(&RdmaHandleManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));
    RdmaHandleManager::GetInstance().tokenInfoMap[rdmaHandle] = make_unique<TokenInfoManager>(0, rdmaHandle);

    auto res = localRmaBufManager.Reg(opTag, bufferType, devBuffer, port, LinkProtocol::UB_CTP);
    EXPECT_NE(nullptr, res);
    EXPECT_EQ(RmaType::UB, res->GetRmaType());
    localRmaBufManager.Destroy();
}
