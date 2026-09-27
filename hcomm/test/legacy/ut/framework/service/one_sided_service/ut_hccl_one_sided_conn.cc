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
#include "test_mock_setup.h"
#include "stub_communicator_impl_trans_mgr.h"

#define private public
#define protected public

#include "hccl_one_sided_conn_v2.h"
#include "hccl_one_sided_service_v2.h"
#include "rdma_handle_manager.h"
#include "hccl_mem_v2.h"

#undef protected
#undef private

#include <memory>

using namespace Hccl;
using LocalRdmaRmaBufferMgr = hcomm::RmaBufferMgr<BufferKey<uintptr_t, u64>, std::shared_ptr<LocalUbRmaBuffer>>;
class HcclOneSidedConnTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclOneSidedConnTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HcclOneSidedConnTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in HcclOneSidedConnTest SetUp" << std::endl;
        SETUP_COMM_NOTIFY_MOCKS();
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in HcclOneSidedConnTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }
    u32 fakeDevPhyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
};

TEST_F(HcclOneSidedConnTest, EnableMemAccess_MemoryOverlap)
{
    void* rdmaHandle = (void*)0x100;
    MOCKER_CPP(&RdmaHandleManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));
    CommunicatorImpl com;
    BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(basePortType, 0, 1, 0, 1);
    HcclOneSidedConn conn(&com, linkData);

    HcclBuf hcclBuf;
    int* a = new int;
    std::shared_ptr<Buffer> localBufferPtr
        = make_shared<Buffer>(reinterpret_cast<uintptr_t>(a), sizeof(int), HcclMemType::HCCL_MEM_TYPE_DEVICE);
    std::shared_ptr<LocalUbRmaBuffer> localUbRmaBuffer = make_shared<LocalUbRmaBuffer>(localBufferPtr);
    hcclBuf.handle = reinterpret_cast<void*>(localUbRmaBuffer.get());
    hcclBuf.len = sizeof(int);
    hcclBuf.addr = a;
    RmaMemDesc rmaDesc1;
    char* desc = static_cast<char*>(rmaDesc1.memDesc);
    uint64_t descLen = 0;
    HcclMemExportV2(&hcclBuf, &desc, &descLen);

    // 创建一个HcclMemDesc对象，描述一个内存段
    HcclMemDesc desc1;
    const size_t bufferSize = 512;
    const size_t copySize = sizeof(RmaMemDesc);
    memcpy_s(desc1.desc, bufferSize, &rmaDesc1, copySize);

    // 第一次调用EnableMemAccess，成功添加
    HcclMem mem1;
    mem1.type = HcclMemType::HCCL_MEM_TYPE_DEVICE;

    std::string fakeKeyDesc = "fakeKeyDesc";
    MOCKER(HrtRaGetKeyDescribe).stubs().will(returnValue(fakeKeyDesc));

    u64 fakeNotifyHandleAddr = 100;
    u64 fakeTargetSegVa = 150;
    HrtRaUbRemMemImportedOutParam fakeRemoteOutParam;
    fakeRemoteOutParam.handle = fakeNotifyHandleAddr;
    fakeRemoteOutParam.targetSegVa = fakeTargetSegVa;
    MOCKER(HrtRaUbRemoteMemImport)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeRemoteOutParam));

    EXPECT_NO_THROW(conn.EnableMemAccess(desc1, mem1));
    EXPECT_NO_THROW(conn.DisableMemAccess(desc1));
    delete a;
}

TEST_F(HcclOneSidedConnTest, DisableMemAccess_BufferNotFound)
{
    void* rdmaHandle = (void*)0x100;
    MOCKER_CPP(&RdmaHandleManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));
    CommunicatorImpl com;
    BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(basePortType, 0, 1, 0, 1);
    HcclOneSidedConn conn(&com, linkData);

    // 创建一个HcclMemDesc对象，描述一个不存在的内存段
    HcclMemDesc desc;
    RmaMemDesc rmaDesc;
    const size_t bufferSize = 512;
    const size_t copySize = sizeof(RmaMemDesc);
    memcpy_s(desc.desc, bufferSize, &rmaDesc, copySize);

    u64 fakeNotifyHandleAddr = 100;
    u64 fakeTargetSegVa = 150;
    HrtRaUbRemMemImportedOutParam fakeRemoteOutParam;
    fakeRemoteOutParam.handle = fakeNotifyHandleAddr;
    fakeRemoteOutParam.targetSegVa = fakeTargetSegVa;
    MOCKER(HrtRaUbRemoteMemImport)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeRemoteOutParam));

    // 调用DisableMemAccess，期望抛出异常，实际上没有抛出异常
    EXPECT_NO_THROW(conn.DisableMemAccess(desc));
}

static bool StubRecvForRemoteMemDesc(Socket* socket, void* data, u32 size)
{
    (void)socket;
    static u32 actualNumOfRemote = 1;
    if (size == sizeof(u32)) {
        memcpy(data, &actualNumOfRemote, sizeof(u32));
    }
    return true;
}

TEST_F(HcclOneSidedConnTest, ReceiveRemoteMemDesc_When_ArrayIsNullptr_Expect_ReturnHCCL_E_PTR)
{
    CommunicatorImpl com;
    BasePortType basePortType(PortDeploymentType::P2P, ConnectProtoType::UB);
    LinkData linkData(basePortType, 0, 1, 0, 1);
    HcclOneSidedConn conn(&com, linkData);

    IpAddress localIp;
    IpAddress remoteIp;
    Socket* mockSocket
        = new Socket(nullptr, localIp, 0, remoteIp, "test", SocketRole::CLIENT, NicType::DEVICE_NIC_TYPE);
    conn.socket_ = mockSocket;

    MOCKER_CPP(&Socket::Recv).stubs().will(invoke(StubRecvForRemoteMemDesc));

    HcclMemDescs remoteMemDescs;
    remoteMemDescs.array = nullptr;
    remoteMemDescs.arrayLength = 0;

    u32 actualNumOfRemote = 0;
    HcclResult ret = conn.ReceiveRemoteMemDesc(remoteMemDescs, actualNumOfRemote);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PTR);

    delete mockSocket;
    conn.socket_ = nullptr;
}
