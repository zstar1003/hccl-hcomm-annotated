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
#include "rma_connection.h"
#include "p2p_connection.h"
#include "local_ipc_rma_buffer_v2.h"
#include "remote_rma_buffer.h"
#include "types.h"
#include "data_type.h"
#include "reduce_op.h"
#include "dev_buffer.h"
#include "rma_buffer.h"

using namespace Hccl;

class P2PConnectionTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "P2PConnection tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "P2PConnection tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in P2PConnection SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in P2PConnection TearDown" << std::endl;
    }
    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100, 0x100);
};

TEST_F(P2PConnectionTest, should_return_true_when_calling_rma_connection_bind_and_unbind_with_valid_params)
{
    u32 id = 0; // port
    BasePortType basePortType(PortDeploymentType::P2P);
    PortData portData(0, basePortType, id, IpAddress());

    LocalIpcRmaBuffer localIpcRmaBuffer(devBuf);

    std::unique_ptr<RemoteIpcRmaBuffer> remoteIpcRmaBuffer1 = make_unique<RemoteIpcRmaBuffer>();
    remoteIpcRmaBuffer1->Describe();

    Socket* socket = nullptr;

    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    std::string tag = "Bind_Unbind";

    P2PConnection p2pConnection(socket, tag);

    auto res2 = p2pConnection.GetRemoteRmaBuffer(BufferType::INPUT);
    EXPECT_EQ(nullptr, res2);

    p2pConnection.Bind(remoteIpcRmaBuffer1.get(), BufferType::INPUT);
    auto res3 = p2pConnection.GetRemoteRmaBuffer(BufferType::INPUT);
    EXPECT_NE(nullptr, res3);
}

TEST_F(P2PConnectionTest, should_return_task_ptr_when_calling_rma_p2p_connection_prepare_read_tasks)
{
    Socket* socket = nullptr;
    std::string tag = "Get";

    P2PConnection p2pConnection(socket, tag);
    std::string p2pConnectionDescribeStr = p2pConnection.Describe();
    std::cout << "rmaConnectionDescribeStr: " << p2pConnectionDescribeStr << std::endl;

    // only one task
    u64 size1 = 0x1000; // 默认 SDMA最大是4GB
    u64 remoteAddr1 = 8;
    u64 localAddr1 = 0;
    MemoryBuffer remoteMemBuf1(remoteAddr1, size1, 0);
    MemoryBuffer localMemBuf1(localAddr1, size1, 0);
    SqeConfig config{};

    auto res1 = p2pConnection.PrepareRead(remoteMemBuf1, localMemBuf1, config);
    TaskP2pMemcpy taskP2pMemcpy1 = static_cast<const TaskP2pMemcpy&>(*res1.get());
    EXPECT_NE(nullptr, res1);
    EXPECT_EQ(remoteAddr1, taskP2pMemcpy1.GetSrcAddr());
    EXPECT_EQ(localAddr1, taskP2pMemcpy1.GetDstAddr());
    EXPECT_EQ(localMemBuf1.size, taskP2pMemcpy1.GetSize());
    EXPECT_EQ(MemcpyKind::D2D, taskP2pMemcpy1.GetKind());

    // zero task
    u64 size2 = 0;
    u64 remoteAddr2 = 8;
    u64 localAddr2 = 0;
    MemoryBuffer remoteMemBuf2(remoteAddr2, size2, 0);
    MemoryBuffer localMemBuf2(localAddr2, size2, 0);

    auto res2 = p2pConnection.PrepareRead(remoteMemBuf2, localMemBuf2, config);
    EXPECT_EQ(nullptr, res2);
    // only one task
    u64 size3 = 0x100000000; // 默认 SDMA最大是4GB
    u64 remoteAddr3 = 8;
    u64 localAddr3 = 0;
    MemoryBuffer remoteMemBuf3(remoteAddr3, size3, 0);
    MemoryBuffer localMemBuf3(localAddr3, size3, 0);

    auto res3 = p2pConnection.PrepareRead(remoteMemBuf3, localMemBuf3, config);
    EXPECT_NE(nullptr, res3);
    TaskP2pMemcpy taskP2pMemcpy3 = static_cast<const TaskP2pMemcpy&>(*res3.get());
    EXPECT_EQ(remoteAddr3, taskP2pMemcpy3.GetSrcAddr());
    EXPECT_EQ(localAddr3, taskP2pMemcpy3.GetDstAddr());
    EXPECT_EQ(localMemBuf3.size, taskP2pMemcpy3.GetSize());
    EXPECT_EQ(MemcpyKind::D2D, taskP2pMemcpy3.GetKind());

    // two task
    u64 size4 = 0x100000010; // 默认 SDMA最大是4GB
    u64 remoteAddr4 = 8;
    u64 localAddr4 = 0;
    MemoryBuffer remoteMemBuf4(remoteAddr4, size4, 0);
    MemoryBuffer localMemBuf4(localAddr4, size4, 0);

    auto res4 = p2pConnection.PrepareRead(remoteMemBuf4, localMemBuf4, config);
    EXPECT_NE(nullptr, res4);
    TaskP2pMemcpy taskP2pMemcpy40 = static_cast<const TaskP2pMemcpy&>(*res4.get());
    EXPECT_EQ(remoteAddr3, taskP2pMemcpy40.GetSrcAddr());
    EXPECT_EQ(localAddr3, taskP2pMemcpy40.GetDstAddr());
    EXPECT_EQ(0x100000010, taskP2pMemcpy40.GetSize());
    EXPECT_EQ(MemcpyKind::D2D, taskP2pMemcpy40.GetKind());

    TaskP2pMemcpy taskP2pMemcpy41 = static_cast<const TaskP2pMemcpy&>(*res4.get());
    EXPECT_EQ(remoteAddr4, taskP2pMemcpy41.GetSrcAddr());
    EXPECT_EQ(localAddr4, taskP2pMemcpy41.GetDstAddr());
    EXPECT_EQ(0x100000010, taskP2pMemcpy41.GetSize());
    EXPECT_EQ(MemcpyKind::D2D, taskP2pMemcpy41.GetKind());

    // different size
    MemoryBuffer remoteMemBufDiff(0, 1000, 0);
    MemoryBuffer localMemBufDiff(2000, 1, 0);

    EXPECT_THROW(p2pConnection.PrepareRead(remoteMemBufDiff, localMemBufDiff, config), InvalidParamsException);
}

TEST_F(P2PConnectionTest, should_return_task_ptr_when_calling_rma_p2p_connection_prepare_read_reduce_tasks)
{
    Socket* socket = nullptr;

    BasePortType basePortType(PortDeploymentType::P2P);
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::UB);
    std::string tag = "Get";

    P2PConnection p2pConnection(socket, tag);
    std::string p2pConnectionDescribeStr = p2pConnection.Describe();
    std::cout << "rmaConnectionDescribeStr: " << p2pConnectionDescribeStr << std::endl;
    DataType dataType = DataType::FP16;
    ReduceOp reduceOp = ReduceOp::SUM;

    // only one task
    u64 size1 = 0x1000; // 默认 SDMA最大是4GB
    u64 remoteAddr1 = 8;
    u64 localAddr1 = 0;
    MemoryBuffer remoteMemBuf1(remoteAddr1, size1, 0);
    MemoryBuffer localMemBuf1(localAddr1, size1, 0);
    SqeConfig config{};

    auto res1 = p2pConnection.PrepareReadReduce(remoteMemBuf1, localMemBuf1, dataType, reduceOp, config);
    TaskSdmaReduce taskSdmaReduce1 = static_cast<const TaskSdmaReduce&>(*res1.get());
    EXPECT_NE(nullptr, res1);
    EXPECT_EQ(remoteAddr1, taskSdmaReduce1.GetSrcAddr());
    EXPECT_EQ(localAddr1, taskSdmaReduce1.GetDstAddr());
    EXPECT_EQ(size1, taskSdmaReduce1.GetSize());
    EXPECT_EQ(dataType, taskSdmaReduce1.GetDataType());
    EXPECT_EQ(reduceOp, taskSdmaReduce1.GetReduceOp());

    // zero task
    u64 size2 = 0;
    u64 remoteAddr2 = 8;
    u64 localAddr2 = 8;
    MemoryBuffer remoteMemBuf2(remoteAddr2, size2, 0);
    MemoryBuffer localMemBuf2(localAddr2, size2, 0);

    auto res2 = p2pConnection.PrepareReadReduce(remoteMemBuf2, localMemBuf2, dataType, reduceOp, config);
    EXPECT_EQ(nullptr, res2);

    // only one task
    u64 size3 = 0x100000000; // 默认 SDMA最大是4GB
    u64 remoteAddr3 = 8;
    u64 localAddr3 = 8;
    MemoryBuffer remoteMemBuf3(remoteAddr3, size3, 0);
    MemoryBuffer localMemBuf3(localAddr3, size3, 0);

    auto res3 = p2pConnection.PrepareReadReduce(remoteMemBuf3, localMemBuf3, dataType, reduceOp, config);
    TaskSdmaReduce taskSdmaReduce3 = static_cast<const TaskSdmaReduce&>(*res3.get());
    EXPECT_NE(nullptr, res3);
    EXPECT_EQ(remoteAddr3, taskSdmaReduce3.GetSrcAddr());
    EXPECT_EQ(localAddr3, taskSdmaReduce3.GetDstAddr());
    EXPECT_EQ(size3, taskSdmaReduce3.GetSize());
    EXPECT_EQ(dataType, taskSdmaReduce3.GetDataType());
    EXPECT_EQ(reduceOp, taskSdmaReduce3.GetReduceOp());

    // two task
    u64 size4 = 0x100000010; // 默认 SDMA最大是4GB
    u64 remoteAddr4 = 8;
    u64 localAddr4 = 8;
    MemoryBuffer remoteMemBuf4(remoteAddr4, size4, 0);
    MemoryBuffer localMemBuf4(localAddr4, size4, 0);

    auto res4 = p2pConnection.PrepareReadReduce(remoteMemBuf4, localMemBuf4, dataType, reduceOp, config);
    TaskSdmaReduce taskSdmaReduce40 = static_cast<const TaskSdmaReduce&>(*res4.get());
    EXPECT_NE(nullptr, res4);
    EXPECT_EQ(remoteAddr4, taskSdmaReduce40.GetSrcAddr());
    EXPECT_EQ(localAddr4, taskSdmaReduce40.GetDstAddr());
    EXPECT_EQ(0x100000010, taskSdmaReduce40.GetSize());
    EXPECT_EQ(dataType, taskSdmaReduce40.GetDataType());
    EXPECT_EQ(reduceOp, taskSdmaReduce40.GetReduceOp());

    TaskSdmaReduce taskSdmaReduce41 = static_cast<const TaskSdmaReduce&>(*res4.get());

    EXPECT_EQ(remoteAddr4, taskSdmaReduce41.GetSrcAddr());
    EXPECT_EQ(localAddr4, taskSdmaReduce41.GetDstAddr());
    EXPECT_EQ(0x100000010, taskSdmaReduce41.GetSize());
    EXPECT_EQ(dataType, taskSdmaReduce41.GetDataType());
    EXPECT_EQ(reduceOp, taskSdmaReduce41.GetReduceOp());

    // different size
    MemoryBuffer remoteMemBufDiff(0, 1000, 0);
    MemoryBuffer localMemBufDiff(2000, 1, 0);

    EXPECT_THROW(
        p2pConnection.PrepareReadReduce(remoteMemBufDiff, localMemBufDiff, dataType, reduceOp, config),
        InvalidParamsException);
}

TEST_F(P2PConnectionTest, should_return_task_ptr_when_calling_rma_p2p_connection_prepare_write)
{
    Socket* socket = nullptr;

    std::string tag = "Write";

    P2PConnection p2pConnection(socket, tag);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};

    auto res1 = p2pConnection.PrepareWrite(remoteMemBuffer, localMemBuffer, config);
    EXPECT_NE(nullptr, res1);
}

TEST_F(P2PConnectionTest, should_return_task_ptr_when_calling_rma_p2p_connection_prepare_write_reduce)
{
    Socket* socket = nullptr;
    std::string tag = "WriteReduce";

    P2PConnection p2pConnection(socket, tag);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);

    DataType dataType = DataType::FP16;
    ReduceOp reduceOp = ReduceOp::SUM;
    SqeConfig config{};

    auto res1 = p2pConnection.PrepareWriteReduce(remoteMemBuffer, localMemBuffer, dataType, reduceOp, config);
    EXPECT_NE(nullptr, res1);
}
