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

#include "local_ipc_rma_buffer_v2.h"
#include "remote_rma_buffer.h"
#include "remote_ipc_rma_buffer.h"
#include "dev_buffer.h"
#include "rma_buffer.h"

using namespace Hccl;

class LocalIpcRmaBufferTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LocalIpcRmaBuffer tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "LocalIpcRmaBuffer tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in LocalIpcRmaBuffer SetUP." << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in LocalIpcRmaBuffer TearDown." << std::endl;
    }
};

TEST_F(LocalIpcRmaBufferTest, should_recover_RmaBuffer_info_after_serialize_and_deserialize)
{
    BufferType type = BufferType::INPUT; // dataBuffer
    void* ptr = nullptr;
    u64 size = 0;
    bool remoteAccess = true;
    ResourceOwner owner = ResourceOwner::CCL;
    shared_ptr<Buffer> buf = DevBuffer::Create(100, 100);

    u32 id = 0; // port
    BasePortType basePortType(PortDeploymentType::P2P);

    PortData port(0, basePortType, id, IpAddress());

    void* ipcPtr = nullptr; // LocalIpcRmaBuffer
    u64 ipcOffset = 0;
    u64 ipcSize = 10;
    string name = "name";
    LocalIpcRmaBuffer localIpcRmaBuffer(buf, port);
};
