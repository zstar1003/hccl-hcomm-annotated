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

#include "hccl_communicator.h"
#include "transport.h"
#include "hccl_types.h"
#include "hccl_socket.h"

using namespace std;
using namespace hccl;

class ErrorCqeTest_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ErrorCqeTest_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "ErrorCqeTest_UT TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

static HcclResult StubHcclNetDevGetLocalIp(HcclNetDevCtx, HcclIpAddress& ip) { return HCCL_SUCCESS; }

static HcclResult
StubGetTransportErrorCqe(const HcclNetDevCtx, std::vector<std::pair<Transport*, CqeInfo>>& infolist, u32& num)
{
    CqeInfo cqe;
    infolist.push_back({reinterpret_cast<Transport*>(0x1000), cqe});
    num = 1;
    return HCCL_SUCCESS;
}

TEST_F(ErrorCqeTest_UT, GetMc2TransportCqeErrors)
{
    HcclCommunicator communicator;
    HcclNetDevCtx ctx = reinterpret_cast<HcclNetDevCtx>(0x1);
    Transport* mapTransport = reinterpret_cast<Transport*>(0x2000);
    MOCKER(HcclNetDevGetLocalIp).stubs().will(invoke(StubHcclNetDevGetLocalIp));

    MOCKER(Transport::GetTransportErrorCqe).stubs().will(invoke(StubGetTransportErrorCqe));

    std::vector<ErrCqeInfo> infos;
    u32 num = 0;
    EXPECT_EQ(communicator.GetTransportCqeErrors(ctx, infos, num), HCCL_SUCCESS);

    EXPECT_EQ(num, 1);
    EXPECT_EQ(infos.size(), 1);
}
