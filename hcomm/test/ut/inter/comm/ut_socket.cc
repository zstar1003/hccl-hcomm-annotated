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
#include <stdio.h>

#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "sal.h"

#include <externalinput_pub.h>
#include "dltdt_function.h"
#include "dlra_function.h"

#define private public
#define protected public
#include "adapter_hccp.h"
#include "hccl_socket.h"
#undef private
#undef protected
#include "hccl_network.h"

using namespace std;
using namespace hccl;

int stub_SocketRaGetAsyncReqResult(void* reqHandle, int* reqResult)
{
    *reqResult = 0;
    return 0;
}

int stub_SocketRaSocketSendAsync(
    const FdHandle fdHandle, const void* data, unsigned long long size, unsigned long long* sentSize, void** reqHandle)
{
    *reqHandle = (Void*)0x01;
    return 0;
}

int stub_SocketRaSocketRecvAsync(
    const FdHandle fdHandle, void* data, unsigned long long size, unsigned long long* receivedSize, void** reqHandle)
{
    *reqHandle = (Void*)0x02;
    return 0;
}

class SocketTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--SocketTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--SocketTest TearDown--\033[0m" << std::endl; }
    virtual void SetUp() {}
    virtual void TearDown() { GlobalMockObject::verify(); }
};

TEST_F(SocketTest, Ut_SendAsync_When_ParamErrOrRaSocketErr_Expect_Error)
{
    NetDevContext devCtx;
    HcclSocket tempSocket(&devCtx, 16666);
    HcclResult ret;

    u8 data[2] = {1, 2};
    u64 sentSize = 0;
    void* handle = nullptr;
    ret = tempSocket.SendAsync(data, 2, &sentSize, &handle); // fdHandle_ is null
    EXPECT_EQ(ret, HCCL_E_PTR);

    tempSocket.fdHandle_ = (void*)0x01;
    ret = tempSocket.SendAsync(nullptr, 2, &sentSize, &handle); // data is null
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = tempSocket.SendAsync(data, 2, &sentSize, nullptr); // handle is null
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = tempSocket.SendAsync(data, 2, nullptr, &handle); // sentSize is null
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = tempSocket.SendAsync(data, 0, &sentSize, &handle); // size is 0
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = tempSocket.SendAsync(data, 2049, &sentSize, &handle); // size > SOCKET_SEND_MAXLEN (2048)
    EXPECT_EQ(ret, HCCL_E_PARA);

    DlRaFunction::GetInstance().dlRaSocketSendAsync = nullptr;
    ret = tempSocket.SendAsync(data, 2, &sentSize, &handle);
    EXPECT_EQ(ret, HCCL_E_NETWORK);

    DlRaFunction::GetInstance().dlRaSocketSendAsync = stub_SocketRaSocketSendAsync;
    MOCKER(stub_SocketRaSocketSendAsync).stubs().will(returnValue(SOCK_EAGAIN));
    ret = tempSocket.SendAsync(data, 2, &sentSize, &handle);
    EXPECT_EQ(ret, HCCL_E_AGAIN);

    HcclResult sendResult;
    ret = tempSocket.GetAsyncReqResult(nullptr, sendResult);
    EXPECT_EQ(ret, HCCL_E_PTR);

    DlRaFunction::GetInstance().dlRaGetAsyncReqResult = nullptr;
    handle = (void*)0x02;
    ret = tempSocket.GetAsyncReqResult(handle, sendResult);
    EXPECT_EQ(ret, HCCL_E_NETWORK);

    tempSocket.fdHandle_ = nullptr;
}

TEST_F(SocketTest, Ut_SendAsync_When_RaSocketSuccess_Expect_Success)
{
    NetDevContext devCtx;
    HcclSocket tempSocket(&devCtx, 16666);
    HcclResult ret;

    tempSocket.fdHandle_ = (void*)0x01;
    DlRaFunction::GetInstance().dlRaSocketSendAsync = stub_SocketRaSocketSendAsync;
    DlRaFunction::GetInstance().dlRaGetAsyncReqResult = stub_SocketRaGetAsyncReqResult;

    u8 data[2] = {1, 2};
    u64 sentSize = 0;
    void* handle = nullptr;
    ret = tempSocket.SendAsync(data, 2, &sentSize, &handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    int sendResultOut = SOCK_EAGAIN;
    MOCKER(stub_SocketRaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&sendResultOut))
        .will(returnValue(OTHERS_EAGAIN))
        .then(returnValue(0));

    HcclResult sendResult;
    ret = tempSocket.GetAsyncReqResult(handle, sendResult);
    EXPECT_EQ(ret, HCCL_E_AGAIN);

    ret = tempSocket.GetAsyncReqResult(handle, sendResult);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(sendResult, HCCL_E_AGAIN);

    tempSocket.fdHandle_ = nullptr;
}

TEST_F(SocketTest, Ut_RecvAsync_When_ParamErrOrRaSocketErr_Expect_Error)
{
    NetDevContext devCtx;
    HcclSocket tempSocket(&devCtx, 16666);
    HcclResult ret;

    u8 data[2];
    u64 recvSize = 0;
    void* handle = nullptr;
    ret = tempSocket.RecvAsync(data, 2, &recvSize, &handle); // fdHandle_ is null
    EXPECT_EQ(ret, HCCL_E_PTR);

    tempSocket.fdHandle_ = (void*)0x01;
    ret = tempSocket.RecvAsync(nullptr, 2, &recvSize, &handle); // recvBuf is null
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = tempSocket.RecvAsync(data, 2, &recvSize, nullptr); // handle is null
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = tempSocket.RecvAsync(data, 2, nullptr, &handle); // recvSize is null
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = tempSocket.RecvAsync(data, 0, &recvSize, &handle); // recvBufLen is 0
    EXPECT_EQ(ret, HCCL_E_PARA);

    DlRaFunction::GetInstance().dlRaSocketRecvAsync = nullptr;
    ret = tempSocket.RecvAsync(data, 2, &recvSize, &handle);
    EXPECT_EQ(ret, HCCL_E_NETWORK);

    DlRaFunction::GetInstance().dlRaSocketRecvAsync = stub_SocketRaSocketRecvAsync;
    MOCKER(stub_SocketRaSocketRecvAsync).stubs().will(returnValue(SOCK_EAGAIN));
    ret = tempSocket.RecvAsync(data, 2, &recvSize, &handle);
    EXPECT_EQ(ret, HCCL_E_AGAIN);

    tempSocket.fdHandle_ = nullptr;
}

TEST_F(SocketTest, Ut_RecvAsync_When_RaSocketSuccess_Expect_Success)
{
    NetDevContext devCtx;
    HcclSocket tempSocket(&devCtx, 16666);
    HcclResult ret;

    tempSocket.fdHandle_ = (void*)0x01;
    DlRaFunction::GetInstance().dlRaSocketRecvAsync = stub_SocketRaSocketRecvAsync;
    DlRaFunction::GetInstance().dlRaGetAsyncReqResult = stub_SocketRaGetAsyncReqResult;

    u8 data[2];
    u64 recvSize = 0;
    void* handle = nullptr;
    ret = tempSocket.RecvAsync(data, 2, &recvSize, &handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    int recvResultOut = SOCK_EAGAIN;
    MOCKER(stub_SocketRaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&recvResultOut))
        .will(returnValue(OTHERS_EAGAIN))
        .then(returnValue(0));

    HcclResult recvResult;
    ret = tempSocket.GetAsyncReqResult(handle, recvResult);
    EXPECT_EQ(ret, HCCL_E_AGAIN);

    ret = tempSocket.GetAsyncReqResult(handle, recvResult);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(recvResult, HCCL_E_AGAIN);

    tempSocket.fdHandle_ = nullptr;
}

TEST_F(SocketTest, Ut_IsSupportAsync_When_RaSocketNotSupportAsync_Expect_ReturnFalse)
{
    bool isSupportRaSocketAsync = false;
    MOCKER(IsSupportHdcAsync).stubs().with(outBound(isSupportRaSocketAsync)).will(returnValue(HCCL_E_NOT_SUPPORT));

    bool support = HcclSocket::IsSupportAsync();
    EXPECT_FALSE(support);
}
