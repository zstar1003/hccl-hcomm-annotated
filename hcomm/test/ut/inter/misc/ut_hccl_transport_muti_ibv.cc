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

#include "stream_pub.h"
#include "dlra_function.h"
#include "llt_hccl_stub_pub.h"

class TransportMutiIbverbsTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "\033[36m--TransportMutiIbverbsTest SetUP--\033[0m" << std::endl;
        DlRaFunction::GetInstance().DlRaFunctionInit();
    }
    static void TearDownTestCase() { std::cout << "\033[36m--TransportMutiIbverbsTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(TransportMutiIbverbsTest, ut_function_for_batchsendrecv_multi_ibv)
{
    s32 ret;
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    s32 mem_size = 256;
    DeviceMem mem = DeviceMem::alloc(mem_size);

    void* dispatcherPtr = nullptr;
    ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(dispatcherPtr, nullptr);
    DispatcherPub* dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);
    MachinePara machinePara;

    std::chrono::milliseconds timeout;
    const std::string tag;

    std::shared_ptr<TransportIbverbs> linktmp = nullptr;
    linktmp.reset(new TransportIbverbs(dispatcher, nullptr, machinePara, timeout));
    linktmp->Init();
    linktmp->TxPrepare(stream);
    linktmp->RxPrepare(stream);
    linktmp->TxData(UserMemType::INPUT_MEM, 0, mem.ptr(), mem_size, stream);
    linktmp->RxData(UserMemType::INPUT_MEM, 0, mem.ptr(), mem_size, stream);
    linktmp->TxDone(stream);
    linktmp->RxDone(stream);
    linktmp->PostFinAck(stream);
    linktmp->WaitFinAck(stream);
    if (dispatcherPtr != nullptr) {
        ret = HcclDispatcherDestroy(dispatcherPtr);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        dispatcherPtr = nullptr;
    }
}
