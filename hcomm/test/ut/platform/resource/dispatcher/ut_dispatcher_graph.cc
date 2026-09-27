/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "dispatcher_graph.h"
#endif
#include "profiling_manager.h"
#include "dlprof_function.h"
#include "profiler_manager.h"
#include "externalinput.h"
#include "adapter_rts.h"
#include "rts_notify.h"
#include "queue_notify_manager.h"

#undef private
#undef protected
#include "graph_ctx_mgr_common.h"
#include "hccl_dispatcher_ctx.h"
#include "dispatcher_ctx.h"

using namespace hccl;

class DispatcherGraph_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DispatcherGraph_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "DispatcherGraph_UT TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        dispatcherGraph->Init();
        std::cout << "DispatcherGraph_UT Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "DispatcherGraph_UT Test TearDown" << std::endl;
    }

    std::unique_ptr<DispatcherGraph> dispatcherGraph
        = std::unique_ptr<DispatcherGraph>(new (std::nothrow) DispatcherGraph(0));
};

bool callbackCalled = false;

void TestCallback(void* userPtr, void* taskPara, unsigned int size)
{
    callbackCalled = true;
    auto* para = static_cast<TaskPara*>(taskPara);
    EXPECT_EQ(para->type, TaskType::TASK_RDMA);
}

TEST_F(DispatcherGraph_UT, RdmaSendPayloadTaskParaDma)
{
    hccl::DispatcherGraph dispatcher(0);
    hccl::Stream stream;

    dispatcher.disableFfts_ = false;
    dispatcher.callback_ = TestCallback;

    MOCKER(DispatcherPub::IsProfSubscribeAdditionInfo).stubs().will(returnValue(true));

    MOCKER(GetExternalInputHcclEnableFfts).stubs().will(returnValue(true));

    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    MOCKER(GetExternalInputTaskExceptionSwitch).stubs().will(returnValue(1));

    SgList sg[1];
    sg[0].addr = 0x1000;
    sg[0].len = 1024;
    SendWr wr{};
    wr.bufList = sg;
    wr.dstAddr = 0x2000;

    auto ret = dispatcher.RdmaSend(0, 123, wr, stream, 2, false);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(callbackCalled);
}

void TestCallbackGraph(void* userPtr, void* taskPara, unsigned int size)
{
    callbackCalled = true;
    return;
}

TEST_F(DispatcherGraph_UT, LaunchTasksEx)
{
    hccl::DispatcherGraph dispatcher(0);
    hccl::Stream stream;
    std::vector<hccl::Stream> subStreams;

    dispatcher.disableFfts_ = false;
    dispatcher.callback_ = TestCallbackGraph;
    int temp = 0;
    dispatcher.fftsCtxsPtr = &temp;

    auto ret = dispatcher.LaunchTasksEx(stream, subStreams);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(callbackCalled);
}

TEST_F(DispatcherGraph_UT, RdmaSendNotifyTaskParaDma)
{
    hccl::DispatcherGraph dispatcher(0);
    hccl::Stream stream;

    dispatcher.disableFfts_ = false;
    dispatcher.callback_ = TestCallback;

    MOCKER(DispatcherPub::IsProfSubscribeAdditionInfo).stubs().will(returnValue(true));

    MOCKER(GetExternalInputHcclEnableFfts).stubs().will(returnValue(true));

    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    MOCKER(GetExternalInputTaskExceptionSwitch).stubs().will(returnValue(1));

    SgList sg[1];
    sg[0].addr = 0x1000;
    sg[0].len = 1024;
    SendWr wr{};
    wr.bufList = sg;
    wr.dstAddr = 0x2000;

    auto ret = dispatcher.RdmaSend(0, 123, wr, stream, 1, 2, false);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(callbackCalled);
}
