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
#include <cstdio>
#include <slog.h>
#include "mem_host_pub.h"
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "llt_hccl_stub_sal_pub.h"
#include "adapter_rts.h"

#define private public
#define protected public
#include "opexecounter_pub.h"
#include "externalinput.h"
#include "profiler_manager.h"
#include "dispatcher_pub.h"
#include "op_base.h"
#undef protected
#undef private

using namespace std;
using namespace hccl;

class OpCounterTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--OpExeCounter SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--OpExeCounter TearDown--\033[0m" << std::endl; }
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(OpCounterTest, ut_op_counter)
{
    ResetInitState();
    InitExternalInput();

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    SetFftsSwitch(false);

    // 创建dispatcher
    HcclResult ret = HCCL_SUCCESS;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::shared_ptr<hccl::ProfilerManager> profilerManager;
    profilerManager.reset(new (std::nothrow) ProfilerManager(0, 0, 2, 2));
    profilerManager->InitProfiler();
    DevType chipType = DevType::DEV_TYPE_910;
    void* dispatcherPtr = nullptr;

    ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(dispatcherPtr, nullptr);
    DispatcherPub* dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);
    ret = OpExeCounter::GetInstance(0).InitCounter();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 申请stream
    rtStream_t rtStream = NULL;
    rtError_t rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtCreateStream(&rtStream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    Stream stream = Stream(rtStream);

    ret = OpExeCounter::GetInstance(0).AddCounter(dispatcher, stream, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = OpExeCounter::GetInstance(0).AddCounter(dispatcher, stream, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::pair<int32_t, int32_t> counter{0, 0};
    ret = OpExeCounter::GetInstance(0).GetCounter(counter);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    OpExeCounter::GetInstance(0).DeInitCounter();

    if (dispatcher != nullptr) {
        ret = HcclDispatcherDestroy(dispatcher);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        dispatcher = nullptr;
    }

    // 销毁资源
    if (rtStream != NULL) {
        rt_ret = aclrtDestroyStream(rtStream);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    rt_ret = hrtResetDevice(0);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ResetInitState();
    GlobalMockObject::verify();
}

TEST_F(OpCounterTest, st_stuck_detect_use_hcclexectime)
{
    ResetInitState();
    setenv("HCCL_EXEC_TIMEOUT", "100", 1);
    HcclResult ret = InitExternalInput();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    unsetenv("HCCL_EXEC_TIMEOUT");
    ResetInitState();
    GlobalMockObject::verify();
}

TEST_F(OpCounterTest, ut_clear_opCounter)
{
    ResetInitState();
    setenv("HCCL_OP_COUNTER_ENABLE", "1", 1);
    InitExternalInput();

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemSet).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    HcclResult ret = OpExeCounter::GetInstance(0).InitCounter();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = ClearOpCounterMem();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_OP_COUNTER_ENABLE");
    ResetInitState();
    GlobalMockObject::verify();
}
