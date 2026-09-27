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
#include "aicpu_daemon_service.h"
#include <thread>

using namespace Hccl;
class AicpuDaemonServiceTest : public testing::Test {
public:
    static void SetUpTestCase() { std::cout << "AicpuDaemonServiceTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "AicpuDaemonServiceTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in AicpuDaemonServiceTest SetUp" << std::endl; }

    virtual void TearDown()
    {
        std::cout << "A Test case in AicpuDaemonServiceTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

class DaemonFuncTest : public DaemonFunc {
public:
    static DaemonFuncTest& GetInstance()
    {
        static DaemonFuncTest daemonFunc;
        return daemonFunc;
    }

    void Call() override { std::cout << "DaemonFuncTest" << std::endl; }
};

TEST_F(AicpuDaemonServiceTest, should_success_when_calling_Register)
{
    AicpuDaemonService::GetInstance().Register(&DaemonFuncTest::GetInstance());
}

TEST_F(AicpuDaemonServiceTest, should_success_when_calling_ServiceRun)
{
    CommandToBackGroud info = CommandToBackGroud::Default;
    auto daemonServiceRun = [](void* info) {
        AicpuDaemonService::GetInstance().ServiceRun(info);
    };
    auto daemonServiceStop = [](void* info) {
        AicpuDaemonService::GetInstance().ServiceStop(info);
    };
    auto threadRun = std::thread(daemonServiceRun, &info);
    auto threadStop = std::thread(daemonServiceStop, &info);
    threadStop.join();
    threadRun.join();
    EXPECT_EQ(CommandToBackGroud::Stop, info);
}

TEST_F(AicpuDaemonServiceTest, should_success_when_calling_ServiceStop)
{
    CommandToBackGroud info = CommandToBackGroud::Default;
    AicpuDaemonService::GetInstance().ServiceStop(&info);
    EXPECT_EQ(CommandToBackGroud::Stop, info);
}

TEST_F(AicpuDaemonServiceTest, should_success_when_calling_Break) { AicpuDaemonService::GetInstance().Break(); }
