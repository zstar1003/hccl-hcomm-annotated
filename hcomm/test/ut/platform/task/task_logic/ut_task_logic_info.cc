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
#include <string>
#include <vector>

#include "task_logic_info.h"

using namespace std;
using namespace hccl;

class TaskLogicInfoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TaskLogicInfoTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "TaskLogicInfoTest TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "TaskLogicInfoTest SetUP per test" << std::endl; }
    virtual void TearDown()
    {
        std::cout << "TaskLogicInfoTest TearDown per test" << std::endl;
        GlobalMockObject::verify();
    }
};

// 构造1: 基础构造(index/taskLogicType/funcType)
TEST_F(TaskLogicInfoTest, ut_BasicConstructor_Expect_FieldsSet)
{
    TaskLogicInfo info(3, TaskLogicType::TRANSPORT_TYPE, TaskLogicFuncType::TRANSPORT_TXACK_TYPE);
    EXPECT_EQ(info.taskLogicCmd.taskLogicType, TaskLogicType::TRANSPORT_TYPE);
    EXPECT_EQ(info.taskLogicCmd.index, 3u);
    EXPECT_EQ(info.taskFuncType, TaskLogicFuncType::TRANSPORT_TXACK_TYPE);
}

// 构造2: txAsync 构造
TEST_F(TaskLogicInfoTest, ut_TxAsyncConstructor_Expect_FieldsSet)
{
    std::vector<TxMemoryInfo> txMems(2);
    TaskLogicInfo info(1, TaskLogicType::DISPATCHER_TYPE, TaskLogicFuncType::TRANSPORT_TXASYNC_TYPE, txMems);
    EXPECT_EQ(info.taskLogicCmd.taskLogicType, TaskLogicType::DISPATCHER_TYPE);
    EXPECT_EQ(info.taskLogicCmd.index, 1u);
    EXPECT_EQ(info.taskFuncType, TaskLogicFuncType::TRANSPORT_TXASYNC_TYPE);
    EXPECT_EQ(info.txAsync.txMems.size(), 2u);
}

// 构造3: rxAsync 构造
TEST_F(TaskLogicInfoTest, ut_RxAsyncConstructor_Expect_FieldsSet)
{
    std::vector<RxMemoryInfo> rxMems(3);
    TaskLogicInfo info(2, TaskLogicType::DISPATCHER_TYPE, TaskLogicFuncType::TRANSPORT_RXASYNC_TYPE, rxMems);
    EXPECT_EQ(info.taskLogicCmd.taskLogicType, TaskLogicType::DISPATCHER_TYPE);
    EXPECT_EQ(info.taskLogicCmd.index, 2u);
    EXPECT_EQ(info.taskFuncType, TaskLogicFuncType::TRANSPORT_RXASYNC_TYPE);
    EXPECT_EQ(info.rxAsync.rxMems.size(), 3u);
}

// 构造4: signalWait 构造
TEST_F(TaskLogicInfoTest, ut_SignalWaitConstructor_Expect_FieldsSet)
{
    void* signal = reinterpret_cast<void*>(0x1000);
    TaskLogicInfo info(
        4u, TaskLogicType::DISPATCHER_TYPE, TaskLogicFuncType::DISPATCHER_SIGNALWAIT_TYPE, signal, 7u, 8u, 9);
    EXPECT_EQ(info.taskLogicCmd.taskLogicType, TaskLogicType::DISPATCHER_TYPE);
    EXPECT_EQ(info.taskLogicCmd.index, 4u);
    EXPECT_EQ(info.taskFuncType, TaskLogicFuncType::DISPATCHER_SIGNALWAIT_TYPE);
    EXPECT_EQ(info.taskLogicPara.dispatcherTaskLogicPara.signalWait.signal, signal);
    EXPECT_EQ(info.taskLogicPara.dispatcherTaskLogicPara.signalWait.userRank, 7u);
    EXPECT_EQ(info.taskLogicPara.dispatcherTaskLogicPara.signalWait.remoteRank, 8u);
    EXPECT_EQ(info.taskLogicPara.dispatcherTaskLogicPara.signalWait.stage, 9);
}

// 构造5: signalRecord 构造
TEST_F(TaskLogicInfoTest, ut_SignalRecordConstructor_Expect_FieldsSet)
{
    void* signal = reinterpret_cast<void*>(0x2000);
    TaskLogicInfo info(
        5u, TaskLogicType::DISPATCHER_TYPE, TaskLogicFuncType::DISPATCHER_SIGNALRECORD_TYPE, signal, 11u, 0xFFFFULL,
        12);
    EXPECT_EQ(info.taskLogicCmd.taskLogicType, TaskLogicType::DISPATCHER_TYPE);
    EXPECT_EQ(info.taskLogicCmd.index, 5u);
    EXPECT_EQ(info.taskFuncType, TaskLogicFuncType::DISPATCHER_SIGNALRECORD_TYPE);
    EXPECT_EQ(info.taskLogicPara.dispatcherTaskLogicPara.signalRecord.signal, signal);
    EXPECT_EQ(info.taskLogicPara.dispatcherTaskLogicPara.signalRecord.userRank, 11u);
    EXPECT_EQ(info.taskLogicPara.dispatcherTaskLogicPara.signalRecord.offset, 0xFFFFu);
    EXPECT_EQ(info.taskLogicPara.dispatcherTaskLogicPara.signalRecord.stage, 12);
}

// 构造6: memAsync 构造
TEST_F(TaskLogicInfoTest, ut_MemAsyncConstructor_Expect_FieldsSet)
{
    void* dst = reinterpret_cast<void*>(0x3000);
    void* src = reinterpret_cast<void*>(0x4000);
    TaskLogicInfo info(
        6, TaskLogicType::DISPATCHER_TYPE, TaskLogicFuncType::DISPATCHER_MEMCPYASYNC_TYPE, dst, 1024, src, 256,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE);
    EXPECT_EQ(info.taskLogicCmd.taskLogicType, TaskLogicType::DISPATCHER_TYPE);
    EXPECT_EQ(info.taskLogicCmd.index, 6u);
    EXPECT_EQ(info.taskFuncType, TaskLogicFuncType::DISPATCHER_MEMCPYASYNC_TYPE);
    EXPECT_EQ(info.taskLogicPara.dispatcherTaskLogicPara.memAsync.dst, dst);
    EXPECT_EQ(info.taskLogicPara.dispatcherTaskLogicPara.memAsync.destMax, 1024u);
    EXPECT_EQ(info.taskLogicPara.dispatcherTaskLogicPara.memAsync.src, src);
    EXPECT_EQ(info.taskLogicPara.dispatcherTaskLogicPara.memAsync.count, 256u);
    EXPECT_EQ(
        info.taskLogicPara.dispatcherTaskLogicPara.memAsync.kind,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE);
}
