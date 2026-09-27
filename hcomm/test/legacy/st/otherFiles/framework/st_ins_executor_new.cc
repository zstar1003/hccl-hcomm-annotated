/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define private public
#define protected public
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <chrono>
#include <mockcpp/mockcpp.hpp>
#include <stdexcept>
#include <string>
#include "communicator_impl_lite_manager.h"
#include "communicator_impl_lite.h"
#include "ins_executor.h"
#include "rtsq_a5.h"
#include "rtsq_base.h"
#include "orion_adapter_rts.h"
#include "one_sided_component_lite.h"
#include "sqe_build_a5.h"
#include "profiling_handler_lite.h"
#include "task_info.h"
#include "task_param.h"
#include "mirror_task_manager.h"
#include "ins_to_sqe_rule.h"
#undef private
#undef protected
#include "internal_exception.h"

using namespace Hccl;
using namespace aicpu;
// Test fixture for InsExecutor tests
class InsExecutorTestNew : public ::testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InsExecutorTestNew SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "InsExecutorTestNew TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(0));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();
        MOCKER(&GetKernelExecTimeoutFromEnvConfig).stubs().with().will(returnValue(68));
        MOCKER(Interpret, void(const InsLocalCopy&, const StreamLite&, ResMgrFetcher*))
            .stubs()
            .will(ignoreReturnValue());
        std::cout << "A Test case in InsExecutorTestNew SetUp" << std::endl;
    }
    virtual void TearDown()
    {
        std::cout << "A Test case in InsExecutorTestNew TearDown" << std::endl;
        GlobalMockObject::verify();
    }
    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

void MockCreateStreamLite(CommunicatorImplLite& communicatorImplLite, u32 streamId)
{
    u32 fakeStreamId = streamId;
    u32 fakeSqId = streamId;
    u32 fakeDevPhyId = 0;
    BinaryStream liteBinaryStream;
    liteBinaryStream << fakeStreamId;
    liteBinaryStream << fakeSqId;
    liteBinaryStream << fakeDevPhyId;
    std::vector<char> uniqueId{};
    liteBinaryStream.Dump(uniqueId);
    communicatorImplLite.GetStreamLiteMgr()->streams.push_back(std::make_unique<StreamLite>(uniqueId));
}

void MockAddPreStreamSyncTask(std::shared_ptr<InsQueue> insQueue)
{
    std::unique_ptr<InsPreStreamSync> insPreStreamSync = std::make_unique<InsPreStreamSync>();
    insQueue->Append(std::move(insPreStreamSync));
}

void MockCreateNotifyLite(CommunicatorImplLite& communicatorImplLite, u32 notifyId)
{
    u32 fakeDevPhyId = 0;
    u32 fakeNotifyId = notifyId;
    u32 index = notifyId;
    BinaryStream notifyStream;
    notifyStream << fakeNotifyId;
    notifyStream << fakeDevPhyId;
    std::vector<char> notifyUniqueId;
    notifyStream.Dump(notifyUniqueId);
    communicatorImplLite.GetHostDeviceSyncNotifyLiteMgr()->notifys[index]
        = std::make_unique<NotifyLite>(notifyUniqueId);
}

void MockRtsqA5(CommunicatorImplLite& communicatorImplLite)
{
    auto rtsq = static_cast<RtsqA5*>(communicatorImplLite.GetStreamLiteMgr()->GetMaster()->GetRtsq());
    rtsq->sqHead_ = 10;
    rtsq->sqTail_ = 500;
    rtsq->sqDepth_ = 1000;
    MOCKER_CPP_VIRTUAL(*rtsq, &RtsqA5::LaunchTask).stubs().with(mockcpp::any());
    MOCKER_CPP_VIRTUAL(*rtsq, &RtsqA5::NotifyWait).stubs().with(mockcpp::any());
    MOCKER_CPP_VIRTUAL(*rtsq, &RtsqA5::NotifyRecordLoc).stubs().with(mockcpp::any());
    MOCKER_CPP_VIRTUAL(*rtsq, &RtsqA5::SdmaReduce).stubs().with(mockcpp::any());
    MOCKER_CPP_VIRTUAL(*rtsq, &RtsqA5::IsRtsqQueueSpaceSufficient).stubs().with(mockcpp::any()).will(returnValue(true));
}

void MockAddTask2InsQueue(std::shared_ptr<InsQueue> insQueue)
{
    DataSlice usrInSlice = DataSlice(BufferType::INPUT, 0, 64);
    DataSlice usrOutSlice = DataSlice(BufferType::INPUT, 0, 64);
    std::unique_ptr<Instruction> insLocalCopy = std::make_unique<InsLocalCopy>(usrInSlice, usrOutSlice);
    insQueue->Append(std::move(insLocalCopy));
}

// Test case 1: Normal flow test
TEST_F(InsExecutorTestNew, ExecuteV82_NormalFlow)
{
    u32 commIdIndex = 0;
    // 初始化commImplLite和InsExecutor
    CommunicatorImplLite communicatorImplLite(commIdIndex);
    communicatorImplLite.insExecutor = std::make_unique<InsExecutor>(&communicatorImplLite);
    // 创建InsQueue，还有subInsQueue
    std::shared_ptr<InsQueue> queue = std::make_shared<InsQueue>();
    auto subQueue = queue->Fork();

    // 初始化流，一条主流一条从流
    MockCreateStreamLite(communicatorImplLite, 0); // 主流
    MockCreateStreamLite(communicatorImplLite, 1); // 从流

    // 初始化notify
    MockCreateNotifyLite(communicatorImplLite, 0);
    MockCreateNotifyLite(communicatorImplLite, 1);

    // 将Task加入InsQueue
    for (u32 i = 0; i < 2; i++) {
        MockAddTask2InsQueue(queue);
        MockAddTask2InsQueue(subQueue);
    }

    MockRtsqA5(communicatorImplLite);
    // 验证
    communicatorImplLite.insExecutor->ExecuteV82(*queue);
}

// Test case 2: Null pointer exception test
TEST_F(InsExecutorTestNew, ExecuteV82_NullMasterStreamException)
{
    u32 commIdIndex = 0;
    // 初始化commImplLite和InsExecutor
    CommunicatorImplLite communicatorImplLite(commIdIndex);
    communicatorImplLite.insExecutor = std::make_unique<InsExecutor>(&communicatorImplLite);
    // 创建InsQueue
    std::shared_ptr<InsQueue> queue = std::make_shared<InsQueue>();

    MockAddTask2InsQueue(queue);

    EXPECT_THROW(communicatorImplLite.insExecutor->ExecuteV82(*queue);, NullPtrException);
}

TEST_F(InsExecutorTestNew, ExecuteV82_SingleMasterTaskQueue)
{
    u32 commIdIndex = 0;
    // 初始化commImplLite和InsExecutor
    CommunicatorImplLite communicatorImplLite(commIdIndex);
    communicatorImplLite.insExecutor = std::make_unique<InsExecutor>(&communicatorImplLite);
    // 创建InsQueue
    std::shared_ptr<InsQueue> queue = std::make_shared<InsQueue>();

    // 初始化流，一条主流一条从流
    MockCreateStreamLite(communicatorImplLite, 0); // 主流
    MockCreateStreamLite(communicatorImplLite, 1); // 从流

    // 初始化notify
    MockCreateNotifyLite(communicatorImplLite, 0);
    MockCreateNotifyLite(communicatorImplLite, 1);

    // 将Task加入InsQueue
    for (u32 i = 0; i < 2; i++) {
        MockAddTask2InsQueue(queue);
    }
    MockRtsqA5(communicatorImplLite);
    // 验证
    communicatorImplLite.insExecutor->ExecuteV82(*queue);
}

TEST_F(InsExecutorTestNew, ExecuteV82_SingleSlaveTaskQueue)
{
    u32 commIdIndex = 0;
    // 初始化commImplLite和InsExecutor
    CommunicatorImplLite communicatorImplLite(commIdIndex);
    communicatorImplLite.insExecutor = std::make_unique<InsExecutor>(&communicatorImplLite);
    // 创建InsQueue
    std::shared_ptr<InsQueue> queue = std::make_shared<InsQueue>();
    auto subQueue = queue->Fork();

    // 初始化流，一条主流一条从流
    MockCreateStreamLite(communicatorImplLite, 0); // 主流
    MockCreateStreamLite(communicatorImplLite, 1); // 从流

    // 初始化notify
    MockCreateNotifyLite(communicatorImplLite, 0);
    MockCreateNotifyLite(communicatorImplLite, 1);

    // 将Task加入InsQueue
    for (u32 i = 0; i < 2; i++) {
        MockAddTask2InsQueue(subQueue);
    }
    MockRtsqA5(communicatorImplLite);
    // 验证
    communicatorImplLite.insExecutor->ExecuteV82(*queue);
}

TEST_F(InsExecutorTestNew, ExecuteV82_MultipleTaskQueue)
{
    u32 commIdIndex = 0;
    // 初始化commImplLite和InsExecutor
    CommunicatorImplLite communicatorImplLite(commIdIndex);
    communicatorImplLite.insExecutor = std::make_unique<InsExecutor>(&communicatorImplLite);
    // 创建InsQueue
    std::shared_ptr<InsQueue> queue = std::make_shared<InsQueue>();
    auto subQueue1 = queue->Fork();
    auto subQueue2 = queue->Fork();

    // 初始化流，一条主流两条从流
    MockCreateStreamLite(communicatorImplLite, 0); // 主流
    MockCreateStreamLite(communicatorImplLite, 1); // 从流1
    MockCreateStreamLite(communicatorImplLite, 2);

    // 初始化notify
    MockCreateNotifyLite(communicatorImplLite, 0);
    MockCreateNotifyLite(communicatorImplLite, 1);

    // 将Task加入InsQueue
    for (u32 i = 0; i < 2; i++) {
        MockAddTask2InsQueue(queue);
    }
    for (u32 i = 0; i < 3; i++) {
        MockAddTask2InsQueue(subQueue1);
    }
    for (u32 i = 0; i < 6; i++) {
        MockAddTask2InsQueue(subQueue2);
    }
    MockRtsqA5(communicatorImplLite);
    MockAddPreStreamSyncTask(queue);
    MockAddPreStreamSyncTask(subQueue1);
    MockAddPreStreamSyncTask(subQueue2);
    // 验证
    communicatorImplLite.insExecutor->ExecuteV82(*queue);
}
