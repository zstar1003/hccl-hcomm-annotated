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
#include <iostream>
#include <thread>
#include <chrono>

#include "task_profiling_pub.h"

#include "llt_hccl_stub_profiling_plugin.h"
#include "transport_p2p_pub.h"
#include "llt_hccl_stub_pub.h"

#include "threadManage.h"

#include "profiler_manager.h"
#include "queue_notify_manager.h"

class ThreadManageTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        s32 ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr);
        if (ret != HCCL_SUCCESS)
            return;
        if (dispatcherPtr == nullptr)
            return;
        dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);
        std::cout << "ThreadManageTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "ThreadManageTest TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A test ThreadManageTest SetUp" << std::endl;
    }
    virtual void TearDown() { std::cout << "A test ThreadManageTest TearDown" << std::endl; }
    static HcclDispatcher dispatcherPtr;
    static DispatcherPub* dispatcher;
};
HcclDispatcher ThreadManageTest::dispatcherPtr = nullptr;
DispatcherPub* ThreadManageTest::dispatcher = nullptr;

TEST_F(ThreadManageTest, threadMange_test_001)
{
    s32 ret = HCCL_SUCCESS;
    s32 device_id = 0;
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ThreadManage* threadM = new (std::nothrow) ThreadManage(device_id, 0, dispatcher);
    EXPECT_NE(threadM, nullptr);
    ret = threadM->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::this_thread::sleep_for(std::chrono::seconds(3));
    threadM->NotifyStart();
    threadM->WaitDone();
    threadM->Finalize();
    MOCKER_CPP(&ThreadManage::Finalize).stubs().will(returnValue(HCCL_E_PARA));
    delete threadM;
    GlobalMockObject::verify();
}

TEST_F(ThreadManageTest, threadMange_test_002)
{
    s32 ret = HCCL_SUCCESS;
    s32 device_id = 0;
    // 创建输入输出内存
    u32 memSize = 1024;
    DeviceMem inputMem = DeviceMem::alloc(memSize);
    DeviceMem outputMem = DeviceMem::alloc(memSize);
    sal_memset(inputMem.ptr(), memSize, 1, memSize);
    sal_memset(outputMem.ptr(), memSize, 0, memSize);
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    u64 count = memSize;
    // 创建流
    rtStream_t rtstream;
    aclrtCreateStream(&rtstream);
    Stream stream(rtstream);
    HcclReduceOp op = HCCL_REDUCE_SUM;
    u32 root = 0;
    std::vector<Slice> slice;
    Slice slice1;
    slice1.size = 1024;
    slice1.offset = 0;
    slice.push_back(slice1);
    u64 baseOffset = 0;
    std::vector<u32> nicRankList;
    nicRankList.push_back(0);
    s32 profStage = 0;
    // 创建exchanger
    std::string unique_id = "ThreadManageTest";
    IntraExchanger exchanger{};
    std::vector<RankInfo> para_vector(1);
    // 创建信号

    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    notifyPool.reset(new (std::nothrow) NotifyPool());
    EXPECT_NE(notifyPool, nullptr);
    ret = notifyPool->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::unique_ptr<QueueNotifyManager> queueNotifyManager = nullptr;
    queueNotifyManager.reset(new (std::nothrow) QueueNotifyManager());
    EXPECT_NE(queueNotifyManager, nullptr);
    ret = queueNotifyManager->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<LocalNotify> signalAux = nullptr;
    std::shared_ptr<LocalNotify> signalMain = nullptr;
    std::string tag = "signal_test";
    std::vector<std::shared_ptr<LocalNotify>> notifys(2, nullptr);
    ret = queueNotifyManager->Alloc(tag, 2, notifys);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    signalMain = notifys[0];
    signalAux = notifys[1];
    // 创建comm
    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    SubCommInfo comm_inner = {0, 1};
    u32 ringIndex = 0;
    ExecutorType type = ExecutorType::REDUCE_SCATTER_RING;
    u64 reduceAttr = 0;
    ThreadManage* threadM = new (std::nothrow) ThreadManage(device_id, 0, dispatcher);
    EXPECT_NE(threadM, nullptr);
    ret = threadM->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = threadM->Prepare(
        inputMem, outputMem, inputMem, count, dataType, stream, op, root, slice, 0, nicRankList, "tag", profStage,
        comm_inner, signalAux, signalMain, ringIndex, ExecutorType::REDUCE_SCATTER_RING, reduceAttr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    threadM->NotifyStart();
    threadM->WaitDone();
    threadM->Finalize();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete threadM;
    signalMain = nullptr;
    signalAux = nullptr;
    ret = aclrtDestroyStream(rtstream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    inputMem.free();
    outputMem.free();
}

TEST_F(ThreadManageTest, threadMange_test_003)
{
    s32 ret = HCCL_SUCCESS;
    s32 device_id = 0;
    // 创建输入输出内存
    u32 memSize = 1024;
    DeviceMem inputMem = DeviceMem::alloc(memSize);
    DeviceMem outputMem = DeviceMem::alloc(memSize);
    sal_memset(inputMem.ptr(), memSize, 1, memSize);
    sal_memset(outputMem.ptr(), memSize, 0, memSize);
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    u64 count = memSize;
    // 创建流
    rtStream_t rtstream;
    aclrtCreateStream(&rtstream);
    Stream stream(rtstream);
    HcclReduceOp op = HCCL_REDUCE_SUM;
    u32 root = 0;
    std::vector<Slice> slice;
    Slice slice1;
    slice1.size = 1024;
    slice1.offset = 0;
    slice.push_back(slice1);
    u64 baseOffset = 0;
    std::vector<u32> nicRankList;
    nicRankList.push_back(0);
    s32 profStage = 0;
    // 创建exchanger
    std::string unique_id = "ThreadManageTest";
    std::vector<RankInfo> para_vector(1);
    // 创建信号
    std::shared_ptr<LocalNotify> signalAux = nullptr;
    std::shared_ptr<LocalNotify> signalMain = nullptr;

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    SubCommInfo comm_inner = {0, 1};
    u32 ringIndex = 0;
    ExecutorType type = ExecutorType::REDUCE_SCATTER_RING;
    u64 reduceAttr = 0;
    ThreadManage* threadM = new (std::nothrow) ThreadManage(device_id, 0, dispatcher);
    EXPECT_NE(threadM, nullptr);
    ret = threadM->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = threadM->Prepare(
        inputMem, outputMem, inputMem, count, dataType, stream, op, root, slice, 0, nicRankList, "tag", profStage,
        comm_inner, signalAux, signalMain, ringIndex, ExecutorType::REDUCE_SCATTER_RING, reduceAttr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    threadM->NotifyStart();
    threadM->WaitDone();
    threadM->Finalize();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete threadM;
    inputMem.free();
    outputMem.free();
    ret = aclrtDestroyStream(rtstream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ThreadManageTest, threadMange_test_004)
{
    s32 ret = HCCL_SUCCESS;
    s32 device_id = 0;
    // 创建输入输出内存
    u32 memSize = 1024;
    DeviceMem inputMem = DeviceMem::alloc(memSize);
    DeviceMem outputMem = DeviceMem::alloc(memSize);
    sal_memset(inputMem.ptr(), memSize, 1, memSize);
    sal_memset(outputMem.ptr(), memSize, 0, memSize);
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    u64 count = memSize;
    // 创建流
    rtStream_t rtstream;
    aclrtCreateStream(&rtstream);
    Stream stream(rtstream);
    HcclReduceOp op = HCCL_REDUCE_SUM;
    u32 root = 0;
    std::vector<Slice> slice;
    Slice slice1;
    slice1.size = 1024;
    slice1.offset = 0;
    slice.push_back(slice1);
    u64 baseOffset = 0;
    std::vector<u32> nicRankList;
    nicRankList.push_back(0);
    s32 profStage = 0;
    // 创建exchanger
    std::string unique_id = "ThreadManageTest";
    IntraExchanger exchanger{};
    std::vector<RankInfo> para_vector(1);
    // 创建信号

    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    notifyPool.reset(new (std::nothrow) NotifyPool());
    EXPECT_NE(notifyPool, nullptr);
    ret = notifyPool->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::unique_ptr<QueueNotifyManager> queueNotifyManager = nullptr;
    queueNotifyManager.reset(new (std::nothrow) QueueNotifyManager());
    EXPECT_NE(queueNotifyManager, nullptr);
    ret = queueNotifyManager->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<LocalNotify> signalAux = nullptr;
    std::shared_ptr<LocalNotify> signalMain = nullptr;
    std::string tag = "signal_test";
    std::vector<std::shared_ptr<LocalNotify>> notifys(2, nullptr);
    ret = queueNotifyManager->Alloc(tag, 2, notifys);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    signalMain = notifys[0];
    signalAux = notifys[1];

    // 创建comm
    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;

    SubCommInfo comm_inner = {0, 1};
    u32 ringIndex = 0;
    ExecutorType type = ExecutorType::REDUCE_SCATTER_RING;
    u64 reduceAttr = 0;
    ThreadManage* threadM = new (std::nothrow) ThreadManage(device_id, 0, dispatcher);
    EXPECT_NE(threadM, nullptr);
    ret = threadM->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = threadM->Prepare(
        inputMem, outputMem, inputMem, count, dataType, stream, op, root, slice, 0, nicRankList, "tag", profStage,
        comm_inner, signalAux, signalMain, ringIndex, ExecutorType::REDUCE_SCATTER_RING, reduceAttr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    threadM->NotifyStart();
    threadM->WaitDone();
    threadM->Finalize();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete threadM;
    signalMain = nullptr;
    signalAux = nullptr;
    ret = aclrtDestroyStream(rtstream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    inputMem.free();
    outputMem.free();
}

TEST_F(ThreadManageTest, threadMange_test_005)
{
    s32 ret = HCCL_SUCCESS;
    s32 device_id = 0;
    // 创建输入输出内存
    u32 memSize = 1024;
    DeviceMem inputMem = DeviceMem::alloc(memSize);
    DeviceMem outputMem = DeviceMem::alloc(memSize);
    sal_memset(inputMem.ptr(), memSize, 1, memSize);
    sal_memset(outputMem.ptr(), memSize, 0, memSize);
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    u64 count = memSize;
    // 创建流
    rtStream_t rtstream;
    aclrtCreateStream(&rtstream);
    Stream stream(rtstream);
    HcclReduceOp op = HCCL_REDUCE_SUM;
    u32 root = 0;
    std::vector<Slice> slice;
    Slice slice1;
    slice1.size = 1024;
    slice1.offset = 0;
    slice.push_back(slice1);
    u64 baseOffset = 0;
    std::vector<u32> nicRankList;
    nicRankList.push_back(0);
    s32 profStage = 0;
    // 创建exchanger
    std::string unique_id = "ThreadManageTest";
    IntraExchanger exchanger{};
    std::vector<RankInfo> para_vector(1);
    // 创建信号

    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    notifyPool.reset(new (std::nothrow) NotifyPool());
    EXPECT_NE(notifyPool, nullptr);
    ret = notifyPool->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::unique_ptr<QueueNotifyManager> queueNotifyManager = nullptr;
    queueNotifyManager.reset(new (std::nothrow) QueueNotifyManager());
    EXPECT_NE(queueNotifyManager, nullptr);
    ret = queueNotifyManager->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<LocalNotify> signalAux = nullptr;
    std::shared_ptr<LocalNotify> signalMain = nullptr;
    std::string tag = "signal_test";
    std::vector<std::shared_ptr<LocalNotify>> notifys(2, nullptr);
    ret = queueNotifyManager->Alloc(tag, 2, notifys);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    signalMain = notifys[0];
    signalAux = notifys[1];
    // 创建comm
    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;

    SubCommInfo comm_inner = {0, 1};
    u32 ringIndex = 0;
    ExecutorType type = ExecutorType::REDUCE_SCATTER_RING;
    u64 reduceAttr = 0;
    ThreadManage* threadM = new (std::nothrow) ThreadManage(device_id, 0, dispatcher);
    EXPECT_NE(threadM, nullptr);
    ret = threadM->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = threadM->Prepare(
        inputMem, outputMem, inputMem, count, dataType, stream, op, root, slice, 0, nicRankList, "tag", profStage,
        comm_inner, signalAux, signalMain, ringIndex, ExecutorType::REDUCE_SCATTER_RING, reduceAttr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    threadM->NotifyStart();
    threadM->WaitDone();
    threadM->Finalize();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete threadM;
    signalMain = nullptr;
    signalAux = nullptr;
    ret = aclrtDestroyStream(rtstream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    inputMem.free();
    outputMem.free();
}

TEST_F(ThreadManageTest, threadMange_test_006)
{
    s32 ret = HCCL_SUCCESS;
    s32 device_id = 0;
    // 创建输入输出内存
    u32 memSize = 1024;
    DeviceMem inputMem = DeviceMem::alloc(memSize);
    DeviceMem outputMem = DeviceMem::alloc(memSize);
    sal_memset(inputMem.ptr(), memSize, 1, memSize);
    sal_memset(outputMem.ptr(), memSize, 0, memSize);
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    u64 count = memSize;
    // 创建流
    rtStream_t rtstream;
    aclrtCreateStream(&rtstream);
    Stream stream(rtstream);
    HcclReduceOp op = HCCL_REDUCE_SUM;
    u32 root = 0;
    std::vector<Slice> slice;
    Slice slice1;
    slice1.size = 1024;
    slice1.offset = 0;
    slice.push_back(slice1);
    u64 baseOffset = 0;
    std::vector<u32> nicRankList;
    nicRankList.push_back(0);
    s32 profStage = 0;
    // 创建exchanger
    std::string unique_id = "ThreadManageTest";
    IntraExchanger exchanger{};
    std::vector<RankInfo> para_vector(1);
    // 创建信号

    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    notifyPool.reset(new (std::nothrow) NotifyPool());
    EXPECT_NE(notifyPool, nullptr);
    ret = notifyPool->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::unique_ptr<QueueNotifyManager> queueNotifyManager = nullptr;
    queueNotifyManager.reset(new (std::nothrow) QueueNotifyManager());
    EXPECT_NE(queueNotifyManager, nullptr);
    ret = queueNotifyManager->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<LocalNotify> signalAux = nullptr;
    std::shared_ptr<LocalNotify> signalMain = nullptr;
    std::string tag = "signal_test";
    std::vector<std::shared_ptr<LocalNotify>> notifys(2, nullptr);
    ret = queueNotifyManager->Alloc(tag, 2, notifys);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    signalMain = notifys[0];
    signalAux = notifys[1];
    // 创建comm
    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;

    SubCommInfo comm_inner = {0, 1};
    u32 ringIndex = 0;
    ExecutorType type = ExecutorType::REDUCE_SCATTER_RING;
    u64 reduceAttr = 0;
    ThreadManage* threadM = new (std::nothrow) ThreadManage(device_id, 0, dispatcher);
    EXPECT_NE(threadM, nullptr);
    ret = threadM->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = threadM->Prepare(
        inputMem, outputMem, inputMem, count, dataType, stream, op, root, slice, 0, nicRankList, "tag", profStage,
        comm_inner, signalAux, signalMain, ringIndex, ExecutorType::ALLGATHER_RING, reduceAttr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    threadM->NotifyStart();
    threadM->WaitDone();
    threadM->Finalize();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete threadM;
    signalMain = nullptr;
    signalAux = nullptr;
    ret = aclrtDestroyStream(rtstream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    inputMem.free();
    outputMem.free();
}

TEST_F(ThreadManageTest, threadMange_test_007)
{
    s32 ret = HCCL_SUCCESS;
    s32 device_id = 0;
    // 创建输入输出内存
    u32 memSize = 1024;
    DeviceMem inputMem = DeviceMem::alloc(memSize);
    DeviceMem outputMem = DeviceMem::alloc(memSize);
    sal_memset(inputMem.ptr(), memSize, 1, memSize);
    sal_memset(outputMem.ptr(), memSize, 0, memSize);
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    u64 count = memSize;
    // 创建流
    rtStream_t rtstream;
    aclrtCreateStream(&rtstream);
    Stream stream(rtstream);
    HcclReduceOp op = HCCL_REDUCE_SUM;
    u32 root = 0;
    std::vector<Slice> slice;
    Slice slice1;
    slice1.size = 1024;
    slice1.offset = 0;
    slice.push_back(slice1);
    u64 baseOffset = 0;
    std::vector<u32> nicRankList;
    nicRankList.push_back(0);
    s32 profStage = 0;
    // 创建exchanger
    std::string unique_id = "ThreadManageTest";
    IntraExchanger exchanger{};
    std::vector<RankInfo> para_vector(1);
    // 创建信号

    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    notifyPool.reset(new (std::nothrow) NotifyPool());
    EXPECT_NE(notifyPool, nullptr);
    ret = notifyPool->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::unique_ptr<QueueNotifyManager> queueNotifyManager = nullptr;
    queueNotifyManager.reset(new (std::nothrow) QueueNotifyManager());
    EXPECT_NE(queueNotifyManager, nullptr);
    ret = queueNotifyManager->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<LocalNotify> signalAux = nullptr;
    std::shared_ptr<LocalNotify> signalMain = nullptr;
    std::string tag = "signal_test";
    std::vector<std::shared_ptr<LocalNotify>> notifys(2, nullptr);
    ret = queueNotifyManager->Alloc(tag, 2, notifys);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    signalMain = notifys[0];
    signalAux = notifys[1];
    // 创建comm
    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    SubCommInfo ringSubCommInfo{0, 1, {}};
    u32 ringIndex = 0;
    ExecutorType type = ExecutorType::REDUCE_SCATTER_RING;
    u64 reduceAttr = 0;
    ThreadManage* threadM = new (std::nothrow) ThreadManage(device_id, 0, dispatcher);
    EXPECT_NE(threadM, nullptr);
    ret = threadM->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = threadM->Prepare(
        inputMem, outputMem, inputMem, count, dataType, stream, op, root, slice, 0, nicRankList, "tag", profStage,
        ringSubCommInfo, signalAux, signalMain, ringIndex, ExecutorType::ALLGATHER_RING, reduceAttr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    threadM->NotifyStart();
    threadM->WaitDone();
    threadM->Finalize();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete threadM;
    signalMain = nullptr;
    signalAux = nullptr;
    ret = aclrtDestroyStream(rtstream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    inputMem.free();
    outputMem.free();
}
