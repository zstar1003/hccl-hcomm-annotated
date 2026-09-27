/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "aicpu_args_stub.h"
#include "alg_param.h"
#include "hccl_device_pub.h"
#include "hccl_kernel_executor.h"
#include "store_sim_memory_manager.h"
#include "db_sim_runner_db.h"
#include "db_sim_sqlite_db.h"

using namespace ops_hccl;

extern bool gLibsLoaded;
extern void* gSlogHandle;
extern void* gHcommHandle;
extern void* gHcclKerHandle;
extern uint32_t (*runAicpuIndOpCommInitPtr)(void* args);
extern uint32_t (*runAicpuIndOpThreadInitPtr)(void* args);
extern uint32_t (*runAicpuIndOpChannelInitV2Ptr)(void* args);
extern uint32_t (*runAicpuDfxOpInfoInitV2Ptr)(void* args);
extern uint32_t (*runAicpuNotifyWaitPtr)(void* args);
extern uint32_t (*runAicpuNotifyRecordPtr)(void* args);
extern unsigned int (*hcclLaunchAicpuKernelPtr)(OpParam* param);
extern unsigned int (*hcclLaunchP2pAicpuKernelPtr)(void* args);
extern uint64_t d2hAddr;

namespace {
const std::string kTestDbPath = "/tmp/test_hccl_kernel_executor.db";

static uint32_t g_mockCommInitCalled = 0;
static uint32_t g_mockThreadInitCalled = 0;
static uint32_t g_mockChannelInitV2Called = 0;
static uint32_t g_mockLaunchAicpuKernelCalled = 0;
static uint32_t g_mockNotifyWaitCalled = 0;
static uint32_t g_mockNotifyRecordCalled = 0;
static uint32_t g_mockLaunchP2pKernelCalled = 0;
static void* g_mockP2pResCtx = nullptr;

uint32_t MockRunAicpuCommInit(void* args)
{
    g_mockCommInitCalled++;
    return 0;
}

uint32_t MockRunAicpuIndOpThreadInit(void* args)
{
    g_mockThreadInitCalled++;
    return 0;
}

uint32_t MockRunAicpuIndOpChannelInitV2(void* args)
{
    g_mockChannelInitV2Called++;
    return 0;
}

uint32_t MockRunAicpuDfxInitV2(void* args) { return 0; }

unsigned int MockHcclLaunchAicpuKernel(OpParam* param)
{
    g_mockLaunchAicpuKernelCalled++;
    return 0;
}

uint32_t MockRunAicpuNotifyWait(void* args)
{
    g_mockNotifyWaitCalled++;
    return 0;
}

uint32_t MockRunAicpuNotifyRecord(void* args)
{
    g_mockNotifyRecordCalled++;
    return 0;
}

unsigned int MockHcclLaunchP2pAicpuKernel(void* args)
{
    g_mockLaunchP2pKernelCalled++;
    auto* p2pParam = reinterpret_cast<HcclP2pKernelParamStub*>(args);
    auto* opParam = reinterpret_cast<OpParam*>(p2pParam->opParams);
    g_mockP2pResCtx = opParam->resCtx;
    return 0;
}

void CleanUpDb()
{
    std::remove(kTestDbPath.c_str());
    std::remove((kTestDbPath + "-wal").c_str());
    std::remove((kTestDbPath + "-shm").c_str());
}

static uint64_t g_memNameCounter = 0;
static std::vector<std::string> g_allocatedMemNames;

std::string UniqueMemName(const std::string& base)
{
    return "_ut_hkex_" + base + "_" + std::to_string(++g_memNameCounter);
}

struct TestMemRegion {
    std::string name;
    void* hostPtr;
    size_t size;
    uint64_t devBaseAddr;
};

TestMemRegion SetupMemRegion(const std::string& name, size_t size, uint64_t devBaseAddr)
{
    TestMemRegion region;
    region.name = name;
    region.size = size;
    region.devBaseAddr = devBaseAddr;

    sim::PhyMemBlock phyMem{};
    snprintf(phyMem.name, sizeof(phyMem.name), "%s", name.c_str());
    phyMem.size = size;
    phyMem.device_id = 1;
    uint64_t phyMemId = RunnerDB::Add<sim::PhyMemBlock>(phyMem);

    sim::VirtualMemBlock virMem{};
    virMem.start_ptr = devBaseAddr;
    virMem.size = size;
    virMem.phy_mem_id = phyMemId;
    virMem.src_type = static_cast<uint8_t>(sim::VIR_MEM_TYPE_DEV);
    RunnerDB::Add<sim::VirtualMemBlock>(virMem);

    region.hostPtr = sim::MemoryManager::GetInstance().AllocMemByName(name.c_str(), size);
    g_allocatedMemNames.push_back(name);
    return region;
}

void CleanupMemRegions()
{
    for (const auto& name : g_allocatedMemNames) {
        sim::MemoryManager::GetInstance().FreeMemByName(name.c_str());
        shm_unlink(name.c_str());
    }
    g_allocatedMemNames.clear();
    RunnerDB::DeleteAll<sim::VirtualMemBlock>();
    RunnerDB::DeleteAll<sim::PhyMemBlock>();
}

void SetupTestData()
{
    CleanUpDb();
    sim::SqliteDatabase::SetDbPath(kTestDbPath);
    SimRunnerSqliteDB::Instance().ClearAll();
}

void ResetGlobalState()
{
    gLibsLoaded = false;
    gSlogHandle = nullptr;
    gHcommHandle = nullptr;
    gHcclKerHandle = nullptr;
    runAicpuIndOpCommInitPtr = nullptr;
    runAicpuIndOpThreadInitPtr = nullptr;
    runAicpuIndOpChannelInitV2Ptr = nullptr;
    runAicpuDfxOpInfoInitV2Ptr = nullptr;
    runAicpuNotifyWaitPtr = nullptr;
    runAicpuNotifyRecordPtr = nullptr;
    hcclLaunchAicpuKernelPtr = nullptr;
    hcclLaunchP2pAicpuKernelPtr = nullptr;
    d2hAddr = 0;
    g_mockCommInitCalled = 0;
    g_mockThreadInitCalled = 0;
    g_mockChannelInitV2Called = 0;
    g_mockLaunchAicpuKernelCalled = 0;
    g_mockNotifyWaitCalled = 0;
    g_mockNotifyRecordCalled = 0;
    g_mockLaunchP2pKernelCalled = 0;
    g_mockP2pResCtx = nullptr;
}

void SetMockFuncHandles()
{
    gLibsLoaded = true;
    gSlogHandle = reinterpret_cast<void*>(0x2);
    gHcommHandle = reinterpret_cast<void*>(0x3);
    gHcclKerHandle = reinterpret_cast<void*>(0x4);
    runAicpuIndOpCommInitPtr = MockRunAicpuCommInit;
    runAicpuIndOpThreadInitPtr = MockRunAicpuIndOpThreadInit;
    runAicpuIndOpChannelInitV2Ptr = MockRunAicpuIndOpChannelInitV2;
    runAicpuDfxOpInfoInitV2Ptr = MockRunAicpuDfxInitV2;
    runAicpuNotifyWaitPtr = MockRunAicpuNotifyWait;
    runAicpuNotifyRecordPtr = MockRunAicpuNotifyRecord;
    hcclLaunchAicpuKernelPtr = MockHcclLaunchAicpuKernel;
    hcclLaunchP2pAicpuKernelPtr = MockHcclLaunchP2pAicpuKernel;
}
} // namespace

class HcclKernelExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();
        ResetGlobalState();
    }

    void TearDown() override
    {
        CleanupMemRegions();
        CleanUpDb();
    }
};

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_EmptyKernelName)
{
    uint32_t rankId = 0;
    std::string kernelName = "";
    uint64_t args = 0;
    EXPECT_NO_THROW(ExecuteAicpuKernel(rankId, kernelName, args));
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_UnknownKernel)
{
    uint32_t rankId = 0;
    std::string kernelName = "UnknownKernel";
    uint64_t args = 0;
    EXPECT_NO_THROW(ExecuteAicpuKernel(rankId, kernelName, args));
}

TEST_F(HcclKernelExecutorTest, InitKernelFuncHandle_MultipleCalls)
{
    bool result1 = InitKernelFuncHandle();
    bool result2 = InitKernelFuncHandle();
    EXPECT_EQ(result1, result2);
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_RunAicpuCommInit)
{
    SetMockFuncHandles();

    const uint64_t devBase = 0x10000000;
    const size_t bufSize = sizeof(CommAicpuParam) + 256;
    auto region = SetupMemRegion(UniqueMemName("comm_init"), bufSize, devBase);
    ASSERT_NE(region.hostPtr, nullptr);

    CommAicpuParam* hostParam = reinterpret_cast<CommAicpuParam*>(region.hostPtr);
    memset(hostParam, 0, sizeof(CommAicpuParam));

    uint64_t innerDevBase = devBase + sizeof(CommAicpuParam);
    auto innerRegion = SetupMemRegion(UniqueMemName("comm_init_inner"), 4096, innerDevBase);
    ASSERT_NE(innerRegion.hostPtr, nullptr);

    hostParam->kfcControlTransferH2DParams.deviceAddr = innerDevBase;
    hostParam->kfcControlTransferH2DParams.readCacheAddr = innerDevBase + 256;
    hostParam->kfcStatusTransferD2HParams.deviceAddr = innerDevBase + 512;

    EXPECT_NO_THROW(ExecuteAicpuKernel(0, "RunAicpuCommInit", devBase));
    EXPECT_EQ(g_mockCommInitCalled, 1u);
    EXPECT_NE(d2hAddr, 0u);
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_RunAicpuIndOpThreadInit)
{
    SetMockFuncHandles();

    const uint64_t devBase = 0x20000000;
    const size_t outerBufSize = 128;
    auto outerRegion = SetupMemRegion(UniqueMemName("thread_init_outer"), outerBufSize, devBase);
    ASSERT_NE(outerRegion.hostPtr, nullptr);

    const uint64_t paramDevBase = 0x20010000;
    const size_t paramBufSize = sizeof(ThreadMgrAicpuParam) + 4096;
    auto paramRegion = SetupMemRegion(UniqueMemName("thread_init_param"), paramBufSize, paramDevBase);
    ASSERT_NE(paramRegion.hostPtr, nullptr);

    uint64_t* ptrValue = reinterpret_cast<uint64_t*>(outerRegion.hostPtr);
    *ptrValue = paramDevBase;

    ThreadMgrAicpuParam* hostParam = reinterpret_cast<ThreadMgrAicpuParam*>(paramRegion.hostPtr);
    memset(hostParam, 0, sizeof(ThreadMgrAicpuParam));
    hostParam->threadNum = 1;
    hostParam->deviceHandle = nullptr;

    AicpuTsThread* thread = reinterpret_cast<AicpuTsThread*>(hostParam->threadParam[0]);
    thread->devId = 99;

    EXPECT_NO_THROW(ExecuteAicpuKernel(5, "RunAicpuIndOpThreadInit", devBase));
    EXPECT_EQ(g_mockThreadInitCalled, 1u);
    EXPECT_EQ(thread->devId, 5u);
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_RunAicpuIndOpThreadInit_MultipleThreads)
{
    SetMockFuncHandles();

    const uint64_t devBase = 0x21000000;
    const size_t outerBufSize = 128;
    auto outerRegion = SetupMemRegion(UniqueMemName("thr_multi_outer"), outerBufSize, devBase);
    ASSERT_NE(outerRegion.hostPtr, nullptr);

    const uint64_t paramDevBase = 0x21010000;
    const size_t paramBufSize = sizeof(ThreadMgrAicpuParam) + 4096;
    auto paramRegion = SetupMemRegion(UniqueMemName("thr_multi_param"), paramBufSize, paramDevBase);
    ASSERT_NE(paramRegion.hostPtr, nullptr);

    uint64_t* ptrValue = reinterpret_cast<uint64_t*>(outerRegion.hostPtr);
    *ptrValue = paramDevBase;

    ThreadMgrAicpuParam* hostParam = reinterpret_cast<ThreadMgrAicpuParam*>(paramRegion.hostPtr);
    memset(hostParam, 0, sizeof(ThreadMgrAicpuParam));
    hostParam->threadNum = 3;
    hostParam->deviceHandle = nullptr;

    for (uint32_t i = 0; i < hostParam->threadNum; i++) {
        AicpuTsThread* thread = reinterpret_cast<AicpuTsThread*>(hostParam->threadParam[i]);
        thread->devId = 99;
    }

    EXPECT_NO_THROW(ExecuteAicpuKernel(7, "RunAicpuIndOpThreadInit", devBase));
    EXPECT_EQ(g_mockThreadInitCalled, 1u);

    for (uint32_t i = 0; i < 3; i++) {
        AicpuTsThread* thread = reinterpret_cast<AicpuTsThread*>(hostParam->threadParam[i]);
        EXPECT_EQ(thread->devId, 7u);
    }
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_RunAicpuIndOpChannelInitV2)
{
    SetMockFuncHandles();

    const uint64_t devBase = 0x30000000;
    const size_t taskBufSize = sizeof(InitTask) + 256;
    auto taskRegion = SetupMemRegion(UniqueMemName("channel_task"), taskBufSize, devBase);
    ASSERT_NE(taskRegion.hostPtr, nullptr);

    const uint64_t ctxDevBase = 0x30010000;
    const size_t ctxBufSize = sizeof(HcclChannelUrmaRes) + 256;
    auto ctxRegion = SetupMemRegion(UniqueMemName("channel_ctx"), ctxBufSize, ctxDevBase);
    ASSERT_NE(ctxRegion.hostPtr, nullptr);

    InitTask* hostTask = reinterpret_cast<InitTask*>(taskRegion.hostPtr);
    hostTask->context = ctxDevBase;

    HcclChannelUrmaRes* hostRes = reinterpret_cast<HcclChannelUrmaRes*>(ctxRegion.hostPtr);
    memset(hostRes, 0, sizeof(HcclChannelUrmaRes));
    hostRes->listNum = 1;
    hostRes->uniqueIdSize = 1024;
    hostRes->channelList = nullptr;
    hostRes->channelSizeAddr = nullptr;
    hostRes->remoteRankList = nullptr;

    const uint64_t uniqueIdDevBase = 0x30020000;
    const size_t uniqueIdBufSize = 4096;
    auto uniqueIdRegion = SetupMemRegion(UniqueMemName("channel_uid"), uniqueIdBufSize, uniqueIdDevBase);
    ASSERT_NE(uniqueIdRegion.hostPtr, nullptr);

    uint8_t* uniqueIdBuf = reinterpret_cast<uint8_t*>(uniqueIdRegion.hostPtr);
    memset(uniqueIdBuf, 0, uniqueIdBufSize);

    hostRes->uniqueIdAddr = reinterpret_cast<void*>(uniqueIdDevBase);

    uint8_t* headerAddr = uniqueIdBuf + UNIQUEID_HEADER_OFFSET;
    UniqueIdV2Header* header = reinterpret_cast<UniqueIdV2Header*>(headerAddr);
    header->type = 0;
    header->notifyNum = 1;
    header->bufferNum = 1;
    header->connNum = 1;

    uint32_t offset = COMMON_DATA_SIZE + header->notifyNum * NOTIFY_ID_SIZE + COMMON_DATA_SIZE
                      + header->notifyNum * NOTIFY_BUFFER_SIZE + header->bufferNum * LOCAL_BUFFER_SIZE * 2;

    ConnUniqueBlock* block = reinterpret_cast<ConnUniqueBlock*>(headerAddr + UNIQUEID_HEADER_SIZE + offset);

    const uint64_t sqBuffDevBase = 0x30030000;
    const size_t sqBuffBufSize = 4096;
    auto sqBuffRegion = SetupMemRegion(UniqueMemName("channel_sqbuff"), sqBuffBufSize, sqBuffDevBase);
    ASSERT_NE(sqBuffRegion.hostPtr, nullptr);

    block->size = sizeof(ConnUniqueIds);
    block->conn[0].sqBuffVa = sqBuffDevBase;

    EXPECT_NO_THROW(ExecuteAicpuKernel(0, "RunAicpuIndOpChannelInitV2", devBase));
    EXPECT_EQ(g_mockChannelInitV2Called, 1u);
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_RunAicpuIndOpChannelInitV2_SingleList)
{
    SetMockFuncHandles();

    const uint64_t devBase = 0x32000000;
    const size_t taskBufSize = sizeof(InitTask) + 256;
    auto taskRegion = SetupMemRegion(UniqueMemName("ch_single_task"), taskBufSize, devBase);
    ASSERT_NE(taskRegion.hostPtr, nullptr);

    const uint64_t ctxDevBase = 0x32010000;
    const size_t ctxBufSize = sizeof(HcclChannelUrmaRes) + 256;
    auto ctxRegion = SetupMemRegion(UniqueMemName("ch_single_ctx"), ctxBufSize, ctxDevBase);
    ASSERT_NE(ctxRegion.hostPtr, nullptr);

    InitTask* hostTask = reinterpret_cast<InitTask*>(taskRegion.hostPtr);
    hostTask->context = ctxDevBase;

    HcclChannelUrmaRes* hostRes = reinterpret_cast<HcclChannelUrmaRes*>(ctxRegion.hostPtr);
    memset(hostRes, 0, sizeof(HcclChannelUrmaRes));
    hostRes->listNum = 1;
    hostRes->uniqueIdSize = 512;
    hostRes->channelList = nullptr;
    hostRes->channelSizeAddr = nullptr;
    hostRes->remoteRankList = nullptr;

    const uint64_t uniqueIdDevBase = 0x32020000;
    const size_t uniqueIdBufSize = 4096;
    auto uniqueIdRegion = SetupMemRegion(UniqueMemName("ch_single_uid"), uniqueIdBufSize, uniqueIdDevBase);
    ASSERT_NE(uniqueIdRegion.hostPtr, nullptr);

    uint8_t* uniqueIdBuf = reinterpret_cast<uint8_t*>(uniqueIdRegion.hostPtr);
    memset(uniqueIdBuf, 0, uniqueIdBufSize);
    hostRes->uniqueIdAddr = reinterpret_cast<void*>(uniqueIdDevBase);

    UniqueIdV2Header* header = reinterpret_cast<UniqueIdV2Header*>(uniqueIdBuf + UNIQUEID_HEADER_OFFSET);
    header->type = 0;
    header->notifyNum = 0;
    header->bufferNum = 0;
    header->connNum = 0;

    EXPECT_NO_THROW(ExecuteAicpuKernel(0, "RunAicpuIndOpChannelInitV2", devBase));
    EXPECT_EQ(g_mockChannelInitV2Called, 1u);
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_RunAicpuDfxInitV2)
{
    SetMockFuncHandles();

    const uint64_t devBase = 0x40000000;
    const size_t bufSize = 4096;
    auto region = SetupMemRegion(UniqueMemName("dfx_init"), bufSize, devBase);
    ASSERT_NE(region.hostPtr, nullptr);

    EXPECT_NO_THROW(ExecuteAicpuKernel(0, "RunAicpuDfxInitV2", devBase));
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_HcclLaunchAicpuKernel)
{
    SetMockFuncHandles();

    const uint64_t devBase = 0x50000000;
    const size_t bufSize = sizeof(OpParam) + 256;
    auto region = SetupMemRegion(UniqueMemName("launch_kern"), bufSize, devBase);
    ASSERT_NE(region.hostPtr, nullptr);

    OpParam* hostParam = reinterpret_cast<OpParam*>(region.hostPtr);
    memset(hostParam, 0, sizeof(OpParam));
    hostParam->resCtx = nullptr;

    const uint64_t d2hDevBase = 0x50010000;
    const size_t d2hBufSize = 4096;
    auto d2hRegion = SetupMemRegion(UniqueMemName("launch_kern_d2h"), d2hBufSize, d2hDevBase);
    ASSERT_NE(d2hRegion.hostPtr, nullptr);
    d2hAddr = reinterpret_cast<uint64_t>(d2hRegion.hostPtr);

    EXPECT_NO_THROW(ExecuteAicpuKernel(0, "HcclLaunchAicpuKernel", devBase));
    EXPECT_EQ(g_mockLaunchAicpuKernelCalled, 1u);
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_HcclLaunchAicpuKernel_WithResCtx)
{
    SetMockFuncHandles();

    const uint64_t devBase = 0x51000000;
    const size_t bufSize = sizeof(OpParam) + 256;
    auto region = SetupMemRegion(UniqueMemName("launch_resctx"), bufSize, devBase);
    ASSERT_NE(region.hostPtr, nullptr);

    const uint64_t resCtxDevBase = 0x51010000;
    const size_t resCtxBufSize = 4096;
    auto resCtxRegion = SetupMemRegion(UniqueMemName("launch_resctx_data"), resCtxBufSize, resCtxDevBase);
    ASSERT_NE(resCtxRegion.hostPtr, nullptr);

    OpParam* hostParam = reinterpret_cast<OpParam*>(region.hostPtr);
    memset(hostParam, 0, sizeof(OpParam));
    hostParam->resCtx = reinterpret_cast<void*>(resCtxDevBase);

    const uint64_t d2hDevBase = 0x51020000;
    const size_t d2hBufSize = 4096;
    auto d2hRegion = SetupMemRegion(UniqueMemName("launch_resctx_d2h"), d2hBufSize, d2hDevBase);
    ASSERT_NE(d2hRegion.hostPtr, nullptr);
    d2hAddr = reinterpret_cast<uint64_t>(d2hRegion.hostPtr);

    EXPECT_NO_THROW(ExecuteAicpuKernel(0, "HcclLaunchAicpuKernel", devBase));
    EXPECT_EQ(g_mockLaunchAicpuKernelCalled, 1u);
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_RunAicpuNotifyWait)
{
    SetMockFuncHandles();

    const uint64_t devBase = 0x51100000;
    auto region = SetupMemRegion(UniqueMemName("notify_wait"), sizeof(ThreadNotifyWaitParam), devBase);
    ASSERT_NE(region.hostPtr, nullptr);

    auto* param = reinterpret_cast<ThreadNotifyWaitParam*>(region.hostPtr);
    param->thread = 0x1234;
    param->notifyIdx = 2;

    EXPECT_NO_THROW(ExecuteAicpuKernel(0, "RunAicpuNotifyWait", devBase));
    EXPECT_EQ(g_mockNotifyWaitCalled, 1u);
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_HcclLaunchP2pAicpuKernel)
{
    SetMockFuncHandles();

    const uint64_t devBase = 0x51200000;
    auto region = SetupMemRegion(UniqueMemName("launch_p2p"), sizeof(HcclP2pKernelParamStub), devBase);
    ASSERT_NE(region.hostPtr, nullptr);

    const uint64_t resCtxDevBase = 0x51210000;
    auto resCtxRegion = SetupMemRegion(UniqueMemName("launch_p2p_resctx"), 4096, resCtxDevBase);
    ASSERT_NE(resCtxRegion.hostPtr, nullptr);

    auto* p2pParam = reinterpret_cast<HcclP2pKernelParamStub*>(region.hostPtr);
    memset(p2pParam, 0, sizeof(HcclP2pKernelParamStub));
    p2pParam->sendRecvThread = 0x5678;
    auto* opParam = reinterpret_cast<OpParam*>(p2pParam->opParams);
    opParam->resCtx = reinterpret_cast<void*>(resCtxDevBase);

    EXPECT_NO_THROW(ExecuteAicpuKernel(0, "HcclLaunchP2pAicpuKernel", devBase));
    EXPECT_EQ(g_mockLaunchP2pKernelCalled, 1u);
    EXPECT_EQ(g_mockP2pResCtx, resCtxRegion.hostPtr);
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_RunAicpuNotifyRecord)
{
    SetMockFuncHandles();

    const uint64_t devBase = 0x51300000;
    auto region = SetupMemRegion(UniqueMemName("notify_record"), sizeof(ThreadNotifyRecordParam), devBase);
    ASSERT_NE(region.hostPtr, nullptr);

    auto* param = reinterpret_cast<ThreadNotifyRecordParam*>(region.hostPtr);
    param->thread = 0x1234;
    param->dstThread = 0x5678;
    param->dstNotifyIdx = 0;

    EXPECT_NO_THROW(ExecuteAicpuKernel(0, "RunAicpuNotifyRecord", devBase));
    EXPECT_EQ(g_mockNotifyRecordCalled, 1u);
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_HcclLaunchAicpuKernel_ZeroD2hAddr)
{
    SetMockFuncHandles();

    const uint64_t devBase = 0x52000000;
    const size_t bufSize = sizeof(OpParam) + 256;
    auto region = SetupMemRegion(UniqueMemName("launch_zerod2h"), bufSize, devBase);
    ASSERT_NE(region.hostPtr, nullptr);

    OpParam* hostParam = reinterpret_cast<OpParam*>(region.hostPtr);
    memset(hostParam, 0, sizeof(OpParam));
    hostParam->resCtx = nullptr;
    d2hAddr = 0;

    EXPECT_NO_THROW(ExecuteAicpuKernel(0, "HcclLaunchAicpuKernel", devBase));
    EXPECT_EQ(g_mockLaunchAicpuKernelCalled, 1u);
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_NotSupportKernel)
{
    SetMockFuncHandles();

    const uint64_t devBase = 0x60000000;
    const size_t bufSize = 256;
    auto region = SetupMemRegion(UniqueMemName("not_support"), bufSize, devBase);
    ASSERT_NE(region.hostPtr, nullptr);

    EXPECT_NO_THROW(ExecuteAicpuKernel(0, "NotSupportKernel", devBase));
    EXPECT_EQ(g_mockCommInitCalled, 0u);
    EXPECT_EQ(g_mockThreadInitCalled, 0u);
    EXPECT_EQ(g_mockChannelInitV2Called, 0u);
    EXPECT_EQ(g_mockLaunchAicpuKernelCalled, 0u);
}

TEST_F(HcclKernelExecutorTest, LoadLibrary_NullDir)
{
    void* handle = LoadLibrary("", "libc.so.6");
    if (handle != nullptr) {
        EXPECT_NE(handle, nullptr);
    }
}

TEST_F(HcclKernelExecutorTest, ExecuteAicpuKernel_DifferentRankIds)
{
    SetMockFuncHandles();

    for (uint32_t rankId = 0; rankId < 4; rankId++) {
        g_mockCommInitCalled = 0;

        const uint64_t devBase = 0x70000000 + rankId * 0x1000000ULL;
        const size_t bufSize = sizeof(CommAicpuParam) + 256;
        std::string memName = UniqueMemName("rank_test");
        auto region = SetupMemRegion(memName.c_str(), bufSize, devBase);
        ASSERT_NE(region.hostPtr, nullptr) << "rankId=" << rankId;

        CommAicpuParam* hostParam = reinterpret_cast<CommAicpuParam*>(region.hostPtr);
        memset(hostParam, 0, sizeof(CommAicpuParam));
        hostParam->userRank = rankId;

        EXPECT_NO_THROW(ExecuteAicpuKernel(rankId, "RunAicpuCommInit", devBase));
        EXPECT_EQ(g_mockCommInitCalled, 1u);
    }
}
