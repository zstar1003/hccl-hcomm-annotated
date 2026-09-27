/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#include <memory>
#include <string>
#include <cstring>
#include "hccl_types.h"
#include "adapter_rts_common.h"
#include "endpoint.h"
#include "urma_endpoint.h"
#include "proc_reged_mem_mgr_cache.h"
#include "log.h"
#define private public
#include "endpoint_monitor.h"
#undef private

using namespace hcomm;

class EndpointMonitorTest : public ::testing::Test {
public:
    void SetUp() override { testing::Test::SetUp(); }
    void TearDown() override
    {
        testing::Test::TearDown();
        GlobalMockObject::verify();
    }

    EndpointMonitor& g_monitor = *EndpointMonitor::GetHolder(0);
};

// GetAsyncEvents 已下移为 UrmaEndpoint 自有方法（基类无默认实现），monitor 按具体类型 downcast 访问；
// 桩改继承 UrmaEndpoint，使 ProcessUbAsyncEvents 的 dynamic_cast 命中，GetAsyncEvents 经 MOCKER_CPP 打桩
class UtStubEndpoint : public UrmaEndpoint {
public:
    explicit UtStubEndpoint(const EndpointDesc& endpointDesc) : UrmaEndpoint(endpointDesc) {}
    ~UtStubEndpoint() noexcept override { (void)ReleaseCache(); }

    void HoldCacheForTest() { cacheKeepAlive_ = ProcRegedMemMgrCache::GetHolder(); }

    // 对齐派生类 ReleaseCache 形态：判空返 HCCL_E_PTR，正常路径返 HCCL_SUCCESS
    HcclResult ReleaseCache()
    {
        if (cacheKeepAlive_ == nullptr) {
            HCCL_WARNING("[UtStubEndpoint][%s] cacheKeepAlive_ is null, nothing to release", __func__);
            return HCCL_E_PTR;
        }
        cacheKeepAlive_->Release(cacheKey_);
        cacheKeepAlive_.reset();
        return HCCL_SUCCESS;
    }

    HcclResult Init() { return HCCL_SUCCESS; }

    HcclResult ServerSocketListen(const uint32_t port) { return HCCL_SUCCESS; }

    // Endpoint 新增纯虚接口，UtStubEndpoint 提供最小实现
    RegedMemMgr* GetRegedMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return nullptr; }
    bool IsCtxHandleValid() const override { return false; }

    // 注册内存
    HcclResult RegisterMemory(HcommMem mem, const char* memTag, void** memHandle) { return HCCL_SUCCESS; }

    // 注销内存
    HcclResult UnregisterMemory(void* memHandle) { return HCCL_SUCCESS; }

    // 导出指定内存描述，用于交换
    HcclResult MemoryExport(void* memHandle, void** memDesc, uint32_t* memDescLen) { return HCCL_SUCCESS; }

    // 基于内存描述，导入获得内存
    HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) { return HCCL_SUCCESS; }

    // 关闭内存
    HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) { return HCCL_SUCCESS; }

    HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) { return HCCL_SUCCESS; }

private:
    MemMgrCacheKey cacheKey_{};
    std::shared_ptr<ProcRegedMemMgrCache> cacheKeepAlive_{};
};

TEST_F(EndpointMonitorTest, Ut_PrintUbAsyncEventsContext_When_ContextLenExceedMax_Expect_Return)
{
    u32 devPhyId = 0;
    struct AsyncEvent event;
    event.resId = 1;
    event.eventType = 2;
    event.len = CONTEXT_MAX_LEN + 1;
    memset(event.context, 0, CONTEXT_MAX_LEN);

    g_monitor.PrintUbAsyncEventsContext(nullptr, 0, event);
}

TEST_F(EndpointMonitorTest, Ut_PrintUbAsyncEventsContext_When_NormalContext_Expect_PrintInfo)
{
    u32 devPhyId = 1;
    struct AsyncEvent event;
    event.resId = 100;
    event.eventType = 200;
    event.len = 11;
    for (unsigned int i = 0; i < event.len; i++) {
        event.context[i] = static_cast<uint8_t>(i);
    }

    g_monitor.PrintUbAsyncEventsContext(nullptr, 0, event);
}

TEST_F(EndpointMonitorTest, Ut_ProcessUbAsyncEvents_When_NoEndpointHandle_Expect_Return)
{
    g_monitor.epHandleSet_.clear();
    g_monitor.ProcessUbAsyncEvents();
}

TEST_F(EndpointMonitorTest, Ut_ProcessUbAsyncEvents_CoverAllBranches)
{
    u32 devPhyId = 0;
    EndpointDesc desc;
    UtStubEndpoint myUtEndpoint(desc);

    EXPECT_EQ(g_monitor.RegisterToEndpointMonitor(-1, reinterpret_cast<EndpointHandle>(&myUtEndpoint)), HCCL_E_PARA);
    EXPECT_EQ(
        g_monitor.RegisterToEndpointMonitor(MAX_MODULE_DEVICE_NUM, reinterpret_cast<EndpointHandle>(&myUtEndpoint)),
        HCCL_E_PARA);
    EXPECT_EQ(g_monitor.RegisterToEndpointMonitor(0, nullptr), HCCL_E_PTR);

    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any(), outBound(devPhyId)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&UrmaEndpoint::GetAsyncEvents).stubs().will(returnValue(HCCL_E_NOT_SUPPORT));
    EXPECT_EQ(g_monitor.RegisterToEndpointMonitor(0, reinterpret_cast<EndpointHandle>(&myUtEndpoint)), HCCL_SUCCESS);
    EXPECT_EQ(g_monitor.UnRegisterToEndpointMonitor(), HCCL_SUCCESS);
    EXPECT_EQ(g_monitor.epHandleSet_.size(), 0);
    GlobalMockObject::verify();

    g_monitor.epHandleSet_.emplace(reinterpret_cast<u64>(&myUtEndpoint));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any(), outBound(devPhyId)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&UrmaEndpoint::GetAsyncEvents)
        .stubs()
        .will(returnValue(HCCL_SUCCESS))
        .then(returnValue(HCCL_E_NOT_SUPPORT))
        .then(returnValue(HCCL_E_INTERNAL));
    MOCKER(&EndpointMonitor::PrintUbAsyncEventsContext).stubs();

    g_monitor.ProcessUbAsyncEvents();
    EXPECT_EQ(g_monitor.epHandleSet_.size(), 1);

    g_monitor.ProcessUbAsyncEvents();
    EXPECT_EQ(g_monitor.epHandleSet_.size(), 0);

    g_monitor.ProcessUbAsyncEvents();
    EXPECT_EQ(g_monitor.epHandleSet_.size(), 0);
    GlobalMockObject::verify();

    g_monitor.epHandleSet_.clear();
}

TEST_F(EndpointMonitorTest, Ut_ProcessUbAsyncEvents_RemoveEpHandleToEndpointMonitor)
{
    u32 devPhyId = 0;
    EndpointDesc desc;
    UtStubEndpoint myUtEndpoint(desc);

    g_monitor.epHandleSet_.emplace(reinterpret_cast<u64>(&myUtEndpoint));
    g_monitor.RemoveEpHandleFromEndpointMonitor(nullptr);
    EXPECT_EQ(g_monitor.epHandleSet_.size(), 1);
    g_monitor.RemoveEpHandleFromEndpointMonitor(reinterpret_cast<EndpointHandle>(&myUtEndpoint));
    EXPECT_EQ(g_monitor.epHandleSet_.size(), 0);
    g_monitor.epHandleSet_.clear();
}

// Destroy / 析构只用成员 shared_ptr，拦截再调 GetHolder。
TEST_F(EndpointMonitorTest, Ut_ReleaseEndpointMonitor_When_NotAttached_Expect_SkipAndNotCallGetHolder)
{
    EndpointDesc desc;
    UtStubEndpoint ep(desc);
    MOCKER_CPP(&EndpointMonitor::GetHolder).expects(never());
    ep.ReleaseEndpointMonitor(reinterpret_cast<EndpointHandle>(&ep));
}

TEST_F(EndpointMonitorTest, Ut_ReleaseEndpointMonitor_When_Attached_Expect_RemoveWithoutGetHolder)
{
    EndpointDesc desc;
    UtStubEndpoint ep(desc);
    auto handle = reinterpret_cast<EndpointHandle>(&ep);
    ep.AttachMonitor(0);
    g_monitor.epHandleSet_.emplace(reinterpret_cast<u64>(handle));

    MOCKER_CPP(&EndpointMonitor::GetHolder).expects(never());
    ep.ReleaseEndpointMonitor(handle);
    EXPECT_EQ(g_monitor.epHandleSet_.size(), 0);
}

TEST_F(EndpointMonitorTest, Ut_ReleaseEndpointMonitor_When_ReleasedTwice_Expect_SecondSkipGetHolder)
{
    EndpointDesc desc;
    UtStubEndpoint ep(desc);
    auto handle = reinterpret_cast<EndpointHandle>(&ep);
    ep.AttachMonitor(0);
    ep.ReleaseEndpointMonitor(handle);

    MOCKER_CPP(&EndpointMonitor::GetHolder).expects(never());
    ep.ReleaseEndpointMonitor(handle);
}

TEST_F(EndpointMonitorTest, Ut_RegisterToEndpointMonitor_When_NotAttached_Expect_InternalAndNotCallGetHolder)
{
    EndpointDesc desc;
    UtStubEndpoint ep(desc);
    MOCKER_CPP(&EndpointMonitor::GetHolder).expects(never());
    EXPECT_EQ(ep.RegisterToEndpointMonitor(0, reinterpret_cast<EndpointHandle>(&ep)), HCCL_E_INTERNAL);
}

TEST_F(EndpointMonitorTest, Ut_Dtor_When_MonitorAttached_Expect_NotCallGetHolder)
{
    EndpointDesc desc;
    auto ep = std::make_unique<UtStubEndpoint>(desc);
    ep->AttachMonitor(0);
    MOCKER_CPP(&EndpointMonitor::GetHolder).expects(never());
    ep.reset();
}

TEST_F(EndpointMonitorTest, Ut_ReleaseCache_When_NotAttached_Expect_SkipAndNotCallGetHolder)
{
    EndpointDesc desc;
    UtStubEndpoint ep(desc);
    MOCKER_CPP(&ProcRegedMemMgrCache::GetHolder).expects(never());
    EXPECT_EQ(ep.ReleaseCache(), HCCL_E_PTR);
}

TEST_F(EndpointMonitorTest, Ut_ReleaseCache_When_Attached_Expect_ReleaseWithoutGetHolder)
{
    EndpointDesc desc;
    UtStubEndpoint ep(desc);
    ep.HoldCacheForTest();
    MOCKER_CPP(&ProcRegedMemMgrCache::GetHolder).expects(never());
    EXPECT_EQ(ep.ReleaseCache(), HCCL_SUCCESS);
}

TEST_F(EndpointMonitorTest, Ut_Dtor_When_CacheAttached_Expect_NotCallGetHolder)
{
    EndpointDesc desc;
    auto ep = std::make_unique<UtStubEndpoint>(desc);
    ep->HoldCacheForTest();
    MOCKER_CPP(&ProcRegedMemMgrCache::GetHolder).expects(never());
    ep.reset();
}

TEST_F(EndpointMonitorTest, Ut_Dtor_When_CacheAndMonitorAttached_Expect_NotCallGetHolder)
{
    EndpointDesc desc;
    auto ep = std::make_unique<UtStubEndpoint>(desc);
    ep->AttachMonitor(0);
    ep->HoldCacheForTest();
    MOCKER_CPP(&EndpointMonitor::GetHolder).expects(never());
    MOCKER_CPP(&ProcRegedMemMgrCache::GetHolder).expects(never());
    ep.reset();
}
