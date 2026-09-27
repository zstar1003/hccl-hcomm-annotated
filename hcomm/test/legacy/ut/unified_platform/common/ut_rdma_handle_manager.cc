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
#define protected public
#define private public
#include "socket_manager.h"
#include "rma_conn_manager.h"
#include "rma_connection.h"
#include "p2p_connection.h"
#include "types.h"
#include "socket.h"
#include "communicator_impl.h"
#include "virtual_topo.h"
#include "op_mode.h"
#include "coll_operator.h"
#include "json_parser.h"
#include "rdma_handle_manager.h"
#include "dev_rdma_connection.h"
#include "rank_table.h"
#include "timeout_exception.h"
#include "ccu_context_mgr_imp.h"
#include "ccu_res_batch_allocator_legacy.h"
#include "ccu_component.h"
#undef protected
#undef private

using namespace Hccl;

class RdmaHandleManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RdmaHandleManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RdmaHandleManagerTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in RdmaHandleManagerTest SetUP" << std::endl;

        rdmaHandle = new int(0);
        MOCKER(HrtRaRdmaInit, RdmaHandle(HrtNetworkMode, RaInterface&))
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(rdmaHandle));

        BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete rdmaHandle;
        std::cout << "A Test case in RdmaHandleManagerTest TearDown" << std::endl;
    }

    IpAddress GetAnIpAddress()
    {
        IpAddress ipAddress("1.0.0.0");
        return ipAddress;
    }

    void* rdmaHandle;
};

TEST_F(RdmaHandleManagerTest, rdma_handle_manager_get_and_create)
{
    // Given
    u32 devicePhyId = 0;
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    PortData localPortData(0, basePortType, 0, IpAddress());

    // when
    auto res = RdmaHandleManager::GetInstance().Get(devicePhyId, localPortData, LinkProtocol::UB_CTP);

    // then
    EXPECT_EQ(rdmaHandle, res);
}

TEST_F(RdmaHandleManagerTest, rdma_handle_manager_get_twice)
{
    // Given
    u32 devicePhyId = 0;
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    PortData localPortData(0, basePortType, 0, IpAddress());

    // when
    auto res1 = RdmaHandleManager::GetInstance().Get(devicePhyId, localPortData, LinkProtocol::UB_CTP);
    auto res2 = RdmaHandleManager::GetInstance().Get(devicePhyId, localPortData, LinkProtocol::UB_CTP);

    // then
    EXPECT_EQ(res1, res2);
}

TEST_F(RdmaHandleManagerTest, rdma_handle_manager_get_jfc)
{
    RdmaHandle rdmaHandle = nullptr;
    HrtUbJfcMode mode;
    struct Hccl::CqCreateInfo cqInfo;
    EXPECT_THROW(RdmaHandleManager::GetInstance().GetJfcHandle(rdmaHandle, cqInfo, mode), InvalidParamsException);
    rdmaHandle = new RdmaHandle();
    EXPECT_THROW(RdmaHandleManager::GetInstance().GetJfcHandle(rdmaHandle, cqInfo, mode), InvalidParamsException);

    RdmaHandle rdmaHandle2 = nullptr;
    EXPECT_THROW(RdmaHandleManager::GetInstance().GetDieAndFuncId(rdmaHandle2), InvalidParamsException);
    delete rdmaHandle;
}

TEST_F(RdmaHandleManagerTest, rdma_handle_manager_get_token_id_handle)
{
    RdmaHandle rdmaHandle = nullptr;
    TokenIdHandle tokenIdHandle;
    EXPECT_THROW(
        RdmaHandleManager::GetInstance().GetTokenIdInfo(rdmaHandle, BufferKey<uintptr_t, u64>{0, 0}),
        InvalidParamsException);

    RdmaHandle rdmaHandle1 = (void*)0x12;
    RdmaHandleManager::GetInstance().tokenInfoMap[rdmaHandle1] = make_unique<TokenInfoManager>(0, rdmaHandle1);
    RdmaHandle rdmaHandle2 = (void*)0x1365;
    EXPECT_THROW(
        RdmaHandleManager::GetInstance().GetTokenIdInfo(rdmaHandle2, BufferKey<uintptr_t, u64>{0, 0}),
        InvalidParamsException);

    std::pair<TokenIdHandle, uint32_t> expectResult(0, 0);
    EXPECT_EQ(
        RdmaHandleManager::GetInstance().GetTokenIdInfo(rdmaHandle1, BufferKey<uintptr_t, u64>{0, 0}), expectResult);
}

// ===================== FindCachedJfcHandle / GetJfcHandle / GetJfcHandleAndCqInfo / cqInfoMap cleanup
// =====================

static void ClearJfcCqCache(RdmaHandleManager& mgr)
{
    mgr.jfcHandleMap.clear();
    mgr.cqInfoMap.clear();
}

TEST_F(RdmaHandleManagerTest, find_cached_jfc_handle_return_false_when_rdma_handle_not_exist)
{
    auto& mgr = RdmaHandleManager::GetInstance();
    ClearJfcCqCache(mgr);

    RdmaHandle handle = (void*)0x9999;
    JfcHandle outHandle = 0;
    Hccl::CqCreateInfo outCqInfo{};
    EXPECT_FALSE(mgr.FindCachedJfcHandle(handle, HrtUbJfcMode::NORMAL, outHandle, outCqInfo));

    ClearJfcCqCache(mgr);
}

TEST_F(RdmaHandleManagerTest, find_cached_jfc_handle_return_false_when_jfc_mode_not_exist)
{
    auto& mgr = RdmaHandleManager::GetInstance();
    ClearJfcCqCache(mgr);

    RdmaHandle handle = (void*)0x9999;
    mgr.jfcHandleMap[handle][HrtUbJfcMode::STARS_POLL] = 100;

    JfcHandle outHandle = 0;
    Hccl::CqCreateInfo outCqInfo{};
    EXPECT_FALSE(mgr.FindCachedJfcHandle(handle, HrtUbJfcMode::NORMAL, outHandle, outCqInfo));

    ClearJfcCqCache(mgr);
}

TEST_F(RdmaHandleManagerTest, find_cached_jfc_handle_return_true_and_fill_outputs_when_hit)
{
    auto& mgr = RdmaHandleManager::GetInstance();
    ClearJfcCqCache(mgr);

    RdmaHandle handle = (void*)0x9999;
    JfcHandle cachedJfc = 0x1234;
    Hccl::CqCreateInfo cachedCqInfo{1, 2, 3, 4, 5, 6};

    mgr.jfcHandleMap[handle][HrtUbJfcMode::NORMAL] = cachedJfc;
    mgr.cqInfoMap[cachedJfc] = cachedCqInfo;

    JfcHandle outHandle = 0;
    Hccl::CqCreateInfo outCqInfo{};
    EXPECT_TRUE(mgr.FindCachedJfcHandle(handle, HrtUbJfcMode::NORMAL, outHandle, outCqInfo));

    EXPECT_EQ(outHandle, cachedJfc);
    EXPECT_EQ(outCqInfo.va, cachedCqInfo.va);
    EXPECT_EQ(outCqInfo.id, cachedCqInfo.id);
    EXPECT_EQ(outCqInfo.cqeSize, cachedCqInfo.cqeSize);
    EXPECT_EQ(outCqInfo.cqDepth, cachedCqInfo.cqDepth);
    EXPECT_EQ(outCqInfo.swdbAddr, cachedCqInfo.swdbAddr);

    ClearJfcCqCache(mgr);
}

TEST_F(RdmaHandleManagerTest, get_jfc_handle_return_cached_and_fill_cq_info_on_hit)
{
    auto& mgr = RdmaHandleManager::GetInstance();
    ClearJfcCqCache(mgr);

    RdmaHandle handle = (void*)0x9999;
    JfcHandle cachedJfc = 0x4321;
    Hccl::CqCreateInfo cachedCqInfo{10, 20, 30, 40, 50, 60};

    mgr.jfcHandleMap[handle][HrtUbJfcMode::NORMAL] = cachedJfc;
    mgr.cqInfoMap[cachedJfc] = cachedCqInfo;

    Hccl::CqCreateInfo outCqInfo{};
    JfcHandle ret = mgr.GetJfcHandle(handle, outCqInfo, HrtUbJfcMode::NORMAL);

    EXPECT_EQ(ret, cachedJfc);
    EXPECT_EQ(outCqInfo.va, cachedCqInfo.va);
    EXPECT_EQ(outCqInfo.id, cachedCqInfo.id);
    EXPECT_EQ(outCqInfo.cqDepth, cachedCqInfo.cqDepth);

    ClearJfcCqCache(mgr);
}

TEST_F(RdmaHandleManagerTest, get_jfc_handle_create_and_cache_on_miss)
{
    auto& mgr = RdmaHandleManager::GetInstance();
    ClearJfcCqCache(mgr);

    RdmaHandle handle = (void*)0x9999;
    JfcHandle newJfc = 0x5555;
    MOCKER(HrtRaUbCreateJfc).stubs().will(returnValue(newJfc));

    Hccl::CqCreateInfo inCqInfo{100, 200, 300, 400, 500, 600};
    Hccl::CqCreateInfo outCqInfo = inCqInfo;

    JfcHandle ret = mgr.GetJfcHandle(handle, outCqInfo, HrtUbJfcMode::STARS_POLL);

    EXPECT_EQ(ret, newJfc);
    EXPECT_EQ(mgr.jfcHandleMap[handle][HrtUbJfcMode::STARS_POLL], newJfc);
    EXPECT_EQ(mgr.cqInfoMap[newJfc].va, inCqInfo.va);
    EXPECT_EQ(mgr.cqInfoMap[newJfc].id, inCqInfo.id);
    EXPECT_EQ(mgr.cqInfoMap[newJfc].cqeSize, inCqInfo.cqeSize);

    ClearJfcCqCache(mgr);
}
