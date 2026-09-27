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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "adapter_rts.h"
#include "coll_comm.h"
#include "hcom_common.h"
#include "hcomm_c_adpt.h"
#include "hcomm_result_defs.h"
#include "my_rank.h"
#include "op_base.h"
#include "coll_comm_mgr.h"

extern thread_local s32 g_hcclDeviceId;

namespace {
constexpr uint32_t DEFAULT_MODE = 0;
constexpr uint32_t TEST_OP_EXPANSION_MODE = 1;

std::atomic<uint32_t> g_commNameIndex(0);
std::atomic<uint32_t> g_initCallCount(0);
std::atomic<uint32_t> g_topoInfoCallCount(0);
std::atomic<bool> g_failFirstTopoInfo(false);
std::mutex g_modeMutex;
std::vector<uint32_t> g_observedConfigModes;

HcclResult StubGetCommNameV2(HcclComm comm, char* commName)
{
    (void)comm;
    int32_t ret = snprintf_s(
        commName, hccl::ROOTINFO_INDENTIFIER_MAX_LENGTH, hccl::ROOTINFO_INDENTIFIER_MAX_LENGTH - 1, "coll_comm_%u",
        g_commNameIndex.fetch_add(1));
    return ret < 0 ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}

HcclResult StubInitCollComm(
    hccl::hcclComm* comm, void* commV2, void* rankGraph, uint32_t userRank, HcclMem cclBuffer,
    const std::string& commName, const HcclCommConfig* config, hccl::CollCommInitMode initMode)
{
    (void)rankGraph;
    (void)cclBuffer;
    uint32_t callIndex = g_initCallCount.fetch_add(1);
    uint32_t configMode = config == nullptr ? DEFAULT_MODE : config->hcclOpExpansionMode;
    {
        std::lock_guard<std::mutex> lock(g_modeMutex);
        if (g_observedConfigModes.size() <= callIndex) {
            g_observedConfigModes.resize(callIndex + 1, DEFAULT_MODE);
        }
        g_observedConfigModes[callIndex] = configMode;
    }

    comm->collComm_ = std::make_unique<hccl::CollComm>(commV2, userRank, commName, hccl::ManagerCallbacks{}, initMode);
    comm->collComm_->myRank_ = std::make_shared<hccl::MyRank>(
        nullptr, userRank, comm->collComm_->GetCommConfig(), hccl::ManagerCallbacks{}, nullptr, nullptr);
    comm->collComm_->myRank_->opExpansionMode_ = configMode;
    return HCCL_SUCCESS;
}

HcclResult StubSetGroupTopoInfo(const char* group, uint32_t rankSize)
{
    (void)group;
    (void)rankSize;
    uint32_t callIndex = g_topoInfoCallCount.fetch_add(1);
    return g_failFirstTopoInfo.load() && callIndex == 0 ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}
} // namespace

class HcomInitCollCommTest : public testing::Test {
protected:
    virtual void SetUp() override
    {
        const char* fakeA5SocName = "Ascend950PR_958b";
        MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA5SocName));
    }

    virtual void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(HcomInitCollCommTest, ut_HcomInitCollComm_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    void* commV2 = nullptr;
    MOCKER(&HcclGetCommNameV2).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&HcclGetCclBuffer).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&HcclGetRankGraphV2).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&hccl::hcclComm::InitCollComm).stubs().will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcomInitCollComm(0, &commV2, hcomInfo.pComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(hcomInfo.pComm, nullptr);
}

TEST_F(HcomInitCollCommTest, ut_HcomInitCollComm_When_commV2IsNullptr_Expect_ReturnIsHCCL_E_PARA)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    HcclResult ret = HcomInitCollComm(0, nullptr, hcomInfo.pComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 看护空输出指针在解引用前被拒绝。
TEST_F(HcomInitCollCommTest, Ut_HcclCommInitCollComm_When_CommIsNullptr_Expect_ReturnHcclEPtr)
{
    void* commV2 = reinterpret_cast<void*>(0x1);
    EXPECT_EQ(HcclCommInitCollComm(0, &commV2, nullptr, nullptr), HCCL_E_PTR);
}

class HcclCommInitCollCommGuardTest : public testing::Test {
protected:
    void SetUp() override
    {
        g_hcclDeviceId = 0;
        g_commNameIndex.store(0);
        g_initCallCount.store(0);
        g_topoInfoCallCount.store(0);
        g_failFirstTopoInfo.store(false);
        {
            std::lock_guard<std::mutex> lock(g_modeMutex);
            g_observedConfigModes.clear();
        }

        const char* fakeA5SocName = "Ascend950PR_958b";
        MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA5SocName));
        MOCKER(hccl::GetMaxDevNum).stubs().with(outBound(8U)).will(returnValue(HCCL_SUCCESS));
        MOCKER(HcommResMgrInit).stubs().will(returnValue(static_cast<HcommResult>(HCOMM_SUCCESS)));
        MOCKER(&HcclGetCommNameV2).stubs().will(invoke(StubGetCommNameV2));
        MOCKER(&HcclGetCclBuffer).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(&HcclGetRankGraphV2).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(&hccl::hcclComm::InitCollComm).stubs().will(invoke(StubInitCollComm));
        MOCKER(HcomSetGroupTopoInfo).stubs().will(invoke(StubSetGroupTopoInfo));

        ResetOpBaseState();
    }

    void TearDown() override
    {
        ResetOpBaseState();
        GlobalMockObject::verify();
    }

    static HcclResult InitComm(const HcclCommConfig& config, HcclComm& comm)
    {
        void* commV2 = reinterpret_cast<void*>(0x1);
        return HcclCommInitCollComm(0, &commV2, &config, &comm);
    }

    static std::vector<uint32_t> GetObservedConfigModes()
    {
        std::lock_guard<std::mutex> lock(g_modeMutex);
        return g_observedConfigModes;
    }

    static void ResetOpBaseState()
    {
        HcclOpInfoCtx& opBaseHcom = hccl::CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
        std::lock_guard<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        opBaseHcom.opGroup2CommMap.clear();
    }
};

// 看护 CollComm 初始化只读传递配置，不得修改调用者输入。
TEST_F(HcclCommInitCollCommGuardTest, Ut_HcclCommInitCollComm_When_UsesConstConfig_Expect_InputConfigUnchanged)
{
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclOpExpansionMode = TEST_OP_EXPANSION_MODE;
    const HcclCommConfig constConfig = config;
    HcclComm comm = nullptr;

    ASSERT_EQ(InitComm(constConfig, comm), HCCL_SUCCESS);

    EXPECT_EQ(constConfig.hcclOpExpansionMode, TEST_OP_EXPANSION_MODE);
    const auto observedModes = GetObservedConfigModes();
    ASSERT_EQ(observedModes.size(), 1U);
    EXPECT_EQ(observedModes[0], TEST_OP_EXPANSION_MODE);
}

// 看护拓扑信息设置失败时清理输出句柄和通信域映射。
TEST_F(HcclCommInitCollCommGuardTest, Ut_HcclCommInitCollComm_When_SetGroupTopoInfoFails_Expect_CompleteRollback)
{
    g_failFirstTopoInfo.store(true);
    HcclCommConfig failedConfig;
    HcclCommConfigInit(&failedConfig);
    failedConfig.hcclOpExpansionMode = TEST_OP_EXPANSION_MODE;
    HcclComm failedComm = nullptr;

    EXPECT_EQ(InitComm(failedConfig, failedComm), HCCL_E_INTERNAL);
    EXPECT_EQ(failedComm, nullptr);
    HcclOpInfoCtx& opBaseHcom = hccl::CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    {
        std::lock_guard<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        EXPECT_TRUE(opBaseHcom.opGroup2CommMap.empty());
    }

    HcclCommConfig retryConfig;
    HcclCommConfigInit(&retryConfig);
    retryConfig.hcclOpExpansionMode = TEST_OP_EXPANSION_MODE;
    HcclComm retryComm = nullptr;
    ASSERT_EQ(InitComm(retryConfig, retryComm), HCCL_SUCCESS);

    const auto observedModes = GetObservedConfigModes();
    ASSERT_EQ(observedModes.size(), 2U);
    EXPECT_EQ(observedModes[0], TEST_OP_EXPANSION_MODE);
    EXPECT_EQ(observedModes[1], TEST_OP_EXPANSION_MODE);
}
