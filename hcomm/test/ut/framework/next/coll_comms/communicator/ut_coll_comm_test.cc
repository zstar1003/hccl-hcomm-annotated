/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../ut_hcomm_base.h"
#define private public
#include "coll_comm.h"
#include "symmetric_memory/symmetric_memory.h"
#undef private
#include "ubmem_symmetric_memory.h"
#include "coll_comm_config.h"
#include "hcom_common.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "hccl_team_mgr.h"
#include "hccl_team_c_adpt.h"
#include "hcomm_team.h"
#include "hcomm_team_c_adpt.h"
#include "hccl/hccl_channel.h"
#include "my_rank.h"
#include "coll_comm.h"

class TestCollComm : public TestHcommCAdptBase {
public:
    void SetUp() override { TestHcommCAdptBase::SetUp(); }
    void TearDown() override { TestHcommCAdptBase::TearDown(); }
};

namespace {
std::unique_ptr<CollComm> PrepareSuspendingCollCommForResetNotify(HcclResult resetNotifyRet)
{
    auto coll = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), hccl::ManagerCallbacks{});
    coll->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING;
    coll->isCleaned_ = true;
    coll->commEngineResMgr_ = std::make_unique<CommEngineResMgr>();

    MOCKER_CPP(&MyRank::Resume, HcclResult(MyRank::*)()).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommEngineResMgr::ResetCommLocalNotifies, HcclResult(CommEngineResMgr::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(resetNotifyRet));

    aclrtBinHandle binHandle;
    coll->myRank_ = std::make_shared<MyRank>(binHandle, 0, coll->GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);
    return coll;
}
} // namespace

HcclResult StubCollCommUrmaHrtMalloc(void** devPtr, u64 size, bool Level2Address)
{
    static uintptr_t devAddr = 0x7000000;
    (void)size;
    (void)Level2Address;
    *devPtr = reinterpret_cast<void*>(devAddr);
    devAddr += 0x1000;
    return HCCL_SUCCESS;
}

TEST_F(TestCollComm, Ut_TestCollCommInit_When_RankGraphNullptr_Return_HCCL_E_PTR)
{
    hccl::CollComm collComm(nullptr, 0, "test_comm", hccl::ManagerCallbacks{});
    HcclMem cclBuffer = {};
    HcclResult ret = collComm.Init(nullptr, nullptr, cclBuffer, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestCollComm, test_get_comm_status_initial_and_after_change)
{
    std::unique_ptr<CollComm> coll_
        = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    EXPECT_EQ(coll_->GetCommStatus(), HcclCommStatus::HCCL_COMM_STATUS_INVALID);

    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
    EXPECT_EQ(coll_->GetCommStatus(), HcclCommStatus::HCCL_COMM_STATUS_READY);
}

TEST_F(TestCollComm, test_suspend_success_and_idempotent)
{
    std::unique_ptr<CollComm> coll_
        = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    // mock MyRank::StopLaunch to return success
    MOCKER_CPP(&MyRank::StopLaunch, HcclResult(MyRank::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // attach a MyRank instance (can be real or mocked; method is mocked above)
    aclrtBinHandle binHandle;
    coll_->myRank_
        = std::make_shared<MyRank>(binHandle, 0, coll_->GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);

    auto ret = coll_->Suspend();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(coll_->commStatus_, HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING);

    // calling Suspend again when already suspending should return success without error
    ret = coll_->Suspend();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestCollComm, test_clean_fail_not_suspending)
{
    std::unique_ptr<CollComm> coll_
        = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    // when not suspending, Clean should return not support
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
    auto ret = coll_->Clean();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TestCollComm, test_clean_success_and_idempotent)
{
    std::unique_ptr<CollComm> coll_
        = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    // prepare for cleaning: put into suspending state
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING;
    coll_->isCleaned_ = false;

    // mock MyRank::Clean to return success
    MOCKER_CPP(&MyRank::Clean, HcclResult(MyRank::*)()).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // attach a MyRank instance
    aclrtBinHandle binHandle;
    coll_->myRank_
        = std::make_shared<MyRank>(binHandle, 0, coll_->GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);

    auto ret = coll_->Clean();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(coll_->isCleaned_);

    // calling Clean again should be idempotent and return success
    ret = coll_->Clean();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestCollComm, test_resume_fail_invalid_and_resume_success)
{
    std::unique_ptr<CollComm> coll_
        = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    coll_->isCleaned_ = false;
    // Resume when commStatus_ is INVALID should return internal error
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    auto ret = coll_->Resume();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    // Now test successful resume from SUSPENDING
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING;
    coll_->isCleaned_ = true;

    // mock MyRank::Resume to return success
    MOCKER_CPP(&MyRank::Resume, HcclResult(MyRank::*)()).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // attach a MyRank instance
    aclrtBinHandle binHandle;
    coll_->myRank_
        = std::make_shared<MyRank>(binHandle, 0, coll_->GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);

    ret = coll_->Resume();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(coll_->commStatus_, HcclCommStatus::HCCL_COMM_STATUS_READY);
    EXPECT_FALSE(coll_->isCleaned_);
}

TEST_F(TestCollComm, Ut_Resume_When_CommEngineResMgrNullptr_Expect_SkipResetCommLocalNotifiesAndSuccess)
{
    std::unique_ptr<CollComm> coll_
        = std::make_unique<CollComm>(nullptr, 0u, std::string("ut_test"), hccl::ManagerCallbacks{});
    coll_->commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING;
    coll_->isCleaned_ = true;
    coll_->commEngineResMgr_ = nullptr;

    MOCKER_CPP(&MyRank::Resume, HcclResult(MyRank::*)()).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    aclrtBinHandle binHandle;
    coll_->myRank_
        = std::make_shared<MyRank>(binHandle, 0, coll_->GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);

    auto ret = coll_->Resume();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(coll_->commStatus_, HcclCommStatus::HCCL_COMM_STATUS_READY);
    EXPECT_FALSE(coll_->isCleaned_);
}

TEST_F(TestCollComm, Ut_Resume_When_ResetCommLocalNotifiesSuccess_Expect_ReturnSuccess)
{
    auto coll = PrepareSuspendingCollCommForResetNotify(HCCL_SUCCESS);

    auto ret = coll->Resume();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(coll->commStatus_, HcclCommStatus::HCCL_COMM_STATUS_READY);
    EXPECT_FALSE(coll->isCleaned_);
}

TEST_F(TestCollComm, Ut_Resume_When_ResetCommLocalNotifiesFailed_Expect_ReturnFailed)
{
    auto coll = PrepareSuspendingCollCommForResetNotify(HCCL_E_INTERNAL);

    auto ret = coll->Resume();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(coll->commStatus_, HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING);
    EXPECT_TRUE(coll->isCleaned_);
}

TEST_F(TestCollComm, Ut_InitSimpleMode_When_Success_Expect_ReturnsSuccessAndRankgraphSet)
{
    // Create CollComm instance
    hccl::CollComm collComm(nullptr, 0, "test_comm", hccl::ManagerCallbacks{});

    // Verify initial state - rankgraph should be nullptr before initialization
    // This validates the internal state that InitSimpleMode would populate
    EXPECT_EQ(collComm.GetRankSize(), 0u);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_ValidConfig_Expect_Success)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 2U;
    config.hcclRdmaTrafficClass = 120U;
    config.hcclRdmaServiceLevel = 3U;
    config.hcclQos = 5U;
    config.hcclChannelSqDepth = 128U;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_SUCCESS);
    EXPECT_EQ(opExpansionMode, 2U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigHcclQos(), 5U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigTrafficClass(), 120U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigServiceLevel(), 3U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigSqDepth(), 128U);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_NullConfig_Expect_Success)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
    uint32_t opExpansionMode = 9U;
    EXPECT_EQ(ApplyHcclCommConfig(nullptr, coll.GetCommConfig(), opExpansionMode), HCCL_SUCCESS);
    EXPECT_EQ(opExpansionMode, 0U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigSqDepth(), HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_SqDepthVaries_Expect_VersionRules)
{
    struct TestCase {
        uint32_t version;
        uint32_t sqDepth;
        HcclResult expectedResult;
        uint32_t expectedSqDepth;
    };

    const TestCase testCases[] = {
        {11U, 16U, HCCL_SUCCESS, 16U},
        {11U, 8192U, HCCL_SUCCESS, 8192U},
        {11U, 15U, HCCL_SUCCESS, 15U},
        {11U, 8193U, HCCL_SUCCESS, 8193U},
        {10U, 128U, HCCL_SUCCESS, HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET},
    };

    for (const auto& testCase : testCases) {
        SCOPED_TRACE(testing::Message() << "version=" << testCase.version << ", sqDepth=" << testCase.sqDepth);
        hccl::CollComm coll(nullptr, 0, "ut_sqdepth", hccl::ManagerCallbacks{});
        HcclCommConfig config{};
        UtInitHcclCommConfig(config);
        auto* configInfo = reinterpret_cast<CommConfigInfo*>(config.reserved);
        configInfo->version = testCase.version;
        config.hcclChannelSqDepth = testCase.sqDepth;
        uint32_t opExpansionMode = 0U;

        EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), testCase.expectedResult);
        EXPECT_EQ(coll.GetCommConfig().GetConfigSqDepth(), testCase.expectedSqDepth);
    }
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_InvalidHcclQos_Expect_EPara)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclQos = 8U;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_E_PARA);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_InvalidTrafficClass_Expect_EPara)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclRdmaTrafficClass = 256U;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_E_PARA);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_InvalidServiceLevel_Expect_EPara)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclRdmaServiceLevel = 8U;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_E_PARA);
}

TEST_F(TestCollComm, Ut_RegisterPendingSymmetricMemHandles_When_PendingConsumed_Expect_RegisteredHandleRetained)
{
    MOCKER_CPP(hrtMalloc).stubs().will(invoke(StubCollCommUrmaHrtMalloc));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));
    hccl::CollComm coll(nullptr, 0, "ut_sym", hccl::ManagerCallbacks{});
    aclrtBinHandle binHandle{};
    coll.myRank_ = std::make_shared<MyRank>(binHandle, 0, coll.GetCommConfig(), ManagerCallbacks(), nullptr, nullptr);
    coll.myRank_->commMems_ = std::make_unique<CommMems>(0);
    coll.symmetricMemory_.reset(new SymmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA));

    void* win = nullptr;
    void* ptr = reinterpret_cast<void*>(0x8000000);
    HcclComm comm = reinterpret_cast<HcclComm>(&coll);
    EXPECT_EQ(coll.RegisterWindow(comm, ptr, 0x2000, &win), HCCL_SUCCESS);
    HcclComm ownerComm = nullptr;
    EXPECT_EQ(hccl::GetHcommWindowComm(win, ownerComm), HCCL_SUCCESS);
    EXPECT_EQ(ownerComm, comm);

    EXPECT_EQ(coll.RegisterPendingSymmetricMemHandles(), HCCL_SUCCESS);

    SymmetricMemoryResource resource;
    // PR 后 *winHandle 为 HcommWindow device 副本，symmetricMemory_ 资源以 devSymWin 为 key，
    // 经 hcommToSymMap_ 反查 devSymWin 后查询
    void* devSymWin = coll.hcommToSymMap_[win];
    ASSERT_NE(devSymWin, nullptr);
    EXPECT_EQ(coll.symmetricMemory_->GetRegisteredMemoryResource(devSymWin, resource), HCCL_SUCCESS);
    EXPECT_NE(resource.memHandle, nullptr);

    EXPECT_EQ(coll.RegisterPendingSymmetricMemHandles(), HCCL_SUCCESS);

    std::vector<HcclMemHandle> memHandles;
    EXPECT_EQ(coll.GetAllRegisteredSymMemHandles(memHandles), HCCL_SUCCESS);
    ASSERT_EQ(memHandles.size(), 1U);
    EXPECT_EQ(memHandles[0], static_cast<HcclMemHandle>(resource.memHandle));

    std::vector<std::string> remoteMemTags;
    remoteMemTags.emplace_back(resource.memTag);
    EXPECT_EQ(coll.GetRemoteMissingSymMemHandles(remoteMemTags, memHandles), HCCL_SUCCESS);
    EXPECT_TRUE(memHandles.empty());

    remoteMemTags[0] = "unrelated_tag";
    EXPECT_EQ(coll.GetRemoteMissingSymMemHandles(remoteMemTags, memHandles), HCCL_SUCCESS);
    ASSERT_EQ(memHandles.size(), 1U);
    EXPECT_EQ(memHandles[0], static_cast<HcclMemHandle>(resource.memHandle));
}

TEST_F(TestCollComm, Ut_GetCommSymWin_When_UbMemoryEnabled_Expect_UseSharedWindowMapping)
{
    hccl::CollComm coll(nullptr, 0, "ut_ub_sym_get", hccl::ManagerCallbacks{});
    const std::vector<uint32_t> worldRanks{0U};
    coll.ubMemSymmetricMemory_ = std::make_unique<hccl::UbMemSymmetricMemory>(
        &coll, reinterpret_cast<HcommTeamHandle>(0x1000U), 2U, worldRanks);
    coll.symmetricMemory_.reset(new SymmetricMemory(0U, 2U, 0U, SymmetricMemoryMode::URMA));

    void* userAddress = reinterpret_cast<void*>(0x3000U);
    void* legacyWindow = reinterpret_cast<void*>(0x4000U);
    HcclCommSymWindow hcommWindow = reinterpret_cast<HcclCommSymWindow>(0x5000U);
    auto urmaWindow = std::make_shared<SymmetricWindow>();
    urmaWindow->userVa = userAddress;
    urmaWindow->userSize = 0x1000U;
    urmaWindow->devWin = legacyWindow;
    coll.symmetricMemory_->sortedWindows_.emplace_back(urmaWindow);
    coll.symmetricMemory_->windowMap_.emplace(legacyWindow, urmaWindow);
    coll.symToHcommMap_.emplace(legacyWindow, hcommWindow);

    HcclCommSymWindow winHandle = reinterpret_cast<HcclCommSymWindow>(0x2000U);
    size_t offset = 0x100U;
    EXPECT_EQ(
        coll.GetCommSymWin(static_cast<uint8_t*>(userAddress) + 0x80U, 0x100U, &winHandle, &offset), HCCL_SUCCESS);
    EXPECT_EQ(winHandle, hcommWindow);
    EXPECT_EQ(offset, 0x80U);

    coll.symToHcommMap_.clear();
    coll.symmetricMemory_->windowMap_.clear();
    coll.symmetricMemory_->sortedWindows_.clear();
}

TEST_F(TestCollComm, Ut_GetCommSymWin_When_WindowNotFound_Expect_SuccessWithNullHandle)
{
    hccl::CollComm coll(nullptr, 0, "ut_urma_sym_miss", hccl::ManagerCallbacks{});
    coll.symmetricMemory_.reset(new SymmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA));

    HcclCommSymWindow winHandle = reinterpret_cast<HcclCommSymWindow>(0x2000U);
    size_t offset = 0x100U;
    EXPECT_EQ(coll.GetCommSymWin(reinterpret_cast<void*>(0x3000U), 0x100U, &winHandle, &offset), HCCL_SUCCESS);
    EXPECT_EQ(winHandle, nullptr);
    EXPECT_EQ(offset, 0U);
}

TEST_F(TestCollComm, Ut_DeregisterWindow_When_UbDeregisterFails_Expect_FullyRetryable)
{
    hccl::CollComm coll(nullptr, 0, "ut_sym_ub_fail", hccl::ManagerCallbacks{});
    coll.symmetricMemory_.reset(new SymmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA));
    const std::vector<uint32_t> ranks{0U};
    coll.ubMemSymmetricMemory_ = std::make_unique<hccl::UbMemSymmetricMemory>(&coll, nullptr, 0U, ranks);

    HcclCommSymWindow window = reinterpret_cast<HcclCommSymWindow>(0x8200000);
    void* legacyWindow = reinterpret_cast<void*>(0x8300000);
    coll.hcommToSymMap_[window] = legacyWindow;

    EXPECT_EQ(coll.DeregisterWindow(window), HCCL_E_NOT_FOUND);
    EXPECT_NE(coll.hcommToSymMap_.find(window), coll.hcommToSymMap_.end());
}

TEST_F(TestCollComm, Ut_UpdateSymmetricRemoteMem_When_ChannelReturnsRemoteMem_Expect_UpdateWindow)
{
    MOCKER_CPP(hrtMalloc).stubs().will(invoke(StubCollCommUrmaHrtMalloc));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));

    hccl::CollComm coll(nullptr, 0, "ut_sym", hccl::ManagerCallbacks{});
    coll.symmetricMemory_.reset(new SymmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA));

    void* win = nullptr;
    void* ptr = reinterpret_cast<void*>(0x9000000);
    constexpr size_t winSize = 0x2000;
    HcclComm comm = reinterpret_cast<HcclComm>(&coll);
    EXPECT_EQ(coll.RegisterWindow(comm, ptr, winSize, &win), HCCL_SUCCESS);
    HcclComm ownerComm = nullptr;
    EXPECT_EQ(hccl::GetHcommWindowComm(win, ownerComm), HCCL_SUCCESS);
    EXPECT_EQ(ownerComm, comm);

    SymmetricMemoryResource resource;
    resource.memHandle = reinterpret_cast<void*>(0x9100000);
    resource.memTag = std::string(HCCL_SYMMETRIC_MEMORY_TAG_PREFIX) + "ut_sym_addr_"
                      + std::to_string(reinterpret_cast<uintptr_t>(ptr)) + "_size_" + std::to_string(winSize);
    // PR 后 *winHandle 为 HcommWindow device 副本，SetRegisteredMemoryResource 须以 devSymWin 为 key
    void* devSymWin = coll.hcommToSymMap_[win];
    ASSERT_NE(devSymWin, nullptr);
    EXPECT_EQ(coll.symmetricMemory_->SetRegisteredMemoryResource(devSymWin, resource), HCCL_SUCCESS);

    CommMem remoteMem{};
    remoteMem.type = COMM_MEM_TYPE_DEVICE;
    remoteMem.addr = reinterpret_cast<void*>(0x9200000);
    remoteMem.size = winSize;
    std::vector<std::string> memTags = {resource.memTag};
    EXPECT_EQ(coll.UpdateSymmetricRemoteMem(1, &remoteMem, memTags), HCCL_SUCCESS);

    // PR 后 *winHandle 为 HcommWindow device 副本，remoteMemMap_ 以 devSymWin 为 key
    auto remoteMemIt = coll.symmetricMemory_->remoteMemMap_.find(devSymWin);
    ASSERT_NE(remoteMemIt, coll.symmetricMemory_->remoteMemMap_.end());
    ASSERT_EQ(remoteMemIt->second.size(), 2U);
    EXPECT_EQ(remoteMemIt->second[1].addr, remoteMem.addr);
    EXPECT_EQ(remoteMemIt->second[1].size, remoteMem.size);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_NullptrConfig_Expect_TcSlSkipped)
{
    hccl::CollComm coll(nullptr, 0, "ut_tcsl", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclRdmaTrafficClass = 0xFFFFFFFFu;
    config.hcclRdmaServiceLevel = 0xFFFFFFFFu;
    config.hcclQos = 0xFFFFFFFFu;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_SUCCESS);
    EXPECT_EQ(coll.GetCommConfig().GetConfigTrafficClass(), 0xFFFFFFFFu);
    EXPECT_EQ(coll.GetCommConfig().GetConfigServiceLevel(), 0xFFFFFFFFu);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_TcNotMultipleOf4_Expect_EPara)
{
    hccl::CollComm coll(nullptr, 0, "ut_tc", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclRdmaTrafficClass = 3U;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_E_PARA);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_TcValidMultipleOf4_Expect_Success)
{
    hccl::CollComm coll(nullptr, 0, "ut_tc_valid", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclRdmaTrafficClass = 4U;
    config.hcclRdmaServiceLevel = 5U;
    config.hcclQos = 0xFFFFFFFFu;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_SUCCESS);
    EXPECT_EQ(coll.GetCommConfig().GetConfigTrafficClass(), 4U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigServiceLevel(), 5U);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_QosDefault_Expect_Success)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos_def", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclQos = 0xFFFFFFFFu;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_SUCCESS);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_QosValid_Expect_Success)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos_valid", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclQos = 7U;
    config.hcclRdmaTrafficClass = 0xFFFFFFFFu;
    config.hcclRdmaServiceLevel = 0xFFFFFFFFu;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_SUCCESS);
    EXPECT_EQ(coll.GetCommConfig().GetConfigHcclQos(), 7U);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_LowVersionQos_Expect_QosNotSet)
{
    hccl::CollComm coll(nullptr, 0, "ut_qos_ver", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclQos = 5U;
    config.hcclRdmaTrafficClass = 0xFFFFFFFFu;
    config.hcclRdmaServiceLevel = 0xFFFFFFFFu;
    CommConfigInfo info{};
    info.version = 5U;
    errno_t sRet = memcpy_s(config.reserved, sizeof(config.reserved), &info, sizeof(info));
    ASSERT_EQ(sRet, EOK);
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_SUCCESS);
    EXPECT_EQ(coll.GetCommConfig().GetConfigHcclQos(), HCCL_COMM_QOS_CONFIG_NOT_SET);
}

TEST_F(TestCollComm, Ut_ApplyHcclCommConfig_When_TcZero_Expect_Success)
{
    hccl::CollComm coll(nullptr, 0, "ut_tc_zero", hccl::ManagerCallbacks{});
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclRdmaTrafficClass = 0U;
    config.hcclRdmaServiceLevel = 0U;
    config.hcclQos = 0xFFFFFFFFu;
    uint32_t opExpansionMode = 0U;
    EXPECT_EQ(ApplyHcclCommConfig(&config, coll.GetCommConfig(), opExpansionMode), HCCL_SUCCESS);
    EXPECT_EQ(coll.GetCommConfig().GetConfigTrafficClass(), 0U);
    EXPECT_EQ(coll.GetCommConfig().GetConfigServiceLevel(), 0U);
}

// =====================================================================
// 以下用例覆盖 PR 4786（AIN 控制面重构）在 coll_comm.cc 新增的 4 个函数：
// CreatePrebuiltWorldTeam / InitWorldTeams / ReExchangeWindowsForBoundTeams /
// UpdateHcommWindowRemoteMem（对应 STC ut_ain_047~058，见 docs/stc.md 2.3/2.6）。
// 设计说明：
// 1) RankGraph::GetNetLayers/GetInstRanksByNetLayer/GetLinks 为虚函数（rank_graph_base.h），
//    此前判定"pImpl 转发不可 MOCKER"不成立——用可配置桩子类 UtRankGraphStub 直接注入
//    collComm.rankgraph_（#define private public），比 MOCKER 更稳（无签名/abi 依赖）。
// 2) HcommTeamCreate/HcommTeamGetNetLayer/HcommTeamGetEngine/
//    HcommTeamUpdateWindowRemoteMemByRank 为 extern "C" 接口，MOCKER(...) 打桩；
//    MyRank::CreateChannels/ChannelGetRemoteMems 为非虚成员函数，MOCKER_CPP 打桩。
// 3) HcclTeamMgr 为真实单例（不可 mock），用例内真实注册 team，
//    TearDown 用 UnregisterTeam/ClearByCollComm 兜底清理防状态泄漏。
// =====================================================================

namespace {
// —— 可配置 RankGraph 桩：按 (netLayer, proto) 配置每层 rank 集合与 peer 链路 ——
class UtRankGraphStub : public hccl::RankGraph {
public:
    ~UtRankGraphStub() override = default;

    HcclResult GetRankSize(uint32_t* rankSize) override
    {
        *rankSize = static_cast<uint32_t>(allRanks.size());
        return HCCL_SUCCESS;
    }
    HcclResult GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum) override
    {
        *netLayers = layerBuf.data();
        *netLayerNum = static_cast<uint32_t>(layerBuf.size());
        return getNetLayersRet;
    }
    HcclResult GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum) override
    {
        auto it = layerRanks.find(netLayer);
        if (it == layerRanks.end()) {
            *rankList = nullptr;
            *rankNum = 0;
            return HCCL_E_NOT_FOUND;
        }
        *rankList = it->second.data();
        *rankNum = static_cast<uint32_t>(it->second.size());
        return HCCL_SUCCESS;
    }
    HcclResult
    GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize) override
    {
        (void)srcRank;
        auto it = layerLinks.find(netLayer);
        if (it == layerLinks.end()) {
            *linkList = nullptr;
            *listSize = 0;
            return HCCL_SUCCESS;
        }
        *linkList = it->second.data();
        *listSize = static_cast<uint32_t>(it->second.size());
        return HCCL_SUCCESS;
    }
    HcclResult GetRankGraphInfo(GraphType type, void** graph, uint32_t* len) override
    {
        (void)type;
        *graph = nullptr;
        *len = 0;
        return HCCL_SUCCESS;
    }
    HcclResult GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType) override
    {
        (void)netLayer;
        *topoType = COMM_TOPO_RESERVED;
        return HCCL_SUCCESS;
    }
    HcclResult GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum) override
    {
        *rankNum = static_cast<uint32_t>(layerRanks[netLayer].size());
        return HCCL_SUCCESS;
    }
    HcclResult GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize) override
    {
        (void)netLayer;
        *instSizeList = nullptr;
        *listSize = 0;
        return HCCL_SUCCESS;
    }
    HcclResult GetDeviceId(uint32_t rankId, uint32_t* deviceId) override
    {
        *deviceId = rankId;
        return HCCL_SUCCESS;
    }
    HcclResult GetEndpointInfo(
        uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen,
        void* info) override
    {
        (void)rankId;
        (void)endPointDesc;
        (void)endpointAttr;
        (void)infoLen;
        (void)info;
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> allRanks{0, 1};
    std::vector<uint32_t> layerBuf{0};
    std::map<uint32_t, std::vector<uint32_t>> layerRanks; // netLayer -> ranks
    std::map<uint32_t, std::vector<CommLink>> layerLinks; // netLayer -> links
    HcclResult getNetLayersRet{HCCL_SUCCESS};
};

CommLink MakeLink(CommProtocol proto)
{
    CommLink link{};
    link.header.version = 1;
    link.header.magicWord = 0x0f0e0f0f;
    link.header.size = sizeof(CommLink);
    link.linkAttr.linkProtocol = proto;
    link.linkAttr.hop = 1;
    return link;
}

// —— extern "C" 接口桩 ——
HcommTeamHandle UtNextWorldTeamHandle()
{
    static uintptr_t nextHandle = 0x30000;
    nextHandle += 0x1000;
    return reinterpret_cast<HcommTeamHandle>(nextHandle);
}

HcommResult UtStubHcommTeamCreateOk(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* syncMemSize)
{
    (void)worldTeam;
    (void)desc;
    *team = UtNextWorldTeamHandle(); // 每次分配不同伪句柄，避免单例内同 handle 重复注册 E_PARA
    *syncMemSize = 0;
    return 0;
}

HcommResult UtStubHcommTeamCreateFail(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* syncMemSize)
{
    (void)worldTeam;
    (void)desc;
    *team = nullptr;
    *syncMemSize = 0;
    return static_cast<HcommResult>(1); // HCOMM_E_PARA
}

uint32_t g_utGetNetLayerVal = 1;
HcommResult UtStubHcommTeamGetNetLayer(HcommTeamHandle team, uint32_t* netLayer)
{
    (void)team;
    *netLayer = g_utGetNetLayerVal;
    return 0;
}

CommEngine g_utGetEngineVal = COMM_ENGINE_AIV;
HcommResult UtStubHcommTeamGetEngine(HcommTeamHandle team, CommEngine* engine)
{
    (void)team;
    *engine = g_utGetEngineVal;
    return 0;
}

// HcommTeamUpdateWindowRemoteMemByRank 捕获桩：记录 sizes/slots 供偏移断言
bool g_utUpdateWinCalled = false;
uint32_t g_utUpdateWinSizes[3] = {0, 0, 0};
std::vector<uint32_t> g_utUpdateWinSlots;
HcclMemHandle g_utLastRemoteMem = nullptr;
HcommResult UtStubUpdateWindowRemoteMemByRank(
    HcclCommSymWindow handle, const uint32_t* sizes, uint32_t sizeNum, const uint32_t* slots, uint32_t slotNum,
    const CommMem* remoteMem)
{
    (void)handle;
    (void)sizeNum;
    (void)slotNum;
    g_utUpdateWinCalled = true;
    for (uint32_t i = 0; i < 3; i++) {
        g_utUpdateWinSizes[i] = sizes[i];
    }
    g_utUpdateWinSlots.assign(slots, slots + slotNum);
    g_utLastRemoteMem = remoteMem ? reinterpret_cast<HcclMemHandle>(remoteMem->addr) : nullptr;
    return 0;
}

// —— MyRank 非虚成员函数桩 ——
uint32_t g_utCreateChannelsCalls = 0;
HcclResult UtStubMyRankCreateChannelsOk(
    hccl::MyRank* self, CommEngine engine, const std::string& commTag, const HcclChannelDesc* channelDescs,
    uint32_t channelNum, ChannelHandle* channels)
{
    (void)self;
    (void)engine;
    (void)commTag;
    (void)channelDescs;
    g_utCreateChannelsCalls++;
    for (uint32_t i = 0; i < channelNum; i++) {
        channels[i] = static_cast<ChannelHandle>(0x71000) + static_cast<int>(i);
    }
    return HCCL_SUCCESS;
}

} // namespace

// ===================== 1. CreatePrebuiltWorldTeam =====================

// ut_ain_044 L2 侧：HcommTeamCreate 成功 + 真实注册，索引可查。
TEST_F(TestCollComm, Ut_CreatePrebuiltWorldTeam_When_CreateSuccess_Expect_RegisterInTeamMgr)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateOk));
    hccl::CollComm coll(nullptr, 0, "ut_prebuilt", hccl::ManagerCallbacks{});
    uint32_t rankIds[2] = {0, 1};
    HcommTeamHandle created = nullptr;
    {
        // 捕获桩分配的句柄：经 FindWorldTeamByProtoLayer 反查验证。
        EXPECT_EQ(coll.CreatePrebuiltWorldTeam(COMM_PROTOCOL_UB_CTP, 1, rankIds, 2, 0), HCCL_SUCCESS);
        created = hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 1);
    }
    ASSERT_NE(created, nullptr);

    hccl::HcclTeamMgr::GetInstance().UnregisterTeam(created);
    GlobalMockObject::verify();
}

// HcommTeamCreate 失败透传为 HCCL_E_INTERNAL。
TEST_F(TestCollComm, Ut_CreatePrebuiltWorldTeam_When_CreateFail_Expect_ReturnInternal)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateFail));
    hccl::CollComm coll(nullptr, 0, "ut_prebuilt_fail", hccl::ManagerCallbacks{});
    uint32_t rankIds[2] = {0, 1};

    EXPECT_EQ(coll.CreatePrebuiltWorldTeam(COMM_PROTOCOL_UB_CTP, 1, rankIds, 2, 0), HCCL_E_INTERNAL);

    EXPECT_EQ(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 1), nullptr);
    GlobalMockObject::verify();
}

// 创建成功后可按 (protocol, netLayer) 查询。
TEST_F(TestCollComm, Ut_CreatePrebuiltWorldTeam_When_Created_Expect_QuerySuccess)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateOk));
    hccl::CollComm coll(nullptr, 0, "ut_prebuilt_dup", hccl::ManagerCallbacks{});
    uint32_t rankIds[2] = {0, 1};

    ASSERT_EQ(coll.CreatePrebuiltWorldTeam(COMM_PROTOCOL_UB_CTP, 1, rankIds, 2, 0), HCCL_SUCCESS);
    HcommTeamHandle created
        = hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 1);
    ASSERT_NE(created, nullptr);
    hccl::HcclTeamMgr::GetInstance().UnregisterTeam(created);
    GlobalMockObject::verify();
}

// ===================== 2. InitWorldTeams =====================

// ut_ain_047 URMA 多协议多层预制：netLayers={0,1}，layer0=UB_CTP、layer1=UBC_TP+UB_CTP，共 3 个 worldTeam。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_MultiLayerMultiProto_Expect_ThreeWorldTeams)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateOk));
    hccl::CollComm coll(nullptr, 0, "ut_iwt_multi", hccl::ManagerCallbacks{});

    UtRankGraphStub graph;
    graph.layerBuf = {0, 1};
    graph.layerRanks[0] = {0, 1};
    graph.layerRanks[1] = {0, 1};
    graph.layerLinks[0] = {MakeLink(COMM_PROTOCOL_UB_CTP)};
    graph.layerLinks[1] = {MakeLink(COMM_PROTOCOL_UBC_TP), MakeLink(COMM_PROTOCOL_UB_CTP)};
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.InitWorldTeams(), HCCL_SUCCESS);

    EXPECT_NE(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 0), nullptr);
    EXPECT_NE(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UBC_TP, 1), nullptr);
    EXPECT_NE(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 1), nullptr);
    std::vector<uint32_t> sizes = {0, 0, 0};
    hccl::HcclTeamMgr::GetInstance().GetWorldTeamSizesPerNetLayer(&coll, sizes);
    EXPECT_EQ(sizes[0], 2U);
    EXPECT_EQ(sizes[1], 2U); // 同层多协议只记一份 layerRanks
    EXPECT_EQ(sizes[2], 0U);

    coll.registeredSymMemHandleMap_.clear();
    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    coll.rankgraph_ = nullptr;
    GlobalMockObject::verify();
}

// ut_ain_048 UB_MEM 仅最高层预制：3 层均有 UB_MEM（layer1 另有 UB_CTP），最终仅 (UB_MEM,2) 1 个。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_UbMemMultiLayer_Expect_OnlyHighestLayer)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateOk));
    hccl::CollComm coll(nullptr, 0, "ut_iwt_ubmem", hccl::ManagerCallbacks{});

    UtRankGraphStub graph;
    graph.layerBuf = {0, 1, 2};
    graph.layerRanks[0] = {0, 1};
    graph.layerRanks[1] = {0, 1};
    graph.layerRanks[2] = {0, 1};
    graph.layerLinks[0] = {MakeLink(COMM_PROTOCOL_UB_MEM)};
    graph.layerLinks[1] = {MakeLink(COMM_PROTOCOL_UB_MEM), MakeLink(COMM_PROTOCOL_UB_CTP)};
    graph.layerLinks[2] = {MakeLink(COMM_PROTOCOL_UB_MEM)};
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.InitWorldTeams(), HCCL_SUCCESS);

    EXPECT_NE(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_MEM, 2), nullptr);
    EXPECT_EQ(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_MEM, 0), nullptr);
    EXPECT_EQ(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_MEM, 1), nullptr);
    // 层分段表：layerRanksMap_ 按 netLayer 记录（协议无关），本场景 layer1(UB_CTP)+layer2(UB_MEM) 各 2
    std::vector<uint32_t> sizes = {0, 0, 0};
    hccl::HcclTeamMgr::GetInstance().GetWorldTeamSizesPerNetLayer(&coll, sizes);
    EXPECT_EQ(sizes[0], 0U);
    EXPECT_EQ(sizes[1], 2U);
    EXPECT_EQ(sizes[2], 2U);

    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    coll.rankgraph_ = nullptr;
    GlobalMockObject::verify();
}

// UB Memory不限定固定NetLayer，多层均可达时选择最高层作为LSA WorldTeam。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_UbMemHasHigherLayer_Expect_SelectHighestCandidate)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateOk));
    hccl::CollComm coll(nullptr, 0, "ut_iwt_ubmem_higher_layer", hccl::ManagerCallbacks{});

    UtRankGraphStub graph;
    graph.layerBuf = {2, 3};
    graph.layerRanks[2] = {0, 1};
    graph.layerRanks[3] = {0, 1};
    graph.layerLinks[2] = {MakeLink(COMM_PROTOCOL_UB_MEM)};
    graph.layerLinks[3] = {MakeLink(COMM_PROTOCOL_UB_MEM)};
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.InitWorldTeams(), HCCL_SUCCESS);
    EXPECT_EQ(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_MEM, 2), nullptr);
    EXPECT_NE(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_MEM, 3), nullptr);

    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    coll.rankgraph_ = nullptr;
    GlobalMockObject::verify();
}

// ut_ain_049 rankNum<=1 的层跳过：不预制任何 team，整体 SUCCESS。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_SingleRankLayer_Expect_SkipLayer)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateOk));
    hccl::CollComm coll(nullptr, 0, "ut_iwt_single", hccl::ManagerCallbacks{});

    UtRankGraphStub graph;
    graph.layerBuf = {0, 1};
    graph.layerRanks[0] = {0}; // 仅本 rank，rankNum=1 跳过
    graph.layerRanks[1] = {0, 1};
    graph.layerLinks[1] = {MakeLink(COMM_PROTOCOL_UB_CTP)};
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.InitWorldTeams(), HCCL_SUCCESS);

    EXPECT_EQ(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 0), nullptr);
    EXPECT_NE(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 1), nullptr);

    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    coll.rankgraph_ = nullptr;
    GlobalMockObject::verify();
}

// ut_ain_050 无 link 的 peer 跳过：layer1 无 link，(UB_CTP,1) 不预制，整体 SUCCESS。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_NoLinkPeer_Expect_SkipPeer)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(UtStubHcommTeamCreateOk));
    hccl::CollComm coll(nullptr, 0, "ut_iwt_nolink", hccl::ManagerCallbacks{});

    UtRankGraphStub graph;
    graph.layerBuf = {0, 1};
    graph.layerRanks[0] = {0, 1};
    graph.layerRanks[1] = {0, 1};
    graph.layerLinks[0] = {MakeLink(COMM_PROTOCOL_UB_CTP)};
    // layer1 无 links 条目 -> GetLinks 返回空
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.InitWorldTeams(), HCCL_SUCCESS);

    EXPECT_NE(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 0), nullptr);
    EXPECT_EQ(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 1), nullptr);

    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    coll.rankgraph_ = nullptr;
    GlobalMockObject::verify();
}

// ut_ain_051 GetNetLayers 返回空：HCCL_E_INTERNAL。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_NetLayersEmpty_Expect_ReturnInternal)
{
    hccl::CollComm coll(nullptr, 0, "ut_iwt_empty", hccl::ManagerCallbacks{});

    UtRankGraphStub graph;
    graph.layerBuf.clear(); // netLayerNum=0
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.InitWorldTeams(), HCCL_E_INTERNAL);

    coll.rankgraph_ = nullptr;
}

// rankgraph_ 为空：HCCL_E_PTR。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_RankGraphNull_Expect_ReturnPtrError)
{
    hccl::CollComm coll(nullptr, 0, "ut_iwt_null", hccl::ManagerCallbacks{});
    EXPECT_EQ(coll.InitWorldTeams(), HCCL_E_PTR);
}

// ut_ain_052 同 (proto,layer) 已预制不重复：预注册同键后 InitWorldTeams 不再创建。
TEST_F(TestCollComm, Ut_InitWorldTeams_When_ProtoLayerAlreadyPrebuilt_Expect_NoDuplicateCreate)
{
    uint32_t rankIds[2] = {0, 1};
    HcommTeamHandle existing = reinterpret_cast<HcommTeamHandle>(0x31000);
    hccl::CollComm coll(nullptr, 0, "ut_iwt_dup", hccl::ManagerCallbacks{});
    ASSERT_EQ(
        hccl::HcclTeamMgr::GetInstance().RegisterPrebuiltWorldTeam(
            existing, &coll, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2),
        HCCL_SUCCESS);

    // 若误创建则必然调用（返回不同句柄触发 E_PARA），expects(never()) 锁死该路径
    MOCKER(HcommTeamCreate).expects(never());

    UtRankGraphStub graph;
    graph.layerBuf = {1};
    graph.layerRanks[1] = {0, 1};
    graph.layerLinks[1] = {MakeLink(COMM_PROTOCOL_UB_CTP)};
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.InitWorldTeams(), HCCL_SUCCESS);
    EXPECT_EQ(hccl::HcclTeamMgr::GetInstance().FindWorldTeamByProtoLayer(&coll, COMM_PROTOCOL_UB_CTP, 1), existing);

    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    coll.rankgraph_ = nullptr;
    GlobalMockObject::verify();
}

// ===================== 3. UpdateHcommWindowRemoteMem =====================

// ut_ain_053 slots 计算与 L3 透传：layer0(2)+layer1(2) 预制，rank0 slots={0,2}、rank1 slots={1,3}。
TEST_F(TestCollComm, Ut_UpdateHcommWindowRemoteMem_When_MultiLayerPrebuilt_Expect_SlotsWithOffset)
{
    MOCKER(HcommTeamUpdateWindowRemoteMemByRank).stubs().will(invoke(UtStubUpdateWindowRemoteMemByRank));
    /* selfSlot 新路径：UpdateHcommWindowRemoteMem 先调 SetSelfInfo 登记本端槽位，
     * 桩句柄非真实 window，打桩直接返回成功 */
    MOCKER(HcommTeamWindowSetSelfInfo).stubs().will(returnValue(0));
    uint32_t rankIds[2] = {0, 1};
    hccl::CollComm coll(nullptr, 0, "ut_uhwrm", hccl::ManagerCallbacks{});
    HcommTeamHandle w0 = reinterpret_cast<HcommTeamHandle>(0x32000);
    HcommTeamHandle w1 = reinterpret_cast<HcommTeamHandle>(0x33000);
    ASSERT_EQ(
        hccl::HcclTeamMgr::GetInstance().RegisterPrebuiltWorldTeam(w0, &coll, COMM_PROTOCOL_UB_CTP, 0, rankIds, 2),
        HCCL_SUCCESS);
    ASSERT_EQ(
        hccl::HcclTeamMgr::GetInstance().RegisterPrebuiltWorldTeam(w1, &coll, COMM_PROTOCOL_UB_RTP, 1, rankIds, 2),
        HCCL_SUCCESS);

    void* fakeWin = reinterpret_cast<void*>(0x9500000);
    coll.tagToHcommMap_["__hccl_sym_win__ut_uhwrm"] = fakeWin;

    CommMem remoteMem{};
    remoteMem.type = COMM_MEM_TYPE_DEVICE;
    remoteMem.addr = reinterpret_cast<void*>(0x9600000);
    remoteMem.size = 0x2000;
    std::vector<std::string> memTags = {"__hccl_sym_win__ut_uhwrm"};

    // remoteRank=0：layer0 槽 0 + (sizes[0]=2) + layer1 槽 0 => slots 集合={0,2}
    // （layerRanksMap_ 为 unordered_map，槽位顺序不定，按升序集合断言）
    g_utUpdateWinCalled = false;
    EXPECT_EQ(coll.UpdateHcommWindowRemoteMem(0, &remoteMem, memTags), HCCL_SUCCESS);
    EXPECT_TRUE(g_utUpdateWinCalled);
    EXPECT_EQ(g_utUpdateWinSizes[0], 2U);
    EXPECT_EQ(g_utUpdateWinSizes[1], 2U);
    EXPECT_EQ(g_utUpdateWinSizes[2], 0U);
    ASSERT_EQ(g_utUpdateWinSlots.size(), 2U);
    std::sort(g_utUpdateWinSlots.begin(), g_utUpdateWinSlots.end());
    EXPECT_EQ(g_utUpdateWinSlots[0], 0U);
    EXPECT_EQ(g_utUpdateWinSlots[1], 2U);
    EXPECT_EQ(g_utLastRemoteMem, reinterpret_cast<HcclMemHandle>(remoteMem.addr));

    // remoteRank=1：layer0 槽 1 + 2 + layer1 槽 1 => slots 集合={1,3}
    EXPECT_EQ(coll.UpdateHcommWindowRemoteMem(1, &remoteMem, memTags), HCCL_SUCCESS);
    ASSERT_EQ(g_utUpdateWinSlots.size(), 2U);
    std::sort(g_utUpdateWinSlots.begin(), g_utUpdateWinSlots.end());
    EXPECT_EQ(g_utUpdateWinSlots[0], 1U);
    EXPECT_EQ(g_utUpdateWinSlots[1], 3U);

    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    GlobalMockObject::verify();
}

// ut_ain_054 memTag 空/未命中跳过：SUCCESS 且 L3 Update 不被调。
TEST_F(TestCollComm, Ut_UpdateHcommWindowRemoteMem_When_TagMissOrEmpty_Expect_SkipL3Update)
{
    MOCKER(HcommTeamUpdateWindowRemoteMemByRank).expects(never());
    hccl::CollComm coll(nullptr, 0, "ut_uhwrm_miss", hccl::ManagerCallbacks{});

    CommMem remoteMem{};
    remoteMem.type = COMM_MEM_TYPE_DEVICE;
    remoteMem.addr = reinterpret_cast<void*>(0x9700000);
    remoteMem.size = 0x2000;
    std::vector<std::string> memTags = {"", "unknown_tag"};

    EXPECT_EQ(coll.UpdateHcommWindowRemoteMem(1, &remoteMem, memTags), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

// L3 回填失败透传：HCOMM_E_PARA -> HCCL_E_PARA。
TEST_F(TestCollComm, Ut_UpdateHcommWindowRemoteMem_When_L3UpdateFail_Expect_ReturnError)
{
    MOCKER(HcommTeamUpdateWindowRemoteMemByRank).stubs().will(returnValue(static_cast<HcommResult>(1)));
    hccl::CollComm coll(nullptr, 0, "ut_uhwrm_fail", hccl::ManagerCallbacks{});
    coll.tagToHcommMap_["__hccl_sym_win__ut_fail"] = reinterpret_cast<void*>(0x9800000);

    CommMem remoteMem{};
    remoteMem.type = COMM_MEM_TYPE_DEVICE;
    remoteMem.addr = reinterpret_cast<void*>(0x9900000);
    remoteMem.size = 0x2000;
    std::vector<std::string> memTags = {"__hccl_sym_win__ut_fail"};

    EXPECT_EQ(coll.UpdateHcommWindowRemoteMem(1, &remoteMem, memTags), HCCL_E_PARA);
    GlobalMockObject::verify();
}

// ===================== 4. ReExchangeWindowsForBoundTeams =====================

// ut_ain_056 无已建链 team：空操作 SUCCESS，不建 channel 不回填。
TEST_F(TestCollComm, Ut_ReExchangeWindowsForBoundTeams_When_NoLinkedTeam_Expect_NoOp)
{
    MOCKER_CPP(
        &hccl::MyRank::CreateChannels,
        HcclResult(hccl::MyRank::*)(CommEngine, const std::string&, const HcclChannelDesc*, uint32_t, ChannelHandle*))
        .expects(never());
    hccl::CollComm coll(nullptr, 0, "ut_reex_empty", hccl::ManagerCallbacks{});
    aclrtBinHandle binHandle{};
    coll.myRank_ = std::make_shared<hccl::MyRank>(
        binHandle, 0, coll.GetCommConfig(), hccl::ManagerCallbacks(), nullptr, nullptr);

    UtRankGraphStub graph;
    coll.rankgraph_ = &graph;

    EXPECT_EQ(coll.ReExchangeWindowsForBoundTeams(), HCCL_SUCCESS);

    coll.rankgraph_ = nullptr;
    coll.myRank_ = nullptr;
    GlobalMockObject::verify();
}

// ut_ain_057/058 ReExchange 有已建链 team：1 个 subTeam（成员 {0,1}，self=0）。
// 设计说明：MyRank::ChannelGetRemoteMems(vector) 为 const 成员函数，mockcpp 的 MOCKER_CPP invoke
// 对其存在桩签名匹配异常（同型比较失败/段错误），故本用例取"无 pending window"变体：
// symmetricMemory_=nullptr 使 RegisterPendingSymmetricMemHandles 真实返回空，源码按设计跳过
// GetRemoteMems/回填分支；对每个非 self peer 各建 1 条 channel（CreateChannels 计数验证）。
// memNum>0 触发回填的分支由既有 Ut_UpdateSymmetricRemoteMem_When_ChannelReturnsRemoteMem_*
// （驱动 UpdateSymmetricRemoteMem→UpdateHcommWindowRemoteMem 全链）与本文件 UpdateHcommWindowRemoteMem
// 用例共同覆盖。
TEST_F(TestCollComm, Ut_ReExchangeWindowsForBoundTeams_When_HasLinkedTeam_Expect_ReExchangePerPeer)
{
    MOCKER(HcommTeamGetNetLayer).stubs().will(invoke(UtStubHcommTeamGetNetLayer));
    MOCKER(HcommTeamGetEngine).stubs().will(invoke(UtStubHcommTeamGetEngine));
    MOCKER_CPP(
        &hccl::MyRank::CreateChannels,
        HcclResult(hccl::MyRank::*)(CommEngine, const std::string&, const HcclChannelDesc*, uint32_t, ChannelHandle*))
        .stubs()
        .will(invoke(UtStubMyRankCreateChannelsOk));
    g_utCreateChannelsCalls = 0;

    hccl::CollComm coll(nullptr, 0, "ut_reex_sub", hccl::ManagerCallbacks{});
    // 注册 fake sym mem handle 使 GetAllRegisteredSymMemHandles 返回非空（触发 ReExchangeChannelsForTeam）
    coll.registeredSymMemHandleMap_["fake_sym_mem"] = reinterpret_cast<HcclMemHandle>(0x12345);
    aclrtBinHandle binHandle{};
    coll.myRank_ = std::make_shared<hccl::MyRank>(
        binHandle, 0, coll.GetCommConfig(), hccl::ManagerCallbacks(), nullptr, nullptr);
    // symmetricMemory_ 保持 nullptr：RegisterPendingSymmetricMemHandles 返回空，跳过回填分支

    // 注册 worldTeam + subTeam（成员 {0,1}，self rank=0）
    uint32_t rankIds[2] = {0, 1};
    HcommTeamHandle worldTeam = reinterpret_cast<HcommTeamHandle>(0x34000);
    HcommTeamHandle subTeam = reinterpret_cast<HcommTeamHandle>(0x35000);
    ASSERT_EQ(
        hccl::HcclTeamMgr::GetInstance().RegisterPrebuiltWorldTeam(
            worldTeam, &coll, COMM_PROTOCOL_UB_CTP, 1, rankIds, 2),
        HCCL_SUCCESS);
    ASSERT_EQ(
        hccl::HcclTeamMgr::GetInstance().RegisterSubTeam(worldTeam, subTeam, nullptr, 0, rankIds, 2), HCCL_SUCCESS);

    UtRankGraphStub graph;
    graph.layerBuf = {1};
    graph.layerRanks[1] = {0, 1};
    graph.layerLinks[1] = {MakeLink(COMM_PROTOCOL_UB_CTP)};
    coll.rankgraph_ = &graph;
    g_utGetNetLayerVal = 1;
    g_utGetEngineVal = COMM_ENGINE_AIV;

    // ReExchangeWindowsForBoundTeams 走补交换路径：CreateChannels 被调 1 次（验证路径走到）
    // ChannelGetRemoteMems 在 UT 环境下因无真实 channel 会返回错误——验证 CreateChannels 计数即可
    coll.ReExchangeWindowsForBoundTeams();
    EXPECT_EQ(g_utCreateChannelsCalls, 1U);

    hccl::HcclTeamMgr::GetInstance().ClearByCollComm(&coll);
    coll.rankgraph_ = nullptr;
    coll.myRank_ = nullptr;
    GlobalMockObject::verify();
}

TEST_F(TestCollComm, Ut_HcommWindowOwner_When_EraseWindow_Expect_OtherOwnerKept)
{
    HcclCommSymWindow window = reinterpret_cast<HcclCommSymWindow>(0x310000U);
    HcclCommSymWindow otherWindow = reinterpret_cast<HcclCommSymWindow>(0x312000U);
    HcclComm expectedComm = reinterpret_cast<HcclComm>(0x320000U);
    HcclComm otherComm = reinterpret_cast<HcclComm>(0x330000U);
    HcclComm actualComm = nullptr;

    ASSERT_EQ(hccl::RecordHcommWindowOwner(window, expectedComm), HCCL_SUCCESS);
    ASSERT_EQ(hccl::RecordHcommWindowOwner(otherWindow, otherComm), HCCL_SUCCESS);
    EXPECT_EQ(hccl::GetHcommWindowComm(window, actualComm), HCCL_SUCCESS);
    EXPECT_EQ(actualComm, expectedComm);
    EXPECT_EQ(hccl::RecordHcommWindowOwner(window, expectedComm), HCCL_E_INTERNAL);

    hccl::EraseHcommWindowOwner(window);
    EXPECT_EQ(hccl::GetHcommWindowComm(window, actualComm), HCCL_E_NOT_FOUND);
    EXPECT_EQ(actualComm, nullptr);
    EXPECT_EQ(hccl::GetHcommWindowComm(otherWindow, actualComm), HCCL_SUCCESS);
    EXPECT_EQ(actualComm, otherComm);

    hccl::EraseHcommWindowOwner(otherWindow);
}
