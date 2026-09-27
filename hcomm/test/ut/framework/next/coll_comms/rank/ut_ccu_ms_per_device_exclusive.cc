/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#include "coll_comm_mgr.h"
#include "my_rank.h"
#undef private

#include "ccu_device_pub.h"
#include "ccu_res.h"
#include "op_base.h"

namespace {
constexpr int32_t DEVICE_0 = 0;
constexpr int32_t DEVICE_1 = 1;
constexpr uint32_t AICPU_TS_MODE = 2;
constexpr uint32_t CCU_MS_MODE = 5;
constexpr uint32_t CCU_SCHED_MODE = 6;
constexpr CcuInsHandle VALID_CCU_INS_HANDLE = 1;

std::atomic<bool> g_reservationHeldDuringInstanceDestroy{false};
std::atomic<bool> g_reservationHeldDuringDriverDeinit{false};
bool g_reservationHeldDuringInitFailure = false;

HcclResult StubDeviceRefresh(s32& deviceLogicId)
{
    deviceLogicId = DEVICE_0;
    return HCCL_SUCCESS;
}

CcuResult StubDestroyCcuInstance(CcuInsHandle ccuInsHandle)
{
    EXPECT_EQ(ccuInsHandle, VALID_CCU_INS_HANDLE);
    bool reserved = false;
    EXPECT_EQ(
        hccl::CollCommMgr::GetInstance().TryReserveCcuMsComm(DEVICE_0, "instance_destroy_probe", reserved),
        HCCL_SUCCESS);
    g_reservationHeldDuringInstanceDestroy.store(!reserved);
    return CcuResult::CCU_SUCCESS;
}

CcuResult StubDeinitCcuFeature(int32_t deviceLogicId)
{
    EXPECT_EQ(deviceLogicId, DEVICE_0);
    bool reserved = false;
    EXPECT_EQ(
        hccl::CollCommMgr::GetInstance().TryReserveCcuMsComm(DEVICE_0, "driver_deinit_probe", reserved), HCCL_SUCCESS);
    g_reservationHeldDuringDriverDeinit.store(!reserved);
    return CcuResult::CCU_SUCCESS;
}

// 简化后 TryInitCcuInstance 只走 OnDemand（拉起驱动 CcuInitFeature），不再创建 ccu instance；
// 驱动拉起失败时的预约对账行为改由 CcuInitFeature 返回值驱动
CcuResult StubCcuInitFeatureFailure(const int32_t devLogicId, std::shared_ptr<hcomm::CcuDrvHandle>& ccuDrvHandle)
{
    EXPECT_EQ(devLogicId, DEVICE_0);
    (void)ccuDrvHandle;
    bool reserved = false;
    EXPECT_EQ(
        hccl::CollCommMgr::GetInstance().TryReserveCcuMsComm(DEVICE_0, "init_failure_probe", reserved), HCCL_SUCCESS);
    g_reservationHeldDuringInitFailure = !reserved;
    return CcuResult::CCU_E_PARA;
}
} // namespace

class CcuMsPerDeviceExclusiveTest : public testing::Test {
protected:
    void SetUp() override
    {
        ResetReservations();
        myRank_ = CreateMyRank("ccu_ms_test_comm");
    }

    void TearDown() override
    {
        myRank_.reset();
        ResetReservations();
        GlobalMockObject::verify();
    }

    static std::unique_ptr<hccl::MyRank> CreateMyRank(const std::string& commName)
    {
        hccl::CommConfig config(commName);
        return std::make_unique<hccl::MyRank>(nullptr, 0, config, hccl::ManagerCallbacks{}, nullptr, nullptr);
    }

    static void ResetReservations()
    {
        auto* manager = &hccl::CollCommMgr::GetInstance();
        std::lock_guard<std::mutex> lock(manager->ccuMsCommMutex_);
        manager->ccuMsCommIds_.fill("");
    }

    static HcclMem CreateCclBuffer()
    {
        HcclMem cclBuffer{};
        cclBuffer.addr = reinterpret_cast<void*>(0xab);
        cclBuffer.size = 1024;
        cclBuffer.type = HCCL_MEM_TYPE_DEVICE;
        return cclBuffer;
    }

    std::unique_ptr<hccl::MyRank> myRank_;
};

TEST_F(CcuMsPerDeviceExclusiveTest, Ut_TryReserveCcuMsComm_When_SameDeviceConcurrent_Expect_OnlyOneReserved)
{
    auto* manager = &hccl::CollCommMgr::GetInstance();
    std::atomic<uint32_t> readyCount{0};
    std::atomic<bool> start{false};
    HcclResult firstRet = HCCL_E_INTERNAL;
    HcclResult secondRet = HCCL_E_INTERNAL;
    bool firstReserved = false;
    bool secondReserved = false;

    auto reserve = [&](const std::string& commId, HcclResult& ret, bool& reserved) {
        readyCount.fetch_add(1);
        while (!start.load()) {
            std::this_thread::yield();
        }
        ret = manager->TryReserveCcuMsComm(DEVICE_0, commId, reserved);
    };
    std::thread firstThread(reserve, "concurrent_owner_0", std::ref(firstRet), std::ref(firstReserved));
    std::thread secondThread(reserve, "concurrent_owner_1", std::ref(secondRet), std::ref(secondReserved));
    while (readyCount.load() != 2U) {
        std::this_thread::yield();
    }
    start.store(true);
    firstThread.join();
    secondThread.join();

    EXPECT_EQ(firstRet, HCCL_SUCCESS);
    EXPECT_EQ(secondRet, HCCL_SUCCESS);
    EXPECT_EQ(static_cast<uint32_t>(firstReserved) + static_cast<uint32_t>(secondReserved), 1U);
}

TEST_F(CcuMsPerDeviceExclusiveTest, Ut_TryReserveCcuMsComm_When_DifferentDevices_Expect_BothReserved)
{
    bool firstReserved = false;
    bool secondReserved = false;
    auto* manager = &hccl::CollCommMgr::GetInstance();

    EXPECT_EQ(manager->TryReserveCcuMsComm(DEVICE_0, "device_0_owner", firstReserved), HCCL_SUCCESS);
    EXPECT_EQ(manager->TryReserveCcuMsComm(DEVICE_1, "device_1_owner", secondReserved), HCCL_SUCCESS);
    EXPECT_TRUE(firstReserved);
    EXPECT_TRUE(secondReserved);
}

TEST_F(CcuMsPerDeviceExclusiveTest, Ut_ReleaseCcuMsComm_When_NonOwnerReleases_Expect_OwnerPreserved)
{
    auto* manager = &hccl::CollCommMgr::GetInstance();
    bool ownerReserved = false;
    ASSERT_EQ(manager->TryReserveCcuMsComm(DEVICE_0, "owner", ownerReserved), HCCL_SUCCESS);
    ASSERT_TRUE(ownerReserved);

    manager->ReleaseCcuMsComm(DEVICE_0, "non_owner");

    bool nextReserved = false;
    EXPECT_EQ(manager->TryReserveCcuMsComm(DEVICE_0, "next_owner", nextReserved), HCCL_SUCCESS);
    EXPECT_FALSE(nextReserved);
}

TEST_F(CcuMsPerDeviceExclusiveTest, Ut_ReleaseCcuMsComm_When_OwnerReleases_Expect_NextCanReserve)
{
    auto* manager = &hccl::CollCommMgr::GetInstance();
    bool ownerReserved = false;
    ASSERT_EQ(manager->TryReserveCcuMsComm(DEVICE_0, "owner", ownerReserved), HCCL_SUCCESS);
    ASSERT_TRUE(ownerReserved);

    manager->ReleaseCcuMsComm(DEVICE_0, "owner");

    bool nextReserved = false;
    EXPECT_EQ(manager->TryReserveCcuMsComm(DEVICE_0, "next_owner", nextReserved), HCCL_SUCCESS);
    EXPECT_TRUE(nextReserved);
}

TEST_F(CcuMsPerDeviceExclusiveTest, Ut_MyRankInit_When_RankNumIsOneAndModeIsMs_Expect_NoReservation)
{
    MOCKER_CPP(&hccl::MyRank::TryInitCcuInstance).expects(never());

    EXPECT_EQ(myRank_->Init(CreateCclBuffer(), CCU_MS_MODE, 1), HCCL_SUCCESS);

    bool nextReserved = false;
    EXPECT_EQ(hccl::CollCommMgr::GetInstance().TryReserveCcuMsComm(DEVICE_0, "next_owner", nextReserved), HCCL_SUCCESS);
    EXPECT_TRUE(nextReserved);
}

TEST_F(CcuMsPerDeviceExclusiveTest, Ut_TryInitCcuInstance_When_SameDeviceAlreadyOwned_Expect_FallbackToSched)
{
    auto* manager = &hccl::CollCommMgr::GetInstance();
    bool ownerReserved = false;
    ASSERT_EQ(manager->TryReserveCcuMsComm(DEVICE_0, "existing_owner", ownerReserved), HCCL_SUCCESS);
    ASSERT_TRUE(ownerReserved);
    myRank_->devLogicId_ = DEVICE_0;
    myRank_->opExpansionMode_ = CCU_MS_MODE;
    // 简化后：MS 回退到 SCHED 由 ReserveCcuMsCommOrFallback 完成（预约失败即回退），
    // 不再依赖 ccu instance 创建；驱动拉起走 CcuInitFeature
    MOCKER(hcomm::CcuInitFeature).expects(once()).will(returnValue(CcuResult::CCU_SUCCESS));

    EXPECT_EQ(myRank_->TryInitCcuInstance(), HCCL_SUCCESS);
    EXPECT_EQ(myRank_->opExpansionMode_, CCU_SCHED_MODE);
    EXPECT_FALSE(myRank_->ccuMsCommReserved_);

    myRank_.reset();
    bool nextReserved = false;
    EXPECT_EQ(manager->TryReserveCcuMsComm(DEVICE_0, "next_owner", nextReserved), HCCL_SUCCESS);
    EXPECT_FALSE(nextReserved);
}

TEST_F(CcuMsPerDeviceExclusiveTest, Ut_TryInitCcuInstance_When_CcuInitFails_Expect_ReservationReleased)
{
    myRank_->devLogicId_ = DEVICE_0;
    myRank_->opExpansionMode_ = CCU_MS_MODE;
    g_reservationHeldDuringInitFailure = false;
    // 简化后 ccu instance 不在 init 时创建，驱动拉起失败由 CcuInitFeature 返回
    MOCKER(hcomm::CcuInitFeature).expects(once()).will(invoke(StubCcuInitFeatureFailure));

    EXPECT_EQ(myRank_->TryInitCcuInstance(), HCCL_E_PARA);
    EXPECT_TRUE(g_reservationHeldDuringInitFailure);
    EXPECT_FALSE(myRank_->ccuMsCommReserved_);

    bool nextReserved = false;
    EXPECT_EQ(hccl::CollCommMgr::GetInstance().TryReserveCcuMsComm(DEVICE_0, "next_owner", nextReserved), HCCL_SUCCESS);
    EXPECT_TRUE(nextReserved);
}

TEST_F(CcuMsPerDeviceExclusiveTest, Ut_TryInitCcuInstance_When_DriverBusyFallsBackToAicpu_Expect_ReservationReleased)
{
    myRank_->devLogicId_ = DEVICE_0;
    myRank_->opExpansionMode_ = CCU_MS_MODE;
    // 简化后驱动忙回退到 aicpu 由 TryInitCcuInstanceOnDemand 中 CcuInitFeature 返回 CCU_E_DRV_BUSY 触发
    MOCKER(hcomm::CcuInitFeature).expects(once()).will(returnValue(CcuResult::CCU_E_DRV_BUSY));

    EXPECT_EQ(myRank_->TryInitCcuInstance(), HCCL_SUCCESS);
    EXPECT_EQ(myRank_->opExpansionMode_, AICPU_TS_MODE);
    EXPECT_FALSE(myRank_->ccuMsCommReserved_);

    bool nextReserved = false;
    EXPECT_EQ(hccl::CollCommMgr::GetInstance().TryReserveCcuMsComm(DEVICE_0, "next_owner", nextReserved), HCCL_SUCCESS);
    EXPECT_TRUE(nextReserved);
}

TEST_F(CcuMsPerDeviceExclusiveTest, Ut_MyRankDestructor_When_HoldsMsReservation_Expect_ReleaseAfterCcuCleanup)
{
    auto* manager = &hccl::CollCommMgr::GetInstance();
    bool ownerReserved = false;
    ASSERT_EQ(manager->TryReserveCcuMsComm(DEVICE_0, "destructor_owner", ownerReserved), HCCL_SUCCESS);
    ASSERT_TRUE(ownerReserved);
    myRank_ = CreateMyRank("destructor_owner");
    myRank_->devLogicId_ = DEVICE_0;
    myRank_->ccuMsCommReserved_ = true;
    myRank_->ccuInsHandle_ = VALID_CCU_INS_HANDLE;
    // 删除 useCcuResStaticAlloc_ 成员；析构路径按 ccuInsHandle_/assignedCcuInsHandle_ 是否非 0 决定清理
    auto* fakeDrvHandle = reinterpret_cast<hcomm::CcuDrvHandle*>(0x1);
    myRank_->ccuDrvHandle_ = std::shared_ptr<hcomm::CcuDrvHandle>(fakeDrvHandle, [](hcomm::CcuDrvHandle*) {});
    g_reservationHeldDuringInstanceDestroy.store(false);
    g_reservationHeldDuringDriverDeinit.store(false);
    MOCKER(HcclDeviceRefresh).expects(once()).will(invoke(StubDeviceRefresh));
    MOCKER(HcommCcuInsDestroy).expects(once()).will(invoke(StubDestroyCcuInstance));
    MOCKER(hcomm::CcuDeinitFeature).expects(once()).will(invoke(StubDeinitCcuFeature));

    myRank_.reset();

    EXPECT_TRUE(g_reservationHeldDuringInstanceDestroy.load());
    EXPECT_TRUE(g_reservationHeldDuringDriverDeinit.load());
    bool nextReserved = false;
    EXPECT_EQ(manager->TryReserveCcuMsComm(DEVICE_0, "next_owner", nextReserved), HCCL_SUCCESS);
    EXPECT_TRUE(nextReserved);
}

TEST_F(CcuMsPerDeviceExclusiveTest, Ut_TryReserveCcuMsComm_When_DeviceIdInvalid_Expect_ReturnError)
{
    auto* manager = &hccl::CollCommMgr::GetInstance();
    bool ownerReserved = false;
    ASSERT_EQ(manager->TryReserveCcuMsComm(DEVICE_0, "device_0_owner", ownerReserved), HCCL_SUCCESS);
    ASSERT_TRUE(ownerReserved);
    bool reserved = true;

    EXPECT_EQ(manager->TryReserveCcuMsComm(-1, "invalid_owner", reserved), HCCL_E_PARA);
    EXPECT_FALSE(reserved);
    reserved = true;
    EXPECT_EQ(manager->TryReserveCcuMsComm(MAX_MODULE_DEVICE_NUM, "invalid_owner", reserved), HCCL_E_PARA);
    EXPECT_FALSE(reserved);

    bool nextReserved = false;
    EXPECT_EQ(manager->TryReserveCcuMsComm(DEVICE_0, "next_owner", nextReserved), HCCL_SUCCESS);
    EXPECT_FALSE(nextReserved);
}
