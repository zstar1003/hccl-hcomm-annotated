/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * UT: HcommCcuCascCnt API — 3 public interfaces + internal components
 *
 *   HcommCcuCascCntAlloc / HcommCcuCascCntGetMem / HcommCcuGetRmtMemToken
 *   CcuResPack::AllocCascCntBlock / AcquireCascCntBlock / GetCascCntBlock
 */

#include <iostream>
#include "hccl_api_base_test.h"

#define private public
#define protected public

#include "log.h"
#include "hcom_common.h"
#include "op_base.h"
#include "ccu_device_pub.h"
#include "ccu_res.h"
#include "ccu_types.h"
#include "ccu_device_res.h"
#include "ccu_res_desc_mgr.h"
#include "ccu_res_specs.h"
#include "ccu_res_batch_allocator.h"
#include "ccu_instance_mgr.h"
#include "ccu_res_pack.h"
#include "ccu_res_repo.h"
#include "hcomm_res_defs.h"

#include "mocks/ccu_device_mock_utils.h"

#undef protected
#undef private

using namespace hcomm;

namespace {
constexpr int32_t TEST_DEVICE_LOGIC_ID = 0;
int32_t g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
HcclResult g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;
// 校验输出句柄时的非 0 初值：若初值就为 0，"失败置 0"/"失败不写"的断言恒真，验证不到实际行为。
constexpr HcommCcuCascCntHandle STALE_CASC_CNT_HANDLE = 0x1234;

HcclResult MockHrtGetDeviceRefresh(int32_t* deviceLogicId)
{
    if (deviceLogicId == nullptr) {
        return HcclResult::HCCL_E_PTR;
    }
    if (g_deviceRefreshResult != HcclResult::HCCL_SUCCESS) {
        return g_deviceRefreshResult;
    }
    *deviceLogicId = g_runtimeDeviceLogicId;
    return HcclResult::HCCL_SUCCESS;
}

int g_allocCallSeq = 0;
int g_releaseCallCount = 0;
// 用全局变量控制第几次 alloc 调用失败，而不是在用例内重复 MOCKER 同名函数。
// mockcpp 中 SetUp 已注册的 stub 会先命中，用例内再次 MOCKER 无法覆盖，故统一走 SetUp 的默认 mock。
// -1 表示所有调用都成功；N 表示第 N 次（0-based）调用返回 CCU_E_UNAVAIL。
int g_allocFailOnCall = -1;

CcuResult MockCcuAllocCntXnBlock(const int32_t devLogicId, const uint8_t dieId, CntXnBlock& cntXnBlock)
{
    (void)devLogicId;
    int seq = g_allocCallSeq++;
    if (g_allocFailOnCall >= 0 && seq == g_allocFailOnCall) {
        return CcuResult::CCU_E_UNAVAIL;
    }
    cntXnBlock.wishCntXns = {0, 1021};
    cntXnBlock.wishCntXnsMem = {reinterpret_cast<void*>(0x1000ULL + dieId * 0x100ULL), 1022 * sizeof(uint64_t)};
    cntXnBlock.totalCntXn = dieId * 4;
    cntXnBlock.expectedCntXn = dieId * 4 + 1;
    cntXnBlock.blockIdx = 0;
    return CcuResult::CCU_SUCCESS;
}

CcuResult MockReleaseTrack(const int32_t, const uint8_t, const CntXnBlock&)
{
    g_releaseCallCount++;
    return CcuResult::CCU_SUCCESS;
}
} // namespace

class HcommCcuCascCntApiTest : public BaseInit {
public:
    void SetUp() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        BaseInit::SetUp();
        ResDescMgr(TEST_DEVICE_LOGIC_ID).Deinit();
        g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
        g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;
        g_allocCallSeq = 0;
        g_releaseCallCount = 0;
        g_allocFailOnCall = -1;

        MOCKER(hrtGetDeviceRefresh).stubs().with(mockcpp::any()).will(invoke(MockHrtGetDeviceRefresh));
        int32_t seededDeviceLogicId = INVALID_INT;
        ASSERT_EQ(HcclDeviceRefresh(seededDeviceLogicId), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(seededDeviceLogicId, TEST_DEVICE_LOGIC_ID);

        MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));

        // Mock CcuAllocCntXnBlock — fill CntXnBlock with test data
        MOCKER(CcuAllocCntXnBlock)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(invoke(MockCcuAllocCntXnBlock));
        MOCKER(CcuReleaseCntXnBlock)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(invoke(MockReleaseTrack));
    }

    void TearDown() override
    {
        g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
        g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;
        int32_t restoredDeviceLogicId = INVALID_INT;
        EXPECT_EQ(HcclDeviceRefresh(restoredDeviceLogicId), HcclResult::HCCL_SUCCESS);

        ResDescMgr(TEST_DEVICE_LOGIC_ID).Deinit();
        BaseInit::TearDown();
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }

protected:
    static CcuResDescMgr& ResDescMgr(int32_t deviceLogicId)
    {
        return CcuInstanceMgr::GetInstance(deviceLogicId).GetResDescMgr();
    }

    static HcommCcuResDescHandle CreateDescDirect(uint32_t dieId)
    {
        HcommCcuResDescHandle handle = 0;
        EXPECT_EQ(ResDescMgr(TEST_DEVICE_LOGIC_ID).Create(dieId, handle), CcuResult::CCU_SUCCESS);
        EXPECT_NE(handle, 0U);
        return handle;
    }

    static std::unique_ptr<CcuResPack> MakeResPackWithBlocks(uint8_t dieId, uint32_t blockCount)
    {
        auto resPack = std::make_unique<CcuResPack>();
        resPack->userDevId_ = TEST_DEVICE_LOGIC_ID;
        for (uint32_t i = 0; i < blockCount; i++) {
            auto cntXnBlock = std::make_unique<CntXnBlock>();
            MockCcuAllocCntXnBlock(TEST_DEVICE_LOGIC_ID, dieId, *cntXnBlock);
            resPack->cascCntBlocks_[dieId].emplace(std::move(cntXnBlock), false);
        }
        return resPack;
    }
};

// ═══════════════════════════════════════════════════════════
// 1. HcommCcuCascCntAlloc — Parameter Validation
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuCascCntAlloc_When_HandleNull_Expect_CCU_E_PTR)
{
    CcuResult ret = HcommCcuCascCntAlloc(1, 0, nullptr);
    EXPECT_EQ(ret, CCU_E_PTR);
}

// 失败路径统一校验 handle 置 0（见 ccu_res.h 对 HcommCcuCascCntAlloc 的约定）：
// 初值刻意设为非 0，否则 handle 本就为 0 时断言恒真，无法验证清零行为。
TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuCascCntAlloc_When_CcuInsHandleZero_Expect_CCU_E_PARA)
{
    HcommCcuCascCntHandle handle = STALE_CASC_CNT_HANDLE;
    CcuResult ret = HcommCcuCascCntAlloc(0, 0, &handle);
    EXPECT_EQ(ret, CCU_E_PARA);
    EXPECT_EQ(handle, 0U);
}

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuCascCntAlloc_When_DieIdInvalid_Expect_CCU_E_PARA)
{
    HcommCcuCascCntHandle handle = STALE_CASC_CNT_HANDLE;
    CcuResult ret = HcommCcuCascCntAlloc(1, CCU_MAX_IODIE_NUM, &handle);
    EXPECT_EQ(ret, CCU_E_PARA);
    EXPECT_EQ(handle, 0U);
}

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuCascCntAlloc_When_DeviceRefreshFails_Expect_Error)
{
    g_deviceRefreshResult = HcclResult::HCCL_E_INTERNAL;
    HcommCcuCascCntHandle handle = STALE_CASC_CNT_HANDLE;
    CcuResult ret = HcommCcuCascCntAlloc(1, 0, &handle);
    EXPECT_NE(ret, CCU_SUCCESS);
    EXPECT_EQ(handle, 0U);
}

// ═══════════════════════════════════════════════════════════
// 2. HcommCcuCascCntGetMem — Parameter Validation
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuCascCntGetMem_When_HandleZero_Expect_CCU_E_PARA)
{
    CommMem commMem{};
    CcuResult ret = HcommCcuCascCntGetMem(0, &commMem);
    EXPECT_EQ(ret, CCU_E_PARA);
}

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuCascCntGetMem_When_CommMemNull_Expect_CCU_E_PTR)
{
    CcuResult ret = HcommCcuCascCntGetMem(1, nullptr);
    EXPECT_EQ(ret, CCU_E_PTR);
}

// ═══════════════════════════════════════════════════════════
// 3. HcommCcuGetRmtMemToken — Parameter Validation
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuGetRmtMemToken_When_ChannelHandleZero_Expect_CCU_E_PARA)
{
    uint64_t tokenInfo = 0;
    CcuResult ret = HcommCcuGetRmtMemToken(0, 0x1000, &tokenInfo);
    EXPECT_EQ(ret, CCU_E_PARA);
}

TEST_F(HcommCcuCascCntApiTest, Ut_HcommCcuGetRmtMemToken_When_TokenInfoNull_Expect_CCU_E_PTR)
{
    CcuResult ret = HcommCcuGetRmtMemToken(1, 0x1000, nullptr);
    EXPECT_EQ(ret, CCU_E_PTR);
}

// ═══════════════════════════════════════════════════════════
// 4. CcuResPack — AllocCascCntBlock
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuCascCntApiTest, Ut_AllocCascCntBlock_When_ValidDescs_Expect_SuccessAndBlocksAllocated)
{
    auto descHandle = CreateDescDirect(0);
    ASSERT_NE(descHandle, 0U);
    EXPECT_EQ(HcommCcuInsResDescSetNum(descHandle, HCOMM_CCU_RES_TYPE_CASC_CNT, 2), CCU_SUCCESS);

    auto* desc = ResDescMgr(TEST_DEVICE_LOGIC_ID).Get(descHandle);
    ASSERT_NE(desc, nullptr);
    const CcuResDesc* descs[] = {desc};

    CcuResPack resPack;
    resPack.userDevId_ = TEST_DEVICE_LOGIC_ID;
    EXPECT_EQ(resPack.AllocCascCntBlock(descs, 1), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resPack.cascCntBlocks_[0].size(), 2U);

    HcommCcuInsResDescDestroy(descHandle);
}

TEST_F(HcommCcuCascCntApiTest, Ut_AllocCascCntBlock_When_CascCntExceedsMax_Expect_CCU_E_UNAVAIL)
{
    auto descHandle = CreateDescDirect(0);
    ASSERT_NE(descHandle, 0U);
    // CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM = 4, request 5 to exceed limit
    EXPECT_EQ(
        HcommCcuInsResDescSetNum(descHandle, HCOMM_CCU_RES_TYPE_CASC_CNT, CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM + 1),
        CCU_SUCCESS);

    auto* desc = ResDescMgr(TEST_DEVICE_LOGIC_ID).Get(descHandle);
    ASSERT_NE(desc, nullptr);
    const CcuResDesc* descs[] = {desc};

    CcuResPack resPack;
    resPack.userDevId_ = TEST_DEVICE_LOGIC_ID;
    EXPECT_EQ(resPack.AllocCascCntBlock(descs, 1), CcuResult::CCU_E_UNAVAIL);

    HcommCcuInsResDescDestroy(descHandle);
}

TEST_F(HcommCcuCascCntApiTest, Ut_AllocCascCntBlock_When_NullDesc_Expect_SuccessNoAlloc)
{
    const CcuResDesc* descs[] = {nullptr};

    CcuResPack resPack;
    resPack.userDevId_ = TEST_DEVICE_LOGIC_ID;
    EXPECT_EQ(resPack.AllocCascCntBlock(descs, 1), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resPack.cascCntBlocks_[0].size(), 0U);
}

TEST_F(HcommCcuCascCntApiTest, Ut_AllocCascCntBlock_When_AllocFails_Expect_ErrorAndNoBlocks)
{
    // 让第一次（也是唯一一次）alloc 调用失败，走 SetUp 已注册的默认 mock
    g_allocFailOnCall = 0;

    auto descHandle = CreateDescDirect(0);
    ASSERT_NE(descHandle, 0U);
    EXPECT_EQ(HcommCcuInsResDescSetNum(descHandle, HCOMM_CCU_RES_TYPE_CASC_CNT, 1), CCU_SUCCESS);

    auto* desc = ResDescMgr(TEST_DEVICE_LOGIC_ID).Get(descHandle);
    ASSERT_NE(desc, nullptr);
    const CcuResDesc* descs[] = {desc};

    CcuResPack resPack;
    resPack.userDevId_ = TEST_DEVICE_LOGIC_ID;
    EXPECT_EQ(resPack.AllocCascCntBlock(descs, 1), CcuResult::CCU_E_UNAVAIL);
    EXPECT_EQ(resPack.cascCntBlocks_[0].size(), 0U);

    HcommCcuInsResDescDestroy(descHandle);
}

TEST_F(HcommCcuCascCntApiTest, Ut_AllocCascCntBlock_When_Die1FailsAfterDie0Succeeds_Expect_RollbackReleasesDie0Blocks)
{
    g_allocCallSeq = 0;
    g_releaseCallCount = 0;
    // die0 第一次 alloc 成功，die1 第二次 alloc 失败，均走 SetUp 已注册的默认 mock
    g_allocFailOnCall = 1;

    auto descHandle0 = CreateDescDirect(0);
    ASSERT_NE(descHandle0, 0U);
    EXPECT_EQ(HcommCcuInsResDescSetNum(descHandle0, HCOMM_CCU_RES_TYPE_CASC_CNT, 1), CCU_SUCCESS);

    auto descHandle1 = CreateDescDirect(1);
    ASSERT_NE(descHandle1, 0U);
    EXPECT_EQ(HcommCcuInsResDescSetNum(descHandle1, HCOMM_CCU_RES_TYPE_CASC_CNT, 1), CCU_SUCCESS);

    auto* desc0 = ResDescMgr(TEST_DEVICE_LOGIC_ID).Get(descHandle0);
    auto* desc1 = ResDescMgr(TEST_DEVICE_LOGIC_ID).Get(descHandle1);
    ASSERT_NE(desc0, nullptr);
    ASSERT_NE(desc1, nullptr);
    const CcuResDesc* descs[] = {desc0, desc1};

    CcuResPack resPack;
    resPack.userDevId_ = TEST_DEVICE_LOGIC_ID;
    EXPECT_EQ(resPack.AllocCascCntBlock(descs, 2), CcuResult::CCU_E_UNAVAIL);
    EXPECT_EQ(resPack.cascCntBlocks_[0].size(), 1U);

    resPack.CcuReleaseRes();
    EXPECT_EQ(g_releaseCallCount, 1);
    EXPECT_EQ(resPack.cascCntBlocks_[0].size(), 0U);
    EXPECT_EQ(resPack.cascCntBlocks_[1].size(), 0U);

    HcommCcuInsResDescDestroy(descHandle0);
    HcommCcuInsResDescDestroy(descHandle1);
}

// ═══════════════════════════════════════════════════════════
// 5. CcuResPack — AcquireCascCntBlock
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuCascCntApiTest, Ut_AcquireCascCntBlock_When_BlocksAvailable_Expect_SuccessAndBlock)
{
    auto resPack = MakeResPackWithBlocks(0, 2);
    // handle 由上层 CcuInstanceMgr 全局唯一分配后按值传入, resPack 仅按其登记, 不再自增
    HcommCcuCascCntHandle handle = 1;
    EXPECT_EQ(resPack->AcquireCascCntBlock(0, handle), CcuResult::CCU_SUCCESS);
    // 传入句柄后应能按值取回块内容（块已 emplace 进 handleMap_）
    CntXnBlock block{};
    EXPECT_EQ(resPack->GetCascCntBlock(handle, block), CcuResult::CCU_SUCCESS);
    EXPECT_NE(block.wishCntXnsMem.first, nullptr);
    EXPECT_GT(block.wishCntXnsMem.second, 0U);
}

TEST_F(HcommCcuCascCntApiTest, Ut_AcquireCascCntBlock_When_AllBlocksUsed_Expect_CCU_E_UNAVAIL)
{
    auto resPack = MakeResPackWithBlocks(0, 1);
    // 取走唯一的块
    ASSERT_EQ(resPack->AcquireCascCntBlock(0, 1), CcuResult::CCU_SUCCESS);
    // 已无空闲块：失败返回 CCU_E_UNAVAIL（即便传入新的合法句柄）
    EXPECT_EQ(resPack->AcquireCascCntBlock(0, 2), CcuResult::CCU_E_UNAVAIL);
}

TEST_F(HcommCcuCascCntApiTest, Ut_AcquireCascCntBlock_When_NoBlocks_Expect_CCU_E_UNAVAIL)
{
    CcuResPack resPack;
    resPack.userDevId_ = TEST_DEVICE_LOGIC_ID;
    EXPECT_EQ(resPack.AcquireCascCntBlock(0, 1), CcuResult::CCU_E_UNAVAIL);
}

TEST_F(HcommCcuCascCntApiTest, Ut_AcquireCascCntBlock_When_DieIdOutOfRange_Expect_CCU_E_PARA)
{
    auto resPack = MakeResPackWithBlocks(0, 1);
    EXPECT_EQ(resPack->AcquireCascCntBlock(CCU_MAX_IODIE_NUM, 1), CcuResult::CCU_E_PARA);
}

// handle 恒由上层保证非 0(全局唯一从 1 起), resPack 对 0 句柄做防御性拒绝
TEST_F(HcommCcuCascCntApiTest, Ut_AcquireCascCntBlock_When_HandleZero_Expect_CCU_E_PARA)
{
    auto resPack = MakeResPackWithBlocks(0, 1);
    EXPECT_EQ(resPack->AcquireCascCntBlock(0, 0), CcuResult::CCU_E_PARA);
}

// ═══════════════════════════════════════════════════════════
// 6. CcuResPack — GetCascCntBlock
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuCascCntApiTest, Ut_GetCascCntBlock_When_ValidHandle_Expect_ReturnBlock)
{
    auto resPack = MakeResPackWithBlocks(0, 1);
    HcommCcuCascCntHandle handle = 1;
    ASSERT_EQ(resPack->AcquireCascCntBlock(0, handle), CcuResult::CCU_SUCCESS);

    CntXnBlock block{};
    EXPECT_EQ(resPack->GetCascCntBlock(handle, block), CcuResult::CCU_SUCCESS);
    EXPECT_NE(block.wishCntXnsMem.first, nullptr);
    EXPECT_GT(block.wishCntXnsMem.second, 0U);
}

TEST_F(HcommCcuCascCntApiTest, Ut_GetCascCntBlock_When_InvalidHandle_Expect_CCU_E_NOT_FOUND)
{
    auto resPack = MakeResPackWithBlocks(0, 1);
    CntXnBlock block{};
    EXPECT_EQ(resPack->GetCascCntBlock(9999, block), CcuResult::CCU_E_NOT_FOUND);
}

TEST_F(HcommCcuCascCntApiTest, Ut_GetCascCntBlock_When_HandleZero_Expect_CCU_E_NOT_FOUND)
{
    auto resPack = MakeResPackWithBlocks(0, 1);
    CntXnBlock block{};
    // handle 由上层全局唯一分配（从 1 起），resPack 不再自增，0 恒为无效
    EXPECT_EQ(resPack->GetCascCntBlock(0, block), CcuResult::CCU_E_NOT_FOUND);
}

// ═══════════════════════════════════════════════════════════
// 7. CcuResPack — 多个互异句柄各自独立登记
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuCascCntApiTest, Ut_AcquireCascCntBlock_When_MultipleDistinctHandles_Expect_AllRegistered)
{
    auto resPack = MakeResPackWithBlocks(0, 3);
    // 句柄由上层(Mgr)全局唯一分配后按值传入, resPack 按值登记;
    // 三个互异句柄应各自独立存入 handleMap_ 且可分别取回
    ASSERT_EQ(resPack->AcquireCascCntBlock(0, 1), CcuResult::CCU_SUCCESS);
    ASSERT_EQ(resPack->AcquireCascCntBlock(0, 2), CcuResult::CCU_SUCCESS);
    ASSERT_EQ(resPack->AcquireCascCntBlock(0, 3), CcuResult::CCU_SUCCESS);

    EXPECT_EQ(resPack->GetCascCntBlocks().size(), 3U);
    CntXnBlock block{};
    EXPECT_EQ(resPack->GetCascCntBlock(1, block), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resPack->GetCascCntBlock(2, block), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resPack->GetCascCntBlock(3, block), CcuResult::CCU_SUCCESS);
}
