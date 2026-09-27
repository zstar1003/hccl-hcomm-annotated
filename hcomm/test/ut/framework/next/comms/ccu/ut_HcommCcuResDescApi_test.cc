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
 * UT: HcommCcuResDesc API — 6 public interfaces
 *
 *   HcommCcuInsResDescCreate / Destroy / SetNum / QueryNum / QueryDieId / QueryRemainResDesc
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

#include "mocks/ccu_device_mock_utils.h"

#undef protected
#undef private

using namespace hcomm;

namespace {
constexpr int32_t TEST_DEVICE_LOGIC_ID = 0;
constexpr int32_t OTHER_TEST_DEVICE_LOGIC_ID = 1;
int32_t g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
uint32_t g_deviceRefreshCalls = 0;
HcclResult g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;

HcclResult MockHrtGetDeviceRefresh(int32_t* deviceLogicId)
{
    ++g_deviceRefreshCalls;
    if (g_deviceRefreshResult != HcclResult::HCCL_SUCCESS) {
        return g_deviceRefreshResult;
    }
    if (deviceLogicId == nullptr) {
        return HcclResult::HCCL_E_PTR;
    }
    *deviceLogicId = g_runtimeDeviceLogicId;
    return HcclResult::HCCL_SUCCESS;
}
} // namespace

class HcommCcuResDescApiTest : public BaseInit {
public:
    void SetUp() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        BaseInit::SetUp();
        ResDescMgr(TEST_DEVICE_LOGIC_ID).Deinit();
        ResDescMgr(OTHER_TEST_DEVICE_LOGIC_ID).Deinit();
        g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
        g_deviceRefreshCalls = 0;
        g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;

        MOCKER(hrtGetDeviceRefresh).stubs().with(mockcpp::any()).will(invoke(MockHrtGetDeviceRefresh));
        int32_t seededDeviceLogicId = INVALID_INT;
        ASSERT_EQ(HcclDeviceRefresh(seededDeviceLogicId), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(seededDeviceLogicId, TEST_DEVICE_LOGIC_ID);
        ASSERT_EQ(HcclGetThreadDeviceId(), TEST_DEVICE_LOGIC_ID);
        g_deviceRefreshCalls = 0;

        // 将 enableEntryLog 默认返回 true
        MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));

        // 初始化 CcuResSpecifications (dieId=0 启用, 设置各资源容量)
        constexpr int32_t fakeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
        constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
        MOCKER(hrtGetDevice).stubs().with(outBoundP(&fakeDeviceLogicId)).will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(hrtGetDevicePhyIdByIndex)
            .stubs()
            .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
            .will(returnValue(HcclResult::HCCL_SUCCESS));
    }
    void TearDown() override
    {
        g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
        g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;
        int32_t restoredDeviceLogicId = INVALID_INT;
        EXPECT_EQ(HcclDeviceRefresh(restoredDeviceLogicId), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(restoredDeviceLogicId, TEST_DEVICE_LOGIC_ID);
        ResDescMgr(TEST_DEVICE_LOGIC_ID).Deinit();
        ResDescMgr(OTHER_TEST_DEVICE_LOGIC_ID).Deinit();
        BaseInit::TearDown();
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }

protected:
    static CcuResDescMgr& ResDescMgr(int32_t deviceLogicId)
    {
        return CcuInstanceMgr::GetInstance(deviceLogicId).GetResDescMgr();
    }

    static HcommCcuResDescHandle CreateDescDirect(int32_t deviceLogicId, uint32_t dieId)
    {
        HcommCcuResDescHandle handle = 0;
        EXPECT_EQ(ResDescMgr(deviceLogicId).Create(dieId, handle), CcuResult::CCU_SUCCESS);
        EXPECT_NE(handle, 0U);
        return handle;
    }
};
// ═══════════════════════════════════════════════════════════
// 1. HcommCcuInsResDescCreate
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescCreate_When_DieIdZero_Expect_Success)
{
    HcommCcuResDescHandle handle = 0;
    CcuResult ret = HcommCcuInsResDescCreate(0, &handle);
    EXPECT_EQ(ret, CCU_SUCCESS);
    EXPECT_NE(handle, 0u);
    HcommCcuInsResDescDestroy(handle);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescCreate_When_ResDescNull_Expect_CCU_E_PTR)
{
    CcuResult ret = HcommCcuInsResDescCreate(0, nullptr);
    EXPECT_EQ(ret, CCU_E_PTR);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescCreate_When_DieIdInvalid_Expect_CCU_E_PARA)
{
    HcommCcuResDescHandle handle = 0;
    CcuResult ret = HcommCcuInsResDescCreate(CCU_MAX_IODIE_NUM, &handle);
    EXPECT_EQ(ret, CCU_E_PARA);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescCreate_When_Multiple_Expect_DifferentHandles)
{
    HcommCcuResDescHandle h1 = 0, h2 = 0;
    EXPECT_EQ(HcommCcuInsResDescCreate(0, &h1), CCU_SUCCESS);
    EXPECT_EQ(HcommCcuInsResDescCreate(0, &h2), CCU_SUCCESS);
    EXPECT_NE(h1, h2);
    HcommCcuInsResDescDestroy(h1);
    HcommCcuInsResDescDestroy(h2);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescCreate_When_ManagerInternalError_Expect_CCU_E_INTERNAL)
{
    MOCKER_CPP(&CcuResDescMgr::Create).stubs().will(returnValue(CCU_E_INTERNAL));

    HcommCcuResDescHandle handle = 0;
    EXPECT_EQ(HcommCcuInsResDescCreate(0, &handle), CCU_E_INTERNAL);
    EXPECT_EQ(handle, 0u);
}

// ═══════════════════════════════════════════════════════════
// 2. HcommCcuInsResDescDestroy
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescDestroy_When_ValidHandle_Expect_Success)
{
    HcommCcuResDescHandle handle = 0;
    HcommCcuInsResDescCreate(0, &handle);
    CcuResult ret = HcommCcuInsResDescDestroy(handle);
    EXPECT_EQ(ret, CCU_SUCCESS);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescDestroy_When_Unregistered_Expect_CCU_E_NOT_FOUND)
{
    // 0 是合法 ID 值但未注册
    CcuResult ret = HcommCcuInsResDescDestroy(0);
    EXPECT_EQ(ret, CCU_E_NOT_FOUND);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescDestroy_When_DoubleDestroy_Expect_Error)
{
    HcommCcuResDescHandle handle = 0;
    HcommCcuInsResDescCreate(0, &handle);
    HcommCcuInsResDescDestroy(handle);
    CcuResult ret = HcommCcuInsResDescDestroy(handle);
    EXPECT_EQ(ret, CCU_E_NOT_FOUND);
}

// ═══════════════════════════════════════════════════════════
// 3. HcommCcuInsResDescSetNum
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescSetNum_When_AllTypes_Expect_Success)
{
    HcommCcuResDescHandle handle = 0;
    HcommCcuInsResDescCreate(0, &handle);

    struct {
        HcommCcuResType type;
        uint32_t val;
    } cases[] = {
        {HCOMM_CCU_RES_TYPE_LOOP, 8},         {HCOMM_CCU_RES_TYPE_CCU_BUF, 16}, {HCOMM_CCU_RES_TYPE_VARIABLE, 24},
        {HCOMM_CCU_RES_TYPE_ADDRESS, 32},     {HCOMM_CCU_RES_TYPE_EVENT, 40},   {HCOMM_CCU_RES_TYPE_CCU_THREAD, 48},
        {HCOMM_CCU_RES_TYPE_INSTRUCTION, 56},
    };
    for (auto& c : cases) {
        EXPECT_EQ(HcommCcuInsResDescSetNum(handle, c.type, c.val), CCU_SUCCESS);
    }
    for (auto& c : cases) {
        uint32_t num = 0;
        EXPECT_EQ(HcommCcuInsResDescQueryNum(handle, c.type, &num), CCU_SUCCESS);
        EXPECT_EQ(num, c.val);
    }
    HcommCcuInsResDescDestroy(handle);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescSetNum_When_ResNumZero_Expect_Success)
{
    HcommCcuResDescHandle handle = 0;
    HcommCcuInsResDescCreate(0, &handle);
    EXPECT_EQ(HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_LOOP, 100), CCU_SUCCESS);
    EXPECT_EQ(HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_LOOP, 0), CCU_SUCCESS);

    uint32_t num = 999;
    HcommCcuInsResDescQueryNum(handle, HCOMM_CCU_RES_TYPE_LOOP, &num);
    EXPECT_EQ(num, 0u);
    HcommCcuInsResDescDestroy(handle);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescSetNum_When_InvalidType_Expect_CCU_E_PARA)
{
    HcommCcuResDescHandle handle = 0;
    HcommCcuInsResDescCreate(0, &handle);
    CcuResult ret = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_INVALID, 10);
    EXPECT_EQ(ret, CCU_E_PARA);
    HcommCcuInsResDescDestroy(handle);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescSetNum_When_HandleUnregistered_Expect_CCU_E_NOT_FOUND)
{
    CcuResult ret = HcommCcuInsResDescSetNum(9999, HCOMM_CCU_RES_TYPE_LOOP, 10);
    EXPECT_EQ(ret, CCU_E_NOT_FOUND);
}

// ═══════════════════════════════════════════════════════════
// 4. HcommCcuInsResDescQueryNum
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescQueryNum_When_NeverSet_Expect_Zero)
{
    HcommCcuResDescHandle handle = 0;
    HcommCcuInsResDescCreate(0, &handle);
    uint32_t num = 999;
    EXPECT_EQ(HcommCcuInsResDescQueryNum(handle, HCOMM_CCU_RES_TYPE_LOOP, &num), CCU_SUCCESS);
    EXPECT_EQ(num, 0u);
    HcommCcuInsResDescDestroy(handle);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescQueryNum_When_NumNull_Expect_CCU_E_PTR)
{
    HcommCcuResDescHandle handle = 0;
    HcommCcuInsResDescCreate(0, &handle);
    CcuResult ret = HcommCcuInsResDescQueryNum(handle, HCOMM_CCU_RES_TYPE_LOOP, nullptr);
    EXPECT_EQ(ret, CCU_E_PTR);
    HcommCcuInsResDescDestroy(handle);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescQueryNum_When_HandleUnregistered_Expect_CCU_E_NOT_FOUND)
{
    uint32_t num = 0;
    CcuResult ret = HcommCcuInsResDescQueryNum(0, HCOMM_CCU_RES_TYPE_LOOP, &num);
    EXPECT_EQ(ret, CCU_E_NOT_FOUND);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescQueryNum_When_ResTypeInvalid_Expect_CCU_E_PARA)
{
    HcommCcuResDescHandle handle = 0;
    HcommCcuInsResDescCreate(0, &handle);
    uint32_t num = 0;
    CcuResult ret = HcommCcuInsResDescQueryNum(handle, HCOMM_CCU_RES_TYPE_INVALID, &num);
    EXPECT_EQ(ret, CCU_E_PARA);
    HcommCcuInsResDescDestroy(handle);
}

// ═══════════════════════════════════════════════════════════
// 5. HcommCcuInsResDescQueryDieId
// ═══════════════════════════════════════════════════════════

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescQueryDieId_When_DieIdZero_Expect_Zero)
{
    HcommCcuResDescHandle handle = 0;
    HcommCcuInsResDescCreate(0, &handle);
    uint32_t dieId = 999;
    EXPECT_EQ(HcommCcuInsResDescQueryDieId(handle, &dieId), CCU_SUCCESS);
    EXPECT_EQ(dieId, 0u);
    HcommCcuInsResDescDestroy(handle);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescQueryDieId_When_DieIdOne_Expect_One)
{
    HcommCcuResDescHandle handle = 0;
    HcommCcuInsResDescCreate(1, &handle);
    uint32_t dieId = 999;
    EXPECT_EQ(HcommCcuInsResDescQueryDieId(handle, &dieId), CCU_SUCCESS);
    EXPECT_EQ(dieId, 1u);
    HcommCcuInsResDescDestroy(handle);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescQueryDieId_When_DieIdNull_Expect_CCU_E_PTR)
{
    HcommCcuResDescHandle handle = 0;
    HcommCcuInsResDescCreate(0, &handle);
    CcuResult ret = HcommCcuInsResDescQueryDieId(handle, nullptr);
    EXPECT_EQ(ret, CCU_E_PTR);
    HcommCcuInsResDescDestroy(handle);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuInsResDescQueryDieId_When_HandleUnregistered_Expect_CCU_E_NOT_FOUND)
{
    uint32_t dieId = 0;
    CcuResult ret = HcommCcuInsResDescQueryDieId(0, &dieId);
    EXPECT_EQ(ret, CCU_E_NOT_FOUND);
}

// ═══════════════════════════════════════════════════════════
// 6. HcommCcuQueryRemainResDesc
// ═══════════════════════════════════════════════════════════
TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuQueryRemainResDesc_When_NoAllocations_Expect_ResNumFilled)
{
    MOCKER(CcuIsInited).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER(CcuGetDieEnableInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(true))
        .will(returnValue(CCU_SUCCESS));
    MOCKER_CPP(&CcuDevMgrImp::QueryRemainRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcommCcuResDescHandle handle = 0;
    HcommCcuInsResDescCreate(0, &handle);

    CcuResult ret = HcommCcuQueryRemainResDesc(handle);
    EXPECT_EQ(ret, CCU_SUCCESS);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuQueryRemainResDesc_When_HandleUnregistered_Expect_CCU_E_NOT_FOUND)
{
    MOCKER(CcuIsInited).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER(CcuGetDieEnableInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(true))
        .will(returnValue(CCU_SUCCESS));
    CcuResult ret = HcommCcuQueryRemainResDesc(9999);
    EXPECT_EQ(ret, CCU_E_NOT_FOUND);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuQueryRemainResDesc_When_DieEnableFlagFalse_Expect_CCU_E_UNAVAIL)
{
    MOCKER(CcuIsInited).stubs().with(mockcpp::any()).will(returnValue(true));
    bool enableFlag = false;
    MOCKER(CcuGetDieEnableInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(enableFlag))
        .will(returnValue(CCU_SUCCESS));
    HcommCcuResDescHandle handle = 0;
    HcommCcuInsResDescCreate(0, &handle);
    CcuResult ret = HcommCcuQueryRemainResDesc(handle);
    EXPECT_EQ(ret, CCU_E_UNAVAIL);
    HcommCcuInsResDescDestroy(handle);
}

TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuQueryRemainResDesc_When_CcuNotInit_Expect_CCU_E_UNAVAIL)
{
    MOCKER(CcuIsInited).stubs().with(mockcpp::any()).will(returnValue(false));
    HcommCcuResDescHandle handle = 0;
    HcommCcuInsResDescCreate(0, &handle);
    CcuResult ret = HcommCcuQueryRemainResDesc(handle);
    EXPECT_EQ(ret, CCU_E_UNAVAIL);
    HcommCcuInsResDescDestroy(handle);
}

// 验证运行时 Device 切换后，资源描述符接口均刷新并操作新 Device 的管理器。
TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuResDescApis_When_RuntimeDeviceSwitches_Expect_OperateOnRefreshedDevice)
{
    MOCKER(CcuIsInited).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER(CcuGetDieEnableInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(true))
        .will(returnValue(CCU_SUCCESS));
    MOCKER_CPP(&CcuDevMgrImp::QueryRemainRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    g_runtimeDeviceLogicId = OTHER_TEST_DEVICE_LOGIC_ID;
    HcommCcuResDescHandle handle = 0;
    EXPECT_EQ(HcommCcuInsResDescCreate(0, &handle), CCU_SUCCESS);
    ASSERT_NE(handle, 0U);
    EXPECT_EQ(ResDescMgr(TEST_DEVICE_LOGIC_ID).Get(handle), nullptr);
    EXPECT_NE(ResDescMgr(OTHER_TEST_DEVICE_LOGIC_ID).Get(handle), nullptr);

    EXPECT_EQ(HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_LOOP, 17), CCU_SUCCESS);
    uint32_t resNum = 0;
    EXPECT_EQ(HcommCcuInsResDescQueryNum(handle, HCOMM_CCU_RES_TYPE_LOOP, &resNum), CCU_SUCCESS);
    EXPECT_EQ(resNum, 17U);
    uint32_t dieId = hcomm::CCU_MAX_IODIE_NUM;
    EXPECT_EQ(HcommCcuInsResDescQueryDieId(handle, &dieId), CCU_SUCCESS);
    EXPECT_EQ(dieId, 0U);
    EXPECT_EQ(HcommCcuQueryRemainResDesc(handle), CCU_SUCCESS);
    EXPECT_EQ(HcommCcuInsResDescDestroy(handle), CCU_SUCCESS);
    EXPECT_EQ(ResDescMgr(OTHER_TEST_DEVICE_LOGIC_ID).Get(handle), nullptr);
    EXPECT_EQ(g_deviceRefreshCalls, 6U);
}

// 验证运行时 Device 切走时资源描述符操作被拒绝，切回后可继续操作原对象。
TEST_F(
    HcommCcuResDescApiTest, Ut_HcommCcuResDescApis_When_RuntimeDeviceSwitchesAwayAndBack_Expect_RejectAwayThenRecover)
{
    MOCKER(CcuIsInited).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER(CcuGetDieEnableInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(true))
        .will(returnValue(CCU_SUCCESS));
    MOCKER_CPP(&CcuDevMgrImp::QueryRemainRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    const auto handle = CreateDescDirect(TEST_DEVICE_LOGIC_ID, 0);
    EXPECT_EQ(ResDescMgr(TEST_DEVICE_LOGIC_ID).SetResNum(handle, Hccl::ResType::LOOP, 23), CCU_SUCCESS);

    g_runtimeDeviceLogicId = OTHER_TEST_DEVICE_LOGIC_ID;
    uint32_t resNum = 999;
    uint32_t dieId = 999;
    EXPECT_EQ(HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_LOOP, 31), CCU_E_NOT_FOUND);
    EXPECT_EQ(HcommCcuInsResDescQueryNum(handle, HCOMM_CCU_RES_TYPE_LOOP, &resNum), CCU_E_NOT_FOUND);
    EXPECT_EQ(resNum, 999U);
    EXPECT_EQ(HcommCcuInsResDescQueryDieId(handle, &dieId), CCU_E_NOT_FOUND);
    EXPECT_EQ(dieId, 999U);
    EXPECT_EQ(HcommCcuQueryRemainResDesc(handle), CCU_E_NOT_FOUND);
    EXPECT_EQ(HcommCcuInsResDescDestroy(handle), CCU_E_NOT_FOUND);
    EXPECT_NE(ResDescMgr(TEST_DEVICE_LOGIC_ID).Get(handle), nullptr);

    g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
    EXPECT_EQ(HcommCcuInsResDescQueryNum(handle, HCOMM_CCU_RES_TYPE_LOOP, &resNum), CCU_SUCCESS);
    EXPECT_EQ(resNum, 23U);
    EXPECT_EQ(HcommCcuInsResDescQueryDieId(handle, &dieId), CCU_SUCCESS);
    EXPECT_EQ(dieId, 0U);
    EXPECT_EQ(HcommCcuQueryRemainResDesc(handle), CCU_SUCCESS);
    EXPECT_EQ(HcommCcuInsResDescDestroy(handle), CCU_SUCCESS);
    EXPECT_EQ(g_deviceRefreshCalls, 9U);
}

// 验证 Device 刷新失败时，所有资源描述符接口返回错误且不改变管理器状态。
TEST_F(HcommCcuResDescApiTest, Ut_HcommCcuResDescApis_When_DeviceRefreshFails_Expect_ReturnErrorWithoutMutation)
{
    MOCKER(CcuGetDieEnableInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(true))
        .will(returnValue(CCU_SUCCESS));
    MOCKER_CPP(&CcuDevMgrImp::QueryRemainRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    const auto destroyHandle = CreateDescDirect(TEST_DEVICE_LOGIC_ID, 0);
    const auto setHandle = CreateDescDirect(TEST_DEVICE_LOGIC_ID, 0);
    const auto queryNumHandle = CreateDescDirect(TEST_DEVICE_LOGIC_ID, 0);
    const auto queryDieHandle = CreateDescDirect(TEST_DEVICE_LOGIC_ID, 0);
    const auto remainHandle = CreateDescDirect(TEST_DEVICE_LOGIC_ID, 0);
    EXPECT_EQ(ResDescMgr(TEST_DEVICE_LOGIC_ID).SetResNum(setHandle, Hccl::ResType::LOOP, 41), CCU_SUCCESS);

    g_deviceRefreshResult = HcclResult::HCCL_E_INTERNAL;
    HcommCcuResDescHandle createdHandle = 0x1234;
    uint32_t resNum = 0x5678;
    uint32_t dieId = 0x9ABC;
    EXPECT_EQ(HcommCcuInsResDescCreate(0, &createdHandle), CCU_E_INTERNAL);
    EXPECT_EQ(createdHandle, 0x1234U);
    EXPECT_EQ(HcommCcuInsResDescDestroy(destroyHandle), CCU_E_INTERNAL);
    EXPECT_EQ(HcommCcuInsResDescSetNum(setHandle, HCOMM_CCU_RES_TYPE_LOOP, 99), CCU_E_INTERNAL);
    EXPECT_EQ(HcommCcuInsResDescQueryNum(queryNumHandle, HCOMM_CCU_RES_TYPE_LOOP, &resNum), CCU_E_INTERNAL);
    EXPECT_EQ(resNum, 0x5678U);
    EXPECT_EQ(HcommCcuInsResDescQueryDieId(queryDieHandle, &dieId), CCU_E_INTERNAL);
    EXPECT_EQ(dieId, 0x9ABCU);
    EXPECT_EQ(HcommCcuQueryRemainResDesc(remainHandle), CCU_E_INTERNAL);

    EXPECT_NE(ResDescMgr(TEST_DEVICE_LOGIC_ID).Get(destroyHandle), nullptr);
    uint32_t storedNum = 0;
    EXPECT_EQ(ResDescMgr(TEST_DEVICE_LOGIC_ID).QueryResNum(setHandle, Hccl::ResType::LOOP, storedNum), CCU_SUCCESS);
    EXPECT_EQ(storedNum, 41U);
    EXPECT_EQ(g_deviceRefreshCalls, 6U);
}
