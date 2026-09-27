/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"
#include "coll_comm_mgr.h"
extern thread_local s32 g_hcclDeviceId;

class HcclCommInitAllTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        devs = 8;
        // 将建链超时时间设置为1s，减少测试用例运行时间
        MOCKER(GetExternalInputHcclLinkTimeOut).stubs().with(mockcpp::any()).will(returnValue(1));
    }
    void TearDown() override
    {
        // 删除所有拓扑建链的线程
        HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
        opBaseInfo.hcclCommTopoInfoDetectServer.clear();
        opBaseInfo.hcclCommTopoInfoDetectAgent.clear();

        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

protected:
    int devs;
};

TEST_F(HcclCommInitAllTest, Ut_HcclCommInitAll_When_NDevIsZero_Expect_ReturnIsHCCL_E_PARA)
{
    const uint32_t ndev = 0;
    int32_t devices[devs] = {0, 1, 2, 3, 4, 5, 6, 7};
    HcclComm comms[devs] = {};

    for (int i = 0; i < devs; i++) {
        HcclResult ret = hrtSetDevice(devices[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    HcclResult ret = HcclCommInitAll(ndev, devices, comms);
    EXPECT_EQ(ret, HCCL_E_PARA);

    for (int i = 0; i < devs; i++) {
        HcclResult ret = hrtResetDevice(devices[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        Ut_Comm_Destroy(comms[i]);
    }
}

TEST_F(HcclCommInitAllTest, Ut_HcclCommInitAll_When_HasDuplicateID_Expect_ReturnIsHCCL_E_PARA)
{
    const uint32_t ndev = 0;
    int32_t devices[devs] = {0, 1, 2, 3, 4, 3, 6, 7};
    HcclComm comms[devs] = {};

    for (int i = 0; i < devs; i++) {
        HcclResult ret = hrtSetDevice(devices[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    HcclResult ret = HcclCommInitAll(ndev, devices, comms);
    EXPECT_EQ(ret, HCCL_E_PARA);

    for (int i = 0; i < devs; i++) {
        HcclResult ret = hrtResetDevice(devices[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        Ut_Comm_Destroy(comms[i]);
    }
}

TEST_F(HcclCommInitAllTest, Ut_HcclCommInitAll_When_DevicesIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    const uint32_t ndev = 8;
    int32_t* pDevices = nullptr;
    HcclComm comms[devs] = {};

    HcclResult ret = HcclCommInitAll(ndev, pDevices, comms);
    EXPECT_EQ(ret, HCCL_E_PTR);

    for (int i = 0; i < devs; i++) {
        Ut_Comm_Destroy(comms[i]);
    }
}

TEST_F(HcclCommInitAllTest, Ut_HcclCommInitAll_When_CommsIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    const uint32_t ndev = 8;
    int32_t devices[devs] = {0, 1, 2, 3, 4, 5, 6, 7};
    HcclComm* pComms = nullptr;

    for (int i = 0; i < devs; i++) {
        HcclResult ret = hrtSetDevice(devices[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    HcclResult ret = HcclCommInitAll(ndev, devices, pComms);
    EXPECT_EQ(ret, HCCL_E_PTR);

    for (int i = 0; i < devs; i++) {
        HcclResult ret = hrtResetDevice(devices[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
}

TEST_F(HcclCommInitAllTest, Ut_HcclCommInitAll_When_2Server4Rank_Expect_ReturnHCCL_SUCCESS) {}
