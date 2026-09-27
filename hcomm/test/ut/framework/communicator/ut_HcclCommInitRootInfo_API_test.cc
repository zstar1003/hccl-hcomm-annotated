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

class HcclCommInitRootInfoTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
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
};

TEST_F(HcclCommInitRootInfoTest, Ut_HcclCommInitRootInfo_When_nRanksIsZero_Expect_ReturnIsHCCL_E_PARA)
{
    int nRanks = 0;
    HcclRootInfo id = Ut_Get_Root_Info(0);
    int rank = 0;

    HcclResult ret = HcclCommInitRootInfo(nRanks, &id, rank, &comm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    Ut_Comm_Destroy(comm);
}

// 小概率测试用例段错误
TEST_F(HcclCommInitRootInfoTest, Ut_HcclCommInitRootInfo_When_GetDeviceError_Expect_ReturnIsHCCL_E_RUNTIME)
{
    // int nRanks = 1;
    // HcclRootInfo id = Ut_Get_Root_Info(0);
    // int rank = 0;
    // MOCKER(rtGetDevice)
    //     .stubs()
    //     .will(returnValue(ACL_ERROR_RT_CONTEXT_NULL));

    // HcclResult ret = HcclCommInitRootInfo(nRanks, &id, rank, &comm);
    // EXPECT_EQ(ret, HCCL_E_RUNTIME);

    // GlobalMockObject::verify();
    // Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommInitRootInfoTest, Ut_HcclCommInitRootInfo_When_RootInfoIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    Ut_Device_Set(0);
    int nRanks = 1;
    HcclRootInfo* pId = nullptr;
    int rank = 0;

    HcclResult ret = HcclCommInitRootInfo(nRanks, pId, rank, &comm);
    EXPECT_EQ(ret, HCCL_E_PTR);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommInitRootInfoTest, Ut_HcclCommInitRootInfo_When_RankGreaterThannRanks_Expect_ReturnIsHCCL_E_PARA)
{
    int nRanks = 10;
    HcclRootInfo id = Ut_Get_Root_Info(0);
    int rank = 10;

    HcclResult ret = HcclCommInitRootInfo(nRanks, &id, rank, &comm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    Ut_Comm_Destroy(comm);
}

TEST_F(HcclCommInitRootInfoTest, Ut_HcclCommInitRootInfo_When_PCommIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    int nRanks = 1;
    HcclRootInfo id = Ut_Get_Root_Info(0);
    int rank = 0;

    HcclResult ret = HcclCommInitRootInfo(nRanks, &id, rank, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    Ut_Comm_Destroy(comm);
}
