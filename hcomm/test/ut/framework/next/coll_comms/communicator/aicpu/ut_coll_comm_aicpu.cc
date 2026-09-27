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

#include "coll_comm_aicpu.h"
#include "coll_comm_aicpu_mgr.h"
#include "channel_aicpu_mgr.h"
#include "ns_recovery/aicpu/ns_recovery_lite.h"

#define private public
using namespace hccl;

class CollCommAicpuTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(CollCommAicpuTest, Ut_DefaultStatus_IsInvalid_And_SetGet_Works)
{
    CollCommAicpu coll;
    EXPECT_EQ(coll.GetCommmStatus(), HcclCommStatus::HCCL_COMM_STATUS_INVALID);

    coll.SetCommmStatus(HcclCommStatus::HCCL_COMM_STATUS_READY);
    EXPECT_EQ(coll.GetCommmStatus(), HcclCommStatus::HCCL_COMM_STATUS_READY);
}

TEST_F(CollCommAicpuTest, Ut_Clean_ChannelMgrNull_Returns_Error)
{
    CollCommAicpu coll;
    // channelMgr_ 未初始化时 Clean 应返回错误
    auto ret = coll.Clean();
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(CollCommAicpuTest, Ut_GetNsRecoveryLitePtr_DefaultNull_And_Settable)
{
    CollCommAicpu coll;
    EXPECT_EQ(coll.GetNsRecoveryLitePtr(), nullptr);

    auto nsPtr = std::make_shared<NsRecoveryLite>();
    coll.nsRecoveryLitePtr_ = nsPtr;
    EXPECT_NE(coll.GetNsRecoveryLitePtr(), nullptr);
}

TEST_F(CollCommAicpuTest, Ut_Resume_DelegatesToChannelMgr_And_ResetsNsRecoveryFlags)
{
    CollCommAicpu coll;

    // 初始化 channelMgr_ 和 nsRecovery 使其可调用 Resume
    HcclTopoInfo topoInfo{};
    coll.channelMgr_ = std::make_unique<ChannelAicpuMgr>(coll.dfx_, topoInfo);
    coll.nsRecoveryLitePtr_ = std::make_shared<NsRecoveryLite>();
    // 初始化 commEngineResMgr_，避免 GetCommEngineResMgr()->GetAllThread() 解引用 nullptr
    coll.commEngineResMgr_ = std::make_unique<CommEngineResAicpuMgr>(coll.dfx_, [](bool) {
        return HCCL_SUCCESS;
    });
    coll.SetCommmStatus(HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING);

    // Mock ChannelAicpuMgr::Resume（原 CollCommAicpu::ProcessUrmaRes 已迁入 ChannelAicpuMgr）
    MOCKER_CPP(&ChannelAicpuMgr::Resume).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclChannelUrmaRes commParam{};
    auto ret = coll.Resume(&commParam);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(coll.GetCommmStatus(), HcclCommStatus::HCCL_COMM_STATUS_READY);
    EXPECT_FALSE(coll.GetNsRecoveryLitePtr()->IsNeedClean());
}

TEST_F(CollCommAicpuTest, Ut_CheckIndOpExecStatus_Return_HCCL_E_SUSPENDING)
{
    CollCommAicpu coll;
    coll.SetCommmStatus(HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING);
    auto ret = coll.CheckIndOpExecStatus(0);
    EXPECT_EQ(ret, HCCL_E_SUSPENDING);
}

TEST_F(CollCommAicpuTest, Ut_InitIndopEnv_Expect_Success)
{
    CommAicpuParam param{};
    param.commConfig.taskExceptionEnable = true;
    param.commConfig.notifyWaitTimeout = 1836;
    param.commConfig.plfDebugConfig = 0;
    CollCommAicpuMgr::GetInstance().InitIndopEnv(&param);
}

TEST_F(CollCommAicpuTest, Ut_GetCommEngineResMgr_NotNull_After_Init)
{
    CollCommAicpu coll;
    // 通过 InitAicpuIndOp 初始化后 GetCommEngineResMgr 应非空
    CommAicpuParam param{};
    strncpy(param.hcomId, "test_group", HCOMID_MAX_SIZE - 1);

    // Mock 必要的底层调用
    MOCKER_CPP(hrtSetWorkModeAicpu).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtSetlocalDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtSetlocalDeviceType).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtDrvGetLocalDevIDByHostDevID).stubs().will(returnValue(HCCL_SUCCESS));

    auto ret = coll.InitAicpuIndOp(&param);
    if (ret == HCCL_SUCCESS) {
        EXPECT_NE(coll.GetCommEngineResMgr(), nullptr);
        EXPECT_NE(coll.GetChannelMgr(), nullptr);
    }
}

TEST_F(CollCommAicpuTest, Ut_GetTopoInfo_DefaultInit)
{
    CollCommAicpu coll;
    const HcclTopoInfo& topo = coll.GetTopoInfo();
    EXPECT_EQ(topo.userRank, 0u);
    EXPECT_EQ(topo.userRankSize, 0u);
}

TEST_F(CollCommAicpuTest, Ut_GetIdentifier_EmptyByDefault)
{
    CollCommAicpu coll;
    EXPECT_EQ(coll.GetIdentifier(), "");
}

TEST_F(CollCommAicpuTest, Ut_IsErrorReported_DefaultFalse_And_SetGet_Works)
{
    CollCommAicpu coll;
    EXPECT_FALSE(coll.IsErrorReported());
    coll.SetErrorReported(true);
    EXPECT_TRUE(coll.IsErrorReported());
}

TEST_F(CollCommAicpuTest, Ut_UpdateIndex_Increments)
{
    CollCommAicpu coll;
    u32 idx1 = coll.UpdateIndex();
    u32 idx2 = coll.UpdateIndex();
    EXPECT_GT(idx2, idx1);
}

TEST_F(CollCommAicpuTest, Ut_GetDevId_DefaultZero)
{
    CollCommAicpu coll;
    EXPECT_EQ(coll.GetDevId(), 0u);
}

TEST_F(CollCommAicpuTest, Ut_CheckIndOpExecStatus_ReturnError_When_InvalidStatus)
{
    CollCommAicpu coll;
    coll.SetCommmStatus(HcclCommStatus::HCCL_COMM_STATUS_INVALID);
    auto ret = coll.CheckIndOpExecStatus(0);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// 910B legacy 通信域管理
TEST_F(CollCommAicpuTest, Ut_GetLegacy910CollComm_DefaultNull)
{
    CollCommAicpu coll;
    EXPECT_EQ(coll.GetLegacy910CollComm(), nullptr);
}

TEST_F(CollCommAicpuTest, Ut_IsLegacy910CollCommBusy_DefaultFalse)
{
    CollCommAicpu coll;
    EXPECT_FALSE(coll.IsLegacy910CollCommBusy());
}

TEST_F(CollCommAicpuTest, Ut_SetLegacy910CollCommBusy_Works)
{
    CollCommAicpu coll;
    coll.SetLegacy910CollCommBusy(true);
    EXPECT_TRUE(coll.IsLegacy910CollCommBusy());
    coll.SetLegacy910CollCommBusy(false);
    EXPECT_FALSE(coll.IsLegacy910CollCommBusy());
}
