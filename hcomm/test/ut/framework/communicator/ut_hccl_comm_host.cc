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
#include <mockcpp/mockcpp.hpp>

#include "hccl_api_base_test.h"
#include "../../stub/llt_hccl_stub_rank_graph.h"
#include "launch_aicpu.h"
#include "adapter_rts.h"

using namespace hccl;

class HcclCommHostTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

// Resume when communicator is V2 -> should call CollComm::Resume
TEST_F(HcclCommHostTest, Ut_ResumeWhenIsCommunicatorV2ExpectSuccess)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer{};
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};

    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, commName);

    // Mock CollComm::Init and GetHDCommunicate so InitCollComm succeeds and collComm_ is created
    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(HCCL_SUCCESS));
    // Stub CollComm::Resume to return success
    MOCKER_CPP(&CollComm::Resume).stubs().will(returnValue(HCCL_SUCCESS));

    HcclCommConfig config{};
    unsetenv("HCCL_DFS_CONFIG");
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // Set devType to V2 and call Resume
    hcclCommPtr->devType_ = DevType::DEV_TYPE_950;
    ret = hcclCommPtr->Resume();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommHostTest, Ut_ResumeWhenIsCommunicatorV2AndCollResumeFailsExpectError)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer{};
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};

    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, commName);

    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(HCCL_SUCCESS));
    // Fail resume
    MOCKER_CPP(&CollComm::Resume).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclCommConfig config{};
    unsetenv("HCCL_DFS_CONFIG");
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclCommPtr->devType_ = DevType::DEV_TYPE_950;
    ret = hcclCommPtr->Resume();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcclCommHostTest, Ut_GetCommStatusWhenIsCommunicatorV2ExpectCollStatus)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer{};
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};

    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, commName);

    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(HCCL_SUCCESS));
    // stub GetCommStatus to return SUSPENDING
    MOCKER_CPP(&CollComm::GetCommStatus).stubs().will(returnValue(HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING));

    HcclCommConfig config{};
    unsetenv("HCCL_DFS_CONFIG");
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclCommPtr->devType_ = DevType::DEV_TYPE_950;
    HcclCommStatus status = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    ret = hcclCommPtr->GetCommStatus(status);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(status, HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING);
}

TEST_F(HcclCommHostTest, Ut_GetCommStatusWhenIsCommunicatorV1ExpectReturnE_NOT_SUPPORT)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>();
    hcclCommPtr->devType_ = DevType::DEV_TYPE_910_93;

    HcclCommStatus status = HcclCommStatus::HCCL_COMM_STATUS_INVALID;
    HcclResult ret = hcclCommPtr->GetCommStatus(status);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcclCommHostTest, Ut_InitCollCommInner_When_Success_Expect_Success)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, "test_comm");

    MOCKER_CPP(&HcclCommunicator::GetConnectMode).stubs().will(returnValue(1));
    MOCKER_CPP(&HcclCommunicator::GetRankGraphV1).stubs().will(returnValue(reinterpret_cast<void*>(0x1000)));
    MOCKER_CPP(&HcclCommunicator::GetInCCLbuffer).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::CreateCommCCLbuffer).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(HCCL_SUCCESS));

    u32 userRank = 0;
    HcclResult ret = hcclCommPtr->InitCollCommInner(userRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommHostTest, Ut_BinaryUnLoad_When_BinHandleNotNullAndUnloadFailed_Expect_WarningLog)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, "test_comm");

    hcclCommPtr->binHandle_ = reinterpret_cast<aclrtBinHandle>(0x1234);

    MOCKER(aclrtBinaryUnLoad).stubs().will(returnValue(1));

    hcclCommPtr->BinaryUnLoad();

    EXPECT_EQ(hcclCommPtr->binHandle_, nullptr);
}

TEST_F(HcclCommHostTest, Ut_BinaryUnLoad_When_BinHandleNull_Expect_Noop)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, "test_comm");

    hcclCommPtr->binHandle_ = nullptr;

    hcclCommPtr->BinaryUnLoad();

    EXPECT_EQ(hcclCommPtr->binHandle_, nullptr);
}

TEST_F(HcclCommHostTest, Ut_BinaryUnLoad_When_BinHandleNotNullAndUnloadSuccess_Expect_BinHandleSetNull)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, "test_comm");

    hcclCommPtr->binHandle_ = reinterpret_cast<aclrtBinHandle>(0x1234);

    MOCKER(aclrtBinaryUnLoad).stubs().will(returnValue(0));

    hcclCommPtr->BinaryUnLoad();

    EXPECT_EQ(hcclCommPtr->binHandle_, nullptr);
}

TEST_F(HcclCommHostTest, Ut_KernelLaunchAicpuCommInit_When_Normal_Expect_ReachesStreamSynchronize)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, "test_comm");
    hcclCommPtr->devType_ = DevType::DEV_TYPE_950;

    // Stream 构造: hrtStreamCreateWithFlags 失败使 stream_ 保持为空，避免析构时销毁真实资源
    MOCKER(hrtStreamCreateWithFlags).stubs().will(returnValue(static_cast<HcclResult>(HCCL_E_RUNTIME)));
    MOCKER(hrtGetHcclV2Support).stubs().will(returnValue(HCCL_SUCCESS));
    // KernelLaunchAicpuCommInit 主路径 mock
    MOCKER(hrtStreamSetMode).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hccl::AicpuAclKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));

    // 覆盖 line 479 (s32 timeout = 1836) 与 line 488 (hcclStreamSynchronize)；
    // collComm_ 为空时 line 490 CHK_PTR_NULL 返回 HCCL_E_PTR
    HcclResult ret = hcclCommPtr->KernelLaunchAicpuCommInit();
    EXPECT_EQ(ret, HCCL_E_PTR);
}
