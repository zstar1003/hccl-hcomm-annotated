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

#include "ns_recovery/ns_recovery.h"
#include "hccl_common.h"
#include "channel_process.h"
#include "acl/acl.h"

using namespace hccl;
using namespace hcomm;

class NsRecoveryProcessorTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    static void TearDownTestCase()
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    void SetUp() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
        processor_ = std::make_unique<NsRecoveryProcessor>();
        mockKfcControl_ = std::make_shared<HDCommunicate>();
        mockKfcStatus_ = std::make_shared<HDCommunicate>();
        processor_->SetKfcControlTransfer(mockKfcControl_, mockKfcStatus_);
    }
    void TearDown() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }

    void StubKfcControlPut(const HcclResult ret)
    {
        MOCKER_CPP(&hccl::HDCommunicate::Put)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(ret));
    }
    void StubKfcStatusGet(const HcclResult ret, const Hccl::KfcStatus status)
    {
        stubKfcStatus_ = status;
        stubKfcRet_ = ret;
        MOCKER_CPP(&hccl::HDCommunicate::Get)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(invoke(GetFixedStatusStub));
    }
    void StubChannelUpdateKernelLaunch(const HcclResult ret)
    {
        MOCKER_CPP(&ChannelProcess::ChannelUpdateKernelLaunch)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(ret));
    }

    static HcclResult GetFixedStatusStub(hccl::HDCommunicate* self, u32 offset, u32 size, uint8_t* data)
    {
        auto* opInfo = reinterpret_cast<Hccl::KfcExecStatus*>(data);
        opInfo->kfcStatus = stubKfcStatus_;
        return stubKfcRet_;
    }

    static HcclResult GetStatusStub(hccl::HDCommunicate* self, u32 offset, u32 size, uint8_t* data)
    {
        getCallCount_++;
        auto* opInfo = reinterpret_cast<Hccl::KfcExecStatus*>(data);
        if (getCallCount_ == 1) {
            opInfo->kfcStatus = Hccl::KfcStatus::STOP_LAUNCH_DONE;
        } else {
            opInfo->kfcStatus = Hccl::KfcStatus::CLEAN_DONE;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    std::unique_ptr<NsRecoveryProcessor> processor_;
    std::shared_ptr<HDCommunicate> mockKfcControl_;
    std::shared_ptr<HDCommunicate> mockKfcStatus_;
    static int getCallCount_;
    static Hccl::KfcStatus stubKfcStatus_;
    static HcclResult stubKfcRet_;
};

int NsRecoveryProcessorTest::getCallCount_ = 0;
Hccl::KfcStatus NsRecoveryProcessorTest::stubKfcStatus_ = Hccl::KfcStatus::NONE;
HcclResult NsRecoveryProcessorTest::stubKfcRet_ = HcclResult::HCCL_SUCCESS;

// Test SetKfcControlTransfer with valid pointers
TEST_F(NsRecoveryProcessorTest, Ut_SetKfcControlTransfer_When_ValidPointers_Expect_Success)
{
    auto kfcControl = std::make_shared<HDCommunicate>();
    auto kfcStatus = std::make_shared<HDCommunicate>();
    processor_->SetKfcControlTransfer(kfcControl, kfcStatus);
    // No exception expected
    SUCCEED();
}

// Test SetKfcControlTransfer with nullptr for kfcControlTransferH2D
TEST_F(NsRecoveryProcessorTest, Ut_SetKfcControlTransfer_When_KfcControlNull_Expect_Success)
{
    auto kfcStatus = std::make_shared<HDCommunicate>();
    processor_->SetKfcControlTransfer(nullptr, kfcStatus);
    // No exception expected
    SUCCEED();
}

// Test SetKfcControlTransfer with nullptr for kfcStatusTransferD2H
TEST_F(NsRecoveryProcessorTest, Ut_SetKfcControlTransfer_When_KfcStatusNull_Expect_Success)
{
    auto kfcControl = std::make_shared<HDCommunicate>();
    processor_->SetKfcControlTransfer(kfcControl, nullptr);
    // No exception expected
    SUCCEED();
}

// Test SetKfcControlTransfer with both nullptr
TEST_F(NsRecoveryProcessorTest, Ut_SetKfcControlTransfer_When_BothNull_Expect_Success)
{
    processor_->SetKfcControlTransfer(nullptr, nullptr);
    // No exception expected
    SUCCEED();
}

// Test AddNsRecoveryData with COMM_ENGINE_AICPU and channelNum 0
TEST_F(NsRecoveryProcessorTest, Ut_AddNsRecoveryData_When_AICPU_ChannelNum0_Expect_Success)
{
    ChannelHandle deviceHandles[0];
    ChannelHandle hostHandles[0];
    processor_->AddNsRecoveryData(COMM_ENGINE_AICPU, deviceHandles, hostHandles, 0, "test_tag");
    // No exception expected
    SUCCEED();
}

// Test AddNsRecoveryData with COMM_ENGINE_AICPU and channelNum 1
TEST_F(NsRecoveryProcessorTest, Ut_AddNsRecoveryData_When_AICPU_ChannelNum1_Expect_Success)
{
    ChannelHandle deviceHandles[1];
    ChannelHandle hostHandles[1];
    processor_->AddNsRecoveryData(COMM_ENGINE_AICPU, deviceHandles, hostHandles, 1, "test_tag");
    // No exception expected
    SUCCEED();
}

// Test AddNsRecoveryData with COMM_ENGINE_AICPU_TS and multiple channels
TEST_F(NsRecoveryProcessorTest, Ut_AddNsRecoveryData_When_AICPUTS_MultipleChannels_Expect_Success)
{
    ChannelHandle deviceHandles[3];
    ChannelHandle hostHandles[3];
    processor_->AddNsRecoveryData(COMM_ENGINE_AICPU_TS, deviceHandles, hostHandles, 3, "test_tag");
    // No exception expected
    SUCCEED();
}

// Test AddNsRecoveryData with other engine type
TEST_F(NsRecoveryProcessorTest, Ut_AddNsRecoveryData_When_OtherEngine_Expect_Success)
{
    ChannelHandle deviceHandles[1];
    ChannelHandle hostHandles[1];
    processor_->AddNsRecoveryData(COMM_ENGINE_CCU, deviceHandles, hostHandles, 1, "test_tag");
    // No exception expected
    SUCCEED();
}

// Test StopLaunch with no AICPU engine
TEST_F(NsRecoveryProcessorTest, Ut_StopLaunch_When_NoAICPU_Expect_Success)
{
    // Add non-AICPU recovery data
    ChannelHandle deviceHandles[1];
    ChannelHandle hostHandles[1];
    processor_->AddNsRecoveryData(COMM_ENGINE_CCU, deviceHandles, hostHandles, 1, "test_tag");

    auto ret = processor_->StopLaunch();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// Test Clean with AICPU engine and ERROR status
TEST_F(NsRecoveryProcessorTest, Ut_Clean_When_AICPU_ErrorStatus_Expect_InternalError)
{
    // Add AICPU recovery data
    ChannelHandle deviceHandles[1];
    ChannelHandle hostHandles[1];
    processor_->AddNsRecoveryData(COMM_ENGINE_AICPU, deviceHandles, hostHandles, 1, "test_tag");

    // Stub HDCommunicate methods
    StubKfcStatusGet(HcclResult::HCCL_SUCCESS, Hccl::KfcStatus::ERROR);

    auto ret = processor_->Clean();
    EXPECT_EQ(ret, HcclResult::HCCL_E_INTERNAL);
}

// Test Clean with AICPU engine and other status
TEST_F(NsRecoveryProcessorTest, Ut_Clean_When_AICPU_OtherStatus_Expect_InternalError)
{
    // Add AICPU recovery data
    ChannelHandle deviceHandles[1];
    ChannelHandle hostHandles[1];
    processor_->AddNsRecoveryData(COMM_ENGINE_AICPU, deviceHandles, hostHandles, 1, "test_tag");

    // Stub HDCommunicate methods
    StubKfcStatusGet(HcclResult::HCCL_SUCCESS, Hccl::KfcStatus::NONE);

    auto ret = processor_->Clean();
    EXPECT_EQ(ret, HcclResult::HCCL_E_INTERNAL);
}

// Test Clean with no AICPU engine
TEST_F(NsRecoveryProcessorTest, Ut_Clean_When_NoAICPU_Expect_Success)
{
    // Add non-AICPU recovery data
    ChannelHandle deviceHandles[1];
    ChannelHandle hostHandles[1];
    processor_->AddNsRecoveryData(COMM_ENGINE_CCU, deviceHandles, hostHandles, 1, "test_tag");

    auto ret = processor_->Clean();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// Test Resume with AICPU engine and ChannelUpdateKernelLaunch succeeds
TEST_F(NsRecoveryProcessorTest, Ut_Resume_When_AICPU_ChannelUpdateSucceeds_Expect_Success)
{
    // Add AICPU recovery data
    ChannelHandle deviceHandles[1];
    ChannelHandle hostHandles[1];
    processor_->AddNsRecoveryData(COMM_ENGINE_AICPU, deviceHandles, hostHandles, 1, "test_tag");

    // Stub ChannelUpdateKernelLaunch
    StubChannelUpdateKernelLaunch(HcclResult::HCCL_SUCCESS);

    aclrtBinHandle binHandle = nullptr;
    auto ret = processor_->Resume(binHandle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// Test Resume with AICPU engine and ChannelUpdateKernelLaunch fails
TEST_F(NsRecoveryProcessorTest, Ut_Resume_When_AICPU_ChannelUpdateFails_Expect_Failure)
{
    // Add AICPU recovery data
    ChannelHandle deviceHandles[1];
    ChannelHandle hostHandles[1];
    processor_->AddNsRecoveryData(COMM_ENGINE_AICPU, deviceHandles, hostHandles, 1, "test_tag");

    // Stub ChannelUpdateKernelLaunch
    StubChannelUpdateKernelLaunch(HcclResult::HCCL_E_INTERNAL);

    aclrtBinHandle binHandle = nullptr;
    auto ret = processor_->Resume(binHandle);
    EXPECT_EQ(ret, HcclResult::HCCL_E_INTERNAL);
}

// Test Resume with no AICPU engine
TEST_F(NsRecoveryProcessorTest, Ut_Resume_When_NoAICPU_Expect_Success)
{
    // Add non-AICPU recovery data
    ChannelHandle deviceHandles[1];
    ChannelHandle hostHandles[1];
    processor_->AddNsRecoveryData(COMM_ENGINE_CCU, deviceHandles, hostHandles, 1, "test_tag");

    aclrtBinHandle binHandle = nullptr;
    auto ret = processor_->Resume(binHandle);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// Test StopLaunch with AICPU engine - success path
TEST_F(NsRecoveryProcessorTest, Ut_StopLaunch_When_AICPU_Expect_Success)
{
    ChannelHandle deviceHandles[1];
    ChannelHandle hostHandles[1];
    processor_->AddNsRecoveryData(COMM_ENGINE_AICPU, deviceHandles, hostHandles, 1, "test_tag");

    StubKfcControlPut(HcclResult::HCCL_SUCCESS);
    StubKfcStatusGet(HcclResult::HCCL_SUCCESS, Hccl::KfcStatus::STOP_LAUNCH_DONE);

    auto ret = processor_->StopLaunch();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

// Test StopLaunch with AICPU_TS engine - success path
TEST_F(NsRecoveryProcessorTest, Ut_StopLaunch_When_AICPUTS_Expect_Success)
{
    ChannelHandle deviceHandles[1];
    ChannelHandle hostHandles[1];
    processor_->AddNsRecoveryData(COMM_ENGINE_AICPU_TS, deviceHandles, hostHandles, 1, "test_tag");

    StubKfcControlPut(HcclResult::HCCL_SUCCESS);
    StubKfcStatusGet(HcclResult::HCCL_SUCCESS, Hccl::KfcStatus::STOP_LAUNCH_DONE);

    auto ret = processor_->StopLaunch();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(NsRecoveryProcessorTest, Ut_Clean_When_AICPU_StopLaunchDoneAndCleanDone_Expect_Success)
{
    ChannelHandle deviceHandles[1];
    ChannelHandle hostHandles[1];
    processor_->AddNsRecoveryData(COMM_ENGINE_AICPU, deviceHandles, hostHandles, 1, "test_tag");

    getCallCount_ = 0;
    MOCKER_CPP(&hccl::HDCommunicate::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(GetStatusStub));
    StubKfcControlPut(HcclResult::HCCL_SUCCESS);

    auto ret = processor_->Clean();
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
