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
#include <string>
#include "hccl/base.h"
#include <hccl/hccl_types.h>

#define private public
#define protected public
#include "task_profiling_pub.h"
#include "profiler_base_pub.h"
#include "alg_profiling.h"
#include "profiling_manager.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

namespace {
constexpr u32 kTestDeviceId = 0;
constexpr s32 kTestPlaneId = 7;
constexpr s32 kCaptureStreamId = 100;
constexpr s32 kRealStreamId = 0;
constexpr u32 kTestTaskId = 1;
constexpr u32 kTestRankSize = 8;
constexpr u32 kTestLocalRank = 2;
constexpr u64 kTestDataSize = 4096;
constexpr u64 kMockHashValue = 0xABCDEF;
constexpr u64 kMockTs = 123456789;
const string kTestTag = "ut_test_tag";
const string kTestGroupName = "ut_test_group";

static HCCLReportData g_capturedReportData{};
static bool g_reportDataCaptured = false;

HcclResult stub_ReportMsprofData_Capture(HCCLReportData& hcclReportData)
{
    g_capturedReportData = hcclReportData;
    g_reportDataCaptured = true;
    return HCCL_SUCCESS;
}

u64 stub_hrtMsprofSysCycleTime() { return kMockTs; }

u64 stub_hrtMsprofGetHashId(const char* buf, size_t len)
{
    (void)buf;
    (void)len;
    return kMockHashValue;
}

class TaskProfilingSaveTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    virtual void SetUp()
    {
        ProfilerBase::streamRecordInfoMap_.fill({});
        ProfilerBase::tagGroupMap_.fill({});
        ProfilerBase::tagModeMap_.fill({});

        g_capturedReportData = HCCLReportData{};
        g_reportDataCaptured = false;

        MOCKER(&TaskProfiling::ReportMsprofData).stubs().will(invoke(stub_ReportMsprofData_Capture));
        MOCKER(hrtMsprofSysCycleTime).stubs().will(returnValue(static_cast<uint64_t>(kMockTs)));
        MOCKER(hrtMsprofGetHashId).stubs().will(returnValue(static_cast<uint64_t>(kMockHashValue)));
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        ProfilerBase::streamRecordInfoMap_.fill({});
        ProfilerBase::tagGroupMap_.fill({});
        ProfilerBase::tagModeMap_.fill({});
    }
};

TaskParaAiv MakeParaAiv()
{
    TaskParaAiv para;
    para.cmdType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    para.tag = 0;
    para.size = kTestDataSize;
    para.numBlocks = 0;
    para.rankSize = kTestRankSize;
    para.aivRdmaStep = 0;
    para.flagMem = nullptr;
    para.rank = kTestLocalRank;
    para.isOpbase = true;
    return para;
}

} // namespace

TEST_F(TaskProfilingSaveTest, Ut_Save_CaptureStreamID_NotInMap_FillsUnknown)
{
    ASSERT_TRUE(ProfilerBase::streamRecordInfoMap_[kTestDeviceId].empty());

    TaskProfiling profiler(kTestDeviceId, kTestLocalRank, kTestRankSize, true);
    TaskParaAiv paraAiv = MakeParaAiv();

    HcclResult ret = profiler.Save(kCaptureStreamId, kRealStreamId, kTestTaskId, paraAiv);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ASSERT_TRUE(g_reportDataCaptured);
    EXPECT_EQ(g_capturedReportData.tag, "unknown");
    EXPECT_EQ(g_capturedReportData.profInfo.planeID, 0u);
    EXPECT_EQ(g_capturedReportData.groupName, "unknown");
    EXPECT_EQ(g_capturedReportData.profInfo.workFlowMode, 0u);
}

TEST_F(TaskProfilingSaveTest, Ut_Save_CaptureStreamID_InMap_PopulatesFromMaps)
{
    ProfilerBase::streamRecordInfoMap_[kTestDeviceId][kCaptureStreamId]
        = StreamRecordInfo(kTestPlaneId, AlgType(), kTestTag);
    ProfilerBase::tagGroupMap_[kTestDeviceId].insert({kTestTag, kTestGroupName});
    ProfilerBase::tagModeMap_[kTestDeviceId].insert({kTestTag, HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE});

    ASSERT_EQ(ProfilerBase::streamRecordInfoMap_[kTestDeviceId].count(kCaptureStreamId), 1u);

    TaskProfiling profiler(kTestDeviceId, kTestLocalRank, kTestRankSize, true);
    TaskParaAiv paraAiv = MakeParaAiv();

    HcclResult ret = profiler.Save(kCaptureStreamId, kRealStreamId, kTestTaskId, paraAiv);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ASSERT_TRUE(g_reportDataCaptured);
    EXPECT_EQ(g_capturedReportData.tag, kTestTag);
    EXPECT_EQ(g_capturedReportData.profInfo.planeID, static_cast<u32>(kTestPlaneId));
    EXPECT_EQ(g_capturedReportData.groupName, kTestGroupName);
    EXPECT_EQ(
        g_capturedReportData.profInfo.workFlowMode, static_cast<u32>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
}

TEST_F(TaskProfilingSaveTest, Ut_Save_CaptureStreamID_InMap_TagNotInGroupModeMaps_DefaultsInserted)
{
    ProfilerBase::streamRecordInfoMap_[kTestDeviceId][kCaptureStreamId]
        = StreamRecordInfo(kTestPlaneId, AlgType(), "orphan_tag");

    TaskProfiling profiler(kTestDeviceId, kTestLocalRank, kTestRankSize, true);
    TaskParaAiv paraAiv = MakeParaAiv();

    HcclResult ret = profiler.Save(kCaptureStreamId, kRealStreamId, kTestTaskId, paraAiv);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ASSERT_TRUE(g_reportDataCaptured);
    EXPECT_EQ(g_capturedReportData.tag, "orphan_tag");
    EXPECT_EQ(g_capturedReportData.profInfo.planeID, static_cast<u32>(kTestPlaneId));
    EXPECT_EQ(g_capturedReportData.groupName, "");
    EXPECT_EQ(g_capturedReportData.profInfo.workFlowMode, 0u);

    ProfilerBase::tagGroupMap_[kTestDeviceId].erase("orphan_tag");
    ProfilerBase::tagModeMap_[kTestDeviceId].erase("orphan_tag");
}
