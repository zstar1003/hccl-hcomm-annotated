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

#define private public
#include "command_handle.h"
#include "profiling_manager.h"
#include "stream_pub.h"
#undef private

using namespace std;
using namespace hccl;

class CommandHandleTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CommandHandleTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "CommandHandleTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(CommandHandleTest, ut_command_handle_init)
{
    struct rtProfCommandHandle handle;
    handle.type = PROF_COMMANDHANDLE_TYPE_INIT;
    rtError_t ret = CommandHandle(RT_PROF_CTRL_SWITCH, static_cast<void*>(&handle), 0);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(CommandHandleTest, ut_command_handle_start)
{
    struct rtProfCommandHandle handle;
    handle.type = PROF_COMMANDHANDLE_TYPE_START;
    rtError_t ret = CommandHandle(RT_PROF_CTRL_SWITCH, static_cast<void*>(&handle), 0);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(CommandHandleTest, ut_command_handle_stop)
{
    struct rtProfCommandHandle handle;
    handle.type = PROF_COMMANDHANDLE_TYPE_STOP;
    rtError_t ret = CommandHandle(RT_PROF_CTRL_SWITCH, static_cast<void*>(&handle), 0);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(CommandHandleTest, ut_command_handle_finalize)
{
    struct rtProfCommandHandle handle;
    handle.type = PROF_COMMANDHANDLE_TYPE_FINALIZE;
    rtError_t ret = CommandHandle(RT_PROF_CTRL_SWITCH, static_cast<void*>(&handle), 0);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(CommandHandleTest, ut_command_handle_subscribe)
{
    struct rtProfCommandHandle handle;
    handle.type = PROF_COMMANDHANDLE_TYPE_MODEL_SUBSCRIBE;
    rtError_t ret = CommandHandle(RT_PROF_CTRL_SWITCH, static_cast<void*>(&handle), 0);
    EXPECT_EQ(ret, SUCCESS);
}
TEST_F(CommandHandleTest, ut_command_handle_unexpected_type)
{
    struct rtProfCommandHandle handle = {};
    handle.type = 999;

    rtError_t ret = CommandHandle(RT_PROF_CTRL_SWITCH, static_cast<void*>(&handle), 0);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(CommandHandleTest, ut_es_command_handle_unexpected_type)
{
    struct rtProfCommandHandle handle = {};
    handle.type = 999;

    rtError_t ret = EsCommandHandle(RT_PROF_CTRL_SWITCH, static_cast<void*>(&handle), 0);
    EXPECT_EQ(ret, SUCCESS);
}
TEST_F(CommandHandleTest, ut_command_handle_unsubscribe)
{
    struct rtProfCommandHandle handle;
    handle.type = PROF_COMMANDHANDLE_TYPE_MODEL_UNSUBSCRIBE;
    rtError_t ret = CommandHandle(RT_PROF_CTRL_SWITCH, static_cast<void*>(&handle), 0);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(CommandHandleTest, ut_es_command_handle_start)
{
    struct rtProfCommandHandle handle;
    EsCommandHandle(RT_PROF_CTRL_REPORTER, static_cast<void*>(&handle), 0);

    handle.type = PROF_COMMANDHANDLE_TYPE_START;
    handle.profSwitch = PROF_TASK_TIME_MASK;
    rtError_t ret = EsCommandHandle(RT_PROF_CTRL_SWITCH, static_cast<void*>(&handle), PROF_TASK_TIME_MASK);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(CommandHandleTest, ut_es_command_handle_stop)
{
    struct rtProfCommandHandle handle;
    handle.type = PROF_COMMANDHANDLE_TYPE_STOP;
    rtError_t ret = EsCommandHandle(RT_PROF_CTRL_SWITCH, static_cast<void*>(&handle), 0);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(CommandHandleTest, ut_es_command_handle_finalize)
{
    struct rtProfCommandHandle handle;
    handle.type = PROF_COMMANDHANDLE_TYPE_FINALIZE;
    rtError_t ret = EsCommandHandle(RT_PROF_CTRL_SWITCH, static_cast<void*>(&handle), 0);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(CommandHandleTest, ut_es_command_handle_subscribe)
{
    struct rtProfCommandHandle handle;
    handle.type = PROF_COMMANDHANDLE_TYPE_MODEL_SUBSCRIBE;
    rtError_t ret = EsCommandHandle(RT_PROF_CTRL_SWITCH, static_cast<void*>(&handle), 0);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(CommandHandleTest, ut_es_command_handle_unsubscribe)
{
    struct rtProfCommandHandle handle;
    handle.type = PROF_COMMANDHANDLE_TYPE_MODEL_UNSUBSCRIBE;
    rtError_t ret = EsCommandHandle(RT_PROF_CTRL_SWITCH, static_cast<void*>(&handle), 0);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(CommandHandleTest, ut_hcclProfilingAdditionalinfo_test)
{
    auto& profilingManager = hccl::ProfilingManager::Instance();
    EsUpdatePara para;
    para.tag = 0;

    profilingManager.CallMsprofReportAdditionInfoForEsUpdate(para, ProfTaskType::TASK_REMOTE_UPDATE_KEY_REDUCE);
}

TEST_F(CommandHandleTest, ut_command_handle_ctrl_reporter)
{
    struct rtProfCommandHandle handle;
    rtError_t ret = CommandHandle(RT_PROF_CTRL_REPORTER, static_cast<void*>(&handle), 0);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(CommandHandleTest, ut_hcclProfilingCompactInfo_test)
{
    auto& profilingManager = hccl::ProfilingManager::Instance();

    MsprofCompactInfo reporterData{};
    reporterData.level = MSPROF_REPORT_NODE_LEVEL;
    reporterData.type = MSPROF_REPORT_NODE_HCCL_OP_INFO_TYPE;
    reporterData.threadId = 1;
    reporterData.dataLen = 10;
    reporterData.timeStamp = 11;
    reporterData.data.hcclopInfo.relay = 0;
    reporterData.data.hcclopInfo.retry = 0;
    reporterData.data.hcclopInfo.dataType = HCCL_DATA_TYPE_FP32;
    reporterData.data.hcclopInfo.algType = 1;
    reporterData.data.hcclopInfo.count = 12;
    reporterData.data.hcclopInfo.groupName = 13;
    s32 deviceLogicId = 0;
    ProfilingManager::storageCompactInfo_[deviceLogicId].push(reporterData);
    profilingManager.ReportStoragedCompactInfo();
}

TEST_F(CommandHandleTest, ut_NodeInfo_Test)
{
    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    HcclResult ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string profName = "AICPUKernel";
    ret = ProfilingManagerPub::CallMsprofReportNodeInfo(0, 0, profName, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    auto& profilingManager = hccl::ProfilingManager::Instance();
    profilingManager.StopSubscribe(0);
    ret = hrtResetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
