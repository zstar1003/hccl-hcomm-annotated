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

#include <vector>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <sys/wait.h>

#include "topoinfo_struct.h"
#include "log.h"
#include "checker_def.h"
#include "topo_meta.h"
#include "testcase_utils.h"
#include "alg_profiling.h"
#include "hccl_aiv.h"
#include "checker.h"
#include "ccl_buffer_manager.h"
#include "topo_matcher.h"

using namespace hccl;

class MiscTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MiscTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "MiscTest tear down." << std::endl; }

    virtual void SetUp()
    {
        const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::stringstream ss;
        ss << "analysis_result_" << std::string(test_info->test_case_name()) << "_" << std::string(test_info->name());
        checker::Checker::SetDumpFileName(ss.str());
    }

    virtual void TearDown()
    {
        checker::Checker::SetDumpFileName("analysis_result");
        // GlobalMockObject::verify();
        // 这边每个case执行完成需要清理所有的环境变量，如果有新增的环境变量，需要在这个函数中进行清理
        ClearHcclEnv();
    }
};

TEST_F(MiscTest, testcase_alg_profiling)
{
    AlgWrap::GetInstance().RegisterAlgCallBack("comm1", nullptr, nullptr, -1);
    AlgWrap::GetInstance().RegisterAlgCallBack("comm2", nullptr, nullptr, 1);
    struct TaskParaGeneral para;
    AlgWrap::GetInstance().TaskAivProfiler("comm1", para);
    AlgWrap::GetInstance().TaskAivProfiler("comm2", para);
    AlgWrap::GetInstance().UnregisterAlgCallBack("comm2");
}

TEST_F(MiscTest, testcase_RegisterKernel) { RegisterKernel(DevType::DEV_TYPE_910_93); }

TEST_F(MiscTest, testcase_ClearAivSyncBufAndTag)
{
    s32 tmp;
    DeviceMem inAIVbuffer = DeviceMem::create(&tmp, 4);
    DeviceMem outAIVbuffer = DeviceMem::create(&tmp, 4);
    std::string identifier = "test";

    HcclResult ret;
    std::string tagKey = "tmp_test";
}

TEST_F(MiscTest, testcase_ExecuteKernelLaunchInner)
{
    std::string tag = "tmp_test";

    void* buffersIn[MAX_RANK_SIZE];
    void* buffersOut[MAX_RANK_SIZE];
    u64 count = 1;
    u32 numBlocks = 1;
    s32 aivTag = TAG_RESET_COUNT;

    AivOpArgs opArgs{HcclCMDType::HCCL_CMD_ALLGATHER,
                     nullptr,
                     nullptr,
                     count,
                     HCCL_DATA_TYPE_RESERVED,
                     HCCL_REDUCE_RESERVED,
                     0,
                     true};
    AivTopoArgs topoArgs{0, 1, MAX_RANK_SIZE, 0, 1, DevType::DEV_TYPE_910_93};
    AivResourceArgs resourceArgs{tag, nullptr, buffersIn, buffersOut, 200 * 1024 * 1024, numBlocks, aivTag};
    AivAlgArgs algArgs{};
    struct AivProfilingInfo aivProfilingInfo;

    ExecuteKernelLaunchInner(opArgs, topoArgs, resourceArgs, algArgs, nullptr, 1, aivProfilingInfo);
}

TEST_F(MiscTest, testcase_CleanAIVbuffer)
{
    s32 tmp;
    DeviceMem aIVbuffer = DeviceMem::create(&tmp, 4);
    CCLBufferManager ccLBufferManager = CCLBufferManager();
    ccLBufferManager.CreateCommAIVbuffer(true);
    ccLBufferManager.CreateCommAIVbuffer(false);
    ccLBufferManager.CreateCommInfoAIVbuffer();
    ccLBufferManager.GetAivCommInfoBuffer();
    ccLBufferManager.GetInAivOpbaseBuffer();
    ccLBufferManager.GetOutAivOpbaseBuffer();
    ccLBufferManager.GetInAivOffloadbuffer();
    ccLBufferManager.GetOutAivOffloadbuffer();
    ccLBufferManager.ClearCommAIVbuffer();

    ccLBufferManager.CleanAIVbuffer(aIVbuffer.ptr());
}

TEST_F(MiscTest, testcase_SetDeterministicConfig)
{
    const std::vector<std::vector<std::vector<u32>>> CommPlaneRanks{};
    std::vector<bool> isBridgeVector{};
    HcclTopoInfo topoInfo{};
    HcclAlgoInfo algoInfo{};
    HcclExternalEnable externalEnable{};
    std::vector<std::vector<std::vector<u32>>> serverAndsuperPodToRank{};
    std::unique_ptr<TopoMatcher> topoMatcher = std::make_unique<TopoMatcher>(
        CommPlaneRanks, isBridgeVector, topoInfo, algoInfo, externalEnable, serverAndsuperPodToRank);
    topoMatcher->SetDeterministicConfig(0);
}
