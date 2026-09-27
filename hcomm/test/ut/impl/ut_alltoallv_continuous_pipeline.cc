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
#include <stdio.h>
#include <chrono>
#include <memory>
#include <vector>

#define protected public
#define private public
#include "alltoallv_continuous_pipeline_pub.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

class AlltoallvContinuousPipelineTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AlltoallvContinuousPipelineTest Testcase SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "AlltoallvContinuousPipelineTest Testcase TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "AlltoallvContinuousPipelineTest SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "AlltoallvContinuousPipelineTest TearDown" << std::endl;
    }
};

// 等待的rank值已经就绪时，WaitValueOfRank应立即返回成功，不依赖超时时间
TEST_F(AlltoallvContinuousPipelineTest, wait_value_of_rank_success)
{
    HcclDispatcher disp;
    AlltoallvContinuousPipeline pipeline(disp);

    std::vector<u8> buffer(4096, 0);
    *reinterpret_cast<u32*>(buffer.data()) = 42;
    pipeline.inBuffer_ = DeviceMem::create(buffer.data(), buffer.size());
    pipeline.infoOffsets_.push_back(0);
    pipeline.flagAreaRefreshFlag_ = 1;

    u32 value = 0;
    HcclUs startTimeUs = std::chrono::steady_clock::now();
    HcclResult ret = pipeline.WaitValueOfRank(0, startTimeUs, value);

    EXPECT_EQ(HcclResult::HCCL_SUCCESS, ret);
    EXPECT_EQ(42u, value);
}

// 等待的rank值一直未就绪时，WaitValueOfRank应按waitFlagTimeoutSec_超时，而不是历史硬编码的1800s
TEST_F(AlltoallvContinuousPipelineTest, wait_value_of_rank_timeout)
{
    HcclDispatcher disp;
    AlltoallvContinuousPipeline pipeline(disp);

    std::vector<u8> buffer(4096, 0);
    pipeline.inBuffer_ = DeviceMem::create(buffer.data(), buffer.size());
    pipeline.infoOffsets_.push_back(0);
    pipeline.flagAreaRefreshFlag_ = 1;
    pipeline.waitFlagTimeoutSec_ = 1; // 超时时间设置为1s

    u32 value = 0;
    // 将起始时间设置为5s前，确保首轮轮询即超过1s超时
    HcclUs startTimeUs = std::chrono::steady_clock::now() - std::chrono::seconds(5);
    HcclResult ret = pipeline.WaitValueOfRank(0, startTimeUs, value);

    EXPECT_EQ(HcclResult::HCCL_E_TIMEOUT, ret);
}

// Prepare成功后，waitFlagTimeoutSec_应被设置为传入的超时时间+60s
TEST_F(AlltoallvContinuousPipelineTest, prepare_sets_wait_flag_timeout)
{
    HcclDispatcher disp;
    AlltoallvContinuousPipeline pipeline(disp);

    std::vector<u8> userIn(1024, 0);
    std::vector<u8> userOut(1024, 0);
    std::vector<u8> cclIn(65536, 0);
    std::vector<u8> cclOut(65536, 0);

    A2aPipelineMemory a2aMem;
    a2aMem.userInput = DeviceMem::create(userIn.data(), userIn.size());
    a2aMem.userOutput = DeviceMem::create(userOut.data(), userOut.size());
    a2aMem.cclInBuffer = DeviceMem::create(cclIn.data(), cclIn.size());
    a2aMem.cclOutBuffer = DeviceMem::create(cclOut.data(), cclOut.size());

    SubCommInfo level0Comm;
    level0Comm.localRank = 0;
    level0Comm.localRankSize = 2;
    SubCommInfo level1Comm;
    level1Comm.localRank = 0;
    level1Comm.localRankSize = 2;

    Stream mainStream;
    std::vector<Stream> subStream(2);
    std::vector<std::shared_ptr<LocalNotify>> notifyMain
        = {std::make_shared<LocalNotify>(), std::make_shared<LocalNotify>()};
    std::vector<std::shared_ptr<LocalNotify>> notifySub
        = {std::make_shared<LocalNotify>(), std::make_shared<LocalNotify>()};

    SendRecvInfo info;
    info.sendCounts.assign(4, 1);
    info.sendDispls.assign(4, 0);
    info.recvCounts.assign(4, 1);
    info.recvDispls.assign(4, 0);
    std::vector<SendRecvInfo> sendRecvInfoList = {info};

    const u32 timeOut = 100;
    HcclResult ret = pipeline.Prepare(
        0, a2aMem, level0Comm, level1Comm, mainStream, subStream, notifyMain, notifySub, sendRecvInfoList,
        HcclDataType::HCCL_DATA_TYPE_INT8, HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE, timeOut);

    EXPECT_EQ(HcclResult::HCCL_SUCCESS, ret);
    EXPECT_EQ(timeOut + 60u, pipeline.waitFlagTimeoutSec_);
}
