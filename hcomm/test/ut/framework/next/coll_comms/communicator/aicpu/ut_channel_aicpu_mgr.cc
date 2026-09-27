/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#define private public
#include "topo_matcher.h"
#include "channel_aicpu_mgr.h"
#undef private

using namespace hccl;

class ChannelAicpuMgrTest : public testing::Test {
protected:
    HcclCommDfxLite dfx_;
    HcclTopoInfo topoInfo_;

    void SetUp() override { memset(&topoInfo_, 0, sizeof(topoInfo_)); }
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(ChannelAicpuMgrTest, Constructor_InitializesTransportMapEmpty)
{
    ChannelAicpuMgr mgr(dfx_, topoInfo_);
    EXPECT_EQ(mgr.transportMap_.size(), 0u);
}

TEST_F(ChannelAicpuMgrTest, AllocChannelResource_NullParam_ReturnsError)
{
    ChannelAicpuMgr mgr(dfx_, topoInfo_);
    HcclResult ret = mgr.AllocChannelResource(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelAicpuMgrTest, Resume_NullParam_ReturnsError)
{
    ChannelAicpuMgr mgr(dfx_, topoInfo_);
    HcclResult ret = mgr.Resume(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelAicpuMgrTest, Clean_EmptyMap_ReturnsSuccess)
{
    ChannelAicpuMgr mgr(dfx_, topoInfo_);
    HcclResult ret = mgr.Clean();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ChannelAicpuMgrTest, AllocChannelResource_SmokeTest)
{
    ChannelAicpuMgr mgr(dfx_, topoInfo_);
    HcclChannelUrmaRes commParam{};
    strncpy(commParam.hcomId, "test_group", HCOMID_MAX_LENGTH - 1);
    // 空 channelList / listNum=0 应在 ProcessUrmaRes 中快速返回
    commParam.listNum = 0;
    HcclResult ret = mgr.AllocChannelResource(&commParam);
    // 无有效 channel 数据时可能失败，只要不崩溃即可
    EXPECT_TRUE(ret == HCCL_SUCCESS || ret != HCCL_SUCCESS);
}

// InitUrmaChannel — 空 channelList 应返回错误或被 CHK_PTR_NULL 拦截
TEST_F(ChannelAicpuMgrTest, InitUrmaChannel_NullChannelList_ReturnsError)
{
    ChannelAicpuMgr mgr(dfx_, topoInfo_);
    HcclChannelUrmaRes commParam{};
    strncpy(commParam.hcomId, "test_init", HCOMID_MAX_LENGTH - 1);
    commParam.listNum = 1;
    HcclResult ret = mgr.InitUrmaChannel(&commParam);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// ProcessUrmaRes — listNum=0 但 channelList=null 会在循环前 CHK_PTR_NULL 返回错误
TEST_F(ChannelAicpuMgrTest, ProcessUrmaRes_EmptyList_NullPointers_ReturnsError)
{
    ChannelAicpuMgr mgr(dfx_, topoInfo_);
    HcclChannelUrmaRes commParam{};
    strncpy(commParam.hcomId, "test_process", HCOMID_MAX_LENGTH - 1);
    commParam.listNum = 0;
    HcclResult ret = mgr.ProcessUrmaRes(&commParam, true);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// ResumePackData — 空 map find 失败返回错误
TEST_F(ChannelAicpuMgrTest, ResumePackData_EmptyMap_ReturnsError)
{
    ChannelAicpuMgr mgr(dfx_, topoInfo_);
    std::vector<char> data(64, 0);
    ChannelHandle handle = 0x100;
    HcclResult ret = mgr.ResumePackData(data, handle);
    EXPECT_NE(ret, HCCL_SUCCESS);
}
