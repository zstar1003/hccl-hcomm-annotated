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
#include <set>
#include "hccl_comm_pub.h"
#include "hccl_api_base_test.h"
#include "ccu_comp.h"
#include "hcomm_c_adpt.h"

#define private public
#define protected public
#include "ccu_kernel.h"
#include "ccu_kernel_mgr.h"
#include "ccu_urma_channel.h"
#include "ccuTaskException.h"
#undef private
#undef protected

using namespace hccl;
using namespace hcomm;

namespace hcomm {
extern std::mutex g_channelMapMutex;
extern std::unordered_map<uint16_t, uint64_t> g_channelIdToHandle;
} // namespace hcomm

class InitChannelMapTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        {
            std::lock_guard<std::mutex> lock(g_channelMapMutex);
            g_channelIdToHandle.clear();
        }
    }
    void TearDown() override
    {
        {
            std::lock_guard<std::mutex> lock(g_channelMapMutex);
            g_channelIdToHandle.clear();
        }
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(InitChannelMapTest, Ut_GetChannels_When_ChannelsEmpty_Expect_ReturnEmptySet)
{
    hcomm::CcuKernel kernel;
    kernel.channels_.clear();

    const auto& channels = kernel.GetChannels();
    EXPECT_TRUE(channels.empty());
}

TEST_F(InitChannelMapTest, Ut_GetChannels_When_ChannelsHasElements_Expect_ReturnSameSet)
{
    hcomm::CcuKernel kernel;
    kernel.channels_ = {100, 200, 300};

    const auto& channels = kernel.GetChannels();
    EXPECT_EQ(channels.size(), 3u);
    EXPECT_NE(channels.find(100), channels.end());
    EXPECT_NE(channels.find(200), channels.end());
    EXPECT_NE(channels.find(300), channels.end());
}

TEST_F(InitChannelMapTest, Ut_GetChannels_When_AddDuplicateChannel_Expect_SetSizeUnchanged)
{
    hcomm::CcuKernel kernel;
    kernel.channels_ = {100, 200};
    kernel.channels_.insert(100);

    const auto& channels = kernel.GetChannels();
    EXPECT_EQ(channels.size(), 2u);
}

TEST_F(InitChannelMapTest, Ut_InitChannelMap_When_KernelNullptr_Expect_ReturnParaError)
{
    MOCKER_CPP(&hcomm::CcuKernelMgr::GetKernel).stubs().will(returnValue(static_cast<hcomm::CcuKernel*>(nullptr)));

    HcclResult ret = CcuTaskException::InitChannelMap(0, 0x1234);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(InitChannelMapTest, Ut_InitChannelMap_When_ChannelsEmpty_Expect_SuccessAndEmptyMap)
{
    hcomm::CcuKernel kernel;
    kernel.channels_.clear();

    MOCKER_CPP(&hcomm::CcuKernelMgr::GetKernel).stubs().will(returnValue(&kernel));

    HcclResult ret = CcuTaskException::InitChannelMap(0, 0x1234);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::lock_guard<std::mutex> lock(g_channelMapMutex);
    EXPECT_TRUE(g_channelIdToHandle.empty());
}

TEST_F(InitChannelMapTest, Ut_InitChannelMap_When_HcommChannelGetFail_Expect_ReturnError)
{
    hcomm::CcuKernel kernel;
    kernel.channels_ = {0xABCD};

    MOCKER_CPP(&hcomm::CcuKernelMgr::GetKernel).stubs().will(returnValue(&kernel));

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_E_PARA)));

    HcclResult ret = CcuTaskException::InitChannelMap(0, 0x1234);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(InitChannelMapTest, Ut_InitChannelMap_When_ChannelPtrNull_Expect_ReturnPtrError)
{
    hcomm::CcuKernel kernel;
    kernel.channels_ = {0xABCD};

    void* nullPtr = nullptr;

    MOCKER_CPP(&hcomm::CcuKernelMgr::GetKernel).stubs().will(returnValue(&kernel));

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&nullPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    HcclResult ret = CcuTaskException::InitChannelMap(0, 0x1234);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(InitChannelMapTest, Ut_InitChannelMap_When_NormalSingleChannel_Expect_ChannelMapPopulated)
{
    hcomm::CcuKernel kernel;
    ChannelHandle handle = 0xAAA;
    kernel.channels_ = {handle};

    HcommChannelDesc desc{};
    CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<Channel*>(&channel);

    MOCKER_CPP(&hcomm::CcuKernelMgr::GetKernel).stubs().will(returnValue(&kernel));

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    HcclResult ret = CcuTaskException::InitChannelMap(0, 0x5678);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint16_t expectedKey = static_cast<uint16_t>(channel.GetChannelId());
    std::lock_guard<std::mutex> lock(g_channelMapMutex);
    EXPECT_EQ(g_channelIdToHandle.size(), 1u);
    EXPECT_EQ(g_channelIdToHandle[expectedKey], handle);
}

TEST_F(InitChannelMapTest, Ut_InitChannelMap_When_MultipleChannels_Expect_ChannelMapPopulated)
{
    hcomm::CcuKernel kernel;
    ChannelHandle handle1 = 0x100;
    ChannelHandle handle2 = 0x200;
    kernel.channels_ = {handle1, handle2};

    HcommChannelDesc desc1{};
    HcommChannelDesc desc2{};
    CcuUrmaChannel channel1(nullptr, desc1);
    CcuUrmaChannel channel2(nullptr, desc2);

    void* channelPtr1 = static_cast<Channel*>(&channel1);
    void* channelPtr2 = static_cast<Channel*>(&channel2);

    MOCKER_CPP(&hcomm::CcuKernelMgr::GetKernel).stubs().will(returnValue(&kernel));

    MOCKER(HcommChannelGet)
        .stubs()
        .with(eq(handle1), outBoundP(&channelPtr1))
        .will(returnValue(static_cast<HcommResult>(0)));
    MOCKER(HcommChannelGet)
        .stubs()
        .with(eq(handle2), outBoundP(&channelPtr2))
        .will(returnValue(static_cast<HcommResult>(0)));

    MOCKER_CPP(&CcuUrmaChannel::GetChannelId).stubs().will(returnValue(10u)).then(returnValue(20u));

    HcclResult ret = CcuTaskException::InitChannelMap(0, 0x1234);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::lock_guard<std::mutex> lock(g_channelMapMutex);
    EXPECT_EQ(g_channelIdToHandle.size(), 2u);
    std::set<uint64_t> values;
    for (auto& kv : g_channelIdToHandle) {
        values.insert(kv.second);
    }
    EXPECT_NE(values.find(handle1), values.end());
    EXPECT_NE(values.find(handle2), values.end());
}
