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

// Expose private members for testing purposes in this TU
#define private public
#include "endpoint_pair_mgr.h"
#include "endpoint_pair.h"
#undef private

using namespace hcomm;

class EndpointPairMgrTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(EndpointPairMgrTest, GetEpChannelMap_EmptyWhenNoEndpointPairs)
{
    EndpointPairMgr mgr;
    auto map = mgr.GetEpChannelMap();
    EXPECT_TRUE(map.empty());
}

TEST_F(EndpointPairMgrTest, GetEpChannelMap_ReturnsInsertedChannelHandles)
{
    EndpointPairMgr mgr;

    // Build endpoint descriptors
    EndpointDesc localEp{};
    localEp.protocol = COMM_PROTOCOL_ROCE;
    localEp.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    localEp.commAddr.addr = Hccl::IpAddress("1.1.1.1").GetBinaryAddress().addr;
    localEp.loc.locType = ENDPOINT_LOC_TYPE_HOST;

    EndpointDesc remoteEp{};
    remoteEp.protocol = COMM_PROTOCOL_ROCE;
    remoteEp.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    remoteEp.commAddr.addr = Hccl::IpAddress("2.2.2.2").GetBinaryAddress().addr;
    remoteEp.loc.locType = ENDPOINT_LOC_TYPE_HOST;

    EndpointDescPair key = std::make_pair(localEp, remoteEp);
    Hccl::RankIpPortMapPtr rankIpPortMap;
    // Prepare an EndpointPair and populate its channelHandles_
    auto epPtr = std::make_unique<EndpointPair>(localEp, remoteEp, rankIpPortMap);
    // Directly set the internal channel handles (we accessed private via #define)
    epPtr->channelHandles_[COMM_ENGINE_CPU] = {(ChannelHandle)0x11, (ChannelHandle)0x22};

    // Insert into mgr's private map
    mgr.endpointPairMap_.emplace(key, std::move(epPtr));

    auto map = mgr.GetEpChannelMap();
    EXPECT_EQ(map.size(), 1u);
    auto it = map.find(key);
    EXPECT_NE(it, map.end());
    auto inner = it->second;
    EXPECT_EQ(inner.size(), 1u);
    auto vec = inner[COMM_ENGINE_CPU];
    ASSERT_EQ(vec.size(), 2u);
    EXPECT_EQ(vec[0], (ChannelHandle)0x11);
    EXPECT_EQ(vec[1], (ChannelHandle)0x22);
}
