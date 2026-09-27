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

#include "rank_pair.h"
#include "endpoint_pair_mgr.h"
#include "endpoint_pair.h"

using namespace hccl;
using namespace hcomm;

class RankPairTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(RankPairTest, GetEpChannelMap_ReturnsEmptyWhenMgrEmpty)
{
    RankIdPair ids = {1, 2};
    RankPair rp(ids, nullptr);
    EXPECT_EQ(rp.Init(), HCCL_SUCCESS);

    // Stub EndpointPairMgr::GetEpChannelMap to return empty map
    hcomm::EpChannelMap emptyMap;
    MOCKER_CPP(&hcomm::EndpointPairMgr::GetEpChannelMap, hcomm::EpChannelMap(hcomm::EndpointPairMgr::*)())
        .stubs()
        .will(returnValue(emptyMap));

    auto ret = rp.GetEpChannelMap();
    EXPECT_TRUE(ret.empty());
}

TEST_F(RankPairTest, GetEpChannelMap_ReturnsProvidedMap)
{
    RankIdPair ids = {10, 20};
    RankPair rp(ids, nullptr);
    EXPECT_EQ(rp.Init(), HCCL_SUCCESS);

    // Build a fake EpChannelMap with one EndpointDescPair -> (COMM_ENGINE_AICPU -> [0x1, 0x2])
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

    EndpointDescPair epPair = std::make_pair(localEp, remoteEp);

    std::unordered_map<CommEngine, std::vector<ChannelHandle>> inner;
    inner[COMM_ENGINE_AICPU] = {(ChannelHandle)0x1, (ChannelHandle)0x2};

    hcomm::EpChannelMap fakeMap;
    fakeMap[epPair] = inner;

    MOCKER_CPP(&hcomm::EndpointPairMgr::GetEpChannelMap, hcomm::EpChannelMap(hcomm::EndpointPairMgr::*)())
        .stubs()
        .will(returnValue(fakeMap));

    auto ret = rp.GetEpChannelMap();
    EXPECT_EQ(ret.size(), 1u);
    auto it = ret.find(epPair);
    EXPECT_NE(it, ret.end());
    auto innerRet = it->second;
    EXPECT_EQ(innerRet.size(), 1u);
    auto vec = innerRet[COMM_ENGINE_AICPU];
    ASSERT_EQ(vec.size(), 2u);
    EXPECT_EQ(vec[0], (ChannelHandle)0x1);
    EXPECT_EQ(vec[1], (ChannelHandle)0x2);
}
