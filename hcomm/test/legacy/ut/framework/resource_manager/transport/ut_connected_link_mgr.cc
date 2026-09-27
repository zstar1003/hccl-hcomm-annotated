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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "connected_link_mgr.h"
#include "rank_gph.h"
#undef private
using namespace Hccl;

class ConnectedLinkMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ConnectedLinkMgrTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "ConnectedLinkMgrTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in ConnectedLinkMgrTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in ConnectedLinkMgrTest TearDown" << std::endl;
    }

    std::shared_ptr<NetInstance::Link> InitBaseLink(
        std::shared_ptr<NetInstance::Node> srcNodePtr, std::shared_ptr<NetInstance::Node> dstNodePtr, u32 hop = 1)
    {
        IpAddress srcAddr = IpAddress(0);
        IpAddress dstAddr = IpAddress(0);
        AddrPosition addrPos = AddrPosition::DEVICE;
        LinkType linkType = LinkType::PEER2PEER;
        std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
        LinkDirection direction = LinkDirection::BOTH;
        std::set<std::string> ports = {"0/1"};

        NetInstance::ConnInterface srcIf = NetInstance::ConnInterface(srcAddr, ports, addrPos, linkType, protocols);

        NetInstance::ConnInterface dstIf = NetInstance::ConnInterface(dstAddr, ports, addrPos, linkType, protocols);

        auto link = std::make_shared<NetInstance::Link>(
            srcNodePtr, dstNodePtr, std::make_shared<NetInstance::ConnInterface>(srcIf),
            std::make_shared<NetInstance::ConnInterface>(dstIf), linkType, protocols, direction, hop);

        return link;
    }
};

void GenerateRankPairLinkDataMap4(
    std::unordered_map<RankId, std::vector<LinkData>>& rankPairLinkDataMap, const LinkData* dto)
{
    auto iter = rankPairLinkDataMap.find(dto->GetRemoteRankId());
    if (iter == rankPairLinkDataMap.end()) {
        std::vector<LinkData> linkDatas{*dto};
        rankPairLinkDataMap[dto->GetRemoteRankId()] = linkDatas;
    } else {
        rankPairLinkDataMap[dto->GetRemoteRankId()].push_back(*dto);
    }
}

ConnectedLinkMgr MakeLinkMgr4(std::vector<LinkData>& links)
{
    ConnectedLinkMgr mgr;
    std::unordered_map<RankId, std::vector<LinkData>> rankPairLinkDataMap;
    auto linkIter = links.begin();
    for (; linkIter != links.end(); linkIter++) {
        GenerateRankPairLinkDataMap4(rankPairLinkDataMap, &(*linkIter));
    }

    std::unordered_map<u32, std::unordered_map<RankId, std::vector<LinkData>>> levelRankPairLinkDataMap;
    u32 level = 0;
    levelRankPairLinkDataMap.insert(std::make_pair(level, rankPairLinkDataMap));
    mgr.levelRankPairLinkDataMap = levelRankPairLinkDataMap;
    return mgr;
}

TEST_F(ConnectedLinkMgrTest, test_update_and_get_links)
{
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);
    std::vector<LinkData> links{link};
    ConnectedLinkMgr mgr = MakeLinkMgr4(links);

    auto res = mgr.GetLinks(1);
    EXPECT_EQ(1, res.size());
    EXPECT_EQ(link, res[0]);

    mgr.Reset();

    res = mgr.GetLinks(1);
    EXPECT_EQ(0, res.size());
}

TEST_F(ConnectedLinkMgrTest, test_parse_packed_data)
{
    u32 level = 0;
    RankId myRank = 8;
    std::vector<RankId> dstRanks{9, 10};
    auto virtualTopo = std::make_unique<RankGraph>(myRank);
    std::shared_ptr<NetInstance::Peer> virtPeer = std::make_shared<NetInstance::Peer>(myRank, 0, 0, 0);
    std::shared_ptr<NetInstance::Peer> virtPeer1 = std::make_shared<NetInstance::Peer>(dstRanks[0], 1, 1, 0);
    std::shared_ptr<NetInstance::Peer> virtPeer2 = std::make_shared<NetInstance::Peer>(dstRanks[1], 2, 2, 0);
    std::shared_ptr<NetInstance::Link> virtlink = InitBaseLink(virtPeer, virtPeer1);
    std::shared_ptr<NetInstance::Link> virtlink1 = InitBaseLink(virtPeer, virtPeer2);
    virtualTopo->AddPeer(virtPeer);
    virtualTopo->AddPeer(virtPeer1);
    virtualTopo->AddPeer(virtPeer2);
    std::shared_ptr<NetInstance> fabGroup = std::make_shared<InnerNetInstance>(level, "test");
    fabGroup->AddNode(virtPeer);
    fabGroup->AddRankId(myRank);
    fabGroup->AddNode(virtPeer1);
    fabGroup->AddRankId(dstRanks[0]);
    fabGroup->AddNode(virtPeer2);
    fabGroup->AddRankId(dstRanks[1]);
    fabGroup->AddLink(virtlink);
    fabGroup->AddLink(virtlink1);
    virtualTopo->AddNetInstance(fabGroup);
    virtPeer->AddNetInstance(fabGroup);
    virtualTopo->InitInnerRanks();

    ConnectedLinkMgr mgr;
    std::vector<std::pair<u32, RankId>> levelRankPairs;
    for (u32 i = 0; i < dstRanks.size(); ++i) {
        levelRankPairs.push_back({level, dstRanks[i]});
    }
    auto data = virtualTopo->GetPackedData(levelRankPairs);
    mgr.ParsePackedData(data);

    EXPECT_EQ(mgr.levelRankPairLinkDataMap[level].size(), dstRanks.size());
    std::vector<LinkData> links;
    for (const auto& it : levelRankPairs) {
        for (const auto& path : fabGroup->GetPaths(myRank, it.second)) {
            links.emplace_back(path);
        }
    }
    for (u32 i = 0; i < links.size(); ++i) {
        EXPECT_EQ(links[i], mgr.levelRankPairLinkDataMap[level][dstRanks[i]][0]);
    }
}
