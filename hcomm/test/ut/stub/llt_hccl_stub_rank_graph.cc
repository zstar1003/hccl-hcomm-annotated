/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llt_hccl_stub_rank_graph.h"
#include "rank_graph_builder.h"

namespace hccl {

std::shared_ptr<Hccl::NetInstance::Peer>
RankGraphStub::InitPeer(Hccl::RankId rankId, Hccl::LocalId localId, Hccl::DeviceId deviceId, uint32_t listenPort)
{
    std::shared_ptr<Hccl::NetInstance::Peer> peer
        = std::make_shared<Hccl::NetInstance::Peer>(rankId, localId, localId, deviceId, listenPort);
    return peer;
}

std::shared_ptr<Hccl::NetInstance> RankGraphStub::InitNetInstance(u32 netLayer, std::string id)
{
    std::shared_ptr<Hccl::NetInstance> netInst;
    if (netLayer == 0) {
        netInst = std::make_shared<Hccl::InnerNetInstance>(netLayer, id);
    } else {
        netInst = std::make_shared<Hccl::ClosNetInstance>(netLayer, id);
    }
    return netInst;
}

std::shared_ptr<Hccl::NetInstance::ConnInterface> RankGraphStub::InitConnInterface(Hccl::IpAddress addr)
{
    Hccl::AddrPosition pos = Hccl::AddrPosition::DEVICE;
    Hccl::LinkType inputLinkType = Hccl::LinkType::PEER2PEER;
    std::set<Hccl::LinkProtocol> inputLinkProtocol = {Hccl::LinkProtocol::UB_CTP};
    std::set<std::string> ports = {"0/1"};
    Hccl::TopoType topoType = Hccl::TopoType::CLOS;
    uint32_t topoInstId = 0;
    std::shared_ptr<Hccl::NetInstance::ConnInterface> iface = std::make_shared<Hccl::NetInstance::ConnInterface>(
        addr, ports, pos, inputLinkType, inputLinkProtocol, topoType, topoInstId);
    return iface;
}

std::shared_ptr<Hccl::RankGraph> RankGraphStub::Create2PGraph()
{
    Hccl::RankGraph rankGraph(0);
    std::shared_ptr<Hccl::NetInstance> netInstLayer0 = InitNetInstance(0, "layer0");
    std::shared_ptr<Hccl::NetInstance::Peer> peer0 = InitPeer(0, 0, 0, 60001);
    std::shared_ptr<Hccl::NetInstance::Peer> peer1 = InitPeer(1, 1, 1, 60002);

    uint32_t topoInstId = 0;
    auto topoInstance = std::make_shared<Hccl::NetInstance::TopoInstance>(topoInstId);
    std::unordered_map<uint32_t, std::shared_ptr<Hccl::NetInstance::TopoInstance>> topoInsts_;
    topoInstance->topoType = Hccl::TopoType::MESH_1D;
    std::set<Hccl::RankId> rankSet = {0, 1};
    topoInstance->ranks = std::move(rankSet);
    topoInsts_[topoInstId] = topoInstance;
    netInstLayer0->topoInsts_ = std::move(topoInsts_);

    peer0->AddNetInstance(netInstLayer0);
    peer1->AddNetInstance(netInstLayer0);
    netInstLayer0->AddRankId(peer0->GetRankId());
    netInstLayer0->AddRankId(peer1->GetRankId());
    netInstLayer0->AddNode(peer0);
    netInstLayer0->AddNode(peer1);

    char rank0Address[] = "192.168.1.0";
    Hccl::IpAddress rank0Addr(rank0Address);
    auto iface0 = InitConnInterface(rank0Addr);

    char rank1Address[] = "192.168.1.1";
    Hccl::IpAddress rank1Addr(rank1Address);
    auto iface1 = InitConnInterface(rank1Addr);

    peer0->AddConnInterface(0, iface0);
    peer1->AddConnInterface(0, iface1);

    Hccl::LinkType type = Hccl::LinkType::PEER2PEER;
    std::set<Hccl::LinkProtocol> protocols = {Hccl::LinkProtocol::UB_CTP};
    auto link = std::make_shared<Hccl::NetInstance::Link>(peer0, peer1, iface0, iface1, type, protocols);
    netInstLayer0->AddLink(link);

    rankGraph.AddPeer(peer0);
    rankGraph.AddPeer(peer1);
    rankGraph.AddNetInstance(netInstLayer0);
    rankGraph.InitInnerRanks();
    return std::make_shared<Hccl::RankGraph>(rankGraph);
}

std::shared_ptr<Hccl::RankGraph> RankGraphStub::Create1PGraph()
{
    Hccl::RankGraph rankGraph(0);
    std::shared_ptr<Hccl::NetInstance> netInstLayer0 = InitNetInstance(0, "layer0");
    std::shared_ptr<Hccl::NetInstance::Peer> peer0 = InitPeer(0, 0, 0, 60001);

    uint32_t topoInstId = 0;
    auto topoInstance = std::make_shared<Hccl::NetInstance::TopoInstance>(topoInstId);
    std::unordered_map<uint32_t, std::shared_ptr<Hccl::NetInstance::TopoInstance>> topoInsts_;
    topoInstance->topoType = Hccl::TopoType::MESH_1D;
    std::set<Hccl::RankId> rankSet = {0};
    topoInstance->ranks = std::move(rankSet);
    topoInsts_[topoInstId] = topoInstance;
    netInstLayer0->topoInsts_ = std::move(topoInsts_);

    peer0->AddNetInstance(netInstLayer0);
    netInstLayer0->AddRankId(peer0->GetRankId());
    netInstLayer0->AddNode(peer0);

    char rank0Address[] = "192.168.1.0";
    Hccl::IpAddress rank0Addr(rank0Address);
    auto iface0 = InitConnInterface(rank0Addr);
    peer0->AddConnInterface(0, iface0);

    rankGraph.AddPeer(peer0);
    rankGraph.AddNetInstance(netInstLayer0);
    rankGraph.InitInnerRanks();
    return std::make_shared<Hccl::RankGraph>(rankGraph);
}
} // namespace hccl
