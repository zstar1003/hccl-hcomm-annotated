/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_UT_RANK_GRAPH_TEST_DATA_BUILDER_H
#define HCCL_UT_RANK_GRAPH_TEST_DATA_BUILDER_H

#include <initializer_list>
#include <set>
#include <string>
#include <vector>

#include "edge_info.h"
#include "new_rank_info.h"
#include "rank_table_info.h"
#include "topo_info.h"

namespace Hccl {
namespace test {
    inline AddressInfo
    MakeAddress(const std::string& ip, std::initializer_list<std::string> ports, const std::string& planeId = "0")
    {
        AddressInfo address;
        address.addr = IpAddress(ip);
        address.addrType = AddrType::IPV4;
        address.ports = std::set<std::string>(ports.begin(), ports.end());
        address.planeId = planeId;
        return address;
    }

    inline RankLevelInfo MakeRankLevel(
        u32 netLayer, const std::string& netInstId, NetType netType, const std::vector<AddressInfo>& rankAddrs)
    {
        RankLevelInfo level;
        level.netLayer = netLayer;
        level.netInstId = netInstId;
        level.netType = netType;
        level.rankAddrs = rankAddrs;
        for (const auto& rankAddr : level.rankAddrs) {
            for (const auto& port : rankAddr.ports) {
                level.portAddrMap[port].push_back(rankAddr.addr);
            }
        }
        return level;
    }

    inline NewRankInfo MakeRankInfo(
        u32 rankId, u32 deviceId, u32 localId, const std::vector<RankLevelInfo>& levels,
        u32 replacedLocalId = UINT32_MAX)
    {
        NewRankInfo rank;
        rank.rankId = rankId;
        rank.deviceId = deviceId;
        rank.localId = localId;
        rank.replacedLocalId = (replacedLocalId == UINT32_MAX) ? localId : replacedLocalId;
        rank.rankLevelInfos = levels;
        return rank;
    }

    inline RankTableInfo MakeRankTable(const std::vector<NewRankInfo>& ranks, bool detour = false)
    {
        RankTableInfo rankTable;
        rankTable.version = "2.0";
        rankTable.rankCount = ranks.size();
        rankTable.ranks = ranks;
        rankTable.detour = detour;
        return rankTable;
    }

    inline PeerInfo MakePeer(u32 localId)
    {
        PeerInfo peer;
        peer.localId = localId;
        return peer;
    }

    inline EdgeInfo MakeEdge(
        u32 netLayer, LinkType linkType, u32 localA, std::initializer_list<std::string> localAPorts, u32 localB = 0,
        std::initializer_list<std::string> localBPorts = {}, LinkProtocol protocol = LinkProtocol::UB_CTP,
        AddrPosition position = AddrPosition::DEVICE, u32 topoInstId = 0, TopoType topoType = TopoType::CLOS)
    {
        EdgeInfo edge;
        edge.netLayer = netLayer;
        edge.linkType = linkType;
        edge.topoType = topoType;
        edge.topoInstId = topoInstId;
        edge.protocols = {protocol};
        edge.localA = localA;
        edge.localAPorts = std::set<std::string>(localAPorts.begin(), localAPorts.end());
        edge.localB = localB;
        edge.localBPorts = std::set<std::string>(localBPorts.begin(), localBPorts.end());
        edge.position = position;
        return edge;
    }

    inline TopoInfo MakeTopoInfo(const std::vector<u32>& localIds, const std::vector<EdgeInfo>& edges)
    {
        TopoInfo topo;
        topo.version = "2.0";
        topo.peerCount = localIds.size();
        for (auto localId : localIds) {
            topo.peers.emplace_back(MakePeer(localId));
        }
        topo.edgeCount = edges.size();
        for (const auto& edge : edges) {
            topo.edges[edge.netLayer].push_back(edge);
        }
        return topo;
    }

    inline TopoInfo MakeTwoPeerTopo()
    {
        return MakeTopoInfo(
            {0, 1}, {
                        MakeEdge(0, LinkType::PEER2PEER, 0, {"0/0"}, 1, {"0/1"}),
                    });
    }

    inline TopoInfo MakeTwoPeerClosTopo()
    {
        return MakeTopoInfo(
            {0, 1},
            {
                MakeEdge(
                    0, LinkType::PEER2NET, 0, {"0/1", "0/2"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0),
                MakeEdge(
                    0, LinkType::PEER2NET, 0, {"1/1", "1/2"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 1),
                MakeEdge(
                    0, LinkType::PEER2NET, 1, {"0/3", "0/4"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0),
                MakeEdge(
                    0, LinkType::PEER2NET, 1, {"1/3", "1/4"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 1),
            });
    }

    inline TopoInfo MakeOnePeerTopo(bool withEdge)
    {
        if (!withEdge) {
            return MakeTopoInfo({0}, {});
        }
        return MakeTopoInfo(
            {0, 1}, {
                        MakeEdge(0, LinkType::PEER2PEER, 1, {"0/1"}, 0, {"0/0"}),
                    });
    }

    inline TopoInfo MakeFourPeerMeshTopo()
    {
        return MakeTopoInfo(
            {0, 1, 2, 3}, {
                              MakeEdge(0, LinkType::PEER2PEER, 0, {"0/0"}, 1, {"0/0"}),
                              MakeEdge(0, LinkType::PEER2PEER, 0, {"1/0"}, 2, {"1/0"}),
                              MakeEdge(0, LinkType::PEER2PEER, 1, {"1/0"}, 3, {"1/0"}),
                              MakeEdge(0, LinkType::PEER2PEER, 2, {"0/0"}, 3, {"0/0"}),
                          });
    }

    inline TopoInfo MakeFourPeerBuilderTopo()
    {
        std::vector<EdgeInfo> edges = {
            MakeEdge(0, LinkType::PEER2PEER, 0, {"0/0"}, 1, {"0/0"}),
            MakeEdge(0, LinkType::PEER2PEER, 0, {"0/1"}, 2, {"0/1"}),
            MakeEdge(0, LinkType::PEER2PEER, 1, {"0/2"}, 3, {"0/2"}),
            MakeEdge(0, LinkType::PEER2PEER, 2, {"0/0"}, 3, {"0/0"}),
        };
        for (u32 localId = 0; localId < 4U; ++localId) {
            edges.emplace_back(MakeEdge(
                1, LinkType::PEER2NET, localId, {"0/7", "0/8"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0));
            edges.emplace_back(MakeEdge(
                2, LinkType::PEER2NET, localId, {"1/7", "1/8"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0));
        }
        return MakeTopoInfo({0, 1, 2, 3}, edges);
    }

    inline TopoInfo MakeTwoByTwoPlusBackupTopo()
    {
        std::vector<EdgeInfo> edges = {
            MakeEdge(0, LinkType::PEER2PEER, 0, {"0/0"}, 1, {"0/0"}),
            MakeEdge(0, LinkType::PEER2PEER, 0, {"1/0"}, 8, {"1/0"}),
            MakeEdge(0, LinkType::PEER2PEER, 1, {"1/0"}, 9, {"1/0"}),
            MakeEdge(0, LinkType::PEER2PEER, 8, {"0/0"}, 9, {"0/0"}),
        };

        for (u32 localId : {0U, 1U, 8U, 9U}) {
            edges.emplace_back(MakeEdge(
                0, LinkType::PEER2NET, localId, {"0/1"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0));
            edges.emplace_back(MakeEdge(
                0, LinkType::PEER2NET, localId, {"0/2"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 1));
            edges.emplace_back(MakeEdge(
                0, LinkType::PEER2NET, localId, {"1/1"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 2));
            edges.emplace_back(MakeEdge(
                0, LinkType::PEER2NET, localId, {"1/2"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 3));
        }
        edges.emplace_back(MakeEdge(
            0, LinkType::PEER2NET, 64, {"0/0", "0/1", "0/2", "0/3"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE,
            0));
        edges.emplace_back(MakeEdge(
            0, LinkType::PEER2NET, 64, {"0/4", "0/5", "0/6", "0/7"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE,
            1));
        edges.emplace_back(MakeEdge(
            0, LinkType::PEER2NET, 64, {"1/0", "1/1", "1/2", "1/3"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE,
            2));
        edges.emplace_back(MakeEdge(
            0, LinkType::PEER2NET, 64, {"1/4", "1/5", "1/6", "1/7"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE,
            3));

        for (u32 localId : {0U, 1U, 8U, 9U, 64U}) {
            edges.emplace_back(
                MakeEdge(1, LinkType::PEER2NET, localId, {"0/8"}, 0, {}, LinkProtocol::UB_TP, AddrPosition::DEVICE, 0));
        }
        return MakeTopoInfo({0, 1, 8, 9, 64}, edges);
    }

    inline TopoInfo MakeHostRdmaTopo()
    {
        return MakeTopoInfo(
            {0, 1}, {
                        MakeEdge(1, LinkType::PEER2NET, 0, {"0/0"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE),
                        MakeEdge(3, LinkType::PEER2NET, 0, {"host0"}, 0, {}, LinkProtocol::ROCE, AddrPosition::HOST),
                    });
    }

    inline RankTableInfo MakeRankTable1p()
    {
        return MakeRankTable({
            MakeRankInfo(
                0, 0, 0,
                {
                    MakeRankLevel(
                        0, "az0-rack0", NetType::TOPO_FILE_DESC,
                        {
                            MakeAddress("223.0.0.28", {"0/0"}),
                        }),
                }),
        });
    }

    inline RankTableInfo MakeRankTable2p()
    {
        return MakeRankTable(
            {
                MakeRankInfo(
                    0, 0, 0,
                    {
                        MakeRankLevel(
                            0, "az0-rack0", NetType::CLOS,
                            {
                                MakeAddress("192.168.100.11", {"0/1", "0/2"}, "planeA"),
                                MakeAddress("192.168.100.10", {"1/1", "1/2"}, "planeB"),
                            }),
                    }),
                MakeRankInfo(
                    1, 1, 1,
                    {
                        MakeRankLevel(
                            0, "az0-rack0", NetType::CLOS,
                            {
                                MakeAddress("192.168.100.20", {"0/3", "0/4"}, "planeA"),
                                MakeAddress("192.168.100.21", {"1/3", "1/4"}, "planeB"),
                            }),
                    }),
            },
            true);
    }

    inline RankTableInfo MakeRankTable4pForBuilder()
    {
        std::vector<NewRankInfo> ranks;
        for (u32 rankId = 0; rankId < 4; ++rankId) {
            const u32 ipOffset = rankId * 10U + 1U;
            ranks.emplace_back(MakeRankInfo(
                rankId, rankId, rankId,
                {
                    MakeRankLevel(
                        0, "az0-rack0", NetType::TOPO_FILE_DESC,
                        {
                            MakeAddress("192.168.100." + std::to_string(ipOffset), {"0/0"}),
                            MakeAddress("192.168.100." + std::to_string(ipOffset + 1U), {"0/1"}),
                            MakeAddress("192.168.100." + std::to_string(ipOffset + 2U), {"0/2"}),
                        }),
                    MakeRankLevel(
                        1, "az0", NetType::CLOS,
                        {
                            MakeAddress("192.168.101." + std::to_string(ipOffset), {"0/7", "0/8"}),
                        }),
                    MakeRankLevel(
                        2, "all", NetType::CLOS,
                        {
                            MakeAddress("192.168.102." + std::to_string(ipOffset), {"1/7", "1/8"}),
                        }),
                }));
        }
        return MakeRankTable(ranks, true);
    }

    inline RankTableInfo MakeRankTable4p64Plus1(bool replaceRank1)
    {
        std::vector<NewRankInfo> ranks;
        for (u32 rankId = 0; rankId < 4; ++rankId) {
            u32 localId = rankId;
            u32 deviceId = rankId;
            u32 replacedLocalId = rankId;
            if (replaceRank1 && rankId == 1U) {
                localId = BACKUP_LOCAL_ID;
                deviceId = BACKUP_LOCAL_ID;
                replacedLocalId = 1U;
            }
            ranks.emplace_back(MakeRankInfo(
                rankId, deviceId, localId,
                {
                    MakeRankLevel(
                        0, "az0-rack0", NetType::TOPO_FILE_DESC,
                        {
                            MakeAddress("192.168.30." + std::to_string(rankId + 1U), {"0/0"}),
                            MakeAddress("192.168.20." + std::to_string(rankId + 1U), {"1/0"}),
                        }),
                },
                replacedLocalId));
        }
        return MakeRankTable(ranks);
    }

    inline std::vector<AddressInfo> MakeTwoByTwoLevel0Addrs(u32 ipThirdOctet, bool backupPorts)
    {
        std::vector<AddressInfo> addrs;
        const u32 maxPort = backupPorts ? 8U : 2U;
        for (u32 plane = 0; plane < 2; ++plane) {
            for (u32 port = 0; port <= maxPort; ++port) {
                if (!backupPorts && port > 2U) {
                    continue;
                }
                addrs.emplace_back(MakeAddress(
                    "192.168." + std::to_string(ipThirdOctet) + "."
                        + std::to_string((plane == 0U ? 100U : 110U) + port),
                    {std::to_string(plane) + "/" + std::to_string(port)}));
            }
        }
        return addrs;
    }

    inline RankTableInfo MakeRankTable2x2(bool replaceRank1)
    {
        const std::vector<u32> localIds = {0, 1, 8, 9};
        std::vector<NewRankInfo> ranks;
        for (u32 rankId = 0; rankId < localIds.size(); ++rankId) {
            u32 localId = localIds[rankId];
            u32 deviceId = localId;
            u32 replacedLocalId = localId;
            bool backupPorts = false;
            u32 level0IpOctet = 100U + localId;
            if (replaceRank1) {
                backupPorts = true;
                level0IpOctet = 164U;
                if (rankId == 1U) {
                    localId = BACKUP_LOCAL_ID;
                    deviceId = BACKUP_LOCAL_ID;
                    replacedLocalId = 1U;
                }
            }
            const u32 level1IpOctet = (replaceRank1 && rankId <= 1U) ? 164U : 100U + localIds[rankId];
            ranks.emplace_back(MakeRankInfo(
                rankId, deviceId, localId,
                {
                    MakeRankLevel(
                        0, "az0-rack0", NetType::TOPO_FILE_DESC, MakeTwoByTwoLevel0Addrs(level0IpOctet, backupPorts)),
                    MakeRankLevel(
                        1, "az0-layer1", NetType::CLOS,
                        {
                            MakeAddress("192.168." + std::to_string(level1IpOctet) + ".108", {"0/8"}),
                        }),
                },
                replacedLocalId));
        }
        return MakeRankTable(ranks);
    }
} // namespace test
} // namespace Hccl

#endif // HCCL_UT_RANK_GRAPH_TEST_DATA_BUILDER_H
