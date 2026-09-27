/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_INSTANCE_H
#define NET_INSTANCE_H

#include <set>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <map>
#include <utility>

#include "graph.h"
#include "ip_address.h"
#include "iterator.h"
#include "types.h"
#include "topo_common_types.h"

namespace Hccl {
constexpr u32 DEFAULT_LISTENING_PORT = 60001;
class NetInstance {
public:
    class ConnInterface {
    public:
        // 使用地址信息、位置信息、链路类型、链路协议构造接口
        explicit ConnInterface(
            const IpAddress inputAddr, const std::set<string> inputPorts, const AddrPosition inputPos,
            const LinkType inputLinkType, const std::set<LinkProtocol> inputLinkProtocol,
            TopoType inputTopoType = TopoType::CLOS, u32 intputTopoInstId = 0)
            : addr(inputAddr),
              ports(inputPorts),
              pos(inputPos),
              linkType(inputLinkType),
              linkProtocols(inputLinkProtocol),
              topoType(inputTopoType),
              topoInstId(intputTopoInstId)
        {}
        IpAddress GetAddr() const;
        AddrPosition GetPos() const;
        std::set<string> GetPorts() const;
        LinkType GetLinkType() const;
        std::set<LinkProtocol> GetLinkProtocols() const;
        void SetLocalDieId(u32 dieId);
        u32 GetLocalDieId() const;
        TopoType GetTopoType() const;
        u32 GetTopoInstId() const;
        std::string Describe() const;
        bool operator==(const ConnInterface& rhs) const;
        bool operator!=(const ConnInterface& rhs) const;

    private:
        IpAddress addr{};
        std::set<string> ports{};
        AddrPosition pos{};
        LinkType linkType{};
        std::set<LinkProtocol> linkProtocols{};
        u32 localDieId_{};
        TopoType topoType{TopoType::CLOS};
        u32 topoInstId{0};
    };

    class Node {
    public:
        MAKE_ENUM(NodeType, PEER, FABRIC)
        explicit Node(NodeType nodeType) : type_(nodeType) {}
        virtual ~Node() = default;

        void AddConnInterface(u32 layer, const shared_ptr<NetInstance::ConnInterface>& interface);
        void AddConnInterfaces(u32 layer, const std::vector<std::shared_ptr<NetInstance::ConnInterface>>& interfaces);
        NodeType GetType() const;
        std::vector<std::shared_ptr<NetInstance::ConnInterface>> GetIfacesByLayer(u32 layer) const;
        std::vector<std::shared_ptr<NetInstance::ConnInterface>> GetIfaces() const;
        void SetEndpointToIface(
            const CommAddr& commAddr, CommProtocol protocol, const std::shared_ptr<NetInstance::ConnInterface>& iface);
        const std::unordered_map<std::pair<CommAddr, CommProtocol>, std::shared_ptr<NetInstance::ConnInterface>>
        GetEndpointToIfaceMap() const;
        NodeId GetNodeId() const;
        string GetNodeIdStr() const;
        const std::unordered_map<u32, std::vector<std::shared_ptr<NetInstance::ConnInterface>>>
        GetInterfacesMap() const;
        virtual std::string Describe() const = 0;

    protected:
        NodeId nodeId_{0};

    private:
        std::unordered_map<u32, std::vector<std::shared_ptr<NetInstance::ConnInterface>>> interfacesMap_;
        std::unordered_map<std::pair<CommAddr, CommProtocol>, std::shared_ptr<NetInstance::ConnInterface>>
            endpointToIfaceMap_;
        NodeType type_;
    };

    class Peer : public Node {
    public:
        using NetInstancePtr = const NetInstance*;
        Peer(
            RankId rankId, LocalId localId, LocalId replacedLocalId, DeviceId deviceId,
            u32 devicePort = DEFAULT_LISTENING_PORT, u32 hostPort = DEFAULT_LISTENING_PORT)
            : Node(NodeType::PEER),
              rankId_(rankId),
              localId_(localId),
              replacedLocalId_(replacedLocalId),
              deviceId_(deviceId),
              devicePort_(devicePort),
              hostPort_(hostPort)
        {
            nodeId_ = GenerateNodeId(rankId);
        }
        static NodeId GenerateNodeId(RankId rankId);
        void AddNetInstance(const std::shared_ptr<NetInstance>& NetInstance);
        LocalId GetLocalId() const;
        LocalId GetReplacedLocalId() const;
        RankId GetRankId() const;
        DeviceId GetDeviceId() const;
        u32 GetDevicePort() const;
        u32 GetHostPort() const;
        std::set<u32> GetLevels() const;
        NetInstancePtr GetNetInstance(u32 level) const;
        std::map<std::string, std::vector<IpAddress>> GetPortAddrMapLayer0() const;
        void SetPortPortAddrMapLayer0(std::map<std::string, std::vector<IpAddress>> portAddrMap);
        std::string Describe() const override;

    private:
        RankId rankId_;
        LocalId localId_;
        LocalId replacedLocalId_;
        DeviceId deviceId_;
        u32 devicePort_;
        u32 hostPort_;
        std::set<u32> netLayers_;
        std::map<std::string, std::vector<IpAddress>> portAddrMapLayer0_{}; // layer0 层端口与IpAddress的映射。
        std::vector<NetInstancePtr> netInsts_;                              // 下标为level，约束：level从0递增
    };

    class Fabric : public Node {
    public:
        explicit Fabric(FabricId fabricId, PlaneId planeId)
            : Node(NodeType::FABRIC),
              fabricId_(fabricId),
              planeId_(planeId)
        {
            nodeId_ = GenerateNodeId(fabricId);
        }

        explicit Fabric(FabricId fabricId) : Node(NodeType::FABRIC), fabricId_(fabricId), planeId_("")
        {
            nodeId_ = GenerateNodeId(fabricId);
        }

        PlaneId GetPlaneId() const;
        std::string Describe() const override;

    private:
        FabricId fabricId_;
        PlaneId planeId_;
        NodeId GenerateNodeId(FabricId fabricId) const;
    };

    class Link {
    public:
        Link(
            std::shared_ptr<NetInstance::Node> source, std::shared_ptr<NetInstance::Node> target,
            std::shared_ptr<NetInstance::ConnInterface> sourceIface,
            std::shared_ptr<NetInstance::ConnInterface> targetIface, LinkType type,
            std::set<LinkProtocol> linkProtocols, LinkDirection direction = LinkDirection::BOTH, u32 hop = 1)
            : source_(source),
              target_(target),
              sourceIface_(sourceIface),
              targetIface_(targetIface),
              type_(type),
              linkProtocols_(linkProtocols),
              direction_(direction),
              hop_(hop)
        {}
        Link() = default;

        u32 GetHop() const;
        LinkType GetType() const;
        std::set<LinkProtocol> GetLinkProtocols() const;
        LinkDirection GetLinkDirection() const;
        std::shared_ptr<NetInstance::ConnInterface> GetSourceIface() const;
        std::shared_ptr<NetInstance::ConnInterface> GetTargetIface() const;
        std::shared_ptr<NetInstance::Node> GetSourceNode() const;
        std::shared_ptr<NetInstance::Node> GetTargetNode() const;
        std::string Describe() const;
        bool IsEmpty() const;

        bool operator==(const Link& rhs) const;
        bool operator!=(const Link& rhs) const;

    private:
        std::shared_ptr<NetInstance::Node> source_{nullptr};
        std::shared_ptr<NetInstance::Node> target_{nullptr};
        std::shared_ptr<NetInstance::ConnInterface> sourceIface_{nullptr};
        std::shared_ptr<NetInstance::ConnInterface> targetIface_{nullptr}; // 如果target为Fabric节点，则为空
        LinkType type_{};
        set<LinkProtocol> linkProtocols_{};
        LinkDirection direction_{LinkDirection::BOTH};
        u32 hop_{1};
    };

    struct Path {
        std::vector<Link> links;
        LinkDirection direction{LinkDirection::BOTH};
    };

    struct TopoInstance {
        u32 topoInstId{0};
        TopoType topoType;
        std::set<RankId> ranks;
        TopoInstance() = default;

        TopoInstance(u32 instId) : topoInstId(instId) {}
    };

    // FabType: Fabric Group的拓扑类型，目前仅支持INNER与CLOS类型
    // INNER: 同Inner Group内Rank间互联
    // CLOS: 不同Rank经Fabric互联
    MAKE_ENUM(FabType, INNER, CLOS);
    std::unordered_map<u32, std::shared_ptr<TopoInstance>> topoInsts_;

    NetInstance(const u32 netLayer, const std::string& netInstId, const NetType netType);
    virtual ~NetInstance() = default;

    u32 GetNetLayer() const;
    std::string GetNetInstId() const;
    NetType GetNetType() const;
    std::set<RankId> GetRankIds() const;
    u32 GetRankSize() const;
    bool HasNode(const NodeId nodeId) const;
    const std::unordered_map<RankId, std::shared_ptr<Peer>>& GetPeers() const;
    const std::vector<std::shared_ptr<Fabric>>& GetFabrics() const;
    Graph<Node, Link>& GetGraph();
    void AddRankId(const RankId rankId);
    void AddNode(const std::shared_ptr<Node>& node);
    void AddLink(const std::shared_ptr<Link>& link);
    void DeleteLink(const NodeId srcNodeId, const NodeId dstNodeId);

    void UpdateTopoInst(u32 topoInstId, TopoType topoType, RankId rankId);
    void GetTopoInstsByLayer(std::vector<u32>& topoInsts, u32& topoInstNum) const;
    HcclResult GetTopoType(const u32 topoInstId, TopoType& topoType) const;
    HcclResult GetRanksByTopoInst(const u32 topoInstId, std::vector<u32>& ranks, u32& rankNum) const;
    virtual std::vector<Path> GetPaths(const RankId srcRankId, const RankId dstRankId) const = 0;
    std::string Describe() const;

protected:
    u32 netLayer{0};
    std::string netInstId{""};
    NetType netType{NetType::CLOS};
    std::set<RankId> rankIds;
    std::unordered_map<RankId, std::shared_ptr<Peer>> peers;
    std::unordered_map<LocalId, RankId> localIdsMap;
    std::vector<std::shared_ptr<Fabric>> fabrics;
    std::unordered_map<PlaneId, NodeId> planeId2Node; // 除了创建时，其他是否需要使用
    Graph<Node, Link> vGraph;

    void AddPeer(const std::shared_ptr<Peer>& peer);
    void AddFabric(const std::shared_ptr<Fabric>& fabric);
};

class InnerNetInstance : public NetInstance {
public:
    InnerNetInstance(const u32 netLayer, const std::string& netInstId)
        : NetInstance(netLayer, netInstId, NetType::TOPO_FILE_DESC) {};

    ~InnerNetInstance() override = default;

    std::vector<Path> GetPaths(const RankId srcRankId, const RankId dstRankId) const override;
};

class ClosNetInstance : public NetInstance {
public:
    ClosNetInstance(const u32 netLayer, const std::string& netInstId)
        : NetInstance(netLayer, netInstId, NetType::CLOS) {};

    ~ClosNetInstance() override = default;

    std::vector<Path> GetPaths(const RankId srcRankId, const RankId dstRankId) const override;
};

} // namespace Hccl

#endif // NET_INSTANCE_H
