/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PHY_TOPO_H
#define PHY_TOPO_H

#include <vector>
#include <set>
#include <unordered_map>
#include <memory>
#include "topo_common_types.h"
#include "iterator.h"
#include "graph.h"

namespace Hccl {

class PhyTopo {
public:
    static std::unique_ptr<PhyTopo>& GetInstance();
    class ConnInterface {
    public:
        // 使用地址信息、位置信息、链路类型、链路协议构造接口
        ConnInterface(
            const std::set<std::string> inputPorts, const AddrPosition inputPos, const LinkType inputLinkType,
            std::set<LinkProtocol> inputLinkProtocols);
        std::set<std::string> GetPorts() const;
        AddrPosition GetPos() const;
        LinkType GetLinkType() const;
        std::set<LinkProtocol> GetLinkProtocols() const;
        std::string Describe() const;
        bool operator==(const ConnInterface& rhs) const;
        bool operator!=(const ConnInterface& rhs) const;

    private:
        std::set<std::string> ports{};
        AddrPosition pos{};
        LinkType linkType{};
        std::set<LinkProtocol> linkProtocols{};
    };

    class Node {
    public:
        using IfaceIterator = BaseConstIterator<std::vector, std::shared_ptr<PhyTopo::ConnInterface>>;
        MAKE_ENUM(NodeType, PEER, FABRIC);
        explicit Node(const NodeType inputType);
        NodeType GetType() const;
        void AddConnInterface(const std::shared_ptr<PhyTopo::ConnInterface>& interface);
        IfaceIterator IterIfaces() const;
        virtual std::string Describe() const;

    private:
        NodeType type{};
        std::vector<std::shared_ptr<PhyTopo::ConnInterface>> interfaces{};
    };

    class Peer : public Node {
    public:
        explicit Peer(const LocalId localId);
        LocalId GetLocalId() const;
        static NodeId GetId(const LocalId localId);
        std::string Describe() const override;

    private:
        LocalId localId{};
    };

    class Fabric : public Node {
    public:
        explicit Fabric();
        static NodeId GetId();
        std::string Describe() const override;
    };

    struct LinkAttributes {
        LinkType linktype;
        std::set<LinkProtocol> protocols;
    };

    class Link {
    public:
        // 使用源节点、目的节点、链路类型、链路协议、topo类型、topoInstId 构造链路
        Link(
            std::shared_ptr<PhyTopo::Node> inputSource, std::shared_ptr<PhyTopo::Node> inputTarget,
            const LinkAttributes& linkAttrs, const TopoType topoType, const u32 topoInstId);
        void SetSourceIface(std::shared_ptr<PhyTopo::ConnInterface> inputSourceIface);
        void SetTargetIface(std::shared_ptr<PhyTopo::ConnInterface> inputTargetIface);
        LinkType GetType() const;
        std::set<LinkProtocol> GetLinkProtocols() const;
        LinkDirection GetLinkDirection() const;
        TopoType GetTopoType() const;
        u32 GetTopoInstId() const;
        u32 GetHop() const;
        std::shared_ptr<PhyTopo::ConnInterface> GetSourceIFace();
        std::shared_ptr<PhyTopo::ConnInterface> GetTargetIFace();
        std::shared_ptr<PhyTopo::Node> GetSourceNode();
        std::shared_ptr<PhyTopo::Node> GetTargetNode();
        std::string Describe() const;

    private:
        std::shared_ptr<PhyTopo::ConnInterface> sourceIface{nullptr};
        std::shared_ptr<PhyTopo::ConnInterface> targetIface{nullptr}; // 如果target为Fabric节点，则为空
        std::shared_ptr<PhyTopo::Node> source{nullptr};
        std::shared_ptr<PhyTopo::Node> target{nullptr};
        std::set<LinkProtocol> linkProtocols{};
        LinkType linkType{};
        LinkDirection direction{LinkDirection::BOTH};
        TopoType topoType{TopoType::CLOS};
        u32 topoInstId{0};
        u32 hop{1};
    };

    void AddTopoGraph(const u32 netLayer, std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> topo);
    std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> GetTopoGraph(const u32 netLayer) const;
    void InitFinish();
    bool IsInitFinished() const;
    void Clear();
    void Dump() const;
    bool IsNetLayerExisted(const u32 netLayer) const;

private:
    std::unordered_map<u32, std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>>> topos;
    bool initFlag{false};
};
} // namespace Hccl

#endif // PHY_TOPO_H
