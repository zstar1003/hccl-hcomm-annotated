/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_PORT_H
#define HCCLV2_PORT_H

#include <memory>
#include <string>
#include <vector>

#include "types.h"
#include "ip_address.h"
#include "hash_utils.h"
#include "invalid_params_exception.h"
#include "topo_common_types.h"
#include "net_instance.h"

namespace Hccl {

using namespace std;

MAKE_ENUM(PortDeploymentType, P2P, DEV_NET, HOST_NET)

MAKE_ENUM(ConnectProtoType, HCCS, PCIE, TCP, RDMA, UB, UBOE, UB_RTP)

MAKE_ENUM(LinkProtoType, HCCS_PCIE, TCP, RDMA, UB)

inline PortDeploymentType AddrPos2PortDeploymentType(AddrPosition addrPosition, LinkProtocol linkProtocol)
{
    PortDeploymentType portDeploymentType{};
    if (addrPosition == AddrPosition::DEVICE) {
        if (linkProtocol == LinkProtocol::PCIE) {
            portDeploymentType = PortDeploymentType::P2P;
        } else {
            portDeploymentType = PortDeploymentType::DEV_NET;
        }
    } else if (addrPosition == AddrPosition::HOST) {
        portDeploymentType = PortDeploymentType::HOST_NET;
    } else {
        THROW<NotSupportException>(
            StringFormat("[AddrPos2PortDeploymentType] addrPosition[%s].", addrPosition.Describe().c_str()));
    }
    return portDeploymentType;
}

inline LinkProtoType LinkProtocol2LinkProtoType(LinkProtocol linkProtocol)
{
    LinkProtoType linkType{};
    if (linkProtocol == LinkProtocol::UB_CTP || linkProtocol == LinkProtocol::UB_TP
        || linkProtocol == LinkProtocol::UB_MEM || linkProtocol == LinkProtocol::UBOE
        || linkProtocol == LinkProtocol::UB_RTP) {
        linkType = LinkProtoType::UB;
    } else if (linkProtocol == LinkProtocol::ROCE) {
        linkType = LinkProtoType::RDMA;
    } else if (linkProtocol == LinkProtocol::PCIE) {
        linkType = LinkProtoType::HCCS_PCIE;
    } else {
        THROW<NotSupportException>(StringFormat(
            "[LinkProtocol2LinkProtoType] linkProtocol[%s] don't support.", linkProtocol.Describe().c_str()));
    }
    return linkType;
}

inline LinkProtoType ConnProto2LinkProto(ConnectProtoType connType)
{
    LinkProtoType linkType{};
    if (connType == ConnectProtoType::HCCS || connType == ConnectProtoType::PCIE) {
        linkType = LinkProtoType::HCCS_PCIE;
    } else if (connType == ConnectProtoType::TCP) {
        linkType = LinkProtoType::TCP;
    } else if (connType == ConnectProtoType::RDMA) {
        linkType = LinkProtoType::RDMA;
    } else if (
        connType == ConnectProtoType::UB || connType == ConnectProtoType::UBOE
        || connType == ConnectProtoType::UB_RTP) {
        linkType = LinkProtoType::UB;
    }
    HCCL_INFO("[ConnProto2LinkProto] linkType is[%s]", linkType.Describe().c_str());
    return linkType;
}

// 该函数仅用于内部构造函数，主流程不使用
inline LinkProtocol ConnProto2LinkProtocol(ConnectProtoType connType)
{
    LinkProtocol linkProto{};
    if (connType == ConnectProtoType::HCCS || connType == ConnectProtoType::PCIE) {
        linkProto = LinkProtocol::HCCS;
    } else if (connType == ConnectProtoType::TCP) {
        linkProto = LinkProtocol::TCP;
    } else if (connType == ConnectProtoType::RDMA) {
        linkProto = LinkProtocol::ROCE;
    } else if (connType == ConnectProtoType::UB) {
        linkProto = LinkProtocol::UB_CTP;
    } else if (connType == ConnectProtoType::UBOE) {
        linkProto = LinkProtocol::UBOE;
    } else if (connType == ConnectProtoType::UB_RTP) {
        linkProto = LinkProtocol::UB_RTP;
    }
    HCCL_INFO("[ConnProto2LinkProtocol] linkProto is[%s]", linkProto.Describe().c_str());
    return linkProto;
}

class BasePortType {
public:
    BasePortType(const BasePortType&) = default;
    BasePortType& operator=(const BasePortType&) = default;

    inline PortDeploymentType GetType() const { return type_; };

    inline ConnectProtoType GetProto() const { return proto_; };

    explicit BasePortType(PortDeploymentType type) : type_(type) {};

    bool operator==(const BasePortType& rhs) const { return type_ == rhs.type_ && proto_ == rhs.proto_; }

    bool operator!=(const BasePortType& rhs) const { return !(rhs == *this); }

    bool operator<(const BasePortType& rhs) const
    {
        if (type_ < rhs.type_)
            return true;
        if (rhs.type_ < type_)
            return false;
        return proto_ < rhs.proto_;
    }

    BasePortType(PortDeploymentType type, ConnectProtoType proto) : type_(type), proto_(proto) {};

    string Describe() const
    {
        return StringFormat("PortType[type=%s, proto=%s]", type_.Describe().c_str(), proto_.Describe().c_str());
    }

protected:
    PortDeploymentType type_;
    ConnectProtoType proto_;
};

class P2PPortType {
public:
    P2PPortType(ConnectProtoType proto) : portType_(PortDeploymentType::P2P, proto)
    {
        if (proto != ConnectProtoType::HCCS && proto != ConnectProtoType::PCIE) {
            THROW<InvalidParamsException>(StringFormat("P2PPortType::P2PPortType proto invalid"));
        }
    };

    PortDeploymentType GetType() const { return portType_.GetType(); }

    ConnectProtoType GetProto() const { return portType_.GetProto(); }

    bool operator==(const BasePortType& rhs) const { return portType_ == rhs; }

    bool operator!=(const BasePortType& rhs) const { return portType_ != rhs; }

    bool operator<(const BasePortType& rhs) const { return portType_ < rhs; }

    string Describe() const { return portType_.Describe(); }

    operator BasePortType() const { return portType_; }

private:
    BasePortType portType_;
};

class DevNetPortType {
public:
    DevNetPortType(ConnectProtoType proto) : portType_(PortDeploymentType::DEV_NET, proto)
    {
        if (proto != ConnectProtoType::TCP && proto != ConnectProtoType::RDMA && proto != ConnectProtoType::UB) {
            THROW<InvalidParamsException>(StringFormat("DevNetPortType::DevNetPortType proto invalid"));
        }
    };

    PortDeploymentType GetType() const { return portType_.GetType(); }

    ConnectProtoType GetProto() const { return portType_.GetProto(); }

    bool operator==(const BasePortType& rhs) const { return portType_ == rhs; }

    bool operator!=(const BasePortType& rhs) const { return portType_ != rhs; }

    bool operator<(const BasePortType& rhs) const { return portType_ < rhs; }

    string Describe() const { return portType_.Describe(); }

    operator BasePortType() const { return portType_; }

private:
    BasePortType portType_;
};

class HostNetPortType {
public:
    HostNetPortType(ConnectProtoType proto) : portType_(PortDeploymentType::HOST_NET, proto)
    {
        if (proto != ConnectProtoType::TCP && proto != ConnectProtoType::RDMA && proto != ConnectProtoType::UB) {
            THROW<InvalidParamsException>(StringFormat("HostNetPortType::HostNetPortType proto invalid"));
        }
    };

    PortDeploymentType GetType() const { return portType_.GetType(); }

    ConnectProtoType GetProto() const { return portType_.GetProto(); }

    bool operator==(const BasePortType& rhs) const { return portType_ == rhs; }

    bool operator!=(const BasePortType& rhs) const { return portType_ != rhs; }

    bool operator<(const BasePortType& rhs) const { return portType_ < rhs; }

    string Describe() const { return portType_.Describe(); }

    operator BasePortType() const { return portType_; }

private:
    BasePortType portType_;
};

class PortData {
public:
    PortData(RankId rankId, const BasePortType& type, u32 id, const IpAddress& addr)
        : rankId(rankId),
          type(type.GetType()),
          protoType(ConnProto2LinkProto(type.GetProto())),
          id(id),
          addr(addr)
    {}

    PortData(RankId rankId, PortDeploymentType type, LinkProtoType protoType, u32 id, const IpAddress& addr)
        : rankId(rankId),
          type(type),
          protoType(protoType),
          id(id),
          addr(addr)
    {}

    PortData(RankId rankId, const NetInstance::ConnInterface& connIface)
        : rankId(rankId),
          type(AddrPos2PortDeploymentType(connIface.GetPos(), *connIface.GetLinkProtocols().begin())),
          protoType(LinkProtocol2LinkProtoType(*connIface.GetLinkProtocols().begin())),
          id(0),
          addr(connIface.GetAddr())
    {}

    string Describe() const
    {
        return StringFormat(
            "PortData[rankId=%d, type=%s, id=%d, addr=%s]", rankId, type.Describe().c_str(), id,
            addr.Describe().c_str());
    }

    RankId GetRankId() const { return rankId; }

    const PortDeploymentType& GetType() const { return type; }

    const LinkProtoType& GetProto() const { return protoType; }

    u32 GetId() const { return id; }

    const IpAddress& GetAddr() const { return addr; }

    bool operator==(const PortData& rhs) const
    {
        return type == rhs.type && id == rhs.id
               && addr == rhs.addr; // TODO: rankId后面应该要删，rankId == rhs.rankId &&
    }

    bool operator!=(const PortData& rhs) const { return !(rhs == *this); }

    bool operator<(const PortData& rhs) const
    {
        if (rankId < rhs.rankId) {
            return true;
        }
        if (rhs.rankId < rankId) {
            return false;
        }
        if (type < rhs.type) {
            return true;
        }
        if (rhs.type < type)
            return false;
        if (addr < rhs.addr) {
            return true;
        }
        if (rhs.addr < addr) {
            return false;
        }
        return id < rhs.id;
    }

    bool operator>(const PortData& rhs) const { return rhs < *this; }

    bool operator<=(const PortData& rhs) const { return !(rhs < *this); }

    bool operator>=(const PortData& rhs) const { return !(*this < rhs); }

private:
    RankId rankId;
    PortDeploymentType type;
    LinkProtoType protoType;
    u32 id;
    IpAddress addr;
};
} // namespace Hccl

namespace std {

template <>
class hash<Hccl::PortData> {
public:
    size_t operator()(const Hccl::PortData& portData) const
    {
        auto typeHash = hash<uint8_t>{}(portData.GetType());
        auto protoHash = hash<uint8_t>{}(portData.GetProto());
        auto addrHash = hash<Hccl::IpAddress>{}(portData.GetAddr());

        return Hccl::HashCombine({addrHash, typeHash, protoHash});
    }
};

template <>
class equal_to<Hccl::PortData> {
public:
    bool operator()(const Hccl::PortData& p1, const Hccl::PortData& p2) const
    {
        return p1.GetAddr() == p2.GetAddr() && p1.GetType() == p2.GetType() && p1.GetProto() == p2.GetProto();
    }
};
} // namespace std

#endif // HCCLV2_PORT_H
