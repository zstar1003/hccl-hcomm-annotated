/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_VIRTUAL_TOPO_H
#define HCCLV2_VIRTUAL_TOPO_H

#include <cassert>
#include <map>
#include <set>
#include <cstdint>
#include "port.h"
#include "iterator.h"
#include "dev_type.h"
#include "log.h"
#include "net_instance.h"
#include "rank_gph.h"

namespace Hccl {

using namespace std;

MAKE_ENUM(PeerType, CPU, NPU)

static constexpr u32 MAX_LINK_PATH_NUM = 2;

class LinkData {
public:
    // 待修改 构造函数不对外开发，LinkData只能由Link生成
    LinkData(BasePortType portType, RankId localRankId, RankId remoteRankId, u32 localPortId, u32 remotePortId)
        : type(portType.GetType()),
          linkProtocol_(ConnProto2LinkProtocol(portType.GetProto())),
          localRankId_(localRankId),
          remoteRankId_(remoteRankId),
          localPortId_(localPortId),
          remotePortId_(remotePortId) {};
    LinkData(
        PortDeploymentType portDeploymentType, LinkProtocol linkProtocol, RankId localRankId, RankId remoteRankId,
        IpAddress localAddr, IpAddress remoteAddr, u32 devicePhyId = 0, u32 remoteDevicePhyId = 0, u32 reuseIdx = 0)
        : type(portDeploymentType),
          linkProtocol_(linkProtocol),
          localRankId_(localRankId),
          remoteRankId_(remoteRankId),
          localAddr_(localAddr),
          remoteAddr_(remoteAddr),
          localDeviceId_(devicePhyId),
          remoteDeviceId_(remoteDevicePhyId),
          reuseIdx_(reuseIdx) {};

    explicit LinkData(const NetInstance::Path& path)
    {
        if (path.links.size() == 1) {
            auto link = path.links[0];
            auto srcPeer = link.GetSourceNode();
            auto targetPeer = link.GetTargetNode();
            shared_ptr<NetInstance::ConnInterface> srcConnIface = link.GetSourceIface();
            auto targetConnIface = link.GetTargetIface();
            linkProtocol_ = *link.GetLinkProtocols().begin();
            type = AddrPos2PortDeploymentType(srcConnIface->GetPos(), linkProtocol_);
            localRankId_ = std::dynamic_pointer_cast<NetInstance::Peer>(srcPeer)->GetRankId();
            remoteRankId_ = std::dynamic_pointer_cast<NetInstance::Peer>(targetPeer)->GetRankId();
            localDeviceId_ = std::dynamic_pointer_cast<NetInstance::Peer>(srcPeer)->GetDeviceId();
            remoteDeviceId_ = std::dynamic_pointer_cast<NetInstance::Peer>(targetPeer)->GetDeviceId();
            localAddr_ = srcConnIface->GetAddr();
            remoteAddr_ = targetConnIface->GetAddr();
            localDieId_ = srcConnIface->GetLocalDieId();
            hop = path.links[0].GetHop();
            fullmesh = true; // 单链路场景，标识为fullmesh
        } else if (path.links.size() == MAX_LINK_PATH_NUM) {
            auto link0 = path.links[0];
            auto link1 = path.links[1];
            auto srcPeer = link0.GetSourceNode();
            auto targetPeer = link1.GetTargetNode();
            auto srcConnIface = link0.GetSourceIface();
            auto targetConnIface = link1.GetTargetIface();
            linkProtocol_ = *link0.GetLinkProtocols().begin();
            type = AddrPos2PortDeploymentType(srcConnIface->GetPos(), linkProtocol_);
            localRankId_ = std::dynamic_pointer_cast<NetInstance::Peer>(srcPeer)->GetRankId();
            remoteRankId_ = std::dynamic_pointer_cast<NetInstance::Peer>(targetPeer)->GetRankId();
            localDeviceId_ = std::dynamic_pointer_cast<NetInstance::Peer>(srcPeer)->GetDeviceId();
            remoteDeviceId_ = std::dynamic_pointer_cast<NetInstance::Peer>(targetPeer)->GetDeviceId();
            localAddr_ = srcConnIface->GetAddr();
            remoteAddr_ = targetConnIface->GetAddr();
            localDieId_ = srcConnIface->GetLocalDieId();
            hop = path.links[0].GetHop();
            portGroupSize = static_cast<u8>(srcConnIface->GetPorts().size());
            auto tgtPortGroupSize = static_cast<u8>(targetConnIface->GetPorts().size());
            if (portGroupSize != tgtPortGroupSize) {
                HCCL_WARNING(
                    "[LinkData][Constructor]srcConnIface.portGroupSize[%u] "
                    "is not equal to targetConnIface.portGroupSize[%u]",
                    static_cast<u32>(portGroupSize), static_cast<u32>(tgtPortGroupSize));
            }
            fullmesh = false; // 多链路场景，非fullmesh
        } else {
            HCCL_ERROR("[LinkData][Constructor]path.links.size()[%u] is invalid", path.links.size());
            fullmesh = false; // 无效场景，默认为false
        }
        UpdateIpAddrWithPCIE();
        direction = path.direction;

        localPortId_ = 0;
        remotePortId_ = 0;
    }

    explicit LinkData(vector<char>& data);

    std::vector<char> GetUniqueId() const;

    bool operator==(const LinkData& rhs) const
    {
        return type == rhs.type && linkProtocol_ == rhs.linkProtocol_ && localRankId_ == rhs.localRankId_
               && remoteRankId_ == rhs.remoteRankId_ && localAddr_ == rhs.localAddr_ && remoteAddr_ == rhs.remoteAddr_
               && hop == rhs.hop && direction == rhs.direction && portGroupSize == rhs.portGroupSize
               && fullmesh == rhs.fullmesh && reuseIdx_ == rhs.reuseIdx_;
    }

    bool operator!=(const LinkData& rhs) const { return !(rhs == *this); }

    bool operator<(const LinkData& rhs) const
    {
        if (type < rhs.type) {
            return true;
        }
        if (rhs.type < type) {
            return false;
        }
        if (linkProtocol_ < rhs.linkProtocol_) {
            return true;
        }
        if (rhs.linkProtocol_ < linkProtocol_) {
            return false;
        }
        if (localRankId_ < rhs.localRankId_) {
            return true;
        }
        if (rhs.localRankId_ < localRankId_) {
            return false;
        }
        if (remoteRankId_ < rhs.remoteRankId_) {
            return true;
        }
        if (rhs.remoteRankId_ < remoteRankId_) {
            return false;
        }
        if (localAddr_ < rhs.localAddr_) {
            return true;
        }
        if (rhs.localAddr_ < localAddr_) {
            return false;
        }
        if (remoteAddr_ < rhs.remoteAddr_) {
            return true;
        }
        if (rhs.remoteAddr_ < remoteAddr_) {
            return false;
        }
        if (hop < rhs.hop) {
            return true;
        }
        if (rhs.hop < hop) {
            return false;
        }
        if (direction < rhs.direction) {
            return true;
        }
        if (rhs.direction < direction) {
            return false;
        }
        if (rhs.portGroupSize < portGroupSize) {
            return false;
        }
        if (fullmesh == false && rhs.fullmesh == true) {
            return true;
        }
        if (rhs.fullmesh == false && fullmesh == true) {
            return false;
        }
        if (localPortId_ < rhs.localPortId_) {
            return true;
        }
        if (rhs.localPortId_ < localPortId_) {
            return false;
        }
        if (reuseIdx_ < rhs.reuseIdx_) {
            return true;
        }
        if (rhs.reuseIdx_ < reuseIdx_) {
            return false;
        }
        return remotePortId_ < rhs.remotePortId_;
    }

    string Describe() const
    {
        return StringFormat(
            "LinkData:type=%s, protocol=%s, localRankId=%d, localAddr=%s, remoteRankId=%d, "
            "remoteAddr=%s, reuseIdx=%u",
            type.Describe().c_str(), linkProtocol_.Describe().c_str(), localRankId_, localAddr_.Describe().c_str(),
            remoteRankId_, remoteAddr_.Describe().c_str(), reuseIdx_);
    };

    PortData GetLocalPort() const
    {
        return {localRankId_, type, LinkProtocol2LinkProtoType(linkProtocol_), localPortId_, localAddr_};
    };

    PortData GetRemotePort() const
    {
        return {remoteRankId_, type, LinkProtocol2LinkProtoType(linkProtocol_), remotePortId_, remoteAddr_};
    };

    bool IsSymetric(const LinkData& rhs) const
    {
        return (type == rhs.type) && (linkProtocol_ == rhs.linkProtocol_) && (localRankId_ == rhs.remoteRankId_)
               && (remoteRankId_ == rhs.localRankId_) && (localAddr_ == rhs.remoteAddr_)
               && (remoteAddr_ == rhs.localAddr_) && (hop == rhs.hop) && (direction == rhs.direction);
    };

    const PortDeploymentType& GetType() const { return type; };

    const LinkProtocol& GetLinkProtocol() const { return linkProtocol_; }

    u32 GetHop() const { return hop; }

    LinkDirection GetDirection() const { return direction; }

    RankId GetLocalRankId() const { return localRankId_; };

    RankId GetRemoteRankId() const { return remoteRankId_; };

    DeviceId GetRemoteDeviceId() const { return remoteDeviceId_; };

    u32 GetLocalPortId() const { return localPortId_; };

    u32 GetRemotePortId() const { return remotePortId_; };

    const IpAddress& GetLocalAddr() const { return localAddr_; };

    const IpAddress& GetRemoteAddr() const { return remoteAddr_; };

    u32 GetLocalDieId() const { return localDieId_; };

    u8 GetPortGroupSize() const { return portGroupSize; };

    bool Readable() const { return readable; };

    bool Writable() const { return writable; };
    void UpdateIpAddrWithPCIE();

    bool GetFullmesh() const { return fullmesh; };

    std::string GetReuseIdx() const { return std::to_string(reuseIdx_); };

private:
    PortDeploymentType type;
    LinkProtocol linkProtocol_;
    RankId localRankId_{0};
    RankId remoteRankId_{0};
    u32 localPortId_{0};
    u32 remotePortId_{0};
    IpAddress localAddr_;
    IpAddress remoteAddr_;
    bool readable{true};
    bool writable{true};
    u32 hop{0};
    LinkDirection direction;
    u32 localDieId_{};
    u8 portGroupSize{1};
    DeviceId localDeviceId_{UINT32_MAX};
    DeviceId remoteDeviceId_{UINT32_MAX};
    bool fullmesh{false}; // 标识是否为全互联单链路场景
    u32 reuseIdx_{0};     // socket复用idx，加在socket建链tag后面
};
} // namespace Hccl

namespace std {

template <>
class hash<Hccl::LinkData> {
public:
    size_t operator()(const Hccl::LinkData& linkData) const
    {
        auto typeHash = hash<uint8_t>{}(linkData.GetType());
        auto linkProtoHash = hash<uint8_t>{}(linkData.GetLinkProtocol());
        auto localRankIdHash = hash<Hccl::RankId>{}(linkData.GetLocalRankId());
        auto remoteRankIdHash = hash<Hccl::RankId>{}(linkData.GetRemoteRankId());
        auto localPortIdHash = hash<u32>{}(linkData.GetLocalPortId());
        auto remotePortIdHash = hash<u32>{}(linkData.GetRemotePortId());
        auto localAddrHash = hash<Hccl::IpAddress>{}(linkData.GetLocalAddr());
        auto remoteAddrHash = hash<Hccl::IpAddress>{}(linkData.GetRemoteAddr());
        auto portGrpSizeHash = hash<uint8_t>{}(linkData.GetPortGroupSize());
        auto fullmeshHash = hash<bool>{}(linkData.GetFullmesh());

        return Hccl::HashCombine(
            {typeHash, linkProtoHash, localRankIdHash, remoteRankIdHash, localPortIdHash, remotePortIdHash,
             localAddrHash, remoteAddrHash, portGrpSizeHash, fullmeshHash});
    }
};
} // namespace std

#endif // HCCLV2_VIRTUAL_TOPO_H
