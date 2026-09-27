/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_RANK_GROUP_H
#define HCCL_CCU_RANK_GROUP_H

#include <vector>
#include <utility>
#include <functional>
#include "types.h"
#include "virtual_topo.h"

namespace Hccl {

class RankGroup {
public:
    RankGroup() = default;
    explicit RankGroup(const std::vector<RankId>& ranks) : ranks(ranks) {}

    ~RankGroup() = default;

    // 添加RankId到ranks中
    void AddRank(RankId rankId) { ranks.emplace_back(rankId); }

    // 获取ranks
    std::vector<RankId> GetRanks() const { return ranks; }

private:
    std::vector<RankId> ranks;
};

struct LinkInfo {
    RankId rankId;
    u32 dieId;
    IpAddress localAddr;
    IpAddress remoteAddr;

    LinkInfo(RankId rId, u32 dId, IpAddress lAddr, IpAddress rAddr)
        : rankId(rId),
          dieId(dId),
          localAddr(lAddr),
          remoteAddr(rAddr) {};

    LinkInfo(const LinkData& linkdata)
        : rankId(linkdata.GetRemoteRankId()),
          dieId(linkdata.GetLocalDieId()),
          localAddr(linkdata.GetLocalAddr()),
          remoteAddr(linkdata.GetRemoteAddr()) {};

    explicit LinkInfo() : rankId(0), dieId(0), localAddr(IpAddress("0.0.0.1")), remoteAddr(IpAddress("0.0.0.1")) {};
};

class LinkGroup {
public:
    LinkGroup() = default;
    explicit LinkGroup(const std::vector<LinkInfo>& links) : links(links) {}

    ~LinkGroup() = default;

    // 添加linkData到links中
    void AddLink(LinkInfo linkInfo) { links.emplace_back(linkInfo); }

    // 获取links
    std::vector<LinkInfo> GetLinks() const { return links; }

private:
    std::vector<LinkInfo> links;
};

} // namespace Hccl

// 在全局作用域定义哈希函数
namespace std {
// 定义一个常量用于哈希计算中的乘法操作
constexpr size_t K_HASH_MULTIPLIER = 31;

template <>
class hash<Hccl::RankGroup> {
public:
    size_t operator()(const Hccl::RankGroup& rg) const
    {
        size_t hashValue = 0;
        for (const auto& id : rg.GetRanks()) {
            hashValue = hashValue * K_HASH_MULTIPLIER + hash<Hccl::RankId>()(id);
        }
        return hashValue;
    }
};

template <>
class equal_to<Hccl::RankGroup> {
public:
    bool operator()(const Hccl::RankGroup& rg1, const Hccl::RankGroup& rg2) const
    {
        if (rg1.GetRanks().size() != rg2.GetRanks().size()) {
            return false;
        } else {
            for (u32 i = 0; i < rg1.GetRanks().size(); i++) {
                if (rg1.GetRanks()[i] != rg2.GetRanks()[i]) {
                    return false;
                }
            }
        }
        return true;
    }
};

template <>
class hash<Hccl::LinkInfo> {
public:
    size_t operator()(const Hccl::LinkInfo& LinkInfo) const
    {
        auto rankIdHash = hash<Hccl::RankId>{}(LinkInfo.rankId);
        auto dieIdHash = hash<u32>{}(LinkInfo.dieId);
        auto localEidHash = hash<Hccl::IpAddress>{}(LinkInfo.localAddr);
        auto remoteEidHash = hash<Hccl::IpAddress>{}(LinkInfo.remoteAddr);

        return Hccl::HashCombine({rankIdHash, dieIdHash, localEidHash, remoteEidHash});
    }
};

template <>
class hash<Hccl::LinkGroup> {
public:
    size_t operator()(const Hccl::LinkGroup& rg) const
    {
        size_t hashValue = 0;
        for (const auto& id : rg.GetLinks()) {
            hashValue = hashValue * K_HASH_MULTIPLIER + hash<Hccl::LinkInfo>()(id);
        }
        return hashValue;
    }
};

template <>
class equal_to<Hccl::LinkGroup> {
public:
    bool operator()(const Hccl::LinkGroup& rg1, const Hccl::LinkGroup& rg2) const
    {
        if (rg1.GetLinks().size() != rg2.GetLinks().size()) {
            return false;
        } else {
            for (u32 i = 0; i < rg1.GetLinks().size(); i++) {
                if ((rg1.GetLinks()[i].rankId != rg2.GetLinks()[i].rankId)
                    || rg1.GetLinks()[i].dieId != rg2.GetLinks()[i].dieId) {
                    return false;
                }
            }
        }
        return true;
    }
};
} // namespace std

#endif // HCCL_CCU_RANK_GROUP_H
