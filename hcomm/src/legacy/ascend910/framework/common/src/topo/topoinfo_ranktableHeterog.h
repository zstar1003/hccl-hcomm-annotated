/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_RANKTABLEHETEROG_H
#define TOPOINFO_RANKTABLEHETEROG_H

#include "topoinfo_ranktableParser_pub.h"
#include "hccl/base.h"
#include "hccl_comm_pub.h"
#include "hccl_types.h"

constexpr u32 HOST_PORT_USED = 1;
namespace hccl {
class TopoinfoRanktableHeterog : public TopoInfoRanktableParser {
public:
    explicit TopoinfoRanktableHeterog(
        const std::string& rankTableM, const std::string& identify, DevType deviceType = DevType::DEV_TYPE_310P1);
    ~TopoinfoRanktableHeterog() override;

    HcclResult Init() override;
    HcclResult GetClusterInfo(RankTable_t& clusterInfo) override;
    HcclResult GetSelfClusterInfo(HcclCommParams& params);
    HcclResult GetClusterInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable) override;

protected:
private:
    // 所有集群信息
    TopoinfoRanktableHeterog(const TopoinfoRanktableHeterog&);
    TopoinfoRanktableHeterog& operator=(const TopoinfoRanktableHeterog&);
    HcclResult ParserClusterInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable);
    HcclResult GetRanktableInfo(RankTable_t& clusterInfo);
    HcclResult GetSingleNode(const nlohmann::json& NodeListObj, u32 objIndex, RankTable_t& clusterInfo);
    HcclResult GetRanks(
        const nlohmann::json& NodeListObj, u32 objIndex, RankTable_t& clusterInfo, std::string& serverId,
        u32& serverIdx, HcclIpAddress& nodeIp);
    HcclResult CheckNicDeployConsistence(RankTable_t& clusterInfo) const;
    HcclResult GetSingleRank(
        const nlohmann::json& ranksObj, u32 objIndex, RankTable_t& clusterInfo, std::string& serverId, u32& serverIdx,
        HcclIpAddress& nodeIp);
    HcclResult GetSingleRank91093(
        const nlohmann::json& ranksObj, u32 objIndex, RankTable_t& clusterInfo, std::string& serverId, u32& serverIdx,
        HcclIpAddress& nodeIp);
    HcclResult CheckMode(std::string& mode) const;
    HcclResult CheckHeterogSubVersion(std::string& subVersion) const;
    HcclResult GetHostPort(const u32& localRank, u32& hostPort);

    bool is91093_{false};
    DevType deviceType_;
    std::map<u32, u32> hostPortMap_;
};
} // namespace hccl
#endif // TOPOINFO_RANKTABLE_HETEROG_PARSER_VER1_1_H
