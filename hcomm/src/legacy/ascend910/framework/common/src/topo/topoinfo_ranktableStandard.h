/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_RANKTABLESTANDARD_H
#define TOPOINFO_RANKTABLESTANDARD_H

#include <string>

#include "topoinfo_ranktableParser_pub.h"
#include "hccl/base.h"
#include "hccl_comm_pub.h"
#include "hccl_types.h"

namespace hccl {
class TopoinfoRanktableStandard : public TopoInfoRanktableParser {
public:
    explicit TopoinfoRanktableStandard(const std::string& rankTableM, const std::string& identify);
    ~TopoinfoRanktableStandard() override;

    HcclResult Init() override;
    HcclResult GetClusterInfo(RankTable_t& clusterInfo) override;
    HcclResult GetSelfClusterInfo(HcclCommParams& params);
    HcclResult GetClusterInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable) override;

protected:
private:
    // 类成员方法
    /* cloud场景解析rank_table */
    TopoinfoRanktableStandard(const TopoinfoRanktableStandard&);
    TopoinfoRanktableStandard& operator=(const TopoinfoRanktableStandard&);
    HcclResult GetCloudHcomInfo(
        hccl::HcclCommParams& params, hccl::RankTable_t& rankTable, const std::string& identify, u32& rank);
    HcclResult ParserClusterInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable);
    // 较验布置模式,cloud场景还是lab场景
    HcclResult GetDeployMode(bool& cloudFlag) const;

    HcclResult GetSortClouldRankList(hccl::RankTable_t& rankTable);

    HcclResult GetServerList(const nlohmann::json& obj, u32 objIndex, hccl::RankTable_t& rankTable, u32 serverNum);
    HcclResult GetSingleServer(const nlohmann::json& serverListObj, u32 objIndex, hccl::RankTable_t& rankTable);
    HcclResult GetHcomInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable);

    /* cloud场景解析rank_table */
    HcclResult GetGroupList(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable);
    HcclResult GetLabSingleGroup(
        const nlohmann::json& obj, u32 objIndex, const hccl::HcclCommParams& params, hccl::RankTable_t& rankTable,
        u32 instanceNum);
    HcclResult GetCloudDevList(
        nlohmann::json& instanceList, u32 podIndex, nlohmann::json& deviceList, std::string& serverId, u32& serverIdx);
    HcclResult GetDevList(
        const nlohmann::json& instanceList, u32 podIndex, const nlohmann::json& deviceList,
        const hccl::HcclCommParams& params, hccl::RankTable_t& rankTable, std::string& serverId, u32& serverIdx);
    HcclResult GetInstanceList(
        nlohmann::json& instanceList, hccl::HcclCommParams& params, hccl::RankTable_t& rankTable, u32 instanceNum,
        u32 deviceNum);

    HcclResult
    GetSingleGroupDeviceCount(const nlohmann::json& obj, u32 objIndex, hccl::RankTable_t& rankTable, u32& deviceNum);

    bool cloudFlag_ = false; // 默认为false
};
} // namespace hccl
#endif
