/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_RANKTABLEOFFLINE_H
#define TOPOINFO_RANKTABLEOFFLINE_H

#include "topoinfo_ranktableParser_pub.h"
#include "hccl/base.h"
#include "hccl_types.h"

namespace hccl {
class TopoinfoRanktableOffline : public TopoInfoRanktableParser {
public:
    explicit TopoinfoRanktableOffline(const std::string& rankTableM);
    ~TopoinfoRanktableOffline() override;

    HcclResult Init() override;
    HcclResult GetClusterInfo(RankTable_t& clusterInfo) override;
    HcclResult GetDeviceNumPerServer(s32& deviceNum);

protected:
private:
    // 所有集群信息
    TopoinfoRanktableOffline(const TopoinfoRanktableOffline&);
    TopoinfoRanktableOffline& operator=(const TopoinfoRanktableOffline&);
    HcclResult ParserClusterInfo(hccl::RankTable_t& rankTable);
    HcclResult GetRanktableInfo(RankTable_t& clusterInfo);
    HcclResult GetSingleNode(const nlohmann::json& NodeListObj, u32 objIndex, RankTable_t& clusterInfo, u32& serverIdx);
    HcclResult GetSingleRank(
        const nlohmann::json& ranksObj, u32 objIndex, RankTable_t& clusterInfo, u32& serverIdx, std::string& nodeId);
    s32 deviceNumPerServer_;
    s32 curServerDeviceNum_;
};
} // namespace hccl
#endif
