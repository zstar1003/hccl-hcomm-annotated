/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktableHeterog.h"

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <arpa/inet.h>

#include "externalinput_pub.h"
#include "hccl_comm_pub.h"
// ltm指定config路径
#include "common/src/config.h"
#include "workflow_pub.h"

using namespace std;
using namespace hccl;

TopoinfoRanktableHeterog::TopoinfoRanktableHeterog(
    const std::string& rankTableM, const std::string& identify, DevType deviceType)
    : TopoInfoRanktableParser(rankTableM, identify),
      deviceType_(deviceType)
{}

TopoinfoRanktableHeterog::~TopoinfoRanktableHeterog() {}

HcclResult TopoinfoRanktableHeterog::Init() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableHeterog::GetSelfClusterInfo([[maybe_unused]] HcclCommParams& params)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::GetClusterInfo(
    [[maybe_unused]] hccl::HcclCommParams& params, [[maybe_unused]] hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::GetClusterInfo([[maybe_unused]] RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::ParserClusterInfo(
    [[maybe_unused]] hccl::HcclCommParams& params, [[maybe_unused]] hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::GetRanktableInfo([[maybe_unused]] RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::CheckNicDeployConsistence([[maybe_unused]] RankTable_t& clusterInfo) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::CheckMode([[maybe_unused]] std::string& mode) const { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableHeterog::CheckHeterogSubVersion([[maybe_unused]] std::string& subVersion) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::GetHostPort([[maybe_unused]] const u32& localRank, [[maybe_unused]] u32& hostPort)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::GetRanks(
    [[maybe_unused]] const nlohmann::json& NodeListObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankTable_t& clusterInfo, [[maybe_unused]] std::string& serverId, [[maybe_unused]] u32& serverIdx,
    [[maybe_unused]] HcclIpAddress& nodeIp)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::GetSingleNode(
    [[maybe_unused]] const nlohmann::json& NodeListObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

// 91093暂定所有字段都是必选字段。除了ranks里面的 bind_device_id
HcclResult TopoinfoRanktableHeterog::GetSingleRank91093(
    [[maybe_unused]] const nlohmann::json& ranksObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankTable_t& clusterInfo, [[maybe_unused]] std::string& serverId, [[maybe_unused]] u32& serverIdx,
    [[maybe_unused]] HcclIpAddress& nodeIp)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableHeterog::GetSingleRank(
    [[maybe_unused]] const nlohmann::json& ranksObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankTable_t& clusterInfo, [[maybe_unused]] std::string& serverId, [[maybe_unused]] u32& serverIdx,
    [[maybe_unused]] HcclIpAddress& nodeIp)
{
    return HCCL_E_NOT_SUPPORT;
}
