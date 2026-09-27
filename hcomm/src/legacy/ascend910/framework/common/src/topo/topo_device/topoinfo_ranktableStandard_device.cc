/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktableStandard.h"

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <arpa/inet.h>

// ltm指定config路径
#include "common/src/config.h"
#include "hccl_comm_pub.h"
#include "workflow_pub.h"

using namespace std;
using namespace hccl;

TopoinfoRanktableStandard::TopoinfoRanktableStandard(const std::string& rankTableM, const std::string& identify)
    : TopoInfoRanktableParser(rankTableM, identify)
{}

TopoinfoRanktableStandard::~TopoinfoRanktableStandard() {}

HcclResult TopoinfoRanktableStandard::Init() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableStandard::GetSelfClusterInfo([[maybe_unused]] HcclCommParams& params)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetClusterInfo(
    [[maybe_unused]] hccl::HcclCommParams& params, [[maybe_unused]] hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}
HcclResult TopoinfoRanktableStandard::GetClusterInfo([[maybe_unused]] RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::ParserClusterInfo(
    [[maybe_unused]] hccl::HcclCommParams& params, [[maybe_unused]] hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetHcomInfo(
    [[maybe_unused]] hccl::HcclCommParams& params, [[maybe_unused]] hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetServerList(
    [[maybe_unused]] const nlohmann::json& obj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] hccl::RankTable_t& rankTable, [[maybe_unused]] u32 serverNum)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetSingleServer(
    [[maybe_unused]] const nlohmann::json& serverListObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetCloudHcomInfo(
    [[maybe_unused]] hccl::HcclCommParams& params, [[maybe_unused]] hccl::RankTable_t& rankTable,
    [[maybe_unused]] const std::string& identify, [[maybe_unused]] u32& rank)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetSortClouldRankList([[maybe_unused]] hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetSingleGroupDeviceCount(
    [[maybe_unused]] const nlohmann::json& obj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] hccl::RankTable_t& rankTable, [[maybe_unused]] u32& deviceNum)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetLabSingleGroup(
    [[maybe_unused]] const nlohmann::json& obj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] const hccl::HcclCommParams& params, [[maybe_unused]] hccl::RankTable_t& rankTable,
    [[maybe_unused]] u32 instanceNum)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetGroupList(
    [[maybe_unused]] hccl::HcclCommParams& params, [[maybe_unused]] hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetInstanceList(
    [[maybe_unused]] nlohmann::json& instanceList, [[maybe_unused]] hccl::HcclCommParams& params,
    [[maybe_unused]] hccl::RankTable_t& rankTable, [[maybe_unused]] u32 instanceNum, [[maybe_unused]] u32 deviceNum)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetCloudDevList(
    [[maybe_unused]] nlohmann::json& instanceList, [[maybe_unused]] u32 podIndex,
    [[maybe_unused]] nlohmann::json& deviceList, [[maybe_unused]] std::string& serverId,
    [[maybe_unused]] u32& serverIdx)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetDevList(
    [[maybe_unused]] const nlohmann::json& instanceList, [[maybe_unused]] u32 podIndex,
    [[maybe_unused]] const nlohmann::json& deviceList, [[maybe_unused]] const hccl::HcclCommParams& params,
    [[maybe_unused]] hccl::RankTable_t& rankTable, [[maybe_unused]] std::string& serverId,
    [[maybe_unused]] u32& serverIdx)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableStandard::GetDeployMode([[maybe_unused]] bool& cloudFlag) const
{
    return HCCL_E_NOT_SUPPORT;
}
