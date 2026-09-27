/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktableConcise.h"
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <arpa/inet.h>

#include "log.h"
#include "env_config.h"
#include "hccl_comm_pub.h"
// ltm指定config路径
#include "common/src/config.h"
#include "workflow_pub.h"
#include "device_capacity.h"

using namespace std;
using namespace hccl;

TopoinfoRanktableConcise::TopoinfoRanktableConcise(const std::string& rankTableM, const std::string& identify)
    : TopoInfoRanktableParser(rankTableM, identify),
      isInterSuperPodRetryEnable_(GetExternalInputInterSuperPodRetryEnable())
{}

TopoinfoRanktableConcise::~TopoinfoRanktableConcise() {}

HcclResult TopoinfoRanktableConcise::Init() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoinfoRanktableConcise::GetClusterInfo([[maybe_unused]] RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSelfClusterInfo([[maybe_unused]] HcclCommParams& params)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetClusterInfo(
    [[maybe_unused]] hccl::HcclCommParams& params, [[maybe_unused]] hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::SetIsInterSuperPodRetryEnable([[maybe_unused]] bool isRetryEnable)
{
    return HCCL_E_NOT_SUPPORT;
}

void TopoinfoRanktableConcise::DetectNicDepoly([[maybe_unused]] RankTable_t& rankTable) { return; }

HcclResult TopoinfoRanktableConcise::ParserClusterInfo(
    [[maybe_unused]] hccl::HcclCommParams& params, [[maybe_unused]] hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::CheckNicDeployConsistence(
    [[maybe_unused]] RankTable_t& clusterInfo, [[maybe_unused]] NICDeployment deploy) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetRanktableInfo([[maybe_unused]] RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetServerList(
    [[maybe_unused]] const nlohmann::json& obj, [[maybe_unused]] RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleServer(
    [[maybe_unused]] const nlohmann::json& serverListObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetDeviceList(
    [[maybe_unused]] const nlohmann::json& serverListObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankTable_t& clusterInfo, [[maybe_unused]] std::string& serverId, [[maybe_unused]] u32& serverIdx,
    [[maybe_unused]] HcclIpAddress& hostIp)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleDevice(
    [[maybe_unused]] const nlohmann::json& deviceListObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankTable_t& clusterInfo, [[maybe_unused]] std::string& serverId, [[maybe_unused]] u32& serverIdx,
    [[maybe_unused]] HcclIpAddress& hostIp)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::SplitString(
    [[maybe_unused]] const std::string& str, [[maybe_unused]] const std::string& strC,
    [[maybe_unused]] std::vector<std::string>& strVector) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleDeviceIp(
    [[maybe_unused]] const nlohmann::json& deviceListObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] const RankTable_t& clusterInfo, [[maybe_unused]] RankInfo_t& rankinfo,
    [[maybe_unused]] DevType deviceType, [[maybe_unused]] bool invalidHostIp)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleBackupDeviceIp(
    [[maybe_unused]] const nlohmann::json& deviceListObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankInfo_t& rankinfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleDeviceHostPort(
    [[maybe_unused]] const nlohmann::json& deviceListObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankInfo_t& rankinfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleDevicePort(
    [[maybe_unused]] const nlohmann::json& deviceListObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankInfo_t& rankinfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleBackupDevicePort(
    [[maybe_unused]] const nlohmann::json& deviceListObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankInfo_t& rankinfo) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::VerifyBackupDeviceIpAndPort(
    [[maybe_unused]] std::vector<RankInfo_t>& rankList, [[maybe_unused]] u32 devIndex)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleSuperDeviceId(
    [[maybe_unused]] const nlohmann::json& deviceListObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] const RankTable_t& clusterInfo, [[maybe_unused]] RankInfo_t& rankinfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSuperPodList(
    [[maybe_unused]] const nlohmann::json& obj, [[maybe_unused]] RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleSuperPod(
    [[maybe_unused]] const nlohmann::json& superPodList, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSuperPodServerList(
    [[maybe_unused]] const nlohmann::json& superPodList, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankTable_t& clusterInfo, [[maybe_unused]] std::string superPodId)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleSuperPodSever(
    [[maybe_unused]] const nlohmann::json& superPodServerList, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankTable_t& clusterInfo, [[maybe_unused]] std::string superPodId)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::CheckSuperPodInfo([[maybe_unused]] RankTable_t& clusterInfo) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoinfoRanktableConcise::GetSingleNicInfo(
    [[maybe_unused]] const nlohmann::json& serverListObj, [[maybe_unused]] u32 objIndex,
    [[maybe_unused]] RankInfo_t& rankinfo)
{
    return HCCL_E_NOT_SUPPORT;
}
