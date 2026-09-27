/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktableParser_pub.h"

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <arpa/inet.h>

#include "hccl_comm_pub.h"
#include "topoinfo_ranktableStandard.h"
#include "topoinfo_ranktableConcise.h"
#include "topoinfo_ranktableHeterog.h"
#include "config.h"
#include "adapter_error_manager_pub.h"
#include "json_utils.h"

using namespace std;
using namespace hccl;

HcclResult CheckAverageDev([[maybe_unused]] u32 uDeviceNum, [[maybe_unused]] u32 uServerNum)
{
    return HCCL_E_NOT_SUPPORT;
}

TopoInfoRanktableParser::TopoInfoRanktableParser(const std::string& rankTableM, const std::string& identify)
    : rankTableFile_(rankTableM),
      identify_(identify),
      statusCompleted_(false),
      uniqueInfoCheckPool_(static_cast<u32>(JsonUniqueInfoType::UNIQUE_INFO_NUM)),
      devMap_()
{}

TopoInfoRanktableParser::~TopoInfoRanktableParser() {}

HcclResult TopoInfoRanktableParser::ReadFile([[maybe_unused]] const std::string& readFile)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoRanktableParser::LoadFile([[maybe_unused]] const std::string& file) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoRanktableParser::LoadRankTableString([[maybe_unused]] const std::string& string)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoRanktableParser::LoadConfigString([[maybe_unused]] const std::string& string)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoRanktableParser::LoadString([[maybe_unused]] const std::string& string)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoRanktableParser::GetClusterInfo([[maybe_unused]] RankTable_t& clusterInfo)
{
    return HCCL_E_NOT_SUPPORT;
}
HcclResult TopoInfoRanktableParser::GetClusterInfo(
    [[maybe_unused]] hccl::HcclCommParams& params, [[maybe_unused]] hccl::RankTable_t& rankTable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoRanktableParser::Init() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoRanktableParser::LoadFileInit([[maybe_unused]] std::string& rankTableM)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoRanktableParser::SetIsInterSuperPodRetryEnable([[maybe_unused]] bool isRetryEnable)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoRanktableParser::GetRanktableVersion([[maybe_unused]] std::string& version)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoRanktableParser::RefreshStatus() { return HCCL_E_NOT_SUPPORT; }

bool TopoInfoRanktableParser::IsReady() const { return this->statusCompleted_; }

HcclResult TopoInfoRanktableParser::GetJsonProperty(
    [[maybe_unused]] const nlohmann::json& obj, [[maybe_unused]] const char* propName,
    [[maybe_unused]] std::string& propValue, [[maybe_unused]] bool optionalProp) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoRanktableParser::GetJsonProperty(
    [[maybe_unused]] const nlohmann::json& obj, [[maybe_unused]] const char* propName,
    [[maybe_unused]] nlohmann::json& propValue, [[maybe_unused]] bool optionalProp) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoRanktableParser::GetJsonArrayMemberProperty(
    [[maybe_unused]] const nlohmann::json& obj, [[maybe_unused]] const u32 index, [[maybe_unused]] const char* propName,
    [[maybe_unused]] std::string& propValue, [[maybe_unused]] bool optionalProp) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoRanktableParser::GetJsonArrayMemberProperty(
    [[maybe_unused]] const nlohmann::json& obj, [[maybe_unused]] const u32 index, [[maybe_unused]] const char* propName,
    [[maybe_unused]] u32& propValue, [[maybe_unused]] bool optionalProp) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoRanktableParser::GetJsonArrayMemberProperty(
    [[maybe_unused]] const nlohmann::json& obj, [[maybe_unused]] const u32 index, [[maybe_unused]] const char* propName,
    [[maybe_unused]] nlohmann::json& propValue, [[maybe_unused]] bool optionalProp) const
{
    return HCCL_E_NOT_SUPPORT;
}

/* 依据检查类型进行入参内容检查，并将检查选项转为strType带出以便后续信息打印 */
HcclResult TopoInfoRanktableParser::CheckUniquePara(
    [[maybe_unused]] const JsonUniqueInfoType& type, [[maybe_unused]] const std::string& value,
    [[maybe_unused]] string& strType) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoRanktableParser::CheckUniqueAndInsertPool(
    [[maybe_unused]] const JsonUniqueInfoType& type, [[maybe_unused]] const std::string& value,
    [[maybe_unused]] const JsonCheckOpType& opType)
{
    return HCCL_E_NOT_SUPPORT;
}

void TopoInfoRanktableParser::GenerateServerIdx(
    [[maybe_unused]] const std::string& serverId, [[maybe_unused]] u32& serverIdx)
{
    return;
}

void TopoInfoRanktableParser::GenerateSuperPodIdx(
    [[maybe_unused]] const std::string& superPodId, [[maybe_unused]] u32& superPodIdx)
{
    return;
}

HcclResult TopoInfoRanktableParser::ConvertIpAddress(
    [[maybe_unused]] const std::string& ipStr, [[maybe_unused]] HcclIpAddress& ipAddr)
{
    return HCCL_E_NOT_SUPPORT;
}
