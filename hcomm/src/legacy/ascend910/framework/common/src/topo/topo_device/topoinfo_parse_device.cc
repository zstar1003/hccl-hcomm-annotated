/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_parse.h"
#include <string>
#include <unordered_set>
#include <algorithm>
// ltm指定config路径
#include "common/src/config.h"

using namespace std;

namespace hccl {
TopoInfoParse::TopoInfoParse() {}

TopoInfoParse::~TopoInfoParse() {}

HcclResult TopoInfoParse::Init(
    [[maybe_unused]] const RankTable_t& rankTable, [[maybe_unused]] const std::string& serverId,
    [[maybe_unused]] const u32 deviceNumPerServer)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoParse::Init(
    [[maybe_unused]] const std::vector<RankInfo>& rankList, [[maybe_unused]] const std::string& serverId,
    [[maybe_unused]] const u32 deviceNumPerServer)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoParse::GetServerInnerLinkInfo(
    [[maybe_unused]] std::unordered_map<u32, u32>& pairLinkCounter,
    [[maybe_unused]] std::unordered_map<u32, std::unordered_map<int, std::vector<int>>>& pairLinkInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoParse::TransformRankInfoByServerId([[maybe_unused]] std::vector<hccl::RankInfo>& serverInnerInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

// nicIdx不只是校验，还有修改
HcclResult TopoInfoParse::ParseAndCheck([[maybe_unused]] std::vector<u32>& nicIdx) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoParse::Check() { return HCCL_E_NOT_SUPPORT; }

// server间device选取是否对称校验
HcclResult TopoInfoParse::CheckInterServerDeviceId() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoParse::CheckAndAssignNicInfo([[maybe_unused]] std::vector<u32>& nicIdx)
{
    return HCCL_E_NOT_SUPPORT;
}

// nicIdx做填充，nicIdx也是deviceId，deviceId做过的校验这里不再重复
HcclResult TopoInfoParse::CheckRankTableNicInfo([[maybe_unused]] std::vector<u32>& nicIdx)
{
    return HCCL_E_NOT_SUPPORT;
}

// 校验server内4p场景下device选取是否合法，2p与标卡场景重合
HcclResult TopoInfoParse::CheckServerInnerRankInfo() { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoParse::IsAllRankSamePlane([[maybe_unused]] bool& isAllRankSamePlane) { return HCCL_E_NOT_SUPPORT; }

HcclResult TopoInfoParse::IsSingleMeshAggregation([[maybe_unused]] bool& isSingleMeshAggregation)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoParse::IsAllRankConnectedWithHCCS([[maybe_unused]] bool& isAllRankConnectedWithHCCS)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TopoInfoParse::GetDeviceNumInPerMeshAggregation(
    [[maybe_unused]] u32 devicePhyId, [[maybe_unused]] u32& perAggregationNum)
{
    return HCCL_E_NOT_SUPPORT;
}
} // namespace hccl
