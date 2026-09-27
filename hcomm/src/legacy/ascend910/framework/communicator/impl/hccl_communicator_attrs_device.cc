/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_communicator_attrs.h"

using namespace std;

namespace hccl {
HcclResult
HcclCommunicatorAttrs::Init([[maybe_unused]] HcclCommParams& params, [[maybe_unused]] const RankTable_t& rankTable)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::Init(
    [[maybe_unused]] HcclCommParams& params, [[maybe_unused]] const RankTable_t& rankTable,
    [[maybe_unused]] const std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoConfigMap)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::Init(
    [[maybe_unused]] HcclCommParams& params, [[maybe_unused]] const std::vector<RankInfo>& rankList,
    [[maybe_unused]] WorldGroupInfo& groupCommonData)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::Init(
    [[maybe_unused]] HcclCommParams& params, [[maybe_unused]] const std::vector<RankInfo>& rankList,
    [[maybe_unused]] WorldGroupInfo& groupCommonData,
    [[maybe_unused]] const std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoConfigMap)
{
    return HCCL_SUCCESS;
}

bool HcclCommunicatorAttrs::IsStandardCard() { return false; }

bool HcclCommunicatorAttrs::Is310PDuoCard() { return false; }

bool HcclCommunicatorAttrs::IsCommon310P3DUO([[maybe_unused]] const std::vector<RankInfo_t>& rankList) { return false; }

bool HcclCommunicatorAttrs::CompareWithUserRank(
    [[maybe_unused]] const RankInfo& left, [[maybe_unused]] const RankInfo& right)
{
    return false;
}

HcclResult HcclCommunicatorAttrs::CheckDeviceType([[maybe_unused]] const DevType deviceType) const
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::GetNicInfo(
    [[maybe_unused]] const NICDeployment& nicDeploy, [[maybe_unused]] const u32 curRankIndex,
    [[maybe_unused]] const std::vector<RankInfo_t>& servRankList, [[maybe_unused]] RankInfo& rankInfo) const
{
    return HCCL_SUCCESS;
}

// private
HcclResult HcclCommunicatorAttrs::InitCommParams([[maybe_unused]] HcclCommParams& params) { return HCCL_SUCCESS; }

HcclResult HcclCommunicatorAttrs::SetServerId([[maybe_unused]] const RankTable_t& rankTable) { return HCCL_SUCCESS; }

HcclResult HcclCommunicatorAttrs::SetServerNum([[maybe_unused]] const std::vector<RankInfo_t>& ranks)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetInnerServerAverageDevice([[maybe_unused]] const RankTable_t& rankTable)
{
    return HCCL_SUCCESS;
}

// sub group适配获取server内设配数
HcclResult HcclCommunicatorAttrs::SetInnerServerAverageDevice([[maybe_unused]] const std::vector<RankInfo>& rankList)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::TransformRankInfoByServerId(
    [[maybe_unused]] const std::vector<RankInfo_t>& rankList, [[maybe_unused]] ServRankInfo& servRankInfo) const
{
    return HCCL_SUCCESS;
}

bool HcclCommunicatorAttrs::CompareWithDevicePhyId(
    [[maybe_unused]] const RankInfo_t& left, [[maybe_unused]] const RankInfo_t& right)
{
    return false;
}

HcclResult HcclCommunicatorAttrs::SetModuleInfo([[maybe_unused]] const std::vector<RankInfo_t>& rankList)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetSuperPodInfo([[maybe_unused]] const std::vector<RankInfo_t>& rankList)
{
    return HCCL_SUCCESS;
}

// 集群中存在910B A+X时，0-7卡: moduleIdx = 2 * serverIdx; 8-15卡: moduleIdx = 2 * serverIdx + 1
// 集群中不存在910B A+X时，moduleIdx = serverIdx
HcclResult
HcclCommunicatorAttrs::GetModuleIdx([[maybe_unused]] const RankInfo_t& rankInfo, [[maybe_unused]] u32& moduleIdx)
{
    return HCCL_SUCCESS;
}

// 用于标识集群中是否存在 910B A+X形态
bool HcclCommunicatorAttrs::IsDiffDeviceModule([[maybe_unused]] const std::vector<RankInfo_t>& rankList) const
{
    return false;
}

HcclResult HcclCommunicatorAttrs::InitHccsPortNum() { return HCCL_SUCCESS; }

HcclResult HcclCommunicatorAttrs::SetRankInfoList([[maybe_unused]] const RankTable_t& rankTable)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::CheckRankTable(
    [[maybe_unused]] const RankTable_t& rankTable, [[maybe_unused]] const ServRankInfo& servRankInfo)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::CheckDevPhyId([[maybe_unused]] const s32& devicePhyId) const { return HCCL_SUCCESS; }

HcclResult HcclCommunicatorAttrs::SortRankInfoList() { return HCCL_SUCCESS; }

HcclResult HcclCommunicatorAttrs::CheckNicDeploy(
    [[maybe_unused]] NICDeployment nicDeploy, [[maybe_unused]] DevType deviceType) const
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::CheckDevCount([[maybe_unused]] const u32 devNum) { return HCCL_SUCCESS; }

bool HcclCommunicatorAttrs::Check2N([[maybe_unused]] u32 num) const { return false; }

HcclResult HcclCommunicatorAttrs::SetLocalRankInfo() { return HCCL_SUCCESS; }

HcclResult HcclCommunicatorAttrs::SetLocalRankInfoSubGroup([[maybe_unused]] const std::vector<RankInfo>& rankList)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::CheckLocalRankInfo() { return HCCL_SUCCESS; }

u32 HcclCommunicatorAttrs::CalMeshAggRankSize([[maybe_unused]] int halfDevNum) const { return 0; }

HcclResult HcclCommunicatorAttrs::SetMeshAggregationRankSize([[maybe_unused]] u32 size) { return HCCL_SUCCESS; }

HcclResult HcclCommunicatorAttrs::CalAndSetMeshAggRankSize() { return HCCL_SUCCESS; }

HcclResult HcclCommunicatorAttrs::SetWorldGroupInfo(
    [[maybe_unused]] std::unordered_map<std::string, std::map<u32, HcclIpAddress>>& phyIdNicInfoMap,
    [[maybe_unused]] std::vector<RankInfo>& worldRankInfoList, [[maybe_unused]] std::vector<u32>& nicRanksPort,
    [[maybe_unused]] std::vector<u32>& vnicRanksPort)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::TransformRankList(
    [[maybe_unused]] const std::vector<RankInfo>& rankListIn,
    [[maybe_unused]] std::vector<RankInfo_t>& rankListOut) const
{
    return HCCL_SUCCESS;
}

bool HcclCommunicatorAttrs::IsEnableRoce() { return false; }

// a+x mesh间需要同时保证ip有效和roce开关打开才能走rdma
bool HcclCommunicatorAttrs::IsUsedRdmaLevel0AndIpInvalid() { return false; }

bool HcclCommunicatorAttrs::IsSupportEnableRoce() { return false; }

void HcclCommunicatorAttrs::GetTopoAttr([[maybe_unused]] HcclTopoAttr& topoAttr) {}

void HcclCommunicatorAttrs::GetAlgoAttr([[maybe_unused]] HcclAlgoAttr& algoAttr) {}

u32 HcclCommunicatorAttrs::GetLocalNicPort([[maybe_unused]] NicType nicType) { return 0; }
} // namespace hccl
