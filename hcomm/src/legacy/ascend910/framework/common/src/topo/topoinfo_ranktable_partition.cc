/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktable_partition.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace hccl {

TopoinfoRanktablePartition::TopoinfoRanktablePartition(
    hccl::HcclCommParams& globalParams, hccl::RankTable_t& globalRankTable)
    : globalParams_(globalParams),
      globalRankTable_(globalRankTable)
{}

TopoinfoRanktablePartition::~TopoinfoRanktablePartition() {}

HcclResult TopoinfoRanktablePartition::GenerateSubRankTable(
    const uint32_t rankNum, const uint32_t* rankIds, hccl::RankTable_t& subRankTable)
{
    subRankTable.nicDeploy = globalRankTable_.nicDeploy;
    std::unordered_map<uint32_t, size_t> rankInfoMap;
    for (size_t i = 0; i < globalRankTable_.rankList.size(); i++) {
        auto rankId = globalRankTable_.rankList[i].rankId;
        rankInfoMap[rankId] = i;
    }
    std::unordered_map<std::string, u32> serverIdMap;
    std::unordered_map<std::string, u32> superPodIdMap;
    std::unordered_set<uint32_t> rankIdSet;
    subRankTable.deviceNum = 0;
    for (size_t i = 0; i < rankNum; i++) {
        CHK_PTR_NULL(rankIds + i);
        uint32_t rankId = rankIds[i];
        CHK_PRT_RET(
            rankIdSet.find(rankId) != rankIdSet.end(),
            HCCL_ERROR(
                "[TopoinfoRanktablePartition][GenerateSubRankTable]errNo[0x%016llx], "
                "duplicated rankId[%u] in rankIds.",
                HCCL_ERROR_CODE(HCCL_E_PARA), rankId),
            HCCL_E_PARA);

        auto iter = rankInfoMap.find(rankId);
        CHK_PRT_RET(
            iter == rankInfoMap.end(),
            HCCL_ERROR(
                "[TopoinfoRanktablePartition][GenerateSubRankTable]errNo[0x%016llx], "
                "fail to find target rank[%u] in the global communicator.",
                HCCL_ERROR_CODE(HCCL_E_PARA), rankId),
            HCCL_E_PARA);

        hccl::RankInfo_t rankInfo = globalRankTable_.rankList[iter->second];
        serverIdMap.emplace(rankInfo.serverId, serverIdMap.size());
        superPodIdMap.emplace(rankInfo.superPodId, superPodIdMap.size());

        rankInfo.rankId = i;
        rankInfo.serverIdx = serverIdMap[rankInfo.serverId];
        rankInfo.superPodIdx = superPodIdMap[rankInfo.superPodId];
        subRankTable.rankList.emplace_back(rankInfo);

        if (rankInfo.deviceInfo.devicePhyId != HOST_DEVICE_ID) {
            subRankTable.deviceNum++;
        }
        HCCL_INFO(
            "[TopoinfoRanktablePartition][GenerateSubRankTable]"
            "Pick rank[%u] from global comm as rank[%u] in sub comm, "
            "severId[%s], serverIdx[%u], superPodId[%s], superDeviceId[%u], devicePhyId[%d].",
            rankId, i, rankInfo.serverId.c_str(), rankInfo.serverIdx, rankInfo.superPodId.c_str(),
            rankInfo.superDeviceId, rankInfo.deviceInfo.devicePhyId);
    }
    CHK_RET(GenerateSubSuperPodId(subRankTable));
    subRankTable.serverNum = serverIdMap.size();
    subRankTable.superPodNum = superPodIdMap.size();
    subRankTable.rankNum = rankNum;
    subRankTable.version = globalRankTable_.version;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktablePartition::GenerateSubSuperPodId(hccl::RankTable_t& subRankTable)
{
    std::map<std::string, std::vector<RankInfo_t*>> podGroupClusters;
    for (auto& rankInfo : subRankTable.rankList) {
        podGroupClusters[rankInfo.originalSuperPodId].emplace_back(&rankInfo);
    }
    std::set<std::string> superPodIdSet;
    std::map<std::string, std::pair<u32, u32>> superPodIdRanges; // 记录每个逻辑超节点的rank id范围
    for (auto& subCluster : podGroupClusters) {
        auto& subClusterInfo = subCluster.second;
        if (subClusterInfo.size() <= 1) {
            continue;
        }
        u32 groupId = 0;
        superPodIdSet.insert(subCluster.first);
        RankInfo_t preRank = *(subClusterInfo[0]);
        superPodIdRanges[preRank.superPodId] = {preRank.rankId, preRank.rankId}; // 初始化范围
        for (u32 i = 1; i < subClusterInfo.size(); ++i) {
            RankInfo_t& curRank = *(subClusterInfo[i]);
            // 当前的curRank和上一个preRank的rankId不连续，分配新的逻辑超节点ID
            if (curRank.rankId != preRank.rankId + 1) {
                std::string newSuperPodId = curRank.originalSuperPodId + "_HCCLSPLIT_" + std::to_string(groupId);
                curRank.superPodId = newSuperPodId;
                groupId++;
                superPodIdRanges[curRank.superPodId] = {curRank.rankId, curRank.rankId}; // 初始化新的范围
            } else {
                // 同一个sub通信域两个rank原始逻辑超节点是一致的
                // rankId连续 上一个rank的superPodId可能已经重新分配，需要更新当前superPodId为上一个rank的
                curRank.superPodId = preRank.superPodId;
                superPodIdRanges[curRank.superPodId].second = curRank.rankId; // 更新最大rank id
            }
            superPodIdSet.insert(curRank.superPodId);
            preRank = curRank;
        }
    }
    // 打印每个逻辑超节点的rank id范围，只打印包含_HCCLSPLIT_的逻辑超节点
    for (const auto& entry : superPodIdRanges) {
        auto superPodId = entry.first;
        if (superPodId.find("_HCCLSPLIT_") != std::string::npos) {
            auto range = entry.second;
            HCCL_RUN_INFO(
                "[TopoinfoRanktablePartition][%s]Split superPod, ID[%s], rank range[%u, %u]", __func__,
                superPodId.c_str(), range.first, range.second);
        }
    }
    subRankTable.superPodNum = superPodIdSet.size();
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktablePartition::GenerateSubParams(
    const hccl::RankTable_t& subRankTable, const uint32_t subCommRankId, hccl::HcclCommParams& subParams)
{
    subParams.rank = subCommRankId;
    subParams.userRank = subRankTable.rankList[subCommRankId].rankId;
    subParams.totalRanks = subRankTable.rankList.size();
    subParams.logicDevId = globalParams_.logicDevId;
    subParams.serverId = subRankTable.rankList[subCommRankId].serverId;
    subParams.deviceType = globalParams_.deviceType;
    subParams.commPortConfig.devPortSwitchOn = globalParams_.commPortConfig.devPortSwitchOn;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktablePartition::GetRankTableStr(const hccl::RankTable_t& subRankTable, std::string& rankTableStr)
{
    nlohmann::json basicJson;
    HcclResult ret = Struct2JsonRankTable(subRankTable, globalParams_.deviceType, basicJson);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_RUN_WARNING("cluster info to json failed, ret[%d].", ret), HCCL_E_INTERNAL);
    rankTableStr = std::move(basicJson.dump());
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktablePartition::TransformRankInfo(
    const RankTable_t& clusterInfo, nlohmann::json& perRankJson, u32 rankIndex)
{
    auto rankInfo = clusterInfo.rankList[rankIndex];
    perRankJson[PROP_HOST_IP] = std::string(rankInfo.hostIp.GetReadableIP());
    perRankJson[PROP_DEV_ID] = std::to_string(rankInfo.deviceInfo.devicePhyId);
    perRankJson[PROP_DEV_NIC_PORT] = std::to_string(rankInfo.deviceInfo.port);
    perRankJson[PROP_DEV_VNIC_PORT] = std::to_string(rankInfo.deviceInfo.vnicPort);
    perRankJson[PROP_BACKUP_DEV_PORT] = std::to_string(rankInfo.deviceInfo.backupPort);
    perRankJson[PROP_RANK_ID] = std::to_string(rankInfo.rankId);
    perRankJson[PROP_SERVER_ID] = rankInfo.serverId;
    perRankJson[PROP_SUPER_POD_ID] = rankInfo.superPodId;
    perRankJson[PROP_SUPER_DEVICE_ID] = std::to_string(rankInfo.superDeviceId);
    if (rankInfo.deviceInfo.deviceIp.size() != 0 && !rankInfo.deviceInfo.deviceIp[0].IsInvalid()) {
        perRankJson[PROP_DEV_IP] = std::string(rankInfo.deviceInfo.deviceIp[0].GetReadableIP());
    }
    if (clusterInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && rankInfo.deviceInfo.backupDeviceIp.size() != 0
        && !rankInfo.deviceInfo.backupDeviceIp[0].IsInvalid()) {
        perRankJson[PROP_BACKUP_DEV_IP] = std::string(rankInfo.deviceInfo.backupDeviceIp[0].GetReadableIP());
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktablePartition::TransformServerList(const RankTable_t& clusterInfo, nlohmann::json& rankListJson)
{
    for (size_t i = 0; i < clusterInfo.rankList.size(); i++) {
        nlohmann::json perRankJson;
        CHK_RET(TransformRankInfo(clusterInfo, perRankJson, i));
        perRankJson[PROP_RANK_ID] = perRankJson;
        rankListJson.push_back(perRankJson);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktablePartition::Struct2JsonRankTable(
    const RankTable_t& clusterInfo, const DevType deviceType, nlohmann::json& ClusterJson)
{
    ClusterJson[PROP_SERVER_COUNT] = std::to_string(clusterInfo.serverNum);
    ClusterJson[PROP_SUPER_POD_NUM] = std::to_string(clusterInfo.superPodNum);
    ClusterJson[PROP_RANK_NUM] = std::to_string(clusterInfo.rankNum);
    ClusterJson[PROP_DEV_NUM] = std::to_string(clusterInfo.deviceNum);

    nlohmann::json rankListJson;
    CHK_RET(TransformServerList(clusterInfo, rankListJson));
    ClusterJson[PROP_RANK_LIST] = rankListJson;

    ClusterJson[PROP_STATUS] = "completed";
    if (!clusterInfo.version.empty()) {
        ClusterJson[PROP_VERSION] = clusterInfo.version;
    } else {
        ClusterJson[PROP_VERSION] = (deviceType == DevType::DEV_TYPE_910_93) ? "1.2" : "1.0";
    }

    return HCCL_SUCCESS;
}
} // namespace hccl
