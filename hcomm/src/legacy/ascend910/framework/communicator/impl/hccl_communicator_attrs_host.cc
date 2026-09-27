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
#include "device_capacity.h"
#include "common/src/config.h"
#include "externalinput_pub.h"
#include "env_config.h"

using namespace std;

namespace hccl {
HcclResult HcclCommunicatorAttrs::Init(HcclCommParams& params, const RankTable_t& rankTable)
{
    CHK_RET(InitCommParams(params));
    CHK_RET(InitRankInfo(rankTable));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::Init(
    HcclCommParams& params, const RankTable_t& rankTable,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoConfigMap)
{
    algoConfigMap_ = algoConfigMap;
    CHK_RET(InitCommParams(params));
    CHK_RET(InitRankInfo(rankTable));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::Init(
    HcclCommParams& params, const std::vector<RankInfo>& rankList, WorldGroupInfo& groupCommonData)
{
    CHK_RET(InitCommParams(params));
    CHK_RET(InitRankInfoSubGroup(rankList, groupCommonData));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::Init(
    HcclCommParams& params, const std::vector<RankInfo>& rankList, WorldGroupInfo& groupCommonData,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoConfigMap)
{
    algoConfigMap_ = algoConfigMap;
    CHK_RET(InitCommParams(params));
    CHK_RET(InitRankInfoSubGroup(rankList, groupCommonData));
    return HCCL_SUCCESS;
}

bool HcclCommunicatorAttrs::IsStandardCard()
{
    if (Is310P3Common()) {
        HCCL_INFO("The current device only supports the StandardCard case.");
        return true;
    }
    if (isDiffDeviceType_) {
        return false;
    }
    return (
        (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == 0)
        && (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)].size() == 0)
        && (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::SIO_TYPE)].size() == 0));
}

bool HcclCommunicatorAttrs::Is310PDuoCard()
{
    return (Is310P3Common() && (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == userRankSize_));
}

bool HcclCommunicatorAttrs::IsCommon310P3DUO(const std::vector<RankInfo_t>& rankList)
{
    std::vector<u32> devIdList;
    std::vector<std::vector<u32>> checkDevList;
    checkDevList.resize(FACTOR_NUM_TWO);

    for (RankInfo_t rankInfo : rankList) {
        u32 curId = rankInfo.deviceInfo.devicePhyId;
        devIdList.push_back(curId);
    }
    if (devIdList.size() == DEVICE_PER_MODULE) {
        return true;
    }
    std::sort(devIdList.begin(), devIdList.end());
    for (u32 i = 0; i < devIdList.size(); i++) {
        if (devIdList[i] % FACTOR_NUM_TWO == 0) {
            checkDevList[0].push_back(devIdList[i]); // 主die
        } else {
            checkDevList[1].push_back(devIdList[i]); // 从die
        }
    }
    if (devIdList.size() == (DEVICE_PER_MODULE / FACTOR_NUM_TWO) && checkDevList[0].size() == checkDevList[1].size()) {
        return ((checkDevList[1][0] - checkDevList[0][0]) == 1) && ((checkDevList[1][1] - checkDevList[0][1]) != 0);
    } else {
        return false;
    }
    return false;
}

bool HcclCommunicatorAttrs::CompareWithUserRank(const RankInfo& left, const RankInfo& right)
{
    return left.userRank < right.userRank;
}

HcclResult HcclCommunicatorAttrs::CheckDeviceType(const DevType deviceType) const
{
    if ((deviceType >= DevType::DEV_TYPE_COUNT) || (deviceType < DevType::DEV_TYPE_910)) {
        HCCL_ERROR(
            "[Check][DeviceType]errNo[0x%016llx] device Type[%d] out of range[%d, %d]", HCCL_ERROR_CODE(HCCL_E_PARA),
            deviceType, DevType::DEV_TYPE_910, DevType::DEV_TYPE_NOSOC);
        return HCCL_E_PARA;
    }
    HCCL_INFO("[HcclCommunicatorAttrs][CheckDeviceType] CheckDeviceType done");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::GetNicInfo(
    const NICDeployment& nicDeploy, const u32 curRankIndex, const std::vector<RankInfo_t>& servRankList,
    RankInfo& rankInfo) const
{
    CHK_PRT_RET(
        servRankList.empty(),
        HCCL_ERROR("[Get][NicInfo]errNo[0x%016llx] server rank list is empty", HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);
    CHK_PRT_RET(
        curRankIndex >= servRankList.size(),
        HCCL_ERROR(
            "[Get][NicInfo]rankindex[%u] invalid,rank list "
            "size is[%zu]",
            curRankIndex, servRankList.size()),
        HCCL_E_PARA);
    rankInfo.nicDeploy = nicDeploy;
    const RankInfo_t& curRankInfo = servRankList[curRankIndex];
    if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST && curRankInfo.deviceInfo.deviceIp.size() == 0) {
        // 检查网卡个数
        // 网卡挂载位置在host时，按rank index从网卡列表中获取
        rankInfo.nicIp.push_back(curRankInfo.hostIp);
    } else {
        CHK_PRT_RET(
            curRankInfo.deviceInfo.deviceIp.size() == 0,
            HCCL_ERROR("[Get][NicInfo]rankindex[%u] invalid,deviceIp is zero", curRankIndex), HCCL_E_PARA);
        rankInfo.nicIp.push_back(curRankInfo.deviceInfo.deviceIp[0]);
        if (curRankInfo.deviceInfo.backupDeviceIp.size() == 0) {
            HcclIpAddress invalidAddr;
            rankInfo.backupNicIp.push_back(invalidAddr);
        } else {
            rankInfo.backupNicIp.push_back(curRankInfo.deviceInfo.backupDeviceIp[0]);
        }
        rankInfo.deviceNicPort = curRankInfo.deviceInfo.port;
        rankInfo.deviceVnicPort = curRankInfo.deviceInfo.vnicPort;
        rankInfo.backupDevicePort = curRankInfo.deviceInfo.backupPort;
        HCCL_INFO(
            "[Get][NicInfo]serverId[%s], serverIdx[%u], rankIndex[%u], nicIp[%s], backupNicIp[%s], "
            "deviceNicPort[%u], deviceVnicPort[%u], backupDevicePort[%u]",
            rankInfo.serverId.c_str(), rankInfo.serverIdx, curRankIndex, rankInfo.nicIp[0].GetReadableIP(),
            rankInfo.backupNicIp[0].GetReadableIP(), rankInfo.deviceNicPort, rankInfo.deviceVnicPort,
            rankInfo.backupDevicePort);
    }
    HCCL_INFO("[HcclCommunicatorAttrs][GetNicInfo] GetNicInfo done");
    return HCCL_SUCCESS;
}

// private
HcclResult HcclCommunicatorAttrs::InitCommParams(HcclCommParams& params)
{
    userRank_ = params.rank;
    realUserRank_ = params.userRank;
    userRankSize_ = params.totalRanks;
    deviceLogicId_ = params.logicDevId;
    deviceType_ = params.deviceType;

    identifier_ = params.identifier;
    collectiveId_ = params.id.internal;
    commWorkMode_ = params.commWorkMode;
    HCCL_DEBUG(
        "userRank_: %u realUserRank_: %u userRankSize_: %u deviceLogicId_: %u deviceType_: %u commWorkMode_: %u.",
        userRank_, realUserRank_, userRankSize_, deviceLogicId_, deviceType_, commWorkMode_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetServerId(const RankTable_t& rankTable)
{
    for (u32 i = 0; i < rankTable.rankList.size(); i++) {
        if (rankTable.rankList[i].rankId == userRank_) {
            serverId_ = rankTable.rankList[i].serverId;
            superPodId_ = rankTable.rankList[i].superPodId;
            superDeviceId_ = rankTable.rankList[i].superDeviceId;
            break;
        }
    }

    if (serverId_.empty()) {
        HCCL_ERROR("[Set][ServerId]SetServerId fail");
        return HCCL_E_PARA;
    }
    HCCL_INFO("[HcclCommunicatorAttrs][SetServerId] SetServerId done");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetServerNum(const std::vector<RankInfo_t>& ranks)
{
    std::vector<std::string> serverIds;
    for (u32 index = 0; index < ranks.size(); index++) {
        std::vector<std::string>::iterator found = find(serverIds.begin(), serverIds.end(), ranks[index].serverId);
        if (found == serverIds.end()) {
            serverIds.push_back(ranks[index].serverId);
        }
    }
    serverNum_ = serverIds.size();
    HCCL_INFO("[HcclCommunicatorAttrs][SetServerNum] SetServerNum done");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetInnerServerAverageDevice(const RankTable_t& rankTable)
{
    deviceNumPerServer_ = 0;
    bool isConnectedWithHCCS = true;
    LinkTypeInServer linkType = LinkTypeInServer::HCCS_TYPE;
    for (u32 i = 0; i < rankTable.rankList.size(); i++) {
        // 同一server的标识IP 是一样的，所以可以以此推算出平均dev个数
        if (rankTable.rankList[i].deviceInfo.devicePhyId == HOST_DEVICE_ID && isHaveCpuRank_ != true) {
            isHaveCpuRank_ = true;
        }
        if (serverId_ == rankTable.rankList[i].serverId
            && rankTable.rankList[i].deviceInfo.devicePhyId != HOST_DEVICE_ID) {
            deviceNumPerServer_++;
        } else {
            continue;
        }
        if (Is310PDevice()) {
            continue;
        }
        CHK_RET(GetPairDeviceLinkType(rankTable, i, isConnectedWithHCCS, linkType));
    }
    if (deviceType_ == DevType::DEV_TYPE_910B && !isConnectedWithHCCS) {
        deviceNumPerAggregation_ = deviceNumPerServer_ / FACTOR_NUM_TWO;
    } else {
        deviceNumPerAggregation_ = deviceNumPerServer_;
    }
    return HCCL_SUCCESS;
}

// sub group适配获取server内设配数
HcclResult HcclCommunicatorAttrs::SetInnerServerAverageDevice(const std::vector<RankInfo>& rankList)
{
    deviceNumPerServer_ = 0;
    bool isConnectedWithHCCS = true;
    LinkTypeInServer linkType = LinkTypeInServer::HCCS_TYPE;
    for (u32 i = 0; i < rankList.size(); i++) {
        // 同一server的标识IP 是一样的，所以可以以此推算出平均dev个数
        if (serverId_ == rankList[i].serverId && rankList[i].devicePhyId != HOST_DEVICE_ID) {
            deviceNumPerServer_++;
        } else {
            continue;
        }
        if (Is310PDevice() || isHaveCpuRank_) {
            // 异构场景无需获取链路类型并校验
            continue;
        }
        for (u32 j = i + 1; j < rankList.size(); j++) {
            if (rankList[i].serverId == rankList[j].serverId) {
                CHK_RET(hrtGetPairDeviceLinkType(rankList[i].devicePhyId, rankList[j].devicePhyId, linkType));
            }
            if (linkType != LinkTypeInServer::HCCS_TYPE) {
                isConnectedWithHCCS = false;
            }
        }
    }
    if (deviceType_ == DevType::DEV_TYPE_910B && !isConnectedWithHCCS) {
        deviceNumPerAggregation_ = deviceNumPerServer_ / FACTOR_NUM_TWO;
    } else {
        deviceNumPerAggregation_ = deviceNumPerServer_;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::TransformRankInfoByServerId(
    const std::vector<RankInfo_t>& rankList, ServRankInfo& servRankInfo) const
{
    for (size_t index = 0; index < rankList.size(); ++index) {
        const RankInfo_t& rankInfo = rankList[index];
        std::string serverId = SalTrim(rankInfo.serverId);
        ServRankInfo::iterator itr = servRankInfo.find(serverId);
        if (itr != servRankInfo.end()) {
            itr->second.push_back(rankInfo);
        } else {
            std::vector<RankInfo_t> rankInfoList;
            rankInfoList.push_back(rankInfo);
            std::pair<std::string, std::vector<RankInfo_t>> rankInfoPair(serverId, rankInfoList);
            servRankInfo.insert(rankInfoPair);
        }
    }
    // 每个server下的rank列表按  设备Id 从小到大的顺序排序
    for (auto& iter : servRankInfo) {
        std::sort(iter.second.begin(), iter.second.end(), CompareWithDevicePhyId);
    }
    return HCCL_SUCCESS;
}

bool HcclCommunicatorAttrs::CompareWithDevicePhyId(const RankInfo_t& left, const RankInfo_t& right)
{
    return left.deviceInfo.devicePhyId < right.deviceInfo.devicePhyId;
}

HcclResult HcclCommunicatorAttrs::SetModuleInfo(const std::vector<RankInfo_t>& rankList)
{
    isDiffDeviceModule_ = IsDiffDeviceModule(rankList);
    multiModuleDiffDeviceNumMode_ = false;
    moduleNum_ = serverNum_;

    std::map<u32, std::vector<RankInfo_t>> moduleMap;
    for (RankInfo_t rankInfo : rankList) {
        u32 moduleIdx = INVALID_UINT;
        CHK_RET(GetModuleIdx(rankInfo, moduleIdx)); // 这里不判断混合组网，只提取每个server实际的moduleidx
        if (static_cast<s32>(rankInfo.deviceInfo.devicePhyId) == HOST_DEVICE_ID) {
            continue;
        }
        auto iter = moduleMap.find(moduleIdx);
        if (iter == moduleMap.end()) {
            std::vector<RankInfo_t> rankInfoList;
            rankInfoList.push_back(rankInfo);
            moduleMap.insert(std::make_pair(moduleIdx, rankInfoList));
        } else {
            iter->second.push_back(rankInfo);
        }
    }
    if (moduleMap.size() == 0) {
        return HCCL_SUCCESS;
    }

    std::vector<u32> moduleDeviceNumVec;

    moduleNum_ = moduleMap.size();
    u32 preDeviceNum = moduleMap.begin()->second.size();
    u32 curDeviceNum = preDeviceNum;
    std::vector<u32> devicePhyIdInfoList;
    for (auto& moduleInfo : moduleMap) {
        curDeviceNum = moduleInfo.second.size();
        if (curDeviceNum != preDeviceNum) {
            multiModuleDiffDeviceNumMode_ = true;
        }

        moduleDeviceNumVec.push_back(curDeviceNum);

        HCCL_INFO("module[%d] contains [%d]devices", moduleInfo.first, moduleInfo.second.size());
        devicePhyIdInfoList.clear();
        for (auto& rankInfo : moduleInfo.second) {
            devicePhyIdInfoList.push_back(rankInfo.deviceInfo.devicePhyId);
            HCCL_INFO(
                "moduleIdx[%d] Info: rankId[%d], serverId[%s], serverIdx[%d], devicePhyId[%d]", moduleInfo.first,
                rankInfo.rankId, rankInfo.serverId.c_str(), rankInfo.serverIdx, rankInfo.deviceInfo.devicePhyId);
        }
        if (!CheckDoubleRingWithRohTopo(devicePhyIdInfoList)) {
            isARSDoubleRing_ = false;
            HCCL_DEBUG("SetModuleInfo isARSDoubleRing[%llu]", isARSDoubleRing_);
        }
    }

    if (isDiffDeviceType_) {
        gcdDeviceNumPerAggregation_ = CalGCD(moduleDeviceNumVec);
        multiModuleDiffDeviceNumMode_ = false;
        deviceNumPerAggregation_ = gcdDeviceNumPerAggregation_;
        useSuperPodMode_ = false;
        HCCL_INFO(
            "[HcclCommunicatorAttrs][SetModuleInfo]mix mode, set multiModuleDiffDeviceNumMode to false, "
            "gcdDeviceNumPerAggregation [%u] deviceNumPerAggregation [%u]",
            gcdDeviceNumPerAggregation_, deviceNumPerAggregation_);
    }

    HCCL_RUN_INFO("different module contains different numbers of cards:[%d]", multiModuleDiffDeviceNumMode_);
    HCCL_RUN_INFO("different module contains different type of cards:[%d]", isDiffDeviceType_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetSuperPodInfo(const std::vector<RankInfo_t>& rankList)
{
    // 1.超节点数目 2.超节点间server数是否一致 3.
    superPodNum_ = 0;
    multiSuperPodDiffServerNumMode_ = false;
    multiSuperPodDiffDeviceNumMode_ = false;
    std::map<std::string, std::set<u32>> superPodToServerNum;           // 记录每个超节点中的server数目
    std::map<std::string, std::vector<RankInfo_t>> superPodToDeviceNum; // 记录每个超节点中的设备(rank)数目
    for (RankInfo_t rankInfo : rankList) {
        // superPodId为空时, 返回超节点数量为0, 按照非超节点模式处理
        CHK_PRT_RET(
            rankInfo.superPodId.empty(),
            HCCL_DEBUG(
                "ranks[%u] superPodId[%s] is empty, set superPodNum to zero", rankInfo.rankId,
                rankInfo.superPodId.c_str()),
            HCCL_SUCCESS);

        superPodToServerNum[rankInfo.superPodId].insert(rankInfo.serverIdx);
        auto iter = superPodToDeviceNum.find(rankInfo.superPodId);
        if (iter == superPodToDeviceNum.end()) {
            std::vector<RankInfo_t> rankInfoList;
            rankInfoList.push_back(rankInfo);
            superPodToDeviceNum.insert(std::make_pair(rankInfo.superPodId, rankInfoList));
        } else {
            iter->second.push_back(rankInfo);
        }
    }
    superPodNum_ = superPodToServerNum.size();
    std::vector<u32> superPodServerNumVec;
    u32 preServerNum = superPodToServerNum.begin()->second.size();
    u32 curServerNum = preServerNum;
    for (auto superPodItem : superPodToServerNum) {
        curServerNum = superPodItem.second.size();
        if (curServerNum != preServerNum) {
            multiSuperPodDiffServerNumMode_ = true;
        }
        superPodServerNumVec.push_back(curServerNum);
        HCCL_INFO(
            "[Set][SuperPodInfo]SuperPod[%s] contains [%d]servers", superPodItem.first.c_str(),
            superPodItem.second.size());
    }
    HCCL_RUN_INFO(
        "[Set][SuperPodInfo]different superPod contains different numbers of servers:[%d]",
        multiSuperPodDiffServerNumMode_);

    for (auto item : superPodToDeviceNum) {
        u32 curDeviceNum = item.second.size();
        if (curDeviceNum != superPodToDeviceNum.begin()->second.size()) {
            multiSuperPodDiffDeviceNumMode_ = true;
        }
        HCCL_INFO("[Set][SuperPodInfo]SuperPod[%s] contains [%d] devices", item.first.c_str(), item.second.size());
    }

    // 计算最大公约数，切分超节点
    if (!IsConfigAHCAlgo(algoConfigMap_) && !multiModuleDiffDeviceNumMode_ && multiSuperPodDiffServerNumMode_) {
        gcdServerNumPerSuperPod_ = CalGCD(superPodServerNumVec);
        multiSuperPodDiffServerNumMode_ = false; // 取公约数不存在server数不一致场景
        multiSuperPodDiffDeviceNumMode_ = false; // 切分后各超节点server数一致，device数也一致
        superPodNum_ = serverNum_ / gcdServerNumPerSuperPod_;
        HCCL_RUN_INFO(
            "[Set][SuperPodInfo] gcdServerNumPerSuperPod[%u] original superPodNum[%u] converted superPodNum[%u]",
            gcdServerNumPerSuperPod_, superPodToServerNum.size(), superPodNum_);
    }

    if (isDiffDeviceType_) {
        multiSuperPodDiffServerNumMode_ = false;
        HCCL_RUN_INFO("mix mode, set multiSuperPodDiffServerNumMode to false");
    }
    return HCCL_SUCCESS;
}

// 集群中存在910B A+X时，0-7卡: moduleIdx = 2 * serverIdx; 8-15卡: moduleIdx = 2 * serverIdx + 1
// 集群中不存在910B A+X时，moduleIdx = serverIdx
HcclResult HcclCommunicatorAttrs::GetModuleIdx(const RankInfo_t& rankInfo, u32& moduleIdx)
{
    CHK_PRT_RET(
        rankInfo.serverIdx == INVALID_UINT,
        HCCL_ERROR("serverIdx is invalid:[%u], rankId:[%u]", rankInfo.serverIdx, rankInfo.rankId), HCCL_E_PARA);
    CHK_PRT_RET(
        deviceType_ == DevType::DEV_TYPE_COUNT,
        HCCL_ERROR("deviceType_ is invalid:[%d], rankId:[%u]", deviceType_, rankInfo.rankId), HCCL_E_PARA);
    u32 serverIdx = rankInfo.serverIdx;
    if (GetRankInfoDevType(rankInfo) == DevType::DEV_TYPE_910B && isDiffDeviceModule_) {
        moduleIdx = serverIdx * FACTOR_NUM_TWO + rankInfo.deviceInfo.devicePhyId / DEVICE_PER_MODULE;
    } else if (isDiffDeviceType_) {
        moduleIdx = serverIdx * FACTOR_NUM_TWO;
    } else {
        moduleIdx = serverIdx;
    }
    CHK_PRT_RET(
        moduleIdx == INVALID_UINT,
        HCCL_ERROR("GetModuleIdx failed. moduleIdx:[%d], rankId:[%u]", moduleIdx, rankInfo.rankId), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

// 用于标识集群中是否存在 910B A+X形态
bool HcclCommunicatorAttrs::IsDiffDeviceModule(const std::vector<RankInfo_t>& rankList) const
{
    bool minDevice = false;
    bool maxDevice = false;
    bool isDiffMeshAggregation = false;
    if (!isDiffDeviceType_ && (deviceType_ != DevType::DEV_TYPE_910B || rankList.size() == 0)) {
        HCCL_INFO("[IsDiffDeviceModule] deviceType_[%d], rankList.size[%u]", deviceType_, rankList.size());
        return false;
    }

    for (const RankInfo_t& rankInfo : rankList) {
        if (GetRankInfoDevType(rankInfo) == DevType::DEV_TYPE_910B && !isStandardCard_) {
            if (rankInfo.deviceInfo.devicePhyId < DEVICE_PER_MODULE) {
                minDevice = true;
            } else {
                maxDevice = true;
            }
        }
    }
    if (minDevice && maxDevice) {
        isDiffMeshAggregation = true;
    }
    return isDiffMeshAggregation;
}

HcclResult HcclCommunicatorAttrs::InitHccsPortNum()
{
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_910_93) {
        CHK_RET(hrtGetHccsPortNum(deviceLogicId_, hccsPortNum_));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetRankInfoList(const RankTable_t& rankTable)
{
    // 检查rank table入参正确性
    CHK_RET(CheckRankTable(rankTable, servRankInfo_));
    // 获取芯片类型
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    // 遍历rank table获取rank信息
    rankInfoList_.clear();
    for (auto iter = servRankInfo_.begin(); iter != servRankInfo_.end(); ++iter) {
        for (u32 index = 0; index < iter->second.size(); ++index) {
            const RankInfo_t& orgRankInfo = iter->second[index];
            // 构建comm 使用的rank 信息
            RankInfo rankInfo;
            rankInfo.userRank = orgRankInfo.rankId;
            rankInfo.worldRank = orgRankInfo.rankId;

            rankInfo.deviceType = GetRankInfoDevType(orgRankInfo);
            CHK_RET(CheckDeviceType(rankInfo.deviceType));

            if (rankInfo.deviceType != DevType::DEV_TYPE_910B || rankInfo.deviceType != DevType::DEV_TYPE_910_93) {
                // 910B、910_93形态不做devicePhyId最大值的判断
                CHK_RET(CheckDevPhyId(orgRankInfo.deviceInfo.devicePhyId));
            }
            rankInfo.devicePhyId = orgRankInfo.deviceInfo.devicePhyId;
            rankInfo.deviceNicPort = orgRankInfo.deviceInfo.port;
            rankInfo.deviceVnicPort = orgRankInfo.deviceInfo.vnicPort;

            rankInfo.serverId = orgRankInfo.serverId;
            rankInfo.serverIdx = orgRankInfo.serverIdx;
            rankInfo.hostIp = orgRankInfo.hostIp;
            rankInfo.hostPort = orgRankInfo.hostPort;
            rankInfo.localRank = orgRankInfo.localRank;
            rankInfo.superDeviceId = orgRankInfo.superDeviceId;
            if (gcdServerNumPerSuperPod_ > 0) {
                u32 gcdSuperPodIdx = rankInfo.serverIdx / gcdServerNumPerSuperPod_;
                rankInfo.superPodId = orgRankInfo.superPodId + "_" + std::to_string(gcdSuperPodIdx);
                rankInfo.superPodIdx = gcdSuperPodIdx;
                if (userRank_ == rankInfo.userRank) {
                    HCCL_RUN_INFO(
                        "[SetRankInfoList] userRank[%u] serverId[%s] serverIdx[%u] original superPodId[%s] "
                        "superPodIdx[%u] converted superPodId[%s] superPodIdx[%u]",
                        userRank_, rankInfo.serverId.c_str(), rankInfo.serverIdx, orgRankInfo.superPodId.c_str(),
                        orgRankInfo.superPodIdx, rankInfo.superPodId.c_str(), rankInfo.superPodIdx);
                }
            } else {
                rankInfo.superPodId = orgRankInfo.superPodId;
                rankInfo.superPodIdx = orgRankInfo.superPodIdx;
            }
            CHK_RET(GetNicInfo(orgRankInfo.deviceInfo.nicDeploy, index, iter->second, rankInfo));
            rankInfo.nicIdx.assign(nicList_.begin(), nicList_.end());
            rankInfoList_.push_back(rankInfo);
        }
    }
    // 将rank id从小到大的顺序返回
    CHK_RET(SortRankInfoList());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::CheckRankTable(const RankTable_t& rankTable, const ServRankInfo& servRankInfo)
{
    // 检查网卡挂载位置
    if (CheckNicDeploy(rankTable.nicDeploy, deviceType_) != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[Check][RankTable]errNo[0x%016llx] nicDeploy[%d] out of range[%d, %d]", HCCL_ERROR_CODE(HCCL_E_PARA),
            rankTable.nicDeploy, static_cast<int32_t>(NICDeployment::NIC_DEPLOYMENT_HOST),
            static_cast<int32_t>(NICDeployment::NIC_DEPLOYMENT_DEVICE));
        return HCCL_E_PARA;
    }

    if (Is310PDevice()) {
        // 异构场景无需检查server内device个数
        return HCCL_SUCCESS;
    }

    if (CheckSuperDeviceId(rankTable) != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[Check][RankTable]errNo[0x%016llx] super_device_id is invalid in ranktable, "
            "ranktable config value: rankId[%u], superDeviceId[0x%x]",
            HCCL_ERROR_CODE(HCCL_E_PARA), userRank_, superDeviceId_);
        return HCCL_E_PARA;
    }

    // 检查服务器上的设备信息
    ServRankInfo::const_iterator iterBegin = servRankInfo.begin();
    u32 devNum = 0;
    CHK_RET(GetDevNum(iterBegin->second, devNum));

    bool multiServerDiffDeviceNumMode = false;
    for (ServRankInfo::const_iterator iter = iterBegin; iter != servRankInfo.end(); ++iter) {
        // 检测每个服务器内的设备数是否相等，如果不相同即为多server不同卡模式
        u32 curServerDevNum = 0;
        CHK_RET(GetDevNum(iter->second, curServerDevNum));
        if (devNum != curServerDevNum) {
            HCCL_WARNING(
                "[Check][RankTable] devnum isn't same,(serverA:[%s],serverB:[%s])"
                "devNum(%u, %u)",
                iterBegin->first.c_str(), iter->first.c_str(), devNum, curServerDevNum);
            multiServerDiffDeviceNumMode = true;
        }
    }

    // 非多server不同卡模式下，判断实际设备数目和userRank_table中的记录一致
    if (multiServerDiffDeviceNumMode == false && rankTable.deviceNum != devNum * servRankInfo.size()) {
        HCCL_WARNING(
            "[Check][RankTable]errNo[0x%016llx] devnum isn't same, number in rankTable:[%u], actual:[%llu]",
            HCCL_ERROR_CODE(HCCL_E_PARA), rankTable.deviceNum, devNum * servRankInfo.size());
        return HCCL_E_PARA;
    }

    // 910模组：服务器内设备的数目必须是2的次幂,在此check(非模组形态无此限制不check)
    // 910B、910_93模组形态未定，服务器内设备的数目校验规则后续补充
    if (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() > 0 && devNum > HCCL_DEVICE_NUM_TWO
        && (deviceType_ != DevType::DEV_TYPE_910B && deviceType_ != DevType::DEV_TYPE_910_93 && !Is310P3Common())) {
        CHK_PRT_RET(
            CheckDevCount(devNum) != HCCL_SUCCESS,
            HCCL_ERROR("[Check][RankTable]errNo[0x%016llx] devnum is invalid in server.", HCCL_ERROR_CODE(HCCL_E_PARA)),
            HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::CheckDevPhyId(const s32& devicePhyId) const
{
    if (devicePhyId > COMM_MAX_DEVICE_ID && devicePhyId != HOST_DEVICE_ID) {
        HCCL_ERROR(
            "[Check][DevPhyId]errNo[0x%016llx] devicePhyId[%d] out of range[-1, %d]", HCCL_ERROR_CODE(HCCL_E_PARA),
            devicePhyId, COMM_MAX_DEVICE_ID);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SortRankInfoList()
{
    // 按rank id从小到大的顺序返回
    std::sort(rankInfoList_.begin(), rankInfoList_.end(), CompareWithUserRank);

    for (u32 index = 0; index < rankInfoList_.size(); ++index) {
        CHK_PRT_RET(
            (index != rankInfoList_[index].userRank),
            HCCL_ERROR(
                "[HcclCommunicatorAttrs][SortRankInfoList]errNo[0x%016llx] index[%u] != rankInfoList.userRank[%u]",
                HCCL_ERROR_CODE(HCCL_E_PARA), index, rankInfoList_[index].userRank),
            HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::CheckNicDeploy(NICDeployment nicDeploy, DevType deviceType) const
{
    (void)deviceType;
    if (nicDeploy >= NICDeployment::NIC_DEPLOYMENT_RESERVED) {
        HCCL_ERROR(
            "[Check][NicDeploy]errNo[0x%016llx] nicDeploy[%u] out of range[%d, %d]", HCCL_ERROR_CODE(HCCL_E_PARA),
            nicDeploy, static_cast<int32_t>(NICDeployment::NIC_DEPLOYMENT_HOST),
            static_cast<int32_t>(NICDeployment::NIC_DEPLOYMENT_DEVICE));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::CheckDevCount(const u32 devNum)
{
    if (devNum > HCCL_AISERVER_DEVICE_NUM) {
        HCCL_ERROR(
            "[Check][DevCount]errNo[0x%016llx] devNum[%u] out of range[%u, %u]", HCCL_ERROR_CODE(HCCL_E_PARA), devNum,
            0, HCCL_AISERVER_DEVICE_NUM);
        return HCCL_E_PARA;
    }
    // 其他拓扑算法设备数目: 1 server: 1, 2, 4, 8
    //                     n server: 1*n, 2*n, 4*n, 8*n
    if (!Check2N(devNum)) {
        RPT_ENV_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(devNum), "devNum", "to be  1, 2 or 4, or a multiple of 8"}));
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] devNum[%u] devNum must be divisible by 8, or equal to 1, 2 or 4",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCCL_ERROR_CODE(HCCL_E_PARA),
            devNum);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

bool HcclCommunicatorAttrs::Check2N(u32 num) const
{
    if (num < 1) {
        return false;
    } else {
        return ((num & (num - 1)) == 0);
    }
}

HcclResult HcclCommunicatorAttrs::SetLocalRankInfo()
{
    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        HCCL_DEBUG(
            " host ip: %s host port: %u dev phy id: %d.", rankInfoList_[i].hostIp.GetReadableAddress(),
            rankInfoList_[i].hostPort, rankInfoList_[i].devicePhyId);
        if (rankInfoList_[i].userRank == userRank_) {
            devicePhyId_ = rankInfoList_[i].devicePhyId;
            devIpAddr_ = rankInfoList_[i].nicIp;
            devBackupIpAddr_ = rankInfoList_[i].backupNicIp;
            devBackupPort_ = rankInfoList_[i].backupDevicePort;
            hostIp_ = rankInfoList_[i].hostIp;
            hostPort_ = rankInfoList_[i].hostPort;
            localRank_ = rankInfoList_[i].localRank;
            HCCL_DEBUG("localRank_[%u].", localRank_);
            break;
        }
    }
    // 在确定 servRankInfo_ 和 serverId_ 信息后，就完成初始判断
    HCCL_DEBUG("[HcclCommunicatorAttrs][Init]deviceType[%u].", deviceType_);
    if (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) {
        HCCL_ERROR("[HcclCommunicatorAttrs][Init]not support cpu rank");
        return HCCL_E_NOT_SUPPORT;
    } else {
        HCCL_DEBUG("[HcclCommunicatorAttrs][Init]devicePhyId[%u] != HOST_DEVICE_ID", devicePhyId_);
        CHK_RET(hrtGetDevice(&deviceLogicId_));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetLocalRankInfoSubGroup(const std::vector<RankInfo>& rankList)
{
    rankInfoList_.assign(rankList.begin(), rankList.end());
    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        if (rankInfoList_[i].userRank == userRank_) {
            devIpAddr_ = rankInfoList_[i].nicIp;
            devBackupIpAddr_ = rankInfoList_[i].backupNicIp;
            devBackupPort_ = rankInfoList_[i].backupDevicePort;
            devicePhyId_ = rankInfoList_[i].devicePhyId;
            superPodId_ = rankInfoList_[i].superPodId;
            superDeviceId_ = rankInfoList_[i].superDeviceId;
            hostIp_ = rankInfoList_[i].hostIp;
            hostPort_ = rankInfoList_[i].hostPort;
            nicList_.assign(rankInfoList_[i].nicIdx.begin(), rankInfoList_[i].nicIdx.end());
            nicDeployment_ = rankInfoList_[i].nicDeploy;
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::CheckLocalRankInfo()
{
    for (u32 i = 0; i < rankInfoList_.size(); ++i) {
        if (userRank_ == rankInfoList_[i].userRank) {
            CHK_PRT_RET(
                static_cast<s32>(devicePhyId_) != rankInfoList_[i].devicePhyId,
                HCCL_ERROR(
                    "[Init][Para]errNo[0x%016llx] parameter check failed, "
                    "userrank[%u] == rankInfoList.userrank[%u], phyid[%d] != rankInfoList.devid[%d]",
                    HCCL_ERROR_CODE(HCCL_E_PARA), userRank_, rankInfoList_[i].userRank, static_cast<s32>(devicePhyId_),
                    rankInfoList_[i].devicePhyId),
                HCCL_E_PARA);
        }
    }
    return HCCL_SUCCESS;
}

u32 HcclCommunicatorAttrs::CalMeshAggRankSize(int halfDevNum) const
{
    u32 size = INVALID_VALUE_RANKSIZE;
    for (auto iter = servRankInfo_.begin(); iter != servRankInfo_.end(); ++iter) {
        u32 aggregationRankSize0 = 0;
        u32 aggregationRankSize1 = 0;
        for (u32 index = 0; index < iter->second.size(); ++index) {
            const RankInfo_t& orgRankInfo = iter->second[index];
            if (orgRankInfo.deviceInfo.devicePhyId < halfDevNum) {
                aggregationRankSize0++;
            } else {
                aggregationRankSize1++;
            }
        }
        u32 tmpsize = INVALID_VALUE_RANKSIZE;
        if ((aggregationRankSize0 != 0) && (aggregationRankSize1 != 0)) {
            tmpsize = aggregationRankSize0;
        } else {
            tmpsize = iter->second.size();
        }
        size = size > tmpsize ? tmpsize : size;
    }
    return size;
}

HcclResult HcclCommunicatorAttrs::SetMeshAggregationRankSize(u32 size)
{
    HCCL_INFO("[Set][HcclCommunicatorAttrs][MeshAggregationRankSize]set MeshAggregationRankSize[%u].", size);
    meshAggregationRankSize_ = size;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::CalAndSetMeshAggRankSize()
{
    u32 size = INVALID_VALUE_RANKSIZE;
    if ((deviceType_ == DevType::DEV_TYPE_910B) && isDiffDeviceModule_) { // 910B 16p场景
        size = CalMeshAggRankSize(HCCL_DEVICE_NUM_EIGHT);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        if (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == 0) { // 标卡
            size = 1;
        } else { // 模组
            size = CalMeshAggRankSize(HCCL_DEVICE_NUM_FOUR);
        }
    } else { // 910B的8卡、310P 直接返回server内的size数量
        size = servRankInfo_.begin()->second.size();
    }
    CHK_RET(SetMeshAggregationRankSize(size));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetWorldGroupInfo(
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>>& phyIdNicInfoMap,
    std::vector<RankInfo>& worldRankInfoList, std::vector<u32>& nicRanksPort, std::vector<u32>& vnicRanksPort)
{
    for (auto& ipInfo : phyIdNicInfoMap) {
        for (auto& devInfo : ipInfo.second) {
            rankDevicePhyIdNicInfoMap_[ipInfo.first][devInfo.first] = devInfo.second;
            HCCL_DEBUG(
                "phyIdNicInfoMap print hostIp[%s] devId[%u] devIp[%s]", ipInfo.first.c_str(), devInfo.first,
                devInfo.second.GetReadableAddress());
        }
    }

    for (auto& rankInfo : worldRankInfoList) {
        worldRankInfoList_.push_back(rankInfo);
    }

    for (auto& port : nicRanksPort) {
        nicRanksPort_.push_back(port);
        HCCL_DEBUG("nicRanksPort port[%u]", port);
    }
    for (auto& port : vnicRanksPort) {
        vnicRanksPort_.push_back(port);
        HCCL_DEBUG("vnicRanksPort port[%u]", port);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::TransformRankList(
    const std::vector<RankInfo>& rankListIn, std::vector<RankInfo_t>& rankListOut) const
{
    for (size_t index = 0; index < rankListIn.size(); ++index) {
        RankInfo_t rankInfoTmp;
        rankInfoTmp.serverId = rankListIn[index].serverId;
        rankInfoTmp.deviceInfo.devicePhyId = rankListIn[index].devicePhyId;
        rankInfoTmp.deviceInfo.deviceType = rankListIn[index].deviceType;
        rankInfoTmp.serverIdx = rankListIn[index].serverIdx;
        rankInfoTmp.rankId = rankListIn[index].userRank;
        rankInfoTmp.hostIp = rankListIn[index].hostIp;
        rankInfoTmp.hostPort = rankListIn[index].hostPort;
        rankInfoTmp.localRank = rankListIn[index].localRank;
        rankInfoTmp.superDeviceId = rankListIn[index].superDeviceId;
        rankInfoTmp.superPodId = rankListIn[index].superPodId;
        rankInfoTmp.superPodIdx = rankListIn[index].superPodIdx;
        rankListOut.push_back(rankInfoTmp);
    }
    return HCCL_SUCCESS;
}

bool HcclCommunicatorAttrs::IsEnableRoce()
{
    // 910B单机两种使能roce场景：1、a+x同时使用两module  2.标卡
    bool roceSwitch = IsSupportEnableRoce();
    bool isInterServerVnic = false;
    // 910_93超节点内节点间走HCCS通信 && Vnic建链, 不需要使能NIC
    if (useSuperPodMode_ && superPodNum_ == 1 && GetExternalInputInterHccsDisable() == false) {
        isInterServerVnic = true;
    }
    bool ret = (interServer_ && !isInterServerVnic) || roceSwitch;
    HCCL_INFO(
        "IsEnableRoce ret: %d, interServer_: %d, isInterServerVnic: %d, roceSwitch: %d, "
        "isSingleMeshAggregation_: %u",
        ret, interServer_, isInterServerVnic, roceSwitch, isSingleMeshAggregation_);
    return ret;
}

// a+x mesh间需要同时保证ip有效和roce开关打开才能走rdma
bool HcclCommunicatorAttrs::IsUsedRdmaLevel0AndIpInvalid()
{
    u32 nicNum = devIpAddr_.size();
    bool ipInvalid = true;
    for (u32 i = 0; i < nicNum; i++) {
        if (devIpAddr_[i].IsInvalid()) {
            HCCL_INFO("[Init][Nic]nic num[%u] deviceip is invalid, total nicNum[%u]", i, nicNum);
            ipInvalid = false;
            continue;
        }
    }
    // 机间卡数不一致场景下，IP有效情况下就走RDMA
    // 机间卡数一致场景下，需环境变量ROCE打开(多机环境下未对IsEnableRoce开关进行控制)且IP有效情况下走RDMA
    return (
        (GetExternalInputIntraRoceSwitch() != 0 || multiModuleDiffDeviceNumMode_ || isDiffDeviceType_) && ipInvalid);
}

bool HcclCommunicatorAttrs::IsSupportEnableRoce()
{
    // 910B单机两种使能roce场景：1、a+x同时使用两module  2.标卡
    bool roceSwitch = false;
    HCCL_INFO("[HcclCommunicator]IsSupportEnableRoce");
    if (isDiffDeviceType_) {
        roceSwitch = true;
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        roceSwitch = ((GetExternalInputIntraRoceSwitch() != 0) && (!isSingleMeshAggregation_ || isStandardCard_))
                     || multiModuleDiffDeviceNumMode_;
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        roceSwitch = multiSuperPodDiffServerNumMode_ || (multiModuleDiffDeviceNumMode_ && superPodNum_ > 1);
    } else { // 其他单机场景为了防止用户误用roce开关
        roceSwitch = isStandardCard_ ? GetExternalInputIntraRoceSwitch() : false;
    }
    return roceSwitch;
}

void HcclCommunicatorAttrs::GetTopoAttr(HcclTopoAttr& topoAttr)
{
    topoAttr.serverNum = serverNum_;
    topoAttr.superPodNum = superPodNum_;
    topoAttr.moduleNum = moduleNum_;
    topoAttr.deviceNumPerServer = deviceNumPerServer_;
    topoAttr.deviceNumPerAggregation = deviceNumPerAggregation_;
    topoAttr.multiModuleDiffDeviceNumMode = multiModuleDiffDeviceNumMode_;
    topoAttr.multiSuperPodDiffServerNumMode = multiSuperPodDiffServerNumMode_;
    topoAttr.multiSuperPodDiffDeviceNumMode = multiSuperPodDiffDeviceNumMode_;
    topoAttr.meshAggregationRankSize = meshAggregationRankSize_;
    topoAttr.isDiffDeviceModule = isDiffDeviceModule_;
    topoAttr.isDiffDeviceType = isDiffDeviceType_;
    topoAttr.gcdDeviceNumPerAggregation = gcdDeviceNumPerAggregation_;
    topoAttr.isSingleMeshAggregation = isSingleMeshAggregation_;
    topoAttr.isAllRankSamePlane = isAllRankSamePlane_;
    topoAttr.userRank = userRank_;
    topoAttr.realUserRank = realUserRank_;
    topoAttr.userRankSize = userRankSize_;
    topoAttr.devicePhyId = devicePhyId_;
    topoAttr.useSuperPodMode = useSuperPodMode_;
    topoAttr.deviceLogicId = deviceLogicId_;
    topoAttr.deviceType = deviceType_;
    topoAttr.isStandardCard = isStandardCard_;
    topoAttr.is310PDuoCard = is310PDuoCard_;
    topoAttr.isCommon310P3DUO = isCommon310P3DUO_;
    topoAttr.hccsPortNum = hccsPortNum_;
    topoAttr.nicList = nicList_;
    topoAttr.pairLinkCounter = pairLinkCounter_;
    topoAttr.pairLinkInfo = pairLinkInfo_;
    topoAttr.rankInfoList = rankInfoList_;
    topoAttr.isSupportRdmaLite = isSupportRdmaLite_;
    topoAttr.isSupportHccsAndSio = isSupportHccsAndSio_;
    topoAttr.localNicPort = GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
    topoAttr.isNeedInitNic = isNeedInitNic_;
    topoAttr.isARSDoubleRing = isARSDoubleRing_;
}

void HcclCommunicatorAttrs::GetAlgoAttr(HcclAlgoAttr& algoAttr)
{
    algoAttr.isHaveCpuRank = isHaveCpuRank_;
    algoAttr.inlineReduceSwitchOn = inlineReduceSwitchOn_;
    algoAttr.isUsedRdmaLevel0 = isUsedRdmaLevel0_;
    HCCL_INFO("[CollectAlgoAttr]:isUsedRdmaLevel0:[%d]", isUsedRdmaLevel0_);
    algoAttr.isUsedInterHccsMode = isUsedInterHccsMode_;
    algoAttr.identifier = identifier_;
    algoAttr.collectiveId = collectiveId_;
    algoAttr.nicDeployment = nicDeployment_;
    algoAttr.commWorkMode = commWorkMode_;
    algoAttr.commAlgoConfig = algoConfigMap_;
}

u32 HcclCommunicatorAttrs::GetLocalNicPort(NicType nicType)
{
    if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
        return GetHostPort(devicePhyId_);
    }
    // isUseRankPort_在ranksPort初始化时一同配置：1. 异构场景 2. 开启device侧端口配置
    // groupRanksPort_为空说明此时处于全局通信域，要从ranksPort_取监听端口；否则取groupRanksPort_
    if (nicType == NicType::HOST_NIC_TYPE) {
        return GetHostPort(devicePhyId_);
    }
    if (nicType == NicType::VNIC_TYPE && GetExternalInputNpuPortSwitch()) {
        // vnic ports仅在开启device侧端口配置时单独配置
        std::vector<u32>& ranksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
        return GetNicPort(devicePhyId_, ranksPorts, userRank_, isUseRankPort_);
    } else {
        // 1. 开启device侧端口配置时的nic port时使用ranksPorts
        // 2. 异构场景使用ranksPorts
        // 3. 其余场景场景isUseRankPort_应当为false，使用默认port
        std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        return GetNicPort(devicePhyId_, ranksPorts, userRank_, isUseRankPort_);
    }
}
} // namespace hccl
