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
#include "externalinput_pub.h"
#include "env_config.h"
#include "search_path.h"

using namespace std;

namespace hccl {
HcclCommunicatorAttrs::HcclCommunicatorAttrs() {}

HcclCommunicatorAttrs::~HcclCommunicatorAttrs() {}

bool HcclCommunicatorAttrs::Is310P3Common()
{
    return !isHaveCpuRank_ && !Is310PDevice() && deviceType_ == DevType::DEV_TYPE_310P3;
}

HcclResult HcclCommunicatorAttrs::GetPairDeviceLinkType(
    const RankTable_t& rankTable, u32 i, bool& isConnectedWithHCCS, LinkTypeInServer& linkType) const
{
    for (u32 j = i + 1; j < rankTable.rankList.size(); j++) {
        if (rankTable.rankList[i].serverId == rankTable.rankList[j].serverId) {
            bool isValidRanki = rankTable.rankList[i].deviceInfo.devicePhyId == HOST_DEVICE_ID;
            bool isValidRankj = rankTable.rankList[j].deviceInfo.devicePhyId == HOST_DEVICE_ID;
            if (isValidRanki || isValidRankj) {
                continue;
            }
            CHK_RET(hrtGetPairDeviceLinkType(
                rankTable.rankList[i].deviceInfo.devicePhyId, rankTable.rankList[j].deviceInfo.devicePhyId, linkType));
        }
        if (linkType != LinkTypeInServer::HCCS_TYPE) {
            isConnectedWithHCCS = false;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::GetMixInnerLinkInfo(
    std::unordered_map<u32, u32>& pairLinkCounter,
    std::unordered_map<u32, std::unordered_map<int, std::vector<int>>>& pairLinkInfo)
{
    pairLinkInfo.clear();
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] = 0;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::PXI_TYPE)] = 0;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::SIO_TYPE)] = 0;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)] = 0;
    for (auto& it_local : nicList_) {
        for (auto& it_dest : nicList_) {
            if (it_local == it_dest || static_cast<s32>(it_local) == HOST_DEVICE_ID
                || static_cast<s32>(it_dest) == HOST_DEVICE_ID) {
                continue;
            }
            LinkTypeInServer linkType;
            CHK_RET(hrtGetPairDeviceLinkType(it_local, it_dest, linkType));
            pairLinkInfo[static_cast<u32>(linkType)][it_local].push_back(it_dest);
            pairLinkCounter[static_cast<u32>(linkType)]++;
        }
    }
    if (HcclCheckLogLevel(DLOG_DEBUG)) {
        for (auto it : pairLinkInfo) {
            HCCL_DEBUG("pair link information linkType[%u], size[%llu]", it.first, it.second.size());
        }
        for (auto it : pairLinkCounter) {
            HCCL_DEBUG("pair link counter information linkType[%u], size[%llu]", it.first, it.second);
        }
    }

    return HCCL_SUCCESS;
}

// 用于标识集群中是否存在 不同芯片形态
bool HcclCommunicatorAttrs::IsDiffDeviceType(const std::vector<RankInfo_t>& rankList) const
{
    if (rankList.size() <= 1 || isHaveCpuRank_) {
        return false;
    }
    for (const RankInfo_t& rankInfo : rankList) {
        if (GetRankInfoDevType(rankInfo) != deviceType_) {
            HCCL_INFO(
                "[IsDiffDeviceType] deviceType_[%d], and ranktable contains devicePhyId[%d]-deviceType[%d]",
                deviceType_, rankInfo.deviceInfo.devicePhyId, rankInfo.deviceInfo.deviceType);
            return true;
        }
    }
    return false;
}

HcclResult HcclCommunicatorAttrs::SetNiclistInfo()
{
    for (auto& iter : servRankInfo_[serverId_]) {
        if (((!iter.hostIp.IsInvalid()) || (!iter.deviceInfo.deviceIp[0].IsInvalid()))
            && (iter.deviceInfo.devicePhyId != HOST_DEVICE_ID)) {
            if (isDiffDeviceType_) {
                u32 gcdIdx = userRank_ / gcdDeviceNumPerAggregation_;
                u32 gcdUserRankMin = gcdIdx * gcdDeviceNumPerAggregation_;
                u32 gcdUserRankMax = (gcdIdx + 1) * gcdDeviceNumPerAggregation_ - 1;
                if (iter.rankId < gcdUserRankMin || iter.rankId > gcdUserRankMax) {
                    continue;
                }
            }
            nicList_.push_back(iter.deviceInfo.devicePhyId);
        }
    }
    std::sort(nicList_.begin(), nicList_.end());
    HCCL_DEBUG("nic isDiffDeviceType[%u] userRank[%u] nicList size[%d]", isDiffDeviceType_, userRank_, nicList_.size());
    if (isDiffDeviceType_) {
        CHK_RET(GetMixInnerLinkInfo(pairLinkCounter_, pairLinkInfo_)); // 获取混合组网场景上HCCS、PXI链接的数目
    }
    CHK_SMART_PTR_NULL(topoInfoParse_);
    CHK_RET(topoInfoParse_->ParseAndCheck(nicList_));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::InitTopoInfo(const RankTable_t& rankTable)
{
    topoInfoParse_.reset(new (std::nothrow) TopoInfoParse());
    CHK_SMART_PTR_NULL(topoInfoParse_);
    CHK_RET(topoInfoParse_->Init(rankTable, serverId_, deviceNumPerServer_));
    if (!isDiffDeviceType_) {
        CHK_RET(topoInfoParse_->GetServerInnerLinkInfo(
            pairLinkCounter_, pairLinkInfo_)); // 获取本Server上HCCS、PXI链接的数目
    }
    // 初始化阶段判断组网状态
    CHK_RET(topoInfoParse_->IsSingleMeshAggregation(isSingleMeshAggregation_)); // 确认集群中只有一个MeshAggregation
    CHK_RET(topoInfoParse_->IsAllRankSamePlane(isAllRankSamePlane_)); // 确认集群所有卡在一个平面上
    isStandardCard_ = IsStandardCard();
    is310PDuoCard_ = Is310PDuoCard();
    if (is310PDuoCard_) {
        isCommon310P3DUO_ = IsCommon310P3DUO(rankTable.rankList);
    }
    CHK_RET(InitHccsPortNum());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::InitTopoInfo(const std::vector<RankInfo>& rankList)
{
    topoInfoParse_.reset(new (std::nothrow) TopoInfoParse());
    CHK_SMART_PTR_NULL(topoInfoParse_);
    CHK_RET(topoInfoParse_->Init(rankList, serverId_, deviceNumPerServer_));
    if (!isDiffDeviceType_) {
        CHK_RET(topoInfoParse_->GetServerInnerLinkInfo(
            pairLinkCounter_, pairLinkInfo_)); // 获取本Server上HCCS、PXI链接的数目
    }
    // 初始化阶段判断组网状态
    CHK_RET(topoInfoParse_->IsSingleMeshAggregation(isSingleMeshAggregation_)); // 确认集群中只有一个MeshAggregation
    CHK_RET(topoInfoParse_->IsAllRankSamePlane(isAllRankSamePlane_)); // 确认集群所有卡在一个平面上
    isStandardCard_ = IsStandardCard();
    is310PDuoCard_ = Is310PDuoCard();
    CHK_RET(InitHccsPortNum());
    if (!isStandardCard_) {
        CHK_RET(topoInfoParse_->Check());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetInterModeInSuperPod()
{
    // 硬件配置为非超节点模式或软件（ranktable）中未配置sdid，后面按照非超节点形态处理
    if (!useSuperPodMode_) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[Set][InterModeInSuperPod]: serverNum[%u], superPodNum[%u].", serverNum_, superPodNum_);
    // 超节点HCCS模式
    if (GetExternalInputInterHccsDisable() == false && serverNum_ > 1 && superPodNum_ > 0) {
        isUsedInterHccsMode_ = true;
        HCCL_RUN_INFO(
            "[Set][InterModeInSuperPod]: will use inter HCCS Mode, superPodId[%s], superDeviceId[0x%x], "
            "superPodNum[%u], serverNum[%u], userRank[%u].",
            superPodId_.c_str(), superDeviceId_, superPodNum_, serverNum_, userRank_);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SethbRankInfo(const std::vector<RankInfo>& rankList, WorldGroupInfo& groupCommonData)
{
    // 记录serverId
    serverId_ = groupCommonData.serverId;
    useSuperPodMode_ = groupCommonData.useSuperPodMode;

    for (auto& rankInfo : rankList) {
        if (rankInfo.devicePhyId == HOST_DEVICE_ID) {
            isHaveCpuRank_ = true;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::CheckSuperDeviceId(const RankTable_t& rankTable)
{
    // 非910_93/910_93非超节点形态 || 用户配置非超节点模式，无需校验SDID合法性
    if (!useSuperPodMode_) {
        return HCCL_SUCCESS;
    }

    for (u32 i = 0; i < rankTable.rankList.size(); i++) {
        if (rankTable.rankList[i].rankId == userRank_) {
            s64 drvSuperDeviceID = 0;
            CHK_RET(hrtGetDeviceInfo(
                deviceLogicId_, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
                HcclRtDeviceInfoType::HCCL_INFO_TYPE_SDID, drvSuperDeviceID));
            if (superDeviceId_ != static_cast<u32>(drvSuperDeviceID)) {
                RPT_INPUT_ERR(
                    true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                    std::vector<std::string>(
                        {std::to_string(superDeviceId_), "super_device_id", std::to_string(drvSuperDeviceID)}));
                HCCL_ERROR(
                    "[%s][%s]errNo[0x%016llx] super_device_id is invalid, "
                    "expect value [0x%x], ranktable config value [0x%x]",
                    LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA),
                    drvSuperDeviceID, superDeviceId_);
                return HCCL_E_PARA;
            }
            break;
        }
    }
    HCCL_RUN_INFO(
        "[Check][SuperDeviceId]: superDevice check success, superPodId[%s], "
        "superDeviceId[0x%x], userRank[%u].",
        superPodId_.c_str(), superDeviceId_, userRank_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::UpdateNicList()
{
    std::vector<u32> subCommNicList;
    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        if (rankInfoList_[i].serverId == serverId_
            && std::find(nicList_.begin(), nicList_.end(), rankInfoList_[i].devicePhyId) != nicList_.end()) {
            if (isDiffDeviceType_) {
                u32 gcdIdx = userRank_ / gcdDeviceNumPerAggregation_;
                u32 gcdUserRankMin = gcdIdx * gcdDeviceNumPerAggregation_;
                u32 gcdUserRankMax = (gcdIdx + 1) * gcdDeviceNumPerAggregation_ - 1;
                if (rankInfoList_[i].userRank < gcdUserRankMin || rankInfoList_[i].userRank > gcdUserRankMax) {
                    continue;
                }
            }
            subCommNicList.push_back(rankInfoList_[i].devicePhyId);
        }
    }
    nicList_ = subCommNicList;
    if (HcclCheckLogLevel(DLOG_DEBUG)) {
        // 打印更新后的nicList_
        std::ostringstream stringRepresentation;
        for (std::vector<uint32_t>::iterator it = nicList_.begin(); it != nicList_.end(); it++) {
            stringRepresentation << *it << " ";
        }
        std::string nicListString = stringRepresentation.str();
        const char* charNicList = nicListString.c_str();
        HCCL_DEBUG("[HcclCommunicatorAttrs][Init] The subcommunication domain related nicList_: %s", charNicList);
    }
    // 将更新的nicList_刷新到rankInfoList_中
    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        rankInfoList_[i].nicIdx.assign(nicList_.begin(), nicList_.end());
    }
    if (isDiffDeviceType_) {
        CHK_RET(GetMixInnerLinkInfo(pairLinkCounter_, pairLinkInfo_)); // 获取混合组网场景上HCCS、PXI链接的数目
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetRanksPort(const std::vector<RankInfo_t>& rankList)
{
    bool devicePortSwitchOn = GetExternalInputNpuPortSwitch();
    if (devicePortSwitchOn) {
        nicRanksPort_.resize(userRankSize_, HCCL_INVALID_PORT);
        vnicRanksPort_.resize(userRankSize_, HCCL_INVALID_PORT);
        for (auto& rankInfo : rankList) {
            nicRanksPort_[rankInfo.rankId]
                = rankInfo.deviceInfo.port == HCCL_INVALID_PORT ? HETEROG_CCL_PORT : rankInfo.deviceInfo.port;
            vnicRanksPort_[rankInfo.rankId]
                = rankInfo.deviceInfo.vnicPort == HCCL_INVALID_PORT ? HETEROG_CCL_PORT : rankInfo.deviceInfo.vnicPort;
        }
    } else {
        nicRanksPort_.resize(userRankSize_, HCCL_INVALID_PORT);
        for (auto& rankInfo : rankList) {
            nicRanksPort_[rankInfo.rankId]
                = rankInfo.deviceInfo.port == HCCL_INVALID_PORT || rankInfo.deviceInfo.port == 0 ?
                      HETEROG_CCL_PORT :
                      rankInfo.deviceInfo.port;
        }
    }
    isUseRankPort_ = ((devicePortSwitchOn && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) || isHaveCpuRank_
                      || nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) ?
                         true :
                         isUseRankPort_;
    HCCL_INFO(
        "[HcclCommunicatorAttrs][SetRanksPort] devicePortSwitchOn[%u], isHaveCpuRank[%u], isUseRankPort[%u], "
        "nicRanksPort size[%u], vnicRanksPort size[%u].",
        devicePortSwitchOn, isHaveCpuRank_, isUseRankPort_, nicRanksPort_.size(), vnicRanksPort_.size());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::InitRankInfo(const RankTable_t& rankTable)
{
    // 获取serverId
    CHK_RET(SetServerId(rankTable));
    // 获取server数
    CHK_RET(SetServerNum(rankTable.rankList));
    CHK_PRT_RET(
        serverNum_ != rankTable.serverNum,
        HCCL_ERROR(
            "[HcclCommunicatorAttrs][InitRankInfo]calculated serverNum[%u] is not equal to ranktable serverNum[%u]",
            serverNum_, rankTable.serverNum),
        HCCL_E_PARA);
    // 本节点的sdid配置有效(ranktable v1.2)且环境配置server id有效时, 使能superPod
    if (superDeviceId_ != INVALID_UINT) {
        CHK_RET(IsSuperPodMode(useSuperPodMode_)); // 使能superPod
    }
    // 获取server内设备数, 赋值 ishavecpurank_
    CHK_RET(SetInnerServerAverageDevice(rankTable));
    // 根据server整理rank信息
    CHK_RET(TransformRankInfoByServerId(rankTable.rankList, servRankInfo_));
    isDiffDeviceType_ = IsDiffDeviceType(rankTable.rankList);
    // 解析拓扑信息
    CHK_RET(InitTopoInfo(rankTable));
    // 获取module相关信息，moduleNum_, isDiffDeviceModule_, multiModuleDiffDeviceNumMode_;
    CHK_RET(SetModuleInfo(rankTable.rankList));
    // 获取超节点相关信息，superPodNum_, multiSuperPodDiffServerNumMode_
    CHK_RET(SetSuperPodInfo(rankTable.rankList));
    // 生成nicList
    CHK_RET(SetNiclistInfo());
    // 设置超节点内节点间模式，包括是否使用sdid获取vnicip、节点间是否使能HCCS
    CHK_RET(SetInterModeInSuperPod());
    // 解析ranktable信息(生成rankInfoList_)，供给commfactory使用
    CHK_RET(SetRankInfoList(rankTable));
    // 解析当前Rank信息
    CHK_RET(SetLocalRankInfo());
    // 解析rank和port的映射信息
    CHK_RET(SetRanksPort(rankTable.rankList));

    // 通过关键字打印通信域及本端的rank关键信息，方便在日志中直接检索
    HCCL_RUN_INFO(
        "[%s]identifier[%s] rankSize[%u] serverNum[%u] moduleNum[%u] superPodNum[%u] "
        "multiModuleDiffDeviceNumMode[%u] multiSuperPodDiffServerNumMode[%u]",
        LOG_KEYWORDS_COMMUNICATOR.c_str(), identifier_.c_str(), userRankSize_, serverNum_, moduleNum_, superPodNum_,
        multiModuleDiffDeviceNumMode_, multiSuperPodDiffServerNumMode_);
    HCCL_RUN_INFO(
        "[%s]userRank[%u] hostIp[%s] devicePhyId[%u] server[%s] deviceIp[%s] superPodId[%s] useSuperPodMode[%d] "
        "isStandardCard[%d]",
        LOG_KEYWORDS_LOCALRANK.c_str(), userRank_, hostIp_.GetReadableAddress(), devicePhyId_, serverId_.c_str(),
        devIpAddr_.empty() ? "" : devIpAddr_[0].GetReadableAddress(), superPodId_.c_str(), useSuperPodMode_,
        isStandardCard_);

    interServer_ = rankTable.serverNum > 1; // serverNum为1时，不进行roce初始化
    nicDeployment_ = rankTable.nicDeploy;
    rankTableVersion_ = rankTable.version;
    return HCCL_SUCCESS;
}

void HcclCommunicatorAttrs::GenCollectiveId(HcclCommParams& params, const RankTable_t& rankTable)
{
    collectiveId_ = rankTable.collectiveId.empty() ? params.id.internal : rankTable.collectiveId;
}

HcclResult
HcclCommunicatorAttrs::InitRankInfoSubGroup(const std::vector<RankInfo>& rankList, WorldGroupInfo& groupCommonData)
{
    // 填充心跳信息
    SethbRankInfo(rankList, groupCommonData);
    // 获取server内平均device数
    CHK_RET(SetInnerServerAverageDevice(rankList));
    // 将子通信域的ranklist结构体形式转换成全局通信域的
    std::vector<RankInfo_t> rankListNew;
    CHK_RET(TransformRankList(rankList, rankListNew));
    // 获取server数
    CHK_RET(SetServerNum(rankListNew));
    isDiffDeviceType_ = IsDiffDeviceType(rankListNew);
    // 解析拓扑信息
    CHK_RET(InitTopoInfo(rankList));
    // 获取module相关信息，moduleNum_, isDiffDeviceModule_, multiModuleDiffDeviceNumMode_;
    CHK_RET(SetModuleInfo(rankListNew));
    // 获取超节点相关信息，superPodNum_, multiSuperPodDiffServerNumMode_
    CHK_RET(SetSuperPodInfo(rankListNew));
    // 根据server整理rank信息
    CHK_RET(TransformRankInfoByServerId(rankListNew, servRankInfo_));
    //  inline reduce 开关
    inlineReduceSwitchOn_ = groupCommonData.inlineReduceSwitchOn;
    // 设置rank关联信息
    CHK_RET(SetLocalRankInfoSubGroup(rankList));
    // 设置超节点内节点间模式，包括是否使用sdid获取vnicip、节点间是否使用HCCS
    CHK_RET(SetInterModeInSuperPod());

    if (HcclCheckLogLevel(DLOG_DEBUG)) {
        // 打印原来的nicList_
        std::ostringstream stringRepresentation;
        for (std::vector<uint32_t>::iterator it = nicList_.begin(); it != nicList_.end(); it++) {
            stringRepresentation << *it << " ";
        }
        std::string nicListString = stringRepresentation.str();
        const char* charNicList = nicListString.c_str();
        HCCL_DEBUG("[HcclCommunicatorAttrs][Init] The original nicList_: %s", charNicList);
    }
    interServer_ = serverNum_ > 1; // serverNum为1时，不进行roce初始化
    // 更新成跟子通信域相关的nicList_
    CHK_RET(UpdateNicList());
    // 检查当前user_rank 对应的devid和rt查到的一致
    CHK_RET(CheckLocalRankInfo());
    CHK_RET(CalAndSetMeshAggRankSize());

    if (IsEnableRoce()) {
        isUsedRdmaLevel0_ = IsUsedRdmaLevel0AndIpInvalid();
    }

    CHK_RET(SetWorldGroupInfo(
        groupCommonData.phyIdNicInfoMap, groupCommonData.worldRankInfoList, groupCommonData.ranksPort,
        groupCommonData.vnicRanksPort));
    for (auto& rankInfo : worldRankInfoList_) {
        if (rankInfo.devicePhyId == HOST_DEVICE_ID) {
            isUseRankPort_ = true;
            break;
        }
    }
    CHK_RET(IsHostUseDevNic(isHostUseDevNic_));

    groupNicRanksPort_.resize(rankInfoList_.size(), HCCL_INVALID_PORT);
    if (nicRanksPort_.size() != 0) {
        for (auto& rankInfo : rankInfoList_) {
            groupNicRanksPort_[rankInfo.userRank] = nicRanksPort_[rankInfo.worldRank];
            HCCL_INFO(
                "hostIp[%s], nicIp[%s], rankInfo.userRank[%u], rankInfo.worldRank[%u], "
                "nic port[%u], devicePhyId[%d]",
                rankInfo.hostIp.GetReadableAddress(), rankInfo.nicIp[0].GetReadableAddress(), rankInfo.userRank,
                rankInfo.worldRank, groupNicRanksPort_[rankInfo.userRank], rankInfo.devicePhyId);
        }
    }
    bool devicePortSwitchOn = groupCommonData.devPortSwitchOn;
    if (devicePortSwitchOn) {
        groupVnicRanksPort_.resize(rankInfoList_.size(), HCCL_INVALID_PORT);
        if (vnicRanksPort_.size() != 0) {
            for (auto& rankInfo : rankInfoList_) {
                groupVnicRanksPort_[rankInfo.userRank] = vnicRanksPort_[rankInfo.worldRank];
                HCCL_INFO(
                    "hostIp[%s], nicIp[%s], rankInfo.userRank[%u], rankInfo.worldRank[%u], "
                    "vnic port[%u], devicePhyId[%d]",
                    rankInfo.hostIp.GetReadableAddress(), rankInfo.nicIp[0].GetReadableAddress(), rankInfo.userRank,
                    rankInfo.worldRank, groupVnicRanksPort_[rankInfo.userRank], rankInfo.devicePhyId);
            }
        }
    }
    isUseRankPort_ = ((devicePortSwitchOn && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) || isHaveCpuRank_
                      || nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) ?
                         true :
                         isUseRankPort_;
    HCCL_INFO("[InitRankInfoSubGroup]:isUsedRdmaLevel0_[%d]", isUsedRdmaLevel0_);
    return HCCL_SUCCESS;
}

void HcclCommunicatorAttrs::GenUsedRdmaLevel0() { isUsedRdmaLevel0_ = IsSupportEnableRoce(); }

#ifndef OPEN_HCCL_TEST
void HcclCommunicatorAttrs::GenSupportRdmaLite() { isSupportRdmaLite_ = IsSupportRDMALite(deviceLogicId_); }
HcclResult HcclCommunicatorAttrs::GenSupportHccsAndSio()
{
    CHK_RET(IsSupportHccsAndSio(isSupportHccsAndSio_));
    return HCCL_SUCCESS;
}
#endif

bool HcclCommunicatorAttrs::GetUsedRdmaLevel0() const { return isUsedRdmaLevel0_; }

bool HcclCommunicatorAttrs::GetSupportRdmaLite() const { return isSupportRdmaLite_; }
bool HcclCommunicatorAttrs::GetSupportHccsAndSio() const { return isSupportHccsAndSio_; }

std::string HcclCommunicatorAttrs::GetServerId() { return serverId_; }

u32 HcclCommunicatorAttrs::GetServerNum() const { return serverNum_; }
std::string HcclCommunicatorAttrs::GetSuperPodId() { return superPodId_; }

u32 HcclCommunicatorAttrs::GetSuperDeviceId() const { return superDeviceId_; }

bool HcclCommunicatorAttrs::GetSuperPodMode() const { return useSuperPodMode_; }

u32 HcclCommunicatorAttrs::GetSuperPodNums() const { return superPodNum_; }

u32 HcclCommunicatorAttrs::GetDeviceNumPerAggregation() const { return deviceNumPerAggregation_; }

u32 HcclCommunicatorAttrs::GetDeviceNumPerServer() const { return deviceNumPerServer_; }

DevType HcclCommunicatorAttrs::GetRankInfoDevType(const RankInfo_t& rankInfo) const
{
    if (rankInfo.deviceInfo.deviceType == DevType::DEV_TYPE_NOSOC) {
        return deviceType_; // 兼容非混合组网场景，rankInfo中deviceType字段可能没有赋值
    }
    return rankInfo.deviceInfo.deviceType;
}

bool HcclCommunicatorAttrs::GetDiffDeviceType() const { return isDiffDeviceType_; }

u32 HcclCommunicatorAttrs::GetGcdDeviceNumPerAggregation() const { return gcdDeviceNumPerAggregation_; }
ServRankInfo HcclCommunicatorAttrs::GetServRankInfo() { return servRankInfo_; }

bool HcclCommunicatorAttrs::GetDiffDeviceModule() const { return isDiffDeviceModule_; }

bool HcclCommunicatorAttrs::GetSupportARS() const { return isARSDoubleRing_; }

u32 HcclCommunicatorAttrs::GetModuleNum() const { return moduleNum_; }

bool HcclCommunicatorAttrs::GetMultiModuleDiffDeviceNumMode() const { return multiModuleDiffDeviceNumMode_; }

bool HcclCommunicatorAttrs::GetMultiSuperPodDiffServerNumMode() const { return multiSuperPodDiffServerNumMode_; }

bool HcclCommunicatorAttrs::GetmultiSuperPodDiffDeviceNumMode() const { return multiSuperPodDiffDeviceNumMode_; }

std::vector<u32> HcclCommunicatorAttrs::GetNicList() { return nicList_; }

bool HcclCommunicatorAttrs::GetSingleMeshAggregation() const { return isSingleMeshAggregation_; }

bool HcclCommunicatorAttrs::GetAllRankSamePlane() const { return isAllRankSamePlane_; }

bool HcclCommunicatorAttrs::GetStandardCard() const { return isStandardCard_; }

bool HcclCommunicatorAttrs::Get310PDuoCard() const { return is310PDuoCard_; }

bool HcclCommunicatorAttrs::GetIsCommon310P3DUO() const { return isCommon310P3DUO_; }

s32 HcclCommunicatorAttrs::GetHccsPortNum() const { return hccsPortNum_; }

void HcclCommunicatorAttrs::GetPairLinkCounter(std::unordered_map<u32, u32>& pairLinkCounter)
{
    pairLinkCounter = pairLinkCounter_;
}

void HcclCommunicatorAttrs::GetPairLinkInfo(
    std::unordered_map<u32, std::unordered_map<int, std::vector<int>>>& pairLinkInfo)
{
    pairLinkInfo = pairLinkInfo_;
}

bool HcclCommunicatorAttrs::GetUsedInterHccsMode() const { return isUsedInterHccsMode_; }

std::vector<RankInfo> HcclCommunicatorAttrs::GetRankInfoList() { return rankInfoList_; }

std::vector<HcclIpAddress> HcclCommunicatorAttrs::GetDevIpAddr() { return devIpAddr_; }

std::vector<HcclIpAddress> HcclCommunicatorAttrs::GetDevBackupIpAddr() { return devBackupIpAddr_; }

u32 HcclCommunicatorAttrs::GetBackupDevPort() const { return devBackupPort_; }

u32 HcclCommunicatorAttrs::GetDevicePhyId() const { return devicePhyId_; }

HcclIpAddress HcclCommunicatorAttrs::GetHostIp() { return hostIp_; }

u32 HcclCommunicatorAttrs::GetHostPort() { return hostPort_; }

u32 HcclCommunicatorAttrs::GetLocalRank() const { return localRank_; }

std::string HcclCommunicatorAttrs::GetCollectiveId() { return collectiveId_; }

std::string HcclCommunicatorAttrs::GetRankTableVersion() { return rankTableVersion_; }

s32 HcclCommunicatorAttrs::GetDeviceLogicId() const { return deviceLogicId_; }

bool HcclCommunicatorAttrs::GetInterServe() const { return interServer_; }

NICDeployment HcclCommunicatorAttrs::GetNicDeployment() const { return nicDeployment_; }

bool HcclCommunicatorAttrs::GetHaveCpuRank() const { return isHaveCpuRank_; }

u32 HcclCommunicatorAttrs::GetMeshAggregationRankSize() const { return meshAggregationRankSize_; }

bool HcclCommunicatorAttrs::GetInlineReduceSwitchOn() const { return inlineReduceSwitchOn_; }

u32 HcclCommunicatorAttrs::GetHostPort(s32 devicePhyId)
{
    if (GetExternalInputHcclIfBasePort() == HCCL_INVALID_PORT) {
        return (devicePhyId + HOST_PARA_BASE_PORT);
    } else {
        return (devicePhyId + GetExternalInputHcclIfBasePort() + HCCL_AISERVER_DEVICE_NUM);
    }
}

void HcclCommunicatorAttrs::SetNeedInitNicFlag(const bool isNeedInitNic) { isNeedInitNic_ = isNeedInitNic; }

// 判断是否是双环
bool CheckDoubleRingWithRohTopo(const std::vector<u32>& nicList)
{
    std::vector<u32> topoList;
    std::vector<u32> tmpNicList(nicList);
    std::sort(tmpNicList.begin(), tmpNicList.end());
    SearchPath searchPath;
    topoList = searchPath.Search(tmpNicList, true);
    if (topoList.empty()) {
        return false;
    }
    return true;
}
} // namespace hccl
