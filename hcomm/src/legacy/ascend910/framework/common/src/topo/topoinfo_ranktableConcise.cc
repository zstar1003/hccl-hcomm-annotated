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
#include <string>
#include <fstream>
#include <algorithm>
#include <map>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <arpa/inet.h>

#include "externalinput.h"
#include "log.h"
// ltm指定config路径
#include "common/src/config.h"
#include "workflow_pub.h"
#include "device_capacity.h"

using namespace std;
using namespace hccl;

constexpr u32 MAX_PORT_NUMBER = 65535;
constexpr u32 HCCL_SOCKET_PORT_RANGE_AUTO = 0;
constexpr u32 HCCL_DEVICE_PORT_DEFAULT = 16666;
constexpr u32 HCCL_BACKUP_DEVICE_PORT_DEFAULT = 16667;

TopoinfoRanktableConcise::TopoinfoRanktableConcise(const std::string& rankTableM, const std::string& identify)
    : TopoInfoRanktableParser(rankTableM, identify),
      isInterSuperPodRetryEnable_(GetExternalInputInterSuperPodRetryEnable())
{}

TopoinfoRanktableConcise::~TopoinfoRanktableConcise() { devIp2ObjIndex_.clear(); }

HcclResult TopoinfoRanktableConcise::Init()
{
    CHK_RET(LoadRankTableString(rankTableFile_));
    CHK_RET(ParserClusterInfo(params_, rankTable_));
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetClusterInfo(RankTable_t& clusterInfo)
{
    clusterInfo.nicDeploy = rankTable_.nicDeploy;
    clusterInfo.deviceNum = rankTable_.deviceNum;
    clusterInfo.serverNum = rankTable_.serverNum;
    clusterInfo.superPodNum = rankTable_.superPodNum;
    clusterInfo.rankNum = rankTable_.rankNum;
    clusterInfo.rankList = rankTable_.rankList;
    clusterInfo.serverList = rankTable_.serverList;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSelfClusterInfo(HcclCommParams& params)
{
    // 获取芯片类型信息
    params.deviceType = params_.deviceType;
    params.rank = params_.rank;
    params.userRank = params_.rank;
    params.logicDevId = params_.logicDevId;
    params.totalRanks = params_.totalRanks;
    params.serverId = params_.serverId;
    params.commPortConfig.devPortSwitchOn = params_.commPortConfig.devPortSwitchOn;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetClusterInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable)
{
    CHK_RET(GetClusterInfo(rankTable));
    CHK_RET(GetSelfClusterInfo(params));
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::SetIsInterSuperPodRetryEnable(bool isRetryEnable)
{
    isInterSuperPodRetryEnable_ = isRetryEnable;
    return HCCL_SUCCESS;
}

void TopoinfoRanktableConcise::DetectNicDepoly(RankTable_t& rankTable)
{
    // 只有当hostIp有效而且deviceIp无效时，才使用HOST侧网卡部署，目前策略要求集群中所有卡的deploy
    // 形式一致，所以取ranklist[0]的方式即可
    auto isIpInvalid = [](const std::vector<HcclIpAddress>& deviceIp) -> bool {
        for (auto& ip : deviceIp) {
            if (!ip.IsInvalid()) { // 遍历vector中所有的IP，只要有任意一个IP有效就认为是有效的，那么返回false
                return false;
            }
        }
        return true;
    };

    rankTable.nicDeploy = (!rankTable.rankList.empty() && !rankTable.rankList[0].hostIp.IsInvalid()
                           && isIpInvalid(rankTable.rankList[0].deviceInfo.deviceIp)) ?
                              NICDeployment::NIC_DEPLOYMENT_HOST :
                              NICDeployment::NIC_DEPLOYMENT_DEVICE;
}

HcclResult TopoinfoRanktableConcise::ParserClusterInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable)
{
    if (!IsTaskNumCalMode()) {
        CHK_RET(hrtGetDeviceType(params.deviceType));
    }
    // 获取ranktable info信息
    CHK_RET(GetRanktableInfo(rankTable));
    for (auto& rankInfo : rankTable.rankList) {
        HCCL_DEBUG(
            "ParserClusterInfo serverId %s, rankId %u, superDeviceId 0x%x, superPodId %s", rankInfo.serverId.c_str(),
            rankInfo.rankId, rankInfo.superDeviceId, rankInfo.superPodId.c_str());
    }

    if (IsTaskNumCalMode()) {
        HCCL_INFO("[ParserClusterInfo] get task num cal mode.");
        return HCCL_SUCCESS;
    }

    std::sort(
        rankTable.rankList.begin(), rankTable.rankList.end(), [&](const RankInfo_t& a, const RankInfo_t& b) -> bool {
            return a.rankId < b.rankId;
        });

    u32 rankId = INVALID_VALUE_RANKID;
    if (IsAllDigit(identify_.c_str()) != HCCL_SUCCESS
        || SalStrToULong(identify_, HCCL_BASE_DECIMAL, rankId) != HCCL_SUCCESS) {
        RPT_INPUT_ERR(
            true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
            std::vector<std::string>(
                {"HcclCommInitRootInfo", identify_.c_str(), "RANK_ID", "a non-negative decimal number"}));
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] rank_id[%s] is invalid", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA), identify_.c_str());
        return HCCL_E_PARA;
    }

    if (rankId >= rankTable.rankList.size()) {
        RPT_ENV_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(rankId), "rank_id", std::to_string(rankTable.rankList[rankId].rankId)}));
        HCCL_ERROR(
            "[%s][%s]rankId[%u] is invalid", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
            rankId);
        return HCCL_E_PARA;
    }

    RPT_ENV_ERR(
        rankId != rankTable.rankList[rankId].rankId, "EI0014",
        std::vector<std::string>({"value", "variable", "expect"}),
        std::vector<std::string>(
            {std::to_string(rankId), "rank_id", std::to_string(rankTable.rankList[rankId].rankId)}));
    CHK_PRT_RET(
        rankId != rankTable.rankList[rankId].rankId,
        HCCL_ERROR(
            "[%s][%s]check rankList[%u] rankId[%u] failed", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), rankId, rankTable.rankList[rankId].rankId),
        HCCL_E_UNAVAIL);

    if (rankId < rankTable.rankList.size()
        && rankTable.rankList[rankId].deviceInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST) {
        rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_HOST;
        SetHostUseDevNicFlag(false);
    } else {
        DetectNicDepoly(rankTable);
    }
    CHK_RET(CheckNicDeployConsistence(rankTable, rankTable.nicDeploy));

    u32 devId = rankTable.rankList[rankId].deviceInfo.devicePhyId;
    CHK_RET(hrtGetDevice(&params.logicDevId));

    u32 phyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(params.logicDevId), phyId));
    RPT_ENV_ERR(
        phyId != static_cast<u32>(devId), "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
        std::vector<std::string>({std::to_string(phyId), "devicePhyId", std::to_string(static_cast<u32>(devId))}));
    CHK_PRT_RET(
        phyId != static_cast<u32>(devId),
        HCCL_ERROR(
            "[%s][%s]ranktable config devId[%d],but local devId[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), devId, phyId),
        HCCL_E_UNAVAIL);

    params.rank = rankId;
    params.serverId = rankTable.rankList[rankId].serverId;
    params.totalRanks = rankTable.rankNum;

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::CheckNicDeployConsistence(RankTable_t& clusterInfo, NICDeployment deploy) const
{
    CHK_PRT_RET(
        clusterInfo.rankList.size() == 0, HCCL_DEBUG("rank list size is 0, skip nic deeply check."), HCCL_SUCCESS);

    for (auto& it : clusterInfo.rankList) {
        CHK_PRT_RET(
            // 因为部分场景下，网卡部署在device侧，但是允许deviceIP无效，所以不检测该场景，只检测使用HOST但是hostIp无效
            (deploy == NICDeployment::NIC_DEPLOYMENT_HOST && it.hostIp.IsInvalid()),
            HCCL_ERROR(
                "[Get][RanktableInfo]errNo"
                "[0x%016llx] hostIp config between ranks is different.",
                HCOM_ERROR_CODE(HCCL_E_PARA)),
            HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetRanktableInfo(RankTable_t& clusterInfo)
{
    // server_list
    CHK_RET(GetServerList(fileContent_, clusterInfo));
    CHK_RET(GetSuperPodList(fileContent_, clusterInfo));
    CHK_RET(GetDevNum(clusterInfo.rankList, clusterInfo.deviceNum));
    clusterInfo.rankNum = clusterInfo.rankList.size();
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetServerList(const nlohmann::json& obj, RankTable_t& clusterInfo)
{
    clusterInfo.serverList.clear();
    nlohmann::json serverList;
    CHK_RET(GetJsonProperty(obj, "server_list", serverList, false));
    HCCL_DEBUG("[%s.json] -> server_list: size:[%zu]", fileName_.c_str(), serverList.size());

    // 获取serverCount并校验
    std::string serverCount;
    HcclResult ret = GetJsonProperty(obj, "server_count", serverCount, true);
    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG("[%s.json] -> group_count: [%s]", fileName_.c_str(), serverCount.c_str());
        CHK_RET(SalStrToULong(serverCount, HCCL_BASE_DECIMAL, clusterInfo.serverNum));

        // 校验serverCount
        if (serverList.size() != clusterInfo.serverNum) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({serverCount, "server_count", std::to_string(serverList.size())}));
            HCCL_ERROR(
                "[%s][%s]errNo[0x%016llx] serverList size[%zu] neq server num[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA), serverList.size(),
                clusterInfo.serverNum);
            return HCCL_E_PARA;
        }
    } else if (ret == HCCL_E_NOT_FOUND) {
        clusterInfo.serverNum = serverList.size();
        HCCL_WARNING(
            "[Get][ServerList]The 'server_count' in ranktable is not found, "
            "set server_count to server_list size[%u]",
            serverList.size());
    } else {
        HCCL_ERROR("[Get][ServerList]get server_count error, ret[%d]", ret);
        return ret;
    }
    if (clusterInfo.serverNum == 0) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(clusterInfo.serverNum), "server_count", "is a positive integer"}));
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] server num is zero", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    HCCL_DEBUG("[Get][ServerList]serverNum is [%u]", clusterInfo.serverNum);

    for (u32 index = 0; index < serverList.size(); index++) {
        // get single server info
        CHK_RET(GetSingleServer(serverList, index, clusterInfo));
    }

    for (u32 index = 0; index < clusterInfo.rankList.size(); index++) {
        CHK_RET(VerifyBackupDeviceIpAndPort(clusterInfo.rankList, index));
    }
    // 获取device
    return HCCL_SUCCESS;
}

HcclResult
TopoinfoRanktableConcise::GetSingleServer(const nlohmann::json& serverListObj, u32 objIndex, RankTable_t& clusterInfo)
{
    HcclResult ret;
    std::string serverId;
    CHK_RET(GetJsonArrayMemberProperty(serverListObj, objIndex, "server_id", serverId, false));
    if (serverId.empty()) {
        HCCL_ERROR(
            "[Get][JsonArrayMemberProperty]errNo[0x%016llx] serverId[%s] is empty", HCOM_ERROR_CODE(HCCL_E_PARA),
            serverId.c_str());
        return HCCL_E_PARA;
    }

    if (serverId.length() > SERVERID_MAX_LENGTH) {
        HCCL_ERROR(
            "[Get][GetSingleServer]errNo[0x%016llx] serverId[%s] length[%u] is more than %u",
            HCOM_ERROR_CODE(HCCL_E_PARA), serverId.c_str(), serverId.length(), SERVERID_MAX_LENGTH);
        return HCCL_E_PARA;
    }

    // 将serverId添加到资源池,内部会进行IP地址校验，如果资源池中有serverId，则报错
    CHK_RET(CheckUniqueAndInsertPool(
        JsonUniqueInfoType::UNIQUE_INFO_TYPE_SERVER_ID, serverId, JsonCheckOpType::CHECK_OP_TYPE_INSERT));
    HCCL_DEBUG("server id[%u]:[%s]", objIndex, serverId.c_str());

    u32 serverIdx;
    GenerateServerIdx(serverId, serverIdx);
    HCCL_DEBUG("server id[%u]:[%s], serverIdx[%u]", objIndex, serverId.c_str(), serverIdx);
    std::string hostNicIp;
    HcclIpAddress hostIp;
    ret = GetJsonArrayMemberProperty(serverListObj, objIndex, "host_ip", hostNicIp, true);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS && ret != HCCL_E_NOT_FOUND, HCCL_ERROR("[Get][SingleServer]get host ip error"), ret);
    HCCL_DEBUG("[%s.json] -> host_ip: [%s]. ret[%u]", fileName_.c_str(), hostNicIp.c_str(), ret);
    if (ret != HCCL_E_NOT_FOUND) {
        CHK_RET(ConvertIpAddress(hostNicIp, hostIp));
    }

    // 处理ranklist
    ret = GetDeviceList(serverListObj, objIndex, clusterInfo, serverId, serverIdx, hostIp);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Get][SingleServer]get dev list error:serverId[%s]", serverId.c_str()), ret);

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetDeviceList(
    const nlohmann::json& serverListObj, u32 objIndex, RankTable_t& clusterInfo, std::string& serverId, u32& serverIdx,
    HcclIpAddress& hostIp)
{
    HCCL_DEBUG("Get GetDeviceList[%u]: serverId[%s]", objIndex, serverId.c_str());

    nlohmann::json deviceList;
    CHK_RET(GetJsonArrayMemberProperty(serverListObj, objIndex, "device", deviceList, false));

    HCCL_DEBUG("[%s.json] -> device_list: size:%zu", fileName_.c_str(), deviceList.size());

    CHK_PRT_RET(deviceList.size() == 0, HCCL_ERROR("[Get][DeviceList]deviceList size is zero"), HCCL_E_PARA);

    for (u32 index = 0; index < deviceList.size(); index++) {
        // get single server info
        CHK_RET(GetSingleDevice(deviceList, index, clusterInfo, serverId, serverIdx, hostIp));
    }

    // 检查devip的数目是否一致
    u32 rankListSize = clusterInfo.rankList.size();
    CHK_PRT_RET(rankListSize == 0, HCCL_ERROR("[Get][DeviceList]get ranklist is zero"), HCCL_E_PARA);

    u32 checkDeviceIpSize = 0;
    for (u32 index = 0; index < clusterInfo.rankList.size(); index++) {
        if (index == 0) {
            checkDeviceIpSize = clusterInfo.rankList[0].deviceInfo.deviceIp.size();
        }
        if (clusterInfo.rankList[index].deviceInfo.deviceIp.size() != checkDeviceIpSize) {
            HCCL_ERROR(
                "[Get][DeviceList]device[%u] size[%u] neq first device size[%u] error", index,
                clusterInfo.rankList[index].deviceInfo.deviceIp.size(), checkDeviceIpSize);
            return HCCL_E_PARA;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult
TopoinfoRanktableConcise::GetSingleNicInfo(const nlohmann::json& serverListObj, u32 objIndex, RankInfo_t& rankinfo)
{
    std::string netPosition;
    rankinfo.deviceInfo.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    HcclResult ret = GetJsonArrayMemberProperty(serverListObj, objIndex, "net_position", netPosition, true);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS && ret != HCCL_E_NOT_FOUND, HCCL_ERROR("[Get][GetSingleNicInfo]get net position error"),
        ret);
    HCCL_DEBUG("[%s.json] -> net_position: [%s]. ret[%u]", fileName_.c_str(), netPosition.c_str(), ret);
    if (ret != HCCL_E_NOT_FOUND) {
        if (netPosition == "host") { // 默认值为Device模式
            rankinfo.deviceInfo.nicDeploy = NICDeployment::NIC_DEPLOYMENT_HOST;
        } else {
            rankinfo.deviceInfo.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
        }
    }

    std::string netProto;
    ret = GetJsonArrayMemberProperty(serverListObj, objIndex, "net_protocol", netProto, true);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS && ret != HCCL_E_NOT_FOUND, HCCL_ERROR("[Get][GetSingleNicInfo]get net protocol error"),
        ret);
    HCCL_DEBUG("[%s.json] -> net_protocol: [%s]. ret[%u]", fileName_.c_str(), netProto.c_str(), ret);
    if (ret != HCCL_E_NOT_FOUND) {
        if (netProto == "rdma") { // 未配置则使用默认值，不处理，使用原有逻辑
            SetExternalInputProtocolType(ProtocolType::RDMA);
            rankinfo.deviceInfo.proto = u32(ProtocolType::RDMA);
        } else if (netProto == "tcp") {
            SetExternalInputProtocolType(ProtocolType::TCP);
            rankinfo.deviceInfo.proto = u32(ProtocolType::TCP);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSingleDevice(
    const nlohmann::json& deviceListObj, u32 objIndex, RankTable_t& clusterInfo, std::string& serverId, u32& serverIdx,
    HcclIpAddress& hostIp)
{
    // 获取rank_id
    std::string rankId;
    CHK_RET(GetJsonArrayMemberProperty(deviceListObj, objIndex, "rank_id", rankId, false));
    HCCL_DEBUG("[%s.json] -> rank_id: [%s]", fileName_.c_str(), rankId.c_str());

    // 获取device type
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    // 获取device_id
    std::string strDevid;
    CHK_RET(GetJsonArrayMemberProperty(deviceListObj, objIndex, "device_id", strDevid, false));

    u32 devicePhyId = 0;
    CHK_RET(SalStrToULong(strDevid, HCCL_BASE_DECIMAL, devicePhyId));

    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    if ((deviceType == DevType::DEV_TYPE_310P3 || deviceType == DevType::DEV_TYPE_910B
         || deviceType == DevType::DEV_TYPE_910_93)
        && devicePhyId > (maxDeviceNum - 1)) {
        // deviceid in 0 ~ maxDeviceNum
        HCCL_ERROR(
            "[Get][SingleDevice]errNo[0x%016llx] device_id[%u] more than maxDeviceNum[%u] invalid",
            HCOM_ERROR_CODE(HCCL_E_PARA), devicePhyId, maxDeviceNum);
        return HCCL_E_PARA;
    } else if (
        (deviceType != DevType::DEV_TYPE_310P3 && deviceType != DevType::DEV_TYPE_910B
         && deviceType != DevType::DEV_TYPE_910_93)
        && devicePhyId > (HCCL_AISERVER_DEVICE_NUM - 1)) {
        // deviceid in 0 ~ 7
        HCCL_ERROR(
            "[Get][SingleDevice]errNo[0x%016llx] device_id[%u] more than 7 is invalid", HCOM_ERROR_CODE(HCCL_E_PARA),
            devicePhyId);
        return HCCL_E_PARA;
    }
    HCCL_DEBUG("[%s.json] -> device_id: [%s]", fileName_.c_str(), strDevid.c_str());

    RankInfo_t rankinfo;
    rankinfo.serverId = serverId;
    rankinfo.serverIdx = serverIdx;
    rankinfo.hostIp = hostIp;
    rankinfo.deviceInfo.devicePhyId = devicePhyId;

    CHK_RET(GetSingleDeviceHostPort(deviceListObj, objIndex, rankinfo));
    CHK_RET(GetSingleDeviceIp(deviceListObj, objIndex, clusterInfo, rankinfo, deviceType, rankinfo.hostIp.IsInvalid()));
    CHK_RET(GetSingleDevicePort(deviceListObj, objIndex, rankinfo));
    CHK_RET(GetSingleBackupDeviceIp(deviceListObj, objIndex, rankinfo));
    CHK_RET(GetSingleNicInfo(deviceListObj, objIndex, rankinfo));

    if (SalStrToULong(rankId, HCCL_BASE_DECIMAL, rankinfo.rankId) != HCCL_SUCCESS) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({rankId, "rank_id", "is a non-negative integer."}));
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] rankid[%s] is invalid", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA), rankId.c_str());
        return HCCL_E_PARA;
    }

    rankinfo.podName = ""; // podname在新场景下置空
    rankId = "";

    string version;
    CHK_RET(GetRanktableVersion(version));
    if (version.compare(SUPERPOD_CLUSTER_VERSION) != 0) {
        clusterInfo.rankList.push_back(rankinfo);
        HCCL_DEBUG(
            "[%s.json]->rankId[%u], serverId[%s], devicePhyId[%d]", fileName_.c_str(), rankinfo.rankId,
            rankinfo.serverId.c_str(), rankinfo.deviceInfo.devicePhyId);

        return HCCL_SUCCESS;
    }

    CHK_RET(GetSingleSuperDeviceId(deviceListObj, objIndex, clusterInfo, rankinfo));
    clusterInfo.rankList.push_back(rankinfo);
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::SplitString(
    const std::string& str, const std::string& strC, std::vector<std::string>& strVector) const
{
    std::string::size_type startPos = 0;
    std::string::size_type foundPos = str.find(strC);

    while (foundPos != std::string::npos) {
        strVector.push_back(str.substr(startPos, foundPos - startPos));
        startPos = foundPos + strC.size();
        foundPos = str.find(strC, startPos);
    }
    if (startPos != str.length()) {
        strVector.push_back(str.substr(startPos));
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSingleDeviceIp(
    const nlohmann::json& deviceListObj, u32 objIndex, const RankTable_t& clusterInfo, RankInfo_t& rankinfo,
    DevType deviceType, bool invalidHostIp)
{
    // 获取device_ip （可能有多个）
    std::string deviceIp;
    HcclResult ret = GetJsonArrayMemberProperty(deviceListObj, objIndex, "device_ip", deviceIp, true);
    // 多机和走roce网卡ranktable必须有“device_ip”字段，单机可以没有
    if (clusterInfo.serverNum > 1 || (GetExternalInputIntraRoceSwitch() == 1)) {
        // 如果没有配置HostIp，那么deviceIp必须有效，否则出错
        bool isDeviceIpError
            = (ret != HCCL_SUCCESS && invalidHostIp) || (deviceType == DevType::DEV_TYPE_910B && deviceIp == "");
        // A3超节点内使用vnic建链，不需要device ip
        if (deviceType == DevType::DEV_TYPE_910_93 && clusterInfo.superPodNum <= 1
            && GetExternalInputIntraRoceSwitch() == 0) {
            isDeviceIpError = false;
        }
        RPT_INPUT_ERR(
            isDeviceIpError, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({deviceIp, "device_ip", "a valid IP address"}));
        CHK_PRT_RET(
            isDeviceIpError,
            HCCL_ERROR(
                "[%s][%s]'device_ip' is not set correctly, "
                "must be set when multi Server or HCCL_INTRA_ROCE_ENABLE enabled",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str()),
            HCCL_E_PARA);
    } else if (clusterInfo.serverNum == 1 && ret == HCCL_E_NOT_FOUND) {
        HcclIpAddress invalidAddr;
        rankinfo.deviceInfo.deviceIp.push_back(invalidAddr);
        HCCL_WARNING("[Get][SingleDeviceIp]'device_ip' in ranktable is not set!");
        return HCCL_SUCCESS;
    } else {
        // 如果没有配置HostIp，那么deviceIp必须有效，否则出错
        CHK_PRT_RET(
            ret != HCCL_SUCCESS && invalidHostIp,
            HCCL_ERROR(
                "[Get][SingleDeviceIp]errNo[0x%016llx] 'device_ip' is not set correctly", HCOM_ERROR_CODE(HCCL_E_PARA)),
            ret);
    }
    HCCL_DEBUG("[%s.json] -> device_ip: [%s]", fileName_.c_str(), deviceIp.c_str());

    // 处理字符串device_ip
    std::vector<std::string> strDeviceIp;
    if (deviceIp != "") {
        CHK_RET(SplitString(deviceIp, ",", strDeviceIp));

        CHK_PRT_RET(
            strDeviceIp.size() == 0, HCCL_ERROR("[Get][SingleDeviceIp]in device:deviceip size is zero"), HCCL_E_PARA);
        for (u32 index = 0; index < strDeviceIp.size(); index++) {
            CHK_RET(CheckUniqueAndInsertPool(
                JsonUniqueInfoType::UNIQUE_INFO_TYPE_DEVICE_IP, strDeviceIp[index],
                JsonCheckOpType::CHECK_OP_TYPE_INSERT));
            HcclIpAddress ipAddr;
            CHK_RET(ConvertIpAddress(strDeviceIp[index], ipAddr));
            rankinfo.deviceInfo.deviceIp.push_back(ipAddr);
            devIp2ObjIndex_.emplace(ipAddr.GetReadableIP(), objIndex);
        }
    } else {
        HcclIpAddress invalidAddr;
        rankinfo.deviceInfo.deviceIp.push_back(invalidAddr);
        HCCL_WARNING("objIndex[%u],'device_ip' is not set", objIndex);
    }

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSingleBackupDeviceIp(
    const nlohmann::json& deviceListObj, u32 objIndex, RankInfo_t& rankinfo)
{
    if (params_.deviceType != DevType::DEV_TYPE_910_93 || !GetExternalInputHcclAicpuUnfold()
        || !isInterSuperPodRetryEnable_) {
        return HCCL_SUCCESS;
    }
    // 获取backup_device_ip（可能有多个）
    std::string backupDeviceIp;
    HcclResult ret = GetJsonArrayMemberProperty(deviceListObj, objIndex, "backup_device_ip", backupDeviceIp);
    // backup_device_ip字段未配置时通过warning日志提示
    if (ret == HCCL_E_NOT_FOUND) {
        HcclIpAddress invalidAddr;
        rankinfo.deviceInfo.backupDeviceIp.push_back(invalidAddr);
        HCCL_WARNING("[Get][SingleDeviceIp]'backup_device_ip' in ranktable is not set!");
        return HCCL_SUCCESS;
    } else if (ret == HCCL_E_PARA) {
        HCCL_ERROR("[Get][SingleDeviceIp]Get json array member property error");
        return HCCL_E_PARA;
    }
    HCCL_DEBUG("[%s.json] -> backup_device_ip: [%s]", fileName_.c_str(), backupDeviceIp.c_str());

    // 处理backup_device_ip字符串
    std::vector<std::string> strBackupDeviceIp;
    if (backupDeviceIp != "") {
        CHK_RET(SplitString(backupDeviceIp, ",", strBackupDeviceIp));

        CHK_PRT_RET(
            strBackupDeviceIp.size() == 0, HCCL_ERROR("[Get][SingleBackupDeviceIp]in device: deviceip size is zero"),
            HCCL_E_PARA);
        for (u32 index = 0; index < strBackupDeviceIp.size(); index++) {
            CHK_RET(CheckUniqueAndInsertPool(
                JsonUniqueInfoType::UNIQUE_INFO_TYPE_BACKUP_DEVICE_IP, strBackupDeviceIp[index],
                JsonCheckOpType::CHECK_OP_TYPE_INSERT));
            HcclIpAddress ipAddr;
            CHK_RET(ConvertIpAddress(strBackupDeviceIp[index], ipAddr));
            rankinfo.deviceInfo.backupDeviceIp.push_back(ipAddr);
        }
        CHK_RET(GetSingleBackupDevicePort(deviceListObj, objIndex, rankinfo));
        HCCL_INFO(
            "[TopoinfoRanktableConcise][GetSingleBackupDeviceIp]devicePhyId[%u], backupDeviceIp[0]:[%s].",
            rankinfo.deviceInfo.devicePhyId, rankinfo.deviceInfo.backupDeviceIp[0].GetReadableIP());
    } else {
        HcclIpAddress invalidAddr;
        rankinfo.deviceInfo.backupDeviceIp.push_back(invalidAddr);
        HCCL_WARNING("objIndex[%u], 'backup_device_ip' is not set", objIndex);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSingleDeviceHostPort(
    const nlohmann::json& deviceListObj, u32 objIndex, RankInfo_t& rankinfo)
{
    std::string hostPortStr;
    rankinfo.hostPort = HCCL_INVALID_PORT;
    HcclResult ret = GetJsonArrayMemberProperty(deviceListObj, objIndex, "host_port", hostPortStr, true);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS && ret != HCCL_E_NOT_FOUND,
        HCCL_ERROR("[Get][SingleDeviceHostPort]get host port error, deviceIndex[%u]", objIndex), ret);
    HCCL_DEBUG("[%s.json] -> host_port: [%s]. ret[%u]", fileName_.c_str(), hostPortStr.c_str(), ret);
    if (ret != HCCL_E_NOT_FOUND) {
        CHK_RET(SalStrToULong(hostPortStr, HCCL_BASE_DECIMAL, rankinfo.hostPort));
        CHK_PRT_RET(
            rankinfo.hostPort == HCCL_SOCKET_PORT_RANGE_AUTO,
            HCCL_ERROR(
                "[Get][SingleDeviceHostPort] deviceIndex[%u], please do not use the reserved port number[%u]", objIndex,
                HCCL_SOCKET_PORT_RANGE_AUTO),
            HCCL_E_PARA);
        CHK_PRT_RET(
            rankinfo.hostPort > MAX_PORT_NUMBER,
            HCCL_ERROR(
                "[Get][SingleDeviceHostPort] deviceIndex[%u], port number[%u] exceed max port number[%u]", objIndex,
                rankinfo.hostPort, MAX_PORT_NUMBER),
            HCCL_E_PARA);
        HCCL_INFO(
            "[TopoinfoRanktableConcise][GetSingleDeviceHostPort] deviceIndex[%u], devicePhyId[%u], "
            "get HOST port[%u].",
            objIndex, rankinfo.deviceInfo.devicePhyId, rankinfo.hostPort);
    } else {
        HCCL_INFO(
            "[Get][SingleDeviceHostPort] deviceIndex[%u], 'host_port' in ranktable is not set. "
            "Multi-process may not be supported for op retry.",
            objIndex);
    }
    return HCCL_SUCCESS;
}

HcclResult
TopoinfoRanktableConcise::GetSingleDevicePort(const nlohmann::json& deviceListObj, u32 objIndex, RankInfo_t& rankinfo)
{
    // 获取device指定的port；如果用户未配置，则置为缺省值
    std::string strDevPort;
    HcclResult ret = GetJsonArrayMemberProperty(deviceListObj, objIndex, "device_port", strDevPort, true);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS && ret != HCCL_E_NOT_FOUND,
        HCCL_ERROR("[Get][SingleDevicePort]Get json array member property error."), ret);
    if (ret == HCCL_E_NOT_FOUND) {
        rankinfo.deviceInfo.port = HCCL_INVALID_PORT;
        HCCL_INFO(
            "[Get][SingleDevicePort]deviceIndex[%u], devicePhyId[%u], 'device_port' in ranktable is not set. "
            "Multi-process may not be supported for device nic.",
            objIndex, rankinfo.deviceInfo.devicePhyId);
        params_.commPortConfig.devPortSwitchOn = false; // 不启用用户指定的port作为device网卡通信的port
        return HCCL_SUCCESS;
    } else {
        CHK_RET(SalStrToULong(strDevPort, HCCL_BASE_DECIMAL, rankinfo.deviceInfo.port));
        CHK_PRT_RET(
            rankinfo.deviceInfo.port == HCCL_SOCKET_PORT_RANGE_AUTO,
            HCCL_ERROR(
                "[Get][SingleDevicePort] deviceIndex[%u], devicePhyId[%u], "
                "please do not use the reserved port number [%u]. ",
                objIndex, rankinfo.deviceInfo.devicePhyId, HCCL_SOCKET_PORT_RANGE_AUTO),
            HCCL_E_PARA);
        CHK_PRT_RET(
            rankinfo.deviceInfo.port > MAX_PORT_NUMBER,
            HCCL_ERROR(
                "[Get][SingleDevicePort] deviceIndex[%u], devicePhyId[%u], "
                "port number[%u] exceed max port number[%u]",
                objIndex, rankinfo.deviceInfo.devicePhyId, rankinfo.deviceInfo.port, MAX_PORT_NUMBER),
            HCCL_E_PARA);
        params_.commPortConfig.devPortSwitchOn = true; // 启用用户指定的port作为device网卡通信的port
    }

    // 获取device指定的vnic port；如果未配置，则与device_port相同（仅用于MasterInfo协商解析Ranktable）
    std::string strVnicPort;
    ret = GetJsonArrayMemberProperty(deviceListObj, objIndex, "device_vnic_port", strVnicPort, true);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS && ret != HCCL_E_NOT_FOUND,
        HCCL_ERROR("[Get][SingleDevicePort]Get json array member property vnic error."), ret);
    if (ret == HCCL_E_NOT_FOUND) {
        rankinfo.deviceInfo.vnicPort = rankinfo.deviceInfo.port;
        HCCL_INFO(
            "[Get][SingleDevicePort]deviceIndex[%u], devicePhyId[%u], "
            "'device_vnic_port' in ranktable is not set. ",
            objIndex, rankinfo.deviceInfo.devicePhyId);
    } else {
        CHK_RET(SalStrToULong(strVnicPort, HCCL_BASE_DECIMAL, rankinfo.deviceInfo.vnicPort));
        CHK_PRT_RET(
            rankinfo.deviceInfo.vnicPort == HCCL_SOCKET_PORT_RANGE_AUTO,
            HCCL_ERROR(
                "[Get][SingleDevicePort] deviceIndex[%u], devicePhyId[%u], "
                "please do not use the reserved port number [%u] as vnic port number. ",
                objIndex, rankinfo.deviceInfo.devicePhyId, HCCL_SOCKET_PORT_RANGE_AUTO),
            HCCL_E_PARA);
        CHK_PRT_RET(
            rankinfo.deviceInfo.vnicPort > MAX_PORT_NUMBER,
            HCCL_ERROR(
                "[Get][SingleDevicePort] deviceIndex[%u], devicePhyId[%u], "
                "vnic port number[%u] exceed max port number[%u]",
                objIndex, rankinfo.deviceInfo.devicePhyId, rankinfo.deviceInfo.vnicPort, MAX_PORT_NUMBER),
            HCCL_E_PARA);
    }

    HCCL_INFO(
        "[TopoinfoRanktableConcise][GetSingleDevicePort] deviceIndex[%u], devicePhyId[%u], get device port[%u], "
        "device vnic port[%u].",
        objIndex, rankinfo.deviceInfo.devicePhyId, rankinfo.deviceInfo.port, rankinfo.deviceInfo.vnicPort);
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSingleBackupDevicePort(
    const nlohmann::json& deviceListObj, u32 objIndex, RankInfo_t& rankinfo) const
{
    // 获取device指定的backup port；如果用户未配置，则置为缺省值
    std::string strBackupPort;
    HcclResult ret = GetJsonArrayMemberProperty(deviceListObj, objIndex, "backup_device_port", strBackupPort, true);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS && ret != HCCL_E_NOT_FOUND,
        HCCL_ERROR("[Get][SingleBackupDevicePort]Get json array member property error."), ret);
    if (ret == HCCL_E_NOT_FOUND) {
        rankinfo.deviceInfo.backupPort = HCCL_INVALID_PORT;
        HCCL_INFO(
            "[Get][SingleBackupDevicePort]deviceIndex[%u], devicePhyId[%u], "
            "'backup_device_port' in ranktable is not set. Multi-process may not be supported for backup nic.",
            objIndex, rankinfo.deviceInfo.devicePhyId);
        return HCCL_SUCCESS;
    }

    CHK_RET(SalStrToULong(strBackupPort, HCCL_BASE_DECIMAL, rankinfo.deviceInfo.backupPort));
    CHK_PRT_RET(
        rankinfo.deviceInfo.backupPort == HCCL_SOCKET_PORT_RANGE_AUTO,
        HCCL_ERROR(
            "[Get][SingleBackupDevicePort] deviceIndex[%u], devicePhyId[%u], "
            "please do not use the reserved port number [%u]. ",
            objIndex, rankinfo.deviceInfo.devicePhyId, HCCL_SOCKET_PORT_RANGE_AUTO),
        HCCL_E_PARA);
    CHK_PRT_RET(
        rankinfo.deviceInfo.backupPort > MAX_PORT_NUMBER,
        HCCL_ERROR(
            "[Get][SingleBackupDevicePort] deviceIndex[%u], devicePhyId[%u], "
            "port number[%u] exceed max port number[%u]",
            objIndex, rankinfo.deviceInfo.devicePhyId, rankinfo.deviceInfo.backupPort, MAX_PORT_NUMBER),
        HCCL_E_PARA);

    HCCL_INFO(
        "[TopoinfoRanktableConcise][GetSingleBackupDevicePort] deviceIndex[%u], devicePhyId[%u], "
        "get backup device port[%u].",
        objIndex, rankinfo.deviceInfo.devicePhyId, rankinfo.deviceInfo.backupPort);
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::VerifyBackupDeviceIpAndPort(std::vector<RankInfo_t>& rankList, u32 devIndex)
{
    if (params_.deviceType != DevType::DEV_TYPE_910_93 || !GetExternalInputHcclAicpuUnfold()
        || !isInterSuperPodRetryEnable_) {
        return HCCL_SUCCESS;
    }
    RankInfo_t& rankInfo = rankList[devIndex];

    for (auto& backupDevIp : rankInfo.deviceInfo.backupDeviceIp) {
        if (backupDevIp.IsInvalid()) {
            // 无效备用IP，无需校验
            continue;
        }
        string backupDevIpStr = backupDevIp.GetReadableIP();
        if (devIp2ObjIndex_.find(backupDevIpStr) == devIp2ObjIndex_.end()) {
            HCCL_RUN_WARNING(
                "[Verify][BackupDeviceIp]"
                "Backup devIp[%s] for devicePhyId[%d] is not in this comm. "
                "The validation of this backup ip could not be verified! "
                "Please notice it might be an invalid backup ip!",
                backupDevIpStr.c_str(), devIndex);
            continue;
        }

        RankInfo_t& backupRankInfo = rankList[devIp2ObjIndex_[backupDevIpStr]];

        s32 backupDevPhyId = backupRankInfo.deviceInfo.devicePhyId;
        CHK_PRT_RET(
            backupDevPhyId == rankInfo.deviceInfo.devicePhyId,
            HCCL_ERROR(
                "[Verify][BackupDeviceIp]"
                "PhyId[%d] for backup devIp[%s] is the same with self device[%u] phyId[%d]. "
                "Please do not use self ip as backup ip!",
                backupDevPhyId, backupDevIpStr.c_str(), devIndex, rankInfo.deviceInfo.devicePhyId),
            HCCL_E_PARA);

        LinkTypeInServer linkType = LinkTypeInServer::RESERVED_LINK_TYPE;
        CHK_RET(hrtGetPairDeviceLinkType(rankInfo.deviceInfo.devicePhyId, backupDevPhyId, linkType));
        CHK_PRT_RET(
            linkType != LinkTypeInServer::SIO_TYPE,
            HCCL_ERROR(
                "[Verify][BackupDeviceIp]errNo[0x%016llx], device[%u], "
                "link between device phyId[%d] and backup device phyId[%d] is not sio link, backup device ip[%s]. "
                "Please check backup ip validation and whether it is on a pair device!",
                HCOM_ERROR_CODE(HCCL_E_PARA), devIndex, rankInfo.deviceInfo.devicePhyId, backupDevPhyId,
                backupDevIpStr.c_str()),
            HCCL_E_PARA);

        // 用于备用网卡的端口不可和用于主网卡的端口冲突
        CHK_PRT_RET(
            rankInfo.deviceInfo.backupPort != HCCL_INVALID_PORT && backupRankInfo.deviceInfo.port != HCCL_INVALID_PORT
                && rankInfo.deviceInfo.backupPort == backupRankInfo.deviceInfo.port,
            HCCL_ERROR(
                "[Verify][BackupDevicePort] deviceIndex[%u], "
                "backup device port[%u] on devPhyId[%u] should not be the same with device port[%u] on devPhyId[%u].",
                devIndex, rankInfo.deviceInfo.backupPort, rankInfo.deviceInfo.devicePhyId,
                backupRankInfo.deviceInfo.port, backupRankInfo.deviceInfo.devicePhyId),
            HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSingleSuperDeviceId(
    const nlohmann::json& deviceListObj, u32 objIndex, [[maybe_unused]] const RankTable_t& clusterInfo,
    RankInfo_t& rankinfo)
{
    // 获取super_device_id
    HcclResult ret;
    std::string strSuperDeviceId;
    ret = GetJsonArrayMemberProperty(deviceListObj, objIndex, "super_device_id", strSuperDeviceId, true);

    CHK_PRT_RET(
        ret == HCCL_E_NOT_FOUND, HCCL_WARNING("[Get][SingleSuperDeviceId]'super_device_id' is not found"),
        HCCL_SUCCESS);

    u32 superDeviceId = 0;
    ret = SalStrToULong(strSuperDeviceId, HCCL_BASE_DECIMAL, superDeviceId);
    if (ret != HCCL_SUCCESS) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({strSuperDeviceId, "super_device_id", "a valid non-negative integer"}));
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] super_device_id[%s] is invalid", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA), strSuperDeviceId.c_str());
        return ret;
    }

    rankinfo.superDeviceId = superDeviceId;
    HCCL_DEBUG("[%s.json] -> super_device_id: [%s]", fileName_.c_str(), strSuperDeviceId.c_str());
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSuperPodList(const nlohmann::json& obj, RankTable_t& clusterInfo)
{
    string version;
    CHK_RET(GetRanktableVersion(version));
    CHK_PRT_RET(
        version.compare(SUPERPOD_CLUSTER_VERSION) != 0,
        HCCL_INFO("[Get][SuperPodList]ranktable version[%s], do nothing.", version.c_str()), HCCL_SUCCESS);

    if (!IsTaskNumCalMode()) { // taskNum评估阶段，无法判断是否超节点模式
        // 环境配置非超节点时，不解析超节点信息, ranktable中记录的superPodId, superDeviceId, superPodNum均为默认值
        bool useSuperPodMode = false;
        CHK_RET(IsSuperPodMode(useSuperPodMode));
        CHK_PRT_RET(!useSuperPodMode, HCCL_INFO("[Get][SuperPodList]not super pod, do nothing"), HCCL_SUCCESS);
    }

    HcclResult ret;
    nlohmann::json superPodList;
    ret = GetJsonProperty(obj, "super_pod_list", superPodList, true);
    CHK_PRT_RET(
        ret == HCCL_E_NOT_FOUND, HCCL_WARNING("[Get][SuperPodList]'super_pod_list' is not found"), HCCL_SUCCESS);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS && ret != HCCL_E_NOT_FOUND,
        HCCL_ERROR("[Get][SuperPodList]'super_pod_list' in ranktable is not set correctly, ret[%d]", ret), ret);
    HCCL_DEBUG("[%s.json]super_pod_list -> : size:[%zu]", fileName_.c_str(), superPodList.size());

    for (u32 index = 0; index < superPodList.size(); index++) {
        CHK_RET(GetSingleSuperPod(superPodList, index, clusterInfo));
    }

    clusterInfo.superPodNum = superPodList.size();
    CHK_RET(CheckSuperPodInfo(clusterInfo));
    return HCCL_SUCCESS;
}

HcclResult
TopoinfoRanktableConcise::GetSingleSuperPod(const nlohmann::json& superPodList, u32 objIndex, RankTable_t& clusterInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string superPodId;
    ret = GetJsonArrayMemberProperty(superPodList, objIndex, "super_pod_id", superPodId, true);
    if (ret != HCCL_SUCCESS || superPodId.empty()) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({superPodId, "super_pod_id", "a valid non-empty string."}));
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] super_pod_id[%s] is invalid", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(ret), superPodId.c_str());
        return ret;
    }

    // 将superPodId添加到资源池进行查重校验
    CHK_RET(CheckUniqueAndInsertPool(
        JsonUniqueInfoType::UNIQUE_INFO_TYPE_SUPER_POD_ID, superPodId, JsonCheckOpType::CHECK_OP_TYPE_INSERT));
    HCCL_DEBUG("superPod id[%u]:[%s]", objIndex, superPodId.c_str());

    // 处理ranklist
    ret = GetSuperPodServerList(superPodList, objIndex, clusterInfo, superPodId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Get][SingleSuperPod]get server list error:superPodId[%s]", superPodId.c_str()), ret);

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSuperPodServerList(
    const nlohmann::json& superPodList, u32 objIndex, RankTable_t& clusterInfo, std::string superPodId)
{
    HCCL_DEBUG("GetSuperPodServerList[%u]: superPodId[%s]", objIndex, superPodId.c_str());

    nlohmann::json superPodServerList;
    CHK_RET(GetJsonArrayMemberProperty(superPodList, objIndex, "server_list", superPodServerList, false));
    for (u32 index = 0; index < superPodServerList.size(); index++) {
        // get single super pod server info
        CHK_RET(GetSingleSuperPodSever(superPodServerList, index, clusterInfo, superPodId));
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::GetSingleSuperPodSever(
    const nlohmann::json& superPodServerList, u32 objIndex, RankTable_t& clusterInfo, std::string superPodId)
{
    std::string serverId;
    HcclResult ret = HCCL_SUCCESS;
    do {
        // 获取server_id
        ret = GetJsonArrayMemberProperty(superPodServerList, objIndex, "server_id", serverId, true);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Get][SingleSuperPodSever]errNo[0x%016llx] server_id is not found or invalid",
                HCCL_ERROR_CODE(ret)), );

        // 将super_pod_list下的server_id在资源池中进行校验
        ret = CheckUniqueAndInsertPool(
            JsonUniqueInfoType::UNIQUE_INFO_TYPE_SERVER_ID, serverId, JsonCheckOpType::CHECK_OP_TYPE_FIND);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Get][SingleSuperPodSever]errNo[0x%016llx]server_id[%s] is not found in server_list",
                HCCL_ERROR_CODE(ret), serverId.c_str()), );
        HCCL_DEBUG("[%s.json]super_pod_list -> server_id: [%s]", fileName_.c_str(), serverId.c_str());
    } while (0);

    // server_id未找到, 或与server_list中的server_id不一致
    if (ret != HCCL_SUCCESS) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({serverId, "server_id", "a valid non-empty string."}));
        return ret;
    }

    u32 superPodIdx = INVALID_UINT;
    GenerateSuperPodIdx(superPodId, superPodIdx);

    bool isFound = false;
    for (RankInfo_t& rankInfo : clusterInfo.rankList) {
        if (rankInfo.serverId == serverId) {
            rankInfo.superPodId = superPodId;
            rankInfo.superPodIdx = superPodIdx;
            isFound = true;
        }
    }
    CHK_PRT_RET(
        isFound == false,
        HCCL_ERROR("[Get][SingleSuperPodSever]server_id[%s] in super_pod_list is not in server_list", serverId.c_str()),
        HCCL_E_PARA);

    HCCL_DEBUG(
        "[%s.json]super_pod_list -> server_id[%s], super_pod_id[%s]", fileName_.c_str(), serverId.c_str(),
        superPodId.c_str());

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableConcise::CheckSuperPodInfo(RankTable_t& clusterInfo) const
{
    std::map<std::string, std::set<std::string>> superPodMap; // superPodId -> serverId
    std::map<std::string, std::set<u32>> superPodSdidMap;     // super_pod_id -> superDeviceId
    for (RankInfo_t& rankInfo : clusterInfo.rankList) {
        // 超节点模式下, 校验superPodId和sdid值有效
        CHK_PRT_RET(
            rankInfo.superPodId.empty() || rankInfo.superDeviceId == INVALID_UINT,
            HCCL_ERROR(
                "[Check][SuperPodInfo]superDeviceId[0x%x] or superPod[%s] in rank[%u] is invalid",
                rankInfo.superDeviceId, rankInfo.superPodId.c_str(), rankInfo.rankId),
            HCCL_E_PARA);

        auto it = superPodMap.find(rankInfo.superPodId);
        if (it == superPodMap.end()) {
            std::set<std::string> serverIdSet;
            serverIdSet.insert(rankInfo.serverId);
            superPodMap.insert({rankInfo.superPodId, serverIdSet});
        } else if (it->second.find(rankInfo.serverId) == it->second.end()) {
            it->second.insert(rankInfo.serverId);
        }
        // 用户忘记配置，superDeviceId等于无效值
        CHK_PRT_RET(
            rankInfo.superDeviceId == INVALID_UINT,
            HCCL_ERROR(
                "[Check][SuperPodInfo]superDeviceId[0x%x] is invalid in rankId[%u], "
                "the configuration may be missing, please check the ranktable config!",
                rankInfo.superDeviceId, rankInfo.rankId),
            HCCL_E_PARA);

        auto iter = superPodSdidMap.find(rankInfo.superPodId);
        if (iter == superPodSdidMap.end()) {
            std::set<u32> superDeviceIdSet;
            superDeviceIdSet.insert(rankInfo.superDeviceId);
            superPodSdidMap.insert({rankInfo.superPodId, superDeviceIdSet});
        } else if (iter->second.find(rankInfo.superDeviceId) == iter->second.end()) {
            iter->second.insert(rankInfo.superDeviceId);
        } else {
            // 超节点内superDeviceId在超节点内唯一
            CHK_PRT_RET(
                iter->second.find(rankInfo.superDeviceId) != iter->second.end(),
                HCCL_ERROR(
                    "[Verify][SuperPodInfo]superDeviceId[0x%x] in superPod[%s] "
                    "already exists.",
                    rankInfo.superDeviceId, iter->first.c_str()),
                HCCL_E_PARA);
        }
    }

    u32 serverNumTotal = 0;
    for (auto it = superPodMap.begin(); it != superPodMap.end(); ++it) {
        serverNumTotal += it->second.size();
    }

    // 校验super_pod_list和原有server_list的server数量一致
    CHK_PRT_RET(
        serverNumTotal != clusterInfo.serverNum,
        HCCL_ERROR(
            "[Get][SuperPodList]serverNum[%u] in super_pod_list and serverNum[%u] in server_list "
            "are inconsistent",
            serverNumTotal, clusterInfo.serverNum),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}
