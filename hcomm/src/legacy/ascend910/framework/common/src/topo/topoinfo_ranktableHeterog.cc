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
// ltm指定config路径
#include "common/src/config.h"
#include "workflow_pub.h"

using namespace std;
using namespace hccl;

constexpr u32 MAX_PORT_ID = 65535;
TopoinfoRanktableHeterog::TopoinfoRanktableHeterog(
    const std::string& rankTableM, const std::string& identify, DevType deviceType)
    : TopoInfoRanktableParser(rankTableM, identify),
      deviceType_(deviceType)
{}

TopoinfoRanktableHeterog::~TopoinfoRanktableHeterog() {}

HcclResult TopoinfoRanktableHeterog::Init()
{
    // 根据rankTable类型标记执行读文件或读字符串,将内容保存在json对象fileContent_中
    CHK_RET(LoadRankTableString(rankTableFile_));
    // 解析rankTable
    CHK_RET(ParserClusterInfo(params_, rankTable_));
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::GetSelfClusterInfo(HcclCommParams& params)
{
    // 获取params
    params.rank = params_.rank;
    params.userRank = params_.rank;
    params.totalRanks = params_.totalRanks;
    params.logicDevId = params_.logicDevId;
    params.serverId = params_.serverId;
    params.deviceType = params_.deviceType;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::GetClusterInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable)
{
    CHK_RET(GetClusterInfo(rankTable));
    CHK_RET(GetSelfClusterInfo(params));
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::GetClusterInfo(RankTable_t& clusterInfo)
{
    // 获取rankInfo
    clusterInfo.deviceNum = rankTable_.deviceNum;
    clusterInfo.serverNum = rankTable_.serverNum;
    clusterInfo.nicDeploy = rankTable_.nicDeploy;
    clusterInfo.rankNum = rankTable_.rankNum;
    clusterInfo.rankList = rankTable_.rankList;
    clusterInfo.serverList = rankTable_.serverList;
    clusterInfo.collectiveId = rankTable_.collectiveId;
    clusterInfo.version = rankTable_.version;
    clusterInfo.mode = rankTable_.mode;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::ParserClusterInfo(hccl::HcclCommParams& params, hccl::RankTable_t& rankTable)
{
    // 获取ranktable info信息
    CHK_RET(GetRanktableInfo(rankTable));
    if (!IsTaskNumCalMode()) {
        CHK_RET(CheckNicDeployConsistence(rankTable));
        rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
        std::sort(
            rankTable.rankList.begin(), rankTable.rankList.end(),
            [&](const RankInfo_t& a, const RankInfo_t& b) -> bool {
                return a.rankId < b.rankId;
            });

        unordered_map<std::string, u32> minRankPerServerMap;
        for (auto& iter : rankTable.rankList) {
            if (minRankPerServerMap.find(iter.serverId) == minRankPerServerMap.end()) {
                minRankPerServerMap[iter.serverId] = iter.rankId;
            } else if (minRankPerServerMap[iter.serverId] > iter.rankId) {
                minRankPerServerMap[iter.serverId] = iter.rankId;
            }
        }
        map<u32, std::string> minRankInServerMap;
        for (auto& iter : minRankPerServerMap) {
            minRankInServerMap.insert(std::make_pair(iter.second, iter.first));
        }

        u32 serverIndex = 0;
        for (auto& iter : minRankInServerMap) {
            for (u32 rankIndex = 0; rankIndex < rankTable.rankList.size(); rankIndex++) {
                if (rankTable.rankList[rankIndex].serverId == iter.second) {
                    rankTable.rankList[rankIndex].serverIdx = serverIndex;
                    ServerInfo_t serverinfo;
                    NetworkInfo_t networkInfo;
                    serverinfo.serverId = iter.second;
                    networkInfo.ipAddr = rankTable.rankList[rankIndex].hostIp;
                    serverinfo.networkInfo.push_back(networkInfo);
                    std::vector<ServerInfo_t>::iterator found
                        = find(rankTable.serverList.begin(), rankTable.serverList.end(), serverinfo);
                    if (found == rankTable.serverList.end()) {
                        rankTable.serverList.push_back(serverinfo);
                    }
                }
            }
            serverIndex++;
        }

        u32 rankId = INVALID_VALUE_RANKID;
        std::string errormessage = "";
        if (SalStrToULong(identify_, HCCL_BASE_DECIMAL, rankId) != HCCL_SUCCESS) {
            errormessage = "Value [" + identify_
                           + "] for rankTable variable [rank_id] is invalid, expected value is a valid integer.";
            HCCL_ERROR(
                "[%s][%s]errNo[0x%016llx] [%s]", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                HCOM_ERROR_CODE(HCCL_E_PARA), errormessage.c_str());
            return HCCL_E_PARA;
        }
        // 校验rank id合法性
        if (rankId >= rankTable.rankList.size()) {
            errormessage = "Value [" + std::to_string(rankId)
                           + "] for rankTable variable [rank_id] is invalid, "
                             "expected value is a valid rank ID in [0, "
                           + std::to_string(rankTable.rankList.size() - 1) + "].";
            HCCL_ERROR(
                "[%s][%s] [%s]", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                errormessage.c_str());
            return HCCL_E_PARA;
        }
        errormessage = "The current node's rankId [" + std::to_string(rankId)
                       + "] does not match "
                         "the rankId ["
                       + std::to_string(rankTable.rankList[rankId].rankId) + "] in the rankTable.";
        CHK_PRT_RET(
            rankId != rankTable.rankList[rankId].rankId,
            HCCL_ERROR(
                "[%s][%s] [%s]", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                errormessage.c_str()),
            HCCL_E_UNAVAIL);

        // params内容填入
        params.rank = rankId;
        params.logicDevId = rankTable.rankList[rankId].deviceInfo.devicePhyId;
        params.serverId = rankTable.rankList[rankId].serverId;
        params.totalRanks = rankTable.rankNum;
        params.deviceType = deviceType_;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::GetRanktableInfo(RankTable_t& clusterInfo)
{
    // 清空list
    clusterInfo.serverList.clear();
    clusterInfo.rankList.clear();

    // 获取信息
    std::string collective_id;
    nlohmann::json node_list;
    std::string version;
    std::string mode;

    std::string heterogSubVersion;
    // 解析91093-ES专属标志
    if (fileContent_.find("heterog_sub_version") != fileContent_.end()) {
        CHK_RET(GetJsonProperty(fileContent_, "heterog_sub_version", heterogSubVersion, false));
        CHK_RET(CheckHeterogSubVersion(heterogSubVersion));
        HCCL_DEBUG("[Get][RanktableInfo] heterog_sub_version exists!");
        is91093_ = true;
    }

    CHK_RET(GetJsonProperty(fileContent_, "collective_id", collective_id, false));
    if (collective_id.length() > COLLECTIVEID_MAX_LEN) {
        HCCL_ERROR(
            "[Get][RanktableInfo]errNo[0x%016llx] collectiveId length is greater than %d bytes.",
            HCOM_ERROR_CODE(HCCL_E_PARA), collective_id.length());
        return HCCL_E_PARA;
    }
    CHK_RET(GetJsonProperty(fileContent_, "node_list", node_list, false));
    CHK_RET(GetJsonProperty(fileContent_, "version", version, false));

    if (fileContent_.find("mode") != fileContent_.end()) {
        CHK_RET(GetJsonProperty(fileContent_, "mode", mode, false));
        CHK_RET(CheckMode(mode));
    }

    HCCL_DEBUG("[rankTableJson] -> collectiveId: [%s], nodeListSize: [%zu]", collective_id.c_str(), node_list.size());

    // 保存serverNum
    clusterInfo.serverNum = node_list.size();
    if (clusterInfo.serverNum == 0) {
        HCCL_ERROR("[Get][RanktableInfo]errNo[0x%016llx] node num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    // 获得single node信息
    for (u32 index = 0; index < clusterInfo.serverNum; index++) {
        CHK_RET(GetSingleNode(node_list, index, clusterInfo));
    }

    // 保存deviceNum和rankNum
    CHK_RET(GetDevNum(clusterInfo.rankList, clusterInfo.deviceNum));
    clusterInfo.rankNum = clusterInfo.rankList.size();
    clusterInfo.collectiveId = collective_id;
    clusterInfo.version = version;
    clusterInfo.mode = mode;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::CheckNicDeployConsistence(RankTable_t& clusterInfo) const
{
    // 检查rankList的大小
    CHK_PRT_RET(
        clusterInfo.rankList.size() == 0, HCCL_DEBUG("rank list size is 0, skip nic deeply check."), HCCL_SUCCESS);
    NICDeployment tmpNicDeply = clusterInfo.rankList.begin()->hostIp.IsInvalid() ?
                                    NICDeployment::NIC_DEPLOYMENT_DEVICE :
                                    NICDeployment::NIC_DEPLOYMENT_HOST;
    for (auto& it : clusterInfo.rankList) {
        CHK_PRT_RET(
            (tmpNicDeply == NICDeployment::NIC_DEPLOYMENT_DEVICE && !it.hostIp.IsInvalid())
                || (tmpNicDeply == NICDeployment::NIC_DEPLOYMENT_HOST && it.hostIp.IsInvalid()),
            HCCL_ERROR(
                "[Get][RanktableInfo] errNo[0x%016llx] "
                "hostIp config between ranks is different.",
                HCOM_ERROR_CODE(HCCL_E_PARA)),
            HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::CheckMode(std::string& mode) const
{
    if (mode == "tcp" || mode == "rdma") {
        return HCCL_SUCCESS;
    }
    HCCL_ERROR(
        "[Get][RanktableInfo]errNo[0x%016llx] mode[%s] is not supported, support for [tcp] and [rdma]",
        HCOM_ERROR_CODE(HCCL_E_PARA), mode.c_str());
    return HCCL_E_PARA;
}

HcclResult TopoinfoRanktableHeterog::CheckHeterogSubVersion(std::string& subVersion) const
{
    if (subVersion == "1.2") {
        return HCCL_SUCCESS;
    }

    HCCL_ERROR(
        "[Get][RanktableInfo]errNo[0x%016llx] subVersion[%s] is not supported, support for 1.2",
        HCOM_ERROR_CODE(HCCL_E_PARA), subVersion.c_str());
    return HCCL_E_PARA;
}

HcclResult TopoinfoRanktableHeterog::GetHostPort(const u32& localRank, u32& hostPort)
{
    u32 basePort = (GetExternalInputHcclIfBasePort() == HCCL_INVALID_PORT) ? HOST_PARA_BASE_PORT :
                                                                             GetExternalInputHcclIfBasePort();
    hostPort = basePort + localRank;
    CHK_PRT_RET(hostPort > MAX_PORT_ID, HCCL_ERROR("[Get][HostPort]invalid port id[%u]", hostPort), HCCL_E_INTERNAL);
    if (hostPortMap_.find(hostPort) == hostPortMap_.end()) {
        return HCCL_SUCCESS;
    }
    while (hostPortMap_[hostPort] == HOST_PORT_USED) {
        hostPort++;
        CHK_PRT_RET(
            hostPort > MAX_PORT_ID, HCCL_ERROR("[Get][HostPort]invalid port id[%u]", hostPort), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::GetRanks(
    const nlohmann::json& NodeListObj, u32 objIndex, RankTable_t& clusterInfo, std::string& serverId, u32& serverIdx,
    HcclIpAddress& nodeIp)
{
    HCCL_DEBUG("Get Node[%u]: nodeAddr:[%s], nodeIdx:[%u]", objIndex, serverId.c_str(), serverIdx);
    // 获取信息
    nlohmann::json Ranks;
    CHK_RET(GetJsonArrayMemberProperty(NodeListObj, objIndex, "ranks", Ranks, false));

    HCCL_DEBUG("[%s.json] -> rank_list: size:%zu", fileName_.c_str(), Ranks.size());
    CHK_PRT_RET(Ranks.size() == 0, HCCL_ERROR("[Get][Ranks]Ranks size is zero"), HCCL_E_PARA);

    // 91093的GetSingleRank单独操作
    if (is91093_) {
        for (u32 index = 0; index < Ranks.size(); index++) {
            CHK_RET(GetSingleRank91093(Ranks, index, clusterInfo, serverId, serverIdx, nodeIp));
        }
    } else {
        // 获取单rank信息
        for (u32 index = 0; index < Ranks.size(); index++) {
            CHK_RET(GetSingleRank(Ranks, index, clusterInfo, serverId, serverIdx, nodeIp));
        }
    }

    // 重新分配host port
    for (auto& rankInfo : clusterInfo.rankList) {
        if (rankInfo.hostPort == HCCL_INVALID_PORT) {
            CHK_RET(GetHostPort(rankInfo.localRank, rankInfo.hostPort));
        }
        HCCL_DEBUG("rank id: %u localRank:%u host port: %u", rankInfo.rankId, rankInfo.localRank, rankInfo.hostPort);
    }
    hostPortMap_.clear();
    return HCCL_SUCCESS;
}

HcclResult
TopoinfoRanktableHeterog::GetSingleNode(const nlohmann::json& NodeListObj, u32 objIndex, RankTable_t& clusterInfo)
{
    // 获取信息
    HcclResult ret;
    std::string nodeAddr;
    std::string serverId;
    CHK_RET(GetJsonArrayMemberProperty(NodeListObj, objIndex, "node_addr", nodeAddr, false));
    // 将serverId添加到资源池,内部会进行IP地址校验，如果资源池中有serverId，则报错
    serverId = nodeAddr;
    CHK_RET(CheckUniqueAndInsertPool(
        JsonUniqueInfoType::UNIQUE_INFO_TYPE_SERVER_ID, serverId, JsonCheckOpType::CHECK_OP_TYPE_INSERT));
    // 设置serverIdx
    u32 serverIdx;
    GenerateServerIdx(serverId, serverIdx);

    // 计算node ip
    HcclIpAddress nodeIp;
    CHK_RET(ConvertIpAddress(nodeAddr, nodeIp));

    // 处理ranklist
    ret = GetRanks(NodeListObj, objIndex, clusterInfo, serverId, serverIdx, nodeIp);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Get][GetRanks]get rank list error:nodeAddr:[%s]", serverId.c_str()), ret);
    return HCCL_SUCCESS;
}

// 91093暂定所有字段都是必选字段。除了ranks里面的 bind_device_id
HcclResult TopoinfoRanktableHeterog::GetSingleRank91093(
    const nlohmann::json& ranksObj, u32 objIndex, RankTable_t& clusterInfo, std::string& serverId, u32& serverIdx,
    HcclIpAddress& nodeIp)
{
    HCCL_INFO("Entry-GetSingleRank91093");
    std::string rankId;
    CHK_RET(GetJsonArrayMemberProperty(ranksObj, objIndex, "rank_id", rankId, false));

    s32 devicePhyId = 0;
    std::string devPhyIdStr;
    CHK_RET(GetJsonArrayMemberProperty(ranksObj, objIndex, "device_id", devPhyIdStr, false));
    CHK_RET(SalStrToInt(devPhyIdStr, HCCL_BASE_DECIMAL, devicePhyId));

    s32 port = HCCL_INVALID_PORT;
    std::string portStr;
    CHK_RET(GetJsonArrayMemberProperty(ranksObj, objIndex, "port", portStr, false));
    CHK_RET(SalStrToInt(portStr, HCCL_BASE_DECIMAL, port));

    // 如果device没有网卡，则缺省，通信使用node_addr
    HcclIpAddress rankIp;
    std::string rankIpStr;
    CHK_RET(GetJsonArrayMemberProperty(ranksObj, objIndex, "rank_ip", rankIpStr, false));
    if (rankIpStr.empty()) {
        rankIp = nodeIp;
    } else {
        CHK_RET(ConvertIpAddress(rankIpStr, rankIp));
    }

    s32 sdid = 0;
    std::string sdidStr;
    CHK_RET(GetJsonArrayMemberProperty(ranksObj, objIndex, "sdid", sdidStr, false));
    CHK_RET(SalStrToInt(sdidStr, HCCL_BASE_DECIMAL, sdid));

    // 如果deviceId == -1, 则去获取可选字段
    s32 bindDeviceId = -1;
    std::string bindDeviceIdStr;
    if (devicePhyId == HOST_DEVICE_ID) {
        CHK_RET(GetJsonArrayMemberProperty(ranksObj, objIndex, "bind_device_id", bindDeviceIdStr, false));
        CHK_RET(SalStrToInt(bindDeviceIdStr, HCCL_BASE_DECIMAL, bindDeviceId));
    }

    // rankList
    RankInfo_t rankInfo;
    rankInfo.localRank = objIndex;
    rankInfo.serverId = serverId;
    rankInfo.serverIdx = serverIdx;
    rankInfo.hostIp = nodeIp;
    rankInfo.superDeviceId = sdid;
    rankInfo.bindDeviceId = bindDeviceId;
    rankInfo.deviceInfo.devicePhyId = devicePhyId;
    rankInfo.deviceInfo.port = port;
    rankInfo.deviceInfo.deviceIp.push_back(rankIp);

    std::string errormessage = "";
    if (SalStrToULong(rankId, HCCL_BASE_DECIMAL, rankInfo.rankId) != HCCL_SUCCESS) {
        errormessage = "Value [" + rankId
                       + "] for rankTable variable [rank_id] is invalid, expected value is a non-negative integer.";
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] [%s]", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
            HCOM_ERROR_CODE(HCCL_E_PARA), errormessage.c_str());
        return HCCL_E_PARA;
    }

    rankInfo.podName = ""; // podname在新场景下置空
    // ranktable中port无效，设置为环境变量HCCL_IF_BASE_PORT+该rank的local_rank_id,否则按照配置的port使用
    if (port == 0) {
        rankInfo.hostPort = HCCL_INVALID_PORT;
    } else {
        rankInfo.hostPort = port;
        hostPortMap_[rankInfo.hostPort] = HOST_PORT_USED;
    }
    clusterInfo.rankList.push_back(rankInfo);
    HCCL_DEBUG("[%s.json]->rankId[%u], nodeAddr[%s]", fileName_.c_str(), rankInfo.rankId, rankInfo.serverId.c_str());

    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktableHeterog::GetSingleRank(
    const nlohmann::json& ranksObj, u32 objIndex, RankTable_t& clusterInfo, std::string& serverId, u32& serverIdx,
    HcclIpAddress& nodeIp)
{
    // 获取rank_id
    std::string rankId;
    CHK_RET(GetJsonArrayMemberProperty(ranksObj, objIndex, "rank_id", rankId, false));

    s32 devicePhyId = 0;
    std::string devPhyIdStr;
    if (GetJsonArrayMemberProperty(ranksObj, objIndex, "device_id", devPhyIdStr, true) == HCCL_E_NOT_FOUND) {
        devicePhyId = HOST_DEVICE_ID;
    } else {
        HCCL_DEBUG("[Get][GetRanks]device_id:[%s]", devPhyIdStr.c_str());
        CHK_RET(SalStrToInt(devPhyIdStr, HCCL_BASE_DECIMAL, devicePhyId));
    }

    s32 port = HCCL_INVALID_PORT;
    std::string portStr;
    if (GetJsonArrayMemberProperty(ranksObj, objIndex, "port", portStr, true) == HCCL_E_NOT_FOUND) {
        port = HCCL_INVALID_PORT;
    } else {
        CHK_RET(SalStrToInt(portStr, HCCL_BASE_DECIMAL, port));
    }

    // 如果device没有网卡，则缺省，通信使用node_addr
    HcclIpAddress rankIp;
    std::string rankIpStr;
    if (GetJsonArrayMemberProperty(ranksObj, objIndex, "rank_ip", rankIpStr, true) == HCCL_E_NOT_FOUND) {
        rankIp = nodeIp;
    } else {
        if (rankIpStr.empty()) {
            rankIp = nodeIp;
        } else {
            CHK_RET(ConvertIpAddress(rankIpStr, rankIp));
        }
    }

    // rankList
    RankInfo_t rankInfo;
    rankInfo.localRank = objIndex;
    rankInfo.serverId = serverId;
    rankInfo.serverIdx = serverIdx;
    rankInfo.hostIp = nodeIp;
    rankInfo.deviceInfo.devicePhyId = devicePhyId;
    rankInfo.deviceInfo.port = port;
    rankInfo.deviceInfo.deviceIp.push_back(rankIp);
    std::string errormessage = "";
    if (SalStrToULong(rankId, HCCL_BASE_DECIMAL, rankInfo.rankId) != HCCL_SUCCESS) {
        errormessage = "Value [" + rankId
                       + "] for rankTable variable [rank_id] is invalid, expected value is a non-negative integer.";
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] [%s]", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
            HCOM_ERROR_CODE(HCCL_E_PARA), errormessage.c_str());
        return HCCL_E_PARA;
    }
    rankInfo.podName = ""; // podname在新场景下置空
    // ranktable中port无效，设置为环境变量HCCL_IF_BASE_PORT+该rank的local_rank_id,否则按照配置的port使用
    if (port == 0) {
        rankInfo.hostPort = HCCL_INVALID_PORT;
    } else {
        rankInfo.hostPort = port;
        hostPortMap_[rankInfo.hostPort] = HOST_PORT_USED;
    }
    clusterInfo.rankList.push_back(rankInfo);
    HCCL_DEBUG("[%s.json]->rankId[%u], nodeAddr[%s]", fileName_.c_str(), rankInfo.rankId, rankInfo.serverId.c_str());

    return HCCL_SUCCESS;
}
