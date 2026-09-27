/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_exchange_base.h"
#include <thread>
#include <iostream>
#include <fstream>
#include "externalinput_pub.h"
#include "mem_host_pub.h"
#include "json_utils.h"

namespace hccl {

std::atomic<BroadcastStage> g_broadcastStage(BroadcastStage::Idle);
std::mutex g_broadcast_stage_mutex;
std::condition_variable g_broadcast_stage_cv;

TopoInfoExchangeBase::TopoInfoExchangeBase() : currentStep_(0) {}

TopoInfoExchangeBase::~TopoInfoExchangeBase() {}

std::string TopoInfoExchangeBase::BuildMeshTag(u32 smallerRoot, u32 largerRoot, u32 largerRootMeshPort)
{
    return TOPO_DETECT_TAG + "_mesh_" + std::to_string(smallerRoot) + "_" + std::to_string(largerRoot) + "_"
           + std::to_string(largerRootMeshPort);
}

HcclResult TopoInfoExchangeBase::DisconnectSocket(std::shared_ptr<HcclSocket> socket) const
{
    if (socket) {
        socket->Close();
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::SendClusterInfoMsg(
    std::shared_ptr<HcclSocket> socket, [[maybe_unused]] const RankTable_t& clusterInfo, const std::string buffer,
    const u32 msgLen)
{
    HcclResult ret = socket->Send(&msgLen, sizeof(msgLen));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send][ClusterInfoMsg]errNo[0x%016llx] ra send msg length failed! "
            "msgLen[%u], ret[%u]",
            HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), msgLen, ret),
        ret);

    ret = socket->Send(buffer.c_str(), msgLen);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Send][ClusterInfoMsg]errNo[0x%016llx] ra send failed! size[%u], ret[%u]",
            HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), msgLen, ret),
        ret);

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::SendClusterInfo(std::shared_ptr<HcclSocket> socket, const RankTable_t& clusterInfo)
{
    nlohmann::json basicJson;
    CHK_RET(Struct2Json(clusterInfo, basicJson));
    basicJson[PROP_STEP] = currentStep_; // add step to verify.
    std::string buffer = basicJson.dump();
    u32 msgLen = buffer.length();
    CHK_RET(SendClusterInfoMsg(socket, clusterInfo, buffer, msgLen));
    currentStep_++;
    return HCCL_SUCCESS;
}

void TopoInfoExchangeBase::PrintRecvFailReasons(std::shared_ptr<HcclSocket> socket, HcclResult ret)
{
    HCCL_ERROR(
        "[%s][%s]receive msg length from fdhandle failed, ret[%d]", LOG_KEYWORDS_INIT_GROUP.c_str(),
        LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), ret);
    HCCL_ERROR(
        "Current rank get socket with server[%s] success, but wait for recv rankTable from server failed, maybe due to "
        "following reasons:",
        socket->GetRemoteIp().GetReadableIP());
    HCCL_ERROR(
        "1. client wait for recv timeout, please check [ERROR] info in server[%s], whether all ranks were executed to "
        "create the communication",
        socket->GetRemoteIp().GetReadableIP());
    HCCL_ERROR("2. in large-scale cluster scenarios, occasional connection failures may occur due to the maximum "
               "connection limit in the system configuration. ");
    HCCL_ERROR("   these issues can be resolved by modifying the system configuration in all node: `sysctl -w "
               "net.core.somaxconn=65535` and `sysctl -w net.ipv4.tcp_max_syn_backlog=65535`");
}

HcclResult TopoInfoExchangeBase::RecvClusterInfoMsg(std::shared_ptr<HcclSocket> socket, RankTable_t& clusterInfo)
{
    const u32 recvBufferLimit = 100 * 1024 * 1024; // 100 * 1024 * 1024 = 100MB
    u32 msgLen = 0;
    std::string errormessage = "";
    HcclResult ret = socket->Recv(reinterpret_cast<char*>(&msgLen), sizeof(msgLen));
    if (ret == HCCL_E_TIMEOUT) {
        errormessage = "Receiving message from the root node timed out. Check whether node "
                       + std::string(socket->GetRemoteIp().GetReadableIP()) + " reports an error";
        RPT_INPUT_ERR(
            true, "EI0015", std::vector<std::string>({"error_reason"}), std::vector<std::string>({errormessage}));
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS, PrintRecvFailReasons(socket, ret), HCCL_E_INTERNAL);
    CHK_PRT_RET(
        ((msgLen == 0) || (msgLen > recvBufferLimit)),
        HCCL_ERROR(
            "[%s][%s]receive msg "
            "length[%u] from fdhandle failed, msg length is beyond [1 ~ %u].",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), msgLen, recvBufferLimit),
        HCCL_E_INTERNAL);

    u32 recvBufferLen = msgLen + 1;
    HostMem recvMsg = HostMem::alloc(recvBufferLen);
    CHK_PTR_NULL(recvMsg.ptr());
    char* recvMsgBuf = static_cast<char*>(recvMsg.ptr());

    s32 sRet = memset_s(recvMsgBuf, recvBufferLen, 0, recvBufferLen);
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[%s][%s]sockBuff memset failed", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str()),
        HCCL_E_MEMORY);
    ret = socket->Recv(recvMsgBuf, msgLen);
    if (ret == HCCL_E_TIMEOUT) {
        RPT_INPUT_ERR(
            true, "EI0015", std::vector<std::string>({"error_reason"}), std::vector<std::string>({errormessage}));
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]receive from fdhandle failed ,ret[%d]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), ret),
        HCCL_E_INTERNAL);
    nlohmann::json jClusterJson;
    CHK_RET(parseJsonBuff(recvMsgBuf, recvBufferLen, jClusterJson));

    // Verify json basic info
    u32 step;
    CHK_RET(JsonUtils::GetJsonProperty(jClusterJson, PROP_STEP, step));

    CHK_PRT_RET(
        step != currentStep_,
        HCCL_ERROR(
            "[Recv][ClusterInfoMsg]RecvClusterInfo step failed "
            "step[%u] vs currentStep_[%u]",
            step, currentStep_),
        HCCL_E_INTERNAL);

    s32 logicDevId = 0;
    u32 devPhyId = 0;
    CHK_RET(hrtGetDevice(&logicDevId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(logicDevId), devPhyId));
    HcclIpAddress localHostIp;
    CHK_RET(GetLocalHostIP(localHostIp, devPhyId));

    bool isRoot
        = (localHostIp == GetExternalInputMasterInfo().serverIp
           && logicDevId == static_cast<s32>(GetExternalInputMasterInfo().serverDeviceId));
    errormessage = "No rank in the communicator can connect to the root node within the timeout period. List of "
                   "unconnected ranks: "
                   + std::string(jClusterJson["fault_info"].dump().c_str());
    if (!isRoot && jClusterJson.find("fault_type") != jClusterJson.end()
        && jClusterJson.find("fault_info") != jClusterJson.end()) {
        RPT_INPUT_ERR(
            true, "EI0015", std::vector<std::string>({"error_reason"}), std::vector<std::string>({errormessage}));
        HCCL_ERROR(
            "[%s][%s] TopoDetect ERROR occur fault_type[%s], fault_info[%s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), jClusterJson["fault_type"].dump().c_str(),
            jClusterJson["fault_info"].dump().c_str());
    }

    ret = Json2Struct(jClusterJson, clusterInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Recv][ClusterInfoMsg]step[%u] json to struct failed!", currentStep_),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::RecvClusterInfo(std::shared_ptr<HcclSocket> socket, RankTable_t& clusterInfo)
{
    CHK_RET(RecvClusterInfoMsg(socket, clusterInfo));
    std::string errormessage = "";
    if (isByMasterInfo_) {
        u32 identify = 0;
        auto ret = socket->Recv(reinterpret_cast<char*>(&identify), sizeof(identify));
        if (ret == HCCL_E_TIMEOUT) {
            errormessage = "Receiving message from the root node timed out. Check whether node "
                           + std::string(socket->GetRemoteIp().GetReadableIP()) + " reports an error";
            RPT_INPUT_ERR(
                true, "EI0015", std::vector<std::string>({"error_reason"}), std::vector<std::string>({errormessage}));
        }
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s][%s] receive identify from fdhandle failed", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_DETECT.c_str()),
            HCCL_E_INTERNAL);
        identifierNum_ = identify;
    }
    currentStep_++;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::RecvClusterJson(std::shared_ptr<HcclSocket> socket, nlohmann::json& jClusterJson)
{
    const u32 recvBufferLimit = 10 * 1024 * 1024; // 10 * 1024 * 1024 = 10MB
    u32 msgLen = 0;
    std::string errormessage = "";
    HcclResult ret = socket->Recv(reinterpret_cast<char*>(&msgLen), sizeof(msgLen));
    if (ret == HCCL_E_TIMEOUT) {
        errormessage = "Receiving message from the root node timed out. Check whether node "
                       + std::string(socket->GetRemoteIp().GetReadableIP()) + " reports an error";
        RPT_INPUT_ERR(
            true, "EI0015", std::vector<std::string>({"error_reason"}), std::vector<std::string>({errormessage}));
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s] receive msg length from fdhandle failed, ret[%d]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), ret),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        ((msgLen == 0) || (msgLen > recvBufferLimit)),
        HCCL_ERROR(
            "[%s][%s]receive msg length "
            "from fdhandle failed, msg length is beyond [1 ~ %u].",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), recvBufferLimit),
        HCCL_E_INTERNAL);

    u32 recvBufferLen = msgLen + 1;
    HostMem recvMsg = HostMem::alloc(recvBufferLen);
    CHK_PTR_NULL(recvMsg.ptr());
    char* recvMsgBuf = static_cast<char*>(recvMsg.ptr());

    s32 sRet = memset_s(recvMsgBuf, recvBufferLen, 0, recvBufferLen);
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Recv][ClusterInfoMsg]sockBuff memset failed"), HCCL_E_MEMORY);
    ret = socket->Recv(recvMsgBuf, msgLen);
    if (ret == HCCL_E_TIMEOUT) {
        errormessage = "Receiving message from the root node timed out. Check whether node "
                       + std::string(socket->GetRemoteIp().GetReadableIP()) + " reports an error";
        RPT_INPUT_ERR(
            true, "EI0015", std::vector<std::string>({"error_reason"}), std::vector<std::string>({errormessage}));
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s] receive from fdhandle failed ,ret[%d]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), ret),
        HCCL_E_INTERNAL);
    CHK_RET(parseJsonBuff(recvMsgBuf, recvBufferLen, jClusterJson));

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::RecvGrpLeaderInfoMsg(std::shared_ptr<HcclSocket> socket, GroupLeader_t& LeaderInfo)
{
    nlohmann::json jClusterJson;
    CHK_RET(RecvClusterJson(socket, jClusterJson));

    // Verify json basic info
    u32 step;
    CHK_RET(JsonUtils::GetJsonProperty(jClusterJson, PROP_STEP, step));

    CHK_PRT_RET(
        step != currentStep_,
        HCCL_ERROR(
            "[Recv][ClusterInfoMsg]RecvClusterInfo step failed "
            "step[%u] vs currentStep_[%u]",
            step, currentStep_),
        HCCL_E_INTERNAL);

    HcclResult ret = Json2GrpLeader(jClusterJson, LeaderInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Recv][ClusterInfoMsg]step[%u] json to struct failed!", currentStep_),
        HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::BlockReceive(std::shared_ptr<HcclSocket> socket, char* buff, u32 size) const
{
    CHK_PTR_NULL(buff);
    CHK_RET(socket->Recv(buff, size));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::parseJsonBuff(const char buff[], u32 buffLen, nlohmann::json& buffJson) const
{
    u32 len = strnlen(buff, buffLen);
    CHK_PRT_RET(
        (len > buffLen || len == 0),
        HCCL_ERROR("[Parse][JsonBuff]buff len invalid, buff len[%u], msgLen[%u]", len, buffLen), HCCL_E_INTERNAL);

    CHK_RET(JsonUtils::ParseInformation(buffJson, buff));
    u32 step;
    CHK_RET(JsonUtils::GetJsonProperty(buffJson, PROP_STEP, step));
    if (step != currentStep_) {
        HCCL_ERROR(
            "[Parse][JsonBuff]errNo[0x%016llx] received step[%u] is invalid , expect step is %u",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), step, currentStep_);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::Json2GrpLeader(const nlohmann::json& jClusterJson, GroupLeader_t& GrpLeaderInfo) const
{
    GrpLeaderInfo.grpLeaderNum = jClusterJson[PROP_RANK_NUM];
    for (auto& leaderInfoJson : jClusterJson[PROP_GROUP_LEADER_LIST]) {
        HcclRankHandle rankHandle;
        std::string strTmp = leaderInfoJson[PROP_NETWORK_IPADDR];
        s32 sRet = memcpy_s(rankHandle.ip, IP_ADDRESS_BUFFER_LEN, strTmp.c_str(), strTmp.size());
        CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Json2GrpLeader]memcpy_s failed, errorno[%d]", sRet), HCCL_E_MEMORY);
        rankHandle.ip[strTmp.size()] = '\0';
        rankHandle.port = leaderInfoJson[PROP_NETWORK_NETWORKPORT];
        strTmp = leaderInfoJson[PROP_NETWORK_IDENTIFIER];
        sRet = memcpy_s(rankHandle.identifier, ROOTINFO_INDENTIFIER_MAX_LENGTH, strTmp.c_str(), strTmp.size());
        CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Json2GrpLeader]memcpy_s failed, errorno[%d]", sRet), HCCL_E_MEMORY);
        rankHandle.identifier[strTmp.size()] = '\0';
        rankHandle.nicDeploy = leaderInfoJson[PROP_DEPLOY_MODE];
        rankHandle.rankId = leaderInfoJson[PROP_RANK_ID];
        GrpLeaderInfo.GroupLeaderList.emplace_back(rankHandle);
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::Json2Struct(const nlohmann::json& jClusterJson, RankTable_t& clusterInfo) const
{
    CHK_RET(SetClusterDeploy(jClusterJson, clusterInfo)); // deploymode为枚举类变量 需单独处理判断
    CHK_RET(JsonUtils::GetJsonProperty(jClusterJson, PROP_DEV_NUM, clusterInfo.deviceNum));
    CHK_RET(JsonUtils::GetJsonProperty(jClusterJson, PROP_SRV_NUM, clusterInfo.serverNum));
    CHK_RET(JsonUtils::GetJsonProperty(jClusterJson, PROP_SUPER_POD_NUM, clusterInfo.superPodNum));
    CHK_RET(JsonUtils::GetJsonProperty(jClusterJson, PROP_RANK_NUM, clusterInfo.rankNum));
    for (auto& rankInfoJson : jClusterJson[PROP_RANK_LIST]) {
        RankInfo_t rankInfo;
        rankInfo.rankId = rankInfoJson[PROP_RANK_ID];
        rankInfo.serverId = rankInfoJson[PROP_SERVER_ID];
        rankInfo.tlsStatus = rankInfoJson[PROP_TLS_STATUS];
        CHK_RET(rankInfo.hostIp.SetReadableAddress(rankInfoJson[PROP_HOST_IP]));
        rankInfo.deviceInfo.devicePhyId = rankInfoJson[PROP_DEV_INFO][PROP_DEV_ID];
        rankInfo.deviceInfo.deviceType = rankInfoJson[PROP_DEV_INFO][PROP_DEV_TYPE];
        CHK_PRT_RET(
            rankInfoJson[PROP_DEV_INFO].find(PROP_DEV_NIC_PORT) == rankInfoJson[PROP_DEV_INFO].end()
                || rankInfoJson[PROP_DEV_INFO].find(PROP_DEV_VNIC_PORT) == rankInfoJson[PROP_DEV_INFO].end()
                || rankInfoJson[PROP_DEV_INFO].find(PROP_BACKUP_DEV_PORT) == rankInfoJson[PROP_DEV_INFO].end(),
            HCCL_ERROR("[Json2Struct] Fail to find port infos in rank info json. "
                       "Please make sure the CANN version is consistent within the communication."),
            HCCL_E_NOT_SUPPORT);
        rankInfo.deviceInfo.port = rankInfoJson[PROP_DEV_INFO][PROP_DEV_NIC_PORT];
        rankInfo.deviceInfo.vnicPort = rankInfoJson[PROP_DEV_INFO][PROP_DEV_VNIC_PORT];
        rankInfo.deviceInfo.backupPort = rankInfoJson[PROP_DEV_INFO][PROP_BACKUP_DEV_PORT];
        for (auto& devIp : rankInfoJson[PROP_DEV_INFO][PROP_DEV_IP]) {
            std::string ipStr = devIp;
            rankInfo.deviceInfo.deviceIp.emplace_back(ipStr);
        }
        if (rankInfoJson[PROP_DEV_INFO].find(PROP_BACKUP_DEV_IP) == rankInfoJson[PROP_DEV_INFO].end()) {
            HCCL_RUN_WARNING("[Json2Struct] Fail to find backup device ip in rank info json. "
                             "Backup device ip will not be parsed. If you want to use backup device ip, "
                             "Please make sure the CANN version is consistent within the communication.");
        } else {
            for (auto& backupDevIp : rankInfoJson[PROP_DEV_INFO][PROP_BACKUP_DEV_IP]) {
                std::string backupIpStr = backupDevIp;
                rankInfo.deviceInfo.backupDeviceIp.emplace_back(backupIpStr);
            }
        }

        rankInfo.superPodId = rankInfoJson[PROP_SUPER_POD_ID];
        rankInfo.superDeviceId = rankInfoJson[PROP_SUPER_DEVICE_ID];

        /* Optional: for second communication stage */
        if (rankInfoJson.find(PROP_TRANS_INFO) != rankInfoJson.end()) {
            for (auto& transInfoJson : rankInfoJson[PROP_TRANS_INFO]) {
                TransportInfo_t transportInfo;
                transportInfo.dstRankId = transInfoJson[PROP_DEST_RANK];
                transportInfo.transportType = transInfoJson[PROP_TRANS_TYPE];
                rankInfo.transportInfo.push_back(transportInfo);
            }
        }
        clusterInfo.rankList.push_back(rankInfo);
    }
    for (auto& serverInfoJson : jClusterJson[PROP_SERVER_LIST]) {
        ServerInfo_t serverInfo;
        serverInfo.serverId = serverInfoJson[PROP_SERVER_ID];
        for (auto& networkInfoJson : serverInfoJson[PROP_NETWORK_INFO_LIST]) {
            NetworkInfo_t networkInfo;
            networkInfo.ethName = networkInfoJson[PROP_NETWORK_ETHNAME];
            CHK_RET(networkInfo.ipAddr.SetReadableAddress(networkInfoJson[PROP_NETWORK_IPADDR]));
            networkInfo.networkPort = networkInfoJson[PROP_NETWORK_NETWORKPORT];
            CHK_RET(networkInfo.refIp.SetReadableAddress(networkInfoJson[PROP_NETWORK_REFIP]));
            networkInfo.planeID = networkInfoJson[PROP_NETWORK_PLANEID];
            serverInfo.networkInfo.push_back(networkInfo);
        }
        clusterInfo.serverList.push_back(serverInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::Struct2Json(const RankTable_t& clusterInfo, nlohmann::json& ClusterJson)
{
    nlohmann::json rankListJson;
    nlohmann::json serverListJson;

    TransformRankListToJson(clusterInfo, rankListJson);
    for (auto& serverInfo : clusterInfo.serverList) {
        nlohmann::json serverJson;
        serverJson[PROP_SERVER_ID] = serverInfo.serverId;
        nlohmann::json networkInfoListJson;
        for (auto& networkInfo : serverInfo.networkInfo) {
            nlohmann::json networkInfoJson;
            networkInfoJson[PROP_NETWORK_ETHNAME] = networkInfo.ethName;
            networkInfoJson[PROP_NETWORK_IPADDR] = std::string(networkInfo.ipAddr.GetReadableIP());
            networkInfoJson[PROP_NETWORK_NETWORKPORT] = networkInfo.networkPort;
            networkInfoJson[PROP_NETWORK_REFIP] = std::string(networkInfo.refIp.GetReadableIP());
            networkInfoJson[PROP_NETWORK_PLANEID] = networkInfo.planeID;
            networkInfoListJson.push_back(networkInfoJson);
        }
        serverJson[PROP_NETWORK_INFO_LIST] = networkInfoListJson;
        serverListJson.push_back(serverJson);
    }

    ClusterJson[PROP_RANK_NUM] = clusterInfo.rankNum;
    ClusterJson[PROP_DEV_NUM] = clusterInfo.deviceNum;
    ClusterJson[PROP_SRV_NUM] = clusterInfo.serverNum;
    ClusterJson[PROP_SUPER_POD_NUM] = clusterInfo.superPodNum;
    ClusterJson[PROP_DEPLOY_MODE] = clusterInfo.nicDeploy;
    ClusterJson[PROP_RANK_LIST] = rankListJson;
    ClusterJson[PROP_SERVER_LIST] = serverListJson;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::GrpLeader2Json(const GroupLeader_t& GrpLeaderInfo, nlohmann::json& GroupLeaderJson)
{
    nlohmann::json leaderListJson;

    for (auto& leaderInfo : GrpLeaderInfo.GroupLeaderList) {
        nlohmann::json leaderJson;

        leaderJson[PROP_NETWORK_IPADDR] = std::string(leaderInfo.ip);
        leaderJson[PROP_NETWORK_NETWORKPORT] = leaderInfo.port;
        leaderJson[PROP_NETWORK_IDENTIFIER] = std::string(leaderInfo.identifier);
        leaderJson[PROP_DEPLOY_MODE] = leaderInfo.nicDeploy;
        leaderJson[PROP_RANK_ID] = leaderInfo.rankId;

        leaderListJson.push_back(leaderJson);
    }

    GroupLeaderJson[PROP_RANK_NUM] = GrpLeaderInfo.grpLeaderNum;
    GroupLeaderJson[PROP_GROUP_LEADER_LIST] = leaderListJson;

    return HCCL_SUCCESS;
}

HcclResult
TopoInfoExchangeBase::TransformRankListToJson(const RankTable_t& clusterInfo, nlohmann::json& rankListJson) const
{
    for (auto& rankInfo : clusterInfo.rankList) {
        nlohmann::json deviceIp;
        for (auto& devIp : rankInfo.deviceInfo.deviceIp) {
            deviceIp.push_back(std::string(devIp.GetReadableIP()));
        }
        nlohmann::json backupDeviceIp;
        for (auto& backupDevIp : rankInfo.deviceInfo.backupDeviceIp) {
            backupDeviceIp.push_back(std::string(backupDevIp.GetReadableIP()));
        }
        nlohmann::json devInfoJson;
        devInfoJson[PROP_DEV_ID] = rankInfo.deviceInfo.devicePhyId;
        devInfoJson[PROP_DEV_TYPE] = rankInfo.deviceInfo.deviceType;
        devInfoJson[PROP_DEV_NIC_PORT] = rankInfo.deviceInfo.port;
        devInfoJson[PROP_DEV_VNIC_PORT] = rankInfo.deviceInfo.vnicPort;
        devInfoJson[PROP_BACKUP_DEV_PORT] = rankInfo.deviceInfo.backupPort;
        devInfoJson[PROP_DEV_IP] = deviceIp;
        devInfoJson[PROP_BACKUP_DEV_IP] = backupDeviceIp;
        nlohmann::json rankJson;
        rankJson[PROP_RANK_ID] = rankInfo.rankId;
        rankJson[PROP_SERVER_ID] = rankInfo.serverId;
        rankJson[PROP_HOST_IP] = std::string(rankInfo.hostIp.GetReadableIP());
        rankJson[PROP_DEV_INFO] = devInfoJson;

        rankJson[PROP_SUPER_POD_ID] = rankInfo.superPodId;
        rankJson[PROP_SUPER_DEVICE_ID] = rankInfo.superDeviceId;
        rankJson[PROP_TLS_STATUS] = rankInfo.tlsStatus;

        /* Optional: for second communication stage */
        nlohmann::json transInfosJson;
        for (auto& transInfo : rankInfo.transportInfo) {
            nlohmann::json transInfoJson;
            transInfoJson[PROP_TRANS_TYPE] = transInfo.transportType;
            transInfoJson[PROP_DEST_RANK] = transInfo.dstRankId;
            transInfosJson.push_back(transInfoJson);
        }
        if (!transInfosJson.empty()) {
            rankJson[PROP_TRANS_INFO] = transInfosJson;
        }
        rankListJson.push_back(rankJson);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoExchangeBase::SetClusterDeploy(const nlohmann::json& jClusterJson, RankTable_t& clusterInfo) const
{
    if (!jClusterJson.contains(PROP_DEPLOY_MODE)) {
        HCCL_ERROR("[SetClusterDeploy]PROP_DEPLOY_MODE is invalid");
        return HCCL_E_INTERNAL;
    }
    clusterInfo.nicDeploy = jClusterJson[PROP_DEPLOY_MODE];
    return HCCL_SUCCESS;
}

} // namespace hccl
