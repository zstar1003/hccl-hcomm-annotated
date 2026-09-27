/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_info_detect_service.h"

#include <stdio.h>
#include "rank_info_dispatcher.h"
#include "env_config/env_config_v2.h"
#include "host_buffer.h"
#include "root_handle_v2.h"
#include "hccp_peer_manager.h"
#include "orion_adapter_rts.h"
#include "preempt_port_manager_v2.h"
#include "host_socket_handle_manager.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"

namespace Hccl {

const u32 DISPLAY_RANKNUM_PERLINE = 8;
const u32 SOCKET_ACCEPT_TIMEOUT = 60; // Server调用Accept等待的最大超时时间 60s
const u32 SOCKET_PRINT_COUNT = 3;     // 未建链打印的数量
const u32 MAX_AGENT_BUF_SIZE = 256;

void RankInfoDetectService::Setup()
{
    // 1. 连接所有rank
    GetConnections();

    // 2. 接收所有rank发来的localRankTable并整合为全局RankTable
    GetRankTable();

    // 3. 将完整RankTable广播给所有rank
    BroadcastRankTable();
}

void RankInfoDetectService::GetConnections()
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    // 超时参数
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    bool isFirstAcceptTimeOut = false;

    // 期望等待连接的rank数量
    u32 expectedSocketNum = 1;

    // 首个connect获取到的rankSize
    u32 previousRankNum = 0;

    // 获取server端socket信息
    u32 hostPort = serverSocket_->GetListenPort();
    auto hccpHostSocketHandle = HostSocketHandleManager::GetInstance().Get(devPhyId_, hostIp_);
    CHK_PRT_THROW(
        hccpHostSocketHandle == nullptr,
        HCCL_ERROR("[RankInfoDetectService::%s] Get hccpHostSocketHandle fail.", __func__), InternalException,
        "get socket handle error");
    std::string connSocketTag = RANK_INFO_DETECT_TAG + "_" + identifier_ + "_" + std::to_string(hostPort);
    SocketStatus status = SocketStatus::INVALID;

    // 连接rankSize个client
    while (expectedSocketNum > 0) {
        auto topoExUsedTime = std::chrono::steady_clock::now() - startTime;
        if (topoExUsedTime >= timeout) {
            RPT_INPUT_ERR(
                true, "EI0015", std::vector<std::string>({"error_reason"}),
                std::vector<std::string>({StringFormat(
                    "Receiving message from the root node timed out. "
                    "Timeout was set to %lld seconds. expected %u nodes, received %u nodes. "
                    "Check whether worker nodes are reachable and report errors.",
                    static_cast<long long>(timeout.count()), expectedSocketNum + previousRankNum, previousRankNum)}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] topo exchange server get socket timeout[%lld s], "
                "expected[%u] nodes, received[%u] nodes",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_TIMEOUT), static_cast<long long>(timeout.count()),
                expectedSocketNum + previousRankNum, previousRankNum);
            break;
        }

        // duration_cast<seconds> 会进行向下取整，不足 1s
        // 时提前跳出，确保建链超时场景，server端在client端退出前发送临终遗言
        auto topoExResTime = timeout - topoExUsedTime;
        u32 topoExRes_i = std::chrono::duration_cast<std::chrono::seconds>(topoExResTime).count();
        if (topoExRes_i == 0) {
            HCCL_ERROR("[RankInfoDetectService::%s] timeout[%lld s] is exhausted", __func__, timeout);
            break;
        }
        std::shared_ptr<Socket> connSocket = std::make_shared<Socket>(
            hccpHostSocketHandle, hostIp_, hostPort, hostIp_, connSocketTag, SocketRole::SERVER,
            NicType::HOST_NIC_TYPE);
        // GetStatus 是阻塞接口，传入剩余时间作为超时上限，避免其内部超时导致外层循环超时处理失效
        EXCEPTION_CATCH(status = connSocket->GetStatus(topoExRes_i), {
            // 非本端client首次连接异常，直接重试
            if (status == SocketStatus::OK) {
                status = SocketStatus::CONNECTING;
            }
            HCCL_ERROR("[RankInfoDetectService::%s] server get socket fail", __func__);
        });
        if (status == SocketStatus::OK) {
            if (!RecvAndVerifyRemoteAgentIdAndRankSize(connSocket, expectedSocketNum, previousRankNum)) {
                break;
            }
            expectedSocketNum--;
            isFirstAcceptTimeOut = false;
            HCCL_INFO("[RankInfoDetectService::%s] socket[%s] connect ok.", __func__, connSocket->Describe().c_str());
        } else if (status == SocketStatus::CONNECTING) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else if (status == SocketStatus::TIMEOUT) {
            // 避免重复打印
            if (isFirstAcceptTimeOut) {
                continue;
            }
            HCCL_ERROR(
                "[%s][%s] topo exchange server get socket timeout[%lld s]", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), static_cast<long long>(timeout.count()));
            DisplayConnectingStatus(previousRankNum, expectedSocketNum);
            isFirstAcceptTimeOut = true;
        } else {
            HCCL_ERROR("[RankInfoDetectService::%s] SocketStatus[%s] error", __func__, status.Describe().c_str());
            break;
        }
    }

    // 如果没有连接成功的rank则退出
    CHK_PRT_THROW(
        connSockets_.size() == 0, HCCL_ERROR("[RankInfoDetectService::%s] no rank connection success.", __func__),
        InternalException, "no rank connection success");

    // 处理异常流程
    if (expectedSocketNum > 0) {
        // 将建立连接超时的client信息添加到failedAgentIdList_
        FailedConnectionAgentIdString(previousRankNum);
        DisplayConnectedRanks();
        HCCL_INFO("[RankInfoDetectService::%s] end, there exist non-connected ranks.", __func__);
    } else {
        HCCL_INFO("[RankInfoDetectService::%s] end, all agentId get connection socket success.", __func__);
    }
}

void RankInfoDetectService::GetRankTable()
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    // 接收localRankTable并组全局RankTableInfo
    rankTable_ = RankTableInfo{};
    for (auto& iter : connSockets_) {
        vector<char> rankInfoMsg{};
        SocketAgent socketAgent(iter.second.get());
        RecvRankInfoMsg(socketAgent, rankInfoMsg);
        ParseRankTable(rankInfoMsg);
    }

    // 按照rankid排序
    SortRankTable();

    // 更新当前阶段
    currentStep_++;

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
}

void RankInfoDetectService::BroadcastRankTable()
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    // 广播全局ranktable
    std::shared_ptr<RankInfoDispatcher> dispatcher = std::make_shared<RankInfoDispatcher>(this);
    dispatcher->BroadcastRankTable(connSockets_, rankTable_, failedAgentIdList_, currentStep_);

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
}

void RankInfoDetectService::Disconnect()
{
    for (auto iter = connSockets_.begin(); iter != connSockets_.end();) {
        iter->second.get()->Close();
        iter = connSockets_.erase(iter);
    }
}

bool RankInfoDetectService::RecvRemoteAgentId(SocketAgent& connSocketAgent, std::string& agentId)
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    // 接收消息
    u64 revMsgLen = 0;
    char msg[MAX_AGENT_BUF_SIZE] = {0};
    bool ret = connSocketAgent.RecvMsg(msg, revMsgLen);
    CHK_PRT_RET(
        !ret || revMsgLen >= MAX_AGENT_BUF_SIZE,
        HCCL_ERROR("[RankInfoDetectService::%s] recv error, revMsgLen[%llu].", __func__, revMsgLen), false);

    // 解析agentId
    msg[revMsgLen] = '\0';
    agentId = msg;

    HCCL_INFO("[RankInfoDetectService::%s] agentId[%s]", __func__, agentId.c_str());
    return true;
}

bool RankInfoDetectService::RecvRemoteRankSize(SocketAgent& connSocketAgent, u32& rankSize)
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    // 接收rankSize
    u64 revMsgLen = 0;
    bool ret = connSocketAgent.RecvMsg(&rankSize, revMsgLen);
    CHK_PRT_RET(
        !ret, HCCL_ERROR("[RankInfoDetectService::%s] RecvMsg fail, revMsgLen[%llu].", __func__, revMsgLen), false);

    HCCL_INFO("[RankInfoDetectService::%s] rankSize[%u]", __func__, rankSize);
    return true;
}

// 接收客户端发送的字节流形式的rankinfo消息
void RankInfoDetectService::RecvRankInfoMsg(SocketAgent& connSocketAgent, vector<char>& rankInfoMsg)
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    u64 revMsgLen = 0;
    std::unique_ptr<HostBuffer> msg = std::make_unique<HostBuffer>(MAX_BUFFER_LEN);
    char* msgAddr = reinterpret_cast<char*>(msg->GetAddr());
    CHK_PRT_THROW(
        !connSocketAgent.RecvMsg(msgAddr, revMsgLen),
        HCCL_ERROR("[RankInfoDetectService::%s] RecvMsg fail, revMsgLen[%llu]", __func__, revMsgLen),
        InvalidParamsException, "RecvMsg fail");

    // 以vector<char>格式保存
    rankInfoMsg.resize(revMsgLen);
    rankInfoMsg.assign(msgAddr, msgAddr + revMsgLen);

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
}

// 解析接收到的rank table信息
void RankInfoDetectService::ParseRankTable(vector<char>& rankInfoMsg)
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    // 消息格式: [ranktable数据(n字节)][step(4字节)]
    BinaryStream binStream(rankInfoMsg);

    // 解析localRankInfo
    RankTableInfo localRankInfo(binStream);
    localRankInfo.Dump();

    // 解析step
    u32 receivedStep;
    binStream >> receivedStep;

    // 校验step是否匹配
    CHK_PRT_THROW(
        receivedStep != currentStep_,
        HCCL_ERROR(
            "[RankInfoDetectService::%s] Step mismatch: received %u, expected %u", __func__, receivedStep,
            currentStep_),
        InvalidParamsException, "Step mismatch");

    // 添加到rankTable_
    rankTable_.UpdateRankTable(localRankInfo);

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
}

bool RankIdCompare(const NewRankInfo& i, const NewRankInfo& j) { return (i.rankId < j.rankId); }

void RankInfoDetectService::SortRankTable()
{
    std::sort(rankTable_.ranks.begin(), rankTable_.ranks.end(), RankIdCompare);
}

void RankInfoDetectService::FailedConnectionAgentIdString(u32 rankSize)
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    std::vector<bool> connectedRank(rankSize, false);
    for (auto it : connSockets_) {
        u32 rankid = 0;
        HcclResult ret = SalStrToULong(it.first, HCCL_BASE_DECIMAL, rankid);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[RankInfoDetectService::%s] agentId[%s] strToULong fail.", __func__, it.first.c_str()), );
        CHK_PRT_RET(
            rankid >= rankSize,
            HCCL_ERROR("[RankInfoDetectService::%s] invalid rank id[%u], rankSize[%u].", __func__, rankid, rankSize), );
        connectedRank[rankid] = true;
    }

    for (u32 i = 0; i < rankSize; i++) {
        if (!connectedRank[i]) {
            if (!failedAgentIdList_.empty()) {
                failedAgentIdList_ += ',';
            }
            failedAgentIdList_ += std::to_string(i);
        }
    }

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
}

// 校验相关方法
bool RankInfoDetectService::RecvAndVerifyRemoteAgentIdAndRankSize(
    std::shared_ptr<Socket> connSocket, u32& expectedSocketNum, u32& previousRankSize)
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);
    SocketAgent socketAgent(connSocket.get());

    // 接收AgentId
    std::string agentId = "";
    bool ret = RecvRemoteAgentId(socketAgent, agentId);
    CHK_PRT_RET(!ret, HCCL_ERROR("[RankInfoDetectService::%s] RecvRemoteAgentId fail.", __func__), false);

    // 保存connSocket
    auto iter = connSockets_.find(agentId);
    CHK_PRT_RET(
        iter != connSockets_.end(),
        HCCL_ERROR("[RankInfoDetectService::%s] agentId[%s] has been connected.", __func__, agentId.c_str()), false);
    connSockets_.insert({agentId, connSocket});

    // 接收RankSize
    u32 rankSize = 0;
    ret = RecvRemoteRankSize(socketAgent, rankSize);
    CHK_PRT_RET(!ret, HCCL_ERROR("[RankInfoDetectService::%s] RecvRemoteAgentId fail.", __func__), false);

    // 校验
    expectedSocketNum = (previousRankSize == 0) ? rankSize : expectedSocketNum;
    CHK_PRT_RET(
        !VerifyRemoteRankSize(previousRankSize, rankSize),
        HCCL_ERROR("[RankInfoDetectService::%s] VerifyRemoteRankSize fail, rankSize[%u]", __func__, rankSize), false);

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
    return true;
}

bool RankInfoDetectService::VerifyRemoteRankSize(u32& previousRankSize, u32 remoteRankSize) const
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    if (previousRankSize == 0) {
        previousRankSize = remoteRankSize;
    } else {
        if (previousRankSize != remoteRankSize) {
            HCCL_ERROR(
                "[RankInfoDetectService::%s] VerifyRemoteRankSize failed. remoteRankSize[%u] is different "
                "from previousRankSize[%u].",
                __func__, remoteRankSize, previousRankSize);
            return false;
        }
    }

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
    return true;
}

// DFX相关方法
void RankInfoDetectService::DisplayConnectedRanks()
{
    vector<std::string> ranksInfo;
    for (const auto& it : connSockets_) {
        ranksInfo.push_back(it.first);
    }
    u64 ranksLen = ranksInfo.size();
    u64 lineNum = (ranksInfo.size() % DISPLAY_RANKNUM_PERLINE == 0) ? (ranksInfo.size() / DISPLAY_RANKNUM_PERLINE) :
                                                                      (ranksInfo.size() / DISPLAY_RANKNUM_PERLINE + 1);
    HCCL_ERROR(
        "[RankInfoDetectService::%s] total connected num is [%llu],line num is [%llu]", __func__, ranksLen, lineNum);
    for (u64 i = 0; i < lineNum; i++) {
        std::string tmpRankList;
        for (u32 j = 0; j < DISPLAY_RANKNUM_PERLINE; j++) {
            u32 ranksInfoIndex = i * DISPLAY_RANKNUM_PERLINE + j;
            if (ranksInfoIndex < ranksInfo.size()) {
                tmpRankList += "[" + ranksInfo[ranksInfoIndex] + "]";
            } else {
                break;
            }
            tmpRankList += ((j == DISPLAY_RANKNUM_PERLINE - 1 || ranksInfoIndex == ranksInfo.size() - 1) ? ";" : ",");
        }
        HCCL_ERROR("[RankInfoDetectService::%s] connected rankinfo[LINE %llu]: %s", __func__, i, tmpRankList.c_str());
    }
}

void RankInfoDetectService::DisplayConnectingStatus(u32 totalSockets, u32 waitSockets)
{
    if (totalSockets == 0 && waitSockets == 1) {
        HCCL_INFO("[RankInfoDetectService::%s] wait for first connection.", __func__);
        return;
    }

    std::vector<bool> rankinfos(totalSockets, false);
    for (auto it : connSockets_) { // 建立映射
        u32 rankid = 0;
        HcclResult ret = SalStrToULong(it.first, HCCL_BASE_DECIMAL, rankid);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[RankInfoDetectService::%s] agentId[%s] strToULong fail.", __func__, it.first.c_str()), );
        CHK_PRT_RET(
            rankid >= totalSockets,
            HCCL_ERROR(
                "[RankInfoDetectService::%s] invalid rankid[%u], rankSize[%u].", __func__, rankid, totalSockets), );
        rankinfos[rankid] = true;
    }

    u32 unRankCount = 0; // 只打印前三条未建链的rank
    std::vector<std::string> unsocketinfos;
    for (u32 rankid = 0; rankid < totalSockets; rankid++) {
        if (unRankCount >= SOCKET_PRINT_COUNT) {
            break;
        }
        if (!rankinfos[rankid]) {
            unRankCount++;
            std::string rankID = std::to_string(rankid);
            std::string agentID = std::string(16 - rankID.length(), '0') + rankID;
            unsocketinfos.push_back(agentID);
        }
    }

    std::string infoStr = "succ sockets are [" + std::to_string((totalSockets - waitSockets))
                          + "], waiting sockets is [" + std::to_string(waitSockets) + "], wait sockets rankid: ";
    for (u32 index = 0; index < unsocketinfos.size(); index++) {
        if (index == (unsocketinfos.size() - 1)) {
            infoStr += "[" + unsocketinfos[index] + "]";
        } else {
            infoStr += "[" + unsocketinfos[index] + "],";
        }
    }

    HCCL_INFO("[RankInfoDetectService::%s] %s", __func__, infoStr.c_str());
}

void RankInfoDetectService::TearDown()
{
    HCCL_INFO("[RankInfoDetectService::%s] start.", __func__);

    CHK_PRT_RET(!serverSocket_, HCCL_INFO("[RankInfoDetectService::%s] serverSocket is null", __func__), );

    // close socket
    Disconnect();

    // 如果白名单使能则删除白名单
    if (!EnvConfig::GetInstance().GetHostNicConfig().GetWhitelistDisable()) {
        CHK_PRT_CONT(wlistInfo_.size() == 0, HCCL_ERROR("whitelist is empty"); break);
        SocketHandle hostSocketHandle = HostSocketHandleManager::GetInstance().Get(devPhyId_, hostIp_);
        HrtRaSocketWhiteListDel(hostSocketHandle, wlistInfo_);
    }

    s32 deviceLogicId = HrtGetDevice();
    if (EnvConfig::GetInstance().GetHostNicConfig().GetHostSocketPortRange().size() > 0
        || EnvConfig::GetInstance().GetHostNicConfig().GetIfBasePort() == HCCL_INVALID_PORT) {
        // 若开启抢占监听端口
        PreemptPortManager::GetInstance(deviceLogicId).Release(serverSocket_);
    } else {
        // 停止监听
        serverSocket_->StopListen();
    }

    // deinit handle
    HostSocketHandleManager::GetInstance().Destroy(devPhyId_, hostIp_);

    // deinit ra
    HccpPeerManager::GetInstance().DeInit(deviceLogicId);

    HCCL_INFO("[RankInfoDetectService::%s] end.", __func__);
}

RankInfoDetectService::~RankInfoDetectService() { DECTOR_TRY_CATCH("RankInfoDetectService", TearDown()); }

} // namespace Hccl
