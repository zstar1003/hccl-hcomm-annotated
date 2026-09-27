/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exchange_info_mgr.h"
#include "env_config/env_config_v2.h"
#include "rank_consistency_checker_v2.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"
#include "comm_engine_utils.h"

namespace hccl {
ExchangeInfoMgr::ExchangeInfoMgr() {}

ExchangeInfoMgr::~ExchangeInfoMgr()
{
    HCCL_INFO("[ExchangeInfoMgr][~ExchangeInfoMgr] CollCommConfigConsistency deinit");
}

HcclResult ExchangeInfoMgr::BatchExchangeAndCheckConsistency(
    const HcclChannelDesc* channelDescs, const std::vector<HcommChannelDesc>& hcommDescs, uint32_t channelNum,
    const std::vector<std::pair<u32, u32>>& newChannels, CollCommConfigConsistency& collCommConfigConsistency,
    CommEngine engine) const
{
    std::vector<Hccl::Socket*> sockets;
    std::vector<u32> remoteRanks;
    std::vector<HcommSocketRole> roles;

    if (channelNum == 0) {
        HCCL_INFO("[BatchExchangeAndCheckConsistency] channelNum is 0.");
        return HCCL_SUCCESS;
    }

    for (uint32_t i = 0; i < channelNum; i++) {
        u32 remoteRank = channelDescs[i].remoteRank;
        HcommSocket rawSocket = hcommDescs[i].socket;
        Hccl::Socket* socket = static_cast<Hccl::Socket*>(rawSocket);
        CHK_PRT_RET(
            socket == nullptr,
            HCCL_ERROR(
                "[BatchExchangeAndCheckConsistency] socket is null for channel[%u] remoteRank[%u].", i, remoteRank),
            HCCL_E_INTERNAL);
        sockets.push_back(socket);
        remoteRanks.push_back(remoteRank);
        roles.push_back(hcommDescs[i].role);
    }

    bool isHostLoc = (channelDescs[0].localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST);
    HCCL_INFO(
        "[BatchExchangeAndCheckConsistency] isHostLoc[%d], engine[%s]", isHostLoc,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());

    // host 网卡使用 Socket 同步收发接口
    if (isHostLoc) {
        // 校验HComm信息
        CHK_RET(CheckHcommInfo(channelDescs, hcommDescs, sockets, remoteRanks, roles, newChannels, false));
        // 交换HCCL算子信息
        CHK_RET(ExchangeUserInfo(sockets, remoteRanks, roles, collCommConfigConsistency));
    } else {
        // 校验HComm信息
        CHK_RET(CheckHcommInfo(channelDescs, hcommDescs, sockets, remoteRanks, roles, newChannels, true));
        // 交换HCCL算子信息
        CHK_RET(ExchangeUserInfoAsync(sockets, remoteRanks, roles, collCommConfigConsistency));
    }
    CHK_RET(collCommConfigConsistency.ResetExchangeInfo());

    return HCCL_SUCCESS;
}

HcclResult ExchangeInfoMgr::CheckHcommInfo(
    const HcclChannelDesc* channelDescs, const std::vector<HcommChannelDesc>& hcommDescs,
    const std::vector<Hccl::Socket*>& sockets, const std::vector<u32>& remoteRanks,
    const std::vector<HcommSocketRole>& roles, const std::vector<std::pair<u32, u32>>& newChannels, bool isAsync) const
{
    s32 deviceLogicId = 0;
    (void)hrtGetDeviceRefresh(&deviceLogicId);
    auto& checker = RankConsistencyCheckerV2::GetInstance(deviceLogicId);
    u64 frameLenV2 = checker.GetCheckFrameLengthV2();
    std::vector<Hccl::Socket*> newSockets; // 新建链的
    std::vector<u32> newRemoteRanks;
    std::vector<HcommSocketRole> newRoles;

    for (const auto& newCh : newChannels) {
        u32 idx = newCh.first;
        u32 remoteRank = channelDescs[idx].remoteRank;
        HcommSocket rawSocket = hcommDescs[idx].socket;
        Hccl::Socket* socket = static_cast<Hccl::Socket*>(rawSocket);
        CHK_PRT_RET(
            socket == nullptr,
            HCCL_ERROR("[CheckHcommInfo] socket is null for channel[%u] remoteRank[%u].", idx, remoteRank),
            HCCL_E_INTERNAL);
        newSockets.push_back(socket);
        newRemoteRanks.push_back(remoteRank);
        newRoles.push_back(hcommDescs[idx].role);
    }
    if (newChannels.size() > 0) {
        checker.SetInconsistentCheckFirstDone(false);
    }
    // 只有rankConsistentState是first或者on时才进行hcomm信息校验
    int32_t mode = Hccl::EnvConfig::GetInstance().GetLogConfig().GetDfsConfig().rankConsistentState;
    bool isFirst = (mode == 0 && !checker.GetInconsistentCheckFirstDone()); // 0:first 1:on
    if (mode == 1 || isFirst) {
        // ====== 生成本端CheckFrameV2 ======
        CheckFrameV2 localFrame;
        CHK_RET(checker.GenerateCheckFrameV2(localFrame));

        // ====== 交换CheckFrameV2（定长，批量并发交换）======
        std::vector<CheckFrameV2> remoteFrames;
        u32 checkSocketSize = newSockets.size();
        if (isFirst) {
            remoteFrames.resize(newSockets.size());
            CHK_RET(BatchExchangeFixedData(
                newSockets, newRemoteRanks, newRoles, reinterpret_cast<const u8*>(&localFrame),
                static_cast<u32>(frameLenV2), reinterpret_cast<u8*>(remoteFrames.data()), static_cast<u32>(frameLenV2),
                isAsync));
        } else {
            checkSocketSize = sockets.size();
            remoteFrames.resize(sockets.size());
            CHK_RET(BatchExchangeFixedData(
                sockets, remoteRanks, roles, reinterpret_cast<const u8*>(&localFrame), static_cast<u32>(frameLenV2),
                reinterpret_cast<u8*>(remoteFrames.data()), static_cast<u32>(frameLenV2), isAsync));
        }

        // ====== 逐个比对CheckFrameV2（精确报错：环境变量名/子通信域参数名等）======
        for (u32 i = 0; i < checkSocketSize; i++) {
            CHK_RET(checker.CompareCheckFrameV2(localFrame, remoteFrames[i]));
        }
        HCCL_INFO("[BatchExchangeAndCheckConsistency] hcomm compare check suc.");
        checker.SetInconsistentCheckFirstDone(true);
    }
    return HCCL_SUCCESS;
}

HcclResult ExchangeInfoMgr::ExchangeUserInfo(
    const std::vector<Hccl::Socket*>& sockets, const std::vector<u32>& remoteRanks,
    const std::vector<HcommSocketRole>& roles, hccl::CollCommConfigConsistency& collCommConfigConsistency) const
{
    (void)roles;

    u32 localExchangeInfoLen = collCommConfigConsistency.GetExchangeInfoLen();
    if (localExchangeInfoLen == 0) {
        HCCL_INFO("[ExchangeUserInfo] localExchangeInfoLen is 0.");
        return HCCL_SUCCESS;
    }

    // 交换infoLen
    std::vector<u32> remoteExchangeInfoLens(sockets.size(), 0);
    u8* recvData = reinterpret_cast<u8*>(remoteExchangeInfoLens.data());
    const u8* sendData = reinterpret_cast<const u8*>(&localExchangeInfoLen);
    u32 len = sizeof(u32);
    for (u32 i = 0; i < sockets.size(); i++) {
        sockets[i]->Send(sendData, len);
        sockets[i]->Recv(recvData + i * len, len);
    }

    // 交换info数据（长度可能不同，需逐个收发）
    std::vector<std::vector<u8>> remoteUserDatas(sockets.size());
    for (u32 i = 0; i < sockets.size(); i++) {
        std::vector<u8> exchangeBuf;
        collCommConfigConsistency.GetExchangeInfoBuf(exchangeBuf);
        remoteUserDatas[i].resize(remoteExchangeInfoLens[i], 0);
        sockets[i]->Send(exchangeBuf.data(), localExchangeInfoLen);
        sockets[i]->Recv(remoteUserDatas[i].data(), remoteExchangeInfoLens[i]);
    }

    // 存储对端交换信息
    for (u32 i = 0; i < sockets.size(); i++) {
        if (remoteExchangeInfoLens[i] > 0 && !remoteUserDatas[i].empty()) {
            CHK_RET(collCommConfigConsistency.StoreRemoteExchangeInfo(remoteRanks[i], remoteUserDatas[i]));
        }
    }

    HCCL_INFO("[ExchangeUserInfo] suc.");
    return HCCL_SUCCESS;
}

HcclResult ExchangeInfoMgr::ExchangeUserInfoAsync(
    const std::vector<Hccl::Socket*>& sockets, const std::vector<u32>& remoteRanks,
    const std::vector<HcommSocketRole>& roles, CollCommConfigConsistency& collCommConfigConsistency) const
{
    u32 localExchangeInfoLen = collCommConfigConsistency.GetExchangeInfoLen();
    if (localExchangeInfoLen == 0) {
        HCCL_INFO("[ExchangeUserInfoAsync] localExchangeInfoLen is 0.");
        return HCCL_SUCCESS;
    }
    // 交换infoLen
    std::vector<u32> remoteExchangeInfoLens(sockets.size(), 0);
    CHK_RET(BatchExchangeFixedData(
        sockets, remoteRanks, roles, reinterpret_cast<const u8*>(&localExchangeInfoLen), sizeof(u32),
        reinterpret_cast<u8*>(remoteExchangeInfoLens.data()), sizeof(u32), true));

    // 交换info数据（两阶段异步收发，防死锁）
    std::vector<std::vector<u8>> remoteUserDatas(sockets.size());
    // 第一阶段：SERVER先Recv/CLIENT先Send
    CHK_RET(ExchangeAsyncDataPhase(
        sockets, roles, remoteRanks, remoteUserDatas, remoteExchangeInfoLens, localExchangeInfoLen,
        collCommConfigConsistency, true));
    CHK_RET(WaitActiveAsyncComplete(sockets, remoteRanks, roles, remoteExchangeInfoLens, localExchangeInfoLen, true));
    // 第二阶段：SERVER再Send/CLIENT再Recv
    CHK_RET(ExchangeAsyncDataPhase(
        sockets, roles, remoteRanks, remoteUserDatas, remoteExchangeInfoLens, localExchangeInfoLen,
        collCommConfigConsistency, false));
    CHK_RET(WaitActiveAsyncComplete(sockets, remoteRanks, roles, remoteExchangeInfoLens, localExchangeInfoLen, false));

    // 存储对端交换信息
    for (u32 i = 0; i < sockets.size(); i++) {
        if (remoteExchangeInfoLens[i] > 0 && !remoteUserDatas[i].empty()) {
            CHK_RET(collCommConfigConsistency.StoreRemoteExchangeInfo(remoteRanks[i], remoteUserDatas[i]));
        }
    }

    HCCL_INFO("[ExchangeUserInfoAsync] suc.");
    return HCCL_SUCCESS;
}

// 异步交换info数据的一个阶段（isServerRecv=true: SERVER先Recv; false: SERVER先Send）
HcclResult ExchangeInfoMgr::ExchangeAsyncDataPhase(
    const std::vector<Hccl::Socket*>& sockets, const std::vector<HcommSocketRole>& roles,
    const std::vector<u32>& remoteRanks, std::vector<std::vector<u8>>& remoteUserDatas,
    const std::vector<u32>& remoteExchangeInfoLens, u32 localExchangeInfoLen,
    CollCommConfigConsistency& collCommConfigConsistency, bool isServerRecv) const
{
    for (u32 i = 0; i < sockets.size(); i++) {
        bool shouldRecv = (roles[i] == HCOMM_SOCKET_ROLE_SERVER) == isServerRecv;
        if (shouldRecv) {
            CHK_PRT_RET(
                remoteExchangeInfoLens[i] > HCCL_EXCHANGE_INFO_LEN,
                HCCL_ERROR(
                    "[ExchangeUserInfoAsync] remoteExchangeInfoLen[%u] for remoteRank[%u] "
                    "exceeds max allowed [%u].",
                    remoteExchangeInfoLens[i], remoteRanks[i], HCCL_EXCHANGE_INFO_LEN),
                HCCL_E_PARA);
            remoteUserDatas[i].resize(remoteExchangeInfoLens[i], 0);
            sockets[i]->RecvAsync(remoteUserDatas[i].data(), remoteExchangeInfoLens[i]);
        } else {
            std::vector<u8> exchangeBuf;
            collCommConfigConsistency.GetExchangeInfoBuf(exchangeBuf);
            sockets[i]->SendAsync(exchangeBuf.data(), localExchangeInfoLen);
        }
    }
    return HCCL_SUCCESS;
}

// 批量异步交换定长数据（SERVER先Recv再Send，CLIENT先Send再Recv，防死锁）
HcclResult ExchangeInfoMgr::BatchExchangeFixedData(
    const std::vector<Hccl::Socket*>& sockets, const std::vector<u32>& remoteRanks,
    const std::vector<HcommSocketRole>& roles, const u8* sendData, u32 sendLen, u8* recvData, u32 recvLen,
    bool isAsync) const
{
    if (!isAsync) {
        // 确保 Socket 连接完成
        CHK_RET(WaitAllAsyncComplete(sockets, remoteRanks));
        // 使用 Socket 同步收发接口
        for (u32 i = 0; i < sockets.size(); i++) {
            sockets[i]->Send(sendData, sendLen);
            sockets[i]->Recv(recvData + i * recvLen, recvLen);
        }
        return HCCL_SUCCESS;
    }

    CHK_RET(WaitAllAsyncComplete(sockets, remoteRanks));
    // SERVER先Recv/CLIENT先Send
    for (u32 i = 0; i < sockets.size(); i++) {
        if (roles[i] == HCOMM_SOCKET_ROLE_SERVER) {
            sockets[i]->RecvAsync(recvData + i * recvLen, recvLen);
        } else {
            sockets[i]->SendAsync(sendData, sendLen);
        }
    }
    CHK_RET(WaitAllAsyncComplete(sockets, remoteRanks));

    // SERVER再Send/CLIENT再Recv
    for (u32 i = 0; i < sockets.size(); i++) {
        if (roles[i] == HCOMM_SOCKET_ROLE_SERVER) {
            sockets[i]->SendAsync(sendData, sendLen);
        } else {
            sockets[i]->RecvAsync(recvData + i * recvLen, recvLen);
        }
    }
    CHK_RET(WaitAllAsyncComplete(sockets, remoteRanks));

    return HCCL_SUCCESS;
}

HcclResult ExchangeInfoMgr::WaitAllAsyncComplete(
    const std::vector<Hccl::Socket*>& sockets, const std::vector<u32>& remoteRanks) const
{
    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    std::vector<bool> done(sockets.size(), false);
    uint32_t doneCount = 0;

    while (doneCount < sockets.size()) {
        for (size_t i = 0; i < sockets.size(); i++) {
            if (done[i]) {
                continue;
            }
            Hccl::SocketStatus status = sockets[i]->GetAsyncStatus();
            if (status == Hccl::SocketStatus::OK) {
                done[i] = true;
                doneCount++;
                continue;
            }
            if (status == Hccl::SocketStatus::TIMEOUT) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - startTime)
                                   .count();
                HCCL_ERROR(
                    "[WaitAllAsyncComplete] socket timeout for remoteRank[%u], elapsed[%lld]ms.", remoteRanks[i],
                    elapsed);
                return HCCL_E_TIMEOUT;
            }
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            for (size_t i = 0; i < sockets.size(); i++) {
                if (!done[i]) {
                    HCCL_ERROR(
                        "[WaitAllAsyncComplete] wall-clock timeout for remoteRank[%u], elapsed[%lld]ms.",
                        remoteRanks[i], elapsed);
                }
            }
            return HCCL_E_TIMEOUT;
        }
    }

    auto elapsed
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
    HCCL_INFO("[WaitAllAsyncComplete] all[%zu] sockets completed, elapsed[%lld]ms.", sockets.size(), elapsed);
    return HCCL_SUCCESS;
}

// 收集并等待有实际异步操作的socket子集
HcclResult ExchangeInfoMgr::WaitActiveAsyncComplete(
    const std::vector<Hccl::Socket*>& sockets, const std::vector<u32>& remoteRanks,
    const std::vector<HcommSocketRole>& roles, const std::vector<u32>& remoteExchangeInfoLens, u32 localExchangeInfoLen,
    bool isFirstPass) const
{
    std::vector<Hccl::Socket*> activeSockets;
    std::vector<u32> activeRanks;
    for (u32 i = 0; i < sockets.size(); i++) {
        bool isActive = isFirstPass ? (roles[i] == HCOMM_SOCKET_ROLE_SERVER && remoteExchangeInfoLens[i] > 0)
                                          || (roles[i] != HCOMM_SOCKET_ROLE_SERVER && localExchangeInfoLen > 0) :
                                      (roles[i] == HCOMM_SOCKET_ROLE_SERVER && localExchangeInfoLen > 0)
                                          || (roles[i] != HCOMM_SOCKET_ROLE_SERVER && remoteExchangeInfoLens[i] > 0);
        if (isActive) {
            activeSockets.push_back(sockets[i]);
            activeRanks.push_back(remoteRanks[i]);
        }
    }
    if (!activeSockets.empty()) {
        CHK_RET(WaitAllAsyncComplete(activeSockets, activeRanks));
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
