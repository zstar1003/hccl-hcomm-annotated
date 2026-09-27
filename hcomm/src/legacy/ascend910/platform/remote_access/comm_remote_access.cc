/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_remote_access.h"
#include "externalinput_pub.h"

namespace hccl {
using namespace std;

CommRemoteAccess::CommRemoteAccess(
    u32 rank, u32 devicePhyId, const std::map<u32, std::vector<HcclIpAddress>>& rankInfo,
    const std::vector<MemRegisterAddr>& addrInfos)
    : remoteTransportMap_(),
      rank_(rank),
      deviceLogicId_(0),
      devicePhyId_(devicePhyId),
      rankSize_(0),
      nicDeployment_(NICDeployment::NIC_DEPLOYMENT_DEVICE),
      rankInfo_(rankInfo),
      addrInfos_(addrInfos),
      dstInterServerMap_(),
      dstInterClientMap_(),
      nicSocketHandle_(),
      tag_("RemoteAccess"),
      threadsApplyNum_(0),
      dispatcher_(nullptr),
      notifyPool_(nullptr)
{}

CommRemoteAccess::~CommRemoteAccess()
{
    HcclResult ret = HCCL_SUCCESS;
    for (u32 index = 0; index < linkThreads_.size(); index++) {
        if (linkThreads_[index]) {
            if (linkThreads_[index]->joinable()) {
                HCCL_DEBUG("Joining Link Thread[%u]", index);
                linkThreads_[index]->join(); // 等待线程执行后释放资源
            }
            ret = hrtResetDevice(deviceLogicId_); // 防止线程里面异常退出，在进程中reset
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[Comm][RemoteAccess]CommRemoteAccess reset device[%d] failed", deviceLogicId_);
            }
        }
    }
    struct SocketCloseInfoT socketCloseInfo = {};
    for (u32 i = 0; i < raSockets_.size(); i++) {
        socketCloseInfo.socketHandle = raSockets_[i].socketHandle; // 带入设备ID为物理ID
        socketCloseInfo.fdHandle = raSockets_[i].fdHandle;
        if ((raSockets_[i].socketHandle != nullptr) && (raSockets_[i].fdHandle != nullptr)) {
            ret = hrtRaSocketBatchClose(&socketCloseInfo, 1); /* 销毁已经创建的socket */
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING("~CommRemoteAccess:socket batch close fail! ret=%d", ret);
            }
        }
    }

    ret = DeleteSocketWhiteList();
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("~CommRemoteAccess:delete Socket whiteList fail! ret=%d", ret);
    }
    remoteTransportMap_.clear();

    ret = CommRemoteDeInitRa();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Comm][RemoteAccess]CommRemoteAccess CommRemoteDeInitRa fail! ret[%d]", ret);
    }
    ret = notifyPool_->UnregisterOp(tag_);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("~CommRemoteAccess:UnregisterOp fail! ret=%d", ret);
    }

    if (dispatcher_ != nullptr) {
        HcclDispatcherDestroy(dispatcher_);
        dispatcher_ = nullptr;
    }
}

HcclResult CommRemoteAccess::Init()
{
    HCCL_INFO("CommRemoteAccess Init start");
    // 获取当前线程操作的设备ID
    CHK_RET(hrtGetDevice(&deviceLogicId_));
    // dispatcher 资源初始化
    CHK_RET(RescoucePrepare());
    // 初始化ra资源，若hcom_init则不会进行再初始化
    CHK_RET(CommRemoteInitRa());
    CHK_RET(NetworkManager::GetInstance(deviceLogicId_).GetRaResourceInfo(raResourceInfo_));
    // 建链关系计算，只和统一平面和本端rank建单向链
    CHK_RET(CalcRemoteLink());
    // socket资源准备（白名单、batch connect
    CHK_RET(PrepareSocket());
    // 创建链接并保存fd_socket_handle
    CHK_RET(CreateLinks());
    HCCL_INFO("CommRemoteAccess Init end");
    return HCCL_SUCCESS;
}

HcclResult CommRemoteAccess::DeleteSocketWhiteList()
{
    if (wlistInfosVec_.size() > 0) {
        for (u32 idx = 0; idx < nicSocketHandle_.size(); idx++) {
            CHK_RET(hrtRaSocketWhiteListDel(nicSocketHandle_[idx], wlistInfosVec_.data(), wlistInfosVec_.size()));
        }
    }
    return HCCL_SUCCESS;
}

std::shared_ptr<TransportRemoteAccess>& CommRemoteAccess::GetTransportByRank(const u32 dstRank)
{
    if (remoteTransportMap_.find(dstRank) == remoteTransportMap_.end()) {
        HCCL_ERROR(
            "[Get][TransportByRank]can not find dstRank[%u] in remoteTransportMap_,"
            "remoteTransportMap_ size is [%llu]",
            dstRank, remoteTransportMap_.size());
        return transportDummy_;
    }

    return remoteTransportMap_.lower_bound(dstRank)->second;
}

HcclResult CommRemoteAccess::RescoucePrepare()
{
    // 根据设备ID创建dispatcher
    CHK_SMART_PTR_NULL(dispatcher_);

    notifyPool_.reset(new (std::nothrow) NotifyPool());
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_RET(notifyPool_->Init(devicePhyId_));
    CHK_RET(notifyPool_->RegisterOp(tag_));
    return HCCL_SUCCESS;
}

HcclResult CommRemoteAccess::CommRemoteInitRa()
{
    CHK_RET(NetworkManager::GetInstance(deviceLogicId_).Init(nicDeployment_));

    auto iter = rankInfo_.find(rank_);
    bool check = (iter == rankInfo_.end());
    CHK_PRT_RET(check, HCCL_ERROR("[Get][Instance]can not find rank[%u] info in rankInfo_", rank_), HCCL_E_PARA);
    HCCL_INFO("in CommRemoteInitRa, rank_[%u], iter->second.size[%d]", rank_, iter->second.size());
    for (size_t ipIdex = 0; ipIdex < iter->second.size(); ipIdex++) {
        if (iter->second[ipIdex].IsInvalid()) {
            continue;
        }
        u32 port = HETEROG_CCL_PORT;
        HcclResult ret = NetworkManager::GetInstance(deviceLogicId_).StartNic(iter->second[ipIdex], port, true);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitRa][CommRemote]start nic ipaddr[%s] failed", iter->second[ipIdex].GetReadableAddress()),
            ret);
    }
    return HCCL_SUCCESS;
}

HcclResult CommRemoteAccess::CommRemoteDeInitRa()
{
    auto iter = rankInfo_.find(rank_);
    bool check = (iter == rankInfo_.end());
    CHK_PRT_RET(check, HCCL_ERROR("[DeInit][Ra]can not find rank[%u] info in rankInfo_", rank_), HCCL_E_PARA);
    HCCL_INFO("in CommRemoteDeInitRa, rank_[%u], iter->second.size[%zu]", rank_, iter->second.size());
    for (size_t ipIdex = 0; ipIdex < iter->second.size(); ipIdex++) {
        if (iter->second[ipIdex].IsInvalid()) {
            continue;
        }
        CHK_RET(NetworkManager::GetInstance(deviceLogicId_).StopNic(iter->second[ipIdex], 0));
    }
    CHK_RET(NetworkManager::GetInstance(deviceLogicId_).DeInit(nicDeployment_));
    return HCCL_SUCCESS;
}

HcclResult CommRemoteAccess::CalcRemoteLink()
{
    rankSize_ = rankInfo_.size();
    if ((rankSize_ == 0)) {
        HCCL_ERROR("[Calc][RemoteLink]invalid rankSize, rankSize:[%zu].", rankSize_);
        return HCCL_E_PARA;
    }

    // 计算同一平面的server端rank信息，小于或等于本rank的为server端（本rank既是server也是client）
    for (u32 serverRank = rank_;; serverRank--) {
        HCCL_INFO("CalcRemoteLink rank[%u] serverRank[%u]", rank_, serverRank);
        auto serverRankInfo = rankInfo_.find(serverRank);
        if (serverRankInfo == rankInfo_.end()) {
            HCCL_ERROR("[Calc][RemoteLink]can not find server rank[%u] in rankInfo_.", serverRank);
            return HCCL_E_PARA;
        }
        std::vector<HcclIpAddress> ipVec(rankInfo_[serverRank]);
        auto serverIter = dstInterServerMap_.find(serverRank);
        if (serverIter == dstInterServerMap_.end()) {
            dstInterServerMap_.insert({serverRank, ipVec});
        }
        if (serverRank == 0) {
            break;
        }
    }
    // 计算同一平面的client端rank信息，大于等于本rank的为client端（本rank既是server也是client）
    for (u32 clientRank = rank_; clientRank < rankSize_; clientRank++) {
        auto clientRankInfo = rankInfo_.find(clientRank);
        if (clientRankInfo == rankInfo_.end()) {
            HCCL_ERROR("[Calc][RemoteLink]can not find client rank[%u] in rankInfo_.", clientRank);
            return HCCL_E_PARA;
        }
        std::vector<HcclIpAddress> ipVec(rankInfo_[clientRank]);
        auto clientIter = dstInterClientMap_.find(clientRank);
        if (clientIter == dstInterClientMap_.end()) {
            dstInterClientMap_.insert({clientRank, ipVec});
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommRemoteAccess::PrepareSocket()
{
    // socket handle 是一一对应关系
    if (dstInterServerMap_.size() * raResourceInfo_.nicSocketMap.size()
            + dstInterClientMap_.size() * raResourceInfo_.nicSocketMap.size()
        > 0) {
        for (u32 idx = 0; idx < rankInfo_[rank_].size(); idx++) {
            if (rankInfo_[rank_][idx].IsInvalid()) {
                HCCL_ERROR("[Prepare][Socket]rank_[%u] nicIp[%u] is 0", rank_, idx);
                continue;
            }
            auto it = raResourceInfo_.nicSocketMap.find(rankInfo_[rank_][idx]);
            if (it == raResourceInfo_.nicSocketMap.end()) {
                HCCL_ERROR(
                    "[Prepare][Socket]can not find nicSocketHandle, ip[%s]",
                    rankInfo_[rank_][idx].GetReadableAddress());
                return HCCL_E_PARA;
            } else {
                if (it->second.nicSocketHandle == nullptr) {
                    HCCL_ERROR(
                        "[Prepare][Socket]CommRemoteAccess prepare socket failed! rank[%u] IP addr[%s]", rank_,
                        rankInfo_[rank_][idx].GetReadableAddress());
                    return HCCL_E_PARA;
                }
                nicSocketHandle_.push_back(it->second.nicSocketHandle);
            }
            HCCL_INFO(
                "rank[%u], nicSocketMap[%u] nicIp.size[%u]", rank_, raResourceInfo_.nicSocketMap.size(),
                rankInfo_[rank_].size());
        }
    }

    HCCL_INFO("In PrepareSocket raResourceInfo_Size[%u]", nicSocketHandle_.size());
    CHK_RET(AddSocketWhiteList());

    // 当前rank作为client端batch connect动作
    u32 dstInterServerNum = dstInterServerMap_.size() * raResourceInfo_.nicSocketMap.size();
    HCCL_INFO("socket batch connect dstInterServerNum[%u]", dstInterServerNum);
    if (dstInterServerNum > 0) {
        std::vector<struct SocketConnectInfoT> conns(dstInterServerNum);
        struct SocketConnectInfoT* conn = conns.data();
        s32 sRet = memset_s(
            conn, sizeof(struct SocketConnectInfoT) * dstInterServerNum, 0,
            sizeof(struct SocketConnectInfoT) * dstInterServerNum);
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR(
                "[Prepare][Socket]memory set failed, return[%d]. params:"
                "destMaxSize[%zu], c[%d], count[%zu]",
                sRet, sizeof(struct SocketConnectInfoT) * dstInterServerNum, 0,
                sizeof(struct SocketConnectInfoT) * dstInterServerNum),
            HCCL_E_MEMORY);

        u32 loop = 0;
        for (auto iter = dstInterServerMap_.begin(); iter != dstInterServerMap_.end(); iter++) {
            for (u32 idx = 0; idx < nicSocketHandle_.size() && idx < (iter->second).size(); idx++) {
                conn[loop].remoteIp.addr = (iter->second)[idx].GetBinaryAddress().addr;
                conn[loop].remoteIp.addr6 = (iter->second)[idx].GetBinaryAddress().addr6;
                conn[loop].socketHandle = nicSocketHandle_[idx];
                conn[loop].port = HETEROG_CCL_PORT;
                if (nicSocketHandle_[idx] == nullptr) {
                    HCCL_ERROR("[Prepare][Socket]index[%u] nicSocketHandle_ is null", idx);
                    return HCCL_E_INTERNAL;
                }
                sRet = memcpy_s(&conn[loop].tag[0], sizeof(conn[loop].tag) - 1, tag_.c_str(), tag_.size());
                CHK_PRT_RET(
                    sRet != EOK,
                    HCCL_ERROR(
                        "[Prepare][Socket]memcpy failed. errorno[%d]:"
                        "destMaxSize[%zu], count[%zu]",
                        sRet, sizeof(conn[loop].tag), tag_.size()),
                    HCCL_E_MEMORY);
            }
            loop++;
        }
        HcclResult ret = hrtRaSocketBatchConnect(conn, dstInterServerNum);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Prepare][Socket]socket batch failed, batch size[%u], loop[%u], "
                "dst_inter_server_map_size[%u], handle size[%u]",
                dstInterServerNum, loop, dstInterServerMap_.size(), nicSocketHandle_.size()),
            ret);
    }

    return HCCL_SUCCESS;
}

HcclResult CommRemoteAccess::AddSocketWhiteList()
{
    // 当前rank作为server端socket白名单下发动作
    struct SocketWlistInfoT wlistInfo = {};
    for (auto iter = dstInterClientMap_.begin(); iter != dstInterClientMap_.end(); iter++) {
        wlistInfo.connLimit = NIC_SOCKET_CONN_LIMIT;
        s32 sRet = memcpy_s(&wlistInfo.tag[0], sizeof(wlistInfo.tag) - 1, tag_.c_str(), tag_.size());
        if (sRet != EOK) {
            HCCL_ERROR(
                "[Add][Socket]memory copy failed. errorno[%d]: dest size[%zu], src[%s],"
                "count[%zu]",
                sRet, sizeof(wlistInfo.tag), tag_.c_str(), tag_.size());
            wlistInfosVec_.clear();
            return HCCL_E_MEMORY;
        }
        for (u32 idx = 0; idx < (iter->second).size(); idx++) {
            wlistInfo.remoteIp.addr = iter->second[idx].GetBinaryAddress().addr;
            wlistInfo.remoteIp.addr6 = iter->second[idx].GetBinaryAddress().addr6;
            wlistInfosVec_.push_back(wlistInfo);
        }
    }

    if (wlistInfosVec_.size() > 0) {
        for (u32 idx = 0; idx < nicSocketHandle_.size(); idx++) {
            CHK_RET(hrtRaSocketWhiteListAdd(nicSocketHandle_[idx], wlistInfosVec_.data(), wlistInfosVec_.size()));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult CommRemoteAccess::CreateLinks()
{
    // 计算建链所需线程
    u32 nicNum = raResourceInfo_.nicSocketMap.size();
    u32 threadsNum = dstInterClientMap_.size() * nicNum + dstInterServerMap_.size() * nicNum;
    HCCL_INFO("threadsNum[%u]", threadsNum);
    HCCL_INFO(
        "CommRemoteAccess CreateLinks rank_[%u] dstInterClientMapSize[%u], nicNum[%u], dstInterServerMapSize[%u]",
        rank_, dstInterClientMap_.size(), nicNum, dstInterServerMap_.size());
    CHK_PRT_RET(
        (threadsNum == 0),
        HCCL_ERROR(
            "[Create][Links]no link to create, please check ranktable to see if"
            "device_ip is configured. nicNum[%u]",
            nicNum),
        HCCL_E_INTERNAL);
    linkThreads_.resize(threadsNum);
    threadsStatus_.resize(threadsNum);
    HCCL_INFO(
        "comm base threads info:link threads size[%llu], dst inter client map size[%llu], "
        "dst inter server map size[%llu]",
        linkThreads_.size(), dstInterClientMap_.size() * nicNum, dstInterServerMap_.size() * nicNum);
    HcclUs startut = TIME_NOW();
    // 获取当前rank作为client端时，获取所有server端的socket
    CHK_RET(CreateInterServerLinks());
    // 获取当前rank作为server端时，获取所有client端的socket
    CHK_RET(CreateInterClientLinks());

    bool check = (threadsApplyNum_ != linkThreads_.size());
    CHK_PRT_RET(
        check,
        HCCL_ERROR(
            "[Create][Links]comm apply num[%u] is not equal to link threads[%llu]", threadsApplyNum_,
            linkThreads_.size()),
        HCCL_E_INTERNAL);

    HCCL_INFO("CommRemoteAccess CreateLinks threadsApplyNum_[%u]", threadsApplyNum_);

    for (u32 index = 0; index < linkThreads_.size(); index++) {
        linkThreads_[index]->join();             // 等待线程执行完毕
        CHK_RET(hrtResetDevice(deviceLogicId_)); // 防止线程里面异常退出，在进程中reset
    }
    for (u32 index = 0; index < threadsStatus_.size(); index++) {
        CHK_PRT_RET(
            threadsStatus_[index] != 0, HCCL_ERROR("[Create][Links]execute the thread[%u] function failed", index),
            HCCL_E_PARA);
    }
    linkThreads_.clear();
    HCCL_DEBUG("rdma_rasocket Time:%lld us", DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CommRemoteAccess::CreateInterServerLinks()
{
    // 获取当前rank作为client端时，获取所有server端的socket
    u32 dstInterServerNum = dstInterServerMap_.size() * nicSocketHandle_.size();
    if (dstInterServerNum > 0) {
        std::vector<struct SocketInfoT> cliConns(dstInterServerNum);
        struct SocketInfoT* cliConn = cliConns.data();
        s32 sRet = memset_s(
            cliConn, sizeof(struct SocketInfoT) * dstInterServerNum, 0, sizeof(struct SocketInfoT) * dstInterServerNum);
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR(
                "[Create][InterServerLinks]memory set failed. return[%d]."
                "params: destMaxSize[%zu], c[%d], count[%zu]",
                sRet, sizeof(struct SocketInfoT) * dstInterServerNum, 0,
                sizeof(struct SocketInfoT) * dstInterServerNum),
            HCCL_E_MEMORY);
        // 构建socket_info_t信息，用于获取fd_socket_handle
        u32 connLoop = 0;
        for (auto iter = dstInterServerMap_.begin(); iter != dstInterServerMap_.end(); iter++) {
            sRet = memcpy_s(&cliConn[connLoop].tag[0], sizeof(cliConn[connLoop].tag) - 1, tag_.c_str(), tag_.size());
            CHK_PRT_RET(
                sRet != EOK,
                HCCL_ERROR(
                    "[Create][InterServerLinks]memcpy failed. errorno[%d],"
                    "params:destMaxSize[%zu], count[%zu]",
                    sRet, sizeof(cliConn[connLoop].tag), tag_.size()),
                HCCL_E_MEMORY);
            for (u32 idx = 0; idx < nicSocketHandle_.size(); idx++) {
                cliConn[connLoop].socketHandle = nicSocketHandle_[idx];
                cliConn[connLoop].remoteIp.addr = iter->second[idx].GetBinaryAddress().addr;
                cliConn[connLoop].remoteIp.addr6 = iter->second[idx].GetBinaryAddress().addr6;
                /* 插入建链状态的指示 */
                LinkStatus_t linkInfo;
                linkInfo.userRank = iter->first;
                linkInfo.status = SOCKET_CONNECT_NO_CONNECTION;
                linkInfo.isLinked = false;
                linkInfo.remoteIp = (iter->second)[idx];
                linkInfo.localIp = rankInfo_[rank_][idx];
                HCCL_DEBUG(
                    "CLIENT rank[%u] LocalIp[%s] RemoteIp[%s]", rank_, linkInfo.localIp.GetReadableAddress(),
                    linkInfo.remoteIp.GetReadableAddress());
                serverLinkStatus_.insert(std::make_pair(iter->second[idx], linkInfo));
                connLoop++;
            }
        }
        CHK_RET(GetRaSocket(CLIENT_ROLE_SOCKET, cliConn, dstInterServerNum));
    }
    return HCCL_SUCCESS;
}

HcclResult CommRemoteAccess::GetRaSocket(const u32 role, const struct SocketInfoT conn[], const u32 num)
{
    HCCL_INFO("get sockets para: socket role[%u], socket num[%u]", role, num);
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    s32 sockRet = 0;
    u32 gotSocketsCnt = 0;
    HCCL_INFO("In GetRaSocket, waiting for all rasockets link up...");
    u32 left = num;
    while (true) {
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            PrintErrorConnection(role, left);
            HCCL_ERROR(
                "[Get][RaSocket]in GetRaSocket, get rasocket error role[%u], rank[%u]num[%u], timeout[%lld s]", role,
                rank_, left, timeout);
            return HCCL_E_TIMEOUT;
        }
        std::vector<struct SocketInfoT> conns(left);
        struct SocketInfoT* tmpConn = conns.data();
        s32 sret = memcpy_s(
            tmpConn, sizeof(struct SocketInfoT) * left, conn + (num - left), sizeof(struct SocketInfoT) * left);
        CHK_PRT_RET(
            sret != EOK,
            HCCL_ERROR(
                "[Get][RaSocket]memcpy failed. errorno[%d], params:"
                "destMaxSize[%zu], count[%zu]",
                sret, sizeof(struct SocketInfoT) * left, sizeof(struct SocketInfoT) * left),
            HCCL_E_MEMORY);
        u32 connectedNum = 0;
        sockRet = hrtRaGetSockets(role, tmpConn, left, &connectedNum);
        if ((connectedNum == 0 && sockRet == 0) || (sockRet == SOCK_EAGAIN)) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else if (sockRet != 0) {
            PrintErrorConnection(role, num);
            HCCL_ERROR(
                "[Get][RaSocket]in GetRaSocket, get rasocket error. role[%u], rank[%u],num[%u] sockRet[%d] > 0", role,
                rank_, num, sockRet);
            return HCCL_E_TCP_CONNECT;
        } else if (connectedNum > 0) {
            u32 sockNum = abs(static_cast<s32>(connectedNum));
            left = left - sockNum;
            // 保存建链成功的socket
            HcclResult ret = DealSuccRasocket(connectedNum, role, tmpConn, sockNum);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Get][RaSocket]in GetRaSocket, save rasocket failed. role[%u], rank[%u]"
                    "num[%u] ret[%d] connectednum[%u]",
                    role, rank_, num, ret, connectedNum),
                ret);
            gotSocketsCnt += sockNum;

            if (gotSocketsCnt == num) {
                break;
            } else if (gotSocketsCnt > num) {
                HCCL_ERROR("[Get][RaSocket]total Sockets[%u], more than needed num[%u]!", gotSocketsCnt, num);
                return HCCL_E_TCP_CONNECT;
            } else {
                SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            }
        }
    }
    HCCL_INFO("In CommRemoteAccess, all rasockets linked up ");
    return HCCL_SUCCESS;
}

HcclResult CommRemoteAccess::CreateInterClientLinks()
{
    // 获取当前rank作为server端时，获取所有client端的socket
    u32 dstInterClientNum = dstInterClientMap_.size() * nicSocketHandle_.size();
    HCCL_INFO("dstInterClientNum[%u]", dstInterClientNum);
    if (dstInterClientNum > 0) {
        std::vector<struct SocketInfoT> srvConns(dstInterClientNum);
        struct SocketInfoT* srvConn = srvConns.data();
        s32 sRet = memset_s(
            srvConn, sizeof(struct SocketInfoT) * dstInterClientNum, 0, sizeof(struct SocketInfoT) * dstInterClientNum);
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR(
                "[Create][InterClientLinks]memory set failed. return[%d]."
                "params: destMaxSize[%zu], c[%d], count[%zu]",
                sRet, sizeof(struct SocketInfoT) * dstInterClientNum, 0,
                sizeof(struct SocketInfoT) * dstInterClientNum),
            HCCL_E_MEMORY);
        u32 loop = 0;
        for (u32 interIndex = 0; interIndex < dstInterClientMap_.size(); interIndex++) {
            sRet = memcpy_s(&srvConn[loop].tag[0], sizeof(srvConn[loop].tag) - 1, tag_.c_str(), tag_.size());
            CHK_PRT_RET(
                sRet != EOK,
                HCCL_ERROR(
                    "[Create][InterClientLinks]memcpy failed. errorno[%d], params:"
                    "destMaxSize[%zu],count[%zu]",
                    sRet, sizeof(srvConn[loop].tag), tag_.size()),
                HCCL_E_MEMORY);
            for (u32 idx = 0; idx < nicSocketHandle_.size(); idx++) {
                srvConn[loop].socketHandle = nicSocketHandle_[idx];
                loop++;
            }
        }
        /* 插入建链状态的指示 */
        for (auto iter = dstInterClientMap_.begin(); iter != dstInterClientMap_.end(); iter++) {
            LinkStatus_t linkInfo;
            linkInfo.userRank = iter->first;
            linkInfo.status = SOCKET_CONNECT_NO_CONNECTION;
            linkInfo.isLinked = false;
            for (u32 idx = 0; idx < (iter->second).size(); idx++) {
                linkInfo.remoteIp = (iter->second)[idx];
                linkInfo.localIp = rankInfo_[rank_][idx];
                HCCL_DEBUG(
                    "CreateInterClientLinks SERVER rank[%u] LocalIp[%s] RemoteIp[%s]", rank_,
                    linkInfo.localIp.GetReadableAddress(), linkInfo.remoteIp.GetReadableAddress());
                clientLinkStatus_.insert(std::make_pair((iter->second)[idx], linkInfo));
            }
        }
        CHK_RET(GetRaSocket(SERVER_ROLE_SOCKET, srvConn, dstInterClientNum));
    }
    return HCCL_SUCCESS;
}

void CommRemoteAccess::PrintErrorConnection(const u32 role, const u32 num)
{
    RPT_INNER_ERR_PRT("remote op nic connect failed, please ensure that collective communication execution status "
                      "of each device is consistent(include network TLS configuration)");

    HCCL_ERROR("Some NPUs get socket timeout, the details are as follows:");
    HCCL_ERROR("   _________________________LINK_ERROR_INFO___________________________");
    HCCL_ERROR("   | comm error, device[%d] num[%u] ", deviceLogicId_, num);
    HCCL_ERROR("   |  dest_ip(user_rank)  |   dest_port   |  src_ip(user_rank)   |   src_port   |   MyRole   "
               "|   Status   |");
    HCCL_ERROR("   |--------------------|--------------------|----------|------------|-----------------"
               "|-----------------|");

    /* 第一行打印deviceIds */
    HcclResult ret = HCCL_SUCCESS;
    if (role == SERVER_ROLE_SOCKET) {
        ret = PrintErrorConnectionInfo(clientLinkStatus_, role);
    } else if (role == CLIENT_ROLE_SOCKET) {
        ret = PrintErrorConnectionInfo(serverLinkStatus_, role);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Print][ErrorConnection]PrintErrorConnectionInfo fail. ret[%d] role[%u]", ret, role);
        return;
    }
    HCCL_ERROR("   ___________________________________________________________________  ");
    HCCL_ERROR("the connection failure between this device and target device may be due to the following reasons:");
    HCCL_ERROR("1. the connection between this device and the target device is abnormal.");
    HCCL_ERROR("2. an exception occurred at the target devices.");
    HCCL_ERROR("3. the time difference between the execution of hcom on this device and the target device exceeds the "
               "timeout threshold, make sure this by keyword [Entry-].");
    HCCL_ERROR("4. the behavior of executing the calculation graph on this device and the target device is "
               "inconsistent. ");
    HCCL_ERROR("5. Now you can freely specify a port for listening and connecting. If an invalid port is chosen, "
               "it may result in failed listening and connection timeouts");
    return;
}

#define TRANSFORM_RASOCKET_STATUS(status, stringStatus) \
    do {                                                \
        switch (status) {                               \
            default:                                    \
            case SOCKET_CONNECT_NO_CONNECTION:          \
                stringStatus = "no connect";            \
                break;                                  \
            case SOCKET_CONNECT_OK:                     \
                stringStatus = "connected";             \
                break;                                  \
            case SOCKET_CONNECT_TIMEOUT:                \
                stringStatus = "connecting";            \
                break;                                  \
        }                                               \
    } while (0)

HcclResult
CommRemoteAccess::PrintErrorConnectionInfo(const std::map<HcclIpAddress, LinkStatus_t>& linkStatusMap, u32 role)
{
    std::string sRole;
    switch (role) {
        case SERVER_ROLE_SOCKET:
            sRole = " server ";
            break;
        case CLIENT_ROLE_SOCKET:
            sRole = " client ";
            break;
        default:
            sRole = "   NA   ";
            break;
    }
    for (auto iter = linkStatusMap.begin(); iter != linkStatusMap.end(); iter++) {
        if (!iter->second.isLinked) {
            std::string connectStatus = "";
            TRANSFORM_RASOCKET_STATUS(iter->second.status, connectStatus);
            HCCL_ERROR(
                "   |  %s(%u)   |  %u  |   %s(%u)   |  %u  | %s | %s |  ", iter->second.remoteIp.GetReadableAddress(),
                iter->second.userRank, HETEROG_CCL_PORT, iter->second.localIp.GetReadableAddress(), rank_,
                HETEROG_CCL_PORT, sRole.c_str(), connectStatus.c_str());
        }
    }
    return HCCL_SUCCESS;
}

// 根据IP信息，获得RANK信息
HcclResult CommRemoteAccess::GetDstRank(
    std::map<u32, std::vector<HcclIpAddress>>& dstMap, const HcclIpAddress& dstIp, u32& dstRank)
{
    for (auto it = dstMap.begin(); it != dstMap.end(); it++) {
        for (u32 idx = 0; idx < it->second.size(); idx++) {
            if (it->second[idx] == dstIp) {
                dstRank = it->first;
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("[Get][DstRank]can't find ip[%s] in dst map", dstIp.GetReadableAddress());
    return HCCL_E_NOT_FOUND;
}

HcclResult CommRemoteAccess::CreateInterThread(const u32 role, const SocketInfoT& socketInfo)
{
    // 线程命名，CommRemoteTerL代表CommRemote Inter Link
    std::string threadStr = "RemoteThrd_" + std::to_string(threadsApplyNum_);
    HcclIpAddress nicIp;
    u32 dstRank = 0;
    threadsStatus_[threadsApplyNum_] = 1;
    CHK_RET(GetNicByHandle(socketInfo.socketHandle, nicIp));
    HcclInAddr temp;
    temp.addr = socketInfo.remoteIp.addr;
    temp.addr6 = socketInfo.remoteIp.addr6;
    HcclIpAddress remoteIP(rankInfo_[rank_][0].GetFamily(), temp);
    CHK_PRT_RET(remoteIP.IsInvalid(), HCCL_ERROR("ip is invalid."), HCCL_E_PARA);
    workflowMode_ = GetWorkflowMode();
    if (role == SERVER_ROLE_SOCKET) {
        CHK_RET(GetDstRank(dstInterClientMap_, remoteIP, dstRank));
        linkThreads_[threadsApplyNum_].reset(new (std::nothrow) std::thread(
            &CommRemoteAccess::InitDestTransport, this, hrtErrMGetErrorContext(), role, nicIp, dstRank, threadStr,
            socketInfo.fdHandle, &threadsStatus_[threadsApplyNum_]));
    }

    if (role == CLIENT_ROLE_SOCKET) {
        CHK_RET(GetDstRank(dstInterServerMap_, remoteIP, dstRank));
        linkThreads_[threadsApplyNum_].reset(new (std::nothrow) std::thread(
            &CommRemoteAccess::InitDestTransport, this, hrtErrMGetErrorContext(), role, nicIp, dstRank, threadStr,
            socketInfo.fdHandle, &threadsStatus_[threadsApplyNum_]));
    }
    bool check = !linkThreads_[threadsApplyNum_];
    CHK_PRT_RET(
        check, HCCL_ERROR("[Create][InterThread]link threads[%u] reset failed.", threadsApplyNum_), HCCL_E_INTERNAL);
    threadsApplyNum_++;
    return HCCL_SUCCESS;
}

HcclResult
CommRemoteAccess::DealSuccRasocket(s32 sockRet, const u32 role, const struct SocketInfoT tmpConn[], const u32 num)
{
    HCCL_DEBUG("CommRemoteAccess DealSuccRasocketNum[%u]", num);
    u32 socketsCnt = static_cast<u32>(sockRet);
    u32 loop = 0;
    for (u32 i = 0; i < num; i++) {
        HcclInAddr temp;
        temp.addr = tmpConn[i].remoteIp.addr;
        temp.addr6 = tmpConn[i].remoteIp.addr6;
        HcclIpAddress remoteIP(rankInfo_[rank_][0].GetFamily(), temp);
        CHK_PRT_RET(remoteIP.IsInvalid(), HCCL_ERROR("ip is invalid."), HCCL_E_PARA);
        if (tmpConn[i].status == SOCKET_CONNECT_OK) {
            raSockets_.push_back(tmpConn[i]);
            CHK_RET(CreateInterThread(role, tmpConn[i]));
            // 建链成功的在本地标志建链成功
            serverLinkStatus_[remoteIP].isLinked = true;
            loop++;
        }
        if (tmpConn[i].status != SOCKET_CONNECT_NO_CONNECTION) {
            clientLinkStatus_[remoteIP].status = tmpConn[i].status;
        }
    }

    if (socketsCnt != loop) {
        HCCL_ERROR(
            "[Deal][SuccRasocket]current socketsCnt[%u], not equal to actual connect number[%u]!", socketsCnt, loop);
        return HCCL_E_TCP_CONNECT;
    }
    return HCCL_SUCCESS;
}

HcclResult CommRemoteAccess::InitDestTransport(
    const ErrContext& error_context, u32 role, const HcclIpAddress& nicIp, const u32 dstRank,
    const std::string& threadStr, FdHandle socketFdHandle, u32* getThreadStatus)
{
    hrtErrMSetErrorContext(error_context);

    // 给当前线程添加名字
    SetThreadName(threadStr);
    CHK_RET(hrtSetDevice(deviceLogicId_));
    SetWorkflowMode(workflowMode_);

    RemoteAccessPara accessPara;
    CHK_RET(SetAccessPara(role, nicIp, dstRank, socketFdHandle, accessPara));
    HCCL_INFO(
        "[InitDestTransport para]local_rank[%u]-localIpAddr[%s],dst rank[%u]-remote_rank[%u]-remote_ip_addr[%s], "
        "role[%u]",
        rank_, rankInfo_[rank_][0].GetReadableAddress(), dstRank, dstRank, rankInfo_[dstRank][0].GetReadableAddress(),
        role);

    std::shared_ptr<TransportRemoteAccess> transportPtr;
    transportPtr.reset(new (std::nothrow) TransportRemoteAccess(
        tag_, dispatcher_, notifyPool_, accessPara, addrInfos_, deviceLogicId_));
    CHK_PRT_RET(!transportPtr, HCCL_ERROR("[Init][DestTransport]InitDestTransport failed"), HCCL_E_PTR);

    std::unique_lock<std::mutex> remoteTransportMapLock(remoteTransportMapLock_);
    remoteTransportMap_.insert(std::make_pair(dstRank, transportPtr));
    remoteTransportMapLock.unlock();

    CHK_RET(transportPtr->Init());
    *getThreadStatus = 0;
    return HCCL_SUCCESS;
}

// 根据socket handle，获取本device所使用的网口IP
HcclResult CommRemoteAccess::GetNicByHandle(const SocketHandle socketHandle, HcclIpAddress& nicIp)
{
    for (auto it = raResourceInfo_.nicSocketMap.begin(); it != raResourceInfo_.nicSocketMap.end(); it++) {
        if (it->second.nicSocketHandle == socketHandle) {
            nicIp = it->first;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[Get][NicByHandle]current socket handle error");
    return HCCL_E_NOT_FOUND;
}

HcclResult CommRemoteAccess::SetAccessPara(
    u32 role, const HcclIpAddress& nicIp, u32 dstRank, FdHandle socketFdhandle, RemoteAccessPara& accessPara)
{
    accessPara.role = role;
    accessPara.localIp = nicIp;
    accessPara.localRank = rank_;
    accessPara.remoteRank = dstRank;
    accessPara.socketFdhandle = socketFdhandle;
    accessPara.raResourceInfo = raResourceInfo_;

    // 获取 nicSocketHandle
    auto itSocket = raResourceInfo_.nicSocketMap.find(nicIp);
    if (itSocket == raResourceInfo_.nicSocketMap.end()) {
        HCCL_ERROR(
            "[Set][AccessPara]In get nic handle, can not find socket handle, handle size[%u], local ip[%s]",
            raResourceInfo_.nicSocketMap.size(), nicIp.GetReadableAddress());
        return HCCL_E_PARA;
    }
    accessPara.nicSocketHandle = itSocket->second.nicSocketHandle;
    CHK_PTR_NULL(accessPara.nicSocketHandle);

    accessPara.nicRdmaHandle = itSocket->second.nicRdmaHandle;
    CHK_PTR_NULL(accessPara.nicRdmaHandle);
    return HCCL_SUCCESS;
}
} // namespace hccl
