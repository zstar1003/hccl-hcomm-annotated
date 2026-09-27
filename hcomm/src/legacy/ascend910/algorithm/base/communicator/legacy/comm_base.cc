/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_base.h"
#include <arpa/inet.h>
#include <securec.h>

#include "externalinput_pub.h"
#include "hccl_common.h"
#include "device_capacity.h"
#include "p2p_mgmt_pub.h"
#include "rank_consistentcy_checker.h"

namespace hccl {
constexpr s32 HCCL_DEFAULT_INITIAL_VALUE = -1;
CommBase::CommBase(
    const std::string& collectiveId, const u32 userRank, const u32 userRankSize, const u32 rank, const u32 rankSize,
    const std::vector<RankInfo> paraVector, const TopoType topoFlag, const HcclDispatcher dispatcher,
    const std::unique_ptr<NotifyPool>& notifyPool, std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap,
    const IntraExchanger& exchanger, const DeviceMem& inputMem, const DeviceMem& outputMem, const bool isUsedRdmaLevel0,
    const std::string& tag, const NICDeployment nicDeployInner, bool isAlltoAllCommMesh, const bool useOneDoorbell,
    const bool isAicpuModeEn, const u32 rankRoot, const bool isHaveCpuRank, const bool useSuperPodMode,
    DeviceMem expMem)
    : linkDummy_(nullptr),
      collectiveId_(collectiveId),
      userRank_(userRank),
      userRankSize_(userRankSize),
      rank_(rank),
      rankSize_(rankSize),
      paraVector_(paraVector),
      transportType_(rankSize, TransportType::TRANS_TYPE_RESERVED),
      deviceLogicId_(HCCL_DEFAULT_INITIAL_VALUE),
      devicePhyId_(INVALID_UINT),
      topoFlag_(topoFlag),
      tag_(tag),
      transportInfo_(rankSize),
      rankMap_(userRankSize, INVALID_VALUE_RANKID),
      userRankMap_(rankSize, INVALID_VALUE_RANKID),
      dispatcher_(dispatcher),
      notifyPool_(notifyPool),
      netDevCtxMap_(netDevCtxMap),
      exchanger_(exchanger),
      inputMem_(inputMem),
      outputMem_(outputMem),
      isUsedRdmaLevel0_(isUsedRdmaLevel0),
      dstInterServerMap_(),
      dstInterClientMap_(),
      dstIntraServerVec_(),
      dstIntraClientVec_(),
      linkThreads_(),
      threadsRapplyNum_(0),
      shmDev_(0),
      isAlltoAllCommMesh_(isAlltoAllCommMesh),
      nicDeployInner_(nicDeployInner),
      isNeedHeterogP2P_(false),
      useOneDoorbell_(useOneDoorbell),
      isAicpuModeEn_(isAicpuModeEn),
      subUserRankRoot_(rankRoot),
      isHaveCpuRank_(isHaveCpuRank),
      useSuperPodMode_(useSuperPodMode),
      expMem_(expMem)
{}

CommBase::~CommBase() { (void)DeInit(); }

HcclResult CommBase::DeInit()
{
    for (u32 index = 0; index < linkThreads_.size(); index++) {
        if (linkThreads_[index]) {
            if (linkThreads_[index]->joinable()) {
                HCCL_DEBUG("Joining Link Thread[%u]", index);
                linkThreads_[index]->join(); // 等待线程执行后释放资源
            }

            HcclResult ret = hrtResetDevice(deviceLogicId_); // 防止线程里面异常退出，在进程中reset
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[CommBase][DeInit]comm base reset device[%d] failed", deviceLogicId_),
                ret);
        }
    }
    linkThreads_.clear();

    for (u32 i = 0; i < transportInfo_.size(); i++) {
        if (transportInfo_[i]) { // 使用对应类型的port销毁
            CHK_RET(transportInfo_[i]->DeInit());
        }
    }

    return HCCL_SUCCESS;
}

HcclResult CommBase::Init()
{
    // 获取rank->userrank以及userrank->rank的映射关系
    CHK_RET(SetRankMap());

    if (!IsGeneralServer()) {
        // 获取当前线程操作的设备ID
        CHK_RET(hrtGetDevice(&deviceLogicId_));
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId_), devicePhyId_));
    }

    intraSocketsMap_.insert(exchanger_.socketsMap.begin(), exchanger_.socketsMap.end());

    // 创建当前rank与其他rank之间的link(RDMA异步、TCP)
    CHK_RET(CreateLinks());

    // 校验当前rank与dst rank建链的链路有效性
    CHK_RET(CheckLinks());

    // task多线程并行下发，根据当前transport创建vtransport信息
    CHK_RET(CreateVirturalTransport());

    return HCCL_SUCCESS;
}

std::shared_ptr<Transport>& CommBase::GetTransportByRank(const u32 dstRank)
{
    if (transportInfo_.size() <= dstRank) {
        HCCL_ERROR("[Get][TransportByRank]dstRank[%u] is bigger than link size[%llu]", dstRank, transportInfo_.size());
        return linkDummy_;
    }

    return transportInfo_[dstRank];
}

HcclResult CommBase::GetRankByUserRank(const u32 userRank, u32& rank) const
{
    if (rankMap_.size() > userRank) {
        rank = rankMap_[userRank];
        if (rank == INVALID_VALUE_RANKID) {
            HCCL_INFO("This userRank[%u] is not in this sub communication. ", userRank);
            return HCCL_E_NOT_FOUND;
        }
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[Get][RankByUserRank]This userRank[%u] is invalid. ", userRank);
    rank = INVALID_VALUE_RANKID;
    return HCCL_E_PARA;
}

HcclResult CommBase::GetUserRankByRank(const u32 rank, u32& userRank) const
{
    if (userRankMap_.size() > rank) {
        if (userRankMap_[rank] == INVALID_VALUE_RANKID) {
            HCCL_INFO("This rank[%u] is not in this sub communication.", rank);
            userRank = INVALID_VALUE_RANKID;
            return HCCL_E_NOT_FOUND;
        }

        userRank = userRankMap_[rank];
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[Get][UserRankByRank]This rank[%u] is invalid.", rank);
    userRank = INVALID_VALUE_RANKID;
    return HCCL_E_PARA;
}

HcclResult CommBase::CreateLinks()
{
    HCCL_DEBUG("[CreateLinks] [comm_base] rankSize_[%u]", rankSize_);
    if (rankSize_ == HCCL_RANK_SIZE_EQ_ONE) {
        HCCL_INFO("comm base needn't to create links, rankSize_[%u].", rankSize_);
        return HCCL_SUCCESS;
    }

    CHK_RET(CalcLink());
    u32 threadsNum
        = dstInterClientMap_.size() + dstIntraClientVec_.size() + dstInterServerMap_.size() + dstIntraServerVec_.size();
    CHK_PRT_RET(
        (threadsNum == 0), HCCL_ERROR("[Create][Links]no link to create, threadsNum[%u]", threadsNum), HCCL_E_INTERNAL);

    linkThreads_.resize(threadsNum);
    HCCL_INFO(
        "comm base threads info:link threads size[%llu], dst inter client map size[%llu], "
        "dst intra client vec size[%llu], dst inter server map size[%llu], dst intra server vec size[%llu]",
        linkThreads_.size(), dstInterClientMap_.size(), dstIntraClientVec_.size(), dstInterServerMap_.size(),
        dstIntraServerVec_.size());

    CHK_RET(CreateExchangerNetwork());

    CHK_RET(CreateIntraLinks());

    CHK_RET(CreateInterLinks());

    bool check = (threadsRapplyNum_ != linkThreads_.size());
    CHK_PRT_RET(
        check,
        HCCL_ERROR(
            "[Create][Links]comm base rapply num[%u] is not equal to link threads[%llu]", threadsRapplyNum_,
            linkThreads_.size()),
        HCCL_E_INTERNAL);

    for (u32 index = 0; index < linkThreads_.size(); index++) {
        if (linkThreads_[index] == nullptr) {
            continue;
        }
        if (linkThreads_[index]->joinable()) {
            HCCL_DEBUG("Joining Link Thread[%u]", index);
            linkThreads_[index]->join(); // 等待线程执行完毕
        }
        if (!IsGeneralServer()) {
            CHK_RET(hrtResetDevice(deviceLogicId_)); // 防止线程里面异常退出，在进程中reset
        }
    }
    linkThreads_.clear();
    // 建链结束立即释放socket资源（添加判断host网卡走的是TCP就不释放资源）
    if (pyhIdResourseSockets_.size()) {
        for (auto& iter : pyhIdResourseSockets_) {
            iter.second->DestroySockets(tag_);
        }
    } else if (!GetExternalInputHcclIsTcpMode() && interSocketManager_ != nullptr) {
        // 建链结束，关闭socket
        interSocketManager_->DestroySockets(tag_);
    }
    return HCCL_SUCCESS;
}

HcclResult CommBase::CalcLink() { return HCCL_SUCCESS; }

u32 CommBase::GetSocketsPerLink() { return 1; }

bool CommBase::NeedDataReceivedAck() { return false; }

// 获取rank间的link type
HcclResult CommBase::SetTransportType(const u32 dstRank)
{
    LinkTypeInServer linkType = LinkTypeInServer::RESERVED_LINK_TYPE;

    // 适配910_93的RDMA+SIO ring,创建RDMA类型下的SIO连接
    if (linkType == LinkTypeInServer::SIO_TYPE && paraVector_[rank_].deviceType == DevType::DEV_TYPE_910_93) {
        transportType_[dstRank] = TransportType::TRANS_TYPE_P2P;
    } else if (paraVector_[rank_].serverId == paraVector_[dstRank].serverId) { // 判断是否在同一个server
        // Server内判断是否使用rdma
        if (isUsedRdmaLevel0_ || isAlltoAllCommMesh_
            || (paraVector_[rank_].deviceType != DevType::DEV_TYPE_310P3
                && (paraVector_[rank_].devicePhyId / HCCL_AISERVER_DEVICE_NUM
                    != paraVector_[dstRank].devicePhyId / HCCL_AISERVER_DEVICE_NUM))) {
            transportType_[dstRank] = TransportType::TRANS_TYPE_IBV_EXP;
        } else {
            transportType_[dstRank] = TransportType::TRANS_TYPE_P2P;
        }
    } else { // server间
        if (IsSupportInterHccs(dstRank)) {
            // 超节点内节点间走HCCS通信
            transportType_[dstRank] = TransportType::TRANS_TYPE_P2P;
        } else {
            transportType_[dstRank] = TransportType::TRANS_TYPE_IBV_EXP;
        }
    }

    HCCL_INFO("SetTransportType: dstRank[%u] transport_type[%d]", dstRank, transportType_[dstRank]);
    return HCCL_SUCCESS;
}

HcclResult CommBase::RunTemplateAlg(const std::unique_ptr<AlgTemplateBase>& tempAlg)
{
    HcclResult ret = tempAlg->RunAsync(Rank(), RankSize(), transportInfo_);
    CHK_PRT_RET(ret == HCCL_E_AGAIN, HCCL_WARNING("[Run][AlgTemplateBase]group has been destroyed. Break!"), ret);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Run][AlgTemplateBase]comm base run tempAlg "
            "rank[%u] rank size[%u] failed",
            rank_, rankSize_),
        ret);
    return HCCL_SUCCESS;
}

HcclResult CommBase::RunTemplateAlgStaged(const std::unique_ptr<AlgTemplateBase>& tempAlg, const RunStage& stage)
{
    HcclResult ret = tempAlg->RunAsyncStaged(Rank(), RankSize(), transportInfo_, stage);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Run][RunTemplateAlgStaged]comm base run tempAlg staged "
            "rank[%u] rank size[%u] failed",
            rank_, rankSize_),
        ret);
    return HCCL_SUCCESS;
}

HcclResult CommBase::SetRankMap()
{
    // 参数有效性校验
    if ((userRankSize_ <= userRank_) || (rankSize_ <= rank_)) {
        HCCL_ERROR(
            "[Set][RankMap]invalid:userRankSize_[%u] userRank_[%u] rankSize_[%u] rank_[%u].", userRankSize_, userRank_,
            rankSize_, rank_);
        return HCCL_E_PARA;
    }

    for (u32 index = 0; index < rankSize_; index++) {
        userRankMap_[index] = paraVector_[index].userRank;

        if (userRankSize_ > userRankMap_[index]) {
            rankMap_[userRankMap_[index]] = index;
        }
        HCCL_INFO("userRankMap: [%u] -> [%u]", index, paraVector_[index].worldRank);
    }

    return HCCL_SUCCESS;
}

HcclResult CommBase::CheckLinks() const
{
    for (u32 index = 0; index < transportType_.size(); index++) {
        bool check = (transportInfo_.size() <= index);
        CHK_PRT_RET(
            check, HCCL_ERROR("[Check][Links]index[%u] is bigger than link size[%llu]", index, transportInfo_.size()),
            HCCL_E_INTERNAL);
        if ((transportType_[index] != TransportType::TRANS_TYPE_RESERVED) && !transportInfo_[index]) {
            HCCL_ERROR(
                "[Check][Links]there is no effective link(type[%d]) between rank[%u] and dst rank[%u]!",
                transportType_[index], rank_, index);
            return HCCL_E_NOT_FOUND;
        }
    }

    return HCCL_SUCCESS;
}

// 只有节点内，采用虚拟网卡时，才会进入该函数，有且仅有一个nic_ip
HcclResult CommBase::CreateIntraLinks()
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;

    auto socketsMap = intraSocketsMap_;
    HCCL_DEBUG("[Create][IntraLinks] dstIntraServerVec size[%u].", dstIntraServerVec_.size());
    for (auto& rank : dstIntraServerVec_) {
        HCCL_DEBUG("[Create][IntraLinks] localrank[%u] remoterank[%u].", userRank_, rank);
        // 与当前Inter的Socket不同, 在Intra Socket创建时, 使用的userRank, 所以这里需要使用 userRank 为 key
        auto item = socketsMap.find(paraVector_[rank].userRank);
        if (item != socketsMap.end()) {
            ret = CreateIntraThread(CLIENT_ROLE_SOCKET, rank, item->second);
        } else {
            HCCL_INFO("[Create][IntraLinks] remoterank[%u] socket item not find.", rank);
            // 异构场景下, 当前上层 CreateCommP2PAsync 并不会创建 IntraExchanger, 后继的
            // TransportHeterogP2P 场景使用原有的 Socket 逻辑, 所以在这里没有找到 socket item 时,
            // 使用一个空的 sockets 作为参数, 创建后继处理线程.
            std::vector<std::shared_ptr<HcclSocket>> sockets;
            ret = CreateIntraThread(CLIENT_ROLE_SOCKET, rank, sockets);
        }
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[Create][IntraLinks] create intra thread failed, socket role is client"),
            ret);
    }

    HCCL_DEBUG("[Create][IntraLinks] dstIntraClientVec size[%u].", dstIntraClientVec_.size());
    for (auto& rank : dstIntraClientVec_) {
        HCCL_DEBUG("[Create][IntraLinks] localrank[%u] remoterank[%u].", userRank_, rank);
        auto item = socketsMap.find(paraVector_[rank].userRank);
        if (item != socketsMap.end()) {
            ret = CreateIntraThread(SERVER_ROLE_SOCKET, rank, item->second);
        } else {
            HCCL_INFO("[Create][IntraLinks] remoterank[%u] socket item not find.", rank);
            // 异构场景下, 当前上层 CreateCommP2PAsync 并不会创建 IntraExchanger, 后继的
            // TransportHeterogP2P 场景使用原有的 Socket 逻辑, 所以在这里没有找到 socket item 时,
            // 使用一个空的 sockets 作为参数, 创建后继处理线程.
            std::vector<std::shared_ptr<HcclSocket>> sockets;
            ret = CreateIntraThread(SERVER_ROLE_SOCKET, rank, sockets);
        }
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[Create][IntraLinks] create intra thread failed, socket role is server"),
            ret);
    }

    HCCL_DEBUG("[Create][IntraLinks] create intra link used time:%lld us.", DURATION_US(TIME_NOW() - startut));
    return ret;
}

HcclResult
CommBase::CreateIntraThread(const u32 role, u32 dstRank, const std::vector<std::shared_ptr<HcclSocket>>& sockets)
{
    if (threadsRapplyNum_ >= linkThreads_.size()) {
        HCCL_ERROR(
            "[Create][InterThread] threadsRapplyNum_[%u] is bigger than link threads size[%llu] ", threadsRapplyNum_,
            linkThreads_.size());
        return HCCL_E_INTERNAL;
    }

    // 线程命名，TraL_ 代表Intra Link
    std::string threadStr = "HcclTraL_" + std::to_string(threadsRapplyNum_);

    // 创建新线程前更新一下最新的workflowMode
    workflowMode_ = GetWorkflowMode();
    if (role == SERVER_ROLE_SOCKET) {
        linkThreads_[threadsRapplyNum_].reset(new (std::nothrow) std::thread(
            &CommBase::CreateDestLink, this, hrtErrMGetErrorContextPub(), MachineType::MACHINE_SERVER_TYPE,
            paraVector_[rank_].serverId, dstRank, threadStr, sockets));
    }

    HCCL_DEBUG("[CommBase][CreateIntraThread]role is %u", role);
    if (role == CLIENT_ROLE_SOCKET) {
        linkThreads_[threadsRapplyNum_].reset(new (std::nothrow) std::thread(
            &CommBase::CreateDestLink, this, hrtErrMGetErrorContextPub(), MachineType::MACHINE_CLIENT_TYPE,
            paraVector_[rank_].serverId, dstRank, threadStr, sockets));
    }

    if (!linkThreads_[threadsRapplyNum_]) {
        HCCL_ERROR("[Create][IntraThread] link threads[%u] reset failed.", threadsRapplyNum_);
        return HCCL_E_INTERNAL;
    }
    threadsRapplyNum_++;

    HCCL_DEBUG(
        "[Create][IntraThread] role[%u], dstRank[%u], sockets size[%u], threadsRapplyNum[%u]", role, dstRank,
        sockets.size(), threadsRapplyNum_);
    return HCCL_SUCCESS;
}

void CommBase::PrintCreateInterLinksInfo()
{
    HCCL_RUN_INFO(
        "[PrintCreateInterLinksInfo] dstInterServerMap size[%llu], dstInterClientMap size[%llu]",
        dstInterServerMap_.size(), dstInterClientMap_.size());

    // 维护建链输出的信息
    std::string outLogInfo = "";
    for (auto iter = dstInterServerMap_.begin(); iter != dstInterServerMap_.end(); iter++) {
        outLogInfo.append(std::to_string(paraVector_[iter->first].userRank));
        outLogInfo.append("/");
        outLogInfo.append(paraVector_[iter->first].serverId);
        outLogInfo.append("/");
        outLogInfo.append(std::to_string(paraVector_[iter->first].devicePhyId));
        outLogInfo.append("; ");
    }

    for (auto iter = dstInterClientMap_.begin(); iter != dstInterClientMap_.end(); iter++) {
        outLogInfo.append(std::to_string(paraVector_[iter->first].userRank));
        outLogInfo.append("/");
        outLogInfo.append(paraVector_[iter->first].serverId);
        outLogInfo.append("/");
        outLogInfo.append(std::to_string(paraVector_[iter->first].devicePhyId));
        outLogInfo.append("; ");
    }

    HCCL_RUN_INFO(
        "serverInterConnectInfo:tag[%s], userRank/serverIp/devicePhyId:[%u/%s/%d], connectRankInfo[%s]", tag_.c_str(),
        userRank_, paraVector_[rank_].serverId.c_str(), paraVector_[rank_].devicePhyId, outLogInfo.c_str());
}

HcclResult CommBase::CreateInterLinks()
{
    interSocketManager_.reset(new (std::nothrow)
                                  HcclSocketManager(nicDeployInner_, deviceLogicId_, devicePhyId_, userRank_));
    CHK_PTR_NULL(interSocketManager_);

    if (dstInterServerMap_.size() + dstInterClientMap_.size() == 0) {
        HCCL_DEBUG("[Create][InterLinks] do not need create links.");
        return HCCL_SUCCESS;
    }

    PrintCreateInterLinksInfo();

    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;
    std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> serverSocketsMap;
    std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> clientSocketsMap;
    ret = interSocketManager_->CreateSockets(
        tag_, true, netDevCtxMap_[paraVector_[rank_].nicIp[0]], dstInterServerMap_, dstInterClientMap_,
        serverSocketsMap, clientSocketsMap);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Create][InterLinks] socket manager create connections failed, ret[%u]", ret),
        ret);

    for (auto& sockets : clientSocketsMap) {
        ret = CreateInterThread(CLIENT_ROLE_SOCKET, sockets.first, sockets.second);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Create][InterLinks] create inter thread failed, socket role[CLIENT_ROLE_SOCKET] "), ret);
    }
    HCCL_DEBUG("[CommBase][CreateInterLinks]create inter thread success");
    for (auto& sockets : serverSocketsMap) {
        ret = CreateInterThread(SERVER_ROLE_SOCKET, sockets.first, sockets.second);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Create][InterLinks] create inter thread failed, socket role[SERVER_ROLE_SOCKET] "), ret);
    }

    HCCL_DEBUG("[Create][InterLinks] create inter link used time:%lld us", DURATION_US(TIME_NOW() - startut));
    return ret;
}

HcclResult
CommBase::CreateInterThread(const u32 role, u32 dstRank, const std::vector<std::shared_ptr<HcclSocket>>& sockets)
{
    if (sockets.empty()) {
        HCCL_ERROR("[Create][InterThread] create inter link failed, rank's sockets is empty");
        return HCCL_E_INTERNAL;
    }

    if (threadsRapplyNum_ >= linkThreads_.size()) {
        HCCL_ERROR(
            "[Create][InterThread] threadsRapplyNum_[%u] is bigger than link threads size[%llu] ", threadsRapplyNum_,
            linkThreads_.size());
        return HCCL_E_INTERNAL;
    }

    // 线程命名，TerL代表Inter Link
    std::string threadStr = "HcclTerL_" + std::to_string(threadsRapplyNum_);

    // 创建新线程前更新一下最新的workflowMode
    workflowMode_ = GetWorkflowMode();
    if (role == SERVER_ROLE_SOCKET) {
        linkThreads_[threadsRapplyNum_].reset(new (std::nothrow) std::thread(
            &CommBase::CreateDestLink, this, hrtErrMGetErrorContextPub(), MachineType::MACHINE_SERVER_TYPE,
            paraVector_[rank_].serverId, dstRank, threadStr, sockets));
    }

    if (role == CLIENT_ROLE_SOCKET) {
        linkThreads_[threadsRapplyNum_].reset(new (std::nothrow) std::thread(
            &CommBase::CreateDestLink, this, hrtErrMGetErrorContextPub(), MachineType::MACHINE_CLIENT_TYPE,
            paraVector_[rank_].serverId, dstRank, threadStr, sockets));
    }

    if (!linkThreads_[threadsRapplyNum_]) {
        HCCL_ERROR("[Create][InterThread] link threads[%u] reset failed.", threadsRapplyNum_);
        return HCCL_E_INTERNAL;
    }
    threadsRapplyNum_++;

    HCCL_DEBUG(
        "[Create][InterThread] role[%u], dstRank[%u], sockets size[%u], threadsRapplyNum[%u]", role, dstRank,
        sockets.size(), threadsRapplyNum_);

    return HCCL_SUCCESS;
}

u32 CommBase::GetInterRemotePort(s32 devicePhyId, u32 dstUserRank)
{
    if (isUseRankPort_ && dstUserRank < ranksPort_.size() && ranksPort_[dstUserRank] != HCCL_INVALID_PORT) {
        HCCL_INFO("[GetInterRemotePort] port[%u] from ranks port", ranksPort_[dstUserRank]);
        return ranksPort_[dstUserRank];
    } else if (!isUseRankPort_ && !Is310PDevice()) {
        HCCL_INFO("[GetInterRemotePort] port[%u]", HETEROG_CCL_PORT);
        return HETEROG_CCL_PORT;
    } else if (GetExternalInputHcclIfBasePort() == HCCL_INVALID_PORT) {
        return (devicePhyId + HOST_PARA_BASE_PORT);
    } else {
        return (devicePhyId + GetExternalInputHcclIfBasePort() + HCCL_AISERVER_DEVICE_NUM);
    }
}

HcclResult CommBase::CalcLinksNum(const MachineType machineType, const u32 dstRank)
{
    bool check = (paraVector_.size() <= dstRank) || (paraVector_.size() <= rank_);
    CHK_PRT_RET(
        check,
        HCCL_ERROR(
            "[Calc][LinksNum]para check failed, para vector size[%llu], dstRank[%u], rank[%u] ", paraVector_.size(),
            dstRank, rank_),
        HCCL_E_INTERNAL);
    // 节点间或者是节点内采用RDMA通信的, 放至dst_inter_client_map_,采用rdma建链
    bool isInterRdma
        = paraVector_[rank_].serverId != paraVector_[dstRank].serverId || isUsedRdmaLevel0_ || isAlltoAllCommMesh_;

    bool isInterHccs = IsSupportInterHccs(dstRank);

    HCCL_DEBUG(
        "[Calc][LinksNum]rank[%u], dstRank[%u], isInterRdma[%d], isInterHccs[%d], machineType[%d]", rank_, dstRank,
        isInterRdma, isInterHccs, machineType);

    auto dstRankInfo = paraVector_[dstRank];
    if (machineType == MachineType::MACHINE_SERVER_TYPE) {
        CHK_RET(MakeClientInfo(dstRank, dstRankInfo, isInterRdma, isInterHccs));
    }

    if (machineType == MachineType::MACHINE_CLIENT_TYPE) {
        CHK_RET(MakeServerInfo(dstRank, dstRankInfo, isInterRdma, isInterHccs));
    }

    return HCCL_SUCCESS;
}

HcclResult CommBase::MakeClientInfo(const u32 dstRank, RankInfo& dstRankInfo, bool isInterRdma, bool isInterHccs)
{
    if (isInterRdma && !isInterHccs) {
        HcclRankLinkInfo tempLinkInfo{};
        tempLinkInfo.userRank = dstRankInfo.userRank;
        tempLinkInfo.devicePhyId = dstRankInfo.devicePhyId;

        tempLinkInfo.ip = dstRankInfo.nicIp[0];
        tempLinkInfo.port = GetInterRemotePort(tempLinkInfo.devicePhyId, dstRankInfo.userRank);
        tempLinkInfo.socketsPerLink = GetSocketsPerLink();

        auto iter = dstInterClientMap_.find(dstRank);
        bool check = (iter != dstInterClientMap_.end());
        CHK_PRT_RET(
            check, HCCL_ERROR("[Make][ClientInfo]dstRank[%u] already exists in dst inter client map. ", dstRank),
            HCCL_E_PARA);
        dstInterClientMap_.insert(std::make_pair(dstRank, tempLinkInfo));
    } else {
        dstIntraClientVec_.push_back(dstRank);
    }
    return HCCL_SUCCESS;
}

HcclResult CommBase::MakeServerInfo(const u32 dstRank, RankInfo& dstRankInfo, bool isInterRdma, bool isInterHccs)
{
    // 节点间或者是节点内采用RDMA通信的，放至dst_inter_client_map_,采用rdma建链
    if (isInterRdma && !isInterHccs) {
        HcclRankLinkInfo tempLinkInfo{};
        tempLinkInfo.userRank = dstRankInfo.userRank;
        tempLinkInfo.devicePhyId = dstRankInfo.devicePhyId;

        HCCL_INFO(
            "dstRank = %u, useRank = %u, ip = %s", dstRank, dstRankInfo.userRank,
            dstRankInfo.nicIp[0].GetReadableAddress());

        tempLinkInfo.ip = dstRankInfo.nicIp[0];
        tempLinkInfo.port = GetInterRemotePort(tempLinkInfo.devicePhyId, dstRankInfo.userRank);
        tempLinkInfo.socketsPerLink = GetSocketsPerLink();

        auto iter = dstInterServerMap_.find(dstRank);
        bool check = (iter != dstInterServerMap_.end());
        CHK_PRT_RET(
            check, HCCL_ERROR("[Make][ServerInfo]dstRank[%u] already exists in dst inter server map", dstRank),
            HCCL_E_PARA);
        dstInterServerMap_.insert(std::make_pair(dstRank, tempLinkInfo));
    } else {
        dstIntraServerVec_.push_back(dstRank);
    }
    return HCCL_SUCCESS;
}

HcclResult CommBase::CreateDestLink(
    const ErrContextPub& error_context, const MachineType machineType, const std::string& serverId, const u32 dstRank,
    const std::string& threadStr, const std::vector<std::shared_ptr<HcclSocket>>& sockets)
{
    hrtErrMSetErrorContextPub(error_context);
    // 给当前线程添加名字
    SetThreadName(threadStr);
    if (!IsGeneralServer()) {
        CHK_RET(hrtSetDevice(deviceLogicId_));
        SetWorkflowMode(workflowMode_); // 新的线程，更新workflowMode
    }

    bool check = (paraVector_.size() <= dstRank) || (paraVector_.size() <= rank_) || (transportInfo_.size() <= dstRank)
                 || (transportType_.size() <= dstRank);
    CHK_PRT_RET(
        check,
        HCCL_ERROR(
            "[Create][DestLink]paraCheck failed, paraVector size[%llu], linkInfo size[%llu], "
            "linkType size[%llu], dstRank[%u], rank[%u] ",
            paraVector_.size(), transportInfo_.size(), transportType_.size(), dstRank, rank_),
        HCCL_E_INTERNAL);

    MachinePara machinePara;
    CHK_RET(SetMachinePara(machineType, serverId, dstRank, sockets, machinePara));
    HCCL_INFO(
        "[createLink para]rank[%u]-localUserrank[%u]-localIpAddr[%s], linkMode[%d] "
        "dst_rank[%u]-remoteUserrank[%u]-remote_ip_addr[%s], machineType[%d], serverId[%s], nicDeploy[%d] ",
        rank_, paraVector_[rank_].worldRank, paraVector_[rank_].serverId.c_str(), machinePara.linkMode, dstRank,
        paraVector_[dstRank].worldRank, paraVector_[dstRank].serverId.c_str(), machinePara.machineType,
        machinePara.serverId.c_str(), machinePara.nicDeploy);

    // transport初始化
    HcclResult ret = TransportInit(dstRank, machinePara);
    if (ret != HCCL_SUCCESS) {
        transportInfo_[dstRank] = nullptr;
        if (ret == HCCL_E_MEMORY) {
            std::string err_str
                = "[Create][DestLink]Transport init error! IPC memory allocation failed due to "
                  "possible memory limit exceeded. Suggested solution: Use 3TB / (ranksize * 2) as the upper limit of "
                  "HCCL_BUFFSIZE.";
            HCCL_ERROR("%s", err_str.c_str());
        }
        const std::string CREATE_LINK_ERR
            = "[Create][DestLink]Create Dest error! createLink para:rank[" + std::to_string(rank_) + "]-localUserrank["
              + std::to_string(paraVector_[rank_].worldRank) + "]-localIpAddr[" + paraVector_[rank_].serverId.c_str()
              + "], dst_rank[" + std::to_string(dstRank) + "]-remoteUserrank["
              + std::to_string(paraVector_[dstRank].worldRank) + "]-remote_ip_addr["
              + paraVector_[dstRank].serverId.c_str() + "]";

        HCCL_ERROR(
            "[Create][DestLink]Transport init error! createLink para:rank[%u]-localUserrank[%u]-localIpAddr[%s], "
            "dst_rank[%u]-remoteUserrank[%u]-remote_ip_addr[%s], machineType[%d], serverId[%s], linkMode[%d], "
            "shmDev_[%u], tag[%s]",
            rank_, paraVector_[rank_].worldRank, paraVector_[rank_].serverId.c_str(), dstRank,
            paraVector_[dstRank].worldRank, paraVector_[dstRank].serverId.c_str(), machinePara.machineType,
            machinePara.serverId.c_str(), machinePara.linkMode, shmDev_, machinePara.tag.c_str());
        return ret;
    }
    HCCL_INFO(
        "[createLink success]:rank[%u]-localUserrank[%u]-localIpAddr[%s], "
        "dst_rank[%u]-remoteUserrank[%u]-remote_ip_addr[%s], transportType_[%d], tag[%s]",
        rank_, paraVector_[rank_].worldRank, paraVector_[rank_].serverId.c_str(), dstRank,
        paraVector_[dstRank].worldRank, paraVector_[dstRank].serverId.c_str(), transportType_[dstRank],
        machinePara.tag.c_str());

    return HCCL_SUCCESS;
}

void CommBase::SetTransportParam(TransportPara& para, [[maybe_unused]] MachinePara& machinePara)
{
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    para.isRootRank = subUserRankRoot_ == rank_ ? true : false;
    para.timeout = kdefaultTimeout;
    para.virtualFlag = false;
}

HcclResult CommBase::TransportInit(const u32 dstRank, MachinePara& machinePara)
{
    CHK_PRT_RET(
        dstRank >= transportInfo_.size(),
        HCCL_ERROR("[TransportQuery] Transport[%u] is invalid, should init before query it.", dstRank), HCCL_E_PARA);
    // 实例化TransportBase
    CHK_RET(SetTransportType(dstRank));
    TransportPara para{};
    SetTransportParam(para, machinePara);

    TransportType type = transportType_[dstRank];
    if (type == TransportType::TRANS_TYPE_P2P) {
        transportInfo_[dstRank].reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else if (type == TransportType::TRANS_TYPE_IBV_EXP) {
        transportInfo_[dstRank].reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else {
        HCCL_ERROR("[Init][Transport]not supported transport type");
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PRT_RET(!transportInfo_[dstRank], HCCL_ERROR("[Init][Transport]In create link, new link failed"), HCCL_E_PTR);

    if (useOneDoorbell_) {
        transportInfo_[dstRank]->EnableUseOneDoorbell();
    }

    CHK_RET(transportInfo_[dstRank]->Init());

    CHK_RET(CheckExchangeInfo(transportInfo_[dstRank], machinePara.localDeviceId));

    return HCCL_SUCCESS;
}

HcclResult CommBase::SetMachinePara(
    MachineType machineType, const std::string& serverId, u32 dstRank,
    const std::vector<std::shared_ptr<HcclSocket>>& socketList, MachinePara& machinePara)
{
    SetMachineLinkMode(machinePara);
    HCCL_INFO("[Set][MachinePara]rankSize %u, linkMode %d", rankSize_, machinePara.linkMode);

    machinePara.machineType = machineType;
    machinePara.serverId = serverId;
    machinePara.localIpAddr = paraVector_[rank_].nicIp[0];
    machinePara.remoteIpAddr = paraVector_[dstRank].nicIp[0];
    machinePara.localUserrank = paraVector_[rank_].userRank;
    machinePara.remoteUserrank = paraVector_[dstRank].userRank;
    machinePara.localWorldRank = paraVector_[rank_].worldRank;
    machinePara.remoteWorldRank = paraVector_[dstRank].worldRank;
    machinePara.collectiveId = collectiveId_;
    machinePara.localDeviceId = paraVector_[rank_].devicePhyId;
    machinePara.remoteDeviceId = paraVector_[dstRank].devicePhyId;
    machinePara.deviceType = static_cast<DevType>(paraVector_[dstRank].deviceType);
    machinePara.inputMem = inputMem_;
    machinePara.outputMem = outputMem_;
    if (expMem_.ptr() != nullptr) {
        machinePara.mem.push_back(expMem_);
    } else {
        machinePara.mem.clear();
    }
    machinePara.linkAttribute = 0x03; /* 0x03同时支持目的端和源端发起 */
    machinePara.tag = tag_;

    // MoE算子优化，MC2 多机场景使用普通QP模式
    const std::string& suffix = HCCL_MC2_MULTISERVER_SUFFIX;
    if (tag_.size() > suffix.size() && tag_.compare(tag_.size() - suffix.size(), suffix.size(), suffix) == 0) {
        bool isSupportNormalQP{false};
        CHK_RET(IsSupportAicpuNormalQP(paraVector_[rank_].devicePhyId, isSupportNormalQP));
        if (isSupportNormalQP) {
            HCCL_INFO("[Set][MachinePara] Set machinePara.qpMode to [NORMAL]");
            machinePara.qpMode = QPMode::NORMAL;
        }
    }

    // 把原来的两层vector变成一层, 方便后继调用
    for (u32 i = 0; i < socketList.size(); i++) {
        machinePara.sockets.push_back(socketList[i]);
    }
    u64 rankConsistentDataLength
        = RankConsistentcyChecker::GetInstance(machinePara.localDeviceId).GetRankConsistentDataLength();
    machinePara.exchangeInfo.resize(rankConsistentDataLength);
    CHK_RET(RankConsistentcyChecker::GetInstance(machinePara.localDeviceId)
                .GetCheckFrame(&machinePara.exchangeInfo[0], rankConsistentDataLength, tag_));
    machinePara.supportDataReceivedAck = NeedDataReceivedAck();
    machinePara.nicDeploy = nicDeployInner_;
    machinePara.localSocketPort = paraVector_[rank_].hostPort;
    machinePara.remoteSocketPort = paraVector_[dstRank].hostPort;
    machinePara.isAicpuModeEn = isAicpuModeEn_;
    machinePara.deviceLogicId = deviceLogicId_;
    machinePara.srcPorts = std::vector<std::uint16_t>(1, 0); /* 默认填充一个元素，0代表默认不配置 */
    return HCCL_SUCCESS;
}

HcclResult CommBase::CreateVirturalTransport()
{
    MachinePara machinePara;
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());

    vTransportInfo_.resize(transportInfo_.size());
    for (u32 i = 0; i < transportInfo_.size(); i++) {
        TransportPara para{};
        para.virtualFlag = true;
        para.timeout = kdefaultTimeout;
        para.index = i;
        vTransportInfo_[i].reset(new (std::nothrow) Transport(
            TransportType::TRANS_TYPE_RESERVED, para, dispatcher_, notifyPool_, machinePara));

        CHK_PRT_RET(
            !vTransportInfo_[i], HCCL_ERROR("[CreateVirturalTransport]In create link, new link failed"), HCCL_E_PTR);
    }

    return HCCL_SUCCESS;
}

std::shared_ptr<Transport>& CommBase::GetTrasportInfoByVTransportInfoIndex(u32 index)
{
    if (vTransportInfo_.size() <= index) {
        HCCL_ERROR(
            "[GetTrasportInfoByVTransportInfoIndex]index[%u] is bigger than vlink size[%llu]", index,
            vTransportInfo_.size());
        return linkDummy_;
    }

    if (transportInfo_.size() <= index) {
        HCCL_ERROR(
            "[GetTrasportInfoByVTransportInfoIndex]index[%u] is bigger than link size[%llu]", index,
            transportInfo_.size());
        return linkDummy_;
    }
    return transportInfo_[index];
}
HcclResult CommBase::BuildAsync(u32& status)
{
    transportStatus_.resize(rankSize_, 1);
    checkStatus_.resize(rankSize_, false);

    // 获取rank->userrank以及userrank->rank的映射关系
    CHK_RET(SetRankMap());

    // 获取当前线程操作的设备ID
    deviceLogicId_ = 0;
    if (paraVector_[rank_].devicePhyId != HOST_DEVICE_ID) {
        CHK_RET(hrtGetDevice(&deviceLogicId_));
    }

    if (rankSize_ == HCCL_RANK_SIZE_EQ_ONE) {
        HCCL_INFO("comm base needn't to create links, rankSize_[%u].", rankSize_);
        status = 0;
        return HCCL_SUCCESS;
    }
    CHK_RET(CalcLink());

    // 当前rank作为client端角色
    u32 dstIntraServerNum = dstIntraServerVec_.size();
    std::vector<std::shared_ptr<HcclSocket>> sockets;
    for (u32 intraIndex = 0; intraIndex < dstIntraServerNum; intraIndex++) {
        HcclResult ret = TransportBuildAsync(
            MachineType::MACHINE_CLIENT_TYPE, paraVector_[rank_].serverId, dstIntraServerVec_[intraIndex], sockets,
            transportStatus_[dstIntraServerVec_[intraIndex]]);
        CHK_PRT_RET(
            ret,
            HCCL_ERROR(
                "[BuildAsync] transport build async failed, self rank[%u], peer rank[%u]", paraVector_[rank_].worldRank,
                paraVector_[dstIntraServerVec_[intraIndex]].worldRank),
            HCCL_E_INTERNAL);
    }

    // 当前rank作为server端角色
    u32 dstIntraClientNum = dstIntraClientVec_.size();
    for (u32 intraIndex = 0; intraIndex < dstIntraClientNum; intraIndex++) {
        HcclResult ret = TransportBuildAsync(
            MachineType::MACHINE_SERVER_TYPE, paraVector_[rank_].serverId, dstIntraClientVec_[intraIndex], sockets,
            transportStatus_[dstIntraClientVec_[intraIndex]]);
        CHK_PRT_RET(
            ret,
            HCCL_ERROR(
                "[BuildAsync] transport build async failed, self rank[%u], peer rank[%u]", paraVector_[rank_].worldRank,
                paraVector_[dstIntraClientVec_[intraIndex]].worldRank),
            HCCL_E_INTERNAL);
    }

    // 暂不支持 跨node通信

    CHK_RET(GetBuildStatus(status));
    return HCCL_SUCCESS;
}

HcclResult CommBase::BuildQuerry(u32& status)
{
    for (u32 i = 0; i < transportStatus_.size(); i++) {
        if (transportStatus_[i] == HETEROG_P2P_WAIT) {
            CHK_RET(TransportBuildQuerry(i, transportStatus_[i]));
        }
    }
    CHK_RET(GetBuildStatus(status));
    HCCL_DEBUG("BuildQuery: %u", status);
    return HCCL_SUCCESS;
}

HcclResult CommBase::GetBuildStatus(u32& status)
{
    u32 transportDoneNum = 0;
    u32 transportErrorNum = 0;
    for (u32 i = 0; i < transportStatus_.size(); i++) {
        if (transportStatus_[i] == HETEROG_P2P_SUCCESS) {
            transportDoneNum++;
        } else if (transportStatus_[i] == HETEROG_P2P_FAILED) {
            transportErrorNum++;
        }
    }
    u32 transportNum
        = dstInterClientMap_.size() + dstIntraClientVec_.size() + dstInterServerMap_.size() + dstIntraServerVec_.size();
    if (transportErrorNum > 0) {
        status = HETEROG_P2P_FAILED;
        HCCL_ERROR("transport error num[%u].", transportErrorNum);
        return HCCL_E_INTERNAL;
    } else if (transportDoneNum == transportNum) {
        status = HETEROG_P2P_SUCCESS;
        HCCL_INFO("CommBase connect complete.");
    } else if (transportDoneNum < transportNum) {
        status = HETEROG_P2P_WAIT;
    } else {
        status = HETEROG_P2P_FAILED;
        HCCL_ERROR("transport done num[%u] invalid, expect[%u].", transportDoneNum, transportNum);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult CommBase::TransportBuildAsync(
    const MachineType machineType, const std::string& serverId, u32 dstRank,
    const std::vector<std::shared_ptr<HcclSocket>>& sockets, u32& status)
{
    CHK_PRT_RET(
        dstRank >= transportInfo_.size(),
        HCCL_ERROR("[TransportQuery] Transport[%u] is invalid, should init before query it.", dstRank), HCCL_E_PARA);
    MachinePara machinePara;
    CHK_RET(SetMachinePara(machineType, serverId, dstRank, sockets, machinePara));
    // 实例化TransportBase
    CHK_RET(SetTransportType(dstRank));
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());

    TransportPara para{};
    para.timeout = kdefaultTimeout;
    para.virtualFlag = false;
    transportInfo_[dstRank].reset(
        new (std::nothrow) Transport(TransportType::TRANS_TYPE_RESERVED, para, dispatcher_, notifyPool_, machinePara));
    CHK_PRT_RET(!transportInfo_[dstRank], HCCL_ERROR("[Init][Transport]In create link, new link failed"), HCCL_E_PTR);

    CHK_RET(transportInfo_[dstRank]->ConnectAsync(status));
    if (status == HETEROG_P2P_SUCCESS && checkStatus_[dstRank] == false) {
        checkStatus_[dstRank] = true;
        CHK_RET(CheckExchangeInfo(transportInfo_[dstRank], machinePara.localDeviceId));
    }
    HCCL_DEBUG("TransportBuildAsync[%u] %u", dstRank, status);
    return HCCL_SUCCESS;
}

HcclResult CommBase::TransportBuildQuerry(u32 dstRank, u32& status)
{
    CHK_PRT_RET(
        dstRank >= transportInfo_.size(),
        HCCL_ERROR("[TransportQuery] Transport[%u] is invalid, should init before query it.", dstRank), HCCL_E_PARA);
    if (transportInfo_[dstRank]) {
        CHK_RET(transportInfo_[dstRank]->ConnectQuerry(status));
        if (status == HETEROG_P2P_SUCCESS && checkStatus_[dstRank] == false) {
            checkStatus_[dstRank] = true;
            CHK_RET(CheckExchangeInfo(transportInfo_[dstRank], paraVector_[rank_].devicePhyId));
        }
    } else {
        status = HETEROG_P2P_WAIT;
    }
    HCCL_DEBUG("TransportBuildQuery[%u] %u", dstRank, status);
    return HCCL_SUCCESS;
}

HcclResult CommBase::CreateExchangerNetwork()
{
    CHK_PRT_RET(
        dstIntraServerVec_.empty() && dstIntraClientVec_.empty(),
        HCCL_DEBUG("[Create][ExchangerNetwork]dstIntraServerVec and dstIntraClientVec is empty, do nothing."),
        HCCL_SUCCESS);

    bool isInterServer = false; // 是否跨server
    bool isInterHccs = true;    // 是否超节点模式
    std::map<u32, HcclSocketRole> rankRole;
    CHK_RET(GetRankLinkInfo(isInterServer, isInterHccs, rankRole));

    // 保持原逻辑不变，将rank&deviceIP信息，构造成 std::map<u32, std::vector<HcclIpAddress> >，使用
    // SocketManager创建链接
    std::string commTag = (Is310PDevice() || isHaveCpuRank_) ? tag_ : collectiveId_;
    HcclIpAddress localIP;
    std::map<u32, HcclRankLinkInfo> dstServerMap;
    std::map<u32, HcclRankLinkInfo> dstClientMap;
    bool isSupportReuse = false;

    CHK_RET(GetRankIPInfo(
        isInterServer, isInterHccs, isSupportReuse, rankRole, localIP, dstServerMap, dstClientMap, socketManager_));

    std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> serverSocketsMap;
    std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> clientSocketsMap;

    HcclUs startut = TIME_NOW();
    HcclResult ret = socketManager_->CreateSockets(
        commTag, false, netDevCtxMap_[localIP], dstServerMap, dstClientMap, serverSocketsMap, clientSocketsMap,
        isSupportReuse);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Create][ExchangerNetwork]sync create connections Failed, ret[%u].", ret),
        ret);

    HCCL_DEBUG(
        "[Create][Exchanger] serverSocketsMap size[%u], clientSocketsMap size[%u]", serverSocketsMap.size(),
        clientSocketsMap.size());
    intraSocketsMap_.insert(serverSocketsMap.begin(), serverSocketsMap.end());
    intraSocketsMap_.insert(clientSocketsMap.begin(), clientSocketsMap.end());

    HCCL_INFO("[Create][ExchangerNetwork]create connections duration time:%lld us.", DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CommBase::GetRankIPInfo(
    bool isInterServer, bool isInterHccs, bool& isSupportReuse, std::map<u32, HcclSocketRole>& rankRole,
    HcclIpAddress& localIP, std::map<u32, HcclRankLinkInfo>& dstServerMap,
    std::map<u32, HcclRankLinkInfo>& dstClientMap, std::shared_ptr<HcclSocketManager>& socketManager)
{
    if (Is310PDevice() || isHaveCpuRank_) {
        // 310P和异构场景
        std::vector<u32> dstIntraVec;
        for (auto it = rankRole.begin(); it != rankRole.end(); ++it) {
            dstIntraVec.push_back(it->first);
        }
        CHK_RET(GetIntraRankIPInfo(dstIntraVec, localIP, dstServerMap, dstClientMap));

        // 不复用，每次都创建
        isSupportReuse = false;
        socketManager.reset(new (std::nothrow)
                                HcclSocketManager(nicDeployInner_, deviceLogicId_, devicePhyId_, userRank_));
        CHK_PTR_NULL(socketManager);
    } else if (isInterServer && isInterHccs) {
        // 超节点间Hccs模式
        CHK_RET(GetSuperNodeIntraRankIPInfo(rankRole, localIP, dstServerMap, dstClientMap));
        isSupportReuse = true;
        socketManager = exchanger_.socketManager;
        CHK_PTR_NULL(socketManager);
    } else if (isInterServer == false) {
        // server内模式
        CHK_RET(GetIntraRankIPInfo(rankRole, localIP, dstServerMap, dstClientMap));
        isSupportReuse = true;
        socketManager = exchanger_.socketManager;
        CHK_PTR_NULL(socketManager);
    } else {
        HCCL_ERROR(
            "[Create][ExchangerNetwork]isInterServer[%d] and isInterHccs[%d] is not support", isInterServer,
            isInterHccs);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult CommBase::GetRankLinkInfo(bool& isInterServer, bool& isInterHccs, std::map<u32, HcclSocketRole>& rankRole)
{
    std::vector<u32> devicePhyIds;
    for (u32 dstRank : dstIntraServerVec_) {
        isInterServer |= (paraVector_[dstRank].serverId != paraVector_[rank_].serverId);
        isInterHccs &= IsSupportInterHccs(dstRank);
        rankRole.insert(std::make_pair(dstRank, HcclSocketRole::SOCKET_ROLE_SERVER));
        devicePhyIds.push_back(paraVector_[dstRank].devicePhyId);
    }
    for (u32 dstRank : dstIntraClientVec_) {
        isInterServer |= (paraVector_[dstRank].serverId != paraVector_[rank_].serverId);
        isInterHccs &= IsSupportInterHccs(dstRank);
        rankRole.insert(std::make_pair(dstRank, HcclSocketRole::SOCKET_ROLE_CLIENT));
        devicePhyIds.push_back(paraVector_[dstRank].devicePhyId);
    }
    rankRole.insert(std::make_pair(rank_, HcclSocketRole::SOCKET_ROLE_RESERVED));
    devicePhyIds.push_back(paraVector_[rank_].devicePhyId);

    if (paraVector_[rank_].deviceType == DevType::DEV_TYPE_310P3) {
        HcclResult ret = P2PMgmtPub::EnableP2P(devicePhyIds);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Get][RankLinkInfo]Enable P2P Failed, devicePhyId[%d], ret[%u]", paraVector_[rank_].devicePhyId, ret),
            ret);
    }
    // server内非异构场景，使能P2P
    // 心跳需要单独WaitP2PEnabled？
    if (!isInterServer && !isHaveCpuRank_) {
        HcclResult ret = P2PMgmtPub::WaitP2PEnabled(devicePhyIds);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Get][RankLinkInfo]Enable P2P Failed, ret[%u]", ret), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult CommBase::GetIntraRankIPInfo(
    std::map<u32, HcclSocketRole>& rankRole, HcclIpAddress& localIP, std::map<u32, HcclRankLinkInfo>& dstServerMap,
    std::map<u32, HcclRankLinkInfo>& dstClientMap)
{
    u32 userRankSize = rankRole.size();
    for (auto rankIter = rankRole.begin(); rankIter != rankRole.end(); rankIter++) {
        u32 dstRank = rankIter->first;
        s32 dstDeviceId = useSuperPodMode_ ? static_cast<s32>(paraVector_[dstRank].superDeviceId) :
                                             paraVector_[dstRank].devicePhyId;
        HcclSocketRole localRole = rankIter->second;

        // Rank devicePhyId 作为地址
        HcclRankLinkInfo linkInfo{};
        linkInfo.userRank = paraVector_[dstRank].userRank;
        linkInfo.devicePhyId = dstDeviceId;
        if (vnicRanksPort_.empty() || (userRankSize > 1 && IsSupportMC2(tag_) >= MC2_PLANE_MODE_COMBINE)) {
            linkInfo.port = GetNicPort(paraVector_[dstRank].devicePhyId, ranksPort_, linkInfo.userRank, isUseRankPort_);
        } else {
            linkInfo.port
                = GetNicPort(paraVector_[dstRank].devicePhyId, vnicRanksPort_, linkInfo.userRank, isUseRankPort_);
        }
        HcclIpAddress ipAddress(linkInfo.devicePhyId);
        DeviceIdType deviceidType
            = useSuperPodMode_ ? (DeviceIdType::DEVICE_ID_TYPE_SDID) : (DeviceIdType::DEVICE_ID_TYPE_PHY_ID);
        // rank个数小于等于1时，没有初始化ra资源，无法调用device侧hccp接口
        if (userRankSize > 1) {
            if (IsSupportMC2(tag_) >= MC2_PLANE_MODE_COMBINE) {
                ipAddress = paraVector_[dstRank].nicIp.front();
                CHK_PRT_RET(
                    ipAddress.IsInvalid(),
                    HCCL_ERROR(
                        "[Get][IntraRankIPInfo] ipAddress is invalid when NIC, check the ip configuration for "
                        "dstRank[%u]",
                        dstRank),
                    HCCL_E_PARA);
            } else {
                CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                    paraVector_[rank_].devicePhyId, deviceidType, linkInfo.devicePhyId, ipAddress));
            }
        }
        linkInfo.ip = ipAddress;
        linkInfo.socketsPerLink = 1;

        HCCL_DEBUG(
            "[Get][IntraRankIPInfo] tag[%s], userRank[%u], destRank[%u], localRole[%d], port[%u], ip[%s], "
            "devicePhyId[%u]",
            tag_.c_str(), rank_, linkInfo.userRank, localRole, linkInfo.port, linkInfo.ip.GetReadableAddress(),
            paraVector_[rank_].devicePhyId);

        if (localRole == HcclSocketRole::SOCKET_ROLE_CLIENT) {
            dstServerMap.insert(std::make_pair(linkInfo.userRank, linkInfo));
        } else if (localRole == HcclSocketRole::SOCKET_ROLE_SERVER) {
            dstClientMap.insert(std::make_pair(linkInfo.userRank, linkInfo));
        } else {
            // 当前上层逻辑，保证 userRank_(当前 Rank) 在 userRanks 中
            localIP = linkInfo.ip;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CommBase::GetIntraRankIPInfo(
    std::vector<u32>& dstIntraVec, HcclIpAddress& localIP, std::map<u32, HcclRankLinkInfo>& dstServerMap,
    std::map<u32, HcclRankLinkInfo>& dstClientMap)
{
    for (u32 dstRank : dstIntraVec) {
        auto& rankInfo = paraVector_[dstRank];
        HcclRankLinkInfo linkInfo{};
        linkInfo.userRank = rankInfo.userRank;
        linkInfo.devicePhyId = rankInfo.devicePhyId;
        linkInfo.ip = isHaveCpuRank_ ? rankInfo.hostIp : rankInfo.nicIp[0];
        if (!vnicRanksPort_.empty()) {
            linkInfo.port = GetNicPort(linkInfo.devicePhyId, vnicRanksPort_, linkInfo.userRank, isUseRankPort_);
        } else {
            linkInfo.port = GetNicPort(linkInfo.devicePhyId, ranksPort_, linkInfo.userRank, isUseRankPort_);
        }
        linkInfo.socketsPerLink = 1;

        HcclSocketRole localRole;
        if (paraVector_[rank_].userRank < linkInfo.userRank) {
            dstClientMap.insert(std::make_pair(linkInfo.userRank, linkInfo));
            localRole = HcclSocketRole::SOCKET_ROLE_CLIENT;
        } else if (paraVector_[rank_].userRank > linkInfo.userRank) {
            dstServerMap.insert(std::make_pair(linkInfo.userRank, linkInfo));
            localRole = HcclSocketRole::SOCKET_ROLE_SERVER;
        } else {
            localIP = linkInfo.ip;
            localRole = HcclSocketRole::SOCKET_ROLE_RESERVED;
        }
        HCCL_DEBUG(
            "[Get][IntraRankIPInfo] userRank[%u], destRank[%u], localRole[%d], port[%u], ip[%s]", userRank_,
            linkInfo.userRank, localRole, linkInfo.port, linkInfo.ip.GetReadableAddress());
    }
    return HCCL_SUCCESS;
}

HcclResult CommBase::GetSuperNodeIntraRankIPInfo(
    std::map<u32, HcclSocketRole>& rankRole, HcclIpAddress& localIP, std::map<u32, HcclRankLinkInfo>& dstServerMap,
    std::map<u32, HcclRankLinkInfo>& dstClientMap)
{
    for (auto rankIter = rankRole.begin(); rankIter != rankRole.end(); rankIter++) {
        u32 dstUserRank = paraVector_[rankIter->first].userRank;
        HcclIpAddress ipAddr(paraVector_[rankIter->first].nicIp[0]);
        if (!GetExternalInputInterHccsDisable()) {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_SDID, paraVector_[rankIter->first].superDeviceId, ipAddr));
        }
        HcclSocketRole localRole = rankIter->second;

        HcclRankLinkInfo linkInfo{};
        u32 dstRank = INVALID_VALUE_RANKID;
        linkInfo.userRank = dstUserRank;
        linkInfo.devicePhyId = -1;
        linkInfo.ip = ipAddr;
        CHK_RET(GetRankByUserRank(linkInfo.userRank, dstRank));
        if (!vnicRanksPort_.empty()) {
            linkInfo.port
                = GetNicPort(paraVector_[dstRank].devicePhyId, vnicRanksPort_, linkInfo.userRank, isUseRankPort_);
        } else {
            linkInfo.port = GetNicPort(paraVector_[dstRank].devicePhyId, ranksPort_, linkInfo.userRank, isUseRankPort_);
        }
        linkInfo.socketsPerLink = 1;

        HCCL_DEBUG(
            "[Get][SuperNodeIntraRankIPInfo] userRank[%u], destRank[%u], localRole[%d], port[%u], ip[%s]", userRank_,
            dstUserRank, localRole, linkInfo.port, linkInfo.ip.GetReadableAddress());

        if (localRole == HcclSocketRole::SOCKET_ROLE_CLIENT) {
            dstServerMap.insert(std::make_pair(dstUserRank, linkInfo));
        } else if (localRole == HcclSocketRole::SOCKET_ROLE_SERVER) {
            dstClientMap.insert(std::make_pair(dstUserRank, linkInfo));
        } else {
            // 当前上层逻辑，保证 userRank_(当前 Rank) 在 userRanks 中
            localIP = linkInfo.ip;
        }
    }
    return HCCL_SUCCESS;
}

bool CommBase::IsSupportInterHccs(const u32 dstRank)
{
    // 仅判断超节点内, 兼容打平通信域同时有server内和server间, 因此不判断server_id
    bool isInterHccs = GetExternalInputInterHccsDisable() == false
                       && paraVector_[rank_].deviceType == DevType::DEV_TYPE_910_93
                       && paraVector_[rank_].superPodId.empty() == false
                       && paraVector_[rank_].superPodId == paraVector_[dstRank].superPodId;

    HCCL_INFO(
        "[IsSupportInterHccs]rank[%u], superPodId[%s], dstRank[%u], dstSuperPodId[%s], isInterHccs[%d]", rank_,
        paraVector_[rank_].superPodId.c_str(), dstRank, paraVector_[dstRank].superPodId.c_str(), isInterHccs);
    return isInterHccs;
}

void CommBase::SetMachineLinkMode(MachinePara& machinePara) { machinePara.linkMode = LinkMode::LINK_DUPLEX_MODE; }

HcclResult CommBase::SetHDCModeInfo(
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>>& rankDevicePhyIdNicInfoMap,
    std::vector<u32>& ranksPort, std::vector<u32>& vnicRanksPort, bool isSetHDCModeInfo, bool isUseRankPort)
{
    rankDevicePhyIdNicInfoMap_ = rankDevicePhyIdNicInfoMap;
    ranksPort_ = ranksPort;
    vnicRanksPort_ = vnicRanksPort;
    isSetHDCModeInfo_ = isSetHDCModeInfo;
    isUseRankPort_ = isUseRankPort;
    return HCCL_SUCCESS;
}

HcclResult CommBase::CheckExchangeInfo(const std::shared_ptr<Transport>& link, const s32 deviceId)
{
    // 算子一致性校验
    u64 exchangeInfoLength = RankConsistentcyChecker::GetInstance(deviceId).GetRankConsistentDataLength();
    std::vector<u8> recvData = link->GetExchangeInfo();
    if (recvData.size() != 0) {
        CHK_PRT_RET(
            recvData.size() != exchangeInfoLength,
            HCCL_ERROR(
                "[Check][ExchangeInfo]remote exchangInfo size[%zu], local exchangeInfo size[%llu]", recvData.size(),
                exchangeInfoLength),
            HCCL_E_INTERNAL);
        CHK_RET(
            RankConsistentcyChecker::GetInstance(deviceId).CheckFrameRecv(&recvData[0], recvData.size(), tag_.c_str()));
    }

    return HCCL_SUCCESS;
}

u32 CommBase::IsSupportMC2(const std::string& tag)
{
    const std::string& suffix = HCCL_MC2_MULTISERVER_SUFFIX;
    u32 mc2MultiServerType = MC2_PLANE_MODE_HOST; // 非直驱场景
    if (tag.size() > suffix.size() && tag.compare(tag.size() - suffix.size(), suffix.size(), suffix) == 0) {
        mc2MultiServerType = MC2_PLANE_MODE_COMBINE; // 非分层建链场景
        if (tag.find("_HIE") != std::string::npos) {
            mc2MultiServerType = MC2_PLANE_MODE_HIERARCHY; // 分层建链场景
        }
    }
    return mc2MultiServerType;
}
} // namespace hccl
