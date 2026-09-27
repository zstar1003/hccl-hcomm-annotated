/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_manager.h"
#include "p2p_mgmt_pub.h"
#include <algorithm>
#include "rank_consistentcy_checker.h"
#include "env_config.h"
#include "detect_connect_anomalies.h"
#include "../../nslbdp/hccl_nslbdp.h"
#include "device_capacity.h"
#include "rt_external.h"

namespace hccl {

TransportManager::TransportManager(
    CCLBufferManager& cclBufferManager, const std::unique_ptr<HcclSocketManager>& socketManager,
    HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, const std::vector<RankInfo>& rankInfoList,
    RankId userRank, const std::string& identifier, s32 deviceLogicId, NICDeployment nicDeployment, bool isHaveCpuRank,
    bool isUseRankPort, bool isUsedRdmaLevel0, const std::vector<u32>& nicRanksPort,
    const std::vector<u32>& vnicRanksPort, bool useSuperPodMode, const std::vector<HcclIpAddress>& devIpAddr,
    const HcclIpAddress& hostIp, const HcclIpAddress& localVnicIp, std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap)
    : cclBufferManager_(cclBufferManager),
      socketManager_(socketManager),
      dispatcher_(dispatcher),
      notifyPool_(notifyPool),
      rankInfoList_(rankInfoList),
      userRank_(userRank),
      identifier_(identifier),
      deviceLogicId_(deviceLogicId),
      nicDeployment_(nicDeployment),
      isHaveCpuRank_(isHaveCpuRank),
      isUseRankPort_(isUseRankPort),
      isUsedRdmaLevel0_(isUsedRdmaLevel0),
      nicRanksPort_(nicRanksPort),
      vnicRanksPort_(vnicRanksPort),
      useSuperPodMode_(useSuperPodMode),
      devIpAddr_(devIpAddr),
      hostIp_(hostIp),
      localVnicIp_(localVnicIp),
      netDevCtxMap_(netDevCtxMap),
      trafficClass_(HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET),
      serviceLevel_(HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET)
{
    rankConsistentDataLength_ = RankConsistentcyChecker::GetInstance().GetRankConsistentDataLength();
}

TransportManager::~TransportManager()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (enableP2PDevices_.size() != 0) {
        (void)P2PMgmtPub::DisableP2P(enableP2PDevices_);
        enableP2PDevices_.clear();
    }
}

constexpr u32 EXCEPTION_DELAY_US_COUNT = 100000;
constexpr u32 MUL_QP_SOCKETS_PER_LINk = 2;
HcclResult TransportManager::ExceptionHandle(const std::string& tag, OpCommTransport& opTransportResponse)
{
    for (auto& levelNSubCommTransport : opTransportResponse) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            for (auto& transportRequest : singleSubCommTransport.transportRequests) {
                if (transportRequest.isValid) {
                    bool isInterRdma;
                    UpdateIsInterRdma(transportRequest.remoteUserRank, isInterRdma, transportRequest.isUsedRdma);

                    HcclRankLinkInfo remoteLinkInfo;
                    MakeRemoteLinkInfo(transportRequest.remoteUserRank, isInterRdma, 1, remoteLinkInfo);

                    HcclIpAddress ipAddr;
                    if (isInterRdma || Is310PDevice()) {
                        ipAddr = nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE ? devIpAddr_[0] : hostIp_;
                    } else {
                        ipAddr = localVnicIp_;
                    }

                    bool isHccs = false;
                    if (!isInterRdma) {
                        isHccs = IsHccsTransport(transportRequest.remoteUserRank, transportRequest.linkType);
                    }
                    std::string newTag;
                    CHK_RET(ConstructTransTag(tag, newTag, isInterRdma, 0, isHccs));
                    CHK_RET(socketManager_->AddWhiteList(newTag, netDevCtxMap_[ipAddr], remoteLinkInfo));
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TransportManager::CreateVirturalTransport(SingleSubCommTransport& singleSubCommTransport)
{
    MachinePara machinePara;
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());

    singleSubCommTransport.virtualLinks.clear();
    singleSubCommTransport.virtualLinks.resize(singleSubCommTransport.transportRequests.size());

    for (u32 i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {
        TransportPara para{};
        para.virtualFlag = true;
        para.timeout = kdefaultTimeout;
        para.index = i;
        singleSubCommTransport.virtualLinks[i].reset(new (std::nothrow) Transport(
            TransportType::TRANS_TYPE_RESERVED, para, dispatcher_, notifyPool_, machinePara));
        CHK_PRT_RET(
            !singleSubCommTransport.virtualLinks[i],
            HCCL_ERROR("[CreateVirturalTransport]In create link,"
                       "new link failed"),
            HCCL_E_PTR);
    }

    return HCCL_SUCCESS;
}

void TransportManager::SetQpQosAttr(u32 trafficClass, u32 serviceLevel)
{
    trafficClass_ = trafficClass;
    serviceLevel_ = serviceLevel;
}

HcclResult TransportManager::AddremoteUserRankToList(
    TransportRequest& transportRequest, std::vector<u32>& rankList, TransportType transportType)
{
    if (!transportRequest.isValid) {
        HCCL_WARNING("[AddremoteUserRankToList]transportRequest is invalid. No need to build a link, skip");
        return HCCL_SUCCESS;
    }
    TransportType type = TransportType::TRANS_TYPE_RESERVED;
    CHK_PRT(GetTransportType(transportRequest.remoteUserRank, transportRequest.isUsedRdma, type));
    if (type == transportType) {
        // 仅添加对应Type类型的对端
        rankList.emplace_back(transportRequest.remoteUserRank);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportManager::GetRemoteRankList(
    OpCommTransport& opTransportResponse, std::vector<u32>& rankList, TransportType transportType)
{
    // 对当前所有的transportLink做判断
    for (auto& levelNSubCommTransport : opTransportResponse) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            for (auto& transportRequest : singleSubCommTransport.transportRequests) {
                CHK_PRT(AddremoteUserRankToList(transportRequest, rankList, transportType));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportManager::createSubCommLinkThreads(
    const std::string& tag, const TransportIOMem& transMem, struct SubCommLinkPara& subCommLinkPara, bool isAicpuModeEn,
    bool isBackup, u32 subCommIndex, bool isCapture, const HcclCMDType& opType, bool isIndOp, bool isNpuDirectRoce)
{
    u32 num = subCommLinkPara.remoteRankIdNum;
    struct SingleSubCommTransport& singleSubCommTransport = subCommLinkPara.singleSubCommTransport;
    subCommLinkPara.linkThreads.resize(num);
    subCommLinkPara.linkResult.resize(num, HCCL_SUCCESS);

    for (u32 i = 0; i < num; i++) {
        u32 index
            = subCommLinkPara
                  .remoteRankMap[(subCommLinkPara.remoteRankIdStartIndex + i) % subCommLinkPara.remoteRankMap.size()]
                  .second;
        auto& transportRequest = singleSubCommTransport.transportRequests[index];
        auto& link = singleSubCommTransport.links[index];

        if ((!transportRequest.isValid) || (link != nullptr) || (isBackup && !transportRequest.isUsedRdma)) {
            HCCL_INFO(
                "[%s]: no need to create p2p back link, remote UserRank[%u], userRank[%u], "
                "isUsedRdma[%u], isBackup[%d]",
                __func__, transportRequest.remoteUserRank, userRank_, transportRequest.isUsedRdma, isBackup);
            continue;
        }

        DeviceMem inputMem;
        DeviceMem outputMem;
        DeviceMem expMem;
        GetIOMem(transMem, transportRequest.inputMemType, transportRequest.outputMemType, inputMem, outputMem, expMem);
        HCCL_INFO(
            "transportRequest.inputMemType[%d] transportRequest.outputMemType[%d], isBackup[%d]",
            transportRequest.inputMemType, transportRequest.outputMemType, isBackup);

        IndOpMem indOpMem;
        if (isIndOp) {
            indOpMem = transMem.indOpMem;
            HCCL_DEBUG(
                "transportRequest indOpMem, userHostMem size[%llu], userDeviceMem size[%llu]",
                indOpMem.userHostMem.size(), indOpMem.userDeviceMem.size());
        }

        std::vector<std::shared_ptr<HcclSocket>> connectSockets;
        bool isInterRdma;
        bool chooseBackup = transportRequest.isUsedRdma ? isBackup : false;
        HcclNetDevCtx netDevCtx;
        HcclResult ret = CreateDestSockets(
            tag, transportRequest.remoteUserRank, singleSubCommTransport.taskNum, connectSockets, netDevCtx,
            isInterRdma, transportRequest.isUsedRdma, chooseBackup, subCommIndex, transportRequest.linkType);
        HCCL_DEBUG("[%s]CreateDestSockets finished, chooseBackup[%d]", __func__, chooseBackup);
        HCCL_DEBUG(
            "[%s]: remoteUserRank[%u], userRank[%u], isUsedRdma[%u]", __func__, transportRequest.remoteUserRank,
            userRank_, transportRequest.isUsedRdma);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Alloc]Create dest sockets failed"), ret);

        MachineType machineType = transportRequest.localUserRank < transportRequest.remoteUserRank ?
                                      MachineType::MACHINE_SERVER_TYPE :
                                      MachineType::MACHINE_CLIENT_TYPE;
        std::string threadStr = (isInterRdma ? "HcclTerL_" : "HcclIntra_") + std::to_string(i);
        subCommLinkPara.linkThreads[i].reset(new (std::nothrow) std::thread(
            &TransportManager::CreateLink, this, tag, hrtErrMGetErrorContextPub(), machineType,
            rankInfoList_[userRank_].serverId, transportRequest.remoteUserRank,
            singleSubCommTransport.supportDataReceivedAck, singleSubCommTransport.linkMode,
            singleSubCommTransport.enableUseOneDoorbell, threadStr, connectSockets, inputMem, outputMem,
            transportRequest.isUsedRdma, std::ref(link), isAicpuModeEn, std::ref(subCommLinkPara.linkResult[i]),
            netDevCtx, transportRequest.notifyNum, chooseBackup, isCapture, expMem, transportRequest.linkType, isIndOp,
            indOpMem, opType, isNpuDirectRoce));
        CHK_SMART_PTR_NULL(subCommLinkPara.linkThreads[i]);            // 异常时其他线程待处理
        singleSubCommTransport.status[index] = TransportStatus::READY; // 建链后 transport设置为ready状态
    }

    return HCCL_SUCCESS;
}

HcclResult TransportManager::waitSubCommLinkThreadsComplete(struct SubCommLinkPara& subCommLinkPara)
{
    for (u32 i = 0; i < subCommLinkPara.linkThreads.size(); i++) {
        if (subCommLinkPara.linkThreads[i] == nullptr || !subCommLinkPara.linkThreads[i]->joinable()) {
            continue;
        }
        subCommLinkPara.linkThreads[i]->join();  // 等待线程执行完毕
        CHK_RET(hrtResetDevice(deviceLogicId_)); // 防止线程里面异常退出，在进程中reset
    }
    subCommLinkPara.linkThreads.clear();
    CHK_PRT_RET(GetStopFlag(), HCCL_ERROR("Terminating operation due to external request"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult TransportManager::checkSubCommLinkThreadsStatus(
    const std::string& tag, struct SubCommLinkPara& subCommLinkPara, bool isBackup)
{
    u32 num = subCommLinkPara.remoteRankIdNum;
    struct SingleSubCommTransport& singleSubCommTransport = subCommLinkPara.singleSubCommTransport;

    for (u32 i = 0; i < subCommLinkPara.linkResult.size(); i++) {
        CHK_RET(subCommLinkPara.linkResult[i]);
    }
    for (u32 i = 0; i < num; i++) {
        u32 index
            = subCommLinkPara
                  .remoteRankMap[(subCommLinkPara.remoteRankIdStartIndex + i) % subCommLinkPara.remoteRankMap.size()]
                  .second;
        auto& transportRequest = singleSubCommTransport.transportRequests[index];
        auto& link = singleSubCommTransport.links[index];

        if (!transportRequest.isValid) {
            continue;
        }

        if (isBackup && !transportRequest.isUsedRdma) {
            // 备用链路不需要创建p2p
            HCCL_INFO(
                "[%s]: no need to check p2p backup link, remoteUserRank[%u], userRank[%u], "
                "isUsedRdma[%u], isBackup[%d]",
                __func__, transportRequest.remoteUserRank, userRank_, transportRequest.isUsedRdma, isBackup);
            continue;
        }

        if (link == nullptr) {
            HCCL_ERROR(
                "[Create]errNo[0x%016llx] transport create fail in thread, local rank[%d] remote rank[%d], "
                "inputMemType[%d], outputMemType[%d]",
                HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), userRank_, transportRequest.remoteUserRank,
                transportRequest.inputMemType, transportRequest.outputMemType);
            SaluSleep(EXCEPTION_DELAY_US_COUNT);
            (void)notifyPool_->UnregisterOp(tag);
            return HCCL_E_NOT_FOUND;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TransportManager::AllocSubCommLinks(
    const std::string& tag, const TransportIOMem& transMem, struct SingleSubCommTransport& singleSubCommTransport,
    bool isAicpuModeEn, bool isBackup, u32 subCommIndex, bool isCapture, const HcclCMDType& opType, bool isIndOp,
    bool isNpuDirectRoce)
{
    const u32 offset = 8;
    std::vector<std::pair<u32, u32>> remoteRankMap;

    for (u32 i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {
        TransportRequest TemptransportRequest = singleSubCommTransport.transportRequests[i];
        bool tempIsInterRdma = false;
        UpdateIsInterRdma(TemptransportRequest.remoteUserRank, tempIsInterRdma, TemptransportRequest.isUsedRdma);
        if (TemptransportRequest.isValid) {
            remoteRankMap.push_back(std::make_pair(TemptransportRequest.remoteUserRank, i));
            if ((rankInfoList_[TemptransportRequest.localUserRank].deviceType == DevType::DEV_TYPE_310P3
                 || isStandardCard_)
                && !tempIsInterRdma && !Is310PDevice()) {
                std::vector<u32> enableP2PDevices;
                enableP2PDevices.push_back(rankInfoList_[TemptransportRequest.remoteUserRank].devicePhyId);
                HCCL_INFO(
                    "[Create][DestSockets]localDevicePhyId[%u] enable p2p with remoteDevicePhyId[%u]",
                    rankInfoList_[TemptransportRequest.localUserRank].devicePhyId,
                    rankInfoList_[TemptransportRequest.remoteUserRank].devicePhyId);
                HcclResult ret = P2PMgmtPub::EnableP2P(enableP2PDevices);
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR(
                        "[Create][DestSockets]Enable P2P Failed, src devicePhyId[%d], dst devicePhyId[%d], ret[%u]",
                        rankInfoList_[TemptransportRequest.localUserRank].devicePhyId,
                        rankInfoList_[TemptransportRequest.remoteUserRank].devicePhyId, ret),
                    ret);
                enableP2PDevices_.push_back(rankInfoList_[TemptransportRequest.remoteUserRank].devicePhyId);
            }
        }
    }
    if (remoteRankMap.empty()) {
        HCCL_INFO("[%s] is empty", __func__);
        return HCCL_SUCCESS;
    }

    if (singleSubCommTransport.needVirtualLink) {
        // task多线程并行下发，根据当前transport创建vtransport信息
        CHK_RET(CreateVirturalTransport(singleSubCommTransport));
    }

    // sort remoteRankMap by remoteRank
    struct LessFirstElement {
        bool operator()(const std::pair<u32, u32>& a, const std::pair<u32, u32>& b) const { return a.first < b.first; }
    };
    std::sort(remoteRankMap.begin(), remoteRankMap.end(), LessFirstElement());
    std::vector<std::pair<u32, u32>> reversedRemoteRankMap(remoteRankMap);
    std::reverse(reversedRemoteRankMap.begin(), reversedRemoteRankMap.end());

    struct SubCommLinkPara nextSubCommLinkPara(singleSubCommTransport, remoteRankMap, 0, offset);
    struct SubCommLinkPara prevSubCommLinkPara(singleSubCommTransport, reversedRemoteRankMap, 0, offset);
    auto find_greater_than_key1 = [this](const std::pair<u32, u32>& pair) {
        return pair.first >= (this->userRank_);
    };
    auto find_less_than_key1 = [this](const std::pair<u32, u32>& pair) {
        return pair.first <= (this->userRank_);
    };
    auto nextIt = find_if(remoteRankMap.begin(), remoteRankMap.end(), find_greater_than_key1);
    auto prevIt = find_if(reversedRemoteRankMap.begin(), reversedRemoteRankMap.end(), find_less_than_key1);
    u32 rankNum = remoteRankMap.size();
    nextSubCommLinkPara.remoteRankIdStartIndex = std::distance(remoteRankMap.begin(), nextIt) % rankNum;
    prevSubCommLinkPara.remoteRankIdStartIndex = std::distance(reversedRemoteRankMap.begin(), prevIt) % rankNum;

    for (u32 i = 0; i < (rankNum / (FACTOR_NUM_TWO * offset)) + 1; i++) {
        if ((i == rankNum / (FACTOR_NUM_TWO * offset)) && (rankNum % (FACTOR_NUM_TWO * offset)) != 0) {
            nextSubCommLinkPara.remoteRankIdNum = (rankNum % (FACTOR_NUM_TWO * offset)) / FACTOR_NUM_TWO
                                                  + ((rankNum % (FACTOR_NUM_TWO * offset)) % FACTOR_NUM_TWO);
            prevSubCommLinkPara.remoteRankIdNum = (rankNum % (FACTOR_NUM_TWO * offset)) / FACTOR_NUM_TWO;
        }

        CHK_RET(createSubCommLinkThreads(
            tag, transMem, nextSubCommLinkPara, isAicpuModeEn, isBackup, subCommIndex, isCapture, opType, isIndOp,
            isNpuDirectRoce));
        CHK_RET(createSubCommLinkThreads(
            tag, transMem, prevSubCommLinkPara, isAicpuModeEn, isBackup, subCommIndex, isCapture, opType, isIndOp,
            isNpuDirectRoce));
        CHK_RET(waitSubCommLinkThreadsComplete(nextSubCommLinkPara));
        CHK_RET(waitSubCommLinkThreadsComplete(prevSubCommLinkPara));
        CHK_RET(checkSubCommLinkThreadsStatus(tag, nextSubCommLinkPara, isBackup));
        CHK_RET(checkSubCommLinkThreadsStatus(tag, prevSubCommLinkPara, isBackup));
        for (auto& tmpTag : socketTagVec_) {
            (void)socketManager_->DestroySockets(tmpTag);
        }
        socketTagVec_.clear();

        nextSubCommLinkPara.remoteRankIdStartIndex += offset;
        prevSubCommLinkPara.remoteRankIdStartIndex += offset;
    }

    return HCCL_SUCCESS;
}

HcclResult TransportManager::CreateBatchSendRecvLinks(
    const std::string& tag, const TransportIOMem& transMem, struct LinkPoolPara& linkPoolPara, bool isAicpuModeEn,
    bool isBackup, u32 subCommIndex, bool isCapture, const HcclCMDType& opType, bool isIndOp)
{
    HcclResult ret = hrtSetDevice(deviceLogicId_);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateBatchSendRecvLinks]hrtSetDevice failed, ret[%d]", ret);
        linkPoolPara.abortFlag = true;
        return ret;
    }
    struct SingleSubCommTransport& singleSubCommTransport = linkPoolPara.singleSubCommTransport;
    u32 currentIdx = 0;
    u32 requestIdx = 0;
    while (true) {
        currentIdx = linkPoolPara.taskIndex.fetch_add(1);
        if (currentIdx >= linkPoolPara.taskList.size() || linkPoolPara.abortFlag) {
            break;
        }
        requestIdx = linkPoolPara.taskList[currentIdx].second;

        auto& transportRequest = singleSubCommTransport.transportRequests[requestIdx];
        auto& link = singleSubCommTransport.links[requestIdx];

        // 无效请求、link已创建、备用链路，这三种情况不需要创建link
        if ((!transportRequest.isValid) || (link != nullptr) || (isBackup && !transportRequest.isUsedRdma)) {
            HCCL_INFO(
                "[%s]: no need to create p2p back link, remote UserRank[%u], userRank[%u], "
                "isUsedRdma[%u], isBackup[%d]",
                __func__, transportRequest.remoteUserRank, userRank_, transportRequest.isUsedRdma, isBackup);
            continue;
        }

        DeviceMem inputMem;
        DeviceMem outputMem;
        DeviceMem expMem;
        GetIOMem(transMem, transportRequest.inputMemType, transportRequest.outputMemType, inputMem, outputMem, expMem);
        HCCL_INFO(
            "[CreateBatchSendRecvLinks]transportRequest.inputMemType[%d] transportRequest.outputMemType[%d], "
            "isBackup[%d]",
            transportRequest.inputMemType, transportRequest.outputMemType, isBackup);

        IndOpMem indOpMem;
        if (isIndOp) {
            indOpMem = transMem.indOpMem;
            HCCL_DEBUG(
                "[CreateBatchSendRecvLinks]transportRequest indOpMem, userHostMem size[%llu], userDeviceMem size[%llu]",
                indOpMem.userHostMem.size(), indOpMem.userDeviceMem.size());
        }

        std::vector<std::shared_ptr<HcclSocket>> connectSockets;
        bool isInterRdma = false;
        bool chooseBackup = transportRequest.isUsedRdma ? isBackup : false;
        HcclNetDevCtx netDevCtx;
        {
            std::lock_guard<std::mutex> lock(createSocketMutex_);
            ret = CreateDestSockets(
                tag, transportRequest.remoteUserRank, singleSubCommTransport.taskNum, connectSockets, netDevCtx,
                isInterRdma, transportRequest.isUsedRdma, chooseBackup, subCommIndex, transportRequest.linkType);
        }
        HCCL_DEBUG(
            "[%s]CreateDestSockets finished, chooseBackup[%d], remoteUserRank[%u], userRank[%u], isUsedRdma[%u]",
            __func__, chooseBackup, transportRequest.remoteUserRank, userRank_, transportRequest.isUsedRdma);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[CreateBatchSendRecvLinks]Create dest sockets failed");
            linkPoolPara.linkResults[currentIdx] = ret;
            linkPoolPara.abortFlag = true;
            return ret;
        }

        MachineType machineType = transportRequest.localUserRank < transportRequest.remoteUserRank ?
                                      MachineType::MACHINE_SERVER_TYPE :
                                      MachineType::MACHINE_CLIENT_TYPE;
        std::string threadStr = (isInterRdma ? "HcclTerL_" : "HcclIntra_") + std::to_string(requestIdx);
        HCCL_INFO("[%s]threadStr[%s], poolName[%s]", __func__, threadStr.c_str(), linkPoolPara.poolName.c_str());
        ret = CreateLink(
            tag, hrtErrMGetErrorContextPub(), machineType, rankInfoList_[userRank_].serverId,
            transportRequest.remoteUserRank, singleSubCommTransport.supportDataReceivedAck,
            singleSubCommTransport.linkMode, singleSubCommTransport.enableUseOneDoorbell, threadStr, connectSockets,
            inputMem, outputMem, transportRequest.isUsedRdma, link, isAicpuModeEn, linkPoolPara.linkResults[currentIdx],
            netDevCtx, transportRequest.notifyNum, chooseBackup, isCapture, expMem, transportRequest.linkType, isIndOp,
            indOpMem, opType, false);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[CreateBatchSendRecvLinks]Create Link failed");
            linkPoolPara.linkResults[currentIdx] = ret;
            (void)hrtResetDevice(deviceLogicId_); // CreateLink会调用一次hrtSetDevice
            linkPoolPara.abortFlag = true;
            return ret;
        }
        ret = hrtResetDevice(deviceLogicId_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[CreateBatchSendRecvLinks]hrtResetDevice failed");
            linkPoolPara.linkResults[currentIdx] = ret;
            linkPoolPara.abortFlag = true;
            return ret;
        }
        singleSubCommTransport.status[requestIdx] = TransportStatus::READY; // 建链后 transport设置为ready状态
    }
    return HCCL_SUCCESS;
}

HcclResult TransportManager::WaitBatchSendRecvThreadsComplete(struct LinkPoolPara& linkPoolPara)
{
    for (u32 i = 0; i < linkPoolPara.linkThreads.size(); i++) {
        if (linkPoolPara.linkThreads[i] == nullptr || !linkPoolPara.linkThreads[i]->joinable()) {
            continue;
        }
        linkPoolPara.linkThreads[i]->join();     // 等待线程执行完毕
        CHK_RET(hrtResetDevice(deviceLogicId_)); // 防止线程里面异常退出，在进程中reset
    }
    linkPoolPara.linkThreads.clear();
    CHK_PRT_RET(GetStopFlag(), HCCL_ERROR("Terminating operation due to external request"), HCCL_E_INTERNAL);

    for (u32 i = 0; i < linkPoolPara.linkResults.size(); i++) {
        CHK_RET(linkPoolPara.linkResults[i]);
    }
    CHK_PRT_RET(
        linkPoolPara.abortFlag, HCCL_ERROR("[WaitBatchSendRecvThreadsComplete] abortFlag is set"), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult TransportManager::CheckBatchSendRecvLinkStatus(
    const std::string& tag, struct SingleSubCommTransport& singleSubCommTransport, bool isBackup)
{
    for (u32 i = 0; i < singleSubCommTransport.transportRequests.size(); ++i) {
        auto& transportRequest = singleSubCommTransport.transportRequests[i];
        if (transportRequest.isValid) {
            // 备用链路不需要创建p2p
            if (isBackup && !transportRequest.isUsedRdma) {
                HCCL_INFO(
                    "[%s]: no need to check p2p backup link, remoteUserRank[%u], userRank[%u], "
                    "isUsedRdma[%u], isBackup[%d]",
                    __func__, transportRequest.remoteUserRank, userRank_, transportRequest.isUsedRdma, isBackup);
                continue;
            }

            if (singleSubCommTransport.links[i] == nullptr) {
                HCCL_ERROR(
                    "[Create]errNo[0x%016llx] transport create fail in thread, local rank[%u] remote rank[%u], "
                    "inputMemType[%d], outputMemType[%d]",
                    HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), userRank_, transportRequest.remoteUserRank,
                    transportRequest.inputMemType, transportRequest.outputMemType);
                SaluSleep(EXCEPTION_DELAY_US_COUNT);
                (void)notifyPool_->UnregisterOp(tag);
                return HCCL_E_NOT_FOUND;
            }
        }
    }

    for (auto& tmpTag : socketTagVec_) {
        (void)socketManager_->DestroySockets(tmpTag);
    }
    socketTagVec_.clear();

    return HCCL_SUCCESS;
}

HcclResult TransportManager::PrepareTaskLists(
    HcclSendRecvItem* sendRecvItemsPtr, u32 itemNum, const SingleSubCommTransport& singleSubCommTransport,
    std::vector<std::pair<u32, u32>>& senderList, std::vector<std::pair<u32, u32>>& receiverList)
{
    if (sendRecvItemsPtr == nullptr || itemNum == 0) {
        HCCL_INFO("[%s] SendRecvItemsPtr is empty", __func__);
        return HCCL_SUCCESS;
    }

    std::unordered_set<u32> senderSet;
    std::unordered_set<u32> receiverSet;

    for (u32 i = 0; i < itemNum; ++i) {
        if (sendRecvItemsPtr[i].sendRecvType == HcclSendRecvType::HCCL_SEND) {
            receiverSet.insert(sendRecvItemsPtr[i].remoteRank);
        } else if (sendRecvItemsPtr[i].sendRecvType == HcclSendRecvType::HCCL_RECV) {
            senderSet.insert(sendRecvItemsPtr[i].remoteRank);
        }
    }

    for (u32 i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {
        if (singleSubCommTransport.transportRequests[i].isValid) {
            u32 remoteRank = singleSubCommTransport.transportRequests[i].remoteUserRank;
            bool isSender = senderSet.count(remoteRank);
            bool isReceiver = receiverSet.count(remoteRank);
            if (isSender && (!isReceiver || remoteRank < userRank_)) {
                senderList.emplace_back(std::make_pair(remoteRank, i));
            } else if (isReceiver) {
                receiverList.emplace_back(std::make_pair(remoteRank, i));
            }
        }
    }

    auto cmp = [](const std::pair<u32, u32>& a, const std::pair<u32, u32>& b) {
        return a.first < b.first;
    };
    std::sort(senderList.begin(), senderList.end(), cmp);
    std::sort(receiverList.begin(), receiverList.end(), cmp);

    return HCCL_SUCCESS;
}

HcclResult TransportManager::AllocBatchSendRecvLinks(
    HcclSendRecvItem* sendRecvItemsPtr, u32 itemNum, const std::string& tag, const TransportIOMem& transMem,
    struct SingleSubCommTransport& singleSubCommTransport, bool isAicpuModeEn, bool isBackup, u32 subCommIndex,
    bool isCapture, const HcclCMDType& opType, bool isIndOp)
{
    // 记录pair<remoteRank, idx>, idx表示remoteRank对应的建链信息在transportRequests中的索引位置
    std::vector<std::pair<u32, u32>> senderList;
    std::vector<std::pair<u32, u32>> receiverList;

    CHK_RET(PrepareTaskLists(sendRecvItemsPtr, itemNum, singleSubCommTransport, senderList, receiverList));
    if (senderList.empty() && receiverList.empty()) {
        HCCL_INFO("[%s] TransportRequests is empty", __func__);
        return HCCL_SUCCESS;
    }

    if (singleSubCommTransport.needVirtualLink) {
        // task多线程并行下发，根据当前transport创建vtransport信息
        CHK_RET(CreateVirturalTransport(singleSubCommTransport));
    }

    struct LinkPoolPara senderLinkPoolPara(singleSubCommTransport, "sender", senderList);
    struct LinkPoolPara receiverLinkPoolPara(singleSubCommTransport, "receiver", receiverList);
    for (u32 i = 0; i < senderLinkPoolPara.linkThreads.size(); ++i) {
        senderLinkPoolPara.linkThreads[i].reset(new (std::nothrow) std::thread(
            &TransportManager::CreateBatchSendRecvLinks, this, tag, std::ref(transMem), std::ref(senderLinkPoolPara),
            isAicpuModeEn, isBackup, subCommIndex, isCapture, opType, isIndOp));

        if (senderLinkPoolPara.linkThreads[i] == nullptr) {
            HCCL_ERROR("[AllocBatchSendRecvLinks] Failed to create sender thread %u", i);
            senderLinkPoolPara.abortFlag = true;
            WaitBatchSendRecvThreadsComplete(senderLinkPoolPara); // 清理已建立的线程
            return HCCL_E_PTR;
        }
    }
    for (u32 i = 0; i < receiverLinkPoolPara.linkThreads.size(); ++i) {
        receiverLinkPoolPara.linkThreads[i].reset(new (std::nothrow) std::thread(
            &TransportManager::CreateBatchSendRecvLinks, this, tag, std::ref(transMem), std::ref(receiverLinkPoolPara),
            isAicpuModeEn, isBackup, subCommIndex, isCapture, opType, isIndOp));

        if (receiverLinkPoolPara.linkThreads[i] == nullptr) {
            HCCL_ERROR("[AllocBatchSendRecvLinks] Failed to create receiver thread %u", i);
            receiverLinkPoolPara.abortFlag = true;
            senderLinkPoolPara.abortFlag = true;
            WaitBatchSendRecvThreadsComplete(senderLinkPoolPara);
            WaitBatchSendRecvThreadsComplete(receiverLinkPoolPara);
            return HCCL_E_PTR;
        }
    }

    CHK_RET(WaitBatchSendRecvThreadsComplete(senderLinkPoolPara));
    CHK_RET(WaitBatchSendRecvThreadsComplete(receiverLinkPoolPara));
    CHK_RET(CheckBatchSendRecvLinkStatus(tag, singleSubCommTransport, isBackup));

    return HCCL_SUCCESS;
}

HcclResult TransportManager::Alloc(
    const std::string& tag, const TransportIOMem& transMem, OpCommTransport& opTransportResponse, bool isAicpuModeEn,
    bool isBackup, bool isZeroCopy, const HcclCMDType& opType, bool isCapture, bool isIndOp, bool isNpuDirectRoce,
    const OpParam* opParam)
{
    std::lock_guard<std::mutex> lock(mutex_);
    CHK_RET(notifyPool_->RegisterOp(tag));
    workflowMode_ = GetWorkflowMode(); // 后续有起新的线程，因此更新一下workflowMode
    for (u32 levelIdx = 0; levelIdx < opTransportResponse.size(); levelIdx++) {
        auto& levelNSubCommTransport = opTransportResponse[levelIdx];
        u32 subCommIndex = 0;
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            subCommIndex++;
            DevType devType;
            CHK_RET(hrtGetDeviceType(devType));
            if (devType == DevType::DEV_TYPE_910_93) {
                // 如果是零拷贝场景下level0通信域交换零拷贝的共享内存
                if (levelIdx == COMM_LEVEL0 && isZeroCopy) {
                    for (auto& transportRequest : singleSubCommTransport.transportRequests) {
                        if (transportRequest.inputMemType != TransportMemType::RESERVED) {
                            transportRequest.inputMemType = TransportMemType::PARAM_INPUT;
                        }
                        if (transportRequest.outputMemType != TransportMemType::RESERVED) {
                            transportRequest.outputMemType = (opType == HcclCMDType::HCCL_CMD_BROADCAST) ?
                                                                 TransportMemType::PARAM_INPUT :
                                                                 TransportMemType::PARAM_OUTPUT;
                        }
                    }
                }
                if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
                    CHK_PTR_NULL(opParam);
                    CHK_RET(AllocBatchSendRecvLinks(
                        opParam->BatchSendRecvDataDes.sendRecvItemsPtr, opParam->BatchSendRecvDataDes.itemNum, tag,
                        transMem, singleSubCommTransport, isAicpuModeEn, isBackup, subCommIndex, isCapture, opType,
                        isIndOp));
                } else {
                    CHK_RET(AllocSubCommLinks(
                        tag, transMem, singleSubCommTransport, isAicpuModeEn, isBackup, subCommIndex, isCapture, opType,
                        isIndOp, isNpuDirectRoce));
                }
                continue;
            }

            std::vector<std::unique_ptr<std::thread>> linkThreads; // 建链所需线程
            std::vector<HcclResult> linkResult;                    // CreateLink返回值出参
            linkThreads.resize(singleSubCommTransport.transportRequests.size());
            linkResult.resize(linkThreads.size(), HCCL_SUCCESS);
            ThreadsGuard threadsGuard(linkThreads); // 确保异常退出场景析构时等待线程join
            u32 threadsRapplyNum{0};                // 线程使用计数器

            if (singleSubCommTransport.needVirtualLink) {
                // task多线程并行下发，根据当前transport创建vtransport信息
                CHK_RET(CreateVirturalTransport(singleSubCommTransport));
            }

            u32 linkIdx = 0;
            for (auto& transportRequest : singleSubCommTransport.transportRequests) {
                if (transportRequest.isValid && singleSubCommTransport.links[linkIdx] == nullptr) {
                    if (isBackup && !transportRequest.isUsedRdma) {
                        // 备用链路不需要创建p2p
                        HCCL_INFO(
                            "[%s]: no need to create p2p backup link, remoteUserRank[%u], userRank[%u], "
                            "isUsedRdma[%u], isBackup[%d]",
                            __func__, transportRequest.remoteUserRank, userRank_, transportRequest.isUsedRdma,
                            isBackup);
                        linkIdx++;
                        continue;
                    }
                    bool tempIsInterRdma = false;
                    UpdateIsInterRdma(transportRequest.remoteUserRank, tempIsInterRdma, transportRequest.isUsedRdma);
                    if ((rankInfoList_[transportRequest.localUserRank].deviceType == DevType::DEV_TYPE_310P3
                         || isStandardCard_)
                        && !tempIsInterRdma && !Is310PDevice()) {
                        std::vector<u32> enableP2PDevices;
                        enableP2PDevices.push_back(rankInfoList_[transportRequest.remoteUserRank].devicePhyId);
                        HCCL_INFO(
                            "[Alloc]localDevicePhyId[%u] enable p2p with remoteDevicePhyId[%u]",
                            rankInfoList_[transportRequest.localUserRank].devicePhyId,
                            rankInfoList_[transportRequest.remoteUserRank].devicePhyId);
                        HcclResult ret = P2PMgmtPub::EnableP2P(enableP2PDevices);
                        CHK_PRT_RET(
                            ret != HCCL_SUCCESS,
                            HCCL_ERROR(
                                "[Alloc]Enable P2P Failed, src devicePhyId[%d], dst devicePhyId[%d], ret[%u]",
                                rankInfoList_[transportRequest.localUserRank].devicePhyId,
                                rankInfoList_[transportRequest.remoteUserRank].devicePhyId, ret),
                            ret);
                        enableP2PDevices_.push_back(rankInfoList_[transportRequest.remoteUserRank].devicePhyId);
                    }
                    DeviceMem inputMem;
                    DeviceMem outputMem;
                    DeviceMem expMem;
                    HCCL_INFO(
                        "transportRequest.inputMemType[%d] transportRequest.outputMemType[%d], isBackup[%d]",
                        transportRequest.inputMemType, transportRequest.outputMemType, isBackup);
                    GetIOMem(
                        transMem, transportRequest.inputMemType, transportRequest.outputMemType, inputMem, outputMem,
                        expMem);

                    IndOpMem indOpMem;
                    if (isIndOp) {
                        indOpMem = transMem.indOpMem;
                        HCCL_DEBUG(
                            "transportRequest indOpMem, userHostMem size[%llu], userDeviceMem size[%llu]",
                            indOpMem.userHostMem.size(), indOpMem.userDeviceMem.size());
                    }

                    std::vector<std::shared_ptr<HcclSocket>> connectSockets;
                    bool isInterRdma;
                    HCCL_DEBUG(
                        "[%s]: remoteUserRank[%u], userRank[%u], isUsedRdma[%u], tag[%s]", __func__,
                        transportRequest.remoteUserRank, userRank_, transportRequest.isUsedRdma, tag.c_str());
                    bool chooseBackup = transportRequest.isUsedRdma ? isBackup : false;
                    HcclNetDevCtx netDevCtx;
                    HcclResult ret = CreateDestSockets(
                        tag, transportRequest.remoteUserRank, singleSubCommTransport.taskNum, connectSockets, netDevCtx,
                        isInterRdma, transportRequest.isUsedRdma, chooseBackup, subCommIndex,
                        transportRequest.linkType);
                    HCCL_DEBUG("[%s]CreateDestSockets finished, chooseBackup[%d]", __func__, chooseBackup);
                    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Alloc]Create dest sockets failed"), ret);

                    MachineType machineType = transportRequest.localUserRank < transportRequest.remoteUserRank ?
                                                  MachineType::MACHINE_SERVER_TYPE :
                                                  MachineType::MACHINE_CLIENT_TYPE;

                    if (transportRequest.isUsedRdma) {
                        HCCL_INFO(
                            "[%s]: create rdma link, remoteUserRank[%u], userRank[%u], "
                            "isBackup[%d], chooseBackup[%d], isInterRdma[%d]",
                            __func__, transportRequest.remoteUserRank, userRank_, isBackup, chooseBackup, isInterRdma);
                    }
                    bool chooseAivRoceDirect = transportRequest.isUsedRdma ? isNpuDirectRoce : false;
                    std::string threadStr
                        = (isInterRdma ? "HcclTerL_" : "HcclIntra_") + std::to_string(threadsRapplyNum);
                    linkThreads[threadsRapplyNum].reset(new (std::nothrow) std::thread(
                        &TransportManager::CreateLink, this, tag, hrtErrMGetErrorContextPub(), machineType,
                        rankInfoList_[userRank_].serverId, transportRequest.remoteUserRank,
                        singleSubCommTransport.supportDataReceivedAck, singleSubCommTransport.linkMode,
                        singleSubCommTransport.enableUseOneDoorbell, threadStr, connectSockets, inputMem, outputMem,
                        transportRequest.isUsedRdma, std::ref(singleSubCommTransport.links[linkIdx]), isAicpuModeEn,
                        std::ref(linkResult[threadsRapplyNum]), netDevCtx, transportRequest.notifyNum, chooseBackup,
                        isCapture, expMem, transportRequest.linkType, isIndOp, indOpMem, opType, chooseAivRoceDirect));
                    CHK_SMART_PTR_NULL(linkThreads[threadsRapplyNum]);               // 异常时其他线程待处理
                    singleSubCommTransport.status[linkIdx] = TransportStatus::READY; // 建链后 transport设置为ready状态
                    threadsRapplyNum++;
                }
                linkIdx++;
            }

            for (u32 index = 0; index < linkThreads.size(); index++) {
                if (linkThreads[index] == nullptr || !linkThreads[index]->joinable()) {
                    continue;
                }
                linkThreads[index]->join();              // 等待线程执行完毕
                CHK_RET(hrtResetDevice(deviceLogicId_)); // 防止线程里面异常退出，在进程中reset
            }
            linkThreads.clear();
            CHK_PRT_RET(GetStopFlag(), HCCL_ERROR("Terminating operation due to external request"), HCCL_E_INTERNAL);
            for (u32 index = 0; index < linkResult.size(); index++) {
                CHK_RET(linkResult[index]);
            }

            linkIdx = 0;
            for (auto& transportRequest : singleSubCommTransport.transportRequests) {
                if (transportRequest.isValid) {
                    if (isBackup && !transportRequest.isUsedRdma) {
                        // 备用链路不需要创建p2p
                        HCCL_INFO(
                            "[%s]: no need to check p2p backup link, remoteUserRank[%u], userRank[%u], "
                            "isUsedRdma[%u], isBackup[%d]",
                            __func__, transportRequest.remoteUserRank, userRank_, transportRequest.isUsedRdma,
                            isBackup);
                        linkIdx++;
                        continue;
                    }
                    if (singleSubCommTransport.links[linkIdx] == nullptr) {
                        HCCL_ERROR(
                            "[Create]errNo[0x%016llx] transport create fail in thread, local rank[%d] remote rank[%d]",
                            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), userRank_, transportRequest.remoteUserRank);
                        (void)ExceptionHandle(tag, opTransportResponse);
                        SaluSleep(EXCEPTION_DELAY_US_COUNT);
                        (void)notifyPool_->UnregisterOp(tag);
                        return HCCL_E_NOT_FOUND;
                    }
                }
                linkIdx++;
            }
            for (auto& tmpTag : socketTagVec_) {
                (void)socketManager_->DestroySockets(tmpTag);
            }
            socketTagVec_.clear();
        }
    }
    CHK_RET(notifyPool_->UnregisterOp(tag));
    return HCCL_SUCCESS;
}

HcclResult TransportManager::GetIncreRemoteRankList(
    OpCommTransport& opTransportReq, std::vector<u32>& rankList, TransportType transportType)
{
    for (u32 levelIndex = 0; levelIndex < opTransportReq.size(); levelIndex++) {
        for (u32 ringIndex = 0; ringIndex < opTransportReq[levelIndex].size(); ringIndex++) {
            SingleSubCommTransport& reqSingleSubComm = opTransportReq[levelIndex][ringIndex];
            for (u32 rankIndex = 0; rankIndex < reqSingleSubComm.transportRequests.size(); rankIndex++) {
                TransportRequest& transportRequest = reqSingleSubComm.transportRequests[rankIndex];
                CHK_PRT(AddremoteUserRankToList(transportRequest, rankList, transportType));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportManager::IncreAlloc(
    const std::string& tag, const TransportIOMem& transMem, OpCommTransport& opTransportReq,
    OpCommTransport& opTransportResponse, bool isAicpuModeEn, bool isBackup, bool isCapture, const HcclCMDType& opType)
{
    std::lock_guard<std::mutex> lock(mutex_);
    CHK_RET(notifyPool_->RegisterOp(tag));

    workflowMode_ = GetWorkflowMode();
    for (u32 levelIndex = 0; levelIndex < opTransportReq.size(); levelIndex++) {
        u32 subCommIndex = 0;
        for (u32 ringIndex = 0; ringIndex < opTransportReq[levelIndex].size(); ringIndex++) {
            subCommIndex++;
            std::vector<std::unique_ptr<std::thread>> linkThreads; // 建链所需线程
            std::vector<HcclResult> linkResult;                    // CreateLink返回值出参
            linkThreads.resize(opTransportReq[levelIndex][ringIndex].transportRequests.size());
            linkResult.resize(linkThreads.size(), HCCL_SUCCESS);
            ThreadsGuard threadsGuard(linkThreads); // 确保异常退出场景析构时等待线程join
            u32 threadsRapplyNum{0};                // 线程使用计数器
            SingleSubCommTransport& reqSingleSubComm = opTransportReq[levelIndex][ringIndex];
            SingleSubCommTransport& respSingleSubComm = opTransportResponse[levelIndex][ringIndex];
            for (u32 rankIndex = 0; rankIndex < reqSingleSubComm.transportRequests.size(); rankIndex++) {
                TransportRequest& transportRequest = reqSingleSubComm.transportRequests[rankIndex];
                CHK_PRT_RET(
                    rankIndex >= respSingleSubComm.links.size(),
                    HCCL_ERROR(
                        "[IncreAlloc] The remote rank_id[%u] is larger than the existent respSingleSubComm map "
                        "size[%u]",
                        rankIndex, respSingleSubComm.links.size()),
                    HCCL_E_PARA);
                if (respSingleSubComm.links[rankIndex] != nullptr
                    && respSingleSubComm.links[rankIndex]->GetLinkType() != hccl::LinkType::LINK_RESERVED) {
                    HCCL_INFO(
                        "[IncreAlloc] The link to remote userRank[%u] has existed", transportRequest.remoteUserRank);
                    continue;
                }
                if (transportRequest.isValid) {
                    if (isBackup && !transportRequest.isUsedRdma) {
                        // 备用链路不需要创建p2p
                        HCCL_INFO(
                            "[%s]: no need to create p2p backup link, remoteUserRank[%u], userRank[%u], "
                            "isUsedRdma[%u], isBackup[%d]",
                            __func__, transportRequest.remoteUserRank, userRank_, transportRequest.isUsedRdma,
                            isBackup);
                        continue;
                    }
                    bool tempIsInterRdma = false;
                    UpdateIsInterRdma(transportRequest.remoteUserRank, tempIsInterRdma, transportRequest.isUsedRdma);
                    if ((rankInfoList_[transportRequest.localUserRank].deviceType == DevType::DEV_TYPE_310P3
                         || isStandardCard_)
                        && !tempIsInterRdma && !Is310PDevice()) {
                        std::vector<u32> enableP2PDevices;
                        enableP2PDevices.push_back(rankInfoList_[transportRequest.remoteUserRank].devicePhyId);
                        HCCL_INFO(
                            "[IncreAlloc]localDevicePhyId[%u] enable p2p with remoteDevicePhyId[%u]",
                            rankInfoList_[transportRequest.localUserRank].devicePhyId,
                            rankInfoList_[transportRequest.remoteUserRank].devicePhyId);
                        HcclResult ret = P2PMgmtPub::EnableP2P(enableP2PDevices);
                        CHK_PRT_RET(
                            ret != HCCL_SUCCESS,
                            HCCL_ERROR(
                                "[IncreAlloc]Enable P2P Failed, src devicePhyId[%d], dst devicePhyId[%d], ret[%u]",
                                rankInfoList_[transportRequest.localUserRank].devicePhyId,
                                rankInfoList_[transportRequest.remoteUserRank].devicePhyId, ret),
                            ret);
                        enableP2PDevices_.push_back(rankInfoList_[transportRequest.remoteUserRank].devicePhyId);
                    }
                    respSingleSubComm.transportRequests[rankIndex] = transportRequest;
                    DeviceMem inputMem;
                    DeviceMem outputMem;
                    DeviceMem expMem;
                    GetIOMem(
                        transMem, transportRequest.inputMemType, transportRequest.outputMemType, inputMem, outputMem,
                        expMem);
                    HCCL_INFO(
                        "[IncreAlloc] transportRequest.inputMemType[%d] transportRequest.outputMemType[%d], "
                        "isBackup[%d]",
                        transportRequest.inputMemType, transportRequest.outputMemType, isBackup);

                    std::vector<std::shared_ptr<HcclSocket>> connectSockets;
                    bool isInterRdma;
                    bool chooseBackup = transportRequest.isUsedRdma ? isBackup : false;
                    HcclNetDevCtx netDevCtx;
                    HcclResult ret = CreateDestSockets(
                        tag, transportRequest.remoteUserRank, reqSingleSubComm.taskNum, connectSockets, netDevCtx,
                        isInterRdma, transportRequest.isUsedRdma, chooseBackup, subCommIndex,
                        transportRequest.linkType);
                    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[IncreAlloc]Create dest sockets failed"), ret);

                    MachineType machineType = transportRequest.localUserRank < transportRequest.remoteUserRank ?
                                                  MachineType::MACHINE_SERVER_TYPE :
                                                  MachineType::MACHINE_CLIENT_TYPE;
                    std::string threadStr
                        = (isInterRdma ? "HcclTerL_" : "HcclIntra_") + std::to_string(threadsRapplyNum);
                    bool isIndOp = false;
                    IndOpMem indOpMem;
                    linkThreads[threadsRapplyNum].reset(new (std::nothrow) std::thread(
                        &TransportManager::CreateLink, this, tag, hrtErrMGetErrorContextPub(), machineType,
                        rankInfoList_[userRank_].serverId, transportRequest.remoteUserRank,
                        reqSingleSubComm.supportDataReceivedAck, reqSingleSubComm.linkMode,
                        reqSingleSubComm.enableUseOneDoorbell, threadStr, connectSockets, inputMem, outputMem,
                        transportRequest.isUsedRdma, std::ref(respSingleSubComm.links[rankIndex]), isAicpuModeEn,
                        std::ref(linkResult[threadsRapplyNum]), netDevCtx, transportRequest.notifyNum, chooseBackup,
                        isCapture, expMem, transportRequest.linkType, isIndOp, indOpMem, opType, false));
                    CHK_SMART_PTR_NULL(linkThreads[threadsRapplyNum]);            // 异常时其他线程待处理
                    respSingleSubComm.status[rankIndex] = TransportStatus::READY; // 建链后 transport设置为ready状态
                    threadsRapplyNum++;
                }
            }
            for (u32 index = 0; index < linkThreads.size(); index++) {
                if (linkThreads[index] != nullptr && linkThreads[index]->joinable()) {
                    linkThreads[index]->join();
                    CHK_RET(hrtResetDevice(deviceLogicId_)); // 防止线程里面异常退出，在进程中reset
                }
            }
            linkThreads.clear();
            for (u32 index = 0; index < linkResult.size(); index++) {
                CHK_RET(linkResult[index]);
            }
            for (auto& tmpTag : socketTagVec_) {
                (void)socketManager_->DestroySockets(tmpTag);
            }
            socketTagVec_.clear();
        }
    }
    CHK_RET(notifyPool_->UnregisterOp(tag));
    return HCCL_SUCCESS;
}

bool TransportManager::IsHccsTransport(u32 remoteRank, TransportLinkType linkType)
{
    // 判断p2p连接中，与remoteRank间的链路是否为hccs链路
    bool isHccs = true;
    if (linkType == TransportLinkType::RESERVED) {
        // 非hccs sio并发场景，直接通过获取底层优选链路类型来判断，获取失败时，默认为HCCS
        LinkTypeInServer linkTypeTmp = LinkTypeInServer::RESERVED_LINK_TYPE;
        HcclResult ret = hrtGetPairDeviceLinkType(
            rankInfoList_[userRank_].devicePhyId, rankInfoList_[remoteRank].devicePhyId, linkTypeTmp);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "fail to get device link type for userRank[%u] remoteRank[%u] ret[%d], default to Hccs", userRank_,
                remoteRank, ret);
            return true;
        }
        if (linkTypeTmp == LinkTypeInServer::SIO_TYPE) {
            isHccs = false;
        }
    } else { // 910_93 2 die concurrent
        // hccs sio并发场景，直接通过linkType判断
        isHccs = linkType == TransportLinkType::HCCS;
    }

    return isHccs;
}

HcclResult TransportManager::ConstructTransTag(
    const std::string& tag, std::string& transTag, bool isInterRdma, u32 subCommIndex, bool isHccs)
{
    transTag = (Is310PDevice() || isHaveCpuRank_) ? tag : identifier_ + "_res_optimize_" + std::to_string(subCommIndex);
    if (isInterRdma) {
        transTag += "_Inter_";
    } else {
        transTag += isHccs ? "_Hccs_" : "_SIO_";
    }
    return HCCL_SUCCESS;
}

HcclResult TransportManager::GetIOMem(
    const TransportIOMem& transMem, const TransportMemType inputMemType, const TransportMemType outputMemType,
    DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& expMem)
{
    if (inputMemType == CCL_INPUT) {
        inputMem = transMem.cclInputMem;
    } else if (inputMemType == SCRATCH) {
        inputMem = transMem.scratchMem;
    } else if (inputMemType == PARAM_INPUT) {
        inputMem = transMem.paramInputMem;
    } else if (inputMemType == AIV_INPUT) {
        inputMem = transMem.aivInputMem;
    } else if (inputMemType == AIV_OUTPUT) {
        inputMem = transMem.aivOutputMem;
    } else if (inputMemType == CCL_OUTPUT) {
        inputMem = transMem.cclOutputMem;
    } else if (inputMemType == USER_MEM) {
        inputMem = transMem.userMem;
    } else {
        HCCL_ERROR("inputMemType is Invalid, inputMem not set");
        return HCCL_E_INTERNAL;
    }

    if (outputMemType == CCL_OUTPUT) {
        outputMem = transMem.cclOutputMem;
    } else if (outputMemType == SCRATCH) {
        outputMem = transMem.scratchMem;
    } else if (outputMemType == PARAM_OUTPUT) {
        outputMem = transMem.paramOutputMem;
    } else if (outputMemType == AIV_INPUT) {
        outputMem = transMem.aivInputMem;
    } else if (outputMemType == AIV_OUTPUT) {
        outputMem = transMem.aivOutputMem;
    } else if (outputMemType == CCL_INPUT) {
        outputMem = transMem.cclInputMem;
    } else if (outputMemType == PARAM_INPUT) {
        outputMem = transMem.paramInputMem;
    } else if (outputMemType == USER_MEM) {
        outputMem = transMem.userMem;
    } else {
        HCCL_ERROR("outputMemType is Invalid, inputMem not set");
        return HCCL_E_INTERNAL;
    }

    expMem = transMem.expMem;
    return HCCL_SUCCESS;
}

u32 TransportManager::GetHostPort(s32 devicePhyId)
{
    if (GetExternalInputHcclIfBasePort() == HCCL_INVALID_PORT) {
        return (devicePhyId + HOST_PARA_BASE_PORT);
    } else {
        return (devicePhyId + GetExternalInputHcclIfBasePort() + HCCL_AISERVER_DEVICE_NUM);
    }
}

u32 TransportManager::GetRemoteNicPort(s32 devicePhyId, u32 dstUserRank, bool isInterRdma)
{
    DevType devType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(devType));
    if ((nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST)
        && !(
            dstUserRank < nicRanksPort_.size() && nicRanksPort_[dstUserRank] != HCCL_INVALID_PORT
            && devType == DevType::DEV_TYPE_910B)) {
        return GetHostPort(devicePhyId);
    }

    // isUseRankPort_在ranksPort初始化时一同配置：1. 异构场景 2. 开启device侧端口配置
    // vnic port仅用于开启device侧端口配置时的sdma场景
    bool useVnicPort = devPortSwitchOn_ && !isInterRdma && !Is310PDevice();
    const std::vector<u32>& ranksPorts = useVnicPort ? vnicRanksPort_ : nicRanksPort_;
    return GetNicPort(devicePhyId, ranksPorts, dstUserRank, isUseRankPort_);
}

uint32_t TransportManager::GetConnectMode(RankId remoteRank)
{
    NICDeployment localNicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    NICDeployment remoteNicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    for (auto it : rankInfoList_) {
        if (it.userRank == remoteRank) {
            remoteNicDeploy = it.nicDeploy;
        }
        if (it.userRank == userRank_) {
            localNicDeploy = it.nicDeploy;
        }
    }
    return static_cast<uint32_t>(localNicDeploy != remoteNicDeploy);
}

HcclResult TransportManager::GetTransNewTag(
    const std::string& tag, std::string& newTag, RankId remoteRank, bool& isInterRdma, u32 subCommIndex,
    TransportLinkType linkType, HcclRankLinkInfo remoteLink, uint32_t mode)
{
    if (mode != 0) {
        if (userRank_ < remoteRank) {
            newTag = identifier_ + "_" + std::to_string(userRank_) + "_" + std::to_string(remoteRank) + "_"
                     + devIpAddr_[0].GetReadableIP() + "_" + remoteLink.ip.GetReadableIP();
        } else {
            newTag = identifier_ + "_" + std::to_string(remoteRank) + "_" + std::to_string(userRank_) + "_"
                     + remoteLink.ip.GetReadableIP() + "_" + devIpAddr_[0].GetReadableIP();
        }
    } else {
        bool isHccs = isInterRdma ? false : IsHccsTransport(remoteRank, linkType);
        CHK_RET(ConstructTransTag(tag, newTag, isInterRdma, subCommIndex, isHccs));
    }

    return HCCL_SUCCESS;
}

HcclResult TransportManager::CreateDestSockets(
    const std::string& tag, RankId remoteRank, u64 taskNum, std::vector<std::shared_ptr<HcclSocket>>& connectSockets,
    HcclNetDevCtx& netDevCtx, bool& isInterRdma, bool forceRdma, bool isBackup, u32 subCommIndex,
    TransportLinkType linkType)
{
    // 改对端的ip和port
    UpdateIsInterRdma(remoteRank, isInterRdma, forceRdma);
    HCCL_INFO(
        "[Create][DestSockets]UpdateIsInterRdma finished. local rank[%u], remote rank[%u],"
        "isInterRdma[%d], forceRdma[%d], tag[%s]",
        userRank_, remoteRank, isInterRdma, forceRdma, tag.c_str());

    u32 socketsPerLink = 1;
    if (isInterRdma) {
        if (!mulQpinfo_) {
            mulQpinfo_.reset(static_cast<MulQpInfo*>(new (std::nothrow) MulQpInfo()));
        }
        CHK_PRT_RET(!mulQpinfo_, HCCL_ERROR("[Init][Transport]In create mulQpinfo failed"), HCCL_E_PTR);
        CHK_RET(mulQpinfo_->Init(InitParams{
            nicDeployment_, static_cast<std::int32_t>(rankInfoList_[userRank_].devicePhyId),
            rankInfoList_[userRank_].deviceType}));
        socketsPerLink = GetSocketsPerLink(taskNum, remoteRank);
    }

    HcclRankLinkInfo remoteLinkInfo;
    MakeRemoteLinkInfo(remoteRank, isInterRdma, socketsPerLink, remoteLinkInfo);
    if (isBackup) {
        remoteLinkInfo.ip = rankInfoList_[remoteRank].backupNicIp[0];
        remoteLinkInfo.port = rankInfoList_[remoteRank].backupDevicePort == HCCL_INVALID_PORT ?
                                  AICPU_RETRY_BACKUP_PORT :
                                  rankInfoList_[remoteRank].backupDevicePort;
    }

    HCCL_INFO(
        "[%s] ip and port info. local rank[%u], remote rank[%u], isBackup[%d], port[%u], ip[%s]", __func__, userRank_,
        remoteRank, isBackup, remoteLinkInfo.port, remoteLinkInfo.ip.GetReadableIP());

    std::string newTag;
    uint32_t mode = GetConnectMode(remoteRank);
    GetTransNewTag(tag, newTag, remoteRank, isInterRdma, subCommIndex, linkType, remoteLinkInfo, mode);

    HcclResult ret = HCCL_SUCCESS;
    if (isInterRdma || Is310PDevice()) {
        netDevCtx = ((nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE)
                     || (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST && !devIpAddr_[0].IsInvalid())) ?
                        netDevCtxMap_[devIpAddr_[0]] :
                        netDevCtxMap_[hostIp_];
        if (isBackup && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
            netDevCtx = netDevCtxMap_[rankInfoList_[userRank_].backupNicIp[0]];
            HCCL_DEBUG(
                "[%s]refresh netDevCtx info. local rank[%u], remote rank[%u], isBackup[%d], port[%u], ip[%s]", __func__,
                userRank_, remoteRank, isBackup, remoteLinkInfo.port,
                (rankInfoList_[userRank_].backupNicIp[0]).GetReadableIP());
        }
        ret = socketManager_->CreateSingleLinkSocket(
            newTag, netDevCtx, remoteLinkInfo, connectSockets, false, false, 0, mode);
        if (!GetExternalInputHcclIsTcpMode()) {
            std::vector<std::string>::iterator iter = std::find(socketTagVec_.begin(), socketTagVec_.end(), newTag);
            if (iter == socketTagVec_.end()) {
                socketTagVec_.push_back(newTag);
            }
        }
    } else {
        // server内非异构场景，使能P2P
        bool isInterServer = false;
        CHK_PRT(IsInterServer(remoteRank, isInterServer));

        if (!isInterServer && !isHaveCpuRank_) {
            std::vector<u32> WaitP2PEnabledDevices;
            WaitP2PEnabledDevices.push_back(rankInfoList_[remoteRank].devicePhyId);
            HCCL_INFO(
                "[Create][DestSockets]localDevicePhyId[%u] wait p2p enable with remoteDevicePhyId[%u]",
                rankInfoList_[userRank_].devicePhyId, rankInfoList_[remoteRank].devicePhyId);
            HcclResult ret = P2PMgmtPub::WaitP2PEnabled(WaitP2PEnabledDevices, [this]() -> bool {
                return this->GetStopFlag();
            });
            if (ret != HCCL_SUCCESS) {
                if (ret == HCCL_E_DRV) {
                    RankInfo loaclRankInfo = rankInfoList_[userRank_];
                    RankInfo remoteRankInfo = rankInfoList_[remoteRank];
                    DetectConnectionAnomalies::GetInstance(deviceLogicId_)
                        .AddIpQueue(loaclRankInfo, remoteRankInfo, NicType::VNIC_TYPE, deviceLogicId_);
                }
                CHK_PRT_RET(
                    true,
                    HCCL_ERROR(
                        "[Create][DestSockets]Wait Enable P2P Failed, src devicePhyId[%d], dst devicePhyId[%d], "
                        "ret[%u]",
                        rankInfoList_[userRank_].devicePhyId, rankInfoList_[remoteRank].devicePhyId, ret),
                    ret);
            }
        }
        netDevCtx = netDevCtxMap_[localVnicIp_];
        ret = socketManager_->CreateSingleLinkSocket(newTag, netDevCtx, remoteLinkInfo, connectSockets, false, true);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Create][DestSockets]Create single link sockets failed, "
            "local rank[%u], remote rank[%u], isInterRdma[%d]",
            userRank_, remoteRank, isInterRdma),
        ret);
    return ret;
}

u32 TransportManager::GetSocketsPerLink(u64 taskNum, u32 remoteRankId)
{
    bool isEnableMulQp = false;
    CHK_RET(mulQpinfo_->IsEnableMulQp(isEnableMulQp));
    if (isEnableMulQp) {
        PortNum portNum;
        CHK_RET(mulQpinfo_->GetPortsNumByIpPair(
            portNum, remoteRankId >= rankInfoList_.size() ?
                         KeyPair() :
                         std::make_pair(rankInfoList_[userRank_].nicIp[0], rankInfoList_[remoteRankId].nicIp[0])));
        if (portNum > HCCL_QPS_PER_CONNECTION_DEFAULT) {
            SetMultiQpMode(dispatcher_, true);
            return MUL_QP_SOCKETS_PER_LINk;
        }
    }
    u32 socketsPerLink = 1;
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (taskNum == 0) {
            taskNum = 1;
        }
        socketsPerLink = (taskNum + (HCCP_SQ_TEMPLATE_CAPACITY - 1)) / HCCP_SQ_TEMPLATE_CAPACITY;
    }
    return socketsPerLink;
}

HcclResult TransportManager::CheckLinkNumAndSwitchLinkType(
    TransportType& type, MachinePara& machinePara, const std::vector<std::shared_ptr<HcclSocket>> sockets)
{
    u32 localCount = ibvCount_;
    u32 remoteCount = 0;
    CHK_RET(sockets[0]->Send(&localCount, sizeof(localCount)));
    CHK_RET(sockets[0]->Recv(&remoteCount, sizeof(remoteCount)));
    if (localCount > MASSIVE_IBV_CONNECTION_COUNT || remoteCount > MASSIVE_IBV_CONNECTION_COUNT) {
        // 走aicpu直驱时暂时不支持iscapture特性
        type = TransportType::TRANS_TYPE_DEVICE_DIRECT;
    }
    HCCL_INFO(
        "[TransportManager][CheckLinkNumAndSwitchLinkType] local ibvCount[%u], remote IbvCount[%u] "
        "localrank[%u], remoterank[%u], type[%d]",
        localCount, remoteCount, machinePara.localUserrank, machinePara.remoteUserrank, type);
    return HCCL_SUCCESS;
}

HcclResult TransportManager::PrintErrorInfo(NicType nicType)
{
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType != DevType::DEV_TYPE_910_93) {
        return HCCL_SUCCESS;
    }
    std::string nicTypeStr;
    switch (nicType) {
        case NicType::VNIC_TYPE:
            nicTypeStr = "VNIC_TYPE";
            break;
        case NicType::DEVICE_NIC_TYPE:
            nicTypeStr = "DEVICE_NIC_TYPE";
            break;
        case NicType::HOST_NIC_TYPE:
            nicTypeStr = "HOST_NIC_TYPE";
            break;
        default:
            nicTypeStr = "unknown";
    }
    s64 phySuperPodId;
    CHK_RET(hrtGetDeviceInfo(
        deviceLogicId_, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
        HcclRtDeviceInfoType::HCCL_INFO_TYPE_SUPER_POD_ID, phySuperPodId));
    std::string logicSuperPodId = GetExternalInputLogicSuperPodId();
    if (logicSuperPodId.empty()) {
        HCCL_ERROR(
            "[TransportManager][%s]local rank information: nicType[%s], logicSuperPodId is not set, "
            "phySuperPodId[%lld].",
            __func__, nicTypeStr.c_str(), phySuperPodId);
    } else {
        HCCL_ERROR(
            "[TransportManager][%s]local rank information: nicType[%s], logicSuperPodId[%s], phySuperPodId[%lld]. "
            "Note: Do not "
            "configure ranks belonging to different physical superpod ID into a single logical superpod ID",
            __func__, nicTypeStr.c_str(), logicSuperPodId.c_str(), phySuperPodId);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportManager::CreateLink(
    const std::string& tag, const ErrContextPub& error_context, const MachineType machineType,
    const std::string& serverId, const u32 remoteRank, const bool supportDataReceivedAck, const LinkMode linkMode,
    const bool enableUseOneDoorbell, const std::string threadStr,
    const std::vector<std::shared_ptr<HcclSocket>> sockets, const DeviceMem inputMem, const DeviceMem outputMem,
    bool isUsedRdma, std::shared_ptr<Transport>& link, bool isAicpuModeEn, HcclResult& retOut,
    const HcclNetDevCtx& netDevCtx, u32 notifyNum, bool isBackup, bool isCapture, const DeviceMem expMem,
    TransportLinkType linkType, bool isIndOp, const IndOpMem indOpMem, const HcclCMDType& opType, bool isNpuDirectRoce)
{
    hrtErrMSetErrorContextPub(error_context);
    // 给当前线程添加名字
    SetThreadName(threadStr);
    link = nullptr;
    retOut = hrtSetDevice(deviceLogicId_);
    CHK_RET(retOut);

    SetWorkflowMode(workflowMode_); // 更新本线程的workflow

    MachinePara machinePara;
    RankInfo loaclRankInfo = rankInfoList_[userRank_];
    RankInfo remoteRankInfo = rankInfoList_[remoteRank];
    HcclResult ret = HCCL_SUCCESS;
    do {
        ret = SetMachinePara(
            tag, machineType, serverId, remoteRank, supportDataReceivedAck, linkMode, sockets, inputMem, outputMem,
            expMem, isAicpuModeEn, isBackup, isCapture, notifyNum, trafficClass_, serviceLevel_, machinePara,
            loaclRankInfo, remoteRankInfo, netDevCtx, linkType, indOpMem, isIndOp, opType, isNpuDirectRoce);
        retOut = ret;
        std::string tmpErrInfo = ret == HCCL_E_TIMEOUT ? LOG_KEYWORDS_TIMEOUT : LOG_KEYWORDS_RUN_FAILED;
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s][%s][%s]SetMachinePara error.", __func__, LOG_KEYWORDS_INIT_CHANNEL.c_str(),
                tmpErrInfo.c_str()), );

        HCCL_INFO(
            "inputMem[%p],outputMem[%p], inputMem size[%llu], outputMem size[%llu]", inputMem.ptr(), outputMem.ptr(),
            inputMem.size(), outputMem.size());
        if (isIndOp) {
            HCCL_DEBUG(
                "userHostMem num[%llu], userDeviceMem num[%llu]", indOpMem.userHostMem.size(),
                indOpMem.userDeviceMem.size());
        }
        HCCL_INFO(
            "[createLink para]tag[%s], rank[%u]-localUserrank[%u]-localIpAddr[%s], linkMode[%d] "
            "dst_rank[%u]-remoteUserrank[%u]-remote_ip_addr[%s], machineType[%d], serverId[%s], "
            "nicDeploy[%d], isBackup[%d], opType[%d]",
            tag.c_str(), userRank_, rankInfoList_[userRank_].worldRank, rankInfoList_[userRank_].serverId.c_str(),
            machinePara.linkMode, remoteRank, rankInfoList_[remoteRank].worldRank,
            rankInfoList_[remoteRank].serverId.c_str(), machinePara.machineType, machinePara.serverId.c_str(),
            machinePara.nicDeploy, isBackup, opType);
        // transport初始化
        TransportType type = TransportType::TRANS_TYPE_RESERVED;
        CHK_PRT(GetTransportType(remoteRank, isUsedRdma, type));
        // A2/A3 batch_send_recv 走roce才切换到新链路
        if (type == TransportType::TRANS_TYPE_IBV_EXP && opType_ == HCCL_CMD_BATCH_SEND_RECV) {
            ret = CheckLinkNumAndSwitchLinkType(type, machinePara, sockets);
            retOut = ret;
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[%s]errNo[0x%016llx]CheckLinkNumAndSwitchLinkType error.", __func__, HCCL_ERROR_CODE(ret)), );
            std::lock_guard<std::mutex> lock(ibvCountMutex_);
            // 之前已经经过链了，则使用之前的老链路，bsr会建2条链路
            if (remoteTransportMap_.find(machinePara.remoteUserrank) != remoteTransportMap_.end()) {
                type = remoteTransportMap_[machinePara.remoteUserrank];
                HCCL_INFO(
                    "[TransportManager][CreateLink] use the same type as before, localRank %u remoteRank %u type %d",
                    machinePara.localUserrank, machinePara.remoteUserrank, type);
            } else {
                remoteTransportMap_.insert(std::make_pair(machinePara.remoteUserrank, type));
                HCCL_INFO(
                    "[TransportManager][CreateLink] transportMap save remoterank %u type %d",
                    machinePara.remoteUserrank, type);
            }
            if (type == TransportType::TRANS_TYPE_IBV_EXP) {
                ibvCount_++;
            } else if (type == TransportType::TRANS_TYPE_DEVICE_DIRECT) {
                machinePara.qpMode = QPMode::NORMAL;
                machinePara.queueDepthAttr.recvCqDepth = RECV_QP_DEPTH_FOR_BSR;
                machinePara.queueDepthAttr.rqDepth = RECV_QP_DEPTH_FOR_BSR;
                machinePara.queueDepthAttr.sendCqDepth = SEND_QP_DEPTH_FOR_BSR;
                machinePara.queueDepthAttr.sqDepth = SEND_QP_DEPTH_FOR_BSR;
            }
        }
        ret = TransportInit(machinePara, link, enableUseOneDoorbell, type);
        retOut = ret;
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx]TransportInit error.", __func__, HCCL_ERROR_CODE(ret)), );
    } while (0);

    if (ret != HCCL_SUCCESS) {
        link = nullptr;
        retOut = ret;
        if (ret == HCCL_E_MEMORY) {
            std::string err_str
                = "[Create][DestLink]Transport init error! IPC memory allocation failed due to "
                  "possible memory limit exceeded. Suggested solution: Use 3TB / (ranksize * 2) as the upper limit of "
                  "HCCL_BUFFSIZE.";
            HCCL_ERROR(
                "[%s][%s]%s", LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_RUN_FAILED.c_str(), err_str.c_str());
        }

        NicType nicType = sockets[0]->GetSocketType();
        DetectConnectionAnomalies::GetInstance(deviceLogicId_)
            .AddIpQueue(loaclRankInfo, remoteRankInfo, nicType, deviceLogicId_);

        char errorLogBuffer[LOG_TMPBUF_SIZE];
        s32 stringRet = snprintf_s(
            errorLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "createLink para:rank[%u]-localUserrank[%u]-localIpAddr[%s/%d], "
            "remoteRank[%u]-remoteUserrank[%u]-remoteIpAddr[%s/%d], "
            "machineType[%d], linkMode[%d], isUsedRdma[%d], tag[%s]",
            userRank_, rankInfoList_[userRank_].worldRank, rankInfoList_[userRank_].serverId.c_str(),
            rankInfoList_[userRank_].devicePhyId, remoteRank, rankInfoList_[remoteRank].worldRank,
            rankInfoList_[remoteRank].serverId.c_str(), rankInfoList_[remoteRank].devicePhyId, machinePara.machineType,
            machinePara.linkMode, isUsedRdma, machinePara.tag.c_str());
        CHK_PRT_CONT(stringRet == -1, HCCL_ERROR("[Create][DestLink]Transport init error! Failed to build log info"));
        std::string tmpErrInfo = ret == HCCL_E_TIMEOUT ? LOG_KEYWORDS_TIMEOUT : LOG_KEYWORDS_RUN_FAILED;
        HCCL_ERROR(
            "[%s][%s]Transport init error! %s", LOG_KEYWORDS_INIT_CHANNEL.c_str(), tmpErrInfo.c_str(), errorLogBuffer);
        CHK_PRT(PrintErrorInfo(nicType));
        return ret;
    }
    HCCL_INFO(
        "[createLink success]:rank[%u]-localUserrank[%u]-localIpAddr[%s], "
        "dst_rank[%u]-remoteUserrank[%u]-remote_ip_addr[%s], tag[%s]",
        userRank_, rankInfoList_[userRank_].worldRank, rankInfoList_[userRank_].serverId.c_str(), remoteRank,
        rankInfoList_[remoteRank].worldRank, rankInfoList_[remoteRank].serverId.c_str(), machinePara.tag.c_str());

    return HCCL_SUCCESS;
}

HcclResult TransportManager::SetMachinePara(
    const std::string& tag, MachineType machineType, const std::string& serverId, u32 dstRank,
    const bool supportDataReceivedAck, const LinkMode linkMode,
    const std::vector<std::shared_ptr<HcclSocket>>& socketList, const DeviceMem& inputMem, const DeviceMem& outputMem,
    const DeviceMem& expMem, bool isAicpuModeEn, bool isBackup, bool isCapture, u32 notifyNum, u32 trafficClass,
    u32 serviceLevel, MachinePara& machinePara, RankInfo& loaclRank, RankInfo& remoteRank,
    const HcclNetDevCtx& netDevCtx, TransportLinkType linkType, const IndOpMem& indOpMem, bool isIndOp,
    const HcclCMDType& opType, bool isNpuDirectRoce)
{
    machinePara.notifyNum = notifyNum;
    machinePara.linkMode = linkMode;
    machinePara.machineType = machineType;
    machinePara.serverId = serverId;
    machinePara.localUserrank = rankInfoList_[userRank_].userRank;
    machinePara.remoteUserrank = rankInfoList_[dstRank].userRank;
    machinePara.localWorldRank = rankInfoList_[userRank_].worldRank;
    machinePara.remoteWorldRank = rankInfoList_[dstRank].worldRank;
    machinePara.collectiveId = identifier_;
    machinePara.deviceType = static_cast<DevType>(rankInfoList_[dstRank].deviceType);
    machinePara.inputMem = inputMem;
    machinePara.outputMem = outputMem;
    machinePara.tc = trafficClass;
    machinePara.sl = serviceLevel;
    if (expMem.ptr() != nullptr) {
        machinePara.mem.push_back(expMem);
    } else {
        machinePara.mem.clear();
    }
    if (isIndOp) {
        machinePara.userHostMem = indOpMem.userHostMem;
        machinePara.userDeviceMem = indOpMem.userDeviceMem;
    }
    machinePara.isIndOp = isIndOp;
    machinePara.linkAttribute = 0x03; /* 0x03同时支持目的端和源端发起 */
    machinePara.tag = tag;
    if (isBackup) {
        machinePara.localIpAddr = rankInfoList_[userRank_].backupNicIp[0];
        machinePara.remoteIpAddr = rankInfoList_[dstRank].backupNicIp[0];
        u32 localDevBackUpPhyId;
        CHK_RET(hrtGetPairDevicePhyId(rankInfoList_[userRank_].devicePhyId, localDevBackUpPhyId));
        machinePara.localDeviceId = static_cast<s32>(localDevBackUpPhyId);
        u32 remoteDevBackUpPhyId;
        CHK_RET(hrtGetPairDevicePhyId(rankInfoList_[dstRank].devicePhyId, remoteDevBackUpPhyId));
        machinePara.remoteDeviceId = static_cast<s32>(remoteDevBackUpPhyId);
        HCCL_DEBUG(
            "[%s]isBackup[%d], machinePara.localIpAddr[%s], machinePara.remoteIpAddr[%s], "
            "machinePara.localDeviceId[%d], machinePara.remoteDeviceId[%d].",
            __func__, isBackup, machinePara.localIpAddr.GetReadableIP(), machinePara.remoteIpAddr.GetReadableIP(),
            machinePara.localDeviceId, machinePara.remoteDeviceId);
    } else {
        machinePara.localIpAddr = rankInfoList_[userRank_].nicIp[0];
        machinePara.remoteIpAddr = rankInfoList_[dstRank].nicIp[0];
        machinePara.localDeviceId = rankInfoList_[userRank_].devicePhyId;
        machinePara.remoteDeviceId = rankInfoList_[dstRank].devicePhyId;
        HCCL_DEBUG(
            "[%s]isBackup[%d], machinePara.localIpAddr[%s], machinePara.remoteIpAddr[%s], "
            "machinePara.localDeviceId[%d], machinePara.remoteDeviceId[%d].",
            __func__, isBackup, machinePara.localIpAddr.GetReadableIP(), machinePara.remoteIpAddr.GetReadableIP(),
            machinePara.localDeviceId, machinePara.remoteDeviceId);
    }
    // 把原来的两层vector变成一层, 方便后继调用
    if (socketList.size() > 0) {
        std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> socketsMap;
        socketsMap[dstRank] = socketList;
        std::map<u32, u32> dstRankToUserRank;
        dstRankToUserRank[dstRank] = dstRank;
        CHK_RET(socketManager_->WaitLinksEstablishCompleted(
            socketList[0]->GetLocalRole(), socketsMap, dstRankToUserRank, loaclRank, remoteRank, netDevCtx));
        machinePara.sockets = socketList;
    }
    machinePara.exchangeInfo.resize(rankConsistentDataLength_);
    CHK_RET(RankConsistentcyChecker::GetInstance().GetCheckFrame(
        &machinePara.exchangeInfo[0], rankConsistentDataLength_, tag));
    machinePara.supportDataReceivedAck = supportDataReceivedAck; /* NeedDataReceivedAck(); */
    machinePara.nicDeploy = nicDeployment_;
    machinePara.localSocketPort = rankInfoList_[userRank_].hostPort;
    machinePara.remoteSocketPort = rankInfoList_[dstRank].hostPort;
    if (isBackup) {
        u32 tempDevBackUpPhyId;
        CHK_RET(hrtGetPairDevicePhyId(rankInfoList_[userRank_].devicePhyId, tempDevBackUpPhyId));
        u32 tempDevBackUpLogicId;
        CHK_RET(hrtGetDeviceIndexByPhyId(tempDevBackUpPhyId, tempDevBackUpLogicId));
        machinePara.deviceLogicId = static_cast<s32>(tempDevBackUpLogicId);
        HCCL_DEBUG("[%s]isBackup[%d], machinePara.deviceLogicId[%d].", __func__, isBackup, machinePara.deviceLogicId);
    } else {
        machinePara.deviceLogicId = deviceLogicId_;
        HCCL_DEBUG("[%s]isBackup[%d], machinePara.deviceLogicId[%d].", __func__, isBackup, machinePara.deviceLogicId);
    }

    machinePara.srcPorts = std::vector<std::uint16_t>(1, 0); /* 默认填充一个元素，0代表默认不配置 */
    machinePara.isAicpuModeEn = isAicpuModeEn;
    if (linkType == TransportLinkType::RESERVED) {
        // 非910_93 2die sio与hccs并发场景，specifyLink设置为RESERVED_LINK_TYPE，平台层将按实际链路类型建链
        machinePara.specifyLink = LinkTypeInServer::RESERVED_LINK_TYPE;
    } else {
        // 910_93 2die sio与hccs并发场景，
        // 并发链路中的的hccs链路specifyLink设置为HCCS_SW_TYPE，平台层将使用hccs链路来建链；
        // 并发链路中的的sio链路specifyLink设置为SIO_TYPE，平台层将使用sio链路来建链
        machinePara.specifyLink
            = (linkType == TransportLinkType::SIO) ? LinkTypeInServer::SIO_TYPE : LinkTypeInServer::HCCS_SW_TYPE;
    }

    if (isCapture) {
        machinePara.qpMode = QPMode::OFFLOAD;
    }

    if (isNpuDirectRoce) {
        // AIV ROCE直驱场景，需要将QPMode更改为NORMAL模式，以避免底层走入stars调度的下发流程
        machinePara.qpMode = QPMode::NORMAL;
    }

    // reduce相关算子需要使能atomic write能力，用于实现rdma wqe(reduce+record)保序
    bool isReduceOp = (opType == HCCL_CMD_ALLREDUCE) || (opType == HCCL_CMD_REDUCE)
                      || (opType == HCCL_CMD_REDUCE_SCATTER) || (opType == HCCL_CMD_REDUCE_SCATTER_V);
    bool isSupportAtomicWrite = false;
    CHK_RET(IsSupportAtomicWrite(machinePara.deviceType, machinePara.localDeviceId, isSupportAtomicWrite));
    machinePara.enableAtomicWrite = isSupportAtomicWrite && isReduceOp;
    HCCL_DEBUG(
        "%s enableAtomicWrite[%d], opType[%d], isSupportAtomicWrite[%d]", __func__, machinePara.enableAtomicWrite,
        opType, isSupportAtomicWrite);
    return HCCL_SUCCESS;
}

HcclResult TransportManager::GetTransportType(const u32 dstRank, bool isUsedRdma, TransportType& transportType)
{
    // 判断是否在同一个server
    bool isInterServer = false;
    CHK_PRT(IsInterServer(dstRank, isInterServer));

    if (!isInterServer) {
        LinkTypeInServer linkType = LinkTypeInServer::RESERVED_LINK_TYPE;
        CHK_RET(hrtGetPairDeviceLinkType(
            rankInfoList_[userRank_].devicePhyId, rankInfoList_[dstRank].devicePhyId, linkType));
        if (isUsedRdma) {
            transportType = TransportType::TRANS_TYPE_IBV_EXP;
        } else {
            transportType = TransportType::TRANS_TYPE_P2P;
        }
    } else { // server间
        if ((!isUsedRdma) && IsSupportInterHccs(dstRank)) {
            // 超节点内节点间走HCCS通信
            transportType = TransportType::TRANS_TYPE_P2P;
        } else {
            transportType = TransportType::TRANS_TYPE_IBV_EXP;
        }
    }

    HCCL_INFO("GetTransportType: srcRank[%u], dstRank[%u], transport_type[%d].", userRank_, dstRank, transportType);
    return HCCL_SUCCESS;
}

void TransportManager::SetTransportParam(TransportPara& para)
{
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    para.timeout = kdefaultTimeout;
    para.virtualFlag = false;
}

HcclResult TransportManager::TransportInit(
    MachinePara& machinePara, std::shared_ptr<Transport>& link, bool useOneDoorbell, TransportType type)
{
    // 实例化TransportBase
    TransportPara para{};
    SetTransportParam(para);

    if (type == TransportType::TRANS_TYPE_P2P) {
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else if (type == TransportType::TRANS_TYPE_IBV_EXP) {
        bool isEnableMulQp = false;
        CHK_RET(mulQpinfo_->IsEnableMulQp(isEnableMulQp));
        if (isEnableMulQp) {
            CHK_RET(mulQpinfo_->GetSpecialSourcePortsByIpPair(
                machinePara.srcPorts, std::make_pair(machinePara.localIpAddr, machinePara.remoteIpAddr)));
        }
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else if (type == TransportType::TRANS_TYPE_DEVICE_DIRECT) {
        bool isEnableMulQp = false;
        CHK_RET(mulQpinfo_->IsEnableMulQp(isEnableMulQp));
        if (isEnableMulQp) {
            CHK_RET(mulQpinfo_->GetSpecialSourcePortsByIpPair(
                machinePara.srcPorts, std::make_pair(machinePara.localIpAddr, machinePara.remoteIpAddr)));
        }
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else {
        HCCL_ERROR("[Init][Transport]not supported transport type");
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PRT_RET(!link, HCCL_ERROR("[Init][Transport]In create link, new link failed"), HCCL_E_PTR);

    if (useOneDoorbell) {
        link->EnableUseOneDoorbell();
    }

    CHK_RET(link->Init());
    // 算子一致性校验
    std::vector<u8> recvData = link->GetExchangeInfo();
    if (recvData.size() != 0) {
        CHK_PRT_RET(
            recvData.size() != machinePara.exchangeInfo.size(),
            HCCL_ERROR(
                "[Check][ExchangeInfo]remote exchangInfo size[%zu], local exchangeInfo size[%zu]", recvData.size(),
                machinePara.exchangeInfo.size()),
            HCCL_E_INTERNAL);
        CHK_RET(RankConsistentcyChecker::GetInstance().CheckFrameRecv(
            &recvData[0], recvData.size(), machinePara.tag.c_str()));
    }
    return HCCL_SUCCESS;
}

bool TransportManager::IsSupportInterHccs(const u32 dstRank)
{
    // 仅判断超节点内, 兼容打平通信域同时有server内和server间, 因此不判断server_id
    bool isInterHccsDisable = GetExternalInputInterHccsDisable();
    const std::string& curSuperPodId = rankInfoList_[userRank_].superPodId;
    const std::string& dstSuperPodId = rankInfoList_[dstRank].superPodId;

    bool isInterHccs = isInterHccsDisable == false && useSuperPodMode_ == true && curSuperPodId.empty() == false
                       && curSuperPodId == dstSuperPodId;

    HCCL_INFO(
        "[IsSupportInterHccs] rank[%u], superPodId[%s], dstRank[%u], dstSuperPodId[%s], useSuperPodMode[%d], "
        "isInterHccsDisable[%d], isInterHccs[%d]",
        userRank_, curSuperPodId.c_str(), dstRank, dstSuperPodId.c_str(), useSuperPodMode_, isInterHccsDisable,
        isInterHccs);
    return isInterHccs;
}

void TransportManager::UpdateIsInterRdma(const u32 remoteRank, bool& isInterRdma, bool forceRdma) // 待确认判断是否完善
{
    // 超节点内节点间采用HCCS通信的, 放至dstIntraClientVec_, 采用p2p建链
    bool isInterHccs = IsSupportInterHccs(remoteRank);
    if (isInterHccs && (!forceRdma)) {
        isInterRdma = false;
    } else if (rankInfoList_[userRank_].serverId != rankInfoList_[remoteRank].serverId) {
        isInterRdma = true;
    } else {
        LinkTypeInServer linkType;
        hrtGetPairDeviceLinkType(rankInfoList_[userRank_].devicePhyId, rankInfoList_[remoteRank].devicePhyId, linkType);
        isInterRdma = (isUsedRdmaLevel0_ && linkType == LinkTypeInServer::PXI_TYPE) || forceRdma;
    }
}

HcclResult TransportManager::MakeRemoteLinkInfo(
    const u32 remoteRank, bool isInterRdma, u32 socketsPerLink, HcclRankLinkInfo& remoteLinkInfo)
{
    RankInfo dstRankInfo = rankInfoList_[remoteRank];
    remoteLinkInfo.userRank = dstRankInfo.userRank;
    remoteLinkInfo.devicePhyId = dstRankInfo.devicePhyId;
    if (isInterRdma || Is310PDevice()) {
        remoteLinkInfo.ip = dstRankInfo.nicIp[0];
        remoteLinkInfo.port = GetRemoteNicPort(remoteLinkInfo.devicePhyId, dstRankInfo.userRank, isInterRdma);
        remoteLinkInfo.socketsPerLink = socketsPerLink;
    } else {
        remoteLinkInfo.ip = HcclIpAddress(dstRankInfo.devicePhyId);
        if (useSuperPodMode_) {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                rankInfoList_[userRank_].devicePhyId, DeviceIdType::DEVICE_ID_TYPE_SDID,
                rankInfoList_[remoteRank].superDeviceId, remoteLinkInfo.ip));
        } else {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                rankInfoList_[userRank_].devicePhyId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID,
                rankInfoList_[remoteRank].devicePhyId, remoteLinkInfo.ip));
        }
        remoteLinkInfo.port = GetRemoteNicPort(
            rankInfoList_[remoteRank].devicePhyId, rankInfoList_[remoteRank].userRank, isInterRdma); // ?
        remoteLinkInfo.socketsPerLink = socketsPerLink;
    }
    HCCL_INFO(
        "[TransportManager][MakeRemoteLinkInfo] isInterRdma[%u], is310PDevice[%u], "
        "remote rank: userRank[%u], devPhyId[%u], ip[%s], port[%u], socketsPerLink[%u]",
        isInterRdma, Is310PDevice(), remoteLinkInfo.userRank, remoteLinkInfo.devicePhyId,
        remoteLinkInfo.ip.GetReadableAddress(), remoteLinkInfo.port, remoteLinkInfo.socketsPerLink);
    return HCCL_SUCCESS;
}

HcclResult TransportManager::SetStopFlag(bool value)
{
    stopFlag_.store(value);
    return HCCL_SUCCESS;
}

void TransportManager::SetIsStandardCard(bool isStandardCard)
{
    isStandardCard_ = isStandardCard;
    return;
}

bool TransportManager::GetStopFlag() { return stopFlag_.load(); }

void TransportManager::SetPortConfig(bool devPortSwitchOn) { devPortSwitchOn_ = devPortSwitchOn; }

void TransportManager::SetOpType(HcclCMDType opType)
{
    opType_ = opType;
    return;
}

HcclResult TransportManager::SetGroupMode(bool groupMode)
{
    isGroupMode_ = groupMode;
    HCCL_INFO("[SetGroupMode] isGroupMode_=[%d]", isGroupMode_);
    return HCCL_SUCCESS;
}

std::map<u32, TransportType> TransportManager::GetRemoteTransportMap() { return remoteTransportMap_; }

HcclResult TransportManager::IsInterServer([[maybe_unused]] const u32 dstRank, [[maybe_unused]] bool& isInterServer)
{
#if !defined(CCL_KERNEL_AICPU) && !defined(HCCD)
    if (rankInfoList_[userRank_].deviceType == DevType::DEV_TYPE_910_93) {
        uint32_t userRankServerId = 0;
        uint32_t remoteRankServerId = 0;
        rtError_t ret = rtGetServerIDBySDID(rankInfoList_[userRank_].superDeviceId, &userRankServerId);
        CHK_PRT_RET(
            ret != RT_ERROR_NONE,
            HCCL_ERROR(
                "[IsInterServer]rtGetServerIDBySDID failed sdid[0x%08x], serverID[%u], ret[%u]",
                rankInfoList_[userRank_].superDeviceId, userRankServerId, ret),
            HCCL_E_RUNTIME);

        ret = rtGetServerIDBySDID(rankInfoList_[dstRank].superDeviceId, &remoteRankServerId);
        CHK_PRT_RET(
            ret != RT_ERROR_NONE,
            HCCL_ERROR(
                "[IsInterServer]rtGetServerIDBySDID failed sdid[0x%08x], serverID[%u], ret[%u]",
                rankInfoList_[dstRank].superDeviceId, remoteRankServerId, ret),
            HCCL_E_RUNTIME);
        isInterServer = (userRankServerId != remoteRankServerId)
                        || (rankInfoList_[userRank_].superPodId != rankInfoList_[dstRank].superPodId);
        HCCL_INFO(
            "[IsInterServer]localSDID[0x%08x], localdevicePhyId[%d], localServerId[%s], localServerIdBySDID[%d], "
            "localSuperPodId[%s], "
            "remoteSDID[0x%08x], remotedevicePhyId[%d], remoteRankServerId[%s], remoteServerIdBySDID[%d], "
            "remoteSuperPodId[%s], "
            "isInterServer[%s]",
            rankInfoList_[userRank_].superDeviceId, rankInfoList_[userRank_].devicePhyId,
            rankInfoList_[userRank_].serverId.c_str(), userRankServerId, rankInfoList_[userRank_].superPodId.c_str(),
            rankInfoList_[dstRank].superDeviceId, rankInfoList_[dstRank].devicePhyId,
            rankInfoList_[dstRank].serverId.c_str(), remoteRankServerId, rankInfoList_[dstRank].superPodId.c_str(),
            isInterServer ? "true" : "false");
    } else {
        isInterServer = rankInfoList_[userRank_].serverId != rankInfoList_[dstRank].serverId;
        HCCL_INFO(
            "[IsInterServer]localdevicePhyId[%d], localServerId[%s], "
            "remotedevicePhyId[%d], remoteServerId[%s], isInterServer[%s]",
            rankInfoList_[userRank_].devicePhyId, rankInfoList_[userRank_].serverId.c_str(),
            rankInfoList_[dstRank].devicePhyId, rankInfoList_[dstRank].serverId.c_str(),
            isInterServer ? "true" : "false");
    }
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[IsInterServer]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
} // namespace hccl
