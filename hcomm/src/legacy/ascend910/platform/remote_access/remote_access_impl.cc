/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "remote_access_impl.h"
#include <algorithm>

#include "transport_remote_access.h"

namespace hccl {
using namespace std;

RemoteAccessImpl::RemoteAccessImpl() : userRank_(0), userRankNum_(0), serverNum_(0), rankNumPerServer_(0) {}

RemoteAccessImpl::~RemoteAccessImpl() {}

HcclResult RemoteAccessImpl::Init(u32 rank, const vector<MemRegisterAddr>& addrInfos, const RmaRankTable& rankTable)
{
    HCCL_INFO("RemoteAccessImpl init start");

    userRank_ = rank;
    userRankNum_ = rankTable.rankNum;
    serverNum_ = rankTable.serverNum;
    if (serverNum_ == 0) {
        HCCL_ERROR("[RemoteAccessImpl][Init]errNo[0x%016llx] server num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    rankNumPerServer_ = userRankNum_ / serverNum_;
    HCCL_INFO(
        "RemoteAccessImpl Init userRank_[%u] userRankNum_[%u] serverNum_[%u] rankNumPerServer_[%u]", userRank_,
        userRankNum_, serverNum_, rankNumPerServer_);

    u32 rankInComm = userRank_ / rankNumPerServer_;
    CHK_PRT_RET(
        rankTable.deviceIps.empty(), HCCL_ERROR("[Init][RemoteAccessImpl]rankTable.rankList is empty"), HCCL_E_PARA);
    u32 devicePhyId = rankTable.devicePhyId;
    std::map<u32, std::vector<HcclIpAddress>> rankInfo; // rankIdInComm - deviceIp
    for (u32 rankIndex = 0; rankIndex < userRankNum_; rankIndex++) {
        if ((rankIndex % rankNumPerServer_) == (userRank_ % rankNumPerServer_)) { // 在同一平面
            u32 curRank = rankIndex / rankNumPerServer_;                          // 通信域内的第几个rank
            rankInfo.insert(std::pair<u32, std::vector<HcclIpAddress>>(curRank, rankTable.deviceIps[rankIndex]));
        }
    }
    comm_.reset(new (std::nothrow) CommRemoteAccess(rankInComm, devicePhyId, rankInfo, addrInfos));
    CHK_SMART_PTR_NULL(comm_);
    CHK_RET(comm_->Init());
    return HCCL_SUCCESS;
}

void RemoteAccessImpl::ParseRemoteAccessAddrInfo(
    const vector<HcomRemoteAccessAddrInfo>& addrInfos, map<u32, vector<HcomRemoteAccessAddrInfo>>& addrInfoMap)
{
    for (u32 i = 0; i < addrInfos.size(); i++) {
        u32 remoteRankInComm = addrInfos[i].remotetRankID / rankNumPerServer_;
        addrInfoMap[remoteRankInComm].push_back(addrInfos[i]);
        HCCL_DEBUG(
            "ParseRemoteAccessAddrInfo localAddr[0x%016lx] remoteAddr[0x%016lx] length[%llu] "
            "remoteRankInComm[%u]",
            addrInfos[i].localAddr, addrInfos[i].remoteAddr, addrInfos[i].length, remoteRankInComm);
    }
}

HcclResult RemoteAccessImpl::IsInSamePlane(const u32 userRank, const vector<HcomRemoteAccessAddrInfo>& addrInfos)
{
    for (u32 i = 0; i < addrInfos.size(); i++) {
        CHK_PRT_RET(
            (userRank % rankNumPerServer_) != (addrInfos[i].remotetRankID % rankNumPerServer_),
            HCCL_ERROR(
                "[Is][InSamePlane]The userrank[%u] and remoterank[%u] must be in the same plane", userRank,
                addrInfos[i].remotetRankID),
            HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

HcclResult RemoteAccessImpl::RemoteWrite(const vector<HcomRemoteAccessAddrInfo>& addrInfos, HcclRtStream stream)
{
    size_t infoSize = addrInfos.size();
    CHK_PRT_RET(addrInfos.empty(), HCCL_ERROR("[Remote][Write]addrInfos is empty!"), HCCL_E_PARA);
    CHK_RET(IsInSamePlane(userRank_, addrInfos));

    Stream streamObj(stream);
    //  GE 保证传入的addrInfos按照remotetRankID排序，如果目标是同一个remotetRank，优化性能
    if (infoSize > 1 && addrInfos[0].remotetRankID == addrInfos[infoSize - 1].remotetRankID) {
        u32 remoteRankInComm = addrInfos[0].remotetRankID / rankNumPerServer_;
        CHK_PRT_RET(
            remoteRankInComm > (serverNum_ - 1),
            HCCL_ERROR(
                "[Remote][Write]remote write invalid rank id [%u] should be in [0, %u]!", remoteRankInComm,
                (serverNum_ - 1)),
            HCCL_E_PARA);
        std::shared_ptr<TransportRemoteAccess> transportPtr = comm_->GetTransportByRank(remoteRankInComm);
        CHK_SMART_PTR_NULL(transportPtr);
        CHK_RET(transportPtr->RemoteWrite(addrInfos, streamObj));
    } else {
        map<u32, vector<HcomRemoteAccessAddrInfo>> addrInfoMap;
        ParseRemoteAccessAddrInfo(addrInfos, addrInfoMap);
        for (auto it = addrInfoMap.begin(); it != addrInfoMap.end(); it++) {
            CHK_PRT_RET(
                it->first > (serverNum_ - 1),
                HCCL_ERROR(
                    "[Remote][Write]remote write invalid rank id [%u] should be in [0, %u]!", it->first,
                    (serverNum_ - 1)),
                HCCL_E_PARA);
            std::shared_ptr<TransportRemoteAccess> transportPtr = comm_->GetTransportByRank(it->first);
            CHK_SMART_PTR_NULL(transportPtr);
            CHK_RET(transportPtr->RemoteWrite(it->second, streamObj));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RemoteAccessImpl::RemoteRead(const vector<HcomRemoteAccessAddrInfo>& addrInfos, HcclRtStream stream)
{
    HCCL_INFO("RemoteAccessImpl::RemoteRead");
    size_t infoSize = addrInfos.size();
    CHK_PRT_RET(addrInfos.empty(), HCCL_ERROR("[Remote][Read]addrInfos is empty!"), HCCL_E_PARA);

    CHK_RET(IsInSamePlane(userRank_, addrInfos));

    Stream streamObj(stream);
    //  GE 保证传入的addrInfos按照remotetRankID排序，如果目标是同一个remotetRank，优化性能
    if (infoSize > 1 && addrInfos[0].remotetRankID == addrInfos[infoSize - 1].remotetRankID) {
        u32 remoteRankInComm = addrInfos[0].remotetRankID / rankNumPerServer_;
        CHK_PRT_RET(
            remoteRankInComm > (serverNum_ - 1),
            HCCL_ERROR(
                "[remote][Read]remote read invalid rank id [%u] should be in [0, %u]!", remoteRankInComm,
                (serverNum_ - 1)),
            HCCL_E_PARA);

        std::shared_ptr<TransportRemoteAccess> transportPtr = comm_->GetTransportByRank(remoteRankInComm);
        CHK_SMART_PTR_NULL(transportPtr);
        CHK_RET(transportPtr->RemoteRead(addrInfos, streamObj));
    } else {
        map<u32, vector<HcomRemoteAccessAddrInfo>> addrInfoMap;
        ParseRemoteAccessAddrInfo(addrInfos, addrInfoMap);
        for (auto it = addrInfoMap.begin(); it != addrInfoMap.end(); it++) {
            CHK_PRT_RET(
                it->first > (serverNum_ - 1),
                HCCL_ERROR(
                    "[remote][Read]remote read invalid rank id [%u] should be in [0, %u]!", it->first,
                    (serverNum_ - 1)),
                HCCL_E_PARA);

            std::shared_ptr<TransportRemoteAccess> transportPtr = comm_->GetTransportByRank(it->first);
            CHK_SMART_PTR_NULL(transportPtr);

            CHK_RET(transportPtr->RemoteRead(it->second, streamObj));
        }
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
