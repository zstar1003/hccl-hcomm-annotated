/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include "sal_pub.h"
#include "hccl_one_sided_conn.h"
#include "p2p_mgmt_pub.h"

namespace hccl {
using namespace std;

HcclOneSidedConn::HcclOneSidedConn(
    const HcclNetDevCtx& netDevCtx, const HcclRankLinkInfo& localRankInfo, const HcclRankLinkInfo& remoteRankInfo,
    std::unique_ptr<HcclSocketManager>& socketManager, std::unique_ptr<NotifyPool>& notifyPool,
    const HcclDispatcher& dispatcher, const bool& useRdma, u32 sdid, u32 serverId, u32 trafficClass, u32 serviceLevel,
    bool aicpuUnfoldMode, bool isStandardCard, bool isNeedEnableP2P)
    : localRankInfo_(localRankInfo),
      socketManager_(socketManager),
      notifyPool_(notifyPool),
      aicpuUnfoldMode_(aicpuUnfoldMode),
      isStandardCard_(isStandardCard),
      isNeedEnableP2P_(isNeedEnableP2P)
{
    netDevCtx_ = netDevCtx;
    remoteRankInfo_ = remoteRankInfo;
    useRdma_ = useRdma;
    TransportMem::AttrInfo attrInfo{};
    attrInfo.localRankId = localRankInfo.userRank;
    attrInfo.remoteRankId = remoteRankInfo.userRank;
    attrInfo.sdid = sdid;
    attrInfo.serverId = serverId;
    attrInfo.trafficClass = trafficClass;
    attrInfo.serviceLevel = serviceLevel;
    if (useRdma) {
        transportMemPtr_ = TransportMem::Create(
            TransportMem::TpType::ROCE, notifyPool, netDevCtx, dispatcher, attrInfo, aicpuUnfoldMode_);
    } else {
        transportMemPtr_ = TransportMem::Create(
            TransportMem::TpType::IPC, notifyPool, netDevCtx, dispatcher, attrInfo, aicpuUnfoldMode_);
    }
    CHK_SMART_PTR_RET_NULL(transportMemPtr_);
}

HcclOneSidedConn::~HcclOneSidedConn()
{
    if ((isStandardCard_ && !useRdma_) && isNeedEnableP2P_) {
        if (!enableP2PDevices_.empty()) {
            P2PMgmtPub::DisableP2P(enableP2PDevices_);
            enableP2PDevices_.clear();
        }
    }
}

HcclResult HcclOneSidedConn::Connect(const std::string& commIdentifier, s32 timeoutSec)
{
    const auto startTime = TIME_NOW();
    if (aicpuUnfoldMode_) {
        CHK_RET(DeviceMem::alloc(transportDataDevice_, sizeof(TransportDeviceNormalData)));
    }
    // 创建socket用于交换数据
    std::string newTag;
    if (localRankInfo_.userRank < remoteRankInfo_.userRank) {
        // 本端为SERVER，对端为CLIENT
        newTag = string(localRankInfo_.ip.GetReadableIP()) + "_" + to_string(localRankInfo_.port) + "_"
                 + string(remoteRankInfo_.ip.GetReadableIP()) + "_" + to_string(remoteRankInfo_.port) + "_"
                 + commIdentifier;
    } else {
        newTag = string(remoteRankInfo_.ip.GetReadableIP()) + "_" + to_string(remoteRankInfo_.port) + "_"
                 + string(localRankInfo_.ip.GetReadableIP()) + "_" + to_string(localRankInfo_.port) + "_"
                 + commIdentifier;
    }
    HCCL_DEBUG("[HcclOneSidedConn][Connect]socket tag:%s", newTag.c_str());

    // 1、通信域初始化时会做非标卡且非310P场景的EnableP2P操作
    // 2、此处补全标卡且不使用RDMA场景下的EnableP2P操作
    HCCL_INFO(
        "[HcclOneSidedConn][Connect]localRankId[%u]-localDevicePhyId[%u], remoteRankId[%u]-remoteDevicePhyId[%u], "
        "isStandardCard[%s], useRdma[%s], isNeedEnableP2P[%s]",
        localRankInfo_.userRank, localRankInfo_.devicePhyId, remoteRankInfo_.userRank, remoteRankInfo_.devicePhyId,
        isStandardCard_ ? "true" : "false", useRdma_ ? "true" : "false", isNeedEnableP2P_ ? "true" : "false");

    if ((isStandardCard_ && !useRdma_) && isNeedEnableP2P_) {
        std::vector<u32> enableP2PDevices;
        enableP2PDevices.push_back(remoteRankInfo_.devicePhyId);
        HCCL_INFO(
            "[HcclOneSidedConn][Connect]localDevicePhyId[%u] enable p2p with remoteDevicePhyId[%u]",
            localRankInfo_.devicePhyId, remoteRankInfo_.devicePhyId);
        HcclResult ret = P2PMgmtPub::EnableP2P(enableP2PDevices);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[HcclOneSidedConn][Connect]Enable P2P Failed, localPhyId[%u], remotephyId[%u], ret[%u]",
                localRankInfo_.devicePhyId, remoteRankInfo_.devicePhyId, ret),
            ret);
        enableP2PDevices_.push_back(remoteRankInfo_.devicePhyId);
    }

    // EnableP2P需要和WaitP2PEnabled匹配使用，此处需要对1、2两处的EnableP2P做WaitP2PEnabled处理
    if ((!isStandardCard_ || !useRdma_) && isNeedEnableP2P_) {
        std::vector<u32> waitP2PEnabledDevices;
        waitP2PEnabledDevices.push_back(remoteRankInfo_.devicePhyId);
        HCCL_INFO(
            "[HcclOneSidedConn][Connect]localDevicePhyId[%u] wait p2p enable with remoteDevicePhyId[%u]",
            localRankInfo_.devicePhyId, remoteRankInfo_.devicePhyId);
        HcclResult ret = P2PMgmtPub::WaitP2PEnabled(waitP2PEnabledDevices, [this]() -> bool {
            return socketManager_->GetStopFlag();
        });
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[HcclOneSidedConn][Connect]Wait Enable P2P Failed, src devicePhyId[%u], dst devicePhyId[%u], ret[%u]",
                localRankInfo_.devicePhyId, remoteRankInfo_.devicePhyId, ret),
            ret);
    }

    std::vector<std::shared_ptr<HcclSocket>> connectSockets;
    CHK_RET(socketManager_->CreateSingleLinkSocket(
        newTag, netDevCtx_, remoteRankInfo_, connectSockets, true, true, timeoutSec));
    CHK_RET(transportMemPtr_->SetDataSocket(connectSockets[0]));
    socket_ = connectSockets[0];

    if (useRdma_) {
        // 创建socket用于QP建链
        newTag += "_QP";
        auto timeCostSec = std::chrono::duration_cast<std::chrono::seconds>(TIME_NOW() - startTime).count();
        auto timeLeft = timeoutSec - timeCostSec;
        CHK_RET(socketManager_->CreateSingleLinkSocket(
            newTag, netDevCtx_, remoteRankInfo_, connectSockets, true, true, timeLeft));
        CHK_RET(transportMemPtr_->SetSocket(connectSockets[0]));
        rdmaSocket_ = connectSockets[0];

        if (timeoutSec == -1) {
            // timeout为-1，超时时间设为最大值
            CHK_RET(transportMemPtr_->Connect(INT_MAX));
        } else {
            // 超时时间减去已消耗的时间，避免接口整体耗时超过入参的秒数
            timeCostSec = std::chrono::duration_cast<std::chrono::seconds>(TIME_NOW() - startTime).count();
            timeLeft = timeoutSec - timeCostSec;
            CHK_PRT_RET(
                timeLeft <= 0,
                HCCL_ERROR(
                    "[HcclOneSidedConn][Connect] Connect timeout. comm[%s], timeoutSec[%d s]", commIdentifier.c_str(),
                    timeoutSec),
                HCCL_E_TIMEOUT);

            // Transport建链：notify资源创建+QP建链
            CHK_RET(transportMemPtr_->Connect(timeLeft));
        }
    }

    return HCCL_SUCCESS;
}

void HcclOneSidedConn::CleanSocketResource(const std::string& commIdentifier)
{
    HcclSocketRole role;
    std::string newTag;
    if (localRankInfo_.userRank < remoteRankInfo_.userRank) {
        // 本端为SERVER，对端为CLIENT
        role = HcclSocketRole::SOCKET_ROLE_SERVER;
        newTag = string(localRankInfo_.ip.GetReadableIP()) + "_" + to_string(localRankInfo_.port) + "_"
                 + string(remoteRankInfo_.ip.GetReadableIP()) + "_" + to_string(remoteRankInfo_.port) + "_"
                 + commIdentifier;
    } else {
        role = HcclSocketRole::SOCKET_ROLE_CLIENT;
        newTag = string(remoteRankInfo_.ip.GetReadableIP()) + "_" + to_string(remoteRankInfo_.port) + "_"
                 + string(localRankInfo_.ip.GetReadableIP()) + "_" + to_string(localRankInfo_.port) + "_"
                 + commIdentifier;
    }
    if (socket_ != nullptr) {
        HCCL_INFO(
            "[HcclOneSidedConn][%s]abort and delete socket with remote[%u] tag[%s]", __func__, remoteRankInfo_.userRank,
            newTag.c_str());
        std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> socketsMap;
        std::vector<std::shared_ptr<HcclSocket>> rankSockets{socket_};
        socketsMap.insert(std::make_pair(remoteRankInfo_.userRank, rankSockets));
        socketManager_->AbortAndDeleteSocket(newTag, role, socketsMap);
    }

    if (rdmaSocket_ != nullptr) {
        newTag += "_QP";
        HCCL_INFO(
            "[HcclOneSidedConn][%s]abort and delete rdmaSocket with remote[%u] tag[%s]", __func__,
            remoteRankInfo_.userRank, newTag.c_str());
        std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> socketsMap;
        std::vector<std::shared_ptr<HcclSocket>> rankSockets{rdmaSocket_};
        socketsMap.insert(std::make_pair(remoteRankInfo_.userRank, rankSockets));
        socketManager_->AbortAndDeleteSocket(newTag, role, socketsMap);
    }
    return;
}

HcclResult HcclOneSidedConn::ExchangeIpcProcessInfo(const ProcessInfo& localProcess, ProcessInfo& remoteProcess)
{
    HCCL_DEBUG(
        "[HcclOneSidedConn][ExchangeIpcProcessInfo] localRank[%u] exchange process info", localRankInfo_.userRank);
    if (socket_->GetLocalRole() == HcclSocketRole::SOCKET_ROLE_CLIENT) {
        CHK_RET(socket_->Recv(&remoteProcess, sizeof(ProcessInfo)));
        CHK_RET(socket_->Send(&localProcess, sizeof(ProcessInfo)));
    } else {
        CHK_RET(socket_->Send(&localProcess, sizeof(ProcessInfo)));
        CHK_RET(socket_->Recv(&remoteProcess, sizeof(ProcessInfo)));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::ExchangeMemDesc(
    const HcclMemDescs& localMemDescs, HcclMemDescs& remoteMemDescs, u32& actualNumOfRemote)
{
    TransportMem::RmaMemDesc* localMemDescArray
        = static_cast<TransportMem::RmaMemDesc*>(static_cast<void*>(localMemDescs.array));
    TransportMem::RmaMemDescs localRmaMemDescs = {localMemDescArray, localMemDescs.arrayLength};
    TransportMem::RmaMemDesc* remoteMemDescArray
        = static_cast<TransportMem::RmaMemDesc*>(static_cast<void*>(remoteMemDescs.array));
    TransportMem::RmaMemDescs remoteRmaMemDescs = {remoteMemDescArray, remoteMemDescs.arrayLength};

    return transportMemPtr_->ExchangeMemDesc(localRmaMemDescs, remoteRmaMemDescs, actualNumOfRemote);
}

HcclResult HcclOneSidedConn::GetMemType(const char* description, RmaMemType& memType)
{
    std::string tempDesc = std::string(description, TRANSPORT_EMD_ESC_SIZE);
    std::istringstream iss(tempDesc);
    // 定义需要跳过的变量的大小
    const std::vector<size_t> skip_sizes = {
        sizeof(u8),    // type
        sizeof(void*), // addr
        sizeof(u64),   // size
        sizeof(void*)  // devAddr
    };
    // 计算偏移量
    size_t offset = std::accumulate(skip_sizes.begin(), skip_sizes.end(), 0);
    // 定位到 memType 的位置
    iss.seekg(offset);
    iss.read(reinterpret_cast<char_t*>(&memType), sizeof(memType));
    CHK_PRT_RET(
        memType >= RmaMemType::TYPE_NUM,
        HCCL_ERROR("[HcclOneSidedConn][GetMemType] get memType failed memType[%d]", static_cast<int>(memType)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

void HcclOneSidedConn::EnableMemAccess(const HcclMemDesc& remoteMemDesc, HcclMem& remoteMem)
{
    // 数据第一次转换
    HCCL_INFO("[HcclOneSidedConn][EnableMemAccess] Enable memory access.");
    const RmaMemDesc* remoteRmaMemDesc = static_cast<const RmaMemDesc*>(static_cast<const void*>(remoteMemDesc.desc));
    string tempStr = RmaMemDescCopyToStr(*remoteRmaMemDesc);
    RmaMemType memType;
    EXCEPTION_THROW_IF_ERR(
        GetMemType(remoteRmaMemDesc->memDesc, memType), "[HcclOneSidedConn][EnableMemAccess] get memType failed");
    auto iter = memDescMap_.find(tempStr);
    if (iter != memDescMap_.end()) {
        HcclBuf& outBuf = iter->second;
        BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(outBuf.addr), outBuf.len);
        auto resultPair = remoteRmaBufferMgr_.Add(tempKey, outBuf.handle);
        EXCEPTION_THROW_IF_COND_ERR(
            resultPair.first == remoteRmaBufferMgr_.End(),
            "[HcclOneSidedConn][EnableMemAccess]The memory that is expected to enable"
            " overlaps with the memory that has been enabled, please check params");
        remoteMem.type = static_cast<HcclMemType>(memType); // GE会检查这个字段先从字符串中获取
        remoteMem.addr = outBuf.addr;
        remoteMem.size = outBuf.len;
        return;
    }

    HcclBuf outBuf;
    EXCEPTION_THROW_IF_ERR(
        HcclMemImport(remoteRmaMemDesc->memDesc, TRANSPORT_EMD_ESC_SIZE, true, &outBuf, netDevCtx_),
        "[HcclOneSidedConn][EnableMemAccess] Enable memory access failed.");
    remoteMem.type = static_cast<HcclMemType>(memType); // GE会检查这个字段先从字符串中获取
    remoteMem.addr = outBuf.addr;
    remoteMem.size = outBuf.len;
    BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(outBuf.addr), outBuf.len);
    auto resultPair = remoteRmaBufferMgr_.Add(tempKey, outBuf.handle);

    EXCEPTION_THROW_IF_COND_ERR(
        resultPair.first == remoteRmaBufferMgr_.End(),
        "[HcclOneSidedConn][EnableMemAccess]The memory that is expected to enable"
        " overlaps with the memory that has been enabled, please check params");
    HCCL_INFO(
        "[HcclOneSidedConn][EnableMemAccess] after insert remoteRmaBufferMgr_ size[%d]", remoteRmaBufferMgr_.size());

    HCCL_INFO("[HcclOneSidedConn][EnableMemAccess] before insert memDescMap_ size[%d]", memDescMap_.size());

    memDescMap_.emplace(tempStr, outBuf);
    HCCL_INFO("[HcclOneSidedConn][EnableMemAccess] after insert memDescMap_ size[%d]", memDescMap_.size());
    HCCL_INFO("[HcclOneSidedConn][EnableMemAccess] Enable memory access success.");
}

void HcclOneSidedConn::DisableMemAccess(const HcclMemDesc& remoteMemDesc)
{
    // 数据第一次转换
    const RmaMemDesc* remoteRmaMemDesc = static_cast<const RmaMemDesc*>(static_cast<const void*>(remoteMemDesc.desc));
    string tempStr = RmaMemDescCopyToStr(*remoteRmaMemDesc);
    auto it = memDescMap_.find(tempStr);
    EXCEPTION_THROW_IF_COND_ERR(it == memDescMap_.end(), "Can't find hcclmem by key");

    BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(it->second.addr), it->second.len);
    HcclBuf& buf = it->second;
    try {
        if (remoteRmaBufferMgr_.Del(tempKey)) {
            EXCEPTION_THROW_IF_COND_ERR(HcclMemClose(&buf) != HCCL_SUCCESS, "Close remote memory failed.");
            HCCL_INFO("[HcclOneSidedConn][DisableMemAccess] before erase memDescMap_ size[%d]", memDescMap_.size());
            memDescMap_.erase(remoteRmaMemDesc->memDesc);
            HCCL_INFO("[HcclOneSidedConn][DisableMemAccess] after erase memDescMap_ size[%d]", memDescMap_.size());
            // 删除成功：输入key是表中某一最相近key的全集，计数-1后为0，返回true
            HCCL_INFO("[TransportIpcMem][DisableMemAccess]Memory reference count is 0, disable memory access.");
        } else {
            // 删除失败：输入key是表中某一最相近key的全集，计数不为0（存在其他remoteRank使用），返回false
            HCCL_INFO("[TransportIpcMem][DisableMemAccess]Memory reference count is larger than 0"
                      "(used by other RemoteRank), do not disable memory.");
        }
    } catch (std::out_of_range& e) {
        HCCL_ERROR("[TransportIpcMem][DisableMemAccess] catch RmaBufferMgr Del exception: %s", e.what());
        EXCEPTION_THROW_IF_COND_ERR(true, "[TransportIpcMem][DisableMemAccess] catch RmaBufferMgr Del exception");
    }
    HCCL_INFO("[HcclOneSidedConn][DisableMemAccess] Disable memory access success.");
}

void HcclOneSidedConn::BatchWrite(const HcclOneSideOpDesc* oneSideDescs, u32 descNum, const rtStream_t& stream)
{
    for (u32 i = 0; i < descNum; i++) {
        if (oneSideDescs[i].count == 0) {
            HCCL_WARNING("[HcclOneSidedConn][BatchWrite] Desc item[%u] count is 0.", i);
        }
        u32 unitSize;
        EXCEPTION_THROW_IF_ERR(
            SalGetDataTypeSize(oneSideDescs[i].dataType, unitSize),
            "[HcclOneSidedConn][BatchWrite] Get dataType size failed!");
        u64 byteSize = oneSideDescs[i].count * unitSize;
        HCCL_DEBUG(
            "[HcclOneSidedConn][BatchWrite] Desc[%u], localMem[%p], remoteMem[%p], size[%llu]", i,
            oneSideDescs[i].localAddr, oneSideDescs[i].remoteAddr, byteSize);

        BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(oneSideDescs[i].remoteAddr), byteSize);

        auto rmaBuffer = remoteRmaBufferMgr_.Find(tempKey);
        EXCEPTION_THROW_IF_COND_ERR(!rmaBuffer.first, "Can't find remoteBuffer by key");

        HcclBuf localMem = {oneSideDescs[i].localAddr, byteSize, nullptr};
        HcclBuf remoteMem = {oneSideDescs[i].remoteAddr, byteSize, rmaBuffer.second};
        EXCEPTION_THROW_IF_ERR(
            transportMemPtr_->Write(remoteMem, localMem, stream),
            "[HcclOneSidedConn][BatchWrite] transportMem WriteAsync failed.");
    }
    EXCEPTION_THROW_IF_ERR(transportMemPtr_->AddOpFence(stream), "[HcclOneSidedConn][BatchWrite] AddOpFence failed.");
}

void HcclOneSidedConn::BatchRead(const HcclOneSideOpDesc* oneSideDescs, u32 descNum, const rtStream_t& stream)
{
    for (u32 i = 0; i < descNum; i++) {
        if (oneSideDescs[i].count == 0) {
            HCCL_WARNING("[HcclOneSidedConn][BatchRead] Desc item[%u] count is 0.", i);
        }
        u32 unitSize;
        EXCEPTION_THROW_IF_ERR(
            SalGetDataTypeSize(oneSideDescs[i].dataType, unitSize),
            "[HcclOneSidedConn][BatchRead] Get dataType size failed!");
        u64 byteSize = oneSideDescs[i].count * unitSize;
        HCCL_DEBUG(
            "[HcclOneSidedConn][BatchRead] Desc[%u], localMem[%p], remoteMem[%p], size[%llu]", i,
            oneSideDescs[i].localAddr, oneSideDescs[i].remoteAddr, byteSize);

        BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(oneSideDescs[i].remoteAddr), byteSize);

        auto rmaBuffer = remoteRmaBufferMgr_.Find(tempKey);
        EXCEPTION_THROW_IF_COND_ERR(!rmaBuffer.first, "Can't find remoteBuffer by key");

        HcclBuf localMem = {oneSideDescs[i].localAddr, byteSize, nullptr};
        HcclBuf remoteMem = {oneSideDescs[i].remoteAddr, byteSize, rmaBuffer.second};
        EXCEPTION_THROW_IF_ERR(
            transportMemPtr_->Read(localMem, remoteMem, stream),
            "[HcclOneSidedConn][BatchRead] transportMem ReadAsync failed.");
    }
    EXCEPTION_THROW_IF_ERR(transportMemPtr_->AddOpFence(stream), "[HcclOneSidedConn][BatchRead] AddOpFence failed.");
}

HcclResult HcclOneSidedConn::GetTransInfo(
    HcclOneSideOpDescParam* descParam, const HcclOneSideOpDesc* desc, u32 descNum, u64& transportDataAddr,
    u64& transportDataSize)
{
    std::vector<u32> lkeys(descNum);
    std::vector<u32> rkeys(descNum);
    std::vector<HcclBuf> localMems(descNum);
    std::vector<HcclBuf> remoteMems(descNum);
    for (u32 i = 0; i < descNum - 1; ++i) { // last element is signal
        u32 unitSize = 0;
        CHK_RET(SalGetDataTypeSize(desc[i].dataType, unitSize));
        u64 bufSize = desc[i].count * unitSize;
        HCCL_DEBUG(
            "[HcclOneSidedConn][GetTransInfo] Desc[%u], localMem[%p], remoteMem[%p], size[%llu]", i, desc[i].localAddr,
            desc[i].remoteAddr, bufSize);

        BufferKey<uintptr_t, u64> bufKey(reinterpret_cast<uintptr_t>(desc[i].remoteAddr), bufSize);
        auto rmaBuffer = remoteRmaBufferMgr_.Find(bufKey);
        CHK_PRT_RET(
            !rmaBuffer.first,
            HCCL_ERROR("[GetTransInfo] Can't find remoteBuffer by key[%p][%llu]", desc[i].remoteAddr, bufSize),
            HCCL_E_PARA);

        localMems[i] = {desc[i].localAddr, bufSize, nullptr};
        remoteMems[i] = {desc[i].remoteAddr, bufSize, rmaBuffer.second};
    }
    CHK_RET(transportMemPtr_->GetTransInfo(
        transportData_.qpInfo, lkeys.data(), rkeys.data(), localMems.data(), remoteMems.data(), descNum));
    CHK_RET(hrtMemSyncCopy(
        transportDataDevice_.ptr(), transportDataDevice_.size(), static_cast<void*>(&transportData_),
        sizeof(transportData_), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    transportDataAddr = reinterpret_cast<u64>(transportDataDevice_.ptr());
    transportDataSize = transportDataDevice_.size();
    for (u32 i = 0; i < descNum - 1; ++i) {
        u32 unitSize = 0;
        CHK_RET(SalGetDataTypeSize(desc[i].dataType, unitSize));
        descParam[i].dataType = static_cast<u8>(desc[i].dataType);
        descParam[i].count = localMems[i].len / unitSize;
        descParam[i].localAddr = reinterpret_cast<u64>(localMems[i].addr);
        descParam[i].remoteAddr = reinterpret_cast<u64>(remoteMems[i].addr);
        descParam[i].lkey = lkeys[i];
        descParam[i].rkey = rkeys[i];
    }
    descParam[descNum - 1].dataType = static_cast<u8>(HcclDataType::HCCL_DATA_TYPE_UINT8);
    descParam[descNum - 1].count = localMems[descNum - 1].len; // 因为dataType是UINT8，所以count等于len
    descParam[descNum - 1].localAddr = reinterpret_cast<u64>(localMems[descNum - 1].addr);
    descParam[descNum - 1].remoteAddr = reinterpret_cast<u64>(remoteMems[descNum - 1].addr);
    descParam[descNum - 1].lkey = lkeys[descNum - 1];
    descParam[descNum - 1].rkey = rkeys[descNum - 1];
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::WaitOpFence(const rtStream_t& stream)
{
    CHK_RET(transportMemPtr_->WaitOpFence(stream));
    return HCCL_SUCCESS;
}

HcclResult
HcclOneSidedConn::ConnectWithRemote(const std::string& commIdentifier, ProcessInfo localProcess, s32 timeoutSec)
{
    CHK_RET(Connect(commIdentifier, timeoutSec));
    if (!useRdma_) {
        CHK_RET(ExchangeIpcProcessInfo(localProcess, remoteProcess_));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::GetRemoteProcessInfo(ProcessInfo& remoteProcess)
{
    remoteProcess = remoteProcess_;
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::ExchangeMemDesc(const HcclMemDescs& localMemDescs)
{
    constexpr u32 exchangeCntPerLoop = MAX_REMOTE_MEM_NUM;
    u32 localMemOffset = 0;
    u32 localMemCnt = localMemDescs.arrayLength;
    remoteMemDescsVec_.resize(exchangeCntPerLoop);
    actualNumOfRemote_ = 0;

    while (true) {
        // 每轮循环最多交换 exchangeCntPerLoop 个 memDesc
        u32 sendLocalCnt = localMemCnt > exchangeCntPerLoop ? exchangeCntPerLoop : localMemCnt;
        TransportMem::RmaMemDesc* localMemDescArray
            = static_cast<TransportMem::RmaMemDesc*>(static_cast<void*>(localMemDescs.array)) + localMemOffset;
        TransportMem::RmaMemDescs localRmaMemDescs = {localMemDescArray, sendLocalCnt};

        if (remoteMemDescsVec_.size() - actualNumOfRemote_ < exchangeCntPerLoop) {
            remoteMemDescsVec_.resize(remoteMemDescsVec_.size() + exchangeCntPerLoop);
        }
        TransportMem::RmaMemDesc* remoteMemDescArray
            = static_cast<TransportMem::RmaMemDesc*>(static_cast<void*>(&remoteMemDescsVec_[actualNumOfRemote_]));
        TransportMem::RmaMemDescs remoteRmaMemDescs = {remoteMemDescArray, exchangeCntPerLoop};

        u32 actualNumOfRemote = 0;
        CHK_RET(transportMemPtr_->ExchangeMemDesc(localRmaMemDescs, remoteRmaMemDescs, actualNumOfRemote));
        localMemOffset += sendLocalCnt;
        localMemCnt -= sendLocalCnt;
        actualNumOfRemote_ += actualNumOfRemote;

        if (actualNumOfRemote < exchangeCntPerLoop && sendLocalCnt < exchangeCntPerLoop) {
            // 循环结束条件，下轮没有memDesc要发 且 对端也没有memDesc要发
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::EnableMemAccess()
{
    CHK_PRT_RET(
        actualNumOfRemote_ > remoteMemDescsVec_.size(),
        HCCL_ERROR(
            "[HcclOneSidedConn][EnableMemAccess] actualNumOfRemote[%u] is larger than remoteMemDescsVec.size[%zu]",
            actualNumOfRemote_, remoteMemDescsVec_.size()),
        HCCL_E_INTERNAL);

    HcclMem remoteMem;
    for (u32 i = 0; i < actualNumOfRemote_; i++) {
        // 创建HcclMemDesc对象
        HcclMemDesc* remoteMemDesc = static_cast<HcclMemDesc*>(static_cast<void*>(&remoteMemDescsVec_.at(i)));
        this->EnableMemAccess(*remoteMemDesc, remoteMem);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::DisableMemAccess()
{
    CHK_PRT_RET(
        actualNumOfRemote_ > remoteMemDescsVec_.size(),
        HCCL_ERROR(
            "[HcclOneSidedConn][DisableMemAccess] actualNumOfRemote[%u] is larger than remoteMemDescsVec.size[%zu]",
            actualNumOfRemote_, remoteMemDescsVec_.size()),
        HCCL_E_INTERNAL);

    for (u32 i = 0; i < actualNumOfRemote_; i++) {
        // 创建HcclMemDesc对象
        HcclMemDesc* remoteMemDesc = static_cast<HcclMemDesc*>(static_cast<void*>(&remoteMemDescsVec_.at(i)));
        this->DisableMemAccess(*remoteMemDesc);
    }

    return HCCL_SUCCESS;
}
} // namespace hccl
