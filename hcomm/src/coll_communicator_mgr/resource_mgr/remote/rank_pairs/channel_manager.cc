/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel_manager.h"
#include "adapter_rts_common.h"
#include "log.h"
#include "comm_configer.h"
#include "launch_aicpu.h"
#include "comm_engine_utils.h"
#include <unordered_set>
#include <string>
#include "adapter_prof.h"
#include "hcom_host_profiling.h"

namespace hccl {

constexpr u32 RDMA_NOTIFY_MIN_NUM = 3;
constexpr u32 NOTIFY_NUM_MAX = 64; // HcclChannelDesc 中 notifynum 的默认限制最大为64

HcclResult ChannelManager::Init(aclrtBinHandle binHandle, u32 userRank, const ManagerCallbacks& callbacks)
{
    binHandle_ = binHandle;
    userRank_ = userRank;
    callbacks_ = callbacks;
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::SetChannelCallbacks(const ChannelManagerCallbacks& channelCallbacks)
{
    channelCallbacks_ = channelCallbacks;
    rankInfoList_ = channelCallbacks_.getRankLists();
    return HCCL_SUCCESS;
}

std::vector<HcclMemHandle> ChannelManager::CollectMemHandles(const std::vector<HcclChannelDesc>& descs) const
{
    std::vector<HcclMemHandle> handles;
    std::unordered_set<HcclMemHandle> seen;
    for (const auto& desc : descs) {
        for (uint32_t i = 0; i < desc.memHandleNum; ++i) {
            if (desc.memHandles != nullptr && seen.insert(desc.memHandles[i]).second) {
                handles.push_back(desc.memHandles[i]);
            }
        }
    }
    return handles;
}

HcclResult ChannelManager::CheckChannelParam(CommEngine engine, const HcclChannelDesc* channelDesc, uint32_t descNum)
{
    std::unordered_set<HcclChannelDesc, std::hash<HcclChannelDesc>, HcclChannelDescEqual> descSet;

    for (uint32_t descIdx = 0; descIdx < descNum; ++descIdx) {
        // 检查notifyNum
        CHK_PRT_RET(
            channelDesc[descIdx].notifyNum > NOTIFY_NUM_MAX,
            HCCL_ERROR(
                "[%s]Channeldesc[%u] invalid notifyNum, notifyNum[%u], max notify num[%u]", __func__, descIdx,
                channelDesc[descIdx].notifyNum, NOTIFY_NUM_MAX),
            HCCL_E_PARA);
        // 检查HcclChannelDesc是否有重复元素
        CHK_PRT_RET(
            descSet.find(channelDesc[descIdx]) != descSet.end(),
            HCCL_ERROR("[%s]Duplicate item found in hcclchanneldesc.", __func__), HCCL_E_PARA);
        descSet.insert(channelDesc[descIdx]);
        // 检查RemoteRank有效性
        CHK_PRT_RET(
            channelDesc[descIdx].remoteRank == userRank_,
            HCCL_ERROR("[%s]Local rank found in channeldesc, userRank_ = %u.", __func__, userRank_), HCCL_E_PARA);
        // 检查是否有不支持协议
        if (engine == COMM_ENGINE_AIV) {
            CHK_PRT_RET(
                channelDesc[descIdx].channelProtocol != COMM_PROTOCOL_HCCS
                    && channelDesc[descIdx].channelProtocol != COMM_PROTOCOL_HCCS_ONLY
                    && channelDesc[descIdx].channelProtocol != COMM_PROTOCOL_SIO,
                HCCL_ERROR(
                    "[%s]AIV engine Unsupported protocol[%d] found in channeldesc, protocol: %d.", __func__, descIdx,
                    channelDesc[descIdx].channelProtocol),
                HCCL_E_PARA);
        } else {
            CHK_PRT_RET(
                channelDesc[descIdx].channelProtocol != COMM_PROTOCOL_HCCS
                    && channelDesc[descIdx].channelProtocol != COMM_PROTOCOL_ROCE
                    && channelDesc[descIdx].channelProtocol != COMM_PROTOCOL_HCCS_ONLY
                    && channelDesc[descIdx].channelProtocol != COMM_PROTOCOL_SIO,
                HCCL_ERROR(
                    "[%s]Unsupported protocol[%d] found in channeldesc, protocol: %d.", __func__, descIdx,
                    channelDesc[descIdx].channelProtocol),
                HCCL_E_PARA);
        }

        // 检查engine支持情况
        if (engine != COMM_ENGINE_CPU && engine != COMM_ENGINE_CPU_TS && engine != COMM_ENGINE_AICPU
            && engine != COMM_ENGINE_AICPU_TS && engine != COMM_ENGINE_AIV) {
            HCCL_ERROR(
                "[%s]Unsupported engine for channel, engine: %s.", __func__,
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

static std::string BuildChannelKey(const std::string& tag, CommEngine engine, const HcclChannelDesc& channelDesc)
{
    std::string key = tag + ":" + std::to_string(engine) + ":" + std::to_string(channelDesc.remoteRank) + ":"
                      + std::to_string(channelDesc.channelProtocol);
    // 将 memHandles 纳入 key，使得上层更换交换内存时会强制新建 channel，避免复用旧 channel 导致远端地址错配
    if (channelDesc.memHandleNum > 0 && channelDesc.memHandles != nullptr) {
        for (uint32_t i = 0; i < channelDesc.memHandleNum; i++) {
            key += ":" + std::to_string(reinterpret_cast<uintptr_t>(channelDesc.memHandles[i]));
        }
    } else {
        key += ":0";
    }
    return key;
}

HcclResult ChannelManager::RegisterHandle(
    const std::string& tag, CommEngine engine, const HcclChannelDesc& channelDesc, ChannelHandle channelHandle)
{
    std::string channelKey = BuildChannelKey(tag, engine, channelDesc);

    CHK_PRT_RET(
        (channelHandleMap_.find(channelKey) != channelHandleMap_.end()),
        HCCL_ERROR(
            "[%s]Channel already exists, tag[%s], engine[%s], remoteRank[%d], channelProtocol[%d].", __func__,
            tag.c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelDesc.remoteRank,
            channelDesc.channelProtocol),
        HCCL_E_PARA);
    channelHandleMap_[channelKey] = channelHandle;
    keyMap_[channelHandle] = channelKey;
    engineMap_[channelHandle] = engine;
    HCCL_INFO("[%s]Register channel handle[%llu], channelKey[%s]", __func__, channelHandle, channelKey.c_str());
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::PrepareHandleArray(
    const std::string& tag, CommEngine engine, const HcclChannelDesc* channelDesc, uint32_t descNum,
    ChannelHandle* channelHandleArray, std::vector<HcclChannelDesc>& needCreateDescs,
    std::vector<uint32_t>& needCreateIndices)
{
    needCreateDescs.clear();
    needCreateIndices.clear();

    for (uint32_t descIdx = 0; descIdx < descNum; descIdx++) {
        // 组合channelKey
        std::string channelKey = BuildChannelKey(tag, engine, channelDesc[descIdx]);
        if (channelHandleMap_.find(channelKey) != channelHandleMap_.end()) {
            channelHandleArray[descIdx] = channelHandleMap_[channelKey];
            continue;
        }
        channelHandleArray[descIdx] = 0;
        needCreateDescs.push_back(channelDesc[descIdx]);
        needCreateIndices.push_back(descIdx);
    }

    return HCCL_SUCCESS;
}

HcclResult ChannelManager::IsChannelExist(ChannelHandle channel)
{
    CHK_PRT_RET(
        (keyMap_.find(channel) == keyMap_.end()), HCCL_ERROR("[%s]ChannelHandle does not exist.", __func__),
        HCCL_E_PARA);
    HCCL_INFO(
        "[%s]ChannelHandle exist, ChannelHandle[%llu], channelKey[%s]", __func__, channel, keyMap_[channel].c_str());
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::UnregisterHandle(ChannelHandle channel)
{
    CHK_PRT_RET(
        (keyMap_.find(channel) == keyMap_.end()), HCCL_ERROR("[%s]ChannelHandle does not exist.", __func__),
        HCCL_E_PARA);

    channelHandleMap_.erase(keyMap_[channel]);
    keyMap_.erase(channel);
    if (engineMap_[channel] == COMM_ENGINE_AICPU || engineMap_[channel] == COMM_ENGINE_AICPU_TS) {
        channelD2HMap_.erase(channel);
    }
    engineMap_.erase(channel);

    HCCL_INFO("[%s]Unregister channel handle success.", __func__);
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::RegisterHandleHDPair(ChannelHandle deviceChannelHandle, ChannelHandle hostChannelHandle)
{
    CHK_PRT_RET(
        (deviceChannelHandle == 0 || hostChannelHandle == 0), HCCL_ERROR("[%s]ChannelHandle is 0.", __func__),
        HCCL_E_PARA);
    CHK_PRT_RET(
        (channelD2HMap_.find(deviceChannelHandle) != channelD2HMap_.end()),
        HCCL_ERROR("[%s]deviceChannelHandle has existed in channelD2HMap_.", __func__), HCCL_E_PARA);

    channelD2HMap_[deviceChannelHandle] = hostChannelHandle;
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::GetHostChannel(ChannelHandle channel, ChannelHandle& hostChannel)
{
    if (engineMap_[channel] == COMM_ENGINE_AICPU || engineMap_[channel] == COMM_ENGINE_AICPU_TS) {
        CHK_PRT_RET(
            (channelD2HMap_.find(channel) == channelD2HMap_.end()),
            HCCL_ERROR("[%s]device channel handle does not exist in channelD2HMap_.", __func__), HCCL_E_PARA);
        hostChannel = channelD2HMap_[channel];
    } else {
        hostChannel = channel;
    }
    return HCCL_SUCCESS;
}

void ChannelManager::ClearOpTransportResponseLinks(OpCommTransport& opTransportResponse)
{
    for (auto& levelNSubCommTransport : opTransportResponse) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            u32 size = singleSubCommTransport.transportRequests.size();
            singleSubCommTransport.links.resize(size, nullptr);
            singleSubCommTransport.status.resize(size, TransportStatus::INIT);
            HCCL_INFO("[%s] size[%u], linksSize[%zu]", __func__, size, singleSubCommTransport.links.size());
        }
    }
}

HcclResult ChannelManager::CheckNotifyOrQPMaxNum(u64& existNum, const u64& MaxNum, const bool& isNotifyRes)
{
    std::string resType = isNotifyRes ? "Notify" : "QP";
    if (existNum + 1 > MaxNum) {
        HCCL_ERROR(
            "[%s]%s resources are insufficient, existNum[%llu], MaxNum is [%llu]", __func__, resType.c_str(), existNum,
            MaxNum);
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG(
        "[%s]%s resources are sufficient, existNum[%llu], MaxNum is [%llu]", __func__, resType.c_str(), existNum,
        MaxNum);
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::CreateWorkSpace(u64 size, DeviceMem& buffer) const
{
    CHK_PRT_RET(
        size == 0, HCCL_INFO("[Create][WorkSpace]work space size is zero, no need to allocate memory"), HCCL_SUCCESS);

    CHK_PRT_RET(
        (size > ULONG_MAX), HCCL_ERROR("[Create][WorkSpace]work space size is greater than %llu", ULONG_MAX),
        HCCL_E_PARA);

    u64 memSize = size;
    buffer = DeviceMem::alloc(memSize);
    CHK_PRT_RET(
        size > 0 && !buffer,
        HCCL_ERROR(
            "[Create][WorkSpace]Create work space failed, size[%llu], "
            "please check workspace size.",
            size),
        HCCL_E_PTR);
    CHK_RET(hrtMemSet(buffer.ptr(), size, size));
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::AllocAndClearHostMem(u64 size, std::shared_ptr<HostMem>& bufferPtr) const
{
    CHK_PRT_RET(
        size == 0,
        HCCL_INFO("[ChannelManager][AllocAndClearHostMem] host memory size is zero, no need to allocate memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET(
        (size > ULONG_MAX),
        HCCL_ERROR("[ChannelManager][AllocAndClearHostMem] host memory size is greater than %llu", ULONG_MAX),
        HCCL_E_PARA);

    HostMem tmpBuffer = HostMem::alloc(size);
    EXCEPTION_CATCH((bufferPtr = std::make_shared<HostMem>(std::move(tmpBuffer))), return HCCL_E_PTR);

    CHK_PRT_RET(
        size > 0 && !bufferPtr.get()->ptr(),
        HCCL_ERROR(
            "[ChannelManager][AllocAndClearHostMem]alloc host memory failed, size[%llu], "
            "please check host memory size.",
            size),
        HCCL_E_PTR);
    CHK_SAFETY_FUNC_RET(memset_s(bufferPtr.get()->ptr(), size, 0, size));
    return HCCL_SUCCESS;
}

template <typename T>
HcclResult ChannelManager::CopyVectorToDeviceMem(const u64 len, DeviceMem& dstDeviceMem, const std::vector<T>& srcVec)
{
    CHK_PRT_RET(
        len == 0, HCCL_INFO("[ChannelManager][CopyVectorToDeviceMem] space size is zero, no need to allocate memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET(
        (len > ULONG_MAX),
        HCCL_ERROR("[ChannelManager][CopyVectorToDeviceMem] space size is greater than %llu", ULONG_MAX), HCCL_E_PARA);

    CHK_RET(CreateWorkSpace(len, dstDeviceMem));
    std::shared_ptr<HostMem> srcHostMem;
    CHK_RET(AllocAndClearHostMem(len, srcHostMem));
    std::copy(srcVec.begin(), srcVec.end(), static_cast<T*>(srcHostMem.get()->ptr()));
    CHK_RET(hrtMemSyncCopy(
        dstDeviceMem.ptr(), len, srcHostMem.get()->ptr(), len, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

OpCommTransport ChannelManager::BuildChannelRequests(const std::vector<HcclChannelDesc>& descs)
{
    OpCommTransport opCommTransport;
    LevelNSubCommTransport level0Transport;
    SingleSubCommTransport commTransport;

    for (auto desc : descs) {
        TransportRequest tmpTransport;
        tmpTransport.isValid = true;
        tmpTransport.localUserRank = userRank_;
        tmpTransport.remoteUserRank = desc.remoteRank;
        tmpTransport.notifyNum = desc.notifyNum;
        tmpTransport.inputMemType = TransportMemType::CCL_INPUT;
        tmpTransport.outputMemType = TransportMemType::CCL_OUTPUT;
        tmpTransport.isUsedRdma = (desc.channelProtocol == CommProtocol::COMM_PROTOCOL_ROCE);
        TransportLinkType linkType = TransportLinkType::RESERVED;
        if (desc.channelProtocol == CommProtocol::COMM_PROTOCOL_HCCS_ONLY) {
            linkType = TransportLinkType::HCCS;
        } else if (desc.channelProtocol == CommProtocol::COMM_PROTOCOL_SIO) {
            linkType = TransportLinkType::SIO;
        }
        tmpTransport.linkType = linkType;
        commTransport.transportRequests.push_back(tmpTransport);
    }

    level0Transport.push_back(commTransport);
    opCommTransport.push_back(level0Transport);
    ClearOpTransportResponseLinks(opCommTransport);

    return opCommTransport;
}

HcclResult ChannelManager::ParseChannelRemoteDataToMem(
    const OpCommTransport& opTransportResponse, HcclIndOpChannelRemoteResV3& channelParam)
{
    uint32_t level0 = 0;
    auto& singleSubCommTransport = opTransportResponse[level0][level0];
    CHK_PRT_RET(
        channelParam.listNum == 0, HCCL_ERROR("[%s]invalid listNum, listNum[%u]", __func__, channelParam.listNum),
        HCCL_E_PARA);
    CHK_PRT_RET(
        (channelParam.listNum != singleSubCommTransport.links.size()),
        HCCL_ERROR(
            "[%s]invalid listNum, listNum[%u] but links size is [%zu]", __func__, channelParam.listNum,
            singleSubCommTransport.links.size()),
        HCCL_E_PARA);
    // 分配 HcclIndOpChannelRemoteResV2 内存，需要手动释放
    channelParam.remoteResV2
        = static_cast<HcclIndOpChannelRemoteResV2*>(malloc(channelParam.listNum * sizeof(HcclIndOpChannelRemoteResV2)));
    CHK_PRT_RET(
        channelParam.remoteResV2 == nullptr, HCCL_ERROR("[%s]channelParam.remoteResV2 is null.", __func__),
        HCCL_E_MEMORY);
    u32 linkIdx = 0;
    for (auto& transportRequest : singleSubCommTransport.transportRequests) {
        auto& tempLink = singleSubCommTransport.links[linkIdx];
        channelParam.remoteResV2[linkIdx].remoteWorldRank = rankInfoList_[transportRequest.remoteUserRank].worldRank;
        channelParam.remoteResV2[linkIdx].remoteRank = transportRequest.remoteUserRank;
        // transport信息保存（notify、qp）
        if (!transportRequest.isUsedRdma) {
            // sdma -> P2P
            CHK_RET(BuildOpRemoteChannelP2pResParam(tempLink, channelParam.remoteResV2[linkIdx]));
            channelParam.remoteResV2[linkIdx].channelP2p.qos = hcclQos_;
            HCCL_INFO(
                "[ChannelManager] [ParseChannelRemoteDataToMem] hcclQos[%u]",
                channelParam.remoteResV2[linkIdx].channelP2p.qos);
        } else {
            // rdma -> roce
            CHK_RET(BuildOpRemoteChannelRoceResParam(tempLink, channelParam.remoteResV2[linkIdx]));
        }
        linkIdx++;
    }
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::BuildOpRemoteChannelP2pResParam(const LINK& link, HcclIndOpChannelRemoteResV2& remoteRes)
{
    remoteRes.isUsedRdma = false;
    HcclChannelP2p& linkp2p = remoteRes.channelP2p;
    // remoteMem, 独立算子localmem是否需要传待确认
    void* bufferPtr = nullptr;
    CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &bufferPtr));
    linkp2p.remoteHcclbuffer.addr = reinterpret_cast<void*>(bufferPtr);
    u64 remotebufferSize;
    CHK_RET(link->GetRemoteMemSize(UserMemType::INPUT_MEM, remotebufferSize));
    linkp2p.remoteHcclbuffer.size = remotebufferSize;
    // 独立算子远端用户内存，linkp2p.remoteUserMem需要手动释放内存
    CHK_RET(link->GetIndOpRemoteMem(&linkp2p.remoteUserMem, &linkp2p.remoteUserMemCount));
    HCCL_DEBUG("[%s] finish set remoteMem info", __func__);

    // localnotify & remotenotify
    u64 notifyNum = 0;
    std::vector<HcclSignalInfo> locIpcSignals;
    std::vector<HcclSignalInfo> rmtIpcSignals;
    CHK_RET(link->GetLocalNotify(locIpcSignals));
    CHK_RET(link->GetRemoteNotify(rmtIpcSignals));

    for (size_t i = 0; i < locIpcSignals.size(); i++) {
        linkp2p.localIpcSignal[notifyNum] = locIpcSignals[i];
        linkp2p.remoteIpcSignal[notifyNum] = rmtIpcSignals[i];
        notifyNum++;
    }
    remoteRes.p2pNotifyNum = link->GetNotifyNum();
    HCCL_DEBUG(
        "[%s] finish set localnotify & remotenotify info, notifyNum[%llu], p2pNotifyNum[%llu]", __func__, notifyNum,
        remoteRes.p2pNotifyNum);
    // transportAttr
    CHK_RET(link->GetTransportAttr(linkp2p.transportAttr));
    HCCL_DEBUG("[%s] finish set RemoteChannelP2pResParam info", __func__);
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::BuildOpRemoteChannelRoceResParam(const LINK& link, HcclIndOpChannelRemoteResV2& remoteRes)
{
    remoteRes.isUsedRdma = true;
    HcclChannelRoce& linkRoce = remoteRes.channelRoce;
    // 填充localMem信息到linkRoce中
    CHK_RET(link->GetLocalMemDetails(UserMemType::INPUT_MEM, linkRoce.localHcclbuffer));
    // 填充remoteMem信息到linkRoce中
    void* bufferPtr = nullptr;
    CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &bufferPtr));
    linkRoce.remoteHcclbuffer.addr = reinterpret_cast<u64>(bufferPtr);
    CHK_RET(link->GetRemoteMemKey(UserMemType::INPUT_MEM, &(linkRoce.remoteHcclbuffer.key)));
    CHK_RET(link->GetRemoteMemSize(UserMemType::INPUT_MEM, linkRoce.remoteHcclbuffer.size));
    // 独立算子远端用户内存，linkRoce.remoteUserHostMem和remoteUserDeviceMem需要手动释放内存
    CHK_RET(link->GetIndOpRemoteMemDetails(
        &linkRoce.remoteUserHostMem, &linkRoce.remoteUserHostMemCount, HcclMemType::HCCL_MEM_TYPE_HOST));
    CHK_RET(link->GetIndOpRemoteMemDetails(
        &linkRoce.remoteUserDeviceMem, &linkRoce.remoteUserHostMemCount, HcclMemType::HCCL_MEM_TYPE_DEVICE));
    HCCL_DEBUG("[%s] finish set remoteMem info", __func__);

    // 填充notifyValue和notifyValueKey信息到linkRoce中
    std::vector<AddrKey> notifyValueAddrKey;
    CHK_RET(link->GetLocalNotifyValueAddrKey(notifyValueAddrKey));
    linkRoce.notifyValue = notifyValueAddrKey[0].addr;
    linkRoce.notifyValueKey = notifyValueAddrKey[0].key;

    // 填充QP信息到linkRoce中
    std::vector<HcclQpInfoV2> aiQpInfos;
    CHK_RET(link->GetAiQpInfo(aiQpInfos));
    u32 qpNum = aiQpInfos.size();
    if (qpNum > RDMA_QP_MAX_NUM || qpNum < 1) {
        return HCCL_E_INTERNAL;
    }
    std::copy_n(aiQpInfos.begin(), qpNum, linkRoce.QpInfo);
    linkRoce.qpsPerConnection = qpNum - static_cast<u32>(qpNum > 1); // 多QP数量或单QP模式

    // 填充localNotify和remoteNotify信息到linkRoce中
    std::vector<AddrKey> notifyAddrKey;
    std::vector<HcclSignalInfo> signalInfos;
    CHK_RET(link->GetLocalRdmaNotify(signalInfos));
    CHK_RET(link->GetRemoteRdmaNotifyAddrKey(notifyAddrKey));
    if ((signalInfos.size() != notifyAddrKey.size()) || (signalInfos.size() < RDMA_NOTIFY_MIN_NUM)
        || (signalInfos.size() > RDMA_NOTIFY_MAX_NUM) || (notifyAddrKey.size() < RDMA_NOTIFY_MIN_NUM)
        || (notifyAddrKey.size() > RDMA_NOTIFY_MAX_NUM)
        || ((signalInfos.size() - RDMA_NOTIFY_MIN_NUM) % linkRoce.qpsPerConnection) != 0
        || ((notifyAddrKey.size() - RDMA_NOTIFY_MIN_NUM) % linkRoce.qpsPerConnection) != 0) {
        return HCCL_E_INTERNAL;
    }
    u64 notifyNum = (notifyAddrKey.size() - RDMA_NOTIFY_MIN_NUM) / linkRoce.qpsPerConnection
                    - static_cast<u32>(linkRoce.qpsPerConnection > 1);
    linkRoce.singleQPNotifyNum = notifyNum;

    u64 len = signalInfos.size() * sizeof(HcclSignalInfo);
    DeviceMem localNotifyListMem;
    CHK_RET(CopyVectorToDeviceMem(len, localNotifyListMem, signalInfos));
    linkRoce.localNotifyList = reinterpret_cast<u64>(localNotifyListMem.ptr());
    channelParamMemList_.emplace_back(std::move(localNotifyListMem));

    len = notifyAddrKey.size() * sizeof(AddrKey);
    DeviceMem remoteNotifyListMem;
    CHK_RET(CopyVectorToDeviceMem(len, remoteNotifyListMem, notifyAddrKey));
    linkRoce.remoteNotifyList = reinterpret_cast<u64>(remoteNotifyListMem.ptr());
    channelParamMemList_.emplace_back(std::move(remoteNotifyListMem));

    remoteRes.roceNotifyNum = linkRoce.singleQPNotifyNum;
    remoteRes.qpNum = linkRoce.qpsPerConnection;

    return HCCL_SUCCESS;
}

HcclResult ChannelManager::DeepCopyH2DchannelParam(
    const HcclIndOpChannelRemoteResV3& hostChannelParam, HcclIndOpChannelRemoteResV3& deviceChannelParam)
{
    deviceChannelParam = hostChannelParam;
    // 拷贝remoteResV2

    if (hostChannelParam.remoteResV2 != nullptr && hostChannelParam.listNum > 0) {
        // 为设备端的remoteResV2数组分配内存（注意：这个数组存放的是HcclIndOpChannelRemoteResV2结构体）
        size_t remoteResV2ArraySize = sizeof(HcclIndOpChannelRemoteResV2) * hostChannelParam.listNum;
        std::shared_ptr<DeviceMem> deviceRemoteResV2Array;
        EXCEPTION_CATCH(
            (deviceRemoteResV2Array = std::make_shared<DeviceMem>(DeviceMem::alloc(remoteResV2ArraySize))),
            return HCCL_E_PTR);

        // 为每个数组元素进行深度拷贝，并保存设备内存和主机结构体（指针已调整）
        std::vector<DeviceMem> elementMemories; // 保存每个元素分配的设备内存（包括内部指针数据）
        std::vector<HcclIndOpChannelRemoteResV2> hostRemoteResV2Array(hostChannelParam.listNum);

        for (uint32_t i = 0; i < hostChannelParam.listNum; ++i) {
            HcclIndOpChannelRemoteResV2 hostElement = hostChannelParam.remoteResV2[i];
            HcclIndOpChannelRemoteResV2 deviceElement;
            // 深度拷贝一个元素到设备内存，并返回设备内存中的结构体布局（host端）
            CHK_RET(DeepCopyH2DChannelRemoteResV2(hostElement, deviceElement));
            // 保存调整后的主机端结构体（其指针指向设备内存）
            hostRemoteResV2Array[i] = deviceElement;
        }

        // 将主机端的结构体数组（指针已调整）拷贝到设备内存数组
        CHK_RET(hrtMemSyncCopy(
            deviceRemoteResV2Array.get()->ptr(), remoteResV2ArraySize, hostRemoteResV2Array.data(),
            remoteResV2ArraySize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

        // 更新设备端参数中的remoteResV2指针
        deviceChannelParam.remoteResV2 = static_cast<HcclIndOpChannelRemoteResV2*>(deviceRemoteResV2Array.get()->ptr());
        channelParamMemVector_.push_back(std::move(deviceRemoteResV2Array));
    } else {
        HCCL_ERROR("[%s]invalid hostChannelParam", __func__);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::DeepCopyH2DChannelRemoteResV2(
    const HcclIndOpChannelRemoteResV2& hostRemoteResV2, HcclIndOpChannelRemoteResV2& deviceRemoteResV2)
{
    // 复制基本成员
    deviceRemoteResV2 = hostRemoteResV2;
    // 根据通信类型处理不同的通道
    if (hostRemoteResV2.isUsedRdma) {
        // 处理RoCE通道
        CHK_RET(DeepCopyH2DChannelRoce(hostRemoteResV2.channelRoce, deviceRemoteResV2.channelRoce));
    } else {
        // 处理P2P通道
        CHK_RET(DeepCopyH2DChannelP2p(hostRemoteResV2.channelP2p, deviceRemoteResV2.channelP2p));
    }
    return HCCL_SUCCESS;
}

HcclResult
ChannelManager::DeepCopyH2DChannelRoce(const HcclChannelRoce& hostChannelRoce, HcclChannelRoce& deviceChannelRoce)
{
    // 复制基本成员
    deviceChannelRoce = hostChannelRoce;
    // 处理remoteUserHostMem
    if (hostChannelRoce.remoteUserHostMem != nullptr && hostChannelRoce.remoteUserHostMemCount > 0) {
        size_t remoteUserHostMemSize = hostChannelRoce.remoteUserHostMemCount * sizeof(MemDetails);
        std::shared_ptr<DeviceMem> deviceMem;
        EXCEPTION_CATCH(
            (deviceMem = std::make_shared<DeviceMem>(DeviceMem::alloc(remoteUserHostMemSize))), return HCCL_E_PTR);
        CHK_RET(hrtMemSyncCopy(
            deviceMem.get()->ptr(), remoteUserHostMemSize, hostChannelRoce.remoteUserHostMem, remoteUserHostMemSize,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        deviceChannelRoce.remoteUserHostMem = reinterpret_cast<MemDetails*>(deviceMem.get()->ptr());
        channelParamMemVector_.push_back(std::move(deviceMem));
    } else {
        deviceChannelRoce.remoteUserHostMem = nullptr;
    }
    // 处理remoteUserDeviceMem
    if (hostChannelRoce.remoteUserDeviceMem != nullptr && hostChannelRoce.remoteUserDeviceMemCount > 0) {
        size_t remoteUserDeviceMemSize = hostChannelRoce.remoteUserDeviceMemCount * sizeof(MemDetails);
        std::shared_ptr<DeviceMem> deviceMem;
        EXCEPTION_CATCH(
            (deviceMem = std::make_shared<DeviceMem>(DeviceMem::alloc(remoteUserDeviceMemSize))), return HCCL_E_PTR);
        CHK_RET(hrtMemSyncCopy(
            deviceMem.get()->ptr(), remoteUserDeviceMemSize, hostChannelRoce.remoteUserDeviceMem,
            remoteUserDeviceMemSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        deviceChannelRoce.remoteUserDeviceMem = reinterpret_cast<MemDetails*>(deviceMem.get()->ptr());
        channelParamMemVector_.push_back(std::move(deviceMem));
    } else {
        deviceChannelRoce.remoteUserDeviceMem = nullptr;
    }

    return HCCL_SUCCESS;
}

HcclResult ChannelManager::DeepCopyH2DChannelP2p(const HcclChannelP2p& hostChannelP2p, HcclChannelP2p& deviceChannelP2p)
{
    // 复制基本成员
    deviceChannelP2p = hostChannelP2p;
    // 处理remoteUserMem
    if (hostChannelP2p.remoteUserMem != nullptr && hostChannelP2p.remoteUserMemCount > 0) {
        size_t remoteUserMemSize = hostChannelP2p.remoteUserMemCount * sizeof(HcclMem);
        std::shared_ptr<DeviceMem> deviceMem;
        EXCEPTION_CATCH(
            (deviceMem = std::make_shared<DeviceMem>(DeviceMem::alloc(remoteUserMemSize))), return HCCL_E_PTR);
        CHK_RET(hrtMemSyncCopy(
            deviceMem.get()->ptr(), remoteUserMemSize, hostChannelP2p.remoteUserMem, remoteUserMemSize,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        deviceChannelP2p.remoteUserMem = reinterpret_cast<HcclMem*>(deviceMem.get()->ptr());
        channelParamMemVector_.push_back(std::move(deviceMem));
    } else {
        deviceChannelP2p.remoteUserMem = nullptr;
    }
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::ReleaseChannelParam(HcclIndOpChannelRemoteResV3& channelParam)
{
    // 释放remoteResV2
    if (channelParam.remoteResV2 != nullptr) {
        for (uint32_t i = 0; i < channelParam.listNum; ++i) {
            HcclIndOpChannelRemoteResV2& remoteRes = channelParam.remoteResV2[i];
            if (remoteRes.isUsedRdma) {
                if (remoteRes.channelRoce.remoteUserHostMem != nullptr) {
                    free(remoteRes.channelRoce.remoteUserHostMem);
                }
                if (remoteRes.channelRoce.remoteUserDeviceMem != nullptr) {
                    free(remoteRes.channelRoce.remoteUserDeviceMem);
                }
            } else {
                if (remoteRes.channelP2p.remoteUserMem != nullptr) {
                    free(remoteRes.channelP2p.remoteUserMem);
                }
            }
        }
    }
    free(channelParam.remoteResV2);
    channelParam.remoteResV2 = nullptr;

    // 将kernel下发时临时分配的deviceMem一起销毁
    channelParamMemVector_.clear();
    channelParamMemList_.clear();
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::AicpuChannelInit(
    const std::string& commId, const std::string& tag, CommEngine engine, const OpCommTransport& opTransportResponse,
    ChannelHandle* channelList, uint32_t listNum)
{
    HcclIndOpChannelRemoteResV3 channelParam{};
    CHK_SAFETY_FUNC_RET(memset_s(&channelParam, sizeof(channelParam), 0, sizeof(channelParam)));
    uint64_t beginTime = hrtMsprofSysCycleTime();
    // channelParam资源参数填充
    CHK_SAFETY_FUNC_RET(strncpy_s(channelParam.hcomId, HCOMID_MAX_LENGTH, commId.c_str(), HCOMID_MAX_LENGTH - 1));
    CHK_SAFETY_FUNC_RET(strncpy_s(channelParam.channelTag, TAG_MAX_LENGTH, tag.c_str(), TAG_MAX_LENGTH - 1));
    channelParam.engine = engine;
    channelParam.localUserRank = userRank_;
    channelParam.multiQpThreshold = GetExternalInputMultiQpThreshold();

    // 为device侧的channelList分配内存
    DeviceMem deviceChannelList = DeviceMem::alloc(listNum * sizeof(ChannelHandle));
    CHK_PTR_NULL(deviceChannelList.ptr());
    channelParam.channelList = static_cast<void*>(deviceChannelList.ptr());
    channelParam.listNum = listNum;

    // 将建链获取的远端数据填充到channelParam
    HcclResult ret = ParseChannelRemoteDataToMem(opTransportResponse, channelParam);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] ParseChannelRemoteDataToMem failed, return [%d].", __func__, ret);
        ReleaseChannelParam(channelParam);
        return ret;
    }

    // 创建局部流
    Stream localStream(StreamType::STREAM_TYPE_ONLINE);
    constexpr u32 aicpuStreamMode = 1;
    CHK_RET(hrtStreamSetMode(localStream.ptr(), aicpuStreamMode));

    // 将channelParam内部的host内存拷贝成device内存
    HcclIndOpChannelRemoteResV3 deviceChannelParam = channelParam;
    CHK_RET(DeepCopyH2DchannelParam(channelParam, deviceChannelParam));

    DeviceMem addr = DeviceMem::alloc(sizeof(deviceChannelParam));
    CHK_PTR_NULL(addr.ptr());
    CHK_RET(hrtMemSyncCopy(
        addr.ptr(), sizeof(deviceChannelParam), &deviceChannelParam, sizeof(deviceChannelParam),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    // 下kernel
    std::string kernelName = "RunAicpuIndOpChannelInit";
    struct InitTask {
        u64 context;
        bool isCustom;
    };
    InitTask customInitTask = {};
    customInitTask.context = reinterpret_cast<u64>(addr.ptr());
    customInitTask.isCustom = false;

    u16 timeOut = NOTIFY_DEFAULT_WAIT_TIME > std::numeric_limits<uint16_t>::max() ?
                      std::numeric_limits<uint16_t>::max() :
                      NOTIFY_DEFAULT_WAIT_TIME;
    CHK_RET(AicpuAclKernelLaunch(
        localStream.ptr(), static_cast<void*>(&customInitTask), sizeof(customInitTask), binHandle_, kernelName, true,
        timeOut));
    CHK_RET(hcclStreamSynchronize(localStream.ptr(), CommConfiger::GetInstance().GetCommConfigExecTimeOut(tag)));

    // 将device侧的channelList拷贝回host侧的channelList
    CHK_RET(hrtMemSyncCopy(
        channelList, listNum * sizeof(ChannelHandle), deviceChannelList.ptr(), listNum * sizeof(ChannelHandle),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    // 手动释放channelParam中申请的内存
    CHK_RET(ReleaseChannelParam(channelParam));
    const std::string profName = "RunAicpuIndOpChannelInit";
    HCCL_DEBUG("[%s] RunAicpuIndOpChannelInit", __func__);
    // 上报初始化kernel的时间
    HcommProfilingReportKernel(beginTime, profName.c_str());
    return HCCL_SUCCESS;
}

const std::map<CommEngine, std::string> COMM_ENGINE_TYPE_STR_MAP{
    {CommEngine::COMM_ENGINE_CPU, "host_cpu"},     {CommEngine::COMM_ENGINE_CPU_TS, "host_cpu_ts"},
    {CommEngine::COMM_ENGINE_AICPU, "aicpu"},      {CommEngine::COMM_ENGINE_AICPU_TS, "aicpu_ts"},
    {CommEngine::COMM_ENGINE_AIV, "aiv"},          {CommEngine::COMM_ENGINE_CCU, "ccu"},
    {CommEngine::COMM_ENGINE_RESERVED, "reserved"}};

std::string GetCommEngineEnumStr(CommEngine engine)
{
    auto iter = COMM_ENGINE_TYPE_STR_MAP.find(engine);
    if (iter == COMM_ENGINE_TYPE_STR_MAP.end()) {
        return "CommEngine=" + std::to_string(engine);
    } else {
        return iter->second;
    }
}

HcclResult ChannelManager::ChannelCommCreate(
    const std::string& commId, CommEngine engine, const HcclChannelDesc* channelDescList, uint32_t listNum,
    ChannelHandle* channelList)
{
    CHK_RET(CheckChannelParam(engine, channelDescList, listNum));

    // channel复用，以tag + engine + remoterank + channelProtocol 作为channel标识
    std::vector<HcclChannelDesc> needCreateDescs;
    std::vector<uint32_t> needCreateIndices;
    std::string tag = commId;
    CHK_RET(PrepareHandleArray(tag, engine, channelDescList, listNum, channelList, needCreateDescs, needCreateIndices));

    // 对未复用的channelDesc进行建链
    if (needCreateDescs.size() > 0) {
        // 构造建链param
        OpCommTransport opCommTransport = BuildChannelRequests(needCreateDescs);
        std::string linkTag = commId + "_" + GetCommEngineEnumStr(engine);
        bool isAicpuModeEn = false;
        if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
            isAicpuModeEn = true;
        }
        std::vector<HcclMemHandle> memHandles = CollectMemHandles(needCreateDescs);
        CHK_RET(channelCallbacks_.indOpTransportAlloc(
            linkTag, opCommTransport, isAicpuModeEn, memHandles.data(), static_cast<uint32_t>(memHandles.size())));

        uint32_t level0 = 0;
        std::vector<LINK> links = opCommTransport[level0][level0].links;
        uint32_t newDescNum = needCreateDescs.size();
        // 创建host或device侧channel句柄
        if (isAicpuModeEn) {
            // Kernel下发恢复
            if (!callbacks_.getAicpuCommState()) {
                HcclResult ret = callbacks_.kernelLaunchAicpuCommInit();
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[%s] kernelLaunchAicpuCommInit failed, return [%d].", __func__, ret), ret);
                callbacks_.setAicpuCommState(true);
            }
            std::unique_ptr<ChannelHandle[]> tmpChannelList = std::make_unique<ChannelHandle[]>(newDescNum);
            CHK_RET(AicpuChannelInit(commId, tag, engine, opCommTransport, tmpChannelList.get(), newDescNum));
            for (u32 i = 0; i < newDescNum; i++) {
                uint32_t arrayIndex = needCreateIndices[i];
                channelList[arrayIndex] = tmpChannelList[i];
                CHK_RET(RegisterHandle(tag, engine, needCreateDescs[i], tmpChannelList[i]));
                ChannelHandle channelHandle = reinterpret_cast<ChannelHandle>(links[i].get());
                CHK_RET(RegisterHandleHDPair(tmpChannelList[i], channelHandle));
            }
        } else {
            for (u32 i = 0; i < newDescNum; i++) {
                uint32_t arrayIndex = needCreateIndices[i];
                ChannelHandle channelHandle = reinterpret_cast<ChannelHandle>(links[i].get());
                channelList[arrayIndex] = channelHandle;
                CHK_RET(RegisterHandle(tag, engine, needCreateDescs[i], channelHandle));
            }
        }
        // 保存link
        for (auto& link : links) {
            // 设置成员变量保存link
            channelLinks_.push_back(link);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::ChannelCommGetNotifyNum(ChannelHandle channel, uint32_t* notifyNum)
{
    CHK_RET(IsChannelExist(channel));
    ChannelHandle hostchannel;
    CHK_RET(GetHostChannel(channel, hostchannel));

    Transport* transportPtr = reinterpret_cast<Transport*>(hostchannel);
    *notifyNum = transportPtr->GetNotifyNum();
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::ChannelCommDestroy(ChannelHandle* channelList, uint32_t channelNum)
{
    for (uint32_t i = 0; i < channelNum; ++i) {
        UnregisterHandle(channelList[i]);
        channelList[i] = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::ChannelCommGetHcclBuffer(ChannelHandle channel, CommBuffer* buffer)
{
    ChannelHandle hostchannel;
    CHK_RET(IsChannelExist(channel));
    CHK_RET(GetHostChannel(channel, hostchannel));
    Transport* transportPtr = reinterpret_cast<Transport*>(hostchannel);

    buffer->addr = nullptr;
    CHK_RET(transportPtr->GetRemoteMem(UserMemType::INPUT_MEM, &buffer->addr));
    CHK_PTR_NULL(buffer->addr);
    u64 tempSize = 0;
    CHK_RET(transportPtr->GetRemoteMemSize(UserMemType::INPUT_MEM, tempSize));
    buffer->size = static_cast<uint64_t>(tempSize);
    buffer->type = HCCL_MEM_TYPE_DEVICE;
    HCCL_INFO(
        "[%s]channel[%llu] channelKey[%s] get remote hccl buffer success, remote addr[%p], size[%llu]", __func__,
        channel, keyMap_[channel].c_str(), buffer->addr, buffer->size);
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::ChannelCommGetRemoteMem(ChannelHandle channel, HcclMem** remoteMem, uint32_t* memNum)
{
    CHK_RET(IsChannelExist(channel));
    ChannelHandle hostchannel;
    CHK_RET(GetHostChannel(channel, hostchannel));
    Transport* transportPtr = reinterpret_cast<Transport*>(hostchannel);

    CHK_RET(transportPtr->GetIndOpRemoteMem(remoteMem, memNum));
    HCCL_INFO("[%s]get remote mem success, mem num[%u]", __func__, *memNum);
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::ReleaseChannel()
{
    for (auto& link : channelLinks_) {
        if (link != nullptr) {
            if (link->DeInit() != HCCL_SUCCESS) {
                HCCL_ERROR("[%s]transport[%p] deinit failed.", __func__, link.get());
            }
        }
    }
    channelLinks_.clear();
    return HCCL_SUCCESS;
}

HcclResult ChannelManager::SetHcclQos(u32 hcclQos)
{
    HCCL_INFO("[ChannelManager] [SetHcclQos] hcclQos[%u]", hcclQos);
    hcclQos_ = hcclQos;
    return HCCL_SUCCESS;
}
} // namespace hccl
