/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel_aicpu_mgr.h"
#include "ub_transport_lite_impl.h"
#include "p2p_transport_lite_impl.h"
#include "roce_transport_lite_impl.h"
#include "aicpu_res_package_helper.h"
#include "aicpu_indop_env.h"
#include "aicpu_task_cache_manager.h"
#include "adapter_hal_pub.h"
#include "log.h"

ChannelAicpuMgr::ChannelAicpuMgr(hccl::HcclCommDfxLite& dfx, const hccl::HcclTopoInfo& topoInfo)
    : dfx_(dfx),
      topoInfo_(topoInfo)
{}

HcclResult ChannelAicpuMgr::AllocChannelResource(HcclChannelUrmaRes* commParam)
{
    CHK_PTR_NULL(commParam);
    HCCL_INFO(
        "[ChannelAicpuMgr][%s] deviceLogicId[%d], devicePhyId[%u], deviceType[%d], commParam->channelList[%p], "
        "commParam->listNum[%u], commParam->uniqueIdAddr[%p], commParam->uniqueIdSize[%u]",
        __func__, topoInfo_.deviceLogicId, topoInfo_.devicePhyId, topoInfo_.deviceType, commParam->channelList,
        commParam->listNum, commParam->uniqueIdAddr, commParam->uniqueIdSize);
    CHK_RET(InitUrmaChannel(commParam));
    return HCCL_SUCCESS;
}

HcclResult ChannelAicpuMgr::InitUrmaChannel(HcclChannelUrmaRes* commParam) { return ProcessUrmaRes(commParam, true); }

HcclResult ChannelAicpuMgr::ProcessUrmaRes(HcclChannelUrmaRes* commParam, bool isInit)
{
    HCCL_INFO(
        "[ChannelAicpuMgr][%s] commParam->uniqueIdAddr[%p], commParam->uniqueIdSize[%u]", __func__,
        commParam->uniqueIdAddr, commParam->uniqueIdSize);
    ChannelHandle* channelList = static_cast<ChannelHandle*>(commParam->channelList);
    u8* currentSrcAddr = static_cast<u8*>(commParam->uniqueIdAddr);
    u32* addSize = static_cast<u32*>(commParam->channelSizeAddr);
    CHK_PTR_NULL(channelList);
    CHK_PTR_NULL(currentSrcAddr);
    CHK_PTR_NULL(addSize);

    for (u32 index = 0; index < commParam->listNum; index++) {
        std::vector<char> data(*addSize);

        CHK_SAFETY_FUNC_RET(memcpy_s(data.data(), data.size(), currentSrcAddr, *addSize));
        currentSrcAddr += *addSize;
        addSize++;
        Hccl::AicpuResPackageHelper helper;
        auto dataVec = helper.ParsePackedData(data);

        Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM;
        if (static_cast<u32>(resType) >= dataVec.size()) {
            HCCL_ERROR("[ChannelAicpuMgr][%s] fail, resType[%d], dataVec size[%zu]", __func__, resType, dataVec.size());
            return HCCL_E_PARA;
        }

        ChannelHandle channelHandle{0};
        if (isInit) {
            CHK_RET(ParsePackData(dataVec[resType].data, channelHandle));
            channelList[index] = channelHandle;
            CHK_RET(RegisterChannelCacheCallback(channelHandle));
            dfx_.AddChannelRemoteRankId(channelHandle, commParam->remoteRankList[index]);
        } else {
            channelHandle = channelList[index];
            if (!transportMap_.count(channelHandle)) {
                HCCL_ERROR("[ChannelAicpuMgr][%s] fail, resType[%d], current ChannelHandle nullptr", __func__, resType);
                return HCCL_E_PARA;
            }
            CHK_RET(ResumePackData(dataVec[resType].data, channelHandle));
        }

        HCCL_INFO(
            "[ChannelAicpuMgr][%s] index[%u], currentSrcAddr[%p], channelSizeAddr[%p], channelHandle[0x%llx]", __func__,
            index, currentSrcAddr, commParam->channelSizeAddr, static_cast<unsigned long long>(channelHandle));
    }

    return HCCL_SUCCESS;
}

namespace {
template <typename T>
HcclResult CreateAndInsertTransport(
    std::vector<char>& uniqueId, ChannelHandle& handle, T*& outPtr,
    std::unordered_map<ChannelHandle, std::unique_ptr<Hccl::BaseTransportLiteImpl>>& transportMap)
{
    std::unique_ptr<T> impl;
    EXCEPTION_CATCH(impl = std::make_unique<T>(uniqueId), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(impl);
    outPtr = impl.get();
    handle = reinterpret_cast<uint64_t>(impl.get());
    transportMap.insert({handle, std::move(impl)});
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ChannelAicpuMgr::ParsePackData(std::vector<char>& data, ChannelHandle& handle)
{
    HCCL_DEBUG("[ChannelAicpuMgr][%s] data: ptr[%p], size[%zu]", __func__, data.data(), data.size());
    Hccl::BinaryStream binaryStream(data);

    std::vector<char> transpUniqueId;
    binaryStream >> transpUniqueId;

    Hccl::BinaryStream binaryStreamForType(transpUniqueId);
    u32 transType;
    binaryStreamForType >> transType;
    HCCL_INFO("[ChannelAicpuMgr][ParsePackData] transType[%u]", transType);

    if (transType == Hccl::TransportType::UB || transType == Hccl::TransportType::UBoE) {
        Hccl::UbTransportLiteImpl* ubPtr = nullptr;
        CHK_RET(CreateAndInsertTransport<Hccl::UbTransportLiteImpl>(transpUniqueId, handle, ubPtr, transportMap_));
        ubPtr->SetTaskExceptionEnable(hcomm::GetTaskExceptionEnable());
    } else if (transType == Hccl::TransportType::P2P) {
        Hccl::P2PTransportLiteImpl* p2pPtr = nullptr;
        CHK_RET(CreateAndInsertTransport<Hccl::P2PTransportLiteImpl>(transpUniqueId, handle, p2pPtr, transportMap_));
    } else if (transType == Hccl::TransportType::ROCE) {
        Hccl::RoceTransportLiteImpl* rocePtr = nullptr;
        CHK_RET(CreateAndInsertTransport<Hccl::RoceTransportLiteImpl>(transpUniqueId, handle, rocePtr, transportMap_));
    } else {
        HCCL_ERROR("[ChannelAicpuMgr][ParsePackData] unsupported transportType[%u]", transType);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult ChannelAicpuMgr::ResumePackData(std::vector<char>& data, ChannelHandle& handle)
{
    Hccl::BinaryStream binaryStream(data);
    std::vector<char> transpUniqueId;
    binaryStream >> transpUniqueId;

    auto it = transportMap_.find(handle);
    CHK_PRT_RET(
        it == transportMap_.end(),
        HCCL_ERROR("[ChannelAicpuMgr][ResumePackData] channel handle[0x%llx] not found", handle), HCCL_E_PARA);

    auto* ub = dynamic_cast<Hccl::UbTransportLiteImpl*>(it->second.get());
    CHK_PRT_RET(
        ub == nullptr,
        HCCL_ERROR("[ChannelAicpuMgr][ResumePackData] transport is not UB type for handle[0x%llx]", handle),
        HCCL_E_INTERNAL);
    return ub->Resume(transpUniqueId);
}

HcclResult ChannelAicpuMgr::Resume(HcclChannelUrmaRes* commParam)
{
    CHK_PTR_NULL(commParam);
    CHK_RET(ProcessUrmaRes(commParam, false));
    return HCCL_SUCCESS;
}

HcclResult ChannelAicpuMgr::RegisterChannelCacheCallback(ChannelHandle channel)
{
    // 目前aicpu task cache只支持UB.URMA协议，从合并后的transportMap_中查找UB条目
    auto it = transportMap_.find(channel);
    if (it != transportMap_.end()) {
        auto* ub = dynamic_cast<Hccl::UbTransportLiteImpl*>(it->second.get());
        if (ub != nullptr) {
            HCCL_INFO(
                "[ChannelAicpuMgr][RegisterChannelCacheCallback] register cache callback for channel[0x%016llx]",
                channel);
            CHK_RET(ub->SetNeedCacheTaskCallback(hcomm::AicpuTaskCacheManager::NeedCacheTask));
            CHK_RET(ub->SetAddWqeArrayCallback(hcomm::AicpuTaskCacheManager::AddWqeArray));
            HCCL_INFO(
                "[ChannelAicpuMgr][RegisterChannelCacheCallback] register cache callback for channel[0x%016llx] "
                "success",
                channel);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ChannelAicpuMgr::Clean()
{
    for (auto& impl_pair : transportMap_) {
        auto& impl = impl_pair.second;
        auto* ub = dynamic_cast<Hccl::UbTransportLiteImpl*>(impl.get());
        if (ub != nullptr) {
            CHK_RET(ub->Clean());
        }
    }
    HCCL_INFO("[%s][ChannelAicpuMgr]Clean() finished", __func__);
    return HCCL_SUCCESS;
}
