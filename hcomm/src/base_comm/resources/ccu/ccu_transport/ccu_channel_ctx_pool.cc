/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_channel_ctx_pool.h"

#include "ccu_device_pub.h"
#include "orion_adpt_utils.h"

namespace hcomm {

constexpr uint32_t CCU_DEFAULT_REQUEST_SQ_SIZE = 128;
constexpr uint32_t CCU_DEFAULT_REQUEST_CHANNEL_NUM = 1;
constexpr uint32_t CCU_DEFAULT_REQUEST_JETTY_NUM = 0; // 申请数量为0时，由平台层决定提供数量

CcuChannelCtxPool::CcuChannelCtxPool(int32_t userDevId) : userDevId_(userDevId) {}

CcuChannelCtxPool::~CcuChannelCtxPool()
{
    // 对象析构时清空多个map，batchMap_中元素的jettys清空触发ccuJetty析构释放
    (void)ReleaseConfirmedChannelRes();
}

HcclResult CcuChannelCtxPool::ResourceBatch::Init(const std::vector<CcuChannelInfo>& channelInfos)
{
    const uint32_t channelNum = channelInfos.size();
    channelIdKeys.reserve(channelNum);
    availableChannelIdKeys.reserve(channelNum);
    for (const auto& channelInfo : channelInfos) {
        const auto dieId = channelInfo.dieId;
        const auto channelId = channelInfo.channelId;
        channelIdKeys.emplace_back(dieId, channelId);
        availableChannelIdKeys.emplace_back(dieId, channelId);

        for (const auto& jettyInfo : channelInfo.jettyInfos) {
            const auto taJettyId = jettyInfo.taJettyId;
            const auto jettyIdKey = std::make_pair(dieId, taJettyId);
            if (jettys.find(jettyIdKey) != jettys.end()) {
                continue;
            }

            std::unique_ptr<CcuJetty> ccuJetty;
            CHK_RET(CcuCreateJetty(key, jettyInfo, ccuJetty));

            jettys[jettyIdKey] = std::move(ccuJetty);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuChannelCtxPool::PrepareCreate(const std::vector<Hccl::LinkData>& links, uint32_t sqSize)
{
    std::lock_guard<std::mutex> lock(mtx_);

    CHK_PRT_RET(
        links.empty(),
        HCCL_INFO("[CcuChannelCtxPool][%s] passed, links is empty, userDevId[%d].", __func__, userDevId_),
        HcclResult::HCCL_SUCCESS);

    for (const auto& link : links) {
        auto it = allocatedChannelIdMap_.find(link);
        if (it != allocatedChannelIdMap_.end()) {
            HCCL_INFO(
                "[CcuChannelCtxPool][%s] passed, link[%s] is already allocated, "
                "userDevId[%d].",
                __func__, link.Describe().c_str(), userDevId_);
            continue;
        }

        const auto& locAddr = link.GetLocalAddr();
        ResourceBatch* batchPtr = nullptr;
        auto ret = GetAvailableBatch(locAddr, batchPtr, sqSize);
        CHK_PRT_RET(
            ret == HcclResult::HCCL_E_UNAVAIL,
            HCCL_WARNING(
                "[CcuChannelCtxPool][%s] failed to alloc ccu channels, ccu resources "
                "are unavailable, locAddr[%s], userDevId[%d], sqSize[%u].",
                __func__, locAddr.Describe().c_str(), userDevId_, sqSize),
            ret);
        CHK_RET(ret);

        ChannelIdKey channelIdKey = batchPtr->availableChannelIdKeys.back();
        batchPtr->availableChannelIdKeys.pop_back();
        allocatedChannelIdMap_[link] = channelIdKey;
        channelRemoteRankIdMap_[channelIdKey] = link.GetRemoteRankId();
        usedChannelCntMap_[channelIdKey.first] += 1;

        HCCL_INFO(
            "[CcuChannelCtxPool][%s] allocated new channelId[%u] of die[%u] to link[%s], "
            "userDevId[%d], sqSize[%u].",
            __func__, channelIdKey.second, channelIdKey.first, link.Describe().c_str(), userDevId_, sqSize);
    }

    isReleased_ = false;
    return HcclResult::HCCL_SUCCESS;
}

// 当前以locAddr为粒度调用，根据locAddr可以找到已申请的批次，如果资源充足则复用，不足则按新批次申请资源
HcclResult CcuChannelCtxPool::GetAvailableBatch(const BatchKey& batchKey, ResourceBatch*& batchPtr, uint32_t sqSize)
{
    // 当前以locAddr作为batchKey，不同本端不能复用资源
    if (FindAvailableBatch(batchKey, batchPtr)) {
        return HcclResult::HCCL_SUCCESS;
    }
    // 已有的资源不足，需要新增资源，获取的channel数量可能超过申请数量
    CommAddr commAddr{};
    CHK_RET(IpAddressToCommAddr(batchKey, commAddr));
    // 使用传入的sqSize，如果为0xFFFFFFFF则使用默认值
    uint32_t actualSqSize = (sqSize != 0xFFFFFFFF) ? sqSize : CCU_DEFAULT_REQUEST_SQ_SIZE;
    const CcuChannelPara channelPara{
        commAddr, CCU_DEFAULT_REQUEST_CHANNEL_NUM, CCU_DEFAULT_REQUEST_JETTY_NUM, actualSqSize};
    std::vector<CcuChannelInfo> channelInfos;
    auto ret = CcuAllocChannels(userDevId_, channelPara, channelInfos);
    CHK_PRT_RET(
        ret == HcclResult::HCCL_E_UNAVAIL,
        HCCL_WARNING(
            "[CcuChannelCtxPool][%s] failed to alloc ccu channels, ccu resources "
            "are unavailable, locAddr[%s] userDevId[%d].",
            __func__, batchKey.Describe().c_str(), userDevId_),
        ret);
    CHK_RET(ret);
    // 如果新增资源保存失败，手动释放避免泄露
    ret = CreateAndSaveNewBatch(batchKey, channelInfos, batchPtr);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CcuChannelCtxPool][%s] failed, try to release temp ccu resources, locAddr[%s], "
            "userDevId[%d], .",
            __func__, batchKey.Describe().c_str(), userDevId_);
        for (const auto& channelInfo : channelInfos) {
            const auto dieId = channelInfo.dieId;
            const auto channelId = channelInfo.channelId;
            CHK_RET(CcuReleaseChannel(userDevId_, dieId, channelId));
        }
        return ret;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuChannelCtxPool::CreateAndSaveNewBatch(
    const BatchKey& batchKey, const std::vector<CcuChannelInfo> channelInfos, ResourceBatch*& batchPtr)
{
    // todo: 需要检查资源管理是否存在泄露可能
    auto& batches = batchMap_[batchKey];
    std::unique_ptr<ResourceBatch> newBatch{nullptr};
    newBatch.reset(new (std::nothrow) ResourceBatch(batchKey));
    CHK_PTR_NULL(newBatch);
    CHK_RET(newBatch->Init(channelInfos));
    for (const auto& channelInfo : channelInfos) {
        const auto dieId = channelInfo.dieId;
        const auto channelIdKey = std::make_pair(dieId, channelInfo.channelId);

        std::vector<CcuJetty*> jettys;
        for (const auto& jettyInfo : channelInfo.jettyInfos) {
            const auto jettyIdKey = std::make_pair(dieId, jettyInfo.taJettyId);
            jettys.emplace_back(newBatch->jettys[jettyIdKey].get());
        }

        channelJettyInfoMap_.emplace(channelIdKey, std::make_pair(channelInfo, jettys));
    }

    batches.push_back(std::move(newBatch));
    ResourceBatch* rawBatch = batches.back().get();
    for (const auto& channelInfo : channelInfos) {
        channelToBatch_[std::make_pair(channelInfo.dieId, channelInfo.channelId)] = rawBatch;
    }
    batchPtr = rawBatch;
    return HcclResult::HCCL_SUCCESS;
}

bool CcuChannelCtxPool::FindAvailableBatch(const BatchKey& batchKey, ResourceBatch*& batchPtr) const
{
    auto it = batchMap_.find(batchKey);
    if (it == batchMap_.end()) {
        return false;
    }

    auto& batches = it->second;
    // 从后往前遍历：越晚创建的 batch 越可能留有可复用槽位（新申请通常分配自尾部），
    // 优先命中可减少扫描；中间 batch 释放出的槽位同样可被后续创建复用
    for (auto batchIter = batches.rbegin(); batchIter != batches.rend(); ++batchIter) {
        if (*batchIter != nullptr && !(*batchIter)->availableChannelIdKeys.empty()) {
            batchPtr = batchIter->get();
            return true;
        }
    }
    return false;
}

HcclResult
CcuChannelCtxPool::GetChannelCtx(const Hccl::LinkData& link, CcuChannelCtxPool::CcuChannelCtx& channelCtx) const
{
    std::lock_guard<std::mutex> lock(mtx_);

    const auto& it = allocatedChannelIdMap_.find(link);
    CHK_PRT_RET(
        it == allocatedChannelIdMap_.end(),
        HCCL_ERROR(
            "[CcuChannelCtxPool][%s] failed to find allocated channelId of link[%s], userDevId[%d].", __func__,
            link.Describe().c_str(), userDevId_),
        HcclResult::HCCL_E_NOT_FOUND);
    // 内部维护数据保证channelJettyInfoMap_记录的资源存在
    channelCtx = channelJettyInfoMap_.at(it->second);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuChannelCtxPool::ReleaseConfirmedChannelRes()
{
    // 析构路径唯一入口，内部持锁保护 map 遍历与设备层归还
    std::lock_guard<std::mutex> lock(mtx_);

    for (const auto& infoEntry : channelJettyInfoMap_) {
        const auto& channelIdKey = infoEntry.first;
        const auto dieId = channelIdKey.first;
        const auto channelId = channelIdKey.second;
        CHK_RET(CcuReleaseChannel(userDevId_, dieId, channelId));
    }
    channelJettyInfoMap_.clear();
    channelToBatch_.clear();
    isReleased_ = true;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuChannelCtxPool::GetCcuChannelCtxById(const std::pair<uint8_t, uint32_t>& key, CcuChannelCtx& ctx)
{
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = channelJettyInfoMap_.find(key);
    if (it == channelJettyInfoMap_.end()) {
        HCCL_ERROR("[%s]fail, key[%u, %u] not found", __func__, key.first, key.second);
        return HCCL_E_NOT_FOUND;
    }
    ctx = it->second;
    return HCCL_SUCCESS;
}

HcclResult CcuChannelCtxPool::ReleaseChannel(const Hccl::LinkData& link)
{
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = allocatedChannelIdMap_.find(link);
    if (UNLIKELY(it == allocatedChannelIdMap_.end())) {
        // 未分配或已释放的 link 直接返回成功：msg-only(资源不足)等未实际分配资源的
        // channel 销毁路径属正常场景，静默返回即可
        HCCL_DEBUG("[CcuChannelCtxPool][%s] link not allocated, userDevId[%d], skip release.", __func__, userDevId_);
        return HcclResult::HCCL_SUCCESS;
    }
    const auto channelIdKey = it->second;
    allocatedChannelIdMap_.erase(link);
    channelRemoteRankIdMap_.erase(channelIdKey);
    auto cntIt = usedChannelCntMap_.find(channelIdKey.first);
    if (cntIt != usedChannelCntMap_.end() && cntIt->second > 0) {
        cntIt->second -= 1;
        if (cntIt->second == 0) {
            usedChannelCntMap_.erase(cntIt);
        }
    }

    ResourceBatch* batch = FindBatchByChannelId(channelIdKey);
    if (batch == nullptr) {
        HCCL_ERROR(
            "[CcuChannelCtxPool][%s] failed to find batch of channelId[%u] die[%u], "
            "userDevId[%d].",
            __func__, channelIdKey.second, channelIdKey.first, userDevId_);
        return HcclResult::HCCL_E_INTERNAL;
    }
    // 槽位压回可复用列表：V2 组内其他 channel 仍活跃时，设备层占用保持不变，
    // 后续创建可直接复用该槽位（连接流程会重新配置 channel 表）。
    batch->availableChannelIdKeys.push_back(channelIdKey);
    // 整组无活跃 channel 时，锁内整组归还设备层并销毁 batch
    HcclResult releaseRet = ReleaseBatchIfIdle(batch);
    if (releaseRet != HcclResult::HCCL_SUCCESS) {
        // 设备层归还失败：该槽位不能被后续创建复用（设备层资源并未真正归还），
        // 从可复用列表回退，资源留待通信域销毁时 ReleaseConfirmedChannelRes 兜底重试
        if (!batch->availableChannelIdKeys.empty() && batch->availableChannelIdKeys.back() == channelIdKey) {
            batch->availableChannelIdKeys.pop_back();
        }
    }
    return releaseRet;
}

CcuChannelCtxPool::ResourceBatch* CcuChannelCtxPool::FindBatchByChannelId(const ChannelIdKey& key) const
{
    auto it = channelToBatch_.find(key);
    return (it == channelToBatch_.end()) ? nullptr : it->second;
}

HcclResult CcuChannelCtxPool::ReleaseBatchIfIdle(CcuChannelCtxPool::ResourceBatch* batch)
{
    if (batch->availableChannelIdKeys.size() != batch->channelIdKeys.size()) {
        // 组内仍有活跃 channel，保留 batch 供槽位复用
        return HcclResult::HCCL_SUCCESS;
    }
    // 整组无活跃 channel：逐 channel 归还设备层（V2 useCnt 递减至 0），全部成功才销毁 batch；
    // 任一失败则保留 batch（channelJettyInfoMap_/channelToBatch_ 条目仍在，host 对象不被销毁），
    // 返回错误供上层感知；通信域销毁时的 ReleaseConfirmedChannelRes 会再次尝试整体归还。
    for (const auto& channelIdKey : batch->channelIdKeys) {
        auto ret = CcuReleaseChannel(userDevId_, channelIdKey.first, channelIdKey.second);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CcuChannelCtxPool][%s] failed to release channel[die%u, id%u] to device, "
                "ret[%d], userDevId[%d], keep batch for retry.",
                __func__, channelIdKey.first, channelIdKey.second, ret, userDevId_);
            return ret;
        }
    }
    RemoveBatch(batch);
    return HcclResult::HCCL_SUCCESS;
}

void CcuChannelCtxPool::RemoveBatch(CcuChannelCtxPool::ResourceBatch* batch)
{
    auto it = batchMap_.find(batch->key);
    if (it == batchMap_.end()) {
        return;
    }
    auto& batches = it->second;
    for (auto bIt = batches.begin(); bIt != batches.end(); ++bIt) {
        if (bIt->get() != batch) {
            continue;
        }
        for (const auto& channelIdKey : batch->channelIdKeys) {
            channelJettyInfoMap_.erase(channelIdKey);
            channelToBatch_.erase(channelIdKey);
        }
        // 锁内销毁 batch：~ResourceBatch → ~CcuJetty → Clean → RaCtxQpDestroy
        batches.erase(bIt);
        break;
    }
    if (batches.empty()) {
        batchMap_.erase(it);
    }
}
} // namespace hcomm
