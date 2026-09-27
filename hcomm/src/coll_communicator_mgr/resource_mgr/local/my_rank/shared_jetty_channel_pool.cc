/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "shared_jetty_channel_pool.h"
#include "my_rank.h"
#include "log.h"
#include <algorithm>
#include <functional>

namespace hccl {

SharedJettyChannelPool& SharedJettyChannelPool::GetInstance()
{
    static SharedJettyChannelPool instance;
    return instance;
}

HcclResult SharedJettyChannelPool::ReturnExistingChannels(
    MyRank* myRank, const std::string& tag, const EndpointDescPair& epPair, uint32_t requestedNum,
    ChannelHandle* outChannels, uint32_t& returnFromExisting, uint32_t& needCreate)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto& tagMap = rankPools_[myRank];
    auto& epPairMap = tagMap[tag];
    auto& epChannels = epPairMap[epPair];

    uint32_t available = static_cast<uint32_t>(epChannels.channels.size());
    if (available >= requestedNum) {
        returnFromExisting = requestedNum;
    } else {
        returnFromExisting = available;
        needCreate = requestedNum - available;
    }

    HCCL_INFO(
        "[%s] myRank[%p], tag[%s], available[%u], requested[%u], returnFromExisting[%u], needCreate[%u].", __func__,
        myRank, tag.c_str(), available, requestedNum, returnFromExisting, needCreate);

    for (uint32_t i = 0; i < returnFromExisting; ++i) {
        uint32_t idx = epChannels.nextReturnIdx % epChannels.channels.size();
        outChannels[i] = epChannels.channels[idx];
        epChannels.nextReturnIdx = (epChannels.nextReturnIdx + 1) % epChannels.channels.size();
        HCCL_INFO("[%s] return existing channel[%u]: handle[0x%llx].", __func__, i, outChannels[i]);
    }
    return HCCL_SUCCESS;
}

HcclResult SharedJettyChannelPool::AcquireChannels(
    MyRank* myRank, const std::string& tag, const EndpointDescPair& epPair, uint32_t requestedNum,
    const std::function<HcclResult(uint32_t, ChannelHandle*)>& createFunc, ChannelHandle* outChannels,
    uint32_t* outReusedCount)
{
    if (myRank == nullptr || requestedNum == 0 || outChannels == nullptr) {
        HCCL_ERROR(
            "[%s] invalid params, myRank[%p], requestedNum[%u], outChannels[%p].", __func__, myRank, requestedNum,
            outChannels);
        return HCCL_E_PARA;
    }

    uint32_t returnFromExisting = 0;
    uint32_t needCreate = 0;

    // 第一段（持锁）：查询已有 channel 并计算需新建数量，取走复用句柄后释放锁。
    CHK_RET(ReturnExistingChannels(myRank, tag, epPair, requestedNum, outChannels, returnFromExisting, needCreate));

    // 第二段（无锁）：执行建链 I/O，避免阻塞其他 myRank/tag 的并发 Acquire。
    if (needCreate > 0) {
        ChannelHandle* newChannels = outChannels + returnFromExisting;
        HcclResult ret = createFunc(needCreate, newChannels);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] createFunc failed, needCreate[%u], ret[%d].", __func__, needCreate, ret);
            // 不回退 nextReturnIdx：第一段与第三段之间可能有并发 Acquire 推进了游标，
            // 回退会错误覆盖其他线程的推进量。channel 仍在池中，后续仍可通过取模访问到，
            // 本轮调用方收到错误，仅失去部分可复用句柄的租约，不影响正确性。
            return ret;
        }

        // 第三段（持锁）：将新建 channel 回填到池，重新定位条目以规避 rehash 导致的引用失效。
        std::lock_guard<std::mutex> lock(mtx_);
        auto& tagMap = rankPools_[myRank];
        auto& epPairMap = tagMap[tag];
        auto& epChannels = epPairMap[epPair];
        for (uint32_t i = 0; i < needCreate; ++i) {
            epChannels.channels.push_back(newChannels[i]);
            HCCL_INFO(
                "[%s] created new channel[%u]: handle[0x%llx], total channels[%zu].", __func__, i, newChannels[i],
                epChannels.channels.size());
        }
    }

    if (outReusedCount != nullptr) {
        *outReusedCount = returnFromExisting;
    }
    return HCCL_SUCCESS;
}

SharedJettyChannelPool::RankPoolIter
SharedJettyChannelPool::CollectMyRankChannelsLocked(MyRank* myRank, std::vector<ChannelHandle>& allChannels)
{
    auto it = rankPools_.find(myRank);
    if (it == rankPools_.end()) {
        return it;
    }
    // 先统计总数并 reserve，避免 push_back 触发多次 realloc
    uint32_t totalChannels = 0;
    for (auto& tagEntry : it->second) {
        for (auto& epEntry : tagEntry.second) {
            totalChannels += static_cast<uint32_t>(epEntry.second.channels.size());
        }
    }
    allChannels.reserve(totalChannels);
    for (auto& tagEntry : it->second) {
        for (auto& epEntry : tagEntry.second) {
            for (ChannelHandle ch : epEntry.second.channels) {
                allChannels.push_back(ch);
            }
        }
    }
    return it;
}

HcclResult SharedJettyChannelPool::DestroyAllByMyRank(MyRank* myRank)
{
    if (myRank == nullptr) {
        return HCCL_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<ChannelHandle> allChannels;
    auto it = CollectMyRankChannelsLocked(myRank, allChannels);
    if (it == rankPools_.end()) {
        return HCCL_SUCCESS;
    }

    if (!allChannels.empty()) {
        HcclResult ret = static_cast<HcclResult>(HcommChannelDestroy(allChannels.data(), allChannels.size()));
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] HcommChannelDestroy failed, channelNum[%zu], ret[%d].", __func__, allChannels.size(), ret);
        }
    }

    rankPools_.erase(it);
    HCCL_INFO("[%s] destroyed myRank[%p] shared jetty channels, total[%zu].", __func__, myRank, allChannels.size());
    return HCCL_SUCCESS;
}

HcclResult SharedJettyChannelPool::CheckMyRankDestroy(MyRank* myRank)
{
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<ChannelHandle> allChannels;
    auto it = CollectMyRankChannelsLocked(myRank, allChannels);
    if (it == rankPools_.end()) {
        return HCCL_SUCCESS;
    }
    if (!allChannels.empty()) {
        HCCL_ERROR(
            "[%s] cannot destroy myRank[%p], still has [%zu] shared jetty channels.", __func__, myRank,
            allChannels.size());
        return HCCL_E_UNAVAIL;
    }
    return HCCL_SUCCESS;
}

void SharedJettyChannelPool::RemoveChannels(
    MyRank* myRank, const std::string& tag, const EndpointDescPair& epPair, const ChannelHandle* channels,
    uint32_t channelNum)
{
    if (myRank == nullptr || channels == nullptr || channelNum == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    auto tagIt = rankPools_.find(myRank);
    if (tagIt == rankPools_.end()) {
        return;
    }
    auto epIt = tagIt->second.find(tag);
    if (epIt == tagIt->second.end()) {
        return;
    }
    auto pairIt = epIt->second.find(epPair);
    if (pairIt == epIt->second.end()) {
        return;
    }
    auto& epChannels = pairIt->second;
    for (uint32_t i = 0; i < channelNum; ++i) {
        auto& vec = epChannels.channels;
        vec.erase(std::remove(vec.begin(), vec.end(), channels[i]), vec.end());
    }
    // 重置游标避免取模越界（channels 已收缩）
    if (epChannels.channels.empty()) {
        epChannels.nextReturnIdx = 0;
    } else {
        epChannels.nextReturnIdx %= epChannels.channels.size();
    }
    HCCL_INFO(
        "[%s] removed [%u] channels from pool, remaining[%zu].", __func__, channelNum, epChannels.channels.size());
}

} // namespace hccl
