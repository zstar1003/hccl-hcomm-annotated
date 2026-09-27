/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_CHANNELCTX_POOLS_H
#define CCU_CHANNELCTX_POOLS_H

#include <vector>
#include <unordered_map>
#include <mutex>

#include "ccu_jetty_.h"
#include "hash_utils.h"
#include "ip_address.h"
#include "virtual_topo.h"

namespace hcomm {

// 管理着有限的硬件资源：ChannelCtx与jetty
class CcuChannelCtxPool final {
public:
    explicit CcuChannelCtxPool(int32_t userDevId);
    ~CcuChannelCtxPool();

    HcclResult PrepareCreate(const std::vector<Hccl::LinkData>& links, uint32_t sqSize = 0);
    using CcuChannelCtx = std::pair<CcuChannelInfo, std::vector<CcuJetty*>>;
    HcclResult GetChannelCtx(const Hccl::LinkData& link, CcuChannelCtx& channelCtx) const;
    HcclResult GetCcuChannelCtxById(const std::pair<uint8_t, uint32_t>& key, CcuChannelCtx& ctx);
    // Channel 销毁时归还 channel ctx / jetty ctx / wqeBB 资源：
    // - V1：所属 batch 立即变空，逐 channel 归还设备层空闲池并销毁 batch；
    // - V2：槽位压回 batch 的可复用列表供后续创建复用，整组无活跃 channel 时
    //       再逐 channel 归还设备层空闲池（与设备层 useCnt 组粒度语义一致）。
    HcclResult ReleaseChannel(const Hccl::LinkData& link);

private:
    struct ResIdHash {
        std::size_t operator()(const std::pair<uint8_t, uint32_t>& p) const
        {
            return Hccl::HashCombine({p.first, p.second});
        }
    };

    using CcuJettyPtr = CcuJetty*;
    using BatchKey = Hccl::IpAddress; // srcIpAddress;
    using ResIdkey = std::pair<uint8_t, uint32_t>;
    using ChannelIdKey = ResIdkey;
    using JettyIdKey = ResIdkey;

    // 平台层每次调用CcuAllocChannels可能提供多个ccu channel，且不同srcIp的jetty不能复用
    // 故以srcIp为粒度，多次调用接口，每次接口结果定义为一个批次资源
    struct ResourceBatch { // 记录该批次申请到的所有channel资源信息
        BatchKey key;
        std::vector<ChannelIdKey> channelIdKeys;
        std::vector<ChannelIdKey> availableChannelIdKeys;
        std::unordered_map<JettyIdKey, std::unique_ptr<CcuJetty>, ResIdHash> jettys;

        ResourceBatch(const BatchKey& batchKey) : key(batchKey) {};
        HcclResult Init(const std::vector<CcuChannelInfo>& channelInfos);
    };

private:
    HcclResult GetAvailableBatch(const BatchKey& batchKey, ResourceBatch*& batchPtr, uint32_t sqSize);
    bool FindAvailableBatch(const BatchKey& batchKey, ResourceBatch*& batchPtr) const;
    HcclResult CreateAndSaveNewBatch(
        const BatchKey& batchKey, const std::vector<CcuChannelInfo> channelInfos, ResourceBatch*& batchPtr);
    HcclResult ReleaseConfirmedChannelRes();
    ResourceBatch* FindBatchByChannelId(const ChannelIdKey& key) const;
    HcclResult ReleaseBatchIfIdle(ResourceBatch* batch);
    // 从 batchMap_ 移除并销毁该 batch（锁内调用）：设备层 CcuReleaseChannel 与
    // ~CcuJetty（RaCtxQpDestroy）在 pool 锁内执行，牺牲并发流畅，换取"释放先于
    // 同一 pool 的后续申请"，资源紧俏场景下保证释放的资源可被立即复用。
    void RemoveBatch(ResourceBatch* batch);

private:
    int32_t userDevId_{0};
    bool isReleased_{true};

    // 保护以下所有 map/batch，PrepareCreate/GetChannelCtx/ReleaseChannel 并发安全
    mutable std::mutex mtx_;
    // 各资源申请记录，当前按SrcIpAddr粒度申请和管理
    std::unordered_map<BatchKey, std::vector<std::unique_ptr<ResourceBatch>>> batchMap_;
    // 各link已分配的channel资源Id信息
    std::unordered_map<Hccl::LinkData, ChannelIdKey> allocatedChannelIdMap_;
    // 全部已申请的channel资源信息，资源申请成功后将要记录到该map中
    std::unordered_map<ChannelIdKey, CcuChannelCtx, ResIdHash> channelJettyInfoMap_;
    // 以die粒度记录已分配channel资源, index: dieId
    std::unordered_map<uint8_t, uint32_t> usedChannelCntMap_;
    // 记录channel与对端rank的映射关系, index: (die, channelId)
    std::unordered_map<ChannelIdKey, Hccl::RankId, ResIdHash> channelRemoteRankIdMap_;
    // channel -> 所属 batch 反向索引，ReleaseChannel 定位 batch 用
    std::unordered_map<ChannelIdKey, ResourceBatch*, ResIdHash> channelToBatch_{};
};

} // namespace hcomm

#endif // CCU_CHANNELCTX_POOLS_H
