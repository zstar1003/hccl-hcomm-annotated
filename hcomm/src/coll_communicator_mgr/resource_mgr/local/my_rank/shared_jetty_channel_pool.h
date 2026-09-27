/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_JETTY_CHANNEL_POOL_H
#define SHARED_JETTY_CHANNEL_POOL_H

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"
#include "hcomm_res.h"
#include "hccl/hccl_types.h"

namespace hccl {

class MyRank;

using EndpointDescPair = std::pair<EndpointDesc, EndpointDesc>;

constexpr std::size_t ENDPOINT_DESC_NUM_PER_PAIR = 2;

// 字段级 hash，规避 EndpointDesc padding 字节未初始化导致的误判。
// 将参与区分的有效字段序列化到 std::string，复用标准库 std::hash<string> 完成组合，
// 避免手写魔数/位运算 combine（如 0x9e3779b9）带来的可读性与稳健性问题。
struct EndpointDescPairHash {
    static void AppendEndpointDesc(std::string& s, const EndpointDesc& d)
    {
        s.append(reinterpret_cast<const char*>(&d.protocol), sizeof(d.protocol));
        s.append(reinterpret_cast<const char*>(&d.commAddr.type), sizeof(d.commAddr.type));
        // commAddr.union 内 raws[36] 覆盖全部 union 存储，可安全用于 hash
        s.append(reinterpret_cast<const char*>(d.commAddr.raws), sizeof(d.commAddr.raws));
        s.append(reinterpret_cast<const char*>(&d.loc.locType), sizeof(d.loc.locType));
        s.append(reinterpret_cast<const char*>(d.loc.raws), sizeof(d.loc.raws));
    }
    std::size_t operator()(const EndpointDescPair& p) const noexcept
    {
        std::string buf;
        buf.reserve(sizeof(EndpointDesc) * ENDPOINT_DESC_NUM_PER_PAIR);
        AppendEndpointDesc(buf, p.first);
        AppendEndpointDesc(buf, p.second);
        return std::hash<std::string>{}(buf);
    }
};

// 字段级比较，规避 EndpointDesc padding 字段未初始化导致的误判
struct EndpointDescPairEqual {
    bool operator()(const EndpointDescPair& a, const EndpointDescPair& b) const noexcept
    {
        return a.first.protocol == b.first.protocol && a.first.commAddr.type == b.first.commAddr.type
               && std::memcmp(a.first.commAddr.raws, b.first.commAddr.raws, sizeof(a.first.commAddr.raws)) == 0
               && a.first.loc.locType == b.first.loc.locType
               && std::memcmp(a.first.loc.raws, b.first.loc.raws, sizeof(a.first.loc.raws)) == 0
               && a.second.protocol == b.second.protocol && a.second.commAddr.type == b.second.commAddr.type
               && std::memcmp(a.second.commAddr.raws, b.second.commAddr.raws, sizeof(a.second.commAddr.raws)) == 0
               && a.second.loc.locType == b.second.loc.locType
               && std::memcmp(a.second.loc.raws, b.second.loc.raws, sizeof(a.second.loc.raws)) == 0;
    }
};

/**
 * @note 职责：MyRank 级别的共享 Jetty Channel 池。
 *       按 myRank -> tag -> (localEndpoint, remoteEndpoint) -> [ChannelHandle] 管理复用的 Channel。
 *       重复调用时，若 tag->endpointPair 下 channel 数量不足则创建后再返回；若足够直接按序返回。
 *       MyRank 析构时通过 DestroyAllByMyRank 统一清理，避免悬挂引用。
 *       约束：池内 Channel 生命周期由池统一管理，调用方不可单独对其调用
 *       HcommChannelDestroy/HcclChannelRelease，否则池内会残留悬挂句柄。
 */
class SharedJettyChannelPool {
public:
    struct EpPairChannels {
        std::vector<ChannelHandle> channels;
        uint32_t nextReturnIdx{0};
    };

    static SharedJettyChannelPool& GetInstance();

    /**
     * @brief 获取或创建共享 Jetty Channel
     * @param[in] myRank 归属的 MyRank 指针（用于销毁时索引）
     * @param[in] tag 共享队列 tag
     * @param[in] epPair 源目的 endpointPair
     * @param[in] requestedNum 请求的 channel 数量
     * @param[in] createFunc 创建新 channel 的回调
     * @param[out] outChannels 输出的 channel 句柄数组（前 outReusedCount 个为池中复用，其余为新建）
     * @param[out] outReusedCount 输出从池中复用的 channel 数量（可选，nullptr 时不输出）
     * @return HcclResult 执行结果
     */
    HcclResult AcquireChannels(
        MyRank* myRank, const std::string& tag, const EndpointDescPair& epPair, uint32_t requestedNum,
        const std::function<HcclResult(uint32_t, ChannelHandle*)>& createFunc, ChannelHandle* outChannels,
        uint32_t* outReusedCount = nullptr);

    /**
     * @brief 销毁 MyRank 下所有共享 Jetty Channel（MyRank 析构时调用）
     * @param[in] myRank MyRank 指针
     */
    HcclResult DestroyAllByMyRank(MyRank* myRank);

    /**
     * @brief 检查 MyRank 是否有共享 Jetty Channel 未销毁
     * @param[in] myRank MyRank 指针
     * @return HcclResult HCCL_SUCCESS 表示可以销毁，HCCL_E_UNAVAIL 表示仍有 Channel
     */
    HcclResult CheckMyRankDestroy(MyRank* myRank);

    /**
     * @brief 从池中移除指定 channel 句柄（建链失败时清理已销毁的句柄）
     * @param[in] myRank 归属的 MyRank 指针
     * @param[in] tag 共享队列 tag
     * @param[in] epPair 源目的 endpointPair
     * @param[in] channels 要移除的 channel 句柄数组
     * @param[in] channelNum 数量
     */
    void RemoveChannels(
        MyRank* myRank, const std::string& tag, const EndpointDescPair& epPair, const ChannelHandle* channels,
        uint32_t channelNum);

private:
    SharedJettyChannelPool() = default;
    ~SharedJettyChannelPool() = default;
    SharedJettyChannelPool(const SharedJettyChannelPool&) = delete;
    SharedJettyChannelPool& operator=(const SharedJettyChannelPool&) = delete;

    HcclResult ReturnExistingChannels(
        MyRank* myRank, const std::string& tag, const EndpointDescPair& epPair, uint32_t requestedNum,
        ChannelHandle* outChannels, uint32_t& returnFromExisting, uint32_t& needCreate);

    using EpPairMap = std::unordered_map<EndpointDescPair, EpPairChannels, EndpointDescPairHash, EndpointDescPairEqual>;
    using TagMap = std::unordered_map<std::string, EpPairMap>;
    using RankPoolIter = std::unordered_map<MyRank*, TagMap>::iterator;

    // 调用者须持有 mtx_。收集 myRank 下全部共享 Jetty channel 句柄。
    // 返回值：rankPools_ 中 myRank 对应的迭代器；未找到返回 rankPools_.end()。
    // allChannels 输出收集到的句柄（已 reserve 预分配）。
    RankPoolIter CollectMyRankChannelsLocked(MyRank* myRank, std::vector<ChannelHandle>& allChannels);

    std::mutex mtx_;
    std::unordered_map<MyRank*, TagMap> rankPools_;
};

} // namespace hccl

#endif // SHARED_JETTY_CHANNEL_POOL_H
