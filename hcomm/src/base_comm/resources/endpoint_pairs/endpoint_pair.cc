/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "endpoint_pair.h"
#include "socket_config.h"
#include "hcomm_c_adpt.h"
#include "orion_adpt_utils.h"
#include "channel_process.h"
#include "comm_engine_utils.h"

#include "hcom_common.h"
#include "exception_handler.h"
#include "config_plf_log_v2.h"

namespace hcomm {
using Hccl::PLF_CHANNEL;

EndpointPair::~EndpointPair()
{
    for (auto& channels : channelHandles_) {
        if (channels.second.empty()) {
            continue;
        }
        (void)ChannelProcess::ChannelDestroy(channels.second.data(), channels.second.size());
    }
}

HcclResult EndpointPair::Init()
{
    std::lock_guard<std::mutex> lock(channelMtx_);
    EXCEPTION_CATCH(socketMgr_ = std::make_unique<SocketMgr>(), return HCCL_E_PTR);
    channelHandles_.clear();
    s32 devLogicId;
    CHK_RET(hrtGetDevice(&devLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), devicePhyId_));

    return HCCL_SUCCESS;
}

HcclResult EndpointPair::GetHostSocketWithRank(
    const uint32_t myRank, const uint32_t rmtRank, const std::string& socketTag, const uint32_t listenPort,
    u32 reuseIdx, Hccl::Socket*& socket)
{
    uint32_t connectMode = 0;
    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEndpointDesc_, remoteEndpointDesc_, linkData, reuseIdx));
    std::string linkTag = socketTag;
    if (linkData.GetReuseIdx() != "0") {
        linkTag += ("_" + linkData.GetReuseIdx());
    }

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_910B && localEndpointDesc_.loc.locType != remoteEndpointDesc_.loc.locType) {
        connectMode = 1;
    }

    /* A2: host nic(cpu roce channel) -- device nic(transport ibv)时，两边ip地址格式不一样，判断大小算法不匹配
     * 修改成按照rank id大小判断server和client */
    Hccl::SocketConfig socketConfig = Hccl::SocketConfig(linkData, listenPort, linkTag, connectMode, myRank, rmtRank);
    CHK_RET(socketMgr_->GetHostSocket(socketConfig, socket));
    return HCCL_SUCCESS;
}

HcclResult EndpointPair::EnsureSocketMgrCompat(const uint32_t myRank, const std::string& socketTag)
{
    {
        std::lock_guard<std::mutex> lock(socketMgrMtx_);
        if (socketMgrCompat_) {
            return HCCL_SUCCESS;
        }
    }

    int32_t devLogicId = HcclGetThreadDeviceId();
    uint32_t devPhyId{0};
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(devLogicId), devPhyId));
    std::unique_ptr<Hccl::SocketManager> newMgr = nullptr;
    EXCEPTION_CATCH(
        newMgr = std::make_unique<Hccl::SocketManager>(myRank, devPhyId, devLogicId, socketTag), return HCCL_E_PTR);
    CHK_PTR_NULL(rankIpPortMap_);
    CHK_RET(newMgr->SetDeviceServerListenPortMap(rankIpPortMap_));

    {
        std::lock_guard<std::mutex> lock(socketMgrMtx_);
        if (socketMgrCompat_) {
            return HCCL_SUCCESS;
        }
        socketMgrCompat_ = std::move(newMgr);
    }

    return HCCL_SUCCESS;
}

Hccl::SocketConfig EndpointPair::BuildSocketConfig(const Hccl::LinkData& linkData, const std::string& socketTag) const
{
    std::string linkTag = socketTag;
    if (linkData.GetReuseIdx() != "0") {
        linkTag += ("_" + linkData.GetReuseIdx());
    }
    return Hccl::SocketConfig(linkData.GetRemoteRankId(), linkData, linkTag);
}

HcclResult EndpointPair::HandleHostSocketOrBuildLinkData(
    const uint32_t myRank, const uint32_t rmtRank, const std::string& socketTag, u32 reuseIdx,
    const uint32_t listenPort, Hccl::Socket*& socket, uint32_t devicePhyId, uint32_t remoteDevicePhyId,
    Hccl::LinkData& linkData, bool& isHost)
{
    if (localEndpointDesc_.loc.locType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
        std::string socketTagPrefix = socketTag;
        if (myRank <= rmtRank) {
            socketTagPrefix += "_" + std::to_string(myRank) + "_" + std::to_string(rmtRank);
        } else {
            socketTagPrefix += "_" + std::to_string(rmtRank) + "_" + std::to_string(myRank);
        }
        CHK_RET(this->GetHostSocketWithRank(myRank, rmtRank, socketTagPrefix, listenPort, reuseIdx, socket));
        isHost = true;
        return HCCL_SUCCESS;
    }
    isHost = false;
    CHK_RET(EndpointDescPairToLinkDataWithRankIds(
        myRank, rmtRank, localEndpointDesc_, remoteEndpointDesc_, linkData, devicePhyId, remoteDevicePhyId, reuseIdx));
    return HCCL_SUCCESS;
}

HcclResult EndpointPair::GetSocketInternal(
    const uint32_t myRank, const uint32_t rmtRank, const std::string& socketTag, u32 reuseIdx,
    const uint32_t listenPort, Hccl::Socket*& socket, uint32_t devicePhyId, uint32_t remoteDevicePhyId,
    bool connectMode)
{
    Hccl::LinkData linkData = BuildDefaultLinkData();
    bool isHost = false;
    CHK_RET(HandleHostSocketOrBuildLinkData(
        myRank, rmtRank, socketTag, reuseIdx, listenPort, socket, devicePhyId, remoteDevicePhyId, linkData, isHost));
    if (isHost) {
        return HCCL_SUCCESS;
    }
    EXCEPTION_HANDLE_BEGIN
    Hccl::SocketConfig socketConfig = BuildSocketConfig(linkData, socketTag);
    if (connectMode) {
        CHK_PTR_NULL(socketMgrCompat_);
        socketMgrCompat_->ConnectSockets(socketConfig);
    } else {
        CHK_RET(EnsureSocketMgrCompat(myRank, socketTag));
        socketMgrCompat_->BatchCreateSockets(socketConfig);
    }
    socket = socketMgrCompat_->GetConnectedSocket(socketConfig);
    CHK_PTR_NULL(socket);
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult EndpointPair::ServerInit(
    const uint32_t myRank, const uint32_t rmtRank, const std::string& socketTag, u32 reuseIdx, uint32_t devicePhyId,
    uint32_t remoteDevicePhyId)
{
    if (localEndpointDesc_.loc.locType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
        // host网卡不走device的socket监听
        return HCCL_SUCCESS;
    }
    // server监听
    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkDataWithRankIds(
        myRank, rmtRank, localEndpointDesc_, remoteEndpointDesc_, linkData, devicePhyId, remoteDevicePhyId, reuseIdx));
    EXCEPTION_HANDLE_BEGIN
    CHK_RET(EnsureSocketMgrCompat(myRank, socketTag));
    Hccl::SocketConfig socketConfig = BuildSocketConfig(linkData, socketTag);
    // 调用sock的server监听接口
    socketMgrCompat_->ServerListen(socketConfig);
    EXCEPTION_HANDLE_END

    return HCCL_SUCCESS;
}

HcclResult EndpointPair::GetConnectedSocket(
    const uint32_t myRank, const uint32_t rmtRank, const std::string& socketTag, u32 reuseIdx,
    const uint32_t listenPort, Hccl::Socket*& socket, uint32_t devicePhyId, uint32_t remoteDevicePhyId)
{
    // 该接口内进行建链和获取socket
    return GetSocketInternal(
        myRank, rmtRank, socketTag, reuseIdx, listenPort, socket, devicePhyId, remoteDevicePhyId, true);
}

HcclResult EndpointPair::GetSocket(
    const uint32_t myRank, const uint32_t rmtRank, const std::string& socketTag, u32 reuseIdx,
    const uint32_t listenPort, Hccl::Socket*& socket, uint32_t devicePhyId, uint32_t remoteDevicePhyId)
{
    // 临时方案：支持混跑新增，非Roce场景走orion socketMgr实现server socket复用
    return GetSocketInternal(
        myRank, rmtRank, socketTag, reuseIdx, listenPort, socket, devicePhyId, remoteDevicePhyId, false);
}

HcclResult EndpointPair::CreateChannel(
    EndpointHandle endpointHandle, CommEngine engine, u32 reuseIdx, HcommChannelDesc* channelDescs,
    ChannelHandle* channels)
{
    std::lock_guard<std::mutex> lock(channelMtx_);
    if (channelHandles_.find(engine) == channelHandles_.end() || channelHandles_[engine].size() <= reuseIdx) {
        CHK_RET_UNAVAIL(
            static_cast<HcclResult>(HcommCollectiveChannelCreate(endpointHandle, engine, channelDescs, 1, channels)));
        channelHandles_[engine].push_back(channels[0]);
        // 记录真实槽位下标：UNREUSE 通道的入参 reuseIdx 为 0xFFFFFFFF，实际槽位是 push_back 后的下标
        handleToLoc_[channels[0]] = {engine, static_cast<u32>(channelHandles_[engine].size() - 1)};
        PLF_CONFIG_INFO(
            PLF_CHANNEL, "EndpointPair::CreateChannel: engine[%s] reuseIdx[%u] channelHandle[0x%llx].",
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx,
            static_cast<unsigned long long>(channels[0]));
        return HCCL_SUCCESS;
    }

    channels[0] = channelHandles_[engine][reuseIdx];
    if (channelDescs->memHandleNum > 1) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelUpdateMemInfo(channelDescs->memHandles + 1, channelDescs->memHandleNum - 1, channels[0])));
    }
    PLF_CONFIG_INFO(
        PLF_CHANNEL, "EndpointPair::CreateChannel: engine[%s] reuseIdx[%u] reuse channelHandle[0x%llx].",
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx,
        static_cast<unsigned long long>(channels[0]));
    return HCCL_SUCCESS;
}

// 找到对应的channel handle，调用HcommChannelDestroy销毁平台层对象，并删除channelHandles_中的channelHandle元素
HcclResult EndpointPair::DestroyChannel(CommEngine engine, u32 reuseIdx)
{
    std::lock_guard<std::mutex> lock(channelMtx_);
    if (channelHandles_.find(engine) == channelHandles_.end() || channelHandles_[engine].size() <= reuseIdx) {
        HCCL_WARNING(
            "EndpointPair::DestroyChannel: engine[%s] reuseIdx[%u], channelHandle size[%u],"
            "channel not found, skip destroy channel",
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx, channelHandles_[engine].size());
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "EndpointPair::DestroyChannel: engine[%s] reuseIdx[%u], channelHandle size[%u],"
        "start destroy channel",
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx, channelHandles_[engine].size());
    ChannelHandle channelHandle = channelHandles_[engine][reuseIdx];
    // 无论 HcommChannelDestroy 成功与否，底层 channel 对象已被从全局 map 移除（channel 不可用），
    // host 侧索引必须同步清理，避免后续复用到失效 handle
    HcclResult destroyRet = static_cast<HcclResult>(HcommChannelDestroy(&channelHandle, 1));
    if (destroyRet != HCCL_SUCCESS) {
        HCCL_WARNING(
            "EndpointPair::DestroyChannel: HcommChannelDestroy failed, ret[%d], still clean host index.", destroyRet);
    }
    // 先删反查索引再 erase 向量: erase 会使后续元素下标前移
    handleToLoc_.erase(channelHandle);
    // 去掉channelHandles_中reuseIdx位置的channelHandle
    channelHandles_[engine].erase(channelHandles_[engine].begin() + reuseIdx);
    // 同 engine 后续 handle 因 erase 下标前移, 需同步修正反查索引
    auto& handlesVec = channelHandles_[engine];
    for (u32 idx = reuseIdx; idx < handlesVec.size(); ++idx) {
        auto locIt = handleToLoc_.find(handlesVec[idx]);
        if (locIt != handleToLoc_.end()) {
            locIt->second.second = idx;
        }
    }
    HCCL_INFO(
        "EndpointPair::DestroyChannel: engine[%s] reuseIdx[%u] destroy channel success,"
        "channelHandle size[%u]",
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx, channelHandles_[engine].size());
    PLF_CONFIG_INFO(
        PLF_CHANNEL, "EndpointPair::DestroyChannel: engine[%s] reuseIdx[%u] channelHandle[0x%llx].",
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx,
        static_cast<unsigned long long>(channelHandle));
    return destroyRet;
}

// 检查channel是否存在，channel不存在则返回true
bool EndpointPair::IsChannelNotExist(CommEngine engine, u32 reuseIdx)
{
    std::lock_guard<std::mutex> lock(channelMtx_);
    return channelHandles_.find(engine) == channelHandles_.end() || channelHandles_[engine].size() <= reuseIdx;
}

std::unordered_map<CommEngine, std::vector<ChannelHandle>> EndpointPair::GetChannelHandles() const
{
    std::lock_guard<std::mutex> lock(channelMtx_);
    return channelHandles_;
}

bool EndpointPair::GetChannelHandle(CommEngine engine, u32 reuseIdx, ChannelHandle& handle) const
{
    std::lock_guard<std::mutex> lock(channelMtx_);
    auto it = channelHandles_.find(engine);
    if (it == channelHandles_.end() || reuseIdx >= it->second.size()) {
        return false;
    }
    handle = it->second[reuseIdx];
    return true;
}

bool EndpointPair::FindChannelLoc(ChannelHandle handle, CommEngine& engine, u32& reuseIdx) const
{
    std::lock_guard<std::mutex> lock(channelMtx_);
    auto it = handleToLoc_.find(handle);
    if (it == handleToLoc_.end()) {
        return false;
    }
    engine = it->second.first;
    reuseIdx = it->second.second;
    return true;
}

} // namespace hcomm
