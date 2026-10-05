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
    // [中文导读] 读取当前逻辑设备并转换为物理设备编号，供后续 Socket 链路描述使用。
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
    // [中文导读] 复用下标非零时加入 Socket 标签，区分同一端点对上并行申请的不同连接槽位。
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
    // [中文导读] 先在锁外准备兼容 SocketManager 并注入 Rank 监听端口表，再持锁确认是否已有并发创建结果。
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
        // [中文导读] Host 连接标签按两端 Rank 的升序组合，让双方使用相同键匹配 Socket。
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
    // [中文导读] Host 分支已取得 Socket，可直接返回；设备分支继续使用兼容 SocketManager 完成建链。
    if (isHost) {
        return HCCL_SUCCESS;
    }
    EXCEPTION_HANDLE_BEGIN
    Hccl::SocketConfig socketConfig = BuildSocketConfig(linkData, socketTag);
    // [中文导读] connectMode 选择连接已准备的 Socket 还是批量创建；取得结果后必须确认连接对象非空。
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

// [中文导读] EndpointPair按Engine和reuseIdx缓存通道。没有可用槽位才调用HcommCollectiveChannelCreate。
// [中文导读] 复用分支直接取原句柄，并按条件更新第一个句柄之后的附加内存；不等于每个算子都重建连接。
// [中文导读] channelMtx_保护缓存表，UNREUSE使用新槽位；连接就绪与跨Rank一致性仍需跟踪MyRank后续步骤。
// [中文导读] [AllReduce逐行 S208] EndpointPair::CreateChannel的接口声明：按引擎和 reuseIdx 创建或复用 Channel；复用时更新第0项之后的附加内存；这些参数属于本函数调用边界。
HcclResult EndpointPair::CreateChannel(
    // [中文导读] [AllReduce逐行 S209] EndpointPair::CreateChannel的接口声明：本地端点句柄、请求的通信引擎、本批通道或Socket复用槽位、域级通道描述数组；这些参数属于本函数调用边界。
    EndpointHandle endpointHandle, CommEngine engine, u32 reuseIdx, HcommChannelDesc* channelDescs,
    // [中文导读] [AllReduce逐行 S210] EndpointPair::CreateChannel的接口声明：通道句柄出参数组；这些参数属于本函数调用边界。
    ChannelHandle* channels)
// [中文导读] [AllReduce逐行 S211] 进入EndpointPair::CreateChannel函数体：按引擎和 reuseIdx 创建或复用 Channel；复用时更新第0项之后的附加内存。
{
    // [中文导读] [AllReduce逐行 S212] 调用lock；保持声明的局部对象用于后续处理。
    std::lock_guard<std::mutex> lock(channelMtx_);
    // [中文导读] 引擎尚无缓存或请求下标超出当前向量时创建新 Channel；输出句柄随后加入可复用槽位表。
    // [中文导读] [AllReduce逐行 S214] 仅当`(channelHandles_.find(engine) == channelHandles_.end() || channelHandles_[engine].size() <= reuseIdx)`（按引擎分组的通道槽位缓存的find字段、请求的通信引擎、按引擎分组的通道槽位缓存的end字段、按引擎分组的通道槽位缓存、本批通道或Socket复用槽位）成立时进入此分支；读取容器登记项数。
    if (channelHandles_.find(engine) == channelHandles_.end() || channelHandles_[engine].size() <= reuseIdx) {
        // [中文导读] [AllReduce逐行 S215] 执行本行包裹的资源/任务调用；返回非成功时由检查宏立即向上传递，UNAVAIL资源不足状态保持可识别。
        CHK_RET_UNAVAIL(
            // [中文导读] [AllReduce逐行 S216] 为集合通信内部入口规范描述后创建Channel对象补入`static_cast<HcclResult>(HcommCollectiveChannelCreate(endpointHandle, engine, channelDescs, 1, channels)))`（本地端点句柄、请求的通信引擎、域级通道描述数组、通道句柄出参数组）；本行是参数/结构化初始化续行。
            static_cast<HcclResult>(HcommCollectiveChannelCreate(endpointHandle, engine, channelDescs, 1, channels)));
        // [中文导读] [AllReduce逐行 S217] 将当前条目追加到对应数组/列表；传入/处理按引擎分组的通道槽位缓存、请求的通信引擎、通道句柄出参数组。
        channelHandles_[engine].push_back(channels[0]);
        // 记录真实槽位下标：UNREUSE 通道的入参 reuseIdx 为 0xFFFFFFFF，实际槽位是 push_back 后的下标
        // [中文导读] [AllReduce逐行 S219] 设置句柄到引擎/真实槽位反查表、通道句柄出参数组为/按`{engine, static_cast<u32>(channelHandles_[engine].size() - 1)}`（请求的通信引擎、按引擎分组的通道槽位缓存）；读取容器登记项数。
        handleToLoc_[channels[0]] = {engine, static_cast<u32>(channelHandles_[engine].size() - 1)};
        // [中文导读] [AllReduce逐行 S220] 记录EndpointPair::CreateChannel的状态/性能诊断；日志本身不执行传输。
        PLF_CONFIG_INFO(
            // [中文导读] [AllReduce逐行 S221] 为前述多行表达式补入`PLF_CHANNEL, "EndpointPair::CreateChannel: engine[%s] reuseIdx[%u] channelHandle[0x%llx].",`；本行是参数/结构化初始化续行。
            PLF_CHANNEL, "EndpointPair::CreateChannel: engine[%s] reuseIdx[%u] channelHandle[0x%llx].",
            // [中文导读] [AllReduce逐行 S222] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx,`（请求的通信引擎、本批通道或Socket复用槽位）；本行是参数/结构化初始化续行。
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx,
            // [中文导读] [AllReduce逐行 S223] 为把枚举转换成诊断名称补入`static_cast<unsigned long long>(channels[0]))`（通道句柄出参数组）；本行是参数/结构化初始化续行。
            static_cast<unsigned long long>(channels[0]));
        // [中文导读] [AllReduce逐行 S224] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S225] 结束`if (channelHandles_.find(engine) == channelHandles_.end() || channelHandles_[engine].size() <= reuseIdx)`（按引擎分组的通道槽位缓存的find字段、请求的通信引擎、按引擎分组的通道槽位缓存的end字段、按引擎分组的通道槽位缓存、本批通道或Socket复用槽位）分支/循环；控制流返回外层。
    }

    // [中文导读] 缓存命中时返回槽位原句柄；额外内存从第 1 项开始更新，第 0 项按既有通道约定保留。
    // [中文导读] [AllReduce逐行 S228] 设置通道句柄出参数组为/按`channelHandles_[engine][reuseIdx]`（按引擎分组的通道槽位缓存、请求的通信引擎、本批通道或Socket复用槽位）。
    channels[0] = channelHandles_[engine][reuseIdx];
    // [中文导读] [AllReduce逐行 S229] 仅当`(channelDescs->memHandleNum > 1)`（域级通道描述数组的memHandleNum字段）成立时进入此分支。
    if (channelDescs->memHandleNum > 1) {
        // [中文导读] [AllReduce逐行 S230] 执行本行包裹的资源/任务调用；返回非成功时由检查宏立即向上传递。
        CHK_RET(static_cast<HcclResult>(
            // [中文导读] [AllReduce逐行 S231] 为复用Channel时更新第0项之后的附加注册内存补入`HcommChannelUpdateMemInfo(channelDescs->memHandles + 1, channelDescs->memHandleNum - 1, channels[0])))`（域级通道描述数组的memHandles字段、域级通道描述数组的memHandleNum字段、通道句柄出参数组）；本行是参数/结构化初始化续行。
            HcommChannelUpdateMemInfo(channelDescs->memHandles + 1, channelDescs->memHandleNum - 1, channels[0])));
    // [中文导读] [AllReduce逐行 S232] 结束`if (channelDescs->memHandleNum > 1)`（域级通道描述数组的memHandleNum字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S233] 记录EndpointPair::CreateChannel的状态/性能诊断；日志本身不执行传输。
    PLF_CONFIG_INFO(
        // [中文导读] [AllReduce逐行 S234] 为前述多行表达式补入`PLF_CHANNEL, "EndpointPair::CreateChannel: engine[%s] reuseIdx[%u] reuse channelHandle[0x%llx].",`；本行是参数/结构化初始化续行。
        PLF_CHANNEL, "EndpointPair::CreateChannel: engine[%s] reuseIdx[%u] reuse channelHandle[0x%llx].",
        // [中文导读] [AllReduce逐行 S235] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx,`（请求的通信引擎、本批通道或Socket复用槽位）；本行是参数/结构化初始化续行。
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx,
        // [中文导读] [AllReduce逐行 S236] 为把枚举转换成诊断名称补入`static_cast<unsigned long long>(channels[0]))`（通道句柄出参数组）；本行是参数/结构化初始化续行。
        static_cast<unsigned long long>(channels[0]));
    // [中文导读] [AllReduce逐行 S237] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S238] 结束EndpointPair::CreateChannel函数体；控制流返回外层。
}

// 找到对应的channel handle，调用HcommChannelDestroy销毁平台层对象，并删除channelHandles_中的channelHandle元素
HcclResult EndpointPair::DestroyChannel(CommEngine engine, u32 reuseIdx)
{
    std::lock_guard<std::mutex> lock(channelMtx_);
    // [中文导读] 找不到待销毁槽位时按成功跳过，避免重复清理已经移除的缓存条目。
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
    // [中文导读] 持锁验证引擎和下标，只把存在的槽位句柄写入出参，查询失败时由调用方处理未命中。
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
