/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHANNEL_PROCESS_H
#define CHANNEL_PROCESS_H

#include "hcomm_c_adpt.h"
#include "channel.h"
#include "mem_host_pub.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace hccl {
class DeviceMem;
}

namespace hcomm {

// HcommChannelGetStatus 出参状态码
enum HcommChannelLinkStatus : int32_t {
    HCOMM_CHANNEL_STATUS_READY = 0,
    HCOMM_CHANNEL_STATUS_CONNECTING = 1,
    HCOMM_CHANNEL_STATUS_FAILED = 2,
    HCOMM_CHANNEL_STATUS_TIMEOUT = 3,
    HCOMM_CHANNEL_STATUS_RES_LOC_UNAVAIL = 4, // 本端资源不足
    HCOMM_CHANNEL_STATUS_RES_RMT_UNAVAIL = 5, // 对端资源不足
};

struct DeviceChannelKey {
    int32_t deviceId;
    ChannelHandle handle;

    bool operator==(const DeviceChannelKey& other) const
    {
        return deviceId == other.deviceId && handle == other.handle;
    }
};

struct DeviceChannelKeyHash {
    std::size_t operator()(const DeviceChannelKey& key) const
    {
        return std::hash<int32_t>()(key.deviceId) ^ (std::hash<ChannelHandle>()(key.handle) << 1);
    }
};

class ChannelProcess {
public:
    ChannelProcess() = default;
    ~ChannelProcess() = default;
    static HcclResult CreateChannelsLoop(
        EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
        ChannelHandle* outHandles, bool isSharedQueue = false);
    static HcclResult ResolveUserHandleToHost(ChannelHandle userHandle, ChannelHandle& hostHandle);
    static HcclResult ResolveHostHandleToDevice(ChannelHandle hostHandle, ChannelHandle& deviceHandle);
    static HcclResult
    ChannelUpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum, ChannelHandle channelHandle);
    static HcclResult GetChannelsInfo(
        const ChannelHandle* channelList, uint32_t listNum, std::vector<CommEngine>& engines,
        std::vector<HcommChannelDesc>& channelDescs, std::vector<ChannelStatus>& statusList);
    static HcclResult HandleStatusByEngine(
        const ChannelHandle* channelList, uint32_t listNum, const std::vector<CommEngine>& engines,
        const std::vector<HcommChannelDesc>& channelDescFinals, const std::vector<ChannelStatus>& internalStatus,
        int32_t* statusList);
    static HcclResult PrepareUserChannels(
        ChannelHandle* targetChannels, ChannelHandle* userChannels, HcommChannelDesc* channelDescs, uint32_t channelNum,
        CommEngine engine);
    static HcclResult ChannelGetStatus(const ChannelHandle* channelList, uint32_t listNum, int32_t* statusList);
    static HcclResult ChannelKernelLaunchForComm(
        ChannelHandle* channelHandles, ChannelHandle* hostChannelHandles, HcommChannelDesc* hcommDesc, uint32_t listNum,
        const std::string& commTag, aclrtBinHandle binHandle);
    static HcclResult ChannelGetNotifyNum(ChannelHandle channelHandle, uint32_t* notifyNum);
    static HcclResult
    ChannelGetRemoteMems(ChannelHandle channelHandle, uint32_t* memNum, CommMem** remoteMem, char*** memInfos);
    static HcclResult CcuChannelGetRmtMemToken(ChannelHandle channelHandle, uint64_t srcVa, uint64_t& tokenInfo);
    static HcclResult ChannelKernelDestroy(ChannelHandle* channelHandles, uint32_t listNum, aclrtBinHandle binHandle);
    static HcclResult
    ChannelDestroy(const ChannelHandle* channels, uint32_t channelNum, aclrtBinHandle binHandle = nullptr);
    static HcclResult ChannelGet(const ChannelHandle channelHandle, void** channel);
    static HcclResult ChannelClean(const ChannelHandle* channelList, uint32_t channelNum);
    static HcclResult ChannelResume(const ChannelHandle* channelList, uint32_t channelNum);
    static HcclResult ChannelUpdateKernelLaunch(
        ChannelHandle* deviceChannelHandles, ChannelHandle* hostChannelHandles, uint32_t listNum,
        const std::string& commTag, aclrtBinHandle binHandle);
    static HcclResult
    RegisterChannelD2HMap(ChannelHandle* deviceChannelHandles, ChannelHandle* hostChannelHandles, uint32_t listNum);
    static HcclResult FillChannelD2HMap(
        const ChannelHandle* deviceChannelHandles, const ChannelHandle* hostChannelHandles, uint32_t listNum);
    static HcclResult LaunchChannelKernel(
        ChannelHandle* channelHandles, ChannelHandle* hostChannelHandles, HcommChannelDesc* hcommDesc, uint32_t listNum,
        aclrtBinHandle binHandle);
    static HcclResult InsertPluginChannelToMap(ChannelHandle handle, std::shared_ptr<Channel> channelPtr);
    static HcclResult RemovePluginChannelFromMap(ChannelHandle handle);

private:
    template <typename Func>
    static HcclResult WithChannelByHandleLocked(ChannelHandle inHandle, Func&& func);

    static HcclResult
    CombineHostMemory(const std::vector<std::vector<char>>& hostPackBuffers, hccl::HostMem& hostPackBuf);
    static HcclResult CopyUpdateKernelPackResToDevice(
        const std::vector<std::vector<char>>& hostPackBuffers, const std::vector<u32>& channelSizeVec,
        uint32_t totalListNum, hccl::DeviceMem& channelSizeAddr, hccl::DeviceMem& devicePackBuf);
    static HcclResult LaunchChannelKernelCommon(
        ChannelHandle* channelHandles, ChannelHandle* hostChannelHandles, const HcommChannelDesc* hcommDesc,
        uint32_t listNum, const std::string& commTag, aclrtBinHandle binHandle, const std::string& kernelName,
        bool needProfiling);
    static HcclResult ChannelKernelLaunchForBase(
        ChannelHandle* channelHandles, ChannelHandle* hostChannelHandles, HcommChannelDesc* hcommDesc, uint32_t listNum,
        aclrtBinHandle binHandle);
    static HcclResult LaunchCommonChannelKernel(
        ChannelHandle* channelHandles, ChannelHandle* hostChannelHandles, uint32_t listNum,
        HcommChannelKind channelKind, aclrtBinHandle binHandle);
    static HcclResult ChannelResumeConcurrency(const ChannelHandle* channelList, uint32_t channelNum);
    static HcclResult
    RemoveSingleChannel(int32_t deviceId, ChannelHandle inHandle, std::vector<ChannelHandle>& deviceHandles);

    static std::unordered_map<ChannelHandle, std::shared_ptr<Channel>> g_ChannelMap;
    static std::unordered_map<DeviceChannelKey, ChannelHandle, DeviceChannelKeyHash> g_ChannelD2HMap;
    static std::unordered_map<DeviceChannelKey, ChannelHandle, DeviceChannelKeyHash> g_ChannelH2DMap;
    static std::mutex g_ChannelMapMtx;
};
} // namespace hcomm
#endif // CHANNEL_PROCESS_H
