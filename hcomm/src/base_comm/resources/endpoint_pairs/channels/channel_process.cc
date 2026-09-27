/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "channel_process.h"
#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include "exception_handler.h"
#include "channel_param.h"
#include "channel.h"
#include "aicpu_ts_channel_helper.h"
#include "aiv_channel_helper.h"
#include "aicpu_ts_urma_channel.h"
#include "aicpu_ts_uboe_channel.h"
#include "aicpu_ts_ub_rtp_channel.h"
#include "aicpu_ts_roce_channel_v2.h"
#include "acl/acl_rt.h"
#include "launch_aicpu.h"
#include "hcclCommDfx.h"
#include "env_config/env_config_v2.h"
#include "aicpu_ts_p2p_channel.h"
#include "aiv_urma_channel.h"
#include "ccu_urma_channel.h"
#include "mem_device_pub.h"
#include "comm_engine_utils.h"
#include "comm_configer.h"
#include "builtin_channel_ops.h"

namespace hcomm {

std::unordered_map<ChannelHandle, std::shared_ptr<Channel>> ChannelProcess::g_ChannelMap;
std::unordered_map<DeviceChannelKey, ChannelHandle, DeviceChannelKeyHash> ChannelProcess::g_ChannelD2HMap;
std::unordered_map<DeviceChannelKey, ChannelHandle, DeviceChannelKeyHash> ChannelProcess::g_ChannelH2DMap;
std::mutex ChannelProcess::g_ChannelMapMtx;

template <typename Func>
HcclResult ChannelProcess::WithChannelByHandleLocked(ChannelHandle inHandle, Func&& func)
{
    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    std::shared_ptr<Channel> channelPtr = nullptr;
    {
        // 单锁：该锁同时保护 g_ChannelD2HMap 和 g_ChannelMap
        std::lock_guard<std::mutex> lock(g_ChannelMapMtx);

        // 1) D2H 映射
        DeviceChannelKey key{deviceId, inHandle};
        auto itH = g_ChannelD2HMap.find(key);
        if (itH == g_ChannelD2HMap.end()) {
            HCCL_ERROR(
                "[%s] handle not found in g_ChannelD2HMap, deviceId[%d], inHandle[0x%llx].", __func__, deviceId,
                inHandle);
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        const ChannelHandle mappedHandle = itH->second;

        // 2) ChannelMap 查找
        auto itC = g_ChannelMap.find(mappedHandle);
        if (itC == g_ChannelMap.end() || !itC->second) {
            HCCL_ERROR(
                "[%s] channel not found in g_ChannelMap, deviceId[%d], inHandle[0x%llx], mappedHandle[0x%llx].",
                __func__, deviceId, inHandle, mappedHandle);
            return HcclResult::HCCL_E_INTERNAL;
        }

        channelPtr = itC->second;
        if (channelPtr == nullptr) {
            HCCL_ERROR(
                "[%s] null channel pointer, deviceId[%d], inHandle[0x%llx], mappedHandle[0x%llx].", __func__, deviceId,
                inHandle, mappedHandle);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    // 3) 执行用户逻辑
    return std::forward<Func>(func)(*channelPtr);
}

HcclResult ChannelProcess::CreateChannelsLoop(
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
    ChannelHandle* outHandles, bool isSharedQueue)
{
    CHK_PTR_NULL(endpointHandle);

    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    for (uint32_t i = 0; i < channelNum; ++i) {
        std::shared_ptr<Channel> tmpPtr = nullptr;
        CHK_RET_UNAVAIL(Channel::CreateChannel(endpointHandle, engine, channelDescs[i], tmpPtr, isSharedQueue));
        CHK_SMART_PTR_NULL(tmpPtr);

        tmpPtr->SetNicChannelCtx(&g_BuiltinChannelOps, tmpPtr.get());

        ChannelHandle handle = ReinterpretAs<ChannelHandle>(tmpPtr.get());
        outHandles[i] = handle;
        HCCL_INFO("%s deviceId[%d], handle[0x%llx], ptr[%p]", __func__, deviceId, handle, tmpPtr.get());

        // 仅在修改全局表时持锁
        {
            std::lock_guard<std::mutex> lock(g_ChannelMapMtx);

            if (g_ChannelMap.find(handle) != g_ChannelMap.end()) {
                HCCL_ERROR("[%s] channel handle already exists [0x%llx] in ChannelMap", __func__, handle);
                return HCCL_E_INTERNAL;
            }
            DeviceChannelKey key{deviceId, handle};
            if (g_ChannelD2HMap.find(key) != g_ChannelD2HMap.end()) {
                HCCL_ERROR(
                    "[%s] channel handle already exists deviceId[%d], handle[0x%llx] in g_ChannelD2HMap", __func__,
                    deviceId, handle);
                return HCCL_E_INTERNAL;
            }

            g_ChannelMap.emplace(handle, std::move(tmpPtr));
            g_ChannelD2HMap.emplace(key, handle);
            // 同步维护 H2D 反向映射（host 句柄恒等映射，后续 kernel 回写真实 device 句柄时覆盖）
            DeviceChannelKey h2dKey{deviceId, handle};
            g_ChannelH2DMap.emplace(h2dKey, handle);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::InsertPluginChannelToMap(ChannelHandle handle, std::shared_ptr<Channel> channelPtr)
{
    std::lock_guard<std::mutex> lock(g_ChannelMapMtx);
    if (g_ChannelMap.find(handle) != g_ChannelMap.end()) {
        HCCL_ERROR("[NicPlugin][%s] channel handle already exists [0x%llx] in ChannelMap", __func__, handle);
        return HCCL_E_INTERNAL;
    }

    g_ChannelMap.emplace(handle, std::move(channelPtr));
    HCCL_INFO("[NicPlugin][%s] plugin channel inserted, handle[0x%llx].", __func__, handle);

    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::RemovePluginChannelFromMap(ChannelHandle handle)
{
    std::lock_guard<std::mutex> lock(g_ChannelMapMtx);

    auto iter = g_ChannelMap.find(handle);
    if (iter == g_ChannelMap.end()) {
        HCCL_ERROR("[%s] channel not found in g_ChannelMap, handle[0x%llx].", __func__, handle);
        return HCCL_E_NOT_FOUND;
    }
    g_ChannelMap.erase(iter);

    HCCL_INFO("[%s] unregister plugin channel, handle[0x%llx].", __func__, handle);
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::ResolveUserHandleToHost(ChannelHandle userHandle, ChannelHandle& hostHandle)
{
    // 0 句柄不是合法的 device 句柄：直接按未映射处理（由调用方按 host 句柄兜底），
    // 避免 {deviceId, 0} 这类异常映射被误当成有效句柄
    if (userHandle == 0) {
        return HCCL_E_NOT_FOUND;
    }

    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    std::lock_guard<std::mutex> lock(g_ChannelMapMtx);
    DeviceChannelKey key{deviceId, userHandle};
    auto it = g_ChannelD2HMap.find(key);
    if (it == g_ChannelD2HMap.end()) {
        return HCCL_E_NOT_FOUND;
    }
    hostHandle = it->second;
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::ResolveHostHandleToDevice(ChannelHandle hostHandle, ChannelHandle& deviceHandle)
{
    // 0 句柄不是合法的 host 句柄：直接按未映射处理（由调用方按原值兜底）
    if (hostHandle == 0) {
        return HCCL_E_NOT_FOUND;
    }

    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    std::lock_guard<std::mutex> lock(g_ChannelMapMtx);
    DeviceChannelKey key{deviceId, hostHandle};
    auto it = g_ChannelH2DMap.find(key);
    if (it == g_ChannelH2DMap.end()) {
        return HCCL_E_NOT_FOUND;
    }
    deviceHandle = it->second;
    return HCCL_SUCCESS;
}

HcclResult
ChannelProcess::ChannelUpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum, ChannelHandle channelHandle)
{
    EXCEPTION_HANDLE_BEGIN
    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    Channel* channel = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_ChannelMapMtx);
        // 1) D2H 映射
        DeviceChannelKey key{deviceId, channelHandle};
        auto itH = g_ChannelD2HMap.find(key);
        if (itH == g_ChannelD2HMap.end()) {
            HCCL_ERROR(
                "[%s] handle not found in g_ChannelD2HMap, deviceId[%d], channelHandle[0x%llx].", __func__, deviceId,
                channelHandle);
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        const ChannelHandle mappedHandle = itH->second;

        // 2) ChannelMap 查找
        auto itC = g_ChannelMap.find(mappedHandle);
        if (itC == g_ChannelMap.end() || !itC->second) {
            HCCL_ERROR(
                "[%s] channel not found in g_ChannelMap, deviceId[%d], channelHandle[0x%llx], mappedHandle[0x%llx].",
                __func__, deviceId, channelHandle, mappedHandle);
            return HcclResult::HCCL_E_INTERNAL;
        }
        channel = itC->second.get();
    }
    // UpdateMemInfo需要rank间交互，若在锁内执行会导致单进程多线程场景其他rank被锁拦住
    CHK_RET(channel->UpdateMemInfo(memHandles, memHandleNum));
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

// 是否为通道失败终态（含资源不足），供 ChannelGetStatus 轮询跳过与结果归类使用
static inline bool IsChannelFailureStatus(int32_t status)
{
    return status == ChannelStatus::FAILED || status == ChannelStatus::SOCKET_TIMEOUT
           || status == ChannelStatus::RES_LOC_UNAVAIL || status == ChannelStatus::RES_RMT_UNAVAIL;
}

// 批量状态归类：仍有通道未就绪返回 AGAIN；整批就绪返回 SUCCESS；
// 设计约定：资源不足错误仅在整批没有其他失败时才返回，混有网络/超时类失败时优先返回 NETWORK
static HcclResult ClassifyBatchStatus(const int32_t* statusList, uint32_t listNum, u32 readyCount, u32 failCount)
{
    if (readyCount + failCount < listNum) {
        return HCCL_E_AGAIN;
    }
    if (readyCount == listNum) {
        return HCCL_SUCCESS;
    }
    bool hasNetworkFail = false;
    bool hasResUnavailable = false;
    for (uint32_t i = 0; i < listNum; ++i) {
        if (statusList[i] == ChannelStatus::FAILED || statusList[i] == ChannelStatus::SOCKET_TIMEOUT) {
            hasNetworkFail = true;
        }
        if (statusList[i] == ChannelStatus::RES_LOC_UNAVAIL || statusList[i] == ChannelStatus::RES_RMT_UNAVAIL) {
            hasResUnavailable = true;
        }
    }
    if (hasNetworkFail) {
        HCCL_ERROR(
            "[%s] NETWORK, readyCount[%u], failCount[%u], listNum[%u]", __func__, readyCount, failCount, listNum);
        return HCCL_E_NETWORK;
    }
    if (hasResUnavailable) {
        HCCL_WARNING(
            "[%s] RESOURCE UNAVAILABLE, readyCount[%u], failCount[%u], listNum[%u]", __func__, readyCount, failCount,
            listNum);
        return HCCL_E_UNAVAIL;
    }
    HCCL_ERROR("[%s] NETWORK, readyCount[%u], failCount[%u], listNum[%u]", __func__, readyCount, failCount, listNum);
    return HCCL_E_NETWORK;
}

HcclResult ChannelProcess::ChannelGetStatus(const ChannelHandle* channelList, uint32_t listNum, int32_t* statusList)
{
    HcclResult result = HCCL_SUCCESS;
    EXCEPTION_HANDLE_BEGIN

    // 不得随意添加无效日志，可能造成刷屏
    CHK_PTR_NULL(channelList);
    CHK_PTR_NULL(statusList);

    u32 readyCount = 0;
    u32 failCount = 0;

    for (uint32_t i = 0; i < listNum; ++i) {
        const ChannelHandle inHandle = channelList[i];
        int32_t status = 0;
        // 当前通道状态如果已为失败终态(FAILED/SOCKET_TIMEOUT/资源不足)，说明前面已经失败过，
        // 无需再重新获取状态，继续轮询下一个通道，避免日志刷屏
        if (IsChannelFailureStatus(statusList[i])) {
            failCount++;
            continue;
        }
        // 单锁：D2H 映射 + 查 map + 锁内调用 GetStatus()
        HcclResult ret = WithChannelByHandleLocked(inHandle, [&status](Channel& channel) -> HcclResult {
            status = channel.GetStatus(); // 锁内调用，防止 destroy 并发释放
            return HcclResult::HCCL_SUCCESS;
        });
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[%s] Get ChannelHandle failed.", __func__);
            return ret;
        }
        // 某一个channel状态为失败终态时不直接返回，否则后面的channel无法轮询完，状态无法到达终态；
        if (status == ChannelStatus::FAILED) {
            HCCL_ERROR("[%s] FAILED, channel idx[%u], status[%d]", __func__, i, status);
            failCount++;
        }
        if (status == ChannelStatus::SOCKET_TIMEOUT) {
            HCCL_ERROR("[%s] TIMEOUT, channel idx[%u], status[%d]", __func__, i, status);
            failCount++;
        }
        if (status == ChannelStatus::RES_LOC_UNAVAIL) {
            HCCL_WARNING("[%s] LOC RESOURCE UNAVAILABLE, channel idx[%u], status[%d]", __func__, i, status);
            failCount++;
        }
        if (status == ChannelStatus::RES_RMT_UNAVAIL) {
            HCCL_WARNING("[%s] RMT RESOURCE UNAVAILABLE, channel idx[%u], status[%d]", __func__, i, status);
            failCount++;
        }

        readyCount += (status == ChannelStatus::READY) ? 1 : 0;
        statusList[i] = status;
    }
    result = ClassifyBatchStatus(statusList, listNum, readyCount, failCount);
    EXCEPTION_HANDLE_END
    return result;
}

static void PrintChannelStatusDistribution(
    const ChannelHandle* channelList, uint32_t listNum, const std::vector<ChannelStatus>& statusList)
{
    uint32_t ready = 0;
    uint32_t socketOk = 0;
    uint32_t failed = 0;
    for (uint32_t i = 0; i < listNum; i++) {
        if (statusList[i] == ChannelStatus::READY) {
            ready++;
        } else if (statusList[i] == ChannelStatus::SOCKET_OK) {
            socketOk++;
        } else if (statusList[i] == ChannelStatus::FAILED || statusList[i] == ChannelStatus::SOCKET_TIMEOUT) {
            failed++;
        }
    }
    constexpr uint32_t statusLogIntervalSec = 1;
    static thread_local auto lastLog = std::chrono::steady_clock::time_point{};
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - lastLog).count() >= statusLogIntervalSec) {
        HCCL_RUN_INFO(
            "[%s] batch[0x%llx] link_ready[%u] socket_ok[%u] connecting[%u] abnormal[%u] total[%u]", __func__,
            static_cast<unsigned long long>(channelList[0]), ready, socketOk, listNum - ready - socketOk - failed,
            failed, listNum);
        lastLog = now;
    }
}
HcclResult ChannelProcess::GetChannelsInfo(
    const ChannelHandle* channelList, uint32_t listNum, std::vector<CommEngine>& engines,
    std::vector<HcommChannelDesc>& channelDescs, std::vector<ChannelStatus>& statusList)
{
    CHK_PTR_NULL(channelList);
    CHK_PRT_RET((listNum == 0), HCCL_ERROR("[%s] Invalid listNum, listNum[%u]", __func__, listNum), HCCL_E_PARA);

    channelDescs.resize(listNum);
    statusList.resize(listNum);
    engines.resize(listNum);
    for (uint32_t i = 0; i < listNum; ++i) {
        HcclResult ret = WithChannelByHandleLocked(
            channelList[i], [i, &engines, &channelDescs, &statusList](Channel& channel) -> HcclResult {
                engines[i] = channel.GetEngine();
                channelDescs[i] = channel.GetChannelDesc();
                statusList[i] = channel.GetStatus();
                return HcclResult::HCCL_SUCCESS;
            });
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[%s] Get channel[%u] info failed.", __func__, i);
            return ret;
        }
        if (statusList[i] == ChannelStatus::FAILED || statusList[i] == ChannelStatus::SOCKET_TIMEOUT) {
            HCCL_RUN_WARNING("[%s] FAILED, channel idx[%u], status[%d]", __func__, i, statusList[i]);
        }
    }
    PrintChannelStatusDistribution(channelList, listNum, statusList);
    HCCL_DEBUG("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

void ConvertToLinkStatus(const std::vector<ChannelStatus>& internalStatus, std::vector<int32_t>& linkStatusList)
{
    for (size_t i = 0; i < internalStatus.size(); i++) {
        switch (internalStatus[i]) {
            case ChannelStatus::FAILED:
                linkStatusList[i] = HCOMM_CHANNEL_STATUS_FAILED;
                break;
            case ChannelStatus::SOCKET_TIMEOUT:
                linkStatusList[i] = HCOMM_CHANNEL_STATUS_TIMEOUT;
                break;
            case ChannelStatus::READY:
                linkStatusList[i] = HCOMM_CHANNEL_STATUS_READY;
                break;
            case ChannelStatus::RES_LOC_UNAVAIL:
                linkStatusList[i] = HCOMM_CHANNEL_STATUS_RES_LOC_UNAVAIL;
                break;
            case ChannelStatus::RES_RMT_UNAVAIL:
                linkStatusList[i] = HCOMM_CHANNEL_STATUS_RES_RMT_UNAVAIL;
                break;
            default:
                linkStatusList[i] = HCOMM_CHANNEL_STATUS_CONNECTING;
                break;
        }
    }
}

void CopyLinkStatusToOutput(const std::vector<int32_t>& linkStatusList, int32_t* statusList, uint32_t listNum)
{
    for (uint32_t i = 0; i < listNum; i++) {
        statusList[i] = linkStatusList[i];
    }
}

HcclResult ChannelProcess::HandleStatusByEngine(
    const ChannelHandle* channelList, uint32_t listNum, const std::vector<CommEngine>& engines,
    const std::vector<HcommChannelDesc>& channelDescFinals, const std::vector<ChannelStatus>& internalStatus,
    int32_t* statusList)
{
    std::vector<int32_t> linkStatusList(listNum);
    ConvertToLinkStatus(internalStatus, linkStatusList);

    std::map<CommEngine, std::vector<uint32_t>> groups;
    for (uint32_t i = 0; i < listNum; i++) {
        groups[engines[i]].push_back(i);
    }

    for (auto& entry : groups) {
        CommEngine engine = entry.first;
        const std::vector<uint32_t>& indices = entry.second;
        uint32_t subNum = static_cast<uint32_t>(indices.size());
        std::vector<ChannelHandle> subChannels(subNum);
        std::vector<HcommChannelDesc> subDescs(subNum);
        std::vector<int32_t> subLinkStatus(subNum);
        for (uint32_t j = 0; j < subNum; j++) {
            subChannels[j] = channelList[indices[j]];
            subDescs[j] = channelDescFinals[indices[j]];
            subLinkStatus[j] = linkStatusList[indices[j]];
        }

        std::vector<int32_t> subStatus(subNum);
        if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
            CHK_RET(AicpuTsChannelHelper::HandleStatus(
                subChannels.data(), subNum, engine, subDescs.data(), subLinkStatus, subStatus.data()));
        } else if (engine == COMM_ENGINE_AIV) {
            CHK_RET(AivChannelHelper::HandleStatus(
                subChannels.data(), subNum, subDescs.data(), subLinkStatus, subStatus.data()));
        } else {
            CopyLinkStatusToOutput(subLinkStatus, subStatus.data(), subNum);
        }

        for (uint32_t j = 0; j < subNum; j++) {
            statusList[indices[j]] = subStatus[j];
        }
    }

    return HCCL_SUCCESS;
}

HcclResult
ChannelProcess::CombineHostMemory(const std::vector<std::vector<char>>& hostPackBuffers, hccl::HostMem& hostPackBuf)
{
    if (hostPackBuffers.empty()) {
        HCCL_ERROR("[%s] hostPackBuffers is empty, please check.", __func__);
        return HCCL_E_PARA;
    }

    // 将离散数据复制到连续内存中
    u8* dstPtr = static_cast<u8*>(hostPackBuf.ptr()); // 目标内存起始地址
    u64 dstMax = hostPackBuf.size();
    u64 packSize = 0;

    for (const auto& mem : hostPackBuffers) {
        packSize += mem.size();
        CHK_PRT_RET(
            packSize > dstMax,
            HCCL_ERROR("[%s] fail, packSize[%llu] is bigger than dstMax[%llu]", __func__, packSize, dstMax),
            HCCL_E_PARA);

        CHK_SAFETY_FUNC_RET(memcpy_s(dstPtr, mem.size(), mem.data(), mem.size()));
        dstPtr += mem.size(); // 移动目标指针
    }

    HCCL_INFO(
        "[%s] end of merging host memory, hostPackBuf.addr[%p], hostPackBuf.size[%zu]", __func__, hostPackBuf.ptr(),
        hostPackBuf.size());

    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::FillChannelD2HMap(
    const ChannelHandle* deviceChannelHandles, const ChannelHandle* hostChannelHandles, uint32_t listNum)
{
    CHK_PTR_NULL(deviceChannelHandles);
    CHK_PTR_NULL(hostChannelHandles);
    CHK_PRT_RET((listNum == 0), HCCL_ERROR("[%s] Invalid listNum, listNum[%u]", __func__, listNum), HCCL_E_PARA);

    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    std::lock_guard<std::mutex> lock(g_ChannelMapMtx);
    for (uint32_t idx = 0; idx < listNum; idx++) {
        auto deviceChannelHandle = deviceChannelHandles[idx];
        auto hostChannelHandle = hostChannelHandles[idx];
        if (deviceChannelHandle == 0 || hostChannelHandle == 0) {
            continue;
        }
        HCCL_INFO(
            "%s deviceId[%d], deviceChannelHandle[0x%llx], hostChannelHandle[0x%llx]", __func__, deviceId,
            deviceChannelHandle, hostChannelHandle);
        DeviceChannelKey key{deviceId, deviceChannelHandle};
        g_ChannelD2HMap[key] = hostChannelHandle;
        // 同步维护 H2D 反向映射：host 句柄 -> 用户实际使用的句柄（AIV/AICPU_TS 为 device 句柄）
        DeviceChannelKey h2dKey{deviceId, hostChannelHandle};
        g_ChannelH2DMap[h2dKey] = deviceChannelHandle;
    }

    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::RegisterChannelD2HMap(
    ChannelHandle* deviceChannelHandles, ChannelHandle* hostChannelHandles, uint32_t listNum)
{
    return FillChannelD2HMap(deviceChannelHandles, hostChannelHandles, listNum);
}

static HcclResult FillChannelParam(
    HcclChannelUrmaRes& channelParam, const std::string& commTag, hccl::DeviceMem& deviceChannelList,
    hccl::DeviceMem& devicePackBuf, uint32_t listNum, uint32_t totalListNum, hccl::DeviceMem& channelSizeAddr)
{
    // channelParam资源参数填充
    s32 sRet = strncpy_s(channelParam.hcomId, HCOMID_MAX_LENGTH, commTag.c_str(), HCOMID_MAX_LENGTH - 1);
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[%s] str copy fail. return[%d]", __func__, sRet), HCCL_E_INTERNAL);

    channelParam.channelList = static_cast<void*>(deviceChannelList.ptr());
    channelParam.listNum = listNum;
    channelParam.uniqueIdAddr = static_cast<void*>(devicePackBuf.ptr());
    channelParam.uniqueIdSize = totalListNum;
    channelParam.channelSizeAddr = static_cast<void*>(channelSizeAddr.ptr());

    CHK_RET(hrtGetDevice(&channelParam.deviceLogicId));
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    channelParam.deviceType = static_cast<u32>(devType);

    return HCCL_SUCCESS;
}

template <typename T>
static HcclResult
LaunchKernelDeviceParam(const T& channelParam, aclrtBinHandle binHandle, const std::string& kernelName)
{
    hccl::Stream localStream = hccl::Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    constexpr u32 aicpuStreamMode = 1;
    CHK_RET(hrtStreamSetMode(localStream.ptr(), aicpuStreamMode));

    hccl::DeviceMem addr = hccl::DeviceMem::alloc(sizeof(T));
    CHK_PTR_NULL(addr.ptr());

    CHK_RET(hrtMemSyncCopy(
        addr.ptr(), sizeof(T), &channelParam, sizeof(T), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    uint64_t context = ReinterpretAs<uint64_t>(addr.ptr());

    uint32_t envTimeout = 0;
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        envTimeout = Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
    } else {
        envTimeout = hccl::CommConfiger::GetInstance().GetCommConfigExecTimeOut("");
    }
    s32 timeOut = envTimeout + 25 > std::numeric_limits<u16>::max() ? std::numeric_limits<u16>::max() :
                                                                      envTimeout + 25; // 多25s，避免超时

    CHK_RET(hccl::AicpuAclKernelLaunch(
        localStream.ptr(), static_cast<void*>(&context), sizeof(context), binHandle, kernelName, true,
        static_cast<u16>(timeOut)));

    CHK_RET(hcclStreamSynchronize(localStream.ptr(), timeOut));

    HCCL_INFO("[%s] kernel[%s] launch success.", __func__, kernelName.c_str());
    return HCCL_SUCCESS;
}

static HcclResult
LaunchKernel(const HcclChannelUrmaRes& channelParam, aclrtBinHandle binHandle, const std::string& kernelName)
{
    return LaunchKernelDeviceParam(channelParam, binHandle, kernelName);
}

static HcclResult PackAicpuTsChannelH2DRes(ChannelHandle hostChannelHandle, std::vector<char>& hostPackBuffer)
{
    CHK_PRT_RET(hostChannelHandle == 0, HCCL_ERROR("[%s] hostChannelHandle is null.", __func__), HCCL_E_PARA);
    Channel* channel = ReinterpretAs<Channel*>(hostChannelHandle);
    switch (channel->GetChannelKind()) {
        case HcommChannelKind::AICPU_TS_URMA:
            return ReinterpretAs<AicpuTsUrmaChannel*>(hostChannelHandle)->H2DResPack(hostPackBuffer);
        case HcommChannelKind::AICPU_TS_UBOE:
            return ReinterpretAs<AicpuTsUboeChannel*>(hostChannelHandle)->H2DResPack(hostPackBuffer);
        case HcommChannelKind::AICPU_TS_UB_RTP:
            return ReinterpretAs<AicpuTsUbRtpChannel*>(hostChannelHandle)->H2DResPack(hostPackBuffer);
        case HcommChannelKind::AICPU_TS_ROCE_V2:
            return ReinterpretAs<AicpuTsRoceChannelV2*>(hostChannelHandle)->H2DResPack(hostPackBuffer);
        default:
            HCCL_ERROR(
                "[%s] unsupported channel kind[%s].", __func__, HcommChannelKindToString(channel->GetChannelKind()));
            return HCCL_E_NOT_SUPPORT;
    }
}

HcclResult ChannelProcess::CopyUpdateKernelPackResToDevice(
    const std::vector<std::vector<char>>& hostPackBuffers, const std::vector<u32>& channelSizeVec,
    uint32_t totalListNum, hccl::DeviceMem& channelSizeAddr, hccl::DeviceMem& devicePackBuf)
{
    channelSizeAddr = hccl::DeviceMem::alloc(channelSizeVec.size() * sizeof(u32));
    CHK_PTR_NULL(channelSizeAddr.ptr());
    CHK_RET(hrtMemSyncCopy(
        channelSizeAddr.ptr(), channelSizeVec.size() * sizeof(u32), channelSizeVec.data(),
        channelSizeVec.size() * sizeof(u32), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    hccl::HostMem hostPackBuf = hccl::HostMem::alloc(totalListNum);
    CHK_PTR_NULL(hostPackBuf.ptr());
    u8* dstPtr = static_cast<u8*>(hostPackBuf.ptr());
    const u64 dstMax = static_cast<u64>(totalListNum);
    u64 copiedSize = 0;
    for (const auto& mem : hostPackBuffers) {
        const u64 memSize = static_cast<u64>(mem.size());
        CHK_PRT_RET(
            copiedSize > dstMax,
            HCCL_ERROR("[%s] copiedSize[%llu] is bigger than dstMax[%llu]", __func__, copiedSize, dstMax), HCCL_E_PARA);
        const u64 remainingSize = dstMax - copiedSize;
        CHK_PRT_RET(
            memSize > remainingSize,
            HCCL_ERROR("[%s] memSize[%llu] is bigger than remainingSize[%llu]", __func__, memSize, remainingSize),
            HCCL_E_PARA);
        CHK_SAFETY_FUNC_RET(memcpy_s(dstPtr, remainingSize, mem.data(), mem.size()));
        dstPtr += mem.size();
        copiedSize += memSize;
    }
    CHK_PRT_RET(
        copiedSize != dstMax,
        HCCL_ERROR("[%s] copiedSize[%llu] is not equal to dstMax[%llu]", __func__, copiedSize, dstMax), HCCL_E_PARA);

    devicePackBuf = hccl::DeviceMem::alloc(totalListNum);
    CHK_PTR_NULL(devicePackBuf.ptr());
    CHK_RET(hrtMemSyncCopy(
        devicePackBuf.ptr(), totalListNum, hostPackBuf.ptr(), totalListNum,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

static HcclResult CopyUpdateKernelChannelListToDevice(
    ChannelHandle* deviceChannelHandles, uint32_t listNum, hccl::DeviceMem& deviceChannelList)
{
    deviceChannelList = hccl::DeviceMem::alloc(listNum * sizeof(ChannelHandle));
    CHK_PTR_NULL(deviceChannelList.ptr());
    CHK_RET(hrtMemSyncCopy(
        deviceChannelList.ptr(), listNum * sizeof(ChannelHandle), deviceChannelHandles, listNum * sizeof(ChannelHandle),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::LaunchChannelKernelCommon(
    ChannelHandle* channelHandles, ChannelHandle* hostChannelHandles, const HcommChannelDesc* hcommDesc,
    uint32_t listNum, const std::string& commTag, aclrtBinHandle binHandle, const std::string& kernelName,
    bool needProfiling)
{
    CHK_PTR_NULL(channelHandles);
    CHK_PTR_NULL(hostChannelHandles);
    CHK_PRT_RET((listNum == 0), HCCL_ERROR("[%s] Invalid listNum, listNum[%u]", __func__, listNum), HCCL_E_PARA);

    HCCL_RUN_INFO("[%s] listNum[%u], commTag[%s]", __func__, listNum, commTag.c_str());
    std::vector<std::vector<char>> hostPackBuffers(listNum);
    HcclChannelUrmaRes channelParam{};
    CHK_SAFETY_FUNC_RET(memset_s(&channelParam, sizeof(channelParam), 0, sizeof(channelParam)));

    // 获取host侧序列化的地址
    std::vector<u32> channelSizeVec{};
    uint32_t totalListNum = 0;
    for (uint32_t index = 0; index < listNum; index++) {
        if (hcommDesc[index].remoteEndpoint.protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
            auto aicpuTsP2pChannel = ReinterpretAs<AicpuTsP2pChannel*>(hostChannelHandles[index]);
            CHK_PRT(aicpuTsP2pChannel->H2DResPack(hostPackBuffers[index]));
        } else if (hcommDesc[index].remoteEndpoint.protocol == CommProtocol::COMM_PROTOCOL_UBOE) {
            auto aicpuTsUboeChannel = ReinterpretAs<AicpuTsUboeChannel*>(hostChannelHandles[index]);
            CHK_PRT(aicpuTsUboeChannel->H2DResPack(hostPackBuffers[index]));
        } else if (hcommDesc[index].remoteEndpoint.protocol == CommProtocol::COMM_PROTOCOL_UB_RTP) {
            auto aicpuTsUbRtpChannel = ReinterpretAs<AicpuTsUbRtpChannel*>(hostChannelHandles[index]);
            CHK_PRT(aicpuTsUbRtpChannel->H2DResPack(hostPackBuffers[index]));
        } else if (hcommDesc[index].remoteEndpoint.protocol == CommProtocol::COMM_PROTOCOL_ROCE) {
            auto aicpuTsRoceChannelV2 = ReinterpretAs<AicpuTsRoceChannelV2*>(hostChannelHandles[index]);
            CHK_PRT(aicpuTsRoceChannelV2->H2DResPack(hostPackBuffers[index]));
        } else {
            auto aicpuTsUrmaChannel = ReinterpretAs<AicpuTsUrmaChannel*>(hostChannelHandles[index]);
            CHK_PRT(aicpuTsUrmaChannel->H2DResPack(hostPackBuffers[index]));
        }
        totalListNum += hostPackBuffers[index].size();
        channelSizeVec.push_back(hostPackBuffers[index].size());
    }
    HCCL_INFO("[%s] totalListNum[%llu]", __func__, totalListNum);

    // 分配连续的host内存，将序列化的地址放入其中
    hccl::HostMem hostPackBuf = hccl::HostMem::alloc(totalListNum);
    CHK_PTR_NULL(hostPackBuf.ptr());
    CHK_RET(CombineHostMemory(hostPackBuffers, hostPackBuf));
    hccl::DeviceMem devicePackBuf = hccl::DeviceMem::alloc(totalListNum);
    CHK_PTR_NULL(devicePackBuf.ptr());

    // 将host侧序列化内容拷贝到device侧内存中
    CHK_RET(hrtMemSyncCopy(
        devicePackBuf.ptr(), totalListNum, hostPackBuf.ptr(), totalListNum,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    hccl::DeviceMem channelSizeAddr = hccl::DeviceMem::alloc(channelSizeVec.size() * sizeof(u32));
    CHK_PTR_NULL(channelSizeAddr.ptr());

    CHK_RET(hrtMemSyncCopy(
        channelSizeAddr.ptr(), channelSizeVec.size() * sizeof(u32), channelSizeVec.data(),
        channelSizeVec.size() * sizeof(u32), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    // 为device侧的channelList分配内存
    hccl::DeviceMem deviceChannelList = hccl::DeviceMem::alloc(listNum * sizeof(ChannelHandle));
    CHK_PTR_NULL(deviceChannelList.ptr());

    // 填充channelParam参数
    CHK_RET(FillChannelParam(
        channelParam, commTag, deviceChannelList, devicePackBuf, listNum, totalListNum, channelSizeAddr));

    // ctx模式：检测channel是否预分配了ctx，复用deviceChannelList填ctx指针，跳过D2H
    bool isCtxMode = false;
    CHK_RET(AicpuTsChannelHelper::TryFillCtxList(
        hostChannelHandles, listNum, deviceChannelList, channelParam.ctxList, isCtxMode));

    // profiling信息
    hccl::DeviceMem remoteRankList = hccl::DeviceMem::alloc(listNum * sizeof(u32));
    CHK_PTR_NULL(remoteRankList.ptr());
    std::vector<u32> remoteRankIdList(listNum);
    // 集合通信场景才能开启
    if (needProfiling) {
        for (u32 i = 0; i < listNum; ++i) {
            CHK_RET(hccl::HcclCommDfx::GetChannelRemoteRankId(commTag, hostChannelHandles[i], remoteRankIdList[i]));
        }
        // 通过安全的内存拷贝将主机内存数据传输到设备内存
        CHK_RET(hrtMemSyncCopy(
            remoteRankList.ptr(), listNum * sizeof(u32), remoteRankIdList.data(), listNum * sizeof(u32),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        channelParam.remoteRankList = static_cast<u32*>(remoteRankList.ptr());
    }

    // 调用抽离的通用内核启动函数
    CHK_RET(LaunchKernel(channelParam, binHandle, kernelName));

    if (!isCtxMode) {
        // 将device侧的channelList拷贝回host侧的channelList
        CHK_RET(hrtMemSyncCopy(
            channelHandles, listNum * sizeof(ChannelHandle), deviceChannelList.ptr(), listNum * sizeof(ChannelHandle),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

        CHK_RET(FillChannelD2HMap(channelHandles, hostChannelHandles, listNum));
    }

    HCCL_INFO("[%s] channel kernel launch success.", __func__);
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelKernelLaunchForComm(
    ChannelHandle* channelHandles, ChannelHandle* hostChannelHandles, HcommChannelDesc* hcommDesc, uint32_t listNum,
    const std::string& commTag, aclrtBinHandle binHandle)
{
    return LaunchChannelKernelCommon(
        channelHandles, hostChannelHandles, hcommDesc, listNum, commTag, binHandle, "RunAicpuIndOpChannelInitV2", true);
}

HcclResult ChannelProcess::ChannelKernelLaunchForBase(
    ChannelHandle* channelHandles, ChannelHandle* hostChannelHandles, HcommChannelDesc* hcommDesc, uint32_t listNum,
    aclrtBinHandle binHandle)
{
    return LaunchChannelKernelCommon(
        channelHandles, hostChannelHandles, hcommDesc, listNum, "", binHandle, "RunAicpuChannelInitV2", false);
}

namespace {

    struct KHost {
        std::vector<std::shared_ptr<hccl::DeviceMem>> mem;
        std::vector<void*> ptr;
        std::vector<u64> sz;
        std::vector<u32> kind;
    };

    struct KDev {
        hccl::DeviceMem data;
        hccl::DeviceMem size;
        hccl::DeviceMem type;
    };

    HcclResult PackHost(ChannelHandle* host, uint32_t n, HcommChannelKind k, KHost& out)
    {
        out.mem.resize(n);
        out.ptr.resize(n);
        out.sz.resize(n);
        out.kind.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            auto* channel = ReinterpretAs<Channel*>(host[i]);
            CHK_PTR_NULL(channel);
            CHK_RET(channel->Serialize(out.mem[i]));
            CHK_PTR_NULL(out.mem[i]);
            CHK_PTR_NULL(out.mem[i]->ptr());
            out.ptr[i] = out.mem[i]->ptr();
            out.sz[i] = out.mem[i]->size();
            out.kind[i] = static_cast<u32>(k);
        }
        return HCCL_SUCCESS;
    }

    HcclResult PushDev(uint32_t n, const KHost& in, KDev& out)
    {
        out.data = hccl::DeviceMem::alloc(n * sizeof(void*));
        CHK_PTR_NULL(out.data.ptr());
        out.size = hccl::DeviceMem::alloc(n * sizeof(u64));
        CHK_PTR_NULL(out.size.ptr());
        out.type = hccl::DeviceMem::alloc(n * sizeof(u32));
        CHK_PTR_NULL(out.type.ptr());

        CHK_RET(hrtMemSyncCopy(
            out.data.ptr(), n * sizeof(void*), in.ptr.data(), n * sizeof(void*),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        CHK_RET(hrtMemSyncCopy(
            out.size.ptr(), n * sizeof(u64), in.sz.data(), n * sizeof(u64),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        CHK_RET(hrtMemSyncCopy(
            out.type.ptr(), n * sizeof(u32), in.kind.data(), n * sizeof(u32),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        return HCCL_SUCCESS;
    }

} // namespace

HcclResult ChannelProcess::LaunchCommonChannelKernel(
    ChannelHandle* channelHandles, ChannelHandle* hostChannelHandles, uint32_t listNum, HcommChannelKind channelKind,
    aclrtBinHandle binHandle)
{
    HCCL_RUN_INFO("[%s] listNum[%u] HcommChannelRes path", __func__, listNum);
    CHK_PRT_RET((listNum == 0), HCCL_ERROR("[%s] Invalid listNum, listNum[%u]", __func__, listNum), HCCL_E_PARA);

    KHost host;
    CHK_RET(PackHost(hostChannelHandles, listNum, channelKind, host));
    KDev dev;
    CHK_RET(PushDev(listNum, host, dev));

    hccl::DeviceMem deviceChannelList = hccl::DeviceMem::alloc(listNum * sizeof(ChannelHandle));
    CHK_PTR_NULL(deviceChannelList.ptr());

    HcommChannelRes channelParam{};
    channelParam.channelList = static_cast<void*>(deviceChannelList.ptr());
    channelParam.listNum = listNum;
    channelParam.channelDataListAddr = static_cast<void*>(dev.data.ptr());
    channelParam.channelDataSizeListAddr = static_cast<void*>(dev.size.ptr());
    channelParam.channelTypeListAddr = static_cast<void*>(dev.type.ptr());
    CHK_RET(hrtGetDevice(&channelParam.deviceInfo.deviceLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(
        static_cast<u32>(channelParam.deviceInfo.deviceLogicId), channelParam.deviceInfo.devicePhyId));
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    channelParam.deviceInfo.deviceType = static_cast<u32>(devType);

    // ctx模式：检测channel是否预分配了ctx，复用deviceChannelList填ctx指针，跳过D2H
    bool isCtxMode = false;
    CHK_RET(AicpuTsChannelHelper::TryFillCtxList(
        hostChannelHandles, listNum, deviceChannelList, channelParam.ctxList, isCtxMode));

    CHK_RET(LaunchKernelDeviceParam(channelParam, binHandle, "RunAicpuChannelInitV3"));

    if (!isCtxMode) {
        CHK_RET(hrtMemSyncCopy(
            channelHandles, listNum * sizeof(ChannelHandle), deviceChannelList.ptr(), listNum * sizeof(ChannelHandle),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

        CHK_RET(FillChannelD2HMap(channelHandles, hostChannelHandles, listNum));
    }
    HCCL_INFO("[%s] channel kernel (HcommChannelRes) launch success.", __func__);
    return HCCL_SUCCESS;
}

static bool Is950PlusChannelKind(HcommChannelKind kind)
{
    return kind == HcommChannelKind::AICPU_TS_URMA || kind == HcommChannelKind::AICPU_TS_UBOE
           || kind == HcommChannelKind::AICPU_TS_UB_RTP || kind == HcommChannelKind::AICPU_TS_ROCE_V2
           || kind == HcommChannelKind::AICPU_TS_PCIE;
}

HcclResult ChannelProcess::LaunchChannelKernel(
    ChannelHandle* channelHandles, ChannelHandle* hostChannelHandles, HcommChannelDesc* hcommDesc, uint32_t listNum,
    aclrtBinHandle binHandle)
{
    HCCL_RUN_INFO("[%s] listNum[%u]", __func__, listNum);
    CHK_PRT_RET(listNum == 0U, HCCL_ERROR("[%s] listNum is 0", __func__), HCCL_E_PARA);
    auto* ch = ReinterpretAs<Channel*>(hostChannelHandles[0]);
    CHK_PTR_NULL(ch);
    HcommChannelKind channelKind = ch->GetChannelKind();
    // 防御性校验：同一批 channel 必须走同一路径（950 或 910），不允许混用
    bool is950PlusChannelKind = Is950PlusChannelKind(channelKind);
    for (uint32_t i = 1; i < listNum; ++i) {
        auto* curCh = reinterpret_cast<Channel*>(hostChannelHandles[i]);
        CHK_PTR_NULL(curCh);
        HcommChannelKind curKind = curCh->GetChannelKind();
        if (Is950PlusChannelKind(curKind) != is950PlusChannelKind) {
            HCCL_ERROR(
                "[%s] mixed channel kind in one batch: index[0] kind[%s] vs index[%u] kind[%s]", __func__,
                HcommChannelKindToString(channelKind), i, HcommChannelKindToString(curKind));
            return HCCL_E_PARA;
        }
    }
    if (is950PlusChannelKind) {
        return ChannelKernelLaunchForBase(channelHandles, hostChannelHandles, hcommDesc, listNum, binHandle);
    }
    return LaunchCommonChannelKernel(channelHandles, hostChannelHandles, listNum, channelKind, binHandle);
}

HcclResult ChannelProcess::PrepareUserChannels(
    ChannelHandle* targetChannels, ChannelHandle* userChannels, HcommChannelDesc* channelDescs, uint32_t channelNum,
    CommEngine engine)
{
    CHK_PTR_NULL(targetChannels);
    CHK_PTR_NULL(userChannels);
    CHK_PRT_RET(
        (channelNum == 0), HCCL_ERROR("[%s] Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);

    HCCL_INFO(
        "[%s] engine[%s], channelNum[%u].", __func__, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),
        channelNum);
    if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
        CHK_RET(AicpuTsChannelHelper::PreAllocChannels(targetChannels, userChannels, channelDescs, channelNum));
    } else if (engine == COMM_ENGINE_AIV) {
        CHK_RET(AivChannelHelper::PreAllocChannels(targetChannels, userChannels, channelDescs, channelNum));
    } else {
        HCCL_INFO(
            "[%s] engine[%s] no need to pre-alloc.", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        for (uint32_t i = 0; i < channelNum; i++) {
            userChannels[i] = targetChannels[i];
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelGetNotifyNum(ChannelHandle channelHandle, uint32_t* notifyNum)
{
    return WithChannelByHandleLocked(channelHandle, [&notifyNum](Channel& channel) -> HcclResult {
        // 锁内调用，避免 destroy 并发释放
        channel.GetNotifyNum(notifyNum);
        return HcclResult::HCCL_SUCCESS;
    });
}

HcclResult ChannelProcess::ChannelGetRemoteMems(
    ChannelHandle channelHandle, uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    CHK_PTR_NULL(remoteMem);
    CHK_PTR_NULL(memNum);
    CHK_PTR_NULL(memInfos);

    return WithChannelByHandleLocked(channelHandle, [&memNum, &remoteMem, &memInfos](Channel& channel) -> HcclResult {
        // 锁内调用，避免 destroy 并发释放
        return channel.GetRemoteMems(memNum, remoteMem, memInfos);
    });
}

HcclResult ChannelProcess::CcuChannelGetRmtMemToken(ChannelHandle channelHandle, uint64_t srcVa, uint64_t& tokenInfo)
{
    return WithChannelByHandleLocked(channelHandle, [&srcVa, &tokenInfo](Channel& channel) -> HcclResult {
        // 锁内调用，避免 destroy 并发释放
        auto* ccuChannel = dynamic_cast<CcuUrmaChannel*>(&channel);
        CHK_PTR_NULL(ccuChannel);
        return ccuChannel->CcuGetRmtMemToken(srcVa, tokenInfo);
    });
}

HcclResult ChannelProcess::ChannelGet(const ChannelHandle channelHandle, void** channel)
{
    CHK_PTR_NULL(channel);
    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    std::lock_guard<std::mutex> lock(g_ChannelMapMtx);
    DeviceChannelKey key{deviceId, channelHandle};
    const auto& D2HhandleIter = g_ChannelD2HMap.find(key);
    if (D2HhandleIter == g_ChannelD2HMap.end()) {
        HCCL_ERROR("[ChannelProcess][%s] deviceId[%d], channel[%llx] not found.", __func__, deviceId, channelHandle);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    const auto handle = D2HhandleIter->second;
    const auto& handleIter = g_ChannelMap.find(handle);
    if (handleIter == g_ChannelMap.end()) {
        HCCL_ERROR("[ChannelProcess][%s] deviceId[%d], channel[%llx] not found.", __func__, deviceId, handle);
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    *channel = ReinterpretAs<void*>(handleIter->second.get());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
ChannelProcess::ChannelKernelDestroy(ChannelHandle* channelHandles, uint32_t listNum, aclrtBinHandle binHandle)
{
    HCCL_RUN_INFO("[%s] listNum[%u]", __func__, listNum);
    HcclChannelUrmaRes channelParam{};
    CHK_SAFETY_FUNC_RET(memset_s(&channelParam, sizeof(channelParam), 0, sizeof(channelParam)));

    // 将 host 侧的 channel handles 拷贝到 device 内存，供内核使用
    hccl::DeviceMem deviceChannelList = hccl::DeviceMem::alloc(listNum * sizeof(ChannelHandle));
    CHK_PTR_NULL(deviceChannelList.ptr());
    CHK_RET(hrtMemSyncCopy(
        deviceChannelList.ptr(), listNum * sizeof(ChannelHandle), channelHandles, listNum * sizeof(ChannelHandle),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    // 填充 channelParam（只需 channelList 和 listNum）
    channelParam.channelList = static_cast<void*>(deviceChannelList.ptr());
    channelParam.listNum = listNum;

    // 下 kernel
    std::string kernelName = "RunAicpuChannelDestroyV2";

    // 调用抽离的通用内核启动函数
    CHK_RET(LaunchKernel(channelParam, binHandle, kernelName));

    HCCL_INFO("[%s] channel kernel destroy success.", __func__);
    return HCCL_SUCCESS;
}

HcclResult
ChannelProcess::RemoveSingleChannel(int32_t deviceId, ChannelHandle inHandle, std::vector<ChannelHandle>& deviceHandles)
{
    DeviceChannelKey key{deviceId, inHandle};
    auto itH = g_ChannelD2HMap.find(key);
    if (itH == g_ChannelD2HMap.end()) {
        HCCL_ERROR(
            "[Hcomm][%s] failed to find handle mapping in g_ChannelD2HMap, deviceId[%d], inHandle[0x%llx].", __func__,
            deviceId, inHandle);
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    const ChannelHandle mappedHandle = itH->second;

    auto itC = g_ChannelMap.find(mappedHandle);
    if (itC == g_ChannelMap.end()) {
        HCCL_ERROR(
            "[Hcomm][%s] failed to find channel in g_ChannelMap, deviceId[%d], inHandle[0x%llx], mappedHandle[0x%llx].",
            __func__, deviceId, inHandle, mappedHandle);
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    deviceHandles.push_back(inHandle);

    HCCL_INFO(
        "[Hcomm][%s] erase channel: deviceId[%d], inHandle[0x%llx], mappedHandle[0x%llx], ptr[%p]", __func__, deviceId,
        inHandle, mappedHandle, itC->second.get());

    g_ChannelMap.erase(itC);

    for (auto it = g_ChannelD2HMap.begin(); it != g_ChannelD2HMap.end();) {
        if (it->first.deviceId == deviceId && it->second == mappedHandle) {
            it = g_ChannelD2HMap.erase(it);
        } else {
            ++it;
        }
    }
    // 同步删除 H2D 反向映射（键为 {deviceId, host 句柄}）
    g_ChannelH2DMap.erase(DeviceChannelKey{deviceId, mappedHandle});
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelDestroy(const ChannelHandle* channels, uint32_t channelNum, aclrtBinHandle binHandle)
{
    CHK_PTR_NULL(channels);
    CHK_PRT_RET((channelNum == 0), HCCL_ERROR("[%s] Invalid channelNum[0]", __func__), HCCL_E_PARA);
    HCCL_INFO("[%s] START. channelNum[%u].", __func__, channelNum);

    int32_t deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    std::vector<ChannelHandle> deviceHandles;

    {
        std::lock_guard<std::mutex> lock(g_ChannelMapMtx);
        for (uint32_t i = 0; i < channelNum; ++i) {
            HcclResult ret = RemoveSingleChannel(deviceId, channels[i], deviceHandles);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
        }
    }

    if (!deviceHandles.empty() && binHandle) {
        CHK_RET(ChannelKernelDestroy(deviceHandles.data(), deviceHandles.size(), binHandle));
    }
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelClean(const ChannelHandle* channelList, uint32_t channelNum)
{
    CHK_PTR_NULL(channelList);

    for (uint32_t i = 0; i < channelNum; ++i) {
        const ChannelHandle inHandle = channelList[i];
        // 单锁：D2H 映射 + 查 map + 锁内调用 Clean()
        HcclResult ret = WithChannelByHandleLocked(inHandle, [](Channel& channel) -> HcclResult {
            return channel.Clean();
        });
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[%s] ChannelHandle Clean failed, ret = 0x%016llx, i = %u", __func__, HCCL_ERROR_CODE(ret), i);
            return ret;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelResumeConcurrency(const ChannelHandle* channelList, uint32_t channelNum)
{
    for (uint32_t i = 0; i < channelNum; ++i) {
        const ChannelHandle inHandle = channelList[i];
        HcclResult ret = WithChannelByHandleLocked(inHandle, [](Channel& channel) -> HcclResult {
            return channel.Resume();
        });
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[%s] Get ChannelHandle failed, ret = 0x%016llx, i = %u", __func__, HCCL_ERROR_CODE(ret), i);
            return ret;
        }
    }
    return HCCL_SUCCESS;
}
HcclResult ChannelProcess::ChannelResume(const ChannelHandle* channelList, uint32_t channelNum)
{
    CHK_PTR_NULL(channelList);

    // 1.resume resource
    HcclResult ret = ChannelResumeConcurrency(channelList, channelNum);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("HcommChannelResumeConcurrency error, ret = 0x%016llx", HCCL_ERROR_CODE(ret));
        return ret;
    }

    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    HCCL_INFO("[%s] start resuming channels, timeout[%u]sec", __func__, timeout);

    std::vector<int32_t> statusVec(channelNum, 0);
    int32_t* statusList = statusVec.data();
    uint32_t retryCount{0};
    while (true) {
        HcclResult ret = ChannelGetStatus(channelList, channelNum, statusList);
        // 1. 检查超时
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_ERROR(
                "[%s] channel resume timeout after %u sec, channelNum[%u], elapsed[%lld]ms, retryCount[%u]", __func__,
                timeout, channelNum, elapsed, retryCount);
            return HCCL_E_TIMEOUT;
        }

        // 2. 处理重试（去除频繁的重试日志，一秒可能重试上千次）
        if (ret == HCCL_E_AGAIN) {
            ++retryCount;
            continue;
        }

        // 3. 处理失败
        if (ret != HCCL_SUCCESS) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_ERROR(
                "[%s] channel connect failed, channelNum[%u], ret[%d], elapsed[%lld]ms, retryCount[%u]", __func__,
                channelNum, ret, elapsed, retryCount);
            return ret;
        }

        // 4. 正常情况：所有通道连接成功
        auto elapsed
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                  .count();
        HCCL_INFO(
            "[%s] all channels connected successfully, channelNum[%u], elapsed[%lld]ms, retryCount[%u]", __func__,
            channelNum, elapsed, retryCount);
        break;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ChannelProcess::ChannelUpdateKernelLaunch(
    ChannelHandle* deviceChannelHandles, ChannelHandle* hostChannelHandles, uint32_t listNum,
    const std::string& commTag, aclrtBinHandle binHandle)
{
    HCCL_RUN_INFO("[%s] listNum[%u], commTag[%s]", __func__, listNum, commTag.c_str());
    std::vector<std::vector<char>> hostPackBuffers(listNum);
    HcclChannelUrmaRes channelParam{};
    CHK_SAFETY_FUNC_RET(memset_s(&channelParam, sizeof(channelParam), 0, sizeof(channelParam)));

    // 获取host侧序列化的地址
    uint32_t totalListNum = 0;
    std::vector<u32> channelSizeVec{};
    for (uint32_t index = 0; index < listNum; index++) {
        Channel* channel = ReinterpretAs<Channel*>(hostChannelHandles[index]);
        if (channel->GetChannelKind() != HcommChannelKind::AICPU_TS_URMA) {
            CHK_RET(PackAicpuTsChannelH2DRes(hostChannelHandles[index], hostPackBuffers[index]));
            totalListNum += hostPackBuffers[index].size();
            channelSizeVec.push_back(hostPackBuffers[index].size());
            continue;
        }
        auto aicpuTsUrmaChannel = ReinterpretAs<AicpuTsUrmaChannel*>(hostChannelHandles[index]);
        CHK_RET(aicpuTsUrmaChannel->H2DResPack(hostPackBuffers[index])); // todo:后续只打包connection
        totalListNum += hostPackBuffers[index].size();
        channelSizeVec.push_back(hostPackBuffers[index].size());
    }
    HCCL_INFO("[%s] totalListNum[%llu]", __func__, totalListNum);

    hccl::DeviceMem channelSizeAddr;
    hccl::DeviceMem devicePackBuf;
    CHK_RET(
        CopyUpdateKernelPackResToDevice(hostPackBuffers, channelSizeVec, totalListNum, channelSizeAddr, devicePackBuf));

    s32 sRet = strncpy_s(channelParam.hcomId, HCOMID_MAX_LENGTH, commTag.c_str(), HCOMID_MAX_LENGTH - 1);
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[%s] str copy fail. return[%d]", __func__, sRet), HCCL_E_INTERNAL);
    channelParam.listNum = listNum;
    channelParam.uniqueIdAddr = static_cast<void*>(devicePackBuf.ptr());
    channelParam.uniqueIdSize = totalListNum;
    channelParam.channelSizeAddr = static_cast<void*>(channelSizeAddr.ptr());

    // 将 host 侧的 channel handles 拷贝到 device 内存，供内核使用
    hccl::DeviceMem deviceChannelList;
    CHK_RET(CopyUpdateKernelChannelListToDevice(deviceChannelHandles, listNum, deviceChannelList));
    channelParam.channelList = static_cast<void*>(deviceChannelList.ptr());

    std::string kernelName = "RunAicpuIndOpChannelUpdateV2";
    CHK_RET(LaunchKernel(channelParam, binHandle, kernelName));

    HCCL_INFO("[%s] channel kernel launch success.", __func__);
    return HCCL_SUCCESS;
}

} // namespace hcomm
