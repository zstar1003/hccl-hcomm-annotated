/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "shared_jetty_mgr.h"
#include "log.h"

namespace hcomm {

SharedJettyMgr& SharedJettyMgr::GetInstance()
{
    static SharedJettyMgr instance;
    return instance;
}

HcclResult
SharedJettyMgr::RegisterChannels(EndpointHandle endpointHandle, const ChannelHandle* channels, uint32_t channelNum)
{
    if (endpointHandle == nullptr || channels == nullptr || channelNum == 0) {
        HCCL_ERROR(
            "[%s] invalid params, endpointHandle[%p], channels[%p], channelNum[%u].", __func__, endpointHandle,
            channels, channelNum);
        return HCCL_E_PARA;
    }

    std::lock_guard<std::mutex> lock(mtx_);
    auto& ctx = contexts_[endpointHandle];
    for (uint32_t i = 0; i < channelNum; ++i) {
        ctx.channelHandles.insert(channels[i]);
    }
    // 以集合实际大小为准：insert 对重复句柄为 no-op，若用 += channelNum 会导致
    // channelCount > channelHandles.size()，UnregisterChannels 永远无法将 count 减到 0，
    // CheckEndpointDestroy 永久阻塞 endpoint 销毁。
    ctx.channelCount = static_cast<uint32_t>(ctx.channelHandles.size());
    HCCL_INFO(
        "[%s] registered %u channels for endpointHandle[%p], total channelCount[%u].", __func__, channelNum,
        endpointHandle, ctx.channelCount);
    return HCCL_SUCCESS;
}

HcclResult SharedJettyMgr::UnregisterChannels(const ChannelHandle* channels, uint32_t channelNum)
{
    if (channels == nullptr || channelNum == 0) {
        return HCCL_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(mtx_);
    for (uint32_t i = 0; i < channelNum; ++i) {
        for (auto it = contexts_.begin(); it != contexts_.end(); ++it) {
            auto handleIt = it->second.channelHandles.find(channels[i]);
            if (handleIt != it->second.channelHandles.end()) {
                it->second.channelHandles.erase(handleIt);
                if (it->second.channelCount > 0) {
                    it->second.channelCount--;
                }
                HCCL_INFO(
                    "[%s] unregistered channel[0x%llx] from endpointHandle[%p], remaining[%u].", __func__, channels[i],
                    it->first, it->second.channelCount);
                if (it->second.channelCount == 0) {
                    EndpointHandle epHandle = it->first;
                    contexts_.erase(it);
                    HCCL_INFO(
                        "[%s] all channels unregistered, context removed for endpointHandle[%p].", __func__, epHandle);
                }
                break;
            }
        }
        // 非共享 jetty 的 channel 也会走 HcommChannelDestroy，此处静默忽略，避免日志刷屏
    }
    return HCCL_SUCCESS;
}

HcclResult SharedJettyMgr::CheckEndpointDestroy(EndpointHandle endpointHandle)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = contexts_.find(endpointHandle);
    if (it == contexts_.end()) {
        return HCCL_SUCCESS;
    }
    HCCL_ERROR(
        "[%s] cannot destroy endpointHandle[%p], still has [%u] shared jetty channels.", __func__, endpointHandle,
        it->second.channelCount);
    return HCCL_E_UNAVAIL;
}

bool SharedJettyMgr::HasContext(EndpointHandle endpointHandle)
{
    std::lock_guard<std::mutex> lock(mtx_);
    return contexts_.find(endpointHandle) != contexts_.end();
}

void SharedJettyMgr::UnregisterEndpoint(EndpointHandle endpointHandle)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = contexts_.find(endpointHandle);
    if (it == contexts_.end()) {
        return;
    }
    if (it->second.channelCount > 0) {
        HCCL_WARNING(
            "[%s] endpointHandle[%p] still has [%u] channels, force remove to avoid handle reuse misjudge.", __func__,
            endpointHandle, it->second.channelCount);
    }
    contexts_.erase(it);
    HCCL_INFO("[%s] unregistered endpointHandle[%p] from shared jetty mgr.", __func__, endpointHandle);
}

} // namespace hcomm
