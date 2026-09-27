/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_CHANNEL_HELPER_H
#define AICPU_TS_CHANNEL_HELPER_H

#include "cast_utils.h"
#include <memory>
#include <mutex>
#include <vector>
#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"
#include "mem_device_pub.h"

static const uint32_t HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD = 0x0fcf0f4fU;
static const uint32_t HCOMM_AICPU_CHANNEL_CTX_VERSION = 0U;

using HcommAicpuChannelCtx = struct {
    CommAbiHeader abiHeader{HCOMM_AICPU_CHANNEL_CTX_VERSION, HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD, 0, 0};
    CommEngine engine{COMM_ENGINE_RESERVED};
    CommProtocol protocol{COMM_PROTOCOL_RESERVED};
    void* deviceChannel{nullptr};
};

/**
 * @brief AICPU 专用辅助类，持有 ctx 内存等 AICPU 专属资源，提供 AICPU 批量建链状态机接口。
 */
class AicpuTsChannelHelper {
public:
    // === per-channel 实例方法 ===
    HcclResult PreAllocCtx(ChannelHandle& outHandle)
    {
        ctxMem_ = std::make_shared<hccl::DeviceMem>(hccl::DeviceMem::alloc(sizeof(HcommAicpuChannelCtx)));
        CHK_PTR_NULL(ctxMem_->ptr());
        outHandle = ReinterpretAs<ChannelHandle>(ctxMem_->ptr());
        HCCL_INFO("[AicpuTsChannelHelper] pre-alloc ctx success, ctxPtr[%p]", ctxMem_->ptr());
        return HCCL_SUCCESS;
    }
    void SetCtxMem(std::shared_ptr<hccl::DeviceMem> mem) { ctxMem_ = std::move(mem); }
    void* GetCtxPtr() const { return ctxMem_ ? ctxMem_->ptr() : nullptr; }

    // === 批量静态方法 ===
    static HcclResult HandleStatus(
        const ChannelHandle* channelList, uint32_t listNum, CommEngine engine, const HcommChannelDesc* channelDescs,
        const std::vector<int32_t>& linkStatusList, int32_t* statusList);
    static HcclResult PreAllocChannels(
        ChannelHandle* targetChannels, ChannelHandle* userChannels, HcommChannelDesc* channelDescs,
        uint32_t channelNum);
    static HcclResult EnsureKernelBinLoaded(CommEngine engine);
    static aclrtBinHandle GetBinHandle() { return g_BinHandle; }

    static HcclResult TryFillCtxList(
        ChannelHandle* hostChannelHandles, uint32_t listNum, const hccl::DeviceMem& deviceChannelList,
        void*& outCtxList, bool& isCtxMode);

private:
    static HcclResult LaunchKernel(
        const ChannelHandle* channelList, uint32_t listNum, CommEngine engine, const HcommChannelDesc* channelDescs,
        aclrtBinHandle binHandle);

    std::shared_ptr<hccl::DeviceMem> ctxMem_;
    static aclrtBinHandle g_BinHandle;
    static std::mutex g_BinHandleMtx;
};

inline HcclResult UnwrapChannelHandle(ChannelHandle& handle)
{
    if (handle == 0) {
        HCCL_ERROR("[%s] handle is 0.", __func__);
        return HCCL_E_PTR;
    }
    const auto* ctx = ReinterpretAs<const HcommAicpuChannelCtx*>(static_cast<uintptr_t>(handle));
    if (ctx->abiHeader.version == HCOMM_AICPU_CHANNEL_CTX_VERSION
        && ctx->abiHeader.magicWord == HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD) {
        if (ctx->deviceChannel == nullptr) {
            HCCL_ERROR("[%s] ctx deviceChannel is null.", __func__);
            return HCCL_E_INTERNAL;
        }
        handle = ReinterpretAs<ChannelHandle>(ctx->deviceChannel);
    }
    return HCCL_SUCCESS;
}

#endif // AICPU_TS_CHANNEL_HELPER_H
