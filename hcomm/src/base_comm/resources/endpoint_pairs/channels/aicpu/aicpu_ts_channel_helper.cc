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
#include "aicpu_ts_channel_helper.h"
#include "channel_process.h"
#include "channel.h"
#include "launch_aicpu.h"
#include "launch_device.h"
#include "comm_engine_utils.h"
#include "adapter_rts_common.h"

using namespace hcomm;

aclrtBinHandle AicpuTsChannelHelper::g_BinHandle = nullptr;
std::mutex AicpuTsChannelHelper::g_BinHandleMtx;

HcclResult AicpuTsChannelHelper::TryFillCtxList(
    ChannelHandle* hostChannelHandles, uint32_t listNum, const hccl::DeviceMem& deviceChannelList, void*& outCtxList,
    bool& isCtxMode)
{
    auto* firstCh = ReinterpretAs<Channel*>(hostChannelHandles[0]);
    CHK_PTR_NULL(firstCh);
    auto* firstHelper = firstCh->GetAicpuTsHelper();
    CHK_PTR_NULL(firstHelper);
    isCtxMode = (firstHelper->GetCtxPtr() != nullptr);
    HCCL_INFO("[%s] isCtxMode[%d].", __func__, isCtxMode);
    if (!isCtxMode) {
        return HCCL_SUCCESS;
    }
    std::vector<void*> ctxVec(listNum);
    for (uint32_t i = 0; i < listNum; i++) {
        auto* ch = ReinterpretAs<Channel*>(hostChannelHandles[i]);
        CHK_PTR_NULL(ch);
        auto* helper = ch->GetAicpuTsHelper();
        ctxVec[i] = helper ? helper->GetCtxPtr() : nullptr;
    }
    HcclResult ret = hrtMemSyncCopy(
        deviceChannelList.ptr(), listNum * sizeof(void*), ctxVec.data(), listNum * sizeof(void*),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[TryFillCtxList] hrtMemSyncCopy failed, ret[%d]", ret);
        return ret;
    }
    outCtxList = deviceChannelList.ptr();
    return HCCL_SUCCESS;
}

HcclResult AicpuTsChannelHelper::EnsureKernelBinLoaded(CommEngine engine)
{
    if (engine != COMM_ENGINE_AICPU && engine != COMM_ENGINE_AICPU_TS) {
        HCCL_INFO(
            "[%s] engine[%s] kernel loading not required", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(g_BinHandleMtx);
    if (g_BinHandle != nullptr) {
        return HCCL_SUCCESS;
    }
    std::string jsonPath;
    CHK_RET(hccl::GetKernelFilePath(jsonPath));
    jsonPath += "ccl_kernel.json";

    HcclResult ret = hccl::LoadBinaryFromFile(jsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0, g_BinHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] load aicpu file fail, path[%s]", __func__, jsonPath.c_str()), ret);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsChannelHelper::LaunchKernel(
    const ChannelHandle* channelList, uint32_t listNum, [[maybe_unused]] CommEngine engine,
    const HcommChannelDesc* channelDescs, aclrtBinHandle binHandle)
{
    CHK_PTR_NULL(channelList);
    CHK_PRT_RET((listNum == 0), HCCL_ERROR("[%s] Invalid listNum, listNum[%u]", __func__, listNum), HCCL_E_PARA);

    // 过滤出未就绪的子集，避免重复下 kernel 导致 device 侧 channel 对象泄漏
    std::vector<ChannelHandle> subHostHandles;
    std::vector<HcommChannelDesc> subDescs;
    std::vector<Channel*> subChannels;
    subHostHandles.reserve(listNum);
    subDescs.reserve(listNum);
    subChannels.reserve(listNum);
    for (uint32_t i = 0; i < listNum; i++) {
        void* ch = nullptr;
        CHK_RET(ChannelProcess::ChannelGet(channelList[i], &ch));
        CHK_PTR_NULL(ch);
        auto* channel = static_cast<Channel*>(ch);
        if (channel->IsDeviceEntityReady()) {
            continue;
        }
        subHostHandles.push_back(ReinterpretAs<ChannelHandle>(ch));
        subDescs.push_back(channelDescs[i]);
        subChannels.push_back(channel);
    }
    if (subHostHandles.empty()) {
        HCCL_INFO("[%s] all channels already ready, skip kernel launch.", __func__);
        return HCCL_SUCCESS;
    }
    // 序列化 + kernel launch
    std::vector<ChannelHandle> devHandles(subHostHandles.size());
    CHK_RET(ChannelProcess::LaunchChannelKernel(
        devHandles.data(), subHostHandles.data(), subDescs.data(), subHostHandles.size(), binHandle));
    for (auto* channel : subChannels) {
        channel->SetDeviceEntityReady();
    }
    HCCL_INFO("[%s] aicpu kernel launch success, launched[%zu]/total[%u].", __func__, subHostHandles.size(), listNum);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsChannelHelper::HandleStatus(
    const ChannelHandle* channelList, uint32_t listNum, CommEngine engine, const HcommChannelDesc* channelDescs,
    const std::vector<int32_t>& linkStatusList, int32_t* statusList)
{
    bool allReady = true;
    for (uint32_t i = 0; i < listNum; i++) {
        if (linkStatusList[i] != HCOMM_CHANNEL_STATUS_READY) {
            allReady = false;
            break;
        }
    }
    if (!allReady) {
        for (uint32_t i = 0; i < listNum; i++) {
            if (linkStatusList[i] == HCOMM_CHANNEL_STATUS_FAILED) {
                statusList[i] = HCOMM_CHANNEL_STATUS_FAILED;
            } else if (linkStatusList[i] == HCOMM_CHANNEL_STATUS_TIMEOUT) {
                statusList[i] = HCOMM_CHANNEL_STATUS_TIMEOUT;
            } else {
                statusList[i] = HCOMM_CHANNEL_STATUS_CONNECTING;
            }
        }
        return HCCL_SUCCESS;
    }
    CHK_RET(EnsureKernelBinLoaded(engine));
    HcclResult kernelRet = LaunchKernel(channelList, listNum, engine, channelDescs, g_BinHandle);
    if (kernelRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] LaunchKernel failed, ret[%d]", __func__, kernelRet);
        return HCCL_E_INTERNAL;
    }
    for (uint32_t i = 0; i < listNum; i++) {
        statusList[i] = linkStatusList[i];
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsChannelHelper::PreAllocChannels(
    ChannelHandle* targetChannels, ChannelHandle* userChannels, [[maybe_unused]] HcommChannelDesc* channelDescs,
    uint32_t channelNum)
{
    CHK_PTR_NULL(targetChannels);
    CHK_PTR_NULL(userChannels);
    CHK_PRT_RET(
        (channelNum == 0), HCCL_ERROR("[%s] Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);

    for (uint32_t i = 0; i < channelNum; i++) {
        auto* channel = ReinterpretAs<Channel*>(targetChannels[i]);
        CHK_PTR_NULL(channel);
        auto* helper = channel->GetAicpuTsHelper();
        CHK_PTR_NULL(helper);
        CHK_RET(helper->PreAllocCtx(userChannels[i]));
        // 清零 ctx device 内存，防止脏数据 magic 误匹配
        CHK_RET(hrtMemSet(helper->GetCtxPtr(), sizeof(HcommAicpuChannelCtx), sizeof(HcommAicpuChannelCtx)));
    }

    CHK_RET(ChannelProcess::FillChannelD2HMap(userChannels, targetChannels, channelNum));
    return HCCL_SUCCESS;
}
