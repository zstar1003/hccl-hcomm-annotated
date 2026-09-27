/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "independent_op.h"
#include "launch_aicpu.h"
#include "manager_common.h"
#include "comm_configer.h"
#include "adapter_prof.h"
#include "hcom_host_profiling.h"

namespace hccl {

IndependentOp::IndependentOp() {};

HcclResult IndependentOp::SetIndependentOpConfig(
    const CommConfig& commConfig, [[maybe_unused]] const RankTable_t& rankTable, const HcclTopoAttr& topoAttr,
    const aclrtBinHandle binHandle, HDCommunicateParams& kfcControlTransferH2DParams,
    HDCommunicateParams& kfcStatusTransferD2HParams, CCLBufferManager& bufferManager)
{
    commEngine_ = HCCL_COMM_ENGINE_CONFIG_NOT_SET;
    threadNum_ = HCCL_COMM_THREADNUM_CONFIG_NOT_SET;
    notifyNumPerThread_ = HCCL_COMM_NOTIFY_NUM_PER_THREAD_CONFIG_NOT_SET;
    cclBufferSize_ = commConfig.GetConfigBufferSize();
    commId_ = commConfig.GetConfigCommName();
    commMemMgr_.CommSetHcclBufferManager(bufferManager);
    binHandle_ = binHandle;

    // aicpu侧初始化状态的回调函数
    ManagerCallbacks callbacks;
    callbacks.getAicpuCommState = [this]() {
        return this->GetAicpuCommState();
    };
    callbacks.setAicpuCommState = [this](bool state) {
        this->SetAicpuCommState(state);
    };
    callbacks.kernelLaunchAicpuCommInit = [this]() {
        return this->KernelLaunchAicpuCommInit();
    };

    CHK_PRT(engineResMgr_.Init(threadNum_, notifyNumPerThread_, commId_, binHandle, callbacks));
    CHK_PRT(channelMgr_.Init(binHandle, topoAttr.userRank, callbacks));

    // Aicpu通信域初始化参数
    snprintf_s(commAicpuParam_.hcomId, HCOMID_MAX_SIZE, HCOMID_MAX_SIZE - 1, "%s", commId_.c_str());
    commAicpuParam_.deviceLogicId = topoAttr.deviceLogicId;
    commAicpuParam_.devicePhyId = topoAttr.devicePhyId;
    commAicpuParam_.deviceType = static_cast<u32>(topoAttr.deviceType);
    commAicpuParam_.kfcControlTransferH2DParams = kfcControlTransferH2DParams;
    commAicpuParam_.kfcStatusTransferD2HParams = kfcStatusTransferD2HParams;
    commAicpuParam_.userRank = topoAttr.userRank;
    commAicpuParam_.userRankSize = topoAttr.userRankSize;
    CHK_PRT(channelMgr_.SetHcclQos(commConfig.GetConfigHcclQos()));
    HCCL_INFO(
        "[IndependentOp][%s] Hcom[%s] threadNum[%u], notifyPerThread[%u], cclBufferSize[%llu], deviceLogicId[%u], "
        "devicePhyId[%u], deviceType[%u], userRank[%u], userRankSize[%u]",
        __func__, commId_.c_str(), threadNum_, notifyNumPerThread_, cclBufferSize_, commAicpuParam_.deviceLogicId,
        commAicpuParam_.devicePhyId, commAicpuParam_.deviceType, commAicpuParam_.userRank,
        commAicpuParam_.userRankSize);
    return HCCL_SUCCESS;
}

bool IndependentOp::GetAicpuCommState() { return isAicpuCommInit_.load(std::memory_order_acquire); }

void IndependentOp::SetAicpuCommState(bool aicpuCommState)
{
    isAicpuCommInit_.store(aicpuCommState, std::memory_order_release);
    return;
}

HcclResult IndependentOp::KernelLaunchAicpuCommInit()
{
    // 创建局部流
    uint64_t beginTime = hrtMsprofSysCycleTime();
    Stream localStream(StreamType::STREAM_TYPE_ONLINE);
    constexpr u32 aicpuStreamMode = 1;
    CHK_RET(hrtStreamSetMode(localStream.ptr(), aicpuStreamMode));

    // 下kernel进行自定义算子aicpu侧通信域的公共初始化
    std::string kernelName = "RunAicpuCommInit";

    u16 timeOut = NOTIFY_DEFAULT_WAIT_TIME > std::numeric_limits<uint16_t>::max() ?
                      std::numeric_limits<uint16_t>::max() :
                      NOTIFY_DEFAULT_WAIT_TIME;
    CHK_RET(AicpuAclKernelLaunch(
        localStream.ptr(), reinterpret_cast<void*>(&commAicpuParam_), sizeof(commAicpuParam_), binHandle_, kernelName,
        true, timeOut));
    CHK_RET(hcclStreamSynchronize(localStream.ptr(), CommConfiger::GetInstance().GetCommConfigExecTimeOut("")));

    // 打印增加初始化对应的参数
    HCCL_RUN_INFO("[%s] KernelLaunchAicpuCommInit Success", __func__);
    const std::string profName = "RunAicpuCommInit";
    HCCL_INFO("[%s] RunAicpuCommInit", __func__);
    // 上报初始化kernel的时间
    HcommProfilingReportKernel(beginTime, profName.c_str());
    return HCCL_SUCCESS;
}

HcclResult IndependentOp::SetChannelCallbacks(const ChannelManagerCallbacks& channelCallbacks)
{
    return channelMgr_.SetChannelCallbacks(channelCallbacks);
}

} // namespace hccl
