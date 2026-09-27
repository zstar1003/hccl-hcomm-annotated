/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ns_recovery.h"
#include "channel_process.h"
#include "log.h"
#include "comm_engine_utils.h"

namespace hccl {

void NsRecoveryProcessor::SetKfcControlTransfer(
    std::shared_ptr<HDCommunicate> kfcControlTransferH2D, std::shared_ptr<HDCommunicate> kfcStatusTransferD2H)
{
    kfcControlTransferH2D_ = kfcControlTransferH2D;
    kfcStatusTransferD2H_ = kfcStatusTransferD2H;
}

void NsRecoveryProcessor::AddNsRecoveryData(
    const CommEngine& engine, const ChannelHandle* const channelHandles,
    const ChannelHandle* const hostChannelHandleList, uint32_t channelNum, const std::string& commTag)
{
    HCCL_INFO(
        "[NsRecovery][AddData] AddNsRecoveryData for engine[%s], channelNum[%u], commTag[%s]",
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum, commTag.c_str());
    std::vector<ChannelHandle> deviceList;
    std::vector<ChannelHandle> hostList;
    for (uint32_t index = 0; index < channelNum; ++index) {
        deviceList.push_back(channelHandles[index]);
        hostList.push_back(hostChannelHandleList[index]);
    }
    NsRecoveryData data{deviceList, hostList, channelNum, commTag};
    nsRecoveryDatas_[engine].emplace_back(std::move(data));
}

constexpr u32 WAIT_CMD_TIMEOUT = 10 * 1000; // 最大等待10秒
HcclResult NsRecoveryProcessor::PollStopStatus()
{
    Hccl::KfcExecStatus opInfo;
    auto timeout = std::chrono::milliseconds(WAIT_CMD_TIMEOUT);
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        CHK_RET(kfcStatusTransferD2H_->Get(0, sizeof(Hccl::KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
        if (opInfo.kfcStatus == Hccl::KfcStatus::STOP_LAUNCH_DONE) {
            HCCL_INFO(
                "[NsRecovery][Suspend] received KfcStatus[%d], which is STOP_LAUNCH_DONE",
                static_cast<int>(opInfo.kfcStatus));
            return HcclResult::HCCL_E_SUSPENDING;
        } else if (opInfo.kfcStatus == Hccl::KfcStatus::ERROR) {
            HCCL_ERROR(
                "[NsRecovery][Suspend] received KfcStatus[%d], which is ERROR", static_cast<int>(opInfo.kfcStatus));
            return HcclResult::HCCL_E_INTERNAL;
        } else {
            if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
                HCCL_ERROR(
                    "[NsRecovery][Suspend] Wait suspend response status timeout[%u ms] and get the kfcStatus is [%d].",
                    WAIT_CMD_TIMEOUT, static_cast<int>(opInfo.kfcStatus));
                return HcclResult::HCCL_E_TIMEOUT;
            }
            continue;
        }
    }
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult NsRecoveryProcessor::ListenBackGround(Hccl::KfcExecStatus& opInfo)
{
    auto timeout = std::chrono::milliseconds(WAIT_CMD_TIMEOUT);
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        CHK_RET(kfcStatusTransferD2H_->Get(0, sizeof(Hccl::KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
        if (opInfo.kfcStatus == Hccl::KfcStatus::CLEAN_DONE) {
            HCCL_INFO(
                "[NsRecovery][Clean] received KfcStatus[%d], which is CLEAN_DONE", static_cast<int>(opInfo.kfcStatus));
            return HcclResult::HCCL_E_SUSPENDING;
        } else if (opInfo.kfcStatus == Hccl::KfcStatus::ERROR) {
            HCCL_ERROR(
                "[NsRecovery][Clean] received KfcStatus[%d], which is ERROR", static_cast<int>(opInfo.kfcStatus));
            return HcclResult::HCCL_E_INTERNAL;
        } else {
            if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
                HCCL_ERROR(
                    "[NsRecovery][Clean] Wait clean response status timeout[%u ms] and get the kfcStatus is [%d].",
                    WAIT_CMD_TIMEOUT, static_cast<int>(opInfo.kfcStatus));
                return HcclResult::HCCL_E_TIMEOUT;
            }
            continue;
        }
    }
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult NsRecoveryProcessor::StopLaunch()
{
    for (const auto& recoveryData : nsRecoveryDatas_) {
        if (recoveryData.first == COMM_ENGINE_AICPU || recoveryData.first == COMM_ENGINE_AICPU_TS) {
            // Aicpu场景
            Hccl::KfcCommand opCmd = Hccl::KfcCommand::NS_STOP_LAUNCH;
            CHK_RET(kfcControlTransferH2D_->Put(0, sizeof(Hccl::KfcCommand), reinterpret_cast<uint8_t*>(&opCmd)));
            HCCL_INFO(
                "[NsRecovery][Suspend] send KfcCommand[%d] success, which is NS_STOP_LAUNCH.", static_cast<int>(opCmd));

            auto ret = PollStopStatus(); // todo：多CommEngine的管理存在问题
            if (ret != HcclResult::HCCL_E_SUSPENDING) {
                HCCL_ERROR("[NsRecovery][Suspend] PollStopStatus failed, ret[%d]", ret);
                return ret;
            }
            return HcclResult::HCCL_SUCCESS;
        } else {
            HCCL_INFO("[NsRecovery][Suspend] Aicpu kernel is not launched yet. Suspend host only.");
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult NsRecoveryProcessor::Clean()
{
    for (const auto& recoveryData : nsRecoveryDatas_) {
        if (recoveryData.first == COMM_ENGINE_AICPU || recoveryData.first == COMM_ENGINE_AICPU_TS) {
            // 再清理device，后续优化全用host管理
            HCCL_INFO("[NsRecovery][Clean] start to clean device, waiting for device STOP_LAUNCH_DONE");
            Hccl::KfcExecStatus opInfo;
            CHK_RET(kfcStatusTransferD2H_->Get(0, sizeof(Hccl::KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
            if (opInfo.kfcStatus == Hccl::KfcStatus::STOP_LAUNCH_DONE) {
                HCCL_INFO(
                    "[NsRecovery][Clean] received KfcStatus[%d], which is STOP_LAUNCH_DONE",
                    static_cast<int>(opInfo.kfcStatus));
                // 通知背景线程清理device侧资源
                Hccl::KfcCommand opCmd = Hccl::KfcCommand::NS_CLEAN;
                CHK_RET(kfcControlTransferH2D_->Put(0, sizeof(Hccl::KfcCommand), reinterpret_cast<uint8_t*>(&opCmd)));
                HCCL_INFO(
                    "[NsRecovery][Clean] send KfcCommand [%d] success, which is NS_CLEAN", static_cast<int>(opCmd));

                // 监听背景线程状态
                auto ret = ListenBackGround(opInfo);
                if (ret != HcclResult::HCCL_E_SUSPENDING) {
                    HCCL_ERROR("[NsRecovery][Clean] ListenBackGround failed, ret[%d]", ret);
                    return ret;
                }
                return HcclResult::HCCL_SUCCESS;
            } else {
                HCCL_ERROR(
                    "[NsRecovery][Clean] Aicpu kernel is not stopped yet. Cannot clean, kfcStatus is [%s]",
                    opInfo.kfcStatus.Describe().c_str());
                return HcclResult::HCCL_E_INTERNAL;
            }
            return HcclResult::HCCL_SUCCESS;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult NsRecoveryProcessor::Resume(aclrtBinHandle binHandle)
{
    for (auto& recoveryData : nsRecoveryDatas_) {
        if (recoveryData.first == COMM_ENGINE_AICPU || recoveryData.first == COMM_ENGINE_AICPU_TS) {
            for (auto& handleData : recoveryData.second) {
                CHK_RET(hcomm::ChannelProcess::ChannelUpdateKernelLaunch(
                    handleData.channelHandles_.data(), handleData.hostChannelHandleList_.data(), handleData.channelNum_,
                    handleData.commTag_, binHandle));
            }
        }
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
