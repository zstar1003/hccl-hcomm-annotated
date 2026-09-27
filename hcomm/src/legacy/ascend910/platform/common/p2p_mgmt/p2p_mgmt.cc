/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "p2p_mgmt.h"

#include <chrono>
#include <thread>
#include "adapter_error_manager.h"
#include "externalinput_pub.h"
#include "device_capacity.h"
#include "mem_name_repository_pub.h"
#include "sal_pub.h"
#include "driver/ascend_hal.h"
#include "workflow_pub.h"

namespace hccl {
const u32 DEVICE_PER_MODULE = 8; // A+X 一个mesh卡数
const u32 DIE_PER_MODULE = 16;   // 910_93 16die
std::atomic<bool> P2PMgmt::initFlag_ = {false};
P2PMgmt& P2PMgmt::Instance()
{
    static P2PMgmt mgmt;
    return mgmt;
}

P2PMgmt::P2PMgmt() : deviceType_(DevType::DEV_TYPE_COUNT) { initFlag_ = true; }

P2PMgmt::~P2PMgmt() { initFlag_ = false; }

HcclResult P2PMgmt::EnableP2P(std::vector<uint32_t> remoteDevices)
{
    if (initFlag_) {
        isStandardCardFor910B_ = IsStandardCardFor910B(remoteDevices);
        for (auto& remoteDevicePhysicID : remoteDevices) {
            if (IsNeedEstablishP2Pconnection(remoteDevicePhysicID)) {
                CHK_RET(EnableP2P(remoteDevicePhysicID));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult P2PMgmt::EnableP2P(uint32_t remoteDevicePhysicID)
{
    if (Is310PDevice()) {
        return HCCL_SUCCESS;
    }
    int32_t localDeviceLogicID;
    CHK_RET(hrtGetDevice(&localDeviceLogicID));
    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    CHK_PRT_RET(
        static_cast<u32>(localDeviceLogicID) >= maxDeviceNum,
        HCCL_ERROR(
            "[EnableP2P]localDeviceLogicID[%d] is bigger than maxDeviceNum[%u]", localDeviceLogicID, maxDeviceNum),
        HCCL_E_INTERNAL);

    std::unique_lock<std::mutex> lock(connectionsLock_[localDeviceLogicID]);
    auto& iterLocalDevice = connectionsInfo_[localDeviceLogicID];
    auto iterRemoteDevice = iterLocalDevice.find(remoteDevicePhysicID);
    if ((iterRemoteDevice == iterLocalDevice.end()) || (iterRemoteDevice->second.reference == 0)) {
        bool isMarsterIdDiff = false;
        u32 localDevicePhysicID = 0;
        CHK_RET(hrtGetDevicePhyIdByIndex(localDeviceLogicID, localDevicePhysicID));
        CHK_RET(CheckMarsterId(remoteDevicePhysicID, localDevicePhysicID, isMarsterIdDiff));
        HCCL_INFO(
            "[EnableP2P][CheckMasterId]localDevicePhysicID[%u], remoteDevicePhysicID[%u], isMarsterIdDiff[%s]",
            localDevicePhysicID, remoteDevicePhysicID, isMarsterIdDiff ? "true" : "false");
        if (isMarsterIdDiff) {
            CHK_RET(hrtEnableP2P(localDeviceLogicID, remoteDevicePhysicID));
            HCCL_INFO(
                "[EnableP2P]enable p2p: local logic id:%d, local physic id:%u, remote physic id:%u.",
                localDeviceLogicID, localDevicePhysicID, remoteDevicePhysicID);
        }
        iterLocalDevice[remoteDevicePhysicID].status = P2PStatus::P2P_STATUS_ENABLING;
        iterLocalDevice[remoteDevicePhysicID].reference++;
        return HCCL_SUCCESS;
    } else {
        // 使已执行过 enable，且未执行过 disable，不重复执行 enable p2p。
        iterLocalDevice[remoteDevicePhysicID].reference++;
    }
    return HCCL_SUCCESS;
}

HcclResult P2PMgmt::DisableAllP2P()
{
    int32_t localDeviceLogicID;
    CHK_RET(hrtGetDevice(&localDeviceLogicID));
    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    CHK_PRT_RET(
        static_cast<u32>(localDeviceLogicID) >= maxDeviceNum,
        HCCL_ERROR(
            "[DisableAllP2P]localDeviceLogicID[%d] is bigger than maxDeviceNum[%u]", localDeviceLogicID, maxDeviceNum),
        HCCL_E_INTERNAL);

    std::map<uint32_t, P2PConnectionInfo> localP2PInfo;
    std::unique_lock<std::mutex> lock(connectionsLock_[localDeviceLogicID]);
    auto& iterLocalDevice = connectionsInfo_[localDeviceLogicID];
    if (iterLocalDevice.empty()) {
        return HCCL_SUCCESS;
    } else {
        localP2PInfo = iterLocalDevice;
    }
    lock.unlock();

    for (auto& iterRemoteDevice : localP2PInfo) {
        if (iterRemoteDevice.second.reference == 0) {
            continue;
        }

        bool isMarsterIdDiff = false;
        u32 localDevicePhysicID = 0;
        CHK_RET(hrtGetDevicePhyIdByIndex(localDeviceLogicID, localDevicePhysicID));
        HcclResult ret = CheckMarsterId(iterRemoteDevice.first, localDevicePhysicID, isMarsterIdDiff);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Disable][AllP2P]check pcie connection failed. device info: local logic id:%d, "
                "remote physic id:%u.",
                localDeviceLogicID, iterRemoteDevice.first),
            ret);
        if (isMarsterIdDiff) {
            HCCL_INFO(
                "there is active p2p connections. in P2PMgmt disable all p2p, it is forced to disable p2p. "
                "device info: local logic id:%d, remote physic id:%u.",
                localDeviceLogicID, iterRemoteDevice.first);
            CHK_RET(hrtDisableP2P(localDeviceLogicID, iterRemoteDevice.first));
        }
    }

    lock.lock();
    connectionsInfo_[localDeviceLogicID].clear();
    lock.unlock();
    return HCCL_SUCCESS;
}

HcclResult P2PMgmt::DisableP2P(std::vector<uint32_t> remoteDevices)
{
    if (initFlag_) {
        if (Is310PDevice()) {
            return HCCL_SUCCESS;
        }
        int32_t localDeviceLogicID;
        CHK_RET(hrtGetDevice(&localDeviceLogicID));
        u32 maxDeviceNum;
        CHK_RET(GetMaxDevNum(maxDeviceNum));
        CHK_PRT_RET(
            static_cast<u32>(localDeviceLogicID) >= maxDeviceNum,
            HCCL_ERROR(
                "[DisableP2P]localDeviceLogicID[%d] is bigger than maxDeviceNum[%u]", localDeviceLogicID, maxDeviceNum),
            HCCL_E_INTERNAL);

        HcclWorkflowMode mode = GetWorkflowMode();
        if (mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
            MemNameRepository::GetInstance(localDeviceLogicID)->ClearMemNameRepository();
        }

        for (auto& remoteDevicePhysicID : remoteDevices) {
            if (IsNeedEstablishP2Pconnection(remoteDevicePhysicID)) {
                CHK_RET(DisableP2P(localDeviceLogicID, remoteDevicePhysicID));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult P2PMgmt::DisableP2P(uint32_t localDeviceLogicID, uint32_t remoteDevicePhysicID)
{
    std::unique_lock<std::mutex> lock(connectionsLock_[localDeviceLogicID]);
    auto& iterLocalDevice = connectionsInfo_[localDeviceLogicID];
    auto iterRemoteDevice = iterLocalDevice.find(remoteDevicePhysicID);
    if ((iterRemoteDevice == iterLocalDevice.end()) || (iterRemoteDevice->second.reference == 0)) {
        HCCL_WARNING(
            "there is no p2p connections, no need to disable p2p. "
            "device info: local logic id:%d, remote physic id:%u.",
            localDeviceLogicID, remoteDevicePhysicID);
        return HCCL_SUCCESS;
    }

    iterRemoteDevice->second.reference--;
    if (iterRemoteDevice->second.reference == 0) {
        bool isMarsterIdDiff = false;
        u32 localDevicePhysicID = 0;
        CHK_RET(hrtGetDevicePhyIdByIndex(localDeviceLogicID, localDevicePhysicID, true));
        HCCL_INFO("local logic id:%d, local physic id:%u.", localDeviceLogicID, localDevicePhysicID);
        HcclResult ret = CheckMarsterId(remoteDevicePhysicID, localDevicePhysicID, isMarsterIdDiff);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Disable][P2P]check pcie connection failed. device info: local logic id:%d, "
                "remote physic id:%u.",
                localDeviceLogicID, remoteDevicePhysicID),
            ret);
        if (isMarsterIdDiff) {
            HCCL_INFO("disable p2p: local logic id:%d, remote physic id:%u.", localDeviceLogicID, remoteDevicePhysicID);
            CHK_RET(hrtDisableP2P(localDeviceLogicID, remoteDevicePhysicID));
        }
        iterLocalDevice[remoteDevicePhysicID].status = P2PStatus::P2P_STATUS_DISABLED;
    }

    return HCCL_SUCCESS;
}

HcclResult P2PMgmt::WaitP2PEnabled(std::vector<uint32_t> remoteDevices, std::function<bool()> needStop)
{
    if (initFlag_) {
        for (auto& remoteDevicePhysicID : remoteDevices) {
            if (IsNeedEstablishP2Pconnection(remoteDevicePhysicID)) {
                CHK_RET(WaitP2PEnabled(remoteDevicePhysicID, needStop));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult P2PMgmt::CheckMarsterId(uint32_t remoteDevicePhysicID, uint32_t localDevicePhysicID, bool& isMarsterIdDiff)
{
    if (localDevicePhysicID == remoteDevicePhysicID) {
        isMarsterIdDiff = false;
        return HCCL_SUCCESS;
    }
    if (deviceType_ == DevType::DEV_TYPE_910B || deviceType_ == DevType::DEV_TYPE_910_93) {
        s64 localDevicePhysicValue, remoteDevicePhysicValue;
        CHK_RET(hrtGetPhyDeviceInfo(
            localDevicePhysicID, MODULE_TYPE_SYSTEM, RT_PHY_INFO_TYPE_MASTER_ID, localDevicePhysicValue));
        CHK_RET(hrtGetPhyDeviceInfo(
            remoteDevicePhysicID, MODULE_TYPE_SYSTEM, RT_PHY_INFO_TYPE_MASTER_ID, remoteDevicePhysicValue));

        isMarsterIdDiff = (localDevicePhysicValue == remoteDevicePhysicValue) ? false : true;
        return HCCL_SUCCESS;
    }
    LinkTypeInServer linkType;
    CHK_RET(hrtGetPairDeviceLinkType(localDevicePhysicID, remoteDevicePhysicID, linkType));

    isMarsterIdDiff = ((linkType != LinkTypeInServer::HCCS_TYPE) && (linkType != LinkTypeInServer::SIO_TYPE)
                       && (linkType != LinkTypeInServer::HCCS_SW_TYPE)) ?
                          true :
                          false;
    return HCCL_SUCCESS;
}

HcclResult P2PMgmt::WaitP2PEnabled(uint32_t remoteDevicePhysicID, std::function<bool()> needStop)
{
    if (Is310PDevice()) {
        return HCCL_SUCCESS;
    }
    int32_t localDeviceLogicID;
    CHK_RET(hrtGetDevice(&localDeviceLogicID));
    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    CHK_PRT_RET(
        static_cast<u32>(localDeviceLogicID) >= maxDeviceNum,
        HCCL_ERROR(
            "[WaitP2PEnabled]localDeviceLogicID[%d] is bigger than maxDeviceNum[%u]", localDeviceLogicID, maxDeviceNum),
        HCCL_E_INTERNAL);
    std::unique_lock<std::mutex> lock(connectionsLock_[localDeviceLogicID]);

    auto& iterLocalDevice = connectionsInfo_[localDeviceLogicID];
    auto iterRemoteDevice = iterLocalDevice.find(remoteDevicePhysicID);
    bool bErr = (iterRemoteDevice == iterLocalDevice.end()) || (iterRemoteDevice->second.reference == 0)
                || (iterRemoteDevice->second.status == P2PStatus::P2P_STATUS_DISABLED);
    CHK_PRT_RET(
        bErr,
        HCCL_ERROR(
            "[Wait][P2PEnabled]wait p2p enabled failed. enable operation has not been executed, "
            "ret[%u]. device info: local logic id:%d, remote physic id:%u.",
            HCCL_E_INTERNAL, localDeviceLogicID, remoteDevicePhysicID),
        HCCL_E_INTERNAL);

    if (iterRemoteDevice->second.status == P2PStatus::P2P_STATUS_ENABLED) {
        return HCCL_SUCCESS;
    } else {
        bool isMarsterIdDiff = false;
        u32 localDevicePhysicID = 0;
        CHK_RET(hrtGetDevicePhyIdByIndex(localDeviceLogicID, localDevicePhysicID));
        HcclResult ret = CheckMarsterId(remoteDevicePhysicID, localDevicePhysicID, isMarsterIdDiff);
        HCCL_INFO(
            "[WaitP2PEnabled][CheckMasterId]localDevicePhysicID[%u], remoteDevicePhysicID[%u], isMarsterIdDiff[%s]",
            localDevicePhysicID, remoteDevicePhysicID, isMarsterIdDiff ? "true" : "false");
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Wait][P2PEnabled]check pcie connection failed. device info: local logic id:%d, "
                "remote physic id:%u.",
                localDeviceLogicID, remoteDevicePhysicID),
            ret);
        if (isMarsterIdDiff) {
            CHK_RET(WaitP2PConnected(localDeviceLogicID, remoteDevicePhysicID, needStop));
            HCCL_INFO(
                "[Wait]enable p2p: local logic id:%d, local physic id:%u, remote physic id:%u.", localDeviceLogicID,
                localDevicePhysicID, remoteDevicePhysicID);
        }
        iterLocalDevice[remoteDevicePhysicID].status = P2PStatus::P2P_STATUS_ENABLED;
    }
    return HCCL_SUCCESS;
}

HcclResult
P2PMgmt::WaitP2PConnected(int32_t localDeviceLogicID, uint32_t remoteDevicePhysicID, std::function<bool()> needStop)
{
    // 读取P2P状态超时时间
    const std::chrono::seconds timeout(GetExternalInputHcclLinkTimeOut());
    const std::chrono::milliseconds checkP2PTimeInterval(1); // 轮询P2P状态时间 1ms
    const auto start = TIME_NOW();

    while (true) {
        CHK_PRT_RET(needStop(), HCCL_ERROR("Terminating operation due to external request"), HCCL_E_INTERNAL);

        bool enabled = false;
        CHK_RET(CheckP2P(remoteDevicePhysicID, enabled));

        if (enabled) {
            HCCL_INFO(
                "connected p2p success, take time [%lld]us. device info: local logic id:%d, remote physic id:%u.",
                DURATION_US(TIME_NOW() - start), localDeviceLogicID, remoteDevicePhysicID);
            return HCCL_SUCCESS;
        }
        std::this_thread::sleep_for(checkP2PTimeInterval);
        /* 获取当前时间，如果耗时超过timeout，则返回错误 */
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(TIME_NOW() - start);
        if (elapsed > timeout) {
            RPT_INNER_ERR_PRT(
                "connected p2p timeout, timeout:%d s. local logicDevid:%d, "
                "remote physic id:%u The possible causes are as follows: 1. the connection "
                "between this device and the target device is abnormal 2. an exception occurred "
                "at the target devices 3. The ranktable is not matched.",
                GetExternalInputHcclLinkTimeOut(), localDeviceLogicID, remoteDevicePhysicID);

            HCCL_ERROR(
                "[Wait][P2PConnected]connected p2p timeout, timeout:%d s. local logicDevid:%d, "
                "remote physic id:%u.",
                GetExternalInputHcclLinkTimeOut(), localDeviceLogicID, remoteDevicePhysicID);
            return HCCL_E_DRV;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult P2PMgmt::CheckP2P(uint32_t remoteDevicePhysicID, bool& enabled)
{
    uint32_t status = DRV_P2P_STATUS_DISABLE;
    int32_t localDeviceLogicID;
    CHK_RET(hrtGetDevice(&localDeviceLogicID));

    CHK_RET(hrtGetP2PStatus(localDeviceLogicID, remoteDevicePhysicID, &status));

    enabled = (status == DRV_P2P_STATUS_ENABLE);
    return HCCL_SUCCESS;
}

/*
 * ****************************************************************************
 * 判断localdevice和remotedevice是否在相同平面，在相同平面内的device间需要做P2P
 * *****************************************************************************
 */
bool P2PMgmt::IsNeedEstablishP2Pconnection(uint32_t remoteDevicePhysicID)
{
    if (static_cast<s32>(remoteDevicePhysicID) == HOST_DEVICE_ID)
        return false;

    int32_t localDeviceLogicID;
    CHK_RET(hrtGetDevice(&localDeviceLogicID));
    u32 localDevicePhysicID = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(localDeviceLogicID, localDevicePhysicID));
    CHK_RET(hrtGetDeviceType(deviceType_));
    if (deviceType_ == DevType::DEV_TYPE_310P3 || isStandardCardFor910B_) {
        return true;
    }
    u32 deviceNum = (deviceType_ == DevType::DEV_TYPE_910_93) ? DIE_PER_MODULE : DEVICE_PER_MODULE;
    return (
        (localDevicePhysicID % deviceNum == remoteDevicePhysicID % deviceNum)
        || (localDevicePhysicID / deviceNum == remoteDevicePhysicID / deviceNum));
}

bool P2PMgmt::IsStandardCardFor910B(std::vector<uint32_t>& remoteDevicePhysicIDs)
{
    // 非910B场景返回false
    CHK_RET(hrtGetDeviceType(deviceType_));
    if (deviceType_ != DevType::DEV_TYPE_910B) {
        return false;
    }

    int32_t localDeviceLogicID;
    CHK_RET(hrtGetDevice(&localDeviceLogicID));
    u32 localDevicePhysicID = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(localDeviceLogicID, localDevicePhysicID));
    LinkTypeInServer linkType;
    for (auto remoteDevicePhysicID : remoteDevicePhysicIDs) {
        CHK_RET(hrtGetPairDeviceLinkType(localDevicePhysicID, remoteDevicePhysicID, linkType));
        // 两卡之间的链路是HCCS或者SIO时，返回false
        if (linkType == LinkTypeInServer::HCCS_TYPE || linkType == LinkTypeInServer::SIO_TYPE
            || linkType == LinkTypeInServer::HCCS_SW_TYPE) {
            return false;
        }
    }
    HCCL_INFO("[IsStandardCardFor910B] isStandardCardFor910B_[true]");
    return true;
}
} // namespace hccl
