/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "p2p_enable_manager.h"
#include "log.h"
#include "env_config_v2.h"

namespace Hccl {

P2PEnableManager& P2PEnableManager::GetInstance()
{
    static P2PEnableManager p2pEnableManager;
    return p2pEnableManager;
}

HcclResult P2PEnableManager::EnableP2P(std::vector<uint32_t> remoteDevices)
{
    auto localDeviceLogicID = HrtGetDevice();

    for (auto remoteDevicePhysicID : remoteDevices) {
        CHK_RET(EnableP2P(localDeviceLogicID, remoteDevicePhysicID));
    }
    return HCCL_SUCCESS;
}

HcclResult P2PEnableManager::EnableP2P(uint32_t localDeviceLogicID, uint32_t remoteDevicePhysicID)
{
    std::unique_lock<std::mutex> lock(connectionsLock_[localDeviceLogicID]);
    auto& iterLocalDevice = connectionsInfo_[localDeviceLogicID];
    auto iterRemoteDevice = iterLocalDevice.find(remoteDevicePhysicID);
    if ((iterRemoteDevice == iterLocalDevice.end()) || (iterRemoteDevice->second.reference == 0)) {
        auto localDevicePhysicID = HrtGetDevicePhyIdByUserDevId(localDeviceLogicID);
        CHK_RET(HrtEnableP2P(localDeviceLogicID, remoteDevicePhysicID));
        HCCL_INFO(
            "[EnableP2P]enable p2p: local logic id:%u, local physic id:%u, remote physic id:%u.", localDeviceLogicID,
            localDevicePhysicID, remoteDevicePhysicID);
        iterLocalDevice[remoteDevicePhysicID].status = P2PStatus::P2P_STATUS_ENABLING;
        iterLocalDevice[remoteDevicePhysicID].reference++;
        return HCCL_SUCCESS;
    } else {
        // 使已执行过 enable，且未执行过 disable，不重复执行 enable p2p。
        iterLocalDevice[remoteDevicePhysicID].reference++;
    }
    return HCCL_SUCCESS;
}

HcclResult P2PEnableManager::WaitP2PEnabled(std::vector<uint32_t> remoteDevices)
{
    auto localDeviceLogicID = HrtGetDevice();

    for (auto& remoteDevicePhysicID : remoteDevices) {
        CHK_RET(WaitP2PEnabled(localDeviceLogicID, remoteDevicePhysicID));
    }
    return HCCL_SUCCESS;
}

HcclResult P2PEnableManager::WaitP2PEnabled(uint32_t localDeviceLogicID, uint32_t remoteDevicePhysicID)
{
    std::unique_lock<std::mutex> lock(connectionsLock_[localDeviceLogicID]);

    auto& iterLocalDevice = connectionsInfo_[localDeviceLogicID];
    auto iterRemoteDevice = iterLocalDevice.find(remoteDevicePhysicID);
    bool bErr = (iterRemoteDevice == iterLocalDevice.end()) || (iterRemoteDevice->second.reference == 0)
                || (iterRemoteDevice->second.status == P2PStatus::P2P_STATUS_DISABLED);
    CHK_PRT_RET(
        bErr,
        HCCL_ERROR(
            "[Wait][P2PEnabled]wait p2p enabled failed. enable operation has not been executed, "
            "ret[%u]. device info: local logic id:%u, remote physic id:%u.",
            HCCL_E_INTERNAL, localDeviceLogicID, remoteDevicePhysicID),
        HCCL_E_INTERNAL);

    if (iterRemoteDevice->second.status == P2PStatus::P2P_STATUS_ENABLED) {
        return HCCL_SUCCESS;
    } else {
        auto localDevicePhysicID = HrtGetDevicePhyIdByUserDevId(localDeviceLogicID);

        CHK_RET(WaitP2PConnected(localDeviceLogicID, remoteDevicePhysicID));
        HCCL_INFO(
            "[Wait]enable p2p: local logic id:%u, local physic id:%u, remote physic id:%u.", localDeviceLogicID,
            localDevicePhysicID, remoteDevicePhysicID);
        iterLocalDevice[remoteDevicePhysicID].status = P2PStatus::P2P_STATUS_ENABLED;
    }
    return HCCL_SUCCESS;
}

HcclResult P2PEnableManager::WaitP2PConnected(uint32_t localDeviceLogicID, uint32_t remoteDevicePhysicID) const
{
    // 读取P2P状态超时时间
    const std::chrono::seconds timeout(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    const std::chrono::milliseconds checkP2PTimeInterval(1); // 轮询P2P状态时间 1ms
    const auto start = TIME_NOW();

    while (true) {
        uint32_t status = DRV_P2P_STATUS_DISABLE;
        CHK_RET(HrtGetP2PStatus(localDeviceLogicID, remoteDevicePhysicID, &status));

        if (status == DRV_P2P_STATUS_ENABLE) {
            HCCL_INFO(
                "connected p2p success, take time [%lld]us. device info: local logic id:%u, remote physic id:%u.",
                DURATION_US(TIME_NOW() - start), localDeviceLogicID, remoteDevicePhysicID);
            return HCCL_SUCCESS;
        }
        std::this_thread::sleep_for(checkP2PTimeInterval);
        /* 获取当前时间，如果耗时超过timeout，则返回错误 */
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(TIME_NOW() - start);
        if (elapsed > timeout) {
            HCCL_ERROR(
                "[Wait][P2PConnected]connected p2p timeout, timeout:%d s. local logicDevid:%u, "
                "remote physic id:%u.",
                EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut(), localDeviceLogicID, remoteDevicePhysicID);
            return HCCL_E_DRV;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult P2PEnableManager::DisableP2P(uint32_t localDeviceLogicID, std::vector<uint32_t> remoteDevices)
{
    try {
        for (auto& remoteDevicePhysicID : remoteDevices) {
            CHK_RET(DisableP2P(localDeviceLogicID, remoteDevicePhysicID));
        }
    } catch (HcclException& e) {
        HCCL_ERROR("%s", e.what());
        return e.GetErrorCode();
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult P2PEnableManager::DisableP2P(uint32_t localDeviceLogicID, uint32_t remoteDevicePhysicID)
{
    std::unique_lock<std::mutex> lock(connectionsLock_[localDeviceLogicID]);
    auto& iterLocalDevice = connectionsInfo_[localDeviceLogicID];
    auto iterRemoteDevice = iterLocalDevice.find(remoteDevicePhysicID);
    if ((iterRemoteDevice == iterLocalDevice.end()) || (iterRemoteDevice->second.reference == 0)) {
        HCCL_WARNING(
            "there is no p2p connections, no need to disable p2p. "
            "device info: local logic id:%u, remote physic id:%u.",
            localDeviceLogicID, remoteDevicePhysicID);
        return HCCL_SUCCESS;
    }

    iterRemoteDevice->second.reference--;
    if (iterRemoteDevice->second.reference == 0) {
        auto localDevicePhysicID = HrtGetDevicePhyIdByUserDevId(localDeviceLogicID);
        HCCL_INFO(
            "disable p2p: local logic id:%u, local physic id:%u, remote physic id:%u.", localDeviceLogicID,
            localDevicePhysicID, remoteDevicePhysicID);
        CHK_RET(HrtDisableP2P(localDeviceLogicID, remoteDevicePhysicID));
        iterLocalDevice[remoteDevicePhysicID].status = P2PStatus::P2P_STATUS_DISABLED;
    }
    return HCCL_SUCCESS;
}

P2PEnableManager::~P2PEnableManager() {}

} // namespace Hccl
