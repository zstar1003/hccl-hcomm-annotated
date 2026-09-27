/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef P2P_MGMT_H
#define P2P_MGMT_H

#include <functional>
#include <array>
#include <mutex>

#include "p2p_mgmt_pub.h"
#include "hccl_common.h"
#include "adapter_rts.h"

namespace hccl {
class P2PMgmt {
public:
    static P2PMgmt& Instance();
    HcclResult EnableP2P(std::vector<uint32_t> remoteDevices);
    HcclResult DisableP2P(std::vector<uint32_t> remoteDevices);
    HcclResult WaitP2PEnabled(
        std::vector<uint32_t> remoteDevices, std::function<bool()> needStop = []() {
            return false;
        });
    HcclResult SetStopFlag(bool value);
    bool GetStopFlag();

private:
    explicit P2PMgmt();
    ~P2PMgmt();
    // 以下接口暂无外部使用场景，置为私有。根据需要再进行公开。
    HcclResult EnableP2P(uint32_t remoteDevicePhysicID);
    HcclResult DisableAllP2P();
    HcclResult DisableP2P(uint32_t localDeviceLogicID, uint32_t remoteDevicePhysicID);
    HcclResult WaitP2PEnabled(
        uint32_t remoteDevicePhysicID, std::function<bool()> needStop = []() {
            return false;
        });

    HcclResult CheckP2P(uint32_t remoteDevicePhysicID, bool& enabled);
    HcclResult WaitP2PConnected(
        int32_t localDeviceLogicID, uint32_t remoteDevicePhysicID, std::function<bool()> needStop = []() {
            return false;
        });
    HcclResult CheckMarsterId(uint32_t remoteDevicePhysicID, uint32_t localDevicePhysicID, bool& isMarsterIdDiff);
    bool IsNeedEstablishP2Pconnection(uint32_t remoteDevicePhysicID);
    bool IsStandardCardFor910B(std::vector<uint32_t>& remoteDevicePhysicIDs);

    // <<remoteDevicePhsicID, P2PConnectionInfo>, localDeviceLogicID >
    std::array<std::map<uint32_t, P2PConnectionInfo>, MAX_MODULE_DEVICE_NUM> connectionsInfo_;
    std::array<std::mutex, MAX_MODULE_DEVICE_NUM> connectionsLock_;
    DevType deviceType_;
    static std::atomic<bool> initFlag_;
    std::atomic<bool> stopFlag_{false};
    bool isStandardCardFor910B_{false};
};
} // namespace hccl
#endif // P2P_MGMT_PUB_H
