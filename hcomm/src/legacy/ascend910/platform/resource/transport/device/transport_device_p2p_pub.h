/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_DEVICE_P2P_PUB_H
#define TRANSPORT_DEVICE_P2P_PUB_H

#include "transport_p2p_pub.h"

namespace hccl {
class TransportDeviceP2p : public TransportP2p {
public:
    explicit TransportDeviceP2p(
        DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
        std::chrono::milliseconds timeout, const TransportDeviceP2pData& transDevP2pData);
    ~TransportDeviceP2p() override;

    HcclResult Init() override;

    HcclResult UpdateRemoteAddr(void* remoteIn, void* remoteOut) override;

protected:
    HcclResult SignalRecord(
        std::shared_ptr<RemoteNotify>& remoteSignal, u64 remoteSignalAddr, u64 remoteSignalOffset,
        Stream& stream) override;

private:
    HcclResult CheckRelationship(u32 relationship);
    HcclResult ConfigUseSdmaCopyToSignalRecord();
    template <typename T>
    HcclResult ModifySignalAddrToVA(s32 deviceId, std::shared_ptr<T>& notify);
    HcclResult GetNotifyAddr(s32 deviceId, const HcclSignalInfo& signalInfo, u64& addr);
    DeviceMem signalMem_;
};
} // namespace hccl

#endif /* TRANSPORT_DEVICE_P2P_PUB_H */
