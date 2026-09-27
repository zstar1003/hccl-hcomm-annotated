/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_NETWORK_H
#define HCCL_NETWORK_H

#include "hccl_network_pub.h"
#include "hccl_common.h"
#include "hccl_ip_address.h"
#include "hccl_net_dev.h"
#include "rma_buffer_mgr.h"
#include "local_ipc_rma_buffer.h"
#include "local_rdma_rma_buffer.h"

namespace hccl {
class NetDevContext {
public:
    using LocalIpcRmaBufferMgr = hcomm::RmaBufferMgr<BufferKey<uintptr_t, u64>, std::shared_ptr<LocalIpcRmaBuffer>>;
    using LocalRdmaRmaBufferMgr = hcomm::RmaBufferMgr<BufferKey<uintptr_t, u64>, std::shared_ptr<LocalRdmaRmaBuffer>>;

    NetDevContext() {}
    ~NetDevContext() {}
    HcclResult Init(
        NicType nicType, s32 devicePhyId, s32 deviceLogicId, HcclIpAddress localIp,
        HcclIpAddress backupIp = HcclIpAddress(0));
    HcclResult Deinit();
    HcclResult InitV2(const HcclNetDevInfos* info);
    HcclResult GetinfoConfig(const HcclNetDevInfos* info);
    HcclResult ConvertIP(const HcclAddress address);
    HcclResult DeinitV2();
    void SetTlsStatus(TlsStatus tlsStatus);
    void SetIsNotNeedGetTlsStatus(bool isNotNeedGetTlsStatus);
    std::mutex mu_;

    NicType GetNicType() const { return nicType_; }

    HcclIpAddress GetLocalIp() const { return localIp_; }

    HcclIpAddress GetBackupIp() const { return backupIp_; }

    s32 GetPhyId() const { return devicePhyId_; }

    s32 GetLogicId() const { return deviceLogicId_; }
    HcclNetDevDeployment GetNetDevDeployment() const { return netDevDeployment_; }

    bool GetIsBackup() const { return isBackup_; }
    NICDeployment GetNicDeployment() const { return nicDeployment_; }
    bool IsNotNeedGetTlsStatus() { return isNotNeedGetTlsStatus_; }
    TlsStatus GettlsStatus() { return tlsStatus_; }
    HcclProtoType GetProtoType() const { return protoType_; }
    void SetProtoType(u32 proto) { protoType_ = (HcclProtoType)proto; }

    std::shared_ptr<LocalIpcRmaBufferMgr> GetlocalIpcRmaBufferMgr()
    {
        if (!localIpcRmaBufferMgr_) {
            EXCEPTION_CATCH((localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>()), return nullptr);
        }
        return localIpcRmaBufferMgr_;
    }

    std::shared_ptr<LocalRdmaRmaBufferMgr> GetlocalRdmaRmaBufferMgr()
    {
        if (!localRdmaRmaBufferMgr_) {
            EXCEPTION_CATCH((localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>()), return nullptr);
        }
        return localRdmaRmaBufferMgr_;
    }

private:
    NICDeployment nicDeployment_{NICDeployment::NIC_DEPLOYMENT_DEVICE};
    s32 devicePhyId_{INVALID_INT};
    s32 deviceLogicId_{INVALID_INT};
    HcclIpAddress localIp_;
    HcclIpAddress backupIp_;
    NicType nicType_{NicType::DEVICE_NIC_TYPE};
    bool isHostUseDevNic_{false};
    SocketHandle hostSocketHandle_{nullptr};
    HcclProtoType protoType_{HCCL_PROTO_TYPE_RESERVED};
    HcclNetDevDeployment netDevDeployment_{HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE};
    void* handle_{nullptr};
    bool isBackup_{false};
    TlsStatus tlsStatus_ = TlsStatus::UNKNOWN;
    bool isNotNeedGetTlsStatus_ = false;
    std::shared_ptr<LocalIpcRmaBufferMgr> localIpcRmaBufferMgr_{nullptr};
    std::shared_ptr<LocalRdmaRmaBufferMgr> localRdmaRmaBufferMgr_{nullptr};
};
} // namespace hccl

#endif
