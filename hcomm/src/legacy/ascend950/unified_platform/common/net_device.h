/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_PLT_NET_DEVICE_H
#define HCCLV2_PLT_NET_DEVICE_H

#include "orion_adapter_hccp.h"
#include "inner_net_dev.h"
#include "buffer_key.h"
#include "hccl_net_dev_defs.h"

namespace Hccl {

class HcclNetDevice {
public:
    static HcclProtoType ConvertHcclProtoToLinkProto(Hccl::LinkProtoType type)
    {
        switch (type) {
            case Hccl::LinkProtoType::HCCS_PCIE:
                return HCCL_PROTO_TYPE_RESERVED;
            case Hccl::LinkProtoType::TCP:
                return HCCL_PROTO_TYPE_TCP;
            case Hccl::LinkProtoType::RDMA:
                return HCCL_PROTO_TYPE_ROCE;
            case Hccl::LinkProtoType::UB:
                return HCCL_PROTO_TYPE_UBC_TP;
            default:
                return HCCL_PROTO_TYPE_RESERVED;
        }
    }

    static HcclNetDevDeployment ConvertDeploymentType(Hccl::PortDeploymentType type)
    {
        if (type == Hccl::PortDeploymentType::DEV_NET) {
            return HCCL_NETDEV_DEPLOYMENT_DEVICE;
        } else if (type == Hccl::PortDeploymentType::HOST_NET) {
            return HCCL_NETDEV_DEPLOYMENT_HOST;
        } else {
            return HCCL_NETDEV_DEPLOYMENT_RESERVED;
        }
    }
    HcclNetDevice(const NetDevInfo& info);
    ~HcclNetDevice();

    // NetDevInfo 读写函数
    NetDevInfo GetNetDevInfo() const;

    // Getter 方法：返回 ndev_ 的当前值
    InnerNetDev* GetInnerNetDev() const;

    // Setter 方法：允许外部修改 ndev_ 的值
    void SetInnerNetDev(InnerNetDev* value);

    RdmaHandle GetRdmaHandle() const;

    std::pair<TokenIdHandle, uint32_t> GetTokenIdInfo(const BufferKey<uintptr_t, u64>& bufKey) const;
    void PutTokenIdInfo(const BufferKey<uintptr_t, u64>& bufKey, TokenIdHandle tokenIdHandle) const;
    bool IsUB();

private:
    NetDevInfo netDevInfo_;
    InnerNetDev* ndev_;
};

} // namespace Hccl

#endif // HCCLV2_PLT_NET_DEVICE_H
