/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPUTS_ROCE_ENDPOINT_H
#define AICPUTS_ROCE_ENDPOINT_H

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include "hccl_mem_defs.h"
#include "endpoint.h"
#include "server_socket_context/aicpu_ts_roce_server_socket_context.h"
#include "aicpu_ts_roce_reged_mem_mgr.h"

namespace hcomm {
struct AicpuTsNetDevSlot {
    HcclNetDev netDev{nullptr};
    uint32_t refCount{0U};
};

class AicpuTsRoceEndpoint : public Endpoint {
public:
    explicit AicpuTsRoceEndpoint(const EndpointDesc& endpointDesc);
    ~AicpuTsRoceEndpoint() override;

    HcclResult Init() override;

    RegedMemMgr* GetRegedMemMgr() override { return regedMemMgr_.get(); }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override;

    ServerSocketContext* GetServerSocketContext() override
    {
        return serverSocketContext_.has_value() ? &serverSocketContext_.value() : nullptr;
    }

    HcclNetDev GetNetDev() const { return netDev_; }

private:
    static std::unordered_map<uint32_t, AicpuTsNetDevSlot>& GetNetDevMap();
    static std::mutex& NetDevMapMutex();
    HcclResult AcquireSharedNetDev(uint32_t devicePhyId, const HcclNetDevInfos& info);
    void ReleaseSharedNetDev();
    void ReleaseNicSocketHandle(HcclNetDev netDev) const;
    HcclResult AcquireRdmaContext(uint32_t devPhyId, const EndpointDesc& endpointDesc);

    void* ctxHandle_{nullptr};
    std::shared_ptr<AicpuTsRoceRegedMemMgr> regedMemMgr_{};
    HcclNetDev netDev_{nullptr};
    uint32_t netDevRefPhyId_{UINT32_MAX};
    // Init 内 AcquireSharedNetDev 成功后构造
    std::optional<AicpuTsRoceServerSocketContext> serverSocketContext_{};
};
} // namespace hcomm
#endif // AICPUTS_ROCE_ENDPOINT_H
