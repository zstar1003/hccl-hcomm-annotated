/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REMOTE_RDMA_RMA_BUFFER_IMPL_H
#define REMOTE_RDMA_RMA_BUFFER_IMPL_H

#include "hccl_common.h"
#include "hccl_network_pub.h"
#include "rma_buffer.h"
#include "mem_name_repository_pub.h"

namespace hccl {
class RemoteRdmaRmaBufferImpl : public RmaBuffer {
public:
    RemoteRdmaRmaBufferImpl();

    HcclResult Deserialize(const std::string& msg);

    ~RemoteRdmaRmaBufferImpl() override;

    RemoteRdmaRmaBufferImpl(const RemoteRdmaRmaBufferImpl& that) = delete;
    RemoteRdmaRmaBufferImpl& operator=(const RemoteRdmaRmaBufferImpl& that) = delete;

    inline u32 GetKey() const { return rkey; }

private:
    u32 rkey{0};
};
} // namespace hccl
#endif //  REMOTE_RDMA_RMA_BUFFER_IMPL_H
