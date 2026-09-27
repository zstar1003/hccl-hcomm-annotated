/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REMOTE_IPC_RMA_BUFFER_IMPL_H
#define REMOTE_IPC_RMA_BUFFER_IMPL_H

#include "hccl_common.h"
#include "hccl_network_pub.h"
#include "rma_buffer.h"
#include "mem_name_repository_pub.h"

namespace hccl {
class RemoteIpcRmaBufferImpl : public RmaBuffer {
public:
    explicit RemoteIpcRmaBufferImpl(const HcclNetDevCtx netDevCtx);

    HcclResult Deserialize(const std::string& msg);
    HcclResult Open();
    HcclResult Close();

    ~RemoteIpcRmaBufferImpl() override;

    RemoteIpcRmaBufferImpl(const RemoteIpcRmaBufferImpl& that) = delete;
    RemoteIpcRmaBufferImpl& operator=(const RemoteIpcRmaBufferImpl& that) = delete;

private:
    const HcclNetDevCtx netDevCtx{nullptr};
    SecIpcName_t memName;
    u64 memOffset{0};
};
} // namespace hccl
#endif //  REMOTE_IPC_RMA_BUFFER_IMPL_H
