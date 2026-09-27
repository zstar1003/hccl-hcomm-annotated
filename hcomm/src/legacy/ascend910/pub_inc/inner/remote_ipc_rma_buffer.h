/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REMOTE_IPC_RMA_BUFFER_H
#define REMOTE_IPC_RMA_BUFFER_H

#include <memory>
#include <string>
#include "hccl_common.h"
#include "hccl_network_pub.h"
#include "rma_buffer.h"

namespace hccl {
class RemoteIpcRmaBufferImpl;

class RemoteIpcRmaBuffer : public RmaBuffer {
public:
    RemoteIpcRmaBuffer(const HcclNetDevCtx netDevCtx);
    HcclResult Deserialize(const std::string&);
    HcclResult Open();
    HcclResult Close();
    ~RemoteIpcRmaBuffer() override;

    RemoteIpcRmaBuffer(const RemoteIpcRmaBuffer& that) = delete;
    RemoteIpcRmaBuffer& operator=(const RemoteIpcRmaBuffer& that) = delete;
    bool IsOpened() const { return isOpened_; }

private:
    std::unique_ptr<RemoteIpcRmaBufferImpl> pimpl_;
    bool isOpened_{false};
};
} // namespace hccl
#endif //  REMOTE_IPC_RMA_BUFFER_H
