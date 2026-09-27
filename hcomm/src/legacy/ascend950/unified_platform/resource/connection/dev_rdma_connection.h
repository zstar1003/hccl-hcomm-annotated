/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RMA_DEV_NET_CONNECTION_H
#define HCCLV2_RMA_DEV_NET_CONNECTION_H

#include "rma_connection.h"
#include "op_mode.h"
#include "orion_adapter_hccp.h"
#include "socket.h"
#include "task.h"
namespace Hccl {

class DevRdmaConnection : public RmaConnection {
public:
    DevRdmaConnection(Socket* socket, RdmaHandle rdmaHandle, OpMode opMode);
    void Connect() override;
    RmaConnStatus GetStatus() override;
    QpHandle GetHandle();

    unique_ptr<BaseTask>
    PrepareWrite(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override;

    ~DevRdmaConnection() override;

    string Describe() const override;

private:
    QpHandle qpHandle{nullptr};
    void CheckQpStatus(QpHandle handle);

    MAKE_ENUM(RdmaConnStatus, INIT, CONNECTING, SOCKET_TIMEOUT)
    RdmaConnStatus rdmaConnStatus{RdmaConnStatus::INIT};

    unique_ptr<BaseTask> PrepareOneRdmaSendForWrite(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, u64 offset, u32 size) const;
};

} // namespace Hccl

#endif // HCCL_RMA_DEV_NET_CONNECTION_H
