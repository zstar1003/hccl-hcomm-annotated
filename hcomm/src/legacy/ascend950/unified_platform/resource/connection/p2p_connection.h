/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RMA_P2P_CONNECTION_H
#define HCCLV2_RMA_P2P_CONNECTION_H

#include <map>
#include <string>

#include "virtual_topo.h"
#include "rma_connection.h"
#include "task.h"

namespace Hccl {

class P2PConnection : public RmaConnection {
public:
    P2PConnection(Socket* socket, const string& tag);
    ~P2PConnection() override = default;

    void Connect() override;
    RmaConnStatus GetStatus() override;
    string Describe() const override;

    unique_ptr<BaseTask>
    PrepareRead(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override;

    unique_ptr<BaseTask> PrepareReadReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        const SqeConfig& config) override;

    unique_ptr<BaseTask>
    PrepareWrite(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override;

    unique_ptr<BaseTask> PrepareWriteReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        const SqeConfig& config) override;

private:
    void EnableP2p() const;
};

} // namespace Hccl

#endif // HCCLV2_RMA_P2P_CONNECTION_H
