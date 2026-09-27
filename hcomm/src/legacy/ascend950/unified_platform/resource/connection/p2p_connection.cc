/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "p2p_connection.h"
#include "socket.h"
#include "log.h"
#include "exception_util.h"
#include "rma_conn_exception.h"

namespace Hccl {

P2PConnection::P2PConnection(Socket* socket, const std::string& tag) : RmaConnection(socket, RmaConnType::P2P)
{
    HCCL_INFO("P2PConnection::P2PConnection tag = [%s]", tag.c_str());
}

void P2PConnection::Connect()
{
    EnableP2p();
    GetStatus();
}

RmaConnStatus P2PConnection::GetStatus()
{
    switch (status) {
        case RmaConnStatus::READY:
            break;
        case RmaConnStatus::INIT:
            if (socket->GetStatus() == SocketStatus::OK) {
                status = RmaConnStatus::READY;
            } else if (socket->GetStatus() == SocketStatus::TIMEOUT) {
                status = RmaConnStatus::CONN_INVALID;
            }
            break;
        case RmaConnStatus::CLOSE:
            break;
        default:
            auto msg = StringFormat("Invalid status of %s", status.Describe().c_str());
            THROW<RmaConnException>(msg);
            break;
    }

    return status;
}

void P2PConnection::EnableP2p() const
{
    // SDMA P2pEnable
}

unique_ptr<BaseTask> P2PConnection::PrepareRead(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, [[maybe_unused]] const SqeConfig& config)
{
    VerifySizeIsEqual(remoteMemBuf, localMemBuf, "P2PConnection PrepareRead");
    if (localMemBuf.size == 0) {
        return nullptr;
    }
    return make_unique<TaskP2pMemcpy>(localMemBuf.addr, remoteMemBuf.addr, localMemBuf.size, MemcpyKind::D2D);
}

unique_ptr<BaseTask> P2PConnection::PrepareReadReduce(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
    [[maybe_unused]] const SqeConfig& config)
{
    VerifySizeIsEqual(remoteMemBuf, localMemBuf, "P2PConnection PrepareReadReduce");
    if (localMemBuf.size == 0) {
        return nullptr;
    }
    return make_unique<TaskSdmaReduce>(localMemBuf.addr, remoteMemBuf.addr, localMemBuf.size, datatype, reduceOp);
}

unique_ptr<BaseTask> P2PConnection::PrepareWrite(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, [[maybe_unused]] const SqeConfig& config)
{
    VerifySizeIsEqual(remoteMemBuf, localMemBuf, "P2PConnection PrepareWrite");
    if (localMemBuf.size == 0) {
        return nullptr;
    }
    return make_unique<TaskP2pMemcpy>(remoteMemBuf.addr, localMemBuf.addr, localMemBuf.size, MemcpyKind::D2D);
}

unique_ptr<BaseTask> P2PConnection::PrepareWriteReduce(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
    [[maybe_unused]] const SqeConfig& config)
{
    VerifySizeIsEqual(remoteMemBuf, localMemBuf, "P2PConnection PrepareWriteReduce");
    if (localMemBuf.size == 0) {
        return nullptr;
    }
    return make_unique<TaskSdmaReduce>(remoteMemBuf.addr, localMemBuf.addr, localMemBuf.size, datatype, reduceOp);
}

string P2PConnection::Describe() const { return StringFormat("P2PConnection[status=%s]", status.Describe().c_str()); }

} // namespace Hccl
