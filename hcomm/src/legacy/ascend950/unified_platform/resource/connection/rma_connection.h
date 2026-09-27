/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RMA_CONNECTION_H
#define HCCLV2_RMA_CONNECTION_H
#include "task.h"
#include "virtual_topo.h"
#include "buffer_type.h"
#include "remote_rma_buffer.h"
#include "stream.h"
#include "serializable.h"
#include "socket.h"

namespace Hccl {

struct MemoryBuffer {
    u64 addr{0};
    u64 size{0};
    u64 memHandle{0};
    MemoryBuffer(u64 addr, u64 size, u64 memHandle) : addr(addr), size(size), memHandle(memHandle) {}
};

inline void VerifySizeIsEqual(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const string& desc)
{
    if (remoteMemBuf.size != localMemBuf.size) {
        string msg = StringFormat(
            "Check %s size is error, localMemBufSize[0x%llx], remoteMemBufSize[0x%llx]", desc.c_str(), localMemBuf.size,
            remoteMemBuf.size);
        THROW<InvalidParamsException>(msg);
    }
}

// EXCHANGEABLE状态用于指示UB Connection可与对端交换信息
MAKE_ENUM(RmaConnStatus, INIT, READY, SUSPENDED, CLOSE, CONN_INVALID, EXCHANGEABLE)
MAKE_ENUM(WqeMode, DB_SEND, DWQE, WRITE_VALUE)
MAKE_ENUM(RmaConnType, P2P, RDMA, UB, CCU)

class SqeConfig {
public:
    WqeMode wqeMode{WqeMode::DB_SEND};
};

class RmaConnection {
public:
    explicit RmaConnection(Socket* socket, const RmaConnType rmaConnType);
    virtual ~RmaConnection();

    RmaConnType GetRmaConnType() const { return rmaConnType; }

    virtual std::vector<char> GetUniqueId() const { MACRO_THROW(NotSupportException, StringFormat("not supported.")); }

    virtual void Connect() = 0;

    virtual void Close();

    virtual RmaConnStatus GetStatus();

    virtual string Describe() const = 0;

    virtual void Bind(RemoteRmaBuffer* remoteRmaBuf, BufferType bufType);

    virtual RemoteRmaBuffer* GetRemoteRmaBuffer(const BufferType& bufType);

    virtual unique_ptr<BaseTask>
    PrepareRead(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config);

    virtual unique_ptr<BaseTask> PrepareReadReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        const SqeConfig& config);

    virtual unique_ptr<BaseTask>
    PrepareWrite(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config);

    virtual unique_ptr<BaseTask> PrepareWriteReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        const SqeConfig& config);

    virtual unique_ptr<BaseTask>
    PrepareInlineWrite(const MemoryBuffer& remoteMemBuf, u64 data, const SqeConfig& config);

    virtual unique_ptr<BaseTask> PrepareWriteWithNotify(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, u64 data,
        const MemoryBuffer& remoteNotifyMemBuf, const SqeConfig& config);

    virtual unique_ptr<BaseTask> PrepareWriteReduceWithNotify(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        u64 data, const MemoryBuffer& remoteNotifyMemBuf, const SqeConfig& config);

    virtual void AddNop(const Stream& stream) { (void)stream; }

    virtual bool Suspend() { MACRO_THROW(NotSupportException, StringFormat("Resume is not supported.")); }

    virtual unique_ptr<Serializable> GetExchangeDto() // 序列化本地数据
    {
        MACRO_THROW(NotSupportException, StringFormat("not support."));
    }

    virtual void ParseRmtExchangeDto(const Serializable& rmtDto) // 解析收到得远端序列化数据
    {
        (void)rmtDto;
        MACRO_THROW(NotSupportException, StringFormat("not support."));
    }

    virtual void ImportRmtDto() // 导入远端的数据
    {
        MACRO_THROW(NotSupportException, StringFormat("not support."));
    }

    virtual HcclResult Describe(std::string& dfxMsg)
    {
        (void)dfxMsg;
        HCCL_ERROR("[RmaConnection::%s] not support.", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

protected:
    RmaConnStatus status;
    Socket* socket{nullptr};
    RmaConnType rmaConnType;

    unordered_map<BufferType, RemoteRmaBuffer*, std::EnumClassHash> remoteBufs;
};

} // namespace Hccl

#endif // HCCLV2_RMA_CONNECTION_H
