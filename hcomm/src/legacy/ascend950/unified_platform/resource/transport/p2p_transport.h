/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef P2P_TRANSPORT_H
#define P2P_TRANSPORT_H

#include "base_mem_transport.h"
#include "virtual_topo.h"
#include "ipc_remote_notify.h"
#include "remote_rma_buffer.h"

namespace Hccl {
class P2PTransport : public BaseMemTransport {
public:
    P2PTransport(CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket);

    P2PTransport(
        CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket,
        std::function<void(u32 streamId, u32 taskId, TaskParam taskParam)> callback);

    ~P2PTransport() override = default;

    std::string Describe() const override;

    TransportStatus GetStatus() override;

    std::vector<char> GetUniqueId() override;

    std::vector<char> GetUniqueIdV2();

    void Post(u32 index, const Stream& stream) override;

    void Read(const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const Stream& stream) override;

    void ReadReduce(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& reduceIn,
        const Stream& stream) override;

    void Write(const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const Stream& stream) override;

    void WriteReduce(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& reduceIn,
        const Stream& stream) override;

    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos);

private:
    MemoryBuffer GetLocMemBuffer(const RmaBufferSlice& locSlice) const;
    MemoryBuffer GetRmtMemBuffer(const RmtRmaBufferSlice& rmtSlice) const;

    MAKE_ENUM(
        P2PStatus, INIT, SOCKET_OK, SEND_PID, RECV_PID, GRANT, SEND_DATA, RECV_DATA, SEND_DATA_SIZE, RECV_DATA_SIZE)
    P2PStatus p2pStatus{P2PStatus::INIT};

    u32 pidMsgSize{0};
    u32 myPid{0};
    u32 rmtPid{0};

    bool rmtPidValid{false};

    std::vector<std::unique_ptr<IpcRemoteNotify>> rmtNotifyVec;
    std::vector<std::unique_ptr<RemoteIpcRmaBuffer>> rmtBufferVec;

    bool cacheValid_ = false;                // 当前缓存是否有效
    std::vector<CommMem> remoteUserMems_;    // 内存基本信息缓存
    std::vector<std::string> memInfoCopies_; // 储存 Tag 字符串副本
    std::vector<char*> memInfoPointers_;     // Tag 缓存
    std::vector<char> sendData;
    std::vector<char> recvData;

    bool IsRmtPidValid() const;
    void SendPid();
    void RecvPid();
    void Grant() const;
    void PrepareSendData();
    void RecvDataSize();
    void SendExchangeData();
    void RecvExchangeData();
    void ProcessRecvData();

    void BufferVecPack(BinaryStream& binaryStream);

    void RmtNotifyVecUnpackProc(BinaryStream& binaryStream);
    void RmtBufferVecUnpackProc(BinaryStream& binaryStream);

    std::vector<char> GetSingleRmtNotifyUniqueId(u64 addr, u64 size, u32 notifyId) const;
    std::vector<char> GetSingleBufferUniqueId(u64 addr, u64 size) const;
    std::vector<char> GetNotifyUniqueIds();
    std::vector<char> GetRmtNotifyUniqueIds() const;
    std::vector<char> GetLocBufferUniqueIds() const;
    std::vector<char> GetRmtBufferUniqueIds() const;

    std::mutex remoteMemsMutex_; // 远端内存列表互斥锁
};

} // namespace Hccl

#endif
