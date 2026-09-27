/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UB_MEMORY_TRANSPORT_H
#define UB_MEMORY_TRANSPORT_H
#include "virtual_topo.h"
#include "dev_buffer.h"
#include "socket.h"
#include "local_ipc_rma_buffer_v2.h"
#include "remote_rma_buffer.h"
#include "base_mem_transport.h"

namespace Hccl {
class UbMemoryTransport {
public:
    MAKE_ENUM(
        UBTransportStatus, INIT, SOCKET_OK, SEND_MEM_INFO, RECV_MEM_INFO, RECV_MEM_INFO_PROCESS, SEND_NAME, RECV_NAME,
        CONNECT_FAILED, SOCKET_TIMEOUT, READY)

    UbMemoryTransport(
        const std::shared_ptr<Buffer> cclBuffer, const std::shared_ptr<Buffer> aivTagBuffer,
        const std::shared_ptr<Buffer> aivOffloadTagBuffer, Socket* socket, int32_t deviceLogicId);

    UbMemoryTransport(const UbMemoryTransport& that) = delete;
    UbMemoryTransport& operator=(const UbMemoryTransport& other) = delete;
    HcclResult Init();
    UBTransportStatus GetStatus();

    // 给算子开发提供的两个接口
    LocalIpcRmaBuffer* GetLocMemBuffer(const u32 bufIndex) const;
    RemoteIpcRmaBuffer* GetRmtMemBuffer(const u32 bufIndex) const;
    std::string Describe() const;

    vector<char> GetRmtHandshakeMsg() // 返回握手消息
    {
        return rmtHandshakeMsg;
    }

    vector<char> GetLocalHandshakeMsg() // 返回握手消息
    {
        return localHandshakeMsg;
    }

    void SetHandshakeMsg(const vector<char>& handshakeMsg) { localHandshakeMsg = handshakeMsg; }

    AcceleratorState& GetRmtOpAcceState() { return rmtOpAcceState; }

    AcceleratorState& GetLocalOpAcceState() { return locOpAcceState; }

    void SetLocalOpAcceState(const AcceleratorState& opAcceState) { locOpAcceState = opAcceState; }

    struct CclBufferInfo {
        uint64_t addr{0};
        uint32_t size{0};
        uint32_t tokenId{0};
        uint32_t tokenValue{0};

        void Pack(BinaryStream& binaryStream) const
        {
            binaryStream << addr << size << tokenId << tokenValue;
            HCCL_INFO("Pack Ccl Buffer Info: addr[%llu] size[%u]", addr, size);
        }

        void Unpack(BinaryStream& binaryStream)
        {
            binaryStream >> addr >> size >> tokenId >> tokenValue;
            HCCL_INFO("Unpack Ccl Buffer Info: addr[%llu] size[%u]", addr, size);
        }
    };

private:
    UBTransportStatus ubStatus{UBTransportStatus::INIT};
    vector<char> rmtHandshakeMsg{0}; // 远端握手消息
    vector<char> localHandshakeMsg{0};
    vector<char> recvDataMsg{0};
    AcceleratorState rmtOpAcceState{AcceleratorState::AIV};
    AcceleratorState locOpAcceState{AcceleratorState::AIV};

    CclBufferInfo locCclBufInfo;
    CclBufferInfo rmtCclBufInfo;
    uint32_t exchangeDataSize{0};

    //  新增
    std::shared_ptr<Buffer> cclBuffer;
    std::shared_ptr<Buffer> aivTagBuffer;
    std::shared_ptr<Buffer> aivOffloadTagBuffer;
    Socket* socket{};
    int32_t deviceLogicId{0};

    std::vector<std::unique_ptr<RemoteIpcRmaBuffer>> rmtBufferVec;
    std::vector<std::unique_ptr<LocalIpcRmaBuffer>> localBufferVec;
    std::vector<RemoteRmaBuffer*> rmtRmaBufferVec;

    void HandshakeMsgPack(BinaryStream& binaryStream);
    void HandshakeMsgUnpack(BinaryStream& binaryStream);
    UBTransportStatus StateMachine();
    void ReleaseRes();
    void SendMemInfo();
    void RecvMemInfo();
    void RecvMemProcess();
    void BufferPack(BinaryStream& binaryStream);
    void RmtBufferUnpackProc(BinaryStream& binaryStream);
    void SendName();
    void RecvName();
};
} // namespace Hccl

#endif
