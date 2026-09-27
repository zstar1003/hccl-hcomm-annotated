/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_MEM_TRANSPORT_H
#define BASE_MEM_TRANSPORT_H

#include <memory>
#include <unordered_map>

#include "task.h"
#include "local_rma_buffer.h"
#include "remote_rma_buffer.h"
#include "../../resource/connection/rma_connection.h"
#include "local_notify_v2.h"
#include "ipc_remote_notify.h"
#include "local_cnt_notify.h"
#include "op_mode.h"
#include "mem_transport_common.h"
#include "task_param.h"
#include "transport_status.h"
#include "socket.h"
#include "virtual_topo.h"

namespace Hccl {

struct RmaBufferSlice {
    u64 addr;
    u64 size;
    LocalRmaBuffer* buf;
    std::string Describe() const
    {
        if (buf == nullptr) {
            return StringFormat("RmaBufferSlice[addr=0x%llx, size=0x%llx, buf is null]", addr, size);
        } else {
            return StringFormat(
                "RmaBufferSlice[addr=0x%llx, size=0x%llx, buf=%s]", addr, size, buf->Describe().c_str());
        }
    }
};

struct RmtRmaBufferSlice {
    u64 addr;
    u64 size;
    RemoteRmaBuffer* buf;
    std::string Describe() const
    {
        if (buf == nullptr) {
            return StringFormat("RmtRmaBufferSlice=[addr=0x%llx, size=0x%llx, buf is null]", addr, size);
        } else {
            return StringFormat(
                "RmtRmaBufferSlice=[addr=0x%llx, size=0x%llx, buf=%s]", addr, size, buf->Describe().c_str());
        }
    }
};

class BaseMemTransport {
public:
    struct CommonLocRes {
        std::vector<BaseLocalNotify*> notifyVec;
        std::vector<LocalRmaBuffer*> bufferVec;
        std::vector<RmaConnection*> connVec;
        string Describe() const
        {
            string msg = StringFormat(
                "MemTransportCommonLocRes=[notifyNum=%zu, bufferNum=%zu, connNum=%zu]", notifyVec.size(),
                bufferVec.size(), connVec.size());
            return msg;
        }
    };

    struct LocCntNotifyRes {
        vector<LocalCntNotify*> vec{};
        vector<char> desc{}; // 将 topicId + index 映射到 index的关系交换对端

        std::string Describe() const
        {
            string msg = StringFormat(
                "LocCntNotifyRes[cntNotifyNum=%zu], desc=%s", vec.size(), Bytes2hex(desc.data(), desc.size()).c_str());
            return msg;
        }
    };

    struct Attribution {
        OpMode opMode;
        u32 devicePhyId{0};
        vector<char> handshakeMsg{};
        AcceleratorState opAcceState{AcceleratorState::AICPU_TS};
        string Describe() const

        {
            return StringFormat(
                "MemTransportAttribution[opMode=%s, devicePhyId=%u, handleshakeMsg=%s]", opMode.Describe().c_str(),
                devicePhyId, Bytes2hex(handshakeMsg.data(), handshakeMsg.size()).c_str());
        }
    };
    BaseMemTransport(
        CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket,
        TransportType type);

    BaseMemTransport(
        CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket,
        TransportType type, std::function<void(u32 streamId, u32 taskId, TaskParam taskParam)> callback);

    virtual ~BaseMemTransport() = default;

    virtual vector<char>& GetRmtHandshakeMsg() // 返回握手消息
    {
        return rmtHandshakeMsg;
    }

    AcceleratorState& GetRmtOpAcceState() { return rmtOpAcceState; }

    virtual std::string Describe() const = 0;

    virtual void Establish();

    virtual TransportStatus GetStatus() { return TransportStatus::READY; }

    virtual std::vector<char> GetUniqueId() { MACRO_THROW(NotSupportException, StringFormat("not supported.")); }

    virtual RemoteRmaBuffer* GetRmtRmaBuffer(u32 index)
    {
        if (index >= rmtRmaBufferVec.size()) {
            MACRO_THROW(
                InvalidParamsException, StringFormat("Get remote rmaBuffer fail, index[%u] is not in range.", index));
        }
        return rmtRmaBufferVec[index];
    }

    virtual void SetConnVec(std::vector<RmaConnection*>& connVec)
    {
        (void)connVec;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual vector<char>& GetRmtCntNotifyDesc() { MACRO_THROW(NotSupportException, StringFormat("not supported.")); }

    virtual void Post(u32 index, const Stream& stream)
    {
        (void)index;
        (void)stream;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void Wait(u32 index, const Stream& stream, u32 timeout)
    {
        (void)index;
        (void)stream;
        (void)timeout;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void Read(const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const Stream& stream)
    {
        (void)locSlice;
        (void)rmtSlice;
        (void)stream;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void ReadReduce(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& reduceIn,
        const Stream& stream)
    {
        (void)locSlice;
        (void)rmtSlice;
        (void)reduceIn;
        (void)stream;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void Write(const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const Stream& stream)
    {
        (void)locSlice;
        (void)rmtSlice;
        (void)stream;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void WriteReduce(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& reduceIn,
        const Stream& stream)
    {
        (void)locSlice;
        (void)rmtSlice;
        (void)reduceIn;
        (void)stream;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void WriteWithNotify(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const WithNotifyIn& withNotify,
        const Stream& stream)
    {
        (void)locSlice;
        (void)rmtSlice;
        (void)withNotify;
        (void)stream;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void WriteReduceWithNotify(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& reduceIn,
        const WithNotifyIn& withNotify, const Stream& stream)
    {
        (void)locSlice;
        (void)rmtSlice;
        (void)reduceIn;
        (void)withNotify;
        (void)stream;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual vector<char>& GetLocalHandshakeMsg() // 返回本端握手消息
    {
        return attr.handshakeMsg;
    }

    AcceleratorState& GetLocalOpAcceState() { return attr.opAcceState; }

    void SetLocalOpAcceState(const AcceleratorState& opAcceState) { attr.opAcceState = opAcceState; }

    void SetIsHost() { isHost_ = true; }

    string GetLinkDescInfo();
    string DescribeSocket() const;

protected:
    CommonLocRes commonLocRes{};
    Attribution attr;
    LinkData linkData;
    Socket* socket{};
    TransportType transportType;
    std::function<void(u32 streamId, u32 taskId, const TaskParam& taskParam)> callback;

    std::vector<RemoteRmaBuffer*> rmtRmaBufferVec;

    TransportStatus baseStatus{TransportStatus::INIT};

    vector<char> rmtHandshakeMsg{0}; // 远端握手消息
    AcceleratorState rmtOpAcceState{AcceleratorState::AICPU_TS};

    u32 notifyNum{0};
    u32 bufferNum{0};
    u32 connNum{0};
    u32 exchangeDataSize{0}; // 交换的消息大小
    bool isHost_{false};

    void SetBaseStatusReady();

    bool IsSocketReady();

    void NotifyVecPack(BinaryStream& binaryStream);

    void ConnVecPack(BinaryStream& binaryStream);

    void HandshakeMsgPack(BinaryStream& binaryStream);

    HcclResult HandshakeMsgUnpack(BinaryStream& binaryStream);

private:
    HcclResult CheckLocNotify(CommonLocRes& res);

    void CheckLocBuffer(CommonLocRes& res);

    HcclResult CheckLocConn(CommonLocRes& res);

    HcclResult CheckCommonLocRes(CommonLocRes& res);
};

} // namespace Hccl
#endif
