/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_HETEROG_PUB_H
#define TRANSPORT_HETEROG_PUB_H
#include <unordered_map>
#include <atomic>
#include <functional>
#include <vector>
#include <queue>
#include <stack>
#include "memory_alloc_ring.h"
#include "private_types.h"
#include "local_ipc_notify.h"

namespace hccl {
enum class ConnState {
    CONN_STATE_IDLE,
    CONN_STATE_CONNECT_CHECK_SOCKET,
    CONN_STATE_GET_CHECK_SOCKET,
    CONN_STATE_SEND_CF,
    CONN_STATE_RECV_CF,
    CONN_STATE_CHECK_CF,
    CONN_STATE_CONNECT_ALL_SOCKET,
    CONN_STATE_GET_ALL_SOCKET,
    CONN_STATE_CONNECT_QP,
    CONN_STATE_GET_QP,
    CONN_STATE_SEND_STATUS,
    CONN_STATE_RECV_STATUS,
    CONN_STATE_FLUSH_QUEUE,
    CONN_STATE_GET_TAG_QP_ATTR,
    CONN_STATE_SEND_TAG_QP_INFO,
    CONN_STATE_RECV_TAG_QP_INFO,
    CONN_STATE_MODIFY_TAG_QP,
    CONN_STATE_GET_DATA_QP_ATTR,
    CONN_STATE_SEND_DATA_QP_INFO,
    CONN_STATE_RECV_DATA_QP_INFO,
    CONN_STATE_MODIFY_DATA_QP,
    CONN_STATE_COMPLETE
};

enum class RdmaNotifyOp { SEND_NOTIFY, RECV_NOTIFY, NUM };
constexpr u32 HETEROG_MAX_FRAME_LEN = 128;
struct InitInfo {
    s32 protocolType = 0; // 0:ROCE; 1:TCP
    u32 role = 0;
    u32 signal = 0;
    u8 checkFrame[HETEROG_MAX_FRAME_LEN] = {0};
    std::vector<SocketInfoT> socketInfo;
    std::vector<SocketConnectInfoT> socketConnInfo;
};

struct InitStateMachine {
    InitInfo locInitInfo;
    InitInfo remInitInfo;
    u64 size = 0;
    u64 completeSize = 0;
    u32 socketNum = 0;
    u32 completeNum = 0;
};

static constexpr u32 SYNC_SIGNAL = 0xFFFFFFFF;
constexpr u32 HCCL_POLL_CQ_DEPTH = 32;

struct TransportEndPointInfoHash {
    std::size_t operator()(const TransportEndPointInfo& t) const
    {
        return std::hash<u32>()(t.commId) ^ std::hash<u32>()(t.rank) ^ std::hash<u32>()(t.tag);
    }
};

using HcclReceivedEnvelope
    = std::unordered_map<TransportEndPointInfo, std::queue<HcclEnvelopeSummary>, TransportEndPointInfoHash>;

class TransportHeterog {
public:
    explicit TransportHeterog(
        const std::string& tag, HcclIpAddress& selfIp, HcclIpAddress& peerIp, u32 peerPort, u32 selfPort,
        const TransportResourceInfo& transportResourceInfo);
    explicit TransportHeterog(const TransportResourceInfo& transportResourceInfo);
    virtual ~TransportHeterog();
    virtual HcclResult Init() = 0;
    virtual HcclResult Init(u32 localUserRank, u32 remoteUserRank);
    virtual HcclResult Init(SocketInfoT& socketInfo, RdmaHandle rdmaHandle, MrHandle mrHandle);
    virtual HcclResult Deinit() = 0;
    virtual HcclResult
    Isend(const TransData& sendData, const TransportEndPointParam& epParam, HcclRequestInfo*& request)
        = 0;
    virtual HcclResult Send(const TransData& sendData, const TransportEndPointParam& epParam) = 0;
    virtual HcclResult
    Improbe(const TransportEndPointParam& epParam, s32& matched, HcclMessageInfo*& msg, HcclStatus& status)
        = 0;
    virtual HcclResult Imrecv(const TransData& recvData, HcclMessageInfo& msg, HcclRequestInfo*& request) = 0;
    virtual HcclResult Test(HcclRequestInfo& request, s32& flag, HcclStatus& compState) = 0;
    virtual HcclResult
    Improbe(const TransportEndPointParam& epParam, s32& matched, HcclMessageInfo*& msg, HcclStatus& status, bool& flag);
    virtual HcclResult
    Imrecv(const TransData& recvData, HcclMessageInfo& msg, HcclRequestInfo*& request, bool flag, bool needRecordFlag);
    virtual HcclResult ImrecvScatter(
        void* buf[], int count[], int bufCount, HcclDataType datatype, HcclMessageInfo& msg, HcclRequestInfo*& request);
    HcclResult SetDeviceIndex(s32 index);
    u32 GetRecvEnvelopNum();
    void AddRecvEnvelopNum();
    void SubRecvEnvelopNum();
    virtual HcclResult BlockSend(
        const TransData& sendData, const TransportEndPointParam& epParam, HcclRequestInfo*& request, s32 waitTimeOut);
    virtual HcclResult BlockRecv(
        const TransData& recvData, bool matched, TransportHeterog*& transport, s32 waitTimeOut, s32 waitPayloadTimeOut);
    HcclResult CheckAndPushBuildLink();
    HcclResult WaitBuildLinkComplete();
    virtual HcclResult Iwrite(const TransData& sendData, const HcclEnvelope& envelope, HcclRequestInfo*& request);
    virtual HcclResult GetRemoteIsendDoneSignal(std::shared_ptr<LocalIpcNotify>& signal);
    virtual HcclResult GetRemoteImrecvDoneSignal(std::shared_ptr<LocalIpcNotify>& signal);

    ConnState GetState();
    virtual void GetLinkTag(std::string& tag);
    void SetForceClose();
    HcclResult SocketSend(const FdHandle fdHandle, void* data, u64 size, u64& sentSize, bool& completed);
    HcclResult SocketRecv(const FdHandle fdHandle, void* data, u64 size, u64& recvSize, bool& completed);
    static void RecordRankTableCrc(const u32 crcValue);

protected:
    HcclResult CheckRecvMsgAndRequestBuffer();
    HcclResult
    GenerateSendRequest(const TransData& sendData, const TransportEndPointParam& epParam, HcclRequestInfo*& request);
    HcclResult GenerateRecvRequest(const TransData& recvData, const HcclMessageInfo& msg, HcclRequestInfo*& request);
    HcclResult GenerateRecvScatterRequest(const HcclMessageInfo& msg, HcclRequestInfo*& request);
    HcclResult FreeRequest(HcclRequestInfo& request) const;
    HcclResult
    CheckTransportEndPointInfo(const TransportEndPointInfo& epInfo, const TransportEndPointInfo& epInfoCheck) const;
    HcclResult CheckRecvEnvelope(const TransData& recvDataCheck, const HcclEnvelopeSummary& envelope);
    HcclResult CheckRecvScatterEnvelope(
        void* buf[], int count[], int bufCount, HcclDataType datatype, const HcclEnvelopeSummary& envelope);
    HcclResult GenerateRecvMessage(HcclEnvelopeSummary& recvEnvelope, HcclMessageInfo*& msg, HcclStatus& status);
    HcclResult FreeRecvMessage(HcclMessageInfo& msg) const;
    HcclResult ProbeNothing(s32& flag, HcclMessageInfo*& msg, HcclStatus& status) const;
    HcclResult ConnectSocket(SocketConnectInfoT conn[], u32 num, bool& completed);
    HcclResult GetSocket(u32 role, struct SocketInfoT info[], u32 num, u32& connectedNum, bool& completed);
    HcclResult SocketClose();
    HcclResult CheckConsistentFrame();
    HcclResult ConnectAsync();
    HcclResult PrepareSocketInfo(s32 type, s32 linkNum, const std::string& clientTag, const std::string& serverTag);
    HcclResult InitTransportConnect(s32 type, s32 linkNum);
    HcclResult InitTransportConnect(s32 type, u32 role, s32 linkNum, u32 tag);
    HcclResult AddSocketWhiteList(std::string& tag);
    HcclResult TryTransition(HcclResult ret, bool completed, ConnState nextState);
    virtual HcclResult EnterStateProcess(ConnState nextState) = 0;
    virtual HcclResult LoopStateProcess() = 0;

    const std::string transTag_;
    SocketHandle nicSocketHandle_;
    HcclIpAddress selfIp_;
    HcclIpAddress peerIp_;
    u32 peerPort_;
    u32 selfPort_;
    struct InitStateMachine initSM_ {};
    std::atomic<ConnState> connState_{ConnState::CONN_STATE_IDLE};
    const std::unique_ptr<LocklessRingMemoryAllocate<HcclMessageInfo>>& pMsgInfosMem_;
    const std::unique_ptr<LocklessRingMemoryAllocate<HcclRequestInfo>>& pReqInfosMem_;
    s32 index_ = 0;
    u32 recvEnvelopNum_{0};
    bool isHdcMode_ = false;
    u32 localRank_ = 0;
    u32 remoteRank_ = 0;
    bool remoteIsHdc_ = false; // 连接对端为310时，为false
    bool isESMode_ = false;
    bool forceClose_ = false; // 设置socket batch close时是否为强制关闭，而非超时“优雅”关闭

    static std::atomic<u32> rankTableCrc_;
};
} // namespace hccl
#endif
