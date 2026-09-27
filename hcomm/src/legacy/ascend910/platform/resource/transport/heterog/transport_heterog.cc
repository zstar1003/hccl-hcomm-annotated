/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_heterog.h"
#include "log.h"
#include "network_manager_pub.h"
#include "hccl_socket.h"
#include "externalinput_pub.h"

using namespace std;
namespace hccl {
constexpr u32 SINGLE_WHITE_LIST_NUM = 1;
constexpr u32 WAIT_LINK_BUILD_DELAY_TIME_US = 10;
constexpr s32 MAX_LINK_NUM = 10;

std::atomic<u32> TransportHeterog::rankTableCrc_ = {0};
TransportHeterog::TransportHeterog(
    const string& tag, HcclIpAddress& selfIp, HcclIpAddress& peerIp, u32 peerPort, u32 selfPort,
    const TransportResourceInfo& transportResourceInfo)
    : transTag_(tag),
      nicSocketHandle_(nullptr),
      selfIp_(selfIp),
      peerIp_(peerIp),
      peerPort_(peerPort),
      selfPort_(selfPort),
      pMsgInfosMem_(transportResourceInfo.pMsgInfosMem),
      pReqInfosMem_(transportResourceInfo.pReqInfosMem),
      recvEnvelopNum_(0)
{}
TransportHeterog::TransportHeterog(const TransportResourceInfo& transportResourceInfo)
    : nicSocketHandle_(nullptr),
      peerPort_(0),
      selfPort_(0),
      pMsgInfosMem_(transportResourceInfo.pMsgInfosMem),
      pReqInfosMem_(transportResourceInfo.pReqInfosMem),
      recvEnvelopNum_(0)
{}

TransportHeterog::~TransportHeterog() {}

HcclResult TransportHeterog::Init([[maybe_unused]] u32 localUserRank, [[maybe_unused]] u32 remoteUserRank)
{
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::Init(
    [[maybe_unused]] SocketInfoT& socketInfo, [[maybe_unused]] RdmaHandle rdmaHandle,
    [[maybe_unused]] MrHandle mrHandle)
{
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::Improbe(
    [[maybe_unused]] const TransportEndPointParam& epParam, [[maybe_unused]] s32& matched,
    [[maybe_unused]] HcclMessageInfo*& msg, [[maybe_unused]] HcclStatus& status, [[maybe_unused]] bool& flag)
{
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::Imrecv(
    [[maybe_unused]] const TransData& recvData, [[maybe_unused]] HcclMessageInfo& msg,
    [[maybe_unused]] HcclRequestInfo*& request, [[maybe_unused]] bool flag, [[maybe_unused]] bool needRecordFlag)
{
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::ImrecvScatter(
    [[maybe_unused]] void* buf[], [[maybe_unused]] int count[], [[maybe_unused]] int bufCount,
    [[maybe_unused]] HcclDataType datatype, [[maybe_unused]] HcclMessageInfo& msg,
    [[maybe_unused]] HcclRequestInfo*& request)
{
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::CheckRecvMsgAndRequestBuffer()
{
    CHK_SMART_PTR_NULL(pMsgInfosMem_);
    CHK_SMART_PTR_NULL(pReqInfosMem_);
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::GenerateSendRequest(
    const TransData& sendData, const TransportEndPointParam& epParam, HcclRequestInfo*& request)
{
    request = pReqInfosMem_->Alloc();
    CHK_PTR_NULL(request);
    request->transportHandle = this;
    request->transportRequest.transData = sendData;
    request->transportRequest.epParam = epParam;
    request->transportRequest.requestType = HcclRequestType::HCCL_REQUEST_SEND;
    request->transportRequest.protocol = 0;
    request->transportRequest.msn = reinterpret_cast<u64>(request);
    request->transportRequest.status = -1;
    request->transportRequest.envoffset = 0;
    request->transportRequest.tranoffset = 0;

    return HCCL_SUCCESS;
}

HcclResult
TransportHeterog::GenerateRecvRequest(const TransData& recvData, const HcclMessageInfo& msg, HcclRequestInfo*& request)
{
    request = pReqInfosMem_->Alloc();
    CHK_PTR_NULL(request);
    request->transportHandle = this;
    request->transportRequest.transData = recvData;
    request->transportRequest.transData.srcBuf = msg.envelope.envelope.transData.srcBuf;
    request->transportRequest.epParam = msg.envelope.envelope.epParam;
    request->transportRequest.requestType = HcclRequestType::HCCL_REQUEST_RECV;
    request->transportRequest.protocol = msg.envelope.envelope.protocol;
    request->transportRequest.msn = msg.envelope.envelope.msn;
    request->transportRequest.status = -1;
    request->transportRequest.envoffset = 0;
    request->transportRequest.tranoffset = 0;

    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::GenerateRecvScatterRequest(const HcclMessageInfo& msg, HcclRequestInfo*& request)
{
    request = pReqInfosMem_->Alloc();
    CHK_PTR_NULL(request);
    request->transportHandle = this;
    request->transportRequest.transData.srcBuf = msg.envelope.envelope.transData.srcBuf;
    request->transportRequest.transData.count = 0;
    request->transportRequest.epParam = msg.envelope.envelope.epParam;
    request->transportRequest.requestType = HcclRequestType::HCCL_REQUEST_RECV;
    request->transportRequest.protocol = msg.envelope.envelope.protocol;
    request->transportRequest.msn = msg.envelope.envelope.msn;
    request->transportRequest.status = -1;
    request->transportRequest.envoffset = 0;
    request->transportRequest.tranoffset = 0;

    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::FreeRequest(HcclRequestInfo& request) const
{
    request.commHandle = nullptr;
    request.transportHandle = nullptr;
    CHK_RET(pReqInfosMem_->Free(&request));
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::CheckTransportEndPointInfo(
    const TransportEndPointInfo& epInfo, const TransportEndPointInfo& epInfoCheck) const
{
    if (epInfo.commId != epInfoCheck.commId) {
        HCCL_ERROR(
            "[Check][Tag]errNo[0x%016llx] commId[%u] is invalid, expect:%u", HCCL_ERROR_CODE(HCCL_E_PARA),
            epInfo.commId, epInfoCheck.commId);
        return HCCL_E_PARA;
    }

    if (epInfo.tag != epInfoCheck.tag) {
        HCCL_ERROR(
            "[Check][Tag]errNo[0x%016llx] tag[%u] is invalid, expect:%u", HCCL_ERROR_CODE(HCCL_E_PARA), epInfo.tag,
            epInfoCheck.tag);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::CheckRecvEnvelope(const TransData& recvDataCheck, const HcclEnvelopeSummary& envelope)
{
    if (envelope.status != 0) {
        HCCL_ERROR("[Check][EnvelopeStatus] envelope status:[%u] is invalid", envelope.status);
        return HCCL_E_PARA;
    }

    CHK_RET(CheckTransportEndPointInfo(envelope.envelope.epParam.src, envelope.envelope.epParam.dst));

    if (recvDataCheck.count < envelope.envelope.transData.count) {
        HCCL_ERROR(
            "[Check][RecvEnvelope]Imrecv input count[%llu] should be not less than Isend count[%llu]",
            recvDataCheck.count, envelope.envelope.transData.count);
        return HCCL_E_PARA;
    }

    if ((recvDataCheck.dstBuf == 0) && ((recvDataCheck.count != 0) || (envelope.envelope.transData.count != 0))) {
        HCCL_ERROR(
            "[Check][RecvEnvelope]Imrecv buffer[%p] or count[%llu] is invalid", recvDataCheck.dstBuf,
            envelope.envelope.transData.count);
        return HCCL_E_PARA;
    }

    if (recvDataCheck.dataType != envelope.envelope.transData.dataType) {
        HCCL_ERROR(
            "[Check][RecvEnvelope]Imrecv input dataType[%s] should be Isend dataType[%s]",
            GetDataTypeEnumStr(recvDataCheck.dataType).c_str(),
            GetDataTypeEnumStr(envelope.envelope.transData.dataType).c_str());
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::CheckRecvScatterEnvelope(
    void* buf[], int count[], int bufCount, HcclDataType datatype, const HcclEnvelopeSummary& envelope)
{
    if (envelope.status != 0) {
        HCCL_ERROR("[Check][EnvelopeStatus] envelope status:[%u] is invalid", envelope.status);
        return HCCL_E_PARA;
    }

    CHK_RET(CheckTransportEndPointInfo(envelope.envelope.epParam.src, envelope.envelope.epParam.dst));

    u32 recvSize = 0;
    u32 envelopSize = envelope.envelope.transData.count * SIZE_TABLE[envelope.envelope.transData.dataType];
    for (s32 i = 0; i < bufCount; i++) {
        if ((reinterpret_cast<u64>(buf[i]) == 0) && ((bufCount != 0) || (envelope.envelope.transData.count != 0))) {
            HCCL_ERROR(
                "[Check][RecvEnvelope]Imrecv buffer[%p] or count[%llu] is invalid", reinterpret_cast<u64>(buf[i]),
                envelope.envelope.transData.count);
            return HCCL_E_PARA;
        }
        recvSize += count[i] * SIZE_TABLE[datatype];
    }

    if (recvSize < envelopSize) {
        HCCL_ERROR(
            "[Check][RecvEnvelope] recvSize[%u Byte] is less than envelop total Size[%u Byte]", recvSize, envelopSize);
        return HCCL_E_PARA;
    }
    if (datatype != envelope.envelope.transData.dataType) {
        HCCL_ERROR(
            "[Check][RecvEnvelope]Imrecv input dataType[%s] should be Isend dataType[%s]",
            GetDataTypeEnumStr(datatype).c_str(), GetDataTypeEnumStr(envelope.envelope.transData.dataType).c_str());
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult
TransportHeterog::GenerateRecvMessage(HcclEnvelopeSummary& recvEnvelope, HcclMessageInfo*& msg, HcclStatus& status)
{
    msg = pMsgInfosMem_->Alloc();
    CHK_PTR_NULL(msg);

    msg->transportHandle = this;
    msg->envelope = recvEnvelope;
    status.srcRank = recvEnvelope.envelope.epParam.src.rank;
    status.tag = recvEnvelope.envelope.epParam.src.tag;
    status.count = recvEnvelope.envelope.transData.count;
    status.error = recvEnvelope.status;
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::FreeRecvMessage(HcclMessageInfo& msg) const
{
    msg.commHandle = nullptr;
    msg.transportHandle = nullptr;
    CHK_RET(pMsgInfosMem_->Free(&msg));
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::ProbeNothing(s32& flag, HcclMessageInfo*& msg, HcclStatus& status) const
{
    flag = HCCL_IMPROBE_INCOMPLETED;
    msg = nullptr;
    status.srcRank = -1;
    status.tag = -1;
    status.error = -1;
    status.count = -1;
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::AddSocketWhiteList(string& tag)
{
    std::vector<SocketWlistInfoT> whiteList(1);
    constexpr u32 connLimit = 4096;
    whiteList[0].remoteIp.addr = peerIp_.GetBinaryAddress().addr;
    whiteList[0].remoteIp.addr6 = peerIp_.GetBinaryAddress().addr6;
    whiteList[0].connLimit = connLimit;
    CHK_SAFETY_FUNC_RET(memcpy_s(&whiteList[0].tag, sizeof(whiteList[0].tag), tag.c_str(), tag.size() + 1));

    CHK_RET(hrtRaSocketWhiteListAdd(nicSocketHandle_, whiteList.data(), SINGLE_WHITE_LIST_NUM));
    HCCL_INFO(
        "TransportHeterogRoce::AddSocketWhiteList ip[%s], tag[%s]", peerIp_.GetReadableAddress(), whiteList[0].tag);
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::PrepareSocketInfo(s32 type, s32 linkNum, const string& clientTag, const string& serverTag)
{
    initSM_.locInitInfo.signal = SYNC_SIGNAL;
    initSM_.locInitInfo.protocolType = type;
    HcclInAddr peerAddr = peerIp_.GetBinaryAddress();
    for (int i = initSM_.locInitInfo.socketInfo.size(); i < linkNum; i++) {
        string tag = transTag_ + "_" + to_string(i) + "_";
        if (initSM_.locInitInfo.role == CLIENT_ROLE_SOCKET) {
            tag += clientTag;
            SocketConnectInfoT tmpConnInfo{};
            tmpConnInfo.socketHandle = nicSocketHandle_;
            tmpConnInfo.remoteIp.addr = peerAddr.addr;
            tmpConnInfo.remoteIp.addr6 = peerAddr.addr6;
            tmpConnInfo.port = peerPort_;
            CHK_SAFETY_FUNC_RET(strncpy_s(tmpConnInfo.tag, SOCK_CONN_TAG_SIZE, tag.c_str(), tag.length() + 1));
            initSM_.locInitInfo.socketConnInfo.emplace_back(tmpConnInfo);
        } else {
            tag += serverTag;
        }

        HCCL_INFO("link[%d] tag[%s]", i, tag.c_str());
        SocketInfoT tmpInfo = {};
        tmpInfo.socketHandle = nicSocketHandle_;
        tmpInfo.fdHandle = nullptr;
        tmpInfo.remoteIp.addr = peerAddr.addr;
        tmpInfo.remoteIp.addr6 = peerAddr.addr6;
        tmpInfo.status = CONNECT_FAIL;
        CHK_SAFETY_FUNC_RET(strncpy_s(tmpInfo.tag, SOCK_CONN_TAG_SIZE, tag.c_str(), tag.length() + 1));
        initSM_.locInitInfo.socketInfo.emplace_back(tmpInfo);
        if (isHdcMode_ || remoteIsHdc_) {
            // hdc模式下hccp默认开启白名单校验,因此要配置tag进入白名单
            CHK_RET(AddSocketWhiteList(tag));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::InitTransportConnect(s32 type, s32 linkNum)
{
    if (selfIp_ == peerIp_) {
        initSM_.locInitInfo.role = (selfPort_ < peerPort_) ? SERVER_ROLE_SOCKET : CLIENT_ROLE_SOCKET;
    } else {
        initSM_.locInitInfo.role = (selfIp_ < peerIp_) ? SERVER_ROLE_SOCKET : CLIENT_ROLE_SOCKET;
    }

    string clientTag = string(peerIp_.GetReadableIP()) + to_string(peerPort_) + string(selfIp_.GetReadableIP())
                       + to_string(selfPort_);
    string serverTag = string(selfIp_.GetReadableIP()) + to_string(selfPort_) + string(peerIp_.GetReadableIP())
                       + to_string(peerPort_);
    CHK_RET(PrepareSocketInfo(type, linkNum, clientTag, serverTag));

    if (initSM_.locInitInfo.role == CLIENT_ROLE_SOCKET) {
        CHK_RET(TryTransition(HCCL_SUCCESS, true, ConnState::CONN_STATE_CONNECT_CHECK_SOCKET));
    } else {
        CHK_RET(TryTransition(HCCL_SUCCESS, true, ConnState::CONN_STATE_GET_CHECK_SOCKET));
    }

    u32 rankTableCrc = TransportHeterog::rankTableCrc_.load();
    // 序列化信息
    std::ostringstream oss;
    oss.write(reinterpret_cast<const char_t*>(&rankTableCrc), sizeof(rankTableCrc));
    oss.write(
        reinterpret_cast<const char_t*>(&initSM_.locInitInfo.protocolType), sizeof(initSM_.locInitInfo.protocolType));

    CHK_SAFETY_FUNC_RET(
        memcpy_s(&(initSM_.locInitInfo.checkFrame[0]), HETEROG_MAX_FRAME_LEN - 1, oss.str().c_str(), oss.str().size()));

    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::InitTransportConnect(s32 type, u32 role, s32 linkNum, u32 tag)
{
    HCCL_DEBUG("TransportHeterog InitTransportConnect start type[%d] role[%u] linkNum[%d]", type, role, linkNum);
    initSM_.locInitInfo.role = role;

    string clientTag = string(selfIp_.GetReadableIP()) + "_" + to_string(0) + "_" + string(peerIp_.GetReadableIP())
                       + "_" + to_string(peerPort_) + "_" + to_string(tag);
    string serverTag = string(peerIp_.GetReadableIP()) + "_" + to_string(0) + "_" + string(selfIp_.GetReadableIP())
                       + "_" + to_string(selfPort_) + "_" + to_string(tag);
    CHK_RET(PrepareSocketInfo(type, linkNum, clientTag, serverTag));

    if (initSM_.locInitInfo.role == CLIENT_ROLE_SOCKET) {
        CHK_RET(TryTransition(HCCL_SUCCESS, true, ConnState::CONN_STATE_CONNECT_ALL_SOCKET));
    } else {
        CHK_RET(TryTransition(HCCL_SUCCESS, true, ConnState::CONN_STATE_GET_ALL_SOCKET));
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::ConnectSocket(SocketConnectInfoT conn[], u32 num, bool& completed)
{
    HcclResult ret = hrtRaSocketNonBlockBatchConnect(conn, num);
    completed = (ret == HCCL_SUCCESS);
    return ret;
}

HcclResult TransportHeterog::GetSocket(u32 role, struct SocketInfoT info[], u32 num, u32& connectedNum, bool& completed)
{
    HcclResult ret = HCCL_SUCCESS;
    for (u32 i = 0; i < num; i++) {
        if (info[i].status == CONNECT_FAIL) {
            SocketInfoT tmpInfo = info[i];
            u32 tmpNum = 0;
            ret = hrtRaNonBlockGetSockets(role, &tmpInfo, 1, &tmpNum);
            if (ret == HCCL_SUCCESS && tmpNum == 1 && tmpInfo.status == CONNECT_OK && tmpInfo.fdHandle != nullptr) {
                info[i].status = CONNECT_OK;
                info[i].fdHandle = tmpInfo.fdHandle;
                connectedNum += 1;
            } else if (ret == HCCL_E_AGAIN) {
                continue;
            } else {
                HCCL_WARNING("hrtRaNonBlockGetSockets ret[%d]", ret);
                return ret;
            }
        }
    }

    completed = connectedNum == num ? true : false;
    return ret;
}

HcclResult TransportHeterog::SocketSend(const FdHandle fdHandle, void* data, u64 size, u64& sentSize, bool& completed)
{
    HCCL_DEBUG("TransportHeterog::SocketSend start fdHandle[%p]", fdHandle);
    u64 tmpSize = 0;
    HcclResult ret
        = hrtRaSocketNonBlockSendHeterog(fdHandle, reinterpret_cast<char*>(data) + sentSize, size - sentSize, &tmpSize);
    if (ret == HCCL_SUCCESS) {
        sentSize += tmpSize;
        if (size == sentSize) {
            completed = true;
        } else if (sentSize > size) {
            HCCL_ERROR(
                "SocketSend sentSize[%llu Byte] bigger than size[%llu Byte] completed[%u Byte] tmpSize[%llu Byte]",
                sentSize, size, completed, tmpSize);
            return HCCL_E_NETWORK;
        }
    } else {
        completed = false;
    }
    if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
        HCCL_ERROR(
            "TransportHeterog::SocketSend size[%llu Byte] recvSize[%llu Byte] completed[%u Byte] tmpSize[%llu Byte]",
            size, sentSize, completed, tmpSize);
    }
    return ret;
}

HcclResult TransportHeterog::SocketRecv(const FdHandle fdHandle, void* data, u64 size, u64& recvSize, bool& completed)
{
    HCCL_DEBUG("TransportHeterog::SocketRecv start fdHandle[%p]", fdHandle);
    u64 tmpSize = 0;
    HcclResult ret
        = hrtRaSocketNonBlockRecvHeterog(fdHandle, reinterpret_cast<char*>(data) + recvSize, size - recvSize, &tmpSize);
    if (ret == HCCL_SUCCESS) {
        recvSize += tmpSize;
        if (size == recvSize) {
            completed = true;
        } else if (recvSize > size) {
            HCCL_ERROR(
                "SocketRecv recvSize[%llu Byte] bigger than size[%llu Byte] completed[%u Byte] tmpSize[%llu Byte]",
                recvSize, size, completed, tmpSize);
            return HCCL_E_NETWORK;
        }
    } else {
        completed = false;
    }
    if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
        HCCL_ERROR(
            "TransportHeterog::SocketRecv size[%llu Byte] recvSize[%llu Byte] completed[%u Byte] tmpSize[%llu Byte]",
            size, recvSize, completed, tmpSize);
    }
    return ret;
}

HcclResult TransportHeterog::SocketClose()
{
    u32 closeConnCount = 0;
    SocketCloseInfoT conns[MAX_LINK_NUM]{};
    CHK_PRT_RET(
        initSM_.locInitInfo.socketInfo.size() > MAX_LINK_NUM,
        HCCL_ERROR(
            "locInitInfo.socketInfo size can't exceed MAX_LINK_NUM, size[%d]", initSM_.locInitInfo.socketInfo.size()),
        HCCL_E_PARA);
    for (size_t i = 0; i < initSM_.locInitInfo.socketInfo.size(); i++) {
        if (initSM_.locInitInfo.socketInfo[i].fdHandle != nullptr) {
            conns[i].socketHandle = nicSocketHandle_;
            conns[i].fdHandle = initSM_.locInitInfo.socketInfo[i].fdHandle;
            conns[i].disuseLinger = static_cast<s32>(forceClose_);
            closeConnCount++;
        }
    }

    if (closeConnCount > 0) {
        if (hrtRaSocketBatchClose(conns, closeConnCount) != HCCL_SUCCESS) {
            HCCL_ERROR("[Destroy][TransportHeterog]ra socket batch close failed");
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::CheckConsistentFrame()
{
    std::string msg;
    msg.resize(sizeof(initSM_.locInitInfo.checkFrame));
    CHK_SAFETY_FUNC_RET(memcpy_s(
        reinterpret_cast<void*>(const_cast<char_t*>(msg.data())), HETEROG_MAX_FRAME_LEN - 1,
        &(initSM_.remInitInfo.checkFrame[0]), HETEROG_MAX_FRAME_LEN - 1));

    std::istringstream iss(msg);
    u32 localRankTableCrc = TransportHeterog::rankTableCrc_.load();
    u32 remoteRankTableCrc = 0;
    iss.read(reinterpret_cast<char_t*>(&remoteRankTableCrc), sizeof(remoteRankTableCrc));
    iss.read(reinterpret_cast<char_t*>(&initSM_.remInitInfo.protocolType), sizeof(initSM_.remInitInfo.protocolType));

    bool bIsDiff = false;
    if (remoteRankTableCrc != localRankTableCrc) {
        RPT_INPUT_ERR(
            true, "EI0005", std::vector<std::string>({"ccl_op", "para_name", "local_para", "remote_para"}),
            std::vector<std::string>(
                {"HcomBatchGet", "ranktable CRC", std::to_string(localRankTableCrc),
                 std::to_string(remoteRankTableCrc)}));
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] ranktable CRC check failed, crcValue[%u], receive crcvalue[%u].",
            LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_PARAMETER_CONFLICT.c_str(),
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), localRankTableCrc, remoteRankTableCrc);
        bIsDiff = true;
    }

    if (initSM_.remInitInfo.protocolType != initSM_.locInitInfo.protocolType) {
        HCCL_ERROR(
            "[CheckConsistentFrame][CompareFrame]errNo[0x%016llx] ProtocolType check fail",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL));
        bIsDiff = true;
    }
    if (bIsDiff) {
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

ConnState TransportHeterog::GetState() { return connState_.load(); }

HcclResult TransportHeterog::TryTransition(HcclResult ret, bool completed, ConnState nextState)
{
    if (ret == HCCL_SUCCESS && completed) {
        HCCL_INFO(
            "link[%s]: state[%d] transfer to state[%d]", initSM_.locInitInfo.socketInfo[0].tag, GetState(), nextState);
        connState_.store(nextState);
        CHK_RET(EnterStateProcess(nextState));
    } else if ((ret == HCCL_SUCCESS && !completed) || ret == HCCL_E_AGAIN) {
        HCCL_DEBUG("link[%s]: state[%d] not complete, hold", initSM_.locInitInfo.socketInfo[0].tag, GetState());
    } else {
        HCCL_ERROR(
            "link[%s]: State[%d] execute failed errno[%d][%s]", initSM_.locInitInfo.socketInfo[0].tag, nextState, errno,
            strerror(errno));
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::ConnectAsync()
{
    if (initSM_.locInitInfo.socketInfo.size() == 0) {
        HCCL_ERROR("[ConnectAsync]initSM_.locInitInfo.socketInfo is invalid!");
        return HCCL_E_PARA;
    }
    HCCL_DEBUG("link[%s]: Call ConnectAsync", initSM_.locInitInfo.socketInfo[0].tag);
    CHK_RET(LoopStateProcess());
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::SetDeviceIndex(s32 index)
{
    index_ = index;
    return HCCL_SUCCESS;
}

void TransportHeterog::AddRecvEnvelopNum()
{
    recvEnvelopNum_++;
    return;
}

void TransportHeterog::SubRecvEnvelopNum()
{
    recvEnvelopNum_--;
    return;
}

u32 TransportHeterog::GetRecvEnvelopNum() { return recvEnvelopNum_; }

HcclResult TransportHeterog::BlockSend(
    [[maybe_unused]] const TransData& sendData, [[maybe_unused]] const TransportEndPointParam& epParam,
    [[maybe_unused]] HcclRequestInfo*& request, [[maybe_unused]] s32 waitTimeOut)
{
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::BlockRecv(
    [[maybe_unused]] const TransData& recvData, [[maybe_unused]] bool matched,
    [[maybe_unused]] TransportHeterog*& transport, [[maybe_unused]] s32 waitTimeOut,
    [[maybe_unused]] s32 waitPayloadTimeOut)
{
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::CheckAndPushBuildLink()
{
    // 建链未完成时，继续推进建链流程；
    if (GetState() != ConnState::CONN_STATE_COMPLETE) {
        CHK_RET(ConnectAsync());
    }

    return (GetState() == ConnState::CONN_STATE_COMPLETE) ? HCCL_SUCCESS : HCCL_E_AGAIN;
}

HcclResult TransportHeterog::WaitBuildLinkComplete()
{
    HCCL_INFO("linkTag[%s] WaitBuildLinkComplete Begin! State[%d]", initSM_.locInitInfo.socketInfo[0].tag, GetState());
    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());

    while ((chrono::steady_clock::now() - startTime) < timeout) {
        HcclResult ret = CheckAndPushBuildLink();
        if (ret == HCCL_E_AGAIN) {
            SaluSleep(WAIT_LINK_BUILD_DELAY_TIME_US);
            continue;
        }

        if (ret == HCCL_SUCCESS) {
            HCCL_INFO(
                "Transport heterog connect success, localRank[%u], localIp[%s], remoteRank[%u], "
                "remoteIp[%s], linkTag[%s]!",
                localRank_, selfIp_.GetReadableAddress(), remoteRank_, peerIp_.GetReadableAddress(),
                initSM_.locInitInfo.socketInfo[0].tag);
        } else {
            HCCL_ERROR("Transport heterog connect failed, ret[%d]!", ret);
        }

        return ret;
    }

    HCCL_ERROR(
        "WaitBuildLinkComplete timeOut[%d] s, localIp[%s], "
        "remoteIp[%s], linkTag[%s], State[%d]",
        GetExternalInputHcclLinkTimeOut(), selfIp_.GetReadableAddress(), peerIp_.GetReadableAddress(),
        initSM_.locInitInfo.socketInfo[0].tag, GetState());

    return HCCL_E_TIMEOUT;
}

HcclResult TransportHeterog::Iwrite(
    [[maybe_unused]] const TransData& sendData, [[maybe_unused]] const HcclEnvelope& envelope,
    [[maybe_unused]] HcclRequestInfo*& request)
{
    HCCL_WARNING("Empty TransportHeterog::Iwrite is called.");
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::GetRemoteIsendDoneSignal([[maybe_unused]] std::shared_ptr<LocalIpcNotify>& signal)
{
    HCCL_WARNING("Empty TransportHeterog::GetRemoteIsendDoneSignal is called.");
    return HCCL_SUCCESS;
}

HcclResult TransportHeterog::GetRemoteImrecvDoneSignal([[maybe_unused]] std::shared_ptr<LocalIpcNotify>& signal)
{
    HCCL_WARNING("Empty TransportHeterog::GetRemoteImrecvDoneSignal is called.");
    return HCCL_SUCCESS;
}

void TransportHeterog::GetLinkTag([[maybe_unused]] std::string& tag)
{
    HCCL_WARNING("Empty TransportHeterog::GetLinkTag is called.");
    return;
}

void TransportHeterog::SetForceClose() { forceClose_ = true; }

void TransportHeterog::RecordRankTableCrc(const u32 crcValue)
{
    rankTableCrc_.store(crcValue);
    return;
}

} // namespace hccl
