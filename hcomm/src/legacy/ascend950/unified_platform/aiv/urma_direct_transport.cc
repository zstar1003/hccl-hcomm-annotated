/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "urma_direct_transport.h"
#include "serializable.h"
#include "exchange_ub_buffer_dto.h"
#include "exchange_ub_conn_dto.h"
#include "local_ub_rma_buffer.h"
#include "orion_adapter_hccp.h"

namespace Hccl {
constexpr u32 FINISH_MSG_SIZE = 128;
constexpr char_t FINISH_MSG[FINISH_MSG_SIZE] = "Ub Comm Pipe ready!";

constexpr size_t RMT_BUFFER_VEC_SIZE = 3;
constexpr size_t RMT_BUFFER_INDEX = 2;
constexpr size_t CONN_NUM = 1;
constexpr u32 WQE_SIZE = 64;

UrmaDirectTransport::UrmaDirectTransport(
    CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket,
    RdmaHandle rdmaHandle1)
    : BaseMemTransport(commonLocRes, attr, linkData, socket, TransportType::UB),
      rdmaHandle(rdmaHandle1)
{}

UrmaDirectTransport::UrmaDirectTransport(
    CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket,
    RdmaHandle rdmaHandle1, std::function<void(u32 streamId, u32 taskId, const TaskParam& taskParam)> callback)
    : BaseMemTransport(commonLocRes, attr, linkData, socket, TransportType::UB, callback),
      rdmaHandle(rdmaHandle1)
{}

RemoteUbRmaBuffer* UrmaDirectTransport::GetRmtBuffer() const
{
    HCCL_INFO("[%s] start", __func__);
    if (rmtBufferVec.size() != RMT_BUFFER_VEC_SIZE) {
        THROW<InternalException>(StringFormat(
            "[%s] rmtBufferVec is not [%zu], size[%zu]", __func__, rmtBufferVec.size(), RMT_BUFFER_VEC_SIZE));
    }
    auto rmtBuf = rmtBufferVec[RMT_BUFFER_INDEX].get();
    CHECK_NULLPTR(rmtBuf, "[UrmaDirectTransport::GetRmtBuffer] rmtBuf is nullptr!");
    return rmtBuf;
}

std::string UrmaDirectTransport::Describe() const
{
    string msg = StringFormat(
        "UbMemTransport=[commonLocRes=%s, urmaStatus=%s, ", commonLocRes.Describe().c_str(),
        urmaStatus.Describe().c_str());
    msg += StringFormat("exchangeDataSize=%u, ", exchangeDataSize);
    return msg;
}

HcclAiRMAWQ UrmaDirectTransport::GetAiRMAWQ()
{
    if (baseStatus != TransportStatus::READY) {
        MACRO_THROW(
            InternalException,
            StringFormat("[UrmaDirectTransport::%s]transport status is not ready, please check", __func__));
    }

    HcclAiRMAWQ wq = {};
    wq.wqeSize = WQE_SIZE;

    size_t connNum = commonLocRes.connVec.size();
    if (connNum != CONN_NUM) {
        THROW<InternalException>("[UrmaDirectTransport::%s] connNum[%zu] is not [%zu]", __func__, connNum, CONN_NUM);
    }
    auto conn = reinterpret_cast<DevUbCtpConnection*>(commonLocRes.connVec[0]);
    CHECK_NULLPTR(conn, StringFormat("[UrmaDirectTransport::%s] failed, connection pointer is nullptr", __func__));
    conn->SetWqInfo(wq);

    for (auto& it : commonLocRes.bufferVec) {
        if (it != nullptr) {
            LocalUbRmaBuffer* localBuffer = dynamic_cast<LocalUbRmaBuffer*>(it);
            CHECK_NULLPTR(
                localBuffer,
                StringFormat("[UrmaDirectTransport::%s] failed, localBuffer pointer is nullptr", __func__));
            HCCL_INFO("get local buffer, %s", localBuffer->Describe().c_str());
            wq.localTokenId = localBuffer->GetTokenId();
        }
    }

    for (auto& it : rmtBufferVec) {
        if (it != nullptr) {
            HCCL_INFO("get remote buffer, %s", it->Describe().c_str());
            wq.rmtObjId = it->GetTokenId();
            wq.rmtTokenValue = it->GetTokenValue();
        }
    }

    return wq;
}

HcclAiRMACQ UrmaDirectTransport::GetAiRMACQ()
{
    if (baseStatus != TransportStatus::READY) {
        MACRO_THROW(
            InternalException,
            StringFormat("[UrmaDirectTransport::%s]transport status is not ready, please check", __func__));
    }
    size_t connNum = commonLocRes.connVec.size();
    if (connNum != CONN_NUM) {
        THROW<InternalException>("[UrmaDirectTransport::%s] connNum[%zu] is not [%zu]", __func__, connNum, CONN_NUM);
    }
    auto conn = reinterpret_cast<DevUbCtpConnection*>(commonLocRes.connVec[0]);
    CHECK_NULLPTR(conn, StringFormat("[UrmaDirectTransport::%s] failed, connection pointer is nullptr", __func__));

    HcclAiRMACQ cq = {};
    conn->SetCqInfo(cq);
    return cq;
}

void UrmaDirectTransport::SendExchangeData()
{
    bufferNum = commonLocRes.bufferVec.size(); // 需要交换的buffer数量
    connNum = commonLocRes.connVec.size();

    HCCL_INFO("bufferNum=%u, connNum=%u", bufferNum, connNum);

    BinaryStream binaryStream;
    HandshakeMsgPack(binaryStream);
    BufferVecPack(binaryStream);
    ConnVecPack(binaryStream);

    binaryStream.Dump(sendData);
    socket->SendAsync(sendData.data(), sendData.size());
    exchangeDataSize = sendData.size();

    HCCL_INFO("send data %s, size=%u", GetLinkDescInfo().c_str(), exchangeDataSize);
}

void UrmaDirectTransport::BufferVecPack(BinaryStream& binaryStream)
{
    binaryStream << bufferNum;
    HCCL_INFO("start pack %s bufferVec", transportType.Describe().c_str());
    u32 pos = 0;
    for (auto& it : commonLocRes.bufferVec) {
        binaryStream << pos;
        if (it != nullptr) { // 非空的buffer，从buffer中获取 dto
            std::unique_ptr<Serializable> dto = it->GetExchangeDto();
            dto->Serialize(binaryStream);
            HCCL_INFO("pack buffer pos=%u dto %s", pos, dto->Describe().c_str());
        } else { // 空的buffer，dto所有字段为0(size=0)
            ExchangeUbBufferDto exchangeDto;
            exchangeDto.Serialize(binaryStream);
            HCCL_INFO("pack buffer pos=%u, dto is null %s", pos, exchangeDto.Describe().c_str());
        }
        pos++;
    }
}

bool UrmaDirectTransport::IsResReady()
{
    for (auto& it : commonLocRes.connVec) {
        CHECK_NULLPTR(it, StringFormat("[UrmaDirectTransport::%s] failed, connection pointer is nullptr", __func__));

        RmaConnType connType = it->GetRmaConnType();
        if (connType != RmaConnType::UB) {
            THROW<InternalException>(
                "[UrmaDirectTransport::%s] connection type[%s] is not ub", __func__, connType.Describe().c_str());
        }

        auto status = it->GetStatus();
        if (status != RmaConnStatus::EXCHANGEABLE && status != RmaConnStatus::READY) {
            return false;
        }
    }

    HCCL_INFO("[UrmaDirectTransport::IsResReady] all resources ready.");
    return true;
}

void UrmaDirectTransport::RecvExchangeData()
{
    recvData.resize(exchangeDataSize);
    socket->RecvAsync(reinterpret_cast<u8*>(recvData.data()), recvData.size());

    HCCL_INFO("recv data %s, size=%zu", GetLinkDescInfo().c_str(), recvData.size());
}

bool UrmaDirectTransport::ConnVecUnpackProc(BinaryStream& binaryStream)
{
    u32 rmtConnNum;
    binaryStream >> rmtConnNum;
    HCCL_INFO("start unpack conn %s connNum=%u, rmtConnNum=%u", GetLinkDescInfo().c_str(), connNum, rmtConnNum);
    if (connNum != rmtConnNum) {
        MACRO_THROW(
            InvalidParamsException, StringFormat("connNum=%u is not equal to rmtConnNum=%u", connNum, rmtConnNum));
    }

    bool result = false; // 不需要发送 finish
    for (u32 i = 0; i < rmtConnNum; i++) {
        u32 pos;
        binaryStream >> pos;
        ExchangeUbConnDto rmtDto;
        rmtDto.Deserialize(binaryStream);
        HCCL_INFO("unpack connection pos=%u dto %s", pos, rmtDto.Describe().c_str());
        if (commonLocRes.connVec[i]->GetStatus() != RmaConnStatus::READY) {
            HCCL_INFO(
                "parse and import pos=%u, rmt dto to connection[%s]", pos, commonLocRes.connVec[i]->Describe().c_str());
            commonLocRes.connVec[i]->ParseRmtExchangeDto(rmtDto);
            commonLocRes.connVec[i]->ImportRmtDto();
            result = true; // connection 建链，需要发送finish
        }
    }
    return result;
}

void UrmaDirectTransport::RmtBufferVecUnpackProc(u32 locNum, BinaryStream& binaryStream, RemoteBufferVec& bufferVec)
{
    u32 rmtNum;
    binaryStream >> rmtNum;

    HCCL_INFO("unpack BUFFER %s, locNum=%u, rmtNum=%u", GetLinkDescInfo().c_str(), locNum, rmtNum);
    if (rmtNum != locNum) {
        MACRO_THROW(
            InvalidParamsException, StringFormat("BUFFER, locNum=%u is not equal to rmtNum=%u", locNum, rmtNum));
    }

    for (u32 i = 0; i < rmtNum; i++) {
        u32 pos;
        binaryStream >> pos;
        ExchangeUbBufferDto dto;
        dto.Deserialize(binaryStream);
        if (bufferVec.size() > pos) {
            // 对于之前已经加过的资源，无需追加
            continue;
        }

        HCCL_INFO("unpack BUFFER pos=%u, dto %s", pos, dto.Describe().c_str());
        if (dto.size == 0) { // size为0，则为 remote 空buffer
            HCCL_INFO("unpack nullptr, pos=%u", pos);
            bufferVec.push_back(nullptr);
            rmtRmaBufferVec.push_back(nullptr);
        } else { // size非0，则构造一个remote buffer
            bufferVec.push_back(make_unique<RemoteUbRmaBuffer>(rdmaHandle, dto));
            rmtRmaBufferVec.push_back(bufferVec.back().get());
            HCCL_INFO("unpack buffer pos=%u, rmtRmaBuffer=%s", pos, bufferVec.back()->Describe().c_str());
        }
    }
}

bool UrmaDirectTransport::RecvDataProcess()
{
    HCCL_INFO(
        "RecvDataProcess: link=%s, size=%zu, exchangeDataSize=%u", GetLinkDescInfo().c_str(), recvData.size(),
        exchangeDataSize);
    BinaryStream binaryStream(recvData);
    HandshakeMsgUnpack(binaryStream);
    RmtBufferVecUnpackProc(bufferNum, binaryStream, rmtBufferVec);
    return ConnVecUnpackProc(binaryStream);
}

bool UrmaDirectTransport::IsConnsReady()
{
    for (u32 i = 0; i < connNum; i++) {
        if (commonLocRes.connVec[i]->GetStatus() != RmaConnStatus::READY) {
            return false;
        }
    }
    HCCL_INFO("conns are ready.");
    return true;
}

void UrmaDirectTransport::SendFinish()
{
    HCCL_INFO("start send Finish Msg %s [%s]", GetLinkDescInfo().c_str(), FINISH_MSG);
    sendFinishMsg = std::vector<char>(FINISH_MSG, FINISH_MSG + FINISH_MSG_SIZE);
    socket->SendAsync(sendFinishMsg.data(), FINISH_MSG_SIZE);
    HCCL_INFO("end send Finish Msg %s [%s]", GetLinkDescInfo().c_str(), FINISH_MSG);
}

void UrmaDirectTransport::RecvFinish()
{
    recvFinishMsg.resize(FINISH_MSG_SIZE);
    HCCL_INFO("start recv Finish Msg %s [%s]", GetLinkDescInfo().c_str(), FINISH_MSG);
    socket->RecvAsync(reinterpret_cast<u8*>(recvFinishMsg.data()), FINISH_MSG_SIZE);
    HCCL_INFO("end recv Finish Msg %s [%s]", GetLinkDescInfo().c_str(), FINISH_MSG);
}

TransportStatus UrmaDirectTransport::GetStatus()
{
    if (baseStatus == TransportStatus::READY) {
        return baseStatus;
    } else if (baseStatus == TransportStatus::INIT) {
        urmaStatus = UrmaStatus::INIT;
    }
    if (!IsSocketReady()) {
        return baseStatus;
    }
    switch (urmaStatus) {
        case UrmaStatus::INIT:
            urmaStatus = UrmaStatus::SOCKET_OK;
            baseStatus = TransportStatus::SOCKET_OK;
            break;
        case UrmaStatus::SOCKET_OK:
            if (IsResReady()) {
                urmaStatus = UrmaStatus::SEND_DATA;
                SendExchangeData();
            }
            break;
        case UrmaStatus::SEND_DATA:
            RecvExchangeData();
            urmaStatus = UrmaStatus::RECV_DATA;
            break;
        case UrmaStatus::RECV_DATA:
            if (RecvDataProcess()) { // 收消息中，如果设置到connection的建链，则需要发送 finish
                urmaStatus = UrmaStatus::PROCESS_DATA;
            } else { // 不需要发送finish，则将transport状态调整为 ready
                urmaStatus = UrmaStatus::RECV_FIN;
                SetBaseStatusReady();
            }
            break;
        case UrmaStatus::PROCESS_DATA:
            if (IsConnsReady()) {
                urmaStatus = UrmaStatus::CONN_OK;
                SendFinish();
            }
            break;
        case UrmaStatus::CONN_OK:
            RecvFinish();
            urmaStatus = UrmaStatus::SEND_FIN;
            break;
        case UrmaStatus::SEND_FIN:
            urmaStatus = UrmaStatus::RECV_FIN;
            SetBaseStatusReady();
            break;
        default:
            break;
    }
    return baseStatus;
}

} // namespace Hccl
