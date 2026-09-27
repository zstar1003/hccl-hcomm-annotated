/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_memory_transport.h"
#include "exchange_ipc_buffer_dto.h"
#include "coll_operator_check.h"

namespace Hccl {
constexpr u32 ONE_HUNDRED_MICROSECOND_OF_USLEEP = 100;
UbMemoryTransport::UbMemoryTransport(
    const std::shared_ptr<Buffer> cclBuffer, const std::shared_ptr<Buffer> aivTagBuffer,
    const std::shared_ptr<Buffer> aivOffloadTagBuffer, Socket* socket, int32_t deviceLogicId)
    : cclBuffer(cclBuffer),
      aivTagBuffer(aivTagBuffer),
      aivOffloadTagBuffer(aivOffloadTagBuffer),
      socket(socket),
      deviceLogicId(deviceLogicId)
{}

HcclResult UbMemoryTransport::Init()
{
    localBufferVec.push_back(make_unique<LocalIpcRmaBuffer>(cclBuffer));
    localBufferVec.push_back(make_unique<LocalIpcRmaBuffer>(aivTagBuffer));
    localBufferVec.push_back(make_unique<LocalIpcRmaBuffer>(aivOffloadTagBuffer));
    return HCCL_SUCCESS;
}

LocalIpcRmaBuffer* UbMemoryTransport::GetLocMemBuffer(const u32 bufIndex) const
{
    HCCL_INFO("[%s] start", __func__);
    if (bufIndex >= localBufferVec.size()) {
        HCCL_ERROR("[%s] bufIndex[%u] is invalid, size[%zu]", __func__, bufIndex, localBufferVec.size());
        THROW<InternalException>(
            StringFormat("[%s] bufIndex[%u] is invalid, size[%zu]", __func__, bufIndex, localBufferVec.size()));
    }
    return localBufferVec[bufIndex].get();
}

RemoteIpcRmaBuffer* UbMemoryTransport::GetRmtMemBuffer(const u32 bufIndex) const
{
    HCCL_INFO("[%s] start", __func__);
    if (bufIndex >= rmtBufferVec.size()) {
        HCCL_ERROR("[%s] bufIndex[%u] is invalid, size[%zu]", __func__, bufIndex, rmtBufferVec.size());
        THROW<InternalException>(
            StringFormat("[%s] bufIndex[%u] is invalid, size[%zu]", __func__, bufIndex, rmtBufferVec.size()));
    }
    return rmtBufferVec[bufIndex].get();
}

UbMemoryTransport::UBTransportStatus UbMemoryTransport::GetStatus()
{
    UbMemoryTransport::UBTransportStatus status = UbMemoryTransport::UBTransportStatus::CONNECT_FAILED;
    try {
        status = StateMachine();
    } catch (HcclException& e) {
        HCCL_ERROR("%s", e.what());
        return UbMemoryTransport::UBTransportStatus::CONNECT_FAILED;
    } catch (exception& e) {
        HCCL_ERROR("%s", e.what());
        return UbMemoryTransport::UBTransportStatus::CONNECT_FAILED;
    } catch (...) {
        HCCL_ERROR("Unknown error occurred when StateMachine!");
        return UbMemoryTransport::UBTransportStatus::CONNECT_FAILED;
    }
    return status;
}

UbMemoryTransport::UBTransportStatus UbMemoryTransport::StateMachine()
{
    if (ubStatus == UBTransportStatus::READY) {
        return ubStatus;
    }
    SocketStatus socketStatus = socket->GetAsyncStatus();
    if (socketStatus == SocketStatus::INIT) {
        THROW<InternalException>("[UbMemoryTransport][GetStatus]socket timeout or no link, please check");
    } else if (socketStatus == SocketStatus::TIMEOUT) {
        return UBTransportStatus::SOCKET_TIMEOUT;
    } else if (socketStatus != SocketStatus::OK) {
        return ubStatus;
    }
    switch (ubStatus) {
        case UBTransportStatus::INIT:
            ubStatus = UBTransportStatus::SOCKET_OK;
            break;
        case UBTransportStatus::SOCKET_OK:
            SendMemInfo();
            ubStatus = UBTransportStatus::SEND_MEM_INFO;
            break;
        case UBTransportStatus::SEND_MEM_INFO:
            RecvMemInfo();
            ubStatus = UBTransportStatus::RECV_MEM_INFO;
            break;
        case UBTransportStatus::RECV_MEM_INFO:
            RecvMemProcess();
            ubStatus = UBTransportStatus::RECV_MEM_INFO_PROCESS;
            break;
        // 预留状态机，当前方案未使用
        case UBTransportStatus::RECV_MEM_INFO_PROCESS:
            ubStatus = UBTransportStatus::SEND_NAME;
            break;
        case UBTransportStatus::SEND_NAME:
            ubStatus = UBTransportStatus::RECV_NAME;
            break;
        case UBTransportStatus::RECV_NAME:
            ubStatus = UBTransportStatus::READY;
            break;
        default:
            break;
    }
    return ubStatus;
}

void UbMemoryTransport::SendMemInfo()
{
    HCCL_INFO("[%s] start", __func__);

    BinaryStream binaryStream;

    HandshakeMsgPack(binaryStream);
    BufferPack(binaryStream);

    std::vector<char> data;
    binaryStream.Dump(data);
    socket->SendAsync(&data[0], data.size());
    exchangeDataSize = data.size();
    HCCL_INFO("[%s] finished", __func__);
}

void UbMemoryTransport::HandshakeMsgPack(BinaryStream& binaryStream)
{
    binaryStream << static_cast<u32>(locOpAcceState);
    binaryStream << localHandshakeMsg;
    HCCL_INFO("[UbMemoryTransport][%s] start pack handshakeMsg", __func__);
}

void UbMemoryTransport::RecvMemInfo()
{
    recvDataMsg.resize(exchangeDataSize);
    socket->RecvAsync(reinterpret_cast<u8*>(&recvDataMsg[0]), recvDataMsg.size());
    HCCL_INFO(
        "recv data, size=%zu, data=%s", recvDataMsg.size(), Bytes2hex(recvDataMsg.data(), recvDataMsg.size()).c_str());
}

void UbMemoryTransport::RecvMemProcess()
{
    BinaryStream binaryStream(recvDataMsg);
    HandshakeMsgUnpack(binaryStream);
    RmtBufferUnpackProc(binaryStream);
}

void UbMemoryTransport::HandshakeMsgUnpack(BinaryStream& binaryStream)
{
    u32 rmtAccelerator{0};
    binaryStream >> rmtAccelerator;
    HCCL_INFO("[UbMemoryTransport::HandshakeMsgUnpack], rmtAccelerator[%u]", rmtAccelerator);
    rmtOpAcceState = static_cast<AcceleratorState::Value>(rmtAccelerator);

    if (rmtOpAcceState != locOpAcceState) {
        THROW<InvalidParamsException>(StringFormat(
            "[UbMemoryTransport::HandshakeMsgUnpack] Accelerator information check fail. "
            "locOpAccelerator[%s], rmtOpAccelerator[%s]",
            locOpAcceState.Describe().c_str(), rmtOpAcceState.Describe().c_str()));
    }

    rmtHandshakeMsg.clear();
    binaryStream >> rmtHandshakeMsg;

    if (localHandshakeMsg.size() != rmtHandshakeMsg.size()) {
        THROW<InvalidParamsException>(StringFormat(
            "handshakeMsg size=%zu is not equal to rmt=%zu", localHandshakeMsg.size(), rmtHandshakeMsg.size()));
    }

    auto localCollOperator = CollOperator::GetPackedData(localHandshakeMsg);
    auto remoteCollOperator = CollOperator::GetPackedData(rmtHandshakeMsg);
    CheckCollOperator(localCollOperator, remoteCollOperator); // 两端算子参数一致性校验

    HCCL_INFO("[UbMemoryTransport][%s] start unpack handshakeMsg", __func__);
}

void UbMemoryTransport::BufferPack(BinaryStream& binaryStream)
{
    u32 vecSize = localBufferVec.size();
    binaryStream << vecSize;
    for (auto& it : localBufferVec) {
        if (it != nullptr) { // 非空的buffer，从buffer中获取 dto
            std::unique_ptr<Serializable> dto = it->GetExchangeDto();
            HCCL_INFO("[%s] dto[%s]", __func__, dto->Describe().c_str());
            dto->Serialize(binaryStream);
        } else { // 空的buffer，dto所有字段为0(size=0)
            ExchangeIpcBufferDto exchangeDto;
            exchangeDto.Serialize(binaryStream);
        }
    }
}

void UbMemoryTransport::RmtBufferUnpackProc(BinaryStream& binaryStream)
{
    rmtBufferVec.clear();
    rmtRmaBufferVec.clear();
    u32 vecSize{0};
    binaryStream >> vecSize;
    HCCL_INFO("vecSize=%u", vecSize);
    for (u32 pos = 0; pos < vecSize; ++pos) {
        ExchangeIpcBufferDto dto;
        dto.Deserialize(binaryStream);
        HCCL_INFO("[%s] dto[%s]", __func__, dto.Describe().c_str());

        if (dto.size == 0) { // size为0，则为 remote 空buffer
            HCCL_INFO("unpack nullptr, pos=%u", pos);
            rmtBufferVec.push_back(nullptr);
            rmtRmaBufferVec.push_back(nullptr);
        } else { // size非0，则构造一个remote buffer
            rmtBufferVec.push_back(make_unique<RemoteIpcRmaBuffer>(dto));
            rmtRmaBufferVec.push_back(rmtBufferVec.back().get());
        }
    }
}

std::string UbMemoryTransport::Describe() const
{
    std::string description = "";

    description = StringFormat("deviceLogicId:%d,", deviceLogicId);
    description += StringFormat("UBTransportStatus:%u", static_cast<u32>(ubStatus));
    return description;
}

} // namespace Hccl
