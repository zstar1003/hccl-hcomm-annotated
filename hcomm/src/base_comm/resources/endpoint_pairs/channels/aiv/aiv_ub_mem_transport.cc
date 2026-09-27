/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_ub_mem_transport.h"
#include "exception_handler.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/buffer/exchange_ipc_buffer_dto.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/mem/user_remote_mem_getter.h"
#include "env_config/env_config_v2.h"

namespace hcomm {

AivUbMemTransport::AivUbMemTransport(Hccl::Socket* socket, HcommChannelDesc& channelDesc)
    : socket_(socket),
      channelDesc_(channelDesc)
{}

HcclResult AivUbMemTransport::FillBufferVec(
    HcommMemHandle* memHandles, uint32_t bufferNum, std::vector<Hccl::LocalIpcRmaBuffer*>& bufferVec)
{
    uint32_t totalBufferNum = localRmaBufferVec_.size() + bufferNum;
    if (UNLIKELY(totalBufferNum > MAX_BUFFER_NUM)) {
        HCCL_ERROR(
            "[AivUbMemTransport][FillBufferVec] totalBufferNum[%u] exceeds limit[%u]", totalBufferNum, MAX_BUFFER_NUM);
        return HCCL_E_PARA;
    }
    for (uint32_t i = 0; i < bufferNum; ++i) {
        auto localIpcRmaBuffer = reinterpret_cast<Hccl::LocalIpcRmaBuffer*>(memHandles[i]);
        CHK_PTR_NULL(localIpcRmaBuffer);
        auto buf = localIpcRmaBuffer->GetBuf();
        CHK_PTR_NULL(buf);
        bufferVec.push_back(localIpcRmaBuffer);
        HCCL_INFO(
            "[AivUbMemTransport][FillBufferVec] memHandleNum[%u] buffer[%s]", i, localIpcRmaBuffer->Describe().data());
    }
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::Init()
{
    uint32_t bufferNum = channelDesc_.memHandleNum;
    if (bufferNum == 0) {
        HCCL_ERROR("[AivUbMemTransport][Init] bufferNum is 0.");
        return HCCL_E_PARA;
    }
    HCCL_INFO("[AivUbMemTransport][Init] channelDesc_.memHandleNum: %u", bufferNum);
    CHK_RET(FillBufferVec(channelDesc_.memHandles, bufferNum, localRmaBufferVec_));

    baseStatus_ = Hccl::TransportStatus::INIT;
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::IsSocketReady(bool& isReady)
{
    CHK_PTR_NULL(socket_);
    EXCEPTION_HANDLE_BEGIN
    Hccl::SocketStatus socketStatus = socket_->GetAsyncStatus();
    if (socketStatus == Hccl::SocketStatus::OK) {
        baseStatus_ = Hccl::TransportStatus::SOCKET_OK;
        isReady = true;
    } else if (socketStatus == Hccl::SocketStatus::TIMEOUT) {
        baseStatus_ = Hccl::TransportStatus::SOCKET_TIMEOUT;
        isReady = false;
    }
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

void AivUbMemTransport::CheckStatusFuncResult(std::string funcName, HcclResult ret)
{
    if (UNLIKELY(ret != HCCL_SUCCESS)) {
        HCCL_ERROR(
            "[%s] fail ret[%d], aivUbStatus_[%d], baseStatus_[%d]", funcName.c_str(), ret, aivUbStatus_, baseStatus_);
        baseStatus_ = Hccl::TransportStatus::INVALID;
    }
}

Hccl::TransportStatus AivUbMemTransport::GetStatus()
{
    if (baseStatus_ == Hccl::TransportStatus::READY || baseStatus_ == Hccl::TransportStatus::INVALID) {
        return baseStatus_;
    } else if (baseStatus_ == Hccl::TransportStatus::INIT) {
        aivUbStatus_ = AivUbMemTransportStatus::INIT;
    }

    bool isReady = false;
    if (UNLIKELY(IsSocketReady(isReady) != HCCL_SUCCESS)) {
        HCCL_ERROR("[%s] IsSocketReady fail, aivUbStatus_[%d], baseStatus_[%d]", __func__, aivUbStatus_, baseStatus_);
        baseStatus_ = Hccl::TransportStatus::INVALID;
        return baseStatus_;
    }
    if (!isReady) {
        return baseStatus_;
    }
    return UpdateStatus();
}

Hccl::TransportStatus AivUbMemTransport::UpdateStatus()
{
    HCCL_INFO(
        "%s aivUbStatus_[%d], baseStatus_[%d] start, aivUbStatus_::SOCKET_OK[%d]", __func__, aivUbStatus_, baseStatus_,
        AivUbMemTransportStatus::SOCKET_OK);
    HcclResult ret;
    switch (aivUbStatus_) {
        case AivUbMemTransportStatus::INIT:
            aivUbStatus_ = AivUbMemTransportStatus::SOCKET_OK;
            baseStatus_ = Hccl::TransportStatus::SOCKET_OK;
            break;
        case AivUbMemTransportStatus::SOCKET_OK:
            ret = SendDataSize();
            CheckStatusFuncResult("SendDataSize", ret);
            aivUbStatus_ = AivUbMemTransportStatus::SEND_DATA_SIZE;
            break;
        case AivUbMemTransportStatus::SEND_DATA_SIZE:
            ret = RecvDataSize();
            CheckStatusFuncResult("RecvDataSize", ret);
            aivUbStatus_ = AivUbMemTransportStatus::RECV_DATA_SIZE;
            break;
        case AivUbMemTransportStatus::RECV_DATA_SIZE:
            ret = SendMemInfo();
            CheckStatusFuncResult("SendMemInfo", ret);
            aivUbStatus_ = AivUbMemTransportStatus::SEND_MEM_INFO;
            break;
        case AivUbMemTransportStatus::SEND_MEM_INFO:
            ret = RecvMemInfo();
            CheckStatusFuncResult("RecvMemInfo", ret);
            aivUbStatus_ = AivUbMemTransportStatus::RECV_MEM_INFO;
            break;
        case AivUbMemTransportStatus::RECV_MEM_INFO:
            ret = RecvDataProcess();
            CheckStatusFuncResult("RecvDataProcess", ret);
            aivUbStatus_ = AivUbMemTransportStatus::RECV_MEM_FIN;
            break;
        case AivUbMemTransportStatus::RECV_MEM_FIN:
            aivUbStatus_ = AivUbMemTransportStatus::READY;
            baseStatus_ = Hccl::TransportStatus::READY;
            break;
        default:
            break;
    }
    HCCL_INFO("%s aivUbStatus_[%d], baseStatus_[%d]", __func__, aivUbStatus_, baseStatus_);
    return baseStatus_;
}

HcclResult AivUbMemTransport::SendDataSize()
{
    HCCL_INFO("[%s] start", __func__);

    Hccl::BinaryStream binaryStream;
    CHK_RET(BufferPack(binaryStream, localRmaBufferVec_));

    binaryStream.Dump(sendData_);
    u32 sendSize = sendData_.size();
    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(&sendSize, sizeof(sendSize));
    EXCEPTION_HANDLE_END
    HCCL_INFO("[%s] finished", __func__);
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::RecvDataSize()
{
    HCCL_INFO("[%s] start", __func__);

    EXCEPTION_HANDLE_BEGIN
    socket_->RecvAsync(reinterpret_cast<u8*>(&exchangeDataSize_), sizeof(exchangeDataSize_));
    EXCEPTION_HANDLE_END
    HCCL_INFO("[%s] finished", __func__);
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::SendMemInfo()
{
    HCCL_INFO("[%s] start", __func__);

    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(&sendData_[0], sendData_.size());
    EXCEPTION_HANDLE_END
    HCCL_INFO("[%s] finished", __func__);
    return HCCL_SUCCESS;
}

HcclResult
AivUbMemTransport::BufferPack(Hccl::BinaryStream& binaryStream, std::vector<Hccl::LocalIpcRmaBuffer*>& bufferVec) const
{
    u32 vecSize = bufferVec.size();
    binaryStream << vecSize;
    HCCL_INFO("BufferPack vecSize=%u", vecSize);

    for (uint32_t i = 0; i < vecSize; ++i) {
        std::unique_ptr<Hccl::Serializable> dto = bufferVec[i]->GetExchangeDto();
        CHK_PTR_NULL(dto);
        dto->Serialize(binaryStream);
        HCCL_INFO("[%s] dto[%s]", __func__, dto->Describe().c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::RecvMemInfo()
{
    recvData_.resize(exchangeDataSize_);
    EXCEPTION_HANDLE_BEGIN
    socket_->RecvAsync(reinterpret_cast<u8*>(&recvData_[0]), recvData_.size());
    EXCEPTION_HANDLE_END
    // HCCL_INFO("recv data, size=%llu, data=%s", data.size(), Hccl::Bytes2hex(data.data(), data.size()).c_str());
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::RecvDataProcess()
{
    Hccl::BinaryStream binaryStream(recvData_);
    rmtBufferVec_.clear();
    rmtRmaBufferVec_.clear();
    EXCEPTION_HANDLE_BEGIN
    RmtBufferUnpackProc(binaryStream);
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

void AivUbMemTransport::RmtBufferUnpackProc(Hccl::BinaryStream& binaryStream)
{
    u32 vecSize{0};
    binaryStream >> vecSize;
    HCCL_INFO("vecSize=%u", vecSize);
    uint32_t totalBufferNum = rmtBufferVec_.size() + vecSize;
    if (UNLIKELY(totalBufferNum > MAX_BUFFER_NUM)) {
        EXCEPTION_THROW_IF_ERR(HCCL_E_PARA, "[AivUbMemTransport][RmtBufferUnpackProc] vecSize exceeds limit.");
    }

    for (u32 pos = 0; pos < vecSize; ++pos) {
        Hccl::ExchangeIpcBufferDto dto;
        dto.Deserialize(binaryStream);
        HCCL_INFO("[%s] dto[%s]", __func__, dto.Describe().c_str());
        if (dto.size == 0) { // size为0，则为 remote 空buffer
            HCCL_INFO("unpack nullptr, pos=%u", pos);
            rmtBufferVec_.push_back(nullptr);
            rmtRmaBufferVec_.push_back(nullptr);
        } else { // size非0，则构造一个remote buffer
            HCCL_INFO("[AivUbMemTransport][RmtBufferUnpackProc] unpack buffer memInfo[%s]", dto.memInfo.c_str());
            rmtBufferVec_.push_back(std::make_unique<Hccl::RemoteIpcRmaBuffer>(dto));
            rmtBufferVec_.back()->SetPathMode(channelDesc_.ubMemAttr.pathMode);
            rmtRmaBufferVec_.push_back(rmtBufferVec_.back().get());
        }
    }
}

HcclResult AivUbMemTransport::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    Hccl::RemoteMemCtx<std::unique_ptr<Hccl::RemoteIpcRmaBuffer>> remoteMemCtx{
        cacheValid_, rmtBufferVec_, remoteUserMems_, memInfoCopies_, memInfoPointers_, remoteMem, memInfos, memNum};
    CHK_RET(GetRemoteUserMems(remoteMemCtx));
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::CheckSocketStatus(std::string socketOperator)
{
    CHK_PTR_NULL(socket_);
    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    uint32_t retryCount = 0;
    while (true) {
        EXCEPTION_HANDLE_BEGIN
        Hccl::SocketStatus socketStatus = socket_->GetAsyncStatus();
        if (socketStatus == Hccl::SocketStatus::OK) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_INFO(
                "[AivUbMemTransport][%s] socket transport operation[%s] success, elapsed[%lld]ms, retryCount[%u]",
                __func__, socketOperator.c_str(), elapsed, retryCount);
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout || socketStatus == Hccl::SocketStatus::TIMEOUT) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_ERROR(
                "[AivUbMemTransport][%s] socket transport operation[%s] timeout after %lld sec, elapsed[%lld]ms, "
                "retryCount[%u]",
                __func__, socketOperator.c_str(), timeout, elapsed, retryCount);
            return HCCL_E_TIMEOUT;
        }
        EXCEPTION_HANDLE_END
        retryCount++;
    }
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum)
{
    if (memHandleNum == 0) {
        HCCL_WARNING("[AivUbMemTransport][UpdateMemInfo] bufferNum is 0.");
        return HCCL_SUCCESS;
    }
    locMemTemp_.clear();
    CHK_RET(FillBufferVec(memHandles, memHandleNum, locMemTemp_));
    HCCL_INFO("[AivUbMemTransport][UpdateMemInfo] bufferNum[%zu]", locMemTemp_.size());
    sendData_.clear();
    Hccl::BinaryStream sendStream;
    CHK_RET(BufferPack(sendStream, locMemTemp_));
    sendStream.Dump(sendData_);
    u32 sendSize = sendData_.size();
    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(&sendSize, sizeof(sendSize));
    EXCEPTION_HANDLE_END
    CHK_RET(CheckSocketStatus("SendDataSize"));
    CHK_RET(RecvDataSize());
    CHK_RET(CheckSocketStatus("RecvDataSize"));
    CHK_RET(SendMemInfo());
    CHK_RET(CheckSocketStatus("SendMemInfo"));
    CHK_RET(RecvMemInfo());
    CHK_RET(CheckSocketStatus("RecvMemInfo"));
    Hccl::BinaryStream recvStream(recvData_);
    EXCEPTION_HANDLE_BEGIN
    RmtBufferUnpackProc(recvStream);
    EXCEPTION_HANDLE_END
    localRmaBufferVec_.insert(localRmaBufferVec_.end(), locMemTemp_.begin(), locMemTemp_.end());
    // 流程中已有新增内存数量判断，故执行到此位置一定存在新增内存，需要将标识置位false，使得再次调用GetRemoteMems时重新构造缓存
    cacheValid_ = false;
    return HCCL_SUCCESS;
}
} // namespace hcomm
