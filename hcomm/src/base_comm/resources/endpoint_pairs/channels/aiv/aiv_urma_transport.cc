/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "aiv_urma_transport.h"

#include <chrono>
#include <thread>

#include "serializable.h"
#include "exchange_ub_buffer_dto.h"
#include "exchange_ub_conn_dto.h"
#include "local_ub_rma_buffer.h"
#include "orion_adapter_hccp.h"
#include "hcomm_adapter_rts.h"
#include "coll_operator_check.h"
#include "user_remote_mem_getter.h"
#include "env_config/env_config_v2.h"

namespace Hccl {
constexpr uint32_t FINISH_MSG_SIZE = 128;
constexpr char_t FINISH_MSG[FINISH_MSG_SIZE] = "Ub Comm Pipe ready!";
constexpr uint32_t WQE_SIZE = 64;
constexpr uint32_t QUEUE_INDEX_MEM_UNIT_SIZE = sizeof(void*);

static hcomm::rtMemUbTokenInfo QueryProcessToken(const LocalRmaBuffer& buffer)
{
    hcomm::rtMemUbTokenInfo processTokenInfo{};
    processTokenInfo.va = buffer.GetAddr();
    processTokenInfo.size = buffer.GetSize();
    HcclResult ret = hcomm::RtsUbDevQueryInfo(QUERY_PROCESS_TOKEN, processTokenInfo);
    if (ret != HCCL_SUCCESS) {
        MACRO_THROW(
            InternalException, StringFormat(
                                   "[AivUrmaTransport::%s] query process token failed, addr[0x%llx], size[%llu], "
                                   "ret[%d]",
                                   __func__, static_cast<unsigned long long>(buffer.GetAddr()),
                                   static_cast<unsigned long long>(buffer.GetSize()), ret));
    }
    return processTokenInfo;
}

AivUrmaTransport::AivUrmaTransport(
    BaseMemTransport::CommonLocRes& commonLocRes, BaseMemTransport::Attribution& attr, const LinkData& linkData,
    const Socket& socket, RdmaHandle rdmaHandle)
    : commonLocRes_(commonLocRes),
      attr_(attr),
      linkData_(linkData),
      socket_(const_cast<Socket*>(&socket)),
      transportType_(TransportType::UB),
      rdmaHandle_(rdmaHandle)
{
    CheckCommonLocRes(commonLocRes);
}

std::string AivUrmaTransport::GetLinkDescInfo()
{
    return StringFormat(
        "rank[%u], rmtRank[%u] linkData=%s, type=%s", linkData_.GetLocalRankId(), linkData_.GetRemoteRankId(),
        linkData_.Describe().c_str(), transportType_.Describe().c_str());
}

void AivUrmaTransport::CheckLocBuffer(BaseMemTransport::CommonLocRes& res)
{
    HCCL_INFO("%s buffer check start, bufferNum=%u", GetLinkDescInfo().c_str(), res.bufferVec.size());
    uint32_t bufIndex = 0;
    for (auto& it : res.bufferVec) {
        if (it == nullptr) {
            HCCL_INFO("bufIndex=%u is nullptr", bufIndex);
        } else {
            HCCL_INFO("bufIndex=%u, buf=%s", bufIndex, it->Describe().c_str());
        }
        bufIndex++;
    }

    HCCL_INFO("%s buffer check ok, bufferNum=%u", GetLinkDescInfo().c_str(), res.bufferVec.size());
}

void AivUrmaTransport::CheckLocConn(BaseMemTransport::CommonLocRes& res)
{
    connNum_ = res.connVec.size();

    HCCL_INFO("%s connection check start, connNum=%u", GetLinkDescInfo().c_str(), connNum_);
    for (auto& it : res.connVec) {
        if (it == nullptr) {
            string msg = StringFormat("%s conn is nullptr", GetLinkDescInfo().c_str());
            MACRO_THROW(InvalidParamsException, msg);
        }
        HCCL_INFO("conn=%s", it->Describe().c_str());
    }
    HCCL_INFO("%s connection check ok, connNum=%u", GetLinkDescInfo().c_str(), connNum_);
}

void AivUrmaTransport::CheckCommonLocRes(BaseMemTransport::CommonLocRes& res)
{
    CheckLocBuffer(res);
    CheckLocConn(res);
}

std::string AivUrmaTransport::Describe() const
{
    string msg = StringFormat(
        "UbMemTransport=[commonLocRes=%s, urmaStatus=%s, ", commonLocRes_.Describe().c_str(),
        urmaStatus_.Describe().c_str());
    msg += StringFormat("exchangeDataSize=%u, ", exchangeDataSize_);
    return msg;
}

void AivUrmaTransport::GetEntityCountsForLayout(uint32_t& bufNum, uint32_t& connNum) const
{
    bufNum = static_cast<uint32_t>(commonLocRes_.bufferVec.size());
    connNum = connNum_;
}

void AivUrmaTransport::EnsureQueueIndexDeviceMem()
{
    if (connNum_ == 0) {
        return;
    }
    if (sqPiMem_ && sqCiMem_ && cqPiMem_ && cqCiMem_) {
        return;
    }

    const size_t memSize = static_cast<size_t>(connNum_) * QUEUE_INDEX_MEM_UNIT_SIZE;
    sqPiMem_ = hccl::DeviceMem::alloc(memSize);
    sqCiMem_ = hccl::DeviceMem::alloc(memSize);
    cqPiMem_ = hccl::DeviceMem::alloc(memSize);
    cqCiMem_ = hccl::DeviceMem::alloc(memSize);
    if (!sqPiMem_ || !sqCiMem_ || !cqPiMem_ || !cqCiMem_) {
        MACRO_THROW(
            InternalException,
            StringFormat(
                "[AivUrmaTransport::%s] DeviceMem::alloc for queue index mem failed, connNum[%u], size[%zu]", __func__,
                connNum_, memSize));
    }
}

void AivUrmaTransport::SetQueueIndexDeviceMem(
    void* sqPiMem, void* sqCiMem, void* cqPiMem, void* cqCiMem, size_t memSize)
{
    sqPiMem_ = hccl::DeviceMem::create(sqPiMem, memSize);
    sqCiMem_ = hccl::DeviceMem::create(sqCiMem, memSize);
    cqPiMem_ = hccl::DeviceMem::create(cqPiMem, memSize);
    cqCiMem_ = hccl::DeviceMem::create(cqCiMem, memSize);
}

void AivUrmaTransport::GetSqContext()
{
    if (transportStatus_ != TransportStatus::READY) {
        MACRO_THROW(
            InternalException,
            StringFormat("[AivUrmaTransport::%s]transport status is not ready, please check", __func__));
    }
    EnsureQueueIndexDeviceMem();

    sqContextVec_.clear();
    sqContextVec_.resize(connNum_);

    for (uint32_t i = 0; i < connNum_; ++i) {
        auto conn = dynamic_cast<DevUbConnection*>(commonLocRes_.connVec[i]);
        CHECK_NULLPTR(conn, StringFormat("[AivUrmaTransport::%s] failed, connection pointer is nullptr", __func__));
        SqContext sqContext{};
        sqContext.type = SQ_CONTEXT_TYPE_UB_JFS;
        sqContext.contextInfo.ubJfs.wqeSize = WQE_SIZE;
        conn->SetSqContextInfo(sqContext);
        sqContext.contextInfo.ubJfs.headAddr
            = ReinterpretAs<uint64_t>(sqPiMem_.ptr()) + static_cast<uint64_t>(i) * QUEUE_INDEX_MEM_UNIT_SIZE;
        sqContext.contextInfo.ubJfs.tailAddr
            = ReinterpretAs<uint64_t>(sqCiMem_.ptr()) + static_cast<uint64_t>(i) * QUEUE_INDEX_MEM_UNIT_SIZE;
        sqContextVec_[i] = sqContext;
    }
}

void AivUrmaTransport::GetCqContext()
{
    if (transportStatus_ != TransportStatus::READY) {
        MACRO_THROW(
            InternalException,
            StringFormat("[AivUrmaTransport::%s]transport status is not ready, please check", __func__));
    }
    EnsureQueueIndexDeviceMem();

    cqContextVec_.clear();
    cqContextVec_.resize(connNum_);

    for (uint32_t i = 0; i < connNum_; ++i) {
        auto conn = dynamic_cast<DevUbConnection*>(commonLocRes_.connVec[i]);
        CHECK_NULLPTR(conn, StringFormat("[AivUrmaTransport::%s] failed, connection pointer is nullptr", __func__));
        CqContext cqContext{};
        cqContext.type = CQ_CONTEXT_TYPE_UB_JFC;
        conn->SetCqContextInfo(cqContext);
        cqContext.contextInfo.ubJfc.headAddr
            = ReinterpretAs<uint64_t>(cqPiMem_.ptr()) + static_cast<uint64_t>(i) * QUEUE_INDEX_MEM_UNIT_SIZE;
        cqContext.contextInfo.ubJfc.tailAddr
            = ReinterpretAs<uint64_t>(cqCiMem_.ptr()) + static_cast<uint64_t>(i) * QUEUE_INDEX_MEM_UNIT_SIZE;
        cqContextVec_[i] = cqContext;
    }
}

void AivUrmaTransport::PrepareHostChannelEntity(ChannelEntity* channelEntitiesHost)
{
    CHECK_NULLPTR(channelEntitiesHost, StringFormat("[AivUrmaTransport::%s]channelEntitiesHost is nullptr", __func__));
    GetProtectionInfo();

    channelEntitiesHost->localBufferNum = localBufferInfo_.size();
    channelEntitiesHost->localBufferAddr = localBufferInfo_.data();
    channelEntitiesHost->remoteBufferNum = remoteBufferInfo_.size();
    channelEntitiesHost->remoteBufferAddr = remoteBufferInfo_.data();
    channelEntitiesHost->sqNum = connNum_;
    channelEntitiesHost->cqNum = connNum_;
}

void AivUrmaTransport::ResolveLocalBufferTokens()
{
    if (localBufferTokensResolved_) {
        return;
    }

    localBufferTokens_.resize(commonLocRes_.bufferVec.size());
    hcomm::rtMemUbTokenInfo processTokenInfo{};
    bool processTokenResolved = false;
    int32_t currentDeviceId = 0;
    aclError getDeviceRet = aclrtGetDevice(&currentDeviceId);
    if (getDeviceRet != ACL_SUCCESS) {
        MACRO_THROW(
            InternalException,
            StringFormat("[AivUrmaTransport::%s] get current device failed, ret[%d]", __func__, getDeviceRet));
    }
    for (size_t i = 0; i < commonLocRes_.bufferVec.size(); ++i) {
        if (commonLocRes_.bufferVec[i] == nullptr) {
            continue;
        }
        auto* localBuffer = dynamic_cast<LocalUbRmaBuffer*>(commonLocRes_.bufferVec[i]);
        CHECK_NULLPTR(
            localBuffer, StringFormat("[AivUrmaTransport::%s] failed, localBuffer pointer is nullptr", __func__));

        localBufferTokens_[i].tokenId = localBuffer->GetTokenId();
        localBufferTokens_[i].tokenValue = localBuffer->GetTokenValue();

        aclrtPtrAttributes attributes{};
        aclError ret = aclrtPointerGetAttributes(ReinterpretAs<const void*>(localBuffer->GetAddr()), &attributes);
        if (ret != ACL_SUCCESS) {
            HCCL_WARNING(
                "[AivUrmaTransport::%s] get buffer attributes failed, use registered token, addr[0x%llx], ret[%d]",
                __func__, static_cast<unsigned long long>(localBuffer->GetAddr()), ret);
            continue;
        }
        if (attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE
            || attributes.location.id != static_cast<uint32_t>(currentDeviceId)) {
            HCCL_INFO(
                "[AivUrmaTransport::%s] buffer is not on channel device, use registered token, addr[0x%llx], "
                "bufferDeviceId[%u], channelDeviceId[%u]",
                __func__, static_cast<unsigned long long>(localBuffer->GetAddr()), attributes.location.id,
                static_cast<uint32_t>(currentDeviceId));
            continue;
        }

        if (!processTokenResolved) {
            processTokenInfo = QueryProcessToken(*localBuffer);
            processTokenResolved = true;
        }
        localBufferTokens_[i].tokenId = processTokenInfo.tokenId;
        localBufferTokens_[i].tokenValue = processTokenInfo.tokenValue;
        localBufferTokens_[i].useProcessToken = true;
    }
    localBufferTokensResolved_ = true;
}

void AivUrmaTransport::GetProtectionInfo()
{
    if (transportStatus_ != TransportStatus::READY) {
        MACRO_THROW(
            InternalException,
            StringFormat("[AivUrmaTransport::%s]transport status is not ready, please check", __func__));
    }

    ResolveLocalBufferTokens();
    size_t localBufSize = commonLocRes_.bufferVec.size();
    localBufferInfo_.clear();
    localBufferInfo_.resize(localBufSize);
    for (size_t i = 0; i < localBufSize; ++i) {
        auto& it = commonLocRes_.bufferVec[i];
        if (it != nullptr) {
            LocalUbRmaBuffer* localBuffer = dynamic_cast<LocalUbRmaBuffer*>(it);
            CHECK_NULLPTR(
                localBuffer, StringFormat("[AivUrmaTransport::%s] failed, localBuffer pointer is nullptr", __func__));
            HCCL_INFO("get local buffer, %s", localBuffer->Describe().c_str());
            localBufferInfo_[i].type = REGED_BUFFER_RMA;
            localBufferInfo_[i].bufferInfo.rma.addr = it->GetAddr();
            localBufferInfo_[i].bufferInfo.rma.size = it->GetSize();
            localBufferInfo_[i].bufferInfo.rma.protectionInfo.type = PROTECTION_TYPE_UB;
            localBufferInfo_[i].bufferInfo.rma.protectionInfo.memInfo.ub.tokenId = localBufferTokens_[i].tokenId;
            localBufferInfo_[i].bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue = localBufferTokens_[i].tokenValue;
        }
    }

    size_t remoteBufSize = rmtBufferVec_.size();
    remoteBufferInfo_.clear();
    remoteBufferInfo_.resize(remoteBufSize);
    for (size_t i = 0; i < remoteBufSize; ++i) {
        auto& it = rmtBufferVec_[i];
        if (it != nullptr) {
            HCCL_INFO("get remote buffer, %s", it->Describe().c_str());
            remoteBufferInfo_[i].type = REGED_BUFFER_RMA;
            remoteBufferInfo_[i].bufferInfo.rma.addr = it->GetAddr();
            remoteBufferInfo_[i].bufferInfo.rma.size = it->GetSize();
            remoteBufferInfo_[i].bufferInfo.rma.protectionInfo.type = PROTECTION_TYPE_UB;
            remoteBufferInfo_[i].bufferInfo.rma.protectionInfo.memInfo.ub.tokenId = it->GetTokenId();
            remoteBufferInfo_[i].bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue = it->GetTokenValue();
        }
    }
}

void AivUrmaTransport::HandshakeMsgPack(BinaryStream& binaryStream)
{
    HCCL_INFO(
        "[AivUrmaTransport::%s] start pack %s handshakeMsg, size=%u, accelerator=%s", __func__,
        transportType_.Describe().c_str(), attr_.handshakeMsg.size(), attr_.opAcceState.Describe().c_str());
    binaryStream << static_cast<uint32_t>(attr_.opAcceState);
    binaryStream << attr_.handshakeMsg;
}

void AivUrmaTransport::HandshakeMsgUnpack(BinaryStream& binaryStream)
{
    uint32_t rmtAccelerator{0};
    binaryStream >> rmtAccelerator;
    rmtOpAcceState_ = static_cast<AcceleratorState::Value>(rmtAccelerator);
    HCCL_INFO(
        "[AivUrmaTransport::%s] locOpAccelerator[%s], rmtOpAccelerator[%s]", __func__,
        attr_.opAcceState.Describe().c_str(), rmtOpAcceState_.Describe().c_str());
    if (rmtOpAcceState_ != attr_.opAcceState) {
        THROW<InvalidParamsException>(StringFormat(
            "[AivUrmaTransport::HandshakeMsgUnpack] Accelerator information check fail. "
            "locOpAccelerator[%s], rmtOpAccelerator[%s]",
            attr_.opAcceState.Describe().c_str(), rmtOpAcceState_.Describe().c_str()));
    }

    rmtHandshakeMsg_.clear();
    binaryStream >> rmtHandshakeMsg_;
    // 这里怎么确认两边的msg一样
    if (attr_.handshakeMsg.size() != rmtHandshakeMsg_.size()) {
        MACRO_THROW(
            InvalidParamsException,
            StringFormat(
                "handshakeMsg size=%u is not equal to rmt=%u", attr_.handshakeMsg.size(), rmtHandshakeMsg_.size()));
    }

    // 单边通信情况下，handshakeMsg的size为0
    if (attr_.handshakeMsg.size() == 0) {
        return;
    }
    auto localCollOperator = CollOperator::GetPackedData(attr_.handshakeMsg);
    auto remoteCollOperator = CollOperator::GetPackedData(rmtHandshakeMsg_);
    CheckCollOperator(localCollOperator, remoteCollOperator); // 两端算子参数一致性校验
}

void AivUrmaTransport::BufferVecPack(BinaryStream& binaryStream)
{
    ResolveLocalBufferTokens();
    binaryStream << static_cast<u32>(commonLocRes_.bufferVec.size());
    HCCL_INFO("start pack %s bufferVec", transportType_.Describe().c_str());
    uint32_t pos = 0;
    for (auto& it : commonLocRes_.bufferVec) {
        binaryStream << pos;
        if (it != nullptr) { // 非空的buffer，从buffer中获取 dto
            std::unique_ptr<Serializable> dto = it->GetExchangeDto();
            auto* ubBufferDto = dynamic_cast<ExchangeUbBufferDto*>(dto.get());
            CHECK_NULLPTR(
                ubBufferDto,
                StringFormat("[AivUrmaTransport::%s] exchange buffer dto is not ExchangeUbBufferDto", __func__));
            if (localBufferTokens_[pos].useProcessToken) {
                ubBufferDto->tokenId = localBufferTokens_[pos].tokenId;
                ubBufferDto->tokenValue = localBufferTokens_[pos].tokenValue;
            }
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

void AivUrmaTransport::ConnVecPack(BinaryStream& binaryStream)
{
    binaryStream << connNum_;
    HCCL_INFO("start pack %s connVec", transportType_.Describe().c_str());
    uint32_t pos = 0;
    for (auto& it : commonLocRes_.connVec) {
        binaryStream << pos;
        std::unique_ptr<Serializable> dto = it->GetExchangeDto();
        dto->Serialize(binaryStream);
        HCCL_INFO("pack connection pos=%u, dto %s", pos, dto->Describe().c_str());
        pos++;
    }
}

void AivUrmaTransport::SendExchangeData()
{
    HCCL_INFO(
        "bufferNum=%u, connNum=%u notifyNum=%u", commonLocRes_.bufferVec.size(), connNum_,
        commonLocRes_.notifyVec.size());

    BinaryStream binaryStream;
    HandshakeMsgPack(binaryStream);
    BufferVecPack(binaryStream);
    ConnVecPack(binaryStream);

    binaryStream.Dump(sendData_);
    socket_->SendAsync(sendData_.data(), sendData_.size());
    exchangeDataSize_ = sendData_.size();

    HCCL_INFO("send data %s, size=%llu", GetLinkDescInfo().c_str(), exchangeDataSize_);
}

bool AivUrmaTransport::IsResReady()
{
    for (auto& it : commonLocRes_.connVec) {
        CHECK_NULLPTR(it, StringFormat("[AivUrmaTransport::%s] failed, connection pointer is nullptr", __func__));

        RmaConnType connType = it->GetRmaConnType();
        if (connType != RmaConnType::UB) {
            THROW<InternalException>(
                "[AivUrmaTransport::%s] connection type[%s] is not ub", __func__, connType.Describe().c_str());
        }

        auto status = it->GetStatus();
        if (status != RmaConnStatus::EXCHANGEABLE && status != RmaConnStatus::READY) {
            return false;
        }
    }

    HCCL_INFO("[AivUrmaTransport::IsResReady] all resources ready.");
    return true;
}

void AivUrmaTransport::RecvExchangeData()
{
    recvData_.clear();
    recvData_.resize(exchangeDataSize_);
    socket_->RecvAsync(ReinterpretAs<u8*>(recvData_.data()), recvData_.size());

    HCCL_INFO("recv data %s, size=%llu", GetLinkDescInfo().c_str(), recvData_.size());
}

bool AivUrmaTransport::ConnVecUnpackProc(BinaryStream& binaryStream)
{
    uint32_t rmtConnNum;
    binaryStream >> rmtConnNum;
    HCCL_INFO("start unpack conn %s connNum=%u, rmtConnNum=%u", GetLinkDescInfo().c_str(), connNum_, rmtConnNum);
    if (connNum_ != rmtConnNum) {
        MACRO_THROW(
            InvalidParamsException, StringFormat("connNum=%u is not equal to rmtConnNum=%u", connNum_, rmtConnNum));
    }

    bool result = false; // 不需要发送 finish
    for (uint32_t i = 0; i < rmtConnNum; i++) {
        uint32_t pos;
        binaryStream >> pos;
        ExchangeUbConnDto rmtDto;
        rmtDto.Deserialize(binaryStream);
        HCCL_INFO("unpack connection pos=%u dto %s", pos, rmtDto.Describe().c_str());
        if (commonLocRes_.connVec[i]->GetStatus() != RmaConnStatus::READY) {
            HCCL_INFO(
                "parse and import pos=%u, rmt dto to connection[%s]", pos,
                commonLocRes_.connVec[i]->Describe().c_str());
            commonLocRes_.connVec[i]->ParseRmtExchangeDto(rmtDto);
            commonLocRes_.connVec[i]->ImportRmtDto();
            result = true; // connection 建链，需要发送finish
        }
    }
    return result;
}

void AivUrmaTransport::RmtBufferVecUnpackProc(uint32_t locNum, BinaryStream& binaryStream, RemoteBufferVec& bufferVec)
{
    uint32_t rmtNum;
    binaryStream >> rmtNum;

    HCCL_INFO("unpack BUFFER %s, locNum=%u, rmtNum=%u", GetLinkDescInfo().c_str(), locNum, rmtNum);
    if (rmtNum != locNum) {
        MACRO_THROW(
            InvalidParamsException, StringFormat("BUFFER, locNum=%u is not equal to rmtNum=%u", locNum, rmtNum));
    }

    for (uint32_t i = 0; i < rmtNum; i++) {
        uint32_t pos;
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
        } else { // size非0，则构造一个remote buffer
            bufferVec.push_back(make_unique<RemoteUbRmaBuffer>(rdmaHandle_, dto));
            HCCL_INFO("unpack buffer pos=%u, rmtRmaBuffer=%s", pos, bufferVec.back()->Describe().c_str());
        }
    }
}

bool AivUrmaTransport::RecvDataProcess()
{
    HCCL_INFO(
        "RecvDataProcess: link=%s, size=%llu, exchangeDataSize=%u", GetLinkDescInfo().c_str(), recvData_.size(),
        exchangeDataSize_);
    BinaryStream binaryStream(recvData_);
    HandshakeMsgUnpack(binaryStream); // 这里怎么确认两边的msg一样
    RmtBufferVecUnpackProc(commonLocRes_.bufferVec.size(), binaryStream, rmtBufferVec_);
    return ConnVecUnpackProc(binaryStream);
}

bool AivUrmaTransport::IsConnsReady()
{
    for (uint32_t i = 0; i < connNum_; i++) {
        if (commonLocRes_.connVec[i]->GetStatus() != RmaConnStatus::READY) {
            return false;
        }
    }
    HCCL_INFO("conns are ready.");
    return true;
}

void AivUrmaTransport::SendFinish()
{
    HCCL_INFO("start send Finish Msg %s [%s]", GetLinkDescInfo().c_str(), FINISH_MSG);
    sendFinishMsg_ = std::vector<char>(FINISH_MSG, FINISH_MSG + FINISH_MSG_SIZE);
    socket_->SendAsync(sendFinishMsg_.data(), FINISH_MSG_SIZE);
    HCCL_INFO("end send Finish Msg %s [%s]", GetLinkDescInfo().c_str(), FINISH_MSG);
}

void AivUrmaTransport::RecvFinish()
{
    recvFinishMsg_.resize(FINISH_MSG_SIZE);
    HCCL_INFO("start recv Finish Msg %s [%s]", GetLinkDescInfo().c_str(), FINISH_MSG);
    socket_->RecvAsync(ReinterpretAs<u8*>(recvFinishMsg_.data()), FINISH_MSG_SIZE);
    HCCL_INFO("end recv Finish Msg %s [%s]", GetLinkDescInfo().c_str(), FINISH_MSG);
}

bool AivUrmaTransport::IsSocketReady()
{
    if (socket_ == nullptr) {
        MACRO_THROW(InternalException, StringFormat("%s socket is nullptr, please check", GetLinkDescInfo().c_str()));
    }

    SocketStatus socketStatus = socket_->GetAsyncStatus();
    if (socketStatus == SocketStatus::OK) {
        transportStatus_ = TransportStatus::SOCKET_OK;
        return true;
    } else if (socketStatus == SocketStatus::TIMEOUT) {
        transportStatus_ = TransportStatus::SOCKET_TIMEOUT;
        return false;
    }

    return false;
}

HcclResult AivUrmaTransport::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    Hccl::RemoteMemCtx<std::unique_ptr<RemoteUbRmaBuffer>> remoteMemCtx{
        cacheValid_, rmtBufferVec_, remoteUserMems_, memInfoCopies_, memInfoPointers_, remoteMem, memInfos, memNum};
    CHK_RET(GetRemoteUserMems(remoteMemCtx));
    return HCCL_SUCCESS;
}

HcclResult AivUrmaTransport::CheckSocketStatus(std::string socketOperator) const
{
    CHK_PTR_NULL(socket_);
    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    uint32_t retryCount = 0;
    while (true) {
        SocketStatus socketStatus = socket_->GetAsyncStatus();
        if (socketStatus == SocketStatus::OK) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_INFO(
                "[AivUrmaTransport][%s] socket transport operation[%s] success, elapsed[%lld]ms, retryCount[%u]",
                __func__, socketOperator.c_str(), elapsed, retryCount);
            return HCCL_SUCCESS;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout || socketStatus == SocketStatus::TIMEOUT) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_ERROR(
                "[AivUrmaTransport][%s] socket transport operation[%s] timeout after %lld sec, elapsed[%lld]ms, "
                "retryCount[%u]",
                __func__, socketOperator.c_str(), timeout, elapsed, retryCount);
            return HCCL_E_TIMEOUT;
        }
        std::this_thread::yield();
        retryCount++;
    }
    return HCCL_SUCCESS;
}

HcclResult AivUrmaTransport::UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum)
{
    if (memHandles == nullptr || memHandleNum == 0) {
        HCCL_WARNING("[AivUrmaTransport][UpdateMemInfo] bufferNum is 0.");
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(socket_);
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);

    // 1. 新增 buffer 转换并追加到本端 bufferVec
    std::vector<Hccl::LocalRmaBuffer*> newBufs;
    for (uint32_t i = 0; i < memHandleNum; ++i) {
        auto localRmaBuffer = ReinterpretAs<Hccl::LocalRmaBuffer*>(memHandles[i]);
        CHK_PTR_NULL(localRmaBuffer);
        auto buf = localRmaBuffer->GetBuf();
        CHK_PTR_NULL(buf);
        newBufs.push_back(localRmaBuffer);
    }
    size_t origBufNum = commonLocRes_.bufferVec.size();
    commonLocRes_.bufferVec.insert(commonLocRes_.bufferVec.end(), newBufs.begin(), newBufs.end());
    HCCL_INFO("[AivUrmaTransport][UpdateMemInfo] bufferNum after append[%zu]", commonLocRes_.bufferVec.size());

    // 2. 双向交换：仅含全量 bufferVec（connVec 已建链不重做）
    BinaryStream binaryStream;
    BufferVecPack(binaryStream);
    sendData_.clear();
    binaryStream.Dump(sendData_);
    exchangeDataSize_ = sendData_.size();
    socket_->SendAsync(sendData_.data(), sendData_.size());
    CHK_RET(CheckSocketStatus("SendData"));

    recvData_.clear();
    recvData_.resize(exchangeDataSize_);
    socket_->RecvAsync(ReinterpretAs<u8*>(recvData_.data()), recvData_.size());
    CHK_RET(CheckSocketStatus("RecvData"));

    // 3. 解析对端全量 buffer 数据（RmtBufferVecUnpackProc 对已存在的 pos 幂等跳过）
    BinaryStream recvStream(recvData_);
    RmtBufferVecUnpackProc(commonLocRes_.bufferVec.size(), recvStream, rmtBufferVec_);
    if (rmtBufferVec_.size() != commonLocRes_.bufferVec.size()) {
        HCCL_ERROR(
            "[AivUrmaTransport][UpdateMemInfo] buffer num mismatch after update, loc[%zu] rmt[%zu]",
            commonLocRes_.bufferVec.size(), rmtBufferVec_.size());
        commonLocRes_.bufferVec.resize(origBufNum);
        return HCCL_E_INTERNAL;
    }

    cacheValid_ = false;
    return HCCL_SUCCESS;
}

bool AivUrmaTransport::PrepareGetStatus()
{
    if (transportStatus_ == TransportStatus::READY) {
        return false;
    } else if (transportStatus_ == TransportStatus::INIT) {
        urmaStatus_ = UrmaStatus::INIT;
    }

    return IsSocketReady();
}

void AivUrmaTransport::ProcessUrmaStatus()
{
    switch (urmaStatus_) {
        case UrmaStatus::INIT:
            urmaStatus_ = UrmaStatus::SOCKET_OK;
            transportStatus_ = TransportStatus::SOCKET_OK;
            break;
        case UrmaStatus::SOCKET_OK:
            if (IsResReady()) {
                urmaStatus_ = UrmaStatus::SEND_DATA;
                SendExchangeData();
            }
            break;
        case UrmaStatus::SEND_DATA:
            RecvExchangeData();
            urmaStatus_ = UrmaStatus::RECV_DATA;
            break;
        case UrmaStatus::RECV_DATA:
            if (RecvDataProcess()) { // 收消息中，如果设置到connection的建链，则需要发送 finish
                urmaStatus_ = UrmaStatus::PROCESS_DATA;
            } else { // 不需要发送finish，则将transport状态调整为 ready
                urmaStatus_ = UrmaStatus::RECV_FIN;
                transportStatus_ = TransportStatus::READY;
            }
            break;
        case UrmaStatus::PROCESS_DATA:
            if (IsConnsReady()) {
                urmaStatus_ = UrmaStatus::CONN_OK;
                SendFinish();
            }
            break;
        case UrmaStatus::CONN_OK:
            RecvFinish();
            urmaStatus_ = UrmaStatus::SEND_FIN;
            break;
        case UrmaStatus::SEND_FIN:
            urmaStatus_ = UrmaStatus::RECV_FIN;
            transportStatus_ = TransportStatus::READY;
            break;
        default:
            break;
    }
}

TransportStatus AivUrmaTransport::GetStatus()
{
    if (PrepareGetStatus()) {
        ProcessUrmaStatus();
    }
    return transportStatus_;
}

void AivUrmaTransport::GetHostChannelEntity(ChannelEntity* channelEntitiesHost)
{
    PrepareHostChannelEntity(channelEntitiesHost);
    GetSqContext();
    GetCqContext();
    channelEntitiesHost->sqNum = sqContextVec_.size();
    channelEntitiesHost->sqContextAddr = sqContextVec_.data();
    channelEntitiesHost->cqNum = cqContextVec_.size();
    channelEntitiesHost->cqContextAddr = cqContextVec_.data();

    HCCL_INFO(
        "localBufferNum[%u] localBufferAddr[0x%x] remoteBufferNum[%u] remoteBufferAddr[0x%x] sqNum[%u] "
        "sqContextAddr[0x%x] cqNum[%u] cqContextAddr[0x%x]",
        channelEntitiesHost->localBufferNum, channelEntitiesHost->localBufferAddr, channelEntitiesHost->remoteBufferNum,
        channelEntitiesHost->remoteBufferAddr, channelEntitiesHost->sqNum, channelEntitiesHost->sqContextAddr,
        channelEntitiesHost->cqNum, channelEntitiesHost->cqContextAddr);
}

} // namespace Hccl
