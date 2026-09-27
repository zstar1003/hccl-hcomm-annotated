/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "ccu_transport_.h"

#include "exception_handler.h"

#include "../../../../../legacy/ascend950/unified_platform/resource/mem/user_remote_mem_getter.h"

#include "env_config/env_config_v2.h"
#include "ccu_assist_v1.h"
#include "hccl_log_keywords.h"

namespace hcomm {

constexpr uint32_t FINISH_MSG_SIZE = 128;
constexpr char FINISH_MSG[FINISH_MSG_SIZE] = "Transport exchange data ready!";

HcclResult BuildCcuConnection(
    const CcuTransport::CcuConnectionInfo& ccuConnectionInfo, std::unique_ptr<CcuConnection>& ccuConnection)
{
    if (ccuConnectionInfo.type == CcuTransport::CcuConnectionType::UB_CTP) {
        ccuConnection.reset(new (std::nothrow) CcuCtpConnection(
            ccuConnectionInfo.locAddr, ccuConnectionInfo.rmtAddr, ccuConnectionInfo.channelInfo,
            ccuConnectionInfo.ccuJettys, ccuConnectionInfo.qos));
    } else {
        ccuConnection.reset(new (std::nothrow) CcuRtpConnection(
            ccuConnectionInfo.locAddr, ccuConnectionInfo.rmtAddr, ccuConnectionInfo.channelInfo,
            ccuConnectionInfo.ccuJettys, ccuConnectionInfo.qos));
    }
    CHK_PTR_NULL(ccuConnection);
    CHK_RET(ccuConnection->Init());
    return HCCL_SUCCESS;
}

HcclResult CcuCreateTransport(
    Hccl::Socket* socket, const CcuTransport::CcuConnectionInfo& ccuConnectionInfo,
    const CcuTransport::CclBufferInfo& cclBufferInfo, std::unique_ptr<CcuTransport>& ccuTransport)
{
    CHK_PTR_NULL(socket);
    std::unique_ptr<CcuConnection> ccuConnection{nullptr};
    CHK_RET(BuildCcuConnection(ccuConnectionInfo, ccuConnection));

    ccuTransport.reset(new (std::nothrow) CcuTransport(socket, std::move(ccuConnection), cclBufferInfo));
    CHK_PTR_NULL(ccuTransport);
    CHK_RET(ccuTransport->Init());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuCreateTransport(
    Hccl::Socket* socket, const CcuTransport::CcuConnectionInfo& ccuConnectionInfo,
    const std::vector<CcuTransport::CclBufferInfo>& bufferInfos, std::unique_ptr<CcuTransport>& ccuTransport)
{
    CHK_PTR_NULL(socket);
    std::unique_ptr<CcuConnection> ccuConnection{nullptr};
    CHK_RET(BuildCcuConnection(ccuConnectionInfo, ccuConnection));

    if (bufferInfos.size() == 0) {
        HCCL_ERROR("[CcuCreateTransport] bufferNum is 0.");
        return HCCL_E_PARA;
    }
    ccuTransport.reset(new (std::nothrow) CcuTransport(socket, std::move(ccuConnection), bufferInfos));
    CHK_PTR_NULL(ccuTransport);
    // 可能申请xn cke失败，需要回退
    auto ret = ccuTransport->Init();
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING("[%s] ccuTransport init failed, ccu transport resources unavailable.", __func__);
        return ret;
    }
    CHK_RET(ret);

    return HcclResult::HCCL_SUCCESS;
}

CcuTransport::CcuTransport(
    Hccl::Socket* socket, std::unique_ptr<CcuConnection>&& connection, const CclBufferInfo& locCclBufInfo)
    : socket_(socket),
      ccuConnection_(std::move(connection))
{
    locBufferInfos_.push_back(locCclBufInfo);
}

CcuTransport::CcuTransport(
    Hccl::Socket* socket, std::unique_ptr<CcuConnection>&& connection, const std::vector<CclBufferInfo>& bufferInfos)
    : socket_(socket),
      ccuConnection_(std::move(connection)),
      locBufferInfos_(bufferInfos)
{}

HcclResult CcuTransport::Init()
{
    dieId_ = ccuConnection_->GetDieId();
    userDevId_ = ccuConnection_->GetUserDevId();
    auto ret = AppendCkes(INIT_CKE_NUM);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        return ret;
    }
    CHK_RET(ret);

    ret = AppendXns(INIT_XN_NUM);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        return ret;
    }
    CHK_RET(ret);

    transStatus_ = TransStatus::INIT;
    // 正常建链路径在 Init 成功时确认本端资源充足；msg-only(资源不足)路径由
    // ConstructMsgOnlyTransport 单独覆盖为 UNAVAIL/FAILED，避免依赖调用方设置
    locResStatus_ = CcuResStatus::RES_OK;
    return HCCL_SUCCESS;
}

CcuTransport::TransStatus CcuTransport::GetStatus()
{
    if (transStatus_ == TransStatus::READY || transStatus_ == TransStatus::CONNECT_FAILED
        || transStatus_ == TransStatus::SOCKET_TIMEOUT) {
        return transStatus_;
    }

    if (StatusMachine() != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CcuTransport][%s] failed, %s.", __func__, transStatus_.Describe().c_str());
        transStatus_ = TransStatus::CONNECT_FAILED;
    }

    return transStatus_;
}

HcclResult CcuTransport::AppendCkes(uint32_t ckesNum)
{
    std::vector<ResInfo> resInfo;
    auto ret = CcuDevMgrImp::AllocCke(userDevId_, dieId_, ckesNum, resInfo);
    CHK_PRT_RET(
        ret == HcclResult::HCCL_E_UNAVAIL,
        HCCL_WARNING("[CcuTransport][%s] failed, the resource is not enough.", __func__), ret);
    CHK_RET(ret);

    const uint32_t resSize = resInfo.size();
    for (uint32_t i = 0; i < resSize; i++) {
        const uint32_t ckeNum = resInfo[i].num;
        const uint32_t ckesSartId = resInfo[i].startId;
        for (uint32_t j = 0; j < ckeNum; j++) {
            locRes_.ckes.emplace_back(ckesSartId + j);
        }
    }
    ckesRes_.push_back(resInfo);
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::AppendXns(uint32_t xnsNum)
{
    std::vector<ResInfo> resInfo;
    auto ret = CcuDevMgrImp::AllocXn(userDevId_, dieId_, xnsNum, resInfo);
    CHK_PRT_RET(
        ret == HcclResult::HCCL_E_UNAVAIL,
        HCCL_WARNING("[CcuTransport][%s] failed, the resource is not enough.", __func__), ret);
    CHK_RET(ret);

    const uint32_t resSize = resInfo.size();
    for (uint32_t i = 0; i < resSize; i++) {
        uint32_t xnNum = resInfo[i].num;
        uint32_t xnsSartId = resInfo[i].startId;
        for (uint32_t j = 0; j < xnNum; j++) {
            locRes_.xns.emplace_back(xnsSartId + j);
        }
    }
    xnsRes_.push_back(resInfo);
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::StatusMachine()
{
    EXCEPTION_HANDLE_BEGIN
    Hccl::SocketStatus socketStatus = socket_->GetAsyncStatus();
    if (socketStatus == Hccl::SocketStatus::INIT || socketStatus == Hccl::SocketStatus::TIMEOUT) {
        HCCL_ERROR("[CcuTransport][GetStatus] socket timeout or no link, please check");
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (socketStatus != Hccl::SocketStatus::OK) {
        return HcclResult::HCCL_SUCCESS; // 操作成功，保持当前状态
    }
    EXCEPTION_HANDLE_END

    switch (transStatus_) {
        case CcuTransport::TransStatus::INIT: {
            if (locResStatus_ != CcuResStatus::RES_OK) {
                // 本端资源不足或资源申请失败, 无ccuConnection_, 直接发状态标记
                CHK_RET(SendDataSize());
                transStatus_ = TransStatus::SEND_DATA_SIZE;
                break;
            }
            auto connStatus = ccuConnection_->GetStatus();
            if (connStatus == CcuConnStatus::CONN_INVALID) {
                HCCL_ERROR(
                    "[CcuTransport][GetStatus] connection status[%s] failed."
                    " please check.",
                    connStatus.Describe().c_str());
                return HcclResult::HCCL_E_INTERNAL;
            }

            if (connStatus == CcuConnStatus::EXCHANGEABLE || connStatus == CcuConnStatus::CONNECTED) {
                // connection完成本端资源创建或复用时，发送本端资源信息
                CHK_RET(SendDataSize());
                transStatus_ = TransStatus::SEND_DATA_SIZE;
            }

            // connection状态非错误但未达到目标状态时，transport保持当前状态
            break;
        }
        case CcuTransport::TransStatus::SEND_DATA_SIZE:
            CHK_RET(RecvDataSize());
            transStatus_ = TransStatus::RECV_DATA_SIZE;
            break;
        case CcuTransport::TransStatus::RECV_DATA_SIZE:
            CHK_RET(SendConnAndTransInfo());
            transStatus_ = TransStatus::SEND_ALL_INFO;
            break;
        case CcuTransport::TransStatus::SEND_ALL_INFO:
            CHK_RET(RecvConnAndTransInfo());
            transStatus_ = TransStatus::RECV_ALL_INFO;
            break;
        case CcuTransport::TransStatus::RECV_ALL_INFO:
            CHK_RET(RecvDataProcess());
            if (IsBothEndResSufficient()) {
                CHK_RET(ccuConnection_->ImportJetty());
            }
            transStatus_ = TransStatus::SEND_FIN;
            break;
        case CcuTransport::TransStatus::SEND_FIN: {
            if (!IsBothEndResSufficient()) {
                CHK_RET(SendFinish());
                transStatus_ = CcuTransport::TransStatus::RECVING_FIN;
                break;
            }
            auto connStatus = ccuConnection_->GetStatus();
            if (connStatus == CcuConnStatus::CONN_INVALID) {
                HCCL_ERROR(
                    "[CcuTransport][GetStatus] connection status[%s] failed."
                    " please check",
                    connStatus.Describe().c_str());
                return HcclResult::HCCL_E_INTERNAL;
            }

            if (connStatus == CcuConnStatus::CONNECTED) {
                CHK_RET(SendFinish());
                transStatus_ = CcuTransport::TransStatus::RECVING_FIN;
            }
            break;
        }
        case CcuTransport::TransStatus::RECVING_FIN:
            CHK_RET(RecvFinish());
            transStatus_ = CcuTransport::TransStatus::RECV_FIN;
            break;
        case CcuTransport::TransStatus::RECV_FIN:
            CHK_RET(CheckFinish());
            if (locResStatus_ != CcuResStatus::RES_OK || rmtResStatus_ != CcuResStatus::RES_OK) {
                HCCL_WARNING(
                    "[CcuTransport][%s] resource status not OK after FIN (loc[%u], rmt[%u]) set CONNECT_FAILED",
                    __func__, static_cast<uint8_t>(locResStatus_), static_cast<uint8_t>(rmtResStatus_));
                transStatus_ = TransStatus::CONNECT_FAILED;
                break;
            }
            transStatus_ = CcuTransport::TransStatus::READY;
            break;
        case CcuTransport::TransStatus::SEND_TRANS_RES:
            CHK_RET(SendTransInfo());
            transStatus_ = CcuTransport::TransStatus::RECVING_TRANS_RES;
            break;
        case CcuTransport::TransStatus::RECVING_TRANS_RES:
            CHK_RET(RecvTransInfo());
            transStatus_ = CcuTransport::TransStatus::RECV_TRANS_RES;
            break;
        case CcuTransport::TransStatus::RECV_TRANS_RES:
            CHK_RET(RecvTransInfoProcess());
            transStatus_ = CcuTransport::TransStatus::SEND_FIN;
            break;
        default:
            HCCL_ERROR("[CcuTransport][%s] failed, error status[%s].", __func__, transStatus_.Describe().c_str());
            transStatus_ = CcuTransport::TransStatus::CONNECT_FAILED;
            break;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::SendDataSize()
{
    Hccl::BinaryStream binaryStream;
    // 所有路径先写 locResStatus (int), 接收端据此分流
    int locResStatus = static_cast<int>(locResStatus_);
    binaryStream << locResStatus;
    if (locResStatus_ == CcuResStatus::RES_OK) {
        CHK_RET(HandshakeMsgPack(binaryStream));
        CHK_RET(ConnInfoPack(binaryStream));
        CHK_RET(TransResPack(binaryStream));
        CHK_RET(BufferInfoPack(binaryStream, locBufferInfos_));
    }
    binaryStream.Dump(sendData_);
    u32 sendSize = sendData_.size();

    // 发送数据包尺寸
    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(&sendSize, sizeof(sendSize));
    EXCEPTION_HANDLE_END
    HCCL_INFO(
        "[CcuTransport::%s] Send size[%u] of data success. [%zu] bytes sent. locResStatus[%u]", __func__, sendSize,
        sizeof(sendSize), static_cast<uint8_t>(locResStatus_));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::RecvDataSize()
{
    // 接收数据包尺寸
    EXCEPTION_HANDLE_BEGIN
    socket_->RecvAsync(ReinterpretAs<u8*>(&exchangeDataSize_), sizeof(exchangeDataSize_));
    EXCEPTION_HANDLE_END
    HCCL_INFO(
        "[CcuTransport::%s] Receive size[%u] of data success. [%zu] bytes received.", __func__, exchangeDataSize_,
        sizeof(exchangeDataSize_));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::SendConnAndTransInfo()
{
    // 当前socket失败会抛异常，需要统一整改
    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(sendData_.data(), sendData_.size());
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::RecvConnAndTransInfo()
{
    recvData_.resize(exchangeDataSize_);
    EXCEPTION_HANDLE_BEGIN
    socket_->RecvAsync(ReinterpretAs<u8*>(recvData_.data()), recvData_.size());
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::RecvDataProcess()
{
    if (exchangeDataSize_ < sizeof(int)) {
        HCCL_ERROR(
            "[CcuTransport][%s] exchangeDataSize[%u] < sizeof(int)[%zu], protocol mismatch.", __func__,
            exchangeDataSize_, sizeof(int));
        return HcclResult::HCCL_E_INTERNAL;
    }
    Hccl::BinaryStream binaryStream(recvData_);
    int rmtResStatusInt = static_cast<int>(CcuResStatus::RES_UNKNOWN);
    binaryStream >> rmtResStatusInt;
    // 校验对端传值范围：越界值强转后比较行为未定义，按协议错误处理
    if (rmtResStatusInt < static_cast<int>(CcuResStatus::RES_UNKNOWN)
        || rmtResStatusInt > static_cast<int>(CcuResStatus::RES_FAILED)) {
        HCCL_ERROR(
            "[CcuTransport][%s] invalid rmtResStatus[%d], out of range[%d,%d].", __func__, rmtResStatusInt,
            static_cast<int>(CcuResStatus::RES_UNKNOWN), static_cast<int>(CcuResStatus::RES_FAILED));
        return HcclResult::HCCL_E_INTERNAL;
    }
    rmtResStatus_ = static_cast<CcuResStatus>(rmtResStatusInt);

    if (rmtResStatus_ != CcuResStatus::RES_OK) {
        HCCL_WARNING(
            "[CcuTransport][%s] remote resource status[%u], skip rest unpack.", __func__,
            static_cast<uint8_t>(rmtResStatus_));
        return HcclResult::HCCL_SUCCESS;
    }
    if (locResStatus_ != CcuResStatus::RES_OK) {
        HCCL_WARNING(
            "[CcuTransport][%s] loc resource status[%u], skip remote full unpack.", __func__,
            static_cast<uint8_t>(locResStatus_));
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_RET(HandshakeMsgUnpack(binaryStream));
    CHK_RET(ConnInfoUnpackProc(binaryStream));
    CHK_RET(TransResUnpackProc(binaryStream));
    rmtBufferVec_.clear();
    CHK_RET(BufferInfoUnpack(binaryStream));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::SendTransInfo()
{
    Hccl::BinaryStream binaryStream;
    TransResPack(binaryStream);
    binaryStream.Dump(sendTrans_);
    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(sendTrans_.data(), sendTrans_.size());
    EXCEPTION_HANDLE_END
    exchangeDataSize_ = sendTrans_.size();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::RecvTransInfo()
{
    recvTrans_.resize(exchangeDataSize_);
    EXCEPTION_HANDLE_BEGIN
    socket_->RecvAsync(ReinterpretAs<u8*>(recvTrans_.data()), recvTrans_.size());
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::RecvTransInfoProcess()
{
    Hccl::BinaryStream binaryStream(recvTrans_);
    TransResUnpackProc(binaryStream);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::HandshakeMsgPack(Hccl::BinaryStream& binaryStream)
{
    binaryStream << attr_.handshakeMsg;
    HCCL_INFO(
        "[CcuTransport][%s] start pack handshakeMsg, attr.handshakeMsg.size[%zu]", __func__, attr_.handshakeMsg.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::ConnInfoPack(Hccl::BinaryStream& binaryStream) const
{
    std::vector<char> dtoData{};
    CHK_RET(ccuConnection_->Serialize(dtoData));
    binaryStream << dtoData;
    HCCL_INFO("[CcuTransport][%s] start pack connInfo, dtoData.size[%zu]", __func__, dtoData.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::TransResPack(Hccl::BinaryStream& binaryStream)
{
    const uint32_t locCkesSize = locRes_.ckes.size();
    binaryStream << locCkesSize;
    const uint32_t locCkeSize = locRes_.ckes.size();
    for (uint32_t i = 0; i < locCkeSize; i++) {
        binaryStream << locRes_.ckes[i];
    }

    const uint32_t locXnsSize = locRes_.xns.size();
    binaryStream << locXnsSize;
    for (uint32_t i = 0; i < locRes_.xns.size(); i++) {
        binaryStream << locRes_.xns[i];
    }

    HCCL_INFO("Send ckesSize[%u], xnsSize[%u]", locCkesSize, locXnsSize);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::BufferInfoPack(Hccl::BinaryStream& binaryStream, std::vector<CclBufferInfo>& bufferVec) const
{
    u32 locBufferNum = bufferVec.size();
    binaryStream << locBufferNum;
    for (u32 pos = 0; pos < locBufferNum; ++pos) {
        bufferVec[pos].Pack(binaryStream);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::HandshakeMsgUnpack(Hccl::BinaryStream& binaryStream)
{
    binaryStream >> rmtHandshakeMsg_;

    if (attr_.handshakeMsg.size() != rmtHandshakeMsg_.size()) {
        HCCL_ERROR("handshakeMsg size=%zu is not equal to rmt=%zu", attr_.handshakeMsg.size(), rmtHandshakeMsg_.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    HCCL_INFO("[CcuTransport][%s] unpack handshakeMsg, rmtHandshakeMsg.size[%zu]", __func__, rmtHandshakeMsg_.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::ConnInfoUnpackProc(Hccl::BinaryStream& binaryStream) const
{
    std::vector<char> dtoData{};
    binaryStream >> dtoData;
    CHK_RET(ccuConnection_->Deserialize(dtoData));
    HCCL_INFO("[CcuTransport][%s] start unpack connInfo, dtoData.size[%zu]", __func__, dtoData.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::TransResUnpackProc(Hccl::BinaryStream& binaryStream)
{
    uint32_t resSize{0};
    binaryStream >> resSize;
    rmtRes_.ckes.clear();
    for (uint32_t i = 0; i < resSize; i++) {
        uint32_t cke{0};
        binaryStream >> cke;
        rmtRes_.ckes.push_back(cke);
    }
    HCCL_INFO("Recv ckesSize[%u]", resSize);

    binaryStream >> resSize;
    rmtRes_.xns.clear();
    for (uint32_t i = 0; i < resSize; i++) {
        uint32_t xn{0};
        binaryStream >> xn;
        rmtRes_.xns.push_back(xn);
    }
    HCCL_INFO("Recv xnsSize[%u]", resSize);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::BufferInfoUnpack(Hccl::BinaryStream& binaryStream)
{
    u32 rmtBufferNum{0};
    binaryStream >> rmtBufferNum;
    CHK_PRT_RET(
        rmtBufferNum == 0 || rmtBufferNum > MAX_BUFFER_NUM,
        HCCL_ERROR(
            "[CcuTransport][BufferInfoUnpack] rmtBufferNum[%u] is zero or exceeds limit[%u]", rmtBufferNum,
            MAX_BUFFER_NUM),
        HCCL_E_PARA);
    HCCL_INFO("[CcuTransport][BufferInfoUnpack] rmtBufferNum[%u]", rmtBufferNum);
    for (u32 pos = 0; pos < rmtBufferNum; ++pos) {
        CclBufferInfo rmtBufferInfo{};
        rmtBufferInfo.Unpack(binaryStream);
        std::string memInfo(rmtBufferInfo.memInfo.data(), strnlen(rmtBufferInfo.memInfo.data(), HCCL_RES_TAG_MAX_LEN));
        if (memInfo == "HcclBuffer") {
            rmtHcclBufferInfo_ = rmtBufferInfo;
        }
        rmtBufferVec_.push_back(std::make_unique<Hccl::RemoteUbRmaBuffer>(
            ReinterpretAs<uintptr_t>(rmtBufferInfo.addr), rmtBufferInfo.size, rmtBufferInfo.tokenId,
            rmtBufferInfo.tokenValue, Hccl::CommMemTypeToHcclMemType(rmtBufferInfo.type), memInfo));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::SendFinish()
{
    sendFinishMsg_ = std::vector<char>(FINISH_MSG, FINISH_MSG + FINISH_MSG_SIZE);
    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(sendFinishMsg_.data(), FINISH_MSG_SIZE);
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::RecvFinish()
{
    recvFinishMsg_.resize(FINISH_MSG_SIZE);
    EXCEPTION_HANDLE_BEGIN
    socket_->RecvAsync(ReinterpretAs<u8*>(recvFinishMsg_.data()), FINISH_MSG_SIZE);
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::CheckFinish()
{
    const std::string sendFinishMsgStr(sendFinishMsg_.begin(), sendFinishMsg_.end());
    const std::string recvFinishMsgStr(recvFinishMsg_.begin(), recvFinishMsg_.end());
    if (sendFinishMsgStr != recvFinishMsgStr) {
        HCCL_ERROR(
            "[CcuTransport][RecvFinish]msgRecv[%s] and msgSend[%s] are not equal", recvFinishMsgStr.c_str(),
            sendFinishMsgStr.c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::ReleaseTransRes()
{
    for (uint32_t i = 0; i < ckesRes_.size(); i++) {
        if (ckesRes_[i].empty()) {
            continue;
        }
        auto ret = CcuDevMgrImp::ReleaseCke(userDevId_, dieId_, ckesRes_[i]);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CcuTransport][%s] release ckes failed but passed, "
                "userDevId[%d] dieId[%u].",
                __func__, userDevId_, dieId_);
        }
    }
    ckesRes_.clear();

    for (uint32_t i = 0; i < xnsRes_.size(); i++) {
        if (xnsRes_[i].empty()) {
            continue;
        }
        auto ret = CcuDevMgrImp::ReleaseXn(userDevId_, dieId_, xnsRes_[i]);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CcuTransport][%s] release xns failed but passed, "
                "userDevId[%d] dieId[%u].",
                __func__, userDevId_, dieId_);
        }
    }
    xnsRes_.clear();

    return HcclResult::HCCL_SUCCESS;
}

uint32_t CcuTransport::GetDieId() const { return dieId_; }

uint32_t CcuTransport::GetChannelId() const
{
    // msg-only(资源不足)transport 无 ccuConnection_，返回无效 id，避免空指针解引用
    if (ccuConnection_ == nullptr) {
        return UINT32_MAX;
    }
    return ccuConnection_->GetChannelId();
}

HcclResult CcuTransport::GetLocCkeByIndex(const uint32_t index, uint32_t& locCkeId) const
{
    CHK_PRT_RET(
        locRes_.ckes.empty(), HCCL_ERROR("[CcuTransport][%s] failed, local resources is empty.", __func__),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        index >= locRes_.ckes.size(),
        HCCL_ERROR(
            "[CcuTransport][%s] failed, index[%u] is larger than size[%u].", __func__, index, locRes_.ckes.size()),
        HcclResult::HCCL_E_PARA);

    locCkeId = locRes_.ckes[index];
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::GetLocXnByIndex(const uint32_t index, uint32_t& locXnId) const
{
    CHK_PRT_RET(
        locRes_.xns.empty(), HCCL_ERROR("[CcuTransport][%s] failed, local resources is empty.", __func__),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        index >= locRes_.xns.size(),
        HCCL_ERROR(
            "[CcuTransport][%s] failed, index[%u] is larger than size[%u].", __func__, index, locRes_.xns.size()),
        HcclResult::HCCL_E_PARA);

    locXnId = locRes_.xns[index];
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::GetRmtCkeByIndex(const uint32_t index, uint32_t& rmtCkeId) const
{
    CHK_PRT_RET(
        rmtRes_.ckes.empty(), HCCL_ERROR("[CcuTransport][%s] failed, local resources is empty.", __func__),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        index >= rmtRes_.ckes.size(),
        HCCL_ERROR(
            "[CcuTransport][%s] failed, index[%u] is larger than size[%u].", __func__, index, rmtRes_.ckes.size()),
        HcclResult::HCCL_E_PARA);

    rmtCkeId = rmtRes_.ckes[index];
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::GetRmtXnByIndex(const uint32_t index, uint32_t& rmtXnId) const
{
    CHK_PRT_RET(
        rmtRes_.xns.empty(), HCCL_ERROR("[CcuTransport][%s] failed, local resources is empty.", __func__),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        index >= rmtRes_.xns.size(),
        HCCL_ERROR(
            "[CcuTransport][%s] failed, index[%u] is larger than size[%u].", __func__, index, rmtRes_.xns.size()),
        HcclResult::HCCL_E_PARA);

    rmtXnId = rmtRes_.xns[index];
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::GetLocBuffer(CclBufferInfo& bufferInfo, const uint32_t& bufNum) const
{
    (void)bufNum;
    bufferInfo = locBufferInfos_[0];
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::GetRmtBuffer(CclBufferInfo& bufferInfo, const uint32_t& bufNum) const
{
    (void)bufNum;
    bufferInfo = rmtHcclBufferInfo_;
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::GetCkeNum(uint32_t& ckeNum) const
{
    ckeNum = locRes_.ckes.size();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTransport::GetRmtSignalAddrByIndex(uint32_t index, uint64_t& rmtCkeAddr) const
{
    uint32_t rmtCkeId{0};
    uint64_t ckeOffsetCcumAddr{0};
    CHK_RET(GetRmtCkeByIndex(index, rmtCkeId));
    CHK_PRT_RET(
        CcuDevMgrImp::GetCkeOffsetCcumAddrById(userDevId_, dieId_, rmtCkeId, ckeOffsetCcumAddr),
        HCCL_ERROR(
            "[CcuTransport][%s] Failed to get cke offset address. userDevId = %d, dieId = %u.", __func__, userDevId_,
            dieId_),
        HCCL_E_INTERNAL);
    uint64_t rmtResourceAddr = ccuConnection_->GetRmtCcuBufAddr();
    HCCL_DEBUG(
        "[CcuTransport][%s] index[%u] rmtCcuBufAddr[0x%llx], ckeAddr[%u][0x%llx]", __func__, index, rmtResourceAddr,
        rmtCkeId, ckeOffsetCcumAddr);
    if (ckeOffsetCcumAddr > UINT64_MAX - rmtResourceAddr) {
        HCCL_ERROR(
            "[CcuTransport][%s] failed, rmtResourceAddr[%llx] + ckeOffsetCcumAddr[%llx] is overflow.", __func__,
            rmtResourceAddr, ckeOffsetCcumAddr);
        return HcclResult::HCCL_E_INTERNAL;
    }
    rmtCkeAddr = rmtResourceAddr + ckeOffsetCcumAddr;
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::GetRmtVarAddrByIndex(uint32_t index, uint64_t& rmtXnAddr) const
{
    uint32_t rmtXnId{0};
    CHK_RET(GetRmtXnByIndex(index, rmtXnId));
    CHK_RET(GetRmtVarAddrByXnId(rmtXnId, rmtXnAddr));
    HCCL_DEBUG("[CcuTransport][%s] index[%u], xnAddr[%u][0x%llx]", __func__, index, rmtXnId, rmtXnAddr);
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::GetRmtVarAddrByXnId(const uint32_t rmtXnId, uint64_t& rmtXnAddr) const
{
    uint64_t xnOffsetCcumAddr = 0;
    CHK_PRT_RET(
        CcuDevMgrImp::GetXnOffsetCcumAddrById(userDevId_, dieId_, rmtXnId, xnOffsetCcumAddr),
        HCCL_ERROR(
            "[CcuTransport][%s] Failed to get xn offset address. userDevId = %d, dieId = %u.", __func__, userDevId_,
            dieId_),
        HCCL_E_INTERNAL);
    const uint64_t rmtResourceAddr = ccuConnection_->GetRmtCcuBufAddr();
    HCCL_DEBUG(
        "[CcuTransport][%s]rmtCcuBufAddr[0x%llx], xnAddr[%u][0x%llx]", __func__, rmtResourceAddr, rmtXnId,
        xnOffsetCcumAddr);
    if (rmtResourceAddr > UINT64_MAX - xnOffsetCcumAddr) {
        HCCL_ERROR(
            "[CcuTransport][%s] failed, CCU resource base address[%llu] is "
            "greater than expected, ccu xn offset[%llu], their sum will exceed the range "
            "of uint64_t.",
            __func__, rmtResourceAddr, xnOffsetCcumAddr);
        return HCCL_E_INTERNAL;
    }
    rmtXnAddr = rmtResourceAddr + xnOffsetCcumAddr;
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::GetRmtCcuBufferTokenInfo(uint32_t& rmtTokenId, uint32_t& rmtTokenValue) const
{
    rmtTokenId = ccuConnection_->GetRmtCcuBufTokenId();
    rmtTokenValue = ccuConnection_->GetRmtCcuBufTokenValue();
    return HcclResult::HCCL_SUCCESS;
}

CcuTransport::~CcuTransport() { (void)ReleaseTransRes(); }

std::string CcuTransport::Describe() const
{
    std::string description = "";

    description = Hccl::StringFormat("DieId: %u, ", dieId_);
    description += transStatus_.Describe();
    description += Hccl::StringFormat(", LocRes: {%u Ckes, %u Xns}, ", locRes_.ckes.size(), locRes_.xns.size());
    description += Hccl::StringFormat("RmtRes: {%u Ckes, %u Xns}, ", rmtRes_.ckes.size(), rmtRes_.xns.size());
    description += Hccl::StringFormat("CkesRes size: %u, ", ckesRes_.size());
    description += Hccl::StringFormat("XnsRes size: %u.", xnsRes_.size());
    return description;
}

HcclResult CcuTransport::Describe(std::string& dfxMsg)
{
    CHK_RET(ccuConnection_->Describe(dfxMsg));
    return HcclResult::HCCL_SUCCESS;
}

void CcuTransport::Clean()
{
    transStatus_ = TransStatus::INIT;
    sendData_.clear();
    // msg-only(资源不足)transport 无 ccuConnection_，直接跳过 connection 清理
    if (ccuConnection_ != nullptr) {
        ccuConnection_->Clean();
    }
}

HcclResult CcuTransport::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    Hccl::RemoteMemCtx<std::unique_ptr<Hccl::RemoteUbRmaBuffer>> remoteMemCtx{
        cacheValid_, rmtBufferVec_, remoteUserMems_, memInfoCopies_, memInfoPointers_, remoteMem, memInfos, memNum};
    CHK_RET(Hccl::GetRemoteUserMems(remoteMemCtx));
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::CheckSocketStatus()
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
            HCCL_INFO("[CcuTransport][%s] success, elapsed[%lld]ms, retryCount[%u]", __func__, elapsed, retryCount);
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout || socketStatus == Hccl::SocketStatus::TIMEOUT) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_ERROR(
                "[%s][%s] wait socket establish timeout, channel connect timeout after %lld sec, "
                "elapsed[%lld]ms, retryCount[%u]",
                LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_TIMEOUT.c_str(), timeout.count(), elapsed, retryCount);
            return HCCL_E_TIMEOUT;
        }
        EXCEPTION_HANDLE_END
        retryCount++;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::UpdateMemInfo(std::vector<CcuTransport::CclBufferInfo>& bufferVecTemp)
{
    if (bufferVecTemp.size() == 0) {
        HCCL_WARNING("[CcuTransport][UpdateMemInfo] bufferNum is 0.");
        return HCCL_SUCCESS;
    }
    uint32_t totalBufferNum = locBufferInfos_.size() + bufferVecTemp.size();
    if (UNLIKELY(totalBufferNum > MAX_BUFFER_NUM)) {
        HCCL_ERROR(
            "[CcuTransport][UpdateMemInfo] totalBufferNum[%u] exceeds limit[%u]", totalBufferNum, MAX_BUFFER_NUM);
        return HCCL_E_PARA;
    }
    HCCL_INFO("[CcuTransport][UpdateMemInfo] bufferNum[%zu]", bufferVecTemp.size());
    sendData_.clear();
    Hccl::BinaryStream sendStream;
    CHK_RET(BufferInfoPack(sendStream, bufferVecTemp));
    sendStream.Dump(sendData_);
    u32 sendSize = sendData_.size();
    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(&sendSize, sizeof(sendSize));
    EXCEPTION_HANDLE_END
    HCCL_INFO(
        "[CcuTransport][UpdateMemInfo] Send size[%u] of data success. [%zu] bytes sent.", sendSize, sizeof(sendSize));
    CHK_RET(CheckSocketStatus());
    CHK_RET(RecvDataSize());
    CHK_RET(CheckSocketStatus());
    CHK_RET(SendConnAndTransInfo());
    CHK_RET(CheckSocketStatus());
    CHK_RET(RecvConnAndTransInfo());
    CHK_RET(CheckSocketStatus());
    Hccl::BinaryStream recvStream(recvData_);
    CHK_RET(BufferInfoUnpack(recvStream));
    locBufferInfos_.insert(locBufferInfos_.end(), bufferVecTemp.begin(), bufferVecTemp.end());
    // 流程中已有新增内存数量判断，故执行到此位置一定存在新增内存，需要将标识置位false，使得再次调用GetRemoteMems时重新构造缓存
    cacheValid_ = false;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuTransport::ConstructMsgOnlyTransport(Hccl::Socket* socket, std::unique_ptr<CcuTransport>& impl, CcuResStatus status)
{
    HCCL_INFO("[CcuTransport][%s] construct msg-only transport, status[%u].", __func__, static_cast<uint8_t>(status));
    CHK_PTR_NULL(socket);
    std::vector<CclBufferInfo> emptyBufferInfos{};
    EXCEPTION_CATCH((impl = std::make_unique<CcuTransport>(socket, nullptr, emptyBufferInfos)), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(impl);
    impl->locResStatus_ = status;
    impl->transStatus_ = CcuTransport::TransStatus::INIT;
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::CcuGetRmtMemToken(uint64_t srcVa, uint64_t& tokenInfo)
{
    for (auto& bufferInfo : rmtBufferVec_) {
        if (bufferInfo->GetAddr() == srcVa) {
            tokenInfo = CcuRep::CcuCombineTokenInfo(bufferInfo->GetTokenId(), bufferInfo->GetTokenValue(), 1);
            HCCL_INFO("[CcuTransport][CcuGetRmtMemToken] srcVa[%llu].", srcVa);
            return HcclResult::HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("[CcuTransport][CcuGetRmtMemToken] srcVa[%llu] not found.", srcVa);
    return HcclResult::HCCL_E_NOT_FOUND;
}
} // namespace hcomm
