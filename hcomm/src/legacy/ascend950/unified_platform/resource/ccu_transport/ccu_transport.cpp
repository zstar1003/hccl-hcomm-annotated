/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_transport.h"
#include "coll_operator_check.h"
#include "exception_util.h"

namespace Hccl {

constexpr uint32_t FINISH_MSG_SIZE = 128;
constexpr char_t FINISH_MSG[FINISH_MSG_SIZE] = "Transport exchange data ready!";

HcclResult CcuCreateTransport(
    Socket* socket, const CcuTransport::CcuConnectionInfo& ccuConnectionInfo,
    const CcuTransport::CclBufferInfo& cclBufferInfo, std::unique_ptr<CcuTransport>& ccuTransport)
{
    CHK_PTR_NULL(socket);
    HCCL_INFO(
        "[%s]ccuConnectionInfo type[%d], locAddr[%s], rmtAddr[%s], channelInfo[channelId %u:dieId %u], "
        "cclBufferInfo addr[%llu], size[%u]",
        __func__, ccuConnectionInfo.type, ccuConnectionInfo.locAddr.GetIpStr().c_str(),
        ccuConnectionInfo.rmtAddr.GetIpStr().c_str(), ccuConnectionInfo.channelInfo.channelId,
        ccuConnectionInfo.channelInfo.dieId, cclBufferInfo.addr, cclBufferInfo.size);
    TRY_CATCH_RETURN(
        std::unique_ptr<CcuConnection> ccuConnection;
        if (ccuConnectionInfo.type == CcuTransport::CcuConnectionType::UB_CTP) {
            ccuConnection = std::make_unique<CcuCtpConnection>(
                ccuConnectionInfo.locAddr, ccuConnectionInfo.rmtAddr, ccuConnectionInfo.channelInfo,
                ccuConnectionInfo.ccuJettys);
        } else {
            ccuConnection = std::make_unique<CcuTpConnection>(
                ccuConnectionInfo.locAddr, ccuConnectionInfo.rmtAddr, ccuConnectionInfo.channelInfo,
                ccuConnectionInfo.ccuJettys);
        }

        auto ret
        = ccuConnection->Init();
        if (ret != HcclResult::HCCL_SUCCESS) {
            ccuConnection = nullptr;
            return ret;
        }

        ccuTransport
        = std::make_unique<CcuTransport>(socket, std::move(ccuConnection), cclBufferInfo);
        ret = ccuTransport->Init(); if (ret != HcclResult::HCCL_SUCCESS) {
            ccuTransport = nullptr;
            return ret;
        });

    return HcclResult::HCCL_SUCCESS;
}

CcuTransport::CcuTransport(
    Socket* socket, std::unique_ptr<CcuConnection>&& connection, const CclBufferInfo& locCclBufInfo)
    : socket(socket),
      ccuConnection(std::move(connection)),
      locCclBufInfo(locCclBufInfo)
{}

HcclResult CcuTransport::Init()
{
    dieId = ccuConnection->GetDieId();
    devLogicId = ccuConnection->GetDevLogicId();
    auto ret = AppendCkes(INIT_CKE_NUM);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING("[AppendCkes] UNAVAIL.");
        return ret;
    }

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("errNo[0x%016llx]:AppendCkes failed.", ret);
        return ret;
    }
    ret = AppendXns(INIT_XN_NUM);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING("[AppendXns] UNAVAIL.");
        return ret;
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("errNo[0x%016llx]:AppendXns failed.", ret);
        return ret;
    }
    transStatus = TransStatus::INIT;
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::AppendRes(uint32_t ckesNum, uint32_t xnsNum)
{
    try {
        std::unique_lock<std::shared_timed_mutex> lock(transMutex);
        auto ret = AppendCkes(ckesNum);
        if (ret == HcclResult::HCCL_E_UNAVAIL) {
            HCCL_WARNING("[AppendCkes] UNAVAIL.");
            return ret;
        }

        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("errNo[0x%016llx]:AppendCkes failed.", ret);
            return ret;
        }
        ret = AppendXns(xnsNum);
        if (ret == HcclResult::HCCL_E_UNAVAIL) {
            HCCL_WARNING("[AppendXns] UNAVAIL.");
            return ret;
        }

        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("errNo[0x%016llx]:AppendXns failed.", ret);
            return ret;
        }
        transStatus = CcuTransport::TransStatus::SEND_TRANS_RES;
    } catch (HcclException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_INTERNAL;
    } catch (exception& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("Unknown error occurred during unimport jetty or destroy jetty!");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::AppendCkes(uint32_t ckesNum)
{
    vector<ResInfo> resInfo;
    auto ret = CcuDeviceManager::AllocCke(devLogicId, dieId, ckesNum, resInfo);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING("[AppendCkes] UNAVAIL.");
        return ret;
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("errNo[0x%016llx]:AppendCkes failed.", ret);
        return ret;
    }

    for (uint32_t i = 0; i < resInfo.size(); i++) {
        uint32_t ckeNum = resInfo[i].num;
        uint32_t ckesSartId = resInfo[i].startId;
        for (uint32_t j = 0; j < ckeNum; j++) {
            locRes.ckes.push_back(ckesSartId + j);
        }
    }
    ckesRes.push_back(resInfo);
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::AppendXns(uint32_t xnsNum)
{
    vector<ResInfo> resInfo;
    auto ret = CcuDeviceManager::AllocXn(devLogicId, dieId, xnsNum, resInfo);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING("[AppendXns] UNAVAIL.");
        return ret;
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("errNo[0x%016llx]:AppendXns failed.", ret);
        return ret;
    }
    for (uint32_t i = 0; i < resInfo.size(); i++) {
        uint32_t xnNum = resInfo[i].num;
        uint32_t xnsSartId = resInfo[i].startId;
        for (uint32_t j = 0; j < xnNum; j++) {
            locRes.xns.push_back(xnsSartId + j);
        }
    }
    xnsRes.push_back(resInfo);
    return HCCL_SUCCESS;
}

void CcuTransport::SetCntCke(const vector<uint32_t>& cntCke)
{
    HCCL_INFO("[%s]cntCke size[%llu]", __func__, cntCke.size());
    locRes.cntCkes = cntCke;
}

CcuTransport::TransStatus CcuTransport::StateMachine()
{
    if (transStatus == TransStatus::READY) {
        return transStatus;
    }
    SocketStatus socketStatus = socket->GetAsyncStatus();
    if (socketStatus == SocketStatus::INIT) {
        THROW<InternalException>("[CcuTransport][GetStatus]socket timeout or no link, please check");
    }

    if (socketStatus == SocketStatus::TIMEOUT) {
        return TransStatus::SOCKET_TIMEOUT;
    }

    if (socketStatus != SocketStatus::OK) {
        return transStatus;
    }

    switch (transStatus) {
        case CcuTransport::TransStatus::INIT: {
            auto connStatus = ccuConnection->GetStatus();
            if (connStatus == CcuConnStatus::CONN_INVALID) {
                THROW<InternalException>(
                    "[CcuTransport][GetStatus] connection status[%s] failed."
                    " please check",
                    connStatus.Describe().c_str());
            }

            if (connStatus == CcuConnStatus::EXCHANGEABLE || connStatus == CcuConnStatus::CONNECTED) {
                transStatus = CcuTransport::TransStatus::SEND_ALL_INFO;
                SendConnAndTransInfo();
            }

            break;
        }
        case CcuTransport::TransStatus::SEND_ALL_INFO:
            transStatus = CcuTransport::TransStatus::RECV_ALL_INFO;
            RecvConnAndTransInfo();
            break;
        case CcuTransport::TransStatus::RECV_ALL_INFO:
            RecvDataProcess();
            ccuConnection->ImportJetty();
            transStatus = CcuTransport::TransStatus::SEND_FIN;
            break;
        case CcuTransport::TransStatus::SEND_FIN: {
            auto connStatus = ccuConnection->GetStatus();
            if (connStatus == CcuConnStatus::CONN_INVALID) {
                THROW<InternalException>(
                    "[CcuTransport][GetStatus] connection status[%s] failed."
                    " please check",
                    connStatus.Describe().c_str());
            }

            if (connStatus == CcuConnStatus::CONNECTED) {
                SendFinish();
                transStatus = CcuTransport::TransStatus::RECVING_FIN;
            }
            break;
        }
        case CcuTransport::TransStatus::RECVING_FIN:
            RecvFinish();
            transStatus = CcuTransport::TransStatus::RECV_FIN;
            break;
        case CcuTransport::TransStatus::RECV_FIN:
            CheckFinish();
            transStatus = CcuTransport::TransStatus::READY;
            break;
        case CcuTransport::TransStatus::SEND_TRANS_RES:
            SendTransInfo();
            transStatus = CcuTransport::TransStatus::RECVING_TRANS_RES;
            break;
        case CcuTransport::TransStatus::RECVING_TRANS_RES:
            RecvTransInfo();
            transStatus = CcuTransport::TransStatus::RECV_TRANS_RES;
            break;
        case CcuTransport::TransStatus::RECV_TRANS_RES:
            RecvTransInfoProcess();
            transStatus = CcuTransport::TransStatus::SEND_FIN;
            break;
        default:
            THROW<InternalException>("[CcuTransport][GetStatus]failed status");
            break;
    }
    return transStatus;
}

CcuTransport::TransStatus CcuTransport::GetStatus()
{
    CcuTransport::TransStatus status = CcuTransport::TransStatus::CONNECT_FAILED;
    auto lockAndStatuMachine = [&]() {
        std::unique_lock<std::shared_timed_mutex> lock(transMutex);
        status = StateMachine();
    };
    TRY_CATCH_PROCESS_THROW(
        InternalException, lockAndStatuMachine(), "CcuTransport GetStatus() Error when creating transport connection",
        { transStatus = CcuTransport::TransStatus::CONNECT_FAILED; });
    return status;
}

void CcuTransport::SendConnAndTransInfo()
{
    BinaryStream binaryStream;
    HandshakeMsgPack(binaryStream);
    ConnInfoPack(binaryStream);
    TransResPack(binaryStream);
    CclBufferInfoPack(binaryStream);
    binaryStream.Dump(sendData);
    socket->SendAsync(sendData.data(), sendData.size());
    exchangeDataSize = sendData.size();
}

void CcuTransport::RecvConnAndTransInfo()
{
    recvData.resize(exchangeDataSize);
    socket->RecvAsync(reinterpret_cast<u8*>(recvData.data()), recvData.size());
}

void CcuTransport::RecvDataProcess()
{
    BinaryStream binaryStream(recvData);
    HandshakeMsgUnpack(binaryStream);
    ConnInfoUnpackProc(binaryStream);
    TransResUnpackProc(binaryStream);
    CclBufferInfoUnpack(binaryStream);
}

void CcuTransport::SendTransInfo()
{
    BinaryStream binaryStream;
    TransResPack(binaryStream);
    binaryStream.Dump(sendTrans);
    socket->SendAsync(sendTrans.data(), sendTrans.size());
    exchangeDataSize = sendTrans.size();
}

void CcuTransport::RecvTransInfo()
{
    recvTrans.resize(exchangeDataSize);
    socket->RecvAsync(reinterpret_cast<u8*>(recvTrans.data()), recvTrans.size());
}

void CcuTransport::RecvTransInfoProcess()
{
    BinaryStream binaryStream(recvTrans);
    TransResUnpackProc(binaryStream);
}

void CcuTransport::HandshakeMsgPack(BinaryStream& binaryStream)
{
    binaryStream << static_cast<u32>(attr.opAcceState);
    binaryStream << attr.handshakeMsg;
    HCCL_INFO(
        "[CcuTransport][%s] start pack handshakeMsg, attr.handshakeMsg.size[%zu]", __func__, attr.handshakeMsg.size());
}

void CcuTransport::ConnInfoPack(BinaryStream& binaryStream) const
{
    std::vector<char> dtoData{};
    ccuConnection->Serialize(dtoData);
    binaryStream << dtoData;
    HCCL_INFO("[CcuTransport][%s] start pack connInfo, dtoData.size[%u]", __func__, dtoData.size());
}

void CcuTransport::TransResPack(BinaryStream& binaryStream)
{
    uint32_t locCkesSize = locRes.ckes.size();
    binaryStream << locCkesSize;
    for (uint32_t i = 0; i < locRes.ckes.size(); i++) {
        binaryStream << locRes.ckes[i];
    }

    uint32_t locCntCkesSize = locRes.cntCkes.size();
    binaryStream << locCntCkesSize;
    for (uint32_t i = 0; i < locRes.cntCkes.size(); i++) {
        binaryStream << locRes.cntCkes[i];
    }

    uint32_t locXnsSize = locRes.xns.size();
    binaryStream << locXnsSize;
    for (uint32_t i = 0; i < locRes.xns.size(); i++) {
        binaryStream << locRes.xns[i];
    }
    HCCL_INFO("Send ckesSize[%u], cntCkesSize[%u], xnsSize[%u]", locCkesSize, locCntCkesSize, locXnsSize);
}

void CcuTransport::CclBufferInfoPack(BinaryStream& binaryStream) const { locCclBufInfo.Pack(binaryStream); }

void CcuTransport::HandshakeMsgUnpack(BinaryStream& binaryStream)
{
    u32 rmtAccelerator{0};
    binaryStream >> rmtAccelerator;
    HCCL_INFO("[CcuTransport::HandshakeMsgUnpack], rmtAccelerator[%u]", rmtAccelerator);
    rmtOpAcceState = static_cast<AcceleratorState::Value>(rmtAccelerator);

    if (rmtOpAcceState != attr.opAcceState) {
        THROW<InvalidParamsException>(StringFormat(
            "[CcuTransport::HandshakeMsgUnpack] Accelerator information check fail. "
            "locOpAccelerator[%s], rmtOpAccelerator[%s]",
            attr.opAcceState.Describe().c_str(), rmtOpAcceState.Describe().c_str()));
    }

    rmtHandshakeMsg.clear();
    binaryStream >> rmtHandshakeMsg;

    if (attr.handshakeMsg.size() != rmtHandshakeMsg.size()) {
        MACRO_THROW(
            InvalidParamsException,
            StringFormat(
                "handshakeMsg size=%u is not equal to rmt=%u", attr.handshakeMsg.size(), rmtHandshakeMsg.size()));
    }

    auto localCollOperator = CollOperator::GetPackedData(attr.handshakeMsg);
    auto remoteCollOperator = CollOperator::GetPackedData(rmtHandshakeMsg);
    CheckCollOperator(localCollOperator, remoteCollOperator); // 两端算子参数一致性校验

    HCCL_INFO("[CcuTransport][%s] start unpack handshakeMsg", __func__);
}

void CcuTransport::ConnInfoUnpackProc(BinaryStream& binaryStream) const
{
    std::vector<char> dtoData{};
    binaryStream >> dtoData;
    ccuConnection->Deserialize(dtoData);
    HCCL_INFO("[CcuTransport][%s] start unpack connInfo, dtoData.size[%zu]", __func__, dtoData.size());
}

void CcuTransport::TransResUnpackProc(BinaryStream& binaryStream)
{
    uint32_t resSzie;
    binaryStream >> resSzie;
    rmtRes.ckes.clear();
    for (uint32_t i = 0; i < resSzie; i++) {
        uint32_t cke;
        binaryStream >> cke;
        rmtRes.ckes.push_back(cke);
    }
    HCCL_INFO("Recv ckesSize[%u]", resSzie);

    binaryStream >> resSzie;
    rmtRes.cntCkes.clear();
    for (uint32_t i = 0; i < resSzie; i++) {
        uint32_t cntCke;
        binaryStream >> cntCke;
        rmtRes.cntCkes.push_back(cntCke);
    }

    binaryStream >> resSzie;
    rmtRes.xns.clear();
    for (uint32_t i = 0; i < resSzie; i++) {
        uint32_t xn;
        binaryStream >> xn;
        rmtRes.xns.push_back(xn);
    }
    HCCL_INFO("Recv xnsSize[%u]", resSzie);
}

void CcuTransport::CclBufferInfoUnpack(BinaryStream& binaryStream) { rmtCclBufInfo.Unpack(binaryStream); }

void CcuTransport::SendFinish()
{
    sendFinishMsg = std::vector<char>(FINISH_MSG, FINISH_MSG + FINISH_MSG_SIZE);
    socket->SendAsync(sendFinishMsg.data(), FINISH_MSG_SIZE);
}

void CcuTransport::RecvFinish()
{
    recvFinishMsg.resize(FINISH_MSG_SIZE);
    socket->RecvAsync(reinterpret_cast<u8*>(recvFinishMsg.data()), FINISH_MSG_SIZE);
}

void CcuTransport::CheckFinish()
{
    std::string sendFinishMsgStr(sendFinishMsg.begin(), sendFinishMsg.end());
    std::string recvFinishMsgStr(recvFinishMsg.begin(), recvFinishMsg.end());
    if (sendFinishMsgStr != recvFinishMsgStr) {
        THROW<InternalException>(
            "[CcuTransport][RecvFinish]msgRecv[%s] and msgSend[%s] are not equal", recvFinishMsgStr.c_str(),
            sendFinishMsgStr.c_str());
    }
}

void CcuTransport::ReleaseTransRes()
{
    for (uint32_t i = 0; i < ckesRes.size(); i++) {
        auto ret = CcuDeviceManager::ReleaseCke(devLogicId, dieId, ckesRes[i]);
        if (ret != HcclResult::HCCL_SUCCESS) {
            THROW<InternalException>("errNo[0x%016llx]:Release ckesRes failed.", ret);
        }
    }

    for (uint32_t i = 0; i < xnsRes.size(); i++) {
        auto ret = CcuDeviceManager::ReleaseXn(devLogicId, dieId, xnsRes[i]);
        if (ret != HcclResult::HCCL_SUCCESS) {
            THROW<InternalException>("errNo[0x%016llx]:Release xnsRes failed.", ret);
        }
    }
}

uint32_t CcuTransport::GetDieId() const { return dieId; }

uint32_t CcuTransport::GetChannelId() const { return ccuConnection->GetChannelId(); }

uint32_t CcuTransport::GetLocCkeByIndex(uint32_t index) const
{
    std::shared_lock<std::shared_timed_mutex> lock(transMutex);
    if (index >= locRes.ckes.size()) {
        THROW<InternalException>(
            "[GetLocCkeByIndex]:index[%u] is bigger than ckes size[%u]", index, locRes.ckes.size());
    }
    return locRes.ckes[index];
}

uint32_t CcuTransport::GetLocCntCkeByIndex(uint32_t index) const
{
    std::shared_lock<std::shared_timed_mutex> lock(transMutex);
    if (index >= locRes.cntCkes.size()) {
        THROW<InternalException>(
            "[GetLocCntCkeByIndex]:index[%u] is bigger than cntCkes size[%u]", index, locRes.cntCkes.size());
    }
    return locRes.cntCkes[index];
}

uint32_t CcuTransport::GetLocXnByIndex(uint32_t index) const
{
    std::shared_lock<std::shared_timed_mutex> lock(transMutex);
    if (index >= locRes.xns.size()) {
        THROW<InternalException>(
            StringFormat("[GetLocXnByIndex]:index[%u] is bigger than xns size[%u]", index, locRes.xns.size()));
    }
    return locRes.xns[index];
}

uint32_t CcuTransport::GetRmtCkeByIndex(uint32_t index) const
{
    std::shared_lock<std::shared_timed_mutex> lock(transMutex);
    if (index >= rmtRes.ckes.size()) {
        THROW<InternalException>(
            StringFormat("[GetRmtCkeByIndex]:index[%u] is bigger than ckes size[%u]", index, rmtRes.ckes.size()));
    }
    return rmtRes.ckes[index];
}

uint32_t CcuTransport::GetRmtCntCkeByIndex(uint32_t index) const
{
    std::shared_lock<std::shared_timed_mutex> lock(transMutex);
    if (index >= rmtRes.cntCkes.size()) {
        THROW<InternalException>(StringFormat(
            "[GetRmtCntCkeByIndex]:index[%u] is bigger than cntCkes size[%u]", index, rmtRes.cntCkes.size()));
    }
    return rmtRes.cntCkes[index];
}

uint32_t CcuTransport::GetRmtXnByIndex(uint32_t index) const
{
    std::shared_lock<std::shared_timed_mutex> lock(transMutex);
    if (index >= rmtRes.xns.size()) {
        THROW<InternalException>(
            StringFormat("[GetRmtXnByIndex]:index[%u] is bigger than xns size[%u]", index, rmtRes.xns.size()));
    }
    return rmtRes.xns[index];
}

HcclResult CcuTransport::GetLocBuffer(CclBufferInfo& bufferInfo, const uint32_t& bufNum) const
{
    (void)bufNum;
    bufferInfo = locCclBufInfo;
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::GetRmtBuffer(CclBufferInfo& bufferInfo, const uint32_t& bufNum) const
{
    (void)bufNum;
    bufferInfo = rmtCclBufInfo;
    return HCCL_SUCCESS;
}

CcuTransport::~CcuTransport() { DECTOR_TRY_CATCH("CcuTransport", ReleaseTransRes()); }

std::string CcuTransport::Describe() const
{
    std::string description = "";

    description = StringFormat("DieId: %u, ", dieId);
    description += transStatus.Describe();
    description += StringFormat(
        ", LocRes: {%u Ckes, %u CntCkes, %u Xns}, ", locRes.ckes.size(), locRes.cntCkes.size(), locRes.xns.size());
    description += StringFormat(
        "RmtRes: {%u Ckes, %u CntCkes, %u Xns}, ", rmtRes.ckes.size(), rmtRes.cntCkes.size(), rmtRes.xns.size());
    description += StringFormat("CkesRes size: %u, ", ckesRes.size());
    description += StringFormat("XnsRes size: %u, ", xnsRes.size());
    description += StringFormat("%s", socket->Describe().c_str());
    return description;
}

std::vector<ConnJettyInfo> CcuTransport::GetDeleteJettyInfo() { return ccuConnection->GetDeleteJettyInfo(); }
std::vector<ConnJettyInfo> CcuTransport::GetUnimportJettyInfo() { return ccuConnection->GetUnimportJettyInfo(); }

HcclResult CcuTransport::Clean()
{
    transStatus = TransStatus::INIT;
    sendData.clear();
    TRY_CATCH_RETURN(ccuConnection->Clean());
    return HCCL_SUCCESS;
}

} // namespace Hccl
