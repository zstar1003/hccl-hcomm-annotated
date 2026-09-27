/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_connection.h"

#include <cstdlib>
#include "hccp_ctx.h"
#include "exception_util.h"
#include "orion_adapter_rts.h"
#include "internal_exception.h"
#include "rdma_handle_manager.h"

namespace Hccl {

CcuConnection::CcuConnection(
    const IpAddress& locAddr, const IpAddress& rmtAddr, const CcuChannelInfo& channelInfo,
    const std::vector<CcuJetty*>& ccuJettys)
    : locAddr_(locAddr),
      rmtAddr_(rmtAddr),
      channelInfo_(channelInfo),
      ccuJettys_(ccuJettys)
{}

CcuTpConnection::CcuTpConnection(
    const IpAddress& locAddr, const IpAddress& rmtAddr, const CcuChannelInfo& channelInfo,
    const std::vector<CcuJetty*>& ccuJettys)
    : CcuConnection(locAddr, rmtAddr, channelInfo, ccuJettys)
{
    tpProtocol = TpProtocol::TP;
}

CcuCtpConnection::CcuCtpConnection(
    const IpAddress& locAddr, const IpAddress& rmtAddr, const CcuChannelInfo& channelInfo,
    const std::vector<CcuJetty*>& ccuJettys)
    : CcuConnection(locAddr, rmtAddr, channelInfo, ccuJettys)
{
    tpProtocol = TpProtocol::CTP;
}

HcclResult CcuConnection::Init()
{
    TRY_CATCH_RETURN(devLogicId = HrtGetDevice(); uint32_t devPhyId = HrtGetDevicePhyIdByUserDevId(devLogicId);

                     auto& rdmaHandleMgr = RdmaHandleManager::GetInstance();
                     rdmaHandle = rdmaHandleMgr.GetByIp(devPhyId, locAddr_);
                     dieId = rdmaHandleMgr.GetDieAndFuncId(rdmaHandle).first;);

    CHK_RET(GetLocalCcuRmaBufferInfo());

    jettyNum = channelInfo_.jettyInfos.size();
    CHK_PRT_RET(
        jettyNum == 0, HCCL_ERROR("[CcuConnection][%s] failed, jetty num[0] is unexpected.", __func__),
        HcclResult::HCCL_E_PARA);

    GenerateLocalPsn();
    status = CcuConnStatus::INIT;
    innerStatus = InnerStatus::INIT;
    return HcclResult::HCCL_SUCCESS;
}

CcuConnStatus CcuConnection::GetStatus()
{
    if (status == CcuConnStatus::CONNECTED || status == CcuConnStatus::CONN_INVALID) {
        return status;
    }

    if (StatusMachine() != HcclResult::HCCL_SUCCESS) {
        status = CcuConnStatus::CONN_INVALID;
        innerStatus = InnerStatus::CONN_INVALID;
    }

    return status;
}
// 获取本端内存，为部分远端不可写的tokenId
HcclResult CcuConnection::GetLocalCcuRmaBufferInfo()
{
    uint64_t ccuBufSize = 0; // 暂未使用
    CHK_RET(CcuDeviceManager::GetCcuResourceSpaceBufInfo(devLogicId, dieId, ccuBufAddr, ccuBufSize));

    uint64_t tokenId = 0;
    uint64_t tokenValue = 0;
    CHK_RET(CcuDeviceManager::GetCcuResourceSpaceTokenInfo(devLogicId, dieId, tokenId, tokenValue));
    ccuBufTokenId = static_cast<uint32_t>(tokenId);
    ccuBufTokenValue = static_cast<uint32_t>(tokenValue);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuConnection::StatusMachine()
{
    TRY_CATCH_RETURN(
        if (status == CcuConnStatus::INIT) {
            UpdateInitStatus();
            return HcclResult::HCCL_SUCCESS;
        }

        if (innerStatus == InnerStatus::JETTY_IMPORTING) {
            UpdateExchangeStatus();
            return HcclResult::HCCL_SUCCESS;
        });

    return HcclResult::HCCL_SUCCESS;
}

void CcuConnection::UpdateInitStatus()
{
    switch (innerStatus) {
        case InnerStatus::INIT:
        case InnerStatus::JETTY_CREATING: {
            if (!CreateJetty()) {
                innerStatus = InnerStatus::JETTY_CREATING;
                break; // 状态不改变退出，下轮状态机进入继续执行
            }

            if (GetTpInfo()) { // 如果有缓存的tp信息，可以直接完成
                innerStatus = InnerStatus::EXCHANGEABLE;
                status = CcuConnStatus::EXCHANGEABLE;
                break;
            }

            innerStatus = InnerStatus::TP_INFO_GETTING; // 不退出继续调用下个异步接口
            break;
        }
        case InnerStatus::TP_INFO_GETTING: {
            if (!GetTpInfo()) {
                break; // 状态不改变退出，下轮状态机进入继续执行
            }
            innerStatus = InnerStatus::EXCHANGEABLE;
            status = CcuConnStatus::EXCHANGEABLE;
            break;
        }
        default:
            ThrowAbnormalStatus(std::string(__func__));
    }
}

bool CcuConnection::CreateJetty()
{
    if (isJettyCreated) {
        return true;
    }

    isJettyCreated = true;
    for (size_t i = 0; i < jettyNum; i++) {
        auto ret = ccuJettys_[i]->CreateJetty();
        if (ret == HcclResult::HCCL_E_AGAIN) {
            // 不提供日志避免刷屏
            isJettyCreated = isJettyCreated && false;
            continue;
        }

        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[CcuConnection][%s] failed, hccl result[%d]", __func__, ret);
            ThrowAbnormalStatus(std::string(__func__));
        }
    }

    return isJettyCreated;
}

inline uint32_t GetRandomNum()
{
    uint32_t randNum = std::rand();
    return randNum;
}

void CcuConnection::GenerateLocalPsn() { jettyImportCfg.localPsn = GetRandomNum(); }

bool CcuConnection::GetTpInfo()
{
    if (tpProtocol == TpProtocol::INVALID) { // 不感知tp建链，当前默认不支持
        HCCL_ERROR(
            "[CcuConnection][%s] failed, tpProtocol[%s] is not expected.", __func__, tpProtocol.Describe().c_str());
        ThrowAbnormalStatus(std::string(__func__));
    }

    HcclResult ret = TpManager::GetInstance(devLogicId).GetTpInfo({locAddr_, rmtAddr_, tpProtocol}, tpInfo);
    if (ret == HcclResult::HCCL_E_AGAIN) {
        // 此处可能刷屏，非必要勿加日志
        return false;
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CcuConnection][%s] failed, hccl result[%d]", __func__, ret);
        ThrowAbnormalStatus(std::string(__func__));
    }

    jettyImportCfg.localTpHandle = tpInfo.tpHandle;
    return true;
}

void CcuConnection::Serialize(std::vector<char>& dtoData)
{
    if (status != CcuConnStatus::EXCHANGEABLE) {
        HCCL_ERROR(
            "[CcuConnection][%s] failed, not init completed yet, "
            "status[%s].",
            __func__, status.Describe().c_str());
        ThrowAbnormalStatus(std::string(__func__));
    }

    BinaryStream dtoStream;
    dtoStream << ccuBufAddr;
    dtoStream << ccuBufTokenId;
    dtoStream << ccuBufTokenValue;
    HCCL_INFO("[CcuConnection][%s], ccuBufAddr[%llx]", __func__, ccuBufAddr);

    dtoStream << jettyNum;
    HCCL_INFO("[CcuConnection][%s], jettyNum[%u]", __func__, jettyNum);
    for (const auto& ccuJetty : ccuJettys_) {
        dtoStream << ccuJetty->GetCreateJettyParam().tokenValue;
        const auto& outParam = ccuJetty->GetJettyedOutParam();
        dtoStream << outParam.key;
        dtoStream << outParam.keySize;
    }

    if (tpProtocol != TpProtocol::INVALID) {
        dtoStream << jettyImportCfg.localTpHandle;
        dtoStream << jettyImportCfg.localPsn;
        HCCL_INFO(
            "[CcuConnection][%s] tpProtocol[%s], localTpHandle[0x%llx], localPsn[%u].", __func__,
            tpProtocol.Describe().c_str(), jettyImportCfg.localTpHandle, jettyImportCfg.localPsn);
    }

    dtoData.clear();
    dtoStream.Dump(dtoData);
}

void CcuConnection::Deserialize(const std::vector<char>& dtoData)
{
    if (status != CcuConnStatus::EXCHANGEABLE) {
        HCCL_ERROR(
            "[CcuConnection][%s] failed, not init completed yet, "
            "status[%s].",
            __func__, status.Describe().c_str());
        ThrowAbnormalStatus(std::string(__func__));
    }

    vector<char> rmtDtoData = dtoData;
    BinaryStream dtoStream(rmtDtoData);
    dtoStream >> rmtCcuBufAddr;
    dtoStream >> rmtCcuBufTokenId;
    dtoStream >> rmtCcuBufTokenValue;
    HCCL_INFO("[CcuConnection][%s], rmtCcuBufAddr[%llx].", __func__, rmtCcuBufAddr);

    uint32_t remoteJettySize{0};
    dtoStream >> remoteJettySize;

    importJettyCtxs.clear();
    importJettyCtxs.resize(remoteJettySize);
    HCCL_INFO("[CcuConnection][%s], remoteJettySize[%u].", __func__, remoteJettySize);

    for (auto& importCtx : importJettyCtxs) {
        dtoStream >> importCtx.inParam.tokenValue;
        dtoStream >> importCtx.remoteQpKey;            // 保存key数组
        importCtx.inParam.key = importCtx.remoteQpKey; // 保存指针用于接口调用
        dtoStream >> importCtx.inParam.keyLen;
    }

    if (tpProtocol != TpProtocol::INVALID) {
        dtoStream >> jettyImportCfg.remoteTpHandle;
        dtoStream >> jettyImportCfg.remotePsn;

        HCCL_INFO(
            "[CcuConnection][%s] tpEnable, remoteTpHandle[0x%llx], remotePsn[%u].", __func__,
            jettyImportCfg.remoteTpHandle, jettyImportCfg.remotePsn);
    }
}

void CcuConnection::ImportJetty()
{
    if (isJettyImported) {
        HCCL_INFO("[CcuConnection][%s] taJettys has been imported already.", __func__);
        return;
    }

    if (innerStatus != InnerStatus::EXCHANGEABLE) {
        ThrowAbnormalStatus(std::string(__func__));
    }

    if (jettyNum != importJettyCtxs.size()) {
        HCCL_ERROR(
            "[CcuConnection][%s] failed to ImportJetty, "
            "jettyNum[%u] is not equal to importJettyCtxs.size[%u].",
            __func__, jettyNum, importJettyCtxs.size());
        ThrowAbnormalStatus(std::string(__func__));
    }

    ResetRequestCtxs();
    for (size_t i = 0; i < jettyNum; i++) {
        if (StartImportJettyRequest(i, reqHandles[i]) != HcclResult::HCCL_SUCCESS) {
            ThrowAbnormalStatus(std::string(__func__));
        }
    }

    innerStatus = InnerStatus::JETTY_IMPORTING;
}

void CcuConnection::ResetRequestCtxs()
{
    reqHandles.clear();
    reqHandles.resize(jettyNum);

    reqDataBuffers.clear();
    reqDataBuffers.resize(jettyNum);

    remoteJettyHandlePtrs.clear();
    remoteJettyHandlePtrs.resize(jettyNum);
}

HcclResult CcuConnection::StartImportJettyRequest(uint32_t jettyIndex, RequestHandle& reqHandle)
{
    if (tpProtocol == TpProtocol::INVALID) {
        ThrowAbnormalStatus(std::string(__func__));
    }

    auto& importCtx = importJettyCtxs[jettyIndex];
    auto& importCtxInParam = importCtx.inParam;
    importCtxInParam.jettyImportCfg = jettyImportCfg;
    importCtxInParam.jettyImportCfg.protocol = tpProtocol;
    TRY_CATCH_RETURN(reqHandle = RaUbTpImportJettyAsync(
                         rdmaHandle, importCtxInParam, reqDataBuffers[jettyIndex], remoteJettyHandlePtrs[jettyIndex]););

    return HcclResult::HCCL_SUCCESS;
}

bool CcuConnection::CheckRequestResults()
{
    if (reqHandles.size() == 0) {
        return true;
    }

    // 检查所有下发异步请求是否完成
    vector<size_t> completedReqs;
    for (size_t i = 0; i < reqHandles.size(); i++) {
        ReqHandleResult result = HrtRaGetAsyncReqResult(reqHandles[i]);
        if (result == ReqHandleResult::NOT_COMPLETED) {
            continue;
        }

        if (result != ReqHandleResult::COMPLETED) {
            THROW<InternalException>(StringFormat(
                "[CcuConnection][%s] failed, result[%s] is unexpected.", __func__, result.Describe().c_str()));
        }

        // 记录已完成的reqHandles
        completedReqs.push_back(i);
    }

    // 删除已完成的reqHandles，避免重复查询
    for (int i = completedReqs.size() - 1; i >= 0; --i) {
        reqHandles.erase(reqHandles.begin() + completedReqs[i]);
    }

    // 检查是否有剩余reqHandles
    return reqHandles.size() == 0;
}

void CcuConnection::UpdateExchangeStatus()
{
    // 状态机保证为 InnerStatus::JETTY_IMPORTING
    if (!CheckRequestResults()) {
        return;
    }

    for (size_t i = 0; i < jettyNum; i++) {
        auto& outParam = importJettyCtxs[i].outParam;
        struct QpImportInfoT* infoPtr = reinterpret_cast<QpImportInfoT*>(reqDataBuffers[i].data());
        outParam.handle = reinterpret_cast<TargetJettyHandle>(remoteJettyHandlePtrs[i]);
        outParam.targetJettyVa = infoPtr->out.ub.tjettyHandle; // 该信息当前未使用
        outParam.tpn = infoPtr->out.ub.tpn;
    }
    isJettyImported = true;

    ConfigChannel();
    status = CcuConnStatus::CONNECTED;
    innerStatus = InnerStatus::CONNECTED;
}

void CcuConnection::ConfigChannel()
{
    if (jettyNum != importJettyCtxs.size()) {
        HCCL_ERROR(
            "[CcuConnection][%s] failed, jettyNum[%u] is not equal to "
            "importJettyCtxs.size[%u].",
            __func__, jettyNum, importJettyCtxs.size());
        ThrowAbnormalStatus(std::string(__func__));
    }

    ChannelCfg cfg{};
    cfg.channelId = channelInfo_.channelId;
    cfg.remoteEid = rmtAddr_.GetReverseEid();
    HCCL_INFO("[CcuComponent::ConfigChannel] cfg.remoteEid=%s", cfg.remoteEid.Describe().c_str());
    cfg.tpn = importJettyCtxs[0].outParam.tpn; // tp handle复用所以tpn一致
    cfg.remoteCcuVa = rmtCcuBufAddr;
    cfg.memTokenId = rmtCcuBufTokenId;
    cfg.memTokenValue = rmtCcuBufTokenValue;

    for (size_t i = 0; i < jettyNum; i++) {
        const auto& ccuJetty = ccuJettys_[i];
        const auto& inParam = ccuJetty->GetCreateJettyParam();
        const auto& outParam = ccuJetty->GetJettyedOutParam();
        const auto& jettyInfo = channelInfo_.jettyInfos[i];
        cfg.jettyCfgs.emplace_back(JettyCfg{
            jettyInfo.jettyCtxId, outParam.dbVa, outParam.dbTokenId,
            inParam.tokenValue}); // 安全问题，禁止打印token相关信息
    }

    if (CcuDeviceManager::ConfigChannel(devLogicId, dieId, cfg) != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CcuConnection][%s] failed, devLogicId[%d], dieId[%u] channelId[%u].", __func__, devLogicId, dieId,
            cfg.channelId);
        ThrowAbnormalStatus(std::string(__func__));
    }
}

CcuConnection::~CcuConnection() { DECTOR_TRY_CATCH("CcuConnection", ReleaseConnRes()); }

HcclResult CcuConnection::ReleaseConnRes()
{
    for (auto& item : importJettyCtxs) {
        if (item.outParam.handle != 0) {
            HrtRaUbUnimportJetty(rdmaHandle, item.outParam.handle);
            item.outParam.handle = 0;
        }
    }

    if (tpInfo.tpHandle != 0) { // tp handle 复用，只释放一次
        (void)TpManager::GetInstance(devLogicId).ReleaseTpInfo({locAddr_, rmtAddr_, tpProtocol}, tpInfo);
        tpInfo.tpHandle = 0;
    }

    // CcuJetty 生命周期跟随通信域CcuJettyMgr
    // 不需要connection主动销毁
    return HcclResult::HCCL_SUCCESS;
}

void CcuConnection::ThrowAbnormalStatus(const string& funcName)
{
    auto errMsg = StringFormat("[CcuConnection][%s] failed, [%s].", funcName.c_str(), Describe().c_str());
    status = CcuConnStatus::CONN_INVALID;
    innerStatus = InnerStatus::CONN_INVALID;
    THROW<InternalException>(errMsg);
}

std::string CcuConnection::Describe()
{
    return StringFormat(
        "[CcuConnection[locAddr=%s, rmtAddr=%s, protocol=%s, "
        "status=%s, innerStatus=%s, [dieId=%u, channelId=%u, jettyNum=%u]]]",
        locAddr_.Describe().c_str(), rmtAddr_.Describe().c_str(), tpProtocol.Describe().c_str(),
        status.Describe().c_str(), innerStatus.Describe().c_str(), dieId, channelInfo_.channelId, jettyNum);
}

uint32_t CcuConnection::GetDieId() const { return dieId; }

uint32_t CcuConnection::GetChannelId() const { return channelInfo_.channelId; }

int32_t CcuConnection::GetDevLogicId() const { return devLogicId; }

std::vector<ConnJettyInfo> CcuConnection::GetDeleteJettyInfo()
{
    std::vector<ConnJettyInfo> connDeleteJettyInfos;
    ConnJettyInfo jettyInfo;
    for (auto& ccuJetty : ccuJettys_) {
        if (ccuJetty != nullptr) {
            ccuJetty->GetJettyInfo(jettyInfo);
            jettyInfo.rdmaHandle = rdmaHandle;
            connDeleteJettyInfos.push_back(jettyInfo);
        }
    }
    return connDeleteJettyInfos;
}

std::vector<ConnJettyInfo> CcuConnection::GetUnimportJettyInfo()
{
    std::vector<ConnJettyInfo> connUnimportJettyInfos;
    ConnJettyInfo jettyInfo;
    for (auto& item : importJettyCtxs) {
        if (item.outParam.handle != 0) {
            jettyInfo.remoteJetty = item.outParam.handle;
            jettyInfo.rdmaHandle = rdmaHandle;
            item.outParam.handle = 0;
            connUnimportJettyInfos.push_back(jettyInfo);
        }
    }
    return connUnimportJettyInfos;
}

void CcuConnection::Clean()
{
    status = CcuConnStatus::INIT;
    innerStatus = InnerStatus::INIT;
    isJettyCreated = false;
    isJettyImported = false;
    ReleaseConnRes();
    GenerateLocalPsn();

    // 销毁jetty要在ReleaseConnRes之后
    for (auto& ccuJetty : ccuJettys_) {
        ccuJetty->Clean();
    }
}

} // namespace Hccl
