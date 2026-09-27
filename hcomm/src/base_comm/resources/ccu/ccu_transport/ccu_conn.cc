/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_conn.h"

#include <random>
#include <sstream>

#include "hcom_common.h"
#include "exception_handler.h"
#include "eid_info_mgr.h"

#include "hccp_ctx.h"

#include "rdma_handle_manager.h"
#include "orion_adapter_hccp.h"
#include "hcomm_res_mgr.h"
#include "env_config/env_config_v2.h"

namespace hcomm {

CcuConnection::CcuConnection(
    const CommAddr& locAddr, const CommAddr& rmtAddr, const CcuChannelInfo& channelInfo,
    const std::vector<CcuJetty*>& ccuJettys, uint32_t qos)
    : locAddr_(locAddr),
      rmtAddr_(rmtAddr),
      channelInfo_(channelInfo),
      ccuJettys_(ccuJettys),
      qos_(qos)
{}

CcuRtpConnection::CcuRtpConnection(
    const CommAddr& locAddr, const CommAddr& rmtAddr, const CcuChannelInfo& channelInfo,
    const std::vector<CcuJetty*>& ccuJettys, uint32_t qos)
    : CcuConnection(locAddr, rmtAddr, channelInfo, ccuJettys, qos)
{
    tpProtocol_ = TpProtocol::RTP;
}

CcuCtpConnection::CcuCtpConnection(
    const CommAddr& locAddr, const CommAddr& rmtAddr, const CcuChannelInfo& channelInfo,
    const std::vector<CcuJetty*>& ccuJettys, uint32_t qos)
    : CcuConnection(locAddr, rmtAddr, channelInfo, ccuJettys, qos)
{
    tpProtocol_ = TpProtocol::CTP;
}

HcclResult CcuConnection::Init()
{
    userDevId_ = HcclGetThreadDeviceId();
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(userDevId_), devPhyId_));

    EXCEPTION_HANDLE_BEGIN
    auto& rdmaHandleMgr = Hccl::RdmaHandleManager::GetInstance();
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(locAddr_, ipAddr));
    ctxHandle_ = rdmaHandleMgr.GetByIp(devPhyId_, ipAddr);
    CHK_PRT_RET(
        !rdmaHandleMgr.IsHandleValid(ctxHandle_),
        HCCL_ERROR(
            "[CcuConnection][%s] ctxHandle_[%p] is not valid, "
            "RdmaHandleManager may have DeInit this device",
            __func__, ctxHandle_),
        HcclResult::HCCL_E_INTERNAL);

    DevEidInfo eidInfo{};
    CHK_RET(EidInfoMgr::GetInstance(devPhyId_).GetEidInfoByAddr(locAddr_, eidInfo));
    dieId_ = static_cast<uint8_t>(eidInfo.dieId);
    funcId_ = eidInfo.funcId;

    EXCEPTION_HANDLE_END

    CHK_RET(GetLocalCcuRmaBufferInfo());

    jettyNum_ = channelInfo_.jettyInfos.size();
    CHK_PRT_RET(
        jettyNum_ == 0, HCCL_ERROR("[CcuConnection][%s] failed, jetty num[0] is unexpected.", __func__),
        HcclResult::HCCL_E_PARA);

    GenerateLocalPsn();
    status_ = CcuConnStatus::INIT;
    innerStatus_ = InnerStatus::INIT;
    return HcclResult::HCCL_SUCCESS;
}

CcuConnStatus CcuConnection::GetStatus()
{
    if (status_ == CcuConnStatus::CONNECTED || status_ == CcuConnStatus::CONN_INVALID) {
        return status_;
    }

    if (StatusMachine() != HcclResult::HCCL_SUCCESS) {
        status_ = CcuConnStatus::CONN_INVALID;
        innerStatus_ = InnerStatus::CONN_INVALID;
    }

    return status_;
}

HcclResult CcuConnection::GetLocalCcuRmaBufferInfo()
{
    uint64_t ccuBufSize = 0; // 暂未使用
    CHK_RET(CcuDevMgrImp::GetCcuResourceSpaceBufInfo(userDevId_, dieId_, ccuBufAddr_, ccuBufSize));

    uint64_t tokenId = 0;
    uint64_t tokenValue = 0;
    CHK_RET(CcuDevMgrImp::GetCcuResourceSpaceTokenInfo(userDevId_, dieId_, tokenId, tokenValue));
    ccuBufTokenId_ = static_cast<uint32_t>(tokenId);
    ccuBufTokenValue_ = static_cast<uint32_t>(tokenValue);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuConnection::StatusMachine()
{
    if (status_ == CcuConnStatus::INIT) {
        CHK_RET(UpdateInitStatus());
        return HcclResult::HCCL_SUCCESS;
    }

    if (innerStatus_ == InnerStatus::JETTY_IMPORTING) {
        CHK_RET(UpdateExchangeStatus());
        return HcclResult::HCCL_SUCCESS;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuConnection::GetTaTimeOut()
{
    if (tpProtocol_ == TpProtocol::CTP) {
        uint32_t taTimeOutValue = 0;
        CHK_RET(hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(taTimeOutValue));
        errTimeout_ = static_cast<uint8_t>(taTimeOutValue);
        HCCL_INFO("[CcuConnection][%s] CTP, env errTimeout[%u].", __func__, errTimeout_);
        return HcclResult::HCCL_SUCCESS;
    }

    uint32_t taTimeOutValue = 0;
    CHK_RET(hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(taTimeOutValue));
    uint8_t envTaTimeOut = static_cast<uint8_t>(taTimeOutValue);
    uint32_t tpTimeOutMs = 0;
    (void)TpMgr::GetTpTotalTimeout(tpAttrInfo_, tpTimeOutMs);
    errTimeout_ = TpMgr::CalcTaTimeout(tpProtocol_, envTaTimeOut, tpTimeOutMs);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuConnection::UpdateInitStatus()
{
    switch (innerStatus_) {
        case InnerStatus::INIT:
        case InnerStatus::TP_INFO_GETTING: {
            auto ret = GetTpInfo();
            if (ret == HcclResult::HCCL_E_AGAIN) {
                innerStatus_ = InnerStatus::TP_INFO_GETTING;
                return HcclResult::HCCL_SUCCESS;
            }
            CHK_RET(ret);
            CHK_PRT_RET(
                !tpInfo_.hasMappedJettyPriority,
                HCCL_ERROR("[CcuConnection][%s] TpMgr did not provide mappedJettyPriority.", __func__),
                HcclResult::HCCL_E_INTERNAL);
            for (auto* jetty : ccuJettys_) {
                CHK_RET(jetty->SetMappedJettyPriority(tpInfo_.mappedJettyPriority));
            }
            innerStatus_ = InnerStatus::TP_ATTR_GETTING;
            return HcclResult::HCCL_SUCCESS;
        }
        case InnerStatus::TP_ATTR_GETTING: {
            auto ret = GetTpAttr();
            if (ret == HcclResult::HCCL_E_AGAIN) {
                innerStatus_ = InnerStatus::TP_ATTR_GETTING;
                return HcclResult::HCCL_SUCCESS;
            }
            CHK_RET(ret);

            CHK_RET(GetTaTimeOut());
            innerStatus_ = InnerStatus::JETTY_CREATING;
            return HcclResult::HCCL_SUCCESS;
        }
        case InnerStatus::JETTY_CREATING: {
            auto ret = CreateJetty();
            if (ret == HcclResult::HCCL_E_AGAIN) {
                return HcclResult::HCCL_SUCCESS;
            }
            CHK_RET(ret);
            innerStatus_ = InnerStatus::EXCHANGEABLE;
            status_ = CcuConnStatus::EXCHANGEABLE;
            return HcclResult::HCCL_SUCCESS;
        }
        default:
            return ReturnErrorStatus(std::string(__func__));
    }
}

HcclResult CcuConnection::CreateJetty()
{
    if (isJettyCreated_) {
        return HcclResult::HCCL_SUCCESS;
    }

    isJettyCreated_ = true;
    for (size_t i = 0; i < jettyNum_; i++) {
        auto ret = ccuJettys_[i]->CreateJetty(errTimeout_);
        if (ret == HcclResult::HCCL_E_AGAIN) {
            // 不提供日志避免刷屏
            isJettyCreated_ = isJettyCreated_ && false;
            continue;
        }

        if (ret != HcclResult::HCCL_SUCCESS) {
            isJettyCreated_ = true;
            HCCL_ERROR("[CcuConnection][%s] failed, hccl result[%d]", __func__, ret);
            return HcclResult::HCCL_E_NETWORK;
        }
    }

    return isJettyCreated_ ? HcclResult::HCCL_SUCCESS : HcclResult::HCCL_E_AGAIN;
}

inline uint32_t GetRandomNum()
{
    uint32_t randNum = std::rand();
    return randNum;
}

void CcuConnection::GenerateLocalPsn() { jettyImportCfg_.localPsn = GetRandomNum(); }

GetTpInfoParam CcuConnection::MakeGetTpInfoParam() const
{
    GetTpInfoParam param;
    param.locAddr = locAddr_;
    param.rmtAddr = rmtAddr_;
    param.tpProtocol = tpProtocol_;
    param.qos = (qos_ > 7U) ? EnvConfig::UB_QOS_DEFAULT : (qos_ & 7U);
    param.slLevelCount = 0;
    param.loopFirstTpLowestSl = false;
    return param;
}

HcclResult CcuConnection::GetTpInfo()
{
    if (tpProtocol_ == TpProtocol::INVALID) { // 不感知tp建链，当前默认不支持
        HCCL_ERROR(
            "[CcuConnection][%s] failed, tpProtocol[%s] is not expected.", __func__, tpProtocol_.Describe().c_str());
        return HcclResult::HCCL_E_PARA;
    }

    HcclResult ret = TpMgr::GetInstance(devPhyId_).GetTpInfo(MakeGetTpInfoParam(), tpInfo_);
    if (ret == HcclResult::HCCL_E_AGAIN) {
        return ret;
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CcuConnection][%s] failed, hccl result[%d]", __func__, ret);
        return HcclResult::HCCL_E_NETWORK;
    }

    jettyImportCfg_.localTpHandle = tpInfo_.tpHandle;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuConnection::GetTpAttr()
{
    if (tpProtocol_ == TpProtocol::CTP) {
        HCCL_INFO("[CcuConnection][%s] CTP.", __func__);
        return HcclResult::HCCL_SUCCESS;
    }

    constexpr uint32_t kTpAttrRetryTimesInitBit = 0U;
    constexpr uint32_t kTpAttrAtBit = 1U;
    constexpr uint32_t TP_ATTR_BITMAP = (1U << kTpAttrRetryTimesInitBit) | (1U << kTpAttrAtBit);
    HcclResult ret
        = TpMgr::GetInstance(devPhyId_).GetTpAttr({tpInfo_.tpHandle, TP_ATTR_BITMAP}, tpAttrInfo_, ctxHandle_);
    if (ret == HcclResult::HCCL_E_AGAIN) {
        return ret;
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CcuConnection][%s] failed, hccl result[%d]", __func__, ret);
        return HcclResult::HCCL_E_NETWORK;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuConnection::Serialize(std::vector<char>& dtoData)
{
    if (status_ != CcuConnStatus::EXCHANGEABLE) {
        HCCL_ERROR(
            "[CcuConnection][%s] failed, not init completed yet, "
            "status[%s].",
            __func__, status_.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }

    Hccl::BinaryStream dtoStream;
    dtoStream << ccuBufAddr_;
    dtoStream << ccuBufTokenId_;
    dtoStream << ccuBufTokenValue_;
    HCCL_INFO("[CcuConnection][%s], ccuBufAddr[%llx]", __func__, ccuBufAddr_);

    dtoStream << jettyNum_;
    HCCL_INFO("[CcuConnection][%s], jettyNum[%u]", __func__, jettyNum_);
    for (const auto& ccuJetty : ccuJettys_) {
        dtoStream << ccuJetty->GetCreateJettyParam().tokenValue;
        const auto& outParam = ccuJetty->GetJettyedOutParam();
        dtoStream << outParam.key;
        dtoStream << outParam.keySize;
    }

    if (tpProtocol_ != TpProtocol::INVALID) {
        dtoStream << jettyImportCfg_.localTpHandle;
        dtoStream << jettyImportCfg_.localPsn;
        HCCL_INFO(
            "[CcuConnection][%s] tpProtocol[%s], localTpHandle[0x%llx], localPsn[%u].", __func__,
            tpProtocol_.Describe().c_str(), jettyImportCfg_.localTpHandle, jettyImportCfg_.localPsn);
    }

    dtoData.clear();
    dtoStream.Dump(dtoData);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuConnection::Deserialize(const std::vector<char>& dtoData)
{
    if (status_ != CcuConnStatus::EXCHANGEABLE) {
        HCCL_ERROR(
            "[CcuConnection][%s] failed, not init completed yet, "
            "status[%s].",
            __func__, status_.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }

    std::vector<char> rmtDtoData = dtoData;
    Hccl::BinaryStream dtoStream(rmtDtoData);
    dtoStream >> rmtCcuBufAddr_;
    dtoStream >> rmtCcuBufTokenId_;
    dtoStream >> rmtCcuBufTokenValue_;
    HCCL_INFO("[CcuConnection][%s], rmtCcuBufAddr[%llx].", __func__, rmtCcuBufAddr_);

    uint32_t remoteJettySize{0};
    dtoStream >> remoteJettySize;

    importJettyCtxs_.clear();
    importJettyCtxs_.resize(remoteJettySize);
    HCCL_INFO("[CcuConnection][%s], remoteJettySize[%u].", __func__, remoteJettySize);

    for (auto& importCtx : importJettyCtxs_) {
        dtoStream >> importCtx.inParam.tokenValue;
        dtoStream >> importCtx.remoteQpKey;            // 保存key数组
        importCtx.inParam.key = importCtx.remoteQpKey; // 保存指针用于接口调用
        dtoStream >> importCtx.inParam.keyLen;
    }

    if (tpProtocol_ != TpProtocol::INVALID) {
        dtoStream >> jettyImportCfg_.remoteTpHandle;
        dtoStream >> jettyImportCfg_.remotePsn;

        HCCL_INFO(
            "[CcuConnection][%s] tpEnable, remoteTpHandle[0x%llx], remotePsn[%u].", __func__,
            jettyImportCfg_.remoteTpHandle, jettyImportCfg_.remotePsn);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuConnection::ImportJetty()
{
    if (isJettyImported_) {
        HCCL_INFO("[CcuConnection][%s] taJettys has been imported already.", __func__);
        return HcclResult::HCCL_SUCCESS;
    }

    if (innerStatus_ != InnerStatus::EXCHANGEABLE) {
        return ReturnErrorStatus(std::string(__func__));
    }

    // importJettyCtxs_.resize(jettyNum_);
    if (jettyNum_ != importJettyCtxs_.size()) {
        HCCL_ERROR(
            "[CcuConnection][%s] failed to ImportJetty, "
            "jettyNum[%u] is not equal to importJettyCtxs.size[%u].",
            __func__, jettyNum_, importJettyCtxs_.size());
        return ReturnErrorStatus(std::string(__func__));
    }

    ResetRequestCtxs();
    for (size_t i = 0; i < jettyNum_; i++) {
        if (StartImportJettyRequest(i, reqHandles_[i]) != HcclResult::HCCL_SUCCESS) {
            return ReturnErrorStatus(std::string(__func__));
        }
    }

    innerStatus_ = InnerStatus::JETTY_IMPORTING;
    return HcclResult::HCCL_SUCCESS;
}

void CcuConnection::ResetRequestCtxs()
{
    reqHandles_.clear();
    reqHandles_.resize(jettyNum_);

    reqDataBuffers_.clear();
    reqDataBuffers_.resize(jettyNum_);

    remoteJettyHandlePtrs_.clear();
    remoteJettyHandlePtrs_.resize(jettyNum_);
}

HcclResult CcuConnection::StartImportJettyRequest(uint32_t jettyIndex, RequestHandle& reqHandle)
{
    if (tpProtocol_ == TpProtocol::INVALID) {
        return ReturnErrorStatus(std::string(__func__));
    }

    auto& importCtx = importJettyCtxs_[jettyIndex];
    auto& importCtxInParam = importCtx.inParam;
    importCtxInParam.jettyImportCfg = jettyImportCfg_;
    importCtxInParam.jettyImportCfg.protocol = tpProtocol_;
    CHK_RET(HccpUbTpImportJettyAsync(
        ctxHandle_, importCtxInParam, reqDataBuffers_[jettyIndex], remoteJettyHandlePtrs_[jettyIndex], reqHandle));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuConnection::CheckRequestResults()
{
    if (reqHandles_.size() == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    // 检查所有下发异步请求是否完成
    std::vector<size_t> completedReqs;
    const uint32_t reqSize = reqHandles_.size();
    for (size_t i = 0; i < reqSize; i++) {
        RequestResult result = HccpGetAsyncReqResult(reqHandles_[i]);
        if (result == RequestResult::NOT_COMPLETED) {
            continue;
        }

        if (result != RequestResult::COMPLETED) {
            HCCL_ERROR("[CcuConnection][%s] failed, result[%s] is unexpected.", __func__, result.Describe().c_str());
            return HcclResult::HCCL_E_NETWORK;
        }

        // 记录已完成的reqHandles
        completedReqs.push_back(i);
    }

    // 删除已完成的reqHandles，避免重复查询
    for (int i = completedReqs.size() - 1; i >= 0; --i) {
        reqHandles_.erase(reqHandles_.begin() + completedReqs[i]);
    }

    // 检查是否有剩余reqHandles
    return reqHandles_.size() == 0 ? HcclResult::HCCL_SUCCESS : HcclResult::HCCL_E_AGAIN;
}

HcclResult CcuConnection::UpdateExchangeStatus()
{
    // 状态机保证为 InnerStatus::JETTY_IMPORTING
    auto ret = CheckRequestResults();
    if (ret == HcclResult::HCCL_E_AGAIN) {
        return HcclResult::HCCL_SUCCESS; // 操作成功，保持当前状态
    }
    CHK_RET(ret);

    for (size_t i = 0; i < jettyNum_; i++) {
        auto& outParam = importJettyCtxs_[i].outParam;
        struct QpImportInfoT* infoPtr = reinterpret_cast<QpImportInfoT*>(reqDataBuffers_[i].data());
        outParam.handle = reinterpret_cast<TargetJettyHandle>(remoteJettyHandlePtrs_[i]);
        outParam.targetJettyVa = infoPtr->out.ub.tjettyHandle; // 该信息当前未使用
        outParam.tpn = infoPtr->out.ub.tpn;
        ccuJettys_[i]->SetTpn(outParam.tpn);
    }
    isJettyImported_ = true;

    CHK_RET(ConfigChannel());
    status_ = CcuConnStatus::CONNECTED;
    innerStatus_ = InnerStatus::CONNECTED;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuConnection::ConfigChannel()
{
    if (jettyNum_ != importJettyCtxs_.size()) {
        HCCL_ERROR(
            "[CcuConnection][%s] failed, jettyNum[%u] is not equal to "
            "importJettyCtxs.size[%u].",
            __func__, jettyNum_, importJettyCtxs_.size());
        return HcclResult::HCCL_E_INTERNAL;
    }

    ChannelCfg cfg{};
    cfg.channelId = channelInfo_.channelId;
    Hccl::IpAddress rmtAddr{};
    CHK_RET(CommAddrToIpAddress(rmtAddr_, rmtAddr));
    CHK_RET(IpAddressToReverseHcclEid(rmtAddr, cfg.remoteEid)); // 配置ccu硬件需要使用反向eid
    cfg.tpn = importJettyCtxs_[0].outParam.tpn;                 // tp handle复用所以tpn一致
    cfg.remoteCcuVa = rmtCcuBufAddr_;
    cfg.memTokenId = rmtCcuBufTokenId_;
    cfg.memTokenValue = rmtCcuBufTokenValue_;

    for (size_t i = 0; i < jettyNum_; i++) {
        const auto& ccuJetty = ccuJettys_[i];
        const auto& inParam = ccuJetty->GetCreateJettyParam();
        const auto& outParam = ccuJetty->GetJettyedOutParam();
        const auto& jettyInfo = channelInfo_.jettyInfos[i];
        cfg.jettyCfgs.emplace_back(JettyCfg{
            jettyInfo.jettyCtxId, outParam.dbVa, outParam.dbTokenId,
            inParam.tokenValue}); // 安全问题，禁止打印token相关信息
    }

    CHK_RET(CcuDevMgrImp::ConfigChannel(userDevId_, dieId_, cfg));
    return HcclResult::HCCL_SUCCESS;
}

CcuConnection::~CcuConnection() { (void)ReleaseConnRes(); }

HcclResult CcuConnection::ReleaseConnRes()
{
    const bool ctxValid = ctxHandle_ != nullptr && Hccl::RdmaHandleManager::GetInstance().IsHandleValid(ctxHandle_);

    for (auto& item : importJettyCtxs_) {
        if (item.outParam.handle != 0) {
            if (!ctxValid) {
                HCCL_WARNING(
                    "[CcuConnection][%s] skip RaCtxQpUnimport, ctxHandle=%p invalid, "
                    "remoteJettyHandle=%p",
                    __func__, ctxHandle_, item.outParam.handle);
                item.outParam.handle = 0;
                continue;
            }
            int32_t ret = RaCtxQpUnimport(ctxHandle_, item.outParam.handle);
            item.outParam.handle = 0;
            if (ret != 0) {
                HCCL_ERROR(
                    "[CcuComponent][%s] failed but passed, ctxHandle[%p] "
                    "remoteJettyHandle[%p], userDevId[%d].",
                    __func__, ctxHandle_, item.outParam.handle, userDevId_);
                status_ = CcuConnStatus::CONN_INVALID;
                innerStatus_ = InnerStatus::CONN_INVALID;
            }
        }
    }
    importJettyCtxs_.clear();

    if (tpProtocol_ == TpProtocol::RTP && tpInfo_.tpHandle != 0) {
        (void)TpMgr::GetInstance(devPhyId_).ReleaseTpAttr(tpInfo_.tpHandle, tpAttrInfo_);
    }

    if (tpInfo_.tpHandle != 0) { // tp handle 复用，只释放一次
        (void)TpMgr::GetInstance(devPhyId_).ReleaseTpInfo(MakeGetTpInfoParam(), tpInfo_);
        tpInfo_.tpHandle = 0;
        tpInfo_.hasMappedJettyPriority = false;
    }
    // CcuJetty 生命周期跟随通信域CcuJettyMgr
    // 不需要connection主动销毁
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuConnection::ReturnErrorStatus(const std::string& funcName)
{
    std::string errMsg = Hccl::StringFormat("[CcuConnection][%s] failed, [%s].", funcName.c_str(), Describe().c_str());
    status_ = CcuConnStatus::CONN_INVALID;
    innerStatus_ = InnerStatus::CONN_INVALID;
    HCCL_ERROR("%s", errMsg.c_str());
    return HcclResult::HCCL_E_INTERNAL;
}

std::string CcuConnection::Describe()
{
    Hccl::IpAddress locAddr{}, rmtAddr{};
    (void)CommAddrToIpAddress(locAddr_, locAddr);
    (void)CommAddrToIpAddress(rmtAddr_, rmtAddr);
    return Hccl::StringFormat(
        "[CcuConnection[locAddr=%s, rmtAddr=%s, protocol=%s, "
        "status=%s, innerStatus=%s, [dieId=%u, channelId=%u, jettyNum=%u]]]",
        locAddr.Describe().c_str(), rmtAddr.Describe().c_str(), tpProtocol_.Describe().c_str(),
        status_.Describe().c_str(), innerStatus_.Describe().c_str(), dieId_, channelInfo_.channelId, jettyNum_);
}

HcclResult CcuConnection::Describe(std::string& dfxMsg)
{
    uint16_t udpSport = 0xFFFF; // 无法获取实际的udpSport，使用0xFFFF表示未知
    if (tpProtocol_ == TpProtocol::RTP) {
        struct TpAttr tpAttr {};
        uint32_t attrBitmap = 1 << 13; // 13对应dataUdpSrcport
        EXCEPTION_HANDLE_BEGIN
        // HrtRaGetTpAttrAsync：封装内已同步等待，返回时 tpAttr 已就绪
        HcclResult ret
            = Hccl::HrtRaGetTpAttrAsync(devPhyId_, ctxHandle_, tpInfo_.tpHandle, attrBitmap, tpAttr, reqHandles_[0]);
        if (ret == HCCL_E_NOT_SUPPORT) {
            HCCL_ERROR(
                "[DevUbConnection::%s] failed, this package does not support RaGetTpAttrAsync for device,"
                " please change new package. devPhyId[%u]",
                __func__, devPhyId_);
            return ret;
        } else if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[DevUbConnection::%s] failed, hccl result[%d]", __func__, ret);
            return ret;
        }
        EXCEPTION_HANDLE_END
        udpSport = tpAttr.dataUdpSrcport;
    }
    udpSport = udpSport & 0xFF;

    std::ostringstream oss;
    for (size_t i = 0; i < ccuJettys_.size(); ++i) {
        uint16_t jettyId = ccuJettys_[i]->GetJettyedOutParam().id;
        if (i != 0) {
            oss << ", ";
        }
        oss << jettyId;
    }
    std::string jettyIds = oss.str();

    Hccl::IpAddress locAddr{}, rmtAddr{};
    CHK_RET(CommAddrToIpAddress(locAddr_, locAddr));
    CHK_RET(CommAddrToIpAddress(rmtAddr_, rmtAddr));
    Hccl::Eid locEid = locAddr.GetEid();
    Hccl::Eid rmtEid = rmtAddr.GetEid();

    std::string dfxStr = Hccl::StringFormat(
        "chip id[%u] die id[%u] func_id[%u] jetty id[%s] "
        "local %s remote %s udp sport[%u]",
        userDevId_, dieId_, funcId_, jettyIds.c_str(), locEid.Describe().c_str(), rmtEid.Describe().c_str(), udpSport);
    dfxMsg += dfxStr;
    HCCL_INFO("[CcuConnection::%s] %s", __func__, dfxStr.c_str());
    return HcclResult::HCCL_SUCCESS;
}

uint32_t CcuConnection::GetDieId() const { return dieId_; }

uint32_t CcuConnection::GetChannelId() const { return channelInfo_.channelId; }

int32_t CcuConnection::GetUserDevId() const { return userDevId_; }

uint64_t CcuConnection::GetRmtCcuBufAddr() const { return rmtCcuBufAddr_; }

uint32_t CcuConnection::GetRmtCcuBufTokenId() const { return rmtCcuBufTokenId_; }

uint32_t CcuConnection::GetRmtCcuBufTokenValue() const { return rmtCcuBufTokenValue_; }

HcclResult CcuConnection::Clean()
{
    status_ = CcuConnStatus::INIT;
    innerStatus_ = InnerStatus::INIT;
    isJettyCreated_ = false;
    isJettyImported_ = false;
    CHK_RET(ReleaseConnRes());
    GenerateLocalPsn();

    // 销毁jetty要在ReleaseConnRes之后
    for (auto& ccuJetty : ccuJettys_) {
        ccuJetty->Clean();
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace hcomm
