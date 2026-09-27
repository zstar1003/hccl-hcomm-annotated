/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dev_ub_connection.h"

#include <cstdlib>

#include "hccp_ctx.h"
#include "exception_util.h"
#include "rma_conn_exception.h"
#include "rdma_handle_manager.h"
#include "exchange_ub_conn_dto.h"
#include "env_config/env_config_v2.h"

namespace Hccl {

constexpr u32 OPBASED_UB_SQ_DEPTH_MAX = 8192;
constexpr u32 UB_SQ_OFFLOAD_DEPTH = 128;
constexpr u32 UB_SQ_WQEBB_SIZE = 64;
constexpr u32 WQE_NUM_PER_SQE = 4;                   // URMA约束每个SQE包含4个WQEBB
constexpr u32 UB_MAX_TRANS_SIZE = 256 * 1024 * 1024; // UB单次最大传输量256*1024*1024 Byte
constexpr uint32_t kTpAttrRetryTimesInitBit = 0U;
constexpr uint32_t kTpAttrAtBit = 1U;

DevUbConnection::DevUbConnection(
    const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
    const bool devUsed, const HrtUbJfcMode jfcMode, const IpAddress& locIpv4Addr, const IpAddress& rmtIpv4Addr, u8 qos,
    u8 taTimeOut, CommEngine engine, u32 inSqDepth, JettyMode jettyMode)
    : RmaConnection(nullptr, RmaConnType::UB),
      rdmaHandle(rdmaHandle),
      locAddr(locAddr),
      rmtAddr(rmtAddr),
      opMode(opMode),
      jfcMode(jfcMode),
      engine_(engine),
      locIpv4Addr(locIpv4Addr),
      rmtIpv4Addr(rmtIpv4Addr),
      rmtEid(rmtAddr.GetEid()),
      locEid(locAddr.GetEid()),
      rmtReverseEid(rmtAddr.GetReverseEid()),
      qos_(qos),
      devUsed_(devUsed),
      taTimeOut_(taTimeOut),
      sqDepth(inSqDepth),
      jettyMode_(jettyMode)
{
    HCCL_INFO(
        "[DevUbConnection::DevUbConnection] rmtEid=%s, engine=%d, jettyMode=%d", rmtEid.Describe().c_str(),
        static_cast<s32>(engine_), static_cast<s32>(jettyMode_));
    devLogicId = HrtGetDevice();

    auto dieIdAndFuncId = RdmaHandleManager::GetInstance().GetDieAndFuncId(rdmaHandle); // 获取dieId和FuncId
    dieId = dieIdAndFuncId.first;
    funcId = dieIdAndFuncId.second;

    // EXTERNAL_INJECT 模式：跳过建 JFC/jetty，等外部调 SetSharedJettyFields 填充
    if (jettyMode_ == JettyMode::EXTERNAL_INJECT) {
        sqDepth = OPBASED_UB_SQ_DEPTH_MAX;
        HCCL_INFO("[DevUbConnection][Constructor] EXTERNAL_INJECT mode, skip JFC/Jetty creation.");
        return;
    }

    if (engine_ == COMM_ENGINE_AIV) {
        CreateAivUrmaJfc();
    } else if (jfcMode == HrtUbJfcMode::USER_CTL) {
        jfcHandle = RdmaHandleManager::GetInstance().GetJfcHandleAndCqInfo(rdmaHandle, cqInfo_, jfcMode);
    } else {
        jfcHandle = RdmaHandleManager::GetInstance().GetJfcHandle(rdmaHandle, cqInfo_, jfcMode);
    }
    if (sqDepth == UB_SQ_DEPTH_NOT_SET) {
        sqDepth = OPBASED_UB_SQ_DEPTH_MAX;
        if (opMode == OpMode::OFFLOAD && !devUsed) {
            sqDepth = UB_SQ_OFFLOAD_DEPTH;
        }
    }
    HCCL_INFO(
        "[DevUbConnection][Constructor] sqDepth[%u], opMode[%d], devUsed[%d]", sqDepth, static_cast<s32>(opMode),
        devUsed);

    if (sqDepth > (UINT32_MAX / UB_SQ_WQEBB_SIZE / WQE_NUM_PER_SQE)) {
        THROW<InternalException>("integer overflow occurs");
    }

    if (!devUsed_) {
        CreateJetty(devUsed_);
    } else {
        HCCL_INFO("[DevUbConnection][Constructor] devUsed: defer CreateJetty until GetTpInfo maps qos.");
    }
}

DevUbTpConnection::DevUbTpConnection(
    const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
    const bool devUsed, const HrtUbJfcMode jfcMode, const IpAddress& locIpv4Addr, const IpAddress& rmtIpv4Addr, u8 qos,
    u8 taTimeOut, CommEngine engine, u32 sqDepth, JettyMode jettyMode)
    : DevUbConnection(
          rdmaHandle, locAddr, rmtAddr, opMode, devUsed, jfcMode, locIpv4Addr, rmtIpv4Addr, qos, taTimeOut, engine,
          sqDepth, jettyMode)
{
    tpProtocol = TpProtocol::TP;
}

DevUbCtpConnection::DevUbCtpConnection(
    const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
    const bool devUsed, const HrtUbJfcMode jfcMode, const IpAddress& locIpv4Addr, const IpAddress& rmtIpv4Addr, u8 qos,
    u8 taTimeOut, CommEngine engine, u32 sqDepth, JettyMode jettyMode)
    : DevUbConnection(
          rdmaHandle, locAddr, rmtAddr, opMode, devUsed, jfcMode, locIpv4Addr, rmtIpv4Addr, qos, taTimeOut, engine,
          sqDepth, jettyMode)
{
    tpProtocol = TpProtocol::CTP;
}

DevUbUboeConnection::DevUbUboeConnection(
    const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
    const bool devUsed, const HrtUbJfcMode jfcMode, const IpAddress& locIpv4Addr, const IpAddress& rmtIpv4Addr, u8 qos,
    u8 taTimeOut, CommEngine engine, u32 sqDepth, JettyMode jettyMode)
    : DevUbConnection(
          rdmaHandle, locAddr, rmtAddr, opMode, devUsed, jfcMode, locIpv4Addr, rmtIpv4Addr, qos, taTimeOut, engine,
          sqDepth, jettyMode)
{
    tpProtocol = TpProtocol::UBOE;
    jettyTimeOut = 16; // UBOE Jetty异步创建超时 hw_val=16 (对应8s)
}

DevUbRtpConnection::DevUbRtpConnection(
    const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
    const bool devUsed, const HrtUbJfcMode jfcMode, const IpAddress& locAddrEid, const IpAddress& rmtAddrEid,
    const u8 qos, u8 taTimeOut, CommEngine engine, u32 sqDepth, JettyMode jettyMode)
    : DevUbConnection(
          rdmaHandle, locAddr, rmtAddr, opMode, devUsed, jfcMode, locAddrEid, rmtAddrEid, qos, taTimeOut, engine,
          sqDepth, jettyMode)
{
    tpProtocol = TpProtocol::UB_RTP;
    // UB_RTP与UBOE同属UB传输，Jetty异步创建超时一致，均为16秒
    jettyTimeOut = 16;
}

std::vector<char> DevUbConnection::GetUniqueId() const
{
    BinaryStream binaryStream;
    binaryStream << dieId;
    binaryStream << funcId;
    binaryStream << jettyId;

    u32 jfcPollMode = 0;          // 待修改，0代表STARS POLL，1代表software Poll
    bool dwqeCacheLocked = false; // 待修改，该jetty是否支持dwqeCachedLocked，默认不支持
    u64 sqCiAddr = 0; // 待修改，软件poll CQ情况下，需要AICPU从该地址中读取CI,依赖UB驱动支持
    binaryStream << jfcPollMode;
    binaryStream << dwqeCacheLocked;
    binaryStream << dbAddr;
    binaryStream << sqCiAddr;
    binaryStream << sqBuffVa;
    binaryStream << sqDepth;
    binaryStream << tpn;
    binaryStream << rmtEid.raw;
    binaryStream << locEid.raw;
    binaryStream << maxReadSize;
    binaryStream << maxWriteSize;
    binaryStream << static_cast<uint64_t>(jettyHandle);

    std::vector<char> result;
    binaryStream.Dump(result);
    HCCL_INFO("DevUbConnection::GetUniqueId:%s", Describe().c_str());
    HCCL_INFO(
        "type=%s, jfcPollMode=%u, dwqeCacheLocked=%d, sqCiAddr=0x%llx", rmaConnType.Describe().c_str(), jfcPollMode,
        dwqeCacheLocked, sqCiAddr);
    return result;
}

void DevUbConnection::SetCqInfo(HcclAiRMACQ& cq) const
{
    cq.jfcId = cqInfo_.id;
    cq.cqVA = cqInfo_.va;
    cq.cqeSize = cqInfo_.cqeSize;
    cq.cqDepth = cqInfo_.cqDepth;
    cq.dbAddr = cqInfo_.swdbAddr;
}

void DevUbConnection::SetWqInfo(HcclAiRMAWQ& wq)
{
    wq.jettyId = jettyId;
    wq.dbAddr = dbAddr;
    wq.sqVA = sqBuffVa;
    wq.sqDepth = sqDepth * WQE_NUM_PER_SQE;
    wq.tp_id = tpn;
    errno_t ret = memcpy_s(wq.rmtEid, sizeof(wq.rmtEid), rmtReverseEid.raw, sizeof(wq.rmtEid));
    if (ret != EOK) {
        HCCL_ERROR("[DevUbConnection][%s] memcpy_s failed, ret=%d", __func__, ret);
        ThrowAbnormalStatus(std::string(__func__));
    }
}

void DevUbConnection::SetSqContextInfo(SqContext& sq)
{
    sq.contextInfo.ubJfs.jfsID = jettyId;
    sq.contextInfo.ubJfs.dbVa = dbAddr;
    sq.contextInfo.ubJfs.sqVa = sqBuffVa;
    sq.contextInfo.ubJfs.sqDepth = sqDepth * WQE_NUM_PER_SQE;
    sq.contextInfo.ubJfs.tpID = tpn;
    errno_t ret = memcpy_s(
        sq.contextInfo.ubJfs.remoteEID, sizeof(sq.contextInfo.ubJfs.remoteEID), rmtReverseEid.raw,
        sizeof(sq.contextInfo.ubJfs.remoteEID));
    if (ret != EOK) {
        HCCL_ERROR("[DevUbConnection][%s] memcpy_s failed, ret=%d", __func__, ret);
        ThrowAbnormalStatus(std::string(__func__));
    }
}

void DevUbConnection::SetCqContextInfo(CqContext& cq) const
{
    cq.contextInfo.ubJfc.jfcID = cqInfo_.id;
    cq.contextInfo.ubJfc.scqVa = cqInfo_.va;
    cq.contextInfo.ubJfc.cqeSize = cqInfo_.cqeSize;
    cq.contextInfo.ubJfc.cqDepth = cqInfo_.cqDepth;
    cq.contextInfo.ubJfc.dbVa = cqInfo_.swdbAddr;
}

void DevUbConnection::Connect() { GetStatus(); }

inline uint32_t GetRandomNum()
{
    uint32_t randNum = std::rand();
    return randNum;
}

HcclResult DevUbConnection::CalcTotalTimeout(uint32_t& outTotalTimeoutMs)
{
    TpHandle tpHandle = tpInfo.tpHandle;
    uint32_t attrBitmap = (1U << kTpAttrRetryTimesInitBit) | (1U << kTpAttrAtBit);
    struct TpAttr tpAttr = {};
    u32 devicePhyId = HrtGetDevicePhyIdByUserDevId(devLogicId);
    CHK_RET(HrtRaGetTpAttrAsync(devicePhyId, rdmaHandle, tpHandle, attrBitmap, tpAttr, reqHandle));
    TpAttrInfo tpAttrInfo = TpAttrInfo(tpAttr);
    CHK_RET(TpManager::GetTpTotalTimeout(tpAttrInfo, outTotalTimeoutMs));
    return HCCL_SUCCESS;
}

void DevUbConnection::GetTimeOut() // 基于调用方按协议从环境变量获取并传入的超时值控制
{
    if (tpProtocol == TpProtocol::INVALID) { // 不感知tp建链，当前默认不支持
        HCCL_ERROR(
            "[DevUbConnection][%s] failed, tpProtocol[%s] is not expected.", __func__, tpProtocol.Describe().c_str());
        ThrowAbnormalStatus(std::string(__func__));
        return;
    }

    // CTP 协议不感知 TP 建链，跳过 CalcTotalTimeout，
    // tpTimeOutMs 保持 0，由 TpManager::CalcTaTimeout 内部按 CTP 规则直接使用 taTimeOut_
    uint32_t tpTimeOutMs = 0;
    if (tpProtocol != TpProtocol::CTP) {
        HcclResult ret = CalcTotalTimeout(tpTimeOutMs);
        if (ret != HCCL_SUCCESS) {
            HCCL_RUN_WARNING(
                "[DevUbConnection][%s] CalcTotalTimeout failed[%d], tpTimeOutMs remains 0.", __func__, ret);
        }
    }
    jettyTimeOut = TpManager::CalcTaTimeout(tpProtocol, taTimeOut_, tpTimeOutMs);
    HCCL_INFO(
        "[DevUbConnection][%s] final TA Timeout [%u] (%ums).", __func__, jettyTimeOut,
        TpManager::TaHwValueToMs(jettyTimeOut));
}

/*
 * UB 建链状态机（GetTpInfo/CreateJetty 异步未完成则停；同步成功时可同次推进）:
 *   INIT                  --GetTpInfo fail--> TP_INFO_GETTING
 *   INIT / TP_INFO_GETTING --GetTpInfo ok--> CreateJetty --> JETTY_CREATING | JETTY_CREATED
 *                         （EXTERNAL_INJECT 模式时跳过 CreateJetty，直接 JETTY_CREATED）
 *   JETTY_CREATING         --create done--> JETTY_CREATED (EXCHANGEABLE)
 *   JETTY_CREATED          --ImportRmtDto--> JETTY_IMPORTING（此处不推进）
 *   JETTY_IMPORTING        --import done--> READY
 */
void DevUbConnection::AdvanceUbConnFromJettyImporting()
{
    SetImportInfo();

    if (jettyMode_ == JettyMode::EXTERNAL_INJECT) {
        if (publishRemoteCb_ == nullptr) {
            THROW<InternalException>("[DevUbConnection][%s] publish callback is null.", __func__);
        }
        HcclResult ret = publishRemoteCb_(remoteQpKey, remoteKeySize, remoteJettyHandle, remoteJettyHandlePtr, tpn);
        if (ret != HCCL_SUCCESS) {
            THROW<InternalException>(
                "[DevUbConnection][%s] publish shared remote jetty failed, ret[%d].", __func__, ret);
        }
    }

    status = RmaConnStatus::READY;
    ubConnStatus = UbConnStatus::READY;
}

void DevUbConnection::AdvanceUbConnFromJettyImportWaiting()
{
    // 超时检查：超过 jettyTimeOut 秒仍未 publish，则对端异常，抛异常避免永久阻塞。
    // 每次轮询都检查，保证超时判定不被退避延迟。
    auto elapsedSec
        = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - importWaitingStart_);
    if (elapsedSec.count() >= static_cast<int64_t>(jettyTimeOut)) {
        HCCL_ERROR(
            "[DevUbConnection][%s] JETTY_IMPORT_WAITING timeout[%us], remote jetty not published.", __func__,
            static_cast<uint32_t>(elapsedSec.count()));
        THROW<InternalException>("[DevUbConnection][%s] wait shared remote jetty publish timeout.");
    }

    // 退避：每 8 次 GetStatus 轮询才真正调 acquireRemoteCb_ 一次，减少 JettyContext 锁竞争。
    // 首次进入立即调用（pollCount==0），后续每 8 次轮询触发一次实际查询。
    if (importWaitingPollCount_ > 0 && (importWaitingPollCount_ % 8U) != 0U) {
        importWaitingPollCount_++;
        return;
    }
    importWaitingPollCount_++;

    AcquireOrWaitSharedRemoteJetty();
}

RmaConnStatus DevUbConnection::GetStatus()
{
    // 稳定态 / 等待外部 ImportRmtDto：无需推进
    if (ubConnStatus == UbConnStatus::READY || ubConnStatus == UbConnStatus::JETTY_CREATED) {
        return status;
    }

    if (!CheckRequestResult()) {
        return status;
    }

    switch (ubConnStatus) {
        case UbConnStatus::INIT:
            ProcessInit();
            break;
        case UbConnStatus::TP_INFO_GETTING:
            if (!GetTpInfo()) {
                break;
            }
            ProcessCreateJetty();
            break;
        case UbConnStatus::JETTY_CREATING:
            SetJettyInfo();
            status = RmaConnStatus::EXCHANGEABLE;
            ubConnStatus = UbConnStatus::JETTY_CREATED;
            break;
        case UbConnStatus::JETTY_IMPORTING:
            AdvanceUbConnFromJettyImporting();
            break;
        case UbConnStatus::JETTY_IMPORT_WAITING:
            AdvanceUbConnFromJettyImportWaiting();
            break;
        case UbConnStatus::READY:
            break;
        default:
            ThrowAbnormalStatus(std::string(__func__));
            break;
    }

    return status;
}

void DevUbConnection::ProcessInit()
{
    HCCL_INFO(
        "[DevUbConnection][%s] start, status[%s], ubConnStatus[%s].", __func__, status.Describe().c_str(),
        ubConnStatus.Describe().c_str());
    if (!GetTpInfo()) {
        ubConnStatus = UbConnStatus::TP_INFO_GETTING;
        return;
    }
    ProcessCreateJetty();
}

void DevUbConnection::ProcessCreateJetty()
{
    GetTimeOut();
    if (jettyMode_ == JettyMode::EXTERNAL_INJECT) {
        // 共享 jetty：句柄已由 SetSharedJettyFields 注入，跳过 CreateJetty
        status = RmaConnStatus::EXCHANGEABLE;
        ubConnStatus = UbConnStatus::JETTY_CREATED;
        HCCL_INFO("[DevUbConnection][%s] shared jetty mode, skip CreateJetty, direct to JETTY_CREATED.", __func__);
        return;
    }
    CreateJetty(devUsed_);
    if (devUsed_ || !CheckRequestResult()) {
        ubConnStatus = UbConnStatus::JETTY_CREATING;
        return;
    }
    SetJettyInfo();
    status = RmaConnStatus::EXCHANGEABLE;
    ubConnStatus = UbConnStatus::JETTY_CREATED;
}

std::unique_ptr<Serializable> DevUbConnection::GetExchangeDto()
{
    if (status != RmaConnStatus::READY && status != RmaConnStatus::EXCHANGEABLE) {
        HCCL_ERROR("[DevUbConnection][%s] status[%s] is not expected.", __func__, status.Describe().c_str());
        ThrowAbnormalStatus(std::string(__func__));
    }

    if (tpProtocol != TpProtocol::INVALID) {
        jettyImportCfg.localTpHandle = tpInfo.tpHandle;

        HCCL_INFO(
            "[DevUbConnection][%s] tpEnable, localTpHandle[0x%llx] localPsn[%u].", __func__,
            jettyImportCfg.localTpHandle, jettyImportCfg.localPsn);
    }

    std::unique_ptr<ExchangeUbConnDto> dto
        = make_unique<ExchangeUbConnDto>(tokenValue, keySize, jettyImportCfg.localTpHandle, jettyImportCfg.localPsn);
    (void)memcpy_s(dto->qpKey, HRT_UB_QP_KEY_MAX_LEN, localQpKey, HRT_UB_QP_KEY_MAX_LEN);
    return std::unique_ptr<Serializable>(dto.release());
}

void DevUbConnection::ParseRmtExchangeDto(const Serializable& rmtDto)
{
    auto dto = dynamic_cast<const ExchangeUbConnDto&>(rmtDto);
    HCCL_INFO("[DevUbConnection][%s] remoteConnDto[%s]", __func__, dto.Describe().c_str());
    if (dto.qpKeySize == 0 || dto.qpKeySize > HRT_UB_QP_KEY_MAX_LEN) {
        THROW<InvalidParamsException>(
            "[DevUbConnection][%s] invalid remote qpKeySize[%u], max[%u].", __func__, dto.qpKeySize,
            HRT_UB_QP_KEY_MAX_LEN);
    }
    remoteTokenValue = dto.tokenValue;
    remoteKeySize = dto.qpKeySize;
    (void)memcpy_s(remoteQpKey, HRT_UB_QP_KEY_MAX_LEN, dto.qpKey, remoteKeySize);

    if (tpProtocol != TpProtocol::INVALID) {
        jettyImportCfg.remoteTpHandle = dto.tpHandle;
        jettyImportCfg.remotePsn = dto.psn;
        HCCL_INFO(
            "[DevUbConnection][%s] tpEnable, remoteTpHandle[0x%llx], remotePsn[%u].", __func__,
            jettyImportCfg.remoteTpHandle, jettyImportCfg.remotePsn);
    }
}

void DevUbConnection::ImportRmtDto()
{
    if (ubConnStatus == UbConnStatus::READY) {
        HCCL_WARNING("[DevUbConnection][%s] import jetty already, %s.", __func__, Describe().c_str());
        return;
    }

    if (ubConnStatus != UbConnStatus::JETTY_CREATED) {
        HCCL_ERROR(
            "[DevUbConnection][%s] failed, ubConnStatus[%s] is not expected.", __func__,
            ubConnStatus.Describe().c_str());
        ThrowAbnormalStatus(std::string(__func__));
    }

    if (jettyMode_ == JettyMode::EXTERNAL_INJECT) {
        AcquireOrWaitSharedRemoteJetty();
        return;
    }

    ImportJetty();
    ubConnStatus = UbConnStatus::JETTY_IMPORTING;
}

void DevUbConnection::ThrowAbnormalStatus(std::string funcName)
{
    auto errMsg = StringFormat("[DevUbConnection][%s] failed, [%s].", funcName.c_str(), Describe().c_str());
    status = RmaConnStatus::CONN_INVALID;
    ubConnStatus = UbConnStatus::CONN_INVALID;
    THROW<RmaConnException>(errMsg);
}

bool DevUbConnection::CheckRequestResult()
{
    if (reqHandle == 0) {
        return true;
    }

    ReqHandleResult result = HrtRaGetAsyncReqResult(reqHandle);
    if (result == ReqHandleResult::NOT_COMPLETED) {
        return false;
    }

    if (result != ReqHandleResult::COMPLETED) {
        THROW<InternalException>(
            "[DevUbConnection][%s] failed, result[%s] is unexpected.", __func__, result.Describe().c_str());
    }

    return true;
}

void DevUbConnection::CreateJetty(const bool devUsed)
{
    if (sqDepth > UINT32_MAX / UB_SQ_WQEBB_SIZE / WQE_NUM_PER_SQE) {
        THROW<InternalException>(
            "[DevUbConnection][%s] failed, sqDepth[%u] times "
            "UB_SQ_WQEBB_SIZE[%u] overflow uint32 max.",
            __func__, sqDepth, UB_SQ_WQEBB_SIZE);
    }
    u32 size = static_cast<u32>(sqDepth) * static_cast<u32>(UB_SQ_WQEBB_SIZE) * static_cast<u32>(WQE_NUM_PER_SQE);
    HrtRaUbCreateJettyParam req{
        jfcHandle,
        jfcHandle,
        GetUbToken(),
        0,
        HrtJettyMode::HOST_OPBASE, // 默认HOST单算子模式
        0,                         // HOST展开与AICPU展开传入jetty id为0，申请一个新的jetty
        0,                         // va由底层分配，此处填0即可。
        size,
        0,
        sqDepth,
        jettyTimeOut}; // 非CCUv2不需要填写sqeBufIndex

    if (opMode == OpMode::OFFLOAD) { // HOST展开图模式切换模式
        req.jettyMode = HrtJettyMode::HOST_OFFLOAD;
    }

    if (devUsed) { // AICPU场景切换模式
        req.jettyMode = HrtJettyMode::DEV_USED;
        HCCL_INFO("[DevUbConnection][%s] HrtJettyMode is DEV_USED.", __func__);
    }

    if (tpInfo.hasMappedJettyPriority) {
        req.qos = static_cast<u8>(tpInfo.mappedJettyPriority & 0xFU);
    }
    HCCL_INFO(
        "[DevUbConnection][%s] jetty create qos[%u] (maps to attr.ub.priority lower 4 bits).", __func__,
        static_cast<unsigned int>(req.qos));

    reqHandle = RaUbCreateJettyAsync(rdmaHandle, req, reqDataBuffer, jettyHandlePtr);
}

HcclResult DevUbConnection::SetSharedJettyFields(
    JettyHandle jettyHdl, void* jettyHdlPtr, uint32_t jId, uint64_t sqVa, uint64_t db, const uint8_t* qpKey,
    uint32_t kSize, uint32_t sDepth, JfcHandle sharedJfc, CqCreateInfo sharedCqInfo, uint32_t sharedLocalPsn,
    void* epTag, std::function<void(void*)> releaseCb, AcquireSharedRemoteJettyCallback acquireRemoteCb,
    PublishSharedRemoteJettyCallback publishRemoteCb)
{
    if (jettyHdl == 0 || jettyHdlPtr == nullptr || sDepth == 0 || acquireRemoteCb == nullptr
        || publishRemoteCb == nullptr) {
        HCCL_ERROR(
            "[DevUbConnection][%s] invalid params, jettyHdl[0x%llx], jettyHdlPtr[%p], sDepth[%u], "
            "acquireRemoteCb[%d], publishRemoteCb[%d].",
            __func__, static_cast<unsigned long long>(jettyHdl), jettyHdlPtr, sDepth,
            acquireRemoteCb == nullptr ? 0 : 1, publishRemoteCb == nullptr ? 0 : 1);
        return HCCL_E_PARA;
    }
    if (jettyMode_ != JettyMode::EXTERNAL_INJECT) {
        HCCL_ERROR(
            "[DevUbConnection][%s] not EXTERNAL_INJECT mode, jettyMode[%d], reject SetSharedJettyFields.", __func__,
            static_cast<s32>(jettyMode_));
        return HCCL_E_INTERNAL;
    }
    // EXTERNAL_INJECT 模式构造时跳过了建 JFC/jetty，此处填充共享字段，无需销毁私有 JFC
    if (qpKey != nullptr && kSize > 0 && kSize <= HRT_UB_QP_KEY_MAX_LEN) {
        s32 ret = memcpy_s(&localQpKey[0], HRT_UB_QP_KEY_MAX_LEN, qpKey, kSize);
        if (ret != EOK) {
            HCCL_ERROR("[DevUbConnection][%s] memcpy_s localQpKey failed, ret[%d].", __func__, ret);
            return HCCL_E_INTERNAL;
        }
    }
    endpointTag_ = epTag;
    releaseCb_ = std::move(releaseCb);
    acquireRemoteCb_ = std::move(acquireRemoteCb);
    publishRemoteCb_ = std::move(publishRemoteCb);
    // 共享 jetty 模式下主 connection 仍各自调 GetTpInfo 申请本 pair 的 tpHandle（TpManager useCnt++），
    // 析构时必须 ReleaseTp 释放引用，否则 useCnt 泄漏导致 tpHandle 永不回收。
    releaseTpOnDestroy_ = true;
    jettyHandle = jettyHdl;
    jettyHandlePtr = jettyHdlPtr;
    jettyId = jId;
    sqBuffVa = sqVa;
    dbAddr = db;
    keySize = kSize;
    sqDepth = sDepth;
    jfcHandle = sharedJfc;
    cqInfo_ = sharedCqInfo;
    // 注入临时 connection 生成的 localPsn，使主 connection 的 GetExchangeDto 发送与共享 jetty
    // 一致的 psn。多个主 connection 共享同一本地 jetty/SQ，必须用同一 localPsn，避免各自
    // GenerateLocalPsn 生成不同 psn 后 import 同一 TP 对时 psn 互相覆盖导致硬件传输错乱。
    // 注意：tpHandle 不在此注入——一对多场景下各主 connection 到不同对端需各自向管控面申请
    // 自己的 tpHandle，否则对端 import 时 peerTpHandle 路由不匹配。
    jettyImportCfg.localPsn = sharedLocalPsn;
    // 注入后不直接跳状态机：仍需走 GetTpInfo 申请本 pair 的 TP，由 ProcessCreateJetty
    // 中 EXTERNAL_INJECT 分支跳过 CreateJetty 直接进入 JETTY_CREATED
    HCCL_INFO(
        "[DevUbConnection][%s] shared jetty fields set, handle[0x%llx], jettyId[%u], sqDepth[%u], "
        "jfcHandle[%llu].",
        __func__, static_cast<unsigned long long>(jettyHandle), jettyId, sqDepth,
        static_cast<unsigned long long>(jfcHandle));
    return HCCL_SUCCESS;
}

void DevUbConnection::DetachJetty()
{
    // SELF_CREATE 模式建好 jetty 后调用：分离 jetty 所有权，析构不销毁 jetty，交由 JettyContext 管理。
    // 注意：不置 releaseTpOnDestroy_=false——临时 connection 自己申请的 TP 引用仍需在析构时
    // ReleaseTp 释放（TpManager 引用计数 -1），否则 useCnt 泄漏。
    // 共享 jetty 模式下主 connection 不复用临时 connection 的 tpHandle：一对多场景各主 connection
    // 到不同对端需各自向 TpManager 申请本 pair 的 tpHandle，否则对端 import 时 peerTpHandle 路由不匹配。
    // 临时 connection 的 tpHandle 引用随析构释放，不影响主 connection 各自申请的 tpHandle。
    jettyDetached_ = true;
    HCCL_INFO(
        "[DevUbConnection][%s] jetty ownership detached, handle[0x%llx].", __func__,
        static_cast<unsigned long long>(jettyHandle));
}

HcclResult DevUbConnection::GetJettyInfo(JettyInfo& info) const
{
    info.handle = jettyHandle;
    info.handlePtr = jettyHandlePtr;
    info.jettyId = jettyId;
    info.sqBuffVa = sqBuffVa;
    info.dbAddr = dbAddr;
    info.keySize = keySize;
    info.sqDepth = sqDepth;
    info.rdmaHandle = rdmaHandle;
    info.jfcHandle = jfcHandle;
    info.cqInfo = cqInfo_;
    info.localPsn = jettyImportCfg.localPsn;
    auto sRet = memcpy_s(&info.localQpKey[0], HRT_UB_QP_KEY_MAX_LEN, localQpKey, HRT_UB_QP_KEY_MAX_LEN);
    if (sRet != EOK) {
        HCCL_ERROR("[DevUbConnection][%s] memcpy_s failed, ret[%d].", __func__, sRet);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

void DevUbConnection::SetJettyInfo()
{
    struct QpCreateInfo* info = reinterpret_cast<QpCreateInfo*>(reqDataBuffer.data());
    jettyId = info->ub.id;
    jettyHandle = reinterpret_cast<JettyHandle>(jettyHandlePtr);
    keySize = info->key.size;
    sqBuffVa = info->ub.sqBuffVa; // hccp提供
    HCCL_RUN_INFO(
        "[DevUbConnection][%s] Get sqBuffVa is %llx. jettyId[%u], jettyHandle[%llx], dieId[%u], funcId[%u]", __func__,
        sqBuffVa, jettyId, jettyHandle, dieId, funcId);

    s32 ret = memcpy_s(&localQpKey[0], HRT_UB_QP_KEY_MAX_LEN, info->key.value, info->key.size);
    if (ret != 0) {
        THROW<InternalException>(StringFormat("[DevUbConnection][%s] memcpy_s failed, ret=%d", __func__, ret));
    }

    dbAddr = info->ub.dbAddr;
}

bool DevUbConnection::GetTpInfo()
{
    if (tpProtocol == TpProtocol::INVALID) { // 不感知tp建链，当前默认不支持
        HCCL_ERROR(
            "[DevUbConnection][%s] failed, tpProtocol[%s] is not expected.", __func__, tpProtocol.Describe().c_str());
        ThrowAbnormalStatus(std::string(__func__));
    }

    // 共享 jetty 模式：tpHandle 不复用临时 connection 的值——一对多场景下各主 connection 到不同对端
    // 必须各自向管控面申请本 pair 的 tpHandle，否则对端 import 时 peerTpHandle 路由不匹配。
    // 但 localPsn 已由 SetSharedJettyFields 注入（来自临时 connection），不在此 GenerateLocalPsn，
    // 避免多主 connection 共享同一本地 jetty/SQ 各自生成不同 psn 后 import 同一 TP 对时互相覆盖。
    const bool isSharedJettyMode = (jettyMode_ == JettyMode::EXTERNAL_INJECT);

    RaUbGetTpInfoParam p{};
    p.locAddr = locAddr;
    p.rmtAddr = rmtAddr;
    p.tpProtocol = tpProtocol;
    p.qos = static_cast<uint32_t>(qos_);
    p.slLevelCount = 0;
    p.loopFirstTpLowestSl = false;
    p.locIpv4Addr = locIpv4Addr;
    p.rmtIpv4Addr = rmtIpv4Addr;

    auto ret = TpManager::GetInstance(devLogicId).GetTpInfo(p, tpInfo);

    switch (ret) {
        case HcclResult::HCCL_SUCCESS:
            if (isSharedJettyMode) {
                HCCL_INFO(
                    "[DevUbConnection][%s] shared jetty mode, apply own tpHandle[0x%llx] for rmtAddr[%s], "
                    "reuse injected localPsn[%u].",
                    __func__, static_cast<unsigned long long>(tpInfo.tpHandle), rmtAddr.Describe().c_str(),
                    jettyImportCfg.localPsn);
            } else if (tpProtocol == TpProtocol::UB_RTP) {
                GenerateLocalPsn();
                const HcclResult psnRet
                    = TpManager::GetInstance(devLogicId).GetOrSetLocalPsn(p, tpInfo, jettyImportCfg.localPsn);
                if (psnRet != HcclResult::HCCL_SUCCESS) {
                    HCCL_ERROR("[DevUbConnection][%s] get UB_RTP local PSN failed, ret[%d].", __func__, psnRet);
                    ThrowAbnormalStatus(std::string(__func__));
                }
            } else {
                GenerateLocalPsn();
            }
            return true;
        case HcclResult::HCCL_E_AGAIN:
            return false;
        case HcclResult::HCCL_E_NOT_FOUND:
        default:
            HCCL_ERROR("[DevUbConnection][%s] failed, hccl result[%d]", __func__, ret);
            ThrowAbnormalStatus(std::string(__func__));
            break;
    }
    return true;
}

void DevUbConnection::GenerateLocalPsn() { jettyImportCfg.localPsn = GetRandomNum(); }

void DevUbConnection::ImportJetty()
{
    HrtRaUbJettyImportedInParam in{};
    in.key = remoteQpKey;
    in.keyLen = remoteKeySize;
    in.tokenValue = remoteTokenValue;
    in.jettyImportCfg = jettyImportCfg;
    in.jettyImportCfg.protocol = tpProtocol;

    if (tpProtocol != TpProtocol::CTP && tpProtocol != TpProtocol::TP && tpProtocol != TpProtocol::UBOE
        && tpProtocol != TpProtocol::UB_RTP) {
        HCCL_ERROR(
            "[DevUbConnection][%s] failed, tp protocol[%s] is not expected, %s.", __func__,
            tpProtocol.Describe().c_str(), Describe().c_str());
        ThrowAbnormalStatus(std::string(__func__));
    }

    reqHandle = RaUbTpImportJettyAsync(rdmaHandle, in, reqDataBuffer, remoteJettyHandlePtr);
}

void DevUbConnection::SetImportInfo()
{
    struct QpImportInfoT* info = reinterpret_cast<QpImportInfoT*>(reqDataBuffer.data());
    remoteJettyHandle = reinterpret_cast<TargetJettyHandle>(remoteJettyHandlePtr);
    tpn = info->out.ub.tpn;
}

void DevUbConnection::SetSharedRemoteJettyInfo(TargetJettyHandle handle, void* handlePtr, uint32_t remoteTpn)
{
    remoteJettyHandle = handle;
    remoteJettyHandlePtr = handlePtr;
    tpn = remoteTpn;
    status = RmaConnStatus::READY;
    ubConnStatus = UbConnStatus::READY;
    HCCL_INFO(
        "[DevUbConnection][%s] reuse shared remote jetty, handle[0x%llx], tpn[%u].", __func__,
        static_cast<unsigned long long>(handle), remoteTpn);
}

void DevUbConnection::AcquireOrWaitSharedRemoteJetty()
{
    if (acquireRemoteCb_ == nullptr) {
        THROW<InternalException>("[DevUbConnection][%s] acquire callback is null.", __func__);
    }

    bool needImport = false;
    TargetJettyHandle cachedHandle = 0;
    void* cachedHandlePtr = nullptr;
    uint32_t cachedTpn = 0;
    HcclResult ret = acquireRemoteCb_(remoteQpKey, remoteKeySize, needImport, cachedHandle, cachedHandlePtr, cachedTpn);
    if (ret != HCCL_SUCCESS) {
        THROW<InternalException>("[DevUbConnection][%s] acquire shared remote jetty failed, ret[%d].", __func__, ret);
    }

    if (cachedHandle != 0) {
        SetSharedRemoteJettyInfo(cachedHandle, cachedHandlePtr, cachedTpn);
        return;
    }
    if (needImport) {
        ImportJetty();
        ubConnStatus = UbConnStatus::JETTY_IMPORTING;
        HCCL_INFO("[DevUbConnection][%s] start shared remote jetty import.", __func__);
        return;
    }
    // 进入 WAITING：记录起始时刻，供 AdvanceUbConnFromJettyImportWaiting 做超时判断
    if (ubConnStatus != UbConnStatus::JETTY_IMPORT_WAITING) {
        importWaitingStart_ = std::chrono::steady_clock::now();
        importWaitingPollCount_ = 0;
    }
    ubConnStatus = UbConnStatus::JETTY_IMPORT_WAITING;
}

void DevUbConnection::ReleaseTp()
{
    ReleaseUbConnectionTp(devLogicId, locAddr, rmtAddr, tpProtocol, tpInfo, static_cast<uint32_t>(qos_));
}

void DevUbConnection::ReleaseRemoteJettyIfImported(bool ctxValid)
{
    // EXTERNAL_INJECT 模式（主 connection）：远端 jetty 由 JettyContext 统一 unimport，不在此处理
    if (jettyMode_ == JettyMode::EXTERNAL_INJECT) {
        return;
    }
    if (remoteJettyHandle == 0) {
        return;
    }
    if (!ctxValid) {
        HCCL_WARNING(
            "[DevUbConnection][%s] skip HrtRaUbUnimportJetty, "
            "rdmaHandle=%p invalid (DeInit/DestroyAll done), remoteJettyHandle=0x%llx",
            __func__, rdmaHandle, static_cast<unsigned long long>(remoteJettyHandle));
    } else {
        HrtRaUbUnimportJetty(rdmaHandle, remoteJettyHandle);
    }
    remoteJettyHandle = 0;
}

void DevUbConnection::ReleaseSharedJettyModeResources(bool ctxValid)
{
    // EXTERNAL_INJECT 模式（主 connection）：构造时跳过 JFC/jetty 创建，jetty + JFC + CQ 全部由
    // JettyContext 统一管理，connection 不销毁。通过 releaseCb_ 通知 Endpoint 减引用计数
    // （refCount 归 0 时 JettyContext 销毁共享资源）。
    jettyHandle = 0;
    jfcHandle = 0;
    if (releaseCb_) {
        releaseCb_(endpointTag_);
        releaseCb_ = nullptr;
    }
    HCCL_INFO("[DevUbConnection][%s] EXTERNAL_INJECT mode, skip DestroyJetty, releaseCb invoked.", __func__);
    (void)ctxValid; // EXTERNAL_INJECT 模式无私有 JFC 需销毁，ctxValid 不影响
}

void DevUbConnection::ReleaseOwnedJettyAndJfc(bool ctxValid)
{
    if (jettyHandle != 0) {
        if (!ctxValid) {
            HCCL_WARNING(
                "[DevUbConnection][%s] skip HrtRaUbDestroyJetty, "
                "rdmaHandle=%p invalid, jettyHandle=0x%llx",
                __func__, rdmaHandle, static_cast<unsigned long long>(jettyHandle));
        } else {
            HrtRaUbDestroyJetty(jettyHandle);
        }
        jettyHandle = 0;
    }

    if (engine_ == COMM_ENGINE_AIV && jfcHandle != 0) {
        if (!ctxValid) {
            HCCL_WARNING(
                "[DevUbConnection][%s] skip HrtRaUbDestroyJfc, "
                "rdmaHandle=%p invalid, jfcHandle=0x%llx",
                __func__, rdmaHandle, static_cast<unsigned long long>(jfcHandle));
        } else {
            HrtRaUbDestroyJfc(rdmaHandle, jfcHandle);
        }
        jfcHandle = 0;
    }
}

void DevUbConnection::ReleaseResource()
{
    const bool ctxValid = (rdmaHandle != nullptr) && RdmaHandleManager::GetInstance().IsHandleValid(rdmaHandle);

    ReleaseRemoteJettyIfImported(ctxValid);

    if (releaseTpOnDestroy_) {
        ReleaseTp();
    }

    // EXTERNAL_INJECT 模式（主 connection）：jetty/JFC 由 JettyContext 统一管理，调 releaseCb_ 减引用
    if (jettyMode_ == JettyMode::EXTERNAL_INJECT) {
        ReleaseSharedJettyModeResources(ctxValid);
        return;
    }

    // SELF_CREATE + DetachJetty（临时 connection）：jetty/JFC 所有权已移交给 JettyContext，不销毁
    if (jettyDetached_) {
        HCCL_INFO("[DevUbConnection][%s] SELF_CREATE + DetachJetty, skip DestroyJetty/Jfc.", __func__);
        jettyHandle = 0;
        jfcHandle = 0;
        return;
    }

    ReleaseOwnedJettyAndJfc(ctxValid);
}

void DevUbConnection::CreateAivUrmaJfc()
{
    jfcHandle = HrtRaUbCreateJfcUserCtl(rdmaHandle, cqInfo_);
    HCCL_INFO("[DevUbConnection][CreateAivUrmaJfc] create jfcHandle[%llu] for rdmaHandle[%p].", jfcHandle, rdmaHandle);
}

DevUbConnection::~DevUbConnection() { DECTOR_TRY_CATCH("DevUbConnection", ReleaseResource()); }

// Suspend接口当前已不使用，由框架调用触发析构流程
bool DevUbConnection::Suspend()
{
    HCCL_WARNING("[DevUbConnection][%s] should not be called.", __func__);
    if (status == RmaConnStatus::SUSPENDED) {
        HCCL_INFO("[DevUbConnection][%s] RmaConnStatus is SUSPENDED, status[%s].", __func__, status.Describe().c_str());
        return true;
    }

    if (status != RmaConnStatus::READY) {
        ThrowAbnormalStatus(std::string(__func__));
    }

    ReleaseResource();
    status = RmaConnStatus::SUSPENDED;
    return true;
}

static void PrepareUbSendWrReqParamForWriteOrRead(
    HrtRaUbSendWrReqParam& sendWrReq, const HrtUbSendWrOpCode sendWrOpCode, const MemoryBuffer& remoteMemBuf,
    const MemoryBuffer& localMemBuf, JettyHandle remoteJettyHandle, const SqeConfig& config, u32 cqeEnable = 1)
{
    sendWrReq.cqeEn = cqeEnable;
    sendWrReq.opcode = sendWrOpCode;
    sendWrReq.size = localMemBuf.size;
    sendWrReq.localAddr = localMemBuf.addr;
    sendWrReq.remoteAddr = remoteMemBuf.addr;

    sendWrReq.lmemHandle = localMemBuf.memHandle;
    sendWrReq.rmemHandle = remoteMemBuf.memHandle;
    sendWrReq.handle = remoteJettyHandle;

    // 打印入参
    HCCL_INFO(
        "PrepareOneUbSendForRead params opCode=[%u], size=[%u], localAddr=[0x%llx], "
        "remoteAddr=[0x%llx], lmemHandle=[0x%llx], rmemHandle=[0x%llx], "
        "jettyHandle=[0x%llx], cqeEn=[%u], config=[%d]",
        static_cast<u32>(sendWrReq.opcode), sendWrReq.size, localMemBuf.addr, remoteMemBuf.addr, localMemBuf.memHandle,
        remoteMemBuf.memHandle, remoteJettyHandle, sendWrReq.cqeEn, config);
}

static void PrepareUbSendWrReqParamReduceInfo(HrtRaUbSendWrReqParam& sendWrReq, DataType dataType, ReduceOp reduceOp)
{
    sendWrReq.inlineReduceFlag = true;
    sendWrReq.dataType = dataType;
    sendWrReq.reduceOp = reduceOp;
    HCCL_INFO(
        "PrepareUbSendWrReqParamReduceInfo params inlineReduceFlag[%u], dataType[%s], reduceOp[%s]",
        sendWrReq.inlineReduceFlag, dataType.Describe().c_str(), reduceOp.Describe().c_str());
}

static void
PrepareUbSendWrReqParamNotifyInfo(HrtRaUbSendWrReqParam& sendWrReq, u64 data, const MemoryBuffer& remoteNotifyMemBuf)
{
    sendWrReq.opcode = HrtUbSendWrOpCode::WRITE_WITH_NOTIFY;
    sendWrReq.notifyData = data;
    sendWrReq.notifyAddr = remoteNotifyMemBuf.addr;
    sendWrReq.notifyHandle = remoteNotifyMemBuf.memHandle;
    HCCL_INFO(
        "PrepareUbSendWrReqParamNotifyInfo params opCode[%u], "
        "notifyData[0x%llx], notifyAddr[0x%llx], notifyHandle[0x%llx]",
        static_cast<u32>(sendWrReq.opcode), sendWrReq.notifyData, sendWrReq.notifyAddr, sendWrReq.notifyHandle);
}

std::unique_ptr<BaseTask>
DevUbConnection::ConstructTaskUbSend(const HrtRaUbSendWrRespParam& sendWrResp, const SqeConfig& config)
{
    unique_ptr<BaseTask> result;
    if (opMode == OpMode::OPBASE) {
        if (config.wqeMode == WqeMode::DWQE) {
            result = make_unique<TaskUbDirectSend>(
                sendWrResp.funcId, sendWrResp.dieId, sendWrResp.jettyId, sendWrResp.dwqeSize, sendWrResp.dwqe);
        } else if (config.wqeMode == WqeMode::DB_SEND) {
            result
                = make_unique<TaskUbDbSend>(sendWrResp.jettyId, sendWrResp.funcId, sendWrResp.piVal, sendWrResp.dieId);
        } else if (config.wqeMode == WqeMode::WRITE_VALUE) {
            HCCL_INFO("[DevUbConnection::%s] dbAddr=[%llx], piVal=[%u]", __func__, dbAddr, sendWrResp.piVal);
            result = make_unique<TaskWriteValue>(dbAddr, sendWrResp.piVal);
        } else {
            auto msg = StringFormat("Invalid WqeMode[%s]", config.wqeMode.Describe().c_str());
            THROW<InvalidParamsException>(msg);
        }
    } else if (opMode == OpMode::OFFLOAD) {
        CHK_PRT_THROW(
            sendWrResp.piVal < piVal,
            HCCL_ERROR(
                "[DevUbConnection::%s] sendWrResp.piVal[%u] is less than piVal[%u]", __func__, sendWrResp.piVal, piVal),
            InvalidParamsException, "sendWrResp.piVal or piVal is invalid");
        u32 sendPiVal = sendWrResp.piVal - piVal;
        result = make_unique<TaskUbDbSend>(sendWrResp.jettyId, sendWrResp.funcId, sendPiVal, sendWrResp.dieId);
        HCCL_INFO(
            "[DevUbConnection::%s] sendPiVal[%u] piVal[%u] sendWrResp.piVal[%u]", __func__, sendPiVal, piVal,
            sendWrResp.piVal);
    } else {
        auto msg = StringFormat("Invalid OpMode[%s]", opMode.Describe().c_str());
        THROW<InvalidParamsException>(msg);
    }

    piVal = sendWrResp.piVal;
    return result;
}

void DevUbConnection::ProcessSlices(
    const MemoryBuffer& loc, const MemoryBuffer& rmt,
    std::function<void(const MemoryBuffer&, const MemoryBuffer&, u32)> processOneSlice, DataType dataType) const
{
    HCCL_INFO("[DevUbConnection::%s] start", __func__);

    // reduce操作需要保证切片大小是数据类型大小的整数倍
    u32 sliceSize = UB_MAX_TRANS_SIZE;
    if (dataType != DataType::INVALID) {
        u32 dataTypeSize = DATA_TYPE_SIZE_MAP.at(dataType);
        sliceSize = UB_MAX_TRANS_SIZE / dataTypeSize * dataTypeSize;
    }

    u64 locBufSize = loc.size;
    u64 sliceNum = locBufSize / sliceSize;
    u64 lastSliceSize = locBufSize % sliceSize;
    u64 totalSize = sliceNum * sliceSize;
    if (loc.addr > UINT64_MAX - totalSize || rmt.addr > UINT64_MAX - totalSize) {
        THROW<InternalException>("integer overflow occurs");
    }
    for (u64 sliceIdx = 0; sliceIdx < sliceNum; sliceIdx++) {
        MemoryBuffer locSlice(loc.addr + sliceIdx * sliceSize, sliceSize, loc.memHandle);
        MemoryBuffer rmtSlice(rmt.addr + sliceIdx * sliceSize, sliceSize, rmt.memHandle);
        // 当前是最后一片，且没有lastSlice时，启用cqe
        u32 cqeEnable = (sliceIdx == sliceNum - 1 && lastSliceSize == 0) ? 1 : 0;
        processOneSlice(locSlice, rmtSlice, cqeEnable);
    }

    if (lastSliceSize > 0) {
        MemoryBuffer lastLocSlice(loc.addr + sliceNum * sliceSize, lastSliceSize, loc.memHandle);
        MemoryBuffer lastRmtSlice(rmt.addr + sliceNum * sliceSize, lastSliceSize, rmt.memHandle);
        processOneSlice(lastLocSlice, lastRmtSlice, 1);
        sliceNum++;
    }

    HCCL_INFO(
        "[DevUbConnection::%s] end, locBufSize[%llu], sliceNum[%llu], sliceSize[%u], lastSliceSize[%llu]", __func__,
        locBufSize, sliceNum, sliceSize, lastSliceSize);
}

void DevUbConnection::ProcessSlicesWithNotify(
    const MemoryBuffer& loc, const MemoryBuffer& rmt,
    std::function<void(const MemoryBuffer&, const MemoryBuffer&, u32)> processOneSlice,
    std::function<void(const MemoryBuffer&, const MemoryBuffer&)> processOneSliceWithNotify, DataType dataType) const
{
    HCCL_INFO("[DevUbConnection::%s] start", __func__);

    // reduce操作需要保证切片大小是数据类型大小的整数倍
    u32 sliceSize = UB_MAX_TRANS_SIZE;
    if (dataType != DataType::INVALID) {
        u32 dataTypeSize = DATA_TYPE_SIZE_MAP.at(dataType);
        sliceSize = UB_MAX_TRANS_SIZE / dataTypeSize * dataTypeSize;
    }

    u64 locBufSize = loc.size;
    u64 sliceNum = locBufSize / sliceSize;
    u64 lastSliceSize = locBufSize % sliceSize;
    if (sliceNum > 0 && lastSliceSize == 0) {
        sliceNum--;
        lastSliceSize = sliceSize;
    }
    u64 totalSize = sliceNum * sliceSize;
    if (loc.addr > UINT64_MAX - totalSize || rmt.addr > UINT64_MAX - totalSize) {
        THROW<InternalException>("integer overflow occurs");
    }

    for (u64 sliceIdx = 0; sliceIdx < sliceNum; sliceIdx++) {
        MemoryBuffer locSlice(loc.addr + sliceIdx * sliceSize, sliceSize, loc.memHandle);
        MemoryBuffer rmtSlice(rmt.addr + sliceIdx * sliceSize, sliceSize, rmt.memHandle);
        // 固定会有lastSlice，则前面的cqe都不启用
        processOneSlice(locSlice, rmtSlice, 0);
    }

    if (lastSliceSize > 0) {
        MemoryBuffer lastLocSlice(loc.addr + sliceNum * sliceSize, lastSliceSize, loc.memHandle);
        MemoryBuffer lastRmtSlice(rmt.addr + sliceNum * sliceSize, lastSliceSize, rmt.memHandle);
        processOneSliceWithNotify(lastLocSlice, lastRmtSlice);
        sliceNum++;
    }

    HCCL_INFO(
        "[DevUbConnection::%s] end, locBufSize[%llu], sliceNum[%llu], sliceSize[%u], lastSliceSize[%llu]", __func__,
        locBufSize, sliceNum, sliceSize, lastSliceSize);
}

unique_ptr<BaseTask>
DevUbConnection::PrepareRead(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config)
{
    VerifySizeIsEqual(remoteMemBuf, localMemBuf, "DevUbConnection::PrepareRead");

    if (localMemBuf.size == 0) {
        return nullptr;
    }

    HrtRaUbSendWrRespParam sendWrResp{};
    ProcessSlices(
        localMemBuf, remoteMemBuf, [&](const MemoryBuffer& locSlice, const MemoryBuffer& rmtSlice, u32 cqeEnable) {
            HrtRaUbSendWrReqParam sendWrReq = {};
            PrepareUbSendWrReqParamForWriteOrRead(
                sendWrReq, HrtUbSendWrOpCode::READ, rmtSlice, locSlice, remoteJettyHandle, config, cqeEnable);

            sendWrResp = HrtRaUbPostSend(jettyHandle, sendWrReq);
        });

    return ConstructTaskUbSend(sendWrResp, config);
}

unique_ptr<BaseTask> DevUbConnection::PrepareReadReduce(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType dataType, ReduceOp reduceOp,
    const SqeConfig& config)
{
    VerifySizeIsEqual(remoteMemBuf, localMemBuf, "DevUbConnection::PrepareReadReduce");

    if (localMemBuf.size == 0) {
        return nullptr;
    }

    HrtRaUbSendWrRespParam sendWrResp{};
    ProcessSlices(
        localMemBuf, remoteMemBuf,
        [&](const MemoryBuffer& locSlice, const MemoryBuffer& rmtSlice, u32 cqeEnable) {
            HrtRaUbSendWrReqParam sendWrReq = {};
            PrepareUbSendWrReqParamForWriteOrRead(
                sendWrReq, HrtUbSendWrOpCode::READ, rmtSlice, locSlice, remoteJettyHandle, config, cqeEnable);
            PrepareUbSendWrReqParamReduceInfo(sendWrReq, dataType, reduceOp);

            sendWrResp = HrtRaUbPostSend(jettyHandle, sendWrReq);
        },
        dataType);

    return ConstructTaskUbSend(sendWrResp, config);
}

unique_ptr<BaseTask> DevUbConnection::PrepareWrite(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config)
{
    VerifySizeIsEqual(remoteMemBuf, localMemBuf, "DevUbConnection::PrepareWrite");

    if (localMemBuf.size == 0) {
        return nullptr;
    }

    HrtRaUbSendWrRespParam sendWrResp{};
    ProcessSlices(
        localMemBuf, remoteMemBuf, [&](const MemoryBuffer& locSlice, const MemoryBuffer& rmtSlice, u32 cqeEnable) {
            HrtRaUbSendWrReqParam sendWrReq = {};
            PrepareUbSendWrReqParamForWriteOrRead(
                sendWrReq, HrtUbSendWrOpCode::WRITE, rmtSlice, locSlice, remoteJettyHandle, config, cqeEnable);
            sendWrResp = HrtRaUbPostSend(jettyHandle, sendWrReq);
        });

    return ConstructTaskUbSend(sendWrResp, config);
}

unique_ptr<BaseTask> DevUbConnection::PrepareWriteReduce(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType dataType, ReduceOp reduceOp,
    const SqeConfig& config)
{
    VerifySizeIsEqual(remoteMemBuf, localMemBuf, "DevUbConnection::PrepareWriteReduce");

    if (localMemBuf.size == 0) {
        return nullptr;
    }

    HrtRaUbSendWrRespParam sendWrResp{};
    ProcessSlices(
        localMemBuf, remoteMemBuf,
        [&](const MemoryBuffer& locSlice, const MemoryBuffer& rmtSlice, u32 cqeEnable) {
            HrtRaUbSendWrReqParam sendWrReq = {};
            PrepareUbSendWrReqParamForWriteOrRead(
                sendWrReq, HrtUbSendWrOpCode::WRITE, rmtSlice, locSlice, remoteJettyHandle, config, cqeEnable);
            PrepareUbSendWrReqParamReduceInfo(sendWrReq, dataType, reduceOp);
            sendWrResp = HrtRaUbPostSend(jettyHandle, sendWrReq);
        },
        dataType);

    return ConstructTaskUbSend(sendWrResp, config);
}

unique_ptr<BaseTask>
DevUbConnection::PrepareInlineWrite(const MemoryBuffer& remoteMemBuf, u64 data, const SqeConfig& config)
{
    HrtRaUbSendWrReqParam sendWrReq = {};
    sendWrReq.opcode = HrtUbSendWrOpCode::WRITE;
    sendWrReq.remoteAddr = remoteMemBuf.addr;
    sendWrReq.rmemHandle = remoteMemBuf.memHandle;
    sendWrReq.handle = remoteJettyHandle;
    sendWrReq.inlineFlag = true;
    sendWrReq.inlineData = reinterpret_cast<u8*>(&data);
    sendWrReq.size = sizeof(data);
    /*
     * 当前只有前后同步使用writeValue任务
     * 由于writeValue任务不使能cqe，
     * writeValue和dwqe混用会有潜在问题，所以后面需要区分开这两种任务模式
     * 不在同一个connection里面既使用writeValue又使用dwqe
     */
    if (config.wqeMode == WqeMode::WRITE_VALUE && opMode == OpMode::OPBASE) {
        // 当前只有inlineWrite使用write value
        // 图模式不能使用writeValue
        // writeValue 不需要使能cqe
        sendWrReq.cqeEn = false;
    }

    HCCL_INFO(
        "DevUbConnection::PrepareInlineWrite params opCode=[%u], "
        "remoteAddr=[0x%llx], rmemHandle=[0x%llx], remoteJettyHandle=[0x%llx], inlineFlag[%u], size=[%u], data=[%u]",
        sendWrReq.opcode, sendWrReq.remoteAddr, sendWrReq.rmemHandle, sendWrReq.handle, sendWrReq.inlineFlag,
        sendWrReq.size, static_cast<u32>(*sendWrReq.inlineData));
    auto res = HrtRaUbPostSend(jettyHandle, sendWrReq);

    return ConstructTaskUbSend(res, config);
}

inline HrtRaUbSendWrReqParam ConstructUbSendWrReqParamForWriteWithNotify(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, u64 data, const MemoryBuffer& remoteNotifyMemBuf)
{
    HrtRaUbSendWrReqParam sendWrReq = {};
    sendWrReq.opcode = HrtUbSendWrOpCode::WRITE_WITH_NOTIFY;
    sendWrReq.size = remoteMemBuf.size;
    sendWrReq.localAddr = localMemBuf.addr;
    sendWrReq.remoteAddr = remoteMemBuf.addr;
    sendWrReq.lmemHandle = localMemBuf.memHandle;
    sendWrReq.rmemHandle = remoteMemBuf.memHandle;
    sendWrReq.notifyData = data;
    sendWrReq.notifyAddr = remoteNotifyMemBuf.addr;
    sendWrReq.notifyHandle = remoteNotifyMemBuf.memHandle;

    return sendWrReq;
}

unique_ptr<BaseTask> DevUbConnection::PrepareWriteWithNotify(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, u64 data, const MemoryBuffer& remoteNotifyMemBuf,
    const SqeConfig& config)
{
    VerifySizeIsEqual(remoteMemBuf, localMemBuf, "DevUbConnection::PrepareWriteWithNotify");

    if (localMemBuf.size == 0) {
        return nullptr;
    }

    HrtRaUbSendWrRespParam sendWrResp{};
    ProcessSlicesWithNotify(
        localMemBuf, remoteMemBuf,
        [&](const MemoryBuffer& locSlice, const MemoryBuffer& rmtSlice, u32 cqeEnable) {
            HrtRaUbSendWrReqParam sendWrReq = {};
            PrepareUbSendWrReqParamForWriteOrRead(
                sendWrReq, HrtUbSendWrOpCode::WRITE, rmtSlice, locSlice, remoteJettyHandle, config, cqeEnable);
            sendWrResp = HrtRaUbPostSend(jettyHandle, sendWrReq);
        },
        [&](const MemoryBuffer& locSlice, const MemoryBuffer& rmtSlice) {
            HrtRaUbSendWrReqParam sendWrReq = {};
            PrepareUbSendWrReqParamForWriteOrRead(
                sendWrReq, HrtUbSendWrOpCode::WRITE, rmtSlice, locSlice, remoteJettyHandle, config);
            PrepareUbSendWrReqParamNotifyInfo(sendWrReq, data, remoteNotifyMemBuf);

            sendWrResp = HrtRaUbPostSend(jettyHandle, sendWrReq);
        });

    return ConstructTaskUbSend(sendWrResp, config);
}

unique_ptr<BaseTask> DevUbConnection::PrepareWriteReduceWithNotify(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType dataType, ReduceOp reduceOp, u64 data,
    const MemoryBuffer& remoteNotifyMemBuf, const SqeConfig& config)
{
    VerifySizeIsEqual(remoteMemBuf, localMemBuf, "DevUbConnection::PrepareWriteReduceWithNotify");

    if (localMemBuf.size == 0) {
        return nullptr;
    }

    HrtRaUbSendWrRespParam sendWrResp{};
    ProcessSlicesWithNotify(
        localMemBuf, remoteMemBuf,
        [&](const MemoryBuffer& locSlice, const MemoryBuffer& rmtSlice, u32 cqeEnable) {
            HrtRaUbSendWrReqParam sendWrReq = {};
            PrepareUbSendWrReqParamForWriteOrRead(
                sendWrReq, HrtUbSendWrOpCode::WRITE, rmtSlice, locSlice, remoteJettyHandle, config, cqeEnable);
            PrepareUbSendWrReqParamReduceInfo(sendWrReq, dataType, reduceOp);
            sendWrResp = HrtRaUbPostSend(jettyHandle, sendWrReq);
        },
        [&](const MemoryBuffer& locSlice, const MemoryBuffer& rmtSlice) {
            HrtRaUbSendWrReqParam sendWrReq = {};
            PrepareUbSendWrReqParamForWriteOrRead(
                sendWrReq, HrtUbSendWrOpCode::WRITE, rmtSlice, locSlice, remoteJettyHandle, config);
            PrepareUbSendWrReqParamReduceInfo(sendWrReq, dataType, reduceOp);
            PrepareUbSendWrReqParamNotifyInfo(sendWrReq, data, remoteNotifyMemBuf);
            sendWrResp = HrtRaUbPostSend(jettyHandle, sendWrReq);
        },
        dataType);

    return ConstructTaskUbSend(sendWrResp, config);
}

string DevUbConnection::Describe() const
{
    return StringFormat(
        "DevUbConnection[locAddr=%s, rmtAddr=%s, status=%s, dieId=%u, funcId=%u, jettyId=%u, sqBuffVa=%llx, "
        "sqDepth=%u, maxReadSize=%u, maxWriteSize=%u, tpn=%u, dbAddr=0x%llx]",
        locAddr.Describe().c_str(), rmtAddr.Describe().c_str(), status.Describe().c_str(), dieId, funcId, jettyId,
        sqBuffVa, sqDepth, maxReadSize, maxWriteSize, tpn, dbAddr);
}

HcclResult DevUbConnection::Describe(std::string& dfxMsg)
{
    uint16_t udpSport = 0xFFFF; // 无法获取实际的udpSport，使用0xFFFF表示未知
    if (tpProtocol == TpProtocol::TP) {
        struct TpAttr tpAttr {};
        uint32_t attrBitmap = 1 << 13; // 13对应dataUdpSrcport
        TRY_CATCH_PRINT_ERROR(
            u32 devicePhyId = HrtGetDevicePhyIdByUserDevId(devLogicId);
            HcclResult ret
            = HrtRaGetTpAttrAsync(devicePhyId, rdmaHandle, tpInfo.tpHandle, attrBitmap, tpAttr, reqHandle);
            if (ret == HCCL_E_NOT_SUPPORT) {
                HCCL_ERROR(
                    "[DevUbConnection::%s] this package does not support RaGetTpAttrAsync for device,"
                    " please change new package, devPhyId[%u]",
                    __func__, devicePhyId);
                return ret;
            } else if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[DevUbConnection::%s] failed, hccl result[%d]", __func__, ret);
                return ret;
            });
        udpSport = tpAttr.dataUdpSrcport;
    }
    udpSport = udpSport & 0xFF;

    std::string dfxStr = StringFormat(
        "chip id[%u] die id[%u] func id[%u] jetty id[%u] "
        "local %s remote %s udp sport[%u]",
        devLogicId, dieId, funcId, jettyId, locEid.Describe().c_str(), rmtEid.Describe().c_str(), udpSport);
    dfxMsg += dfxStr;
    HCCL_INFO("[DevUbConnection::%s] %s", __func__, dfxStr.c_str());
    return HCCL_SUCCESS;
}

void DevUbConnection::AddNop(const Stream& stream)
{
    if (opMode != OpMode::OFFLOAD) {
        HCCL_WARNING("[DevUbConnection][AddNop]Invalid OpMode[%s]", opMode.Describe().c_str());
        return;
    }
    if (sqDepth < piVal) {
        auto msg = StringFormat("Invalid piVal[%u], piVal should be less than or equal to sqDepth[%u]", piVal, sqDepth);
        THROW<InvalidParamsException>(msg);
    }
    if (sqDepth == piVal) {
        return;
    }
    u32 numNop = sqDepth - piVal;
    HrtRaUbPostNops(jettyHandle, remoteJettyHandle, numNop);

    HrtUbDbInfo info;
    info.dbNum = 1;
    info.wrCqe = 0; // 默认值是0 不会cqe  如果传1，驱动分发，会给hccl cqe，用于维护ci指针。
    info.info[0].functionId = funcId;
    info.info[0].dieId = dieId;
    info.info[0].jettyId = jettyId;
    info.info[0].piValue = numNop;
    HrtUbDbSend(info, stream.GetPtr());

    piVal = sqDepth;
}

HrtUbJfcMode DevUbConnection::GetUbJfcMode() const { return jfcMode; }

JettyHandle& DevUbConnection::GetJettyHandle() { return jettyHandle; }

JettyHandle& DevUbConnection::GetRemoteJettyHandle() { return remoteJettyHandle; }

RdmaHandle& DevUbConnection::GetRdmaHandle() { return rdmaHandle; }

u32 DevUbConnection::GetPiVal() const { return piVal; }

u32 DevUbConnection::GetCiVal() const { return ciVal; }

u32 DevUbConnection::GetSqDepth() const { return sqDepth; }

void DevUbConnection::UpdateCiVal(u32 ci) { ciVal = ci; }

std::vector<DevUbConnection*> GetStarsPollUbConns(const std::vector<RmaConnection*>& rmaConns)
{
    std::vector<DevUbConnection*> ubConns;
    for (auto& rmaConn : rmaConns) {
        if (rmaConn->GetRmaConnType() == RmaConnType::UB) {
            if (dynamic_cast<DevUbConnection*>(rmaConn)->GetUbJfcMode() == HrtUbJfcMode::STARS_POLL) {
                ubConns.emplace_back(dynamic_cast<DevUbConnection*>(rmaConn));
            }
        }
    }
    return ubConns;
}

bool IfNeedUpdatingUbCi(const std::vector<DevUbConnection*>& ubConns)
{
    for (auto& ubConn : ubConns) {
        u32 pi = ubConn->GetPiVal();
        u32 ci = ubConn->GetCiVal();
        u32 sqDepth = ubConn->GetSqDepth();
        // 考虑pi翻转场景
        u32 extra = pi >= ci ? 0 : sqDepth;
        constexpr u32 thresholdDivisor = 2;

        if (static_cast<double>(pi + extra - ci)
            >= static_cast<double>(sqDepth) / thresholdDivisor) { // 当pi和ci差距大于sqDepth/2时，更新ci
            return true;
        }
    }
    return false;
}

void DevUbConnection::SetMaxReadSize(u32 value) { maxReadSize = value; }

void DevUbConnection::SetMaxWriteSize(u32 value) { maxWriteSize = value; }

} // namespace Hccl
