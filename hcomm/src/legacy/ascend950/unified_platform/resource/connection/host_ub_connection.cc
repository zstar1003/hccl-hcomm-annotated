/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_ub_connection.h"

#include <cstdlib>

#include "exception_util.h"
#include "rma_conn_exception.h"
#include "rdma_handle_manager.h"
#include "exchange_ub_conn_dto.h"

namespace Hccl {

constexpr u32 OPBASED_UB_SQ_DEPTH_MAX = 8192;
constexpr u32 UB_SQ_OFFLOAD_DEPTH = 128;
constexpr u32 UB_SQ_WQEBB_SIZE = 64;
constexpr u32 UB_MAX_TRANS_SIZE = 256 * 1024 * 1024; // UB单次最大传输量256*1024*1024 Byte
constexpr u32 WQE_NUM_PER_SQE = 4;                   // URMA约束每个SQE包含4个WQEBB

HostUbConnection::HostUbConnection(
    const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
    const HrtUbJfcMode jfcMode, u8 qos)
    : RmaConnection(nullptr, RmaConnType::UB),
      rdmaHandle(rdmaHandle),
      locAddr(locAddr),
      rmtAddr(rmtAddr),
      opMode(opMode),
      jfcMode(jfcMode),
      rmtEid(rmtAddr.GetEid()),
      locEid(locAddr.GetEid()),
      rmtReverseEid(rmtAddr.GetReverseEid()),
      qos_(qos)
{
    HCCL_INFO("[HostUbConnection::HostUbConnection] rmtEid=%s", rmtEid.Describe().c_str());

    auto dieIdAndFuncId = RdmaHandleManager::GetInstance().GetDieAndFuncId(rdmaHandle); // 获取dieId和FuncId
    dieId = dieIdAndFuncId.first;
    funcId = dieIdAndFuncId.second;

    jfcHandle = RdmaHandleManager::GetInstance().GetJfcHandle(rdmaHandle, cqInfo_, jfcMode);

    sqDepth = OPBASED_UB_SQ_DEPTH_MAX;
    if (opMode == OpMode::OFFLOAD) {
        sqDepth = UB_SQ_OFFLOAD_DEPTH;
    }
    HCCL_INFO(
        "rdmaHandle[%p] locAddr[%s] rmtAddr[%s] opMode[%u] jfcMode[%s] dieId[%u] funcId[%u] jfcHandle[%llu] "
        "sqDepth[%u]",
        rdmaHandle, locAddr.Describe().c_str(), rmtAddr.Describe().c_str(), opMode, jfcMode.Describe().c_str(), dieId,
        funcId, jfcHandle, sqDepth);
    if (sqDepth > (UINT32_MAX / UB_SQ_WQEBB_SIZE / WQE_NUM_PER_SQE)) {
        THROW<InternalException>("integer overflow occurs");
    }
}

HostUbTpConnection::HostUbTpConnection(
    const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
    const HrtUbJfcMode jfcMode, u8 qos)
    : HostUbConnection(rdmaHandle, locAddr, rmtAddr, opMode, jfcMode, qos)
{
    tpProtocol = TpProtocol::TP;
}

HostUbCtpConnection::HostUbCtpConnection(
    const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
    const HrtUbJfcMode jfcMode, u8 qos)
    : HostUbConnection(rdmaHandle, locAddr, rmtAddr, opMode, jfcMode, qos)
{
    tpProtocol = TpProtocol::CTP;
}

std::vector<char> HostUbConnection::GetUniqueId() const
{
    BinaryStream binaryStream;
    binaryStream << dieId;
    binaryStream << funcId;
    binaryStream << jettyId_;

    bool dwqeCacheLocked = false; // 待修改，该jetty是否支持dwqeCachedLocked，默认不支持
    u32 jfcPollMode = 0;          // 待修改，0代表STARS POLL，1代表software Poll
    u64 sqCiAddr = 0; // 待修改，软件poll CQ情况下，需要AICPU从该地址中读取CI,依赖UB驱动支持
    std::vector<char> result;
    binaryStream << jfcPollMode;
    binaryStream << dwqeCacheLocked;
    binaryStream << dbAddr;
    binaryStream << sqCiAddr;
    binaryStream << sqBuffVa;
    binaryStream << sqDepth;
    binaryStream << tpn;
    binaryStream << rmtEid.raw;
    binaryStream << locEid.raw;

    binaryStream.Dump(result);
    HCCL_INFO("HostUbConnection::GetUniqueId:%s", Describe().c_str());
    HCCL_INFO(
        "type=%s, jfcPollMode=%u, dwqeCacheLocked=%d, sqCiAddr=0x%llx", rmaConnType.Describe().c_str(), jfcPollMode,
        dwqeCacheLocked, sqCiAddr);
    return result;
}

void HostUbConnection::SetCqInfo(HcclAiRMACQ& cq)
{
    cq.jfcId = cqInfo_.id;
    cq.cqVA = cqInfo_.va;
    cq.cqeSize = cqInfo_.cqeSize;
    cq.cqDepth = cqInfo_.cqDepth;
    cq.dbAddr = cqInfo_.swdbAddr;
}

void HostUbConnection::SetWqInfo(HcclAiRMAWQ& wq)
{
    wq.jettyId = jettyId_;
    wq.dbAddr = dbAddr;
    wq.sqVA = sqBuffVa;
    wq.sqDepth = sqDepth * WQE_NUM_PER_SQE;
    wq.tp_id = tpn;
    errno_t ret = memcpy_s(wq.rmtEid, sizeof(wq.rmtEid), rmtReverseEid.raw, sizeof(wq.rmtEid));
    if (ret != EOK) {
        HCCL_ERROR("[HostUbConnection][%s] memcpy_s failed, ret=%d", __func__, ret);
        ThrowAbnormalStatus(std::string(__func__));
    }
}

void HostUbConnection::Connect() { GetStatus(); }

inline uint32_t GetRandomNum()
{
    uint32_t randNum = std::rand();
    return randNum;
}

RmaConnStatus HostUbConnection::GetStatus()
{
    switch (ubConnStatus) {
        case UbConnStatus::INIT: {
            HCCL_INFO(
                "[HostUbConnection][%s] start, status[%s], ubConnStatus[%s].", __func__, status.Describe().c_str(),
                ubConnStatus.Describe().c_str());
            if (!GetTpInfo()) {
                ubConnStatus = UbConnStatus::TP_INFO_GETTING;
                break;
            }
            CreateJetty();
            SetJettyInfo();
            ubConnStatus = UbConnStatus::JETTY_CREATED;
            status = RmaConnStatus::EXCHANGEABLE;
            break;
        }
        case UbConnStatus::TP_INFO_GETTING: {
            if (GetTpInfo()) {
                CreateJetty();
                SetJettyInfo();
                ubConnStatus = UbConnStatus::JETTY_CREATED;
                status = RmaConnStatus::EXCHANGEABLE;
            }
            break;
        }
        case UbConnStatus::JETTY_CREATED: {
            HCCL_INFO(
                "[HostUbConnection][%s] status[%s] will not change, "
                "should call ImportRmtDto to change status.",
                __func__, status.Describe().c_str());
            break;
        }
        case UbConnStatus::JETTY_IMPORTING: {
            SetImportInfo();
            ubConnStatus = UbConnStatus::READY;
            status = RmaConnStatus::READY;
            break;
        }
        case UbConnStatus::READY:
            break;
        default:
            ThrowAbnormalStatus(std::string(__func__));
    }

    return status;
}

std::unique_ptr<Serializable> HostUbConnection::GetExchangeDto()
{
    if (status != RmaConnStatus::READY && status != RmaConnStatus::EXCHANGEABLE) {
        HCCL_ERROR("[HostUbConnection][%s] status[%s] is not expected.", __func__, status.Describe().c_str());
        ThrowAbnormalStatus(std::string(__func__));
    }

    if (tpProtocol != TpProtocol::INVALID) {
        jettyImportCfg.localTpHandle = tpInfo.tpHandle;

        HCCL_INFO(
            "[HostUbConnection][%s] tpEnable, localTpHandle[0x%llx] localPsn[%u].", __func__,
            jettyImportCfg.localTpHandle, jettyImportCfg.localPsn);
    }

    std::unique_ptr<ExchangeUbConnDto> dto
        = make_unique<ExchangeUbConnDto>(tokenValue, keySize, jettyImportCfg.localTpHandle, jettyImportCfg.localPsn);
    errno_t ret = memcpy_s(dto->qpKey, HRT_UB_QP_KEY_MAX_LEN, repJetty_.key, HRT_UB_QP_KEY_MAX_LEN);
    if (ret != EOK) {
        HCCL_ERROR("[HostUbConnection][%s] memcpy_s failed, ret=%d", __func__, ret);
        ThrowAbnormalStatus(std::string(__func__));
    }
    return std::unique_ptr<Serializable>(dto.release());
}

void HostUbConnection::ParseRmtExchangeDto(const Serializable& rmtDto)
{
    auto dto = dynamic_cast<const ExchangeUbConnDto&>(rmtDto);
    HCCL_INFO("[HostUbConnection][%s] remoteConnDto[%s]", __func__, dto.Describe().c_str());
    if (dto.qpKeySize == 0 || dto.qpKeySize > HRT_UB_QP_KEY_MAX_LEN) {
        HCCL_ERROR(
            "[HostUbConnection][%s] invalid remote qpKeySize[%u], max[%u].", __func__, dto.qpKeySize,
            HRT_UB_QP_KEY_MAX_LEN);
        ThrowAbnormalStatus(std::string(__func__));
    }
    remoteTokenValue = dto.tokenValue;
    remoteKeySize = dto.qpKeySize;
    errno_t ret = memcpy_s(remoteQpKey, HRT_UB_QP_KEY_MAX_LEN, dto.qpKey, remoteKeySize);
    if (ret != EOK) {
        HCCL_ERROR("[HostUbConnection][%s] memcpy_s failed, ret=%d", __func__, ret);
        ThrowAbnormalStatus(std::string(__func__));
    }

    if (tpProtocol != TpProtocol::INVALID) {
        jettyImportCfg.remoteTpHandle = dto.tpHandle;
        jettyImportCfg.remotePsn = dto.psn;
        HCCL_INFO(
            "[HostUbConnection][%s] tpEnable, remoteTpHandle[0x%llx], remotePsn[%u].", __func__,
            jettyImportCfg.remoteTpHandle, jettyImportCfg.remotePsn);
    }
}

void HostUbConnection::ImportRmtDto()
{
    if (ubConnStatus == UbConnStatus::READY) {
        HCCL_WARNING("[HostUbConnection][%s] import jetty already, %s.", __func__, Describe().c_str());
        return;
    }

    if (ubConnStatus != UbConnStatus::JETTY_CREATED) {
        HCCL_ERROR(
            "[HostUbConnection][%s] failed, ubConnStatus[%s] is not expected.", __func__,
            ubConnStatus.Describe().c_str());
        ThrowAbnormalStatus(std::string(__func__));
    }

    ImportJetty();
    ubConnStatus = UbConnStatus::JETTY_IMPORTING;
}

void HostUbConnection::ThrowAbnormalStatus(std::string funcName)
{
    auto errMsg = StringFormat("[HostUbConnection][%s] failed, [%s].", funcName.c_str(), Describe().c_str());
    status = RmaConnStatus::CONN_INVALID;
    ubConnStatus = UbConnStatus::CONN_INVALID;
    THROW<RmaConnException>(errMsg);
}

bool HostUbConnection::CheckRequestResult()
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
            "[HostUbConnection][%s] failed, result[%s] is unexpected.", __func__, result.Describe().c_str());
    }

    return true;
}

void HostUbConnection::CreateJetty()
{
    if (sqDepth > UINT32_MAX / UB_SQ_WQEBB_SIZE / WQE_NUM_PER_SQE) {
        THROW<InternalException>(
            "[HostUbConnection][%s] failed, sqDepth[%u] times "
            "UB_SQ_WQEBB_SIZE[%u] overflow uint32 max.",
            __func__, sqDepth, UB_SQ_WQEBB_SIZE);
    }
    u32 size = static_cast<u32>(sqDepth) * static_cast<u32>(UB_SQ_WQEBB_SIZE) * static_cast<u32>(WQE_NUM_PER_SQE);
    HrtRaUbCreateJettyParam req{
        jfcHandle,
        jfcHandle,
        GetUbToken(),
        0,
        HrtJettyMode::STANDARD, // peer模式只支持JETTY_MODE_URMA_NORMAL
        0,                      // HOST展开与AICPU展开传入jetty id为0，申请一个新的jetty
        0,                      // va由底层分配，此处填0即可。
        size,
        0,
        sqDepth}; // 非CCUv2不需要填写sqeBufIndex
    if (tpInfo.hasMappedJettyPriority) {
        req.qos = static_cast<u8>(tpInfo.mappedJettyPriority & 0xFU);
    }
    HCCL_INFO(
        "[HostUbConnection][%s] jetty create qos[%u] (maps to attr.ub.priority lower 4 bits).", __func__,
        static_cast<unsigned int>(req.qos));

    repJetty_ = HrtRaUbCreateJetty(rdmaHandle, req);
}

void HostUbConnection::SetJettyInfo()
{
    jettyId_ = repJetty_.id;
    jettyHandle_ = repJetty_.handle;
    jettyVa_ = repJetty_.jettyVa;
    sqBuffVa = repJetty_.sqBuffVa; // hccp提供
    HCCL_INFO("[HostUbConnection][%s] Get sqBuffVa is %llx.", __func__, sqBuffVa);
    keySize = repJetty_.keySize;
    dbAddr = repJetty_.dbVa;
}

bool HostUbConnection::GetTpInfo()
{
    if (tpProtocol == TpProtocol::INVALID) { // 不感知tp建链，当前默认不支持
        HCCL_ERROR(
            "[HostUbConnection][%s] failed, tpProtocol[%s] is not expected.", __func__, tpProtocol.Describe().c_str());
        ThrowAbnormalStatus(std::string(__func__));
    }

    int32_t devLogicId = HrtGetDevice();
    RaUbGetTpInfoParam p{};
    p.locAddr = locAddr;
    p.rmtAddr = rmtAddr;
    p.tpProtocol = tpProtocol;
    p.qos = static_cast<uint32_t>(qos_);
    auto ret = TpManager::GetInstance(devLogicId).GetTpInfo(p, tpInfo, true);

    switch (ret) {
        case HcclResult::HCCL_E_AGAIN:
            return false;
        case HcclResult::HCCL_SUCCESS:
            GenerateLocalPsn();
            return true;
        case HcclResult::HCCL_E_NOT_FOUND:
        default:
            HCCL_ERROR("[HostUbConnection][%s] failed, hccl result[%d]", __func__, ret);
            ThrowAbnormalStatus(std::string(__func__));
            break;
    }
    return true;
}

void HostUbConnection::GenerateLocalPsn() { jettyImportCfg.localPsn = GetRandomNum(); }

void HostUbConnection::ImportJetty()
{
    HrtRaUbJettyImportedInParam in{};
    in.key = remoteQpKey;
    in.keyLen = remoteKeySize;
    in.tokenValue = remoteTokenValue;
    in.jettyImportCfg = jettyImportCfg;
    in.jettyImportCfg.protocol = tpProtocol;

    if (tpProtocol != TpProtocol::CTP && tpProtocol != TpProtocol::TP) {
        HCCL_ERROR(
            "[HostUbConnection][%s] failed, tp protocol[%s] is not expected, %s.", __func__,
            tpProtocol.Describe().c_str(), Describe().c_str());
        ThrowAbnormalStatus(std::string(__func__));
    }

    remOutParam_ = RaUbTpImportJetty(rdmaHandle, in.key, in.keyLen, in.tokenValue, in.jettyImportCfg);
}

void HostUbConnection::SetImportInfo()
{
    remoteJettyVa_ = remOutParam_.targetJettyVa;
    remoteJettyHandle_ = remOutParam_.handle;
    tpn = remOutParam_.tpn;
    return;
}

void HostUbConnection::ReleaseTp()
{
    ReleaseUbConnectionTp(HrtGetDevice(), locAddr, rmtAddr, tpProtocol, tpInfo, static_cast<uint32_t>(qos_));
}

void HostUbConnection::ReleaseResource()
{
    const bool ctxValid = (rdmaHandle != nullptr) && RdmaHandleManager::GetInstance().IsHandleValid(rdmaHandle);

    if (rdmaHandle && remoteJettyHandle_ != 0) {
        if (!ctxValid) {
            HCCL_WARNING(
                "[HostUbConnection][%s] skip HrtRaUbUnimportJetty, "
                "rdmaHandle=%p invalid (DeInit/DestroyAll done), remoteJettyHandle=0x%llx",
                __func__, rdmaHandle, static_cast<unsigned long long>(remoteJettyHandle_));
        } else {
            HrtRaUbUnimportJetty(rdmaHandle, remoteJettyHandle_);
        }
        remoteJettyHandle_ = 0;
    }

    ReleaseTp();

    if (jettyHandle_ != 0) {
        if (!ctxValid) {
            HCCL_WARNING(
                "[HostUbConnection][%s] skip HrtRaUbDestroyJetty, "
                "rdmaHandle=%p invalid, jettyHandle=0x%llx",
                __func__, rdmaHandle, static_cast<unsigned long long>(jettyHandle_));
        } else {
            HrtRaUbDestroyJetty(jettyHandle_);
        }
        jettyHandle_ = 0;
    }
}

HostUbConnection::~HostUbConnection() { DECTOR_TRY_CATCH("HostUbConnection", ReleaseResource()); }

// Suspend接口当前已不使用，由框架调用触发析构流程
bool HostUbConnection::Suspend()
{
    HCCL_WARNING("[HostUbConnection][%s] should not be called.", __func__);
    return true;
}

std::unique_ptr<BaseTask>
HostUbConnection::ConstructTaskUbSend(const HrtRaUbSendWrRespParam& sendWrResp, const SqeConfig& config) const
{
    (void)sendWrResp;
    (void)config;
    unique_ptr<BaseTask> result;
    return result;
}

void HostUbConnection::ProcessSlices(
    const MemoryBuffer& loc, const MemoryBuffer& rmt,
    std::function<void(const MemoryBuffer&, const MemoryBuffer&, u32)> processOneSlice, DataType dataType) const
{
    (void)loc;
    (void)rmt;
    (void)processOneSlice;
    (void)dataType;
}

unique_ptr<BaseTask> HostUbConnection::PrepareRead(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config)
{
    (void)remoteMemBuf;
    (void)localMemBuf;
    (void)config;
    HCCL_INFO("[HostUbConnection::%s] not supported yet.", __func__);
    return nullptr;
}

unique_ptr<BaseTask> HostUbConnection::PrepareReadReduce(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType dataType, ReduceOp reduceOp,
    const SqeConfig& config)
{
    (void)remoteMemBuf;
    (void)localMemBuf;
    (void)dataType;
    (void)reduceOp;
    (void)config;
    HCCL_INFO("[HostUbConnection::%s] not supported yet.", __func__);
    return nullptr;
}

unique_ptr<BaseTask> HostUbConnection::PrepareWrite(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config)
{
    (void)remoteMemBuf;
    (void)localMemBuf;
    (void)config;
    HCCL_INFO("[HostUbConnection::%s] not supported yet.", __func__);
    return nullptr;
}

unique_ptr<BaseTask> HostUbConnection::PrepareWriteReduce(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType dataType, ReduceOp reduceOp,
    const SqeConfig& config)
{
    (void)remoteMemBuf;
    (void)localMemBuf;
    (void)dataType;
    (void)reduceOp;
    (void)config;
    HCCL_INFO("[HostUbConnection::%s] not supported yet.", __func__);
    return nullptr;
}

unique_ptr<BaseTask>
HostUbConnection::PrepareInlineWrite(const MemoryBuffer& remoteMemBuf, u64 data, const SqeConfig& config)
{
    (void)remoteMemBuf;
    (void)data;
    (void)config;
    HCCL_INFO("[HostUbConnection::%s] not supported yet.", __func__);
    return nullptr;
}

string HostUbConnection::Describe() const
{
    return StringFormat(
        "HostUbConnection[locAddr=%s, rmtAddr=%s, status=%s, dieId=%u, funcId=%u, jettyId=%u, sqBuffVa=%llx, "
        "sqDepth=%u, tpn=%u, dbAddr=0x%llx]",
        locAddr.Describe().c_str(), rmtAddr.Describe().c_str(), status.Describe().c_str(), dieId, funcId, jettyId_,
        sqBuffVa, sqDepth, tpn, dbAddr);
}

void HostUbConnection::AddNop(const Stream& stream) { (void)stream; }

HrtUbJfcMode HostUbConnection::GetUbJfcMode() const { return jfcMode; }

JettyHandle& HostUbConnection::GetJettyHandle() { return jettyHandle_; }

JettyHandle& HostUbConnection::GetRemoteJettyHandle() { return remoteJettyHandle_; }

RdmaHandle& HostUbConnection::GetRdmaHandle() { return rdmaHandle; }

u32 HostUbConnection::GetPiVal() const { return piVal; }

u32 HostUbConnection::GetCiVal() const { return ciVal; }

u32 HostUbConnection::GetSqDepth() const { return sqDepth; }

uint64_t HostUbConnection::GetCqVa() const { return cqInfo_.va; }

u64 HostUbConnection::GetJettyVa() const { return jettyVa_; }

JettyHandle HostUbConnection::GetTJettyVa() const { return remoteJettyVa_; }

void HostUbConnection::UpdateCiVal(u32 ci) { ciVal = ci; }

bool IfNeedUpdatingUbCi(const std::vector<HostUbConnection*>& ubConns)
{
    for (auto& ubConn : ubConns) {
        u32 pi = ubConn->GetPiVal();
        u32 ci = ubConn->GetCiVal();
        u32 sqDepth = ubConn->GetSqDepth();
        // 考虑pi翻转场景
        u32 extra = pi >= ci ? 0 : sqDepth;
        constexpr u32 thresholdDivisor = 2;

        if (static_cast<double>(pi + extra - ci) >= static_cast<double>(sqDepth) / thresholdDivisor) {
            // 当pi和ci差距大于sqDepth/2时，更新ci
            return true;
        }
    }
    return false;
}

} // namespace Hccl
