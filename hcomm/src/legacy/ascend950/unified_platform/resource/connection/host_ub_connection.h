/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_HOST_UB_CONNECTION_H
#define HCCLV2_HOST_UB_CONNECTION_H

#include "rma_connection.h"
#include "op_mode.h"
#include "orion_adapter_hccp.h"
#include "tp_manager.h"
#include "local_ub_rma_buffer.h"
#include "stream.h"
#include "task.h"
#include "mc2_type.h"
#include "env_config/env_config_v2.h"

namespace Hccl {

class HostUbConnection : public RmaConnection {
public:
    HostUbConnection(
        const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
        const HrtUbJfcMode jfcMode = HrtUbJfcMode::NORMAL, u8 qos = static_cast<u8>(UB_QOS_DEFAULT));
    void Connect() override;
    RmaConnStatus GetStatus() override;
    bool Suspend() override;

    std::unique_ptr<Serializable> GetExchangeDto() override;
    void ParseRmtExchangeDto(const Serializable& rmtDto) override;
    void ImportRmtDto() override;

    std::vector<char> GetUniqueId() const override;

    void SetCqInfo(HcclAiRMACQ& cq);

    void SetWqInfo(HcclAiRMAWQ& wq);

    unique_ptr<BaseTask>
    PrepareRead(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override;

    unique_ptr<BaseTask> PrepareReadReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType dataType, ReduceOp reduceOp,
        const SqeConfig& config) override;

    unique_ptr<BaseTask>
    PrepareWrite(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override;

    unique_ptr<BaseTask> PrepareWriteReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType dataType, ReduceOp reduceOp,
        const SqeConfig& config) override;

    unique_ptr<BaseTask>
    PrepareInlineWrite(const MemoryBuffer& remoteMemBuf, u64 data, const SqeConfig& config) override;

    class UbCiUpdater;

    void AddNop(const Stream& stream) override;

    void ReleaseTp();
    ~HostUbConnection() override;

    string Describe() const override;

    HrtUbJfcMode GetUbJfcMode() const;
    JettyHandle& GetJettyHandle();
    JettyHandle& GetRemoteJettyHandle();
    RdmaHandle& GetRdmaHandle();
    u32 GetPiVal() const;
    u32 GetCiVal() const;
    u32 GetSqDepth() const;

    uint64_t GetCqVa() const;
    u64 GetJettyVa() const;
    JettyHandle GetTJettyVa() const;

protected:
    TpProtocol tpProtocol{TpProtocol::INVALID};

private:
    MAKE_ENUM(UbConnStatus, INIT, TP_INFO_GETTING, JETTY_CREATED, JETTY_IMPORTING, READY, CONN_INVALID);

    UbConnStatus ubConnStatus{UbConnStatus::INIT};

    RdmaHandle rdmaHandle{nullptr};
    IpAddress locAddr{};
    IpAddress rmtAddr{};
    OpMode opMode{OpMode::OPBASE};
    HrtUbJfcMode jfcMode{HrtUbJfcMode::NORMAL}; // Peer模式只支持jfcMode为NORMAL类型
    u32 tokenValue{GetUbToken()};
    Eid rmtEid{};
    Eid locEid{};
    Eid rmtReverseEid{}; // 反序Eid，仅用于传递给硬件

    u32 dieId{0};
    u32 funcId{0};
    JfcHandle jfcHandle{0};
    u32 sqDepth{0};
    uint64_t sqBuffVa{0};

    RequestHandle reqHandle{0};
    vector<char_t> reqDataBuffer;
    HrtRaUbJettyCreatedOutParam repJetty_{};
    HrtRaUbJettyImportedOutParam remOutParam_{};

    u8 remoteQpKey[HRT_UB_QP_KEY_MAX_LEN] = {0};
    u32 keySize{0};
    u32 remoteKeySize{0};
    u32 remoteTokenValue{0};
    JettyImportCfg jettyImportCfg{};

    JettyHandle jettyHandle_{0};
    u64 jettyVa_{0};
    void* jettyHandlePtr{nullptr};
    JettyHandle remoteJettyVa_{0};
    TargetJettyHandle remoteJettyHandle_{0};
    u8 localQpKey[HRT_UB_QP_KEY_MAX_LEN]{0};

    u32 jettyId_{0};
    u64 dbAddr{0};
    u32 tpn{0};

    u32 localTpnStart{0};
    u32 localTpNum{0};
    TpInfo tpInfo{};

    u8 qos_{static_cast<u8>(UB_QOS_DEFAULT)}; // 业务 QoS，GetTpInfo / ReleaseTpInfo 缓存键

    u32 piVal{0};
    u32 ciVal{0};

    CqCreateInfo cqInfo_{};

    bool CheckRequestResult();
    void ThrowAbnormalStatus(std::string funcName);

    void GenerateLocalPsn();
    void CreateJetty();
    void SetJettyInfo();
    bool GetTpInfo();
    void ImportJetty();
    void SetImportInfo();
    void ReleaseResource();

    void ProcessSlices(
        const MemoryBuffer& loc, const MemoryBuffer& rmt,
        std::function<void(const MemoryBuffer&, const MemoryBuffer&, u32)> processOneSlice,
        DataType dataType = DataType::INVALID) const;

    std::unique_ptr<BaseTask>
    ConstructTaskUbSend(const HrtRaUbSendWrRespParam& sendWrResp, const SqeConfig& config) const;
    void UpdateCiVal(u32 ci);
};

class HostUbTpConnection : public HostUbConnection {
public:
    HostUbTpConnection(
        const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
        const HrtUbJfcMode jfcMode = HrtUbJfcMode::NORMAL, u8 qos = static_cast<u8>(UB_QOS_DEFAULT));
};

class HostUbCtpConnection : public HostUbConnection {
public:
    HostUbCtpConnection(
        const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
        const HrtUbJfcMode jfcMode = HrtUbJfcMode::NORMAL, u8 qos = static_cast<u8>(UB_QOS_DEFAULT));
};

bool IfNeedUpdatingUbCi(const std::vector<HostUbConnection*>& ubConns);

} // namespace Hccl

#endif // HCCLV2_HOST_UB_CONNECTION_H
