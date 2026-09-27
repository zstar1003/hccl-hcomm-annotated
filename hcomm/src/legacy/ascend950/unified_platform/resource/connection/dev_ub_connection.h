/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_DEV_UB_CONNECTION_H
#define HCCLV2_DEV_UB_CONNECTION_H

#include "rma_connection.h"
#include "op_mode.h"
#include "orion_adapter_hccp.h"
#include "../../../framework/env_config/env_config_v2.h"
#include "tp_manager.h"
#include "local_ub_rma_buffer.h"
#include "stream.h"
#include "task.h"
#include "mc2_type.h"
#include "hcomm/hcomm_res_entity_defs.h"
#include <chrono>
#include <functional>

namespace Hccl {

class DevUbConnection : public RmaConnection {
public:
    using AcquireSharedRemoteJettyCallback
        = std::function<HcclResult(const uint8_t*, uint32_t, bool&, TargetJettyHandle&, void*&, uint32_t&)>;
    using PublishSharedRemoteJettyCallback
        = std::function<HcclResult(const uint8_t*, uint32_t, TargetJettyHandle, void*, uint32_t)>;

    /**
     * @brief jetty 生命周期模式，构造时确定，替代旁路方法 + 事后标记。
     *        SELF_CREATE（默认）：原逻辑，构造时建 JFC/jetty，析构销毁。
     *        EXTERNAL_INJECT：跳过建 JFC/jetty，等外部调 SetSharedJettyFields 填充，析构不销毁。
     */
    enum class JettyMode { SELF_CREATE, EXTERNAL_INJECT };

    DevUbConnection(
        const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
        const bool devUsed = false, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL,
        const IpAddress& locIpv4Addr = IpAddress(), const IpAddress& rmtIpv4Addr = IpAddress(),
        u8 qos = static_cast<u8>(UB_QOS_DEFAULT), u8 taTimeOut = TpManager::TA_TIMEOUT_NOT_SET,
        CommEngine engine = COMM_ENGINE_RESERVED, u32 inSqDepth = UB_SQ_DEPTH_NOT_SET,
        JettyMode jettyMode = JettyMode::SELF_CREATE);
    void Connect() override;
    RmaConnStatus GetStatus() override;
    bool Suspend() override;

    std::unique_ptr<Serializable> GetExchangeDto() override;
    void ParseRmtExchangeDto(const Serializable& rmtDto) override;
    void ImportRmtDto() override;

    std::vector<char> GetUniqueId() const override;

    void SetCqInfo(HcclAiRMACQ& cq) const;

    void SetWqInfo(HcclAiRMAWQ& wq);

    void SetCqContextInfo(CqContext& cq) const;
    void SetSqContextInfo(SqContext& sq);

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

    unique_ptr<BaseTask> PrepareWriteWithNotify(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, u64 data,
        const MemoryBuffer& remoteNotifyMemBuf, const SqeConfig& config) override;

    unique_ptr<BaseTask> PrepareWriteReduceWithNotify(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType dataType, ReduceOp reduceOp,
        u64 data, const MemoryBuffer& remoteNotifyMemBuf, const SqeConfig& config) override;

    class UbCiUpdater;

    void AddNop(const Stream& stream) override;

    /**
     * @brief 填充共享 jetty 字段（EXTERNAL_INJECT 模式专用）。
     *        构造时 JettyMode::EXTERNAL_INJECT 跳过建 JFC/jetty，预留空位由本方法填充。
     *        状态机据此跳过 JETTY_CREATING 直接进入 JETTY_CREATED。
     * @note 架构说明：本组方法属 base_comm 共享 jetty 特性（IS_SHARED_QUEUE）的实现细节，
     *       因 DevUbConnection 当前仍位于 legacy/ 而暂置于此。base_comm 侧通过
     *       shared_jetty_connection_adapter 适配层调用，不直接依赖本类。
     *       迁移跟踪：DevUbConnection 迁入 base_comm 后本组方法随之脱离 legacy，
     *       shared_jetty_channel_helper.h 对 legacy 的 include 一并清除。
     *       在迁移完成前，本目录仅作过渡技术债承载，禁止继续扩展共享 jetty 新特性。
     */
    HcclResult SetSharedJettyFields(
        JettyHandle jettyHdl, void* jettyHdlPtr, uint32_t jId, uint64_t sqVa, uint64_t db, const uint8_t* qpKey,
        uint32_t kSize, uint32_t sDepth, JfcHandle sharedJfc, CqCreateInfo sharedCqInfo, uint32_t sharedLocalPsn,
        void* epTag, std::function<void(void*)> releaseCb, AcquireSharedRemoteJettyCallback acquireRemoteCb,
        PublishSharedRemoteJettyCallback publishRemoteCb);

    /**
     * @brief 分离 jetty 所有权（SELF_CREATE 模式建好 jetty 后调用）：
     *        之后析构不销毁 jetty/JFC，生命周期交由 Endpoint::JettyContext 管理。
     *        仅当 connection 已完成 SetJettyInfo 后调用。
     */
    void DetachJetty();

    /**
     * @brief jetty 衍生字段集合，供共享模式下提取后填充给主 connection
     */
    struct JettyInfo {
        JettyHandle handle{0};
        void* handlePtr{nullptr};
        uint32_t jettyId{0};
        uint64_t sqBuffVa{0};
        uint64_t dbAddr{0};
        uint8_t localQpKey[HRT_UB_QP_KEY_MAX_LEN]{0};
        uint32_t keySize{0};
        uint32_t sqDepth{0};
        RdmaHandle rdmaHandle{nullptr}; // 销毁 JFC 所需的 RDMA 句柄
        JfcHandle jfcHandle{0};         // 临时 connection 创建的 JFC, 由 Endpoint 统一销毁
        CqCreateInfo cqInfo{};          // 临时 connection 创建的 CQ 信息, 注入给主 connection 共享
        uint32_t localPsn{0}; // 临时 connection 生成的 psn, 注入给主 connection 复用, 避免多 connection 共享同一本地
                              // jetty/SQ 时各自 GenerateLocalPsn 导致 import 同一 TP 对时 psn 互相覆盖
    };

    /** 获取当前 connection 的 jetty 衍生字段（共享模式下用于 Adopt 到 Holder） */
    HcclResult GetJettyInfo(JettyInfo& info) const;

    void ReleaseTp();
    ~DevUbConnection() override;

    string Describe() const override;
    HcclResult Describe(std::string& dfxMsg) override;

    HrtUbJfcMode GetUbJfcMode() const;
    JettyHandle& GetJettyHandle();
    JettyHandle& GetRemoteJettyHandle();
    RdmaHandle& GetRdmaHandle();
    u32 GetPiVal() const;
    u32 GetCiVal() const;
    u32 GetSqDepth() const;

    void SetMaxReadSize(u32 value);
    void SetMaxWriteSize(u32 value);

protected:
    TpProtocol tpProtocol{TpProtocol::INVALID};
    void GetTimeOut();
    u8 jettyTimeOut{8};

private:
    MAKE_ENUM(
        UbConnStatus, INIT, TP_INFO_GETTING, JETTY_CREATING, JETTY_CREATED, JETTY_IMPORTING, JETTY_IMPORT_WAITING,
        READY, CONN_INVALID);

    UbConnStatus ubConnStatus{UbConnStatus::INIT};

    RdmaHandle rdmaHandle{nullptr};
    IpAddress locAddr{};
    IpAddress rmtAddr{};
    OpMode opMode{OpMode::OPBASE};
    HrtUbJfcMode jfcMode{HrtUbJfcMode::STARS_POLL};
    CommEngine engine_{COMM_ENGINE_RESERVED};
    IpAddress locIpv4Addr{};
    IpAddress rmtIpv4Addr{};
    u32 tokenValue{GetUbToken()};
    Eid rmtEid{};
    Eid locEid{};
    Eid rmtReverseEid{};                      // 反序Eid，仅用于传递给硬件
    u8 qos_{static_cast<u8>(UB_QOS_DEFAULT)}; // 业务 QoS，GetTpInfo / ReleaseTpInfo 缓存键

    bool devUsed_{false};

    // 由调用方根据协议从环境变量获取并传入；TA_TIMEOUT_NOT_SET 表示未传入
    u8 taTimeOut_{TpManager::TA_TIMEOUT_NOT_SET};

    int32_t devLogicId{0};
    u32 dieId{0};
    u32 funcId{0};
    JfcHandle jfcHandle{0};
    u32 sqDepth{0};
    uint64_t sqBuffVa{0};

    RequestHandle reqHandle{0};
    vector<char_t> reqDataBuffer;

    u8 remoteQpKey[HRT_UB_QP_KEY_MAX_LEN] = {0};
    u32 keySize{0};
    u32 remoteKeySize{0};
    u32 remoteTokenValue{0};
    JettyImportCfg jettyImportCfg{};
    void* remoteJettyHandlePtr{nullptr};

    JettyHandle jettyHandle{0};
    void* jettyHandlePtr{nullptr};
    JettyHandle remoteJettyHandle{0};
    u8 localQpKey[HRT_UB_QP_KEY_MAX_LEN]{0};

    u32 jettyId{0};
    u64 dbAddr{0};
    u32 tpn{0};

    u32 localTpnStart{0};
    u32 localTpNum{0};
    TpInfo tpInfo{};

    u32 piVal{0};
    u32 ciVal{0};

    CqCreateInfo cqInfo_{};

    // 最大传输size，切片使用
    u32 maxReadSize{0};
    u32 maxWriteSize{0};

    // jetty 生命周期模式：SELF_CREATE 自建自销毁；EXTERNAL_INJECT 外部填充不自销毁
    JettyMode jettyMode_{JettyMode::SELF_CREATE};
    bool jettyDetached_{false};  // SELF_CREATE 模式建好 jetty 后调 DetachJetty 置 true，析构不销毁
    void* endpointTag_{nullptr}; // 共享模式下透传给 releaseCb_ 的 Endpoint 标签
    std::function<void(void*)> releaseCb_{nullptr}; // 共享 jetty 释放回调（调 Endpoint::ReleaseSharedJetty）
    AcquireSharedRemoteJettyCallback acquireRemoteCb_{nullptr};
    PublishSharedRemoteJettyCallback publishRemoteCb_{nullptr};
    bool releaseTpOnDestroy_{true};

    // JETTY_IMPORT_WAITING 状态的超时与退避：避免对端异常未 PublishSharedRemoteJetty 时无限轮询。
    // importWaitingStart_ 记录进入 WAITING 的起始时刻；importWaitingPollCount_ 累计轮询次数用于退避。
    std::chrono::steady_clock::time_point importWaitingStart_{};
    uint32_t importWaitingPollCount_{0};

    bool CheckRequestResult();
    void ThrowAbnormalStatus(std::string funcName);
    void AdvanceUbConnFromJettyImporting();
    void AdvanceUbConnFromJettyImportWaiting();

    void ProcessInit();
    void ProcessCreateJetty();
    void GenerateLocalPsn();
    void CreateJetty(const bool devUsed);
    void CreateAivUrmaJfc();
    void SetJettyInfo();
    bool GetTpInfo();
    void UpdateLocTpInfo();
    TpInfo SelectTpInfo();
    void ImportJetty();
    void SetImportInfo();
    void AcquireOrWaitSharedRemoteJetty();
    void SetSharedRemoteJettyInfo(TargetJettyHandle handle, void* handlePtr, uint32_t remoteTpn);
    void UnImportJetty();
    void DestroyJetty();
    void ReleaseResource();
    void ReleaseRemoteJettyIfImported(bool ctxValid);
    void ReleaseSharedJettyModeResources(bool ctxValid);
    void ReleaseOwnedJettyAndJfc(bool ctxValid);

    void ProcessSlices(
        const MemoryBuffer& loc, const MemoryBuffer& rmt,
        std::function<void(const MemoryBuffer&, const MemoryBuffer&, u32)> processOneSlice,
        DataType dataType = DataType::INVALID) const;

    void ProcessSlicesWithNotify(
        const MemoryBuffer& loc, const MemoryBuffer& rmt,
        std::function<void(const MemoryBuffer&, const MemoryBuffer&, u32)> processOneSlice,
        std::function<void(const MemoryBuffer&, const MemoryBuffer&)> processOneSliceWithNotify,
        DataType dataType = DataType::INVALID) const;

    std::unique_ptr<BaseTask> ConstructTaskUbSend(const HrtRaUbSendWrRespParam& sendWrResp, const SqeConfig& config);
    void UpdateCiVal(u32 ci);
    HcclResult CalcTotalTimeout(uint32_t& outTotalTimeoutMs);
};

class DevUbTpConnection : public DevUbConnection {
public:
    DevUbTpConnection(
        const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
        const bool devUsed = false, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL,
        const IpAddress& locIpv4Addr = IpAddress(), const IpAddress& rmtIpv4Addr = IpAddress(),
        u8 qos = static_cast<u8>(UB_QOS_DEFAULT), u8 taTimeOut = TpManager::TA_TIMEOUT_NOT_SET,
        CommEngine engine = COMM_ENGINE_RESERVED, u32 sqDepth = UB_SQ_DEPTH_NOT_SET,
        JettyMode jettyMode = JettyMode::SELF_CREATE);
};

class DevUbCtpConnection : public DevUbConnection {
public:
    DevUbCtpConnection(
        const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
        const bool devUsed = false, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL,
        const IpAddress& locIpv4Addr = IpAddress(), const IpAddress& rmtIpv4Addr = IpAddress(),
        u8 qos = static_cast<u8>(UB_QOS_DEFAULT), u8 taTimeOut = TpManager::TA_TIMEOUT_NOT_SET,
        CommEngine engine = COMM_ENGINE_RESERVED, u32 sqDepth = UB_SQ_DEPTH_NOT_SET,
        JettyMode jettyMode = JettyMode::SELF_CREATE);
};

class DevUbUboeConnection : public DevUbConnection {
public:
    DevUbUboeConnection(
        const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
        const bool devUsed = false, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL,
        const IpAddress& locIpv4Addr = IpAddress(), const IpAddress& rmtIpv4Addr = IpAddress(),
        u8 qos = static_cast<u8>(UB_QOS_DEFAULT), u8 taTimeOut = TpManager::TA_TIMEOUT_NOT_SET,
        CommEngine engine = COMM_ENGINE_RESERVED, u32 sqDepth = UB_SQ_DEPTH_NOT_SET,
        JettyMode jettyMode = JettyMode::SELF_CREATE);
};

class DevUbRtpConnection : public DevUbConnection {
public:
    DevUbRtpConnection(
        const RdmaHandle rdmaHandle, const IpAddress& locAddr, const IpAddress& rmtAddr, const OpMode opMode,
        const bool devUsed = false, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL,
        const IpAddress& locAddrEid = IpAddress(), const IpAddress& rmtAddrEid = IpAddress(),
        u8 qos = static_cast<u8>(UB_QOS_DEFAULT), u8 taTimeOut = TpManager::TA_TIMEOUT_NOT_SET,
        CommEngine engine = COMM_ENGINE_RESERVED, u32 sqDepth = UB_SQ_DEPTH_NOT_SET,
        JettyMode jettyMode = JettyMode::SELF_CREATE);
};

std::vector<DevUbConnection*> GetStarsPollUbConns(const std::vector<RmaConnection*>& rmaConns);

bool IfNeedUpdatingUbCi(const std::vector<DevUbConnection*>& ubConns);

} // namespace Hccl

#endif // HCCLV2_DEV_UB_CONNECTION_H
