/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_CONN_H
#define CCU_CONN_H

#include "env_config.h"
#include "tp_mgr.h"
#include "ccu_jetty_.h"
#include "ccu_dev_mgr_imp.h"
#include "hcomm_adapter_hccp.h"

namespace hcomm {

MAKE_ENUM(
    CcuConnStatus,
    INIT,          // 初始化
    EXCHANGEABLE,  // 可与对端交换
    CONNECTED,     // 建链完成
    CONN_INVALID); // 链接错误

class CcuConnection {
public:
    CcuConnection(
        const CommAddr& locAddr, const CommAddr& rmtAddr, const CcuChannelInfo& channelInfo,
        const std::vector<CcuJetty*>& ccuJettys, uint32_t qos);
    CcuConnection(const CcuConnection& that) = delete;
    CcuConnection& operator=(const CcuConnection& other) = delete;
    ~CcuConnection();

    // 用于建链过程CcuTransport调用
    HcclResult Init();
    CcuConnStatus GetStatus();
    HcclResult Serialize(std::vector<char>& dtoData);
    HcclResult Deserialize(const std::vector<char>& dtoData);
    HcclResult ImportJetty();
    HcclResult Clean();

    uint32_t GetChannelId() const;
    uint32_t GetDieId() const;
    int32_t GetUserDevId() const;
    uint64_t GetRmtCcuBufAddr() const;
    uint32_t GetRmtCcuBufTokenId() const;
    uint32_t GetRmtCcuBufTokenValue() const;
    HcclResult Describe(std::string& dfxMsg);

protected:
    TpProtocol tpProtocol_{TpProtocol::INVALID};

private:
    MAKE_ENUM(
        InnerStatus, INIT, TP_INFO_GETTING, TP_ATTR_GETTING, JETTY_CREATING, EXCHANGEABLE, JETTY_IMPORTING, CONNECTED,
        CONN_INVALID);

    struct ImportJettyCtx {
        // 保存对端的key，传递数据结构时不需要整个copy
        u8 remoteQpKey[HRT_UB_QP_KEY_MAX_LEN]{0};
        HccpUbJettyImportedInParam inParam{};
        HrtRaUbJettyImportedOutParam outParam{};
    };

private:
    HcclResult StatusMachine();
    HcclResult UpdateInitStatus();
    HcclResult UpdateExchangeStatus();

    HcclResult GetLocalCcuRmaBufferInfo();
    HcclResult CreateJetty();
    HcclResult GetTpInfo();
    HcclResult GetTpAttr();
    HcclResult GetTaTimeOut();
    GetTpInfoParam MakeGetTpInfoParam() const;
    void GenerateLocalPsn();
    void ResetRequestCtxs();
    HcclResult StartImportJettyRequest(uint32_t jettyIndex, RequestHandle& reqHandle);
    HcclResult CheckRequestResults();
    HcclResult ConfigChannel();
    HcclResult ReleaseConnRes();
    HcclResult ReturnErrorStatus(const std::string& funcName);
    std::string Describe();

private:
    CcuConnStatus status_{CcuConnStatus::CONN_INVALID};
    InnerStatus innerStatus_{InnerStatus::CONN_INVALID};
    bool isJettyCreated_{false};
    bool isJettyImported_{false};

    CommAddr locAddr_{};
    CommAddr rmtAddr_{};
    CcuChannelInfo channelInfo_{};
    std::vector<CcuJetty*> ccuJettys_;
    uint32_t qos_{EnvConfig::UB_QOS_DEFAULT};

    int32_t userDevId_{0};
    uint32_t devPhyId_{0};
    uint32_t dieId_{0};
    uint32_t funcId_{0};
    CtxHandle ctxHandle_{nullptr};
    uint32_t jettyNum_{0};

    // 通过ccu comp 获取 ccu buffer信息
    uint64_t ccuBufAddr_{0};
    uint32_t ccuBufTokenId_{0};
    uint32_t ccuBufTokenValue_{0};

    std::vector<ImportJettyCtx> importJettyCtxs_; // 记录import jetty相关信息
    JettyImportCfg jettyImportCfg_{};             // import配置信息，因复用TpHandle只需一份

    // 交换后获取对端ccu buffer信息
    uint64_t rmtCcuBufAddr_{0};
    uint32_t rmtCcuBufTokenId_{0};
    uint32_t rmtCcuBufTokenValue_{0};

    // 感知tp获取tp handle，import jetty后urma提供tpn
    TpInfo tpInfo_{};
    TpAttrInfo tpAttrInfo_{};

    // 异步import上下文信息
    std::vector<RequestHandle> reqHandles_;
    std::vector<std::vector<char>> reqDataBuffers_;
    std::vector<void*> remoteJettyHandlePtrs_;

    u8 errTimeout_{8};

    HcclResult CalcTotalTimeout(CtxHandle ctxHandle, TpHandle tpHandle, uint32_t& outTotalTimeoutMs);
};

class CcuRtpConnection : public CcuConnection {
public:
    CcuRtpConnection(
        const CommAddr& locAddr, const CommAddr& rmtAddr, const CcuChannelInfo& channelInfo,
        const std::vector<CcuJetty*>& ccuJettys, uint32_t qos);
};

class CcuCtpConnection : public CcuConnection {
public:
    CcuCtpConnection(
        const CommAddr& locAddr, const CommAddr& rmtAddr, const CcuChannelInfo& channelInfo,
        const std::vector<CcuJetty*>& ccuJettys, uint32_t qos);
};

} // namespace hcomm

#endif // CCU_CONN_H
