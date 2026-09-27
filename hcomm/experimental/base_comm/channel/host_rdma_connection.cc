/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_rdma_connection.h"
#include "dtype_common.h"
#include "orion_adapter_rts.h"
#include "exchange_rdma_conn_dto.h"
#include "hccp.h"
#include "sal.h"

namespace hcomm_experimental {
constexpr u32 WAIT_US_COUNT = 1000;
constexpr uint32_t TC_TEMP = 132;
constexpr uint32_t SL_TEMP = 4;
constexpr uint32_t RETRY_CNT_TEMP = 7;
constexpr uint32_t RETRY_TIME_TEMP = 20;

namespace {
    HcclResult GetHostQpMode(int& qpMode)
    {
        qpMode = Hccl::OPBASE_QP_MODE;
        HCCL_INFO("[HostRdmaConnection][%s] host-only default qpMode[%d].", __func__, qpMode);
        return HCCL_SUCCESS;
    }
} // namespace

HostRdmaConnection::HostRdmaConnection(Hccl::Socket* socket, RdmaHandle rdmaHandle)
    : socket_(socket),
      rdmaHandle_(rdmaHandle)
{}

HcclResult HostRdmaConnection::Init()
{
    if (rdmaConnStatus_ != RdmaConnStatus::CLOSED) {
        HCCL_INFO(
            "[HostRdmaConnection][%s] status[%s] is not need init.", __func__, rdmaConnStatus_.Describe().c_str());
        return HCCL_SUCCESS;
    }

    int qpMode = 0;
    CHK_RET(GetHostQpMode(qpMode));
    qpInfo_.qpMode = qpMode;
    qpInfo_.rdmaHandle = rdmaHandle_;
    rdmaConnStatus_ = RdmaConnStatus::INIT;
    return HCCL_SUCCESS;
}

HostRdmaConnection::~HostRdmaConnection()
{
    if (rdmaConnStatus_ == RdmaConnStatus::CLOSED || rdmaConnStatus_ == RdmaConnStatus::INIT) {
        return;
    }
    HcclResult ret = DestroyQp();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s]call DestroyQp failed: hcclRet -> %d", __func__, ret);
    }
}

std::string HostRdmaConnection::Describe() const
{
    return Hccl::StringFormat("HostRdmaConnection[status=%s]", rdmaConnStatus_.Describe().c_str());
}

HcclResult HostRdmaConnection::CreateQp()
{
    if (socket_->GetStatus() != Hccl::SocketStatus::OK) {
        HCCL_WARNING("[HostRdmaConnection::CreateQp] socket status is not ok, please check socket connection");
        return HCCL_E_AGAIN;
    }

    // 创建receive & send channel，用于poll cq，避免软件一直轮询cq
    HCCL_INFO("HostRdmaConnection CreateCompChannel");
    s32 ret = RaCreateCompChannel(qpInfo_.rdmaHandle, &sendCompChannel_);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[HostRdmaConnection::CreateQp][CreateSendCompChannel]errNo[0x%016llx] RaCreateCompChannel fail. "
            "return[%d], params: rdmaHandle[%p], sendCompChannel[%p]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, qpInfo_.rdmaHandle, &sendCompChannel_),
        HCCL_E_NETWORK);
    ret = RaCreateCompChannel(qpInfo_.rdmaHandle, &recvCompChannel_);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[HostRdmaConnection::CreateQp][CreateReceiveCompChannel]errNo[0x%016llx] RaCreateCompChannel fail. "
            "return[%d], params: rdmaHandle[%p], rcvCompChannel[%p]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, qpInfo_.rdmaHandle, &recvCompChannel_),
        HCCL_E_NETWORK);

    // 创建CQ和QP
    // qp创建时不指定srq/srq cq/srq context，由qp创建时创建独立的sq和rq，并创建对应的cq
    // cq for sq句柄保存在qpInfo_.sendCq中; cq for rq句柄保存在qpInfo_.receiveCq变量中
    HCCL_INFO("HostRdmaConnection CreateCqAndQp");
    CHK_RET(
        Hccl::HrtRaCreateQpWithCq(qpInfo_.rdmaHandle, -1, -1, sendCompChannel_, recvCompChannel_, qpInfo_, isHdcMode_));

    struct QosAttr qosAttr {};
    qosAttr.tc = qpInfo_.trafficClass;
    qosAttr.sl = qpInfo_.serviceLevel;
    HCCL_INFO("[%s]Set qp qos success by config, TC[%u] SL[%u]", __func__, qosAttr.tc, qosAttr.sl);
    roceAttr_.tc = qpInfo_.trafficClass;
    roceAttr_.sl = qpInfo_.serviceLevel;
    roceAttr_.retryCnt = qpInfo_.retryCnt;
    roceAttr_.retryInterval = qpInfo_.retryInterval;

    if (qpInfo_.lbValue > 0) {
        ret = RaSetQpLbValue(qpInfo_.qpHandle, qpInfo_.lbValue);
        CHK_PRT_RET(
            ret != 0,
            HCCL_ERROR(
                "[HostRdmaConnection::CreateQp][SetQpLbValue]errNo[0x%016llx] RaSetQpLbValue fail. "
                "return[%d], params: qpHandle[%p], lbValue[%u]",
                HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, qpInfo_.qpHandle, qpInfo_.lbValue),
            HCCL_E_NETWORK);
    }
    ret = RaSetQpAttrQos(qpInfo_.qpHandle, &qosAttr);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[HostRdmaConnection::CreateQp][SetQpAttrQos]errNo[0x%016llx] RaSetQpAttrQos fail. "
            "return[%d], params: qpHandle[%p], trafficClass[%u], serviceLevel[%u]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, qpInfo_.qpHandle, qosAttr.tc, qosAttr.sl),
        HCCL_E_NETWORK);
    ret = RaSetQpAttrTimeout(qpInfo_.qpHandle, &(qpInfo_.retryInterval));
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[HostRdmaConnection::CreateQp][SetQpAttrTimeout]errNo[0x%016llx] RaSetQpAttrTimeout fail. "
            "return[%d], params: qpHandle[%p], retryInterval[%u]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, qpInfo_.qpHandle, qpInfo_.retryInterval),
        HCCL_E_NETWORK);
    ret = RaSetQpAttrRetryCnt(qpInfo_.qpHandle, &(qpInfo_.retryCnt));
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[HostRdmaConnection::CreateQp][SetQpAttrRetryCnt]errNo[0x%016llx] RaSetQpAttrRetryCnt fail. "
            "return[%d], params: qpHandle[%p], retryCnt[%u]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, qpInfo_.qpHandle, qpInfo_.retryCnt),
        HCCL_E_NETWORK);
    rdmaConnStatus_ = RdmaConnStatus::QP_CREATED;
    return HCCL_SUCCESS;
}

HcclResult HostRdmaConnection::DestroyQp()
{
    if (rdmaConnStatus_ == RdmaConnStatus::CLOSED || rdmaConnStatus_ == RdmaConnStatus::INIT) {
        return HCCL_SUCCESS;
    }

    CHK_RET(Hccl::HrtRaDestroyQpWithCq(qpInfo_, isHdcMode_));

    s32 ret = RaDestroyCompChannel(qpInfo_.rdmaHandle, sendCompChannel_);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[HostRdmaConnection::DestroyQp]errNo[0x%016llx] ra_destroy_comp_channel fail. "
            "return[%d], params: rdmaHandle[%p], sendCompChannel[%p]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, qpInfo_.rdmaHandle, &sendCompChannel_),
        HCCL_E_NETWORK);
    ret = RaDestroyCompChannel(qpInfo_.rdmaHandle, recvCompChannel_);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[HostRdmaConnection::DestroyQp]errNo[0x%016llx] ra_destroy_comp_channel fail. "
            "return[%d], params: rdmaHandle[%p], rcvCompChannel[%p]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, qpInfo_.rdmaHandle, &recvCompChannel_),
        HCCL_E_NETWORK);

    qpInfo_ = Hccl::QpInfo();
    rdmaConnStatus_ = RdmaConnStatus::CLOSED;
    return HCCL_SUCCESS;
}

HcclResult HostRdmaConnection::GetExchangeDto(std::unique_ptr<Hccl::Serializable>& locQpAttrserial)
{
    if (rdmaConnStatus_ != RdmaConnStatus::QP_CREATED && rdmaConnStatus_ != RdmaConnStatus::QP_MODIFIED) {
        HCCL_ERROR(
            "[HostRdmaConnection][%s] status[%s] is not expected.", __func__, rdmaConnStatus_.Describe().c_str());
        return HCCL_E_AGAIN;
    }

    struct QpAttr localQpAttr;
    s32 ret = RaGetQpAttr(qpInfo_.qpHandle, &localQpAttr);
    if (ret != 0) {
        HCCL_ERROR("[HostRdmaConnection::GetExchangeDto]RaGetQpAttr failed, ret(%d)", ret);
        return HCCL_E_ROCE_CONNECT;
    }
    std::unique_ptr<ExchangeRdmaConnDto> dto = nullptr;
    EXCEPTION_CATCH(
        dto = std::make_unique<ExchangeRdmaConnDto>(localQpAttr.qpn, localQpAttr.psn, localQpAttr.gidIdx),
        return HCCL_E_PTR);
    CHK_SAFETY_FUNC_RET(memcpy_s(dto->gid_, HCCP_GID_RAW_LEN, localQpAttr.gid, HCCP_GID_RAW_LEN));
    locQpAttrserial = std::unique_ptr<Hccl::Serializable>(std::move(dto));
    return HCCL_SUCCESS;
}

HcclResult HostRdmaConnection::ParseRmtExchangeDto(const Hccl::Serializable& rmtQpAttrSerial)
{
    auto dto = dynamic_cast<const ExchangeRdmaConnDto&>(rmtQpAttrSerial);
    HCCL_INFO("[HostRdmaConnection][%s] remoteConnDto[%s]", __func__, dto.Describe().c_str());
    rmtQpAttr_.psn = dto.psn_;
    rmtQpAttr_.qpn = dto.qpn_;
    rmtQpAttr_.gid_idx = dto.gid_idx_;
    CHK_SAFETY_FUNC_RET(memcpy_s(rmtQpAttr_.gid, HCCP_GID_RAW_LEN, dto.gid_, HCCP_GID_RAW_LEN));
    return HCCL_SUCCESS;
}

HcclResult HostRdmaConnection::ModifyQp()
{
    if (rdmaConnStatus_ == RdmaConnStatus::QP_MODIFIED) {
        HCCL_WARNING(
            "[HostRdmaConnection][%s] modify qp already, status[%s].", __func__, rdmaConnStatus_.Describe().c_str());
        return HCCL_SUCCESS;
    }
    if (rdmaConnStatus_ != RdmaConnStatus::QP_CREATED) {
        HCCL_ERROR(
            "[HostRdmaConnection][%s] status[%s] is not expected.", __func__, rdmaConnStatus_.Describe().c_str());
        return HCCL_E_AGAIN;
    }

    if (!rmtQpAttr_.IsValid()) {
        HCCL_ERROR("[HostRdmaConnection][%s] remote Qp Attr is empty, exchange qp attr first", __func__);
        return HCCL_E_INTERNAL;
    }

    struct QpAttr localQpAttr;
    s32 ret = RaGetQpAttr(qpInfo_.qpHandle, &localQpAttr);
    if (ret != 0) {
        return HCCL_E_ROCE_CONNECT;
    }

    HCCL_INFO(
        "[HostRdmaConnection::ModifyQp] HostRdmaConnection qpInfo_: serviceLevel[%d], trafficClass[%d], retryCnt[%d], "
        "retryInterval[%d]. "
        "roceAttr_: sl[%d], tc[%d], retryCnt[%d], retryInterval[%d]",
        qpInfo_.serviceLevel, qpInfo_.trafficClass, qpInfo_.retryCnt, qpInfo_.retryInterval, roceAttr_.sl, roceAttr_.tc,
        roceAttr_.retryCnt, roceAttr_.retryInterval);
    HCCL_DEBUG(
        "[HostRdmaConnection::ModifyQp] HostRdmaConnection localQp: qpn[%u], psn[%u], gidIdx[%u]", localQpAttr.qpn,
        localQpAttr.psn, localQpAttr.gidIdx);
    HCCL_DEBUG(
        "[HostRdmaConnection::ModifyQp] HostRdmaConnection remoteQp: qpn[%u], psn[%u], gidIdx[%u]", rmtQpAttr_.qpn,
        rmtQpAttr_.psn, rmtQpAttr_.gid_idx);

    struct TypicalQp localQp;
    struct TypicalQp rmtQp;
    localQp.sl = roceAttr_.sl;
    localQp.tc = roceAttr_.tc;
    localQp.retryCnt = roceAttr_.retryCnt;
    localQp.retryTime = roceAttr_.retryInterval;
    localQp.qpn = localQpAttr.qpn;
    localQp.psn = localQpAttr.psn;
    localQp.gidIdx = localQpAttr.gidIdx;
    (void)memcpy_s(localQp.gid, HCCP_GID_RAW_LEN, localQpAttr.gid, HCCP_GID_RAW_LEN);
    rmtQp.sl = qpInfo_.serviceLevel;
    rmtQp.tc = qpInfo_.trafficClass;
    rmtQp.retryCnt = qpInfo_.retryCnt;
    rmtQp.retryTime = qpInfo_.retryInterval;
    rmtQp.qpn = rmtQpAttr_.qpn;
    rmtQp.psn = rmtQpAttr_.psn;
    rmtQp.gidIdx = rmtQpAttr_.gid_idx;
    (void)memcpy_s(rmtQp.gid, HCCP_GID_RAW_LEN, rmtQpAttr_.gid, HCCP_GID_RAW_LEN);
    ret = RaTypicalQpModify(qpInfo_.qpHandle, &localQp, &rmtQp);
    if (ret != 0) {
        HCCL_ERROR("[modify][ra_qp]modify qp failed, ret(%d)", ret);
        return HCCL_E_ROCE_CONNECT;
    }
    rdmaConnStatus_ = RdmaConnStatus::QP_MODIFIED;
    return HCCL_SUCCESS;
}

} // namespace hcomm_experimental
