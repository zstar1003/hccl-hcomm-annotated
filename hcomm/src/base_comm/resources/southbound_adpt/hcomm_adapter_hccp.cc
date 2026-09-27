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
#include "hcomm_adapter_hccp.h"

#include <algorithm>

#include "securec.h"
#include "log.h"
#include "config_plf_log_v2.h"
#include "orion_adpt_utils.h"
#include "hccp_tlv.h"
#include "hccp_common.h"

#include "hccp_async.h"
#include "hccp_async_ctx.h"
#include "enum_factory.h"

#include "hccp_tlv_hdc_manager.h"
#include "exception_handler.h"
#include "hccp_ctx_dfx.h"

namespace hcomm {
using Hccl::PLF_CHANNEL;

HcclResult IpAddressToHccpEid(const Hccl::IpAddress& ipAddr, Eid& eid)
{
    HCCL_INFO("EID ipAddr[%s]", ipAddr.Describe().c_str());
    int32_t sRet = memcpy_s(eid.raw, sizeof(eid.raw), ipAddr.GetEid().raw, sizeof(ipAddr.GetEid().raw));
    if (sRet != EOK) {
        HCCL_ERROR("[%s] memcpy failed[%d].", __func__, sRet);
        return HcclResult::HCCL_E_MEMORY;
    }
    HCCL_INFO(
        "[IpAddressToHccpEid] hccpEid.in6.subnetPrefix[%016llx], hccpEid.in6.interfaceId[%016llx]",
        eid.in6.subnetPrefix, eid.in6.interfaceId);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult IpAddressToReverseHcclEid(const Hccl::IpAddress& ipAddr, Hccl::Eid& eid)
{
    HCCL_INFO("EID ipAddr[%s]", ipAddr.Describe().c_str());
    int32_t sRet = memcpy_s(eid.raw, sizeof(eid.raw), ipAddr.GetReverseEid().raw, sizeof(ipAddr.GetReverseEid().raw));
    if (sRet != EOK) {
        HCCL_ERROR("[%s] memcpy failed[%d].", __func__, sRet);
        return HcclResult::HCCL_E_MEMORY;
    }
    HCCL_INFO(
        "[IpAddressToHccpEid] hccpEid.in6.subnetPrefix[%016llx], hccpEid.in6.interfaceId[%016llx]",
        eid.in6.subnetPrefix, eid.in6.interfaceId);
    return HcclResult::HCCL_SUCCESS;
}

inline Hccl::IpAddress HccpEidToIpAddress(Eid& hccpEid)
{
    Hccl::Eid eid{};
    HCCL_INFO(
        "[HccpEidToIpAddress] hccpEid.in6.subnetPrefix[%016llx], hccpEid.in6.interfaceId[%016llx]",
        hccpEid.in6.subnetPrefix, hccpEid.in6.interfaceId);
    s32 sRet = memcpy_s(eid.raw, sizeof(eid.raw), hccpEid.raw, sizeof(hccpEid.raw));
    if (sRet != EOK) {
        HCCL_ERROR("failed to change eid to ip");
        return Hccl::IpAddress{}; // 暂时不处理
    }
    return Hccl::IpAddress(eid);
}

HcclResult RaGetDevEidInfos(const RaInfo& raInfo, std::vector<DevEidInfo>& devEidInfos)
{
    uint32_t num = 0;
    int32_t ret = RaGetDevEidInfoNum(raInfo, &num);
    if (ret != 0) {
        HCCL_ERROR("call RaGetDevEidInfoNum failed, error code =%d.", ret);
        return HcclResult::HCCL_E_NETWORK;
    }

    struct HccpDevEidInfo infoList[num] = {};
    ret = RaGetDevEidInfoList(raInfo, infoList, &num);
    if (ret != 0) {
        HCCL_ERROR("call RaGetDevEidInfoList failed, error code =%d.", ret);
        return HcclResult::HCCL_E_NETWORK;
    }

    devEidInfos.resize(num);
    for (uint32_t i = 0; i < num; i++) {
        devEidInfos[i].name = (infoList[i].name);
        Hccl::IpAddress ipAddr = HccpEidToIpAddress(infoList[i].eid);
        CHK_RET(IpAddressToCommAddr(ipAddr, devEidInfos[i].commAddr));
        devEidInfos[i].type = infoList[i].type;
        devEidInfos[i].eidIndex = infoList[i].eidIndex;
        devEidInfos[i].dieId = infoList[i].dieId;
        devEidInfos[i].chipId = infoList[i].chipId;
        devEidInfos[i].funcId = infoList[i].funcId;
        devEidInfos[i].devFeature = infoList[i].devFeature;
        PLF_CONFIG_INFO(
            PLF_CHANNEL, "Get DevEidInfo: name[%s] eidIdx[%u] dieId[%u] chipId[%u] feId[%u] devFeature[0x%x].",
            devEidInfos[i].name.c_str(), devEidInfos[i].eidIndex, devEidInfos[i].dieId, devEidInfos[i].chipId,
            devEidInfos[i].funcId, devEidInfos[i].devFeature);
    }

    return HcclResult::HCCL_SUCCESS;
}

RequestResult HccpGetAsyncReqResult(RequestHandle& reqHandle)
{
    if (reqHandle == 0) {
        HCCL_ERROR("[%s] failed, reqHandle is 0.", __func__);
        return RequestResult::INVALID_PARA;
    }

    int reqResult = 0;
    int32_t ret = RaGetAsyncReqResult(ReinterpretAs<void*>(reqHandle), &reqResult);
    // 返回 OTHERS_EAGAIN 代表查询到异步任务未完成，需要重新查询，此时保留handle
    if (ret == OTHERS_EAGAIN) {
        return RequestResult::NOT_COMPLETED;
    }

    // 返回码非0代表调用查询接口失败，当前仅入参错误时触发
    if (ret != 0) {
        HCCL_ERROR(
            "[%s] failed to get asynchronous request result[%d], "
            "reqhandle[%llx].",
            __func__, ret, reqHandle);
        return RequestResult::GET_REQ_RESULT_FAILED;
    }

    RequestHandle tmpReqHandle = reqHandle;
    // 返回码为 0 时，reqResult为异步任务完成结果，0代表成功，其他值代表失败
    // SOCK_EAGAIN 为 socket 类执行结果，代表 socket 接口失败需要重试
    if (reqResult == SOCK_EAGAIN) {
        return RequestResult::SOCK_E_AGAIN;
    }

    if (reqResult != 0) {
        HCCL_ERROR(
            "[%s] failed, the asynchronous request "
            "error[%d], reqhandle[%llx].",
            __func__, reqResult, tmpReqHandle);
        return RequestResult::ASYNC_REQUEST_FAILED;
    }

    return RequestResult::COMPLETED;
}

const std::map<HrtTransportMode, TransportModeT> HRT_TRANSPORT_MODE_MAP
    = {{HrtTransportMode::RM, TransportModeT::CONN_RM}};
const std::map<HrtJettyMode, JettyMode> HRT_JETTY_MODE_MAP
    = {{HrtJettyMode::STANDARD, JettyMode::JETTY_MODE_URMA_NORMAL},
       {HrtJettyMode::HOST_OFFLOAD, JettyMode::JETTY_MODE_USER_CTL_NORMAL},
       {HrtJettyMode::HOST_OPBASE, JettyMode::JETTY_MODE_USER_CTL_NORMAL},
       {HrtJettyMode::DEV_USED, JettyMode::JETTY_MODE_USER_CTL_NORMAL},
       {HrtJettyMode::CACHE_LOCK_DWQE, JettyMode::JETTY_MODE_CACHE_LOCK_DWQE},
       {HrtJettyMode::CCU_CCUM_CACHE, JettyMode::JETTY_MODE_CCU},
       {HrtJettyMode::CCU_TA_CACHE, JettyMode::JETTY_MODE_CCU_TA_CACHE}};

constexpr uint8_t RNR_RETRY = 7;
constexpr uint32_t RQ_DEPTH = 64;

HcclResult
HccpUbCreateJetty(const CtxHandle ctxhandle, const HrtRaUbCreateJettyParam& in, HrtRaUbJettyCreatedOutParam& out)
{
    struct QpCreateAttr attr {};
    attr.scqHandle = ReinterpretAs<void*>(in.sjfcHandle);
    attr.rcqHandle = ReinterpretAs<void*>(in.rjfcHandle);
    attr.srqHandle = ReinterpretAs<void*>(in.sjfcHandle);
    attr.rqDepth = RQ_DEPTH;
    attr.sqDepth = in.sqDepth;
    attr.transportMode = HRT_TRANSPORT_MODE_MAP.at(in.transMode);
    attr.ub.mode = HRT_JETTY_MODE_MAP.at(in.jettyMode);

    attr.ub.tokenValue = in.tokenValue;
    attr.ub.tokenIdHandle = ReinterpretAs<void*>(in.tokenIdHandle);
    attr.ub.flag.value = 0;
    /* errTime配置值：0-31
       0-7代表芯片配置值b00:512ms
       8-15代表芯片配置值b01:4s
       16-23代表芯片配置值b10:8s
       24-31代表芯片配置值b11:32s
    */
    attr.ub.errTimeout = in.errTimeout;
    attr.ub.priority = static_cast<uint8_t>(in.qos & 0xFU);
    attr.ub.rnrRetry = RNR_RETRY;
    attr.ub.flag.bs.shareJfr = 1;
    attr.ub.jettyId = in.jettyId;
    // 在continue模式下+配置了wqe的fence标记，并且远端有一些权限校验错误/内存异常错误，硬件会直接挂死
    // jfs_flag 的 error_suspend 设置为 1，
    attr.ub.jfsFlag.bs.errorSuspend = 1;

    if (in.jettyMode == HrtJettyMode::CCU_TA_CACHE) {
        attr.ub.tokenValue = in.tokenValue;
        attr.ub.taCacheMode.lockFlag = true;
        attr.ub.taCacheMode.sqeBufIdx = in.sqeBufIndex;
    } else {
        attr.ub.extMode.sqebbNum = in.sqDepth;
    }

    if (in.jettyMode == HrtJettyMode::HOST_OFFLOAD) {
        attr.ub.extMode.piType = 1;
    } else if (in.jettyMode == HrtJettyMode::CCU_CCUM_CACHE) {
        attr.ub.tokenValue = in.tokenValue;
        attr.ub.extMode.cstmFlag.bs.sqCstm = 1;
        attr.ub.extMode.sq.buffSize = in.sqBufSize;
        attr.ub.extMode.sq.buffVa = in.sqBufVa;
    } else if (in.jettyMode == HrtJettyMode::DEV_USED) {
        attr.ub.extMode.cstmFlag.bs.sqCstm = 1;
        attr.ub.extMode.sq.buffSize = in.sqBufSize;
        attr.ub.extMode.sq.buffVa = in.sqBufVa;
    }

    // 其他Mode暂时不需要额外更新特定字段
    HCCL_INFO(
        "Create jetty, input params: attr.ub.jettyId[%u], attr.rqDepth[%u], "
        "attr.sqDepth[%u], attr.transportMode[%d], attr.ub.mode[%d], "
        "attr.ub.extMode.sqebbNum[%u], attr.ub.extMode.sq.buffVa[%llx], "
        "attr.ub.extMode.sq.buffSize[%u], attr.ub.extMode.piType[%u], "
        "attr.ub.priority[%u], timeout[%u], attr.ub.taCacheMode.sqeBufIdx[%u].",
        attr.ub.jettyId, attr.rqDepth, attr.sqDepth, attr.transportMode, attr.ub.mode, attr.ub.extMode.sqebbNum,
        attr.ub.extMode.sq.buffVa, attr.ub.extMode.sq.buffSize, attr.ub.extMode.piType, attr.ub.priority,
        attr.ub.errTimeout, attr.ub.taCacheMode.sqeBufIdx);

    struct QpCreateInfo info {};
    void* qpHandle = nullptr;
    int32_t ret = RaCtxQpCreate(ctxhandle, &attr, &info, &qpHandle);
    if (ret != 0) {
        HCCL_ERROR(
            "[%s] failed, ctxHandle[%p] jetty_id[%u] JettyMode[%s] "
            "sqDepth[%u] sq.buffVa[%llx] sq.buffSize[%u].",
            __func__, ctxhandle, attr.ub.jettyId, in.jettyMode.Describe().c_str(), attr.sqDepth,
            attr.ub.extMode.sq.buffVa, attr.ub.extMode.sq.buffSize);
        return HcclResult::HCCL_E_NETWORK;
    }

    // 适配URMA，直接组装WQE的TOKENID需要进行移位，包括CCU与AICPU
    constexpr u32 URMA_TOKEN_ID_RIGHT_SHIFT = 8;

    out.handle = ReinterpretAs<JettyHandle>(qpHandle);
    out.id = info.ub.id;
    out.uasid = info.ub.uasid;
    out.jettyVa = info.va;
    out.dbVa = info.ub.dbAddr;
    out.dbTokenId = info.ub.dbTokenId >> URMA_TOKEN_ID_RIGHT_SHIFT;

    int32_t sRet = memcpy_s(out.key, sizeof(out.key), info.key.value, info.key.size);
    if (sRet != 0) {
        HCCL_ERROR("[%s] failed, memcpy failed[%d].", __func__, sRet);
        return HcclResult::HCCL_E_MEMORY;
    }
    out.keySize = info.key.size;
    attr.ub.tokenValue = 0; // 清理栈中的敏感信息
    PLF_CONFIG_INFO(
        PLF_CHANNEL, "Create Jetty: jettyId[%u] jettyVa[0x%llx] dbVa[0x%llx] uasid[%u] keySize[%u].", out.id,
        out.jettyVa, out.dbVa, out.uasid, out.keySize);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult HccpUbCreateJettyAsync(
    const CtxHandle ctxhandle, const HrtRaUbCreateJettyParam& in, std::vector<char>& out, void*& jettyHandle,
    RequestHandle& reqHandle)
{
    struct QpCreateAttr attr {};
    attr.scqHandle = ReinterpretAs<void*>(in.sjfcHandle);
    attr.rcqHandle = ReinterpretAs<void*>(in.rjfcHandle);
    attr.srqHandle = ReinterpretAs<void*>(in.sjfcHandle);
    attr.rqDepth = RQ_DEPTH;
    attr.sqDepth = in.sqDepth;
    attr.transportMode = HRT_TRANSPORT_MODE_MAP.at(in.transMode);
    attr.ub.mode = HRT_JETTY_MODE_MAP.at(in.jettyMode);

    attr.ub.tokenValue = in.tokenValue;
    attr.ub.tokenIdHandle = ReinterpretAs<void*>(in.tokenIdHandle);
    attr.ub.flag.value = 0;
    /* errTime配置值：0-31
       0-7代表芯片配置值b00:512ms
       8-15代表芯片配置值b01:4s
       16-23代表芯片配置值b10:8s
       24-31代表芯片配置值b11:32s
    */
    attr.ub.errTimeout = in.errTimeout;
    attr.ub.priority = static_cast<uint8_t>(in.qos & 0xFU);
    attr.ub.rnrRetry = RNR_RETRY;
    attr.ub.flag.bs.shareJfr = 1;
    attr.ub.jettyId = in.jettyId;
    // 在continue模式下+配置了wqe的fence标记，并且远端有一些权限校验错误/内存异常错误，硬件会直接挂死
    // jfs_flag 的 error_suspend 设置为 1，
    attr.ub.jfsFlag.bs.errorSuspend = 1;

    if (in.jettyMode == HrtJettyMode::CCU_TA_CACHE) {
        attr.ub.tokenValue = in.tokenValue;
        attr.ub.taCacheMode.lockFlag = true;
        attr.ub.taCacheMode.sqeBufIdx = in.sqeBufIndex;
    } else {
        attr.ub.extMode.sqebbNum = in.sqDepth;
    }

    if (in.jettyMode == HrtJettyMode::HOST_OFFLOAD) {
        attr.ub.extMode.piType = 1;
    } else if (in.jettyMode == HrtJettyMode::CCU_CCUM_CACHE) {
        attr.ub.tokenValue = in.tokenValue;
        attr.ub.extMode.cstmFlag.bs.sqCstm = 1;
        attr.ub.extMode.sq.buffSize = in.sqBufSize;
        attr.ub.extMode.sq.buffVa = in.sqBufVa;
    } else if (in.jettyMode == HrtJettyMode::DEV_USED) {
        attr.ub.extMode.cstmFlag.bs.sqCstm = 1;
        attr.ub.extMode.sq.buffSize = in.sqBufSize;
        attr.ub.extMode.sq.buffVa = in.sqBufVa;
    }

    // 其他Mode暂时不需要额外更新特定字段
    HCCL_INFO(
        "Create jetty, input params: attr.ub.jettyId[%u], attr.rqDepth[%u], "
        "attr.sqDepth[%u], attr.transportMode[%d], attr.ub.mode[%d], "
        "attr.ub.extMode.sqebbNum[%u], attr.ub.extMode.sq.buffVa[%llx], "
        "attr.ub.extMode.sq.buffSize[%u], attr.ub.extMode.piType[%u], "
        "attr.ub.priority[%u], timeout[%u], attr.ub.taCacheMode.sqeBufIdx[%u].",
        attr.ub.jettyId, attr.rqDepth, attr.sqDepth, attr.transportMode, attr.ub.mode, attr.ub.extMode.sqebbNum,
        attr.ub.extMode.sq.buffVa, attr.ub.extMode.sq.buffSize, attr.ub.extMode.piType, attr.ub.priority,
        attr.ub.errTimeout, attr.ub.taCacheMode.sqeBufIdx);

    void* raReqHandle = nullptr;
    out.resize(sizeof(QpCreateInfo));
    s32 ret
        = RaCtxQpCreateAsync(ctxhandle, &attr, ReinterpretAs<QpCreateInfo*>(out.data()), &jettyHandle, &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        HCCL_ERROR(
            "[%s] failed, call interface error[%d], raReqHandle[%p], "
            "ctxHandle[%p].",
            __func__, ret, raReqHandle, ctxhandle);
        return HcclResult::HCCL_E_NETWORK;
    }
    attr.ub.tokenValue = 0; // 清理栈中的token信息
    HCCL_INFO("[%s] ok, get handle[%llu].", __func__, ReinterpretAs<RequestHandle>(raReqHandle));
    reqHandle = ReinterpretAs<RequestHandle>(raReqHandle);
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult ImportJetty(
    const CtxHandle ctxHandle, u8* key, const u32 keyLen, const u32 tokenValue, const JettyImportExpCfg& cfg,
    const JettyImportMode mode, const TpProtocol protocol, HrtRaUbJettyImportedOutParam& out)
{
    if (mode == JettyImportMode::JETTY_IMPORT_MODE_NORMAL) {
        HCCL_ERROR("[%s] currently not support JETTY_IMPORT_MODE_NORMAL.", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    if (protocol != TpProtocol::RTP && protocol != TpProtocol::CTP) {
        HCCL_ERROR("[%s] failed, tp protocol[%s] is not expected.", __func__, protocol.Describe().c_str());
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    struct QpImportInfoT info {};
    int res = memcpy_s(info.in.key.value, sizeof(info.in.key.value), key, keyLen);
    if (res != 0) {
        HCCL_ERROR("[%s] memcpy_s failed, ret = %d", __func__, res);
        return HcclResult::HCCL_E_MEMORY;
    }
    info.in.key.size = keyLen;

    info.in.ub.mode = mode;
    info.in.ub.tokenValue = tokenValue;
    info.in.ub.policy = JettyGrpPolicy::JETTY_GRP_POLICY_RR;
    info.in.ub.type = TargetType::TARGET_TYPE_JETTY;

    info.in.ub.flag.value = 0;
    info.in.ub.flag.bs.tokenPolicy = TOKEN_POLICY_PLAIN_TEXT;

    info.in.ub.expImportCfg = cfg;
    // tp_type: 0->RTP, 1->CTP
    info.in.ub.tpType = protocol == TpProtocol::RTP ? 0 : 1;

    void* remQpHandle = nullptr;
    int32_t ret = RaCtxQpImport(ctxHandle, &info, &remQpHandle);
    if (ret != 0) {
        HCCL_ERROR(
            "[%s] failed, ctxHandle[%p] loc tp handle[%llx] "
            "rmt tp handle[%llx] loc tag[%llu] loc psn[%u] rmt psn[%u]"
            "protocol[%s].",
            __func__, ctxHandle, cfg.tpHandle, cfg.peerTpHandle, cfg.tag, cfg.txPsn, cfg.rxPsn,
            protocol.Describe().c_str());
        return HcclResult::HCCL_E_NETWORK;
    }

    out.handle = ReinterpretAs<TargetJettyHandle>(remQpHandle);
    out.targetJettyVa = info.out.ub.tjettyHandle;
    out.tpn = info.out.ub.tpn;
    info.in.ub.tokenValue = 0; // 清理栈中的敏感信息
    PLF_CONFIG_INFO(PLF_CHANNEL, "Import Jetty: tjettyHandle[0x%llx] tpn[%u].", out.targetJettyVa, out.tpn);
    return HcclResult::HCCL_SUCCESS;
}

static struct JettyImportExpCfg GetTpImportCfg(const JettyImportCfg& jettyImportCfg)
{
    struct JettyImportExpCfg cfg = {};

    cfg.tpHandle = jettyImportCfg.localTpHandle;
    cfg.peerTpHandle = jettyImportCfg.remoteTpHandle;
    cfg.tag = jettyImportCfg.localTag;
    cfg.txPsn = jettyImportCfg.localPsn;
    cfg.rxPsn = jettyImportCfg.remotePsn;

    return cfg;
}

HcclResult HccpUbTpImportJetty(
    const CtxHandle ctxHandle, u8* key, const u32 keyLen, const u32 tokenValue, const JettyImportCfg& jettyImportCfg,
    HrtRaUbJettyImportedOutParam& out)
{
    struct JettyImportExpCfg cfg = GetTpImportCfg(jettyImportCfg);
    const auto mode = JettyImportMode::JETTY_IMPORT_MODE_EXP;
    return ImportJetty(ctxHandle, key, keyLen, tokenValue, cfg, mode, jettyImportCfg.protocol, out);
}

static HcclResult ImportJettyAsync(
    CtxHandle ctxHandle, const HccpUbJettyImportedInParam& in, std::vector<char>& out, void*& remQpHandle,
    const JettyImportExpCfg& cfg, JettyImportMode mode, TpProtocol protocol, RequestHandle& reqHandle)
{
    if (mode == JettyImportMode::JETTY_IMPORT_MODE_NORMAL) {
        HCCL_ERROR("[%s] currently not support JETTY_IMPORT_MODE_NORMAL.", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    out.resize(sizeof(QpImportInfoT));
    struct QpImportInfoT* info = ReinterpretAs<QpImportInfoT*>(out.data());

    s32 ret = memcpy_s(info->in.key.value, sizeof(info->in.key.value), in.key, in.keyLen);
    if (ret != 0) {
        HCCL_ERROR("[%s] memcpy_s failed, ret=%d.", __func__, ret);
        return HcclResult::HCCL_E_MEMORY;
    }

    info->in.key.size = in.keyLen;
    info->in.ub.mode = mode;
    info->in.ub.tokenValue = in.tokenValue;
    info->in.ub.policy = JettyGrpPolicy::JETTY_GRP_POLICY_RR;
    info->in.ub.type = TargetType::TARGET_TYPE_JETTY;

    info->in.ub.flag.value = 0;
    info->in.ub.flag.bs.tokenPolicy = TOKEN_POLICY_PLAIN_TEXT;

    info->in.ub.expImportCfg = cfg;

    if (protocol != TpProtocol::RTP && protocol != TpProtocol::CTP) {
        HCCL_ERROR("[%s] failed, tp protocol[%s] is not expected.", __func__, protocol.Describe().c_str());
        return HcclResult::HCCL_E_PARA;
    }
    // tp_type: 0->RTP, 1->CTP
    info->in.ub.tpType = protocol == TpProtocol::RTP ? 0 : 1;

    void* raReqHandle = nullptr;
    ret = RaCtxQpImportAsync(ctxHandle, info, &remQpHandle, &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        HCCL_ERROR(
            "[%s] failed, call interface error[%d] raReqHandle[%p], "
            "ctxHandle[%p].",
            __func__, ret, raReqHandle, ctxHandle);
        return HcclResult::HCCL_E_NETWORK;
    }
    info->in.ub.tokenValue = 0;
    HCCL_INFO("[%s] ok, get handle[%llu]", __func__, ReinterpretAs<RequestHandle>(raReqHandle));
    reqHandle = ReinterpretAs<RequestHandle>(raReqHandle);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HccpUbTpImportJettyAsync(
    const CtxHandle ctxHandle, const HccpUbJettyImportedInParam& in, std::vector<char>& out, void*& remQpHandle,
    RequestHandle& reqHandle)
{
    struct JettyImportExpCfg cfg = GetTpImportCfg(in.jettyImportCfg);
    const auto mode = JettyImportMode::JETTY_IMPORT_MODE_EXP;
    return ImportJettyAsync(ctxHandle, in, out, remQpHandle, cfg, mode, in.jettyImportCfg.protocol, reqHandle);
}

HcclResult HccpRaTlvCcuCustomChannel(int32_t devLogicId, void* customIn, void* customOut)
{
    // 当前复用legacy流程单例，后续需整改
    EXCEPTION_HANDLE_BEGIN
    auto tlvHandle = Hccl::HccpTlvHdcManager::GetInstance().GetTlvHandle(devLogicId);
    CHK_RET(HccpRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, customIn, customOut));
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HccpRaTlvRequestForCustomChannel(void* tlvHandle, unsigned int msgType, void* customIn, void* customOut)
{
    CHK_PTR_NULL(tlvHandle);
    CHK_PTR_NULL(customIn);
    CHK_PTR_NULL(customOut);

    struct TlvMsg sendMsg {};
    sendMsg.type = msgType;
    sendMsg.length = sizeof(CustomChanInfoIn);
    sendMsg.data = static_cast<char*>(customIn);

    struct TlvMsg recvMsg {};
    recvMsg.type = msgType;
    recvMsg.length = sizeof(CustomChanInfoOut);
    recvMsg.data = static_cast<char*>(customOut);

    int ret = RaTlvRequest(tlvHandle, TLV_MODULE_TYPE_CCU, &sendMsg, &recvMsg);
    if (ret != 0) {
        HCCL_ERROR("[%s] RaTlvRequest fail, tlvHandle[%p], ret[%d]", __func__, tlvHandle, ret);
        return HCCL_E_NETWORK;
    }
    return HCCL_SUCCESS;
}

HcclResult HccpBatchQueryJettyStatus(
    const CtxHandle ctxHandle, const std::vector<JettyHandle>& jettyHandles, std::vector<JettyStatus>& jettyAttrs,
    u32& num)
{
    CHK_PTR_NULL(ctxHandle);
    if (jettyHandles.size() != num) {
        HCCL_ERROR("jettyHandles size[%zu] not equal to num[%u]", jettyHandles.size(), num);
        return HCCL_E_PARA;
    }
    std::vector<struct JettyAttr> raJettyAttrs(MAX_JETTY_QUERY_NUM);
    std::vector<void*> qp_handle(jettyHandles.size());
    for (size_t i = 0; i < jettyHandles.size(); ++i) {
        qp_handle[i] = ReinterpretAs<void*>(jettyHandles[i]);
    }
    auto ret = RaCtxQpQueryBatch(qp_handle.data(), raJettyAttrs.data(), &num);
    if (ret != 0) {
        HCCL_ERROR("RaBatchQueryJettyAttr failed.ret[%d]", ret);
        return HCCL_E_NETWORK;
    }
    if (num != jettyHandles.size()) {
        HCCL_ERROR("jettyAttrs num[%zu] not equal to input jettyHandles size[%zu]", num, jettyHandles.size());
        return HCCL_E_PARA;
    }

    for (u32 i = 0; i < num; i++) {
        JettyStatus jettyStatus = static_cast<JettyStatus::Value>(static_cast<int>(raJettyAttrs[i].state));
        jettyAttrs.push_back(jettyStatus);
    }
    return HCCL_SUCCESS;
}

HcclResult HccpGetUboeFlagEnable(const u32 devPhyId)
{
    u32 uboeVersion = 0;
    s32 versionRet = RaGetInterfaceVersion(devPhyId, GET_UBOE_FLAG_ENABLE_OPCODE, &uboeVersion);
    CHK_PRT_RET(
        versionRet != 0,
        HCCL_ERROR("[%s] RaGetInterfaceVersion failed, devPhyId=%u, versionRet=%d", __func__, devPhyId, versionRet),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        uboeVersion < GET_UBOE_FLAG_ENABLE_VERSION,
        HCCL_ERROR(
            "[%s] this package does not support to get uboe flag, "
            "please change new package. uboeVersion[%u].",
            __func__, uboeVersion),
        HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult HccpGetIpByEid(void* ctxHandle, const CommAddr& eidAddr, CommAddr& ipAddr)
{
    ipAddr = {};
    ipAddr.type = COMM_ADDR_TYPE_RESERVED;
    CHK_PTR_NULL(ctxHandle);
    CHK_PRT_RET(
        eidAddr.type != COMM_ADDR_TYPE_EID,
        HCCL_ERROR("[%s] invalid address type[%d], expected COMM_ADDR_TYPE_EID.", __func__, eidAddr.type), HCCL_E_PARA);

    union HccpEid hccpEid {};
    CHK_SAFETY_FUNC_RET(memcpy_s(hccpEid.raw, sizeof(hccpEid.raw), eidAddr.eid, sizeof(eidAddr.eid)));

    struct IpInfo ipInfo {};
    uint32_t num = 1U;
    const int32_t ret = RaGetIpByEid(ctxHandle, &hccpEid, &ipInfo, &num);
    CHK_PRT_RET(
        ret != 0, HCCL_ERROR("[%s] RaGetIpByEid failed, ctxHandle[%p], ret[%d].", __func__, ctxHandle, ret),
        HCCL_E_NETWORK);
    CHK_PRT_RET(
        num == 0, HCCL_ERROR("[%s] RaGetIpByEid returned no IPv4 address, ctxHandle[%p].", __func__, ctxHandle),
        HCCL_E_NOT_FOUND);
    CHK_PRT_RET(
        num != 1U,
        HCCL_ERROR("[%s] RaGetIpByEid returned unexpected address count[%u], ctxHandle[%p].", __func__, num, ctxHandle),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        ipInfo.family != AF_INET,
        HCCL_ERROR(
            "[%s] RaGetIpByEid returned unsupported address family[%d], expected AF_INET.", __func__, ipInfo.family),
        HCCL_E_NOT_SUPPORT);

    ipAddr.type = COMM_ADDR_TYPE_IP_V4;
    ipAddr.addr = ipInfo.ip.addr;
    HCCL_INFO(
        "[%s] query UBoE IPv4 success, ctxHandle[%p], IPv4[0x%08x].", __func__, ctxHandle, ntohl(ipAddr.addr.s_addr));
    return HCCL_SUCCESS;
}

HcclResult HccpRaGetDevBaseAttr(void* ctxHandle, struct DevBaseAttr* attr)
{
    int ret = RaGetDevBaseAttr(ctxHandle, attr);
    if (ret != 0) {
        HCCL_ERROR("[%s] RaGetDevBaseAttr failed, ctxHandle[%p], attr[%p], ret[%d]", __func__, ctxHandle, attr, ret);
        return HCCL_E_NETWORK;
    }
    HCCL_INFO(
        "HccpRaGetDevBaseAttr success, sqMaxDepth[%u], rqMaxDepth[%u], sqMaxSge[%u], rqMaxSge[%u], maxReadSize[%u], "
        "maxWriteSize[%u]",
        attr->sqMaxDepth, attr->rqMaxDepth, attr->sqMaxSge, attr->rqMaxSge, attr->maxReadSize, attr->maxWriteSize);
    return HCCL_SUCCESS;
}

HcclResult HccpGetCtpEnable(void* ctxHandle, bool& ctpEnable)
{
    ctpEnable = false;
    CHK_PTR_NULL(ctxHandle);
    DevBaseAttr attr{};
    CHK_RET(HccpRaGetDevBaseAttr(ctxHandle, &attr));

    for (uint32_t i = 0; i < MAX_PRIORITY_CNT; ++i) {
        if (attr.ub.priorityInfo[i].tpType.bs.ctp == 1) {
            ctpEnable = true;
            break;
        }
    }
    HCCL_INFO("[%s] ctxHandle[%p], ctpEnable[%d].", __func__, ctxHandle, ctpEnable);
    return HCCL_SUCCESS;
}

HcclResult HrtRaDumpJettyContext(JettyHandle jettyHandle, u32 jettyId)
{
    CHK_PTR_NULL(jettyHandle);

    uint8_t context[CONTEXT_MAX_LEN] = {0};
    unsigned int len = CONTEXT_MAX_LEN;
    int ret = RaCtxGetJettyContext(static_cast<void*>(jettyHandle), context, &len);

    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[HrtRaDumpJettyContext] RaCtxGetJettyContext failed, "
            "jettyId[%u], ret=%d",
            jettyId, ret),
        HCCL_E_INTERNAL);

    CHK_PRT_RET(
        len == 0 || len > CONTEXT_MAX_LEN,
        HCCL_ERROR("[HrtRaDumpJettyContext] invalid context len=%u, jettyId[%u]", len, jettyId), HCCL_E_INTERNAL);

    constexpr u32 bytesPerLine = 64;
    constexpr u32 kHexCharsPerByte = 2;
    for (u32 offset = 0; offset < len; offset += bytesPerLine) {
        u32 bytesThisLine = std::min(len - offset, bytesPerLine);

        char hexBuf[bytesPerLine * kHexCharsPerByte + 1] = {0};
        for (u32 i = 0; i < bytesThisLine; i++) {
            int sret = snprintf_s(
                hexBuf + i * kHexCharsPerByte, sizeof(hexBuf) - i * kHexCharsPerByte, kHexCharsPerByte, "%02x",
                context[offset + i]);
            CHK_PRT_RET(
                sret <= 0,
                HCCL_ERROR(
                    "[HrtRaDumpJettyContext] snprintf_s failed, dest[%p], destMax[%zu], count[%u], "
                    "contextVal[%u], ret[%d].",
                    static_cast<void*>(hexBuf + i * kHexCharsPerByte), sizeof(hexBuf) - i * kHexCharsPerByte,
                    kHexCharsPerByte, context[offset + i], sret),
                HCCL_E_INTERNAL);
        }

        HCCL_ERROR("[HrtRaDumpJettyContext] jettyId=%u, len=%u, JettyContext:%s", jettyId, len, hexBuf);
    }
    return HCCL_SUCCESS;
}

} // namespace hcomm
