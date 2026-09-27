/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tp_mgr.h"

#include <algorithm>
#include <vector>

#include "hccp_ctx.h"
#include "hccp_async_ctx.h"

#include "hccl_common.h"
#include "exception_handler.h"
#include "network_api_exception.h"
#include "orion_adapter_hccp.h"
#include "rdma_handle_manager.h"
#include "dev_type.h"
#include "orion_adapter_rts.h"
#include "env_config/env_config_v2.h"
#include "tp_qos.h"

namespace hcomm {

namespace {
    constexpr uint32_t kTpAttrSlAvailableBit = 17U;
    static constexpr uint32_t kTpAttrBitmapSl = (1U << 10U);
    static constexpr uint32_t kTpAttrBitmapDscp = (1U << 8U);
    static constexpr uint32_t kTpAttrDscpConfigModeBit = 18U;

    static constexpr QosKey QosMapKey(uint32_t qos) noexcept { return static_cast<QosKey>(qos & 0xFFU); }

    // MAINBOARD_PCIE_STD（PCIE 标卡）：跳过 GetTpAttr/SL 策略，固定使用 TP 列表首个 TP；
    // jetty priority（SL）取 2，为标卡 UB 互通方案约定档位，与现网标卡环境对齐。
    static constexpr uint32_t kPcieStdMappedSl = 2U;

    static HcclResult IsPcieStdMainboardByPhyId(uint32_t devPhyId, bool& isPcieStd)
    {
        isPcieStd = false;
        u32 devLogicId = 0U;
        CHK_RET(hrtGetDeviceIndexByPhyId(devPhyId, devLogicId));
        Hccl::HcclMainboardId mainboardId = Hccl::HcclMainboardId::MAINBOARD_OTHERS;
        CHK_RET(Hccl::HrtGetMainboardId(devLogicId, mainboardId));
        isPcieStd = (mainboardId == Hccl::HcclMainboardId::MAINBOARD_PCIE_STD);
        return HcclResult::HCCL_SUCCESS;
    }

    struct TpInfoAddrKey {
        Hccl::IpAddress locAddr{};
        Hccl::IpAddress rmtAddr{};
        QosKey qosKey{0};
    };

    static HcclResult ResolveTpInfoAddrKey(const GetTpInfoParam& param, TpInfoAddrKey& out)
    {
        CHK_RET(CommAddrToIpAddress(param.locAddr, out.locAddr));
        CHK_RET(CommAddrToIpAddress(param.rmtAddr, out.rmtAddr));
        out.qosKey = QosMapKey(param.qos);
        return HcclResult::HCCL_SUCCESS;
    }

    static uint32_t CalSlAvailableCnt(uint32_t mask)
    {
        uint32_t c = 0;
        for (uint32_t i = 0; i < 16U; ++i) {
            if ((mask & (1U << i)) != 0U) {
                ++c;
            }
        }
        return c;
    }

    static uint32_t SlValueAtRankInMask16(uint32_t mask, uint32_t rank)
    {
        uint32_t seen = 0;
        for (uint32_t bit = 0; bit < 16U; ++bit) {
            if ((mask & (1U << bit)) != 0U) {
                if (seen == rank) {
                    return bit;
                }
                ++seen;
            }
        }
        return 0;
    }

    static uint16_t ReadSlAvailableMask16(const struct TpAttr& attr) { return static_cast<uint16_t>(attr.slBitmap); }

    static uint32_t ResolveSlAvailableCntForPolicy(uint16_t slMask, uint32_t slLevelCount)
    {
        uint32_t slAvailableCnt = CalSlAvailableCnt(slMask);
        if (slLevelCount != 0U) {
            slAvailableCnt = std::min(slLevelCount, slAvailableCnt);
        }
        return slAvailableCnt;
    }

    static bool
    ApplyQosTpSlPolicy(const GetTpInfoParam& param, uint16_t slMask, uint32_t& tpListIndexOut, uint32_t& mappedSlOut)
    {
        const uint32_t slAvailableCnt = ResolveSlAvailableCntForPolicy(slMask, param.slLevelCount);
        if (slAvailableCnt == 0U) {
            return false;
        }
        if (param.loopFirstTpLowestSl) {
            tpListIndexOut = 0U;
            mappedSlOut = SlValueAtRankInMask16(slMask, 0U);
            return true;
        }

        const uint32_t qos = param.qos;
        const uint32_t numGroups = slAvailableCnt;
        const uint32_t groupIdx = Hccl::TpQosResolveQosSlGroupIdx(qos, numGroups);
        if (groupIdx >= numGroups) {
            HCCL_ERROR(
                "[TpMgr][%s] groupIdx out of range: groupIdx[%u] numGroups[%u] qos[%u] slAvailableCnt[%u].", __func__,
                groupIdx, numGroups, qos, slAvailableCnt);
            return false;
        }

        tpListIndexOut = 0U;
        const uint32_t slRank = (slAvailableCnt - 1U) - groupIdx;
        mappedSlOut = SlValueAtRankInMask16(slMask, slRank);
        return true;
    }

    static uint8_t ResolveUboeDscpLookupQos(const GetTpInfoParam& param, uint32_t nTp, uint16_t slMask)
    {
        (void)nTp;
        (void)slMask;
        if (param.loopFirstTpLowestSl) {
            return 0U;
        }
        return static_cast<uint8_t>(param.qos & 0xFFU);
    }

    /// isSync=false（异步 GetTpInfo 写回 SL/DSCP）：HrtRaSetTpAttrAsync。
    /// 阻塞等待在 adapter 内（RaSetTpAttrAsync + WaitRequestResult），本函数返回时 Set 已生效。
    /// 不用 RaCtxSetTpAttr，避免 Rs 路径 phyId 无效（与 TpManager::SetTpAttrAsync 一致）。
    static HcclResult SetTpAttrAsync(
        const Hccl::RdmaHandle rdmaHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr& attr,
        const char* logTag)
    {
        Hccl::RequestHandle reqHandle = 0;
        try {
            const HcclResult hret = Hccl::HrtRaSetTpAttrAsync(rdmaHandle, tpHandle, attrBitmap, attr, reqHandle);
            if (hret != HcclResult::HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[TpMgr][%s] HrtRaSetTpAttrAsync failed hcclRet[%d] tpHandle[%llu].", logTag,
                    static_cast<int>(hret), static_cast<unsigned long long>(tpHandle));
            }
            return hret;
        } catch (const Hccl::NetworkApiException& ex) {
            HCCL_ERROR(
                "[TpMgr][%s] HrtRaSetTpAttrAsync exception: %s tpHandle[%llu].", logTag, ex.what(),
                static_cast<unsigned long long>(tpHandle));
            return HcclResult::HCCL_E_NETWORK;
        }
    }

    static HcclResult
    CommitMappedSlToTpAttr(const uint32_t devPhyId, const CommAddr& locCommAddr, uint64_t tpHandle, uint32_t mappedSl)
    {
        if (tpHandle == 0U) {
            HCCL_ERROR("[TpMgr][CommitMappedSlToTpAttr] tpHandle is 0");
            return HcclResult::HCCL_E_INTERNAL;
        }
        Hccl::IpAddress locAddr{};
        CHK_RET(CommAddrToIpAddress(locCommAddr, locAddr));
        const Hccl::RdmaHandle rdmaHandle = Hccl::RdmaHandleManager::GetInstance().GetByIp(devPhyId, locAddr);
        CHK_PTR_NULL(rdmaHandle);

        struct TpAttr tpSlAttr {};
        tpSlAttr.sl = static_cast<uint8_t>(mappedSl & 0xFU);
        const HcclResult hret
            = SetTpAttrAsync(rdmaHandle, tpHandle, kTpAttrBitmapSl, tpSlAttr, "CommitMappedSlToTpAttr");
        if (hret == HcclResult::HCCL_SUCCESS) {
            HCCL_INFO(
                "[TpMgr][CommitMappedSlToTpAttr] ok tpHandle[%llu] sl[%u].", static_cast<unsigned long long>(tpHandle),
                static_cast<unsigned>(mappedSl & 0xFU));
        }
        return hret;
    }

    static HcclResult
    CommitUboeDscpToTpAttr(const uint32_t devPhyId, const CommAddr& locCommAddr, uint64_t tpHandle, uint8_t dscp)
    {
        if (tpHandle == 0U) {
            HCCL_ERROR("[TpMgr][CommitUboeDscpToTpAttr] tpHandle is 0");
            return HcclResult::HCCL_E_INTERNAL;
        }
        Hccl::IpAddress locAddr{};
        CHK_RET(CommAddrToIpAddress(locCommAddr, locAddr));
        const Hccl::RdmaHandle rdmaHandle = Hccl::RdmaHandleManager::GetInstance().GetByIp(devPhyId, locAddr);
        CHK_PTR_NULL(rdmaHandle);

        struct TpAttr tpDscpAttr {};
        tpDscpAttr.dscp = static_cast<uint8_t>(dscp & 0x3FU);
        const HcclResult hret
            = SetTpAttrAsync(rdmaHandle, tpHandle, kTpAttrBitmapDscp, tpDscpAttr, "CommitUboeDscpToTpAttr");
        if (hret == HcclResult::HCCL_SUCCESS) {
            HCCL_INFO(
                "[TpMgr][CommitUboeDscpToTpAttr] ok tpHandle[%llu] dscp[%u].",
                static_cast<unsigned long long>(tpHandle), static_cast<unsigned>(tpDscpAttr.dscp));
        }
        return hret;
    }

} // namespace

TpMgr& TpMgr::GetInstance(const uint32_t devicePhyId)
{
    static TpMgr tpMgr[MAX_MODULE_DEVICE_NUM + 1];

    uint32_t devPhyId = devicePhyId;
    if (devPhyId >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[TpMgr][%s] use the backup device, devPhyId[%u] should be "
            "less than %u.",
            __func__, devPhyId, MAX_MODULE_DEVICE_NUM);
        devPhyId = MAX_MODULE_DEVICE_NUM;
    }

    tpMgr[devPhyId].devPhyId_ = devPhyId;

    return tpMgr[devPhyId];
}

static HcclResult CheckRequestResult(RequestHandle& reqHandle)
{
    if (reqHandle == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    RequestResult result = HccpGetAsyncReqResult(reqHandle);
    if (result == RequestResult::NOT_COMPLETED) {
        return HcclResult::HCCL_E_AGAIN;
    }

    if (result != RequestResult::COMPLETED) {
        HCCL_ERROR("[TpMgr][%s] failed, result[%s] is unexpected.", __func__, result.Describe().c_str());
        return HcclResult::HCCL_E_NETWORK;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CheckTpProtocol(const TpProtocol tpProtocol)
{
    if (tpProtocol != TpProtocol::CTP && tpProtocol != TpProtocol::RTP && tpProtocol != TpProtocol::UBOE) {
        HCCL_ERROR("[TpMgr][%s] failed, tpProtocol[%s] is not supported.", __func__, tpProtocol.Describe().c_str());
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpMgr::LookupInfoCtxEntry(
    InfoCtxMap& infoMap, const Hccl::IpAddress& locAddr, const Hccl::IpAddress& rmtAddr, const QosKey qosKey,
    InfoCtxMap::iterator& lit, InfoRmtMap::iterator& rit, InfoQosMap::iterator& qosIt) const
{
    lit = infoMap.find(locAddr);
    if (lit == infoMap.end()) {
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    rit = lit->second.find(rmtAddr);
    if (rit == lit->second.end()) {
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    qosIt = rit->second.find(qosKey);
    if (qosIt == rit->second.end()) {
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpMgr::FindAndGetTpInfo(const GetTpInfoParam& param, TpInfo& tpInfo)
{
    TpInfoAddrKey key{};
    CHK_RET(ResolveTpInfoAddrKey(param, key));
    std::lock_guard<std::mutex> lock(GetInfoCtxMutex(param.tpProtocol));
    auto& infoMap = GetInfoCtxMap(param.tpProtocol);
    InfoCtxMap::iterator lit;
    InfoRmtMap::iterator rit;
    InfoQosMap::iterator qosIt;
    const auto lookupRet = LookupInfoCtxEntry(infoMap, key.locAddr, key.rmtAddr, key.qosKey, lit, rit, qosIt);
    if (lookupRet != HcclResult::HCCL_SUCCESS) {
        return lookupRet;
    }
    // 复用缓存：useCnt 仅在此处（命中）递增，与 CommitTpInfoToCache 写入路径分离。
    qosIt->second.useCnt += 1;
    tpInfo = qosIt->second.tpInfo;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpMgr::BeginGetTpInfoListRequest(const GetTpInfoParam& param, ReqQosMap& qosMap, const QosKey qosKey) const
{
    RequestCtx& reqCtx = qosMap[qosKey];
    CHK_RET(StartGetTpInfoListRequest(param, reqCtx));
    HCCL_INFO(
        "[TpMgr][GetTpInfo] RaGetTpInfoListAsync submitted, devPhyId[%u] reqHandle[%llu] phase[WAIT_LIST] "
        "param[%s].",
        devPhyId_, static_cast<unsigned long long>(reqCtx.handle), param.Describe().c_str());
    return HcclResult::HCCL_E_AGAIN;
}

HcclResult TpMgr::AdvanceGetTpInfoWaitList(
    const GetTpInfoParam& param, RequestCtx& reqCtx, ReqQosMap& qosMap, const ReqQosMap::iterator it,
    std::unique_lock<std::mutex>& reqCtxLock, TpInfo& tpInfo)
{
    if (reqCtx.tpInfoNum == 0U) {
        qosMap.erase(it);
        reqCtxLock.unlock();
        HCCL_WARNING(
            "[TpMgr][%s] failed to find tp info, tpInfoNum is 0, param[%s].", __func__, param.Describe().c_str());
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    bool isPcieStd = false;
    CHK_RET(IsPcieStdMainboardByPhyId(devPhyId_, isPcieStd));
    if (isPcieStd) {
        const struct HccpTpInfo* list = reinterpret_cast<const struct HccpTpInfo*>(reqCtx.dataBuffer.data());
        HCCL_INFO(
            "[TpMgr][%s] pcie std mainboard: skip GetTpAttr, devPhyId[%u] tpInfoNum[%u] mappedSl[%u] "
            "tpHandle[%llu] param[%s].",
            __func__, devPhyId_, reqCtx.tpInfoNum, kPcieStdMappedSl, static_cast<unsigned long long>(list[0].tpHandle),
            param.Describe().c_str());
        RequestCtx completedReqCtx = std::move(it->second);
        qosMap.erase(it);
        reqCtxLock.unlock();
        CHK_RET(HandleCompletedRequest(std::move(completedReqCtx), param, tpInfo));
        return HcclResult::HCCL_SUCCESS;
    }
    const struct HccpTpInfo* list = reinterpret_cast<const struct HccpTpInfo*>(reqCtx.dataBuffer.data());
    HCCL_INFO(
        "[TpMgr][GetTpInfo] list stage ok, devPhyId[%u] tpInfoNum[%u] firstTpHandle[%llu] param[%s].", devPhyId_,
        reqCtx.tpInfoNum, static_cast<unsigned long long>(list[0].tpHandle), param.Describe().c_str());
    try {
        CHK_RET(StartGetTpAttrForFirstTp(param, reqCtx));
    } catch (...) {
        qosMap.erase(it);
        throw;
    }
    HCCL_INFO(
        "[TpMgr][GetTpInfo] RaGetTpAttrAsync submitted, devPhyId[%u] reqHandle[%llu] phase[WAIT_TP_ATTR] "
        "tpAttrBitmap[0x%x] param[%s].",
        devPhyId_, static_cast<unsigned long long>(reqCtx.handle), reqCtx.tpAttrBitmap, param.Describe().c_str());
    return HcclResult::HCCL_E_AGAIN;
}

HcclResult
TpMgr::PollGetTpInfoReqCtx(std::unique_lock<std::mutex>& reqCtxLock, const GetTpInfoParam& param, TpInfo& tpInfo)
{
    auto& reqCtxMap = GetReqCtxMap(param.tpProtocol);
    TpInfoAddrKey key{};
    CHK_RET(ResolveTpInfoAddrKey(param, key));
    auto& qosMap = reqCtxMap[key.locAddr][key.rmtAddr];
    auto it = qosMap.find(key.qosKey);
    if (it == qosMap.end()) {
        return BeginGetTpInfoListRequest(param, qosMap, key.qosKey);
    }

    RequestCtx& reqCtx = it->second;
    const auto ret = CheckRequestResult(reqCtx.handle);
    if (ret == HcclResult::HCCL_E_AGAIN) {
        return ret;
    }
    CHK_RET(ret);

    if (reqCtx.phase == ReqPhase::WAIT_LIST) {
        return AdvanceGetTpInfoWaitList(param, reqCtx, qosMap, it, reqCtxLock, tpInfo);
    }

    // 先 move 出槽位再 erase，避免 erase 析构槽内对象后再 move（UB / double free）
    RequestCtx completedReqCtx = std::move(it->second);
    qosMap.erase(it);
    reqCtxLock.unlock();
    CHK_RET(HandleCompletedRequest(std::move(completedReqCtx), param, tpInfo));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpMgr::GetTpInfo(const GetTpInfoParam& param, TpInfo& tpInfo)
{
    CHK_RET(CheckTpProtocol(param.tpProtocol));
    if (FindAndGetTpInfo(param, tpInfo) == HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_SUCCESS;
    }

    std::unique_lock<std::mutex> reqCtxLock(GetReqCtxMutex(param.tpProtocol));
    return PollGetTpInfoReqCtx(reqCtxLock, param, tpInfo);
}

HcclResult TpMgr::ReleaseTpInfo(const GetTpInfoParam& param, const TpInfo& tpInfo)
{
    TpInfoAddrKey key{};
    CHK_RET(ResolveTpInfoAddrKey(param, key));
    std::lock_guard<std::mutex> lock(GetInfoCtxMutex(param.tpProtocol));
    auto& infoMap = GetInfoCtxMap(param.tpProtocol);
    InfoCtxMap::iterator lit;
    InfoRmtMap::iterator rmtIt;
    InfoQosMap::iterator qosIt;
    const auto lookupRet = LookupInfoCtxEntry(infoMap, key.locAddr, key.rmtAddr, key.qosKey, lit, rmtIt, qosIt);
    if (lookupRet != HcclResult::HCCL_SUCCESS) {
        if (lit == infoMap.end()) {
            HCCL_ERROR("[TpMgr][%s] failed, tp info is not found, param[%s].", __func__, param.Describe().c_str());
        } else if (rmtIt == lit->second.end()) {
            HCCL_ERROR("[TpMgr][%s] failed, tp info is not found, param[%s].", __func__, param.Describe().c_str());
        } else {
            HCCL_ERROR(
                "[TpMgr][%s] failed, tp info is not found for qosKey[%u], param[%s].", __func__,
                static_cast<unsigned>(key.qosKey), param.Describe().c_str());
        }
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    // 未入缓存的并发 GetTpInfo 结果：与缓存 tpHandle 不一致，无需操作缓存。
    if (tpInfo.tpHandle != qosIt->second.tpInfo.tpHandle) {
        return HcclResult::HCCL_SUCCESS;
    }

    if (qosIt->second.useCnt > 1) {
        qosIt->second.useCnt -= 1;
        return HcclResult::HCCL_SUCCESS;
    }

    rmtIt->second.erase(qosIt);
    if (rmtIt->second.empty()) {
        lit->second.erase(rmtIt);
    }
    if (lit->second.empty()) {
        infoMap.erase(lit);
    }
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult GetTpInfoListAsync(
    const CtxHandle ctxHandle, const GetTpInfoParam& param, std::vector<char>& out, uint32_t& num,
    RequestHandle& reqHandle)
{
    Hccl::IpAddress locAddr{};
    Hccl::IpAddress rmtAddr{};
    CHK_RET(CommAddrToIpAddress(param.locAddr, locAddr));
    CHK_RET(CommAddrToIpAddress(param.rmtAddr, rmtAddr));
    const auto& tpProtocol = param.tpProtocol;

    struct GetTpCfg cfg {};
    cfg.flag.bs.rtp = tpProtocol == TpProtocol::RTP ? 1 : 0;
    cfg.flag.bs.ctp = tpProtocol == TpProtocol::CTP ? 1 : 0;
    cfg.flag.bs.uboe = tpProtocol == TpProtocol::UBOE ? 1 : 0;
    cfg.transMode = TransportModeT::CONN_RM;
    CHK_RET(IpAddressToHccpEid(locAddr, cfg.localEid));
    HCCL_INFO(
        "[TpMgr][GetTpInfoListAsync] cfg.local_eid[subnetPrefix[%016llx], interfaceId[%016llx]]",
        static_cast<unsigned long long>(cfg.localEid.in6.subnetPrefix),
        static_cast<unsigned long long>(cfg.localEid.in6.interfaceId));
    CHK_RET(IpAddressToHccpEid(rmtAddr, cfg.peerEid));
    HCCL_INFO(
        "[TpMgr][GetTpInfoListAsync] cfg.peer_eid[subnetPrefix[%016llx], interfaceId[%016llx]]",
        static_cast<unsigned long long>(cfg.peerEid.in6.subnetPrefix),
        static_cast<unsigned long long>(cfg.peerEid.in6.interfaceId));

    // buffer 须至少容纳本次请求的个数，避免 RS 按 num 写多条 HccpTpInfo 时越界破坏堆
    out.resize(static_cast<size_t>(Hccl::TP_HANDLE_REQUEST_NUM) * sizeof(struct HccpTpInfo));
    struct HccpTpInfo* info = reinterpret_cast<struct HccpTpInfo*>(out.data());

    void* raReqHandle = nullptr;
    num = Hccl::TP_HANDLE_REQUEST_NUM; // 指定需要从管控面申请 tp handle 的上限；完成后 num 为实际个数
    const s32 ret = RaGetTpInfoListAsync(ctxHandle, &cfg, info, &num, &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        HCCL_ERROR(
            "[TpMgr][%s] failed, call interface error[%d] raReqHandle[%p], ctxHandle[%p] locAddr[%s] rmtAddr[%s].",
            __func__, ret, raReqHandle, ctxHandle, locAddr.Describe().c_str(), rmtAddr.Describe().c_str());
        return HcclResult::HCCL_E_NETWORK;
    }

    reqHandle = reinterpret_cast<RequestHandle>(raReqHandle);
    HCCL_INFO("[TpMgr][%s] get request handle[%llu].", __func__, static_cast<unsigned long long>(reqHandle));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpMgr::StartGetTpInfoListRequest(const GetTpInfoParam& param, RequestCtx& reqCtx) const
{
    EXCEPTION_HANDLE_BEGIN
    reqCtx.phase = ReqPhase::WAIT_LIST;
    reqCtx.tpAttrBitmap = 0;
    (void)memset_s(&reqCtx.tpAttr, sizeof(reqCtx.tpAttr), 0, sizeof(reqCtx.tpAttr));

    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(param.locAddr, ipAddr));
    const CtxHandle ctxHandle
        = static_cast<CtxHandle>(Hccl::RdmaHandleManager::GetInstance().GetByIp(devPhyId_, ipAddr));
    CHK_PTR_NULL(ctxHandle);

    CHK_RET(GetTpInfoListAsync(ctxHandle, param, reqCtx.dataBuffer, reqCtx.tpInfoNum, reqCtx.handle));
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpMgr::StartGetTpAttrForFirstTp(const GetTpInfoParam& param, RequestCtx& reqCtx) const
{
    EXCEPTION_HANDLE_BEGIN(void) memset_s(&reqCtx.tpAttr, sizeof(reqCtx.tpAttr), 0, sizeof(reqCtx.tpAttr));
    reqCtx.tpAttrBitmap = (1U << kTpAttrSlAvailableBit) | kTpAttrBitmapSl;
    if (param.tpProtocol == TpProtocol::UBOE) {
        reqCtx.tpAttrBitmap |= kTpAttrBitmapDscp | (1U << kTpAttrDscpConfigModeBit);
    }

    const struct HccpTpInfo* list = reinterpret_cast<const struct HccpTpInfo*>(reqCtx.dataBuffer.data());
    const uint64_t firstTpHandle = list[0].tpHandle;

    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(param.locAddr, ipAddr));
    const CtxHandle ctxHandle
        = static_cast<CtxHandle>(Hccl::RdmaHandleManager::GetInstance().GetByIp(devPhyId_, ipAddr));
    CHK_PTR_NULL(ctxHandle);

    void* raReqHandle = nullptr;
    const s32 ret = RaGetTpAttrAsync(ctxHandle, firstTpHandle, &reqCtx.tpAttrBitmap, &reqCtx.tpAttr, &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        HCCL_ERROR(
            "[TpMgr][%s] RaGetTpAttrAsync failed ret[%d] raReqHandle[%p] ctx[%p] tpHandle[%llu].", __func__, ret,
            raReqHandle, ctxHandle, static_cast<unsigned long long>(firstTpHandle));
        return HcclResult::HCCL_E_NETWORK;
    }
    reqCtx.handle = reinterpret_cast<RequestHandle>(raReqHandle);
    reqCtx.phase = ReqPhase::WAIT_TP_ATTR;
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpMgr::FindAndGetTpAttr(const TpHandle tpHandle, TpAttrInfo& tpAttrInfo)
{
    std::lock_guard<std::mutex> lock(tpAttrCtxMutex_);
    auto attrIter = tpAttrCtxMap_.find(tpHandle);
    if (attrIter != tpAttrCtxMap_.end()) {
        attrIter->second.useCnt += 1;
        tpAttrInfo = attrIter->second.tpAttrInfo;
        return HcclResult::HCCL_SUCCESS;
    }

    return HcclResult::HCCL_E_NOT_FOUND;
}

HcclResult TpMgr::GetTpAttr(const GetTpAttrParam& param, TpAttrInfo& tpAttrInfo, CtxHandle ctxHandle)
{
    const TpHandle tpHandle = param.tpHandle;
    if (FindAndGetTpAttr(tpHandle, tpAttrInfo) == HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_SUCCESS;
    }

    std::unique_lock<std::mutex> reqCtxLock(tpAttrReqMutex_);
    auto reqCtxIter = tpAttrReqCtxMap_.find(tpHandle);
    if (reqCtxIter == tpAttrReqCtxMap_.end()) {
        HCCL_INFO("[TpMgr][%s] get new tpAttr, param[%s].", __func__, param.Describe().c_str());

        TpAttrRequestCtx& reqCtx = tpAttrReqCtxMap_[tpHandle];
        CHK_RET(StartGetTpAttrRequest(param, reqCtx, ctxHandle));
        return HcclResult::HCCL_E_AGAIN;
    }

    auto& reqCtx = reqCtxIter->second;
    auto ret = CheckRequestResult(reqCtx.handle);
    if (ret == HcclResult::HCCL_E_AGAIN) {
        return ret;
    }
    CHK_RET(ret);

    TpAttrRequestCtx completedReqCtx = reqCtxIter->second;
    tpAttrReqCtxMap_.erase(reqCtxIter);
    reqCtxLock.unlock();
    CHK_RET(HandleCompletedTpAttrRequest(std::move(completedReqCtx), tpHandle, tpAttrInfo));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
TpMgr::StartGetTpAttrRequest(const GetTpAttrParam& param, TpMgr::TpAttrRequestCtx& reqCtx, CtxHandle ctxHandle) const
{
    void* raReqHandle = nullptr;
    s32 ret = RaGetTpAttrAsync(
        ctxHandle, param.tpHandle, const_cast<uint32_t*>(&param.attrBitmap), &reqCtx.tpAttr, &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        HCCL_ERROR(
            "[TpMgr][%s] failed, call RaGetTpAttrAsync error[%d] raReqHandle[%p], "
            "tpHandle[0x%llx] attrBitmap[0x%x].",
            __func__, ret, raReqHandle, static_cast<unsigned long long>(param.tpHandle), param.attrBitmap);
        return HcclResult::HCCL_E_NETWORK;
    }

    reqCtx.handle = reinterpret_cast<RequestHandle>(raReqHandle);
    HCCL_INFO(
        "[TpMgr][%s] success, tpHandle[0x%llx] reqHandle[%llu].", __func__,
        static_cast<unsigned long long>(param.tpHandle), static_cast<unsigned long long>(reqCtx.handle));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpMgr::HandleCompletedTpAttrRequest(
    const TpMgr::TpAttrRequestCtx reqCtx, const TpHandle tpHandle, TpAttrInfo& tpAttrInfo)
{
    TpAttrInfo tmpTpAttrInfo(reqCtx.tpAttr);

    std::lock_guard<std::mutex> lock(tpAttrCtxMutex_);
    tpAttrCtxMap_[tpHandle] = {std::move(tmpTpAttrInfo), 1};

    tpAttrInfo = tpAttrCtxMap_[tpHandle].tpAttrInfo;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpMgr::ReleaseTpAttr(const TpHandle tpHandle, [[maybe_unused]] const TpAttrInfo& tpAttrInfo)
{
    std::lock_guard<std::mutex> lock(tpAttrCtxMutex_);
    auto attrIter = tpAttrCtxMap_.find(tpHandle);
    if (attrIter == tpAttrCtxMap_.end()) {
        HCCL_ERROR(
            "[TpMgr][%s] failed, tp attr is not found, "
            "tpHandle[0x%llx].",
            __func__, static_cast<unsigned long long>(tpHandle));
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    if (attrIter->second.useCnt > 1) {
        attrIter->second.useCnt -= 1;
        return HcclResult::HCCL_SUCCESS;
    }

    tpAttrCtxMap_.erase(attrIter);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpMgr::GetTpTotalTimeout(const TpAttrInfo& tpAttrInfo, uint32_t& tpTimeOutMs)
{
    uint8_t rawAtGear = tpAttrInfo.tpAttr.at;
    uint8_t rawRetryTimes = tpAttrInfo.tpAttr.retryTimesInit;

    uint8_t finalAtGear = rawAtGear;
    if (rawAtGear > AT_GEAR_MAX) {
        finalAtGear = AT_GEAR_DEFAULT;
        HCCL_WARNING(
            "[TpMgr][%s] Invalid at gear[%u], expect [%u, %u], use default gear[%u].", __func__,
            static_cast<unsigned>(rawAtGear), static_cast<unsigned>(AT_GEAR_MIN), static_cast<unsigned>(AT_GEAR_MAX),
            static_cast<unsigned>(finalAtGear));
    }

    uint32_t singleAtTimeoutMs = AT_TIMEOUT_MAP[finalAtGear];
    tpTimeOutMs = singleAtTimeoutMs * static_cast<uint32_t>(rawRetryTimes + 1);

    HCCL_INFO(
        "[TpMgr][%s] TP timeout calc success: raw_at_gear[%u], final_at_gear[%u], "
        "single_timeout[%ums], retry_times[%u], total_timeout[%ums].",
        __func__, static_cast<unsigned>(rawAtGear), static_cast<unsigned>(finalAtGear), singleAtTimeoutMs,
        static_cast<unsigned>(rawRetryTimes), tpTimeOutMs);

    return HcclResult::HCCL_SUCCESS;
}

static uint32_t TaHwValueToMs(uint8_t hwValue)
{
    uint8_t gear = hwValue / 8;
    switch (gear) {
        case TA_GEAR_INDEX_0:
            return TA_TIMEOUT_MS_GEAR0;
        case TA_GEAR_INDEX_1:
            return TA_TIMEOUT_MS_GEAR1;
        case TA_GEAR_INDEX_2:
            return TA_TIMEOUT_MS_GEAR2;
        case TA_GEAR_INDEX_3:
            return TA_TIMEOUT_MS_GEAR3;
        default:
            return TA_TIMEOUT_MS_GEAR2;
    }
}

static uint8_t FindMinTaHwValue(uint32_t tpTotalTimeoutMs)
{
    if (tpTotalTimeoutMs < TA_TIMEOUT_MS_GEAR0) {
        return TA_HW_GEAR0_BASE;
    }
    if (tpTotalTimeoutMs < TA_TIMEOUT_MS_GEAR1) {
        return TA_HW_GEAR1_BASE;
    }
    if (tpTotalTimeoutMs < TA_TIMEOUT_MS_GEAR2) {
        return TA_HW_GEAR2_BASE;
    }
    return TA_HW_GEAR3_BASE;
}

uint8_t TpMgr::CalcTaTimeout(TpProtocol tpProtocol, uint8_t taTimeOut, uint32_t tpTimeOutMs)
{
    // 未传入时回退到协议默认值（CTP=8，其他=16）
    uint8_t envValue = (taTimeOut != TA_TIMEOUT_NOT_SET) ? taTimeOut : ((tpProtocol == TpProtocol::CTP) ? 8 : 16);
    uint32_t envTimeOutMs = TaHwValueToMs(envValue);

    // CTP 协议直接使用 envValue，不与 TP 总超时比较
    if (tpProtocol == TpProtocol::CTP) {
        HCCL_INFO("[TpMgr][%s] [UbCtp] Env Value [%u] (%ums).", __func__, envValue, envTimeOutMs);
        return envValue;
    }

    // 其他协议：需要与 TP 总超时比较决定是否自动升挡
    uint8_t jettyTimeOut = envValue;
    if (envTimeOutMs <= tpTimeOutMs) {
        jettyTimeOut = FindMinTaHwValue(tpTimeOutMs);
        HCCL_WARNING(
            "[TpMgr][%s] Env timeout [%ums] <= TP timeout [%ums]. Auto upgrade TA to hw_val[%u] (%ums).", __func__,
            envTimeOutMs, tpTimeOutMs, jettyTimeOut, TaHwValueToMs(jettyTimeOut));
    } else {
        HCCL_INFO(
            "[TpMgr][%s] Env timeout [%ums] > TP timeout [%ums]. Use env gear base hw_val[%u] (%ums).", __func__,
            envTimeOutMs, tpTimeOutMs, envValue, envTimeOutMs);
    }

    return jettyTimeOut;
}

HcclResult TpMgr::BuildTpInfoAndCommitQosAttr(
    const GetTpInfoParam& param, const RequestCtx& reqCtx, const struct HccpTpInfo* baseInfoPtr,
    const uint32_t tpListIndex, const uint32_t mappedSl, TpInfo& tpInfo) const
{
    tpInfo = TpInfo{};
    tpInfo.tpHandle = baseInfoPtr[tpListIndex].tpHandle;
    tpInfo.mappedJettyPriority = mappedSl & 0xFU;
    tpInfo.hasMappedJettyPriority = true;

    bool isPcieStd = false;
    CHK_RET(IsPcieStdMainboardByPhyId(devPhyId_, isPcieStd));
    if (isPcieStd) {
        HCCL_INFO(
            "[TpMgr][%s] pcie std mainboard: skip SetTpAttr, devPhyId[%u] tpProtocol[%s] tpHandle[%llu] "
            "param[%s].",
            __func__, devPhyId_, param.tpProtocol.Describe().c_str(), static_cast<unsigned long long>(tpInfo.tpHandle),
            param.Describe().c_str());
    } else if (param.tpProtocol == TpProtocol::RTP || param.tpProtocol == TpProtocol::UBOE) {
        CHK_RET(CommitMappedSlToTpAttr(devPhyId_, param.locAddr, tpInfo.tpHandle, mappedSl));
    }
    if (!isPcieStd && param.tpProtocol == TpProtocol::UBOE && reqCtx.tpAttr.dscpConfigMode == 0) {
        const uint8_t dscpBefore = static_cast<uint8_t>(reqCtx.tpAttr.dscp & 0x3FU);
        const uint8_t requestQos = static_cast<uint8_t>(param.qos & 0xFFU);
        const uint16_t slMask = ReadSlAvailableMask16(reqCtx.tpAttr);
        const uint8_t dscpLookupQos = ResolveUboeDscpLookupQos(param, reqCtx.tpInfoNum, slMask);
        uint8_t dscp = Hccl::kUboeDefaultDscp;
        CHK_RET(Hccl::GetDscpByQos(devPhyId_, dscpLookupQos, dscp));
        CHK_RET(CommitUboeDscpToTpAttr(devPhyId_, param.locAddr, tpInfo.tpHandle, dscp));
        HCCL_INFO(
            "[TpMgr][%s] UBOE dscp updated: tpHandle[%llu] requestQos[%u] dscpLookupQos[%u] dscpBefore[%u] "
            "dscpAfter[%u].",
            __func__, static_cast<unsigned long long>(tpInfo.tpHandle), static_cast<unsigned>(requestQos),
            static_cast<unsigned>(dscpLookupQos), static_cast<unsigned>(dscpBefore), static_cast<unsigned>(dscp));
    }
    HCCL_INFO(
        "[TpMgr][%s] tp qos mapping ok: tpHandle[%llu] tpListIndex[%u] mappedSl[%u] jettyPriority[%u] qos[%u] "
        "param[%s].",
        __func__, static_cast<unsigned long long>(tpInfo.tpHandle), tpListIndex, static_cast<unsigned>(mappedSl & 0xFU),
        tpInfo.mappedJettyPriority, param.qos & 0xFFU, param.Describe().c_str());
    return HcclResult::HCCL_SUCCESS;
}

// GetTpInfo 完成后写入缓存。useCnt 仅在 FindAndGetTpInfo 命中时 +1，此处不做引用计数。
// 并发首次 GetTpInfo 时，先完成者写入缓存；后完成者若 tpHandle 不同则跳过写入，直接使用本地结果。
HcclResult TpMgr::CommitTpInfoToCache(const GetTpInfoParam& param, TpInfo& tpInfo)
{
    Hccl::IpAddress locAddr{};
    Hccl::IpAddress rmtAddr{};
    CHK_RET(CommAddrToIpAddress(param.locAddr, locAddr));
    CHK_RET(CommAddrToIpAddress(param.rmtAddr, rmtAddr));
    const QosKey qosKey = QosMapKey(param.qos);

    std::lock_guard<std::mutex> lock(GetInfoCtxMutex(param.tpProtocol));
    auto& infoMap = GetInfoCtxMap(param.tpProtocol);
    auto& rmtMap = infoMap[locAddr][rmtAddr];
    const auto qIt = rmtMap.find(qosKey);
    if (qIt == rmtMap.end()) {
        rmtMap[qosKey] = TpInfoCtx{tpInfo, 1U};
        return HcclResult::HCCL_SUCCESS;
    }

    // 缓存已存在：不再覆盖（避免并发后写覆盖先写的 tpHandle）；tpInfo 保持 GetTpInfo 本地结果。
    if (qIt->second.tpInfo.tpHandle != tpInfo.tpHandle) {
        HCCL_WARNING(
            "[TpMgr][%s] skip cache store, cached tpHandle[%llu] != local tpHandle[%llu] param[%s].", __func__,
            static_cast<unsigned long long>(qIt->second.tpInfo.tpHandle),
            static_cast<unsigned long long>(tpInfo.tpHandle), param.Describe().c_str());
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpMgr::HandleCompletedRequest(RequestCtx reqCtx, const GetTpInfoParam& param, TpInfo& tpInfo)
{
    const uint32_t tpInfoNum = reqCtx.tpInfoNum;
    if (tpInfoNum == 0U) {
        HCCL_WARNING(
            "[TpMgr][%s] failed to find tp info, tpInfoNum is 0, param[%s].", __func__, param.Describe().c_str());
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    tpInfo = TpInfo{};

    const struct HccpTpInfo* baseInfoPtr = reinterpret_cast<const struct HccpTpInfo*>(reqCtx.dataBuffer.data());
    bool isPcieStd = false;
    CHK_RET(IsPcieStdMainboardByPhyId(devPhyId_, isPcieStd));
    if (isPcieStd) {
        tpInfo.tpHandle = baseInfoPtr[0].tpHandle;
        tpInfo.mappedJettyPriority = kPcieStdMappedSl;
        tpInfo.hasMappedJettyPriority = true;
        HCCL_INFO(
            "[TpMgr][%s] pcie std mainboard: skip GetTpAttr/SetTpAttr, devPhyId[%u] tpInfoNum[%u] "
            "mappedSl[%u] tpHandle[%llu] param[%s].",
            __func__, devPhyId_, tpInfoNum, kPcieStdMappedSl, static_cast<unsigned long long>(tpInfo.tpHandle),
            param.Describe().c_str());
        return CommitTpInfoToCache(param, tpInfo);
    }

    const uint16_t slMask = ReadSlAvailableMask16(reqCtx.tpAttr);
    const uint32_t slAvailableCnt = CalSlAvailableCnt(slMask);
    HCCL_INFO(
        "[TpMgr][%s] after get_tp_attr: slMask[0x%04x] slAvailableCnt[%u] slBitmap[0x%x] dscp[%u] dscpConfigMode[%u] "
        "tpAttrBitmap[0x%x] param[%s].",
        __func__, static_cast<unsigned>(slMask), slAvailableCnt, static_cast<unsigned>(reqCtx.tpAttr.slBitmap),
        static_cast<unsigned>(reqCtx.tpAttr.dscp & 0x3FU), static_cast<unsigned>(reqCtx.tpAttr.dscpConfigMode & 1U),
        reqCtx.tpAttrBitmap, param.Describe().c_str());
    if (slAvailableCnt == 0U) {
        HCCL_ERROR(
            "[TpMgr][%s] sl_available mask empty after get_tp_attr, param[%s].", __func__, param.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint32_t tpListIndex = 0;
    uint32_t mappedSl = 0;
    if (!ApplyQosTpSlPolicy(param, slMask, tpListIndex, mappedSl)) {
        HCCL_ERROR(
            "[TpMgr][%s] ApplyQosTpSlPolicy failed, param[%s] nTp[%u] slAvailableCnt[%u] mask[%u].", __func__,
            param.Describe().c_str(), tpInfoNum, slAvailableCnt, static_cast<unsigned>(slMask));
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (tpListIndex >= tpInfoNum) {
        HCCL_ERROR(
            "[TpMgr][%s] tpListIndex out of range: tpListIndex[%u] tpInfoNum[%u] mappedSl[%u] param[%s].", __func__,
            tpListIndex, tpInfoNum, static_cast<unsigned>(mappedSl & 0xFU), param.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }

    CHK_RET(BuildTpInfoAndCommitQosAttr(param, reqCtx, baseInfoPtr, tpListIndex, mappedSl, tpInfo));
    return CommitTpInfoToCache(param, tpInfo);
}

TpMgr::InfoCtxMap& TpMgr::GetInfoCtxMap(const TpProtocol tpProtocol)
{
    switch (tpProtocol) {
        case TpProtocol::CTP:
            return ctpInfoMap_;
        case TpProtocol::RTP:
            return rtpInfoMap_;
        case TpProtocol::UBOE:
            return uboeInfoMap_;
        default:
            return rtpInfoMap_;
    }
}

TpMgr::ReqCtxMap& TpMgr::GetReqCtxMap(const TpProtocol tpProtocol)
{
    switch (tpProtocol) {
        case TpProtocol::CTP:
            return ctpReqMap_;
        case TpProtocol::RTP:
            return rtpReqMap_;
        case TpProtocol::UBOE:
            return uboeReqMap_;
        default:
            return rtpReqMap_;
    }
}

std::mutex& TpMgr::GetInfoCtxMutex(const TpProtocol tpProtocol)
{
    switch (tpProtocol) {
        case TpProtocol::CTP:
            return ctpInfoMutex_;
        case TpProtocol::RTP:
            return rtpInfoMutex_;
        case TpProtocol::UBOE:
            return uboeInfoMutex_;
        default:
            return rtpInfoMutex_;
    }
}

std::mutex& TpMgr::GetReqCtxMutex(const TpProtocol tpProtocol)
{
    switch (tpProtocol) {
        case TpProtocol::CTP:
            return ctpReqMutex_;
        case TpProtocol::RTP:
            return rtpReqMutex_;
        case TpProtocol::UBOE:
            return uboeReqMutex_;
        default:
            return rtpReqMutex_;
    }
}

} // namespace hcomm
