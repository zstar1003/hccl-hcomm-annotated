/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tp_manager.h"

#include <algorithm>
#include <arpa/inet.h>
#include <string>

#include "exception_util.h"
#include "hccl_common_v2.h"
#include "invalid_params_exception.h"
#include "env_config/env_config_v2.h"
#include "network_api_exception.h"
#include "tp_qos.h"

#include "dev_type.h"
#include "orion_adapter_rts.h"
#include "rdma_handle_manager.h"
#include "securec.h"

namespace Hccl {

// GetTpAttr 属性位图常量；匿名命名空间内工具与 TpManager 成员函数共用
constexpr uint32_t kTpAttrSlAvailableBit = 17U;
constexpr uint32_t kTpAttrBitmapSl = (1U << 10U);
constexpr uint32_t kTpAttrBitmapDscp = (1U << 8U);
constexpr uint32_t kTpAttrDscpConfigModeBit = 18U;
/// UBOE SetTpAttr 网络属性位图：bit2~8（0x1FC，sip/dip/smac/dmac/vlan + dscp，不含 sl）；HCCP 自动
/// urma_get_smac/get_dmac
constexpr uint32_t kTpAttrBitmapUboeNetWithDscp = 0x1FCU;

namespace {

    static constexpr QosKey QosMapKey(uint32_t qos) noexcept { return static_cast<QosKey>(qos & 0xFFU); }

    // MAINBOARD_PCIE_STD（PCIE 标卡）：跳过 GetTpAttr/SL 策略，固定使用 TP 列表首个 TP；
    // jetty priority（SL）取 2，为标卡 UB 互通方案约定档位，与现网标卡环境对齐。
    static constexpr uint32_t kPcieStdMappedSl = 2U;
    // TpAttr::sip/dip 为 16 字节；IPv4 映射时前 12 字节填 0，后 4 字节填 IPv4 四段地址（网络字节序）。
    static constexpr size_t kMappedIpArrayLen = 16U;
    static constexpr size_t kIpv4OctetCount = 4U;
    static constexpr size_t kIpv4MappedOffset = kMappedIpArrayLen - kIpv4OctetCount;

    static HcclResult IsPcieStdMainboard(uint32_t devLogicId, bool& isPcieStd)
    {
        isPcieStd = false;
        HcclMainboardId mainboardId = HcclMainboardId::MAINBOARD_OTHERS;
        CHK_RET(HrtGetMainboardId(devLogicId, mainboardId));
        isPcieStd = (mainboardId == HcclMainboardId::MAINBOARD_PCIE_STD);
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

    static bool ApplyLoopFirstTpLowestSl(
        const RaUbGetTpInfoParam& param, const uint16_t slMask, const uint32_t slRawCnt, const uint32_t slAvailableCnt,
        uint32_t& tpListIndexOut, uint32_t& mappedSlOut)
    {
        (void)param;
        tpListIndexOut = 0U;
        mappedSlOut = SlValueAtRankInMask16(slMask, 0U);
        HCCL_INFO(
            "[TpManager][ApplyQosTpSlPolicy] loopFirstTpLowestSl: slRawCnt[%u] slAvailableCnt[%u(after cap)] "
            "slMask[0x%x] tpListIdx[0] mappedSl[%u] param[%s].",
            slRawCnt, slAvailableCnt, static_cast<unsigned>(slMask), static_cast<unsigned>(mappedSlOut & 0xFU),
            param.Describe().c_str());
        return true;
    }

    static bool ApplyQosTpSlPolicy(
        const RaUbGetTpInfoParam& param, uint16_t slMask, uint32_t& tpListIndexOut, uint32_t& mappedSlOut)
    {
        const uint32_t slRawCnt = CalSlAvailableCnt(slMask);
        const uint32_t slAvailableCnt = ResolveSlAvailableCntForPolicy(slMask, param.slLevelCount);
        if (slAvailableCnt == 0U) {
            return false;
        }
        if (param.loopFirstTpLowestSl) {
            return ApplyLoopFirstTpLowestSl(param, slMask, slRawCnt, slAvailableCnt, tpListIndexOut, mappedSlOut);
        }

        const uint32_t qos = param.qos;
        const uint32_t numGroups = slAvailableCnt;
        const uint32_t groupIdx = TpQosResolveQosSlGroupIdx(qos, numGroups);
        if (groupIdx >= numGroups) {
            HCCL_ERROR(
                "[TpManager][%s] groupIdx out of range: groupIdx[%u] numGroups[%u] qos[%u] slAvailableCnt[%u].",
                __func__, groupIdx, numGroups, qos, slAvailableCnt);
            return false;
        }

        tpListIndexOut = 0U;
        // groupIdx 越大 SL 越低：slRank = (numGroups - 1) - groupIdx，从 slMask 中取对应 jetty priority。
        const uint32_t slRank = (slAvailableCnt - 1U) - groupIdx;
        mappedSlOut = SlValueAtRankInMask16(slMask, slRank);
        return true;
    }

    static uint32_t BuildGetTpAttrBitmapForSlPolicy(TpProtocol tpProtocol)
    {
        uint32_t bitmap = (1U << kTpAttrSlAvailableBit) | kTpAttrBitmapSl;
        if (tpProtocol == TpProtocol::UBOE || tpProtocol == TpProtocol::UB_RTP) {
            bitmap |= kTpAttrBitmapDscp | (1U << kTpAttrDscpConfigModeBit);
        }
        return bitmap;
    }

    /// isSync=false（异步 GetTpInfo 写回 SL/DSCP）：HrtRaSetTpAttrAsync。
    /// 阻塞等待在 adapter 内，本函数返回时 Set 已生效。
    /// 不用同步 HrtRaSetTpAttr，避免 Rs 路径 phyId 无效。
    static HcclResult SetTpAttrAsync(
        RdmaHandle ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr& attr, const char* logTag)
    {
        RequestHandle reqHandle = 0;
        HcclResult hret = HcclResult::HCCL_SUCCESS;
        TRY_CATCH_RETURN(hret = HrtRaSetTpAttrAsync(ctxHandle, tpHandle, attrBitmap, attr, reqHandle););
        if (hret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR(
                "[TpManager][%s] HrtRaSetTpAttrAsync failed hcclRet[%d] tpHandle[%llu].", logTag,
                static_cast<int>(hret), tpHandle);
        }
        return hret;
    }

    /// isSync=true（同步 GetTpInfo 写回 SL/DSCP）：HrtRaSetTpAttr，与 HrtRaGetTpAttr / RaUbGetTpInfo 成对。
    static HcclResult
    SetTpAttrSync(RdmaHandle ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr& attr, const char* logTag)
    {
        const HcclResult hret = HrtRaSetTpAttr(ctxHandle, tpHandle, attrBitmap, attr);
        if (hret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR(
                "[TpManager][%s] HrtRaSetTpAttr failed hcclRet[%d] tpHandle[%llu] attrBitmap[0x%x].", logTag,
                static_cast<int>(hret), tpHandle, attrBitmap);
        }
        return hret;
    }

    /// 按 GetTpInfo 的 isSync 选择 SetTpAttr：true → SetTpAttrSync；false → SetTpAttrAsync。
    static HcclResult SetTpAttrByPath(
        const bool isSync, RdmaHandle ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr& attr,
        const char* logTag)
    {
        if (isSync) {
            return SetTpAttrSync(ctxHandle, tpHandle, attrBitmap, attr, logTag);
        }
        return SetTpAttrAsync(ctxHandle, tpHandle, attrBitmap, attr, logTag);
    }

    static HcclResult
    CommitMappedSlToTpAttr(const bool isSync, RdmaHandle ctxHandle, uint64_t tpHandle, uint32_t mappedSl)
    {
        if (tpHandle == 0U) {
            HCCL_ERROR("[TpManager][CommitMappedSlToTpAttr] tpHandle is 0");
            return HcclResult::HCCL_E_INTERNAL;
        }
        if (!ctxHandle) {
            HCCL_ERROR("[TpManager][CommitMappedSlToTpAttr] ctxHandle is null tpHandle[%llu]", tpHandle);
            return HcclResult::HCCL_E_INTERNAL;
        }
        struct TpAttr tpSlAttr {};
        tpSlAttr.sl = static_cast<uint8_t>(mappedSl & 0xFU);
        return SetTpAttrByPath(isSync, ctxHandle, tpHandle, kTpAttrBitmapSl, tpSlAttr, "CommitMappedSlToTpAttr");
    }

    static RdmaHandle ResolveUbRdmaHandle(const bool isSync, const uint32_t devPhyId, const IpAddress& locAddr)
    {
        if (isSync) {
            Hccl::IpAddress addr = locAddr;
            return RdmaHandleManager::GetInstance().GetByAddr(
                devPhyId, LinkProtoType::UB, addr, Hccl::PortDeploymentType::HOST_NET);
        }
        return RdmaHandleManager::GetInstance().GetByIp(devPhyId, locAddr);
    }

    // 将 IPv4 点分十进制字符串解析为 TpAttr::sip/dip 所需的 16 字节映射格式：
    // 前 12 字节填 0，后 4 字节为 inet_pton 得到的网络字节序 IPv4 地址。
    static HcclResult Ipv4ToIpArray(const char* ipv4Str, uint8_t ipArr[kMappedIpArrayLen])
    {
        if (ipv4Str == nullptr || ipArr == nullptr) {
            return HcclResult::HCCL_E_PARA;
        }
        struct in_addr addr {};
        if (inet_pton(AF_INET, ipv4Str, &addr) != 1) {
            return HcclResult::HCCL_E_PARA;
        }
        const errno_t memsetRet = memset_s(ipArr, kMappedIpArrayLen, 0, kMappedIpArrayLen);
        if (memsetRet != EOK) {
            return HcclResult::HCCL_E_INTERNAL;
        }
        const errno_t cpyRet = memcpy_s(ipArr + kIpv4MappedOffset, kIpv4OctetCount, &addr.s_addr, kIpv4OctetCount);
        if (cpyRet != EOK) {
            return HcclResult::HCCL_E_INTERNAL;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    static HcclResult CommitUboeNetAttrsToTpAttr(
        const bool isSync, RdmaHandle ctxHandle, uint64_t tpHandle, const TpAttr& tpAttr, const IpAddress& locIpv4Addr,
        const IpAddress& rmtIpv4Addr, bool setDscp, uint8_t dscp)
    {
        if (tpHandle == 0U || !ctxHandle) {
            return HcclResult::HCCL_E_INTERNAL;
        }
        struct TpAttr netAttr = tpAttr;
        const std::string localIp = locIpv4Addr.GetIpStr();
        const std::string rmtIp = rmtIpv4Addr.GetIpStr();
        CHK_RET(Ipv4ToIpArray(localIp.c_str(), netAttr.sip));
        CHK_RET(Ipv4ToIpArray(rmtIp.c_str(), netAttr.dip));
        if (setDscp) {
            netAttr.dscp = static_cast<uint8_t>(dscp & 0x3FU);
        }
        HCCL_INFO(
            "[TpManager][CommitUboeNetAttrsToTpAttr] tpHandle[%llu] localIpv4[%s] rmtIpv4[%s] setDscp[%d] "
            "dscp[%u] attrBitmap[0x%x].",
            tpHandle, localIp.c_str(), rmtIp.c_str(), static_cast<int>(setDscp),
            static_cast<unsigned>(netAttr.dscp & 0x3FU), kTpAttrBitmapUboeNetWithDscp);
        return SetTpAttrByPath(
            isSync, ctxHandle, tpHandle, kTpAttrBitmapUboeNetWithDscp, netAttr, "CommitUboeNetAttrsToTpAttr");
    }

    static HcclResult CommitUbRtpDscpToTpAttr(const bool isSync, RdmaHandle ctxHandle, uint64_t tpHandle, uint8_t dscp)
    {
        if (tpHandle == 0U || !ctxHandle) {
            return HcclResult::HCCL_E_INTERNAL;
        }

        struct TpAttr dscpAttr {};
        dscpAttr.dscp = dscp;
        HCCL_INFO(
            "[TpManager][%s] tpHandle[%llu] dscp[%u] attrBitmap[0x%x].", __func__, tpHandle,
            static_cast<unsigned>(dscpAttr.dscp), kTpAttrBitmapDscp);
        return SetTpAttrByPath(isSync, ctxHandle, tpHandle, kTpAttrBitmapDscp, dscpAttr, "CommitUbRtpDscpToTpAttr");
    }

    static HcclResult CommitTpAttrsAfterSlMapping(
        const uint32_t devLogicId, const uint32_t devPhyId, const bool isSync, const RaUbGetTpInfoParam& param,
        const TpAttr& tpAttr, uint64_t tpHandle, uint32_t mappedSl, [[maybe_unused]] uint32_t nTp,
        [[maybe_unused]] uint16_t slMask)
    {
        bool isPcieStd = false;
        CHK_RET(IsPcieStdMainboard(devLogicId, isPcieStd));
        if (isPcieStd) {
            HCCL_INFO(
                "[TpManager][%s] pcie std mainboard: skip SetTpAttr, devPhyId[%u] tpProtocol[%s] tpHandle[%llu] "
                "param[%s].",
                __func__, devPhyId, param.tpProtocol.Describe().c_str(), tpHandle, param.Describe().c_str());
            return HcclResult::HCCL_SUCCESS;
        }
        const RdmaHandle ctxHandle = ResolveUbRdmaHandle(isSync, devPhyId, param.locAddr);
        if (!ctxHandle) {
            HCCL_ERROR(
                "[TpManager][%s] ctxHandle null devPhyId[%u] isSync[%d] loc[%s].", __func__, devPhyId,
                static_cast<int>(isSync), param.locAddr.Describe().c_str());
            return HcclResult::HCCL_E_INTERNAL;
        }
        // TP / UBOE / UB_RTP：将 TP QoS/SL 策略得到的 mapped SL 写回 TP；CTP 不向 TP 写 SL（与 Next TpMgr 一致）
        if (param.tpProtocol == TpProtocol::TP || param.tpProtocol == TpProtocol::UBOE
            || param.tpProtocol == TpProtocol::UB_RTP) {
            CHK_RET(CommitMappedSlToTpAttr(isSync, ctxHandle, tpHandle, mappedSl));
        }
        if (param.tpProtocol != TpProtocol::UBOE && param.tpProtocol != TpProtocol::UB_RTP) {
            return HcclResult::HCCL_SUCCESS;
        }
        if (param.tpProtocol == TpProtocol::UB_RTP) {
            // UB_RTP：仅当 dscp mode == 0 时下发 DSCP。
            if (tpAttr.dscpConfigMode != 0) {
                return HcclResult::HCCL_SUCCESS;
            }

            uint8_t dscp = kUboeDefaultDscp;
            CHK_RET(GetDscpByQos(devPhyId, param.qos, dscp));
            CHK_RET(CommitUbRtpDscpToTpAttr(isSync, ctxHandle, tpHandle, dscp));
            HCCL_INFO(
                "[TpManager][%s] UB_RTP dscp committed: tpHandle[%llu] dscpAfter[%u].", __func__, tpHandle,
                static_cast<unsigned>(dscp));
            return HcclResult::HCCL_SUCCESS;
        }
        if (tpAttr.dscpConfigMode == 1) {
            CHK_RET(CommitUboeNetAttrsToTpAttr(
                isSync, ctxHandle, tpHandle, tpAttr, param.locIpv4Addr, param.rmtIpv4Addr, false, 0U));
            return HcclResult::HCCL_SUCCESS;
        }

        uint8_t dscp = kUboeDefaultDscp;
        CHK_RET(GetDscpByQos(devPhyId, param.qos, dscp));
        CHK_RET(CommitUboeNetAttrsToTpAttr(
            isSync, ctxHandle, tpHandle, tpAttr, param.locIpv4Addr, param.rmtIpv4Addr, true, dscp));
        HCCL_INFO(
            "[TpManager][%s] UBOE net attrs updated: tpHandle[%llu] dscpAfter[%u].", __func__, tpHandle,
            static_cast<unsigned>(dscp));
        return HcclResult::HCCL_SUCCESS;
    }

} // namespace

TpManager& TpManager::GetInstance(const int32_t deviceLogicId)
{
    static TpManager tpManager[MAX_MODULE_DEVICE_NUM + 1];

    if (deviceLogicId < 0 || static_cast<uint32_t>(deviceLogicId) > MAX_MODULE_DEVICE_NUM) {
        THROW<InvalidParamsException>(
            "[TpManager][%s] failed to get instance, "
            "devLogicId[%d] should be less than %u.",
            __func__, deviceLogicId, MAX_MODULE_DEVICE_NUM);
    }

    tpManager[deviceLogicId].devLogicId = deviceLogicId;

    return tpManager[deviceLogicId];
}

void TpManager::Init()
{
    if (initFlag) {
        return;
    }

    devPhyId = HrtGetDevicePhyIdByUserDevId(devLogicId);
    initFlag = true;
}

bool TpManager::CheckRequestResult(RequestHandle& reqHandle) const
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
            "[TpManager][%s] failed, result[%s] is unexpected.", __func__, result.Describe().c_str());
    }

    return true;
}

HcclResult CheckTpProtocol(const TpProtocol tpProtocol)
{
    if (tpProtocol != TpProtocol::CTP && tpProtocol != TpProtocol::TP && tpProtocol != TpProtocol::UBOE
        && tpProtocol != TpProtocol::UB_RTP) {
        HCCL_WARNING("[TpManager][%s] failed, tpProtocol[%d] is not supported.", __func__, tpProtocol);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::AdvanceDeviceWaitListPhase(
    const RaUbGetTpInfoParam& param, RequestCtx& reqCtx, ReqQosMap& qosReqMap, ReqQosMap::iterator it,
    std::unique_lock<std::mutex>& reqCtxLock, TpInfo& tpInfo)
{
    if (reqCtx.tpInfoNum == 0U) {
        qosReqMap.erase(it);
        HCCL_ERROR(
            "[TpManager][%s] failed to find tp info, tpInfoNum is 0, param[%s].", __func__, param.Describe().c_str());
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    bool isPcieStd = false;
    CHK_RET(IsPcieStdMainboard(devLogicId, isPcieStd));
    if (isPcieStd) {
        const struct HccpTpInfo* list
            = static_cast<const struct HccpTpInfo*>(static_cast<const void*>(reqCtx.dataBuffer.data()));
        HCCL_INFO(
            "[TpManager][%s] pcie std mainboard: skip GetTpAttr, devPhyId[%u] tpInfoNum[%u] mappedSl[%u] "
            "tpHandle[%llu] param[%s].",
            __func__, devPhyId, reqCtx.tpInfoNum, kPcieStdMappedSl, static_cast<unsigned long long>(list[0].tpHandle),
            param.Describe().c_str());
        RequestCtx completedReqCtx = std::move(it->second);
        qosReqMap.erase(it);
        reqCtxLock.unlock();
        CHK_RET(HandleCompletedRequest(std::move(completedReqCtx), param, tpInfo, false));
        return HcclResult::HCCL_SUCCESS;
    }
    if (HrtRaSupportsGetTpAttr(devPhyId)) {
        const struct HccpTpInfo* list
            = static_cast<const struct HccpTpInfo*>(static_cast<const void*>(reqCtx.dataBuffer.data()));
        HCCL_INFO(
            "[TpManager][GetTpInfo] list stage ok, devPhyId[%u] tpInfoNum[%u] firstTpHandle[%llu] param[%s].", devPhyId,
            reqCtx.tpInfoNum, static_cast<unsigned long long>(list[0].tpHandle), param.Describe().c_str());
        TRY_CATCH_PROCESS_THROW(
            NetworkApiException, StartGetTpAttrForFirstTpDevice(param, reqCtx),
            "[TpManager][AdvanceDeviceWaitListPhase] StartGetTpAttrForFirstTpDevice failed", qosReqMap.erase(it));
        return HcclResult::HCCL_E_AGAIN;
    }
    RequestCtx completedReqCtx = std::move(it->second);
    qosReqMap.erase(it);
    reqCtxLock.unlock();
    CHK_RET(HandleCompletedRequest(std::move(completedReqCtx), param, tpInfo, false));
    return HcclResult::HCCL_SUCCESS;
}

// GetTpInfo 完成后写入缓存。useCnt 仅在 FindAndGetTpInfo 命中时 +1，此处不做引用计数。
// 并发首次 GetTpInfo 时，先完成者写入缓存；后完成者若 tpHandle 不同则跳过写入，直接使用本地结果。
HcclResult TpManager::StoreTpInfoResult(const RaUbGetTpInfoParam& param, TpInfo& tpInfo)
{
    const QosKey qosKey = QosMapKey(param.qos);
    std::lock_guard<std::mutex> lock(GetInfoCtxMutex(param.tpProtocol));
    auto& infoMap = GetInfoCtxMap(param.tpProtocol);
    auto& qosMap = infoMap[param.locAddr][param.rmtAddr];
    const auto qIt = qosMap.find(qosKey);
    if (qIt == qosMap.end()) {
        qosMap[qosKey] = TpInfoCtx{tpInfo, 1U};
        return HcclResult::HCCL_SUCCESS;
    }

    // 缓存已存在：不再覆盖（避免并发后写覆盖先写的 tpHandle）；tpInfo 保持 GetTpInfo 本地结果。
    if (qIt->second.tpInfo.tpHandle != tpInfo.tpHandle) {
        HCCL_WARNING(
            "[TpManager][%s] skip cache store, cached tpHandle[%llu] != local tpHandle[%llu] param[%s].", __func__,
            qIt->second.tpInfo.tpHandle, tpInfo.tpHandle, param.Describe().c_str());
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::SyncGetFirstTpAttrForSlPolicy(
    const RaUbGetTpInfoParam& param, uint64_t firstTpHandle, TpAttr& tpAttr, uint32_t& attrBitmap) const
{
    (void)memset_s(&tpAttr, sizeof(tpAttr), 0, sizeof(tpAttr));
    attrBitmap = BuildGetTpAttrBitmapForSlPolicy(param.tpProtocol);
    const RdmaHandle rdmaHandle = ResolveUbRdmaHandle(true, devPhyId, param.locAddr);
    if (!rdmaHandle) {
        HCCL_ERROR(
            "[TpManager][%s] can not find host rdmaHandle, devPhyId[%u] locAddr[%s].", __func__, devPhyId,
            param.locAddr.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }
    CHK_RET(HrtRaGetTpAttr(rdmaHandle, firstTpHandle, attrBitmap, tpAttr));
    HCCL_INFO(
        "[TpManager][%s] HrtRaGetTpAttr ok, tpHandle[%llu] attrBitmap[0x%x] param[%s].", __func__, firstTpHandle,
        attrBitmap, param.Describe().c_str());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::RunSyncGetTpInfo(const RaUbGetTpInfoParam& param, TpInfo& tpInfo)
{
    RequestCtx reqCtx{};
    StartGetTpInfoListRequest(param, reqCtx, true);

    if (reqCtx.tpInfoNum == 0U) {
        HCCL_ERROR(
            "[TpManager][%s] failed to find tp info, tpInfoNum is 0, param[%s].", __func__, param.Describe().c_str());
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    const struct HccpTpInfo* list
        = static_cast<const struct HccpTpInfo*>(static_cast<const void*>(reqCtx.dataBuffer.data()));
    HCCL_INFO(
        "[TpManager][%s] sync GetTpList ok, devPhyId[%u] tpInfoNum[%u] firstTpHandle[%llu] param[%s].", __func__,
        devPhyId, reqCtx.tpInfoNum, static_cast<unsigned long long>(list[0].tpHandle), param.Describe().c_str());

    tpInfo = TpInfo{};
    bool isPcieStd = false;
    CHK_RET(IsPcieStdMainboard(devLogicId, isPcieStd));
    if (isPcieStd) {
        tpInfo.tpHandle = list[0].tpHandle;
        tpInfo.mappedJettyPriority = kPcieStdMappedSl;
        tpInfo.hasMappedJettyPriority = true;
        HCCL_INFO(
            "[TpManager][%s] pcie std mainboard: skip GetTpAttr/SetTpAttr, mappedSl[%u] tpHandle[%llu] "
            "param[%s].",
            __func__, kPcieStdMappedSl, tpInfo.tpHandle, param.Describe().c_str());
    } else {
        CHK_RET(SyncGetFirstTpAttrForSlPolicy(param, list[0].tpHandle, reqCtx.tpAttr, reqCtx.tpAttrBitmap));
        CHK_RET(MapTpInfoFromTpAttr(param, reqCtx, tpInfo, true));
    }

    return StoreTpInfoResult(param, tpInfo);
}

HcclResult TpManager::RunAsyncGetTpInfo(const RaUbGetTpInfoParam& param, TpInfo& tpInfo)
{
    RequestCtx completedReqCtx{};
    bool withSlPolicy = false;

    {
        std::unique_lock<std::mutex> reqCtxLock(GetReqCtxMutex(param.tpProtocol));
        auto& reqCtxMap = GetReqCtxMap(param.tpProtocol);
        const auto& locAddr = param.locAddr;
        const auto& rmtAddr = param.rmtAddr;
        const QosKey qosKey = QosMapKey(param.qos);

        auto& rmtReqMap = reqCtxMap[locAddr];
        auto& qosReqMap = rmtReqMap[rmtAddr];
        auto it = qosReqMap.find(qosKey);
        if (it == qosReqMap.end()) {
            HCCL_INFO("[TpManager][%s] get new tpInfo, param[%s].", __func__, param.Describe().c_str());

            RequestCtx& reqCtx = qosReqMap[qosKey];
            StartGetTpInfoListRequest(param, reqCtx, false);
            return HcclResult::HCCL_E_AGAIN;
        }

        RequestCtx& reqCtx = it->second;

        if (reqCtx.handle != 0U && !CheckRequestResult(reqCtx.handle)) {
            return HcclResult::HCCL_E_AGAIN;
        }

        switch (reqCtx.phase) {
            case RequestCtx::ReqPhase::WAIT_LIST:
                return AdvanceDeviceWaitListPhase(param, reqCtx, qosReqMap, it, reqCtxLock, tpInfo);
            case RequestCtx::ReqPhase::WAIT_TP_ATTR:
                completedReqCtx = std::move(it->second);
                qosReqMap.erase(it);
                withSlPolicy = true;
                break;
            default:
                completedReqCtx = std::move(it->second);
                qosReqMap.erase(it);
                withSlPolicy = false;
                break;
        }
    }

    CHK_RET(HandleCompletedRequest(std::move(completedReqCtx), param, tpInfo, withSlPolicy));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::GetTpInfo(const RaUbGetTpInfoParam& param, TpInfo& tpInfo, bool isSync)
{
    CHK_RET(CheckTpProtocol(param.tpProtocol));
    if (FindAndGetTpInfo(param, tpInfo) == HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_SUCCESS;
    }

    if (isSync) {
        return RunSyncGetTpInfo(param, tpInfo);
    }
    return RunAsyncGetTpInfo(param, tpInfo);
}

HcclResult TpManager::GetOrSetLocalPsn(const RaUbGetTpInfoParam& param, const TpInfo& tpInfo, uint32_t& localPsn)
{
    CHK_RET(CheckTpProtocol(param.tpProtocol));
    const QosKey qosKey = QosMapKey(param.qos);
    std::lock_guard<std::mutex> lock(GetInfoCtxMutex(param.tpProtocol));
    auto& infoMap = GetInfoCtxMap(param.tpProtocol);
    const auto locIt = infoMap.find(param.locAddr);
    if (locIt == infoMap.end()) {
        HCCL_INFO(
            "[TpManager][%s] local PSN cache miss, locAddr not found, param[%s].", __func__, param.Describe().c_str());
        return HcclResult::HCCL_SUCCESS;
    }
    const auto rmtIt = locIt->second.find(param.rmtAddr);
    if (rmtIt == locIt->second.end()) {
        HCCL_INFO(
            "[TpManager][%s] local PSN cache miss, rmtAddr not found, param[%s].", __func__, param.Describe().c_str());
        return HcclResult::HCCL_SUCCESS;
    }
    const auto qosIt = rmtIt->second.find(qosKey);
    if (qosIt == rmtIt->second.end()) {
        HCCL_INFO(
            "[TpManager][%s] local PSN cache miss, qos not found, param[%s].", __func__, param.Describe().c_str());
        return HcclResult::HCCL_SUCCESS;
    }

    TpInfoCtx& infoCtx = qosIt->second;
    if (infoCtx.tpInfo.tpHandle != tpInfo.tpHandle) {
        // 并发首次申请可能返回不同 TP，StoreTpInfoResult 会保留先写入者、让后完成者使用自己的 TP。
        // 这种情况下不能错误复用缓存 TP 的 PSN，也不能把原有可用路径改成失败。
        HCCL_WARNING(
            "[TpManager][%s] skip PSN reuse, cached tpHandle[0x%llx] != current tpHandle[0x%llx], param[%s].", __func__,
            static_cast<unsigned long long>(infoCtx.tpInfo.tpHandle), static_cast<unsigned long long>(tpInfo.tpHandle),
            param.Describe().c_str());
        return HcclResult::HCCL_SUCCESS;
    }

    const bool reuse = infoCtx.hasLocalPsn;
    if (!reuse) {
        infoCtx.localPsn = localPsn;
        infoCtx.hasLocalPsn = true;
    }
    localPsn = infoCtx.localPsn;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::FindAndGetTpAttr(const TpHandle tpHandle, TpAttrInfo& tpAttrInfo)
{
    std::lock_guard<std::mutex> lock(tpAttrCtxMutex);
    auto attrIter = tpAttrCtxMap.find(tpHandle);
    if (attrIter != tpAttrCtxMap.end()) {
        attrIter->second.useCnt += 1;
        tpAttrInfo = attrIter->second.tpAttrInfo;
        HCCL_INFO(
            "[TpManager][%s] cache hit, tpHandle[0x%llx] useCnt[%u].", __func__,
            static_cast<unsigned long long>(tpHandle), attrIter->second.useCnt);
        return HcclResult::HCCL_SUCCESS;
    }

    return HcclResult::HCCL_E_NOT_FOUND;
}

HcclResult TpManager::GetTpAttr(const GetTpAttrParam& param, TpAttrInfo& tpAttrInfo, RdmaHandle rdmaHandle)
{
    const TpHandle tpHandle = param.tpHandle;
    if (FindAndGetTpAttr(tpHandle, tpAttrInfo) == HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_SUCCESS;
    }

    std::unique_lock<std::mutex> reqCtxLock(tpAttrReqMutex);
    auto reqCtxIter = tpAttrReqCtxMap.find(tpHandle);
    if (reqCtxIter == tpAttrReqCtxMap.end()) {
        HCCL_INFO("[TpManager][%s] get new tpAttr, param[%s].", __func__, param.Describe().c_str());

        TpAttrRequestCtx& reqCtx = tpAttrReqCtxMap[tpHandle];
        CHK_RET(StartGetTpAttrRequest(param, reqCtx, rdmaHandle));
        return HcclResult::HCCL_E_AGAIN;
    }

    auto& reqCtx = reqCtxIter->second;
    if (!CheckRequestResult(reqCtx.handle)) {
        return HcclResult::HCCL_E_AGAIN;
    }

    TpAttrRequestCtx completedReqCtx = reqCtxIter->second;
    tpAttrReqCtxMap.erase(reqCtxIter);
    reqCtxLock.unlock();
    CHK_RET(HandleCompletedTpAttrRequest(std::move(completedReqCtx), tpHandle, tpAttrInfo));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::StartGetTpAttrRequest(
    const GetTpAttrParam& param, TpManager::TpAttrRequestCtx& reqCtx, RdmaHandle rdmaHandle) const
{
    reqCtx.attrBitmap = param.attrBitmap;
    CHK_RET(HrtRaStartGetTpAttrAsync(rdmaHandle, param.tpHandle, reqCtx.attrBitmap, reqCtx.tpAttr, reqCtx.handle));
    HCCL_INFO("[TpManager][%s] success, tpHandle[0x%llx] reqHandle[%llu].", __func__, param.tpHandle, reqCtx.handle);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::HandleCompletedTpAttrRequest(
    const TpManager::TpAttrRequestCtx reqCtx, const TpHandle tpHandle, TpAttrInfo& tpAttrInfo)
{
    TpAttrInfo tmpTpAttrInfo(reqCtx.tpAttr);

    std::lock_guard<std::mutex> lock(tpAttrCtxMutex);
    tpAttrCtxMap[tpHandle] = {std::move(tmpTpAttrInfo), 1};

    tpAttrInfo = tpAttrCtxMap[tpHandle].tpAttrInfo;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::ReleaseTpInfo(const RaUbGetTpInfoParam& param, const TpInfo& tpInfo)
{
    const QosKey qosKey = QosMapKey(param.qos);
    std::lock_guard<std::mutex> lock(GetInfoCtxMutex(param.tpProtocol));
    auto& infoMap = GetInfoCtxMap(param.tpProtocol);
    auto lit = infoMap.find(param.locAddr);
    if (lit == infoMap.end()) {
        HCCL_ERROR("[TpManager][%s] failed, tp info is not found, param[%s].", __func__, param.Describe().c_str());
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    auto rit = lit->second.find(param.rmtAddr);
    if (rit == lit->second.end()) {
        HCCL_ERROR("[TpManager][%s] failed, tp info is not found, param[%s].", __func__, param.Describe().c_str());
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    auto qit = rit->second.find(qosKey);
    if (qit == rit->second.end()) {
        HCCL_ERROR(
            "[TpManager][%s] failed, tp info is not found for qosKey[%u], param[%s].", __func__,
            static_cast<unsigned>(qosKey), param.Describe().c_str());
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    // 未入缓存的并发 GetTpInfo 结果：与缓存 tpHandle 不一致，无需操作缓存。
    if (tpInfo.tpHandle != qit->second.tpInfo.tpHandle) {
        HCCL_INFO(
            "[TpManager][%s] skip, tpHandle mismatch, local[%llu] cached[%llu] locAddr[%s] rmtAddr[%s].", __func__,
            static_cast<unsigned long long>(tpInfo.tpHandle),
            static_cast<unsigned long long>(qit->second.tpInfo.tpHandle), param.locAddr.Describe().c_str(),
            param.rmtAddr.Describe().c_str());
        return HcclResult::HCCL_SUCCESS;
    }

    if (qit->second.useCnt > 1) {
        qit->second.useCnt -= 1;
        HCCL_INFO(
            "[TpManager][%s] ref decrement, useCnt[%u -> %u] tpHandle[%llu] locAddr[%s] rmtAddr[%s].", __func__,
            qit->second.useCnt + 1U, qit->second.useCnt, static_cast<unsigned long long>(tpInfo.tpHandle),
            param.locAddr.Describe().c_str(), param.rmtAddr.Describe().c_str());
        return HcclResult::HCCL_SUCCESS;
    }

    HCCL_INFO(
        "[TpManager][%s] last ref, erase cache entry, useCnt[%u] tpHandle[%llu] locAddr[%s] rmtAddr[%s].", __func__,
        qit->second.useCnt, static_cast<unsigned long long>(tpInfo.tpHandle), param.locAddr.Describe().c_str(),
        param.rmtAddr.Describe().c_str());
    rit->second.erase(qit);
    if (rit->second.empty()) {
        lit->second.erase(rit);
    }
    if (lit->second.empty()) {
        infoMap.erase(lit);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::ReleaseTpAttr(const TpHandle tpHandle, const TpAttrInfo& tpAttrInfo)
{
    (void)tpAttrInfo;
    std::lock_guard<std::mutex> lock(tpAttrCtxMutex);
    auto attrIter = tpAttrCtxMap.find(tpHandle);
    if (attrIter == tpAttrCtxMap.end()) {
        HCCL_ERROR(
            "[TpManager][%s] failed, tp attr is not found, "
            "tpHandle[0x%llx].",
            __func__, tpHandle);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    if (attrIter->second.useCnt > 1) {
        attrIter->second.useCnt -= 1;
        HCCL_INFO(
            "[TpManager][%s] ref decrement, useCnt[%u -> %u] tpHandle[0x%llx].", __func__, attrIter->second.useCnt + 1U,
            attrIter->second.useCnt, static_cast<unsigned long long>(tpHandle));
        return HcclResult::HCCL_SUCCESS;
    }

    HCCL_INFO(
        "[TpManager][%s] last ref, erase cache entry, useCnt[%u] tpHandle[0x%llx].", __func__, attrIter->second.useCnt,
        static_cast<unsigned long long>(tpHandle));
    tpAttrCtxMap.erase(attrIter);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::GetTpTotalTimeout(const TpAttrInfo& tpAttrInfo, uint32_t& tpTimeOutMs)
{
    uint8_t rawAtGear = tpAttrInfo.tpAttr.at;
    uint8_t rawRetryTimes = tpAttrInfo.tpAttr.retryTimesInit;

    uint8_t finalAtGear = rawAtGear;
    if (rawAtGear > AT_GEAR_MAX) {
        finalAtGear = AT_GEAR_DEFAULT;
        HCCL_WARNING(
            "[TpManager][%s] Invalid at gear[%u], expect [%u, %u], use default gear[%u].", __func__, rawAtGear,
            AT_GEAR_MIN, AT_GEAR_MAX, finalAtGear);
    }

    uint32_t singleAtTimeoutMs = AT_TIMEOUT_MAP[finalAtGear];
    tpTimeOutMs = singleAtTimeoutMs * static_cast<uint32_t>(rawRetryTimes + 1);

    HCCL_INFO(
        "[TpManager][%s] TP timeout calc success: raw_at_gear[%u], final_at_gear[%u], "
        "single_timeout[%ums], retry_times[%u], total_timeout[%ums].",
        __func__, rawAtGear, finalAtGear, singleAtTimeoutMs, rawRetryTimes, tpTimeOutMs);

    return HcclResult::HCCL_SUCCESS;
}

uint32_t TpManager::TaHwValueToMs(uint8_t hwValue)
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

uint8_t TpManager::FindMinTaHwValue(uint32_t tpTotalTimeoutMs)
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

uint8_t TpManager::CalcTaTimeout(TpProtocol tpProtocol, uint8_t taTimeOut, uint32_t tpTimeOutMs)
{
    // 调用方（base_comm 通道）按协议从环境变量获取后传入；未传入时回退到协议默认值（CTP=8，其他=16）
    uint8_t envValue = (taTimeOut != TA_TIMEOUT_NOT_SET) ? taTimeOut : ((tpProtocol == TpProtocol::CTP) ? 8 : 16);
    uint32_t envTimeOutMs = TaHwValueToMs(envValue);

    // CTP 协议直接使用 envValue，不与 TP 总超时比较（对齐原有 GetTimeOut 逻辑）
    if (tpProtocol == TpProtocol::CTP) {
        HCCL_INFO("[TpManager][%s] [UbCtp] Env Value [%u] (%ums).", __func__, envValue, envTimeOutMs);
        return envValue;
    }

    // UBOE / UBG / TP 协议：需要与 TP 总超时比较决定是否自动升挡
    uint8_t jettyTimeOut = envValue;
    if (envTimeOutMs <= tpTimeOutMs) {
        // 规则: 如果环境变量时间 <= TP总超时，选择大于TP总超时的最小TA挡位
        jettyTimeOut = FindMinTaHwValue(tpTimeOutMs);
        HCCL_WARNING(
            "[TpManager][%s] Env timeout [%ums] <= TP timeout [%ums]. Auto upgrade TA to hw_val[%u] (%ums).", __func__,
            envTimeOutMs, tpTimeOutMs, jettyTimeOut, TaHwValueToMs(jettyTimeOut));
    } else {
        // 规则: 否则，直接使用环境变量对应的挡位
        HCCL_INFO(
            "[TpManager][%s] Env timeout [%ums] > TP timeout [%ums]. Use env gear base hw_val[%u] (%ums).", __func__,
            envTimeOutMs, tpTimeOutMs, envValue, envTimeOutMs);
    }

    return jettyTimeOut;
}

HcclResult TpManager::FindAndGetTpInfo(const RaUbGetTpInfoParam& param, TpInfo& tpInfo)
{
    const QosKey qosKey = QosMapKey(param.qos);
    std::lock_guard<std::mutex> lock(GetInfoCtxMutex(param.tpProtocol));
    auto& infoMap = GetInfoCtxMap(param.tpProtocol);
    auto lit = infoMap.find(param.locAddr);
    if (lit == infoMap.end()) {
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    auto rit = lit->second.find(param.rmtAddr);
    if (rit == lit->second.end()) {
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    auto qit = rit->second.find(qosKey);
    if (qit == rit->second.end()) {
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    // 复用缓存：useCnt 仅在此处（命中）递增，与 StoreTpInfoResult 写入路径分离。
    qit->second.useCnt += 1;
    tpInfo = qit->second.tpInfo;
    HCCL_INFO(
        "[TpManager][%s] cache hit, tpHandle[%llu] mappedJettyPriority[%u] hasMappedJettyPriority[%d] "
        "useCnt[%u] param[%s].",
        __func__, static_cast<unsigned long long>(tpInfo.tpHandle), tpInfo.mappedJettyPriority,
        static_cast<int>(tpInfo.hasMappedJettyPriority), qit->second.useCnt, param.Describe().c_str());
    return HcclResult::HCCL_SUCCESS;
}

void TpManager::StartGetTpInfoListRequest(
    const RaUbGetTpInfoParam& param, TpManager::RequestCtx& reqCtx, bool isSync) const
{
    reqCtx.phase = RequestCtx::ReqPhase::WAIT_LIST;
    (void)memset_s(&reqCtx.tpAttr, sizeof(reqCtx.tpAttr), 0, sizeof(reqCtx.tpAttr));
    reqCtx.tpAttrBitmap = 0;

    Hccl::IpAddress localIp = param.locAddr;
    const RdmaHandle rdmaHandle = ResolveUbRdmaHandle(isSync, devPhyId, localIp);
    if (!rdmaHandle) {
        THROW<InternalException>(
            "[TpManager][%s] can not find rdmaHandle, "
            "devPhyId[%u] locAddr[%s].",
            __func__, devPhyId, param.locAddr.Describe().c_str());
    }
    if (isSync) {
        RaUbGetTpInfo(rdmaHandle, param, reqCtx.dataBuffer, reqCtx.tpInfoNum);
        return;
    }
    reqCtx.handle = RaUbGetTpInfoAsync(rdmaHandle, param, reqCtx.dataBuffer, reqCtx.tpInfoNum);
}

void TpManager::StartGetTpAttrForFirstTpDevice(const RaUbGetTpInfoParam& param, RequestCtx& reqCtx) const
{
    (void)memset_s(&reqCtx.tpAttr, sizeof(reqCtx.tpAttr), 0, sizeof(reqCtx.tpAttr));
    reqCtx.tpAttrBitmap = BuildGetTpAttrBitmapForSlPolicy(param.tpProtocol);
    const struct HccpTpInfo* list
        = static_cast<const struct HccpTpInfo*>(static_cast<const void*>(reqCtx.dataBuffer.data()));
    const uint64_t firstTpHandle = list[0].tpHandle;
    const RdmaHandle rdmaHandle = ResolveUbRdmaHandle(false, devPhyId, param.locAddr);
    if (!rdmaHandle) {
        THROW<InternalException>(
            "[TpManager][%s] can not find rdmaHandle for GetTpAttrAsync, devPhyId[%u].", __func__, devPhyId);
    }
    const HcclResult hret
        = HrtRaStartGetTpAttrAsync(rdmaHandle, firstTpHandle, reqCtx.tpAttrBitmap, reqCtx.tpAttr, reqCtx.handle);
    if (hret != HcclResult::HCCL_SUCCESS) {
        THROW<NetworkApiException>(StringFormat(
            "[TpManager][StartGetTpAttrForFirstTpDevice] HrtRaStartGetTpAttrAsync "
            "failed hcclRet[%d] tpHandle[%llu].",
            static_cast<int>(hret), firstTpHandle));
    }
    reqCtx.phase = RequestCtx::ReqPhase::WAIT_TP_ATTR;
}

HcclResult TpManager::MapTpInfoFromTpAttr(
    const RaUbGetTpInfoParam& param, const RequestCtx& reqCtx, TpInfo& outTpInfo, bool isSync) const
{
    const uint32_t tpInfoNum = reqCtx.tpInfoNum;
    const struct HccpTpInfo* baseInfoPtr
        = static_cast<const struct HccpTpInfo*>(static_cast<const void*>(reqCtx.dataBuffer.data()));
    const uint16_t slMask = ReadSlAvailableMask16(reqCtx.tpAttr);
    const uint32_t slAvailableCnt = CalSlAvailableCnt(slMask);
    HCCL_INFO(
        "[TpManager][%s] after get_tp_attr: slMask[0x%04x] slAvailableCnt[%u] slBitmap[0x%x] dscp[%u] "
        "dscpConfigMode[%u] tpAttrBitmap[0x%x] param[%s].",
        __func__, static_cast<unsigned>(slMask), slAvailableCnt, static_cast<unsigned>(reqCtx.tpAttr.slBitmap),
        static_cast<unsigned>(reqCtx.tpAttr.dscp & 0x3FU), static_cast<unsigned>(reqCtx.tpAttr.dscpConfigMode & 1U),
        reqCtx.tpAttrBitmap, param.Describe().c_str());
    if (slAvailableCnt == 0U) {
        HCCL_ERROR(
            "[TpManager][%s] sl_available mask empty after get_tp_attr, param[%s].", __func__,
            param.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint32_t tpListIndex = 0;
    uint32_t mappedSl = 0;
    if (!ApplyQosTpSlPolicy(param, slMask, tpListIndex, mappedSl)) {
        HCCL_ERROR(
            "[TpManager][%s] ApplyQosTpSlPolicy failed, param[%s] nTp[%u] slAvailableCnt[%u] mask[%u].", __func__,
            param.Describe().c_str(), tpInfoNum, slAvailableCnt, static_cast<unsigned>(slMask));
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (tpListIndex >= tpInfoNum) {
        HCCL_ERROR(
            "[TpManager][%s] tpListIndex out of range: tpListIndex[%u] tpInfoNum[%u] mappedSl[%u] param[%s].", __func__,
            tpListIndex, tpInfoNum, static_cast<unsigned>(mappedSl & 0xFU), param.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }

    outTpInfo.tpHandle = baseInfoPtr[tpListIndex].tpHandle;
    outTpInfo.mappedJettyPriority = mappedSl & 0xFU;
    outTpInfo.hasMappedJettyPriority = true;

    CHK_RET(CommitTpAttrsAfterSlMapping(
        devLogicId, devPhyId, isSync, param, reqCtx.tpAttr, outTpInfo.tpHandle, mappedSl, tpInfoNum, slMask));

    HCCL_INFO(
        "[TpManager][%s] tp qos mapping ok: tpInfoNum[%u] tpHandle[%llu] tpListIndex[%u] "
        "mappedJettyPriority[%u] qos[%u] param[%s].",
        __func__, tpInfoNum, outTpInfo.tpHandle, tpListIndex, outTpInfo.mappedJettyPriority, param.qos & 0xFFU,
        param.Describe().c_str());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::HandleCompletedRequest(
    const TpManager::RequestCtx reqCtx, const RaUbGetTpInfoParam& param, TpInfo& tpInfo, bool withSlPolicy)
{
    const uint32_t tpInfoNum = reqCtx.tpInfoNum;
    if (tpInfoNum == 0) {
        HCCL_ERROR(
            "[TpManager][%s] failed to find tp info, tpInfoNum is 0, "
            "param[%s].",
            __func__, param.Describe().c_str());
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    tpInfo = TpInfo{};

    HCCL_INFO(
        "[TpManager][%s] RaGetTpInfoList completed: tpInfoNum[%u] withSlPolicy[%d] devPhyId[%u] param[%s].", __func__,
        tpInfoNum, static_cast<int>(withSlPolicy), devPhyId, param.Describe().c_str());

    bool isPcieStd = false;
    CHK_RET(IsPcieStdMainboard(devLogicId, isPcieStd));
    if (isPcieStd) {
        const struct HccpTpInfo* baseInfoPtr
            = static_cast<const struct HccpTpInfo*>(static_cast<const void*>(reqCtx.dataBuffer.data()));
        tpInfo.tpHandle = baseInfoPtr[0].tpHandle;
        tpInfo.mappedJettyPriority = kPcieStdMappedSl;
        tpInfo.hasMappedJettyPriority = true;
        HCCL_INFO(
            "[TpManager][%s] pcie std mainboard: skip GetTpAttr/SetTpAttr, devPhyId[%u] tpInfoNum[%u] "
            "mappedSl[%u] tpHandle[%llu] param[%s].",
            __func__, devPhyId, tpInfoNum, kPcieStdMappedSl, tpInfo.tpHandle, param.Describe().c_str());
    } else if (withSlPolicy) {
        CHK_RET(MapTpInfoFromTpAttr(param, reqCtx, tpInfo, false));
    } else {
        const struct HccpTpInfo* baseInfoPtr
            = static_cast<const struct HccpTpInfo*>(static_cast<const void*>(reqCtx.dataBuffer.data()));
        tpInfo.tpHandle = baseInfoPtr[0].tpHandle;
        tpInfo.hasMappedJettyPriority = false;
    }

    return StoreTpInfoResult(param, tpInfo);
}

TpManager::InfoCtxMap& TpManager::GetInfoCtxMap(const TpProtocol tpProtocol)
{
    switch (tpProtocol) {
        case TpProtocol::CTP:
            return ctpInfoMap;
        case TpProtocol::TP:
            return tpInfoMap;
        case TpProtocol::UBOE:
            return uboeInfoMap;
        case TpProtocol::UB_RTP:
            return ubRtpInfoMap;
        default:
            return tpInfoMap;
    }
}

TpManager::ReqCtxMap& TpManager::GetReqCtxMap(const TpProtocol tpProtocol)
{
    switch (tpProtocol) {
        case TpProtocol::CTP:
            return ctpReqMap;
        case TpProtocol::TP:
            return tpReqMap;
        case TpProtocol::UBOE:
            return uboeReqMap;
        case TpProtocol::UB_RTP:
            return ubRtpReqMap;
        default:
            return tpReqMap;
    }
}

std::mutex& TpManager::GetInfoCtxMutex(const TpProtocol tpProtocol)
{
    switch (tpProtocol) {
        case TpProtocol::CTP:
            return ctpInfoMutex;
        case TpProtocol::TP:
            return tpInfoMutex;
        case TpProtocol::UBOE:
            return uboeInfoMutex;
        case TpProtocol::UB_RTP:
            return ubRtpInfoMutex;
        default:
            return tpInfoMutex;
    }
}

std::mutex& TpManager::GetReqCtxMutex(const TpProtocol tpProtocol)
{
    switch (tpProtocol) {
        case TpProtocol::CTP:
            return ctpReqMutex;
        case TpProtocol::TP:
            return tpReqMutex;
        case TpProtocol::UBOE:
            return uboeReqMutex;
        case TpProtocol::UB_RTP:
            return ubRtpReqMutex;
        default:
            return tpReqMutex;
    }
}

void ReleaseUbConnectionTp(
    int32_t devLogicId, const IpAddress& locAddr, const IpAddress& rmtAddr, TpProtocol tpProtocol, TpInfo& tpInfo,
    uint32_t requestQos)
{
    if (tpInfo.tpHandle == 0) {
        HCCL_WARNING(
            "[TpManager][%s] skip release, tpHandle is 0, devLogicId[%d] loc[%s] rmt[%s] tpProtocol[%s] "
            "qos[%u].",
            __func__, devLogicId, locAddr.Describe().c_str(), rmtAddr.Describe().c_str(), tpProtocol.Describe().c_str(),
            requestQos);
        return;
    }
    RaUbGetTpInfoParam relParam(locAddr, rmtAddr, tpProtocol);
    relParam.qos = requestQos;
    (void)TpManager::GetInstance(devLogicId).ReleaseTpInfo(relParam, tpInfo);
    tpInfo.tpHandle = 0;
}

} // namespace Hccl
