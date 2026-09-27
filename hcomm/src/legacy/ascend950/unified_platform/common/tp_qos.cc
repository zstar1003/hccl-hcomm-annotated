/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tp_qos.h"

#include <cctype>
#include <cerrno>
#include <algorithm>
#include <string>
#include <vector>

#include "log.h"
#include "hccp.h"

#ifndef ENOTSUPP
#define ENOTSUPP ENOTSUP
#endif

namespace Hccl {

namespace {

    static constexpr uint32_t slPolicy2 = 2U;
    static constexpr uint32_t slPolicy3 = 3U;
    static constexpr uint32_t slPolicy4 = 4U;
    static constexpr uint32_t slPolicy5 = 5U;
    static constexpr uint32_t slPolicy7 = 7U;
    static constexpr uint32_t slPolicy8 = 8U;
    static constexpr uint32_t slGroupIdx0 = 0U;
    static constexpr uint32_t slGroupIdx1 = 1U;
    static constexpr uint32_t slGroupIdx2 = 2U;

    constexpr size_t kMaxQosDscpPairs = 8U;
    constexpr uint32_t kDecimalBase = 10U;
    constexpr uint32_t kQosMax = 7U; // 与通道/通信域 hcclQos 取值 [0, 7] 对齐
    constexpr uint32_t kDscpMax = 255U;
    constexpr unsigned int kHccnCfgValueBufLen = 2048U;

    static bool ParseUint32Field(const std::string& cfg, size_t& pos, uint32_t& out)
    {
        if (pos >= cfg.size() || std::isdigit(static_cast<unsigned char>(cfg[pos])) == 0) {
            return false;
        }
        uint32_t val = 0U;
        while (pos < cfg.size() && std::isdigit(static_cast<unsigned char>(cfg[pos])) != 0) {
            val = val * kDecimalBase + static_cast<uint32_t>(cfg[pos] - '0');
            ++pos;
        }
        out = val;
        return true;
    }

    // HCCN cfg value format: "qos:dscp,qos:dscp,..." (e.g. "0:33,1:65"), at most 8 pairs.
    // SUCCESS: 命中 qos；NOT_FOUND: 串合法但未配置该 qos；PARA: 格式非法。
    static HcclResult ParseDscpByQos(const std::string& cfg, uint8_t qos, uint8_t& dscp)
    {
        size_t pos = 0U;
        for (size_t pairIdx = 0U; pairIdx < kMaxQosDscpPairs; ++pairIdx) {
            uint32_t cfgQos = 0U;
            uint32_t cfgDscp = 0U;
            if (!ParseUint32Field(cfg, pos, cfgQos)) {
                return HcclResult::HCCL_E_PARA;
            }
            if (pos >= cfg.size() || cfg[pos] != ':') {
                return HcclResult::HCCL_E_PARA;
            }
            ++pos;
            if (!ParseUint32Field(cfg, pos, cfgDscp)) {
                return HcclResult::HCCL_E_PARA;
            }
            if (cfgQos > kQosMax || cfgDscp > kDscpMax) {
                return HcclResult::HCCL_E_PARA;
            }

            if (static_cast<uint8_t>(cfgQos) == qos) {
                dscp = static_cast<uint8_t>(cfgDscp);
                return HcclResult::HCCL_SUCCESS;
            }

            if (pos >= cfg.size()) {
                return HcclResult::HCCL_E_NOT_FOUND;
            }
            if (cfg[pos] != ',') {
                return HcclResult::HCCL_E_PARA;
            }
            ++pos;
        }
        return HcclResult::HCCL_E_NOT_FOUND;
    }

} // namespace

uint32_t TpQosResolveQosSlGroupIdx(const uint32_t qos, const uint32_t numGroups)
{
    if (numGroups == slPolicy3) {
        if (qos < slPolicy3) {
            return slGroupIdx0;
        }
        if (qos < slPolicy5) {
            return slGroupIdx1;
        }
        return slGroupIdx2;
    }
    if (numGroups >= slPolicy4 && numGroups <= slPolicy7) {
        return qos / slPolicy2;
    }
    return (qos * numGroups) / slPolicy8;
}

HcclResult GetDscpByQos(const uint32_t devPhyId, uint8_t qos, uint8_t& dscp, NetworkMode networkMode)
{
    struct RaInfo info {};
    info.mode = static_cast<int>(networkMode);
    info.phyId = devPhyId;

    std::vector<char> value(kHccnCfgValueBufLen, 0);
    unsigned int valueLen = kHccnCfgValueBufLen;
    const int ret = RaGetHccnCfg(&info, HCCN_CFG_QOS_DSCP, value.data(), &valueLen);
    unsigned int logLen = valueLen;
    if (logLen > kHccnCfgValueBufLen) {
        logLen = kHccnCfgValueBufLen;
    }
    const std::string cfgLog(value.data(), logLen);
    HCCL_INFO(
        "[%s] RaGetHccnCfg ret[%d] phyId[%u] mode[%d] valueLen[%u] qos_dscp[%s].", __func__, ret, devPhyId,
        static_cast<int>(networkMode), valueLen, cfgLog.c_str());
    if (ret != 0) {
        HCCL_ERROR(
            "[%s] RaGetHccnCfg failed, ret[%d] phyId[%u] mode[%d] valueLen[%u].", __func__, ret, devPhyId,
            static_cast<int>(networkMode), valueLen);
        if (ret == -EINVAL) {
            return HcclResult::HCCL_E_PARA;
        }
        if (ret == -ENOTSUPP) {
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (valueLen == 0U) {
        dscp = kUboeDefaultDscp;
        HCCL_RUN_WARNING(
            "[%s] qos_dscp empty, phyId[%u] mode[%d] qos[%u], use default dscp[%u].", __func__, devPhyId,
            static_cast<int>(networkMode), static_cast<unsigned>(qos), static_cast<unsigned>(dscp));
        return HcclResult::HCCL_SUCCESS;
    }

    unsigned int cfgLen = std::min(valueLen, kHccnCfgValueBufLen);
    if (cfgLen > 0U && value[cfgLen - 1U] == '\0') {
        --cfgLen;
    }
    const std::string cfg(value.data(), cfgLen);
    const HcclResult parseRet = ParseDscpByQos(cfg, qos, dscp);
    if (parseRet == HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_SUCCESS;
    }
    if (parseRet == HcclResult::HCCL_E_NOT_FOUND) {
        dscp = kUboeDefaultDscp;
        HCCL_RUN_WARNING(
            "[%s] qos[%u] not found in qos_dscp, phyId[%u] mode[%d], use default dscp[%u].", __func__,
            static_cast<unsigned>(qos), devPhyId, static_cast<int>(networkMode), static_cast<unsigned>(dscp));
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_ERROR(
        "[%s] invalid qos_dscp format, phyId[%u] mode[%d] qos[%u] cfg[%s].", __func__, devPhyId,
        static_cast<int>(networkMode), static_cast<unsigned>(qos), cfg.c_str());
    return parseRet;
}

} // namespace Hccl
