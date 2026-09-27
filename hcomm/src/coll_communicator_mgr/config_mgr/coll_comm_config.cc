/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_config.h"
#include <cstring> // 包含strnlen函数

namespace hccl {
constexpr uint32_t MULTIPLE = 4;             // 用于A5判断TC是否为4的倍数
constexpr uint32_t TC_MAX = 255;             // TC的最大值（不区分芯片类型）
constexpr uint32_t SL_MAX = 7u;              // sl范围的最大值，sl即serviceLevel（不区分芯片类型）
constexpr uint32_t TC_DEFAULT = 0xFFFFFFFFu; // TC的默认值（不区分芯片类型）
constexpr uint32_t SL_DEFAULT = 0xFFFFFFFFu; // SL的默认值（不区分芯片类型）
constexpr uint32_t HCCL_COMM_CONFIG_QOS_VERSION = 10U;

static HcclResult GetHcclCommConfigVersion(const HcclCommConfig* config, uint32_t& version)
{
    CHK_PTR_NULL(config);

    CommConfigInfo info{};
    s32 sRet = memcpy_s(&info, sizeof(info), config->reserved, sizeof(info));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[GetHcclCommConfigVersion] memcpy_s failed, errNo[%d]", sRet), HCCL_E_MEMORY);
    version = info.version;
    return HCCL_SUCCESS;
}

static HcclResult ApplyHcclQos(const HcclCommConfig* hcclCommConfig, CommConfig& commConfig)
{
    if (hcclCommConfig == nullptr) {
        return HCCL_SUCCESS;
    }

    // hcclQos 自 CommConfig version 10 起引入；低版本按未配置处理
    uint32_t configVersion = 0U;
    CHK_RET(GetHcclCommConfigVersion(hcclCommConfig, configVersion));
    if (configVersion < HCCL_COMM_CONFIG_QOS_VERSION) {
        HCCL_INFO(
            "[ApplyHcclQos] skip hcclQos by version, configVersion[%u] < HCCL_COMM_CONFIG_QOS_VERSION[%u]",
            configVersion, HCCL_COMM_CONFIG_QOS_VERSION);
        CHK_RET(commConfig.SetConfigHcclQos(HCCL_COMM_QOS_CONFIG_NOT_SET));
        return HCCL_SUCCESS;
    }

    u32 qos = hcclCommConfig->hcclQos;
    CHK_PRT_RET(
        (qos != HCCL_COMM_QOS_CONFIG_NOT_SET) && (qos > 7u),
        HCCL_ERROR(
            "[ApplyHcclQos]errNo[0x%016llx] invalid hcclQos[%u], must be 0xFFFFFFFF or in [0,7]",
            HCCL_ERROR_CODE(HCCL_E_PARA), qos),
        HCCL_E_PARA);
    CHK_RET(commConfig.SetConfigHcclQos(qos));
    HCCL_INFO("[ApplyHcclQos] hcclQos[%u]", qos);
    return HCCL_SUCCESS;
}

static HcclResult ApplyHcclSqDepth(const HcclCommConfig* hcclCommConfig, CommConfig& commConfig)
{
    if (hcclCommConfig == nullptr) {
        return HCCL_SUCCESS;
    }

    uint32_t configVersion = 0U;
    CHK_RET(GetHcclCommConfigVersion(hcclCommConfig, configVersion));
    if (configVersion < HCCL_COMM_CONFIG_SQ_DEPTH_VERSION) {
        HCCL_INFO(
            "[ApplyHcclSqDepth] skip hcclSqDepth by version, configVersion[%u] < "
            "HCCL_COMM_CONFIG_SQ_DEPTH_VERSION[%u]",
            configVersion, HCCL_COMM_CONFIG_SQ_DEPTH_VERSION);
        return commConfig.SetConfigSqDepth(HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET);
    }

    const uint32_t sqDepth = hcclCommConfig->hcclChannelSqDepth;
    CHK_RET(commConfig.SetConfigSqDepth(sqDepth));
    HCCL_INFO("[ApplyHcclSqDepth] hcclSqDepth[%u]", sqDepth);
    return HCCL_SUCCESS;
}

static HcclResult ApplyTrafficClassAndServiceLevel(const HcclCommConfig* hcclCommConfig, CommConfig& commConfig)
{
    if (hcclCommConfig == nullptr) {
        return HCCL_SUCCESS;
    }

    u32 tc = hcclCommConfig->hcclRdmaTrafficClass;
    CHK_PRT_RET(
        (tc != TC_DEFAULT) && (tc > TC_MAX || (tc % MULTIPLE != 0)),
        HCCL_ERROR(
            "[ApplyTrafficClassAndServiceLevel]errNo[0x%016llx] invalid hcclRdmaTrafficClass[%u], "
            "must be 0xFFFFFFFF or in [0,255] and a multiple of 4",
            HCCL_ERROR_CODE(HCCL_E_PARA), tc),
        HCCL_E_PARA);
    CHK_RET(commConfig.SetConfigTrafficClass(tc));

    u32 sl = hcclCommConfig->hcclRdmaServiceLevel;
    CHK_PRT_RET(
        (sl != SL_DEFAULT) && (sl > SL_MAX),
        HCCL_ERROR(
            "[ApplyTrafficClassAndServiceLevel]errNo[0x%016llx] invalid hcclRdmaServiceLevel[%u], "
            "must be 0xFFFFFFFF or in [0,7]",
            HCCL_ERROR_CODE(HCCL_E_PARA), sl),
        HCCL_E_PARA);
    CHK_RET(commConfig.SetConfigServiceLevel(sl));
    return HCCL_SUCCESS;
}

HcclResult ApplyHcclCommConfig(const HcclCommConfig* hcclCommConfig, CommConfig& commConfig, uint32_t& opExpansionMode)
{
    opExpansionMode = 0;
    if (hcclCommConfig == nullptr) {
        return HCCL_SUCCESS;
    }

    opExpansionMode = hcclCommConfig->hcclOpExpansionMode;
    CHK_RET(ApplyTrafficClassAndServiceLevel(hcclCommConfig, commConfig));
    CHK_RET(ApplyHcclQos(hcclCommConfig, commConfig));

    if (hcclCommConfig->hcclAlgo[0] != '\0') {
        size_t algoLen = strnlen(hcclCommConfig->hcclAlgo, static_cast<size_t>(HCCL_COMM_ALGO_MAX_LENGTH));
        CHK_RET(commConfig.SetConfigHcclAlgoStr(std::string(hcclCommConfig->hcclAlgo, algoLen)));
    }
    CHK_RET(ApplyHcclSqDepth(hcclCommConfig, commConfig));
    return HCCL_SUCCESS;
}
} // namespace hccl
