/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_channel_config.h"
#include "hccl/hccl_channel.h"
#include "log.h"
#include "exception_handler.h"

HcclResult HcclChannelConfigCreate(HcclChannelConfig* config)
{
    CHK_PTR_NULL(config);
    NEW_NOTHROW(*config, hccl::HcclChannelConfigData(), return HCCL_E_PTR);
    return HCCL_SUCCESS;
}

HcclResult HcclChannelConfigDestroy(HcclChannelConfig config)
{
    CHK_PTR_NULL(config);
    auto* cfg = static_cast<hccl::HcclChannelConfigData*>(config);
    if (cfg->destroyed_.exchange(true)) {
        HCCL_WARNING("[%s] config[%p] is already destroyed, skip to avoid double-free.", __func__, config);
        return HCCL_SUCCESS;
    }
    delete cfg;
    return HCCL_SUCCESS;
}

HcclResult HcclChannelConfigSetInt(HcclChannelConfig config, HcclChannelConfigType type, uint32_t value)
{
    CHK_PTR_NULL(config);
    auto* cfg = static_cast<hccl::HcclChannelConfigData*>(config);
    switch (type) {
        case HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE:
            cfg->isSharedQueue = (value != 0);
            HCCL_INFO("[%s] set IS_SHARED_QUEUE=%d.", __func__, static_cast<int>(cfg->isSharedQueue));
            break;
        default:
            HCCL_ERROR("[%s] invalid int config type[%d].", __func__, static_cast<int>(type));
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclChannelConfigSetStr(HcclChannelConfig config, HcclChannelConfigType type, const char* value)
{
    CHK_PTR_NULL(config);
    CHK_PTR_NULL(value);
    auto* cfg = static_cast<hccl::HcclChannelConfigData*>(config);
    switch (type) {
        case HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG:
            CHK_PRT_RET(
                strlen(value) == 0 || strlen(value) > HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN,
                HCCL_ERROR(
                    "[%s] SHARED_QUEUE_TAG length[%zu] must be in (0, %u].", __func__, strlen(value),
                    HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN),
                HCCL_E_PARA);
            cfg->sharedQueueTag = std::string(value);
            HCCL_INFO("[%s] set SHARED_QUEUE_TAG=%s.", __func__, cfg->sharedQueueTag.c_str());
            break;
        default:
            HCCL_ERROR("[%s] invalid str config type[%d].", __func__, static_cast<int>(type));
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
