/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel_config.h"
#include "log.h"
#include "exception_handler.h"

namespace hcomm {

HcommResult ChannelConfigCreate(HcommChannelConfig* config)
{
    CHK_PTR_NULL(config);
    NEW_NOTHROW(*config, HcommChannelConfigData(), return HCCL_E_PTR);
    return HCCL_SUCCESS;
}

HcommResult ChannelConfigDestroy(HcommChannelConfig config)
{
    CHK_PTR_NULL(config);
    auto* cfg = static_cast<HcommChannelConfigData*>(config);
    if (cfg->destroyed_.exchange(true)) {
        HCCL_WARNING("[%s] config[%p] is already destroyed, skip to avoid double-free.", __func__, config);
        return HCCL_SUCCESS;
    }
    delete cfg;
    return HCCL_SUCCESS;
}

HcommResult ChannelConfigSetInt(HcommChannelConfig config, HcommChannelConfigType type, uint32_t value)
{
    CHK_PTR_NULL(config);
    auto* cfg = static_cast<HcommChannelConfigData*>(config);
    switch (type) {
        case HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE:
            cfg->isSharedQueue = (value != 0);
            HCCL_INFO("[%s] set IS_SHARED_QUEUE=%d.", __func__, static_cast<int>(cfg->isSharedQueue));
            break;
        default:
            HCCL_ERROR("[%s] invalid int config type[%d].", __func__, static_cast<int>(type));
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

} // namespace hcomm
