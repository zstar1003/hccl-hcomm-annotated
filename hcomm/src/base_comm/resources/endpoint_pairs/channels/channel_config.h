/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHANNEL_CONFIG_H
#define CHANNEL_CONFIG_H

#include <atomic>
#include <cstdint>
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"

namespace hcomm {

/**
 * @note 职责：HcommChannelCreateWithConfig 的配置对象，记录是否共享 Jetty。
 *       对外通过 HcommChannelConfig 不透明句柄暴露，本 struct 为其实现细节。
 */
struct HcommChannelConfigData {
    bool isSharedQueue{false};
    std::atomic<bool> destroyed_{false};
};

HcommResult ChannelConfigCreate(HcommChannelConfig* config);
HcommResult ChannelConfigDestroy(HcommChannelConfig config);
HcommResult ChannelConfigSetInt(HcommChannelConfig config, HcommChannelConfigType type, uint32_t value);

} // namespace hcomm

#endif // CHANNEL_CONFIG_H
