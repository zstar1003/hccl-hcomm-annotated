/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CHANNEL_CONFIG_H
#define HCCL_CHANNEL_CONFIG_H

#include <atomic>
#include <string>

namespace hccl {

/**
 * @note 职责：HcclChannelAcquireWithConfig 的配置对象（HCCM 层）。
 *       记录是否共享 Jetty 及其 tag，用于通信域层 channel 复用管理。
 *       对外通过 HcclChannelConfig 不透明句柄暴露，本 struct 为其实现细节。
 */
struct HcclChannelConfigData {
    bool isSharedQueue{false};
    std::string sharedQueueTag;
    std::atomic<bool> destroyed_{false};
};

} // namespace hccl

#endif // HCCL_CHANNEL_CONFIG_H
