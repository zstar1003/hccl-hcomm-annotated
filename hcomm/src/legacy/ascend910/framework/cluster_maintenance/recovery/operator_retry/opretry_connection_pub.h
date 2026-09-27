/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_OPRETRY_CONNECTION_PUB_H
#define HCCL_OPRETRY_CONNECTION_PUB_H

#include <string>
#include "hccl_socket.h"
#include "hccl_op_retry_pub.h"

namespace hccl {
class OpRetryConnectionPub {
public:
    /* 配置是否开启该建链功能 */
    static void SetOpRetryConnEnable(bool enable);
    static bool IsOpRetryConnEnable();

    /* 初始化group中的链接 */
    static HcclResult Init(
        const std::string& group, u32 rankSize, const OpRetryServerInfo& serverInfo, const OpRetryAgentInfo& agentInfo,
        u32 rootRank = 0);
    /* 释放group对应的链接资源 */
    static void DeInit(const std::string& group);
    /* 获取对应链路，只有isRoot为真时，server中才会有有效链接 */
    static HcclResult GetConns(
        const std::string& group, bool& isRoot, std::shared_ptr<HcclSocket>& agent,
        std::map<u32, std::shared_ptr<HcclSocket>>& server);
};
} // namespace hccl

#endif
