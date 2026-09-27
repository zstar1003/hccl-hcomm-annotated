/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_JETTY_CHANNEL_HELPER_H
#define SHARED_JETTY_CHANNEL_HELPER_H

#include "endpoint.h"
#include "comm_queue_context/jetty_context.h"
#include "dev_ub_connection.h"
#include "shared_jetty_connection_adapter.h"
#include <functional>

namespace hcomm {

/**
 * @brief 创建临时 connection 的工厂回调类型（由各 channel 提供，确保 tpProtocol 正确）
 * @note 返回 Hccl::DevUbConnection 是历史现状（channel 侧已深度依赖该类型构造子类）；
 *       共享 jetty 对 connection 的操作统一走 SetSharedJettyFieldsToConn / ExtractJettyInfoFromConn / DetachConnJetty
 * 适配层，不直接调 legacy 新方法。
 */
using TempConnFactory = std::function<std::unique_ptr<Hccl::DevUbConnection>()>;

/**
 * @note 职责：channel 层 BuildConnection 共享 jetty 接入辅助。
 *       共享模式下：命中已有 jetty 则填充 connection 字段；未命中则用临时 connection 同步创建并缓存到 Endpoint。
 *       调用方在 BuildConnection 中，构造 connection 后调用本函数。
 * @param[in] endpoint channel 关联的 Endpoint 对象指针（由 EndpointHandle 转换得到）
 * @param[in] connection 已构造的 connection
 * @param[in] tempConnFactory 创建临时 connection 的回调（首次创建 jetty 时使用）
 * @param[out] outCtx 输出的共享 jetty 上下文（含 PI/CI 共享内存指针，供 channel 构建 entity 时复用）
 * @return HcclResult 成功后 connection 已注入共享 jetty；调用方仍需把 connection 加入 connections_ 并由状态机推进
 * Import。
 */
HcclResult AcquireSharedJettyForChannel(
    Endpoint* endpoint, Hccl::DevUbConnection* connection, const TempConnFactory& tempConnFactory,
    JettyContext::Ctx& outCtx);

} // namespace hcomm

#endif // SHARED_JETTY_CHANNEL_HELPER_H
