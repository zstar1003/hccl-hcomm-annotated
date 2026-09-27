/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_JETTY_CONNECTION_ADAPTER_H
#define SHARED_JETTY_CONNECTION_ADAPTER_H

#include <functional>
#include "endpoint.h"
#include "comm_queue_context/jetty_context.h"

namespace hcomm {

/**
 * @note 职责：共享 jetty 与底层 connection 交互的适配层。
 *       base_comm 的 shared_jetty_channel_helper 通过本适配层操作 connection，
 *       不直接 #include legacy 的 dev_ub_connection.h，便于将来 DevUbConnection 迁入 base_comm 时仅改本文件。
 *       当前实现内部转发到 Hccl::DevUbConnection 的新增方法（dev_ub_connection.h）。
 *
 *       rawConnection 形参为不透明指针，调用方从具体 connection 类型 reinterpret 而来；
 *       适配层实现内部 static_cast 回 Hccl::DevUbConnection*。
 *       该设计牺牲一层类型安全，换取 base_comm 头文件不暴露 legacy 依赖。
 */

/**
 * @brief 填充共享 jetty 字段到 EXTERNAL_INJECT 模式的 connection（命中复用路径）
 * @param[in] rawConnection 不透明 connection 指针（构造时传 JettyMode::EXTERNAL_INJECT）
 * @param[in] ctx 已创建/复用的共享 jetty 上下文
 * @param[in] endpointTag Endpoint 不透明标签（透传给 releaseCb）
 * @param[in] releaseCb connection 销毁时回调（由调用方注入 jetty 减引用）
 */
HcclResult SetSharedJettyFieldsToConn(
    void* rawConnection, const JettyContext::Ctx& ctx, void* endpointTag, std::function<void(void*)> releaseCb);

/**
 * @brief 从已完成 jetty 创建的 connection 提取衍生字段（首次创建路径）
 * @param[in] rawConnection 不透明 connection 指针
 * @param[out] ctx 输出的 jetty 上下文（仅 jetty 相关字段, refCount 不填）
 */
HcclResult ExtractJettyInfoFromConn(void* rawConnection, JettyContext::Ctx& ctx);

/**
 * @brief 分离 connection 的 jetty 所有权，析构不再销毁 jetty（首次创建后调用）
 */
HcclResult DetachConnJetty(void* rawConnection);

} // namespace hcomm

#endif // SHARED_JETTY_CONNECTION_ADAPTER_H
