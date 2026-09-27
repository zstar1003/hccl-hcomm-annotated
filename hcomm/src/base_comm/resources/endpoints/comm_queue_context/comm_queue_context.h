/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_QUEUE_CONTEXT_H
#define COMM_QUEUE_CONTEXT_H

namespace hcomm {

/**
 * @note 职责：通信队列上下文（CommQueueContext）体系基类。
 *       仅定义生命周期契约（虚析构），不承载具体资源，Endpoint 基类不感知派生细节；
 *       具体队列资源由派生类聚合（如共享 Jetty 数据面资源由 JettyContext 承载），
 *       访问入口为 Endpoint 基类虚函数 GetCommQueueContext()，调用方按需 downcast JettyContext 使用。
 */
class CommQueueContext {
public:
    CommQueueContext() = default;
    virtual ~CommQueueContext() = default;
};

} // namespace hcomm

#endif // COMM_QUEUE_CONTEXT_H
