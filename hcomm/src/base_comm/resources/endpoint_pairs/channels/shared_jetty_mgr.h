/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_JETTY_MGR_H
#define SHARED_JETTY_MGR_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"

namespace hcomm {

/**
 * @note 职责：管理共享 Jetty 的 channel 注册记录，用于 endpoint 销毁前校验。
 *       当 IS_SHARED_QUEUE=true 时，HcommChannelCreateWithConfig 通过本管理器注册 Channel，
 *       HcommChannelDestroy 通过本管理器注销 Channel，
 *       HcommEndpointDestroy 通过本管理器校验所有共享 Jetty 的 Channel 是否已销毁。
 *
 *       endpointHandle 作为不透明 key（builtin 路径为 Endpoint*，plugin 路径为 PluginEndpointCtx*），
 *       本管理器不做 static_cast 解引用，plugin/builtin 路径均安全。
 *       底层 jetty 句柄的缓存与引用计数由 JettyContext 管理（见 comm_queue_context/jetty_context.h），
 *       本管理器仅维护 channelHandle 集合用于销毁校验，不持有 jetty 句柄。
 *
 *       Endpoint/plugin 析构时调 UnregisterEndpoint 摘除记录，避免 endpointHandle 复用导致误判。
 */
class SharedJettyMgr {
public:
    struct SharedJettyContext {
        uint32_t channelCount{0};
        std::unordered_set<ChannelHandle> channelHandles;
    };

    static SharedJettyMgr& GetInstance();

    /**
     * @brief 注册 Channel 到共享 Jetty 上下文
     * @param[in] endpointHandle Endpoint 句柄（不透明 key，builtin 为 Endpoint*，plugin 为 PluginEndpointCtx*）
     * @param[in] channels 要注册的 Channel 句柄数组
     * @param[in] channelNum Channel 数量
     * @return HcclResult 执行结果
     */
    HcclResult RegisterChannels(EndpointHandle endpointHandle, const ChannelHandle* channels, uint32_t channelNum);

    /**
     * @brief 注销 Channel
     * @param[in] channels 要注销的 Channel 句柄数组
     * @param[in] channelNum Channel 数量
     * @return HcclResult 执行结果
     */
    HcclResult UnregisterChannels(const ChannelHandle* channels, uint32_t channelNum);

    /**
     * @brief 校验 Endpoint 是否可以销毁（所有共享 Jetty 的 Channel 已销毁）
     * @param[in] endpointHandle Endpoint 句柄（不透明 key）
     * @return HcclResult HCCL_SUCCESS 表示可以销毁，HCCL_E_UNAVAIL 表示仍有 Channel 未销毁
     */
    HcclResult CheckEndpointDestroy(EndpointHandle endpointHandle);

    /**
     * @brief 判断 Endpoint 是否已存在共享 Jetty 上下文
     */
    bool HasContext(EndpointHandle endpointHandle);

    /**
     * @brief Endpoint/plugin 析构时摘除该 endpointHandle 的注册记录，避免句柄复用误判
     * @note builtin 路径由 Endpoint::~Endpoint 调用，plugin 路径由 DestroyPluginEndpoint 调用
     */
    void UnregisterEndpoint(EndpointHandle endpointHandle);

private:
    SharedJettyMgr() = default;
    ~SharedJettyMgr() = default;
    SharedJettyMgr(const SharedJettyMgr&) = delete;
    SharedJettyMgr& operator=(const SharedJettyMgr&) = delete;

    std::mutex mtx_;
    std::unordered_map<EndpointHandle, SharedJettyContext> contexts_;
};

} // namespace hcomm

#endif // SHARED_JETTY_MGR_H
