/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENDPOINT_H
#define ENDPOINT_H

#include <memory>
#include <vector>
#include <string>
#include "reged_mem_mgr.h"
#include "socket/socket.h"
#include "socket_handle_manager.h"
#include "rdma_handle_manager.h"
#include "hccp_hdc_manager.h"
#include "proc_reged_mem_mgr_cache.h"
#include "comm_queue_context/comm_queue_context.h"
#include "dfx/endpoint_monitor.h"
#include "server_socket_context/server_socket_context.h"
#include "mgr/endpoint_ctx_mgr.h"
#include "../../common/orion_adpt_utils.h"

namespace hcomm {

class EndpointMonitor;
/**
 * @note 职责：通信设备Endpoint的C++纯虚接口类，只承担 endpoint 生命周期与工厂职责。
 *       数据成员仅保留 endpointDesc_/monitorKeepAlive_（各子类无差别），其余全部移至各具体派生类。
 *       内存注册接口由 RegedMemMgr 承载（经 GetRegedMemMgr() 访问）；
 *       Socket 监听接口由 ServerSocketContext 承载（经 GetServerSocketContext() 访问）；
 *       NIC 插件方法由 PluginEndpointHolder 承载；
 *       RegedMemMgr 缓存管理（AttachCache/ReleaseCache）移至各派生类 private。
 */
class Endpoint {
public:
    explicit Endpoint(const EndpointDesc& endpointDesc);

    virtual ~Endpoint();

    static HcclResult CreateEndpoint(const EndpointDesc& endpointDesc, std::unique_ptr<Endpoint>& endpointPtr);

    virtual HcclResult Init() = 0;

    // 返回组合持有的 RegedMemMgr（各派生类 override 返回自身成员）
    // 返回裸指针视图：生命周期由宿主 endpoint 保证，调用方不得在 endpoint 存活期外持有
    virtual RegedMemMgr* GetRegedMemMgr() = 0;

    // 返回 RDMA 句柄（各派生类 override 返回自身 ctxHandle_）
    virtual void* GetRdmaHandle() = 0;
    virtual bool IsCtxHandleValid() const = 0;

    // 返回 ServerSocketContext（仅 Socket 相关类 override 返回非 nullptr）
    virtual ServerSocketContext* GetServerSocketContext() { return nullptr; }

    // 返回 CommQueueContext（仅 UB 类 endpoint override 返回非 nullptr，当前唯一产出为 JettyContext）
    virtual CommQueueContext* GetCommQueueContext() { return nullptr; }

    EndpointDesc GetEndpointDesc() { return endpointDesc_; }

    static HcclResult CheckFeature(const EndpointDesc& endpointDesc, HcommEndpointFeatureType featureType, bool& value);

    // Register：AttachMonitor 拷贝 GetHolder，再用持有的指针 RegisterToEndpointMonitor。
    // Destroy / 析构走 ReleaseEndpointMonitor。未 Register 的空指针直接返回。不要再调 GetHolder()。
    void AttachMonitor(s32 logicId);
    HcclResult RegisterToEndpointMonitor(s32 logicId, EndpointHandle handle);
    void ReleaseEndpointMonitor(EndpointHandle handle);

protected:
    static HcclResult CreateEndpointBase(const EndpointDesc& endpointDesc, std::unique_ptr<Endpoint>& endpointPtr);
    EndpointDesc endpointDesc_;
    std::shared_ptr<EndpointMonitor> monitorKeepAlive_;
};

} // namespace hcomm
#endif // ENDPOINT_H
