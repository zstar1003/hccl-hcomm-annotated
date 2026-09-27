/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_EXPERIMENTAL_ENDPOINT_H
#define HCOMM_EXPERIMENTAL_ENDPOINT_H

#include <memory>
#include <vector>
#include <string>
#include "reged_mem_mgr.h"
#include "hcomm_c_adpt.h"
#include "socket/socket.h"
#include "socket_handle_manager.h"
#include "rdma_handle_manager.h"
#include "orion_adpt_utils.h"
#include "hccp_hdc_manager.h"

namespace hcomm_experimental {
/**
 * @note 职责：通信设备Endpoint的C++抽象接口类，管理通信设备上下文，以及设备上的注册内存。
 */
class Endpoint {
public:
    explicit Endpoint(const EndpointDesc& endpointDesc);

    virtual ~Endpoint() = default;

    static HcclResult CreateEndpoint(const EndpointDesc& endpointDesc, std::unique_ptr<Endpoint>& endpointPtr);

    virtual HcclResult Init() = 0;

    virtual HcclResult ServerSocketListen(const uint32_t port) = 0;

    virtual HcclResult ServerSocketStopListen(const uint32_t port)
    {
        (void)port;
        return HCCL_E_NOT_SUPPORT;
    }

    virtual HcclResult ServerSocketGetListenPort(uint32_t* port)
    {
        (void)port;
        return HCCL_E_NOT_SUPPORT;
    }

    virtual std::shared_ptr<RegedMemMgr> GetRegedMemMgr() { return regedMemMgr_; }

    void* GetRdmaHandle() { return ctxHandle_; }

    EndpointDesc GetEndpointDesc() { return endpointDesc_; }

    // 注册内存
    virtual HcclResult RegisterMemory(HcommMem mem, const char* memTag, void** memHandle) = 0;

    // 注销内存
    virtual HcclResult UnregisterMemory(void* memHandle) = 0;

    // 导出指定内存描述，用于交换
    virtual HcclResult MemoryExport(void* memHandle, void** memDesc, uint32_t* memDescLen) = 0;

    // 基于内存描述，导入获得内存
    virtual HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) = 0;

    // 关闭内存
    virtual HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) = 0;

    virtual HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) = 0;

    virtual HcclResult MemoryGrant(const HcommMemGrantInfo* remoteGrantInfo)
    {
        (void)remoteGrantInfo;
        return HCCL_SUCCESS;
    }

    // 获取UB异步事件
    HcclResult GetAsyncEventsContext(uint32_t devPhyId, struct AsyncEvent events[], uint32_t& num);

protected:
    static HcclResult CreateEndpointBase(const EndpointDesc& endpointDesc, std::unique_ptr<Endpoint>& endpointPtr);
    void* ctxHandle_{nullptr};
    std::shared_ptr<RegedMemMgr> regedMemMgr_{};
    EndpointDesc endpointDesc_;
};

} // namespace hcomm_experimental
#endif // HCOMM_EXPERIMENTAL_ENDPOINT_H
