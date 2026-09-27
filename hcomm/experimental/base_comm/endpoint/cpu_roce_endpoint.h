/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_EXPERIMENTAL_CPU_ROCE_ENDPOINT_H
#define HCOMM_EXPERIMENTAL_CPU_ROCE_ENDPOINT_H

#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include "endpoint.h"
#include "externalinput_pub.h"

namespace hcomm_experimental {
/**
 * @note 职责：Host CPU通信引擎+RoCE协议的通信设备Endpoint，管理通信设备上下文，以及设备上的注册内存。
 */
class CpuRoceEndpoint : public Endpoint {
public:
    explicit CpuRoceEndpoint(const EndpointDesc& endpointDesc);

    ~CpuRoceEndpoint() noexcept override;

    HcclResult Init() override;

    HcclResult ServerSocketListen(const uint32_t port) override;
    HcclResult ServerSocketStopListen(const uint32_t port) override;
    HcclResult ServerSocketGetListenPort(uint32_t* port) override;

    HcclResult RegisterMemory(HcommMem mem, const char* memTag, void** memHandle) override;
    HcclResult UnregisterMemory(void* memHandle) override;
    HcclResult MemoryExport(void* memHandle, void** memDesc, uint32_t* memDescLen) override;
    HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) override;
    HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) override;
    HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) override;

    struct Capabilities {
        uint64_t maxMsgSize{0};
        int lbMax{0};
        // 按需扩展
    };
    HcclResult GetCapabilities(Capabilities& caps);

private:
    std::mutex portMutex_;
    u32 dynamicPort_{HCCL_INVALID_PORT};
    Capabilities capabilities_{};
    bool isCapabilitiesAvailable_{false};
};
} // namespace hcomm_experimental
#endif // HCOMM_EXPERIMENTAL_CPU_ROCE_ENDPOINT_H
