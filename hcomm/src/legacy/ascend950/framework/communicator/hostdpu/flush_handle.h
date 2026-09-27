/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_FLUSH_HANDLE_H
#define HCCL_FLUSH_HANDLE_H

#include <string>

#include "hccp_common.h"
#include "hccl_types.h"
#include "rdma_handle_manager.h"

namespace Hccl {
constexpr u64 FLUSH_BUFFER_SIZE = 8; // 每次读取8字节
class FlushHandle {
public:
    FlushHandle();
    ~FlushHandle();

    // 初始化 handle
    HcclResult Init(IpAddress ip, u32 devPhyId);

    // 销毁 handle
    HcclResult Destroy();

    // 成员变量
    MrInfoT loopBackQpMrRemoteInfo = {};
    MrInfoT loopBackQpMrLocalInfo = {};
    LoopbackQpPair loopBackQpParam = {};

    bool GetFlushOpcodeSupport() const { return flushOpcodeSupport_; }

    void SetFlushOpcodeSupport() { flushOpcodeSupport_ = true; }

private:
    bool flushOpcodeSupport_{false};
    bool flushIsInitialized{false};
    void* localMem{nullptr};
    void* deviceMem{nullptr};
    MrHandle localMrHandle{nullptr};
    MrHandle remoteMrHandle{nullptr};
    QpHandle qpHandle{nullptr};
    RdmaHandle rdmaHandle{nullptr};

    // 初始化方法
    HcclResult GetRdmaHandle(IpAddress ip, u32 devPhyId, void** rdmaHandle) const;
    HcclResult GetLbMax(int* lbMax) const;
    HcclResult AllocateDeviceMemory();
    HcclResult AllocateLocalMemory();
    HcclResult CreateLoopbackQp();
    HcclResult RegisterLocalMr();
    HcclResult RegisterRemoteMr();

    // 销毁方法
    HcclResult DeregisterMr(MrHandle& mrHandle, std::string logTag) const;
    HcclResult DestroyLoopbackQp();
    HcclResult FreeLocalMemory();
    HcclResult FreeDeviceMemory();
};

} // namespace Hccl

#endif
