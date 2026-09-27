/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ts_hccs_server_socket_context.h"
#include "log.h"
#include "net_dev/global_net_dev_manager.h"

namespace hcomm {

// ---- AicpuTsHccsServerSocketContext（GlobalNetDevMgr 路径，从 AicpuTsHccsEndpoint 迁入）----

AicpuTsHccsServerSocketContext::AicpuTsHccsServerSocketContext(uint32_t devPhyId, uint32_t serverPort)
    : devPhyId_(devPhyId),
      serverPort_(serverPort)
{}

AicpuTsHccsServerSocketContext::~AicpuTsHccsServerSocketContext()
{
    // 幂等兜底：宿主 endpoint 析构体内已显式停止监听时（serverListened_==false）为空操作
    if (serverListened_) {
        (void)hccl::GlobalNetDevMgr::GetInstance(devPhyId_).ServerDeInit(serverPort_);
        serverListened_ = false;
    }
}

HcclResult AicpuTsHccsServerSocketContext::ServerSocketListen([[maybe_unused]] const uint32_t port)
{
    CHK_RET(hccl::GlobalNetDevMgr::GetInstance(devPhyId_).ServerInit(serverPort_));
    serverListened_ = true;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsHccsServerSocketContext::ServerSocketStopListen(const uint32_t port)
{
    if (serverListened_) {
        CHK_RET(hccl::GlobalNetDevMgr::GetInstance(devPhyId_).ServerDeInit(port));
        serverListened_ = false;
    }
    return HCCL_SUCCESS;
}

} // namespace hcomm
