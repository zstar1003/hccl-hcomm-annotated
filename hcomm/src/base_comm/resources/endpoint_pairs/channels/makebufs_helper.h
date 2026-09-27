/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHANNEL_MAKEBUFS_HELPER_H
#define CHANNEL_MAKEBUFS_HELPER_H

#include "buffer.h"
#include "local_rma_buffer.h"
#include "log.h"
#include "hcomm_c_adpt.h"

namespace hcomm {

inline HcclResult MakeRmaBufferVecFromMemHandles(
    HcommMemHandle* memHandles, uint32_t memHandleNum, std::vector<Hccl::LocalRmaBuffer*>& bufferVec,
    const char* channelName)
{
    bufferVec.clear();
    for (uint32_t i = 0; i < memHandleNum; ++i) {
        auto localRmaBuffer = static_cast<Hccl::LocalRmaBuffer*>(memHandles[i]);
        CHK_PTR_NULL(localRmaBuffer);
        auto buf = localRmaBuffer->GetBuf();
        CHK_PTR_NULL(buf);
        HCCL_INFO(
            "[%s][%s] addr[0x%llx], size[0x%llx], memType[%d], memInfo[%s]", channelName, __func__,
            static_cast<unsigned long long>(localRmaBuffer->GetAddr()),
            static_cast<unsigned long long>(localRmaBuffer->GetSize()), static_cast<int>(buf->GetMemType()),
            buf->GetMemInfo().c_str());
        bufferVec.push_back(localRmaBuffer);
    }
    return HCCL_SUCCESS;
}

} // namespace hcomm

#endif // CHANNEL_MAKEBUFS_HELPER_H
