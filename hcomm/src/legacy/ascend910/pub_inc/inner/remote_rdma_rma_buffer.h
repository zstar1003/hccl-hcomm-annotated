/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REMOTE_RDMA_RMA_BUFFER_H
#define REMOTE_RDMA_RMA_BUFFER_H

#include <memory>
#include <string>
#include "hccl_common.h"
#include "hccl_network_pub.h"
#include "rma_buffer.h"

namespace hccl {
class RemoteRdmaRmaBufferImpl;

class RemoteRdmaRmaBuffer : public RmaBuffer {
public:
    RemoteRdmaRmaBuffer();
    HcclResult Deserialize(const std::string&);
    ~RemoteRdmaRmaBuffer() override;

    RemoteRdmaRmaBuffer(const RemoteRdmaRmaBuffer& that) = delete;
    RemoteRdmaRmaBuffer& operator=(const RemoteRdmaRmaBuffer& that) = delete;

    // Deserialize成功后调用获取属性接口
    u32 GetKey() const;

private:
    std::unique_ptr<RemoteRdmaRmaBufferImpl> pimpl_;
};
} // namespace hccl
#endif //  REMOTE_RDMA_RMA_BUFFER_H
