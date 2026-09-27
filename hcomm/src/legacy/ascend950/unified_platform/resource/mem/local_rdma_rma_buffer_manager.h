/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOCAL_RDMA_RMA_BUFFER_MANAGER_H
#define LOCAL_RDMA_RMA_BUFFER_MANAGER_H

#include <memory>

#include "hccl/base.h"
#include "buffer_key.h"
#include "../buffer/local_rdma_rma_buffer_v2.h"
#include "rma_buffer_mgr.h"

namespace Hccl {

using LocalRdmaRmaBufferMgr = hcomm::RmaBufferMgr<BufferKey<uintptr_t, u64>, std::shared_ptr<LocalRdmaRmaBuffer>>;

class LocalRdmaRmaBufferManager {
public:
    static LocalRdmaRmaBufferMgr* GetInstance();

private:
    LocalRdmaRmaBufferManager() = default;
    ~LocalRdmaRmaBufferManager() = default;
    LocalRdmaRmaBufferManager(const LocalRdmaRmaBufferManager&) = delete;
    LocalRdmaRmaBufferManager& operator=(const LocalRdmaRmaBufferManager&) = delete;
};

} // namespace Hccl

#endif // LOCAL_RDMA_RMA_BUFFER_MANAGER_H
