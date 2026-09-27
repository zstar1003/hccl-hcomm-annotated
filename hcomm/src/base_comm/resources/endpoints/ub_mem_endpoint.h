/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UB_MEM_ENDPOINT_H
#define UB_MEM_ENDPOINT_H

#include <memory>
#include <vector>
#include "endpoint.h"
#include "ub_mem_reged_mem_mgr.h"

namespace hcomm {
class UbMemEndpoint : public Endpoint {
public:
    explicit UbMemEndpoint(const EndpointDesc& endpointDesc);
    ~UbMemEndpoint() noexcept override;
    HcclResult Init() override;
    RegedMemMgr* GetRegedMemMgr() override { return regedMemMgr_.get(); }
    void* GetRdmaHandle() override { return nullptr; }
    bool IsCtxHandleValid() const override { return false; }

private:
    HcclResult AttachCache(const MemMgrCacheKey& key, const std::function<std::shared_ptr<RegedMemMgr>()>& creator);
    HcclResult ReleaseCache();

    std::shared_ptr<UbMemRegedMemMgr> regedMemMgr_{};
    MemMgrCacheKey cacheKey_{};
    std::shared_ptr<ProcRegedMemMgrCache> cacheKeepAlive_{};
};

} // namespace hcomm

#endif // UB_MEM_ENDPOINT_H
