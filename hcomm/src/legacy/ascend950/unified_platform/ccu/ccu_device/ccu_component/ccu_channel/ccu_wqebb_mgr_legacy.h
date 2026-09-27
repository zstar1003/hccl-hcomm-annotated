/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_WQEBB_MGR_H
#define HCCL_CCU_WQEBB_MGR_H

#include <memory>

#include "ccu_res_allocator_legacy.h"
#include "ccu_device_manager.h"

namespace Hccl {

class CcuWqeBBMgr {
public:
    CcuWqeBBMgr(const int32_t devLogicId, const uint8_t dieId);
    CcuWqeBBMgr() = default;
    ~CcuWqeBBMgr() = default;

    HcclResult Alloc(const uint32_t sqSize, ResInfo& wqeBBInfo);
    HcclResult Release(const ResInfo& wqeBBInfo);

private:
    int32_t devLogicId{0};
    uint8_t dieId{0};

    std::unique_ptr<CcuResIdAllocator> idAllocator{nullptr};
};

}; // namespace Hccl

#endif
