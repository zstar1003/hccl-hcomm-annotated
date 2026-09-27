/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_RES_PACK_H
#define HCCL_CCU_RES_PACK_H

#include "cast_utils.h"
#include <vector>

#include "ccu_device_manager.h"

namespace Hccl {

struct CcuResPack {
    std::vector<CcuResHandle> handles{};
    u32 count{0}; // resPack可能为多个ccuContext共用，计数清零时，才能释放ccu资源
    // 提供方法获取首handles
    uintptr_t GetId() const
    {
        if (handles.empty()) {
            return 0;
        }
        return ReinterpretAs<uintptr_t>(handles.front());
    }
    ~CcuResPack() { HCCL_DEBUG("~CcuResPack"); }
};

}; // namespace Hccl

#endif // HCCL_CCU_RES_PACK_H
