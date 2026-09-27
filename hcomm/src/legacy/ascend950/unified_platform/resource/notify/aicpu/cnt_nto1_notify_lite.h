/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CNT_NTO1_NOTIFY_LITE_H
#define HCCLV2_CNT_NTO1_NOTIFY_LITE_H

#include <string>
#include <vector>
#include "types.h"

namespace Hccl {
class CntNto1NotifyLite {
public:
    explicit CntNto1NotifyLite(std::vector<char>& uniqueId);

    u32 GetId() const;
    u32 GetDevPhyId() const;

    std::string Describe() const;

private:
    u32 notifyId;
    u32 devPhyId;
};
} // namespace Hccl

#endif // HCCLV2_CNT_NTO1_NOTIFY_LITE_H
