/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COMMON_INTERFACE_H
#define HCCLV2_COMMON_INTERFACE_H

#include "task_param.h"
#include <vector>
#include <string>

namespace Hccl {
void DumpCcuProfilingInfo(const std::vector<CcuProfilingInfo>& profInfos);
void CheckProfilingInfo(
    const std::vector<CcuProfilingInfo>& profInfos, uint32_t sqeCnt, uint32_t ckeCnt, uint32_t loopGroupCnt);

} // namespace Hccl

#endif // HCCLV2_COMMON_INTERFACE_H
