/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_base_stub.h"

#include <cstdint>

namespace AscendC {
int64_t block_num = 0;
thread_local int64_t block_idx = 0;

std::string GetPipeName(pipe_t pipe)
{
    switch (pipe) {
        case PIPE_ALL:
            return "PIPE_ALL";
        case PIPE_MTE2:
            return "PIPE_MTE2";
        case PIPE_MTE3:
            return "PIPE_MTE3";
        case PIPE_S:
            return "PIPE_SCALAR";
        default:
            return "UNKNOWN";
    }
}
} // namespace AscendC
