/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hash_utils.h"

namespace Hccl {

std::size_t HashCombine(std::initializer_list<std::size_t> hashItem)
{
    std::size_t res = 17;
    std::size_t padding = 31;
    for (auto begin = hashItem.begin(); begin != hashItem.end(); ++begin) {
        res = padding * res + (*begin);
    }
    return res;
}

} // namespace Hccl
