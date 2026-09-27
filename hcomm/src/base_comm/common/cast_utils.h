/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CAST_UTILS_H
#define HCCL_CAST_UTILS_H

// reinterpret_cast 封装函数，统一替代各场景的 reinterpret_cast 调用
template <typename TargetType, typename SourceType>
inline TargetType ReinterpretAs(SourceType val)
{
    return reinterpret_cast<TargetType>(val);
}

#endif // HCCL_CAST_UTILS_H
