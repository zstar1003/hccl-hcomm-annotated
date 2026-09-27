/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RANGE_UTIL_H
#define HCCLV2_RANGE_UTIL_H
#include <limits>

namespace Hccl {
/**
 * 判断以begin1为起点，长度为len1的范围是否包含以begin2为起点，长度为len2的范围
 * @tparam T1
 * @tparam T2
 * @param begin1  范围1的起点
 * @param len1    范围1的长度
 * @param begin2  范围2的起点
 * @param len2    范围2的长度
 * @return 范围1包含范围2则返回true；否则返回false
 */
template <typename T1, typename T2>
inline bool IsRangeInclude(T1 begin1, T2 len1, T1 begin2, T2 len2)
{
    // 检查begin1 + len1是否会溢出
    if (len1 > static_cast<T2>(std::numeric_limits<T1>::max() - begin1)) {
        return false;
    }
    // 检查begin2 + len2是否会溢出
    if (len2 > static_cast<T2>(std::numeric_limits<T1>::max() - begin2)) {
        return false;
    }
    return (begin1 <= begin2 && begin1 + len1 >= begin2 + len2);
}
} // namespace Hccl

#endif // HCCLV2_RANGE_UTIL_H
