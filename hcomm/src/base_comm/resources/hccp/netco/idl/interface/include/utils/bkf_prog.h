/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_FIBO_H
#define BKF_FIBO_H

#include "bkf_comm.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/**
 * @brief 获取斐波那契数列值
 *
 * @param[in] idx : 斐波那契数列下标，从0开始
 * @return 斐波那契数列值
 *   @retval >=0  正常值
 *   @retval <0   超出int32_t表示范围，返回负数
 */
int32_t BkfFibo32Get(int32_t idx);

/**
 * @brief 获取斐波那契数列下一个有效值
 *
 * @param[in] itorInOut : 迭代器
 * @return 斐波那契数列值
 *   @retval >= 0  正常值
 *   @retval < 0   超出int32_t表示范围，返回负数
 * @note 外部不要关注itor
 */
int32_t BkfFibo32GetNextValid(int32_t *itorInOut, int32_t max);
#define BKF_PROG_INIT_ITOR_FIRST(itor)                                                                                 \
    do {                                                                                                               \
        *(itor) = -1;                                                                                                  \
    } while (0)

/**
 * @brief 获取斐波那契数列前一个有效值
 *
 * @param[in] itorInOut : 迭代器
 * @return 斐波那契数列值
 *   @retval >= 0  正常值
 *   @retval < 0   超出int32_t表示范围，返回负数
 * @note 外部不要关注idxItor
 */
int32_t BkfFibo32GetPrevValid(int32_t *itorInOut, int32_t max);
#define BKF_PROG_INIT_ITOR_LAST(itor)                                                                                  \
    do {                                                                                                               \
        *(itor) = 0x7fffffff;                                                                                          \
    } while (0)

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
