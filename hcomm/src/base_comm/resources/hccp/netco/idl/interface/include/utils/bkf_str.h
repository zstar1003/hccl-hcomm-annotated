/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_STR_H
#define BKF_STR_H

#include "bkf_comm.h"
#include "bkf_mem.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
#define BKF_STR_LEN_MAX (1024)

/**
 * @brief 在堆上创建一个字串。注意字串长度不能大于 @see BKF_STR_LEN_MAX
 *
 * @param[in] memMng 内存句柄
 * @param[in] fmt 格式化字串
 * @param[in] ... 格式化字串相关的可变参数。
 * @return 字串
 *   @retval 非空 成功创建的字串
 *   @retval 空 失败
 */
char *BkfStrNew(BkfMemMng *memMng, const char *fmt, ...);

/**
 * @brief 删除字串
 *
 * @param[in] str 字串。 注意一定是 @see BkfStrNew 创建的。
 * @return 无
 */
void BkfStrDel(char *str);

/**
 * @brief 判断字串是否有效。注意一定是 @see BkfStrNew 创建的。
 *
 * @param[in] str 字串
 * @return 是否有效
 *   @retval 非零 无效
 *   @retval 零 有效
 */
BOOL BkfStrIsValid(char *str);

/**
 * @brief 获取内存使用的长度。注意不是字串长度。
 *
 * @param[in] str 字串
 * @return 内存使用长度
 *   @retval >=0 内存使用的有效长度
 *   @retval <0 无效长度，比如入参空，非 @see BKF_STR_LEN_MAX 创建的str，内存写坏等
 */
int32_t BkfStrGetMemUsedLen(char *str);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
