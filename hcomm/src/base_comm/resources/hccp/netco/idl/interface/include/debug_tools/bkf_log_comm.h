/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_LOG_COMM_H
#define BKF_LOG_COMM_H

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/**
 * 模块名易用宏。\n
 * 在一份文件(.c)的开头定义，那该文件后续调用log就不用反复输入。\n
 * 并且，不同文件定义不同的以区分。
 */
#ifdef BKF_MOD_NAME
#define BKF_MOD_NAME_ BKF_MOD_NAME
#else
#define BKF_MOD_NAME_ "logDefault"
#endif

/**
 * 模块位置易用宏。\n
 * 一份文件(.c)的开头定义，那该文件后续调用log就不用反复输入。\n
 */
#ifdef BKF_LINE
#define BKF_LINE_ BKF_LINE
#else
#define BKF_LINE_ __LINE__
#endif

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
