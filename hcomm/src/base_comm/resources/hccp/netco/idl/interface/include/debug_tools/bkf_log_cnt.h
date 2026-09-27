/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_LOG_CNT_H
#define BKF_LOG_CNT_H

#include "bkf_comm.h"
#include "bkf_mem.h"
#include "bkf_disp.h"
#include "bkf_log_comm.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */
/**
 * @brief logCnt库句柄
 */
typedef struct tagBkfLogCnt BkfLogCnt;

/* init */
/**
 * @brief logCnt库初始化参数
 */
typedef struct tagBkfLogCntInitArg {
    char *name;        /**< 名称 */
    BkfMemMng *memMng; /**< 内存库句柄,见bkf_mem.h,同一app内可复用 */
    BkfDisp *disp;     /**< 诊断显示库句柄,见bkf_disp.h,同一app内可复用 */
    int32_t modCntMax; /**< 最大记录模块数量 */
    uint8_t rsv[0x10];
} BkfLogCntInitArg;

/* func */
/**
 * @brief log count 库初始化
 *
 * @param[in] *arg 参数
 * @return 库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfLogCnt *BkfLogCntInit(BkfLogCntInitArg *arg);

/**
 * @brief log count 库反初始化
 *
 * @param[in] *logCnt 库句柄
 * @return none
 */
void BkfLogCntUninit(BkfLogCnt *logCnt);

/**
 * @brief 记录count
 *
 * @param[in] *logCnt 库句柄
 * @param[in] *modName 模块名。
 * @param[in] line log位置标识。
 * @return none
 */
void BkfLogCntFunc(BkfLogCnt *logCnt, char *modName, uint16_t line);

/* 宏，便于使用 */
/**
 * @brief debug log宏封装。记录细粒度的诊断信息。
 *
 * @param[in] logCnt logCnt库句柄
 * @return none
 */
#define BKF_LOG_CNT(logCnt)                                                                                            \
    do {                                                                                                               \
        BkfLogCntFunc((logCnt), (char *)(BKF_MOD_NAME_), (BKF_LINE_));                                                 \
    } while (0)

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
