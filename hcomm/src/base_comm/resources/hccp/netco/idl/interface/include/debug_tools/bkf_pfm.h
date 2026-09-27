/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_PFM_H
#define BKF_PFM_H

#include "bkf_comm.h"
#include "bkf_mem.h"
#include "bkf_disp.h"
#include "bkf_log.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */
/**
 * @brief 性能测量库句柄,用于对某个流程进行耗时打点
 */
typedef struct tagBkfPfm BkfPfm;

/* init */
/**
 * @brief 性能测量库初始化参数
 */
typedef struct tagBkfPfmInitArg {
    char *name;        /**< 初始化库名称 */
    BkfMemMng *memMng; /**< 内存库句柄,见bkf_mem.h,同一app内可复用 */
    BkfDisp *disp;     /**< 诊断显示库句柄,见bkf_disp.h,同一app内可复用 */
    BkfLogCnt *logCnt; /**< log cnt库句柄,见bkf_log_cnt.h,日志记录次数，同一app内可复用 */
    BkfLog *log;       /**< log库句柄,见bkf_log.h,同一app内可复用 */
    BOOL enable;       /**< 测量开关，初始化后，在运行期也可以通过测试命令开启关闭 */
    uint8_t rsv[0x10];
} BkfPfmInitArg;

/* func */
/**
 * @brief 性能测量库初始化
 *
 * @param[in] *arg 初始化参数
 * @return 性能测量库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfPfm *BkfPfmInit(BkfPfmInitArg *arg);

/**
 * @brief 性能测量库去初始化
 *
 * @param[in] *pfm 性能测量库句柄
 * @return none
 */
void BkfPfmUninit(BkfPfm *pfm);
/*
1. 统计以meaStr为粒度，[begin, end]的耗时
2. 因为字串本身的比对操作也会耗时，影响测试准确度。因此，字串不要太长，不要太相似。
*/
/**
 * @brief 开启一个性能测量实例 \n
 1.统计以meaStr为粒度，[begin, end]的耗时, \n
 2.因为字串本身的比对操作也会耗时，影响测试准确度。因此，字串不要太长，不要太相似。
 * @param[in] *pfm 性能测量库句柄
 * @param[in] *meaStr 性能测量实例key
 * @return 开启结果
 *   @retval BKF_OK 开启成功
 *   @retval BKF_ERR 开启失败
 */
uint32_t BkfPfmBegin(BkfPfm *pfm, char *meaStr);

/**
 * @brief 结束一个性能测量实例，测量结果记录在内存中，通过测试命令输出
 * @param[in] *pfm 性能测量库句柄
 * @param[in] *meaStr 性能测量实例key
 * @return 结果
 *   @retval BKF_OK 成功
 *   @retval BKF_ERR 失败
 */
uint32_t BkfPfmEnd(BkfPfm *pfm, char *meaStr);

/**
 * @brief 获取性能测量汇总输出字符串
 * @param[in] *pfm 性能测量库句柄
 * @param[in] *buf 输出缓冲区
 * @param[in] bufLen 输出缓冲区长度
 * @return 输出字符串
 */
char *BkfPfmGetSummaryStr(BkfPfm *pfm, uint8_t *buf, int32_t bufLen);

/**
 * @brief 获取第一个性能测量实例输出字符串
 * @param[in] *pfm 性能测量库句柄
 * @param[out] **meaStrOut 第一个性能测量实例
 * @param[in] *buf 输出缓冲区
 * @param[in] bufLen 输出缓冲区长度
 * @return 输出字符串
 *   @retval 非空 获取成功
 *   @retval NULL 获取失败
 */
char *BkfPfmGetFirstMeaStr(BkfPfm *pfm, char **meaStrOut, uint8_t *buf, int32_t bufLen);

/**
 * @brief 获取下一个性能测量实例输出字符串
 * @param[in] *pfm 性能测量库句柄
 * @param[in] **meaStrInOut 输入性能测量实例，同时为出参：下一个实例
 * @param[in] *buf 输出缓冲区
 * @param[in] bufLen 输出缓冲区长度
 * @return 输出字符串
 *   @retval 非空 获取成功
 *   @retval NULL 获取失败
 */
char *BkfPfmGetNextMeaStr(BkfPfm *pfm, char **meaStrInOut, uint8_t *buf, int32_t bufLen);

/**
 * @brief 测试接口,测试性能库准确度，业务无需调用
 * @param[in] *pfm 性能测量库句柄
 * @return None
 */
void BkfPfmTestAddMea(BkfPfm *pfm);

/* 宏，便于使用 */
#ifdef BKF_PFM_ON
/**
 * @brief 性能测量实例开启易用宏封装,需定义BKF_PFM_ON后生效
 *
 * @param[in] pfm 性能测量库句柄
 * @param[in] mstr 性能测量实例字符串
 * @return none
 */
#define BKF_PFM_BEGIN(pfm, mstr)                                                                                       \
    do {                                                                                                               \
        (void)BkfPfmBegin((pfm), (char *)(mstr));                                                                      \
    } while (0)

/**
 * @brief 性能测量实例结束易用宏封装,需定义BKF_PFM_ON后生效
 *
 * @param[in] pfm 性能测量库句柄
 * @param[in] mstr 性能测量实例字符串
 * @return none
 */
#define BKF_PFM_END(pfm, mstr)                                                                                         \
    do {                                                                                                               \
        (void)BkfPfmEnd((pfm), (char *)(mstr));                                                                        \
    } while (0)

#else
/**
 * @brief 性能测量实例开启易用宏封装
 *
 * @param[in] pfm 性能测量库句柄
 * @param[in] mstr 性能测量实例字符串
 * @return none
 */
#define BKF_PFM_BEGIN(pfm, mstr)                                                                                       \
    do {                                                                                                               \
    } while (0)

/**
 * @brief 性能测量实例结束易用宏封装
 *
 * @param[in] pfm 性能测量库句柄
 * @param[in] mstr 性能测量实例字符串
 * @return none
 */
#define BKF_PFM_END(pfm, mstr)                                                                                         \
    do {                                                                                                               \
    } while (0)

#endif
#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
