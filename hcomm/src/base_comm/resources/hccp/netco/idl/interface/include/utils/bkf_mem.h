/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_MEM_H
#define BKF_MEM_H

#include "bkf_comm.h"

/*
!!! 非常重要 !!!
1. init之后，“库” 线程安全。
   并且 BKF_MEM_STAT 不打开，也不会影响库性能。
2. 如果一个库多线程使用，还需要 malloc 和 free 回调实现线程安全，才能保证 “端到端”线程安全
*/

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */
typedef struct tagBkfMemMng BkfMemMng;

typedef struct tagBkfMemStatKey {
    char *str;
    uint16_t num;
    uint8_t pad1[2];
} BkfMemStatKey;

/* init */
typedef void *(*F_BKF_IMEM_MALLOC)(void *cookieIMem, uint32_t len, const char *str, const uint16_t num);
typedef void (*F_BKF_IMEM_FREE)(void *cookieIMem, void *ptr, const char *str, const uint16_t num);
typedef void *(*F_BKF_IMEM_GET_ADPEE_MEM_HND)(void *cookieIMem);
typedef struct tagBkfIMem {
    char *name;
    void *cookie;
    F_BKF_IMEM_MALLOC malloc;
    F_BKF_IMEM_FREE free;
    F_BKF_IMEM_GET_ADPEE_MEM_HND getAdpeeMemHndOrNull;
    uint8_t rsv1[0x10];
} BkfIMem;

/* func */
/**
 * @brief 内存库初始化
 *
 * @param[in] *IMem 内存接口(interface)
 * @return 内存库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfMemMng *BkfMemInit(BkfIMem *IMem);

/**
 * @brief 内存库反初始化
 *
 * @param[in] *memMng 内存库句柄
 * @return none
 */
void BkfMemUninit(BkfMemMng *memMng);

/**
 * @brief 内存申请
 *
 * @param[in] *memMng 内存库句柄
 * @param[in] len 内存申请长度
 * @param[in] *str 申请内存标记(可输出文件名)
 * @param[in] num 申请内存标记(可输出文件行号)
 * @return 内存申请结果地址
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
void *BkfMalloc(BkfMemMng *memMng, uint32_t len, const char *str, const uint16_t num);

/**
 * @brief 内存释放
 *
 * @param[in] *memMng 内存库句柄
 * @param[in] *ptr 内存地址
 * @param[in] *str 申请内存标记(可输出文件名)
 * @param[in] num 申请内存标记(可输出文件行号)
 * @return none
 */
void BkfFree(BkfMemMng *memMng, void *ptr, const char *str, const uint16_t num);

/**
 * @brief 内存释放，不带句柄
 *
 * @param[in] *ptr 内存地址
 * @param[in] *str 申请内存标记(可输出文件名)
 * @param[in] num 申请内存标记(可输出文件行号)
 * @return none
 */
void BkfFreeNHnd(void *ptr, const char *str, const uint16_t num);

/**
 * @brief 获取adpee 实际内存的句柄。
 *
 * @param[in] *memMng 内存库句柄
 * @return adpee实际内存的句柄
 *   @retval 非空 实际可用句柄
 *   @retval 空    无可用句柄
 * @note 内存采用的是适配器模式。目前vrp类型的adpee有vrp hnd。在业务某些场景需要获取。因此，提供此函数。但尽量少使用。
 */
void *BkfMemGetAdpeeMemHnd(BkfMemMng *memMng);

/**
 * @brief 获取内存库诊断信息
 *
 * @param[in] *memMng 内存库句柄
 * @param[in] *buf 信息缓冲
 * @param[in] bufLen 缓冲长度
 * @return 诊断信息
 *   @retval 非空 诊断信息字串。如果内部判断错误，返回一个常量串。所以不会返回空。
 * @note 由于@BkfMemGetFirstStatStr和@BkfMemGetNextStatStr可能不在一个消息中被调用，期间内存占用可能会变化，\n
 *       所以，两函数内存占用和，未必等于此函数。
 */
char *BkfMemGetSummaryStr(BkfMemMng *memMng, uint8_t *buf, int32_t bufLen);

/**
 * @brief 获取内存库统计信息
 *
 * @param[in] *memMng 内存库句柄
 * @param[in] *keyOut 输出统计key
 * @param[in] *buf 信息缓冲
 * @param[in] bufLen 缓冲长度
 * @return 诊断信息
 *   @retval 非空 有诊断信息
 *   @retval 空 无诊断信息
 */
char *BkfMemGetFirstStatStr(BkfMemMng *memMng, BkfMemStatKey *keyOut, uint8_t *buf, int32_t bufLen);

/**
 * @brief 获取内存库统计信息
 *
 * @param[in] *memMng 内存库句柄
 * @param[in] *keyInOut 输入/输出统计key
 * @param[in] *buf 信息缓冲
 * @param[in] bufLen 缓冲长度
 * @return 诊断信息
 *   @retval 非空 有诊断信息
 *   @retval 空 无诊断信息
 */
char *BkfMemGetNextStatStr(BkfMemMng *memMng, BkfMemStatKey *keyInOut, uint8_t *buf, int32_t bufLen);

/* 宏，便于使用 */
#define BKF_MEM_STR __FILE__
#define BKF_MEM_NUM __LINE__

#ifdef BKF_MEM_DEBUG_STR
#define BKF_MALLOC(memMng, len) BkfMalloc((memMng), (len), (BKF_MEM_STR), (BKF_MEM_NUM))
#define BKF_FREE(memMng, ptr) BkfFree((memMng), (ptr), (BKF_MEM_STR), (BKF_MEM_NUM))
#define BKF_FREE_NHND(ptr) BkfFreeNHnd((ptr), (BKF_MEM_STR), (BKF_MEM_NUM))

#else
#define BKF_MALLOC(memMng, len) BkfMalloc((memMng), (len), "mallocRsv", 123)
#define BKF_FREE(memMng, ptr) BkfFree((memMng), (ptr), "freeRsv", 123)
#define BKF_FREE_NHND(ptr) BkfFreeNHnd((ptr), "freeRsv", 123)

#endif

/*
1. mockcpp中多线程调用桩函数，会有问题
2. 为了保持API不变，多两个桩：
   1) 在mock cpp中会被替换。
   2) 如果stub init返回非空，init的cookie会被替换。
*/
void *BkfMemBaseStubInit(char *name);
void BkfMemBaseStubUninit(void *cookie);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
