/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_TMR_H
#define BKF_TMR_H

#include "bkf_comm.h"
#include "bkf_mem.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */
typedef struct tagBkfTmrMng BkfTmrMng;
typedef struct tagBkfTmrId BkfTmrId;
/*
1. 定时器回调中一定会有 paramTmrStart 这个参数。
2. 一些timer的实现，为了减少一些内存消耗等原因，回调会有第二个 void * 的参数。
   这个参数的含义，取决于timer自己的实现，而且不同实现可能不一。
3. 原先回调的第二参数名是 paramTmrLibUnknown。bkf 不使用回调的第二参数；但app可能要用。
   1) bkf tmr是适配器模式的target，而adpee是业务初始化的。
      此时，adpee初始化时 可以(may) 增加cookie，并在timeout的回调中给出。
      因此，第二参数修改成 cookieOfAdpeeInit
   2) 对bkf lib来说，还是不使用（不知道其含义）
4. 注意，由于实现的约束，在once定时器回调中删除此定时器会有严重问题。
*/
typedef uint32_t (*F_BKF_TMR_TIMEOUT_PROC)(void *paramTmrStart, void *cookieOfAdpeeInit);

/* init */
typedef void *(*F_BKF_ITMR_START_ONCE)(void *cookieITmr, F_BKF_TMR_TIMEOUT_PROC proc, uint32_t intervalMs, void *param);
typedef void *(*F_BKF_ITMR_START_LOOP)(void *cookieITmr, F_BKF_TMR_TIMEOUT_PROC proc, uint32_t intervalMs, void *param);
typedef void (*F_BKF_ITMR_STOP)(void *cookieITmr, void *iTmrId);
typedef uint32_t (*F_BKF_ITMR_REFRESH)(void *cookieITmr, BkfTmrId *tmrId, uint32_t newIntervalMs);
typedef void *(*F_BKF_ITMR_GET_PARAM_START)(void *cookieITmr, BkfTmrId *tmrId);
typedef uint32_t (*F_BKF_ITMR_GET_REMAIN_TIME)(void *cookieITmr, BkfTmrId *tmrId);
typedef struct tagBkfITmr {
    char *name;
    BkfMemMng *memMng;
    void *cookie;
    F_BKF_ITMR_START_ONCE startOnce;
    F_BKF_ITMR_START_LOOP startLoop;
    F_BKF_ITMR_STOP stop;
    F_BKF_ITMR_REFRESH refreshOrNull;
    F_BKF_ITMR_GET_PARAM_START getParamStartOrNull;
    F_BKF_ITMR_GET_REMAIN_TIME getRemainTimeOrNull;
    uint8_t rsv1[0x10];
} BkfITmr;

/* func */
/**
 * @brief 定时器库初始化
 *
 * @param[in] *ITmr 定时器接口(interface)
 * @return 定时器库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfTmrMng *BkfTmrInit(BkfITmr *ITmr);

/**
 * @brief 定时器库反初始化
 *
 * @param[in] *tmrMng 定时器库句柄
 * @return none
 */
void BkfTmrUninit(BkfTmrMng *tmrMng);

/**
 * @brief 启动一次定时器
 *
 * @param[in] *tmrMng 定时器库句柄
 * @param[in] proc 超时回调函数
 * @param[in] intervalMs 定时器间隔
 * @param[in] param 超时回调函数传回的参数
 * @return 定时器创建结果，id号
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfTmrId *BkfTmrStartOnce(BkfTmrMng *tmrMng, F_BKF_TMR_TIMEOUT_PROC proc, uint32_t intervalMs, void *param);

/**
 * @brief 启动循环定时器
 *
 * @param[in] *tmrMng 定时器库句柄
 * @param[in] proc 超时回调函数
 * @param[in] intervalMs 定时器间隔
 * @param[in] param 超时回调函数传回的参数
 * @return 定时器创建结果，id号
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfTmrId *BkfTmrStartLoop(BkfTmrMng *tmrMng, F_BKF_TMR_TIMEOUT_PROC proc, uint32_t intervalMs, void *param);

/**
 * @brief 停止定时器
 *
 * @param[in] *tmrMng 定时器库句柄
 * @param[in] *tmrId 定时器id
 * @return none
 */
void BkfTmrStop(BkfTmrMng *tmrMng, BkfTmrId *tmrId);

/**
 * @brief 刷新定时器
 *
 * @param[in] *tmrMng 定时器库句柄
 * @param[in] *tmrId 定时器id
 * @param[in] newIntervalMs 新的定时器间隔
 * @return 是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfTmrRefresh(BkfTmrMng *tmrMng, BkfTmrId *tmrId, uint32_t newIntervalMs);

/**
 * @brief 获取定时器用户参数
 *
 * @param[in] *tmrMng 定时器库句柄
 * @param[in] *tmrId 定时器id
 * @return 用户参数
 *   @retval VOS_NULL tmrId不存在，或者用户参数本身为空
 *   @retval 其他 用户参数
 */
void *BkfTmrGetParamStart(BkfTmrMng *tmrMng, BkfTmrId *tmrId);

/**
 * @brief 获取定时器节点弹出剩余时间单位ms
 *
 * @param[in] *tmrMng 定时器库句柄
 * @param[in] *tmrId 定时器id
 * @return 用户参数
 *   @retval 入参非法返回0
 *   @retval 入参合法返回下次弹出的剩余时间单位ms
 */
uint32_t BkfTmrGetRemainTime(BkfTmrMng *tmrMng, BkfTmrId *tmrId);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
