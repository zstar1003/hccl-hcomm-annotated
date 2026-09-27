/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_XMAP_H
#define BKF_XMAP_H

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
typedef struct tagBkfXMap BkfXMap;
typedef uint32_t (*F_BKF_XMAP_PROC)(void *cookieReg, void *paramDispatch1, void *paramDispatch2);
typedef uint32_t (*F_BKF_XMAP_PROC_NO_COOKIE)(void *paramDispatch1, void *paramDispatch2);

/* init */
typedef struct tagBkfXMapInitArg {
    char *name;
    BkfMemMng *memMng;
    uint8_t rsv1[0x10];
} BkfXMapInitArg;

#define BKF_XMAP_DISPATCH_NOT_FIND ((uint32_t)1360202108)

#define BKF_XMAP_REG_PRIO_DFT ((uint8_t)128)

/* func */
/**
 * @brief 注册派发库初始化
 *
 * @param[in] *arg 初始化参数
 * @return 注册派发库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfXMap *BkfXMapInit(BkfXMapInitArg *arg);

/**
 * @brief 注册派发反初始化
 *
 * @param[in] *xMap 注册派发库
 * @return none
 */
void BkfXMapUninit(BkfXMap *xMap);

/**
 * @brief 注册有cookie的回调
 *
 * @param[in] *xMap 注册派发库句柄
 * @param[in] key 注册key
 * @param[in] *keyStr key字串信息，用于诊断。一定是常量串，不能释放。
 * @param[in] proc 接收到key之后，派发回调函数。
 * @param[in] *cookie 注册参数
 * @param[in] prio 优先级，影响回调顺序。值越大优先级越高，回调越先被调用, 相同优先级随机。
 * @return 注册结果
 *   @retval BKF_OK 注册成功
 *   @retval 其他 注册失败
 * @note proc和paramReg构成了注册实例的标识。
 */
uint32_t BkfXMapRegEx(BkfXMap *xMap, uintptr_t key, const char *keyStr, F_BKF_XMAP_PROC proc, void *cookie,
    uint8_t prio);

/**
 * @brief 去注册有cookie的回调
 *
 * @param[in] *xMap 注册派发库句柄
 * @param[in] key 注册key
 * @param[in] *keyStr key字串信息，用于诊断。一定是常量串，不能释放。
 * @param[in] proc 接收到key之后，派发回调函数。
 * @param[in] *cookie 注册参数
 * @param[in] prio 优先级，影响回调顺序。值越大优先级越高，回调越先被调用, 相同优先级随机。
 * @return None
 * @note proc和paramReg构成了注册实例的标识。
 */
void BkfXMapUnRegEx(BkfXMap *xMap, uintptr_t key, F_BKF_XMAP_PROC proc, void *cookie, uint8_t prio);
/**
 * @brief 注册无cookie的回调
 *
 * @param[in] *xMap 注册派发库句柄
 * @param[in] key 注册key
 * @param[in] *keyStr key字串信息，用于诊断。一定是常量串，不能释放。
 * @param[in] proc 接收到key之后，派发回调函数。
 * @param[in] prio 优先级，影响回调顺序。值越大优先级越高，回调越先被调用, 相同优先级随机。
 * @return 注册结果
 *   @retval BKF_OK 注册成功
 *   @retval 其他 注册失败
 * @note proc和paramReg构成了注册实例的标识。
 */
uint32_t BkfXMapRegExNoCookie(BkfXMap *xMap, uintptr_t key, const char *keyStr, F_BKF_XMAP_PROC_NO_COOKIE proc,
    uint8_t prio);

/**
 * @brief 派发
 *
 * @param[in] *xMap 注册派发库句柄
 * @param[in] key 派发key
 * @param[in] *paramDispatch 派发参数
 * @return 派发结果
 *   @retval 多种返回
 * 值取决于业务proc回调返回结果。如果有多个回调，是所有回调返回的或。因此，需要协调好所有的回调的返回值。
 *                    一般来说，外部不关注返回值。
 */
uint32_t BkfXMapDispatch(BkfXMap *xMap, uintptr_t key, void *paramDispatch1, void *paramDispatch2);

/* 宏，方便使用 */
/* msg */
/**
 * @brief 注册消息, 基本
 *
 * @param[in] *xMap 注册派发库句柄
 * @param[in] msg 注册消息，长度不超过4字节
 * @param[in] proc 接收到key之后，派发回调函数
 * @param[in] *paramReg 注册参数
 * @return 注册结果
 *   @retval BKF_OK 注册成功
 *   @retval 其他 注册失败
 */
#define BKF_MSG_REG(xMap, msgId, proc, cookie)                                                                         \
    BkfXMapRegEx((xMap), (uintptr_t)(msgId), (#msgId), (F_BKF_XMAP_PROC)(proc), (cookie), BKF_XMAP_REG_PRIO_DFT)

/* 宏，方便使用 */
/* msg */
/**
 * @brief 去注册消息, 基本
 *
 * @param[in] *xMap 注册派发库句柄
 * @param[in] msg 注册消息，长度不超过4字节
 * @param[in] proc 接收到key之后，派发回调函数
 * @param[in] *paramReg 注册参数
 * @return NONE
 */
#define BKF_MSG_UNREG(xMap, msgId, proc, cookie)                                                                       \
    BkfXMapUnRegEx((xMap), (uintptr_t)(msgId), (F_BKF_XMAP_PROC)(proc), (cookie), BKF_XMAP_REG_PRIO_DFT)

/**
 * @brief 注册消息
 *
 * @param[in] *xMap 注册派发库句柄
 * @param[in] msg 注册消息，长度不超过4字节
 * @param[in] proc 接收到key之后，派发回调函数
 * @param[in] *paramReg 注册参数
 * @param[in] prio 优先级
 * @return 注册结果
 *   @retval BKF_OK 注册成功
 *   @retval 其他 注册失败
 */
#define BKF_MSG_REG_EX(xMap, msgId, proc, cookie, prio)                                                                \
    BkfXMapRegEx((xMap), (uintptr_t)(msgId), (#msgId), (F_BKF_XMAP_PROC_NO_COOKIE)(proc), (cookie), (prio))

/**
 * @brief 派发消息
 *
 * @param[in] *xMap 注册派发库句柄
 * @param[in] msgId 派发消息
 * @param[in] *paramDispatch 派发参数
 * @return 派发结果
 *   @retval BKF_OK 派发成功
 *   @retval 其他 派发失败
 */
#define BKF_MSG_DISPATCH(xMap, msgId, paramDispatch1, paramDispatch2)                                                  \
    BkfXMapDispatch((xMap), (uintptr_t)(msgId), (paramDispatch1), (paramDispatch2))

/* job */
/**
 * @brief 注册job, 基本
 *
 * @param[in] *xMap 注册派发库句柄
 * @param[in] jobTypeId 注册job类型，长度不超过4字节
 * @param[in] proc 接收到key之后，派发回调函数
 * @return 注册结果
 *   @retval BKF_OK 注册成功
 *   @retval 其他 注册失败
 */
#define BKF_JOB_REG(xMap, jobTypeId, proc)                                                                             \
    BkfXMapRegExNoCookie((xMap), (uintptr_t)(jobTypeId), (#jobTypeId), (F_BKF_XMAP_PROC_NO_COOKIE)(proc),              \
        BKF_XMAP_REG_PRIO_DFT)

/**
 * @brief 注册job
 *
 * @param[in] *xMap 注册派发库句柄
 * @param[in] msg 注册消息，长度不超过4字节
 * @param[in] proc 接收到key之后，派发回调函数
 * @param[in] *paramReg 注册参数
 * @param[in] prio 优先级
 * @return 注册结果
 *   @retval BKF_OK 注册成功
 *   @retval 其他 注册失败
 */
#define BKF_JOB_REG_EX(xMap, jobTypeId, proc, prio)                                                                    \
    BkfXMapRegExNoCookie((xMap), (uintptr_t)(jobTypeId), (#jobTypeId), (F_BKF_XMAP_PROC_NO_COOKIE)(proc), (prio))

/**
 * @brief 派发job
 *
 * @param[in] *xMap 注册派发库句柄
 * @param[in] jobTypeId 派发job类型
 * @param[in] *paramDispatch 派发参数
 * @return 派发结果
 *   @retval BKF_OK 派发成功
 *   @retval 其他 派发失败
 */
#define BKF_JOB_DISPATCH(xMap, jobTypeId, paramDispatch1, paramDispatch2)                                              \
    BkfXMapDispatch((xMap), (uintptr_t)(jobTypeId), (paramDispatch1), (paramDispatch2))

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
