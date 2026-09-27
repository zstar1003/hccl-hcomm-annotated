/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_TIMER_H
#define UDF_TIMER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(4)

#define UDF_TIMER_NAME_LEN_MAX 16
#define UDF_TIMER_DEPLOY_LEN_MAX 16
#define UDF_TIMER_LAST_PROC_TIME_LEN_MAX 32
#define UDF_TIMER_USR_FUNC_NAME_MAX 32

/**
 * @ingroup udf_timer
 * @brief 定时器句柄
 */
typedef void *UdfTimerHandle;

/**
 * @ingroup udf_timer
 * @brief 定时器回调函数原型
 *
 * @param[in] tmrHandle 定时器句柄
 * @param[in] rdVal     定时器超时重置时读取的值， resize 后 rdVal 会清零。
 * @param[in] param     透传在创建定时器时传入的参数
 */
typedef uint32_t (*UdfTimerProc)(UdfTimerHandle tmrHandle, uint64_t rdVal, void *param);

/**
 * @ingroup udf_timer
 * @brief 定时器用户回调以及回调参数
 * @attention param 生命周期由用户管理，定时器只做透传
 */
typedef struct {
    UdfTimerProc proc;
    void *param;
} UdfTimerUsrCb;

/**
 * @ingroup udf_timer
 * @brief 定时器状态
 */
typedef enum {
    UDF_TIMER_STATUS_STOPED, /* 停止中 */
    UDF_TIMER_STATUS_ACTIVE, /* 运行中 */
    UDF_TIMER_STATUS_BUTT
} UdfTimerStatus;

/**
 * @ingroup udf_timer
 * @brief 定时器模式
 */
typedef enum {
    UDF_TIMER_MODE_ONE_SHOT = 0, /* 单次定时器，定时器超时后资源自动回收 */
    UDF_TIMER_MODE_PERIOD,       /* 循环定时器 */
} UdfTimerMode;

/**
 * @ingroup udf_timer
 * @brief 定时器信息
 */
typedef struct {
    UdfTimerHandle handle;
    char timerName[UDF_TIMER_NAME_LEN_MAX];
    char lastTime[UDF_TIMER_LAST_PROC_TIME_LEN_MAX];
    char usrFunc[UDF_TIMER_USR_FUNC_NAME_MAX];
    uint32_t interval;
    UdfTimerMode mode;
    uint32_t maxProcTime;
    UdfTimerStatus status;
    uint64_t timeoutCnt;
} UdfTimerMntInfo;

/**
 * @ingroup udf_timer
 * @brief 定时器销毁用户析构函数回调原型
 * @param[in] param     透传在创建定时器时传入的参数
 */
typedef void (*TimerDestructProc)(void *param);

#pragma pack()

#ifdef __cplusplus
}
#endif

#endif
