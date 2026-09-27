/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_TWL_TMR_H
#define UDF_TWL_TMR_H

#include <stdint.h>
#include <sys/epoll.h>
#include "udf_timer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define UDF_TWL_TMR_NAME_LEN 16

#pragma pack(4)

/**
 * @ingroup udf_twltimer
 * @brief 时间轮定时器参数设置命令字
 */
typedef enum {
    UDF_TWL_TMR_ATTR_NAME = 0, /* (非必选项) 时间轮定时器名称, 类型:字符串,最大长度见模块规格文件 */
    UDF_TWL_TMR_ATTR_BUTT
} UdfTwlTimerAttrCmd;

typedef void (*TimerFdEventCB)(struct epoll_event *event, void *timerEventParam);
typedef struct {
    int operate;                /* epoll ctrl时的操作：EPOLL_CTL_ADD/EPOLL_CTL_MOD/EPOLL_CTL_DEL */
    TimerFdEventCB timerFdProc; /* timer lib的fd事件处理回调 */
    void *timerEventParam;      /* timer fd事件回调参数 */
    struct epoll_event event;   /* timer fd event */
} UdfTimerFdParams;

typedef void *(*TimerMallocCB)(void *handle, size_t size, const char *name, uint32_t line);
typedef void (*TimerFreeCB)(void *handle, void *ptr, const char *name, uint32_t line);
typedef uint32_t (*EpollCtrCallBack)(void *appHandle, uint32_t cid, int fd, UdfTimerFdParams *timerFdParam);

typedef struct {
    char timerInstName[UDF_TWL_TMR_NAME_LEN];
    void *appHandle; /* app的handle，不能为NULL，在注册timer fd事件和定时器超时回调时传给app */
    uint32_t appCid; /* app的cid，如果没有填0，在注册timer fd事件时传给app */
    void *memHandle; /* app内存操作handle，如果为NULL，则使用appHandle替代 */
    EpollCtrCallBack epollCtlCB; /* 由app在初始化时提供，用于timer lib将timer fd事件注册 */
    TimerMallocCB mallocCB;      /* 由app在初始化时提供，用于timer lib申请内存 */
    TimerFreeCB freeCB;          /* 由app在初始化时提供，用于timer lib释放内存 */
} UdfTimerInitParam;

/**
 * @ingroup udf_twltimer
 * @brief 时间轮定时器控制句柄
 */
typedef void *UdfTwlTimerHandle;

/**
 * @ingroup udf_twltimer
 * @brief 创建一个时间轮定时器控制句柄
 * @par 描述:时间轮定时器作为调度器的一个 task ，将加入到线程运行的调度器内统一调度
 * @attention: 该接口UdfTwlTimerCreate会创建一个timerfd
 * @param[in]  context      时间轮定时器部署的上下文
 * @param[in]  attr         时间轮定时器扩展属性
 * @param[out] handle       时间轮定时器控制句柄
 * @retval UDF_OK 成功
 * @retval UDF_ERROR 通用错误
 * @par 依赖
 * <ul><li> libudfsch.so：该接口所属的开发包。</li>
 * <li> udf_twl_timer.h：该接口声明所在的头文件。</li></ul>
 * @since V100R023C10
 */
uint32_t UdfTwlTimerCreate(UdfTimerInitParam *initParam, UdfTwlTimerHandle *handle);

/**
 * @ingroup udf_twltimer
 * @brief 销毁一个时间轮定时器控制句柄
 * @par 描述：将时间轮定时器从调度器中删除
 * @param[in] handle  时间轮定时器控制句柄
 * @par 依赖
 * <ul><li> libudfsch.so：该接口所属的开发包。</li>
 * <li> udf_twl_timer.h：该接口声明所在的头文件。</li></ul>
 * @since V100R023C10
 */
void UdfTwlTimerDestroy(UdfTwlTimerHandle handle);

/**
 * @ingroup udf_twltimer
 * @brief 时间轮定时器实例句柄。
 */
typedef void *UdfTmrInstHandle;

/**
 * @ingroup udf_twltimer
 * @brief 定时器实例用户回调函数原型。
 * @param[in] tmrHandle 定时器句柄
 * @param[in] rdVal     定时器超时重置时读取的值， resize 后 rdVal 会清零。
 * @param[in] param     透传在创建定时器时传入的参数
 */
typedef uint32_t (*UdfTmrInstCallBack)(UdfTmrInstHandle tmrHandle, void *param);

typedef struct {
    UdfTimerMode tmrMode;               /* 标识定时器周期性：单次或周期 */
    uint32_t interval;                  /* 定时器超时时间(ms) */
    char tmrName[UDF_TWL_TMR_NAME_LEN]; /* 定时器名称(16字节) */
    UdfTmrInstCallBack callBack;        /* 定时器超时回调 */
    void *usrData;                      /* 定时器超时回调参数 */
} UdfTmrInstParam;

/**
 * @ingroup udf_twltimer
 * @brief timer create
 * @param[in]  handle: Got by UdfTwlTimerCreate
 * @param[in]  timerParam
 * @param[out] tmrInstHandle Create one timer instance in timerHandler
 * @attention: 该接口UdfTimerInstCreate不会创建timerfd，会作为一个调度实例添加到handle对应的时间轮中。
               即handle与timerfd对应关系为1:1，tmrInstHandle与timerfd关系为n:1。
 * @retval UDF_OK 成功
 * @retval UDF_ERROR 通用错误
 * @par 依赖
 * <ul><li> libudfsch.so：该接口所属的开发包。</li>
 * <li> udf_twl_timer.h：该接口声明所在的头文件。</li></ul>
 * @since V100R023C10
 */
uint32_t UdfTimerInstCreate(UdfTwlTimerHandle handle, UdfTmrInstParam *timerParam, UdfTmrInstHandle *tmrInstHandle);

/**
 * @ingroup udf_twltimer
 * @brief timer instance delete
 * @param[in] tmrInst ： Got by UdfTimerInstCreate
 * @retval void
 * @par 依赖
 * <ul><li> libudfsch.so：该接口所属的开发包。</li>
 * <li> udf_twl_timer.h：该接口声明所在的头文件。</li></ul>
 * @since V100R023C10
 */
void UdfTimerInstDelete(UdfTmrInstHandle tmrInst);

/**
 * @ingroup udf_twltimer
 * @brief timer instance resize
 * @param[in] tmrInst ： Got by UdfTimerInstCreate
 * @retval UDF_OK 成功
 * @retval UDF_ERROR 通用错误
 * @par 依赖
 * <ul><li> libudfsch.so：该接口所属的开发包。</li>
 * <li> udf_twl_timer.h：该接口声明所在的头文件。</li></ul>
 * @since V100R023C10
 */
uint32_t UdfTimerInstResize(UdfTmrInstHandle tmrInst, uint32_t timeOutMs);
void *UdfTwlTimerGetInstUsrData(UdfTmrInstHandle tmrInst);
uint32_t UdfTwlTimerRunningInstGetCnt(UdfTwlTimerHandle timerHandle);

#pragma pack()

#ifdef __cplusplus
}
#endif

#endif /* End of File */
