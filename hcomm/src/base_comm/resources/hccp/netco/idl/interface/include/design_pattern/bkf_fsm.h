/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_FSM_H
#define BKF_FSM_H

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
 * @brief fsm模板。
 */
typedef struct tagBkfFsmTmpl BkfFsmTmpl;

/**
 * @brief fsm统计。
 */
typedef struct tagBkfFsmStatInfo BkfFsmStatInfo;

/**
* @brief fsm（实例）\n
1. 基于模板，会有多个fsm实例。
2. fsm内嵌在业务数据(appData)中。
*/
typedef struct tagBkfFsm {
    uint8_t sign;
    uint8_t state;
    uint16_t pad1;
    BkfFsmTmpl *tmpl;
    BkfFsmStatInfo *statInfo;
} BkfFsm;

/**
 * @brief state: [0, 0xff)
 */
#define BKF_FSM_STATE_INVALID ((uint8_t)0xff)

/* tmpl init */
/**
 * @brief 事件分发回调处理虚函数
 * @param[in] *dispatchParam1 分发参数，使用者自定义
 * @param[in] *dispatchParam2 分发参数，使用者自定义
 * @param[in] *dispatchParam3 分发参数，使用者自定义
 * @return 分发结果
 */
typedef uint32_t (*F_BKF_FSM_PROC)(void *dispatchParam1, void *dispatchParam2, void *dispatchParam3);

/**
 * @brief 事件分发回调处理
 */
typedef struct tagBkfFsmProcItem {
    F_BKF_FSM_PROC proc; /**< 回调接口 */
    const char *procStr; /**< 处理名称字符串，用于诊断 */
} BkfFsmProcItem;

/**
 * @brief 事件分发回调处理实体函数构造宏
 * @param[in] func app回调函数实体
 */
#define BKF_FSM_BUILD_PROC_ITEM(func) (F_BKF_FSM_PROC)(func), #func

/**
 * @brief 基于状态机获取app诊断虚函数
 * @param[in] *fsm 分发参数,状态机句柄
 * @param[in] *buf 分发参数,信息缓冲区
 * @param[in] bufLen 分发参数,缓冲区长度
 * @return 诊断信息字串
 */
typedef char *(*F_BKF_GET_APP_DATA_STR_BY_FSM_NODE)(BkfFsm *fsm, uint8_t *buf, int32_t bufLen);

/**
 * @brief 状态机模板初始化参数
 */
typedef struct tagBkfFsmTmplInitArg {
    char *name;        /**< 库名称 */
    BOOL dbgOn;        /**< 诊断开关 */
    BkfMemMng *memMng; /**< 内存库句柄,见bkf_mem.h,同一app内可复用 */
    BkfDisp *disp;     /**< 诊断显示库句柄,见bkf_disp.h,同一app内可复用 */
    BkfLog *log;       /**< log库句柄,见bkf_log.h,同一app内可复用 */

    uint8_t stateCnt;  /**< 状态数量 */
    uint8_t stateInit; /**< 初始状态 */
    uint8_t evtCnt;    /**< 事件数量 */
    uint8_t pad1[1];
    const char **stateStrTbl;                               /**< 状态字串表 */
    const char **evtStrTbl;                                 /**< 事件字串表 */
    const BkfFsmProcItem *stateEvtProcItemMtrx;             /**< 状态事件分发表 */
    F_BKF_GET_APP_DATA_STR_BY_FSM_NODE getAppDataStrOrNull; /**< 诊断信息获取接口 */
    uint8_t rsv[0x10];
} BkfFsmTmplInitArg;

/**
 * @brief 状态机中不可能事件处理
 */
#define BKF_FSM_PROC_IMPOSSIBLE ((F_BKF_FSM_PROC)(VOS_NULL))

/**
 * @brief 状态机中忽略事件处理
 */
#define BKF_FSM_PROC_IGNORE ((F_BKF_FSM_PROC)(1))

/* fsm dispatch */
/**
 * @brief 状态机分发遇到不可能事件返回值
 */
#define BKF_FSM_DISPATCH_RET_IMPOSSIBLE ((uint32_t)1380202103)

/* func */
/**
 * @brief 状态机模板初始化
 *
 * @param[in] *arg 参数
 * @return 模板是否创建成功
 *   @retval 非空 成功
 *   @retval 空 失败
 */
BkfFsmTmpl *BkfFsmTmplInit(BkfFsmTmplInitArg *arg);

/**
 * @brief 状态机模板反初始化
 *
 * @param[in] *fsmTmpl 状态机模板句柄
 * @return none
 */
void BkfFsmTmplUninit(BkfFsmTmpl *fsmTmpl);

/**
 * @brief 获取状态机状态诊断信息
 *
 * @param[in] *fsmTmpl 状态机模板句柄
 * @param[in] state 状态
 * @return 诊断信息
 *   @retval 非空 诊断信息字串。如果内部判断错误，返回一个常量串。所以不会返回空。
 */
const char *BkfFsmTmplGetStateStr(BkfFsmTmpl *fsmTmpl, uint8_t state);

/**
 * @brief 获取状态机事件诊断信息
 *
 * @param[in] *fsmTmpl 状态机模板句柄
 * @param[in] evt 事件
 * @return 诊断信息
 *   @retval 非空 诊断信息字串。如果内部判断错误，返回一个常量串。所以不会返回空。
 */
const char *BkfFsmTmplGetEvtStr(BkfFsmTmpl *fsmTmpl, uint8_t evt);

/**
 * @brief 状态机初始化。
 *
 * @param[in] *fsm 状态机（实例）。内嵌在业务数据（appData）中。
 * @param[in] *fsmTmpl 状态机模板。
 * @return 状态机初始化是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfFsmInit(BkfFsm *fsm, BkfFsmTmpl *fsmTmpl);

/**
 * @brief 状态机反初始化。
 *
 * @param[in] *fsm 状态机（实例）。
 * @return none
 */
void BkfFsmUninit(BkfFsm *fsm);

/**
 * @brief 状态机事件派发。一般不直接使用。@see BKF_FSM_DISPATCH
 *
 * @param[in] *fsm 状态机（实例）。
 * @param[in] evt 事件
 * @param[in] dispatchParam1 参数1，用于状态机的状态处理函数。@see F_BKF_FSM_PROC
 * @param[in] dispatchParam2 参数2，同上。
 * @param[in] dispatchParam3 参数3，同上。
 * @return 事件派发结果
 *   @retval BKF_FSM_DISPATCH_RET_IMPOSSIBLE 不可能的派发。代表一种错误。业务需要考虑状态机的设计是否正确，并对
                                             这种返回值做一定的处理（比如重置状态机）。
 *   @retval BKF_OK 正常
 *   @retval 其他 其他业务处理的结果情况
 */
uint32_t BkfFsmDispatch(BkfFsm *fsm, uint8_t evt, void *dispatchParam1, void *dispatchParam2, void *dispatchParam3);

/**
 * @brief 获取状态机当前状态。一般不直接使用。@see BKF_FSM_GET_STATE
 *
 * @param[in] *fsm 状态机（实例）。
 * @return 状态机当前状态
 *   @retval BKF_FSM_STATE_INVALID 无效状态
 *   @retval 其他 状态机状态值
 */
uint8_t BkfFsmGetState(BkfFsm *fsm);

/**
 * @brief 改变状态机当前状态。一般不直接使用。@see BKF_FSM_CHG_STATE
 *
 * @param[in] *fsm 状态机（实例）。
 * @param[in] newState 新状态
 * @return 改变状态是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfFsmChgState(BkfFsm *fsm, uint8_t newState);

/**
 * @brief 监控状态机改变。用于测试。
 *
 * @param[in] *fsm 状态机（实例）。
 * @param[in] newState 新状态
 * @return 无
 */
void BkfFsmStateChgSpy(BkfFsm *fsm, uint8_t newState);

/**
 * @brief 获取状态机状态信息
 *
 * @param[in] *fsm 状态机（实例）。
 * @return 诊断信息
 *   @retval 非空 诊断信息字串。如果内部判断错误，返回一个常量串。所以不会返回空。
 */
const char *BkfFsmGetStateStr(BkfFsm *fsm);

/**
 * @brief 获取状态机诊断信息
 *
 * @param[in] *fsm 状态机（实例）。
 * @param[in] *buf 信息缓冲
 * @param[in] bufLen 缓冲长度
 * @return 诊断信息
 *   @retval 非空 诊断信息字串。如果内部判断错误，返回一个常量串。所以不会返回空。
 */
char *BkfFsmGetStr(BkfFsm *fsm, uint8_t *buf, int32_t bufLen);

/* 宏，便于使用 */
/**
 * @brief 状态机事件派发宏封装。@see BkfFsmDispatch
 */
#define BKF_FSM_DISPATCH(fsm, evt, dispatchParam1, dispatchParam2, dispatchParam3)                                     \
    BkfFsmDispatch((fsm), (evt), (dispatchParam1), (dispatchParam2), (dispatchParam3))

/**
 * @brief 获取状态机当前状态宏封装。@see BkfFsmGetState
 */
#define BKF_FSM_GET_STATE(fsm) BkfFsmGetState(fsm)

/**
 * @brief 改变状态机当前状态宏封装。@see BkfFsmChgState
 */
#define BKF_FSM_CHG_STATE(fsm, newState) BkfFsmChgState((fsm), (newState))

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
