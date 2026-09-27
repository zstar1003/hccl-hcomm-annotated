/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WFW_JOB_BY_MUX_H
#define WFW_JOB_BY_MUX_H

#include "wfw_mux.h"
#include "bkf_disp.h"
#include "bkf_log_cnt.h"
#include "bkf_log.h"
#include "bkf_pfm.h"
#include "bkf_job.h"

/*
1. 业务回调过程中删除当前 jobId 节点，或者回调结束返回 finish (*) ，都认为当前的job节点不需要再调度，会删除当前节点
   此算法对齐v8原先实现
2. 调度算法做了简化，并不对齐原先算法。
具体来说，job的调度算法仅对优先级、runCost和业务回调返回值做约定。
未明确说明的，认为是可能随时调整，业务不要对此做假定。
2.1 type有优先级：
   a) 不同优先级严格按照优先级调度（运行期不做动态提权操作）
2.2 对业务回调后返回的处理：
   a) 如果当前jobId不再需要调度：
      a-1) 如果不使用runCost，则放权
      a-2) 否则，只有等累计的runCost大于runCostMax，才放权
   b) 否则，如果业务回调返回 continue， 放权 （此处算法做了简化）
   c) 否则，不放权
*/

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */

/**
 * @brief WfwJobxMng库句柄，基于mux实现job
 */
typedef struct tagWfwJobxMng WfwJobxMng;

/* init */
/**
 * @brief WfwJobxMng库句柄，初始化参数
 */
typedef struct tagWfwJobxInitArg {
    char *name;
    BOOL dbgOn;
    BkfMemMng *memMng;
    BkfDisp *disp;
    BkfLogCnt *logCnt;
    BkfLog *log;
    BkfPfm *pfm;
    WfwMux *mux;
    uint32_t selfCid;
    uint32_t runCostMax;
    uint8_t rsv1[0x10];
} WfwJobxInitArg;

/* func */
/**
 * @brief WfwJobxMng库初始化
 *
 * @param[in] *arg 参数
 * @return WfwJobxMng库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
WfwJobxMng *WfwJobxInit(WfwJobxInitArg *arg);

/**
 * @brief WfwJobxMng库反初始化
 *
 * @param[in] *jobxMng WfwJobxMng库句柄
 * @return none
 */
void WfwJobxUninit(WfwJobxMng *jobxMng);

/**
 * @brief 注入WFW实现的job回调函数
 *
 * @param[in] *jobxMng WfwJobxMng库句柄
 * @param[in] *temp 注入Job结构体指针，堆变量或栈变量均可
 * @return 构造结果
 *   @retval VOS_NULL 失败
 *   @retval 其他 成功
 */
BkfIJob *WfwJobxBuildIJob(WfwJobxMng *jobxMng, BkfIJob *temp);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
