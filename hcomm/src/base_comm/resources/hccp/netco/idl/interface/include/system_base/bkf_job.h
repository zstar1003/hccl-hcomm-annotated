/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_JOB_H
#define BKF_JOB_H

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
typedef struct tagBkfJobMng BkfJobMng;
typedef struct tagBkfJobId BkfJobId;

/** job proc返回BKF_JOB_CONTINUE， 下次会再调度这个job。@see F_BKF_JOB_PROC */
#define BKF_JOB_CONTINUE 0x11
/** job proc返回BKF_JOB_FINSIH，下次不会再调度这个job。@see F_BKF_JOB_PROC */
#define BKF_JOB_FINSIH 0
/** 无效cost，初始化为此值，代表不使用runCost */
#define BKF_JOB_RUN_COST_INVALID 0xffffffff

/**
 * @brief job 回调
 *
 * @param[in] *paramJobCreate BkfJobCreate中填写param
 * @param[in] *runCost 权重。如果回调中返回finish累计的cost达到初始化设定的最大值，此次job消息不再调度其他job。
 * @return job是否执行完毕
 *   @retval BKF_JOB_CONTINUE 没有完毕，需要继续调度
 *   @retval BKF_JOB_FINSIH 完毕，不需要继续调度
 * @note 因为系统实现原因，需要“**全局**”规划出type出来，并且业务回调相关是挂在type上的。
 *       注意，由于实现的约束，在job回调中删除此job会有严重问题。
 */
typedef uint32_t (*F_BKF_JOB_PROC)(void *paramJobCreate, uint32_t *runCost);

/* reg */
#define BKF_JOB_PRIO_MIN 1
#define BKF_JOB_PRIO_MAX 8
#define BKF_JOB_PRIO_IS_VALID(prio) (((prio) >= BKF_JOB_PRIO_MIN) && ((prio) <= BKF_JOB_PRIO_MAX))
typedef uint32_t (*F_BKF_IJOB_REG_TYPE)(void *cookieIJob, uint32_t jobTypeId, F_BKF_JOB_PROC proc, uint32_t prio);
typedef void *(*F_BKF_IJOB_CREATE)(void *cookieIJob, uint32_t jobTypeId, char *name, void *param);
typedef void (*F_BKF_IJOB_DELETE)(void *cookieIJob, void *iJobId);
typedef struct tagBkfIJob {
    char *name;
    BkfMemMng *memMng;
    uint32_t runCostMax;
    void *cookie;
    F_BKF_IJOB_REG_TYPE regType;
    F_BKF_IJOB_CREATE createJobId;
    F_BKF_IJOB_DELETE deleteJobId;
    uint8_t rsv1[0x10];
} BkfIJob;

/* create */
#define BKF_JOB_NAME_LEN_MAX 19

/* func */
/**
 * @brief job库初始化
 *
 * @param[in] *IJob job接口(interface)
 * @return job库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfJobMng *BkfJobInit(BkfIJob *IJob);

/**
 * @brief job库反初始化
 *
 * @param[in] *jobMng job库句柄
 * @return none
 */
void BkfJobUninit(BkfJobMng *jobMng);

/**
 * @brief 注册job类型
 *
 * @param[in] *jobMng job库句柄
 * @param[in] jobTypeId job类型
 * @param[in] proc job回调函数
 * @param[in] prio 优先级，[BKF_JOB_PRIO_MIN, BKF_JOB_PRIO_MAX],值越大优先级越高
 * @return job类型注册结果
 *   @retval BKF_OK 注册成功
 *   @retval 其他 注册失败
 */
uint32_t BkfJobRegType(BkfJobMng *jobMng, uint32_t jobTypeId, F_BKF_JOB_PROC proc, uint32_t prio);

/**
 * @brief 创建job
 *
 * @param[in] *jobMng job库句柄
 * @param[in] jobTypeId job类型
 * @param[in] *name job名字
 * @param[in] *param 回调函数传回的参数
 * @return job创建结果，id号
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfJobId *BkfJobCreate(BkfJobMng *jobMng, uint32_t jobTypeId, char *name, void *param);

/**
 * @brief 删除job
 *
 * @param[in] *jobMng job库句柄
 * @param[in] *jobId job id
 * @return none
 */
void BkfJobDelete(BkfJobMng *jobMng, BkfJobId *jobId);

/**
 * @brief 获取设置的runCostMax
 *
 * @param[in] *jobMng job库句柄
 * @return runCostMax
 */
uint32_t BkfJobGetRunCostMax(BkfJobMng *jobMng);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
