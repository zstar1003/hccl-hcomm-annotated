/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sys/prctl.h>
#include <pthread.h>
#include "securec.h"
#include "user_log.h"
#include "config_log.h"
#include "ra_hdc.h"
#include "ra_rs_err.h"
#include "ra_adp_pool.h"

STATIC void *RaHdcWorkerThread(void *arg)
{
    struct RaHdcThreadPool *pool = (struct RaHdcThreadPool *)arg;
    pthread_t tidp = pthread_self();
    struct RaHdcTask task = {0};

    (void)prctl(PR_SET_NAME, (uintptr_t) "hccp_rs_work", 0, 0, 0);

    while (1) {
        RA_PTHREAD_MUTEX_LOCK(&pool->poolMutex);
        // block thread until task received
        while (pool->taskNum == 0 && pool->shutdown != SHUTDOWN_SIGNAL) {
            pthread_cond_wait(&pool->condition, &pool->poolMutex);
        }
        if (pool->shutdown == SHUTDOWN_SIGNAL) {
            pool->threadNum--;
            RA_PTHREAD_MUTEX_UNLOCK(&pool->poolMutex);
            break;
        }

        // consume a task
        task.func = pool->taskQueue[pool->queueCi].func;
        task.args.chipId = pool->taskQueue[pool->queueCi].args.chipId;
        task.args.recvBuf = pool->taskQueue[pool->queueCi].args.recvBuf;
        task.args.recvLen = pool->taskQueue[pool->queueCi].args.recvLen;
        pool->queueCi = (pool->queueCi + 1) % pool->queueSize;
        pool->taskNum--;

        // notify manager to produce
        pthread_cond_signal(&pool->condition);
        RA_PTHREAD_MUTEX_UNLOCK(&pool->poolMutex);

        // do task
        task.func(task.args.chipId, task.args.recvBuf, task.args.recvLen);
        free(task.args.recvBuf);
        task.args.recvBuf = NULL;
    }

    hccp_run_info("tidp:%ld exit", tidp);
    return NULL;
}

STATIC int RaHdcPoolMutexCondInit(struct RaHdcThreadPool *pool)
{
    int ret;

    ret = pthread_mutex_init(&pool->poolMutex, NULL);
    CHK_PRT_RETURN(ret != 0, hccp_err("pool_mutex mutex_init failed ret %d", ret), -ESYSFUNC);

    ret = pthread_cond_init(&pool->condition, NULL);
    if (ret != 0) {
        hccp_err("condition cond_init failed ret %d", ret);
        ret = -ESYSFUNC;
        goto deinit_pool_mutex;
    }

    return 0;

deinit_pool_mutex:
    pthread_mutex_destroy(&pool->poolMutex);
    return ret;
}

STATIC void RaHdcPoolMutexCondDeinit(struct RaHdcThreadPool *pool)
{
    pthread_cond_destroy(&pool->condition);
    pthread_mutex_destroy(&pool->poolMutex);
}

STATIC void RaHdcPoolFreeWorkers(struct RaHdcThreadPool *pool)
{
    int timeout = RA_THREAD_TRY_TIME;
    unsigned int i;

    RA_PTHREAD_MUTEX_LOCK(&pool->poolMutex);
    pool->shutdown = SHUTDOWN_SIGNAL;
    for (i = 0; i < pool->threadNum; i++) {
        pthread_cond_signal(&pool->condition);
    }
    RA_PTHREAD_MUTEX_UNLOCK(&pool->poolMutex);

    // wait for all threads exit until time out: RA_THREAD_TRY_TIME * RA_THREAD_SLEEP_TIME us
    while (pool->threadNum > 0 && timeout > 0) {
        usleep(RA_THREAD_SLEEP_TIME);
        timeout--;
    }
    if (pool->threadNum > 0 && timeout <= 0) {
        hccp_warn_others("destroy thread pool timeout, threadNum:%u > 0 and timeout:%d <= 0", pool->threadNum, timeout);
    }
}

struct RaHdcThreadPool *RaHdcPoolCreate(unsigned int queueSize, unsigned int threadNum)
{
    struct RaHdcThreadPool *pool = NULL;
    unsigned int i;
    int ret;

    pool = (struct RaHdcThreadPool *)calloc(1, sizeof(struct RaHdcThreadPool));
    CHK_PRT_RETURN(pool == NULL, hccp_err("calloc pool failed"), NULL);
    pool->taskQueue = (struct RaHdcTask *)calloc(queueSize, sizeof(struct RaHdcTask));
    if (pool->taskQueue == NULL) {
        hccp_err("calloc task_queue failed, queueSize:%u", queueSize);
        goto free_pool;
    }
    pool->queueSize = queueSize;

    ret = RaHdcPoolMutexCondInit(pool);
    if (ret != 0) {
        hccp_err("ra_hdc_pool_mutex_cond_init failed, ret:%d", ret);
        goto free_queue;
    }

    pool->workerThreads = (pthread_t *)calloc(threadNum, sizeof(pthread_t));
    if (pool->workerThreads == NULL) {
        hccp_err("calloc worker_threads failed, threadNum:%u", threadNum);
        goto free_cond;
    }
    for (i = 0; i < threadNum; i++) {
        ret = pthread_create(&pool->workerThreads[i], NULL, (void *)RaHdcWorkerThread, pool);
        if (ret != 0) {
            hccp_err("Create pthread i:%u failed, ret:%d", i, ret);
            pool->threadNum = i;
            goto free_thread;
        }
    }
    pool->threadNum = threadNum;

    return pool;
free_thread:
    RaHdcPoolFreeWorkers(pool);
    free(pool->workerThreads);
    pool->workerThreads = NULL;
free_cond:
    RaHdcPoolMutexCondDeinit(pool);
free_queue:
    free(pool->taskQueue);
    pool->taskQueue = NULL;
free_pool:
    free(pool);
    pool = NULL;
    return NULL;
}

void RaHdcPoolAddTask(struct RaHdcThreadPool *pool, TaskFuncT func, unsigned int chipId, void *recvBuf,
    unsigned int recvLen)
{
    RA_PTHREAD_MUTEX_LOCK(&pool->poolMutex);
    // block until task can be received
    while (pool->taskNum == pool->queueSize) {
        pthread_cond_wait(&pool->condition, &pool->poolMutex);
    }

    // produce a task
    pool->taskQueue[pool->queuePi].func = func;
    pool->taskQueue[pool->queuePi].args.chipId = chipId;
    pool->taskQueue[pool->queuePi].args.recvBuf = recvBuf;
    pool->taskQueue[pool->queuePi].args.recvLen = recvLen;
    pool->queuePi = (pool->queuePi + 1) % pool->queueSize;
    pool->taskNum++;

    // notify worker to consume
    pthread_cond_signal(&pool->condition);
    RA_PTHREAD_MUTEX_UNLOCK(&pool->poolMutex);
}

int RaHdcPoolDestroy(struct RaHdcThreadPool *pool)
{
    CHK_PRT_RETURN(pool == NULL, hccp_err("param invalid, pool is NULL"), -EINVAL);

    RaHdcPoolFreeWorkers(pool);
    if (pool->taskQueue != NULL) {
        free(pool->taskQueue);
        pool->taskQueue = NULL;
    }
    if (pool->workerThreads) {
        free(pool->workerThreads);
        pool->workerThreads = NULL;
    }
    RaHdcPoolMutexCondDeinit(pool);
    free(pool);
    pool = NULL;
    return 0;
}
