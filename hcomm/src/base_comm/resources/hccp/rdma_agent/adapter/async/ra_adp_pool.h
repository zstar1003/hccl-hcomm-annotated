/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_ADP_POOL_H
#define RA_ADP_POOL_H

#include <pthread.h>

#define SHUTDOWN_SIGNAL 1U

typedef int (*TaskFuncT)(unsigned int chipId, void *recvBuf, unsigned int recvLen);

struct RaHdcTask {
    TaskFuncT func;
    struct {
        unsigned int chipId;
        void *recvBuf;
        unsigned int recvLen;
    } args;
};

struct RaHdcThreadPool {
    struct RaHdcTask *taskQueue;
    unsigned int queueSize;

    unsigned int taskNum;
    unsigned int queuePi;
    unsigned int queueCi;

    pthread_t *workerThreads;
    unsigned int threadNum;
    pthread_mutex_t poolMutex;
    pthread_cond_t condition;

    unsigned int shutdown;
};

struct RaHdcThreadPool *RaHdcPoolCreate(unsigned int queueSize, unsigned int threadNum);
int RaHdcPoolDestroy(struct RaHdcThreadPool *pool);
void RaHdcPoolAddTask(struct RaHdcThreadPool *pool, TaskFuncT func, unsigned int chipId, void *recvBuf,
    unsigned int recvLen);
#endif // RA_ADP_POOL_H
