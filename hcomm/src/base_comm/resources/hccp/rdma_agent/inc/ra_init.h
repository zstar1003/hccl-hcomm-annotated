/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_INIT_H
#define RA_INIT_H
#include <pthread.h>

#define RA_MAX_INSTANCES RA_MAX_PHY_ID_NUM

/* instance is used for ra_is_first_used and ra_is_last_used */
typedef struct {
    int refCount;
    pthread_mutex_t mutex;
} RaInstance;

void RaRdevSetHandle(unsigned int phyId, void *rdmaHandle);
void RaRdevIncSendWrNum(void);
#endif // RA_INIT_H
