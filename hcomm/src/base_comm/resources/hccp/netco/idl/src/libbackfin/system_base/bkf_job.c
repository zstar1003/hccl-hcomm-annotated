/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_job.h"
#include "bkf_str.h"
#include "bkf_assert.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)

struct tagBkfJobMng {
    BkfIJob IJob;
    char *name;
};

#pragma pack()

/* func */
BkfJobMng *BkfJobInit(BkfIJob *IJob)
{
    BkfJobMng *jobMng = VOS_NULL;
    uint32_t len;

    if ((IJob == VOS_NULL) || (IJob->name == VOS_NULL) || (IJob->memMng == VOS_NULL) || (IJob->regType == VOS_NULL) ||
        (IJob->createJobId == VOS_NULL) || (IJob->deleteJobId == VOS_NULL)) {
        BKF_ASSERT(0);
        goto error;
    }

    len = sizeof(BkfJobMng);
    jobMng = BKF_MALLOC(IJob->memMng, len);
    if (jobMng == VOS_NULL) {
        BKF_ASSERT(0);
        goto error;
    }
    (void)memset_s(jobMng, len, 0, len);
    jobMng->IJob = *IJob;
    jobMng->name = BkfStrNew(IJob->memMng, "%s_job", IJob->name);
    jobMng->IJob.name = jobMng->name;
    return jobMng;
error:

    BkfJobUninit(jobMng);
    return VOS_NULL;
}

void BkfJobUninit(BkfJobMng *jobMng)
{
    if ((jobMng == VOS_NULL) || (jobMng->IJob.memMng == VOS_NULL)) {
        return;
    }
    BkfStrDel(jobMng->name);
    BKF_FREE(jobMng->IJob.memMng, jobMng);
    return;
}

uint32_t BkfJobRegType(BkfJobMng *jobMng, uint32_t jobTypeId, F_BKF_JOB_PROC proc, uint32_t prio)
{
    if ((jobMng == VOS_NULL) || (proc == VOS_NULL)) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    return jobMng->IJob.regType(jobMng->IJob.cookie, jobTypeId, proc, prio);
}

BkfJobId *BkfJobCreate(BkfJobMng *jobMng, uint32_t jobTypeId, char *name, void *param)
{
    if ((jobMng == VOS_NULL) || (name == VOS_NULL)) {
        BKF_ASSERT(0);
        return VOS_NULL;
    }

    return jobMng->IJob.createJobId(jobMng->IJob.cookie, jobTypeId, name, param);
}

void BkfJobDelete(BkfJobMng *jobMng, BkfJobId *jobId)
{
    if ((jobMng == VOS_NULL) || (jobId == VOS_NULL)) {
        BKF_ASSERT(0);
        return;
    }

    jobMng->IJob.deleteJobId(jobMng->IJob.cookie, jobId);
    return;
}

uint32_t BkfJobGetRunCostMax(BkfJobMng *jobMng)
{
    if (jobMng == VOS_NULL) {
        BKF_ASSERT(0);
        return 0;
    }

    return jobMng->IJob.runCostMax;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
