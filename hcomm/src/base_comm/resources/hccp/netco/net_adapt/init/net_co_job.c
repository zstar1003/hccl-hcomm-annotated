/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((co)->log)
#define BKF_MOD_NAME ((co)->name)
#include "net_co_job.h"
#include "net_co_main_data.h"
#include "wfw_job_by_mux.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif
uint32_t NetCoJobInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    char name[BKF_NAME_LEN_MAX + 1];
    int err = snprintf_truncated_s(name, sizeof(name), "%s_jobx", co->name);
    if (err <= 0) {
        return BKF_ERR;
    }
    WfwJobxInitArg arg = {.name = name,
        .dbgOn = co->dbgOn,
        .memMng = co->memMng,
        .disp = co->disp,
        .logCnt = co->logCnt,
        .log = co->log,
        .pfm = co->pfm,
        .mux = co->mux,
        .selfCid = co->selfCid,
        .runCostMax = BKF_JOB_RUN_COST_INVALID};
    co->jobMngAdpee = WfwJobxInit(&arg);
    if (co->jobMngAdpee == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "jobMngAdpee(%#x), ng\n", BKF_MASK_ADDR(co->jobMngAdpee));
        return BKF_ERR;
    }
    BkfIJob temp;
    co->jobMng = BkfJobInit(WfwJobxBuildIJob((WfwJobxMng *)co->jobMngAdpee, &temp));
    if (co->jobMng == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "jobMng(%#x), ng\n", BKF_MASK_ADDR(co->jobMng));
        return BKF_ERR;
    }

    return BKF_OK;
}

void NetCoJobUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "jobMng(%#x)/jobMngAdpee(%#x)\n", BKF_MASK_ADDR(co->jobMng),
        BKF_MASK_ADDR(co->jobMngAdpee));

    if (co->jobMng) {
        BkfJobUninit(co->jobMng);
        co->jobMng = VOS_NULL;
    }
    if (co->jobMngAdpee) {
        WfwJobxUninit((WfwJobxMng *)co->jobMngAdpee);
        co->jobMngAdpee = VOS_NULL;
    }
}

#ifdef __cplusplus
}
#endif
