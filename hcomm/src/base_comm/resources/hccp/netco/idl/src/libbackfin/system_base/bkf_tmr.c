/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_tmr.h"
#include "v_stringlib.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#pragma pack(4)

struct tagBkfTmrMng {
    BkfITmr ITmr;
    char name[0];
};

#pragma pack()

/* func */
BkfTmrMng *BkfTmrInit(BkfITmr *ITmr)
{
    BkfTmrMng *tmrMng = VOS_NULL;
    uint32_t strLen;
    uint32_t len;
    int32_t err;

    if ((ITmr == VOS_NULL) || (ITmr->name == VOS_NULL) || (ITmr->memMng == VOS_NULL) || (ITmr->startOnce == VOS_NULL) ||
        (ITmr->startLoop == VOS_NULL) || (ITmr->stop == VOS_NULL)) {
        goto error;
    }

    strLen = VOS_StrLen(ITmr->name);
    len = sizeof(BkfTmrMng) + strLen + 1;
    tmrMng = BKF_MALLOC(ITmr->memMng, len);
    if (tmrMng == VOS_NULL) {
        goto error;
    }
    (void)memset_s(tmrMng, len, 0, len);
    tmrMng->ITmr = *ITmr;
    err = snprintf_truncated_s(tmrMng->name, strLen + 1, "%s", ITmr->name);
    if (err < 0) {
        goto error;
    }
    tmrMng->ITmr.name = tmrMng->name;

    return tmrMng;

error:

    BkfTmrUninit(tmrMng);
    return VOS_NULL;
}

void BkfTmrUninit(BkfTmrMng *tmrMng)
{
    if ((tmrMng == VOS_NULL) || (tmrMng->ITmr.memMng == VOS_NULL)) {
        return;
    }

    BKF_FREE(tmrMng->ITmr.memMng, tmrMng);
    return;
}

BkfTmrId *BkfTmrStartOnce(BkfTmrMng *tmrMng, F_BKF_TMR_TIMEOUT_PROC proc, uint32_t intervalMs, void *param)
{
    if ((tmrMng == VOS_NULL) || (proc == VOS_NULL)) {
        return VOS_NULL;
    }

    return tmrMng->ITmr.startOnce(tmrMng->ITmr.cookie, proc, intervalMs, param);
}

BkfTmrId *BkfTmrStartLoop(BkfTmrMng *tmrMng, F_BKF_TMR_TIMEOUT_PROC proc, uint32_t intervalMs, void *param)
{
    if ((tmrMng == VOS_NULL) || (proc == VOS_NULL)) {
        return VOS_NULL;
    }

    return tmrMng->ITmr.startLoop(tmrMng->ITmr.cookie, proc, intervalMs, param);
}

void BkfTmrStop(BkfTmrMng *tmrMng, BkfTmrId *tmrId)
{
    if ((tmrMng == VOS_NULL) || (tmrId == VOS_NULL)) {
        return;
    }

    tmrMng->ITmr.stop(tmrMng->ITmr.cookie, tmrId);
    return;
}

uint32_t BkfTmrRefresh(BkfTmrMng *tmrMng, BkfTmrId *tmrId, uint32_t newIntervalMs)
{
    if ((tmrMng == VOS_NULL) || (tmrId == VOS_NULL)) {
        return BKF_ERR;
    }

    if (tmrMng->ITmr.refreshOrNull == VOS_NULL) {
        return BKF_ERR;
    }
    return tmrMng->ITmr.refreshOrNull(tmrMng->ITmr.cookie, tmrId, newIntervalMs);
}

void *BkfTmrGetParamStart(BkfTmrMng *tmrMng, BkfTmrId *tmrId)
{
    if ((tmrMng == VOS_NULL) || (tmrId == VOS_NULL)) {
        return VOS_NULL;
    }

    if (tmrMng->ITmr.getParamStartOrNull == VOS_NULL) {
        return VOS_NULL;
    }
    return tmrMng->ITmr.getParamStartOrNull(tmrMng->ITmr.cookie, tmrId);
}

uint32_t BkfTmrGetRemainTime(BkfTmrMng *tmrMng, BkfTmrId *tmrId)
{
    if ((tmrMng == VOS_NULL) || (tmrId == VOS_NULL)) {
        return 0;
    }
    if (tmrMng->ITmr.getRemainTimeOrNull == VOS_NULL) {
        return 0;
    }
    return tmrMng->ITmr.getRemainTimeOrNull(tmrMng->ITmr.cookie, tmrId);
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
