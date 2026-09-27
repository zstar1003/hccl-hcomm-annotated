/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bifrost_cncoi_puber.h"
#include "bifrost_cncoi_puber_pri.h"
#include "bifrost_cncoi_svc.h"
#include "bkf_puber.h"
#include "securec.h"

#if __cplusplus
extern "C" {
#endif

BifrostCncoiPuber *BifrostCncoiPuberInit(BifrostCncoiPuberInitArg *arg)
{
    BifrostCncoiPuber *bifrostCncoiPuber = VOS_NULL;
    uint32_t len;
    BkfPuberInitArg puberArg = {0};
    if (arg == VOS_NULL) {
        goto error;
    }

    len = sizeof(BifrostCncoiPuber);
    bifrostCncoiPuber = BKF_MALLOC(arg->memMng, len);
    if (bifrostCncoiPuber == VOS_NULL) {
        goto error;
    }
    (void)memset_s(bifrostCncoiPuber, len, 0, len);
    bifrostCncoiPuber->argInit = *arg;

    puberArg.name = arg->name;
    puberArg.svcName = BIFROST_CNCOI_SVC_NAME;
    puberArg.idlVersionMajor = arg->idlVersionMajor;
    puberArg.idlVersionMinor = arg->idlVersionMinor;
    puberArg.dbgOn = arg->dbgOn;
    puberArg.memMng = arg->memMng;
    puberArg.disp = arg->disp;
    puberArg.log = arg->log;
    puberArg.pfm = arg->pfm;
    puberArg.xMap = arg->xMap;
    puberArg.tmrMng = arg->tmrMng;
    puberArg.jobMng = arg->jobMng;
    puberArg.sysLogMng = arg->sysLogMng;
    puberArg.dc = arg->dc;
    puberArg.jobTypeId = arg->jobTypeId;
    puberArg.jobPrio = arg->jobPrio;
    puberArg.cookie = arg->cookie;
    puberArg.verifyMayAccelerate = arg->verifyMayAccelerate;
    puberArg.chMng = arg->chMng;
    puberArg.connCntMax = arg->connCntMax;
    puberArg.onConnect = arg->onConnect;
    puberArg.onDisConnect = arg->onDisconnect;
    puberArg.onConnectOver = arg->onConnectOver;
    bifrostCncoiPuber->puber = BkfPuberInit(&puberArg);
    if (bifrostCncoiPuber->puber == VOS_NULL) {
        goto error;
    }
    return bifrostCncoiPuber;

error:

    BifrostCncoiPuberUninit(bifrostCncoiPuber);
    return VOS_NULL;
}

void BifrostCncoiPuberUninit(BifrostCncoiPuber *bifrostCncoiPuber)
{
    if (bifrostCncoiPuber == VOS_NULL) {
        return;
    }

    if (bifrostCncoiPuber->puber != VOS_NULL) {
        BkfPuberUninit(bifrostCncoiPuber->puber);
    }
    BKF_FREE(bifrostCncoiPuber->argInit.memMng, bifrostCncoiPuber);
    return;
}

uint32_t BifrostCncoiPuberEnable(BifrostCncoiPuber *bifrostCncoiPuber)
{
    if (bifrostCncoiPuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfPuberEnable(bifrostCncoiPuber->puber);
}

uint32_t BifrostCncoiPuberSetSelfSvcInstId(BifrostCncoiPuber *bifrostCncoiPuber, uint32_t instId)
{
    if (bifrostCncoiPuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfPuberSetSelfSvcInstId(bifrostCncoiPuber->puber, instId);
}

uint32_t BifrostCncoiPuberSetSelfUrl(BifrostCncoiPuber *bifrostCncoiPuber, BkfUrl *url)
{
    if (bifrostCncoiPuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfPuberSetSelfUrl(bifrostCncoiPuber->puber, url);
}

void BifrostCncoiPuberUnsetSelfUrl(BifrostCncoiPuber *bifrostCncoiPuber, BkfUrl *url)
{
    if (bifrostCncoiPuber == VOS_NULL) {
        return;
    }

    return BkfPuberUnsetSelfUrl(bifrostCncoiPuber->puber, url);
}

uint32_t BifrostCncoiPuberCreateSlice(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiSliceKeyT *sliceKey)
{
    if (bifrostCncoiPuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfDcCreateSlice(bifrostCncoiPuber->argInit.dc, sliceKey);
}

void BifrostCncoiPuberDeleteSlice(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiSliceKeyT *sliceKey)
{
    if (bifrostCncoiPuber == VOS_NULL) {
        return;
    }

    BkfDcDeleteSlice(bifrostCncoiPuber->argInit.dc, sliceKey);
    return;
}

uint32_t BifrostCncoiPuberSetConnUpLimit(BifrostCncoiPuber *bifrostCncoiPuber, uint32_t connMax)
{
    if (bifrostCncoiPuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfPuberSetConnUpLimit(bifrostCncoiPuber->puber, connMax);
}
#if __cplusplus
}
#endif
