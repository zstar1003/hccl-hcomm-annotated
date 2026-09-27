/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bifrost_cncoi_suber.h"
#include "bifrost_cncoi_suber_pri.h"
#include "bifrost_cncoi_svc.h"
#include "securec.h"

#if __cplusplus
extern "C" {
#endif

BifrostCncoiSuber *BifrostCncoiSuberInit(BifrostCncoiSuberInitArg *arg)
{
    if (arg == VOS_NULL) {
        return VOS_NULL;
    }
    uint32_t len = sizeof(BifrostCncoiSuber);
    BifrostCncoiSuber *bifrostCncoiSuber = BKF_MALLOC(arg->memMng, len);
    if (bifrostCncoiSuber == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(bifrostCncoiSuber, len, 0, len);
    bifrostCncoiSuber->argInit = *arg;
    BkfSuberInitArg initArg = {0};
    initArg.name = arg->name;
    initArg.idlVersionMajor = arg->idlVersionMajor;
    initArg.idlVersionMinor = arg->idlVersionMinor;
    initArg.subMod = arg->subMod;
    initArg.dbgOn = arg->dbgOn;
    initArg.memMng = arg->memMng;
    initArg.disp = arg->disp;
    initArg.log = arg->log;
    initArg.pfm = arg->pfm;
    initArg.xMap = arg->xMap;
    initArg.tmrMng = arg->tmrMng;
    initArg.jobMng = arg->jobMng;
    initArg.jobTypeId1 = arg->jobTypeId1;
    initArg.jobPrioH = arg->jobPrioH;
    initArg.jobPrioL = arg->jobPrioL;
    initArg.sysLogMng = arg->sysLogMng;
    initArg.chCliMng = arg->chCliMng;
    initArg.sliceKeyLen = arg->sliceKeyLen;
    initArg.sliceKeyCmp = arg->sliceKeyCmp;
    initArg.sliceKeyGetStrOrNull = arg->sliceKeyGetStrOrNull;
    initArg.sliceKeyCodec = arg->sliceKeyCodec;
    initArg.svcName = BIFROST_CNCOI_SVC_NAME;
    initArg.lsnUrl = arg->lsnUrl;
    bifrostCncoiSuber->suber = BkfSuberInit(&initArg);
    if (bifrostCncoiSuber->suber == VOS_NULL) {
        goto error;
    }
    return bifrostCncoiSuber;

error:
    BifrostCncoiSuberUninit(bifrostCncoiSuber);
    return VOS_NULL;
}

void BifrostCncoiSuberUninit(BifrostCncoiSuber *bifrostCncoiSuber)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return;
    }

    if (bifrostCncoiSuber->suber != VOS_NULL) {
        BkfSuberUninit(bifrostCncoiSuber->suber);
    }

    BKF_FREE(bifrostCncoiSuber->argInit.memMng, bifrostCncoiSuber);
    return;
}

uint32_t BifrostCncoiSuberEnable(BifrostCncoiSuber *bifrostCncoiSuber)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfSuberEnable(bifrostCncoiSuber->suber);
}

uint32_t BifrostCncoiSuberSetSelfUrl(BifrostCncoiSuber *bifrostCncoiSuber, BkfUrl *url)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfSuberSetSelfUrl(bifrostCncoiSuber->suber, url);
}

uint32_t BifrostCncoiSuberCreateSvcInst(BifrostCncoiSuber *bifrostCncoiSuber, uint32_t instId)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfSuberCreateSvcInst(bifrostCncoiSuber->suber, instId);
}

uint32_t BifrostCncoiSuberCreateSvcInstEx(BifrostCncoiSuber *bifrostCncoiSuber, uint64_t instId)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfSuberCreateSvcInstEx(bifrostCncoiSuber->suber, instId);
}

void BifrostCncoiSuberDeleteSvcInst(BifrostCncoiSuber *bifrostCncoiSuber, uint32_t instId)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return;
    }

    BkfSuberDeleteSvcInst(bifrostCncoiSuber->suber, instId);
    return;
}

void BifrostCncoiSuberDeleteSvcInstEx(BifrostCncoiSuber *bifrostCncoiSuber, uint64_t instId)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return;
    }

    BkfSuberDeleteSvcInstEx(bifrostCncoiSuber->suber, instId);
    return;
}

uint32_t BifrostCncoiSuberSetSvcInstUrl(BifrostCncoiSuber *bifrostCncoiSuber, uint32_t instId, BkfUrl *puberUrl)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfSuberSetSvcInstUrl(bifrostCncoiSuber->suber, instId, puberUrl);
}

uint32_t BifrostCncoiSuberSetSvcInstUrlEx(BifrostCncoiSuber *bifrostCncoiSuber, uint64_t instId, BkfUrl *puberUrl)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfSuberSetSvcInstUrlEx(bifrostCncoiSuber->suber, instId, puberUrl);
}

uint32_t BifrostCncoiSuberSetSvcInstLocalUrl(BifrostCncoiSuber *bifrostCncoiSuber, uint64_t instId, BkfUrl *localUrl)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfSuberSetSvcInstLocalUrl(bifrostCncoiSuber->suber, instId, localUrl);
}

void BifrostCncoiSuberUnsetSvcInstUrl(BifrostCncoiSuber *bifrostCncoiSuber, uint32_t instId)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return;
    }

    BkfSuberUnSetSvcInstUrl(bifrostCncoiSuber->suber, instId);
    return;
}

void BifrostCncoiSuberUnsetSvcInstUrlEx(BifrostCncoiSuber *bifrostCncoiSuber, uint64_t instId)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return;
    }

    BkfSuberUnSetSvcInstUrlEx(bifrostCncoiSuber->suber, instId);
    return;
}

void BifrostCncoiSuberUnsetSvcInstLocalUrl(BifrostCncoiSuber *bifrostCncoiSuber, uint64_t instId)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return;
    }
    BkfSuberUnSetSvcInstLocalUrl(bifrostCncoiSuber->suber, instId);
    return;
}

uint32_t BifrostCncoiSuberBindSliceInstId(BifrostCncoiSuber *bifrostCncoiSuber, BifrostCncoiSliceKeyT *sliceKey,
    uint32_t instId)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfSuberSetSliceInstId(bifrostCncoiSuber->suber, sliceKey, instId);
}

uint32_t BifrostCncoiSuberBindSliceInstIdEx(BifrostCncoiSuber *bifrostCncoiSuber, BifrostCncoiSliceKeyT *sliceKey,
    uint64_t instId)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfSuberSetSliceInstIdEx(bifrostCncoiSuber->suber, sliceKey, instId);
}

void BifrostCncoiSuberUnbindSliceInstId(BifrostCncoiSuber *bifrostCncoiSuber, BifrostCncoiSliceKeyT *sliceKey)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return;
    }
    BkfSuberUnsetSliceInstId(bifrostCncoiSuber->suber, sliceKey);
    return;
}

void BifrostCncoiSuberUnbindSliceSpecInstId(BifrostCncoiSuber *bifrostCncoiSuber, BifrostCncoiSliceKeyT *sliceKey,
    uint64_t instId)
{
    if (bifrostCncoiSuber == VOS_NULL) {
        return;
    }
    BkfSuberUnsetSliceSpecInstId(bifrostCncoiSuber->suber, sliceKey, instId);
    return;
}

#if __cplusplus
}
#endif
