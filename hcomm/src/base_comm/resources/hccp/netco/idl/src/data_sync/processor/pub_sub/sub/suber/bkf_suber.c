/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_mem.h"
#include "bkf_subscriber.h"
#include "bkf_suber_private.h"
#include "bkf_suber_data.h"
#include "bkf_suber_conn.h"
#include "bkf_suber_env.h"
#include "bkf_suber_disp.h"
#include "bkf_str.h"
#include "bkf_url.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
STATIC uint32_t BkfSuberInitEnv(BkfSuberEnv *env, BkfSuberInitArg *arg);
STATIC void BkfSuberUnInitEnv(BkfSuberEnv *env);
#pragma pack()
#endif

#if BKF_BLOCK("公有函数定义")
BkfSuber *BkfSuberInit(BkfSuberInitArg *arg)
{
    BKF_RETURNNULL_IF(arg == VOS_NULL);
    BKF_RETURNNULL_IF(arg->name == VOS_NULL);
    BKF_RETURNNULL_IF(arg->memMng == VOS_NULL);
    BKF_RETURNNULL_IF(arg->svcName == VOS_NULL);
    BKF_RETURNNULL_IF(arg->disp == VOS_NULL);
    BKF_RETURNNULL_IF(arg->pfm == VOS_NULL);
    BKF_RETURNNULL_IF(arg->xMap == VOS_NULL);
    BKF_RETURNNULL_IF(arg->tmrMng == VOS_NULL);
    BKF_RETURNNULL_IF(arg->jobMng == VOS_NULL);
    BKF_RETURNNULL_IF(arg->sysLogMng == VOS_NULL);
    BKF_RETURNNULL_IF(arg->chCliMng == VOS_NULL);
    BKF_RETURNNULL_IF(arg->sliceKeyCodec == VOS_NULL);
    BKF_RETURNNULL_IF(arg->sliceKeyGetStrOrNull == VOS_NULL);
    BKF_RETURNNULL_IF(arg->sliceKeyCmp == VOS_NULL);

    uint32_t len = sizeof(BkfSuber);
    BkfSuber *suber = BKF_MALLOC(arg->memMng, len);
    if (suber == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(suber, len, 0, len);
    if (BkfSuberInitEnv(&suber->env, arg) != BKF_OK) {
        goto error;
    }

    if (suber->env.dbgOn) {
        suber->log = suber->env.log;
    }

    BkfSuberConnMngInitArg connMngInitArg = {.env = &suber->env};
    suber->connMng = BkfSuberConnMngInit(&connMngInitArg);
    if (suber->connMng == VOS_NULL) {
        goto error;
    }

    BkfSuberSubDataMngInitArg dataInitArg = {.env = &suber->env,
        .cookie = suber->connMng,
        .ntfPuberUrlAdd = (F_BKF_SUBER_DATA_NTF_PUBER_URL_OPER)BkfSuberConnProcInstAddUrl,
        .ntfPuberUrlDel = (F_BKF_SUBER_DATA_NTF_PUBER_URL_OPER)BkfSuberConnProcInstDelUrl,
        .ntfSliceInstAdd = (F_BKF_SUBER_DATA_NTF_SLICE_INST_OPER)BkfSuberConnProcSliceAddInst,
        .ntfSliceInstDel = (F_BKF_SUBER_DATA_NTF_SLICE_INST_OPER)BkfSuberConnProcSliceDelInst,
        .ntfAppSubAdd = (F_BKF_SUBER_DATA_NTF_SLICE_APP_SUB_OPER)BkfSuberConnProcAddSub,
        .ntfAppSubDel = (F_BKF_SUBER_DATA_NTF_SLICE_APP_SUB_OPER)BkfSuberConnProcDelSub};

    suber->dataMng = BkfSuberDataMngInit(&dataInitArg);
    if (suber->dataMng == VOS_NULL) {
        goto error;
    }

    BkfSuberDispInit(suber);
    BKF_LOG_DEBUG(suber->log, "init suber success, name %s\n", suber->env.name);
    return suber;

error:
    BkfSuberUninit(suber);
    return VOS_NULL;
}

void BkfSuberUninit(BkfSuber *suber)
{
    if (suber == VOS_NULL) {
        return;
    }

    BKF_LOG_DEBUG(suber->log, "uninit suber, name %s\n", suber->env.name);

    BkfSuberDispUninit(suber);

    /* 注意初始化时，将connMng注入到datamng，因此销毁时需要先销毁dataMng */
    if (suber->dataMng != VOS_NULL) {
        BkfSuberDataMngUnInit(suber->dataMng);
    }

    if (suber->connMng != VOS_NULL) {
        BkfSuberConnMngUnInit(suber->connMng);
    }

    BkfSuberUnInitEnv(&suber->env);
    BKF_FREE(suber->env.memMng, suber);
}

uint32_t BkfSuberRegisterTableTypeEx(BkfSuber *suber, BkfSuberTableTypeVTbl *vTbl, void *userData, uint16_t userDataLen)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }

    if (suber->env.subMod != BKF_SUBER_MODE_DEFAULT) {
        return BKF_ERR;
    }

    BKF_LOG_DEBUG(suber->log, "reg table suber, name %s, table id %u\n", suber->env.name, vTbl->tableTypeId);
    return BkfSuberDataTableTypeAdd(suber->dataMng, suber->env.subMod, (void *)vTbl, userData, userDataLen);
}

uint32_t BkfSuberRegisterTableTypeExVTbl(BkfSuber *suber, BkfSuberTableTypeVTblEx *vTbl, void *userData,
    uint16_t userDataLen)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }

    if (suber->env.subMod != BKF_SUBER_MODE_EX) {
        return BKF_ERR;
    }

    BKF_LOG_DEBUG(suber->log, "reg table suberEx, name %s, table id %u\n", suber->env.name, vTbl->tableTypeId);
    return BkfSuberDataTableTypeAdd(suber->dataMng, suber->env.subMod, (void *)vTbl, userData, userDataLen);
}

uint32_t BkfSuberUnRegisterTableType(BkfSuber *suber, uint16_t tabType)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(suber->log, "Unreg table name %s, table id %u\n", suber->env.name, tabType);
    return BkfSuberDataTableTypeDel(suber->dataMng, tabType);
}

void *BkfSuberGetTableTypeUserData(BkfSuber *suber, uint16_t tableTypeId)
{
    if (suber == VOS_NULL) {
        return VOS_NULL;
    }

    return BkfSuberDataTableTypeGetUserData(suber->dataMng, tableTypeId);
}

BkfSuberTableTypeVTbl *BkfSuberGetTableTypeVTbl(BkfSuber *suber, uint16_t tableTypeId)
{
    if (suber == VOS_NULL) {
        return VOS_NULL;
    }

    return BkfSuberDataTableTypeGetVtbl(suber->dataMng, tableTypeId);
}

uint32_t BkfSuberEnable(BkfSuber *suber)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }

    BKF_LOG_DEBUG(suber->log, "suber enable\n");
    return BkfSuberConnEnable(suber->connMng);
}

uint32_t BkfSuberCreateSvcInst(BkfSuber *suber, uint32_t instId)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }
    uint64_t instIdVal = instId;

    BKF_LOG_DEBUG(suber->log, "create inst, name %s, inst id %llu\n", suber->env.name, instIdVal);
    BkfSuberInst *inst = BkfSuberDataAddInst(suber->dataMng, instIdVal);
    if (inst == VOS_NULL) {
        BKF_LOG_ERROR(suber->log, "create inst fail, name %s, inst id %llu\n", suber->env.name, instIdVal);
        return BKF_ERR;
    }

    return BKF_OK;
}

uint32_t BkfSuberCreateSvcInstEx(BkfSuber *suber, uint64_t instId)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(suber->log, "create inst, name %s, inst id %llu\n", suber->env.name, instId);
    BkfSuberInst *inst = BkfSuberDataAddInst(suber->dataMng, instId);
    if (inst == VOS_NULL) {
        BKF_LOG_ERROR(suber->log, "create inst fail, name %s, inst id %llu\n", suber->env.name, instId);
        return BKF_ERR;
    }

    return BKF_OK;
}

void BkfSuberDeleteSvcInst(BkfSuber *suber, uint32_t instId)
{
    if (suber == VOS_NULL) {
        return;
    }
    uint64_t instIdVal = instId;
    BKF_LOG_DEBUG(suber->log, "del inst, name %s, inst id %llu\n", suber->env.name, instIdVal);
    BkfSuberDataDelInstByKey(suber->dataMng, instIdVal);
}

void BkfSuberDeleteSvcInstEx(BkfSuber *suber, uint64_t instId)
{
    if (suber == VOS_NULL) {
        return;
    }
    BKF_LOG_DEBUG(suber->log, "del inst, name %s, inst id %llu\n", suber->env.name, instId);
    BkfSuberDataDelInstByKey(suber->dataMng, instId);
}

uint32_t BkfSuberSetSvcInstUrl(BkfSuber *suber, uint32_t instId, BkfUrl *puberUrl)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }
    uint64_t instIdVal = instId;
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    BKF_LOG_DEBUG(suber->log, "set inst url, name %s, inst id %llu, url %s\n", suber->env.name, instIdVal,
        BkfUrlGetStr(puberUrl, urlStr, sizeof(urlStr)));
    BkfSuberInst *inst = BkfSuberDataFindInst(suber->dataMng, instIdVal);
    if (inst == VOS_NULL) {
        BKF_LOG_ERROR(suber->log, "set inst url fail because no inst, name %s, inst id %llu, url %s\n", suber->env.name,
            instIdVal, BkfUrlGetStr(puberUrl, urlStr, sizeof(urlStr)));
        return BKF_ERR;
    }

    return BkfSuberDataInstAddPuberUrl(inst, puberUrl);
}

uint32_t BkfSuberSetSvcInstUrlEx(BkfSuber *suber, uint64_t instId, BkfUrl *puberUrl)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    BKF_LOG_DEBUG(suber->log, "set inst urlEx, name %s, inst id %llu, url %s\n", suber->env.name, instId,
        BkfUrlGetStr(puberUrl, urlStr, sizeof(urlStr)));
    BkfSuberInst *inst = BkfSuberDataFindInst(suber->dataMng, instId);
    if (inst == VOS_NULL) {
        BKF_LOG_ERROR(suber->log, "set inst urlEx fail because no inst, name %s, inst id %llu, url %s\n",
            suber->env.name, instId, BkfUrlGetStr(puberUrl, urlStr, sizeof(urlStr)));
        return BKF_ERR;
    }

    if (suber->env.subMod == BKF_SUBER_MODE_DEFAULT) {
        return BkfSuberDataInstAddPuberUrl(inst, puberUrl);
    }
    return BkfSuberDataInstAddPuberUrlEx(inst, puberUrl);
}

uint32_t BkfSuberSetSvcInstLocalUrl(BkfSuber *suber, uint64_t instId, BkfUrl *localUrl)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }

    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    BKF_LOG_DEBUG(suber->log, "set inst locurl, name %s, inst id %llu, url %s\n", suber->env.name, instId,
        BkfUrlGetStr(localUrl, urlStr, sizeof(urlStr)));
    BkfSuberInst *inst = BkfSuberDataFindInst(suber->dataMng, instId);
    if (inst == VOS_NULL) {
        BKF_LOG_ERROR(suber->log, "set inst locurl fail because no inst, name %s, inst id %llu, url %s\n",
            suber->env.name, instId, BkfUrlGetStr(localUrl, urlStr, sizeof(urlStr)));
        return BKF_ERR;
    }
    return BkfSuberDataInstAddLocalUrl(inst, localUrl);
}

uint32_t BkfSuberUnSetSvcInstUrl(BkfSuber *suber, uint32_t instId)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }
    uint64_t instIdVal = instId;
    BKF_LOG_DEBUG(suber->log, "unset inst url, name %s, inst id %llu\n", suber->env.name, instIdVal);

    BkfSuberInst *inst = BkfSuberDataFindInst(suber->dataMng, instIdVal);
    if (inst == VOS_NULL) {
        BKF_LOG_ERROR(suber->log, "unset inst url fail, name %s, inst id %llu\n", suber->env.name, instIdVal);
        return BKF_ERR;
    }

    BkfSuberDataInstDelPuberUrl(inst);
    return BKF_OK;
}

uint32_t BkfSuberUnSetSvcInstUrlEx(BkfSuber *suber, uint64_t instId)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(suber->log, "unset inst url, name %s, inst id %llu\n", suber->env.name, instId);

    BkfSuberInst *inst = BkfSuberDataFindInst(suber->dataMng, instId);
    if (inst == VOS_NULL) {
        BKF_LOG_ERROR(suber->log, "unset inst url fail, name %s, inst id %llu\n", suber->env.name, instId);
        return BKF_ERR;
    }

    BkfSuberDataInstDelPuberUrl(inst);
    return BKF_OK;
}

uint32_t BkfSuberUnSetSvcInstLocalUrl(BkfSuber *suber, uint64_t instId)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(suber->log, "unset inst locurl, name %s, inst id %llu\n", suber->env.name, instId);

    BkfSuberInst *inst = BkfSuberDataFindInst(suber->dataMng, instId);
    if (inst == VOS_NULL) {
        BKF_LOG_ERROR(suber->log, "unset inst locurl fail, name %s, inst id %llu\n", suber->env.name, instId);
        return BKF_ERR;
    }
    BkfSuberDataInstDelLocalUrl(inst);
    return BKF_OK;
}

uint32_t BkfSuberSetSelfUrl(BkfSuber *suber, BkfUrl *selfUrl)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }

    uint8_t urlStr[BKF_URL_STR_LEN_MAX] = {0};
    BKF_LOG_DEBUG(suber->log, "set self url, name %s, url %s\n", suber->env.name,
        BkfUrlGetStr(selfUrl, urlStr, sizeof(urlStr)));
    return BkfSuberConnMngSetSelfUrl(suber->connMng, selfUrl);
}

void BkfSuberUnSetSelfUrl(BkfSuber *suber, BkfUrl *selfUrl)
{
    (void)suber;
    (void)selfUrl;
    BKF_ASSERT(0);
}

uint32_t BkfSuberSetSliceInstId(BkfSuber *suber, void *sliceKey, uint32_t instId)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }
    uint64_t instIdVal = instId;
    uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN] = {0};
    (void)BkfSuberGetSliceKeyStr(&suber->env, sliceKey, dispSliceStr, sizeof(dispSliceStr));
    BKF_LOG_DEBUG(suber->log, "set slice inst, name %s, slice %s, inst id %llu\n", suber->env.name, dispSliceStr,
        instIdVal);
    BkfSuberSlice *slice = BkfSuberDataAddSlice(suber->dataMng, sliceKey);
    if (slice == VOS_NULL) {
        BKF_LOG_ERROR(suber->log, "set slice inst fail, slice is not exist, name %s, slice %s, inst id %llu\n",
            suber->env.name, dispSliceStr, instIdVal);
        return BKF_ERR;
    }
    return BkfSuberDataSliceAddInst(slice, instIdVal);
}

uint32_t BkfSuberSetSliceInstIdEx(BkfSuber *suber, void *sliceKey, uint64_t instId)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }
    uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN] = {0};
    (void)BkfSuberGetSliceKeyStr(&suber->env, sliceKey, dispSliceStr, sizeof(dispSliceStr));
    BKF_LOG_DEBUG(suber->log, "set slice inst, name %s, slice %s, inst id %llu\n", suber->env.name, dispSliceStr,
        instId);
    BkfSuberSlice *slice = BkfSuberDataAddSlice(suber->dataMng, sliceKey);
    if (slice == VOS_NULL) {
        BKF_LOG_ERROR(suber->log, "set slice inst fail, slice is not exist, name %s, slice %s, inst id %llu\n",
            suber->env.name, dispSliceStr, instId);
        return BKF_ERR;
    }
    return BkfSuberDataSliceAddInst(slice, instId);
}

void BkfSuberUnsetSliceInstId(BkfSuber *suber, void *sliceKey)
{
    if (suber == VOS_NULL) {
        return;
    }
    uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN];
    BKF_LOG_DEBUG(suber->log, "unset slice inst, name %s, slice %s\n", suber->env.name,
        BkfSuberGetSliceKeyStr(&suber->env, sliceKey, dispSliceStr, BKF_SUBER_DISP_SLICELEN));
    BkfSuberDataDelSliceByKey(suber->dataMng, sliceKey);
}

void BkfSuberUnsetSliceSpecInstId(BkfSuber *suber, void *sliceKey, uint64_t instId)
{
    if (suber == VOS_NULL) {
        return;
    }
    uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN];
    BKF_LOG_DEBUG(suber->log, "unset slice inst, name %s, slice %s, instId %llu\n", suber->env.name,
        BkfSuberGetSliceKeyStr(&suber->env, sliceKey, dispSliceStr, BKF_SUBER_DISP_SLICELEN), instId);
    BkfSuberSlice *slice = BkfSuberDataFindSlice(suber->dataMng, sliceKey);
    if (slice == VOS_NULL) {
        return;
    }
    BkfSuberDataDelSliceSpecInst(slice, instId);
}

uint32_t BkfSuberSubEx(BkfSuber *suber, BkfSuberSubArgEx *subArg)
{
    /* 增量订阅和对账请求 */
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }
    uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN] = {0};
    (void)BkfSuberGetSliceKeyStr(&suber->env, subArg->sliceKey, dispSliceStr, sizeof(dispSliceStr));
    BKF_LOG_DEBUG(suber->log, "suber subEx, name %s, table id %u, slice %s instId %llu\n", suber->env.name,
        subArg->tableTypeId, dispSliceStr, subArg->instId);

    BkfSuberTableTypeVTbl *vtbl = BkfSuberDataTableTypeGetVtbl(suber->dataMng, subArg->tableTypeId);
    if (vtbl == VOS_NULL) {
        BKF_LOG_ERROR(suber->log, "suberEx sub fail, no vtbl, name %s, table id %u, slice %s\n", suber->env.name,
            subArg->tableTypeId, dispSliceStr);
        return BKF_ERR;
    }
    uint32_t ret = BKF_OK;

    /* 指定instId sub */
    ret = BkfSuberDataAddAppSub(suber->dataMng, subArg->sliceKey, subArg->tableTypeId, subArg->instId,
        subArg->isVerify);
    if (ret != BKF_OK) {
        BKF_LOG_DEBUG(suber->log, "suber subEx fail, add app sub fail, name %s, table id %u, slice %s instId %llu\n",
            suber->env.name, subArg->tableTypeId, dispSliceStr, subArg->instId);
        return BKF_ERR;
    }
    return BKF_OK;
}

uint32_t BkfSuberSub(BkfSuber *suber, BkfSuberSubArg *subArg)
{
    if (suber == VOS_NULL) {
        return BKF_ERR;
    }

    uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN] = {0};
    (void)BkfSuberGetSliceKeyStr(&suber->env, subArg->sliceKey, dispSliceStr, sizeof(dispSliceStr));
    BKF_LOG_DEBUG(suber->log, "suber sub, name %s, table id %u, slice %s\n", suber->env.name, subArg->tableTypeId,
        dispSliceStr);

    BkfSuberTableTypeVTbl *vtbl = BkfSuberDataTableTypeGetVtbl(suber->dataMng, subArg->tableTypeId);
    if (vtbl == VOS_NULL) {
        BKF_LOG_DEBUG(suber->log, "suber sub fail, no vtbl, name %s, table id %u, slice %s\n", suber->env.name,
            subArg->tableTypeId, dispSliceStr);
        return BKF_ERR;
    }
    uint32_t ret = BkfSuberDataAddAppSub(suber->dataMng, subArg->sliceKey, subArg->tableTypeId,
        BKF_SUBER_INVALID_INST_ID, subArg->isVerify);
    if (ret != BKF_OK) {
        BKF_LOG_DEBUG(suber->log, "suber sub fail, add app sub fail, name %s, table id %u, slice %s\n", suber->env.name,
            subArg->tableTypeId, dispSliceStr);
        return BKF_ERR;
    }
    return BKF_OK;
}

void BkfSuberUnsub(BkfSuber *suber, BkfSuberSubArg *subArg)
{
    if (suber == VOS_NULL) {
        return;
    }

    uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN];
    BKF_LOG_DEBUG(suber->log, "suber unsub, name %s, table id %u, slice %s\n", suber->env.name, subArg->tableTypeId,
        BkfSuberGetSliceKeyStr(&suber->env, subArg->sliceKey, dispSliceStr, BKF_SUBER_DISP_SLICELEN));
    BkfSuberDataDelAppSubByKey(suber->dataMng, subArg->sliceKey, subArg->tableTypeId, BKF_SUBER_INVALID_INST_ID,
        subArg->isVerify);
}

void BkfSuberUnsubEx(BkfSuber *suber, BkfSuberSubArgEx *subArg)
{
    if (suber == VOS_NULL) {
        return;
    }

    uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN];
    BKF_LOG_DEBUG(suber->log, "suber unsubEx, name %s, table id %u, slice %s instId %llu\n", suber->env.name,
        subArg->tableTypeId,
        BkfSuberGetSliceKeyStr(&suber->env, subArg->sliceKey, dispSliceStr, BKF_SUBER_DISP_SLICELEN), subArg->instId);
    /* 指定instId unsub */
    BkfSuberDataDelAppSubByKey(suber->dataMng, subArg->sliceKey, subArg->tableTypeId, subArg->instId, subArg->isVerify);
    return;
}

#endif

#if BKF_BLOCK("私有函数定义")

STATIC uint32_t BkfSuberInitEnv(BkfSuberEnv *env, BkfSuberInitArg *arg)
{
    env->idlVersionMajor = arg->idlVersionMajor;
    env->idlVersionMinor = arg->idlVersionMinor;
    env->dbgOn = arg->dbgOn;
    env->subMod = arg->subMod;
    env->memMng = arg->memMng;
    env->disp = arg->disp;
    env->log = arg->log;
    env->pfm = arg->pfm;
    env->xmap = arg->xMap;
    env->tmrMng = arg->tmrMng;
    env->jobMng = arg->jobMng;
    env->chCliMng = arg->chCliMng;
    env->jobPrioH = arg->jobPrioH;
    env->jobPrioL = arg->jobPrioL;
    env->jobTypeId1 = arg->jobTypeId1;
    env->sysLogMng = arg->sysLogMng;
    env->sliceVTbl.keyLen = arg->sliceKeyLen;
    env->sliceVTbl.keyCodec = arg->sliceKeyCodec;
    env->sliceVTbl.keyGetStrOrNull = arg->sliceKeyGetStrOrNull;
    env->sliceVTbl.keyCmp = arg->sliceKeyCmp;
    env->lsnUrl = arg->lsnUrl;

    if (arg->subConnState) {
        env->cookie = arg->cookie;
        env->onConn = arg->onConn;
        env->onDisConn = arg->onDisConn;
    }
    env->msgLenMax = BKF_CH_CLI_MALLOC_DATA_BUF_LEN_MAX;
    env->name = BkfStrNew(arg->memMng, "%s", arg->name);
    if (env->name == VOS_NULL) {
        goto error;
    }
    env->svcName = BkfStrNew(arg->memMng, "%s", arg->svcName);
    if (env->svcName == VOS_NULL) {
        goto error;
    }

    return BKF_OK;

error:
    BkfSuberUnInitEnv(env);
    return BKF_ERR;
}

STATIC void BkfSuberUnInitEnv(BkfSuberEnv *env)
{
    if (env->name != VOS_NULL) {
        BkfStrDel(env->name);
    }
    if (env->svcName != VOS_NULL) {
        BkfStrDel(env->svcName);
    }
}
#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif