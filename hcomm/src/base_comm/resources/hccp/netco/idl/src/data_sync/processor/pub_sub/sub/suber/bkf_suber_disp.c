/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_suber_private.h"
#include "bkf_suber_data.h"
#include "bkf_subscriber.h"
#include "securec.h"
#include "vos_base.h"
#include "bkf_suber_disp.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#if BKF_BLOCK("内部函数")
typedef struct tagBkfSuberDispAppSubCtx {
    uint32_t appSubCnt;
    uint16_t tableTypeId;
    uint16_t pad;
    uint64_t instId;
    uint8_t sliceKey[BKF_SUBER_SLICE_KEY_LEN_MAX];
} BkfSuberDispAppSubCtx;

void BkfSuberDisp(BkfSuber *suber);
void BkfSuberDispSlice(BkfSuber *suber);
void BkfSuberDispInst(BkfSuber *suber);
void BkfSuberDispAppSub(BkfSuber *suber);
#endif

#if BKF_BLOCK("内部函数")

void BkfSuberDisp(BkfSuber *suber)
{
    BkfDisp *disp = suber->env.disp;

    BKF_DISP_PRINTF(disp, "Suber: %s/%s\n", suber->env.name, suber->env.svcName);
    BKF_DISP_PRINTF(disp, "DbgOn(%u)\n", suber->env.dbgOn);
    BKF_DISP_PRINTF(disp, "MemMng(%#x)\n", BKF_MASK_ADDR(suber->env.memMng));
    BKF_DISP_PRINTF(disp, "Disp(%#x)\n", BKF_MASK_ADDR(suber->env.disp));
    BKF_DISP_PRINTF(disp, "Xmap(%#x)\n", BKF_MASK_ADDR(suber->env.xmap));
    BKF_DISP_PRINTF(disp, "Tmr(%#x)\n", BKF_MASK_ADDR(suber->env.tmrMng));
    BKF_DISP_PRINTF(disp, "Log(%#x)\n", BKF_MASK_ADDR(suber->env.log));
    BKF_DISP_PRINTF(disp, "msgLenMax(%d)\n", suber->env.msgLenMax);
    BKF_DISP_PRINTF(disp, "JobMng(%#x)/jobTypeId1(%u)/jobPrioL(%u),JobPrioH(%u)\n", BKF_MASK_ADDR(suber->env.jobMng),
        suber->env.jobTypeId1, suber->env.jobPrioL, suber->env.jobPrioH);
    BKF_DISP_PRINTF(disp, "Slice_keyLen(%u)/keyCmp(%#x)/keyGetStrOrNull(%#x)/keyCodec(%#x)\n",
        suber->env.sliceVTbl.keyLen, BKF_MASK_ADDR(suber->env.sliceVTbl.keyCmp),
        BKF_MASK_ADDR(suber->env.sliceVTbl.keyGetStrOrNull), BKF_MASK_ADDR(suber->env.sliceVTbl.keyCodec));
    BKF_DISP_PRINTF(disp, "SeqNum(%" VOS_PRIu64 ")\n", suber->env.seeds);

    return;
}

void BkfSuberDispSlice(BkfSuber *suber)
{
    BkfDisp *disp = suber->env.disp;
    BkfSuberSlice *slice = VOS_NULL;
    void *itor;
    uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN];

    BKF_DISP_PRINTF(disp, "================================\n");
    slice = BkfSuberDataGetFirstSlice(suber->dataMng, &itor);
    if (slice != VOS_NULL) {
        BKF_DISP_PRINTF(disp, "    instId                                        sliceKey\n");
    }
    while (slice != VOS_NULL) {
        BKF_DISP_PRINTF(disp, "%llu%48s\n", slice->instId,
            BkfSuberGetSliceKeyStr(&suber->env, slice->sliceKey, dispSliceStr, BKF_SUBER_DISP_SLICELEN));
        slice = BkfSuberDataGetNextSlice(suber->dataMng, &itor);
    }

    BKF_DISP_PRINTF(disp, "================================\n");
    return;
}

void BkfSuberDispAppSub(BkfSuber *suber)
{
    BkfDisp *disp = suber->env.disp;
    BkfSuberDispAppSubCtx *lastCtx = VOS_NULL;
    BkfSuberDispAppSubCtx curCtx = {0};
    BkfSuberAppSub *appSub = VOS_NULL;

    lastCtx = BKF_DISP_GET_LAST_CTX(disp, VOS_NULL);
    if (lastCtx == VOS_NULL) {
        appSub = BkfSuberDataGetFirstAppSub(suber->dataMng, VOS_NULL);
        BKF_DISP_PRINTF(disp, "============================================\n");
    } else {
        curCtx = *lastCtx;
        appSub = BkfSuberDataFindNextAppSub(suber->dataMng, curCtx.sliceKey, curCtx.tableTypeId, curCtx.instId);
    }

    if (appSub != VOS_NULL) {
        curCtx.appSubCnt++;
        uint8_t dispSliceStr[BKF_SUBER_DISP_SLICELEN] = {0};
        BKF_DISP_PRINTF(disp, "==========\n");
        BKF_DISP_PRINTF(disp, "No.%u\n", curCtx.appSubCnt);
        BKF_DISP_PRINTF(disp, "SliceKey: %s\n",
            BkfSuberGetSliceKeyStr(&suber->env, appSub->key.sliceKey, dispSliceStr, BKF_SUBER_DISP_SLICELEN));
        BKF_DISP_PRINTF(disp, "TableTypeId: %u\n", appSub->key.tableTypeId);
        BKF_DISP_PRINTF(disp, "instId: %llu\n", appSub->key.instId);
        BKF_DISP_PRINTF(disp, "IsVerify: %u\n", appSub->isVerify);
        BkfSuberTableTypeVTbl *vtbl = BkfSuberDataTableTypeGetVtbl(suber->dataMng, appSub->key.tableTypeId);
        if (vtbl == VOS_NULL) {
            BKF_DISP_PRINTF(disp, "Not register vtbl for table type\n");
        } else {
            BKF_DISP_PRINTF(disp, "TableIsNeedComplete: %u\n", vtbl->needTableComplete);
        }

        int32_t err = memcpy_s(curCtx.sliceKey, sizeof(curCtx.sliceKey), appSub->key.sliceKey,
            suber->env.sliceVTbl.keyLen);
        BKF_ASSERT(err == EOK);
        curCtx.tableTypeId = appSub->key.tableTypeId;
        curCtx.instId = appSub->key.instId;
        BKF_DISP_SAVE_CTX(disp, &curCtx, sizeof(curCtx), VOS_NULL, 0);
    } else {
        BKF_DISP_PRINTF(disp, "============================================\n");
        BKF_DISP_PRINTF(disp, "Total App Sub Cnt: %u\n", curCtx.appSubCnt);
    }

    return;
}

void BkfSuberDispInst(BkfSuber *suber)
{
    BkfDisp *disp = suber->env.disp;
    BkfSuberInst *inst = VOS_NULL;
    void *itor = VOS_NULL;
    uint8_t buf[BKF_DISP_PRINTF_BUF_LEN_MAX];
    uint32_t cnt = 0;

    BKF_DISP_PRINTF(disp, "============================================\n");
    BKF_DISP_PRINTF(disp, " No|    InstId|Url\n");
    inst = BkfSuberDataGetFirstInst(suber->dataMng, &itor);
    while (inst != VOS_NULL) {
        cnt++;
        BKF_DISP_PRINTF(disp, "%3u|%llu|%s\n", cnt, inst->instId, BkfUrlGetStr(&inst->puberUrl, buf, sizeof(buf)));
        inst = BkfSuberDataGetNextInst(suber->dataMng, &itor);
    }

    BKF_DISP_PRINTF(disp, "==========================================\n");
}

void BkfSuberDispTestUnSetInstUrl(BkfSuber *suber, uint64_t instId)
{
    BkfDisp *disp = suber->env.disp;

    BKF_DISP_PRINTF(disp, "=========begin==============\n");
    BkfSuberInst *inst;
    inst = BkfSuberDataFindInst(suber->dataMng, instId);
    if (inst == VOS_NULL) {
        BKF_DISP_PRINTF(disp, "========inst %llu not exist =========\n", instId);
        return;
    }

    BkfSuberUnSetSvcInstUrlEx(suber, instId);
    BKF_DISP_PRINTF(disp, "==========unset OK!=================\n");
    return;
}

void BkfSuberDispTestSetInstUrl(BkfSuber *suber, uint64_t instId, char *url)
{
    BkfDisp *disp = suber->env.disp;
    BKF_DISP_PRINTF(disp, "=========begin==============\n");
    BkfSuberInst *inst = VOS_NULL;
    inst = BkfSuberDataFindInst(suber->dataMng, instId);
    if (inst == VOS_NULL) {
        BKF_DISP_PRINTF(disp, "========inst %llu not exist =========\n", instId);
        return;
    }

    BkfUrl svcUrl;
    if (BkfUrlSet(&svcUrl, url) != BKF_OK) {
        BKF_DISP_PRINTF(disp, "========url %s invalid =========\n", url);
        return;
    }

    uint32_t ret;
    ret = BkfSuberSetSvcInstUrlEx(suber, instId, &svcUrl);
    if (ret != BKF_OK) {
        BKF_DISP_PRINTF(disp, "========set url %s fail, ret is %u =========\n", url, ret);
    } else {
        BKF_DISP_PRINTF(disp, "==========set OK!=================\n");
    }
    return;
}

void BkfSuberDispTestAddSvcInst(BkfSuber *suber, uint64_t instId)
{
    BkfDisp *disp = suber->env.disp;
    BKF_DISP_PRINTF(disp, "=========begin==============\n");
    BkfSuberInst *inst = VOS_NULL;
    inst = BkfSuberDataFindInst(suber->dataMng, instId);
    if (inst != VOS_NULL) {
        BKF_DISP_PRINTF(disp, "========inst %llu already exists! =========\n", instId);
        return;
    }
    uint32_t ret;
    ret = BkfSuberCreateSvcInstEx(suber, instId);
    if (ret != BKF_OK) {
        BKF_DISP_PRINTF(disp, "========add inst %llu fail! ret is %u =========\n", instId, ret);
        return;
    }

    BKF_DISP_PRINTF(disp, "==========add inst id %u OK!=================\n", instId);
    return;
}

void BkfSuberDispTestDelSvcInst(BkfSuber *suber, uint64_t instId)
{
    BkfDisp *disp = suber->env.disp;
    BKF_DISP_PRINTF(disp, "=========begin==============\n");
    BkfSuberInst *inst = VOS_NULL;
    inst = BkfSuberDataFindInst(suber->dataMng, instId);
    if (inst == VOS_NULL) {
        BKF_DISP_PRINTF(disp, "========inst %llu not exist! =========\n", instId);
        return;
    }

    BkfSuberDeleteSvcInstEx(suber, instId);
    BKF_DISP_PRINTF(disp, "==========del inst id %llu OK!=================\n", instId);
    return;
}
#endif

#if BKF_BLOCK("外部函数")
void BkfSuberDispInit(BkfSuber *suber)
{
    BkfDisp *disp = suber->env.disp;
    char *objName = (char *)suber->env.name;
    uint32_t ret;

    ret = BkfDispRegObj(disp, objName, suber);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return;
    }

    (void)BKF_DISP_REG_FUNC(disp, BkfSuberDisp, "disp suber", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfSuberDispSlice, "disp suber slice", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfSuberDispAppSub, "disp suber sess", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfSuberDispInst, "disp suber inst", objName, 0);
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfSuberDispTestUnSetInstUrl, "disp suber test unset inst url", objName, 1,
        "uniInstId");
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfSuberDispTestSetInstUrl, "disp suber test set inst url", objName, BKF_NUM2,
        "uniInstId", "pcUrlStr");
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfSuberDispTestAddSvcInst, "disp suber test add svc inst", objName, 1,
        "uniInstId");
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfSuberDispTestDelSvcInst, "disp suber test del svc inst", objName, 1,
        "uniInstId");
}

void BkfSuberDispUninit(BkfSuber *suber)
{
    BkfDispUnregObj(suber->env.disp, suber->env.name);
}

char *BkfSuberGetSliceKeyStr(BkfSuberEnv *env, void *sliceKey, uint8_t *buf, int32_t bufLen)
{
    if ((env == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "sliceKeyNg";
    }

    if (sliceKey == VOS_NULL) {
        return "sliceKeyNg_Null";
    }

    if (env->sliceVTbl.keyGetStrOrNull != VOS_NULL) {
        return env->sliceVTbl.keyGetStrOrNull(sliceKey, buf, bufLen);
    } else {
        return BKF_GET_MEM_STD_STR(sliceKey, env->sliceVTbl.keyLen, buf, bufLen);
    }
}

#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
