/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_MOD_NAME ((pfmMng)->name)

#include <unistd.h>
#include "securec.h"
#include "bkf_str.h"
#include "bkf_assert.h"
#include "v_avll.h"
#include "v_stringlib.h"
#include "vos_cputick.h"
#include "bkf_pfm.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#define BKF_PFM_BITNUM_32 (32)
#define BKF_PFM_RANDOMNUM_1001 (1001)
#define BKF_PFM_RANDOMNUM_1002 (1002)
#define BKF_PFM_RANDOMNUM_1003 (1003)
#define BKF_PFM_IMMNUM_100000 (100000)

#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
typedef struct tagBkfPfmMea BkfPfmMea;
/*
此cache不是预先分配的，是第一次mea的时候生成的。
正式版本不会mea，不会占用cache内存；pfm版本会mea，设定较大hash mod，更多加速
*/
static inline BOOL BkfPfmTimeDiff(VOS_CPUTICK_S *begin, VOS_CPUTICK_S *end, uint64_t *diffTick)
{
    uint64_t beginTick = (((uint64_t)begin->uiHigh) << BKF_PFM_BITNUM_32) | ((uint64_t)begin->uiLow);
    uint64_t endTick = (((uint64_t)end->uiHigh) << BKF_PFM_BITNUM_32) | ((uint64_t)end->uiLow);

    if (endTick >= beginTick) {
        *diffTick = (endTick - beginTick);
        return VOS_TRUE;
    } else {
        return VOS_FALSE;
    }
}
static inline uint64_t BkfPfmTick2Ms(uint64_t tick)
{
    VOS_CPUTICK_S tickStru;
    uint32_t msHigh = 0;
    uint32_t msLow = 0;

    tickStru.uiHigh = (uint32_t)(tick >> BKF_PFM_BITNUM_32);
    tickStru.uiLow = (uint32_t)tick;
    (void)VOS_CpuTick2MsEx(&tickStru, &msHigh, &msLow);
    return (((uint64_t)msHigh) << BKF_PFM_BITNUM_32) | (uint64_t)msLow;
}

#define BKF_PFM_MEA_STR_2HASH_CODE_LEN_MAX 8
#define BKF_PFM_MEA_STR_HASH_MOD 0x0fff
#define BKF_PFM_MEA_STR_2CACHE_IDX(str)                                                                                \
    ((BkfGetStrHashCode((str), BKF_PFM_MEA_STR_2HASH_CODE_LEN_MAX)) & BKF_PFM_MEA_STR_HASH_MOD)

typedef struct tagBkfPfmMeaCache {
    BkfPfmMea *mea[BKF_PFM_MEA_STR_HASH_MOD + 1];
} BkfPfmMeaCache;

struct tagBkfPfm {
    BkfPfmInitArg argInit;
    char *name;
    uint8_t enable : 1;
    uint8_t rsv1 : 7;
    uint8_t pad1[3];
    AVLL_TREE meaSet;
    BkfPfmMeaCache *meaCache;
};

typedef struct tagBkfPfmMea {
    AVLL_NODE avlNode;

    uint8_t hasBegin : 1;
    uint8_t rsv1 : 7;
    uint8_t pad1[3];
    uint32_t beginCnt;
    uint32_t beginNgCnt;
    VOS_CPUTICK_S beginTime;
    uint32_t endCnt;
    uint32_t endNgCnt;
    uint32_t okCnt;
    uint64_t okUseTickPeak;
    uint64_t okkUseTickTotal;
    uint32_t cacheIdx; /* 比较加速用 */
    char meaStr[0];    /* key */
} BkfPfmMea;
BKF_STATIC_ASSERT(BKF_MBR_IS_LAST(BkfPfmMea, meaStr));

/* data op */
STATIC BkfPfmMea *BkfPfmAddMea(BkfPfm *pfm, char *meaStr);
STATIC void BkfPfmDelMea(BkfPfm *pfm, BkfPfmMea *mea);
STATIC void BkfPfmDelAllMea(BkfPfm *pfm);
STATIC BkfPfmMea *BkfPfmFindMea(BkfPfm *pfm, char *meaStr);
STATIC BkfPfmMea *BkfPfmFindNextMea(BkfPfm *pfm, char *meaStr);
STATIC BkfPfmMea *BkfPfmGetFirstMea(BkfPfm *pfm, void **itorOutOrNull);
STATIC BkfPfmMea *BkfPfmGetNextMea(BkfPfm *pfm, void **itorInOut);

/* disp */
STATIC void BkfPfmDispInit(BkfPfm *pfm);
STATIC void BkfPfmDispUninit(BkfPfm *pfm);

#pragma pack()
#endif

#if BKF_BLOCK("公有函数定义")
/* func */
BkfPfm *BkfPfmInit(BkfPfmInitArg *arg)
{
    BOOL paramIsInvalid = VOS_FALSE;
    BkfPfm *pfm = VOS_NULL;
    uint32_t len;

    paramIsInvalid = (arg == VOS_NULL) || (arg->name == VOS_NULL) || (arg->memMng == VOS_NULL) ||
                     (arg->disp == VOS_NULL);
    if (paramIsInvalid) {
        return VOS_NULL;
    }

    len = sizeof(BkfPfm);
    pfm = BKF_MALLOC(arg->memMng, len);
    if (pfm == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(pfm, len, 0, len);
    pfm->argInit = *arg;
    pfm->name = BkfStrNew(arg->memMng, "%s_pfm", arg->name);
    pfm->argInit.name = pfm->name;
    pfm->enable = BKF_COND_2BIT_FIELD(arg->enable);
    VOS_AVLL_INIT_TREE(pfm->meaSet, (AVLL_COMPARE)VOS_StrCmp, BKF_OFFSET(BkfPfmMea, meaStr[0]),
        BKF_OFFSET(BkfPfmMea, avlNode));
    BkfPfmDispInit(pfm);

    return pfm;
}

void BkfPfmUninit(BkfPfm *pfm)
{
    if (pfm == VOS_NULL) {
        return;
    }

    BkfPfmDispUninit(pfm);
    BkfPfmDelAllMea(pfm);
    BkfStrDel(pfm->name);
    if (pfm->meaCache != VOS_NULL) {
        BKF_FREE(pfm->argInit.memMng, pfm->meaCache);
    }
    BKF_FREE(pfm->argInit.memMng, pfm);
    return;
}

uint32_t BkfPfmBegin(BkfPfm *pfm, char *meaStr)
{
    BkfPfmMea *mea = VOS_NULL;

    if ((pfm == VOS_NULL) || (meaStr == VOS_NULL)) {
        return BKF_ERR;
    }
    if (!pfm->enable) {
        return BKF_OK;
    }

    mea = BkfPfmFindMea(pfm, meaStr);
    if (mea == VOS_NULL) {
        mea = BkfPfmAddMea(pfm, meaStr);
        if (mea == VOS_NULL) {
            return BKF_ERR;
        }
    }
    if (mea->hasBegin) {
        mea->beginNgCnt++;
    }
    /* 继续 */
    mea->hasBegin = VOS_TRUE;
    mea->beginCnt++;
    VOS_CpuTickGet(&mea->beginTime);
    return BKF_OK;
}

uint32_t BkfPfmEnd(BkfPfm *pfm, char *meaStr)
{
    BkfPfmMea *mea = VOS_NULL;

    if ((pfm == VOS_NULL) || (meaStr == VOS_NULL)) {
        return BKF_ERR;
    }
    if (!pfm->enable) {
        return BKF_OK;
    }

    mea = BkfPfmFindMea(pfm, meaStr);
    if (mea == VOS_NULL) {
        mea = BkfPfmAddMea(pfm, meaStr);
        if (mea == VOS_NULL) {
            return BKF_ERR;
        }
    }
    mea->endCnt++;
    if (mea->hasBegin) {
        VOS_CPUTICK_S endTime;
        uint64_t diffTick;
        BOOL diffOk = VOS_FALSE;

        mea->hasBegin = VOS_FALSE;
        VOS_CpuTickGet(&endTime);
        diffOk = BkfPfmTimeDiff(&mea->beginTime, &endTime, &diffTick);
        if (diffOk) {
            mea->okCnt++;
            mea->okUseTickPeak = BKF_GET_MAX(mea->okUseTickPeak, diffTick);
            mea->okkUseTickTotal += diffTick;
        } else {
            mea->endNgCnt++;
        }
    } else {
        mea->endNgCnt++;
    }

    return BKF_OK;
}

char *BkfPfmGetSummaryStr(BkfPfm *pfm, uint8_t *buf, int32_t bufLen)
{
    BkfPfmMea *mea = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t i;
    uint32_t meaCnt = 0;
    uint32_t meaCacheSize = 0;
    uint32_t validMeaCntInCache = 0;
    int32_t ret;

    if ((pfm == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "pfm_arg_error";
    }

    for (mea = BkfPfmGetFirstMea(pfm, &itor); mea != VOS_NULL; mea = BkfPfmGetNextMea(pfm, &itor)) {
        meaCnt++;
    }
    if (pfm->meaCache != VOS_NULL) {
        meaCacheSize = BKF_GET_ARY_COUNT(pfm->meaCache->mea);
        for (i = 0; i < meaCacheSize; i++) {
            if (pfm->meaCache->mea[i] != VOS_NULL) {
                validMeaCntInCache++;
            }
        }
    }
    ret = snprintf_truncated_s((char *)buf, (uint32_t)bufLen,
        "pfm(%#x, %s), memMng(%#x)/disp(%#x)/log(%#x), "
        "enable(%u/%u), meaCnt(%u), meaCache(%#x)/meaCacheSize(%u)/validMeaCntInCache(%u)",
        BKF_MASK_ADDR(pfm), pfm->name, BKF_MASK_ADDR(pfm->argInit.memMng), BKF_MASK_ADDR(pfm->argInit.disp),
        BKF_MASK_ADDR(pfm->argInit.log), pfm->enable, pfm->argInit.enable, meaCnt, BKF_MASK_ADDR(pfm->meaCache),
        meaCacheSize, validMeaCntInCache);
    if (ret < 0) {
        return "pfm_snprintf_ng";
    }
    return (char *)buf;
}

STATIC char *BkfPfmGetMeaStr(BkfPfm *pfm, BkfPfmMea *mea, uint8_t *buf, int32_t bufLen)
{
    (void)pfm;
    uint64_t meaOkUseMsTotal = BkfPfmTick2Ms(mea->okkUseTickTotal);
    uint64_t meaOkUseMsPeak = BkfPfmTick2Ms(mea->okUseTickPeak);
    uint64_t meaOkUseMsAvg = 0;
    int32_t ret;

    if (mea->okCnt > 0) {
        meaOkUseMsAvg = meaOkUseMsTotal / mea->okCnt;
    }
    ret = snprintf_truncated_s((char *)buf, (uint32_t)bufLen,
        "meaOkMs_avg(%-6" VOS_PRIu64 ")/t(%-12" VOS_PRIu64 ")/p(%-10" VOS_PRIu64 ")"
        "/okCnt(%-12u), "
        "begin(%u)/cnt(%u/%u), endCnt(%u/%u), cacheIdx(%u), "
        "meaOkTick_t(%" VOS_PRIu64 ")/p(%" VOS_PRIu64 ")",
        meaOkUseMsAvg, meaOkUseMsTotal, meaOkUseMsPeak, mea->okCnt, mea->hasBegin, mea->beginCnt, mea->beginNgCnt,
        mea->endCnt, mea->endNgCnt, mea->cacheIdx, mea->okkUseTickTotal, mea->okUseTickPeak);
    if (ret < 0) {
        return VOS_NULL;
    }
    return (char *)buf;
}
char *BkfPfmGetFirstMeaStr(BkfPfm *pfm, char **meaStrOut, uint8_t *buf, int32_t bufLen)
{
    BOOL paramIsInvalid = VOS_FALSE;
    BkfPfmMea *mea = VOS_NULL;

    paramIsInvalid = (pfm == VOS_NULL) || (meaStrOut == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0);
    if (paramIsInvalid) {
        return VOS_NULL;
    }

    mea = BkfPfmGetFirstMea(pfm, VOS_NULL);
    if (mea != VOS_NULL) {
        *meaStrOut = mea->meaStr;
        return BkfPfmGetMeaStr(pfm, mea, buf, bufLen);
    } else {
        return VOS_NULL;
    }
}

char *BkfPfmGetNextMeaStr(BkfPfm *pfm, char **meaStrInOut, uint8_t *buf, int32_t bufLen)
{
    BOOL paramIsInvalid = VOS_FALSE;
    BkfPfmMea *mea = VOS_NULL;

    paramIsInvalid = (pfm == VOS_NULL) || (meaStrInOut == VOS_NULL) || (*meaStrInOut == VOS_NULL) ||
                     (buf == VOS_NULL) || (bufLen <= 0);
    if (paramIsInvalid) {
        return VOS_NULL;
    }

    mea = BkfPfmFindNextMea(pfm, *meaStrInOut);
    if (mea != VOS_NULL) {
        *meaStrInOut = mea->meaStr;
        return BkfPfmGetMeaStr(pfm, mea, buf, bufLen);
    } else {
        return VOS_NULL;
    }
}

static inline void BkfPfmTestUseTime(BkfPfm *pfm, uint32_t loopMs)
{
    (void)pfm;
    (void)usleep(loopMs * BKF_US_PER_MS);
}
void BkfPfmTestAddMea(BkfPfm *pfm)
{
    uint32_t i;
    uint32_t cntMax = BKF_PFM_IMMNUM_100000;
    VOS_CPUTICK_S t2;

    if (!pfm->enable) {
        return;
    }

    BKF_PFM_BEGIN(pfm, "2CpuTickGet");
    for (i = 0; i < cntMax * BKF_NUM2; i++) {
        (void)VOS_CpuTickGet(&t2);
    }
    BKF_PFM_END(pfm, "2CpuTickGet");

    BKF_PFM_BEGIN(pfm, "3++");
    for (i = 0; i < cntMax; i++) {
    }
    BKF_PFM_END(pfm, "3++");

    BKF_PFM_BEGIN(pfm, "4pNone");
    BKF_PFM_END(pfm, "4pNone");

    BKF_PFM_BEGIN(pfm, "5pLoop");
    for (i = 0; i < cntMax; i++) {
        BKF_PFM_BEGIN(pfm, "6pNone");
        BKF_PFM_END(pfm, "6pNone");
    }
    BKF_PFM_END(pfm, "5pLoop");

    BKF_PFM_BEGIN(pfm, "t1");
    BKF_PFM_BEGIN(pfm, "t1"); /* last err1 */
    BkfPfmTestUseTime(pfm, BKF_PFM_RANDOMNUM_1001);
    BKF_PFM_BEGIN(pfm, "t22");
    BkfPfmTestUseTime(pfm, BKF_PFM_RANDOMNUM_1002);
    BKF_PFM_BEGIN(pfm, "t3333");
    BkfPfmTestUseTime(pfm, BKF_PFM_RANDOMNUM_1003);

    BKF_PFM_END(pfm, "t3333");
    BKF_PFM_END(pfm, "t1");
    BKF_PFM_END(pfm, "t1"); /* last err11 */
    BKF_PFM_END(pfm, "t1"); /* last err12 */
    BKF_PFM_END(pfm, "t22");
    BKF_PFM_END(pfm, "t1");  /* last err13 */
    BKF_PFM_END(pfm, "t22"); /* last err21 */
    BKF_PFM_END(pfm, "t22"); /* last err22 */
}

#endif

#if BKF_BLOCK("私有函数定义")
/* data op */
STATIC BkfPfmMea *BkfPfmAddMea(BkfPfm *pfm, char *meaStr)
{
    BkfPfmMea *mea = VOS_NULL;
    uint32_t strLen;
    uint32_t len;
    int32_t ret;
    BOOL insOk = VOS_FALSE;

    strLen = VOS_StrLen(meaStr);
    len = sizeof(BkfPfmMea) + strLen + 1;
    mea = BKF_MALLOC(pfm->argInit.memMng, len);
    if (mea == VOS_NULL) {
        goto error;
    }
    (void)memset_s(mea, len, 0, len);
    mea->cacheIdx = BKF_PFM_MEA_STR_2CACHE_IDX(meaStr);
    ret = snprintf_truncated_s(mea->meaStr, strLen + 1, "%s", meaStr);
    if (ret < 0) {
        goto error;
    }
    VOS_AVLL_INIT_NODE(mea->avlNode);
    insOk = VOS_AVLL_INSERT(pfm->meaSet, mea->avlNode);
    if (!insOk) {
        goto error;
    }
#ifndef BKF_CUT_AVL_CACHE
    if (pfm->meaCache == VOS_NULL) {
        len = sizeof(BkfPfmMeaCache);
        pfm->meaCache = BKF_MALLOC(pfm->argInit.memMng, len);
        if (pfm->meaCache != VOS_NULL) {
            (void)memset_s(pfm->meaCache, len, 0, len);
        }
    }
#endif
    if (pfm->meaCache != VOS_NULL) {
        pfm->meaCache->mea[mea->cacheIdx] = mea;
    }
    return mea;

error:

    if (mea != VOS_NULL) {
        /* insOk一定为false */
        BKF_FREE(pfm->argInit.memMng, mea);
    }
    return VOS_NULL;
}

STATIC void BkfPfmDelMea(BkfPfm *pfm, BkfPfmMea *mea)
{
    if ((pfm->meaCache != VOS_NULL) && (pfm->meaCache->mea[mea->cacheIdx] == mea)) {
        pfm->meaCache->mea[mea->cacheIdx] = VOS_NULL;
    }
    VOS_AVLL_DELETE(pfm->meaSet, mea->avlNode);
    BKF_FREE(pfm->argInit.memMng, mea);
    return;
}

STATIC void BkfPfmDelAllMea(BkfPfm *pfm)
{
    BkfPfmMea *mea = VOS_NULL;
    void *itor = VOS_NULL;

    for (mea = BkfPfmGetFirstMea(pfm, &itor); mea != VOS_NULL; mea = BkfPfmGetNextMea(pfm, &itor)) {
        BkfPfmDelMea(pfm, mea);
    }
    return;
}

STATIC BkfPfmMea *BkfPfmFindMea(BkfPfm *pfm, char *meaStr)
{
    uint32_t cacheIdx = BKF_PFM_MEA_STR_2CACHE_IDX(meaStr);
    BkfPfmMea *mea = VOS_NULL;
    BOOL hit = VOS_FALSE;

    if (pfm->meaCache != VOS_NULL) {
        mea = pfm->meaCache->mea[cacheIdx];
        hit = (mea != VOS_NULL) && (VOS_StrCmp(mea->meaStr, meaStr) == 0);
        if (hit) {
            return mea;
        }
    }

    mea = VOS_AVLL_FIND(pfm->meaSet, meaStr);
    if ((mea != VOS_NULL) && (pfm->meaCache != VOS_NULL)) {
        pfm->meaCache->mea[cacheIdx] = mea;
    }
    return mea;
}

STATIC BkfPfmMea *BkfPfmFindNextMea(BkfPfm *pfm, char *meaStr)
{
    BkfPfmMea *mea = VOS_NULL;

    mea = VOS_AVLL_FIND_NEXT(pfm->meaSet, meaStr);
    return mea;
}

STATIC BkfPfmMea *BkfPfmGetFirstMea(BkfPfm *pfm, void **itorOutOrNull)
{
    BkfPfmMea *mea = VOS_NULL;

    mea = VOS_AVLL_FIRST(pfm->meaSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (mea != VOS_NULL) ? VOS_AVLL_NEXT(pfm->meaSet, mea->avlNode) : VOS_NULL;
    }
    return mea;
}

STATIC BkfPfmMea *BkfPfmGetNextMea(BkfPfm *pfm, void **itorInOut)
{
    BkfPfmMea *mea = VOS_NULL;

    mea = (*itorInOut);
    *itorInOut = (mea != VOS_NULL) ? VOS_AVLL_NEXT(pfm->meaSet, mea->avlNode) : VOS_NULL;
    return mea;
}

#endif

#if BKF_BLOCK("disp")
/* disp */
void BkfPfmResetAllMea(BkfPfm *pfm)
{
    BkfDisp *disp = pfm->argInit.disp;

    BkfPfmDelAllMea(pfm);
    BKF_DISP_PRINTF(disp, "ok\n");
}

void BkfPfmEnable(BkfPfm *pfm)
{
    BkfDisp *disp = pfm->argInit.disp;

    if (pfm->enable) {
        BKF_DISP_PRINTF(disp, "has enable, not change\n");
        return;
    }
    pfm->enable = VOS_TRUE;
    BKF_DISP_PRINTF(disp, "ok\n");
}

void BkfPfmDisable(BkfPfm *pfm)
{
    BkfDisp *disp = pfm->argInit.disp;

    if (!pfm->enable) {
        BKF_DISP_PRINTF(disp, "has disable, not change\n");
        return;
    }
    pfm->enable = VOS_FALSE;
    BKF_DISP_PRINTF(disp, "ok\n");
}

void BkfPfmDispMea(BkfPfm *pfm)
{
    BkfDisp *disp = pfm->argInit.disp;
    char *lastMeaStr = VOS_NULL;
    BkfPfmMea *mea = VOS_NULL;
    int32_t cnt = 0;
    char *tempStr = VOS_NULL;
    uint8_t buf[BKF_1K];

    lastMeaStr = BKF_DISP_GET_LAST_CTX(disp, &cnt);
    if (lastMeaStr == VOS_NULL) {
        cnt = 0;
        BKF_DISP_PRINTF(disp, "%s\n", BkfPfmGetSummaryStr(pfm, buf, sizeof(buf)));
        mea = BkfPfmGetFirstMea(pfm, VOS_NULL);
    } else {
        mea = BkfPfmFindNextMea(pfm, lastMeaStr);
    }

    if (mea != VOS_NULL) {
        tempStr = BkfPfmGetMeaStr(pfm, mea, buf, sizeof(buf));
        if (tempStr == VOS_NULL) {
            tempStr = "errGetMeaStr";
        }
        BKF_DISP_PRINTF(disp, "[%-4d] - [%-20s]: %s\n", cnt, mea->meaStr, tempStr);
        cnt++;
        BKF_DISP_SAVE_CTX(disp, mea->meaStr, VOS_StrLen(mea->meaStr) + 1, &cnt, sizeof(cnt));
    } else {
        BKF_DISP_PRINTF(disp, " ***total %d mea(s)***\n", cnt);
    }
}

void BkfPfmDispTestAddMea(BkfPfm *pfm)
{
    BkfDisp *disp = pfm->argInit.disp;

    if (!pfm->enable) {
        BKF_DISP_PRINTF(disp, "enable(%u), pls enable\n", pfm->enable);
        return;
    }
    BkfPfmTestAddMea(pfm);
    BKF_DISP_PRINTF(disp, "ok, display result by BkfPfmDispMea\n");
}

STATIC void BkfPfmDispInit(BkfPfm *pfm)
{
    BkfDisp *disp = pfm->argInit.disp;
    char *objName = pfm->name;
    uint32_t ret;

    ret = BkfDispRegObj(disp, objName, pfm);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return;
    }

    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfPfmEnable, "perf measure enable", objName, 0);
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfPfmDisable, "perf measure disable", objName, 0);
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfPfmDispMea, "disp perf measure data", objName, 0);
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfPfmResetAllMea, "reset all perf measure all", objName, 0);
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfPfmDispTestAddMea, "test add perf measure", objName, 0);
}

STATIC void BkfPfmDispUninit(BkfPfm *pfm)
{
    BkfDispUnregObj(pfm->argInit.disp, pfm->name);
}

#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
