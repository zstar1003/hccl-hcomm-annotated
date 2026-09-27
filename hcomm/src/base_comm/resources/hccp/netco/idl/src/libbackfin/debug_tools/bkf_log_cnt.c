/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_MOD_NAME ((logCnt)->name)

#include "bkf_log_cnt.h"
#include "bkf_str.h"
#include "bkf_assert.h"
#include "v_avll.h"
#include "v_stringlib.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
typedef struct tagBkfLogCntMod BkfLogCntMod;

#define BKF_LOG_CNT_MOD_HASH_MOD 0x1ff
#define BKF_LOG_CNT_GET_MOD_HASH_IDX(line) ((line) & BKF_LOG_CNT_MOD_HASH_MOD)
struct tagBkfLogCnt {
    BkfLogCntInitArg argInit;
    char *name;

    AVLL_TREE modSet;
    AVLL_TREE modSetByNameOrder;
#ifndef BKF_CUT_AVL_CACHE
    BkfLogCntMod *modCache[BKF_LOG_CNT_MOD_HASH_MOD + 1];
#endif
    uint32_t seqSeed; /* 会溢出，没有关系 */
    int32_t modCnt;
    int64_t callCntNotLogInMod; /* 因为mod有数量限制，某些可能无法明细记录。记录在此 */
};

typedef struct tagBkfLogCntModKey {
    char *name;
    uint16_t line;
    uint8_t pad1[2];
} BkfLogCntModKey;
struct tagBkfLogCntMod {
    AVLL_NODE avlNode;
    AVLL_NODE avlNodeByNameOrder;

    uint32_t lastCallSeq;
    uint32_t callCnt;
    BkfLogCntModKey key;
    char name[0];
};
BKF_STATIC_ASSERT(BKF_MBR_IS_LAST(BkfLogCntMod, name));

/* data op */
STATIC int32_t BkfLogCntCmpMod(const BkfLogCntModKey *key1Input, const BkfLogCntModKey *key2InDs);
STATIC int32_t BkfLogCntCmpModNameFirst(const BkfLogCntModKey *key1Input, const BkfLogCntModKey *key2InDs);
STATIC BkfLogCntMod *BkfLogCntAddMod(BkfLogCnt *logCnt, char *modName, uint16_t line);
STATIC void BkfLogCntDelMod(BkfLogCnt *logCnt, BkfLogCntMod *mod);
STATIC void BkfLogCntDelAllMod(BkfLogCnt *logCnt);
STATIC BkfLogCntMod *BkfLogCntFindMod(BkfLogCnt *logCnt, char *modName, uint16_t line);
STATIC BkfLogCntMod *BkfLogCntFindNextModByNameOrder(BkfLogCnt *logCnt, char *modName, uint16_t line);
STATIC BkfLogCntMod *BkfLogCntGetFirstModByNameOrder(BkfLogCnt *logCnt, void **itorOutOrNull);
STATIC BkfLogCntMod *BkfLogCntGetNextModByNameOrder(BkfLogCnt *logCnt, void **itorInOut);

/* disp */
STATIC void BkfLogCntDispInit(BkfLogCnt *logCnt);
STATIC void BkfLogCntDispUninit(BkfLogCnt *logCnt);

#pragma pack()
#endif

#if BKF_BLOCK("公有函数定义")
BkfLogCnt *BkfLogCntInit(BkfLogCntInitArg *arg)
{
    BkfLogCnt *logCnt = VOS_NULL;
    uint32_t len;

    if ((arg == VOS_NULL) || (arg->name == VOS_NULL) || (arg->memMng == VOS_NULL) || (arg->disp == VOS_NULL) ||
        (arg->modCntMax <= 0)) {
        goto error;
    }

    len = sizeof(BkfLogCnt);
    logCnt = BKF_MALLOC(arg->memMng, len);
    if (logCnt == VOS_NULL) {
        goto error;
    }
    (void)memset_s(logCnt, len, 0, len);
    logCnt->argInit = *arg;
    logCnt->name = BkfStrNew(arg->memMng, "%s_logCnt", arg->name);
    logCnt->argInit.name = logCnt->name;
    VOS_AVLL_INIT_TREE(logCnt->modSet, (AVLL_COMPARE)BkfLogCntCmpMod, BKF_OFFSET(BkfLogCntMod, key),
        BKF_OFFSET(BkfLogCntMod, avlNode));
    VOS_AVLL_INIT_TREE(logCnt->modSetByNameOrder, (AVLL_COMPARE)BkfLogCntCmpModNameFirst, BKF_OFFSET(BkfLogCntMod, key),
        BKF_OFFSET(BkfLogCntMod, avlNodeByNameOrder));
    BkfLogCntDispInit(logCnt);

    return logCnt;

error:

    if (logCnt != VOS_NULL) {
        BkfLogCntUninit(logCnt);
    }
    return VOS_NULL;
}

void BkfLogCntUninit(BkfLogCnt *logCnt)
{
    if (logCnt == VOS_NULL) {
        return;
    }

    BkfLogCntDispUninit(logCnt);
    BkfLogCntDelAllMod(logCnt);
    BkfStrDel(logCnt->name);
    BKF_FREE(logCnt->argInit.memMng, logCnt);
    return;
}

void BkfLogCntFunc(BkfLogCnt *logCnt, char *modName, uint16_t line)
{
    BkfLogCntMod *mod = VOS_NULL;

    if ((logCnt == VOS_NULL) || (modName == VOS_NULL)) {
        return;
    }

    mod = BkfLogCntFindMod(logCnt, modName, line);
    if (mod == VOS_NULL) {
        mod = BkfLogCntAddMod(logCnt, modName, line);
        if (mod == VOS_NULL) {
            logCnt->callCntNotLogInMod++;
            return;
        }
    }

    mod->lastCallSeq = BKF_GET_NEXT_VAL(logCnt->seqSeed);
    mod->callCnt++;
    return;
}

#endif

#if BKF_BLOCK("私有函数定义")
/* data op */
STATIC int32_t BkfLogCntCmpMod(const BkfLogCntModKey *key1Input, const BkfLogCntModKey *key2InDs)
{
    if (key1Input->line > key2InDs->line) {
        return 1;
    } else if (key1Input->line < key2InDs->line) {
        return -1;
    } else {
        return VOS_StrCmp(key1Input->name, key2InDs->name);
    }
}

STATIC int32_t BkfLogCntCmpModNameFirst(const BkfLogCntModKey *key1Input, const BkfLogCntModKey *key2InDs)
{
    int32_t ret = VOS_StrCmp(key1Input->name, key2InDs->name);
    if (ret != 0) {
        return ret;
    }
    if (key1Input->line > key2InDs->line) {
        return 1;
    } else if (key1Input->line < key2InDs->line) {
        return -1;
    } else {
        return 0;
    }
}

STATIC BkfLogCntMod *BkfLogCntAddMod(BkfLogCnt *logCnt, char *modName, uint16_t line)
{
    BkfLogCntMod *mod = VOS_NULL;
    uint32_t strLen;
    uint32_t len;
    int32_t ret;
    BOOL isInsOk = VOS_FALSE;
    BOOL isInsByOrderOk = VOS_FALSE;

    if (logCnt->modCnt >= logCnt->argInit.modCntMax) {
        return VOS_NULL;
    }

    strLen = VOS_StrLen(modName);
    len = sizeof(BkfLogCntMod) + strLen + 1;
    mod = BKF_MALLOC(logCnt->argInit.memMng, len);
    if (mod == VOS_NULL) {
        goto error;
    }
    (void)memset_s(mod, len, 0, len);
    mod->key.name = mod->name;
    mod->key.line = line;
    ret = snprintf_truncated_s(mod->name, strLen + 1, "%s", modName);
    if (ret < 0) {
        goto error;
    }
    VOS_AVLL_INIT_NODE(mod->avlNode);
    isInsOk = VOS_AVLL_INSERT(logCnt->modSet, mod->avlNode);
    if (!isInsOk) {
        goto error;
    }
    VOS_AVLL_INIT_NODE(mod->avlNodeByNameOrder);
    isInsByOrderOk = VOS_AVLL_INSERT(logCnt->modSetByNameOrder, mod->avlNodeByNameOrder);
    if (!isInsByOrderOk) {
        goto error;
    }
#ifndef BKF_CUT_AVL_CACHE
    uint32_t hashIdx = BKF_LOG_CNT_GET_MOD_HASH_IDX(line);
    logCnt->modCache[hashIdx] = mod;
#endif
    logCnt->modCnt++;
    return mod;

error:

    if (mod != VOS_NULL) {
        /* isInsByOrderOk一定为false */
        if (isInsOk) {
            VOS_AVLL_DELETE(logCnt->modSet, mod->avlNode);
        }
        BKF_FREE(logCnt->argInit.memMng, mod);
    }
    return VOS_NULL;
}

STATIC void BkfLogCntDelMod(BkfLogCnt *logCnt, BkfLogCntMod *mod)
{
    logCnt->modCnt--;
#ifndef BKF_CUT_AVL_CACHE
    uint32_t hashIdx = BKF_LOG_CNT_GET_MOD_HASH_IDX(mod->key.line);
    if (logCnt->modCache[hashIdx] == mod) {
        logCnt->modCache[hashIdx] = VOS_NULL;
    }
#endif
    VOS_AVLL_DELETE(logCnt->modSet, mod->avlNode);
    VOS_AVLL_DELETE(logCnt->modSetByNameOrder, mod->avlNodeByNameOrder);
    BKF_FREE(logCnt->argInit.memMng, mod);
    return;
}

STATIC void BkfLogCntDelAllMod(BkfLogCnt *logCnt)
{
    BkfLogCntMod *mod = VOS_NULL;
    void *itor = VOS_NULL;

    for (mod = BkfLogCntGetFirstModByNameOrder(logCnt, &itor); mod != VOS_NULL;
         mod = BkfLogCntGetNextModByNameOrder(logCnt, &itor)) {
        BkfLogCntDelMod(logCnt, mod);
    }
    return;
}

STATIC BkfLogCntMod *BkfLogCntFindMod(BkfLogCnt *logCnt, char *modName, uint16_t line)
{
    BkfLogCntMod *mod = VOS_NULL;
    BkfLogCntModKey key = {modName, line, {}};
#ifndef BKF_CUT_AVL_CACHE
    uint32_t hashIdx = BKF_LOG_CNT_GET_MOD_HASH_IDX(line);
    mod = logCnt->modCache[hashIdx];
    if ((mod != VOS_NULL) && (BkfLogCntCmpMod(&key, &mod->key) == 0)) {
        return mod;
    }
#endif
    mod = VOS_AVLL_FIND(logCnt->modSet, &key);
    if (mod != VOS_NULL) {
#ifndef BKF_CUT_AVL_CACHE
        logCnt->modCache[hashIdx] = mod;
#endif
        return mod;
    }

    return VOS_NULL;
}

STATIC BkfLogCntMod *BkfLogCntFindNextModByNameOrder(BkfLogCnt *logCnt, char *modName, uint16_t line)
{
    BkfLogCntMod *mod = VOS_NULL;
    BkfLogCntModKey key = {modName, line, {}};

    mod = VOS_AVLL_FIND_NEXT(logCnt->modSetByNameOrder, &key);
    return mod;
}

STATIC BkfLogCntMod *BkfLogCntGetFirstModByNameOrder(BkfLogCnt *logCnt, void **itorOutOrNull)
{
    BkfLogCntMod *mod = VOS_NULL;

    mod = VOS_AVLL_FIRST(logCnt->modSetByNameOrder);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (mod != VOS_NULL) ? VOS_AVLL_NEXT(logCnt->modSetByNameOrder, mod->avlNodeByNameOrder)
                                           : VOS_NULL;
    }
    return mod;
}

STATIC BkfLogCntMod *BkfLogCntGetNextModByNameOrder(BkfLogCnt *logCnt, void **itorInOut)
{
    BkfLogCntMod *mod = VOS_NULL;

    mod = (*itorInOut);
    *itorInOut = (mod != VOS_NULL) ? VOS_AVLL_NEXT(logCnt->modSetByNameOrder, mod->avlNodeByNameOrder) : VOS_NULL;
    return mod;
}

#endif

#if BKF_BLOCK("disp")
STATIC int32_t BkfLogCntGetModCnt(BkfLogCnt *logCnt)
{
    BkfLogCntMod *mod = VOS_NULL;
    void *itor = VOS_NULL;
    int32_t modCnt = 0;

    for (mod = BkfLogCntGetFirstModByNameOrder(logCnt, &itor); mod != VOS_NULL;
         mod = BkfLogCntGetNextModByNameOrder(logCnt, &itor)) {
        modCnt++;
    }
    return modCnt;
}
STATIC int64_t BkfLogCntGetCallCnt(BkfLogCnt *logCnt)
{
    BkfLogCntMod *mod = VOS_NULL;
    void *itor = VOS_NULL;
    int64_t callCnt = 0;

    for (mod = BkfLogCntGetFirstModByNameOrder(logCnt, &itor); mod != VOS_NULL;
         mod = BkfLogCntGetNextModByNameOrder(logCnt, &itor)) {
        callCnt++;
    }
    return callCnt;
}
void BkfLogCntDisp(BkfLogCnt *logCnt)
{
    BkfDisp *disp = logCnt->argInit.disp;
    int32_t modCnt;
    int64_t callCntLogInMod;

    modCnt = BkfLogCntGetModCnt(logCnt);
    callCntLogInMod = BkfLogCntGetCallCnt(logCnt);

    BKF_DISP_PRINTF(disp, "%s\n", logCnt->name);
    BKF_DISP_PRINTF(disp, "modCnt(%d/%d)/modCntMax(%d)/seqSeed(%d)\n", modCnt, logCnt->modCnt,
        logCnt->argInit.modCntMax, logCnt->seqSeed);
    BKF_DISP_PRINTF(disp, "callCntLogInMod(%" VOS_PRId64 ")/callCntNotLogInMod(%" VOS_PRId64 ")\n", callCntLogInMod,
        logCnt->callCntNotLogInMod);
}

typedef struct tagLogCntDispModTemp {
    uint16_t line;
    uint8_t pad1[2];
    int32_t modCnt;
} LogCntDispModTemp;
void BkfLogCntDispMod(BkfLogCnt *logCnt)
{
    BkfDisp *disp = logCnt->argInit.disp;
    char *lastModName = VOS_NULL;
    LogCntDispModTemp curTemp = {0};
    BkfLogCntMod *mod = VOS_NULL;

    lastModName = BKF_DISP_GET_LAST_CTX(disp, &curTemp);
    if (lastModName == VOS_NULL) {
        mod = BkfLogCntGetFirstModByNameOrder(logCnt, VOS_NULL);
    } else {
        mod = BkfLogCntFindNextModByNameOrder(logCnt, lastModName, curTemp.line);
    }

    if (mod != VOS_NULL) {
        BKF_DISP_PRINTF(disp, "mod[%d, %s/%u]: callCnt(%d)/lastCallSeq(%d)\n", curTemp.modCnt, mod->key.name,
            mod->key.line, mod->callCnt, mod->lastCallSeq);

        curTemp.modCnt++;
        curTemp.line = mod->key.line;
        BKF_DISP_SAVE_CTX(disp, mod->key.name, VOS_StrLen(mod->key.name) + 1, &curTemp, sizeof(curTemp));
    } else {
        BKF_DISP_PRINTF(disp, " ***total %d logCntMod(s), ***\n", curTemp.modCnt);
    }
}

void BkfLogCntTest(BkfLogCnt *logCnt)
{
    BkfDisp *disp = logCnt->argInit.disp;

    BKF_LOG_CNT(logCnt);
    BKF_DISP_PRINTF(disp, "ok\n");
}

void BkfLogCntReset(BkfLogCnt *logCnt)
{
    BkfDisp *disp = logCnt->argInit.disp;

    BkfLogCntDelAllMod(logCnt);
    logCnt->callCntNotLogInMod = 0;
    BKF_DISP_PRINTF(disp, "ok\n");
}

STATIC void BkfLogCntDispInit(BkfLogCnt *logCnt)
{
    BkfDisp *disp = logCnt->argInit.disp;
    char *objName = logCnt->name;
    uint32_t ret;

    ret = BkfDispRegObj(disp, objName, logCnt);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return;
    }

    (void)BKF_DISP_REG_FUNC(disp, BkfLogCntDisp, "disp logCnt info", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfLogCntDispMod, "disp logCnt each mod", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfLogCntTest, "test logCnt", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfLogCntReset, "reset logCnt", objName, 0);
}

STATIC void BkfLogCntDispUninit(BkfLogCnt *logCnt)
{
    BkfDispUnregObj(logCnt->argInit.disp, logCnt->name);
}

#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
