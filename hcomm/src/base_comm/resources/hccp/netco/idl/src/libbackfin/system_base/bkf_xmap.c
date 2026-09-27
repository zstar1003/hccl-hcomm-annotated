/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_xmap.h"
#include "bkf_bas_type_mthd.h"
#include "bkf_dl.h"
#include "bkf_str.h"
#include "bkf_assert.h"
#include "v_avll.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)

typedef struct tagBkfXMapX BkfXMapX;

#define BKF_XMAP_X_HASH_MOD 0xff
#define BKF_XMAP_GET_HASH_IDX(key) ((uint32_t)(key) & BKF_XMAP_X_HASH_MOD)
struct tagBkfXMap {
    BkfXMapInitArg argInit;
    char *name;
    AVLL_TREE xSet;
    uint64_t findTotalCnt;
    uint64_t findHitCacheCnt;
    uint64_t findMissCacheCnt;
    uint64_t findNullCnt;
#ifndef BKF_CUT_AVL_CACHE
    BkfXMapX *xCache[BKF_XMAP_X_HASH_MOD + 1];
#endif
};

struct tagBkfXMapX {
    BkfXMap *xMap;
    AVLL_NODE avlNode;
    uintptr_t key;
    const char *keyStr;

    AVLL_TREE cbSet;
    BkfDl cbSetByPrio;
};

typedef struct tagBkfXMapCbKeyWithCookie {
    F_BKF_XMAP_PROC proc;
    void *cookie;
} BkfXMapCbKeyWithCookie;

enum {
    BKF_XMAP_PROC_TYPE_WITH_COOKIE,
    BKF_XMAP_PROC_TYPE_WITHOUT_COOKIE,

    BKF_XMAP_PROC_TYPE_CNT
};

typedef struct tagBkfXMapCbKey {
    uint8_t type;
    uint8_t pad1[3];
    union {
        BkfXMapCbKeyWithCookie procCookie;
        F_BKF_XMAP_PROC_NO_COOKIE proc;
    };
} BkfXMapCbKey;

typedef struct tagBkfXMapCb {
    BkfXMapX *x;
    AVLL_NODE avlNode;
    BkfDlNode dlNode;

    BkfXMapCbKey key;
    uint8_t prio;
    uint8_t pad1[3];
} BkfXMapCb;

/* data op */
STATIC BkfXMapX *BkfXMapAddX(BkfXMap *xMap, uintptr_t key, const char *keyStr);
STATIC void BkfXMapDelX(BkfXMapX *x);
STATIC void BkfXMapDelAllX(BkfXMap *xMap);
STATIC BkfXMapX *BkfXMapFindX(BkfXMap *xMap, uintptr_t key);
STATIC BkfXMapX *BkfXMapGetFirstX(BkfXMap *xMap, void **outItorOrNull);
STATIC BkfXMapX *BkfXMapGetNextX(BkfXMap *xMap, void **inOutItor);

STATIC int32_t BkfXMapCmpCbKey(const BkfXMapCbKey *key1Input, const BkfXMapCbKey *key2InDs);
STATIC BkfXMapCb *BkfXMapAddCb(BkfXMap *xMap, BkfXMapX *x, BkfXMapCbKey *cbKey, uint8_t prio);
STATIC void BkfXMapDelCb(BkfXMap *xMap, BkfXMapCb *cb);
STATIC void BkfXMapDelAllCb(BkfXMap *xMap, BkfXMapX *x);
STATIC BkfXMapCb *BkfXMapFindCb(BkfXMap *xMap, BkfXMapX *x, BkfXMapCbKey *key);
STATIC BkfXMapCb *BkfXMapGetFirstCbByPrio(BkfXMap *xMap, BkfXMapX *x, void **outItorOrNull);
STATIC BkfXMapCb *BkfXMapGetNextCbByPrio(BkfXMap *xMap, BkfXMapX *x, void **inOutItor);
#pragma pack()

/* func */
BkfXMap *BkfXMapInit(BkfXMapInitArg *arg)
{
    BkfXMap *xMap = VOS_NULL;
    uint32_t len;

    if ((arg == VOS_NULL) || (arg->name == VOS_NULL) || (arg->memMng == VOS_NULL)) {
        BKF_ASSERT(0);
        goto error;
    }

    len = sizeof(BkfXMap);
    xMap = BKF_MALLOC(arg->memMng, len);
    if (xMap == VOS_NULL) {
        BKF_ASSERT(0);
        goto error;
    }
    (void)memset_s(xMap, len, 0, len);
    xMap->argInit = *arg;
    VOS_AVLL_INIT_TREE(xMap->xSet, (AVLL_COMPARE)BkfUIntptrCmp, BKF_OFFSET(BkfXMapX, key),
        BKF_OFFSET(BkfXMapX, avlNode));
    xMap->name = BkfStrNew(arg->memMng, "%s_xmap", arg->name);
    xMap->argInit.name = xMap->name;

    return xMap;

error:

    return VOS_NULL;
}

void BkfXMapUninit(BkfXMap *xMap)
{
    if (xMap == VOS_NULL) {
        return;
    }

    BkfXMapDelAllX(xMap);
    BkfStrDel(xMap->name);
    BKF_FREE(xMap->argInit.memMng, xMap);
    return;
}

STATIC BOOL BkfXmapCbKeyIsValid(BkfXMapCbKey *cbKey)
{
    if (cbKey->type == BKF_XMAP_PROC_TYPE_WITH_COOKIE) {
        if (cbKey->procCookie.proc == VOS_NULL) {
            return VOS_FALSE;
        }
    } else {
        if (cbKey->proc == VOS_NULL) {
            return VOS_FALSE;
        }
    }

    return VOS_TRUE;
}

STATIC uint32_t BkfXMapRegExProc(BkfXMap *xMap, uintptr_t key, const char *keyStr, BkfXMapCbKey *cbKey, uint8_t prio)
{
    BkfXMapX *x = VOS_NULL;
    BkfXMapCb *cb = VOS_NULL;

    if ((xMap == VOS_NULL) || (keyStr == VOS_NULL) || (cbKey == VOS_NULL) || (!BkfXmapCbKeyIsValid(cbKey))) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    x = BkfXMapFindX(xMap, key);
    if (x == VOS_NULL) {
        x = BkfXMapAddX(xMap, key, keyStr);
        if (x == VOS_NULL) {
            BKF_ASSERT(0);
            return BKF_ERR;
        }
    }
    cb = BkfXMapAddCb(xMap, x, cbKey, prio);
    if (cb == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    return BKF_OK;
}

uint32_t BkfXMapRegEx(BkfXMap *xMap, uintptr_t key, const char *keyStr, F_BKF_XMAP_PROC proc, void *cookie,
    uint8_t prio)
{
    BkfXMapCbKey cbKey = {0};

    cbKey.type = BKF_XMAP_PROC_TYPE_WITH_COOKIE;
    cbKey.procCookie.proc = proc;
    cbKey.procCookie.cookie = cookie;

    return BkfXMapRegExProc(xMap, key, keyStr, &cbKey, prio);
}

void BkfXMapUnRegEx(BkfXMap *xMap, uintptr_t key, F_BKF_XMAP_PROC proc, void *cookie, uint8_t prio)
{
    (void)prio;
    if (xMap == VOS_NULL) {
        BKF_ASSERT(0);
        return;
    }

    BkfXMapX *x = BkfXMapFindX(xMap, key);
    if (x == VOS_NULL) {
        return;
    }

    BkfXMapCbKey cbKey = {0};
    cbKey.type = BKF_XMAP_PROC_TYPE_WITH_COOKIE;
    cbKey.procCookie.proc = proc;
    cbKey.procCookie.cookie = cookie;

    BkfXMapCb *cb = BkfXMapFindCb(xMap, x, &cbKey);
    if (cb == VOS_NULL) {
        return;
    }
    BkfXMapDelCb(xMap, cb);
    if (BKF_DL_IS_EMPTY(&(x->cbSetByPrio))) {
        BkfXMapDelX(x);
    }
    return;
}

uint32_t BkfXMapRegExNoCookie(BkfXMap *xMap, uintptr_t key, const char *keyStr, F_BKF_XMAP_PROC_NO_COOKIE proc,
    uint8_t prio)
{
    BkfXMapCbKey cbKey = {0};

    cbKey.type = BKF_XMAP_PROC_TYPE_WITHOUT_COOKIE;
    cbKey.proc = proc;

    return BkfXMapRegExProc(xMap, key, keyStr, &cbKey, prio);
}

STATIC uint32_t BkfXMapCbDispatch(BkfXMapCb *cb, void *paramDispatch1, void *paramDispatch2)
{
    if (cb->key.type == BKF_XMAP_PROC_TYPE_WITH_COOKIE) {
        return cb->key.procCookie.proc(cb->key.procCookie.cookie, paramDispatch1, paramDispatch2);
    } else {
        return cb->key.proc(paramDispatch1, paramDispatch2);
    }
}

uint32_t BkfXMapDispatch(BkfXMap *xMap, uintptr_t key, void *paramDispatch1, void *paramDispatch2)
{
    BkfXMapX *x = VOS_NULL;
    BkfXMapCb *cb = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t ret;

    if (xMap == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    x = BkfXMapFindX(xMap, key);
    if (x == VOS_NULL) {
        return BKF_XMAP_DISPATCH_NOT_FIND;
    }
    ret = 0;
    for (cb = BkfXMapGetFirstCbByPrio(xMap, x, &itor); cb != VOS_NULL; cb = BkfXMapGetNextCbByPrio(xMap, x, &itor)) {
        ret |= BkfXMapCbDispatch(cb, paramDispatch1, paramDispatch2);
    }

    return ret;
}

/* data op */
STATIC BkfXMapX *BkfXMapAddX(BkfXMap *xMap, uintptr_t key, const char *keyStr)
{
    BkfXMapX *x = VOS_NULL;
    uint32_t len;
    BOOL isInsOk = VOS_FALSE;

    len = sizeof(BkfXMapX);
    x = BKF_MALLOC(xMap->argInit.memMng, len);
    if (x == VOS_NULL) {
        goto error;
    }
    (void)memset_s(x, len, 0, len);
    x->xMap = xMap;
    VOS_AVLL_INIT_NODE(x->avlNode);
    x->key = key;
    x->keyStr = keyStr;
    VOS_AVLL_INIT_TREE(x->cbSet, (AVLL_COMPARE)BkfXMapCmpCbKey, BKF_OFFSET(BkfXMapCb, key),
        BKF_OFFSET(BkfXMapCb, avlNode));
    BKF_DL_INIT(&x->cbSetByPrio);
    isInsOk = VOS_AVLL_INSERT(xMap->xSet, x->avlNode);
    if (!isInsOk) {
        goto error;
    }
#ifndef BKF_CUT_AVL_CACHE
    uint32_t hashIdx = BKF_XMAP_GET_HASH_IDX(key);
    xMap->xCache[hashIdx] = x;
#endif
    return x;

error:

    if (x != VOS_NULL) {
        /* isInsOk一定为false */
        BKF_FREE(xMap->argInit.memMng, x);
    }
    return VOS_NULL;
}

STATIC void BkfXMapDelX(BkfXMapX *x)
{
    BkfXMap *xMap = x->xMap;

    BkfXMapDelAllCb(xMap, x);
#ifndef BKF_CUT_AVL_CACHE
    uint32_t hashIdx = BKF_XMAP_GET_HASH_IDX(x->key);
    if (xMap->xCache[hashIdx] == x) {
        xMap->xCache[hashIdx] = VOS_NULL;
    }
#endif
    VOS_AVLL_DELETE(xMap->xSet, x->avlNode);
    BKF_FREE(xMap->argInit.memMng, x);
    return;
}

STATIC void BkfXMapDelAllX(BkfXMap *xMap)
{
    BkfXMapX *x = VOS_NULL;
    void *itor = VOS_NULL;

    for (x = BkfXMapGetFirstX(xMap, &itor); x != VOS_NULL; x = BkfXMapGetNextX(xMap, &itor)) {
        BkfXMapDelX(x);
    }
    return;
}

STATIC BkfXMapX *BkfXMapFindX(BkfXMap *xMap, uintptr_t key)
{
    BkfXMapX *x = VOS_NULL;

    xMap->findTotalCnt++;
#ifndef BKF_CUT_AVL_CACHE
    uint32_t hashIdx = BKF_XMAP_GET_HASH_IDX(key);
    x = xMap->xCache[hashIdx];
    if ((x != VOS_NULL) && (key == x->key)) {
        xMap->findHitCacheCnt++;
        return x;
    }
#endif
    x = VOS_AVLL_FIND(xMap->xSet, &key);
    if (x != VOS_NULL) {
        xMap->findMissCacheCnt++;
#ifndef BKF_CUT_AVL_CACHE
        xMap->xCache[hashIdx] = x;
#endif
        return x;
    }

    xMap->findNullCnt++;
    return VOS_NULL;
}

STATIC BkfXMapX *BkfXMapGetFirstX(BkfXMap *xMap, void **outItorOrNull)
{
    BkfXMapX *x = VOS_NULL;

    x = VOS_AVLL_FIRST(xMap->xSet);
    if (outItorOrNull != VOS_NULL) {
        *outItorOrNull = (x != VOS_NULL) ? VOS_AVLL_NEXT(xMap->xSet, x->avlNode) : VOS_NULL;
    }
    return x;
}

STATIC BkfXMapX *BkfXMapGetNextX(BkfXMap *xMap, void **inOutItor)
{
    BkfXMapX *x = VOS_NULL;

    x = (*inOutItor);
    *inOutItor = (x != VOS_NULL) ? VOS_AVLL_NEXT(xMap->xSet, x->avlNode) : VOS_NULL;
    return x;
}

STATIC int32_t BkfXMapCmpCbKey(const BkfXMapCbKey *key1Input, const BkfXMapCbKey *key2InDs)
{
    int32_t ret = key1Input->type - key2InDs->type;
    if (ret != 0) {
        return ret;
    }

    if (key1Input->type == BKF_XMAP_PROC_TYPE_WITH_COOKIE) {
        ret = key1Input->procCookie.proc - key2InDs->procCookie.proc;
        if (ret != 0) {
            return ret;
        } else {
            return BKF_CMP_X(key1Input->procCookie.cookie, key2InDs->procCookie.cookie);
        }
    } else {
        ret = key1Input->proc - key2InDs->proc;
        return ret;
    }
}

STATIC BkfXMapCb *BkfXMapAddCb(BkfXMap *xMap, BkfXMapX *x, BkfXMapCbKey *cbKey, uint8_t prio)
{
    BkfXMapCb *cb = VOS_NULL;
    uint32_t len;
    BOOL insOk = VOS_FALSE;
    BkfXMapCb *temp = VOS_NULL;
    void *itor = VOS_NULL;

    len = sizeof(BkfXMapCb);
    cb = BKF_MALLOC(xMap->argInit.memMng, len);
    if (cb == VOS_NULL) {
        goto error;
    }
    (void)memset_s(cb, len, 0, len);
    cb->x = x;
    VOS_AVLL_INIT_NODE(cb->avlNode);
    BKF_DL_NODE_INIT(&cb->dlNode);
    cb->key = *cbKey;
    cb->prio = prio;
    insOk = VOS_AVLL_INSERT(x->cbSet, cb->avlNode);
    if (!insOk) {
        goto error;
    }
    for (temp = BkfXMapGetFirstCbByPrio(xMap, x, &itor); temp != VOS_NULL;
         temp = BkfXMapGetNextCbByPrio(xMap, x, &itor)) {
        if (cb->prio > temp->prio) {
            BKF_DL_ADD_BEFORE(&temp->dlNode, &cb->dlNode);
            break;
        }
    }
    if (!BKF_DL_NODE_IS_IN(&cb->dlNode)) {
        BKF_DL_ADD_LAST(&x->cbSetByPrio, &cb->dlNode);
    }

    return cb;

error:

    if (cb != VOS_NULL) {
        /* isInsOk一定为false */
        BKF_FREE(xMap->argInit.memMng, cb);
    }
    return VOS_NULL;
}

STATIC void BkfXMapDelCb(BkfXMap *xMap, BkfXMapCb *cb)
{
    BkfXMapX *x = cb->x;

    BKF_DL_REMOVE(&cb->dlNode);
    VOS_AVLL_DELETE(x->cbSet, cb->avlNode);
    BKF_FREE(xMap->argInit.memMng, cb);
    return;
}

STATIC void BkfXMapDelAllCb(BkfXMap *xMap, BkfXMapX *x)
{
    BkfXMapCb *cb = VOS_NULL;
    void *itor = VOS_NULL;

    for (cb = BkfXMapGetFirstCbByPrio(xMap, x, &itor); cb != VOS_NULL; cb = BkfXMapGetNextCbByPrio(xMap, x, &itor)) {
        BkfXMapDelCb(xMap, cb);
    }
    return;
}

STATIC BkfXMapCb *BkfXMapFindCb(BkfXMap *xMap, BkfXMapX *x, BkfXMapCbKey *key)
{
    (void)xMap;
    BkfXMapCb *cb = VOS_NULL;

    cb = VOS_AVLL_FIND(x->cbSet, key);
    return cb;
}

STATIC BkfXMapCb *BkfXMapGetFirstCbByPrio(BkfXMap *xMap, BkfXMapX *x, void **outItorOrNull)
{
    (void)xMap;
    BkfDlNode *tempNode = VOS_NULL;
    BkfXMapCb *cb = VOS_NULL;

    tempNode = BKF_DL_GET_FIRST(&x->cbSetByPrio);
    if (tempNode != VOS_NULL) {
        cb = BKF_DL_GET_ENTRY(tempNode, BkfXMapCb, dlNode);
        if (outItorOrNull != VOS_NULL) {
            *outItorOrNull = BKF_DL_GET_NEXT(&x->cbSetByPrio, &cb->dlNode);
        }
    } else {
        cb = VOS_NULL;
        if (outItorOrNull != VOS_NULL) {
            *outItorOrNull = VOS_NULL;
        }
    }
    return cb;
}

STATIC BkfXMapCb *BkfXMapGetNextCbByPrio(BkfXMap *xMap, BkfXMapX *x, void **inOutItor)
{
    (void)xMap;
    BkfDlNode *tempNode = VOS_NULL;
    BkfXMapCb *cb = VOS_NULL;

    tempNode = *inOutItor;
    if (tempNode != VOS_NULL) {
        cb = BKF_DL_GET_ENTRY(tempNode, BkfXMapCb, dlNode);
        *inOutItor = BKF_DL_GET_NEXT(&x->cbSetByPrio, &cb->dlNode);
    } else {
        cb = VOS_NULL;
        *inOutItor = VOS_NULL;
    }
    return cb;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
