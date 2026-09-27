/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((chMng)->log)
#define BKF_MOD_NAME ((chMng)->name)

#include "bkf_ch_ser.h"
#include "bkf_ch_ser_conn_id_base.h"
#include "bkf_bas_type_mthd.h"
#include "bkf_str.h"
#include "bkf_assert.h"
#include "v_avll.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
#define BKF_CH_SER_TYPE_HASH_MOD 0x0f
#define BKF_CH_SER_GET_TYPE_HASH_IDX(typeId) BKF_GET_U8_FOLD4_VAL((uint8_t)(typeId))
typedef struct tagBkfChSerType BkfChSerType;
struct tagBkfChSerMng {
    BkfChSerMngInitArg argInit;
    char *name;
    BkfLog *log;

    uint8_t hasSetSelfCid : 1;
    uint8_t hasEnable : 1;
    uint8_t rsv1 : 6;
    uint8_t pad1[3];
    uint32_t selfCid;
    AVLL_TREE typeSet;
#ifndef BKF_CUT_AVL_CACHE
    BkfChSerType *typeCache[BKF_CH_SER_TYPE_HASH_MOD + 1];
#endif
};

struct tagBkfChSerType {
    AVLL_NODE avlNode;
    BkfChSerTypeVTbl vTbl;
    BkfChSer *ch;
};

/* data op */
STATIC BkfChSerType *BkfChSerAddType(BkfChSerMng *chMng, BkfChSerTypeVTbl *vTbl);
STATIC void BkfChSerDelType(BkfChSerMng *chMng, BkfChSerType *chType);
STATIC void BkfChSerDelAllType(BkfChSerMng *chMng);
STATIC BkfChSerType *BkfChSerFindType(BkfChSerMng *chMng, uint8_t typeId);
STATIC BkfChSerType *BkfChSerFindNextType(BkfChSerMng *chMng, uint8_t typeId);
STATIC BkfChSerType *BkfChSerGetFirstType(BkfChSerMng *chMng, void **itorOutOrNull);
STATIC BkfChSerType *BkfChSerGetNextType(BkfChSerMng *chMng, void **itorInOut);

#pragma pack()
#endif

#if BKF_BLOCK("公有函数定义")
/* func */
BkfChSerMng *BkfChSerInit(BkfChSerMngInitArg *arg)
{
    BOOL argIsInvalid = VOS_FALSE;
    BkfChSerMng *chMng = VOS_NULL;
    uint32_t len;

    argIsInvalid = (arg == VOS_NULL) || (arg->name == VOS_NULL) || (arg->memMng == VOS_NULL) ||
                   (arg->disp == VOS_NULL) || (arg->logCnt == VOS_NULL) || (arg->pfm == VOS_NULL) ||
                   (arg->xMap == VOS_NULL) || (arg->tmrMng == VOS_NULL) || (arg->jobMng == VOS_NULL);
    if (argIsInvalid) {
        goto error;
    }

    len = sizeof(BkfChSerMng);
    chMng = BKF_MALLOC(arg->memMng, len);
    if (chMng == VOS_NULL) {
        goto error;
    }
    (void)memset_s(chMng, len, 0, len);
    chMng->argInit = *arg;
    chMng->log = arg->log;

    VOS_AVLL_INIT_TREE(chMng->typeSet, (AVLL_COMPARE)BkfUCharCmp,
        BKF_OFFSET(BkfChSerType, vTbl) + BKF_OFFSET(BkfChSerTypeVTbl, typeId), BKF_OFFSET(BkfChSerType, avlNode));
    chMng->name = BkfStrNew(arg->memMng, "%s_ch_ser", arg->name);
    if (chMng->name == VOS_NULL) {
        BKF_ASSERT(0);
        goto error;
    }
    chMng->argInit.name = chMng->name;

    BKF_LOG_INFO(BKF_LOG_HND, "chMng(%#x), init ok\n", BKF_MASK_ADDR(chMng));
    return chMng;

error:

    BkfChSerUninit(chMng);
    return VOS_NULL;
}

void BkfChSerUninit(BkfChSerMng *chMng)
{
    if (chMng == VOS_NULL) {
        return;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "chMng(%#x), 2uninit\n", BKF_MASK_ADDR(chMng));

    BkfChSerDelAllType(chMng);
    if (chMng->name != VOS_NULL) {
        BkfStrDel(chMng->name);
    }
    BKF_FREE(chMng->argInit.memMng, chMng);
    return;
}

uint32_t BkfChSerSetSelfCid(BkfChSerMng *chMng, uint32_t selfCid)
{
    BkfChSerType *chType = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t ret;

    if (chMng == VOS_NULL) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "chMng(%#x), selfCid(%#x)/hasSetSelfCid(%u)\n", BKF_MASK_ADDR(chMng), selfCid,
        chMng->hasSetSelfCid);
    if (chMng->hasSetSelfCid) {
        return BKF_ERR;
    }

    chMng->hasSetSelfCid = VOS_TRUE;
    chMng->selfCid = selfCid;
    ret = BKF_OK;
    for (chType = BkfChSerGetFirstType(chMng, &itor); chType != VOS_NULL; chType = BkfChSerGetNextType(chMng, &itor)) {
        if (chType->vTbl.setSelfCid != VOS_NULL) {
            ret |= chType->vTbl.setSelfCid(chType->ch, selfCid);
        }
    }
    return ret;
}

uint32_t BkfChSerSetCera(BkfChSerMng *chMng, BkfCera *cera)
{
    if (chMng == VOS_NULL || cera == VOS_NULL) {
        return BKF_ERR;
    }
    BkfChSerType *chType = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t ret = BKF_OK;
    uint8_t buf[BKF_1K / 8];
    BKF_LOG_DEBUG(BKF_LOG_HND, "chMng(%#x), cera %s\n", BKF_MASK_ADDR(chMng), BkfCeraGetStr(cera, buf, sizeof(buf)));
    for (chType = BkfChSerGetFirstType(chMng, &itor); chType != VOS_NULL; chType = BkfChSerGetNextType(chMng, &itor)) {
        if (chType->vTbl.setCera != VOS_NULL) {
            ret |= chType->vTbl.setCera(chType->ch, cera);
        }
    }
    return ret;
}

uint32_t BkfChSerUnSetCera(BkfChSerMng *chMng)
{
    if (chMng == VOS_NULL) {
        return BKF_ERR;
    }
    BkfChSerType *chType = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t ret = BKF_OK;
    BKF_LOG_DEBUG(BKF_LOG_HND, "chMng(%#x)\n", BKF_MASK_ADDR(chMng));
    for (chType = BkfChSerGetFirstType(chMng, &itor); chType != VOS_NULL; chType = BkfChSerGetNextType(chMng, &itor)) {
        if (chType->vTbl.unSetCera != VOS_NULL) {
            ret |= chType->vTbl.unSetCera(chType->ch);
        }
    }
    return ret;
}

uint32_t BkfChSerRegType(BkfChSerMng *chMng, BkfChSerTypeVTbl *vTbl)
{
    BOOL argIsInvalid = VOS_FALSE;
    BkfChSerType *chType = VOS_NULL;
    uint32_t ret;

    argIsInvalid = (chMng == VOS_NULL) || (vTbl == VOS_NULL) || (vTbl->name == VOS_NULL) || (vTbl->init == VOS_NULL) ||
                   (vTbl->uninit == VOS_NULL) || (vTbl->enable == VOS_NULL) || (vTbl->listen == VOS_NULL) ||
                   (vTbl->unlisten == VOS_NULL) || (vTbl->mallocDataBuf == VOS_NULL) ||
                   (vTbl->freeDataBuf == VOS_NULL) || (vTbl->write == VOS_NULL) || (vTbl->close == VOS_NULL);
    if (argIsInvalid) {
        return BKF_ERR;
    }

    BKF_LOG_DEBUG(BKF_LOG_HND, "chMng(%#x)/type(%u, %s), vTbl(%#x) enable %u\n", BKF_MASK_ADDR(chMng), vTbl->typeId,
        vTbl->name, BKF_MASK_ADDR(vTbl), chMng->hasEnable);

    chType = BkfChSerAddType(chMng, vTbl);
    if (chType == VOS_NULL) {
        return BKF_ERR;
    }
    if (chMng->hasSetSelfCid && (chType->vTbl.setSelfCid != VOS_NULL)) {
        ret = chType->vTbl.setSelfCid(chType->ch, chMng->selfCid);
        if (ret != BKF_OK) {
            BkfChSerDelType(chMng, chType);
            return BKF_ERR;
        }
    }

    return BKF_OK;
}

void BkfChSerUnRegType(BkfChSerMng *chMng, uint8_t chType)
{
    BkfChSerType *chSerType = BkfChSerFindType(chMng, chType);
    if (chSerType) {
        BkfChSerDelType(chMng, chSerType);
    }
    return;
}

uint32_t BkfChSerEnable(BkfChSerMng *chMng, BkfChSerEnableArg *arg)
{
    BkfChSerType *chType = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t ret;

    if ((chMng == VOS_NULL) || (arg == VOS_NULL)) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "chMng(%#x), arg(%#x), hasEnable(%u)\n", BKF_MASK_ADDR(chMng), BKF_MASK_ADDR(arg),
        chMng->hasEnable);

    if (chMng->hasEnable) {
        return BKF_ERR;
    }
    chMng->hasEnable = VOS_TRUE;

    ret = BKF_OK;
    for (chType = BkfChSerGetFirstType(chMng, &itor); chType != VOS_NULL; chType = BkfChSerGetNextType(chMng, &itor)) {
        ret |= chType->vTbl.enable(chType->ch, arg);
    }
    return ret;
}

uint32_t BkfChSerListen(BkfChSerMng *chMng, BkfUrl *urlSelf)
{
    BOOL argIsInvalid = VOS_FALSE;
    BkfChSerType *chType = VOS_NULL;
    uint8_t buf[BKF_1K / 8];

    argIsInvalid = (chMng == VOS_NULL) || (urlSelf == VOS_NULL) || !BkfUrlIsValid(urlSelf);
    if (argIsInvalid) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "chMng(%#x), urlSelf(%s)\n", BKF_MASK_ADDR(chMng),
        BkfUrlGetStr(urlSelf, buf, sizeof(buf)));

    chType = BkfChSerFindType(chMng, urlSelf->type);
    if (chType == VOS_NULL) {
        return BKF_ERR;
    }

    return chType->vTbl.listen(chType->ch, urlSelf);
}

void BkfChSerUnlisten(BkfChSerMng *chMng, BkfUrl *urlSelf)
{
    BOOL argIsInvalid = VOS_FALSE;
    BkfChSerType *chType = VOS_NULL;
    uint8_t buf[BKF_1K / 8];

    argIsInvalid = (chMng == VOS_NULL) || (urlSelf == VOS_NULL) || !BkfUrlIsValid(urlSelf);
    if (argIsInvalid) {
        return;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "chMng(%#x), urlSelf(%s)\n", BKF_MASK_ADDR(chMng),
        BkfUrlGetStr(urlSelf, buf, sizeof(buf)));

    chType = BkfChSerFindType(chMng, urlSelf->type);
    if (chType == VOS_NULL) {
        return;
    }

    chType->vTbl.unlisten(chType->ch, urlSelf);
    return;
}

STATIC BkfChSerType *BkfChSerGetTypeByConnId(BkfChSerMng *chMng, BkfChSerConnId *connId)
{
    BkfChSerConnIdBase *base = VOS_NULL;

    base = BkfChSerConnIdGetBase(connId);
    if (base == VOS_NULL) {
        return VOS_NULL;
    }

    return BkfChSerFindType(chMng, base->urlType);
}
void *BkfChSerMallocDataBuf(BkfChSerMng *chMng, BkfChSerConnId *connId, int32_t dataBufLen)
{
    BOOL argIsInvalid = VOS_FALSE;
    BkfChSerType *chType = VOS_NULL;

    argIsInvalid = (chMng == VOS_NULL) || (connId == VOS_NULL);
    if (argIsInvalid) {
        return VOS_NULL;
    }

    chType = BkfChSerGetTypeByConnId(chMng, connId);
    if (chType == VOS_NULL) {
        return VOS_NULL;
    }

    return chType->vTbl.mallocDataBuf(connId, dataBufLen);
}

void BkfChSerFreeDataBuf(BkfChSerMng *chMng, BkfChSerConnId *connId, void *dataBuf)
{
    BOOL argIsInvalid = VOS_FALSE;
    BkfChSerType *chType = VOS_NULL;

    argIsInvalid = (chMng == VOS_NULL) || (connId == VOS_NULL);
    if (argIsInvalid) {
        return;
    }

    chType = BkfChSerGetTypeByConnId(chMng, connId);
    if (chType == VOS_NULL) {
        return;
    }

    chType->vTbl.freeDataBuf(connId, dataBuf);
    return;
}

int32_t BkfChSerRead(BkfChSerMng *chMng, BkfChSerConnId *connId, void *dataBuf, int32_t bufLen)
{
    BOOL argIsInvalid = VOS_FALSE;
    BkfChSerType *chType = VOS_NULL;

    argIsInvalid = (chMng == VOS_NULL) || (connId == VOS_NULL);
    if (argIsInvalid) {
        return -1;
    }

    chType = BkfChSerGetTypeByConnId(chMng, connId);
    if (chType == VOS_NULL) {
        return -1;
    }
    if (chType->vTbl.read == VOS_NULL) {
        BKF_ASSERT(0);
        return -1;
    }

    return chType->vTbl.read(connId, dataBuf, bufLen);
}

int32_t BkfChSerWrite(BkfChSerMng *chMng, BkfChSerConnId *connId, void *dataBuf, int32_t dataLen)
{
    BOOL argIsInvalid = VOS_FALSE;
    BkfChSerType *chType = VOS_NULL;

    argIsInvalid = (connId == VOS_NULL);
    if (argIsInvalid) {
        return -1;
    }

    chType = BkfChSerGetTypeByConnId(chMng, connId);
    if (chType == VOS_NULL) {
        return -1;
    }

    return chType->vTbl.write(connId, dataBuf, dataLen);
}

void BkfChSerClose(BkfChSerMng *chMng, BkfChSerConnId *connId)
{
    BkfChSerType *chType = VOS_NULL;

    if ((chMng == VOS_NULL) || (connId == VOS_NULL)) {
        return;
    }

    chType = BkfChSerGetTypeByConnId(chMng, connId);
    if (chType == VOS_NULL) {
        return;
    }

    chType->vTbl.close(connId);
    return;
}

#endif

#if BKF_BLOCK("私有函数定义")
/* data op */
STATIC BkfChSerType *BkfChSerAddType(BkfChSerMng *chMng, BkfChSerTypeVTbl *vTbl)
{
    BkfChSerType *chType = VOS_NULL;
    uint32_t len;
    BkfChSerInitArg arg = {.base = &chMng->argInit, .name = vTbl->name};
    BOOL insOk = VOS_FALSE;

    len = sizeof(BkfChSerType);
    chType = BKF_MALLOC(chMng->argInit.memMng, len);
    if (chType == VOS_NULL) {
        goto error;
    }
    (void)memset_s(chType, len, 0, len);
    chType->vTbl = *vTbl;
    VOS_AVLL_INIT_NODE(chType->avlNode);
    insOk = VOS_AVLL_INSERT(chMng->typeSet, chType->avlNode);
    if (!insOk) {
        goto error;
    }
    chType->ch = chType->vTbl.init(&arg);
    if (chType->ch == VOS_NULL) {
        goto error;
    }
#ifndef BKF_CUT_AVL_CACHE
    uint8_t hashIdx = BKF_CH_SER_GET_TYPE_HASH_IDX(vTbl->typeId);
    chMng->typeCache[hashIdx] = chType;
#endif
    return chType;

error:

    if (chType != VOS_NULL) {
        /* chType->ch 一定为空 */
        if (insOk) {
            VOS_AVLL_DELETE(chMng->typeSet, chType->avlNode);
        }
        BKF_FREE(chMng->argInit.memMng, chType);
    }
    return VOS_NULL;
}

STATIC void BkfChSerDelType(BkfChSerMng *chMng, BkfChSerType *chType)
{
#ifndef BKF_CUT_AVL_CACHE
    uint8_t hashIdx = BKF_CH_SER_GET_TYPE_HASH_IDX(chType->vTbl.typeId);
    if (chMng->typeCache[hashIdx] == chType) {
        chMng->typeCache[hashIdx] = VOS_NULL;
    }
#endif
    chType->vTbl.uninit(chType->ch);
    VOS_AVLL_DELETE(chMng->typeSet, chType->avlNode);
    BKF_FREE(chMng->argInit.memMng, chType);
    return;
}

STATIC void BkfChSerDelAllType(BkfChSerMng *chMng)
{
    BkfChSerType *chType = VOS_NULL;
    void *itor = VOS_NULL;

    for (chType = BkfChSerGetFirstType(chMng, &itor); chType != VOS_NULL; chType = BkfChSerGetNextType(chMng, &itor)) {
        BkfChSerDelType(chMng, chType);
    }
    return;
}

STATIC BkfChSerType *BkfChSerFindType(BkfChSerMng *chMng, uint8_t typeId)
{
    BkfChSerType *chType = VOS_NULL;
#ifndef BKF_CUT_AVL_CACHE
    BOOL hit = VOS_FALSE;
    uint8_t hashIdx = BKF_CH_SER_GET_TYPE_HASH_IDX(typeId);
    chType = chMng->typeCache[hashIdx];
    hit = (chType != VOS_NULL) && (chType->vTbl.typeId == typeId);
    if (hit) {
        return chType;
    }
#endif
    chType = VOS_AVLL_FIND(chMng->typeSet, &typeId);
#ifndef BKF_CUT_AVL_CACHE
    if (chType != VOS_NULL) {
        chMng->typeCache[hashIdx] = chType;
    }
#endif
    return chType;
}

STATIC BkfChSerType *BkfChSerFindNextType(BkfChSerMng *chMng, uint8_t typeId)
{
    BkfChSerType *chType = VOS_NULL;

    chType = VOS_AVLL_FIND_NEXT(chMng->typeSet, &typeId);
    return chType;
}

STATIC BkfChSerType *BkfChSerGetFirstType(BkfChSerMng *chMng, void **itorOutOrNull)
{
    BkfChSerType *chType = VOS_NULL;

    chType = VOS_AVLL_FIRST(chMng->typeSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (chType != VOS_NULL) ? VOS_AVLL_NEXT(chMng->typeSet, chType->avlNode) : VOS_NULL;
    }
    return chType;
}

STATIC BkfChSerType *BkfChSerGetNextType(BkfChSerMng *chMng, void **itorInOut)
{
    BkfChSerType *chType = VOS_NULL;

    chType = (*itorInOut);
    *itorInOut = (chType != VOS_NULL) ? VOS_AVLL_NEXT(chMng->typeSet, chType->avlNode) : VOS_NULL;
    return chType;
}

#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
