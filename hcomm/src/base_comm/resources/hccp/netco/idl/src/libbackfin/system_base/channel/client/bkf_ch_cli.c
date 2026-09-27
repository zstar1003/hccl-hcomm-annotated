/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((chMng)->argInit.dbgOn ? (chMng)->argInit.log : VOS_NULL)
#define BKF_MOD_NAME ((chMng)->argInit.name)

#include "bkf_ch_cli.h"
#include "bkf_ch_cli_pri.h"
#include "bkf_ch_cli_disp.h"
#include "bkf_ch_cli_conn_id_base.h"
#include "bkf_bas_type_mthd.h"
#include "bkf_str.h"
#include "v_avll.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#if BKF_BLOCK("公有函数定义")
/* func */
STATIC uint32_t BkfChCliInitChkArg(BkfChCliMngInitArg *arg)
{
    BOOL argIsInvalid = VOS_FALSE;

    argIsInvalid = (arg == VOS_NULL) || (arg->name == VOS_NULL) || (arg->memMng == VOS_NULL) ||
                   (arg->disp == VOS_NULL) || (arg->pfm == VOS_NULL) || (arg->xMap == VOS_NULL) ||
                   (arg->tmrMng == VOS_NULL) || (arg->jobMng == VOS_NULL);
    if (argIsInvalid) {
        return BKF_ERR;
    }

    return BKF_OK;
}
BkfChCliMng *BkfChCliInit(BkfChCliMngInitArg *arg)
{
    BkfChCliMng *chMng = VOS_NULL;
    uint32_t len;
    uint32_t ret;

    ret = BkfChCliInitChkArg(arg);
    if (ret != BKF_OK) {
        goto error;
    }

    len = sizeof(BkfChCliMng);
    chMng = BKF_MALLOC(arg->memMng, len);
    if (chMng == VOS_NULL) {
        goto error;
    }
    (void)memset_s(chMng, len, 0, len);
    chMng->argInit = *arg;
    VOS_AVLL_INIT_TREE(chMng->typeSet, (AVLL_COMPARE)BkfUCharCmp,
        BKF_OFFSET(BkfChCliType, vTbl) + BKF_OFFSET(BkfChCliTypeVTbl, typeId), BKF_OFFSET(BkfChCliType, avlNode));
    chMng->name = BkfStrNew(arg->memMng, "%s_ch_cli", arg->name);
    if (chMng->name == VOS_NULL) {
        BKF_ASSERT(0);
        goto error;
    }
    chMng->argInit.name = chMng->name;
    BkfChCliDispInit(chMng);

    BKF_LOG_INFO(BKF_LOG_HND, "chMng(%#x), init ok\n", BKF_MASK_ADDR(chMng));
    return chMng;

error:

    BkfChCliUninit(chMng);
    return VOS_NULL;
}

void BkfChCliUninit(BkfChCliMng *chMng)
{
    if (chMng == VOS_NULL) {
        return;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "chMng(%#x), 2uninit\n", BKF_MASK_ADDR(chMng));

    BkfChCliDispUnInit(chMng);
    BkfChCliDelAllType(chMng);
    BkfStrDel(chMng->name);
    BKF_FREE(chMng->argInit.memMng, chMng);
    return;
}

uint32_t BkfChCliRegType(BkfChCliMng *chMng, BkfChCliTypeVTbl *vTbl)
{
    BOOL argIsInvalid = VOS_FALSE;
    BkfChCliType *chType = VOS_NULL;

    argIsInvalid = (chMng == VOS_NULL) || (vTbl == VOS_NULL) || (vTbl->name == VOS_NULL) || (vTbl->init == VOS_NULL) ||
                   (vTbl->uninit == VOS_NULL);
    if (argIsInvalid) {
        return BKF_ERR;
    }

    BKF_LOG_DEBUG(BKF_LOG_HND, "chMng(%#x)/type(%u, %s), vTbl(%#x), enable %u\n", BKF_MASK_ADDR(chMng), vTbl->typeId,
        vTbl->name, BKF_MASK_ADDR(vTbl), chMng->hasEnable);

    chType = BkfChCliAddType(chMng, vTbl);
    if (chType == VOS_NULL) {
        return BKF_ERR;
    }

    return BKF_OK;
}

uint32_t BkfChCliUnRegType(BkfChCliMng *chMng, uint8_t chType)
{
    BkfChCliType *chCliType = VOS_NULL;
    if (chMng == VOS_NULL) {
        return BKF_OK;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "chMng(%#x)/type(%u) unreg.\n", BKF_MASK_ADDR(chMng), chType);

    chCliType = BkfChCliFindType(chMng, chType);
    if (chCliType == VOS_NULL) {
        return BKF_OK;
    }
    BkfChCliDelType(chMng, chCliType);
    return BKF_OK;
}

uint32_t BkfChCliEnable(BkfChCliMng *chMng, BkfChCliEnableArg *arg)
{
    BkfChCliType *chType = VOS_NULL;
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
    for (chType = BkfChCliGetFirstType(chMng, &itor); chType != VOS_NULL; chType = BkfChCliGetNextType(chMng, &itor)) {
        if (chType->vTbl.enable != VOS_NULL) {
            ret |= chType->vTbl.enable(chType->ch, arg);
        }
    }
    return ret;
}

uint32_t BkfChCliSetSelfCid(BkfChCliMng *chMng, uint32_t selfCid)
{
    BkfChCliType *chType = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t ret;

    if (chMng == VOS_NULL) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "chMng(%#x), selfCid(%#x)\n", BKF_MASK_ADDR(chMng), selfCid);

    ret = BKF_OK;
    for (chType = BkfChCliGetFirstType(chMng, &itor); chType != VOS_NULL; chType = BkfChCliGetNextType(chMng, &itor)) {
        if (chType->vTbl.setSelfCid != VOS_NULL) {
            ret |= chType->vTbl.setSelfCid(chType->ch, selfCid);
        }
    }
    return ret;
}

uint32_t BkfChCliSetCera(BkfChCliMng *chMng, BkfCera *cera)
{
    if (chMng == VOS_NULL || cera == VOS_NULL) {
        return BKF_ERR;
    }
    BkfChCliType *chType = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t ret = BKF_OK;

    for (chType = BkfChCliGetFirstType(chMng, &itor); chType != VOS_NULL; chType = BkfChCliGetNextType(chMng, &itor)) {
        if (chType->vTbl.setCera != VOS_NULL) {
            ret |= chType->vTbl.setCera(chType->ch, cera);
        }
    }
    return ret;
}

uint32_t BkfChCliUnSetCera(BkfChCliMng *chMng)
{
    if (chMng == VOS_NULL) {
        return BKF_ERR;
    }
    BkfChCliType *chType = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t ret = BKF_OK;

    for (chType = BkfChCliGetFirstType(chMng, &itor); chType != VOS_NULL; chType = BkfChCliGetNextType(chMng, &itor)) {
        if (chType->vTbl.unSetCera != VOS_NULL) {
            ret |= chType->vTbl.unSetCera(chType->ch);
        }
    }
    return ret;
}

uint32_t BkfChCliSetSelfUrl(BkfChCliMng *chMng, BkfUrl *selfUrl)
{
    BkfChCliType *chType = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t ret;
    uint8_t buf[BKF_1K / 8];

    if (chMng == VOS_NULL) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "chMng(%#x), selfUrl(%s)\n", BKF_MASK_ADDR(chMng),
        BkfUrlGetStr(selfUrl, buf, sizeof(buf)));

    ret = BKF_OK;
    for (chType = BkfChCliGetFirstType(chMng, &itor); chType != VOS_NULL; chType = BkfChCliGetNextType(chMng, &itor)) {
        if (chType->vTbl.setSelfUrl != VOS_NULL) {
            ret |= chType->vTbl.setSelfUrl(chType->ch, selfUrl);
        }
    }
    return ret;
}

STATIC BkfChCliType *BkfChCliGetTypeByConnId(BkfChCliMng *chMng, BkfChCliConnId *connId)
{
    BkfChCliConnIdBase *base = VOS_NULL;

    base = BkfChCliConnIdGetBase(connId);
    if (base == VOS_NULL) {
        return VOS_NULL;
    }

    return BkfChCliFindType(chMng, base->urlType);
}

BkfChCliConnId *BkfChCliConn(BkfChCliMng *chMng, BkfUrl *urlServer)
{
    BkfChCliType *chType = VOS_NULL;
    void *itor = VOS_NULL;
    BkfChCliConnId *conn = VOS_NULL;

    if ((chMng == VOS_NULL) || (urlServer == VOS_NULL)) {
        return VOS_NULL;
    }

    for (chType = BkfChCliGetFirstType(chMng, &itor); chType != VOS_NULL; chType = BkfChCliGetNextType(chMng, &itor)) {
        if (chType->vTbl.conn != VOS_NULL) {
            conn = chType->vTbl.conn(chType->ch, urlServer);
            return conn;
        }
    }
    return VOS_NULL;
}

BkfChCliConnId *BkfChCliConnEx(BkfChCliMng *chMng, BkfUrl *urlServer, BkfUrl *urlLocal)
{
    BkfChCliType *chType = VOS_NULL;
    void *itor = VOS_NULL;
    BkfChCliConnId *conn = VOS_NULL;

    if ((chMng == VOS_NULL) || (urlServer == VOS_NULL || urlLocal == VOS_NULL)) {
        return VOS_NULL;
    }

    for (chType = BkfChCliGetFirstType(chMng, &itor); chType != VOS_NULL; chType = BkfChCliGetNextType(chMng, &itor)) {
        if (chType->vTbl.connEx != VOS_NULL) {
            conn = chType->vTbl.connEx(chType->ch, urlServer, urlLocal);
            return conn;
        }
    }
    return VOS_NULL;
}

void BkfChCliDisConn(BkfChCliMng *chMng, BkfChCliConnId *connId)
{
    BkfChCliType *chType = VOS_NULL;

    if ((chMng == VOS_NULL) || (connId == VOS_NULL)) {
        return;
    }

    chType = BkfChCliGetTypeByConnId(chMng, connId);
    if ((chType != VOS_NULL) && (chType->vTbl.disConn != VOS_NULL)) {
        chType->vTbl.disConn(chType->ch, connId);
    }

    return;
}

void *BkfChCliMallocDataBuf(BkfChCliMng *chMng, BkfChCliConnId *connId, int32_t dataBufLen)
{
    BkfChCliType *chType = VOS_NULL;

    if ((chMng == VOS_NULL) || (connId == VOS_NULL) || (dataBufLen <= 0)) {
        return VOS_NULL;
    }

    chType = BkfChCliGetTypeByConnId(chMng, connId);
    if ((chType == VOS_NULL) || (chType->vTbl.mallocDataBuf == VOS_NULL)) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "chMng(%#x) type(%#x) null.\n", BKF_MASK_ADDR(chMng), BKF_MASK_ADDR(chType));
        return VOS_NULL;
    }

    return chType->vTbl.mallocDataBuf(connId, dataBufLen);
}

void BkfChCliFreeDataBuf(BkfChCliMng *chMng, BkfChCliConnId *connId, void *dataBuf)
{
    BkfChCliType *chType = VOS_NULL;

    if ((chMng == VOS_NULL) || (connId == VOS_NULL) || (dataBuf == VOS_NULL)) {
        return;
    }

    chType = BkfChCliGetTypeByConnId(chMng, connId);
    if ((chType == VOS_NULL) || (chType->vTbl.freeDataBuf == VOS_NULL)) {
        return;
    }

    chType->vTbl.freeDataBuf(connId, dataBuf);
    return;
}

int32_t BkfChCliRead(BkfChCliMng *chMng, BkfChCliConnId *connId, void *dataBuf, int32_t bufLen)
{
    BkfChCliType *chType = VOS_NULL;

    if ((chMng == VOS_NULL) || (connId == VOS_NULL) || (dataBuf == VOS_NULL) || (bufLen < 0)) {
        return -1;
    }
    if (bufLen == 0) {
        return 0;
    }

    chType = BkfChCliGetTypeByConnId(chMng, connId);
    if ((chType == VOS_NULL) || (chType->vTbl.read == VOS_NULL)) {
        return -1;
    }

    return chType->vTbl.read(connId, dataBuf, bufLen);
}

int32_t BkfChCliWrite(BkfChCliMng *chMng, BkfChCliConnId *connId, void *dataBuf, int32_t dataLen)
{
    BkfChCliType *chType = VOS_NULL;

    if ((chMng == VOS_NULL) || (connId == VOS_NULL) || (dataBuf == VOS_NULL) || (dataLen < 0)) {
        return -1;
    }
    if (dataLen == 0) {
        return 0;
    }

    chType = BkfChCliGetTypeByConnId(chMng, connId);
    if ((chType == VOS_NULL) || (chType->vTbl.write == VOS_NULL)) {
        return -1;
    }

    return chType->vTbl.write(connId, dataBuf, dataLen);
}

#endif

#if BKF_BLOCK("私有函数定义")
/* data op */
STATIC BkfChCliType *BkfChCliAddType(BkfChCliMng *chMng, BkfChCliTypeVTbl *vTbl)
{
    BkfChCliType *chType = VOS_NULL;
    uint32_t len;
    BkfChCliInitArg arg = {.base = &chMng->argInit, .name = vTbl->name};
    BOOL insOk = VOS_FALSE;

    len = sizeof(BkfChCliType);
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
    uint8_t hashIdx = BKF_CH_CLI_GET_TYPE_HASH_IDX(vTbl->typeId);
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

STATIC void BkfChCliDelType(BkfChCliMng *chMng, BkfChCliType *chType)
{
#ifndef BKF_CUT_AVL_CACHE
    uint8_t hashIdx = BKF_CH_CLI_GET_TYPE_HASH_IDX(chType->vTbl.typeId);
    if (chMng->typeCache[hashIdx] == chType) {
        chMng->typeCache[hashIdx] = VOS_NULL;
    }
#endif
    chType->vTbl.uninit(chType->ch);
    VOS_AVLL_DELETE(chMng->typeSet, chType->avlNode);
    BKF_FREE(chMng->argInit.memMng, chType);
    return;
}

STATIC void BkfChCliDelAllType(BkfChCliMng *chMng)
{
    BkfChCliType *chType = VOS_NULL;
    void *itor = VOS_NULL;

    for (chType = BkfChCliGetFirstType(chMng, &itor); chType != VOS_NULL; chType = BkfChCliGetNextType(chMng, &itor)) {
        BkfChCliDelType(chMng, chType);
    }
    return;
}

STATIC BkfChCliType *BkfChCliFindType(BkfChCliMng *chMng, uint8_t typeId)
{
    BkfChCliType *chType = VOS_NULL;
#ifndef BKF_CUT_AVL_CACHE
    BOOL hit = VOS_FALSE;
    uint8_t hashIdx = BKF_CH_CLI_GET_TYPE_HASH_IDX(typeId);

    chType = chMng->typeCache[hashIdx];
    hit = (chType != VOS_NULL) && (chType->vTbl.typeId == typeId);
    if (hit) {
        return chType;
    }
#endif

    chType = VOS_AVLL_FIND(chMng->typeSet, &typeId);
    if (chType != VOS_NULL) {
#ifndef BKF_CUT_AVL_CACHE
        chMng->typeCache[hashIdx] = chType;
#endif
    }
    return chType;
}

STATIC BkfChCliType *BkfChCliFindNextType(BkfChCliMng *chMng, uint8_t typeId)
{
    BkfChCliType *chType = VOS_NULL;

    chType = VOS_AVLL_FIND_NEXT(chMng->typeSet, &typeId);
    return chType;
}

STATIC BkfChCliType *BkfChCliGetFirstType(BkfChCliMng *chMng, void **itorOutOrNull)
{
    BkfChCliType *chType = VOS_NULL;

    chType = VOS_AVLL_FIRST(chMng->typeSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (chType != VOS_NULL) ? VOS_AVLL_NEXT(chMng->typeSet, chType->avlNode) : VOS_NULL;
    }
    return chType;
}

STATIC BkfChCliType *BkfChCliGetNextType(BkfChCliMng *chMng, void **itorInOut)
{
    BkfChCliType *chType = VOS_NULL;

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
