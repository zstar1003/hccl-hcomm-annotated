/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((co)->log)
#define BKF_MOD_NAME ((co)->name)
#include "net_co_out_data.h"
#include "net_co_main_data.h"
#include "bkf_str.h"
#include "securec.h"
#include "vos_assert.h"

#ifdef __cplusplus
extern "C" {
#endif
uint32_t NetCoOutDataNew(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    BkfMemMng *memMng = co->memMng;
    uint32_t len = sizeof(NetCoOut);
    NetCoOut *out = BKF_MALLOC(memMng, len);
    if (out == VOS_NULL) {
        return BKF_ERR;
    }
    (void)memset_s(out, len, 0, len);
    VOS_AVLL_INIT_TREE(out->tblTypeSet, (AVLL_COMPARE)Bkfuint16_tCmp,
        BKF_OFFSET(NetCoOutTblType, vTbl) + BKF_OFFSET(NetCoOutTblTypeVTbl, typeId),
        BKF_OFFSET(NetCoOutTblType, avlNode));

    co->out = out;
    return BKF_OK;
}

void NetCoOutDataDelete(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    NetCoOutDelAllTblType(co);
    BkfMemMng *memMng = co->memMng;
    BKF_FREE(memMng, co->out);
    co->out = VOS_NULL;
}

#define NET_CO_OUT_LOG_TBL_TYPE(co, tblType)                                                                           \
    do {                                                                                                               \
        if ((co)->dbgOn) {                                                                                             \
            uint8_t buf1_[BKF_1K / 2];                                                                                 \
            BKF_LOG_DEBUG(BKF_LOG_HND, "[%s]\n", NetCoOutGetTblTypeStr((tblType), buf1_, sizeof(buf1_)));              \
        }                                                                                                              \
    } while (0)
NetCoOutTblType *NetCoOutAddTblType(NetCo *co, NetCoOutTblTypeVTbl *vTbl)
{
    if (!NET_TBL_TYPE_IS_VALID(vTbl->typeId)) {
        VOS_ASSERT(0);
        return VOS_NULL;
    }
    if (!NET_TBL_KEY_LEN_IS_VALID(vTbl->keyLen)) {
        VOS_ASSERT(0);
        return VOS_NULL;
    }

    BkfMemMng *memMng = co->memMng;
    uint32_t len = sizeof(NetCoOutTblType);
    NetCoOutTblType *tblType = BKF_MALLOC(memMng, len);
    if (tblType == VOS_NULL) {
        return VOS_NULL;
    }

    (void)memset_s(tblType, len, 0, len);
    tblType->vTbl = *vTbl;
    tblType->vTbl.name = BkfStrNew(memMng, "%s", vTbl->name);
    if (tblType->vTbl.name == VOS_NULL) {
        BKF_FREE(memMng, tblType);
        return VOS_NULL;
    }
    VOS_AVLL_INIT_TREE(tblType->tblSet, (AVLL_COMPARE)tblType->vTbl.keyCmp, BKF_OFFSET(NetCoOutTbl, key),
        BKF_OFFSET(NetCoOutTbl, avlNode));
    VOS_AVLL_INIT_NODE(tblType->avlNode);
    if (!VOS_AVLL_INSERT(co->out->tblTypeSet, tblType->avlNode)) {
        BkfStrDel(tblType->vTbl.name);
        BKF_FREE(memMng, tblType);
        return VOS_NULL;
    }

    BKF_SIGN_SET(tblType->sign, NET_CO_OUT_TBL_TYPE_SIGN);
    NET_CO_OUT_LOG_TBL_TYPE(co, tblType);
    return tblType;
}

void NetCoOutDelTblType(NetCo *co, NetCoOutTblType *tblType)
{
    if (!BKF_SIGN_IS_VALID(tblType->sign, NET_CO_OUT_TBL_TYPE_SIGN)) {
        VOS_ASSERT(0);
        return;
    }
    NET_CO_OUT_LOG_TBL_TYPE(co, tblType);

    NetCoOutDelAllTbl(co, tblType);
    BkfMemMng *memMng = co->memMng;
    BkfStrDel(tblType->vTbl.name);
    VOS_AVLL_DELETE(co->out->tblTypeSet, tblType->avlNode);
    BKF_SIGN_CLR(tblType->sign);
    BKF_FREE(memMng, tblType);
}

void NetCoOutDelAllTblType(NetCo *co)
{
    NetCoOutTblType *tblType;
    void *itor;
    for (tblType = NetCoOutGetFirstTblType(co, &itor); tblType != VOS_NULL;
         tblType = NetCoOutGetNextTblType(co, &itor)) {
        NetCoOutDelTblType(co, tblType);
    }
}

NetCoOutTblType *NetCoOutFindTblType(NetCo *co, uint16_t tblTypeId)
{
    return VOS_AVLL_FIND(co->out->tblTypeSet, &tblTypeId);
}

NetCoOutTblType *NetCoOutFindNextTblType(NetCo *co, uint16_t tblTypeId)
{
    return VOS_AVLL_FIND_NEXT(co->out->tblTypeSet, &tblTypeId);
}

NetCoOutTblType *NetCoOutGetFirstTblType(NetCo *co, void **itorOutOrNull)
{
    NetCoOutTblType *tblType = VOS_AVLL_FIRST(co->out->tblTypeSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (tblType != VOS_NULL) ? VOS_AVLL_NEXT(co->out->tblTypeSet, tblType->avlNode) : VOS_NULL;
    }
    return tblType;
}

NetCoOutTblType *NetCoOutGetNextTblType(NetCo *co, void **itorInOut)
{
    NetCoOutTblType *tblType = (*itorInOut);
    *itorInOut = (tblType != VOS_NULL) ? VOS_AVLL_NEXT(co->out->tblTypeSet, tblType->avlNode) : VOS_NULL;
    return tblType;
}

#define NET_CO_OUT_LOG_TBL(co, tbl)                                                                                    \
    do {                                                                                                               \
        if ((co)->dbgOn) {                                                                                             \
            uint8_t buf1_[BKF_1K / 2];                                                                                 \
            BKF_LOG_DEBUG(BKF_LOG_HND, "[%s]\n", NetCoOutGetTblStr((tbl), buf1_, sizeof(buf1_)));                      \
        }                                                                                                              \
    } while (0)
NetCoOutTbl *NetCoOutAddTbl(NetCo *co, NetCoOutTblType *tblType, void *tblKey, void *tblValOrNull, uint16_t valLen)
{
    BkfMemMng *memMng = co->memMng;
    uint32_t len = sizeof(NetCoOutTbl) + tblType->vTbl.keyLen;
    NetCoOutTbl *tbl = BKF_MALLOC(memMng, len);
    if (tbl == VOS_NULL) {
        return VOS_NULL;
    }

    (void)memset_s(tbl, len, 0, len);
    tbl->valLen = valLen;
    tbl->tblType = tblType;
    (void)memcpy_s(tbl->key, tblType->vTbl.keyLen, tblKey, tblType->vTbl.keyLen);
    if (valLen > 0) {
        tbl->val = BKF_MALLOC(memMng, valLen);
        if (tbl->val == VOS_NULL) {
            BKF_FREE(memMng, tbl);
            return VOS_NULL;
        }
        (void)memcpy_s(tbl->val, valLen, tblValOrNull, valLen);
    }
    VOS_AVLL_INIT_NODE(tbl->avlNode);
    if (!VOS_AVLL_INSERT(tblType->tblSet, tbl->avlNode)) {
        if (tbl->val != VOS_NULL) {
            BKF_FREE(memMng, tbl->val);
        }
        BKF_FREE(memMng, tbl);
        return VOS_NULL;
    }

    BKF_SIGN_SET(tbl->sign, NET_CO_OUT_TBL_SIGN);
    tblType->tblCnt++;
    NET_CO_OUT_LOG_TBL(co, tbl);
    return tbl;
}

uint32_t NetCoOutUpdTbl(NetCo *co, NetCoOutTbl *tbl, void *tblValOrNullNew, uint16_t valLenNew)
{
    if (!BKF_SIGN_IS_VALID(tbl->sign, NET_CO_OUT_TBL_SIGN)) {
        VOS_ASSERT(0);
        return BKF_ERR;
    }

    BkfMemMng *memMng = co->memMng;
    if (valLenNew > tbl->valLen) {
        if (tbl->val != VOS_NULL) {
            BKF_FREE(memMng, tbl->val);
        }

        tbl->val = BKF_MALLOC(memMng, valLenNew);
        if (tbl->val == VOS_NULL) {
            tbl->valLen = 0; /* 强制赋0，实际不会跑到  */
            return BKF_ERR;
        }

        errno_t errNo = memcpy_s(tbl->val, valLenNew, tblValOrNullNew, valLenNew);
        if (errNo != EOK) {
            return BKF_ERR;
        }
        tbl->valLen = valLenNew;
    } else if (valLenNew > 0) {
        errno_t errNo = memcpy_s(tbl->val, tbl->valLen, tblValOrNullNew, valLenNew);
        if (errNo != EOK) {
            return BKF_ERR;
        }
        VOS_ASSERT(errNo == EOK);
        tbl->valLen = valLenNew;
    } else {
        if (tbl->val != VOS_NULL) {
            BKF_FREE(memMng, tbl->val);
            tbl->val = VOS_NULL;
        }
        tbl->valLen = 0;
    }

    NET_CO_OUT_LOG_TBL(co, tbl);
    return BKF_OK;
}

void NetCoOutDelOneTbl(NetCo *co, NetCoOutTbl *tbl)
{
    if (!BKF_SIGN_IS_VALID(tbl->sign, NET_CO_OUT_TBL_SIGN)) {
        VOS_ASSERT(0);
        return;
    }
    NET_CO_OUT_LOG_TBL(co, tbl);

    tbl->tblType->tblCnt--;
    BkfMemMng *memMng = co->memMng;
    if (tbl->val != VOS_NULL) {
        BKF_FREE(memMng, tbl->val);
    }
    VOS_AVLL_DELETE(tbl->tblType->tblSet, tbl->avlNode);
    BKF_SIGN_CLR(tbl->sign);
    BKF_FREE(memMng, tbl);
}

void NetCoOutDelAllTbl(NetCo *co, NetCoOutTblType *tblType)
{
    NetCoOutTbl *tbl;
    void *itor;
    for (tbl = NetCoOutGetFirstTbl(co, tblType, &itor); tbl != VOS_NULL; tbl = NetCoOutGetNextTbl(co, tblType, &itor)) {
        NetCoOutDelOneTbl(co, tbl);
    }
}

NetCoOutTbl *NetCoOutFindTbl(NetCo *co, NetCoOutTblType *tblType, void *tblKey)
{
    (void)co;
    return VOS_AVLL_FIND(tblType->tblSet, tblKey);
}

NetCoOutTbl *NetCoOutFindNextTbl(NetCo *co, NetCoOutTblType *tblType, void *tblKey)
{
    (void)co;
    return VOS_AVLL_FIND_NEXT(tblType->tblSet, tblKey);
}

NetCoOutTbl *NetCoOutGetFirstTbl(NetCo *co, NetCoOutTblType *tblType, void **itorOutOrNull)
{
    (void)co;
    NetCoOutTbl *tbl = VOS_AVLL_FIRST(tblType->tblSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (tbl != VOS_NULL) ? VOS_AVLL_NEXT(tblType->tblSet, tbl->avlNode) : VOS_NULL;
    }
    return tbl;
}

NetCoOutTbl *NetCoOutGetNextTbl(NetCo *co, NetCoOutTblType *tblType, void **itorInOut)
{
    (void)co;
    NetCoOutTbl *tbl = (*itorInOut);
    *itorInOut = (tbl != VOS_NULL) ? VOS_AVLL_NEXT(tblType->tblSet, tbl->avlNode) : VOS_NULL;
    return tbl;
}

#if BKF_BLOCK("display使用")
char *NetCoOutGetTblTypeStr(NetCoOutTblType *tblType, uint8_t *buf, int32_t bufLen)
{
    if ((buf == VOS_NULL) || (bufLen <= 0)) {
        return "__NetCoOutGetTblTypeStrArgNg";
    }

    NetCoOutTblTypeVTbl *vTbl = &tblType->vTbl;
    int32_t ret = snprintf_truncated_s((char *)buf, bufLen,
        "sign(%#x), typeId(%u)/name(%s), "
        "keyLen(%u)/keyCmp(%#x)/keyGetStr(%#x)/valGetStr(%#x), dataCnt(%d)",
        tblType->sign, vTbl->typeId, vTbl->name, vTbl->keyLen, BKF_MASK_ADDR(vTbl->keyCmp),
        BKF_MASK_ADDR(vTbl->keyGetStr), BKF_MASK_ADDR(vTbl->valGetStr), tblType->tblCnt);
    if (ret <= 0) {
        return "__NetCoOutGetTblTypeStrSnprintfNg";
    }

    return (char *)buf;
}

char *NetCoOutGetTblStr(NetCoOutTbl *tbl, uint8_t *buf, int32_t bufLen)
{
    if ((buf == VOS_NULL) || (bufLen <= 0)) {
        return "__NetCoOutGetTblStrArgNg";
    }

    NetCoOutTblTypeVTbl *vTbl = &tbl->tblType->vTbl;
    uint8_t buf1[BKF_1K];
    char *keyStr = vTbl->keyGetStr(&tbl->key[0], buf1, sizeof(buf1));
    uint8_t buf2[BKF_1K];
    char *valStr = vTbl->valGetStr(tbl->val, buf2, sizeof(buf2));
    int32_t ret = snprintf_truncated_s((char *)buf, bufLen, "sign(%#x), typeId(%u)/key(%s)/valLen(%u)/val(%s)",
        tbl->sign, vTbl->typeId, keyStr, tbl->valLen, valStr);
    if (ret <= 0) {
        return "__NetCoOutGetTblStrSnprintfNg";
    }

    return (char *)buf;
}

void NetCoOutGetTblStatCnt(NetCo *co, NetCoOutTblStatCnt *cnt)
{
    NetCoOutTblType *tblType;
    void *itor;
    for (tblType = NetCoOutGetFirstTblType(co, &itor); tblType != VOS_NULL;
         tblType = NetCoOutGetNextTblType(co, &itor)) {
        cnt->tblTypeCnt++;
        NetCoOutGetTblStatCntOfTblType(co, tblType, cnt);
    }
}

void NetCoOutGetTblStatCntOfTblType(NetCo *co, NetCoOutTblType *tblType, NetCoOutTblStatCnt *cnt)
{
    NetCoOutTbl *tbl;
    void *itor;
    for (tbl = NetCoOutGetFirstTbl(co, tblType, &itor); tbl != VOS_NULL; tbl = NetCoOutGetNextTbl(co, tblType, &itor)) {
        cnt->tblCnt++;
    }
}

char *NetCoOutGetTblStatCntStr(NetCoOutTblStatCnt *cnt, uint8_t *buf, int32_t bufLen)
{
    if ((buf == VOS_NULL) || (bufLen <= 0)) {
        return "__NetCoOutGetTblStatCntStrArgNg";
    }

    int32_t ret = snprintf_truncated_s((char *)buf, bufLen, "tblTypeCnt(%d)/tblCnt(%d)", cnt->tblTypeCnt, cnt->tblCnt);
    if (ret <= 0) {
        return "__NetCoOutGetTblStatCntStrSnprintfNg";
    }

    return (char *)buf;
}

STATIC void NetCoOutInitGetTblCxt(NetCo *co, NetCoOutGetTblCtx *ctx)
{
    (void)co;
    (void)memset_s(ctx, sizeof(NetCoOutGetTblCtx), 0, sizeof(NetCoOutGetTblCtx));
    BKF_SIGN_SET(ctx->sign, NET_CO_OUT_GET_TBL_CTX_SIGN);
}
STATIC NetCoOutTblType *NetCoOutUpdGetTblCtx(NetCo *co, NetCoOutGetTblCtx *ctx, NetCoOutTblType *tblTypeOrNull,
    BOOL tblTypeIsNew, NetCoOutTbl *tblOrNull)
{
    (void)co;
    /* 设置一些默认值，用于中途返回 */
    ctx->hasTblType = VOS_FALSE;
    ctx->tblTypeIsNew = VOS_FALSE;
    ctx->tbl = VOS_NULL;

    if (tblTypeOrNull == VOS_NULL) {
        return VOS_NULL;
    }
    ctx->hasTblType = VOS_TRUE;
    ctx->tblTypeIsNew = tblTypeIsNew;
    ctx->tblTypeId = tblTypeOrNull->vTbl.typeId;
    if (ctx->tblTypeIsNew) {
        ctx->tblTypeCnt++;
    }

    if (tblOrNull == VOS_NULL) {
        return tblTypeOrNull;
    }
    errno_t err = memcpy_s(ctx->tblKey, sizeof(ctx->tblKey), tblOrNull->key, tblTypeOrNull->vTbl.keyLen);
    if (err != EOK) {
        VOS_ASSERT(0);
        return tblTypeOrNull;
    }
    ctx->tbl = tblOrNull;
    ctx->tblCntTotal++;
    ctx->tblCntCurTblType = ctx->tblTypeIsNew ? 1 : (ctx->tblCntCurTblType + 1);
    return tblTypeOrNull;
}
NetCoOutTblType *NetCoOutGetFirstTblByCtx(NetCo *co, NetCoOutGetTblCtx *ctx)
{
    NetCoOutInitGetTblCxt(co, ctx);

    NetCoOutTblType *tblType = NetCoOutGetFirstTblType(co, VOS_NULL);
    NetCoOutTbl *tbl = (tblType != VOS_NULL) ? NetCoOutGetFirstTbl(co, tblType, VOS_NULL) : VOS_NULL;
    return NetCoOutUpdGetTblCtx(co, ctx, tblType, VOS_TRUE, tbl);
}

NetCoOutTblType *NetCoOutGetNextTblByCtxNextTblType(NetCo *co, NetCoOutGetTblCtx *ctx)
{
    NetCoOutTblType *tblType = NetCoOutFindNextTblType(co, ctx->tblTypeId);
    NetCoOutTbl *tbl = (tblType != VOS_NULL) ? NetCoOutGetFirstTbl(co, tblType, VOS_NULL) : VOS_NULL;
    return NetCoOutUpdGetTblCtx(co, ctx, tblType, VOS_TRUE, tbl);
}
NetCoOutTblType *NetCoOutGetNextTblByCtx(NetCo *co, NetCoOutGetTblCtx *ctx)
{
    if (!BKF_SIGN_IS_VALID(ctx->sign, NET_CO_OUT_GET_TBL_CTX_SIGN)) {
        VOS_ASSERT(0);
        return VOS_NULL;
    }
    if (!ctx->hasTblType) {
        return VOS_NULL;
    }

    NetCoOutTblType *tblType = NetCoOutFindTblType(co, ctx->tblTypeId);
    if (tblType == VOS_NULL) {
        return NetCoOutGetNextTblByCtxNextTblType(co, ctx);
    }

    if (ctx->tbl == VOS_NULL) {
        return NetCoOutGetNextTblByCtxNextTblType(co, ctx);
    }
    NetCoOutTbl *tbl = NetCoOutFindNextTbl(co, tblType, ctx->tblKey);
    if (tbl == VOS_NULL) {
        return NetCoOutGetNextTblByCtxNextTblType(co, ctx);
    }
    return NetCoOutUpdGetTblCtx(co, ctx, tblType, VOS_FALSE, tbl);
}

#endif

#ifdef __cplusplus
}
#endif
