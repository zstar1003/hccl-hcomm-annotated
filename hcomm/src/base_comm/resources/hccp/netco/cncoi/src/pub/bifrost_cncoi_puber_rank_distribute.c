/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bifrost_cncoi_puber_rank_distribute.h"
#include "bifrost_cncoi_svc.h"
#include "bifrost_cncoi_table.h"
#include "bifrost_cncoi_puber_pri.h"
#include "vos_assert.h"
#include "bifrost_cncoi_reader.h"
#if __cplusplus
extern "C" {
#endif

STATIC int32_t BifrostCncoiPuberRankDistributeOnUpdateCode(BifrostCncoiPuber *bifrostCncoiPuber,
    BifrostCncoiSliceKeyT *sliceKey, BifrostCncoiRankDistributeKeyT *tupleKey, void *tupleVal, void *codeBuf,
    int32_t bufLen)
{
    BifrostCncoiPuberRankDistributeVTbl *appVTbl = VOS_NULL;
    int32_t ret = -1;
    void *builder;

    appVTbl = BkfPuberGetTableTypeUserData(bifrostCncoiPuber->puber, BIFROST_CNCOI_TABLE_TYPE_RANK_DISTRIBUTE);
    if (appVTbl == VOS_NULL) {
        VOS_ASSERT(0);
        goto error;
    }

    if (appVTbl->onFillUpdateData == VOS_NULL) {
        VOS_ASSERT(0);
        goto error;
    }

    builder = bifrostCncoiPuber->argInit.simpoBuilder;
    if (builder == VOS_NULL) {
        VOS_ASSERT(0);
        goto error;
    }
    ret = appVTbl->onFillUpdateData(appVTbl->cookie, builder, sliceKey, tupleKey, tupleVal, codeBuf, bufLen);
    if (ret == 0 || ret > bufLen) {
        ret = -1;
        VOS_ASSERT(0);
        goto error;
    }
    if (ret < 0) {
        ret = BKF_PUBER_CODE_BUF_NOT_ENOUGH;
    }

error:

    return ret;
}

STATIC int32_t BifrostCncoiPuberRankDistributeOnDeleteCode(BifrostCncoiPuber *bifrostCncoiPuber,
    BifrostCncoiSliceKeyT *sliceKey, BifrostCncoiRankDistributeKeyT *tupleKey, void *codeBuf, int32_t bufLen)
{
    (void)sliceKey;
    SimpoBuilderT *builder;
    int32_t ret = -1;

    builder = bifrostCncoiPuber->argInit.simpoBuilder;
    if (builder == VOS_NULL) {
        VOS_ASSERT(0);
        goto error;
    }

    ret = BifrostCncoiRankDistributeDeleteStartAsRoot(builder, codeBuf, bufLen);
    if (ret < 0) {
        return BKF_PUBER_CODE_BUF_NOT_ENOUGH;
    }
    BifrostCncoiRankDistributeDeleteKeyCreate(builder, tupleKey);
    ret = BifrostCncoiRankDistributeDeleteEndAsRoot(builder);
    if (ret == 0 || ret > bufLen) {
        ret = -1;
        VOS_ASSERT(0);
        goto error;
    }
    if (ret < 0) {
        return BKF_PUBER_CODE_BUF_NOT_ENOUGH;
    }

error:

    return ret;
}

STATIC void BifrostCncoiPuberRankDistributeOnSub(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiSliceKeyT *sliceKey,
    void *data, int32_t len)
{
    (void)data;
    (void)len;
    BifrostCncoiPuberRankDistributeVTbl *appVTbl = VOS_NULL;

    appVTbl = BkfPuberGetTableTypeUserData(bifrostCncoiPuber->puber, BIFROST_CNCOI_TABLE_TYPE_RANK_DISTRIBUTE);
    if (appVTbl == VOS_NULL) {
        VOS_ASSERT(0);
        goto error;
    }

    if (appVTbl->onSub == VOS_NULL) {
        goto error;
    }

    appVTbl->onSub(appVTbl->cookie, sliceKey);
error:
    return;
}

STATIC void BifrostCncoiPuberRankDistributeOnUnsub(BifrostCncoiPuber *bifrostCncoiPuber,
    BifrostCncoiSliceKeyT *sliceKey)
{
    BifrostCncoiPuberRankDistributeVTbl *appVTbl = VOS_NULL;

    appVTbl = BkfPuberGetTableTypeUserData(bifrostCncoiPuber->puber, BIFROST_CNCOI_TABLE_TYPE_RANK_DISTRIBUTE);
    if (appVTbl == VOS_NULL) {
        VOS_ASSERT(0);
        goto error;
    }

    if (appVTbl->onUnSub == VOS_NULL) {
        goto error;
    }

    appVTbl->onUnSub(appVTbl->cookie, sliceKey);
error:
    return;
}

uint32_t BifrostCncoiPuberRankDistributeReg(BifrostCncoiPuber *bifrostCncoiPuber,
    BifrostCncoiPuberRankDistributeVTbl *appVTbl)
{
    BkfDcTableTypeVTbl dcVTbl = {0};
    BkfPuberTableTypeVTbl puberVTbl = {0};
    uint32_t ret;

    if (bifrostCncoiPuber == VOS_NULL || appVTbl == VOS_NULL) {
        return BKF_ERR;
    }

    dcVTbl.name = BIFROST_CNCOI_SVC_NAME "_rank_distribute";
    dcVTbl.tableTypeId = BIFROST_CNCOI_TABLE_TYPE_RANK_DISTRIBUTE;
    dcVTbl.cookie = bifrostCncoiPuber;
    dcVTbl.tupleCntMax = appVTbl->tupleCntMax;
    dcVTbl.tupleKeyLen = sizeof(BifrostCncoiRankDistributeKeyT);
    dcVTbl.tupleValLen = appVTbl->tupleValLen;
    dcVTbl.tupleKeyCmp = (F_BKF_CMP)BifrostCncoiRankDistributeKeyCmp;
    dcVTbl.tupleKeyGetStrOrNull = (F_BKF_GET_STR)BifrostCncoiRankDistributeKeyGetStr;
    dcVTbl.tupleValGetStrOrNull = (F_BKF_GET_STR)appVTbl->onGetTupleValStr;
    ret = BkfDcRegTableType(bifrostCncoiPuber->argInit.dc, &dcVTbl);
    if (ret != BKF_OK) {
        return ret;
    }

    puberVTbl.tableTypeId = BIFROST_CNCOI_TABLE_TYPE_RANK_DISTRIBUTE;
    puberVTbl.cookie = bifrostCncoiPuber;
    puberVTbl.tupleUpdateCode = (F_BKF_PUBER_TUPLE_UPDATE_CODE)BifrostCncoiPuberRankDistributeOnUpdateCode;
    puberVTbl.tupleDeleteCode = (F_BKF_PUBER_TUPLE_DELETE_CODE)BifrostCncoiPuberRankDistributeOnDeleteCode;
    puberVTbl.tableOnSub = (F_BKF_PUBER_TABLE_ONSUB)BifrostCncoiPuberRankDistributeOnSub;
    puberVTbl.tableOnUnsub = (F_BKF_PUBER_TABLE_ONUNSUB)BifrostCncoiPuberRankDistributeOnUnsub;
    ret = BkfPuberAttachTableTypeEx(bifrostCncoiPuber->puber, &puberVTbl, appVTbl,
        sizeof(BifrostCncoiPuberRankDistributeVTbl));
    if (ret != BKF_OK) {
        return ret;
    }

    return BKF_OK;
}

uint32_t BifrostCncoiPuberRankDistributeNotifyTableComplete(BifrostCncoiPuber *bifrostCncoiPuber)
{
    if (bifrostCncoiPuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfDcNotifyTableTypeComplete(bifrostCncoiPuber->argInit.dc, BIFROST_CNCOI_TABLE_TYPE_RANK_DISTRIBUTE);
}

uint32_t BifrostCncoiPuberRankDistributeCreateTable(BifrostCncoiPuber *bifrostCncoiPuber,
    BifrostCncoiSliceKeyT *sliceKey)
{
    if (bifrostCncoiPuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfDcCreateTable(bifrostCncoiPuber->argInit.dc, sliceKey, BIFROST_CNCOI_TABLE_TYPE_RANK_DISTRIBUTE);
}

void BifrostCncoiPuberRankDistributeDeleteTable(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiSliceKeyT *sliceKey)
{
    if (bifrostCncoiPuber == VOS_NULL) {
        return;
    }

    BkfDcDeleteTable(bifrostCncoiPuber->argInit.dc, sliceKey, BIFROST_CNCOI_TABLE_TYPE_RANK_DISTRIBUTE);
}

void BifrostCncoiPuberRankDistributeReleaseTable(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiSliceKeyT *sliceKey)
{
    if (bifrostCncoiPuber == VOS_NULL) {
        return;
    }

    BkfDcReleaseTable(bifrostCncoiPuber->argInit.dc, sliceKey, BIFROST_CNCOI_TABLE_TYPE_RANK_DISTRIBUTE);
}

uint32_t BifrostCncoiPuberRankDistributeUpdate(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiRankDistributeKeyT *tupleKey, void *val)
{
    if (bifrostCncoiPuber == VOS_NULL) {
        return BKF_ERR;
    }

    return BkfDcUpdateTuple(bifrostCncoiPuber->argInit.dc, sliceKey, BIFROST_CNCOI_TABLE_TYPE_RANK_DISTRIBUTE, tupleKey,
        val, VOS_NULL);
}

void BifrostCncoiPuberRankDistributeDelete(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiRankDistributeKeyT *tupleKey)
{
    if (bifrostCncoiPuber == VOS_NULL) {
        return;
    }

    BkfDcDeleteTuple(bifrostCncoiPuber->argInit.dc, sliceKey, BIFROST_CNCOI_TABLE_TYPE_RANK_DISTRIBUTE, tupleKey);
}

#if __cplusplus
}
#endif

// auto-generated information:MS4wOzEwOzIyNTsyMDI0MTAxNDtiaWZyb3N0X2NuY29pX3B1YmVyX3JhbmtfZGlzdHJpYnV0ZS5j
// auto-generated check_sum:94296802b5e576596c761b55d6c9d3188e46e6c11fff28939192568ab38c757e