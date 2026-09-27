/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_suber_conn_data.h"
#include "bkf_suber_conn.h"
#include "bkf_str.h"
#include "securec.h"
#include "bkf_suber_conn_fsm.h"

#ifdef __cplusplus
extern "C" {
#endif

#if BKF_BLOCK("私有函数定义")
void BkfSuberConnDataDelConnByUrlTypeMng(BkfSuberConnUrlTypeMng *urlTypeMng)
{
    BkfSuberConn *conn = VOS_NULL;
    BkfSuberConn *connNext = VOS_NULL;

    for (conn = VOS_AVLL_FIRST(urlTypeMng->connSet); conn != VOS_NULL; conn = connNext) {
        connNext = VOS_AVLL_NEXT(urlTypeMng->connSet, conn->avlNode);
        BkfSuberConnDataDel(conn);
    }
}

void BkfSuberConnDataUrlTypeMngDel(BkfSuberConnMng *connMng, BkfSuberConnUrlTypeMng *urlTypeMng)
{
    BkfSuberConnDataDelConnByUrlTypeMng(urlTypeMng);
    BKF_FREE(connMng->env->memMng, urlTypeMng);
}
#endif

#if BKF_BLOCK("公有函数定义")
BkfSuberConnMng *BkfSuberConnDataMngInit(BkfSuberConnMngInitArg *initArg)
{
    uint32_t len = sizeof(BkfSuberConnMng);
    BkfSuberEnv *env = initArg->env;
    BkfSuberConnMng *connMng = BKF_MALLOC(env->memMng, len);
    if (connMng == VOS_NULL) {
        return VOS_NULL;
    }

    (void)memset_s(connMng, len, 0, len);
    connMng->name = BkfStrNew(env->memMng, "%s_conn", env->name);
    if (connMng->name == VOS_NULL) {
        BKF_FREE(env->memMng, connMng);
        return VOS_NULL;
    }
    connMng->env = env;
    if (env->dbgOn) {
        connMng->log = env->log;
    }

    VOS_AVLL_INIT_TREE(connMng->connSetByConnId, (AVLL_COMPARE)BkfCmpPtrAddr, BKF_OFFSET(BkfSuberConn, connId),
        BKF_OFFSET(BkfSuberConn, connIdKeyNode));
    return connMng;
}

void BkfSuberConnDataMngUnInit(BkfSuberConnMng *connMng)
{
    uint32_t urlType;
    for (urlType = 0; urlType < BKF_GET_ARY_COUNT(connMng->urlTypeSet); urlType++) {
        if (connMng->urlTypeSet[urlType] != VOS_NULL) {
            BkfSuberConnDataUrlTypeMngDel(connMng, connMng->urlTypeSet[urlType]);
            connMng->urlTypeSet[urlType] = VOS_NULL;
        }
    }
    if (connMng->name != VOS_NULL) {
        BkfStrDel(connMng->name);
    }
    BkfSuberConnFsmUnInitFsmTmp(connMng);
    BKF_FREE(connMng->env->memMng, connMng);
}

BkfSuberConn *BkfSuberConnDataAdd(BkfSuberConnMng *connMng, BkfUrl *puberUrl, BkfUrl *localUrl)
{
    BkfSuberConnUrlTypeMng *urlTypeMng = BkfSuberConnDataCreateAndGetUrlTypeMng(connMng, puberUrl->type);
    if (urlTypeMng == VOS_NULL) {
        return VOS_NULL;
    }

    BkfSuberConn *conn = VOS_NULL;
    BkfSuberConn tmpConn;
    tmpConn.puberUrl = *puberUrl;
    tmpConn.localUrl = *localUrl;
    conn = VOS_AVLL_FIND(urlTypeMng->connSet, &tmpConn);
    if (conn != VOS_NULL) {
        return conn;
    }

    uint32_t len = sizeof(BkfSuberConn);
    conn = BKF_MALLOC(connMng->env->memMng, len);
    if (conn == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(conn, len, 0, len);

    uint32_t ret = BkfFsmInit(&conn->fsm, connMng->connFsmTmpl);
    if (ret != BKF_OK) {
        BKF_FREE(connMng->env->memMng, conn);
        return VOS_NULL;
    }

    conn->puberUrl = *puberUrl;
    conn->localUrl = *localUrl;
    conn->urlTypeMng = urlTypeMng;
    VOS_AVLL_INIT_NODE(conn->connIdKeyNode);
    BKF_DL_INIT(&conn->obInst);
    VOS_AVLL_INIT_NODE(conn->avlNode);
    BOOL isInsOk = VOS_AVLL_INSERT(urlTypeMng->connSet, conn->avlNode);
    if (!isInsOk) {
        BkfFsmUninit(&conn->fsm);
        BKF_FREE(connMng->env->memMng, conn);
        return VOS_NULL;
    }

    return conn;
}

void BkfSuberConnDataDel(BkfSuberConn *conn)
{
    BkfSuberConnUrlTypeMng *urlTypeMng = conn->urlTypeMng;
    BkfSuberConnMng *connMng = urlTypeMng->connMng;

    if (VOS_AVLL_IN_TREE(conn->avlNode)) {
        VOS_AVLL_DELETE(urlTypeMng->connSet, conn->avlNode);
    }

    if (VOS_AVLL_IN_TREE(conn->connIdKeyNode)) {
        VOS_AVLL_DELETE(connMng->connSetByConnId, conn->connIdKeyNode);
    }

    BkfFsmUninit(&conn->fsm);
    if (conn->rcvDataBuf != VOS_NULL) {
        BkfBufqUninit(conn->rcvDataBuf);
        conn->rcvDataBuf = VOS_NULL;
    }
    if (conn->lastSendLeftDataBuf != VOS_NULL) {
        BkfBufqUninit(conn->lastSendLeftDataBuf);
        conn->lastSendLeftDataBuf = VOS_NULL;
    }
    if (conn->reConnTmrId != VOS_NULL) {
        BkfTmrStop(connMng->env->tmrMng, conn->reConnTmrId);
    }
    BKF_FREE(connMng->env->memMng, conn);
}

BkfSuberConn *BkfSuberConnDataFind(BkfSuberConnMng *connMng, BkfUrl *puberUrl, BkfUrl *localUrl)
{
    BkfSuberConnUrlTypeMng *urlTypeMng = BkfSuberConnDataGetUrlTypeMng(connMng, puberUrl->type);
    if (urlTypeMng == VOS_NULL) {
        return VOS_NULL;
    }
    BkfSuberConn tmpConn;
    tmpConn.puberUrl = *puberUrl;
    tmpConn.localUrl = *localUrl;
    return VOS_AVLL_FIND(urlTypeMng->connSet, &tmpConn);
}

BkfSuberConn *BkfSuberConnDataFindByConnId(BkfSuberConnMng *connMng, BkfChCliConnId *connId)
{
    return VOS_AVLL_FIND(connMng->connSetByConnId, &connId);
}

uint32_t BkfSuberConnDataAddToConnIdSet(BkfSuberConn *conn, BkfChCliConnId *connId)
{
    BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;

    if (unlikely(VOS_AVLL_IN_TREE(conn->connIdKeyNode))) {
        BKF_ASSERT(0);
        VOS_AVLL_DELETE(connMng->connSetByConnId, conn->connIdKeyNode);
    }

    BkfSuberConn *connTmp = VOS_AVLL_FIND(connMng->connSetByConnId, &connId);
    if (unlikely(connTmp != VOS_NULL && connTmp != conn)) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    conn->connId = connId;
    BOOL isInsOk = VOS_AVLL_INSERT(connMng->connSetByConnId, conn->connIdKeyNode);
    if (!isInsOk) {
        return BKF_ERR;
    }
    return BKF_OK;
}

void BkfSuberConnDataDelFromConnIdSet(BkfSuberConn *conn)
{
    BkfSuberConnMng *connMng = conn->urlTypeMng->connMng;

    if (VOS_AVLL_IN_TREE(conn->connIdKeyNode)) {
        VOS_AVLL_DELETE(connMng->connSetByConnId, conn->connIdKeyNode);
    }
    conn->connId = VOS_NULL;
}

BkfSuberConn *BkfSuberConnDataFindNext(BkfSuberConnMng *connMng, BkfUrl *puberUrl, BkfUrl *localUrl)
{
    BkfSuberConnUrlTypeMng *urlTypeMng = BkfSuberConnDataGetUrlTypeMng(connMng, puberUrl->type);
    if (urlTypeMng == VOS_NULL) {
        return VOS_NULL;
    }
    BkfSuberConn tmpConn;
    tmpConn.puberUrl = *puberUrl;
    tmpConn.localUrl = *localUrl;
    return VOS_AVLL_FIND_NEXT(urlTypeMng->connSet, &tmpConn);
}

BkfSuberConn *BkfSuberConnDataGetFirstByUrlType(BkfSuberConnMng *connMng, uint8_t urlType, void **itorOutOrNull)
{
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_NULL;
    }

    BkfSuberConnUrlTypeMng *urlTypeMng = BkfSuberConnDataGetUrlTypeMng(connMng, urlType);
    if (urlTypeMng == VOS_NULL) {
        return VOS_NULL;
    }

    BkfSuberConn *conn = VOS_AVLL_FIRST(urlTypeMng->connSet);
    if (conn != VOS_NULL && itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_AVLL_NEXT(urlTypeMng->connSet, conn->avlNode);
    }
    return conn;
}
BkfSuberConn *BkfSuberConnDataGetNextByUrlType(BkfSuberConnMng *connMng, uint8_t urlType, void **itorInOut)
{
    BkfSuberConnUrlTypeMng *urlTypeMng = BkfSuberConnDataGetUrlTypeMng(connMng, urlType);
    if (urlTypeMng == VOS_NULL) {
        *itorInOut = VOS_NULL;
        return VOS_NULL;
    }

    BkfSuberConn *conn = *(BkfSuberConn **)itorInOut;
    if (conn == VOS_NULL) {
        *itorInOut = VOS_NULL;
    } else {
        *itorInOut = VOS_AVLL_NEXT(urlTypeMng->connSet, conn->avlNode);
    }
    return conn;
}

int32_t BkfSuberConnDataAvllCmp(void *input1, void *input2)
{
    BkfSuberConn *subConn1 = (BkfSuberConn *)input1;
    BkfSuberConn *subConn2 = (BkfSuberConn *)input2;

    int32_t ret = BkfUrlCmp(&subConn1->puberUrl, &subConn2->puberUrl);
    if (ret != 0) {
        return ret;
    }

    ret = BKF_CMP_X(subConn1->localUrl.type, subConn2->localUrl.type);
    if (ret != 0) {
        return ret;
    }
    if (subConn1->localUrl.type == BKF_URL_TYPE_INVALID) {
        return 0;
    }
    return BkfUrlCmp(&subConn1->localUrl, &subConn2->localUrl);
}

BkfSuberConnUrlTypeMng *BkfSuberConnDataCreateAndGetUrlTypeMng(BkfSuberConnMng *connMng, uint8_t urlType)
{
    if (urlType >= BKF_URL_TYPE_CNT || urlType == BKF_URL_TYPE_INVALID) {
        BKF_ASSERT(0);
        return VOS_NULL;
    }

    BkfSuberConnUrlTypeMng *urlTypeMng = connMng->urlTypeSet[urlType];
    if (urlTypeMng != VOS_NULL) {
        return urlTypeMng;
    }

    uint32_t len = sizeof(BkfSuberConnUrlTypeMng);
    urlTypeMng = BKF_MALLOC(connMng->env->memMng, len);
    if (urlTypeMng == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "malloc fail");
        return VOS_NULL;
    }

    (void)memset_s(urlTypeMng, len, 0, len);
    VOS_AVLL_INIT_TREE(urlTypeMng->connSet, (AVLL_COMPARE)BkfSuberConnDataAvllCmp, 0,
        BKF_OFFSET(BkfSuberConn, avlNode));
    urlTypeMng->urlType = urlType;
    urlTypeMng->connMng = connMng;
    connMng->urlTypeSet[urlType] = urlTypeMng;
    return urlTypeMng;
}

BkfSuberConnUrlTypeMng *BkfSuberConnDataGetUrlTypeMng(BkfSuberConnMng *connMng, uint8_t urlType)
{
    if (urlType >= BKF_URL_TYPE_CNT || urlType == BKF_URL_TYPE_INVALID) {
        BKF_ASSERT(0);
        return VOS_NULL;
    }

    return connMng->urlTypeSet[urlType];
}

uint32_t BkfSuberConnStartJob(BkfSuberConn *conn)
{
    if (conn->jobId == VOS_NULL) {
        conn->jobId = BkfJobCreate(conn->urlTypeMng->connMng->env->jobMng, conn->urlTypeMng->connMng->env->jobTypeId1,
            "jobIdSched", conn);
        if (conn->jobId == VOS_NULL) {
            BKF_ASSERT(0);
            return BKF_ERR;
        }
    }

    return BKF_OK;
}

void BkfSuberConnStopJob(BkfSuberConn *conn)
{
    if (conn->jobId == VOS_NULL) {
        return;
    }

    BkfJobDelete(conn->urlTypeMng->connMng->env->jobMng, conn->jobId);
    conn->jobId = VOS_NULL;
    return;
}
#endif

#ifdef __cplusplus
}
#endif