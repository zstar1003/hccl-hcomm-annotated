/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_suber_conn.h"
#include "bkf_suber_conn_data.h"
#include "bkf_suber_conn_pri.h"
#include "bkf_suber_conn_disp.h"
#include "bkf_suber_conn_fsm.h"

#ifdef __cplusplus
extern "C" {
#endif

#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
typedef struct {
    BkfSuberConnMng *connMng;
    BkfUrl *puberUrl;
    BkfUrl *localUrl;
} BkfSuberConnTrigSessPara;
#pragma pack()
#endif

#if BKF_BLOCK("私有函数定义")
void BkfSuberConnProcDelConnByUrlType(BkfSuberConnMng *connMng, uint8_t urlType)
{
    BkfSuberConn *conn = VOS_NULL;
    void *itor = VOS_NULL;

    for (conn = BkfSuberConnDataGetFirstByUrlType(connMng, urlType, &itor); conn != VOS_NULL;
         conn = BkfSuberConnDataGetNextByUrlType(connMng, urlType, &itor)) {
        BkfSuberConnProcDelConn(conn);
    }

    return;
}

void BkfSuberConnProcDelAllConn(BkfSuberConnMng *connMng)
{
    uint32_t urlType;
    for (urlType = BKF_URL_TYPE_INVALID + 1; urlType < BKF_GET_ARY_COUNT(connMng->urlTypeSet); urlType++) {
        BkfSuberConnProcDelConnByUrlType(connMng, urlType);
    }

    return;
}

void BkfSuberConnProcStartConnByUrlType(BkfSuberConnMng *connMng, uint8_t urlType)
{
    BkfSuberConn *conn = VOS_NULL;
    void *itor = VOS_NULL;

    for (conn = BkfSuberConnDataGetFirstByUrlType(connMng, urlType, &itor); conn != VOS_NULL;
         conn = BkfSuberConnDataGetNextByUrlType(connMng, urlType, &itor)) {
        BkfSuberConnProcConn(conn);
    }

    return;
}

void BkfSuberConnProcStartConnByAllUrlType(BkfSuberConnMng *connMng)
{
    uint32_t urlType;
    for (urlType = BKF_URL_TYPE_INVALID + 1; urlType < BKF_GET_ARY_COUNT(connMng->urlTypeSet); urlType++) {
        BkfSuberConnProcStartConnByUrlType(connMng, urlType);
    }

    return;
}

static inline BOOL BkfSuberConnNeedProcConn(BkfSuberConn *conn)
{
    return (conn->obInstCnt == 1) ? VOS_TRUE : VOS_FALSE;
}

static inline BOOL BkfSuberConnNeedProcDelConn(BkfSuberConn *conn)
{
    return (conn->obInstCnt == 0) ? VOS_TRUE : VOS_FALSE;
}

static inline void BkfSuberConnIncInstCnt(BkfSuberConn *conn)
{
    if (conn->obInstCnt < 0xFFFFFFFF) {
        conn->obInstCnt++;
    }
    return;
}

static inline void BkfSuberConnDeccInstCnt(BkfSuberConn *conn)
{
    if (conn->obInstCnt > 0) {
        conn->obInstCnt--;
    }
    return;
}

uint32_t BkfSuberConnProcAddPuberUrl(BkfSuberConnMng *connMng, BkfSuberInst *inst)
{
    BkfSuberConn *conn = BkfSuberConnDataFind(connMng, &inst->puberUrl, &inst->localUrl);
    if (conn == VOS_NULL) {
        conn = BkfSuberConnCreateConn(connMng, inst);
        if (conn == VOS_NULL) {
            BKF_LOG_ERROR(connMng->log, "create conn fail.\n");
            return BKF_ERR;
        }
    }

    if (BKF_DL_NODE_IS_IN(&inst->nodeObBySubConn)) {
        BKF_ASSERT(0); // 有残留
        BKF_DL_REMOVE(&inst->nodeObBySubConn);
    }
    BKF_DL_ADD_LAST(&conn->obInst, &inst->nodeObBySubConn);

    BkfSuberConnIncInstCnt(conn);
    if (BkfSuberConnNeedProcConn(conn)) {
        BkfSuberConnProcConn(conn);
    }
    return BKF_OK;
}

void BkfSuberConnProcDelPuberUrl(BkfSuberConnMng *connMng, BkfUrl *puberUrl, BkfUrl *localUrl, BkfDlNode *instNodeForOb)
{
    BkfSuberConn *conn = BkfSuberConnDataFind(connMng, puberUrl, localUrl);
    if (conn == VOS_NULL) {
        BKF_ASSERT(0); // conn必须存在
        return;
    }

    if (BKF_DL_NODE_IS_IN(instNodeForOb)) {
        BKF_DL_REMOVE(instNodeForOb);
    }

    BkfSuberConnDeccInstCnt(conn);
    if (BkfSuberConnNeedProcDelConn(conn)) {
        BkfSuberConnProcDelConn(conn);
    }

    return;
}

void BkfSuberConnProcCreateSess(BkfSuberAppSub *appSub, BkfSuberConnTrigSessPara *para, BOOL *needContinue)
{
    BkfSuberConnSessArg arg = {.puberUrl = para->puberUrl,
        .sliceKey = appSub->key.sliceKey,
        .tableTypeId = appSub->key.tableTypeId,
        .isVerify = appSub->isVerify,
        .localUrl = para->localUrl};
    uint32_t ret = BkfSuberConnCreateSess(para->connMng, &arg);
    if (needContinue != VOS_NULL) {
        if (ret != BKF_OK) {
            *needContinue = VOS_FALSE;
        } else {
            *needContinue = VOS_TRUE;
        }
    }

    return;
}

void BkfSuberConnProcCreateSessBySlice(BkfSuberConnMng *connMng, BkfSuberSlice *slice, BkfUrl *puberUrl,
    BkfUrl *localUrl, uint64_t instId)
{
    BkfSuberConnTrigSessPara para = {.connMng = connMng, .puberUrl = puberUrl, .localUrl = localUrl};
    BkfSuberDataScanAppSubBySlice(slice, instId, (F_BKF_SUBER_DATA_SCAN_APPSUB_BY_SLICE_PROC)BkfSuberConnProcCreateSess,
        &para);

    return;
}

void BkfSuberConnProcCreateSessByInst(BkfSuberConnMng *connMng, BkfSuberInst *inst)
{
    BkfDlNode *node = VOS_NULL;
    BkfDlNode *nextNode = VOS_NULL;

    for (node = BKF_DL_GET_FIRST(&inst->obSlice); node != VOS_NULL; node = nextNode) {
        nextNode = BKF_DL_GET_NEXT(&inst->obSlice, node);
        BkfSuberSliceInst *sliceInst = BKF_DL_GET_ENTRY(node, BkfSuberSliceInst, nodeObByInst);
        BkfSuberSlice *slice = sliceInst->slice;
        if (slice) {
            BkfSuberConnProcCreateSessBySlice(connMng, slice, &inst->puberUrl, &inst->localUrl, inst->instId);
        }
    }

    return;
}

void BkfSuberConnProcDelSess(BkfSuberAppSub *appSub, BkfSuberConnTrigSessPara *para, BOOL *needContinue)
{
    BkfSuberConnSessArg arg = {.puberUrl = para->puberUrl,
        .sliceKey = appSub->key.sliceKey,
        .tableTypeId = appSub->key.tableTypeId,
        .isVerify = appSub->isVerify,
        .localUrl = para->localUrl};
    BkfSuberConnDelSess(para->connMng, &arg);
    if (needContinue != VOS_NULL) {
        *needContinue = VOS_TRUE;
    }

    return;
}

void BkfSuberConnProcDelSessBySlice(BkfSuberConnMng *connMng, BkfSuberSlice *slice, BkfUrl *puberUrl, BkfUrl *localUrl,
    uint64_t instId)
{
    BkfSuberConnTrigSessPara para = {.connMng = connMng, .puberUrl = puberUrl, .localUrl = localUrl};
    BkfSuberDataScanAppSubBySlice(slice, instId, (F_BKF_SUBER_DATA_SCAN_APPSUB_BY_SLICE_PROC)BkfSuberConnProcDelSess,
        &para);

    return;
}

BkfSuberInst *BkfSuberConnGetValidInstByAppSub(BkfSuberAppSub *appSub)
{
    BkfSuberInst *inst = BkfSuberDataFindInst(appSub->dataMng, appSub->key.instId);
    if (inst == VOS_NULL || inst->isTemp || !BkfUrlIsValid(&inst->puberUrl)) {
        return VOS_NULL;
    }

    return inst;
}
#endif

#if BKF_BLOCK("公有函数定义")
BkfSuberConnMng *BkfSuberConnMngInit(BkfSuberConnMngInitArg *initArg)
{
    BkfSuberConnMng *connMng = BkfSuberConnDataMngInit(initArg);
    if (connMng == VOS_NULL) {
        return VOS_NULL;
    }

    uint32_t ret = BkfSuberConnFsmInitFsmTmp(connMng);
    if (ret != BKF_OK) {
        BkfSuberConnMngUnInit(connMng);
        return VOS_NULL;
    }

    ret = BkfSuberConnJobRegType(connMng);
    if (ret != BKF_OK) {
        BkfSuberConnMngUnInit(connMng);
        return VOS_NULL;
    }

    BkfSuberConnDispInit(connMng);
    return connMng;
}

void BkfSuberConnMngUnInit(BkfSuberConnMng *connMng)
{
    BkfSuberConnDispUninit(connMng);
    /* suber析构时会先删除数据，此时会联动删除conn，下面的处理为容错编码 */
    BkfSuberConnProcDelAllConn(connMng);
    BkfSuberConnDataMngUnInit(connMng);
    return;
}

uint32_t BkfSuberConnMngSetSelfUrl(BkfSuberConnMng *connMng, BkfUrl *selfUrl)
{
    BkfSuberConnUrlTypeMng *tableTypeMng = BkfSuberConnDataCreateAndGetUrlTypeMng(connMng, selfUrl->type);
    if (tableTypeMng == VOS_NULL) {
        BKF_LOG_ERROR(connMng->log, "get url type mng fail");
        return BKF_ERR;
    }

    if (BkfUrlIsValid(&tableTypeMng->selfUrl)) {
        BKF_ASSERT(0); // 不允许修改self url
        return BKF_ERR;
    }
    uint32_t ret = BkfChCliSetSelfUrl(connMng->env->chCliMng, selfUrl);
    if (ret != BKF_OK) {
        uint8_t urlStr[BKF_URL_STR_LEN_MAX] = {0};
        BKF_LOG_ERROR(connMng->log, "ch set self url fail, ret %u, url %s", ret,
            BkfUrlGetStr(selfUrl, urlStr, sizeof(urlStr)));
        return ret;
    }

    tableTypeMng->selfUrl = *selfUrl;

    if (selfUrl->type == BKF_URL_TYPE_V8TLS || selfUrl->type == BKF_URL_TYPE_V8TCP) {
        /* v8tls类型,不依赖selfurl:todo:可以整体url封装，mesh suber也不依赖selfurl */
        return BKF_OK;
    }
    if (connMng->isEnable) {
        BkfSuberConnProcStartConnByUrlType(connMng, selfUrl->type);
    }
    return BKF_OK;
}

uint32_t BkfSuberConnEnable(BkfSuberConnMng *connMng)
{
    if (connMng->isEnable) {
        return BKF_OK;
    }

    BkfChCliEnableArg chCliEnableArg = {.cookie = connMng,
        .onRcvData = (F_BKF_CH_CLI_ON_RCV_DATA)BkfSuberConnRcvData,
        .onRcvDataEvent = (F_BKF_CH_CLI_ON_RCV_DATA_EVENT)BkfSuberConnRcvDataEvent,
        .onUnblock = (F_BKF_CH_CLI_ON_UNBLOCK)BkfSuberConnUnBlock,
        .onDisconn = (F_BKF_CH_CLI_ON_DISCONN)BkfSuberConnDisConn,
        .onDisconnEx = (F_BKF_CH_CLI_ON_DISCONNEX)BkfSuberConnDisConnEx,
        .onConn = (F_BKF_CH_CLI_ON_CONN)BkfSuberConnOnConn};
    uint32_t ret = BkfChCliEnable(connMng->env->chCliMng, &chCliEnableArg);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(connMng->log, "ch enable fail, ret %u", ret);
        return ret;
    }

    connMng->isEnable = VOS_TRUE;
    BkfSuberConnProcStartConnByAllUrlType(connMng);
    return BKF_OK;
}

void BkfSuberConnProcInstAddUrl(BkfSuberConnMng *connMng, BkfSuberInst *inst)
{
    if (!BkfUrlIsValid(&inst->puberUrl)) {
        return;
    }

    uint32_t ret = BkfSuberConnProcAddPuberUrl(connMng, inst);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(connMng->log, "add url fail.\n");
        return;
    }

    BkfSuberConnProcCreateSessByInst(connMng, inst);
    return;
}

void BkfSuberConnProcInstDelUrl(BkfSuberConnMng *connMng, BkfSuberInst *inst)
{
    if (!BkfUrlIsValid(&inst->puberUrl)) {
        return;
    }

    BkfSuberConnProcDelPuberUrl(connMng, &inst->puberUrl, &inst->localUrl, &inst->nodeObBySubConn);
    return;
}

void BkfSuberConnProcSliceAddInst(BkfSuberConnMng *connMng, BkfSuberSlice *slice, uint64_t instId)
{
    BkfSuberInst *inst = BkfSuberDataFindInst(slice->dataMng, instId);
    if (inst == VOS_NULL) {
        return;
    }

    if (inst->isTemp || !BkfUrlIsValid(&inst->puberUrl)) {
        return;
    }

    BkfSuberConnProcCreateSessBySlice(connMng, slice, &inst->puberUrl, &inst->localUrl, instId);
    return;
}

void BkfSuberConnProcSliceDelInst(BkfSuberConnMng *connMng, BkfSuberSlice *slice, uint64_t instId)
{
    BkfSuberInst *inst = BkfSuberDataFindInst(slice->dataMng, instId);
    if (inst == VOS_NULL) {
        return;
    }

    if (inst->isTemp || !BkfUrlIsValid(&inst->puberUrl)) {
        return;
    }

    BkfSuberConnProcDelSessBySlice(connMng, slice, &inst->puberUrl, &inst->localUrl, instId);
    return;
}

void BkfSuberConnProcAddSub(BkfSuberConnMng *connMng, BkfSuberAppSub *appSub)
{
    BkfSuberInst *inst = BkfSuberConnGetValidInstByAppSub(appSub);
    if (inst == VOS_NULL) {
        return;
    }

    BkfSuberConnTrigSessPara para = {.connMng = connMng, .puberUrl = &inst->puberUrl, .localUrl = &inst->localUrl};
    BkfSuberConnProcCreateSess(appSub, &para, VOS_NULL);
    return;
}

void BkfSuberConnProcDelSub(BkfSuberConnMng *connMng, BkfSuberAppSub *appSub)
{
    BkfSuberInst *inst = BkfSuberConnGetValidInstByAppSub(appSub);
    if (inst == VOS_NULL) {
        return;
    }

    BkfSuberConnTrigSessPara para = {.connMng = connMng, .puberUrl = &inst->puberUrl, .localUrl = &inst->localUrl};
    BkfSuberConnProcDelSess(appSub, &para, VOS_NULL);
    return;
}
#endif

#ifdef __cplusplus
}
#endif