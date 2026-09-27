/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_SUBER_CONN_DATA_H
#define BKF_SUBER_CONN_DATA_H

#include "bkf_suber_sess.h"
#include "bkf_suber_conn.h"
#include "bkf_suber_env.h"
#include "bkf_url.h"
#include "bkf_bufq.h"
#include "v_avll.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma pack(4)
typedef struct tagBkfSuberConn BkfSuberConn;
typedef struct {
    BkfUrl selfUrl;
    AVLL_TREE connSet;
    uint32_t connCnt;
    BkfSuberConnMng *connMng;
    uint8_t urlType;
    uint8_t pad[3];
} BkfSuberConnUrlTypeMng;

struct tagBkfSuberConnMng {
    char *name;
    BkfSuberEnv *env;
    BkfLog *log;
    BkfSuberConnUrlTypeMng *urlTypeSet[BKF_URL_TYPE_CNT];
    AVLL_TREE connSetByConnId;
    BkfFsmTmpl *connFsmTmpl;
    char connFsmTmplName[BKF_NAME_LEN_MAX + 1];
    uint8_t isEnable : 1;
    uint8_t rsv : 7;
    uint8_t pad[3];
};

struct tagBkfSuberConn {
    BkfSuberConnUrlTypeMng *urlTypeMng;
    BkfSuberSessMng *sessMng;
    AVLL_NODE avlNode;
    AVLL_NODE connIdKeyNode;
    BkfChCliConnId *connId;
    BkfFsm fsm;
    BkfUrl puberUrl;
    BkfUrl localUrl;
    BkfDl obInst;
    uint32_t obInstCnt;
    BkfJobId *jobId;
    uint8_t lockDel : 1;
    uint8_t isTrigConn : 1;
    uint8_t softDel : 1;
    uint8_t pad : 5;
    BkfBufq *rcvDataBuf;
    BkfBufq *lastSendLeftDataBuf;
    BkfTmrId *reConnTmrId;
    uint16_t srcPort; /* 算网dfx */
    uint16_t pad2;
};
#pragma pack()
BkfSuberConnMng *BkfSuberConnDataMngInit(BkfSuberConnMngInitArg *initArg);
void BkfSuberConnDataMngUnInit(BkfSuberConnMng *connMng);
BkfSuberConnUrlTypeMng *BkfSuberConnDataGetUrlTypeMng(BkfSuberConnMng *connMng, uint8_t urlType);
BkfSuberConnUrlTypeMng *BkfSuberConnDataCreateAndGetUrlTypeMng(BkfSuberConnMng *connMng, uint8_t urlType);
BkfSuberConn *BkfSuberConnDataAdd(BkfSuberConnMng *connMng, BkfUrl *puberUrl, BkfUrl *localUrl);
void BkfSuberConnDataDel(BkfSuberConn *conn);
BkfSuberConn *BkfSuberConnDataFind(BkfSuberConnMng *connMng, BkfUrl *puberUrl, BkfUrl *localUrl);
BkfSuberConn *BkfSuberConnDataFindNext(BkfSuberConnMng *connMng, BkfUrl *puberUrl, BkfUrl *localUrl);
BkfSuberConn *BkfSuberConnDataGetFirstByUrlType(BkfSuberConnMng *connMng, uint8_t urlType, void **itorOutOrNull);
BkfSuberConn *BkfSuberConnDataGetNextByUrlType(BkfSuberConnMng *connMng, uint8_t urlType, void **itorInOut);
BkfSuberConn *BkfSuberConnDataFindByConnId(BkfSuberConnMng *connMng, BkfChCliConnId *connId);
uint32_t BkfSuberConnDataAddToConnIdSet(BkfSuberConn *conn, BkfChCliConnId *connId);
void BkfSuberConnDataDelFromConnIdSet(BkfSuberConn *conn);
void BkfSuberConnStopJob(BkfSuberConn *conn);
uint32_t BkfSuberConnStartJob(BkfSuberConn *conn);
#ifdef __cplusplus
}
#endif
#endif