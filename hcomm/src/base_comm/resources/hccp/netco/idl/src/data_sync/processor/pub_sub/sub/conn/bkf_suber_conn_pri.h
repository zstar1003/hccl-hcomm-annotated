/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_SUBER_CONN_PRI_H
#define BKF_SUBER_CONN_PRI_H

#include "bkf_suber_conn.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(4)
typedef struct {
    BkfUrl *puberUrl;
    void *sliceKey;
    uint16_t tableTypeId;
    uint16_t pad;
    BOOL isVerify;
    BkfUrl *localUrl;
} BkfSuberConnSessArg;
#pragma pack()

uint32_t BkfSuberConnCreateSess(BkfSuberConnMng *connMng, BkfSuberConnSessArg *arg);
void BkfSuberConnDelSess(BkfSuberConnMng *connMng, BkfSuberConnSessArg *arg);
BkfSuberConn *BkfSuberConnCreateConn(BkfSuberConnMng *connMng, BkfSuberInst *inst);
void BkfSuberConnProcConn(BkfSuberConn *conn);
void BkfSuberConnProcDisConn(BkfSuberConn *conn);
void BkfSuberConnProcDelConn(BkfSuberConn *conn);
void BkfSuberConnTrigSched(BkfSuberConn *conn);
uint32_t BkfSuberConnJobRegType(BkfSuberConnMng *connMng);
void BkfSuberConnUnBlock(BkfSuberConnMng *connMng, BkfChCliConnId *connId);
void BkfSuberConnDisConn(BkfSuberConnMng *connMng, BkfChCliConnId *connId);
void BkfSuberConnDisConnEx(BkfSuberConnMng *connMng, BkfChCliConnId *connId, BOOL isPeerFin);
void BkfSuberConnRcvDataEvent(BkfSuberConnMng *connMng, BkfChCliConnId *connId);
void BkfSuberConnRcvData(BkfSuberConnMng *connMng, BkfChCliConnId *connId, uint8_t *data, int32_t dataLen);
void BkfSuberConnOnConn(BkfSuberConnMng *connMng, BkfChCliConnId *connId, uint16_t srcPort);
#ifdef __cplusplus
}
#endif
#endif