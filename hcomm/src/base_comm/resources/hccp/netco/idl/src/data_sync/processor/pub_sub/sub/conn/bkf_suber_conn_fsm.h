/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_SUBER_SUB_CONN_FSM_H
#define BKF_SUBER_SUB_CONN_FSM_H

#include "bkf_suber_conn.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    BACKFIN_SUB_CONN_STATE_CRTINGFD,
    BACKFIN_SUB_CONN_STATE_WAIT_SND_HELLO,
    BACKFIN_SUB_CONN_STATE_WAIT_HELLO_ACK,
    BACKFIN_SUB_CONN_STATE_UP,
    BACKFIN_SUB_CONN_STATE_MAX
};

enum {
    BACKFIN_SUB_CONN_INPUT_STARTINST,
    BACKFIN_SUB_CONN_INPUT_UNBLOCK,
    BACKFIN_SUB_CONN_INPUT_RCVHELLOACK,
    BACKFIN_SUB_CONN_INPUT_TCPDISCONN,
    BACKFIN_SUB_CONN_INPUT_STOPINST,
    BACKFIN_SUB_CONN_INPUT_JOB,
    BACKFIN_SUB_CONN_INPUT_MAX
};

static inline uint32_t BkfSuberConnFsmInputEvt(BkfSuberConn *conn, uint8_t evt, void *para1orNull, void *para2orNull)
{
    return BKF_FSM_DISPATCH(&conn->fsm, evt, conn, para1orNull, para2orNull);
}
uint32_t BkfSuberConnFsmInitFsmTmp(BkfSuberConnMng *connMng);
void BkfSuberConnFsmUnInitFsmTmp(BkfSuberConnMng *connMng);

#ifdef __cplusplus
}
#endif
#endif