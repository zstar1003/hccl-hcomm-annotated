/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_puber_conn_sys_log.h"
#include "bkf_puber_conn_data.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOG_ID_PUBER_CONN_EXCEED 0x089611b7
/* PUBER_CONN_EXCEED(D):(urlCli=%s, connectionCount=%d, connectionCountMax=%d) */
#define DIAG_ID_FMT_PUBER_CONN_EXCEED LOG_ID_PUBER_CONN_EXCEED, "%s%d%d"

STATIC uint32_t BkfPuberConnLimitOnSysLogCode(BkfPuberConnMng *connMng, BkfUrl *urlCli, int32_t *logTimeConnCnt,
    F_BKF_SYS_LOG_PRINTF logPrintf, uint32_t logPrintfParam1)
{
    uint8_t buf[BKF_LOG_LEN] = {0};
    (void)logPrintf(logPrintfParam1, DIAG_ID_FMT_PUBER_CONN_EXCEED, BkfUrlGetStr(urlCli, buf, sizeof(buf)),
        *logTimeConnCnt, connMng->argInit->connCntMax);
    return BKF_OK;
}

uint32_t BkfPuberConnSysLogInit(BkfPuberConnMng *connMng)
{
    BkfSysLogTypeVTbl vTbl = {0};
    vTbl.name = "LOG_ID_PUBER_CONN_EXCEED";
    vTbl.cookie = connMng;
    vTbl.typeId = LOG_ID_PUBER_CONN_EXCEED;
    vTbl.needRestrain = VOS_TRUE;
    vTbl.keyLen = sizeof(BkfUrl);
    vTbl.valLen = sizeof(connMng->connCnt);
    vTbl.keyCmp = (F_BKF_CMP)BkfUrlCmp;
    vTbl.keyGetStrOrNull = (F_BKF_GET_STR)BkfUrlGetStr;
    vTbl.valGetStrOrNull = (F_BKF_GET_STR)BkfInt32GetStr;
    vTbl.out = (F_BKF_SYS_LOG_OUT)BkfPuberConnLimitOnSysLogCode;
    return BkfSysLogReg(connMng->argInit->sysLogMng, &vTbl);
}

void BkfPuberConnSysLogUninit(BkfPuberConnMng *connMng)
{
    (void)connMng;
    return; /* 默认空实现 */
}

uint32_t BkfPuberConnLimitSysLog(BkfPuberConnMng *connMng, BkfUrl *urlCli)
{
    return BkfSysLogFunc(connMng->argInit->sysLogMng, LOG_ID_PUBER_CONN_EXCEED, urlCli, &connMng->connCnt);
}

#ifdef __cplusplus
}
#endif
