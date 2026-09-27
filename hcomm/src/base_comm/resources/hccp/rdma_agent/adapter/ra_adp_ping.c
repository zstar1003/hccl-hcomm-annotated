/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ra_hdc_ping.h"
#include "ra_hdc.h"
#include "ra_adp.h"
#include "rs_ping.h"
#include "config_log.h"
#include "ra_adp_ping.h"

struct RsPingOps gPingOps = {
    .pingInit = RsPingInit,
    .pingTargetAdd = RsPingTargetAdd,
    .pingTaskStart = RsPingTaskStart,
    .pingGetResults = RsPingGetResults,
    .pingTaskStop = RsPingTaskStop,
    .pingTargetDel = RsPingTargetDel,
    .pingDeinit = RsPingDeinit,
};

int RaRsPingInit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpPingInitData *pingDataOut = (union OpPingInitData *)(outBuf + sizeof(struct MsgHead));
    union OpPingInitData *pingData = (union OpPingInitData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpPingInitData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gPingOps.pingInit(&pingData->txData.attr, &pingDataOut->rxData.info, &pingDataOut->rxData.devIndex);
    if (*opResult != 0) {
        hccp_err("ping_init failed ret[%d].", *opResult);
    }
    // only negative return value will be parsed
    if (*opResult > 0) {
        *opResult = -*opResult;
    }

    return 0;
}

int RaRsPingTargetAdd(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpPingAddData *pingData = (union OpPingAddData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpPingAddData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gPingOps.pingTargetAdd(&pingData->txData.rdev, &pingData->txData.target);
    if (*opResult != 0) {
        hccp_err("ping_target_add failed ret[%d].", *opResult);
    }
    // only negative return value will be parsed
    if (*opResult > 0) {
        *opResult = -*opResult;
    }

    return 0;
}

int RaRsPingTaskStart(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpPingStartData *pingData = (union OpPingStartData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpPingStartData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gPingOps.pingTaskStart(&pingData->txData.rdev, &pingData->txData.attr);
    if (*opResult != 0) {
        hccp_err("ping_task_start failed ret[%d].", *opResult);
    }
    // only negative return value will be parsed
    if (*opResult > 0) {
        *opResult = -*opResult;
    }

    return 0;
}

int RaRsPingGetResults(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpPingResultsData *pingDataOut = (union OpPingResultsData *)(outBuf + sizeof(struct MsgHead));
    union OpPingResultsData *pingData = (union OpPingResultsData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpPingResultsData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(pingData->txData.num, 0, RA_MAX_PING_TARGET_NUM, opResult);

    *opResult = gPingOps.pingGetResults(&pingData->txData.rdev, pingData->txData.target, &pingData->txData.num,
        pingDataOut->rxData.target);
    // caller needs to retry, degrade log level
    if (*opResult == -EAGAIN) {
        hccp_warn_rma("ping_get_results unsuccessful, ret[%d].", *opResult);
    } else if (*opResult != 0) {
        hccp_err("ping_get_results failed, ret[%d].", *opResult);
    }
    // only negative return value will be parsed
    if (*opResult > 0) {
        *opResult = -*opResult;
        return 0;
    }
    pingDataOut->rxData.num = pingData->txData.num;

    return 0;
}

int RaRsPingTaskStop(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpPingStopData *pingData = (union OpPingStopData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpPingStopData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gPingOps.pingTaskStop(&pingData->txData.rdev);
    if (*opResult != 0) {
        hccp_err("ping_task_stop failed ret[%d].", *opResult);
    }
    // only negative return value will be parsed
    if (*opResult > 0) {
        *opResult = -*opResult;
    }

    return 0;
}

int RaRsPingTargetDel(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpPingDelData *pingData = (union OpPingDelData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpPingDelData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(pingData->txData.num, 0, RA_MAX_PING_TARGET_NUM, opResult);

    *opResult = gPingOps.pingTargetDel(&pingData->txData.rdev, pingData->txData.target, &pingData->txData.num);
    if (*opResult != 0) {
        hccp_err("ping_target_del failed ret[%d].", *opResult);
    }
    // only negative return value will be parsed
    if (*opResult > 0) {
        *opResult = -*opResult;
    }

    return 0;
}

int RaRsPingDeinit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpPingDeinitData *pingData = (union OpPingDeinitData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpPingDeinitData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gPingOps.pingDeinit(&pingData->txData.rdev);
    if (*opResult != 0) {
        hccp_err("ping_deinit failed ret[%d].", *opResult);
    }
    // only negative return value will be parsed
    if (*opResult > 0) {
        *opResult = -*opResult;
    }

    return 0;
}
