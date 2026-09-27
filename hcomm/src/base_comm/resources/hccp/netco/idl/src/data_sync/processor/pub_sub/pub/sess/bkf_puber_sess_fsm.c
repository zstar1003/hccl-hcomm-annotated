/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((sessMng)->log)
#define BKF_MOD_NAME ((sessMng)->name)
#define BKF_LINE (__LINE__ + 10000)

#include "bkf_puber_sess_fsm.h"
#include "bkf_puber_sess_utils.h"
#include "securec.h"
#include "vos_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#if BKF_BLOCK("私有宏结构函数")
/* fsm proc */
STATIC uint32_t BkfPuberSessBatchProcSub(BkfPuberSess *sess, BkfMsgDecoder *decoder, void *unused2);
STATIC uint32_t BkfPuberSessProcSub(BkfPuberSess *sess, BkfMsgDecoder *decoder, void *unused2);
STATIC uint32_t BkfPuberSessBatchProcUnsub(BkfPuberSess *sess, BkfMsgDecoder *decoder, void *unused2);
STATIC uint32_t BkfPuberSessProcUnsub(BkfPuberSess *sess, BkfMsgDecoder *decoder, void *unused2);
STATIC uint32_t BkfPuberSessTry2SendSubAck(BkfPuberSess *sess, BkfMsgCoder *coder, void *unused2);
STATIC uint32_t BkfPuberSessChgState2WaitSendBatchBegin(BkfPuberSess *sess, void *unused1, void *unused2);
STATIC uint32_t BkfPuberSessTry2SendBatchBegin(BkfPuberSess *sess, BkfMsgCoder *coder, void *unused2);
STATIC uint32_t BkfPuberSessTry2SendBatchEnd(BkfPuberSess *sess, BkfMsgCoder *coder, void *unused2);
STATIC uint32_t BkfPuberSessTry2SendSubDelNtf(BkfPuberSess *sess, BkfMsgCoder *coder, void *unused2);
STATIC uint32_t BkfPuberSessIdleProcTupleChg(BkfPuberSess *sess, void *unused1, void *unused2);
STATIC uint32_t BkfPuberSessSlowBatchProcTupleChg(BkfPuberSess *sess, void *unused1, void *unused2);
STATIC uint32_t BkfPuberSessProcTableRel(BkfPuberSess *sess, void *unused1, void *unused2);
STATIC uint32_t BkfPuberSessBatchProcSched(BkfPuberSess *sess, BkfMsgCoder *coder, void *unused2);
STATIC uint32_t BkfPuberSessRealProcSched(BkfPuberSess *sess, BkfMsgCoder *coder, void *unused2);
STATIC uint32_t BkfPuberSessSlowBatchProcSlowSched(BkfPuberSess *sess, BkfMsgCoder *coder,
    BkfPuberSessSlowSchedCtx *ctx);
STATIC uint32_t BkfPuberSessRealSlowBatchProcSched(BkfPuberSess *sess, BkfMsgCoder *coder, void *unused2);
STATIC void BkfPuberSessBatchChkTmrStart(BkfPuberSess *sess);
/* other proc */
STATIC BOOL BkfPuberSessProcSubMayProc(BkfPuberSess *sess, BkfMsgHead *msgHead, BkfTlvTransNum *tlvTransNum,
    BOOL inBatchState);
STATIC uint32_t BkfPuberSessProcSubDo(BkfPuberSess *sess, BkfMsgHead *msgHead, BkfTlvTransNum *tlvTransNum);
STATIC BOOL BkfPuberSessProcUnsubMayProc(BkfPuberSess *sess, BkfMsgHead *msgHead, BkfTlvTransNum *tlvTransNum,
    BOOL inBatchState);
STATIC uint32_t BkfPuberSessProcUnsubDo(BkfPuberSess *sess, BkfMsgHead *msgHead, BkfTlvTransNum *tlvTransNum);
STATIC uint32_t BkfPuberSessTry2SendBatchData(BkfPuberSess *sess, BkfMsgCoder *coder, BkfPuberSessSlowSchedCtx *ctx);
STATIC uint32_t BkfPuberSessTry2SendRealData(BkfPuberSess *sess, BkfMsgCoder *coder);

#endif

#if BKF_BLOCK("表")
const char *g_BkfPuberSessFsmStateStrTbl[] = {
    "BKF_PUBER_SESS_STATE_WAIT_SUB",
    "BKF_PUBER_SESS_STATE_WAIT_SEND_SUB_ACK",
    "BKF_PUBER_SESS_STATE_WAIT_MAY_SEND_BATCH_BEGIN",
    "BKF_PUBER_SESS_STATE_WAIT_SEND_BATCH_BEGIN",
    "BKF_PUBER_SESS_STATE_WAIT_SEND_BATCH_END",
    "BKF_PUBER_SESS_STATE_WAIT_SEND_DEL_SUB_NTF",
    "BKF_PUBER_SESS_STATE_BATCH_DATA",
    "BKF_PUBER_SESS_STATE_REAL_DATA",
    "BKF_PUBER_SESS_STATE_IDLE",
    "BKF_PUBER_SESS_STATE_SLOW_BATCH_DATA",
    "BKF_PUBER_SESS_STATE_REAL_SLOW_BATCH_DATA",
};
BKF_STATIC_ASSERT(BKF_GET_ARY_COUNT(g_BkfPuberSessFsmStateStrTbl) == BKF_PUBER_SESS_STATE_CNT);

const char *g_BkfPuberSessFsmEvtStrTbl[] = {
    "BKF_PUBER_SESS_EVT_SUB",
    "BKF_PUBER_SESS_EVT_UNSUB",
    "BKF_PUBER_SESS_EVT_TUPLE_CHG",
    "BKF_PUBER_SESS_EVT_TABLE_CPLT",
    "BKF_PUBER_SESS_EVT_TABLE_REL",
    "BKF_PUBER_SESS_EVT_SCHED",
    "BKF_PUBER_SESS_EVT_SLOW_SCHED",
};
BKF_STATIC_ASSERT(BKF_GET_ARY_COUNT(g_BkfPuberSessFsmEvtStrTbl) == BKF_PUBER_SESS_EVT_CNT);

const BkfFsmProcItem g_BkfPuberSessFsm[][BKF_PUBER_SESS_EVT_CNT] = {
    /* BKF_PUBER_SESS_STATE_WAIT_SUB */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcSub)}, /* BKF_PUBER_SESS_EVT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},  /* BKF_PUBER_SESS_EVT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},  /* BKF_PUBER_SESS_EVT_TUPLE_CHG */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},  /* BKF_PUBER_SESS_EVT_TABLE_CPLT */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},  /* BKF_PUBER_SESS_EVT_TABLE_REL */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},  /* BKF_PUBER_SESS_EVT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},  /* BKF_PUBER_SESS_EVT_SLOW_SCHED */
    },
    /* BKF_PUBER_SESS_STATE_WAIT_SEND_SUB_ACK */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcSub)},   /* BKF_PUBER_SESS_EVT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcUnsub)}, /* BKF_PUBER_SESS_EVT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},        /* BKF_PUBER_SESS_EVT_TUPLE_CHG */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},        /* BKF_PUBER_SESS_EVT_TABLE_CPLT */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessProcTableRel)},   /* BKF_PUBER_SESS_EVT_TABLE_REL */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessTry2SendSubAck)}, /* BKF_PUBER_SESS_EVT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},    /* BKF_PUBER_SESS_EVT_SLOW_SCHED */
    },
    /* BKF_PUBER_SESS_STATE_WAIT_MAY_SEND_BATCH_BEGIN */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcSub)},                /* BKF_PUBER_SESS_EVT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcUnsub)},              /* BKF_PUBER_SESS_EVT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},                     /* BKF_PUBER_SESS_EVT_TUPLE_CHG */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessChgState2WaitSendBatchBegin)}, /* BKF_PUBER_SESS_EVT_TABLE_CPLT */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessProcTableRel)},                /* BKF_PUBER_SESS_EVT_TABLE_REL */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},                 /* BKF_PUBER_SESS_EVT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},                 /* BKF_PUBER_SESS_EVT_SLOW_SCHED */
    },
    /* BKF_PUBER_SESS_STATE_WAIT_SEND_BATCH_BEGIN */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcSub)},       /* BKF_PUBER_SESS_EVT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcUnsub)},     /* BKF_PUBER_SESS_EVT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},            /* BKF_PUBER_SESS_EVT_TUPLE_CHG */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},            /* BKF_PUBER_SESS_EVT_TABLE_CPLT */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessProcTableRel)},       /* BKF_PUBER_SESS_EVT_TABLE_REL */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessTry2SendBatchBegin)}, /* BKF_PUBER_SESS_EVT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},        /* BKF_PUBER_SESS_EVT_SLOW_SCHED */
    },
    /* BKF_PUBER_SESS_STATE_WAIT_SEND_BATCH_END */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcSub)},     /* BKF_PUBER_SESS_EVT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcUnsub)},   /* BKF_PUBER_SESS_EVT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},          /* BKF_PUBER_SESS_EVT_TUPLE_CHG */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},          /* BKF_PUBER_SESS_EVT_TABLE_CPLT */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessProcTableRel)},     /* BKF_PUBER_SESS_EVT_TABLE_REL */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessTry2SendBatchEnd)}, /* BKF_PUBER_SESS_EVT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},      /* BKF_PUBER_SESS_EVT_SLOW_SCHED */
    },
    /* BKF_PUBER_SESS_STATE_WAIT_SEND_DEL_SUB_NTF */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},           /* BKF_PUBER_SESS_EVT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},           /* BKF_PUBER_SESS_EVT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},           /* BKF_PUBER_SESS_EVT_TUPLE_CHG */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},           /* BKF_PUBER_SESS_EVT_TABLE_CPLT */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},           /* BKF_PUBER_SESS_EVT_TABLE_REL */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessTry2SendSubDelNtf)}, /* BKF_PUBER_SESS_EVT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},       /* BKF_PUBER_SESS_EVT_SLOW_SCHED */
    },
    /* BKF_PUBER_SESS_STATE_BATCH_DATA */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcSub)},   /* BKF_PUBER_SESS_EVT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcUnsub)}, /* BKF_PUBER_SESS_EVT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},        /* BKF_PUBER_SESS_EVT_TUPLE_CHG */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},        /* BKF_PUBER_SESS_EVT_TABLE_CPLT */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessProcTableRel)},   /* BKF_PUBER_SESS_EVT_TABLE_REL */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcSched)}, /* BKF_PUBER_SESS_EVT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},    /* BKF_PUBER_SESS_EVT_SLOW_SCHED */
    },
    /* BKF_PUBER_SESS_STATE_REAL_DATA */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessProcSub)},       /* BKF_PUBER_SESS_EVT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessProcUnsub)},     /* BKF_PUBER_SESS_EVT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},       /* BKF_PUBER_SESS_EVT_TUPLE_CHG */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},       /* BKF_PUBER_SESS_EVT_TABLE_CPLT */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessProcTableRel)},  /* BKF_PUBER_EVT_TABLE_DEL */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessRealProcSched)}, /* BKF_PUBER_SESS_EVT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},   /* BKF_PUBER_SESS_EVT_SLOW_SCHED */
    },
    /* BKF_PUBER_SESS_STATE_IDLE */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessProcSub)},          /* BKF_PUBER_SESS_EVT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessProcUnsub)},        /* BKF_PUBER_SESS_EVT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessIdleProcTupleChg)}, /* BKF_PUBER_SESS_EVT_TUPLE_CHG */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},          /* BKF_PUBER_SESS_EVT_TABLE_CPLT */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessProcTableRel)},     /* BKF_PUBER_SESS_EVT_TABLE_REL */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},      /* BKF_PUBER_SESS_EVT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},      /* BKF_PUBER_SESS_EVT_SLOW_SCHED */
    },
    /* BKF_PUBER_SESS_STATE_SLOW_BATCH_DATA */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcSub)},           /* BKF_PUBER_SESS_EVT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcUnsub)},         /* BKF_PUBER_SESS_EVT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessSlowBatchProcTupleChg)},  /* BKF_PUBER_SESS_EVT_TUPLE_CHG */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},                /* BKF_PUBER_SESS_EVT_TABLE_CPLT */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessProcTableRel)},           /* BKF_PUBER_SESS_EVT_TABLE_REL */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},            /* BKF_PUBER_SESS_EVT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessSlowBatchProcSlowSched)}, /* BKF_PUBER_SESS_EVT_SLOW_SCHED */
    },
    /* BKF_PUBER_SESS_STATE_REAL_SLOW_BATCH_DATA */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcSub)},           /* BKF_PUBER_SESS_EVT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessBatchProcUnsub)},         /* BKF_PUBER_SESS_EVT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},                /* BKF_PUBER_SESS_EVT_TUPLE_CHG */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},                /* BKF_PUBER_SESS_EVT_TABLE_CPLT */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessProcTableRel)},           /* BKF_PUBER_SESS_EVT_TABLE_REL */
        {BKF_FSM_BUILD_PROC_ITEM(BkfPuberSessRealSlowBatchProcSched)}, /* BKF_PUBER_SESS_EVT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},            /* BKF_PUBER_SESS_EVT_SLOW_SCHED */
    },
};
BKF_STATIC_ASSERT(BKF_GET_ARY_COUNT(g_BkfPuberSessFsm) == BKF_PUBER_SESS_STATE_CNT);

#endif

#if BKF_BLOCK("公有函数定义")
BkfFsmTmpl *BkfPuberSessFsmTmplInit(BkfPuberSessMng *sessMng)
{
    char tempName[BKF_NAME_LEN_MAX + 1] = {0};
    int32_t err = snprintf_truncated_s(tempName, sizeof(tempName), "%s_sessFsmTmpl_%" VOS_PRIu64 "",
        sessMng->argInit->name, BKF_GET_NEXT_VAL(sessMng->argInit->xSeed));
    if (err <= 0) {
        return VOS_NULL;
    }

    BkfFsmTmplInitArg arg = {0};
    arg.name = tempName;
    arg.dbgOn = sessMng->argInit->dbgOn;
    arg.memMng = sessMng->argInit->memMng;
    arg.disp = sessMng->argInit->disp;
    arg.log = sessMng->log;
    arg.stateCnt = BKF_PUBER_SESS_STATE_CNT;
    arg.stateInit = BKF_PUBER_SESS_STATE_WAIT_SUB;
    arg.evtCnt = BKF_PUBER_SESS_EVT_CNT;
    arg.stateStrTbl = g_BkfPuberSessFsmStateStrTbl;
    arg.evtStrTbl = g_BkfPuberSessFsmEvtStrTbl;
    arg.stateEvtProcItemMtrx = (const BkfFsmProcItem *)g_BkfPuberSessFsm;
    arg.getAppDataStrOrNull = VOS_NULL;
    return BkfFsmTmplInit(&arg);
}

uint32_t BkfPuberSessFsmProc(BkfPuberSess *sess, uint8_t evtId, void *param1, void *param2)
{
    BkfPuberSessMng *sessMng = sess->sessMng;
    uint32_t ret = BKF_FSM_DISPATCH(&sess->fsm, evtId, sess, param1, param2);
    if (evtId != BKF_PUBER_SESS_EVT_TUPLE_CHG) {
        BKF_BLACKBOX_LOG(sessMng->argInit->log, BKF_BLACKBOX_TYPE_SESS, (void *)sess, 0, "E %u S %u R %u", evtId,
            sess->fsm.state, ret);
    }

    if ((ret == BKF_PUBER_SESS_FATAL_ERR) || (ret == BKF_FSM_DISPATCH_RET_IMPOSSIBLE)) {
        BKF_LOG_WARN(BKF_LOG_HND, "sess(%#x)/evtId(%u, %s), ret(%u), 2del conn\n", BKF_MASK_ADDR(sess), evtId,
            BkfFsmTmplGetEvtStr(sessMng->sessFsmTmpl, evtId), ret);
        return BKF_PUBER_SESS_FATAL_ERR;
    }

    BKF_LOG_DEBUG(BKF_LOG_HND, "sess(%#x)/evtId(%u, %s), ret(%u)\n", BKF_MASK_ADDR(sess), evtId,
        BkfFsmTmplGetEvtStr(sessMng->sessFsmTmpl, evtId), ret);
    return ret;
}

#endif

#if BKF_BLOCK("私有函数定义")
STATIC uint32_t BkfPuberSessBatchProcSub(BkfPuberSess *sess, BkfMsgDecoder *decoder, void *unused2)
{
    (void)unused2;
    BkfPuberSessBatchChkTmrStop(sess);
    BkfTlvTransNum *tlvTransNum = BkfPuberSessParseTransNum(decoder);
    if (tlvTransNum == VOS_NULL) {
        return BKF_PUBER_SESS_FATAL_ERR;
    }

    BOOL mayProc = BkfPuberSessProcSubMayProc(sess, decoder->curMsgHead, tlvTransNum, VOS_TRUE);
    if (!mayProc) {
        return BKF_ERR;
    }

    return BkfPuberSessProcSubDo(sess, decoder->curMsgHead, tlvTransNum);
}

STATIC uint32_t BkfPuberSessProcSub(BkfPuberSess *sess, BkfMsgDecoder *decoder, void *unused2)
{
    (void)unused2;
    BkfTlvTransNum *tlvTransNum = BkfPuberSessParseTransNum(decoder);
    if (tlvTransNum == VOS_NULL) {
        return BKF_PUBER_SESS_FATAL_ERR;
    }

    BOOL mayProc = BkfPuberSessProcSubMayProc(sess, decoder->curMsgHead, tlvTransNum, VOS_FALSE);
    if (!mayProc) {
        return BKF_ERR;
    }

    return BkfPuberSessProcSubDo(sess, decoder->curMsgHead, tlvTransNum);
}

STATIC uint32_t BkfPuberSessBatchProcUnsub(BkfPuberSess *sess, BkfMsgDecoder *decoder, void *unused2)
{
    (void)unused2;
    BkfPuberSessBatchChkTmrStop(sess);
    BkfTlvTransNum *tlvTransNum = BkfPuberSessParseTransNum(decoder);
    if (tlvTransNum == VOS_NULL) {
        return BKF_PUBER_SESS_FATAL_ERR;
    }

    BOOL mayProc = BkfPuberSessProcUnsubMayProc(sess, decoder->curMsgHead, tlvTransNum, VOS_TRUE);
    if (!mayProc) {
        return BKF_ERR;
    }

    return BkfPuberSessProcUnsubDo(sess, decoder->curMsgHead, tlvTransNum);
}

STATIC uint32_t BkfPuberSessProcUnsub(BkfPuberSess *sess, BkfMsgDecoder *decoder, void *unused2)
{
    (void)unused2;
    BkfTlvTransNum *tlvTransNum = BkfPuberSessParseTransNum(decoder);
    if (tlvTransNum == VOS_NULL) {
        return BKF_PUBER_SESS_FATAL_ERR;
    }

    BOOL mayProc = BkfPuberSessProcUnsubMayProc(sess, decoder->curMsgHead, tlvTransNum, VOS_FALSE);
    if (!mayProc) {
        return BKF_ERR;
    }

    return BkfPuberSessProcUnsubDo(sess, decoder->curMsgHead, tlvTransNum);
}

STATIC uint32_t BkfPuberSessTry2SendSubAck(BkfPuberSess *sess, BkfMsgCoder *coder, void *unused2)
{
    (void)unused2;
    uint32_t ret = BkfPuberSessPackSubAck(sess, coder);
    if (ret != BKF_OK) {
        return BKF_PUBER_SESS_SEND_BUF_NOT_ENOUGH;
    }

    BkfPuberSessMng *sessMng = sess->sessMng;

    BOOL needWaitTblCmpt = sess->subWithNeedTblCplt &&
                           !BkfDcIsTableTypeComplete(sessMng->argInit->dc, sess->key.tableTypeId);
    uint8_t newState = needWaitTblCmpt ? BKF_PUBER_SESS_STATE_WAIT_MAY_SEND_BATCH_BEGIN
                                       : BKF_PUBER_SESS_STATE_WAIT_SEND_BATCH_BEGIN;
    BkfPuberSessChgStateAndTrigSched(sess, newState);
    return BKF_OK;
}

STATIC uint32_t BkfPuberSessChgState2WaitSendBatchBegin(BkfPuberSess *sess, void *unused1, void *unused2)
{
    (void)unused1;
    (void)unused2;
    BkfPuberSessChgStateAndTrigSched(sess, BKF_PUBER_SESS_STATE_WAIT_SEND_BATCH_BEGIN);
    return BKF_OK;
}

STATIC uint32_t BkfPuberSessTry2SendBatchBegin(BkfPuberSess *sess, BkfMsgCoder *coder, void *unused2)
{
    (void)unused2;
    uint32_t ret = BkfPuberSessPackBatchBegin(sess, coder);
    if (ret != BKF_OK) {
        return BKF_PUBER_SESS_SEND_BUF_NOT_ENOUGH;
    }

    if (sess->subWithVerify) {
        /* 有实时变化数据，优先调度 */
        BkfPuberSessMng *sessMng = sess->sessMng;
        BkfDcTupleInfo tupleInfo = {0};
        BOOL schedReal = (sess->itorRealData != VOS_NULL) &&
                         (BkfDcGetTupleBySeqItor(sessMng->argInit->dc, sess->itorRealData, &tupleInfo) != VOS_FALSE);
        if (schedReal) {
            BkfPuberSessChgStateAndTrigSched(sess, BKF_PUBER_SESS_STATE_REAL_SLOW_BATCH_DATA);
        } else {
            BkfPuberSessChgStateAndTrigSlowSched(sess, BKF_PUBER_SESS_STATE_SLOW_BATCH_DATA);
        }
    } else {
        BkfPuberSessBatchChkTmrStart(sess);
        BkfPuberSessChgStateAndTrigSched(sess, BKF_PUBER_SESS_STATE_BATCH_DATA);
    }

    return BKF_OK;
}

STATIC uint32_t BkfPuberSessTry2SendBatchEnd(BkfPuberSess *sess, BkfMsgCoder *coder, void *unused2)
{
    (void)unused2;
    uint32_t ret = BkfPuberSessPackBatchEnd(sess, coder);
    if (ret != BKF_OK) {
        return BKF_PUBER_SESS_SEND_BUF_NOT_ENOUGH;
    }

    BkfPuberSessBatchChkTmrStop(sess);

    BkfPuberSessChgStateAndTrigSched(sess, BKF_PUBER_SESS_STATE_REAL_DATA);
    return BKF_OK;
}

STATIC uint32_t BkfPuberSessTry2SendSubDelNtf(BkfPuberSess *sess, BkfMsgCoder *coder, void *unused2)
{
    (void)unused2;
    uint32_t ret = BkfPuberSessPackSubDelNtf(sess, coder);
    return (ret == BKF_OK) ? BKF_PUBER_SESS_NEED_DELETE : BKF_PUBER_SESS_SEND_BUF_NOT_ENOUGH;
}

STATIC uint32_t BkfPuberSessIdleProcTupleChg(BkfPuberSess *sess, void *unused1, void *unused2)
{
    (void)unused1;
    (void)unused2;
    BkfPuberSessChgStateAndTrigSched(sess, BKF_PUBER_SESS_STATE_REAL_DATA);
    return BKF_OK;
}

STATIC uint32_t BkfPuberSessSlowBatchProcTupleChg(BkfPuberSess *sess, void *unused1, void *unused2)
{
    (void)unused1;
    (void)unused2;
    BkfPuberSessChgStateAndTrigSched(sess, BKF_PUBER_SESS_STATE_REAL_SLOW_BATCH_DATA);
    return BKF_OK;
}

STATIC uint32_t BkfPuberSessProcTableRel(BkfPuberSess *sess, void *unused1, void *unused2)
{
    (void)unused1;
    (void)unused2;
    BkfPuberSessMng *sessMng = sess->sessMng;
    if (sess->itorBatchData != VOS_NULL) {
        BkfDcDeleteTupleKeyItor(sessMng->argInit->dc, sess->itorBatchData);
        sess->itorBatchData = VOS_NULL;
    }

    if (sess->itorRealData != VOS_NULL) {
        BkfDcDeleteTupleSeqItor(sessMng->argInit->dc, sess->itorRealData);
        sess->itorRealData = VOS_NULL;
    }

    BkfPuberSessBatchChkTmrStop(sess);
    BkfPuberSessChgStateAndTrigSched(sess, BKF_PUBER_SESS_STATE_WAIT_SEND_DEL_SUB_NTF);
    return BKF_OK;
}

STATIC uint32_t BkfPuberSessBatchProcSched(BkfPuberSess *sess, BkfMsgCoder *coder, void *unused2)
{
    (void)unused2;
    uint32_t ret = BkfPuberSessTry2SendBatchData(sess, coder, VOS_NULL);
    if (ret == BKF_PUBER_SESS_SCHED_YIELD) {
        BKF_ASSERT(0);
        return BKF_PUBER_SESS_FATAL_ERR;
    }

    if (ret != BKF_OK) {
        return ret;
    }

    BkfPuberSessChgStateAndTrigSched(sess, BKF_PUBER_SESS_STATE_WAIT_SEND_BATCH_END);
    return BKF_OK;
}

STATIC uint32_t BkfPuberSessRealProcSched(BkfPuberSess *sess, BkfMsgCoder *coder, void *unused2)
{
    (void)unused2;
    uint32_t ret = BkfPuberSessTry2SendRealData(sess, coder);
    if (ret != BKF_OK) {
        return ret;
    }

    BkfPuberSessChgStateAndTrigSched(sess, BKF_PUBER_SESS_STATE_IDLE);
    return BKF_OK;
}

STATIC uint32_t BkfPuberSessRealSlowBatchProcSched(BkfPuberSess *sess, BkfMsgCoder *coder, void *unused2)
{
    (void)unused2;
    uint32_t ret = BkfPuberSessTry2SendRealData(sess, coder);
    if (ret != BKF_OK) {
        return ret;
    }

    BkfPuberSessChgStateAndTrigSlowSched(sess, BKF_PUBER_SESS_STATE_SLOW_BATCH_DATA);
    return BKF_OK;
}

STATIC uint32_t BkfPuberSessSlowBatchProcSlowSched(BkfPuberSess *sess, BkfMsgCoder *coder,
    BkfPuberSessSlowSchedCtx *ctx)
{
    uint32_t ret = BkfPuberSessTry2SendBatchData(sess, coder, ctx);
    if (ret != BKF_OK) {
        return ret;
    }

    BkfPuberSessChgStateAndTrigSched(sess, BKF_PUBER_SESS_STATE_WAIT_SEND_BATCH_END);
    return BKF_OK;
}

/*
1) 先前初始状态，启动流程
2) 否则，非batch状态, 事务号不一样, 启动流程
3) 否则，batch 状态：
    3-1) 事务号不一样, 订阅 中断 订阅和对账
    3-2) 无论事物号是否一样，对账 不中断 订阅
    3-3) 事务号不一样, 对账 中断 对账
*/
STATIC BOOL BkfPuberSessProcSubMayProc(BkfPuberSess *sess, BkfMsgHead *msgHead, BkfTlvTransNum *tlvTransNum,
    BOOL inBatchState)
{
    BOOL isInit = (sess->subTransNum == BKF_TRANS_NUM_INVALID);
    BOOL transNumIsDiff = (sess->subTransNum != tlvTransNum->num);
    BOOL curIsVerify = BKF_BIT_TEST(msgHead->flag, BKF_FLAG_VERIFY);
    BOOL mayProc = isInit || (!inBatchState && transNumIsDiff) || (inBatchState && transNumIsDiff && !curIsVerify) ||
                   (inBatchState && transNumIsDiff && curIsVerify && sess->subWithVerify);
    if (mayProc) {
        return VOS_TRUE;
    }

    BkfPuberSessMng *sessMng = sess->sessMng;
    BKF_LOG_DEBUG(BKF_LOG_HND,
        "sess(%#x), inBatchState(%u)/isInit(%u)/transNumIsDiff(%u)/"
        "curIsVerify(%u)/subWithVerify(%u), can not proc & ret\n",
        BKF_MASK_ADDR(sess), inBatchState, isInit, transNumIsDiff, curIsVerify, sess->subWithVerify);
    return VOS_FALSE;
}

STATIC uint32_t BkfPuberSessProcSubDo(BkfPuberSess *sess, BkfMsgHead *msgHead, BkfTlvTransNum *tlvTransNum)
{
    sess->subTransNum = tlvTransNum->num;
    sess->subWithVerify = BKF_COND_2BIT_FIELD(BKF_BIT_TEST(msgHead->flag, BKF_FLAG_VERIFY));
    sess->subWithNeedTblCplt = BKF_COND_2BIT_FIELD(BKF_BIT_TEST(msgHead->flag, BKF_FLAG_NEED_TABLE_COMPLETE));

    /* 重置itor */
    BkfPuberSessMng *sessMng = sess->sessMng;
    if (sess->itorBatchData != VOS_NULL) {
        BkfDcDeleteTupleKeyItor(sessMng->argInit->dc, sess->itorBatchData);
    }

    sess->itorBatchData = BkfDcNewTupleKeyItor(sessMng->argInit->dc, sess->key.sliceKey, sess->key.tableTypeId);
    if (sess->itorBatchData == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_PUBER_SESS_FATAL_ERR;
    }

    BkfPuberSessChgStateAndTrigSched(sess, BKF_PUBER_SESS_STATE_WAIT_SEND_SUB_ACK);
    return BKF_OK;
}

/*
1) 非batch状态, 事务号一样, 启动流程
2) 否则，batch状态
    2-1) 事务号一样, 去订阅 中断 订阅和对账
    2-2) 无论事物号是否一样, 去对账 不能中断 订阅
    2-3) 事务号一样, 去对账 中断 对账
*/
STATIC BOOL BkfPuberSessProcUnsubMayProc(BkfPuberSess *sess, BkfMsgHead *msgHead, BkfTlvTransNum *tlvTransNum,
    BOOL inBatchState)
{
    BOOL transNumIsSame = (sess->subTransNum == tlvTransNum->num);
    BOOL curIsVerify = BKF_BIT_TEST(msgHead->flag, BKF_FLAG_VERIFY);
    BOOL mayProc = (!inBatchState && transNumIsSame) || (inBatchState && transNumIsSame && !curIsVerify) ||
                   (inBatchState && transNumIsSame && curIsVerify && sess->subWithVerify);
    if (mayProc) {
        return VOS_TRUE;
    }

    BkfPuberSessMng *sessMng = sess->sessMng;
    BKF_LOG_DEBUG(BKF_LOG_HND,
        "sess(%#x), inBatchState(%u)/transNumIsSame(%u)/curIsVerify(%u)/"
        "subWithVerify(%u), can not proc & ret\n",
        BKF_MASK_ADDR(sess), inBatchState, transNumIsSame, curIsVerify, sess->subWithVerify);
    return VOS_FALSE;
}

STATIC uint32_t BkfPuberSessProcUnsubDo(BkfPuberSess *sess, BkfMsgHead *msgHead, BkfTlvTransNum *tlvTransNum)
{
    (void)tlvTransNum;
    sess->subWithVerify = BKF_COND_2BIT_FIELD(BKF_BIT_TEST(msgHead->flag, BKF_FLAG_VERIFY));
    if (sess->subWithVerify) {
        BkfPuberSessChgStateAndTrigSched(sess, BKF_PUBER_SESS_STATE_REAL_DATA);
        return BKF_OK;
    }

    return BKF_PUBER_SESS_NEED_DELETE;
}

STATIC uint32_t BkfPuberSessTry2SendOneTuple(BkfPuberSess *sess, BkfDcTupleInfo *tupleInfo, BkfMsgCoder *coder)
{
    uint32_t ret = BkfMsgCodeTLV(coder, BKF_TLV_TUPLE_IDL_DATA, 0, &tupleInfo->seq, VOS_FALSE);
    if (ret != BKF_OK) {
        return ret;
    }

    int32_t leftBufLen;
    uint8_t *leftBuf = BkfMsgCodeGetLeft(coder, &leftBufLen);
    if (leftBuf == VOS_NULL) {
        return BKF_PUBER_CODE_BUF_NOT_ENOUGH;
    }

    BkfPuberSessMng *sessMng = sess->sessMng;
    int32_t codeLen = BkfPuberSessNtfAppCode(sess, tupleInfo, leftBuf, leftBufLen);
    if (codeLen == BKF_PUBER_CODE_BUF_NOT_ENOUGH) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "codeLen(%d), not enough\n", codeLen);
        return BKF_PUBER_CODE_BUF_NOT_ENOUGH;
    }

    if ((codeLen < 0) || (codeLen > leftBufLen)) {
        BKF_LOG_ERROR(BKF_LOG_HND, "codeLen(%d), ng\n", codeLen);
        return BKF_PUBER_SESS_FATAL_ERR;
    }

    uint8_t flag = tupleInfo->isAddUpd ? BKF_FLAG_TUPLE_UPDATE : 0;
    return BkfMsgCodeAppendValLen(coder, flag, codeLen, VOS_TRUE);
}

static inline uint32_t BkfPuberSessPackDataHeadIfNeed(BkfPuberSess *sess, BkfMsgCoder *coder, BOOL *msgHeadHasAdd)
{
    if (*msgHeadHasAdd) {
        return BKF_OK;
    }

    *msgHeadHasAdd = VOS_TRUE;
    return BkfPuberSessPackDataHead(sess, coder);
}

static inline uint32_t BkfPuberSessConvertSendOneTupleRet(uint32_t sndRet)
{
    if (sndRet == BKF_PUBER_SESS_FATAL_ERR) {
        return sndRet;
    }

    if (sndRet != BKF_OK) {
        return BKF_PUBER_SESS_SEND_BUF_NOT_ENOUGH;
    }

    return BKF_OK;
}

STATIC uint32_t BkfPuberSessTry2SendBatchData(BkfPuberSess *sess, BkfMsgCoder *coder, BkfPuberSessSlowSchedCtx *ctx)
{
    BkfPuberSessMng *sessMng = sess->sessMng;
    BOOL msgHeadHasAdd = VOS_FALSE;
    BkfDcTupleInfo tupleInfo = {0};

    while (BkfDcGetTupleByKeyItorFromDcOrApp(sessMng->argInit->dc, sess->itorBatchData, &tupleInfo)) {
        uint32_t ret = BkfPuberSessPackDataHeadIfNeed(sess, coder, &msgHeadHasAdd);
        if (ret != BKF_OK) {
            return BKF_PUBER_SESS_SEND_BUF_NOT_ENOUGH;
        }

        ret = BkfPuberSessTry2SendOneTuple(sess, &tupleInfo, coder);
        ret = BkfPuberSessConvertSendOneTupleRet(ret);
        if (ret != BKF_OK) {
            return ret;
        }

        BkfDcForwordTupleKeyItor(sessMng->argInit->dc, sess->itorBatchData);
        if (ctx != VOS_NULL) {
            ctx->procCnt++;
            if (ctx->procCnt >= ctx->procCntMax) {
                return BKF_PUBER_SESS_SCHED_YIELD;
            }
        }
    }

    return BKF_OK;
}

STATIC uint32_t BkfPuberSessTry2SendRealData(BkfPuberSess *sess, BkfMsgCoder *coder)
{
    BkfPuberSessMng *sessMng = sess->sessMng;
    BOOL msgHeadHasAdd = VOS_FALSE;
    BkfDcTupleInfo tupleInfo = {0};

    while (BkfDcGetTupleBySeqItor(sessMng->argInit->dc, sess->itorRealData, &tupleInfo)) {
        uint32_t ret = BkfPuberSessPackDataHeadIfNeed(sess, coder, &msgHeadHasAdd);
        if (ret != BKF_OK) {
            return BKF_PUBER_SESS_SEND_BUF_NOT_ENOUGH;
        }

        ret = BkfPuberSessTry2SendOneTuple(sess, &tupleInfo, coder);
        ret = BkfPuberSessConvertSendOneTupleRet(ret);
        if (ret != BKF_OK) {
            return ret;
        }

        BkfDcForwordTupleSeqItor(sessMng->argInit->dc, sess->itorRealData);
    }

    return BKF_OK;
}

/* 平滑超时检查 */
STATIC uint32_t BkfPuberSessBatchChkTimeout(BkfPuberSess *sess, void *noUse)
{
    (void)noUse;
    if (sess->batchChkTmr == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_OK;
    }
    sess->batchChkTmr = VOS_NULL;

    /* 仅平滑超时的时候通知, 不包括对账 */
    if (sess->subWithVerify) {
        return BKF_OK;
    }
    uint8_t state = BKF_FSM_GET_STATE(&sess->fsm);
    if (state == BKF_PUBER_SESS_STATE_WAIT_SEND_BATCH_END || state == BKF_PUBER_SESS_STATE_BATCH_DATA) {
        BkfPuberSessNtfAppBatchTimeout(sess->sessMng, sess->key.sliceKey, sess->key.tableTypeId);
    }
    return BKF_OK;
}

void BkfPuberSessBatchChkTmrStart(BkfPuberSess *sess)
{
    BkfPuberSessMng *sessMng = sess->sessMng;

    if (sessMng->batchTimeoutChkFlag == VOS_TRUE) {
        BkfPuberSessBatchChkTmrStop(sess);
        return;
    }

    BkfPuberTableTypeVTbl *vTbl = BkfPuberTableTypeGetVTbl(sessMng->tableTypeMng, sess->key.tableTypeId);
    if (vTbl == VOS_NULL) {
        BKF_ASSERT(0);
        return;
    }

    if (vTbl->batchTimeout == 0) {
        return;
    }

    if (sess->batchChkTmr != VOS_NULL) {
        BkfTmrStop(sessMng->argInit->tmrMng, sess->batchChkTmr);
    }

    sess->batchChkTmr = BkfTmrStartOnce(sessMng->argInit->tmrMng, (F_BKF_TMR_TIMEOUT_PROC)BkfPuberSessBatchChkTimeout,
        vTbl->batchTimeout * BKF_MS_PER_S, (void *)sess);
}

void BkfPuberSessBatchChkTmrStop(BkfPuberSess *sess)
{
    if (sess->batchChkTmr == VOS_NULL) {
        return;
    }

    BkfPuberSessMng *sessMng = sess->sessMng;
    BkfTmrStop(sessMng->argInit->tmrMng, sess->batchChkTmr);
    sess->batchChkTmr = VOS_NULL;
}
#endif

#ifdef __cplusplus
}
#endif
