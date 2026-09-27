/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_SUBSESS_PKT_DISP_LEN (512)
#define BKF_SUBSESS_PKT_DISP_RESV_LEN (8)

#include "bkf_suber_sess_fsm.h"
#include "bkf_suber_sess.h"
#include "bkf_suber_sess_data.h"
#include "bkf_msg.h"
#include "bkf_msg_codec.h"
#include "bkf_subscriber.h"
#include "bkf_suber_data.h"
#include "securec.h"
#include "bkf_comm.h"
#include "bkf_fsm.h"
#include "vos_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
STATIC uint32_t BkfSubSessProcNtf(BkfSuberSess *sess, BkfMsgDecoder *decoder);
STATIC uint32_t BkfSubSessActRcvNtf(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *parm);
STATIC uint32_t BkfSubSessActRcvNtfNotReSub(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *parm);
STATIC uint32_t BkfSubSessProcUnSub(BkfSuberSess *sess, void *nouse1, void *nouse2);
STATIC uint32_t BkfSubSessProcSub(BkfSuberSess *sess, void *nouse1, void *nouse2);
STATIC uint32_t BkfSubSessProcVerifySub(BkfSuberSess *sess, void *nouse1, void *nouse2);
void BkfSubSessChangeStateAndTirg(BkfSuberSess *sess, uint8_t state);
STATIC uint32_t BkfSubSessActStartSub(BkfSuberSess *sess, BkfMsgCoder *coder, void *param);
STATIC uint32_t BkfSubSessActRcvSubAck(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param);
STATIC void BkfSubSessProcDisconn(BkfSuberSess *sess);
STATIC uint32_t BkfSubSessActDisconnNotReSub(BkfSuberSess *sess, BkfMsgDecoder *noUse, void *param);
STATIC uint32_t BkfSubSessActDisconn(BkfSuberSess *sess, BkfMsgDecoder *noUse, void *param);
STATIC uint32_t BkfSubSessActRcvBatchBegin(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param);
STATIC uint32_t BkfSubSessActRcvBatchEnd(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param);
STATIC uint32_t BkfSubSessSndUnSub(BkfSuberSess *sess, BkfMsgCoder *coder, void *param);
STATIC uint32_t BkfSubSessActRcvData(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param);
STATIC uint32_t BkfSubSessActStartVerifySub(BkfSuberSess *sess, BkfMsgCoder *coder, void *param);
STATIC uint32_t BkfSubSessActRcvVerifyAck(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param);
STATIC uint32_t BkfSubSessActRcvVerifyBegin(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param);
STATIC uint32_t BkfSubSessActRcvVerifyEnd(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param);
void BkfSuberSessBatchChkTmrStart(BkfSuberSess *sess);
#pragma pack()
#endif

#if BKF_BLOCK("表")
const char *subSessFsmStr = "SubscribeSessionFsm";

const char *subSessStateStrTbl[] = {
    "BACKFIN_SUB_SESS_STATE_DOWN",
    "BACKFIN_SUB_SESS_STATE_WAITSUBACK",
    "BACKFIN_SUB_SESS_STATE_UP",
    "BACKFIN_SUB_SESS_STATE_SNDUNSUB",
    "BACKFIN_SUB_SESS_STATE_BATCH",
    "BACKFIN_SUB_SESS_STATE_BATOK",
    "BACKFIN_SUB_SESS_STATE_VERIFYWAITACK",
    "BACKFIN_SUB_SESS_STATE_VERIFYRDY",
    "BACKFIN_SUB_SESS_STATE_VERIFY",
};

const char *subSessEvtStrTbl[] = {
    "BACKFIN_SUB_SESS_INPUT_SUB",
    "BACKFIN_SUB_SESS_INPUT_SUBACK",
    "BACKFIN_SUB_SESS_INPUT_UNSUB",
    "BACKFIN_SUB_SESS_INPUT_BATCHBEGIN",
    "BACKFIN_SUB_SESS_INPUT_BATCHEND",
    "BACKFIN_SUB_SESS_INPUT_DATA",
    "BACKFIN_SUB_SESS_INPUT_NTF",
    "BACKFIN_SUB_SESS_INPUT_DISCONN",
    "BACKFIN_SUB_SESS_INPUT_SCHED",
    "BACKFIN_SUB_SESS_INPUT_VERIFYSUB",
    "BACKFIN_SUB_SESS_INPUT_VERIFYSUBACK",
    "BACKFIN_SUB_SESS_INPUT_VERIFYBEGIN",
    "BACKFIN_SUB_SESS_INPUT_VERIFYEND",
};

/*

Y    BACKFIN_SUB_SESS_ACTION_NULL            空操作
Z    BACKFIN_SUB_SESS_ACTION_ASSERT          异常，不应该进入的状态
A    BACKFIN_SUB_SESS_ACTION_SUB             启动发送订阅消息，发送成功：WAITSUBACK/发送失败：DOWN
B    BACKFIN_SUB_SESS_ACTION_RDY_SUB         更新序列号，准备发送sub消息,job驱动conn发包
C    BACKFIN_SUB_SESS_ACTION_RDY_UNSUB       准备发送unSub消息,job驱动conn发包
D    BACKFIN_SUB_SESS_ACTION_UNSUB           发送去订阅消息，成功则：非对账释放E，对账切换5。不成功则等待
E    BACKFIN_SUB_SESS_ACTION_DELETE          资源释放
F    BACKFIN_SUB_SESS_ACTION_BATCHBEGIN      批备开始
G    BACKFIN_SUB_SESS_ACTION_BATCHEND        批备结束
H    BACKFIN_SUB_SESS_ACTION_RCVDATA         收到批备实时数据
I    BACKFIN_SUB_SESS_ACTION_NTF             通知app表删除，更新序列号，准备发送sub消息，job驱动conn发包
J    BACKFIN_SUB_SESS_ACTION_DISCONN         连接中断，通告app，更新序列号，准备发送sub消息，job驱动conn发包
K    BACKFIN_SUB_SESS_ACTION_VERIFYSUB       发送对账消息，发送成功：WAITVERIFYACK/发送失败：BATCH_OK
L    BACKFIN_SUB_SESS_ACTION_RDY_SNDVERIFY   准备发送对账消息,job驱动conn发包
M    BACKFIN_SUB_SESS_ACTION_VERIFYBEGIN     对账开始
N    BACKFIN_SUB_SESS_ACTION_VERIFYEND       对账结束
O    BACKFIN_SUB_SESS_ACTION_RDY_VERIFY      准备开始对账

-----------------------------------------------------------------------------------------------------------------
                                           DOWN  WSUBACK   UP   SNDUNSUB BATCH  BATCH_OK  WVACK  VERRDY   VERIFY
-----------------------------------------------------------------------------------------------------------------
Inputs                           States |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |
-----------------------------------------------------------------------------------------------------------------
00 BACKFIN_SUB_SESS_INPUT_SUB           |  1 B  |  0 B  |  0 B  |  0 B  |  0 B  |  0 B  |  0 B  |  0 B  |  0 B  |
01 BACKFIN_SUB_SESS_INPUT_SUBACK        |  - Y  |  2 C  |  - Y  |  - Y  |  - Y  |  - Y  |  - Z  |  - Z  |  - Z  |
02 BACKFIN_SUB_SESS_INPUT_UNSUB         |  - E  |  3 C  |  3 C  |  - Y  |  3 C  |  3 C  |  3 C  |  3 C  |  3 C  |
03 BACKFIN_SUB_SESS_INPUT_BATCHBEGIN    |  - Y  |  - Y  |  4 F  |  - Y  |  - Y  |  4 F  |  - Z  |  - Z  |  - Z  |
04 BACKFIN_SUB_SESS_INPUT_BATCHEND      |  - Y  |  - Y  |  - Y  |  - Y  |  5 G  |  - Y  |  - Z  |  - Z  |  - Z  |
05 BACKFIN_SUB_SESS_INPUT_DATA          |  - Y  |  - Y  |  - Y  |  - Y  |  4 H  |  - H  |  - H  |  - H  |  - H  |
06 BACKFIN_SUB_SESS_INPUT_NTF           |  - Y  |  0 I  |  0 I  |  0 I  |  0 I  |  0 I  |  0 I  |  0 I  |  0 I  |
07 BACKFIN_SUB_SESS_INPUT_DISCONN       |  - Y  |  0 J  |  0 J  |  0 J  |  0 J  |  0 J  |  0 J  |  0 J  |  0 J  |
08 BACKFIN_SUB_SESS_INPUT_SCHED         |  1 A  |  - Y  |  - Z  |  - D  |  - Z  |  6 K  |  - Z  |  - Y  |  - Y  |
09 BACKFIN_SUB_SESS_INPUT_VERIFYSUB     |  - Y  |  - Y  |  0 B  |  - Y  |  - Y  |  - L  |  5 L  |  5 L  |  5 L  |
10 BACKFIN_SUB_SESS_INPUT_VERIFYSUBACK  |  - Y  |  - Z  |  0 B  |  - Y  |  - Y  |  - Z  |  6 O  |  - Y  |  - Y  |
11 BACKFIN_SUB_SESS_INPUT_VERIFYBEGIN   |  - Y  |  - Y  |  0 B  |  - Y  |  - Y  |  - Z  |  - Z  |  8 M  |  - Y  |
12 BACKFIN_SUB_SESS_INPUT_VERIFYEND     |  - Y  |  - Y  |  0 B  |  - Y  |  - Y  |  - Z  |  - Z  |  - Y  |  5 N  |
-----------------------------------------------------------------------------------------------------------------

*/

BkfFsmProcItem subSessStateEvtProcItemMtrx[BACKFIN_SUB_SESS_STATE_MAX][BACKFIN_SUB_SESS_INPUT_MAX] = {
    /* BACKFIN_SUB_SESS_STATE_DOWN */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcSub)},     /* BACKFIN_SUB_SESS_INPUT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},   /* BACKFIN_SUB_SESS_INPUT_SUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcUnSub)},   /* BACKFIN_SUB_SESS_INPUT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},   /* BACKFIN_SUB_SESS_INPUT_BATCHBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},   /* BACKFIN_SUB_SESS_INPUT_BATCHEND */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},   /* BACKFIN_SUB_SESS_INPUT_DATA */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},   /* BACKFIN_SUB_SESS_INPUT_NTF */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},   /* BACKFIN_SUB_SESS_INPUT_DISCONN */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActStartSub)}, /* BACKFIN_SUB_SESS_INPUT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},   /* BACKFIN_SUB_SESS_INPUT_VERIFYSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},   /* BACKFIN_SUB_SESS_INPUT_VERIFYSUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},   /* BACKFIN_SUB_SESS_INPUT_VERIFYBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},   /* BACKFIN_SUB_SESS_INPUT_VERIFYEND */
    },
    /* BACKFIN_SUB_SESS_STATE_WAITSUBACK */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcSub)},       /* BACKFIN_SUB_SESS_INPUT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvSubAck)},  /* BACKFIN_SUB_SESS_INPUT_SUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcUnSub)},     /* BACKFIN_SUB_SESS_INPUT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},     /* BACKFIN_SUB_SESS_INPUT_BATCHBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},     /* BACKFIN_SUB_SESS_INPUT_BATCHEND */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},     /* BACKFIN_SUB_SESS_INPUT_DATA */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvNtf)},     /* BACKFIN_SUB_SESS_INPUT_NTF */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActDisconn)},    /* BACKFIN_SUB_SESS_INPUT_DISCONN */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActStartSub)},   /* BACKFIN_SUB_SESS_INPUT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},     /* BACKFIN_SUB_SESS_INPUT_VERIFYSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)}, /* BACKFIN_SUB_SESS_INPUT_VERIFYSUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},     /* BACKFIN_SUB_SESS_INPUT_VERIFYBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},     /* BACKFIN_SUB_SESS_INPUT_VERIFYEND */
    },
    /* BACKFIN_SUB_SESS_STATE_UP */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcSub)},          /* BACKFIN_SUB_SESS_INPUT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},        /* BACKFIN_SUB_SESS_INPUT_SUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcUnSub)},        /* BACKFIN_SUB_SESS_INPUT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvBatchBegin)}, /* BACKFIN_SUB_SESS_INPUT_BATCHBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},        /* BACKFIN_SUB_SESS_INPUT_BATCHEND */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},        /* BACKFIN_SUB_SESS_INPUT_DATA */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvNtf)},        /* BACKFIN_SUB_SESS_INPUT_NTF */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActDisconn)},       /* BACKFIN_SUB_SESS_INPUT_DISCONN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},    /* BACKFIN_SUB_SESS_INPUT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},        /* BACKFIN_SUB_SESS_INPUT_VERIFYSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},        /* BACKFIN_SUB_SESS_INPUT_VERIFYSUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},        /* BACKFIN_SUB_SESS_INPUT_VERIFYBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},        /* BACKFIN_SUB_SESS_INPUT_VERIFYEND */
    },
    /* BACKFIN_SUB_SESS_STATE_SNDUNSUB */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcSub)},            /* BACKFIN_SUB_SESS_INPUT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},          /* BACKFIN_SUB_SESS_INPUT_SUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},          /* BACKFIN_SUB_SESS_INPUT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},          /* BACKFIN_SUB_SESS_INPUT_BATCHBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},          /* BACKFIN_SUB_SESS_INPUT_BATCHEND */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},          /* BACKFIN_SUB_SESS_INPUT_DATA */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvNtfNotReSub)},  /* BACKFIN_SUB_SESS_INPUT_NTF */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActDisconnNotReSub)}, /* BACKFIN_SUB_SESS_INPUT_DISCONN */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessSndUnSub)},           /* BACKFIN_SUB_SESS_INPUT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},          /* BACKFIN_SUB_SESS_INPUT_VERIFYSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},          /* BACKFIN_SUB_SESS_INPUT_VERIFYSUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},          /* BACKFIN_SUB_SESS_INPUT_VERIFYBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},          /* BACKFIN_SUB_SESS_INPUT_VERIFYEND */
    },
    /* BACKFIN_SUB_SESS_STATE_BATCH */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcSub)},        /* BACKFIN_SUB_SESS_INPUT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},      /* BACKFIN_SUB_SESS_INPUT_SUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcUnSub)},      /* BACKFIN_SUB_SESS_INPUT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},      /* BACKFIN_SUB_SESS_INPUT_BATCHBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvBatchEnd)}, /* BACKFIN_SUB_SESS_INPUT_BATCHEND */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvData)},     /* BACKFIN_SUB_SESS_INPUT_DATA */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvNtf)},      /* BACKFIN_SUB_SESS_INPUT_NTF */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActDisconn)},     /* BACKFIN_SUB_SESS_INPUT_DISCONN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},  /* BACKFIN_SUB_SESS_INPUT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},      /* BACKFIN_SUB_SESS_INPUT_VERIFYSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},      /* BACKFIN_SUB_SESS_INPUT_VERIFYSUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},      /* BACKFIN_SUB_SESS_INPUT_VERIFYBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},      /* BACKFIN_SUB_SESS_INPUT_VERIFYEND */
    },
    /* BACKFIN_SUB_SESS_STATE_BATOK */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcSub)},           /* BACKFIN_SUB_SESS_INPUT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},         /* BACKFIN_SUB_SESS_INPUT_SUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcUnSub)},         /* BACKFIN_SUB_SESS_INPUT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvBatchBegin)},  /* BACKFIN_SUB_SESS_INPUT_BATCHBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},         /* BACKFIN_SUB_SESS_INPUT_BATCHEND */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvData)},        /* BACKFIN_SUB_SESS_INPUT_DATA */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvNtf)},         /* BACKFIN_SUB_SESS_INPUT_NTF */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActDisconn)},        /* BACKFIN_SUB_SESS_INPUT_DISCONN */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActStartVerifySub)}, /* BACKFIN_SUB_SESS_INPUT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcVerifySub)},     /* BACKFIN_SUB_SESS_INPUT_VERIFYSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},     /* BACKFIN_SUB_SESS_INPUT_VERIFYSUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},     /* BACKFIN_SUB_SESS_INPUT_VERIFYBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},     /* BACKFIN_SUB_SESS_INPUT_VERIFYEND */
    },
    /* BACKFIN_SUB_SESS_STATE_VERIFYWAITACK */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcSub)},         /* BACKFIN_SUB_SESS_INPUT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},   /* BACKFIN_SUB_SESS_INPUT_SUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcUnSub)},       /* BACKFIN_SUB_SESS_INPUT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},   /* BACKFIN_SUB_SESS_INPUT_BATCHBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},   /* BACKFIN_SUB_SESS_INPUT_BATCHEND */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvData)},      /* BACKFIN_SUB_SESS_INPUT_DATA */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvNtf)},       /* BACKFIN_SUB_SESS_INPUT_NTF */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActDisconn)},      /* BACKFIN_SUB_SESS_INPUT_DISCONN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},   /* BACKFIN_SUB_SESS_INPUT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcVerifySub)},   /* BACKFIN_SUB_SESS_INPUT_VERIFYSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvVerifyAck)}, /* BACKFIN_SUB_SESS_INPUT_VERIFYSUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},   /* BACKFIN_SUB_SESS_INPUT_VERIFYBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},   /* BACKFIN_SUB_SESS_INPUT_VERIFYEND */
    },
    /* BACKFIN_SUB_SESS_STATE_VERIFYRDY */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcSub)},           /* BACKFIN_SUB_SESS_INPUT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},     /* BACKFIN_SUB_SESS_INPUT_SUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcUnSub)},         /* BACKFIN_SUB_SESS_INPUT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},     /* BACKFIN_SUB_SESS_INPUT_BATCHBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},     /* BACKFIN_SUB_SESS_INPUT_BATCHEND */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvData)},        /* BACKFIN_SUB_SESS_INPUT_DATA */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvNtf)},         /* BACKFIN_SUB_SESS_INPUT_NTF */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActDisconn)},        /* BACKFIN_SUB_SESS_INPUT_DISCONN */
        {BKF_FSM_BUILD_PROC_ITEM((BKF_FSM_PROC_IGNORE))},       /* BACKFIN_SUB_SESS_INPUT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcVerifySub)},     /* BACKFIN_SUB_SESS_INPUT_VERIFYSUB */
        {BKF_FSM_BUILD_PROC_ITEM((BKF_FSM_PROC_IGNORE))},       /* BACKFIN_SUB_SESS_INPUT_VERIFYSUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvVerifyBegin)}, /* BACKFIN_SUB_SESS_INPUT_VERIFYBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},         /* BACKFIN_SUB_SESS_INPUT_VERIFYEND */
    },
    /* BACKFIN_SUB_SESS_STATE_VERIFY */
    {
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcSub)},         /* BACKFIN_SUB_SESS_INPUT_SUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},   /* BACKFIN_SUB_SESS_INPUT_SUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcUnSub)},       /* BACKFIN_SUB_SESS_INPUT_UNSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},   /* BACKFIN_SUB_SESS_INPUT_BATCHBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},   /* BACKFIN_SUB_SESS_INPUT_BATCHEND */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvData)},      /* BACKFIN_SUB_SESS_INPUT_DATA */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvNtf)},       /* BACKFIN_SUB_SESS_INPUT_NTF */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActDisconn)},      /* BACKFIN_SUB_SESS_INPUT_DISCONN */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IMPOSSIBLE)},   /* BACKFIN_SUB_SESS_INPUT_SCHED */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessProcVerifySub)},   /* BACKFIN_SUB_SESS_INPUT_VERIFYSUB */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},       /* BACKFIN_SUB_SESS_INPUT_VERIFYSUBACK */
        {BKF_FSM_BUILD_PROC_ITEM(BKF_FSM_PROC_IGNORE)},       /* BACKFIN_SUB_SESS_INPUT_VERIFYBEGIN */
        {BKF_FSM_BUILD_PROC_ITEM(BkfSubSessActRcvVerifyEnd)}, /* BACKFIN_SUB_SESS_INPUT_VERIFYEND */
    },
};
#endif

#if BKF_BLOCK("公有函数定义")
uint32_t BkfSuberSessFsmInputEvt(BkfSuberSess *sess, uint32_t event, void *param1OrNull, void *param2OrNull)
{
    return BKF_FSM_DISPATCH(&sess->fsm, event, sess, param1OrNull, param2OrNull);
}
#endif

void BkfSubSessChangeStateAndTirg(BkfSuberSess *sess, uint8_t state)
{
    (void)BKF_FSM_CHG_STATE(&sess->fsm, state);
    sess->sessMng->tirgSchedSelf(sess->sessMng->cookie);
}

STATIC uint32_t BkfSubSessProcUnSub(BkfSuberSess *sess, void *nouse1, void *nouse2)
{
    (void)nouse1;
    (void)nouse2;
    if (BKF_DL_NODE_IS_IN(&sess->dlStateNode)) {
        BKF_DL_REMOVE(&sess->dlStateNode);
    }
    BKF_DL_ADD_LAST(&sess->sessMng->unSubSess, &sess->dlStateNode);
    BkfSubSessChangeStateAndTirg(sess, BACKFIN_SUB_SESS_STATE_SNDUNSUB);
    if (BKF_DL_IS_EMPTY(&sess->sessMng->subSess)) {
        BkfSuberSessDeleteReSubTmr(sess->sessMng);
    }
    BkfSuberSessBatchChkTmrStop(sess);
    return BKF_OK;
}

STATIC uint32_t BkfSubSessProcSub(BkfSuberSess *sess, void *nouse1, void *nouse2)
{
    (void)nouse1;
    (void)nouse2;
    BkfSuberEnv *env = sess->sessMng->env;

    sess->seq = BKF_GET_NEXT_VAL(env->seeds);
    sess->isVerify = 0;
    if (BKF_DL_NODE_IS_IN(&sess->dlStateNode)) {
        BKF_DL_REMOVE(&sess->dlStateNode);
    }
    BKF_DL_ADD_LAST(&sess->sessMng->subSess, &sess->dlStateNode);
    BkfSubSessChangeStateAndTirg(sess, BACKFIN_SUB_SESS_STATE_DOWN);
    BkfSuberSessCreateReSubTmr(sess->sessMng);
    BkfSuberSessBatchChkTmrStop(sess);
    return BKF_OK;
}

STATIC uint32_t BkfSubSessProcVerifySub(BkfSuberSess *sess, void *nouse1, void *nouse2)
{
    (void)nouse1;
    (void)nouse2;
    BkfSuberEnv *env = sess->sessMng->env;

    sess->seq = BKF_GET_NEXT_VAL(env->seeds);
    if (BKF_DL_NODE_IS_IN(&sess->dlStateNode)) {
        BKF_DL_REMOVE(&sess->dlStateNode);
    }
    BKF_DL_ADD_LAST(&sess->sessMng->verifySubSess, &sess->dlStateNode);
    BkfSubSessChangeStateAndTirg(sess, BACKFIN_SUB_SESS_STATE_BATOK);
    return BKF_OK;
}

uint32_t BkfSubSessSndSub(BkfSuberSess *sess, BkfMsgCoder *coder, void *param)
{
    (void)param;
    BkfSuberTableTypeVTbl *vtbl = BkfSuberDataTableTypeGetVtbl(sess->sessMng->dataMng, sess->key.tableTypeId);
    if (vtbl == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    uint8_t flag = 0;
    if (sess->isVerify) {
        BKF_BIT_SET(flag, BKF_FLAG_VERIFY);
    }

    if (vtbl->needTableComplete) {
        BKF_BIT_SET(flag, BKF_FLAG_NEED_TABLE_COMPLETE);
    }

    uint32_t ret = BkfMsgCodeMsgHead(coder, BKF_MSG_SUB, flag);
    ret |= BkfMsgCodeRawTLV(coder, BKF_TLV_SLICE_KEY, 0, sess->key.sliceKey, sess->sessMng->env->sliceVTbl.keyLen,
        VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_TABLE_TYPE, 0, &sess->key.tableTypeId, VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_TRANS_NUM, 0, &sess->seq, VOS_TRUE);

    uint8_t buf[BKF_SUBER_DISP_SLICELEN] = {0};
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint8_t urlStr2[BKF_URL_STR_LEN_MAX];
    BKF_LOG_WARN(sess->sessMng->log,
        "Snd subSess(%#x) SubMsg tableType:%u, seq:%" VOS_PRIu64 ", %s, ret:%u.p %s l %s\n", BKF_MASK_ADDR(sess),
        sess->key.tableTypeId, sess->seq,
        BkfSuberGetSliceKeyStr(sess->sessMng->env, sess->key.sliceKey, buf, sizeof(buf)), ret,
        BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
        BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));
    if (ret != BKF_OK) {
        return BKF_SUBER_SESS_SEND_BUF_NOT_ENOUGH;
    }

    return BKF_OK;
}

STATIC uint32_t BkfSubSessActStartSub(BkfSuberSess *sess, BkfMsgCoder *coder, void *param)
{
    uint32_t ret = BKF_OK;
    if (sess->sessMng->deCongest == VOS_FALSE || BKF_FSM_GET_STATE(&sess->fsm) == BACKFIN_SUB_SESS_STATE_DOWN) {
        ret = BkfSubSessSndSub(sess, coder, param);
    }
    if (ret == BKF_OK) {
        if (sess->isVerify) {
            (void)BKF_FSM_CHG_STATE(&sess->fsm, BACKFIN_SUB_SESS_STATE_VERIFYWAITACK);
        } else {
            (void)BKF_FSM_CHG_STATE(&sess->fsm, BACKFIN_SUB_SESS_STATE_WAITSUBACK);
        }
    }

    return ret;
}

uint32_t BkfSubSessActDownAndUpdSeq(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param)
{
    (void)decoder;
    (void)param;
    if (sess->fsm.state == BACKFIN_SUB_SESS_STATE_SNDUNSUB) {
        return BKF_OK;
    }

    sess->isVerify = 0;
    sess->seq = BKF_GET_NEXT_VAL(sess->sessMng->env->seeds);

    if (BKF_DL_NODE_IS_IN(&sess->dlStateNode)) {
        BKF_DL_REMOVE(&sess->dlStateNode);
    }
    BKF_DL_ADD_LAST(&sess->sessMng->subSess, &sess->dlStateNode);
    BkfSubSessChangeStateAndTirg(sess, BACKFIN_SUB_SESS_STATE_DOWN);
    return BKF_OK;
}

uint32_t BkfSubSessCheckTransValid(BkfSuberSess *sess, BkfMsgDecoder *decoder)
{
    BkfTL *tl = VOS_NULL;
    BkfTlvTransNum *tlvTransNum = VOS_NULL;
    BkfTlvReasonCode *tlvReasonCode = VOS_NULL;
    uint32_t errCode;
    BOOL isInValid = VOS_FALSE;

    BKF_LOG_INFO(sess->sessMng->log, "Session Check TransNum Valid.\n");
    while ((tl = BkfMsgDecodeTLV(decoder, &errCode)) != VOS_NULL) {
        BKF_ASSERT(errCode == BKF_OK);

        if (tl->typeId == BKF_TLV_TRANS_NUM) {
            tlvTransNum = (BkfTlvTransNum *)tl;
            if (tlvTransNum->num == BKF_TRANS_NUM_INVALID) {
                return BKF_ERR;
            }

            isInValid = (sess->seq == BKF_TRANS_NUM_INVALID) || (sess->seq != tlvTransNum->num);
            if (isInValid) {
                BKF_LOG_ERROR(sess->sessMng->log,
                    "subSess(%#x), subTransNum(L:%" VOS_PRIu64 "/P:%" VOS_PRIu64 ")"
                    " not same, check error.\n",
                    BKF_MASK_ADDR(sess), sess->seq, tlvTransNum->num);
                /* 驱动会话进入初始状态，重新sub */
                (void)BkfSubSessActDownAndUpdSeq(sess, decoder, VOS_NULL);
                return BKF_ERR;
            }
        } else if (tl->typeId == BKF_TLV_REASON_CODE) {
            tlvReasonCode = (BkfTlvReasonCode *)tl;
            if (tlvReasonCode->reasonCode != BKF_RC_OK) {
                BKF_LOG_ERROR(sess->sessMng->log, "subSess(%#x), ReasonCode(%u) check error.\n", BKF_MASK_ADDR(sess),
                    tlvReasonCode->reasonCode);
                /* 驱动会话进入初始状态，重新sub */
                (void)BkfSubSessActDownAndUpdSeq(sess, decoder, VOS_NULL);
                return BKF_ERR;
            }
        } else {
            continue;
        }
    }

    return BKF_OK;
}

STATIC uint32_t BkfSubSessActRcvSubAck(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param)
{
    (void)param;
    uint32_t errCode;

    uint8_t buf[BKF_SUBER_DISP_SLICELEN] = {0};
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint8_t urlStr2[BKF_URL_STR_LEN_MAX];
    BKF_LOG_WARN(sess->sessMng->log, "RcvAck subSess(%#x) tableType:%u, seq:%" VOS_PRIu64 ", %s p %s l %s\n",
        BKF_MASK_ADDR(sess), sess->key.tableTypeId, sess->seq,
        BkfSuberGetSliceKeyStr(sess->sessMng->env, sess->key.sliceKey, buf, sizeof(buf)),
        BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
        BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));

    errCode = BkfSubSessCheckTransValid(sess, decoder);
    if (errCode != BKF_OK) {
        return errCode;
    }

    (void)BKF_FSM_CHG_STATE(&sess->fsm, BACKFIN_SUB_SESS_STATE_UP);
    if (BKF_DL_NODE_IS_IN(&(sess->dlStateNode))) {
        BKF_DL_REMOVE(&(sess->dlStateNode));
    }
    return BKF_OK;
}

STATIC uint32_t BkfSubSessSndUnSub(BkfSuberSess *sess, BkfMsgCoder *coder, void *param)
{
    (void)param;
    BkfSuberTableTypeVTbl *vtbl = BkfSuberDataTableTypeGetVtbl(sess->sessMng->dataMng, sess->key.tableTypeId);
    if (vtbl == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    BKF_LOG_WARN(sess->sessMng->log, "Session (%#x) Send UnSub.\n", BKF_MASK_ADDR(sess));

    uint8_t flag = 0;
    if (sess->isVerify) {
        BKF_BIT_SET(flag, BKF_FLAG_VERIFY);
    }
    if (vtbl->needTableComplete) {
        BKF_BIT_SET(flag, BKF_FLAG_NEED_TABLE_COMPLETE);
    }
    uint32_t ret = BkfMsgCodeMsgHead(coder, BKF_MSG_UNSUB, flag);
    ret |= BkfMsgCodeRawTLV(coder, BKF_TLV_SLICE_KEY, 0, sess->key.sliceKey, sess->sessMng->env->sliceVTbl.keyLen,
        VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_TABLE_TYPE, 0, &sess->key.tableTypeId, VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_TRANS_NUM, 0, &sess->seq, VOS_TRUE);
    if (ret != BKF_OK) {
        return BKF_SUBER_SESS_SEND_BUF_NOT_ENOUGH;
    }

    if (BKF_DL_NODE_IS_IN(&(sess->dlStateNode))) {
        BKF_DL_REMOVE(&(sess->dlStateNode));
    }

    /* 1.对账->去对账->发送unsub后切换为batok, sess标记外面不改变，此处置为false */
    /* 2.any->去订阅->发送unsub后删除sess，sess标记外面强制为false，此处返回删除 */
    /* 3.非对账->去对账->不处理 */
    if (sess->isVerify) {
        sess->isVerify = VOS_FALSE;
        (void)BKF_FSM_CHG_STATE(&sess->fsm, BACKFIN_SUB_SESS_STATE_BATOK);
        return BKF_OK;
    } else {
        BkfSuberSessBatchChkTmrStop(sess);
        return BKF_SUBER_SESS_NEED_DELETE;
    }
}

STATIC uint32_t BkfSubSessActRcvBatchBegin(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param)
{
    (void)param;
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint8_t urlStr2[BKF_URL_STR_LEN_MAX];
    BKF_LOG_WARN(sess->sessMng->log, "Session (%#x) Receive Batch Begin, p %s l %s\n", BKF_MASK_ADDR(sess),
        BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
        BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));

    uint32_t errCode = BkfSubSessCheckTransValid(sess, decoder);
    if (errCode != BKF_OK) {
        BKF_LOG_ERROR(sess->sessMng->log, "batchBegin(typeId:%u, dataLen:%d)\n", sess->key.tableTypeId,
            decoder->rcvDataLen);
        return errCode;
    }

    (void)BKF_FSM_CHG_STATE(&sess->fsm, BACKFIN_SUB_SESS_STATE_BATCH);
    BkfSuberSessBatchChkTmrStart(sess);

    BkfSuberTableTypeVTbl *vtbl = BkfSuberDataTableTypeGetVtbl(sess->sessMng->dataMng, sess->key.tableTypeId);
    if (vtbl == VOS_NULL) {
        BKF_LOG_ERROR(sess->sessMng->log, "batchBegin(typeId:%u, dataLen:%d) Vtl Null.\n", sess->key.tableTypeId,
            decoder->rcvDataLen);
        return BKF_ERR;
    }
    if (BkfSuberDataTableTypeGetMode(vtbl) == BKF_SUBER_MODE_DEFAULT) {
        if (vtbl->onTableBatchBegin != VOS_NULL) {
            vtbl->onTableBatchBegin(vtbl->cookie, sess->key.sliceKey);
        }
    } else {
        BkfSuberTableTypeVTblEx *vtblEx = (BkfSuberTableTypeVTblEx *)vtbl;
        if (vtblEx->onTableBatchBeginEx != VOS_NULL) {
            vtblEx->onTableBatchBeginEx(vtbl->cookie, sess->key.sliceKey, &sess->sessMng->pubUrl,
                &sess->sessMng->locUrl);
        }
    }
    return BKF_OK;
}

STATIC uint32_t BkfSubSessActRcvBatchEnd(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param)
{
    (void)param;
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint8_t urlStr2[BKF_URL_STR_LEN_MAX];
    BKF_LOG_WARN(sess->sessMng->log, "Session (%#x) Receive Batch End. p %s l %s\n", BKF_MASK_ADDR(sess),
        BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
        BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));

    uint32_t errCode = BkfSubSessCheckTransValid(sess, decoder);
    if (errCode != BKF_OK) {
        BKF_LOG_ERROR(sess->sessMng->log, "batchEnd(typeId:%u, dataLen:%d)\n", sess->key.tableTypeId,
            decoder->rcvDataLen);
        return errCode;
    }

    BkfSuberSessBatchChkTmrStop(sess);
    (void)BKF_FSM_CHG_STATE(&sess->fsm, BACKFIN_SUB_SESS_STATE_BATOK);
    BkfSuberTableTypeVTbl *vtbl = BkfSuberDataTableTypeGetVtbl(sess->sessMng->dataMng, sess->key.tableTypeId);
    if (vtbl == VOS_NULL) {
        BKF_LOG_ERROR(sess->sessMng->log, "batchEnd(typeId:%u, dataLen:%d) Vtl Null.\n", sess->key.tableTypeId,
            decoder->rcvDataLen);
        return BKF_ERR;
    }

    if (BkfSuberDataTableTypeGetMode(vtbl) == BKF_SUBER_MODE_DEFAULT) {
        if (vtbl->onTableBatchEnd != VOS_NULL) {
            vtbl->onTableBatchEnd(vtbl->cookie, sess->key.sliceKey);
        }
    } else {
        BkfSuberTableTypeVTblEx *vtblEx = (BkfSuberTableTypeVTblEx *)vtbl;
        if (vtblEx->onTableBatchEndEx != VOS_NULL) {
            vtblEx->onTableBatchEndEx(vtbl->cookie, sess->key.sliceKey, &sess->sessMng->pubUrl, &sess->sessMng->locUrl);
        }
    }
    return BKF_OK;
}

void BkfSubSessRcvDataCallBack(BkfSuberSess *sess, BkfSuberTableTypeVTbl *vtbl, BkfTlvTupleIdlData *idlData)
{
    uint32_t valLen = idlData->tl.valLen;
    BkfSuberSessKey *sessKey = &sess->key;
    if (BkfSuberDataTableTypeGetMode(vtbl) == BKF_SUBER_MODE_DEFAULT) {
        if (idlData->tl.flag == BKF_FLAG_TUPLE_UPDATE) {
            vtbl->onDataUpdate(vtbl->cookie, sessKey->sliceKey, idlData->idlData,
                valLen - BKF_MBR_SIZE(BkfTlvTupleIdlData, seq), VOS_TRUE);
        } else {
            vtbl->onDataDelete(vtbl->cookie, sessKey->sliceKey, idlData->idlData,
                valLen - BKF_MBR_SIZE(BkfTlvTupleIdlData, seq), VOS_TRUE);
        }
        return;
    }
    BkfSuberTableTypeVTblEx *vtblEx = (BkfSuberTableTypeVTblEx *)vtbl;
    if (idlData->tl.flag == BKF_FLAG_TUPLE_UPDATE) {
        vtblEx->onDataUpdateEx(vtbl->cookie, sessKey->sliceKey, idlData->idlData,
            valLen - BKF_MBR_SIZE(BkfTlvTupleIdlData, seq), VOS_TRUE, &sess->sessMng->pubUrl, &sess->sessMng->locUrl);
    } else {
        vtblEx->onDataDeleteEx(vtbl->cookie, sessKey->sliceKey, idlData->idlData,
            valLen - BKF_MBR_SIZE(BkfTlvTupleIdlData, seq), VOS_TRUE, &sess->sessMng->pubUrl, &sess->sessMng->locUrl);
    }
}

uint32_t BkfSubSessDecodeIdlData(BkfSuberTableTypeVTbl *vtbl, BkfSuberSess *sess, BkfMsgDecoder *decoder)
{
    uint32_t errCode;
    BkfTL *tl = VOS_NULL;
    uint32_t valLen;
    BkfTlvTupleIdlData *idlData = VOS_NULL;
    BkfTlvTransNum *tlvTransNum = VOS_NULL;
    BOOL isValid = VOS_FALSE;
    BkfSuberSessKey *sessKey = &sess->key;
    uint8_t suberIdlData[BKF_SUBSESS_PKT_DISP_LEN];
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint8_t urlStr2[BKF_URL_STR_LEN_MAX];
    tl = BkfMsgDecodeTLV(decoder, &errCode);
    while (tl != VOS_NULL) {
        if ((tl->typeId == BKF_TLV_TRANS_NUM)) {
            tlvTransNum = (BkfTlvTransNum *)tl;
        } else if (tl->typeId == BKF_TLV_TUPLE_IDL_DATA) {
            isValid = (tlvTransNum == VOS_NULL) ||
                      ((sess->seq == BKF_TRANS_NUM_INVALID) || (sess->seq != tlvTransNum->num));
            if (isValid) {
                BKF_LOG_ERROR(sess->sessMng->log,
                    "suber sess(%#x) seq err: NULL(ctlSeq:%" VOS_PRIu64 ", tlvNum:%" VOS_PRIu64 "). p %s l %s\n",
                    BKF_MASK_ADDR(sess), sess->seq, tlvTransNum != VOS_NULL ? tlvTransNum->num : 0,
                    BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
                    BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));
                return BKF_ERR;
            }
            valLen = tl->valLen;
            idlData = (BkfTlvTupleIdlData *)tl;
            (void)BKF_GET_MEM_STD_STR(idlData->idlData, tl->valLen - sizeof(BkfTL), suberIdlData,
                BKF_SUBSESS_PKT_DISP_LEN);

            uint8_t buf[BKF_SUBER_DISP_SLICELEN] = {0};
            /* 日志数量较多，不记录里程碑 */
            BKF_LOG_DEBUG(sess->sessMng->log,
                "recvIdlData(sess (%#x) typeId:%u, flag:%u, Key:%s, Len:%u, Addr:%#x, conText:\n%s). p %s l %s\n",
                BKF_MASK_ADDR(sess), sessKey->tableTypeId, tl->flag,
                BkfSuberGetSliceKeyStr(sess->sessMng->env, sess->key.sliceKey, buf, sizeof(buf)),
                valLen - sizeof(BkfTL), BKF_MASK_ADDR(idlData->idlData), suberIdlData,
                BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
                BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));
            BkfSubSessRcvDataCallBack(sess, vtbl, idlData);
        }

        tl = BkfMsgDecodeTLV(decoder, &errCode);
    }

    return BKF_OK;
}

STATIC uint32_t BkfSubSessActRcvData(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param)
{
    (void)param;
    BkfSuberSessKey *sessKey = &sess->key;
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint8_t urlStr2[BKF_URL_STR_LEN_MAX];
    if (sess->fsm.state < BACKFIN_SUB_SESS_STATE_BATCH) {
        BKF_LOG_ERROR(sess->sessMng->log, "Suber (%#x) Fsm ERR(typeId: %u, state: %u) p %s l %s\n", BKF_MASK_ADDR(sess),
            sessKey->tableTypeId, sess->fsm.state, BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
            BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));
        return BKF_ERR;
    }

    /* 查询控制块数据 */
    BkfSuberTableTypeVTbl *vtbl = BkfSuberDataTableTypeGetVtbl(sess->sessMng->dataMng, sessKey->tableTypeId);
    if (vtbl == VOS_NULL) {
        BKF_LOG_ERROR(sess->sessMng->log, "Suber sess(%#x) VTBL NULL(typeId: %u, state: %u) p %s l %s\n",
            BKF_MASK_ADDR(sess), sessKey->tableTypeId, sess->fsm.state,
            BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
            BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));
        return BKF_ERR;
    }

    return BkfSubSessDecodeIdlData(vtbl, sess, decoder);
}

/* 驱动连接状态机启动订阅 */
STATIC void BkfSubSessProcDisconn(BkfSuberSess *sess)
{
    BkfSuberTableTypeVTbl *vtbl = VOS_NULL;
    BkfSuberSessKey *sessKey = VOS_NULL;
    sessKey = &sess->key;
    vtbl = BkfSuberDataTableTypeGetVtbl(sess->sessMng->dataMng, sessKey->tableTypeId);
    if (vtbl == VOS_NULL) {
        return;
    }

    if (BkfSuberDataTableTypeGetMode(vtbl) == BKF_SUBER_MODE_DEFAULT) {
        if (vtbl->onTableDisconn != VOS_NULL) {
            vtbl->onTableDisconn(vtbl->cookie, sess->key.sliceKey);
        }
    } else {
        BkfSuberTableTypeVTblEx *vtblEx = (BkfSuberTableTypeVTblEx *)vtbl;
        if (vtblEx->onTableDisconnEx != VOS_NULL) {
            vtblEx->onTableDisconnEx(vtbl->cookie, sess->key.sliceKey, &sess->sessMng->pubUrl, &sess->sessMng->locUrl,
                sess->sessMng->disconnReason);
        }
    }
}

STATIC uint32_t BkfSubSessActDisconnNotReSub(BkfSuberSess *sess, BkfMsgDecoder *noUse, void *param)
{
    (void)noUse;
    (void)param;
    BkfSubSessProcDisconn(sess);
    BkfSuberSessBatchChkTmrStop(sess);
    return BKF_SUBER_SESS_NEED_DELETE;
}

STATIC uint32_t BkfSubSessActDisconn(BkfSuberSess *sess, BkfMsgDecoder *noUse, void *param)
{
    (void)noUse;
    BkfSubSessProcDisconn(sess);
    /* 序列号增加，重新进行状态机 */
    uint32_t ret = BkfSubSessActDownAndUpdSeq(sess, VOS_NULL, param);
    return ret;
}

STATIC uint32_t BkfSubSessActStartVerifySub(BkfSuberSess *sess, BkfMsgCoder *coder, void *param)
{
    (void)param;
    BkfSuberTableTypeVTbl *vtbl = BkfSuberDataTableTypeGetVtbl(sess->sessMng->dataMng, sess->key.tableTypeId);
    if (vtbl == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    uint8_t flag = 0;
    if (sess->isVerify) {
        BKF_BIT_SET(flag, BKF_FLAG_VERIFY);
    }
    if (vtbl->needTableComplete) {
        BKF_BIT_SET(flag, BKF_FLAG_NEED_TABLE_COMPLETE);
    }

    uint32_t ret = BkfMsgCodeMsgHead(coder, BKF_MSG_SUB, flag);
    ret |= BkfMsgCodeRawTLV(coder, BKF_TLV_SLICE_KEY, 0, sess->key.sliceKey, sess->sessMng->env->sliceVTbl.keyLen,
        VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_TABLE_TYPE, 0, &sess->key.tableTypeId, VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_TRANS_NUM, 0, &sess->seq, VOS_TRUE);

    uint8_t buf[BKF_SUBER_DISP_SLICELEN] = {0};
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint8_t urlStr2[BKF_URL_STR_LEN_MAX];
    BKF_LOG_WARN(sess->sessMng->log, "Snd verifysub, Sess(%#x) tableType:%u, seq:%" VOS_PRIu64 ", %s p %s, l %s\n",
        BKF_MASK_ADDR(sess), sess->key.tableTypeId, sess->seq,
        BkfSuberGetSliceKeyStr(sess->sessMng->env, sess->key.sliceKey, buf, sizeof(buf)),
        BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
        BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));

    if (ret == BKF_OK) {
        (void)BKF_FSM_CHG_STATE(&sess->fsm, BACKFIN_SUB_SESS_STATE_VERIFYWAITACK);
    }

    return BKF_OK;
}

STATIC uint32_t BkfSubSessActRcvVerifyAck(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param)
{
    (void)param;
    uint32_t errCode;

    BKF_RETURNVAL_IF(sess == VOS_NULL, BKF_ERR);
    BKF_RETURNVAL_IF(decoder == VOS_NULL, BKF_ERR);

    uint8_t buf[BKF_1K / 4] = {0};
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint8_t urlStr2[BKF_URL_STR_LEN_MAX];
    BKF_LOG_WARN(sess->sessMng->log, "Rcv verifyAck, Sess(%#x) tableType:%u, seq:%" VOS_PRIu64 ", %s p %s l %s\n",
        BKF_MASK_ADDR(sess), sess->key.tableTypeId, sess->seq,
        BkfSuberGetSliceKeyStr(sess->sessMng->env, sess->key.sliceKey, buf, sizeof(buf)),
        BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
        BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));

    errCode = BkfSubSessCheckTransValid(sess, decoder);
    if (errCode != BKF_OK) {
        return errCode;
    }

    if (BKF_DL_NODE_IS_IN(&(sess->dlStateNode))) {
        BKF_DL_REMOVE(&(sess->dlStateNode));
    }

    (void)BKF_FSM_CHG_STATE(&sess->fsm, BACKFIN_SUB_SESS_STATE_VERIFYRDY);
    return BKF_OK;
}

STATIC uint32_t BkfSubSessActRcvVerifyBegin(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param)
{
    (void)param;
    uint32_t errCode;
    BkfSuberSessKey *sessKey = VOS_NULL;
    BkfSuberTableTypeVTbl *vtbl = VOS_NULL;

    BKF_LOG_WARN(sess->sessMng->log, "Session (%#x) Receive Verify Begin.\n", BKF_MASK_ADDR(sess));

    sessKey = &sess->key;
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint8_t urlStr2[BKF_URL_STR_LEN_MAX];
    errCode = BkfSubSessCheckTransValid(sess, decoder);
    if (errCode != BKF_OK) {
        BKF_LOG_ERROR(sess->sessMng->log, "Sess (%#x)VerifyBegin(typeId:%u, dataLen:%d) p %s l %s\n",
            BKF_MASK_ADDR(sess), sessKey->tableTypeId, decoder->rcvDataLen,
            BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
            BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));
        return errCode;
    }

    (void)BKF_FSM_CHG_STATE(&sess->fsm, BACKFIN_SUB_SESS_STATE_VERIFY);

    /* 通知业务对账开始 */
    vtbl = BkfSuberDataTableTypeGetVtbl(sess->sessMng->dataMng, sessKey->tableTypeId);
    if (vtbl == VOS_NULL) {
        BKF_LOG_ERROR(sess->sessMng->log, "Sess (%#x) VerifyBegin(typeId:%u, dataLen:%d) Vtl Null, p %s l %s\n",
            BKF_MASK_ADDR(sess), sessKey->tableTypeId, decoder->rcvDataLen,
            BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
            BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));
        return BKF_OK;
    }

    if (BkfSuberDataTableTypeGetMode(vtbl) == BKF_SUBER_MODE_DEFAULT) {
        if (vtbl->onTableVerifyBegin != VOS_NULL) {
            vtbl->onTableVerifyBegin(vtbl->cookie, sess->key.sliceKey);
        }
    } else {
        BkfSuberTableTypeVTblEx *vtblEx = (BkfSuberTableTypeVTblEx *)vtbl;
        if (vtblEx->onTableVerifyBeginEx != VOS_NULL) {
            vtblEx->onTableVerifyBeginEx(vtbl->cookie, sess->key.sliceKey, &sess->sessMng->pubUrl,
                &sess->sessMng->locUrl);
        }
    }
    return BKF_OK;
}

STATIC uint32_t BkfSubSessActRcvVerifyEnd(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param)
{
    (void)param;
    uint32_t errCode;
    BkfSuberSessKey *sessKey = VOS_NULL;
    BkfSuberTableTypeVTbl *vtbl = VOS_NULL;

    BKF_LOG_WARN(sess->sessMng->log, "Session (%#x) Receive Verify End.\n", BKF_MASK_ADDR(sess));

    sessKey = &sess->key;
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint8_t urlStr2[BKF_URL_STR_LEN_MAX];
    errCode = BkfSubSessCheckTransValid(sess, decoder);
    if (errCode != BKF_OK) {
        BKF_LOG_ERROR(sess->sessMng->log, "Sess (%#x) VerifyEnd(typeId:%u, dataLen:%d) p %s l %s\n",
            BKF_MASK_ADDR(sess), sessKey->tableTypeId, decoder->rcvDataLen,
            BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
            BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));
        return errCode;
    }
    sess->isVerify = 0;
    (void)BKF_FSM_CHG_STATE(&sess->fsm, BACKFIN_SUB_SESS_STATE_BATOK);

    vtbl = BkfSuberDataTableTypeGetVtbl(sess->sessMng->dataMng, sessKey->tableTypeId);
    if (vtbl == VOS_NULL) {
        BKF_LOG_ERROR(sess->sessMng->log, "Sess (%#x) VerifyEnd(typeId:%u, dataLen:%d) Vtl Null p %s l %s\n",
            BKF_MASK_ADDR(sess), sessKey->tableTypeId, decoder->rcvDataLen,
            BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
            BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));
        return BKF_OK;
    }

    if (BkfSuberDataTableTypeGetMode(vtbl) == BKF_SUBER_MODE_DEFAULT) {
        if (vtbl->onTableVerifyEnd != VOS_NULL) {
            vtbl->onTableVerifyEnd(vtbl->cookie, sess->key.sliceKey);
        }
    } else {
        BkfSuberTableTypeVTblEx *vtblEx = (BkfSuberTableTypeVTblEx *)vtbl;
        if (vtblEx->onTableVerifyEndEx != VOS_NULL) {
            vtblEx->onTableVerifyEndEx(vtbl->cookie, sess->key.sliceKey, &sess->sessMng->pubUrl,
                &sess->sessMng->locUrl);
        }
    }

    return BKF_OK;
}

STATIC uint32_t BkfSubSessProcNtf(BkfSuberSess *sess, BkfMsgDecoder *decoder)
{
    uint32_t errCode;
    BkfSuberSessKey *sessKey = VOS_NULL;
    BkfSuberTableTypeVTbl *vtbl = VOS_NULL;

    BKF_LOG_WARN(sess->sessMng->log, "Session (%#x) Receive notify.\n", BKF_MASK_ADDR(sess));
    sessKey = &sess->key;
    uint8_t urlStr[BKF_URL_STR_LEN_MAX];
    uint8_t urlStr2[BKF_URL_STR_LEN_MAX];
    errCode = BkfSubSessCheckTransValid(sess, decoder);
    if (errCode != BKF_OK) {
        BKF_LOG_ERROR(sess->sessMng->log, "Sess (%#x) ntf(typeId:%u, dataLen:%d) p %s l %s\n", BKF_MASK_ADDR(sess),
            sessKey->tableTypeId, decoder->rcvDataLen, BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
            BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));
        return errCode;
    }

    vtbl = BkfSuberDataTableTypeGetVtbl(sess->sessMng->dataMng, sessKey->tableTypeId);
    if (vtbl == VOS_NULL) {
        BKF_LOG_ERROR(sess->sessMng->log, "Sess (%#x) ntf(typeId:%u, dataLen:%d) Vtl Null p %s l %s\n",
            BKF_MASK_ADDR(sess), sessKey->tableTypeId, decoder->rcvDataLen,
            BkfUrlGetStr(&sess->sessMng->pubUrl, urlStr, sizeof(urlStr)),
            BkfUrlGetStr(&sess->sessMng->locUrl, urlStr2, sizeof(urlStr2)));
        return BKF_OK;
    }
    if (BkfSuberDataTableTypeGetMode(vtbl) == BKF_SUBER_MODE_DEFAULT) {
        if (vtbl->onTableDelete != VOS_NULL) {
            vtbl->onTableDelete(vtbl->cookie, sess->key.sliceKey);
        }
    } else {
        BkfSuberTableTypeVTblEx *vtblEx = (BkfSuberTableTypeVTblEx *)vtbl;
        if (vtblEx->onTableDeleteEx != VOS_NULL) {
            vtblEx->onTableDeleteEx(vtbl->cookie, sess->key.sliceKey, &sess->sessMng->pubUrl, &sess->sessMng->locUrl);
        }
    }
    return BKF_OK;
}

STATIC uint32_t BkfSubSessActRcvNtf(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *param)
{
    uint32_t ret = BkfSubSessProcNtf(sess, decoder);
    if (ret != BKF_OK) {
        return ret;
    }
    (void)BKF_FSM_CHG_STATE(&sess->fsm, BACKFIN_SUB_SESS_STATE_DOWN);
    // 重新建立session
    BkfSubSessActDownAndUpdSeq(sess, decoder, param);
    BkfSuberSessCreateReSubTmr(sess->sessMng);
    BkfSuberSessBatchChkTmrStop(sess);
    return BKF_OK;
}

STATIC uint32_t BkfSubSessActRcvNtfNotReSub(BkfSuberSess *sess, BkfMsgDecoder *decoder, void *parm)
{
    (void)parm;
    uint32_t ret = BkfSubSessProcNtf(sess, decoder);
    if (ret != BKF_OK) {
        return ret;
    }

    BkfSuberSessBatchChkTmrStop(sess);
    return BKF_SUBER_SESS_NEED_DELETE;
}

uint32_t BkfSuberSessFsmInitFsmTmp(BkfSuberSessMng *sessMng)
{
    BkfFsmTmplInitArg fsmArg;
    int32_t err;

    err = snprintf_truncated_s(sessMng->sessFsmTmplName, sizeof(sessMng->sessFsmTmplName),
        "%s_sessFsmTmpl_%" VOS_PRIu64 "", sessMng->env->name, BKF_GET_NEXT_VAL(sessMng->env->seeds));
    if (err <= 0) {
        return BKF_ERR;
    }

    (void)memset_s(&fsmArg, sizeof(BkfFsmTmplInitArg), 0, sizeof(BkfFsmTmplInitArg));
    fsmArg.evtCnt = BACKFIN_SUB_SESS_INPUT_MAX;
    fsmArg.stateCnt = BACKFIN_SUB_SESS_STATE_MAX;
    fsmArg.name = sessMng->sessFsmTmplName;
    fsmArg.dbgOn = sessMng->env->dbgOn;
    fsmArg.disp = sessMng->env->disp;
    fsmArg.log = sessMng->env->log;
    fsmArg.memMng = sessMng->env->memMng;
    fsmArg.stateStrTbl = subSessStateStrTbl;
    fsmArg.evtStrTbl = subSessEvtStrTbl;
    fsmArg.stateEvtProcItemMtrx = (const BkfFsmProcItem *)subSessStateEvtProcItemMtrx;
    fsmArg.getAppDataStrOrNull = VOS_NULL;
    sessMng->sessFsmTmpl = BkfFsmTmplInit(&fsmArg);
    if (sessMng->sessFsmTmpl == VOS_NULL) {
        return BKF_ERR;
    }
    return BKF_OK;
}

void BkfSuberSessFsmUnInitFsmTmp(BkfSuberSessMng *sessMng)
{
    if (sessMng->sessFsmTmpl != VOS_NULL) {
        BkfFsmTmplUninit(sessMng->sessFsmTmpl);
    }
}

/* 平滑超时检查 */
void BkfSuberSessNtfyAppBatchTimeout(BkfSuberSess *sess)
{
    BkfSuberSessKey *sessKey = &sess->key;
    BkfSuberTableTypeVTbl *vtbl = BkfSuberDataTableTypeGetVtbl(sess->sessMng->dataMng, sessKey->tableTypeId);
    if (vtbl == VOS_NULL) {
        BKF_ASSERT(0);
        return;
    }

    if (vtbl->batchTimeout == 0) {
        return;
    }

    if (BkfSuberDataTableTypeGetMode(vtbl) == BKF_SUBER_MODE_DEFAULT) {
        if (vtbl->onBatchTimeout != VOS_NULL) {
            vtbl->onBatchTimeout(vtbl->cookie, sess->key.sliceKey);
        }
    } else {
        BkfSuberTableTypeVTblEx *vtblEx = (BkfSuberTableTypeVTblEx *)vtbl;
        if (vtblEx->onBatchTimeoutEx != VOS_NULL) {
            vtblEx->onBatchTimeoutEx(vtbl->cookie, sess->key.sliceKey, &sess->sessMng->pubUrl, &sess->sessMng->locUrl);
        }
    }
}

STATIC uint32_t BkfSuberSessBatchChkTimeout(BkfSuberSess *sess, void *noUse)
{
    (void)noUse;
    if (sess->batchChkTmr == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_OK;
    }
    sess->batchChkTmr = VOS_NULL;

    /* 仅平滑超时的时候通知, 不包括对账 */
    if (sess->isVerify) {
        return BKF_OK;
    }
    uint8_t state = BKF_FSM_GET_STATE(&sess->fsm);
    if (state == BACKFIN_SUB_SESS_STATE_BATCH) {
        BkfSuberSessNtfyAppBatchTimeout(sess);
    }
    return BKF_OK;
}

void BkfSuberSessBatchChkTmrStart(BkfSuberSess *sess)
{
    BkfSuberSessMng *sessMng = sess->sessMng;
    if (sessMng->batchTimeoutChkFlag == VOS_TRUE) {
        BkfSuberSessBatchChkTmrStop(sess);
        return;
    }

    BkfSuberSessKey *sessKey = &sess->key;
    BkfSuberTableTypeVTbl *vTbl = BkfSuberDataTableTypeGetVtbl(sessMng->dataMng, sessKey->tableTypeId);
    if (vTbl == VOS_NULL) {
        BKF_ASSERT(0);
        return;
    }

    if (vTbl->batchTimeout == 0) {
        return;
    }

    if (sess->batchChkTmr != VOS_NULL) {
        BkfTmrStop(sessMng->env->tmrMng, sess->batchChkTmr);
    }

    sess->batchChkTmr = BkfTmrStartOnce(sessMng->env->tmrMng, (F_BKF_TMR_TIMEOUT_PROC)BkfSuberSessBatchChkTimeout,
        vTbl->batchTimeout * BKF_MS_PER_S, (void *)sess);
    return;
}

void BkfSuberSessBatchChkTmrStop(BkfSuberSess *sess)
{
    if (sess->batchChkTmr == VOS_NULL) {
        return;
    }

    BkfSuberSessMng *sessMng = sess->sessMng;
    BkfTmrStop(sessMng->env->tmrMng, sess->batchChkTmr);
    sess->batchChkTmr = VOS_NULL;
}

#ifdef __cplusplus
}
#endif
