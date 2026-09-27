/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_ADP_HW_H
#define RA_ADP_HW_H
#include "ra.h"
#include "ra_hdc.h"
#include "ra_hdc_ping.h"
#include "ra_adp_async.h"
#include "rs.h"

#define BUCKET_DEPTH (1024 * 2)
#define MAX_CFG_OP_NUM (1024 * 64)
#define TOKEN_RATE 400            /* 400 tokens per 1ms */
#define HDC_ACCEPT_SLEEP_TIME 100 /* 100us */
#define HCCP_MAX_CHIP_ID 63U
#define HAVE_OP_RIGHT 1
#define HAVE_NOT_OP_RIGHT 0
#define OP_RIGHT_QUERY_ERR (-1)
#define TGID_INVALID 2
#define RECV_BUF_LEN_INVALID 3
#define HCCP_RUN_CPU_CORE 0

struct RaHdcServer {
    HDC_SERVER hdcServer;
    HDC_SESSION hdcSession;
    struct RaHdcOpSec opSec;
};

struct RaHdcInitPara {
    unsigned int chipId;
    pid_t hostTgid;
    pthread_mutex_t mutex;
    unsigned int hdcFlag;
    unsigned int connectStatus;
    unsigned int threadStatus; /* 0: sleep or init; 1: running; 2: destroying */
    char pidSign[PROCESS_RA_SIGN_LENGTH];
};

struct RaOpHandle {
    unsigned int opcode;
    int (*opHandle)(char *, char *, int *, int *, int);
    unsigned int dataSize;
};

#define RA_ADP_ATTRI_VISI_DEF __attribute__((visibility("default")))
RA_ADP_ATTRI_VISI_DEF int HccpInit(unsigned int chipId, pid_t pid, int hdcType, unsigned int whiteListStatus);
RA_ADP_ATTRI_VISI_DEF int HccpDeinit(unsigned int chipId);

#define HCCP_CHECK_PARAM_LEN(data_size, head_size, rcv_buf_len)                                                        \
    do {                                                                                                               \
        if ((size_t)(data_size) + (size_t)(head_size) != (size_t)(rcv_buf_len)) {                                      \
            hccp_err("op data size is invalid. data size:[%zu] head size:[%zu] recv buff len:[%zu]",                   \
                (size_t)(data_size), (size_t)(head_size), (size_t)(rcv_buf_len));                                      \
            return (-EINVAL);                                                                                          \
        }                                                                                                              \
    } while (0)

#if defined(HNS_ROCE_LLT) || defined(DEFINE_HNS_LLT)
#define HCCP_CHECK_PARAM_LEN_RET_HOST(data_size, head_size, rcv_buf_len, pResult)                                      \
    do {                                                                                                               \
    } while (0)
#else
#define HCCP_CHECK_PARAM_LEN_RET_HOST(data_size, head_size, rcv_buf_len, pResult)                                      \
    do {                                                                                                               \
        if ((size_t)(data_size) + (size_t)(head_size) > (size_t)(rcv_buf_len)) {                                       \
            hccp_err("op data size is invalid. data size:[%zu] head size:[%zu] recv buff len:[%zu]",                   \
                (size_t)(data_size), (size_t)(head_size), (size_t)(rcv_buf_len));                                      \
            *pResult = -EINVAL;                                                                                        \
            return 0;                                                                                                  \
        }                                                                                                              \
    } while (0)
#endif

#define HCCP_CHECK_PARAM_NUM(num, max_num)                                                                             \
    do {                                                                                                               \
        if ((num) > (max_num)) {                                                                                       \
            hccp_err("conn number is invalid");                                                                        \
            return (-EINVAL);                                                                                          \
        }                                                                                                              \
    } while (0)

static inline bool RaIsOpcodeLogSuppressed(unsigned int opcode)
{
    if (opcode != RA_RS_GET_SOCKET) {
        return false;
    }
    return true;
}

void RaHdcInitOpSec(struct RaHdcOpSec *opSec, unsigned long long tokenNum, bool isAsyncOp);
int RaHdcSessionAccept(unsigned int chipId, HDC_SESSION *session, int initHostTgid);
int RaHdcAsyncRecvPkt(struct RaHdcAsyncInfo *asyncInfo, unsigned int chipId, void **recvBuf, unsigned int *recvLen);
int RaHandle(struct RaHdcOpSec *opSec, char *recvBuf, int rcvBufLen, char **sendBuf, int *sndBufLen,
    unsigned int *closeSession);
int RaHdcAsyncSendPkt(struct RaHdcAsyncInfo *asyncInfo, unsigned int chipId, void *sendBuf, unsigned int sendLen);
void RaHdcCloseSession(HDC_SESSION *session);
#endif // RA_ADP_HW_H
