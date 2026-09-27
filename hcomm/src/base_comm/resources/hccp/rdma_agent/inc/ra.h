/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_H
#define RA_H

#include <dlfcn.h>
#include <sched.h>
#include "stdio.h"
#include "hccp_common.h"
#ifndef HNS_ROCE_LLT
#include "dlog_pub.h"
#endif
#include "user_log.h"
#include "ra_rs_comm.h"
#include "rdma_lite.h"

#if defined(HNS_ROCE_LLT) || defined(DEFINE_HNS_LLT)
#define STATIC
#else
#define STATIC static
#endif

#define RA_MAX_PHY_ID_NUM 64
#define MAX_SUPPORT_IFNUM 65536
#define SOCKET_SEND_MAXLEN 2048
#define MAX_HDC_DATA 65536
#define MAX_SOCKET_NUM 16

#define MAX_WLIST_NUM 16
#define MAX_WLIST_NUM_V1 32
#define MAX_SG_LIST_LEN_MAX 2147483648

#define ra_conn_para_check(conn, num) ((num) == 0 || (num) > MAX_SOCKET_NUM)

#define PROCESS_RA_SIGN_LENGTH 49
#define PROCESS_RA_RESV_LENGTH 4

#define MAX_IP_LEN 64

#define MAX_POLL_CQE_NUM 100

struct ProcessRaSign {
    pid_t tgid;
    char sign[PROCESS_RA_SIGN_LENGTH];
};

#define container_of(ptr, type, member)                                                                                \
    ({                                                                                                                 \
        const typeof(((type *)0)->member) *__mptr = (ptr);                                                             \
        (type *)((char *)__mptr - offsetof(type, member));                                                             \
    })

#define list_entry(n, type, member) container_of(n, type, member)

struct RaListHead {
    struct RaListHead *next, *prev;
};

static inline void RA_INIT_LIST_HEAD(struct RaListHead *list)
{
    list->next = list;
    list->prev = list;
}

#define RA_LIST_GET_HEAD_ENTRY(pos, n, head, member, type)                                                             \
    do {                                                                                                               \
        (pos) = list_entry((head)->next, type, member);                                                                \
        (n) = list_entry((pos)->member.next, type, member);                                                            \
    } while (0)

static inline bool RaListEmpty(struct RaListHead *head)
{
    return head->next == head;
}

static inline void ra_list_add_(struct RaListHead *xnew, struct RaListHead *prev, struct RaListHead *next)
{
    next->prev = xnew;
    xnew->next = next;
    xnew->prev = prev;
    prev->next = xnew;
}

static inline void RaListAddTail(struct RaListHead *xnew, struct RaListHead *head)
{
    ra_list_add_(xnew, head->prev, head);
}

static inline void ra_list_del_(struct RaListHead *prev, struct RaListHead *next)
{
    next->prev = prev;
    prev->next = next;
}

static inline void RaListDel(struct RaListHead *entry)
{
    ra_list_del_(entry->prev, entry->next);
}

struct RaBackupInfo {
    bool backupFlag;
    struct rdev rdevInfo;
};

struct RaCqeErrInfo {
    pthread_mutex_t mutex;
    struct CqeErrInfo info;
};

enum RdmaLiteThreadStatus {
    LITE_THREAD_STATUS_DESTROY = 0,
    LITE_THREAD_STATUS_RUNNING = 1,
    LITE_THREAD_STATUS_FINISH_RUNNING = 2,
    LITE_THREAD_STATUS_SUSPEND = 3,
};

struct RaRdmaHandle {
    unsigned int rdevIndex;
    struct rdev rdevInfo;
    struct RaBackupInfo backupInfo;
    struct RaRdmaOps *rdmaOps;
    int supportLite;
    struct rdma_lite_context *liteCtx;
    struct RaListHead qpList;
    pthread_mutex_t rdevMutex;
    pthread_mutex_t cqeErrCntMutex;
    unsigned int cqeErrCnt;
    pthread_t tid;
    enum RdmaLiteThreadStatus threadStatus;
    bool disabledLiteThread;
    bool enabled910aLite;
    unsigned int logicDevid;
    int sensorUpdateCnt;
    uint64_t sensorHandle;
    uint64_t qpCnt; // record the number of ra_qp_create_with_attrs function calls
    bool enabled2mbLite;
    uint8_t gid[HCCP_GID_RAW_LEN];
    uint64_t notifyVa;
    uint64_t notifySize;
    struct RaListHead typicalLiteCqList;
    pthread_mutex_t typicalLiteCqMutex;
};

struct RaSocketHandle {
    int scopeId;
    struct rdev rdevInfo;
    struct RaSocketOps *socketOps;
    uint64_t closeCnt;   // record the number of ra_socket_batch_close function calls
    uint64_t connectCnt; // record the number of ra_socket_batch_connect function calls
    uint64_t abortCnt;   // record the number of ra_socket_batch_abort function calls
};

struct RaLoopbackInfo {
    struct ibv_cq *ibSendCq;
    struct ibv_cq *ibRecvCq;
    void *cqContext;
    struct ibv_comp_channel *compChannel;
};

struct RaCqHandleExt {
    unsigned long long addr;
};

struct RaTypicalCqHandle {
    unsigned int cqn;
    unsigned int phyId;
    unsigned int rdevIndex;
    struct RaRdmaHandle *rdmaHandle;
    struct RaRdmaOps *rdmaOps;
    struct rdma_lite_cq *liteCq;
};

struct RaQpHandle {
    unsigned int qpn;
    int qpMode;
    int flag;
    unsigned int phyId;
    unsigned int rdevIndex;
    struct RaRdmaOps *rdmaOps; // only ra use
    int supportLite;
    struct rdma_lite_cq *sendLiteCq;
    struct rdma_lite_cq *recvLiteCq;
    struct rdma_lite_qp *liteQp;
    unsigned int liteQpState;
    struct LiteMrInfo localMr[RA_MR_MAX_NUM];
    struct LiteMrInfo remMr[RA_MR_MAX_NUM];
    pthread_mutex_t qpMutex;
    struct RaCqeErrInfo cqeErrInfo;
    int dbIndex;
    unsigned int sendWrNum;
    unsigned int pollCqeNum;
    unsigned int recvWrNum;
    unsigned int pollRecvCqeNum;
    struct RaListHead list;
    struct RaRdmaHandle *rdmaHandle;
    struct rdma_lite_wc *liteWc;
    unsigned int memIdx;
    int sqSigAll;
    struct TypicalQpAttr typicalQpAttr;
    unsigned int psn;
    unsigned int gidIdx;
    unsigned int sqDepth; // only valid in RDMA Lite scenario
    unsigned int bpCnt;   // only valid in RDMA Lite scenario
    struct RaLoopbackInfo *loopbackInfo;
    struct RaQpHandle *loopbackQpHandle;
    int directFlag;
};

struct RaMrHandle {
    uint64_t addr;
};

enum GetIfaddrsVersion {
    GET_IFADDRS_VERSION_1 = 1,
    GET_IFADDRS_VERSION_2,
    GET_IFADDRS_VERSION_3,
    GET_IFADDRS_MAX_VERSION,
};

enum RaQpStatus {
    RA_QP_STATUS_DISCONNECT = 0,
    RA_QP_STATUS_CONNECTED = 1,
    RA_QP_STATUS_TIMEOUT = 2,
    RA_QP_STATUS_CONNECTING = 3,
    RA_QP_STATUS_REM_FD_CLOSE = 4,
    RA_QP_STATUS_PAUSE = 5,
};
#endif // RA_H
