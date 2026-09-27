/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_RDMA_INNER_H
#define RS_RDMA_INNER_H

#include "rs_common_inner.h"
#include "rs_list.h"

#define RS_WC_NUM 16384
#define RS_QP_ATTR_MIN_RNR_TIMER 12
#define RS_QP_ATTR_TIMEOUT 16
#define RS_QP_ATTR_RETRY_CNT 7
#define RS_QP_ATTR_RNR_RETRY 7
#define RS_QP_ATTR_MAX_SEND_SGE 8
#define RS_QP_ATTR_GID_LEN 16
#define RS_MAX_RD_ATOMIC_NUM 128
#define RS_QP_TX_DEPTH_PEER_ONLINE 4096     // host RDMA adapt
#define RS_MAX_RD_ATOMIC_NUM_PEER_ONLINE 16 // host RDMA adapt
#define RS_BUF_SIZE 2048
#define RS_PORT_DEF 1

#define RS_QP_PARA_CHECK(phyId)                                                                                        \
    do {                                                                                                               \
        if ((phyId) >= RS_MAX_DEV_NUM) {                                                                               \
            hccp_err("rs qp param error ! physical_id:%d", phyId);                                                     \
            return (-EINVAL);                                                                                          \
        }                                                                                                              \
    } while (0)

enum RsCmdOpcode {
    RS_CMD_QP_INFO = 0x12345678,
    RS_CMD_MR_INFO = 0x12345687,
    RS_CMD_LEN_INFO = 0x12345867,
};

struct RsMrInfo {
    uint32_t cmd; /* MUST be the first element */

    uint32_t rkey;
    uint64_t addr;
    uint64_t len;
};

struct RsRdevCb {
    struct rs_cb *rsCb;
    unsigned int rdevIndex;
    struct RsIpAddrInfo localIp;
    int devNum;
    const char *devName;
    int pollCqeNum;
    unsigned char ibPort;
    unsigned int qpCnt;
    unsigned int qpMaxNum;
    unsigned int txDepth;
    unsigned int rxDepth;
    unsigned int notifyType;
    unsigned long long notifyPaBase;
    unsigned long long notifyVaBase;
    unsigned long long notifySize;
    int notifyAccess;
    unsigned int cqeErrCnt;
    pthread_mutex_t cqeErrCntMutex;

    pthread_mutex_t rdevMutex;

    struct ibv_device_attr deviceAttr;
    struct ibv_mr *notifyMr;
    struct ibv_pd *ibPd;
    struct ibv_context *ibCtx;
    struct ibv_device **devList;

    struct ibv_context_extend *ibCtxEx;
    int directFlag;

    struct RsListHead qpList;
    struct RsListHead typicalMrList;
    struct RsListHead list;

    int supportLite;
    struct {
        bool backupFlag;
        struct rdev rdevInfo;
        struct ibv_context *ibCtx;
    } backupInfo;
};
#endif // RS_RDMA_INNER_H
