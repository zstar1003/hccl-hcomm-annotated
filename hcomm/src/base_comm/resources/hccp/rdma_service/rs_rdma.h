/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_RDMA_H
#define RS_RDMA_H

#include "rs_inner.h"

#define MAX_NOTIFY_SIZE_CLOUD 8192
#define NOTIFY_NUM_MAX_V2 32768
#define NOTIFY_NUM_MAX_V3 1048576
#define WRITE_NOTIFY_OFFSET_MASK 0xffffff
#define WRITE_NOTIFY_VALUE_RECORD 0x1000000
#define RS_MR_NUM_MAX 256
#define RS_QP_TX_DEPTH_OFFLINE 128
#define RS_QP_RX_DEPTH_OFFLINE 128
#define RS_QP_TX_DEPTH_ONLINE 4096
#define RS_QP_RX_DEPTH_ONLINE 4096
#define RS_QP_TX_DEPTH 8191
#define RS_QP_32K_DEPTH 32767
#define RS_ROCE_4_SL 4
#define RS_QP_NUM_MAX 8192
#define RS_SGLIST_LEN_MAX 2147483648
#define RS_MIN_TEMPTH_DEPTH 8
#define RS_MAX_TEMPTH_DEPTH 4096
#define RS_MR_STATE_SYNCED 1

struct RsQpLenInfo {
    uint32_t cmd;
    uint32_t len;
};

struct RsTypicalCqEntry {
    unsigned int phyId;
    unsigned int rdevIndex;
    unsigned int cqn;
    struct ibv_cq *ibCq;
    struct rdma_lite_device_cq_attr deviceCqAttr;
    struct RsListHead list;
};

int RsQueryRdevCb(unsigned int phyId, unsigned int rdevIndex, struct RsRdevCb **rdevCb);
void RsMrRelease(struct RsQpCb *qpCb);

#endif // RS_RDMA_H
