/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include "rdma_lite.h"

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short int u16;
typedef unsigned char u8;

struct rdma_lite_context* rdma_lite_alloc_context(u8 phyId, struct dev_cap_info* cap)
{
    struct rdma_lite_context* lite_ctx = calloc(1, sizeof(struct rdma_lite_context));
    return lite_ctx;
}

void rdma_lite_free_context(struct rdma_lite_context* lite_ctx) { free(lite_ctx); }

int rdma_lite_init_mem_pool(struct rdma_lite_context* lite_ctx, struct rdma_lite_mem_attr* lite_mem_attr) { return 0; }

int rdma_lite_deinit_mem_pool(struct rdma_lite_context* lite_ctx, u32 mem_idx) { return 0; }

struct rdma_lite_cq* rdma_lite_create_cq(struct rdma_lite_context* lite_ctx, struct rdma_lite_cq_attr* lite_cq_attr)
{
    struct rdma_lite_cq* lite_cq = calloc(1, sizeof(struct rdma_lite_cq));
    return lite_cq;
}

int rdma_lite_destroy_cq(struct rdma_lite_cq* lite_cq)
{
    free(lite_cq);

    return 0;
}

int rdma_lite_poll_cq(struct rdma_lite_cq* lite_cq, int num_entries, struct rdma_lite_wc* lite_wc) { return 0; }

int rdma_lite_poll_cq_v2(struct rdma_lite_cq* lite_cq, int num_entries, struct rdma_lite_wc_v2* lite_wc) { return 0; }

struct rdma_lite_qp* rdma_lite_create_qp(struct rdma_lite_context* lite_ctx, struct rdma_lite_qp_attr* lite_qp_attr)
{
    struct rdma_lite_qp* lite_qp = calloc(1, sizeof(struct rdma_lite_qp));
    return lite_qp;
}

int rdma_lite_destroy_qp(struct rdma_lite_qp* lite_qp)
{
    free(lite_qp);

    return 0;
}

int rdma_lite_post_send(
    struct rdma_lite_qp* lite_qp, struct rdma_lite_send_wr* wr, struct rdma_lite_send_wr** bad_wr,
    struct rdma_lite_post_send_attr* attr, struct rdma_lite_post_send_resp* resp)
{
    return 0;
}

int rdma_lite_post_recv(struct rdma_lite_qp* lite_qp, struct rdma_lite_recv_wr* wr, struct rdma_lite_recv_wr** bad_wr)
{
    return 0;
}

int rdma_lite_set_qp_sl(struct rdma_lite_qp* lite_qp, int sl) { return 0; }

int rdma_lite_clean_qp(struct rdma_lite_qp* lite_qp) { return 0; }

int rdma_lite_restore_snapshot(struct rdma_lite_context* lite_ctx) { return 0; }
