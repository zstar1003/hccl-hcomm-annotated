/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INFINIBAND_VERBS_EXP_H
#define INFINIBAND_VERBS_EXP_H

#include <infiniband/verbs.h>
#include <infiniband/driver.h>
#include "peer_ops.h"
#include "rdma_lite_common.h"

int ibv_dontfork_range(void *base, size_t size);
int ibv_dofork_range(void *base, size_t size);
// gdr
#define IBV_EXP_WQE_ENTRY_LENGTH 64
#define IBV_EXP_TEMP_WQE_ENTRY_LENGTH 128
#define IBV_EXP_WQE_DEPTH_MASK (128 - 1)
#define IBV_EXP_SHARED_SQ_DEPTH 128
#define IBV_EXP_SHARED_TEMP_DEPTH 12

#define VERBS_NULL_POINT_RETURN_NULL(p)                                                                                \
    {                                                                                                                  \
        if ((p) == NULL) {                                                                                             \
            return NULL;                                                                                               \
        }                                                                                                              \
    }

#define VERBS_NULL_POINT_RETURN_ERR(p)                                                                                 \
    {                                                                                                                  \
        if ((p) == NULL) {                                                                                             \
            return (-EINVAL);                                                                                          \
        }                                                                                                              \
    }

#define ROCE_MIN(a, b) (((a) < (b)) ? (a) : (b))

#define PROCESS_PSIZE_LENGTH 49
#define PROCESS_PRESV_LENGTH 4

enum {
    IBV_WR_RDMA_WRITE_WITH_NOTIFY = 0x16,
};

struct roce_process_sign {
    int tgid;
    unsigned int devid; /* chip_id */
    unsigned int vfid;
    char sign[PROCESS_PSIZE_LENGTH];
};

struct ibv_exp_gdr_share_sq {
    int index;
    int sq_depth;
    int temp_depth;
    int db_depth;
    int dfx_depth;
    int sq_offset;
    int temp_offset;
    int db_offset;
    int dfx_offset;
    int max_sq_num;
};

struct ibv_exp_gdr_temp_wqe {
    pthread_mutex_t wqe_mutex;
    int temp_offset;
    int wqe_num;
    int use_cnt;
    unsigned long *bitmap;
    int bitmap_len;
};

struct ibv_exp_qp_init_attr {
    struct ibv_qp_init_attr attr;
    int gdr_enable;
    int lite_op_support;
    int mem_align; // 0,1:4KB, 2:2MB
    unsigned int mem_idx;
    unsigned int udp_sport;
    unsigned int ai_op_support;
    unsigned int grp_id;
    unsigned int qp_cstm_flag;
    unsigned int use_resv_mem;
    unsigned int resv_mem_pool_id;
    unsigned int resv[16U];
};

struct wr_exp_rsp {
    unsigned int wqe_index;
    unsigned long db_info;
};

struct ibv_post_send_ext_resp {
    unsigned int wqe_index;
    unsigned long db_info;
};

struct ibv_post_send_ext_attr {
    uint8_t reduce_op;
    uint8_t reduce_type;
};

struct ibv_exp_ah_attr {
    struct ibv_ah_attr attr;
    uint32_t udp_sport;
    uint32_t resv[4U];
};

struct verbs_context_exp {
    int (*drv_exp_ibv_poll_cq)(struct ibv_cq *ibcq, int num_entries, struct ibv_wc *wc);
    int (*exp_peer_commit_qp)(struct ibv_qp *qp, struct ibv_exp_peer_commit *peer);
    int (*drv_exp_post_send)(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr,
        struct wr_exp_rsp *exp_rsp);
    struct ibv_qp *(*drv_exp_create_qp)(struct ibv_pd *pd, struct ibv_exp_qp_init_attr *init_attr,
        struct rdma_lite_device_qp_attr *qp_resp);
    struct ibv_cq *(*drv_exp_create_cq)(struct ibv_context *context, int cqe, struct ibv_comp_channel *channel,
        int vector, struct rdma_lite_device_cq_init_attr *attr, struct rdma_lite_device_cq_attr *cq_resp);
    int (*drv_exp_query_notify)(struct ibv_context *ibv_context, unsigned long long *notify_va,
        unsigned long long *size);

    struct ibv_mr *(*drv_exp_reg_mr)(struct ibv_pd *pd, void *addr, size_t length, int access,
        struct roce_process_sign roce_sign);

    int (*drv_exp_query_device)(struct ibv_context *ibv_context, struct dev_cap_info *cap);

    size_t sz;
};

struct verbs_context_exp *verbs_get_exp_ctx(struct ibv_context *ctx);

#define verbs_get_exp_ctx_op(ctx, op)                                                                                  \
    ({                                                                                                                 \
        struct verbs_context_exp *_vctx = verbs_get_exp_ctx(ctx);                                                      \
        (!_vctx || (_vctx->sz < sizeof(*_vctx) - offsetof(struct verbs_context_exp, op)) || !_vctx->op) ? NULL         \
                                                                                                        : _vctx;       \
    })

struct ibv_qp *ibv_exp_create_qp(struct ibv_pd *pd, struct ibv_exp_qp_init_attr *qp_init_attr,
    struct rdma_lite_device_qp_attr *qp_resp);

struct ibv_cq *ibv_exp_create_cq(struct ibv_context *context, int cqe, void *cq_context,
    struct ibv_comp_channel *channel, int vector, struct rdma_lite_device_cq_init_attr *attr,
    struct rdma_lite_device_cq_attr *cq_resp);

int ibv_exp_query_get_ctx(struct ibv_context *context, unsigned long long *notify_va, unsigned long long *size);

int ibv_exp_query_notify(struct ibv_context *context, unsigned long long *notify_va, unsigned long long *size);

int ibv_exp_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr,
    struct wr_exp_rsp *exp_rsp);

struct ibv_mr *ibv_exp_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access,
    struct roce_process_sign roce_sign);

int ibv_exp_query_device(struct ibv_context *context, struct dev_cap_info *cap);

struct ibv_ah *ibv_exp_create_ah(struct ibv_pd *pd, struct ibv_exp_ah_attr *attrx);

#endif
